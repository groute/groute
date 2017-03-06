// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __GROUTE_WORKLIST_WORKERS_H
#define __GROUTE_WORKLIST_WORKERS_H

#include <vector>
#include <algorithm>
#include <thread>
#include <memory>

#include <cub/grid/grid_barrier.cuh>

#include <groute/context.h>
#include <groute/dwl/work_kernels.cuh>

#include <gflags/gflags.h>
#include <utils/markers.h>

DEFINE_int32(fused_chunk_size, INT_MAX, "Intermediate peer transfer");

namespace groute {

    // Templated APIs:
    /*

    ----

    DistributedWorklist Worker API: (two types provided by the library, fused and unopt, below)
    struct Worker
    {
    public:
        static const int num_workspaces = ...;    
        static const bool soft_prio = ...;

        Worker(Context& context, Endpoint endpoint) { ... }
        void Work(
            DistributedWorklist& distributed_worklist,
            DistributedWorklistPeer* peer, Stream& stream, const WorkArgs&... args)
        { 
            ...
        }
    };

    ----

    WorkSource API:
    struct WorkSource
    {
        __device__ uint32_t get_size() const { return ...; }
        __device__ T get_work(uint32_t i) const { return ...; }
    };

    ----

    WorkTarget API:
    template<typename TLocal, typename TRemote>
    struct WorkTarget
    {
        __device__ void append_work(const TLocal& work) { ... }
        __device__ void append_work(const TRemote& work) { ... }
    };

    ----

    Work API: (user application needs to implement this)
    struct Work
    {
        template<typename WorkSource, typename WorkTarget, typename... WorkArgs>
        __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
            WorkArgs&... args
            )
        { ... }
    };
    */

    /*
    * @brief An optimized DW worker, with complete kernel fusion and soft-priority scheduling   
    */
    template<bool IterationFusion, typename TLocal, typename TRemote, typename TPrio, typename DWCallbacks, typename TWork, typename... WorkArgs>
    class FusedWorker
    {
    public:
        static const int num_workspaces = 2; // Part of the contract used by the DistributedWorklist    
        static const bool soft_prio = true;  //

        typedef typename std::conditional<IterationFusion, NeverStop, RunNTimes<1>>::type StoppingCondition;
        
    private:
        enum
        {
            IMMEDIATE_COUNTER = 0,
            DEFERRED_COUNTER = 1,
            BLOCK_SIZE = 256 
        };
        static const char* WorkerName() { return "FusedWorker"; }
        static const char* KernelName() { return "FusedWorkKernel"; }

        Endpoint m_endpoint;
        
        dim3 m_grid_dims, m_block_dims;
        cub::GridBarrierLifetime m_barrier_lifetime;

        // For reporting immediate/deferred work from fused kernel
        volatile int* m_work_counters[2];    

        // Used for globally deciding on work within the fused kernel  
        uint32_t *m_kernel_internal_counter;  

        // Used for signaling remote work was pushed into the circular-queue and can be sent to peers through the router
        Signal m_work_signal;   

        std::vector<int> m_work_counts; // Trace work for each iteration  

    public:
        FusedWorker(Context& context, Endpoint endpoint) : m_endpoint(endpoint)
        {
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_work_counters[IMMEDIATE_COUNTER], sizeof(int)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_work_counters[DEFERRED_COUNTER], sizeof(int)));

            *m_work_counters[IMMEDIATE_COUNTER] = 0;
            *m_work_counters[DEFERRED_COUNTER] = 0;

            GROUTE_CUDA_CHECK(cudaMalloc(&m_kernel_internal_counter, sizeof(int)));
            GROUTE_CUDA_CHECK(cudaMemset(m_kernel_internal_counter, 0, sizeof(int)));

            int dev;
            cudaDeviceProp props;

            GROUTE_CUDA_CHECK(cudaGetDevice(&dev));
            GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

            int occupancy_per_MP = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                groute::FusedWorkKernel <StoppingCondition, TLocal, TRemote, TPrio, DWCallbacks, TWork, WorkArgs... >, BLOCK_SIZE, 0);

            size_t fused_work_blocks 
                = (props.multiProcessorCount - 1) * occupancy_per_MP; // -1 for split-receive  

            if (FLAGS_verbose)
            {
                printf(
                    "%d - fused kernel, multi processor count: %d, occupancy: %d, blocks: %llu [(mp-1)*occupancy]\n", 
                    dev, props.multiProcessorCount, occupancy_per_MP, fused_work_blocks);
            }

            // Setup the fused block/grid dimensions 
            m_block_dims = dim3(BLOCK_SIZE, 1, 1);
            m_grid_dims = dim3(fused_work_blocks, 1, 1);

            // Setup the barrier
            m_barrier_lifetime.Setup(fused_work_blocks);
        }

        ~FusedWorker()
        {
            if (FLAGS_verbose)
            {
                for (size_t i = 0; i < m_work_counts.size(); i++)
                {
                    printf("%d, %llu, %d\n", (Endpoint::identity_type)m_endpoint, i, m_work_counts[i]);
                }
            }

            GROUTE_CUDA_CHECK(cudaFreeHost((void*)m_work_counters[IMMEDIATE_COUNTER]));
            GROUTE_CUDA_CHECK(cudaFreeHost((void*)m_work_counters[DEFERRED_COUNTER]));
            GROUTE_CUDA_CHECK(cudaFree((void*)m_kernel_internal_counter));
        }

        void Work(
            IDistributedWorklist <TLocal, TRemote>& distributed_worklist,
            IDistributedWorklistPeer <TLocal, TRemote, DWCallbacks>* peer, Stream& stream, const WorkArgs&... args)
        {
            Worklist<TLocal>* immediate_worklist = &peer->GetLocalWorkspace(0);
            Worklist<TLocal>* deferred_worklist = &peer->GetLocalWorkspace(1);

            CircularWorklist<TRemote>*  remote_output = &peer->GetRemoteOutputWorklist();
            CircularWorklist<TLocal>*  remote_input = &peer->GetLocalInputWorklist();

            DWCallbacks callbacks = peer->GetDeviceCallbacks();

            volatile int *immediate_work_counter, *deferred_work_counter;

            GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&immediate_work_counter, (int*)m_work_counters[IMMEDIATE_COUNTER], 0));
            GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&deferred_work_counter, (int*)m_work_counters[DEFERRED_COUNTER], 0));

            int prev_signal = m_work_signal.Peek();

            while (distributed_worklist.HasWork())
            {
                int priority_threshold = distributed_worklist.GetPriorityThreshold();

                if (FLAGS_verbose)
                {
                    int immediate_in = immediate_worklist->GetLength(stream);
                    int deferred_in = deferred_worklist->GetLength(stream);
                    int remote_in = remote_input->GetLength(stream);
                    int remote_out = remote_output->GetLength(stream);

                    printf(
                        "%d - start kernel, prio %d, LH: %d, LL: %d, RI: %d, RO: %d\n", 
                        (Endpoint::identity_type)m_endpoint, priority_threshold, immediate_in, deferred_in, remote_in, remote_out);
                }

                if (FLAGS_count_work)
                {
                    Marker::MarkWorkitems(distributed_worklist.GetCurrentWorkCount(m_endpoint), KernelName());
                }
                
                groute::FusedWorkKernel <StoppingCondition, TLocal, TRemote, TPrio, DWCallbacks, TWork, WorkArgs... >

                    <<< m_grid_dims, m_block_dims, 0, stream.cuda_stream >>> (

                    immediate_worklist->DeviceObject(), deferred_worklist->DeviceObject(),
                    remote_input->DeviceObject(), remote_output->DeviceObject(),
                    FLAGS_fused_chunk_size, priority_threshold,
                    immediate_work_counter, deferred_work_counter,
                    m_kernel_internal_counter, m_work_signal.GetDevPtr(),
                    m_barrier_lifetime,                       
                    callbacks,
                    args...
                    );

                stream.BeginSync(); // Records an event  

                while (true) // Loop on work signals from the running kernel
                {
                    int signal = m_work_signal.WaitForSignal(prev_signal, stream);
                    if (signal == prev_signal) break; // Exit was signaled  

                    int work = signal - prev_signal;
                    distributed_worklist.ReportWork(work, 0, m_endpoint);

                    prev_signal = signal; // Update
                    peer->SignalRemoteWork(Event()); // Trigger work sending to router
                }

                stream.EndSync(); // Sync on the kernel        

                // Some work was done by the kernel, report it   
                distributed_worklist.ReportDeferredWork(*m_work_counters[DEFERRED_COUNTER], 0, m_endpoint);
                distributed_worklist.ReportWork(*m_work_counters[IMMEDIATE_COUNTER], 0, m_endpoint);

                if (FLAGS_verbose)
                {
                    printf(
                        "%d - done kernel, LWC: %d, HWC: %d\n", 
                        (Endpoint::identity_type)m_endpoint, *m_work_counters[DEFERRED_COUNTER], *m_work_counters[IMMEDIATE_COUNTER]);
                }

                if (FLAGS_count_work)
                {
                    m_work_counts.push_back((int)(*m_work_counters[DEFERRED_COUNTER]) + (int)(*m_work_counters[IMMEDIATE_COUNTER]));
                }

                auto segs = peer->WaitForLocalWork(stream, priority_threshold);

                if (FLAGS_verbose)
                {
                    int segs_size = 0;
                    for (auto& seg : segs)
                    {
                        segs_size += seg.GetSegmentSize();
                    }

                    printf(
                        "%d - after wait, segs_total: %d, prev prio: %d, curr prio: %d\n", 
                        (Endpoint::identity_type)m_endpoint, segs_size, priority_threshold, distributed_worklist.GetPriorityThreshold());
                }

                if (priority_threshold < distributed_worklist.GetPriorityThreshold()) // priority is up
                {
                    // Assuming that immediate worklist is empty. 
                    // This will process all of deferred_worklist in the following iteration
                    std::swap(immediate_worklist, deferred_worklist);
                }
            }
        }
    };

    /*
    * @brief A simple unoptimized DW worker 
    */
    template<typename TLocal, typename TRemote, typename DWCallbacks, typename TWork, typename... WorkArgs>
    class Worker
    {
        Endpoint m_endpoint;

        static const char* WorkerName() { return "UnoptimizedWorker"; }
        static const char* KernelName() { return "WorkKernel"; }
        
    public:
        static const int num_workspaces = 1;    
        static const bool soft_prio = false;

        Worker(Context& context, Endpoint endpoint) : m_endpoint(endpoint) { }
                
        void Work(
            IDistributedWorklist <TLocal, TRemote>& distributed_worklist,
            IDistributedWorklistPeer <TLocal, TRemote, DWCallbacks>* peer, Stream& stream, const WorkArgs&... args)
        {
                auto& input_worklist = peer->GetLocalInputWorklist();
                auto& workspace = peer->GetLocalWorkspace(0); 
                DWCallbacks callbacks = peer->GetDeviceCallbacks();

                while (distributed_worklist.HasWork())
                {
                    auto input_segs = peer->WaitForLocalWork(stream);
                    size_t new_work = 0, performed_work = 0;

                    if (input_segs.empty()) continue;

                    // If circular queue passed the end, 2 buffers will be given: [s,capacity) and [0,e)
                    for (auto subseg : input_segs)
                    {
                        dim3 grid_dims, block_dims;
                        KernelSizing(grid_dims, block_dims, subseg.GetSegmentSize());

                        Marker::MarkWorkitems(subseg.GetSegmentSize(), KernelName());
                        WorkKernel <dev::WorkSourceArray<index_t>, TLocal, TRemote, DWCallbacks, TWork, WorkArgs...>

                            <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (

                                dev::WorkSourceArray<index_t>(subseg.GetSegmentPtr(), subseg.GetSegmentSize()),
                                workspace.DeviceObject(),
                                callbacks,
                                args...
                                );

                        input_worklist.PopItemsAsync(subseg.GetSegmentSize(), stream.cuda_stream);
                        performed_work += subseg.GetSegmentSize();
                    }

                    auto output_seg = workspace.ToSeg(stream);
                    new_work += output_seg.GetSegmentSize(); // Add the new work 
                    peer->SplitSend(output_seg, stream); // Call split-send

                    workspace.ResetAsync(stream.cuda_stream); // Reset the locak workspace  

                    // Report work
                    distributed_worklist.ReportWork((int)new_work, (int)performed_work, m_endpoint);
                }
            }
    };
}

#endif // __GROUTE_WORKLIST_WORKERS_H
