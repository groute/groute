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

#ifndef __GROUTE_GRAPHS_FUSED_SOLVER_H
#define __GROUTE_GRAPHS_FUSED_SOLVER_H

#include <vector>
#include <algorithm>
#include <thread>
#include <memory>

#include <cub/grid/grid_barrier.cuh>

#include <groute/event_pool.h>
#include <groute/fused_worker.h>

#include <groute/graphs/traversal_algo.h>

DEFINE_int32(fused_chunk_size, INT_MAX, "Intermediate peer transfer");

namespace groute {
namespace graphs {

    namespace traversal
    {
        template<typename Algo, typename ProblemType, typename TLocal ,typename TRemote, typename PrioT, typename SplitOps, typename... WorkArgs>
        struct FusedSolver
        {
            ProblemType& m_problem;
            cub::GridBarrierLifetime m_barrier_lifetime;

            dim3 m_grid_dims;
            enum { BlockSize = 256 };

            volatile int* m_work_counters[2];
            enum
            {
                HIGH_COUNTER = 0,
                LOW_COUNTER
            };
            uint32_t *m_kernel_internal_counter;
            Signal m_work_signal;

            Endpoint m_endpoint;
            std::vector<int> m_work_counts;

        public:
            FusedSolver(Context<Algo>& context, ProblemType& problem) : m_problem(problem), m_endpoint()
            {
                GROUTE_CUDA_CHECK(cudaMallocHost(&m_work_counters[HIGH_COUNTER], sizeof(int)));
                GROUTE_CUDA_CHECK(cudaMallocHost(&m_work_counters[LOW_COUNTER], sizeof(int)));

                *m_work_counters[HIGH_COUNTER] = 0;
                *m_work_counters[LOW_COUNTER] = 0;

                GROUTE_CUDA_CHECK(cudaMalloc(&m_kernel_internal_counter, sizeof(int)));
                GROUTE_CUDA_CHECK(cudaMemset(m_kernel_internal_counter, 0, sizeof(int)));

                int dev;
                cudaDeviceProp props;

                GROUTE_CUDA_CHECK(cudaGetDevice(&dev));
                GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

                int fused_work_residency = 0;

                if (FLAGS_iteration_fusion)
                {
                    if (FLAGS_cta_np)
                    {
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &fused_work_residency,
                            groute::FusedWork 
                            < groute::NeverStop, TLocal, 
                              TRemote, PrioT, SplitOps, 
                              typename ProblemType::WorkTypeCTA, WorkArgs... >,
                            BlockSize, 0);
                    }
                    else
                    {
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &fused_work_residency,
                            groute::FusedWork
                            < groute::NeverStop, TLocal, 
                              TRemote, PrioT, SplitOps, 
                              typename ProblemType::WorkType, WorkArgs... >,
                            BlockSize, 0);
                    }
                }
                else
                {
                    if (FLAGS_cta_np)
                    {
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &fused_work_residency,
                            groute::FusedWork 
                            < groute::RunNTimes<1>, TLocal, 
                              TRemote, PrioT, SplitOps, 
                              typename ProblemType::WorkTypeCTA, WorkArgs... >,
                            BlockSize, 0);
                    }
                    else
                    {
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &fused_work_residency,
                            groute::FusedWork 
                            < groute::RunNTimes<1>, TLocal, 
                              TRemote, PrioT, SplitOps, 
                              typename ProblemType::WorkType, WorkArgs... >,
                            BlockSize, 0);
                    }
                }

                size_t fused_work_blocks = (props.multiProcessorCount - 1) * fused_work_residency; // -1 for split-receive  

                if (FLAGS_verbose)
                {
                    printf("%d - fused kernel, multi processor count: %d, residency: %d, blocks: %llu [(mp-1)*residency]\n", dev, props.multiProcessorCount, fused_work_residency, fused_work_blocks);
                }

                m_grid_dims = dim3(fused_work_blocks, 1, 1);

                // Setup the barrier
                m_barrier_lifetime.Setup(fused_work_blocks);
            }

            ~FusedSolver()
            {
                if (FLAGS_verbose)
                {
                    for (size_t i = 0; i < m_work_counts.size(); i++)
                    {
                        printf("%d, %llu, %d\n", (groute::Endpoint::identity_type)m_endpoint, i, m_work_counts[i]);
                    }
                }

                GROUTE_CUDA_CHECK(cudaFreeHost((void*)m_work_counters[HIGH_COUNTER]));
                GROUTE_CUDA_CHECK(cudaFreeHost((void*)m_work_counters[LOW_COUNTER]));
                GROUTE_CUDA_CHECK(cudaFree((void*)m_kernel_internal_counter));
            }

            void PersistenceKernelSizing(dim3& grid_dims, dim3& block_dims) const
            {
                dim3 bd(BlockSize, 1, 1);
                grid_dims = m_grid_dims;
                block_dims = bd;
            }

            void Solve(
                groute::Context& context,
                groute::Endpoint endpoint,
                groute::DistributedWorklist<TLocal, TRemote, SplitOps>& dwl,
                groute::IDistributedWorklistPeer<TLocal, TRemote>* peer,
                groute::Stream& stream)
            {
                m_endpoint = endpoint;

                groute::Worklist<TLocal>* immediate_worklist = &peer->GetLocalWorkspace(0);
                groute::Worklist<TLocal>* deferred_worklist = &peer->GetLocalWorkspace(1);

                groute::CircularWorklist<TRemote>*  remote_output = &peer->GetRemoteOutputWorklist();
                groute::CircularWorklist<TLocal>*  remote_input = &peer->GetLocalInputWorklist();

                volatile int *immediate_work_counter, *deferred_work_counter;

                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&immediate_work_counter, (int*)m_work_counters[HIGH_COUNTER], 0));
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&deferred_work_counter, (int*)m_work_counters[LOW_COUNTER], 0));
                
                dim3 grid_dims;
                dim3 block_dims;
                PersistenceKernelSizing(grid_dims, block_dims);

                int prev_signal = m_work_signal.Peek();

                if(m_problem.DoFusedInit(
                        immediate_worklist, deferred_worklist,
                        remote_input, remote_output,
                        FLAGS_fused_chunk_size,
                        dwl.GetPriorityThreshold(),
                        immediate_work_counter, deferred_work_counter,
                        m_kernel_internal_counter,
                        m_work_signal.GetDevPtr(),
                        m_barrier_lifetime,
                        grid_dims, block_dims, stream))
                {
                    stream.BeginSync();

                    while (true) // Loop on work signals from the running kernel
                    {
                        int signal = m_work_signal.WaitForSignal(prev_signal, stream);
                        if (signal == prev_signal) break; // Exit was signaled  

                        int work = signal - prev_signal;
                        dwl.ReportWork(work, 0, "SplitSend", m_endpoint);

                        prev_signal = signal; // Update
                        peer->SignalRemoteWork(Event()); // Trigger work sending to router
                    }

                    stream.EndSync(); // Sync             
                    
                    // Some work was done by init, report it   
                    dwl.ReportDeferredWork(*m_work_counters[LOW_COUNTER], 0, Algo::Name(), endpoint);
                    dwl.ReportWork(*m_work_counters[HIGH_COUNTER], 0, Algo::Name(), endpoint);
                }

                while (dwl.HasWork())
                {
                    int priority_threshold = dwl.GetPriorityThreshold();

                    if (FLAGS_verbose)
                    {
                        int high_in = immediate_worklist->GetLength(stream);
                        int low_in = deferred_worklist->GetLength(stream);
                        int remote_in = remote_input->GetLength(stream);
                        int remote_out = remote_output->GetLength(stream);

                        printf("%d - start kernel, prio %d, LH: %d, LL: %d, RI: %d, RO: %d\n", (groute::Endpoint::identity_type)endpoint, priority_threshold, high_in, low_in, remote_in, remote_out);
                    }

                    if (FLAGS_count_work)
                    {
                        Marker::MarkWorkitems(dwl.GetCurrentWorkCount(endpoint), "FusedWork");
                    }

                    m_problem.DoFusedWork(
                        immediate_worklist, deferred_worklist,
                        remote_input, remote_output,
                        FLAGS_fused_chunk_size,
                        priority_threshold,
                        immediate_work_counter, deferred_work_counter,
                        m_kernel_internal_counter,
                        m_work_signal.GetDevPtr(),
                        m_barrier_lifetime,
                        grid_dims, block_dims, stream
                        );
                    
                    stream.BeginSync();

                    while (true) // Loop on work signals from the running kernel
                    {
                        int signal = m_work_signal.WaitForSignal(prev_signal, stream);
                        if (signal == prev_signal) break; // Exit was signaled  

                        int work = signal - prev_signal;
                        dwl.ReportWork(work, 0, "SplitSend", m_endpoint);

                        prev_signal = signal; // Update
                        peer->SignalRemoteWork(Event()); // Trigger work sending to router
                    }

                    stream.EndSync(); // Sync             
                    
                    // Some work was done by the kernel, report it   
                    dwl.ReportDeferredWork(*m_work_counters[LOW_COUNTER], 0, Algo::Name(), endpoint);
                    dwl.ReportWork(*m_work_counters[HIGH_COUNTER], 0, Algo::Name(), endpoint);

                    if (FLAGS_verbose)
                    {
                        printf("%d - done kernel, LWC: %d, HWC: %d\n", (groute::Endpoint::identity_type)endpoint, *m_work_counters[LOW_COUNTER], *m_work_counters[HIGH_COUNTER]);
                    }

                    if (FLAGS_count_work)
                    {
                        m_work_counts.push_back((int)(*m_work_counters[LOW_COUNTER]) + (int)(*m_work_counters[HIGH_COUNTER]));
                    }

                    auto segs = peer->WaitForLocalWork(stream, priority_threshold);

                    if (FLAGS_verbose)
                    {
                        int segs_size = 0;
                        for (auto& seg : segs)
                        {
                            segs_size += seg.GetSegmentSize();
                        }

                        printf("%d - after wait, segs_total: %d, prev prio: %d, curr prio: %d\n", (groute::Endpoint::identity_type)endpoint, segs_size, priority_threshold, dwl.GetPriorityThreshold());
                    }

                    if (priority_threshold < dwl.GetPriorityThreshold()) // priority is up
                    {
                        // Assuming that immediate worklist is empty. 
                        // This will process all of deferred_worklist in the following iteration
                        std::swap(immediate_worklist, deferred_worklist);
                    }
                }
            }
        };
    }
}
}

#endif // __GROUTE_GRAPHS_FUSED_SOLVER_H
