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

            groute::Worklist<TLocal> m_worklist_a;
            groute::Worklist<TLocal> m_worklist_b;

            dim3 m_grid_dims;
            enum
            {
                BlockSize = 256
            };

            volatile int* m_work_counters[2];
            enum
            {
                HIGH_COUNTER = 0,
                LOW_COUNTER
            };
            uint32_t *m_kernel_internal_counter;

            device_t m_dev;
            std::vector<int> m_work_counts;

        public:
            FusedSolver(Context<Algo>& context, ProblemType& problem) : m_problem(problem), m_dev(groute::Device::Null)
            {
                void* mem_buffer;
                size_t mem_size;
                
                // groute::opt::DistributedWorklistPeer allocates 0.3 + 0.15 + 0.15
                // so here below we use the remaining 0.2 + 0.2  

                mem_buffer = context.Alloc(0.2, mem_size);
                m_worklist_a = groute::Worklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal));
                
                mem_buffer = context.Alloc(0.2, mem_size);
                m_worklist_b = groute::Worklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal));

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
                              typename ProblemType::WorkTypeNP, WorkArgs... >,
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
                              typename ProblemType::WorkTypeNP, WorkArgs... >,
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
                        printf("%d, %llu, %d\n", m_dev, i, m_work_counts[i]);
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
                groute::device_t dev,
                groute::opt::DistributedWorklist<TLocal, TRemote, SplitOps>& dwl,
                groute::opt::IDistributedWorklistPeer<TLocal, TRemote>* peer,
                groute::Stream& stream)
            {
                m_dev = dev;

                m_worklist_a.ResetAsync(stream.cuda_stream);
                m_worklist_b.ResetAsync(stream.cuda_stream);

                groute::Worklist<TLocal>* lwl_high = &m_worklist_a;
                groute::Worklist<TLocal>* lwl_low = &m_worklist_b;

                groute::CircularWorklist<TRemote>*  rwl_out = &peer->GetRemoteOutputWorklist();
                groute::CircularWorklist<TLocal>*  rwl_in = &peer->GetLocalInputWorklist();

                volatile int *send_signal_ptr;
                volatile int *high_work_counter, *low_work_counter;

                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&send_signal_ptr, (int*)peer->GetSendSignalPtr(), 0));
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&high_work_counter, (int*)m_work_counters[HIGH_COUNTER], 0));
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&low_work_counter, (int*)m_work_counters[LOW_COUNTER], 0));
                
                dim3 grid_dims;
                dim3 block_dims;
                PersistenceKernelSizing(grid_dims, block_dims);

                if(m_problem.DoFusedInit(
                        lwl_high, lwl_low,
                        rwl_in, rwl_out,
                        FLAGS_fused_chunk_size,
                        dwl.GetCurrentPrio(),
                        high_work_counter, low_work_counter,
                        m_kernel_internal_counter,
                        send_signal_ptr,
                        m_barrier_lifetime,
                        grid_dims, block_dims, stream))
                {
                    // Sync
                    stream.Sync();                 
                    
                    // Wait until the signal has been processed by the sender thread
                    volatile int *host_send_signal = peer->GetSendSignalPtr();
                    while (*host_send_signal != peer->GetLastSendSignal())
                        std::this_thread::yield();
                    
                    // Some work was done by init, report it   
                    dwl.ReportLowPrioWork(*m_work_counters[LOW_COUNTER], 0, Algo::Name(), dev);
                    dwl.ReportHighPrioWork(*m_work_counters[HIGH_COUNTER], 0, Algo::Name(), dev);
                }

                while (dwl.HasWork())
                {
                    int global_prio = dwl.GetCurrentPrio();

                    if (FLAGS_verbose)
                    {
                        int high_in = lwl_high->GetLength(stream);
                        int low_in = lwl_low->GetLength(stream);
                        int remote_in = rwl_in->GetLength(stream);
                        int remote_out = rwl_out->GetLength(stream);

                        printf("%d - start kernel, prio %d, LH: %d, LL: %d, RI: %d, RO: %d\n", dev, global_prio, high_in, low_in, remote_in, remote_out);
                    }

                    if (FLAGS_count_work)
                    {
                        Marker::MarkWorkitems(dwl.GetCurrentWorkCount(dev), "FusedWork");
                    }

                    m_problem.DoFusedWork(
                        lwl_high, lwl_low,
                        rwl_in, rwl_out,
                        FLAGS_fused_chunk_size,
                        global_prio,
                        high_work_counter, low_work_counter,
                        m_kernel_internal_counter,
                        send_signal_ptr,
                        m_barrier_lifetime,
                        grid_dims, block_dims, stream
                        );
                    stream.Sync();

                    if (FLAGS_verbose)
                    {
                        printf("%d - done kernel, LWC: %d, HWC: %d\n", dev, *m_work_counters[LOW_COUNTER], *m_work_counters[HIGH_COUNTER]);
                    }

                    // Wait until the signal has been processed by the sender thread
                    volatile int *host_send_signal = peer->GetSendSignalPtr();
                    while (*host_send_signal != peer->GetLastSendSignal())
                        std::this_thread::yield();

                    dwl.ReportLowPrioWork(*m_work_counters[LOW_COUNTER], 0, Algo::Name(), dev);
                    dwl.ReportHighPrioWork(*m_work_counters[HIGH_COUNTER], 0, Algo::Name(), dev);

                    if (FLAGS_count_work)
                    {
                        m_work_counts.push_back((int)(*m_work_counters[LOW_COUNTER]) + (int)(*m_work_counters[HIGH_COUNTER]));
                    }

                    auto segs = peer->WaitForPrioOrWork(global_prio, stream);

                    if (FLAGS_verbose)
                    {
                        int segs_size = 0;
                        for (auto& seg : segs)
                        {
                            segs_size += seg.GetSegmentSize();
                        }

                        printf("%d - after wait, segs_total: %d, prev prio: %d, curr prio: %d\n", dev, segs_size, global_prio, dwl.GetCurrentPrio());
                    }

                    if (global_prio < dwl.GetCurrentPrio()) // priority is up
                    {
                        // Assuming that high priority is empty
                        std::swap(lwl_high, lwl_low);
                        // THIS WILL RUN ALL OF lwl_low NEXT TIME
                    }
                }
            }
        };
    }
}
}

#endif // __GROUTE_GRAPHS_FUSED_SOLVER_H
