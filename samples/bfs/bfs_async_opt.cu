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
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>

#include <gflags/gflags.h>

#include <groute/event_pool.h>
#include <groute/fused_distributed_worklist.h>
#include <groute/fused_worker.h>
#include <groute/cta_work.h>

#include <groute/graphs/csr_graph.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/graphs/fused_solver.h>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include "bfs_common.h"

DECLARE_int32(source_node);

DEFINE_bool(exitonerror, false, "exit on error");


namespace bfs {
    namespace opt {
        
        const level_t INF = UINT_MAX;

        struct LevelData
        {
            index_t node;
            level_t level;

            __device__ __host__ __forceinline__ LevelData(index_t node, level_t level) : node(node), level(level) { }
            __device__ __host__ __forceinline__ LevelData() : node(INF), level(INF) { }
        };

        typedef index_t local_work_t;
        typedef LevelData remote_work_t;

        __global__ void BFSInit(level_t* levels, int nnodes)
        {
            int tid = GTID;
            if (tid < nnodes)
            {
                levels[tid] = INF;
            }
        }

        template<
            typename TGraph,
            typename TGraphDatum>
        struct BFSWorkNP
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph, TGraphDatum& levels_datum
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                uint32_t work_size = work_source.get_size();
                uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<level_t> np_local = { 0, 0, 0 };

                    if (i < work_size)
                    {
                        index_t node = work_source.get_work(i);
                        np_local.start = graph.begin_edge(node);
                        np_local.size = graph.end_edge(node) - np_local.start;
                        np_local.meta_data = levels_datum.get_item(node) + 1;
                    }

                    groute::dev::CTAWorkScheduler<level_t>::template schedule(
                        np_local, 
                        [&graph, &levels_datum, &rwl_in, &rwl_out](index_t edge, level_t next_level)
                        {
                            index_t dest = graph.edge_dest(edge);
                            if (next_level < atomicMin(levels_datum.get_item_ptr(dest), next_level))
                            {
                                int is_owned = graph.owns(dest);

                                // TODO: move ballot logic to a device structure   

                                int owned_mask = __ballot(is_owned ? 1 : 0);
                                int remote_mask = __ballot(is_owned ? 0 : 1);

                                if (is_owned)
                                {
                                    int high_leader = __ffs(owned_mask) - 1;
                                    int thread_offset = __popc(owned_mask & ((1 << lane_id()) - 1));
                                    rwl_in.prepend_warp(dest, high_leader, __popc(owned_mask), thread_offset);
                                }
                                else
                                {
                                    int low_leader = __ffs(remote_mask) - 1;
                                    int thread_offset = __popc(remote_mask & ((1 << lane_id()) - 1));
                                    rwl_out.append_warp(LevelData(dest, next_level), low_leader, __popc(remote_mask), thread_offset);
                                }
                            }
                        }
                        ); 
                }
            }
        };

        template<
            typename TGraph,
            typename TGraphDatum>
        struct BFSWork
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph, TGraphDatum& levels_datum
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                uint32_t work_size = work_source.get_size();

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
                    level_t next_level = levels_datum.get_item(node) + 1;

                    for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge)
                    {
                        index_t dest = graph.edge_dest(edge);
                        if (next_level < atomicMin(levels_datum.get_item_ptr(dest), next_level))
                        {
                            int is_owned = graph.owns(dest);

                            // TODO: move ballot logic to a device structure   

                            int owned_mask = __ballot(is_owned ? 1 : 0);
                            int remote_mask = __ballot(is_owned ? 0 : 1);

                            if (is_owned)
                            {
                                int high_leader = __ffs(owned_mask) - 1;
                                int thread_offset = __popc(owned_mask & ((1 << lane_id()) - 1));
                                rwl_in.prepend_warp(dest, high_leader, __popc(owned_mask), thread_offset);
                            }
                            else
                            {
                                int low_leader = __ffs(remote_mask) - 1;
                                int thread_offset = __popc(remote_mask & ((1 << lane_id()) - 1));
                                rwl_out.append_warp(LevelData(dest, next_level), low_leader, __popc(remote_mask), thread_offset);
                            }
                        }
                    }
                }
            }
        };

        struct SplitOps
        {
        private:
            groute::graphs::dev::CSRGraphSeg m_graph_seg;
            groute::graphs::dev::GraphDatum<level_t> m_levels_datum;

        public:
            template<typename...UnusedData>
            SplitOps(const groute::graphs::dev::CSRGraphSeg& graph_seg, const groute::graphs::dev::GraphDatum<level_t>& levels_datum, UnusedData&... data)
                : m_graph_seg(graph_seg), m_levels_datum(levels_datum)
            {
            }

            __device__ __forceinline__ groute::opt::SplitFlags on_receive(const remote_work_t& work)
            {
                if (m_graph_seg.owns(work.node))
                {
                    return (work.level < atomicMin(m_levels_datum.get_item_ptr(work.node), work.level))
                        ? groute::opt::SF_Take
                        : groute::opt::SF_None; // filter
                }

                return groute::opt::SF_Pass;
            }

            __device__ __forceinline__ bool is_high_prio(const local_work_t& work, const level_t& global_prio)
            {
                return m_levels_datum[work] <= global_prio;
            }

            __device__ __forceinline__ groute::opt::SplitFlags on_send(local_work_t work)
            {
                return (m_graph_seg.owns(work))
                    ? groute::opt::SF_Take
                    : groute::opt::SF_Pass;
            }

            __device__ __forceinline__ remote_work_t pack(local_work_t work)
            {
                return LevelData(work, m_levels_datum.get_item(work));
            }

            __device__ __forceinline__ local_work_t unpack(const remote_work_t& work)
            {
                return work.node;
            }
        };

        template<typename TGraph, typename TGraphDatum>
        struct FusedProblem
        {
            TGraph m_graph;
            TGraphDatum m_levels_datum;

            typedef BFSWork<TGraph, TGraphDatum> WorkType;
            typedef BFSWorkNP<TGraph, TGraphDatum> WorkTypeNP;

        public:
            FusedProblem(const TGraph& graph, const TGraphDatum& levels_datum) :
                m_graph(graph), m_levels_datum(levels_datum)
            {
            }

            // Called before a global CPU+GPU barrier
            void Init(groute::Stream& stream) const
            {
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, m_levels_datum.size);

                BFSInit <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_levels_datum.data_ptr, m_levels_datum.size);
            }

            bool DoFusedInit(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, level_t global_prio,
                volatile int *high_work_counter, volatile int *low_work_counter,
                uint32_t *kernel_internal_counter, volatile int *send_signal_ptr,
                cub::GridBarrierLifetime& barrier_lifetime,
                dim3 grid_dims, dim3 block_dims, groute::Stream& stream)
            {
                return false; // no work was done here
            }

            void DoFusedWork(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, level_t global_prio,
                volatile int *high_work_counter, volatile int *low_work_counter,
                uint32_t *kernel_internal_counter, volatile int *send_signal_ptr,
                cub::GridBarrierLifetime& barrier_lifetime,
                dim3 grid_dims, dim3 block_dims, groute::Stream& stream)
            {
                if (FLAGS_iteration_fusion)
                {
                    if (FLAGS_cta_np)
                    {
                        groute::FusedWork <
                            groute::NeverStop, local_work_t, remote_work_t, level_t, SplitOps,
                            WorkTypeNP,
                            TGraph, TGraphDatum >

                            <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,                       
                            bfs::opt::SplitOps(m_graph, m_levels_datum),
                            m_graph, m_levels_datum
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::NeverStop, local_work_t, remote_work_t, level_t, SplitOps,
                            WorkType,
                            TGraph, TGraphDatum >

                            <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            bfs::opt::SplitOps(m_graph, m_levels_datum),
                            m_graph, m_levels_datum
                            );
                    }
                }
                else
                {
                    if (FLAGS_cta_np)
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, level_t, SplitOps,
                            WorkTypeNP,
                            TGraph, TGraphDatum >

                            <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            bfs::opt::SplitOps(m_graph, m_levels_datum),
                            m_graph, m_levels_datum
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, level_t, SplitOps,
                            WorkType,
                            TGraph, TGraphDatum >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            bfs::opt::SplitOps(m_graph, m_levels_datum),
                            m_graph, m_levels_datum
                            );
                    }
                }
            }
        };

        struct Algo
        {
            static const char* NameLower()      { return "bfs"; }
            static const char* Name()           { return "BFS"; }

            static void Init(
                groute::graphs::traversal::Context<bfs::opt::Algo>& context,
                groute::graphs::multi::CSRGraphAllocator& graph_manager,
                groute::router::Router<remote_work_t>& worklist_router,
                groute::opt::DistributedWorklist<local_work_t, remote_work_t, bfs::opt::SplitOps>& distributed_worklist)
            {
                index_t source_node = min(max(0, FLAGS_source_node), context.host_graph.nnodes - 1);

                auto partitioner = graph_manager.GetGraphPartitioner();
                if (partitioner->NeedsReverseLookup())
                {
                    source_node = partitioner->GetReverseLookupFunc()(source_node);
                }

                // Report the initial work
                distributed_worklist.ReportHighPrioWork(1, 0, "Host", groute::Device::Host, true);

                std::vector<remote_work_t> initial_work;
                initial_work.push_back(remote_work_t(source_node, 0));

                groute::router::ISender<remote_work_t>* work_sender = worklist_router.GetSender(groute::Device::Host);
                work_sender->Send(
                    groute::Segment<remote_work_t>(&initial_work[0], 1), groute::Event());
                work_sender->Shutdown();
            }

            template<typename TGraphAllocator, typename TGraphDatum, typename...UnusedData>
            static std::vector<level_t> Gather(TGraphAllocator& graph_allocator, TGraphDatum& levels_datum, UnusedData&... data)
            {
                graph_allocator.GatherDatum(levels_datum);
                return levels_datum.GetHostData();
            }

            template<typename...UnusedData>
            static std::vector<level_t> Host(groute::graphs::host::CSRGraph& graph, UnusedData&... data)
            {
                return BFSHost(graph, min(max(0, FLAGS_source_node), graph.nnodes - 1));
            }

            static int Output(const char *file, const std::vector<level_t>& levels)
            {
                return BFSOutput(file, levels);
            }

            static int CheckErrors(const std::vector<level_t>& levels, const std::vector<level_t>& regression)
            {
                return BFSCheckErrors(levels, regression);
            }
        };
    }
}

bool TestBFSAsyncMultiOptimized(int ngpus)
{
    typedef bfs::opt::FusedProblem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum<level_t>> ProblemType;
    typedef groute::graphs::traversal::FusedSolver<
        bfs::opt::Algo, ProblemType, 
        bfs::opt::local_work_t , bfs::opt::remote_work_t, level_t, 
        bfs::opt::SplitOps, 
        groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum<level_t>> SolverType;

    groute::graphs::traversal::__MultiRunner__Opt__ <
        bfs::opt::Algo,
        ProblemType,
        SolverType,
        bfs::opt::SplitOps,
        bfs::opt::local_work_t,
        bfs::opt::remote_work_t,
        groute::graphs::multi::NodeOutputGlobalDatum<level_t> > runner;

    groute::graphs::multi::NodeOutputGlobalDatum<level_t> levels_datum;
    
    bool retval = runner(ngpus, levels_datum);
    if(FLAGS_exitonerror && !retval)
        exit(100);
    return retval;
}
