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

#include <cub/grid/grid_barrier.cuh>

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

#include "pr_common.h"



namespace pr {
    namespace opt {

        struct RankData
        {
            index_t node;
            rank_t rank;

            __host__ __device__ __forceinline__ RankData(index_t node, rank_t rank) : node(node), rank(rank) { }
            __host__ __device__ __forceinline__ RankData() : node(UINT_MAX), rank(-1.0f) { }
        };

        typedef index_t local_work_t;
        typedef RankData remote_work_t;

        template<
            typename TGraph,
            typename RankDatum,
            typename ResidualDatum>
        struct PageRankWorkNP
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph,
                RankDatum& current_ranks, ResidualDatum& residual
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                uint32_t work_size = work_source.get_size();
                uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<rank_t> np_local = { 0, 0, 0.0 };

                    if (i < work_size)
                    {
                        index_t node = work_source.get_work(i);
                        rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                        if (res > 0)
                        {
                            current_ranks[node] += res;

                            np_local.start = graph.begin_edge(node);
                            np_local.size = graph.end_edge(node) - np_local.start;

                            if (np_local.size > 0) // Just in case
                            {
                                rank_t update = res * ALPHA / np_local.size;
                                np_local.meta_data = update;
                            }
                        }
                    }

                    groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                        np_local, 
                        [&graph, &residual, &rwl_in, &rwl_out](index_t edge, rank_t update)
                        {
                            index_t dest = graph.edge_dest(edge);
                            rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                            if (graph.owns(dest))
                            {
                                if (prev + update > EPSILON && prev <= EPSILON)
                                {
                                    rwl_in.prepend_warp(dest);
                                }
                            }

                            else
                            {
                                if (prev == 0) // no EPSILON check for remote nodes
                                {
                                    rwl_out.append_warp(
                                        RankData(dest, atomicExch(residual.get_item_ptr(dest), 0)));
                                }
                            }
                        }
                        ); 
                }
            }
        };

        template<
            typename TGraph,
            typename RankDatum,
            typename ResidualDatum>
        struct PageRankWork
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph,
                RankDatum& current_ranks, ResidualDatum& residual
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                uint32_t work_size = work_source.get_size();

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);

                    rank_t res = atomicExch(residual.get_item_ptr(node), 0);
                    if (res == 0) continue; // might happen if work_source has duplicates  

                    current_ranks[node] += res;

                    index_t
                        begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                    if (out_degree == 0) continue;

                    rank_t update = res * ALPHA / out_degree;

                    for (index_t edge = begin_edge; edge < end_edge; ++edge)
                    {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        if (graph.owns(dest))
                        {
                            if (prev + update > EPSILON && prev <= EPSILON)
                            {
                                rwl_in.prepend_warp(dest);
                            }
                        }

                        else
                        {
                            if (prev == 0) // no EPSILON check for remote nodes
                            {
                                rwl_out.append_warp(
                                    RankData(dest, atomicExch(residual.get_item_ptr(dest), 0)));
                            }
                        }
                    }
                }
            }
        };

        template<
            typename TGraph,
            typename RankDatum,
            typename ResidualDatum>
            __global__ void PageRankFusedInit(TGraph graph,
            RankDatum current_ranks, ResidualDatum residual,
            groute::dev::CircularWorklist<local_work_t> rwl_in,   // prepending work here
            groute::dev::CircularWorklist<remote_work_t> rwl_out,  // appending work here
            volatile int*     host_high_work_counter,
            volatile int*     host_low_work_counter,
            volatile int *    send_signal_ptr,
            cub::GridBarrier gbar)
        {
            unsigned tid = TID_1D;
            unsigned nthreads = TOTAL_THREADS_1D;

            index_t start_node = graph.owned_start_node();
            index_t end_node = start_node + graph.owned_nnodes();

            // Do init step 1
            //
            for (index_t node = start_node + tid; node < end_node; node += nthreads)
            {
                current_ranks[node] = 1.0 - ALPHA;

                index_t
                    begin_edge = graph.begin_edge(node),
                    end_edge = graph.end_edge(node),
                    out_degree = end_edge - begin_edge;

                if (out_degree == 0) continue;

                rank_t update = ((1.0 - ALPHA) * ALPHA) / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; ++edge)
                {
                    index_t dest = graph.edge_dest(edge);

                    if (graph.owns(dest))
                    {
                        atomicAdd(residual.get_item_ptr(dest), update);
                    }
                    else // we only append remote nodes, since all owned nodes are processed at step 2
                    {
                        // Write directly to remote out without atomics
                        rwl_out.append_warp(RankData(dest, update));
                    }
                }
            }

            gbar.Sync();

            int prev_start;

            // Transmit work
            if (GTID == 0)
            {
                uint32_t remote_work_count = rwl_out.get_alloc_count_and_sync();
                if (remote_work_count > 0) IncreaseHostFlag(send_signal_ptr, remote_work_count);

                prev_start = rwl_in.get_start();
            }

            gbar.Sync();

            // Do init step 2
            //
            PageRankWork<TGraph, RankDatum, ResidualDatum>::work(
                groute::dev::WorkSourceRange<index_t>(
                graph.owned_start_node(),
                graph.owned_nnodes()),
                rwl_in, rwl_out,
                graph, current_ranks, residual
                );

            gbar.Sync();

            // Transmit and report work
            if (GTID == 0)
            {
                uint32_t remote_work_count = rwl_out.get_alloc_count_and_sync();
                if (remote_work_count > 0) IncreaseHostFlag(send_signal_ptr, remote_work_count);

                __threadfence();
                // Report work
                *host_high_work_counter = rwl_in.get_start_diff(prev_start) - graph.owned_nnodes();
                *host_low_work_counter = 0;
            }
        }


        struct SplitOps
        {
        private:
            groute::graphs::dev::CSRGraphSeg m_graph_seg;
            groute::graphs::dev::GraphDatum<rank_t> m_residual;

        public:
            template<typename...UnusedData>
            SplitOps(
                const groute::graphs::dev::CSRGraphSeg& graph_seg,
                const groute::graphs::dev::GraphDatum<rank_t>& residual,
                const groute::graphs::dev::GraphDatumSeg<rank_t>& current_ranks,
                UnusedData&... data)
                :
                m_graph_seg(graph_seg),
                m_residual(residual)
            {
            }

            SplitOps(
                const groute::graphs::dev::CSRGraphSeg& graph_seg,
                const groute::graphs::dev::GraphDatum<rank_t>& residual)
                :
                m_graph_seg(graph_seg),
                m_residual(residual)
            {
            }

            __device__ __forceinline__ groute::opt::SplitFlags on_receive(const remote_work_t& work)
            {
                if (m_graph_seg.owns(work.node))
                {
                    rank_t prev = atomicAdd(m_residual.get_item_ptr(work.node), work.rank);
                    return (prev + work.rank > EPSILON && prev < EPSILON)
                        ? groute::opt::SF_Take
                        : groute::opt::SF_None;
                }

                return groute::opt::SF_Pass;
            }

            __device__ __forceinline__ bool is_high_prio(const local_work_t& work, const rank_t& global_prio)
            {
                return true; // NOTE: Can soft-priority be supported for PR?
            }

            __device__ __forceinline__ groute::opt::SplitFlags on_send(local_work_t work)
            {
                return (m_graph_seg.owns(work))
                    ? groute::opt::SF_Take
                    : groute::opt::SF_Pass;
            }

            __device__ __forceinline__ remote_work_t pack(local_work_t work)
            {
                return RankData(work, atomicExch(m_residual.get_item_ptr(work), 0));
            }

            __device__ __forceinline__ local_work_t unpack(const remote_work_t& work)
            {
                return work.node;
            }
        };

        /*
        * The per-device Page Rank problem
        */
        template<
            typename TGraph,
            template <typename> class ResidualDatum, template <typename> class RankDatum>
        struct FusedProblem
        {
            TGraph m_graph;
            ResidualDatum<rank_t> m_residual;
            RankDatum<rank_t> m_current_ranks;

            typedef PageRankWork<TGraph, RankDatum<rank_t>, ResidualDatum<rank_t>> WorkType;
            typedef PageRankWorkNP<TGraph, RankDatum<rank_t>, ResidualDatum<rank_t>> WorkTypeNP;

            FusedProblem(
                const TGraph& graph,
                const ResidualDatum<rank_t>& residual,
                const RankDatum<rank_t>& current_ranks) :
                m_graph(graph), m_residual(residual), m_current_ranks(current_ranks)
            {
            }

            // Initial init. Called before a global CPU+GPU barrier
            void Init(groute::Stream& stream) const
            {
                GROUTE_CUDA_CHECK(
                    cudaMemsetAsync(
                    m_residual.data_ptr, 0,
                    m_residual.size * sizeof(rank_t),
                    stream.cuda_stream));
            }

            bool DoFusedInit(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, rank_t global_prio,
                volatile int *high_work_counter, volatile int *low_work_counter,
                uint32_t *kernel_internal_counter, volatile int *send_signal_ptr,
                cub::GridBarrierLifetime& barrier_lifetime,
                dim3 grid_dims, dim3 block_dims, groute::Stream& stream)
            {
                PageRankFusedInit <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (
                    m_graph, m_current_ranks, m_residual,
                    rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                    high_work_counter, low_work_counter,
                    send_signal_ptr,
                    barrier_lifetime
                    );

                return true;
            }

            void DoFusedWork(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, rank_t global_prio,
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
                            groute::NeverStop, local_work_t, remote_work_t, rank_t, SplitOps,
                            WorkTypeNP,
                            TGraph, RankDatum<rank_t>, ResidualDatum<rank_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            pr::opt::SplitOps(m_graph, m_residual),
                            m_graph, m_current_ranks, m_residual
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::NeverStop, local_work_t, remote_work_t, rank_t, SplitOps,
                            WorkType,
                            TGraph, RankDatum<rank_t>, ResidualDatum<rank_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            pr::opt::SplitOps(m_graph, m_residual),
                            m_graph, m_current_ranks, m_residual
                            );
                    }
                }

                else
                {
                    if (FLAGS_cta_np)
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, rank_t, SplitOps,
                            WorkTypeNP,
                            TGraph, RankDatum<rank_t>, ResidualDatum<rank_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            pr::opt::SplitOps(m_graph, m_residual),
                            m_graph, m_current_ranks, m_residual
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, rank_t, SplitOps,
                            WorkType,
                            TGraph, RankDatum<rank_t>, ResidualDatum<rank_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            pr::opt::SplitOps(m_graph, m_residual),
                            m_graph, m_current_ranks, m_residual
                            );
                    }
                }
            }
        };

        struct Algo
        {
            static const char* NameLower()      { return "pr"; }
            static const char* Name()           { return "PR"; }

            static void Init(
                groute::graphs::traversal::Context<pr::opt::Algo>& context,
                groute::graphs::multi::CSRGraphAllocator& graph_manager,
                groute::router::Router<remote_work_t>& worklist_router,
                groute::opt::DistributedWorklist<local_work_t, remote_work_t, SplitOps>& distributed_worklist)
            {
                distributed_worklist.ReportHighPrioWork(context.host_graph.nnodes, 0, "Host", groute::Device::Host, true); // PR starts with all nodes
            }

            template<
                typename TGraphAllocator,
                template <typename> class ResidualDatum, template <typename> class RankDatum,
                typename...UnusedData>
                static std::vector<rank_t> Gather(
                TGraphAllocator& graph_allocator,
                ResidualDatum<rank_t>& residual,
                RankDatum<rank_t>& current_ranks,
                UnusedData&... data)
            {
                graph_allocator.GatherDatum(current_ranks);
                return current_ranks.GetHostData();
            }

            template<
                template <typename> class ResidualDatum, template <typename> class RankDatum,
                typename...UnusedData>
                static std::vector<rank_t> Host(
                groute::graphs::host::CSRGraph& graph,
                ResidualDatum<rank_t>& residual,
                RankDatum<rank_t>& current_ranks,
                UnusedData&... data)
            {
                return PageRankHost(graph);
            }

            static int Output(const char *file, const std::vector<rank_t>& ranks)
            {
                return PageRankOutput(file, ranks);
            }

            static int CheckErrors(std::vector<rank_t>& ranks, std::vector<rank_t>& regression)
            {
                return PageRankCheckErrors(ranks, regression);
            }
        };
    }
}

bool TestPageRankAsyncMultiOptimized(int ngpus)
{
    typedef groute::graphs::multi::CSRGraphAllocator GraphAllocator;
    typedef groute::graphs::multi::NodeOutputGlobalDatum<rank_t> ResidualDatum;
    typedef groute::graphs::multi::NodeOutputLocalDatum<rank_t> RankDatum;
    
    typedef pr::opt::FusedProblem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatumSeg> ProblemType;
    typedef groute::graphs::traversal::FusedSolver<
        pr::opt::Algo, ProblemType, 
        pr::opt::local_work_t , pr::opt::remote_work_t, rank_t, 
        pr::opt::SplitOps, 
        groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg<rank_t>, groute::graphs::dev::GraphDatum<rank_t>> SolverType;
    
    groute::graphs::traversal::__MultiRunner__Opt__ <
        pr::opt::Algo,
        ProblemType,
        SolverType,
        pr::opt::SplitOps,
        pr::opt::local_work_t,
        pr::opt::remote_work_t,
        ResidualDatum, RankDatum > runner;
    
    ResidualDatum residual;
    RankDatum current_ranks;
    
    return runner(ngpus, residual, current_ranks);
}
