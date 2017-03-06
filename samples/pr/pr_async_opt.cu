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

#include <groute/device/cta_scheduler.cuh>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/distributed_worklist.cuh>
#include <groute/dwl/workers.cuh>

#include <utils/graphs/traversal.h>

#include "pr_common.h"



namespace pr {

    struct RankData
    {
        index_t node;
        rank_t rank;

        __host__ __device__ __forceinline__ RankData(index_t node, rank_t rank) : node(node), rank(rank) { }
        __host__ __device__ __forceinline__ RankData() : node(UINT_MAX), rank(-1.0f) { }
    };

    typedef index_t local_work_t;
    typedef RankData remote_work_t;

    template<typename TGraph, typename ResidualDatum>
    __global__ void PageRankInitKernel(TGraph graph, ResidualDatum residual)
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        index_t start_node = graph.owned_start_node();
        index_t end_node = start_node + graph.owned_nnodes();

        for (index_t node = start_node + tid; node < end_node; node += nthreads)
        {
            residual[node] = 1.0 - ALPHA;
        }
    }

    /*
    ---- PageRank Algorithm (data-driven, push-based) ----

    Based on Algorithm 4. in http://www.cs.utexas.edu/~inderjit/public_papers/scalable_pagerank_europar15.pdf
    Terminology: T[v]: All outgoing neighbors of node v
                 S[v]: All incoming neighbors of node v

    Input: graph G = (V, E), alpha, epsilon
    Output: PageRank x
    
    Initialize x : x[i] = 1 - alpha                         // Ranks vector
    Initialize r : r[i] = 0                                 // Residual vector

    // Initialization round
    for v in V:
        for w in T[v]:                                      // We perform also the initialization round with push-style, unlike the paper 
            r[w] = r[w] + ( (1-alpha)*alpha / |T[v]| )      // This line fuses lines 5 and 7 in the paper (maybe with some numeric cost)
        worklist.push(v)

    // Relaxation
    while worklist.not_empty
        v = worklist.pop()
        x[v] = x[v] + r[v]
        for w in T[v]:  
            o = r[w]
            r[w] = r[w] + r[v]*alpha/|T[v]|
            if r[w] >= epsilon and o < epsilon
                worklist.push(w)
        r[v] = 0

    // L1 Normalization
    x = x / ||x||_1
    */

    //template<
    //    typename TGraph, typename RankDatum, typename ResidualDatum,  typename WorkTarget>
    //__global__ void PageRankInitKernel(
    //    TGraph graph, RankDatum current_ranks, ResidualDatum residual, WorkTarget remote_work_target)
    //{
    //    unsigned tid = TID_1D;
    //    unsigned nthreads = TOTAL_THREADS_1D;

    //    index_t start_node = graph.owned_start_node();
    //    index_t end_node = start_node + graph.owned_nnodes();

    //    for (index_t node = start_node + tid; node < end_node; node += nthreads)
    //    {
    //        current_ranks[node] = 1.0 - ALPHA;

    //        index_t
    //            begin_edge = graph.begin_edge(node),
    //            end_edge = graph.end_edge(node),
    //            out_degree = end_edge - begin_edge;

    //        if (out_degree == 0) continue;

    //        rank_t update = ((1.0 - ALPHA) * ALPHA) / out_degree;

    //        for (index_t edge = begin_edge; edge < end_edge; ++edge)
    //        {
    //            index_t dest = graph.edge_dest(edge);
    //            if( atomicAdd(residual.get_item_ptr(dest), update) == 0 )
    //            {
    //                if (!graph.owns(dest)) // Push only remote nodes into the worklist since we process all 
    //                {
    //                    remote_work_target.append_work(dest);
    //                }
    //            }
    //        }
    //    }
    //}

    template<bool CTAScheduling = true> 
    /// PR work with Collective Thread Array scheduling for exploiting nested parallelism 
    struct PageRankWork
    {
        template<
            typename WorkSource, typename WorkTarget, 
            typename TGraph, typename ResidualDatum, typename RankDatum>
        __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
            const TGraph& graph, ResidualDatum& residual, RankDatum& current_ranks
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

                        if (np_local.size > 0) // Skip zero-degree nodes 
                        {
                            rank_t update = res * ALPHA / np_local.size;
                            np_local.meta_data = update;
                        }
                    }
                }

                groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    np_local,
                    [&work_target, &graph, &residual](index_t edge, rank_t update)
                    {
                        index_t dest = graph.edge_dest(edge);
                        rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                        // The EPSILON test must be decided by the owner, so if 
                        // dest belongs to another device the threshold is 0
                        rank_t threshold = graph.owns(dest) ? EPSILON : 0;

                        if (prev <= threshold && prev + update > threshold)
                        {
                            work_target.append_work(dest);
                        }
                    }
                );
            }
        }
    };
       
    template<>
    /// PR work without CTA support
    struct PageRankWork< false >
    {
        template<
            typename WorkSource, typename WorkTarget, 
            typename TGraph, typename ResidualDatum, typename RankDatum>
        __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
            const TGraph& graph, ResidualDatum& residual, RankDatum& current_ranks
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

                    // The EPSILON test must be decided by the owner, so if 
                    // dest belongs to another device the threshold is 0
                    rank_t threshold = graph.owns(dest) ? EPSILON : 0;

                    if (prev <= threshold && prev + update > threshold)
                    {
                        work_target.append_work(dest);
                    }
                }
            }
        }
    };

    /*
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
            if (remote_work_count > 0) groute::dev::Signal::Increase(send_signal_ptr, remote_work_count);

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
            if (remote_work_count > 0) groute::dev::Signal::Increase(send_signal_ptr, remote_work_count);

            __threadfence();
            // Report work
            *host_high_work_counter = rwl_in.get_start_diff(prev_start) - graph.owned_nnodes();
            *host_low_work_counter = 0;
        }
    }
    */

    struct DWCallbacks
    {
    private:
        groute::graphs::dev::CSRGraphSeg m_graph_seg;
        groute::graphs::dev::GraphDatum<rank_t> m_residual;

    public:
        template<typename...UnusedData>
        DWCallbacks(
            const groute::graphs::dev::CSRGraphSeg& graph_seg,
            const groute::graphs::dev::GraphDatum<rank_t>& residual,
            const groute::graphs::dev::GraphDatumSeg<rank_t>& current_ranks,
            UnusedData&... data)
            :
            m_graph_seg(graph_seg),
            m_residual(residual)
        {
        }

        DWCallbacks(
            const groute::graphs::dev::CSRGraphSeg& graph_seg,
            const groute::graphs::dev::GraphDatum<rank_t>& residual)
            :
            m_graph_seg(graph_seg),
            m_residual(residual)
        {
        }

        DWCallbacks() { }

        __device__ __forceinline__ groute::SplitFlags on_receive(const remote_work_t& work)
        {
            if (m_graph_seg.owns(work.node))
            {
                rank_t prev = atomicAdd(m_residual.get_item_ptr(work.node), work.rank);
                return (prev + work.rank > EPSILON && prev <= EPSILON)
                    ? groute::SF_Take
                    : groute::SF_None;
            }

            return groute::SF_Pass;
        }

        __device__ __forceinline__ bool should_defer(const local_work_t& work, const rank_t& global_threshold)
        {
            return false; // TODO: How can soft-priority be helpfull for PR?
        }

        __device__ __forceinline__ groute::SplitFlags on_send(local_work_t work)
        {
            return (m_graph_seg.owns(work))
                ? groute::SF_Take
                : groute::SF_Pass;
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

    struct Algo
    {
        static const char* NameLower()      { return "pr"; }
        static const char* Name()           { return "PR"; }

        static void HostInit(
            utils::traversal::Context<pr::Algo>& context,
            groute::graphs::multi::CSRGraphAllocator& graph_manager,
            groute::IDistributedWorklist<local_work_t, remote_work_t>& distributed_worklist)
        {
            // PR starts with all nodes
            distributed_worklist.ReportInitialWork(context.host_graph.nnodes, groute::Endpoint::HostEndpoint(0)); 
        }

        template<typename TGraph, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static void DeviceMemset(groute::Stream& stream, TGraph graph, ResidualDatum residual, RankDatum ranks)
        {
            GROUTE_CUDA_CHECK(
                cudaMemsetAsync(residual.data_ptr, 0, residual.size * sizeof(rank_t), stream.cuda_stream));
            GROUTE_CUDA_CHECK(
                cudaMemsetAsync(ranks.data_ptr, 0, ranks.size * sizeof(rank_t), stream.cuda_stream));
        }

        template<typename TGraph, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static void DeviceInit(
            groute::Endpoint endpoint, groute::Stream& stream, 
            groute::IDistributedWorklist<local_work_t, remote_work_t>& distributed_worklist, 
            groute::IDistributedWorklistPeer<local_work_t, remote_work_t, DWCallbacks>* peer, 
            TGraph graph, ResidualDatum residual, RankDatum ranks)
        {   
            auto& workspace = peer->GetLocalWorkspace(0);
            DWCallbacks callbacks = peer->GetDeviceCallbacks();

            dim3 grid_dims, block_dims;

            // Init residual values
            KernelSizing(grid_dims, block_dims, graph.owned_nnodes());
            PageRankInitKernel <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(graph, residual);

            KernelSizing(grid_dims, block_dims, graph.owned_nnodes());
            groute::WorkKernel <groute::dev::WorkSourceRange<index_t>, local_work_t, remote_work_t, DWCallbacks, PageRankWork<>, TGraph, ResidualDatum, RankDatum>

                <<< grid_dims, block_dims, 0, stream.cuda_stream >>> (

                    groute::dev::WorkSourceRange<index_t>(graph.owned_start_node(), graph.owned_nnodes()),
                    workspace.DeviceObject(),
                    callbacks,
                    graph, residual, ranks
                    );

            auto output_seg = workspace.ToSeg(stream);
            int work = output_seg.GetSegmentSize(); 
            peer->SplitSend(output_seg, stream); 

            workspace.ResetAsync(stream); 
            distributed_worklist.ReportWork(work, graph.owned_nnodes(), endpoint);

            ///

            //auto& input_worklist = peer->GetLocalInputWorklist();
            //auto& temp_worklist = peer->GetLocalWorkspace(0); // local output worklist
            //
            //m_problem.Init__Multi__(temp_worklist, stream);
            //
            //auto seg1 = temp_worklist.ToSeg(stream);

            //// report work
            //distributed_worklist.ReportWork((int)seg1.GetSegmentSize(), (int)m_problem.m_graph.owned_nnodes(), endpoint );

            //peer->SplitSend(seg1, stream); // call split-send
            //
            //temp_worklist.ResetAsync(stream.cuda_stream); // reset the temp output worklist

            //// First relax is a special case, starts from all owned nodes
            //m_problem.Relax__Multi__(
            //    groute::dev::WorkSourceRange<index_t>(
            //        m_problem.m_graph.owned_start_node(),
            //        m_problem.m_graph.owned_nnodes()),
            //        temp_worklist, stream);

            //auto seg2 = temp_worklist.ToSeg(stream);
            //
            //// report work
            //distributed_worklist.ReportWork((int)seg2.GetSegmentSize(), 0, endpoint);

            //peer->SplitSend(seg2, stream); // call split-send
            //temp_worklist.ResetAsync(stream.cuda_stream); // reset the temp output worklist
        }

        template<
            typename TGraphAllocator, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static const std::vector<rank_t>& Gather(
            TGraphAllocator& graph_allocator, ResidualDatum& residual, RankDatum& current_ranks, UnusedData&... data)
        {
            graph_allocator.GatherDatum(current_ranks);
            return current_ranks.GetHostData();
        }

        template<
            typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static std::vector<rank_t> Host(
            groute::graphs::host::CSRGraph& graph, ResidualDatum& residual, RankDatum& current_ranks, UnusedData&... data)
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

    using NodeResidualDatumType = groute::graphs::multi::NodeOutputGlobalDatum < rank_t > ;
    using NodeRankDatumType = groute::graphs::multi::NodeOutputLocalDatum < rank_t > ;

    template<bool IterationFusion = true, bool CTAScheduling = true>
    using FusedWorkerType = groute::FusedWorker <
        IterationFusion, local_work_t, remote_work_t, int, DWCallbacks, PageRankWork<CTAScheduling>,
        groute::graphs::dev::CSRGraphSeg, NodeResidualDatumType::DeviceObjectType, NodeRankDatumType::DeviceObjectType> ;
    
    template<bool CTAScheduling = true>
    using WorkerType = groute::Worker <
        local_work_t, remote_work_t, DWCallbacks, PageRankWork<CTAScheduling>,
        groute::graphs::dev::CSRGraphSeg, NodeResidualDatumType::DeviceObjectType, NodeRankDatumType::DeviceObjectType> ;

    template<typename TWorker>
    using RunnerType = utils::traversal::Runner <
        Algo, TWorker, DWCallbacks, local_work_t, remote_work_t,
        NodeResidualDatumType, NodeRankDatumType > ;
}

template<typename TWorker>
bool TestPageRankAsyncMultiTemplate(int ngpus)
{
    pr::RunnerType<TWorker> runner;

    pr::NodeResidualDatumType residual;
    pr::NodeRankDatumType ranks;

    return runner(ngpus, 0, residual, ranks);
}

bool TestPageRankAsyncMultiOptimized(int ngpus)
{
    return FLAGS_cta_np
        ? FLAGS_iteration_fusion
            ? TestPageRankAsyncMultiTemplate< pr::FusedWorkerType< true, true   >>(ngpus)
            : TestPageRankAsyncMultiTemplate< pr::FusedWorkerType< false, true  >>(ngpus)
        : FLAGS_iteration_fusion                               
            ? TestPageRankAsyncMultiTemplate< pr::FusedWorkerType< true, false  >>(ngpus)
            : TestPageRankAsyncMultiTemplate< pr::FusedWorkerType< false, false >>(ngpus);
}

bool TestPageRankAsyncMulti(int ngpus)
{
    return FLAGS_cta_np
        ? TestPageRankAsyncMultiTemplate< pr::WorkerType< true  >>(ngpus)
        : TestPageRankAsyncMultiTemplate< pr::WorkerType< false >>(ngpus);
}

bool TestPageRankSingle()
{
    return TestPageRankAsyncMultiOptimized(1);
}
