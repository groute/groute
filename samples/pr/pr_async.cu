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
#include <groute/distributed_worklist.h>
#include <groute/cta_work.h>

#include <groute/graphs/csr_graph.h>
#include <groute/graphs/traversal_algo.h>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>

#include "pr_common.h"

DECLARE_int32(max_pr_iterations);
DECLARE_bool(verbose);

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)


namespace pr
{
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
        template <typename> class RankDatum, template <typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual)
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        index_t start_node = graph.owned_start_node();
        index_t end_node = start_node + graph.owned_nnodes();

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
                atomicAdd(residual.get_item_ptr(dest), update);
            }
        }
    }

    template<
        typename TGraph,
        template <typename> class RankDatum, 
        template <typename> class ResidualDatum, 
        typename WorkTarget>
    __global__ void PageRankInit__Multi__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual, 
        WorkTarget remote_work_target
        )
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        index_t start_node = graph.owned_start_node();
        index_t end_node = start_node + graph.owned_nnodes();

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
                if( atomicAdd(residual.get_item_ptr(dest), update) == 0 )
                {
                    if (!graph.owns(dest)) // 
                    {
                        remote_work_target.append_work(dest);
                    }
                }
            }
        }
    }

    template<
        typename TGraph,
        template <typename> class RankDatum, template <typename> class ResidualDatum,
        typename WorkSource,
        template <typename> class TWorklist>
    __global__ void PageRankKernel__Single__NestedParallelism__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
        WorkSource work_source, TWorklist<index_t> output_worklist
        )
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

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
                [&graph, &residual, &output_worklist](index_t edge, rank_t update)
                {
                    index_t dest = graph.edge_dest(edge);
                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);
                    if (prev + update > EPSILON && prev < EPSILON)
                    {
                        output_worklist.append_warp(dest);
                    }
                    }
                ); 
        }
    }

    template<
        typename TGraph,
        template <typename> class RankDatum, template <typename> class ResidualDatum,
        typename WorkSource,
        template <typename> class TWorklist>
    __global__ void PageRankKernel__Single__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual,
        WorkSource work_source, TWorklist<index_t> output_worklist
        )
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

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
                if (prev + update > EPSILON && prev < EPSILON)
                {
                    output_worklist.append_warp(dest);
                }
            }
        }
    }

    template<
        typename TGraph,
        template <typename> class RankDatum, 
        template <typename> class ResidualDatum,
        typename WorkSource,
        template <typename> class TWorklist, 
        typename WorkTarget>
    __global__ void PageRankKernel__Multi__NestedParallelism__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, 
        ResidualDatum<rank_t> residual,
        WorkSource work_source, 
        TWorklist<index_t> local_output_worklist,
        WorkTarget remote_work_target
        )
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

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
                [&graph, &residual, &local_output_worklist, &remote_work_target](index_t edge, rank_t update)
                {
                    index_t dest = graph.edge_dest(edge);
                    rank_t prev = atomicAdd(residual.get_item_ptr(dest), update);

                    if (graph.owns(dest))
                    {
                        if (prev + update > EPSILON && prev <= EPSILON)
                        {
                            local_output_worklist.append_warp(dest);
                        }
                    }

                    else
                    {
                        if (prev == 0) // no EPSILON check for remote nodes
                        {
                            remote_work_target.append_work(dest);
                        }
                    }
                }
                ); 
        }
    }

    template<
        typename TGraph,
        template <typename> class RankDatum, 
        template <typename> class ResidualDatum,
        typename WorkSource,
        template <typename> class TWorklist, 
        typename WorkTarget>
    __global__ void PageRankKernel__Multi__(
        TGraph graph,
        RankDatum<rank_t> current_ranks, 
        ResidualDatum<rank_t> residual,
        WorkSource work_source, 
        TWorklist<index_t> local_output_worklist,
        WorkTarget remote_work_target
        )
    {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

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
                        local_output_worklist.append_warp(dest);
                    }
                }

                else
                {
                    if (prev == 0) // no EPSILON check for remote nodes
                    {
                        remote_work_target.append_work(dest);
                    }
                }
            }
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

        __device__ __forceinline__ groute::SplitFlags on_receive(const remote_work_t& work)
        {
            if (m_graph_seg.owns(work.node))
            {
                rank_t prev = atomicAdd(m_residual.get_item_ptr(work.node), work.rank);
                return (prev + work.rank > EPSILON && prev < EPSILON) 
                    ? groute::SF_Take
                    : groute::SF_None;
            }

            return groute::SF_Pass;
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

    __global__ void PageRankPackKernel(
        groute::graphs::dev::GraphDatum<rank_t> residual,
        groute::graphs::dev::GraphDatum<index_t> halos_datum,
        groute::graphs::dev::GraphDatum<mark_t> halos_marks,
        groute::dev::CircularWorklist<RankData> remote_worklist)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (uint32_t i = 0 + tid; i < halos_datum.size; i += nthreads)
        {
            index_t halo_node = halos_datum[i]; // promised to be unique
            if (halos_marks[halo_node] == 1)
            {
                remote_worklist.append_warp(
                    RankData(halo_node, atomicExch(residual.get_item_ptr(halo_node), 0)));
                halos_marks[halo_node] = 0;
            }
        }
    }

    /*
    * The per-device Page Rank problem
    */
    template<
        typename TGraph,
        template <typename> class ResidualDatum, template <typename> class RankDatum>
    struct Problem
    {
        TGraph m_graph;
        ResidualDatum<rank_t> m_residual;
        RankDatum<rank_t> m_current_ranks;

        Problem(
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

        void Init__Single__(groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Single__");
            PageRankInit__Single__ <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                m_graph, m_current_ranks, m_residual);
        }
        
        void Init__Multi__(groute::Worklist<index_t>& output_worklist, groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_graph.owned_nnodes());

            Marker::MarkWorkitems(m_graph.owned_nnodes(), "PageRankInit__Multi__");
            PageRankInit__Multi__ <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                m_graph, m_current_ranks, m_residual, WorkTargetWorklist(output_worklist));
        }

        template<
            typename WorkSource,
            template <typename> class TWorklist>
        void Relax__Single__(const WorkSource& work_source, TWorklist<index_t>& output_worklist, groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__NestedParallelism__");
                PageRankKernel__Single__NestedParallelism__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_current_ranks, m_residual,
                    work_source,
                    output_worklist.DeviceObject());
            }
            else
            {
                Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Single__");
                PageRankKernel__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_current_ranks, m_residual,
                    work_source,
                    output_worklist.DeviceObject());
            }
        }

        template<
            typename WorkSource,
            template <typename> class TWorklist>
        void Relax__Multi__(const WorkSource& work_source, TWorklist<index_t>& output_worklist, groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Multi__NestedParallelism__");
                PageRankKernel__Multi__NestedParallelism__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_current_ranks, m_residual,
                    work_source,
                    output_worklist.DeviceObject(),
                    WorkTargetWorklist(output_worklist));
            }
            else
            {
                Marker::MarkWorkitems(work_source.get_size(), "PageRankKernel__Multi__");
                PageRankKernel__Multi__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_current_ranks, m_residual,
                    work_source,
                    output_worklist.DeviceObject(),
                    WorkTargetWorklist(output_worklist));
            }
        }
    };

    /*
    * The per-device Page Rank solver
    */
    template<
        typename TGraph,
        template <typename> class ResidualDatum, template <typename> class RankDatum>
    class Solver
    {
    public:
        typedef Problem<TGraph, ResidualDatum, RankDatum> ProblemType;

    private:
        ProblemType& m_problem;

    public:
        Solver(groute::Context& context, ProblemType& problem) : m_problem(problem) { }

        void Solve(
            groute::Context& context,
            groute::Endpoint endpoint,
            groute::DistributedWorklist<local_work_t, remote_work_t>& distributed_worklist,
            groute::IDistributedWorklistPeer<local_work_t, remote_work_t>* worklist_peer,
            groute::Stream& stream)
        {
            auto& input_worklist = worklist_peer->GetLocalInputWorklist();
            auto& temp_worklist = worklist_peer->GetTempWorklist(); // local output worklist
            
            m_problem.Init__Multi__(temp_worklist, stream);
            
            auto seg1 = temp_worklist.ToSeg(stream);

            // report work
            distributed_worklist.ReportWork(
                (int)seg1.GetSegmentSize(),
                (int)m_problem.m_graph.owned_nnodes(),
                "PR", endpoint
                );

            worklist_peer->PerformSplitSend(seg1, stream); // call split-send
            
            temp_worklist.ResetAsync(stream.cuda_stream); // reset the temp output worklist

            // First relax is a special case, starts from all owned nodes
            m_problem.Relax__Multi__(
                groute::dev::WorkSourceRange<index_t>(
                    m_problem.m_graph.owned_start_node(),
                    m_problem.m_graph.owned_nnodes()),
                    temp_worklist, stream);

            auto seg2 = temp_worklist.ToSeg(stream);
            
            // report work
            distributed_worklist.ReportWork(
                (int)seg2.GetSegmentSize(),
                0,
                "PR", endpoint
                );

            worklist_peer->PerformSplitSend(seg2, stream); // call split-send
            
            int iteration = 0;
            int async_iteration_factor = 3;
            // because of the asynchronicity of our model, and all the data exchanges, we must allow more iterations  
            int max_iterations = FLAGS_max_pr_iterations * async_iteration_factor;  

            while (distributed_worklist.HasWork() /*&& distributed_worklist.HasActivePeers()*/)
            {
                temp_worklist.ResetAsync(stream.cuda_stream); // reset the temp output worklist

                auto input_segs = worklist_peer->GetLocalWork(stream);
                size_t new_work = 0, performed_work = 0;

                if (input_segs.empty()) continue;

                // If circular buffer passed the end, 2 buffers will be given: [s,end) and [0,e)
                for (auto input_seg : input_segs)
                {
                    m_problem.Relax__Multi__(
                        groute::dev::WorkSourceArray<index_t>(input_seg.GetSegmentPtr(), input_seg.GetSegmentSize()),
                        temp_worklist, stream);

                    input_worklist.PopItemsAsync(input_seg.GetSegmentSize(), stream.cuda_stream);
                    performed_work += input_seg.GetSegmentSize();
                }

                auto output_seg = temp_worklist.ToSeg(stream);
                new_work = output_seg.GetSegmentSize(); // add the new work 

                // report work
                distributed_worklist.ReportWork(
                    (int)new_work,
                    (int)performed_work,
                    "PR", endpoint
                    );

                worklist_peer->PerformSplitSend(output_seg, stream); // call split-send 

                if (iteration++ == max_iterations) {
                    //distributed_worklist.ReportPeerTermination();
                    //terminated = true;
                    //
                    //printf("Endpoint %d is terminating after %d iterations (max: %d * %d)\n\n", dev, iteration-1, FLAGS_max_pr_iterations, async_iteration_factor);
                    //continue; // Go on until all devices report termination, and router shutsdown  
                }
            }

            if (FLAGS_verbose) {
                printf("Endpoint %d is exiting after %d iterations (max: %d * %d)\n\n", (groute::Endpoint::identity_type)endpoint, iteration, FLAGS_max_pr_iterations, async_iteration_factor);
            }
        }
    };

    struct Algo
    {
        static const char* NameLower()      { return "pr"; }
        static const char* Name()           { return "PR"; }

        static void Init(
            groute::graphs::traversal::Context<pr::Algo>& context,
            groute::graphs::multi::CSRGraphAllocator& graph_manager,
            groute::Link<remote_work_t>& input_link,
            groute::DistributedWorklist<local_work_t, remote_work_t>& distributed_worklist)
        {
            distributed_worklist.ReportWork(context.host_graph.nnodes); // PR starts with all nodes
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

bool TestPageRankAsyncMulti(int ngpus)
{
    typedef groute::graphs::multi::CSRGraphAllocator GraphAllocator;
    typedef groute::graphs::multi::NodeOutputGlobalDatum<rank_t> ResidualDatum;
    typedef groute::graphs::multi::NodeOutputLocalDatum<rank_t> RankDatum;
    
    typedef pr::Solver<GraphAllocator::DeviceObjectType, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatumSeg> SolverType;
    
    groute::graphs::traversal::__MultiRunner__ <
        pr::Algo,
        SolverType::ProblemType,
        SolverType,
        pr::SplitOps,
        pr::local_work_t,
        pr::remote_work_t,
        ResidualDatum, RankDatum > runner;
    
    ResidualDatum residual;
    RankDatum current_ranks;
    
    return runner(ngpus, residual, current_ranks);
}

bool TestPageRankSingle()
{
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;

    groute::graphs::traversal::Context<pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
        dev_graph_allocator(context.host_graph);
    
    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(residual, current_ranks);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device 

    pr::Problem<
        groute::graphs::dev::CSRGraph,
        groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>
        solver(
        dev_graph_allocator.DeviceObject(),
        residual.DeviceObject(),
        current_ranks.DeviceObject());

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;

    groute::Stream stream;

    groute::Worklist<index_t> wl1(max_work_size), wl2(max_work_size);

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    stream.Sync();

    Stopwatch sw(true);

    groute::Worklist<index_t>* in_wl = &wl1, *out_wl = &wl2;

    solver.Init__Single__(stream);
    
    // First relax is a special case, starts from all owned nodes
    solver.Relax__Single__( 
        groute::dev::WorkSourceRange<index_t>(
            dev_graph_allocator.DeviceObject().owned_start_node(), 
            dev_graph_allocator.DeviceObject().owned_nnodes()), 
            *in_wl, stream);

    groute::Segment<index_t> work_seg;
    work_seg = in_wl->ToSeg(stream);

    int iteration = 0;

    while (work_seg.GetSegmentSize() > 0)
    {
        solver.Relax__Single__(
            groute::dev::WorkSourceArray<index_t>(
                work_seg.GetSegmentPtr(), 
                work_seg.GetSegmentSize()), 
            *out_wl, stream);

        if (++iteration > FLAGS_max_pr_iterations) break;

        in_wl->ResetAsync(stream.cuda_stream);
        std::swap(in_wl, out_wl);
        work_seg = in_wl->ToSeg(stream);
    }

    sw.stop();

    if (FLAGS_repetitions > 1)
        printf("\nWarning: ignoring repetitions flag, running just one repetition (not implemented)\n");

    printf("\n%s: %f ms. <filter>\n\n", pr::Algo::Name(), sw.ms() / FLAGS_repetitions);
    printf("%s terminated after %d iterations (max: %d)\n\n", pr::Algo::Name(), iteration, FLAGS_max_pr_iterations);

    // Gather
    auto gathered_output = pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);

    if (FLAGS_output.length() != 0)
        pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = pr::Algo::Host(context.host_graph, residual, current_ranks);
        return pr::Algo::CheckErrors(gathered_output, regression) == 0;
    }
    else {
        printf("Warning: Result not checked\n");
        return true;
    }
}
