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

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include <groute/graphs/csr_graph.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/cta_work.h>

#include "bfs_common.h"

DEFINE_int32(source_node, 0, "The source node for the BFS traversal (clamped to [0, nnodes-1])");

const level_t INF = UINT_MAX;

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)


namespace bfs
{
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

    __global__ void BFSInit(level_t* levels, int nnodes, index_t source)
    {
        int tid = GTID;
        if (tid < nnodes)
        {
            levels[tid] = tid == source ? 0 : INF;
        }
    }



    // -------------- Nested Parallelism kernel ----------------

    template<
        typename TGraph,
        typename TGraphDatum,
        typename WorkTarget>
    struct BFSEdgeOperation
    {
        TGraph&         graph;
        TGraphDatum&    levels_datum;
        WorkTarget&     work_target;

        __device__ __forceinline__ BFSEdgeOperation(TGraph& graph, TGraphDatum& levels_datum, WorkTarget& work_target) :
            graph(graph), levels_datum(levels_datum), work_target(work_target)
        {

        }

        __device__ __forceinline__ void operator()(index_t edge, level_t next_level)
        {
            index_t dest = graph.edge_dest(edge);
            if (next_level < atomicMin(levels_datum.get_item_ptr(dest), next_level))
            {
                work_target.append_work(graph, dest);
            }
        }
    };

   /*
    * Optimized Nested Parallelism version of the BFS kernel
    */
    template<
        typename TGraph,
        typename TGraphDatum,
        typename WorkSource, typename WorkTarget>
        __global__ void BFSKernel__NestedParallelism__(TGraph graph, TGraphDatum levels_datum, WorkSource work_source, WorkTarget work_target)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

#ifndef NP_LAMBDA
        BFSEdgeOperation<TGraph, TGraphDatum, WorkTarget> op(graph, levels_datum, work_target);
#endif

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

#ifdef NP_LAMBDA
            groute::dev::CTAWorkScheduler<level_t>::template schedule(
                np_local, 
                [&graph, &levels_datum, &work_target](index_t edge, level_t next_level)
                {
                    index_t dest = graph.edge_dest(edge);
                    if (next_level < atomicMin(levels_datum.get_item_ptr(dest), next_level))
                    {
                        work_target.append_work(graph, dest);
                    }
                }
                );      
#else
            groute::dev::CTAWorkScheduler<level_t>::schedule(np_local, op);
#endif
        }
    }

    // -------------- Nested Parallelism end ----------------




    template<
        typename TGraph,
        typename TGraphDatum,
        typename WorkSource, typename WorkTarget>
        __global__ void BFSKernel(TGraph graph, TGraphDatum levels_datum, WorkSource work_source, WorkTarget work_target)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

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
                    work_target.append_work(graph, dest);
                }
            }
        }
    }

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

        __device__ __forceinline__ groute::SplitFlags on_receive(const remote_work_t& work)
        {
            if (m_graph_seg.owns(work.node))
            {
                return (work.level < atomicMin(m_levels_datum.get_item_ptr(work.node), work.level))
                    ? groute::SF_Take
                    : groute::SF_None; // filter
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
            return LevelData(work, m_levels_datum.get_item(work));
        }

        __device__ __forceinline__ local_work_t unpack(const remote_work_t& work)
        {
            return work.node;
        }
    };

    /*
    * @brief A per device BFS problem
    */
    template<typename TGraph, typename TGraphDatum>
    class Problem
    {
    private:
        TGraph m_graph;
        TGraphDatum m_levels_datum;

    public:
        Problem(const TGraph& graph, const TGraphDatum& levels_datum) :
            m_graph(graph), m_levels_datum(levels_datum)
        {
        }

        void Init(groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_levels_datum.size);

            Marker::MarkWorkitems(m_levels_datum.size, "BFSInit");
            BFSInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_levels_datum.data_ptr, m_levels_datum.size);
        }

        void Init(groute::Worklist<index_t>& in_wl, groute::Stream& stream) const
        {
            index_t source_node = min(max(0, FLAGS_source_node), m_graph.nnodes - 1);

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_levels_datum.size);

            Marker::MarkWorkitems(m_levels_datum.size, "BFSInit");

            BFSInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_levels_datum.data_ptr, m_levels_datum.size, source_node);

            in_wl.AppendItemAsync(stream.cuda_stream, source_node); // add the first item to the worklist
        }

        template<typename TWorklist, bool WarpAppend = true>
        void Relax(const groute::Segment<index_t>& work, TWorklist& output_worklist, groute::Stream& stream)
        {
            if (work.Empty()) return;

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "BFSKernel__NestedParallelism__");
                BFSKernel__NestedParallelism__
                    <TGraph, TGraphDatum, groute::dev::WorkSourceArray<index_t>, WorkTargetWorklist>
                    <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_graph, m_levels_datum,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    WorkTargetWorklist(output_worklist)
                    );
            }
            else
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "BFSKernel");
                BFSKernel
                    <TGraph, TGraphDatum, groute::dev::WorkSourceArray<index_t>, WorkTargetWorklist>
                    <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_graph, m_levels_datum,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    WorkTargetWorklist(output_worklist)
                    );
            }
        }
    };

    struct Algo
    {
        static const char* NameLower()      { return "bfs"; }
        static const char* Name()           { return "BFS"; }

        static void Init(
            groute::graphs::traversal::Context<bfs::Algo>& context,
            groute::graphs::multi::CSRGraphAllocator& graph_manager,
            groute::router::Router<remote_work_t>& worklist_router,
            groute::DistributedWorklist<local_work_t, remote_work_t>& distributed_worklist)
        {
            index_t source_node = min(max(0, FLAGS_source_node), context.host_graph.nnodes - 1);

            auto partitioner = graph_manager.GetGraphPartitioner();
            if (partitioner->NeedsReverseLookup())
            {
                source_node = partitioner->GetReverseLookupFunc()(source_node);
            }

            // report the initial work
            distributed_worklist.ReportWork(1);

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

bool TestBFSAsyncMulti(int ngpus)
{
    typedef bfs::Problem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum<level_t>> Problem;

    groute::graphs::traversal::__MultiRunner__ <
        bfs::Algo,
        Problem,
        groute::graphs::traversal::__GenericMultiSolver__<bfs::Algo, Problem, bfs::local_work_t, bfs::remote_work_t>,
        bfs::SplitOps,
        bfs::local_work_t,
        bfs::remote_work_t,
        groute::graphs::multi::NodeOutputGlobalDatum<level_t> > runner;

    groute::graphs::multi::NodeOutputGlobalDatum<level_t> levels_datum;

    return runner(ngpus, levels_datum);
}

bool TestBFSSingle()
{
    groute::graphs::traversal::__SingleRunner__ <
        bfs::Algo,
        bfs::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum<level_t>>,
        groute::graphs::single::NodeOutputDatum<level_t> > runner;

    groute::graphs::single::NodeOutputDatum<level_t> levels_datum;

    return runner(levels_datum);
}
