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
#include <groute/worklist/distributed_worklist.cu.h>
#include <groute/worklist/work_kernels.cu.h>
#include <groute/device/cta_scheduler.cu.h>

#include <groute/graphs/csr_graph.h>
#include <groute/worklist/workers.cu.h>

#include <utils/graphs/traversal.h>

#include "bfs_common.h"

DEFINE_int32(source_node, 0, "The source node for the BFS traversal (clamped to [0, nnodes-1])");
const level_t INF = UINT_MAX;

namespace bfs {

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

    template<bool CTAScheduling = true> 
    // BFS work with Collective Thread Array scheduling for exploiting nested parallelism 
    struct BFSWork  
    {
        template<typename WorkSource, typename WorkTarget, typename TGraph, typename TGraphDatum>
        __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
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
                    [&work_target, &graph, &levels_datum](index_t edge, level_t next_level)
                {
                    index_t dest = graph.edge_dest(edge);
                    if (next_level < atomicMin(levels_datum.get_item_ptr(dest), next_level))
                    {
                        work_target.append_work(LevelData(dest, next_level));
                    }
                }
                );
            }
        }
    };

    // BFS work without CTA support
    template<>
    struct BFSWork<false> 
    {
        template<typename WorkSource, typename WorkTarget, typename TGraph, typename TGraphDatum>
        __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
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
                        work_target.append_work(LevelData(dest, next_level));
                    }
                }
            }
        }
    };

    struct DWCallbacks
    {
    private:
        groute::graphs::dev::CSRGraphSeg m_graph_seg;
        groute::graphs::dev::GraphDatum<level_t> m_levels_datum;

    public:
        template<typename...UnusedData>
        DWCallbacks(const groute::graphs::dev::CSRGraphSeg& graph_seg, const groute::graphs::dev::GraphDatum<level_t>& levels_datum, UnusedData&... data)
            : m_graph_seg(graph_seg), m_levels_datum(levels_datum)
        {
        }

        DWCallbacks() { }

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

        __device__ __forceinline__ bool should_defer(const local_work_t& work, const level_t& global_threshold)
        {
            return m_levels_datum[work] > global_threshold;
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

    struct Algo
    {
        static const char* NameLower()      { return "bfs"; }
        static const char* Name()           { return "BFS"; }

        static void Init(
            groute::graphs::traversal::Context<bfs::Algo>& context,
            groute::graphs::multi::CSRGraphAllocator& graph_manager,
            groute::IDistributedWorklist<local_work_t, remote_work_t>& distributed_worklist)
        {
            index_t source_node = min(max((index_t)0, (index_t)FLAGS_source_node), context.host_graph.nnodes - 1);

            auto partitioner = graph_manager.GetGraphPartitioner();
            if (partitioner->NeedsReverseLookup())
            {
                source_node = partitioner->GetReverseLookupFunc()(source_node);
            }

            // Host endpoint for sending initial work  
            groute::Endpoint host = groute::Endpoint::HostEndpoint(0);

            // Report the initial work
            distributed_worklist.ReportWork(1, 0, Name(), host, true);

            std::vector<remote_work_t> initial_work;
            initial_work.push_back(remote_work_t(source_node, 0));
            distributed_worklist
                .GetLink(host)
                .Send(groute::Segment<remote_work_t>(&initial_work[0], 1), groute::Event());
        }

        template<typename TGraph, typename TGraphDatum>
        static void DeviceInit(groute::Stream& stream, TGraph graph, TGraphDatum levels_datum)
        {

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, levels_datum.size);

            BFSInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                levels_datum.data_ptr, levels_datum.size);
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
            return BFSHost(graph, min(max((index_t)0, (index_t)FLAGS_source_node), graph.nnodes - 1));
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

    template<bool IterationFusion = true, bool CTAScheduling = true>
    using FusedWorkerType = groute::FusedWorker <
        IterationFusion, local_work_t, remote_work_t, int, DWCallbacks, BFSWork<CTAScheduling>,
        groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum < level_t >> ;
    
    template<bool CTAScheduling = true>
    using WorkerType = groute::Worker <
        local_work_t, remote_work_t, DWCallbacks, BFSWork<CTAScheduling>,
        groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatum < level_t >> ;

    template<typename TWorker>
    using RunnerType = groute::graphs::traversal::Runner <
        Algo, TWorker, DWCallbacks, local_work_t, remote_work_t,
        groute::graphs::multi::NodeOutputGlobalDatum<level_t> > ;
}

template<typename TWorker>
bool TestBFSAsyncMultiTemplate(int ngpus)
{
    bfs::RunnerType<TWorker> runner;
    groute::graphs::multi::NodeOutputGlobalDatum<level_t> levels_datum;
    return runner(ngpus, FLAGS_prio_delta, levels_datum);
}

bool TestBFSAsyncMultiOptimized(int ngpus)
{
    return FLAGS_cta_np
        ? FLAGS_iteration_fusion
            ? TestBFSAsyncMultiTemplate< bfs::FusedWorkerType< true, true   >>(ngpus)
            : TestBFSAsyncMultiTemplate< bfs::FusedWorkerType< false, true  >>(ngpus)
        : FLAGS_iteration_fusion                               
            ? TestBFSAsyncMultiTemplate< bfs::FusedWorkerType< true, false  >>(ngpus)
            : TestBFSAsyncMultiTemplate< bfs::FusedWorkerType< false, false >>(ngpus);
}

bool TestBFSAsyncMulti(int ngpus)
{
    return FLAGS_cta_np
        ? TestBFSAsyncMultiTemplate< bfs::WorkerType< true  >>(ngpus)
        : TestBFSAsyncMultiTemplate< bfs::WorkerType< false >>(ngpus);
}

bool TestBFSSingle()
{
    return TestBFSAsyncMultiOptimized(1);
}
