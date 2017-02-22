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

#include "sssp_common.h"

const distance_t INF = UINT_MAX;

DEFINE_int32(source_node, 0, "The source node for the SSSP traversal (clamped to [0, nnodes-1])");

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)


namespace sssp
{
    struct DistanceData
    {
        index_t node;
        distance_t distance;

        __device__ __host__ __forceinline__ DistanceData(index_t node, distance_t distance) : node(node), distance(distance) { }
        __device__ __host__ __forceinline__ DistanceData() : node(INF), distance(INF) { }
    };

    typedef index_t local_work_t;
    typedef DistanceData remote_work_t;

    struct WorkTargetRemoteWorklist
    {
    private:
        groute::dev::CircularWorklist<remote_work_t> m_worklist;

    public:
        WorkTargetRemoteWorklist(groute::CircularWorklist<remote_work_t>& worklist) : m_worklist(worklist.DeviceObject()) { }

        __device__ __forceinline__ void append_work(const remote_work_t& work)
        {
            m_worklist.append_warp(work);
        }
    };

    struct WorkTargetDummy
    {

    public:
        WorkTargetDummy() { }

        __device__ __forceinline__ void append_work(const remote_work_t& work)
        {
        }
    };

    struct WorkTargetRemoteMark
    {
    private:
        groute::graphs::dev::GraphDatum<mark_t> m_remote_marks;
        groute::dev::Counter m_remote_counter;

    public:
        WorkTargetRemoteMark(
            groute::graphs::dev::GraphDatum<mark_t> remote_marks,
            groute::Counter& remote_counter) :
            m_remote_marks(remote_marks), m_remote_counter(remote_counter.DeviceObject())
        {

        }

        __device__ __forceinline__ void append_work(const remote_work_t& work)
        {
            if (m_remote_marks[work.node] == 0)
            {
                m_remote_marks[work.node] = 1; // mark
                m_remote_counter.add_one_warp();
            }
        }
    };

    __global__ void SSSPInit(distance_t* distances, int nnodes)
    {
        int tid = GTID;
        if (tid < nnodes)
        {
            distances[tid] = INF;
        }
    }

    __global__ void SSSPInit(distance_t* distances, int nnodes, index_t source)
    {
        int tid = GTID;
        if (tid < nnodes)
        {
            distances[tid] = tid == source ? 0 : INF;
        }
    }

    template<
        typename TGraph, 
        typename TWeightDatum, typename TDistanceDatum, 
        typename WorkSource, typename WorkTarget>
    __global__ void SSSPKernelCTA(
        TGraph graph, 
        TWeightDatum edge_weights, TDistanceDatum node_distances,
        WorkSource work_source, WorkTarget work_target)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x; // we want all threads in active blocks to enter the loop

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
        {
            groute::dev::np_local<distance_t> np_local = { 0, 0, 0 };

            if (i < work_size)
            {
                index_t node = work_source.get_work(i);
                np_local.start = graph.begin_edge(node);
                np_local.size = graph.end_edge(node) - np_local.start;
                np_local.meta_data = node_distances.get_item(node);
            }

            groute::dev::CTAWorkScheduler<distance_t>::template schedule(
                np_local, 
                [&graph, &edge_weights, &node_distances, &work_target](index_t edge, distance_t distance)
                {
                    index_t dest = graph.edge_dest(edge);
                    distance_t weight = edge_weights.get_item(edge);

                    if (distance + weight < atomicMin(node_distances.get_item_ptr(dest), distance + weight))
                    {
                        work_target.append_work(graph, dest);
                    }
                }
                ); 
        }
    }

    template<
        typename TGraph, 
        typename TWeightDatum, typename TDistanceDatum, 
        typename WorkSource, typename WorkTarget>
    __global__ void SSSPKernel(
        TGraph graph, 
        TWeightDatum edge_weights, TDistanceDatum node_distances,
        WorkSource work_source, WorkTarget work_target)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
        {
            index_t node = work_source.get_work(i);
            distance_t distance = node_distances.get_item(node);

            for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge)
            {
                index_t dest = graph.edge_dest(edge);
                distance_t weight = edge_weights.get_item(edge);

                if (distance + weight < atomicMin(node_distances.get_item_ptr(dest), distance + weight))
                {
                    work_target.append_work(graph, dest);
                }
            }
        }
    }

    struct DWCallbacks
    {
    private:
        groute::graphs::dev::CSRGraphSeg m_graph_seg;
        groute::graphs::dev::GraphDatum<distance_t> m_distances_datum;

    public:
        template<typename...UnusedData>
        DWCallbacks(
            const groute::graphs::dev::CSRGraphSeg& graph_seg, 
            const groute::graphs::dev::GraphDatumSeg<distance_t>& weights_datum, 
            const groute::graphs::dev::GraphDatum<distance_t>& distances_datum, 
            UnusedData&... data)
            : m_graph_seg(graph_seg), m_distances_datum(distances_datum)
        {
        }

        __device__ __forceinline__ groute::SplitFlags on_receive(const remote_work_t& work)
        {
            if (m_graph_seg.owns(work.node))
            {
                return (work.distance < atomicMin(m_distances_datum.get_item_ptr(work.node), work.distance))
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
            return DistanceData(work, m_distances_datum.get_item(work));
        }

        __device__ __forceinline__ local_work_t unpack(const remote_work_t& work)
        {
            return work.node;
        }
    };

    template<
        typename TGraph, 
        template <typename> class TWeightDatum, template <typename> class TDistanceDatum
        >
    struct Problem
    {
        TGraph m_graph;
        TWeightDatum<distance_t> m_weights_datum;
        TDistanceDatum<distance_t> m_distances_datum;

    public:
        Problem(const TGraph& graph, const TWeightDatum<distance_t>& weights_datum, const TDistanceDatum<distance_t>& distances_datum) :
            m_graph(graph), m_weights_datum(weights_datum), m_distances_datum(distances_datum)
        {
        }

        void Init(groute::Stream& stream) const
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_distances_datum.size);

            Marker::MarkWorkitems(m_distances_datum.size, "SSSPInit");

            SSSPInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_distances_datum.data_ptr, m_distances_datum.size);
        }      
        
        void Init(groute::Worklist<index_t>& in_wl, groute::Stream& stream) const
        {
            index_t source_node = min(max(0, FLAGS_source_node), m_graph.nnodes - 1);

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, m_distances_datum.size);

            Marker::MarkWorkitems(m_distances_datum.size, "SSSPInit");

            SSSPInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_distances_datum.data_ptr, m_distances_datum.size, source_node);
            
            in_wl.AppendItemAsync(stream.cuda_stream, source_node); // add the first item to the worklist
        }

        template<typename TWorklist, bool WarpAppend = true>
        void Relax(const groute::Segment<index_t>& work, TWorklist& output_worklist, groute::Stream& stream) const
        {
            if (work.Empty()) return;

            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernelCTA");
                SSSPKernelCTA << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    WorkTargetWorklist(output_worklist)
                    );
            }
            else
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernel");
                SSSPKernel << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    WorkTargetWorklist(output_worklist)
                    );
            }
        }
    };

    struct Algo
    {
        static const char* NameLower()      { return "sssp"; }
        static const char* Name()           { return "SSSP"; }

        static void Init(
            groute::graphs::traversal::Context<sssp::Algo>& context,
            groute::graphs::multi::CSRGraphAllocator& graph_manager,
            groute::Link<remote_work_t>& input_link,
            groute::DistributedWorklist<local_work_t, remote_work_t, DWCallbacks>& distributed_worklist)
        {
            index_t source_node = min(max((index_t)0, (index_t)FLAGS_source_node), context.host_graph.nnodes - 1);
            
            auto partitioner = graph_manager.GetGraphPartitioner();
            if (partitioner->NeedsReverseLookup())
            {
                source_node = partitioner->GetReverseLookupFunc()(source_node);
            }

            // Report the initial work
            distributed_worklist.ReportWork(1, 0, Name(), groute::Endpoint::HostEndpoint(0), true);

            std::vector<remote_work_t> initial_work;
            initial_work.push_back(remote_work_t(source_node, 0));
            input_link.Send(groute::Segment<remote_work_t>(&initial_work[0], 1), groute::Event());
        }

        template<
            typename TGraphAllocator, 
            template <typename> class TWeightDatum, template <typename> class TDistanceDatum, typename...UnusedData>
        static std::vector<distance_t> Gather(
            TGraphAllocator& graph_allocator, 
            TWeightDatum<distance_t>& weights_datum, TDistanceDatum<distance_t>& distances_datum, 
            UnusedData&... data)
        {
            graph_allocator.GatherDatum(distances_datum);
            return distances_datum.GetHostData();
        }

        template<
            template <typename> class TWeightDatum, 
            template <typename> class TDistanceDatum, 
            typename...UnusedData>
        static std::vector<distance_t> Host(
            groute::graphs::host::CSRGraph& graph, 
            TWeightDatum<distance_t>& weights_datum, TDistanceDatum<distance_t>& distances_datum, 
            UnusedData&... data)
        {
            return SSSPHostNaive(graph, weights_datum.GetHostDataPtr(), min( max((index_t)0, (index_t)FLAGS_source_node), graph.nnodes-1));
        }

        static int Output(const char *file, const std::vector<distance_t>& distances)
        {
            return SSSPOutput(file, distances);
        }

        static int CheckErrors(const std::vector<distance_t>& distances, const std::vector<distance_t>& regression)
        {
            return SSSPCheckErrors(distances, regression);
        }
    };
}

bool TestSSSPAsyncMulti(int ngpus)
{
    typedef sssp::Problem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg, groute::graphs::dev::GraphDatum> ProblemType;
    typedef groute::graphs::traversal::__MultiSolver__<sssp::Algo, ProblemType, sssp::DWCallbacks, sssp::local_work_t, sssp::remote_work_t> SolverType;
    
    groute::graphs::traversal::__MultiRunner__ <
        sssp::Algo,
        ProblemType,
        SolverType,
        sssp::DWCallbacks,
        sssp::local_work_t,
        sssp::remote_work_t,
        groute::graphs::multi::EdgeInputDatum<distance_t>,
        groute::graphs::multi::NodeOutputGlobalDatum<distance_t> > runner;
    
    groute::graphs::multi::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::multi::NodeOutputGlobalDatum<distance_t> node_distances;
    
    return runner(ngpus, 1, edge_weights, node_distances);
}

bool TestSSSPSingle()
{
    groute::graphs::traversal::__SingleRunner__ < 
        sssp::Algo, 
        sssp::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum>, 
        groute::graphs::single::EdgeInputDatum<distance_t>,
        groute::graphs::single::NodeOutputDatum<distance_t> > runner;
    
    groute::graphs::single::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::single::NodeOutputDatum<distance_t> node_distances;

    return runner(edge_weights, node_distances);
}
