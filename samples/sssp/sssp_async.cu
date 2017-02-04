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

DECLARE_bool(nf);
DEFINE_int32(nf_delta, 10000, "The delta for SSSP-nf");

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
    __global__ void SSSPKernel__NestedParallelism__(
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

    template<
        typename TGraph, 
        template <typename> class TWeightDatum, template <typename> class TDistanceDatum,
        typename WorkSource, typename WorkTarget>
    __global__ void SSSPKernel__NF__NestedParallelism__(
        TGraph graph, 
        TWeightDatum<distance_t> edge_weights, TDistanceDatum<distance_t> node_distances,
        int delta,
        WorkSource work_source, 
        groute::dev::Worklist<index_t> near_worklist, groute::dev::Worklist<index_t> far_worklist, 
        WorkTarget remote_work_target)
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
                [&graph, &edge_weights, &node_distances, &near_worklist, &far_worklist, &remote_work_target, delta](index_t edge, distance_t distance)
                {
                    index_t dest = graph.edge_dest(edge);
                    distance_t weight = edge_weights.get_item(edge);

                    if (distance + weight < atomicMin(node_distances.get_item_ptr(dest), distance + weight))
                    {
                        if (graph.owns(dest))
                        {
                            int near_mask = __ballot(distance + weight <= delta ? 1 : 0);
                            int far_mask = __ballot(distance + weight <= delta ? 0 : 1);

                            if (distance + weight <= delta)
                            {
                                int near_leader = __ffs(near_mask) - 1;
                                int thread_offset = __popc(near_mask & ((1 << lane_id()) - 1));
                                near_worklist.append_warp(dest, near_leader, __popc(near_mask), thread_offset);
                            }
                            else
                            {
                                int far_leader = __ffs(far_mask) - 1;
                                int thread_offset = __popc(far_mask & ((1 << lane_id()) - 1));
                                far_worklist.append_warp(dest, far_leader, __popc(far_mask), thread_offset);
                            }
                        }
                        else
                        {
                            remote_work_target.append_work(DistanceData(dest, distance + weight));
                        }
                    }
                }
                ); 
        }
    }

    template<
        typename TGraph, 
        template <typename> class TWeightDatum, template <typename> class TDistanceDatum,
        typename WorkSource, typename WorkTarget>
    __global__ void SSSPKernel__NF__(
        TGraph graph, 
        TWeightDatum<distance_t> edge_weights, TDistanceDatum<distance_t> node_distances,
        int delta,
        WorkSource work_source, 
        groute::dev::Worklist<index_t> near_worklist, groute::dev::Worklist<index_t> far_worklist, 
        WorkTarget remote_work_target)
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
                    if (graph.owns(dest))
                    {
                        int near_mask = __ballot(distance + weight <= delta ? 1 : 0);
                        int far_mask = __ballot(distance + weight <= delta ? 0 : 1);

                        if (distance + weight <= delta)
                        {
                            int near_leader = __ffs(near_mask) - 1;
                            int thread_offset = __popc(near_mask & ((1 << lane_id()) - 1));
                            near_worklist.append_warp(dest, near_leader, __popc(near_mask), thread_offset);
                        }
                        else
                        {
                            int far_leader = __ffs(far_mask) - 1;
                            int thread_offset = __popc(far_mask & ((1 << lane_id()) - 1));
                            far_worklist.append_warp(dest, far_leader, __popc(far_mask), thread_offset);
                        }
                    }
                    else
                    {
                        remote_work_target.append_work(DistanceData(dest, distance + weight));
                    }
                }
            }
        }
    }

    template<typename WorkSource>
    __global__ void SSSPNearFarSplit__NF__(
        int delta, 
        WorkSource work_source, 
        groute::graphs::dev::GraphDatum<distance_t> node_distances,
        groute::dev::Worklist<index_t> near_worklist, groute::dev::Worklist<index_t> far_worklist)
    {
        int tid = GTID;

        uint32_t work_size = work_source.get_size();

        if (tid < work_size)
        {
            index_t node = work_source.get_work(tid);
            distance_t distance = node_distances.get_item(node);

            int near_mask = __ballot(distance <= delta ? 1 : 0);
            int far_mask = __ballot(distance <= delta ? 0 : 1);

            if (distance <= delta)
            {
                int near_leader = __ffs(near_mask) - 1;
                int thread_offset = __popc(near_mask & ((1 << lane_id()) - 1));
                near_worklist.append_warp(node, near_leader, __popc(near_mask), thread_offset);
            }
            else
            {
                int far_leader = __ffs(far_mask) - 1;
                int thread_offset = __popc(far_mask & ((1 << lane_id()) - 1));
                far_worklist.append_warp(node, far_leader, __popc(far_mask), thread_offset);
            }
        }
    }

    struct SplitOps
    {
    private:
        groute::graphs::dev::CSRGraphSeg m_graph_seg;
        groute::graphs::dev::GraphDatum<distance_t> m_distances_datum;

    public:
        template<typename...UnusedData>
        SplitOps(
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
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernel__NestedParallelism__");
                SSSPKernel__NestedParallelism__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
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

        template< template <typename> class LocalWorklist, template <typename> class RemoteWorklist>
        void Relax__NF__(
            const groute::Segment<index_t>& work, int delta,
            LocalWorklist<index_t>& near_worklist, LocalWorklist<index_t>& far_worklist, 
            RemoteWorklist<DistanceData>& remote_worklist, groute::Stream& stream) const
        {
            if (work.Empty()) return;
            
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernel__NF__NestedParallelism__");
                SSSPKernel__NF__NestedParallelism__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum, delta,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    near_worklist.DeviceObject(), far_worklist.DeviceObject(),
                    WorkTargetRemoteWorklist(remote_worklist)
                    );
            }
            else
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernel__NF__");
                SSSPKernel__NF__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum, delta,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    near_worklist.DeviceObject(), far_worklist.DeviceObject(),
                    WorkTargetRemoteWorklist(remote_worklist)
                    );
            }
        }

        void RelaxSingle__NF__(
            const groute::Segment<index_t>& work, int delta,
            groute::Worklist<index_t>& near_worklist, groute::Worklist<index_t>& far_worklist,  
            groute::Stream& stream) const
        {
            if (work.Empty()) return;
            
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work.GetSegmentSize());

            if (FLAGS_cta_np)
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSPKernel__NF__NestedParallelism__ (single)");
                SSSPKernel__NF__NestedParallelism__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum, delta,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    near_worklist.DeviceObject(), far_worklist.DeviceObject(),
                    WorkTargetDummy()
                    );
            }
            else
            {
                Marker::MarkWorkitems(work.GetSegmentSize(), "SSSP-NF Relax (single)");
                SSSPKernel__NF__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_graph, m_weights_datum, m_distances_datum, delta,
                    groute::dev::WorkSourceArray<index_t>(work.GetSegmentPtr(), work.GetSegmentSize()),
                    near_worklist.DeviceObject(), far_worklist.DeviceObject(),
                    WorkTargetDummy()
                    );
            }
        }

        uint32_t SplitRemoteInput__NF__(
            const std::vector< groute::Segment<index_t> >& work_segs, int delta,
            groute::Worklist<index_t>& near_worklist, groute::Worklist<index_t>& far_worklist, 
            groute::Stream& stream) const
        {
            uint32_t work_size = 0;
            dim3 grid_dims, block_dims;

            switch (work_segs.size())
            {
            case 0: break;
            case 1:
                work_size = work_segs[0].GetSegmentSize();
                KernelSizing(grid_dims, block_dims, work_size);
                SSSPNearFarSplit__NF__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    delta,
                    groute::dev::WorkSourceArray<index_t>(
                        work_segs[0].GetSegmentPtr(), work_segs[0].GetSegmentSize()),
                    m_distances_datum,
                    near_worklist.DeviceObject(), far_worklist.DeviceObject()
                    );
                break;
            case 2:
                work_size = work_segs[0].GetSegmentSize() + work_segs[1].GetSegmentSize();
                KernelSizing(grid_dims, block_dims, work_size);
                SSSPNearFarSplit__NF__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    delta,
                    groute::dev::WorkSourceTwoArrays<index_t>(                        // using a two seg template 
                        work_segs[0].GetSegmentPtr(), work_segs[0].GetSegmentSize(),
                        work_segs[1].GetSegmentPtr(), work_segs[1].GetSegmentSize()),
                    m_distances_datum,
                    near_worklist.DeviceObject(), far_worklist.DeviceObject()
                    );
                break;
            default:
                printf("\n\nWarning: work_segs has more then two segments, something is wrong\n\n");
                assert(false);
            }

            return work_size;
        }
    };

    template<
        typename Algo, typename Problem>
    class SSSPSolver__NF__
    {
        Problem& m_problem;
        std::unique_ptr< groute::Worklist<local_work_t> > m_worklist1;
        std::unique_ptr< groute::Worklist<local_work_t> > m_worklist2;

public:
    SSSPSolver__NF__(groute::graphs::traversal::Context<Algo>& context, Problem& problem) : 
        m_problem(problem)
    {
        size_t max_work_size = (context.host_graph.nedges / context.ngpus) * FLAGS_wl_alloc_factor;
        if (FLAGS_wl_alloc_abs > 0)
            max_work_size = FLAGS_wl_alloc_abs;

        m_worklist1 = groute::make_unique< groute::Worklist<local_work_t> >(max_work_size);
        m_worklist2 = groute::make_unique< groute::Worklist<local_work_t> >(max_work_size);
    }

    void Solve(
        groute::graphs::traversal::Context<Algo>& context,
        groute::device_t dev,
        groute::DistributedWorklist<local_work_t, remote_work_t>& distributed_worklist,
        groute::IDistributedWorklistPeer<local_work_t, remote_work_t>* worklist_peer,
        groute::Stream& stream)
        {
            m_worklist1->ResetAsync(stream.cuda_stream);
            m_worklist2->ResetAsync(stream.cuda_stream);

            int current_delta = FLAGS_nf_delta;

            auto& remote_input_worklist = worklist_peer->GetLocalInputWorklist();
            auto& remote_output_worklist = worklist_peer->GetRemoteOutputWorklist();

            groute::Worklist<local_work_t>* input_worklist = &worklist_peer->GetTempWorklist(); // near output worklist
            groute::Worklist<local_work_t>* near_worklist = m_worklist1.get(); // near output worklist
            groute::Worklist<local_work_t>* far_worklist = m_worklist2.get(); // far output worklist

            groute::Segment<index_t> input_seg;

            while (distributed_worklist.HasWork())
            {
                int overall_far_work = 0;

                while (true)
                {
                    size_t new_work = 0, performed_work = 0;

                    m_problem.Relax__NF__(
                        input_seg, current_delta, *near_worklist, *far_worklist, remote_output_worklist, stream);

                    performed_work += input_seg.GetSegmentSize();

                    // Merge remote work into the local near-far worklists  
                    auto remote_input_segs 
                        = input_seg.Empty() 
                            ? worklist_peer->GetLocalWork(stream) // blocking call
                            : remote_input_worklist.ToSegs(stream);
                                        
                    int remote_input_work = m_problem.SplitRemoteInput__NF__(
                        remote_input_segs, current_delta, *near_worklist, *far_worklist, stream);
                    remote_input_worklist.PopItemsAsync(remote_input_work, stream.cuda_stream);

                    performed_work += remote_input_work;

                    // Get state of near-far work
                    int current_near_work = near_worklist->GetLength(stream);
                    int current_far_work = far_worklist->GetLength(stream);
                    
                    new_work += current_near_work;
                    new_work += (current_far_work - overall_far_work);
                    new_work += remote_output_worklist.GetAllocCountAndSync(stream); // get the work pushed and sync alloc-end  

                    worklist_peer->SignalRemoteWork(context.RecordEvent(dev, stream.cuda_stream)); // signal

                    // Report overall work
                    distributed_worklist.ReportWork(
                        new_work,
                        performed_work,
                        Algo::Name(), dev
                        );

                    overall_far_work = current_far_work;
                    input_worklist->ResetAsync(stream.cuda_stream);
                    input_seg = groute::Segment<index_t>(near_worklist->GetDataPtr(), current_near_work);

                    if (input_seg.Empty()) break; // break to the far worklist
                    std::swap(near_worklist, input_worklist);
                }
                
                current_delta += FLAGS_nf_delta; 

                input_seg = groute::Segment<index_t>(far_worklist->GetDataPtr(), overall_far_work);
                std::swap(far_worklist, input_worklist);
            }
        }
    };

    struct Algo
    {
        static const char* NameLower()      { return FLAGS_nf ? "sssp-nf" : "sssp"; }
        static const char* Name()           { return FLAGS_nf ? "SSSP-nf" : "SSSP"; }

        static void Init(
            groute::graphs::traversal::Context<sssp::Algo>& context,
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
            return SSSPHostNaive(graph, weights_datum.GetHostDataPtr(), min( max(0, FLAGS_source_node), graph.nnodes-1));
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

bool TestSSSPAsyncMulti__NF__(int ngpus)
{
    typedef sssp::Problem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg, groute::graphs::dev::GraphDatum> Problem;
    
    groute::graphs::traversal::__MultiRunner__ <
        sssp::Algo,
        Problem,
        sssp::SSSPSolver__NF__<sssp::Algo, Problem>, // The NF solver  
        sssp::SplitOps,
        sssp::local_work_t,
        sssp::remote_work_t,
        groute::graphs::multi::EdgeInputDatum<distance_t>,
        groute::graphs::multi::NodeOutputGlobalDatum<distance_t> > runner;
    
    groute::graphs::multi::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::multi::NodeOutputGlobalDatum<distance_t> node_distances;
    
    return runner(ngpus, edge_weights, node_distances);
}

bool TestSSSPAsyncMulti(int ngpus)
{
    typedef sssp::Problem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg, groute::graphs::dev::GraphDatum> Problem;
    
    groute::graphs::traversal::__MultiRunner__ <
        sssp::Algo,
        Problem,
        groute::graphs::traversal::__GenericMultiSolver__<sssp::Algo, Problem, sssp::local_work_t, sssp::remote_work_t>,
        sssp::SplitOps,
        sssp::local_work_t,
        sssp::remote_work_t,
        groute::graphs::multi::EdgeInputDatum<distance_t>,
        groute::graphs::multi::NodeOutputGlobalDatum<distance_t> > runner;
    
    groute::graphs::multi::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::multi::NodeOutputGlobalDatum<distance_t> node_distances;
    
    return runner(ngpus, edge_weights, node_distances);
}

bool TestSSSPSingle__NF__()
{
    typedef sssp::Problem<groute::graphs::dev::CSRGraph, groute::graphs::dev::GraphDatum, groute::graphs::dev::GraphDatum> Problem;

    groute::graphs::traversal::Context<sssp::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator
        dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    groute::graphs::single::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::single::NodeOutputDatum<distance_t> node_distances;

    dev_graph_allocator.AllocateDatumObjects(edge_weights, node_distances);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device 

    Problem problem(dev_graph_allocator.DeviceObject(), edge_weights.DeviceObject(), node_distances.DeviceObject());

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;

    groute::Stream stream;

    groute::Worklist<index_t> 
        wl1(max_work_size), 
        wl2(max_work_size), 
        wl3(max_work_size);

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    wl3.ResetAsync(stream.cuda_stream);
    stream.Sync();

    Stopwatch sw(true);
    IntervalRangeMarker algo_rng(context.host_graph.nedges, "SSSP-nf start (hardwired single GPU)");

    groute::Worklist<index_t>* input_worklist = &wl1, *near_worklist = &wl2, *far_worklist = &wl3;

    problem.Init(*input_worklist, stream);

    groute::Segment<index_t> work_seg;
    work_seg = input_worklist->ToSeg(stream);

    int current_delta = FLAGS_nf_delta;
    
    while (!work_seg.Empty())
    {
        while (!work_seg.Empty())
        {
            problem.RelaxSingle__NF__(work_seg, current_delta, *near_worklist, *far_worklist, stream);
            
            input_worklist->ResetAsync(stream.cuda_stream);
            work_seg = near_worklist->ToSeg(stream);

            std::swap(near_worklist, input_worklist);
        }
        
        current_delta += FLAGS_nf_delta; 
        work_seg = far_worklist->ToSeg(stream);

        std::swap(far_worklist, input_worklist);
    }

    algo_rng.Stop();
    sw.stop();

    if (FLAGS_repetitions > 1)
        printf("\nWarning: ignoring repetitions flag, running just one repetition (not implemented)\n");

    printf("\n%s: %f ms. <filter>\n\n", sssp::Algo::Name(), sw.ms() / FLAGS_repetitions);

    // Gather
    auto gathered_output = sssp::Algo::Gather(dev_graph_allocator, edge_weights, node_distances);

    if (FLAGS_output.length() != 0)
        sssp::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = sssp::Algo::Host(context.host_graph, edge_weights, node_distances);
        return sssp::Algo::CheckErrors(gathered_output, regression) == 0;
    }
    else {
        printf("Warning: Result not checked\n");
        return true;
    }
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
