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
#include <groute/cta_work.h>

#include <groute/graphs/csr_graph.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/graphs/fused_solver.h>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>

#include "sssp_common.h"

DECLARE_int32(source_node);


namespace sssp {
    namespace opt {
        
        const distance_t INF = UINT_MAX;

        struct DistanceData
        {
            index_t node;
            distance_t distance;

            __device__ __host__ __forceinline__ DistanceData(index_t node, distance_t distance) : node(node), distance(distance) { }
            __device__ __host__ __forceinline__ DistanceData() : node(INF), distance(INF) { }
        };

        typedef index_t local_work_t;
        typedef DistanceData remote_work_t;

        __global__ void SSSPInit(distance_t* distances, int nnodes)
        {
            int tid = GTID;
            if (tid < nnodes)
            {
                distances[tid] = INF;
            }
        }

        template<
            typename TGraph,
            typename TWeightDatum, typename TDistanceDatum>
        struct SSSPWorkNP
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph,
                TWeightDatum& edge_weights, TDistanceDatum& node_distances
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

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
                        [&graph, &edge_weights, &node_distances, &rwl_in, &rwl_out](index_t edge, distance_t distance)
                        {
                            index_t dest = graph.edge_dest(edge);
                            distance_t weight = edge_weights.get_item(edge);

                            if (distance + weight < atomicMin(node_distances.get_item_ptr(dest), distance + weight))
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
                                    rwl_out.append_warp(DistanceData(dest, distance + weight), low_leader, __popc(remote_mask), thread_offset);
                                }
                            }
                        }
                        ); 
                }
            }
        };

        template<
            typename TGraph,
            typename TWeightDatum, typename TDistanceDatum>
        struct SSSPWork
        {
            template<typename WorkSource>
            __device__ static void work(
                const WorkSource& work_source,
                groute::dev::CircularWorklist<local_work_t>& rwl_in,
                groute::dev::CircularWorklist<remote_work_t>& rwl_out,
                const TGraph& graph,
                TWeightDatum& edge_weights, TDistanceDatum& node_distances
                )
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

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
                                rwl_out.append_warp(DistanceData(dest, distance + weight), low_leader, __popc(remote_mask), thread_offset);
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

            __device__ __forceinline__ groute::opt::SplitFlags on_receive(const remote_work_t& work)
            {
                if (m_graph_seg.owns(work.node))
                {
                    return (work.distance < atomicMin(m_distances_datum.get_item_ptr(work.node), work.distance))
                        ? groute::opt::SF_Take
                        : groute::opt::SF_None; // filter
                }

                return groute::opt::SF_Pass;
            }

            __device__ __forceinline__ bool is_high_prio(const local_work_t& work, const distance_t& global_prio)
            {
                return m_distances_datum[work] <= global_prio;
            }

            __device__ __forceinline__ groute::opt::SplitFlags on_send(local_work_t work)
            {
                return (m_graph_seg.owns(work))
                    ? groute::opt::SF_Take
                    : groute::opt::SF_Pass;
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
        struct FusedProblem
        {
            TGraph m_graph;
            TWeightDatum<distance_t> m_weights_datum;
            TDistanceDatum<distance_t> m_distances_datum;

            typedef SSSPWork<TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t>> WorkType;
            typedef SSSPWorkNP<TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t>> WorkTypeNP;

        public:
            FusedProblem(const TGraph& graph, const TWeightDatum<distance_t>& weights_datum, const TDistanceDatum<distance_t>& distances_datum) :
                m_graph(graph), m_weights_datum(weights_datum), m_distances_datum(distances_datum)
            {
            }

            // Initial init. Called before a global CPU+GPU barrier
            void Init(groute::Stream& stream) const
            {
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, m_distances_datum.size);

                SSSPInit << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_distances_datum.data_ptr, m_distances_datum.size);
            }

            bool DoFusedInit(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, distance_t global_prio,
                volatile int *high_work_counter, volatile int *low_work_counter,
                uint32_t *kernel_internal_counter, volatile int *send_signal_ptr,
                cub::GridBarrierLifetime& barrier_lifetime,
                dim3 grid_dims, dim3 block_dims, groute::Stream& stream)
            {
                return false; // no work was done here
            }

            void DoFusedWork(groute::Worklist<local_work_t>* lwl_high, groute::Worklist<local_work_t>* lwl_low,
                groute::CircularWorklist<local_work_t>*  rwl_in, groute::CircularWorklist<remote_work_t>*  rwl_out,
                int fused_chunk_size, distance_t global_prio,
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
                            groute::NeverStop, local_work_t, remote_work_t, distance_t, SplitOps,
                            WorkTypeNP,
                            TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            sssp::opt::SplitOps(m_graph, m_weights_datum, m_distances_datum),
                            m_graph, m_weights_datum, m_distances_datum
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::NeverStop, local_work_t, remote_work_t, distance_t, SplitOps,
                            WorkType,
                            TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            sssp::opt::SplitOps(m_graph, m_weights_datum, m_distances_datum),
                            m_graph, m_weights_datum, m_distances_datum
                            );
                    }
                }

                else
                {
                    if (FLAGS_cta_np)
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, distance_t, SplitOps,
                            WorkTypeNP,
                            TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            sssp::opt::SplitOps(m_graph, m_weights_datum, m_distances_datum),
                            m_graph, m_weights_datum, m_distances_datum
                            );
                    }
                    else
                    {
                        groute::FusedWork <
                            groute::RunNTimes<1>, local_work_t, remote_work_t, distance_t, SplitOps,
                            WorkType,
                            TGraph, TWeightDatum<distance_t>, TDistanceDatum<distance_t> >

                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (

                            lwl_high->DeviceObject(), lwl_low->DeviceObject(),
                            rwl_in->DeviceObject(), rwl_out->DeviceObject(),
                            fused_chunk_size, global_prio,
                            high_work_counter, low_work_counter,
                            kernel_internal_counter, send_signal_ptr,
                            barrier_lifetime,
                            sssp::opt::SplitOps(m_graph, m_weights_datum, m_distances_datum),
                            m_graph, m_weights_datum, m_distances_datum
                            );
                    }
                }
            }
        };

        struct Algo
        {
            static const char* NameLower()      { return "sssp"; }
            static const char* Name()           { return "SSSP"; }

            static void Init(
                groute::graphs::traversal::Context<sssp::opt::Algo>& context,
                groute::graphs::multi::CSRGraphAllocator& graph_manager,
                groute::router::Router<remote_work_t>& worklist_router,
                groute::opt::DistributedWorklist<local_work_t, remote_work_t, SplitOps>& distributed_worklist)
            {
                index_t source_node = min(max(0, FLAGS_source_node), context.host_graph.nnodes - 1);

                auto partitioner = graph_manager.GetGraphPartitioner();
                if (partitioner->NeedsReverseLookup())
                {
                    source_node = partitioner->GetReverseLookupFunc()(source_node);
                }

                // report the initial work
                distributed_worklist.ReportHighPrioWork(1, 0, "Host", groute::Device::Host, true);

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
                return SSSPHostNaive(graph, weights_datum.GetHostDataPtr(), min(max(0, FLAGS_source_node), graph.nnodes - 1));
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
}

bool TestSSSPAsyncMultiOptimized(int ngpus)
{
    typedef sssp::opt::FusedProblem<groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg, groute::graphs::dev::GraphDatum> ProblemType;
    typedef groute::graphs::traversal::FusedSolver<
        sssp::opt::Algo, ProblemType, 
        sssp::opt::local_work_t , sssp::opt::remote_work_t, distance_t, 
        sssp::opt::SplitOps, 
        groute::graphs::dev::CSRGraphSeg, groute::graphs::dev::GraphDatumSeg<distance_t>, groute::graphs::dev::GraphDatum<distance_t>> SolverType;

    groute::graphs::traversal::__MultiRunner__Opt__ <
        sssp::opt::Algo,
        ProblemType,
        SolverType,
        sssp::opt::SplitOps,
        sssp::opt::local_work_t,
        sssp::opt::remote_work_t,
        groute::graphs::multi::EdgeInputDatum<distance_t>,
        groute::graphs::multi::NodeOutputGlobalDatum<distance_t> > runner;
    
    groute::graphs::multi::EdgeInputDatum<distance_t> edge_weights;
    groute::graphs::multi::NodeOutputGlobalDatum<distance_t> node_distances;
    
    return runner(ngpus, edge_weights, node_distances);
}
