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

#ifndef __GRAPHS_TRAVERSAL_H
#define __GRAPHS_TRAVERSAL_H

#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cassert>
#include <sstream>

#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <groute/dwl/distributed_worklist.cuh>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include <gflags/gflags.h>

DECLARE_string(output);
DECLARE_bool(check);
DECLARE_bool(verbose);
DECLARE_bool(trace);

DECLARE_string(graphfile);
DECLARE_bool(ggr);

DECLARE_bool(gen_graph);
DECLARE_int32(gen_nnodes);
DECLARE_int32(gen_factor);
DECLARE_int32(gen_method);
DECLARE_bool(gen_weights);
DECLARE_int32(gen_weight_range);

DECLARE_double(pipe_alloc_factor);
DECLARE_int32(pipe_alloc_size);
DECLARE_double(pipe_size_factor);
DECLARE_int32(pipe_size);

DECLARE_int32(fragment_size);
DECLARE_int32(cached_events);
DECLARE_int32(block_size);

DECLARE_bool(iteration_fusion);
DECLARE_int32(fused_chunk_size);
DECLARE_int32(prio_delta);
DECLARE_bool(count_work);

DECLARE_double(wl_alloc_factor_local);
DECLARE_double(wl_alloc_factor_in);
DECLARE_double(wl_alloc_factor_out);
DECLARE_double(wl_alloc_factor_pass);

DECLARE_bool(stats);
DECLARE_bool(cta_np);
DECLARE_bool(pn);

using std::min;
using std::max;

inline void KernelSizing(dim3& grid_dims, dim3& block_dims, uint32_t work_size)
{
    dim3 bd(FLAGS_block_size, 1, 1);
    dim3 gd(round_up(work_size, bd.x), 1, 1);

    grid_dims = gd;
    block_dims = bd;
}

namespace utils {
    namespace traversal
    {
        /*
        * @brief A global context for graph traversal workers
        */
        template<typename Algo>
        class Context : public groute::Context
        {
        public:
            groute::graphs::host::CSRGraph host_graph;

            int ngpus;
            int nvtxs, nedges;

            Context(int ngpus) :
                groute::Context(ngpus), ngpus(ngpus)
            {
                if (FLAGS_gen_graph)
                {
                    printf("\nGenerating graph, nnodes: %d, gen_factor: %d", FLAGS_gen_nnodes, FLAGS_gen_factor);

                    switch (FLAGS_gen_method)
                    {
                    case 1: // No intersection chain 
                        {
                            groute::graphs::host::NoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    case 2: // Chain 
                        {
                            groute::graphs::host::ChainGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    case 3: // Full cliques no intersection 
                        {
                            groute::graphs::host::CliquesNoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    default:
                        {
                            // Generates a simple random graph with 'nnodes' nodes and maximum 'gen_factor' neighbors
                            groute::graphs::host::CSRGraphGenerator generator(FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    }
                }

                else
                {
                    graph_t *graph;

                    if (FLAGS_graphfile == "") {
                        printf("A Graph File must be provided\n");
                        exit(0);
                    }

                    printf("\nLoading graph %s (%d)\n", FLAGS_graphfile.substr(FLAGS_graphfile.find_last_of('\\') + 1).c_str(), FLAGS_ggr);
                    graph = GetCachedGraph(FLAGS_graphfile, FLAGS_ggr);

                    if (graph->nvtxs == 0) {
                        printf("Empty graph!\n");
                        exit(0);
                    }

                    host_graph.Bind(
                        graph->nvtxs, graph->nedges, 
                        graph->xadj, graph->adjncy,
                        graph->readew ? graph->adjwgt : nullptr, graph->readvw ? graph->vwgt : nullptr // avoid binding to the default 1's weight arrays
                        );

                    if (FLAGS_stats)
                    {
                        printf(
                            "The graph has %d vertices, and %d edges (avg. degree: %f, max. degree: %d)\n", 
                            host_graph.nnodes, host_graph.nedges, (float)host_graph.nedges / host_graph.nnodes, host_graph.max_degree());

                        CleanupGraphs();
                        exit(0);
                    }
                }

                if (host_graph.edge_weights == nullptr && FLAGS_gen_weights) {

                    if (FLAGS_verbose)
                        printf("\nNo edge data in the input graph, generating edge weights from the range [%d, %d]\n", 1, FLAGS_gen_weight_range);

                    // Generate edge data
                    std::default_random_engine generator;
                    std::uniform_int_distribution<int> distribution(1, FLAGS_gen_weight_range);

                    host_graph.AllocWeights();

                    for (int i = 0; i < host_graph.nedges; i++)
                    {
                        host_graph.edge_weights[i] = distribution(generator);
                    }
                }

                nvtxs = host_graph.nnodes;
                nedges = host_graph.nedges;

                printf("\n----- Running %s -----\n\n", Algo::Name());

                if (FLAGS_verbose) {
                    printf("The graph has %d vertices, and %d edges (average degree: %f)\n", nvtxs, nedges, (float)nedges / nvtxs);
                }
            }
        };

        /*
        * @brief A generic runner for multi-GPU traversal workers
        */
        template<typename Algo, typename TWorker, typename DWCallbacks, typename TLocal, typename TRemote, typename ...TGraphData>
        struct Runner
        {
            bool operator() (int ngpus, int prio_delta, TGraphData&... args)
            {
                Context<Algo> context(ngpus);

                context.configuration.verbose = FLAGS_verbose;
                context.configuration.trace = FLAGS_trace;

                context.CacheEvents(FLAGS_cached_events);
                if (FLAGS_fragment_size > 0) {
                    context.EnableFragmentation(FLAGS_fragment_size);
                }

                groute::graphs::multi::CSRGraphAllocator
                    dev_graph_allocator(context, context.host_graph, ngpus, FLAGS_pn);

                dev_graph_allocator.AllocateDatumObjects(args...);

                // Setup pipeline paramenters for the DWL router
                size_t num_exch_buffs = (FLAGS_pipe_size <= 0)
                    ? ngpus*FLAGS_pipe_size_factor : FLAGS_pipe_size;
                size_t max_exch_size = (FLAGS_pipe_alloc_size <= 0)
                    ? max((size_t)(context.host_graph.nnodes * FLAGS_pipe_alloc_factor), (size_t)1) : FLAGS_pipe_alloc_size;

                // Prepare DistributedWorklist parameters
                groute::Endpoint host = groute::Endpoint::HostEndpoint(0);
                groute::EndpointList worker_endpoints = groute::Endpoint::Range(ngpus);
                std::map<groute::Endpoint, DWCallbacks> callbacks;
                for (int i = 0; i < ngpus; ++i)
                {
                    callbacks[worker_endpoints[i]] = DWCallbacks(dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...);
                }

                groute::DistributedWorklistConfiguration configuration;
                configuration.fused_chunk_size      = FLAGS_fused_chunk_size;
                configuration.count_work            = FLAGS_count_work;
                configuration.alloc_factor_in       = FLAGS_wl_alloc_factor_in;
                configuration.alloc_factor_out      = FLAGS_wl_alloc_factor_out;
                configuration.alloc_factor_pass     = FLAGS_wl_alloc_factor_pass;
                configuration.alloc_factor_local    = FLAGS_wl_alloc_factor_local;

                groute::DistributedWorklist<TLocal, TRemote, DWCallbacks, TWorker> 
                    distributed_worklist(context, { host }, worker_endpoints, callbacks, max_exch_size, num_exch_buffs, prio_delta, configuration);
                
                context.SyncAllDevices(); // Allocations are on default streams, syncing all devices 

                std::vector<std::thread> workers;
                groute::internal::Barrier barrier(ngpus + 1);

                for (int ii = 0; ii < ngpus; ++ii)
                {
                    auto dev_func = [&](size_t i)
                    {
                        context.SetDevice(i);
                        groute::Stream stream = context.CreateStream(i);
                        
                        // Perform algorithm specific device memsets nedded before Algo::HostInit (excluded from timing) 
                        Algo::DeviceMemset(stream, dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...);

                        stream.Sync();

                        barrier.Sync(); // Signal to host
                        barrier.Sync(); // Receive signal from host
                        
                        // Perform algorithm specific initialization (included in timing) 
                        Algo::DeviceInit(
                            i, stream, distributed_worklist, distributed_worklist.GetPeer(i), 
                            dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...);

                        // Loop over the work until convergence  
                        distributed_worklist.Work(i, stream, dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...);

                        barrier.Sync(); // Signal completion to host
                    };

                    workers.push_back(std::thread(dev_func, ii));
                }

                barrier.Sync(); // Wait for devices to init 

                Algo::HostInit(context, dev_graph_allocator, distributed_worklist); // Init from host

                Stopwatch sw(true); // All threads are running, start timing
                
                IntervalRangeMarker range_marker(context.nedges, "work");

                barrier.Sync(); // Signal to devices  

                barrier.Sync(); // Wait for devices to end  

                range_marker.Stop();
                sw.stop();

                printf("\n\n%s: %f ms. <filter>\n\n", Algo::Name(), sw.ms());

                for (int i = 0; i < ngpus; ++i)
                {
                    // Join workers  
                    workers[i].join();
                }

                // Gather
                auto gathered_output = Algo::Gather(dev_graph_allocator, args...);
                
                // Output
                if (FLAGS_output.length() != 0)
                    Algo::Output(FLAGS_output.c_str(), gathered_output);

                // Check results
                if (FLAGS_check) {
                    auto regression = Algo::Host(context.host_graph, args...);
                    return Algo::CheckErrors(gathered_output, regression) == 0;
                }
                else {
                    printf("Warning: Result not checked\n");
                    return true;
                }
            }
        };
    }
}

#endif // __GRAPHS_TRAVERSAL_H
