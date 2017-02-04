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
#include <stdio.h>

#include <gflags/gflags.h>

#include <utils/utils.h>
#include <utils/stopwatch.h>

#include <atomic>
#include <thread>
#include <algorithm>

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/event_pool.h>
#include <groute/groute.h>

#include "cc_context.h"
#include "cc_common.h"
#include "cc_config.h"
#include "cc_partitioner.h"

DEFINE_int32(edges_chunk, 0, "Define the chunk size for the dynamic workload of edges");
DEFINE_int32(parents_chunk, 0, "Define the chunk size for the dynamic reduction of parents");

DEFINE_int32(edge_segs, 0, "Define the number of edge segments");
DEFINE_int32(parent_segs, 0, "Define the number of parent segments");

DEFINE_int32(input_buffers, 0, "Define the number of buffers for the input dynamic workload");
DEFINE_int32(reduce_buffers, 0, "Define the number of buffers for the reduction dynamic workload");
DEFINE_int32(nonatomic_rounds, -1, "Define the number of non atomic rounds to run per input segment");

DEFINE_bool(vertex_partitioning, false, "Perform hirarchic biased vertex partitioning");
DEFINE_bool(auto_config, true, "Deduce configuration automatically");
DEFINE_double(compute_latency_ratio, 0.12, "Hint the compute_time / memory_latency ratio");
DEFINE_int32(degree_threshold, 8, "Threshold degree for the auto config");

DEFINE_bool(tree_topology, true, "Use an hierarchical log_2(n) tree topology");
DEFINE_bool(inverse_topology, false, "Inverse the top down topology (problem converges at the N'th device)");

#ifndef NDEBUG
#define MASYNC_BS 32
#else
#define MASYNC_BS 512
#endif

bool RunCCMAsyncAtomic(int ngpus)
{
    cc::Context context(FLAGS_graphfile, FLAGS_ggr, FLAGS_verbose, ngpus);

    cc::Configuration configuration;
    if (FLAGS_auto_config)
        cc::BuildConfigurationAuto( // Deduce the configuration automatically  
        ngpus, context.nedges, context.nvtxs,
        FLAGS_compute_latency_ratio, FLAGS_degree_threshold,
        FLAGS_nonatomic_rounds,
        configuration
        );
    else
        cc::BuildConfiguration(
        ngpus, context.nedges, context.nvtxs,
        FLAGS_edge_segs, FLAGS_parent_segs,
        FLAGS_edges_chunk, FLAGS_parents_chunk,
        FLAGS_input_buffers, FLAGS_reduce_buffers,
        FLAGS_nonatomic_rounds, FLAGS_vertex_partitioning,
        configuration);

    if (FLAGS_verbose) configuration.print();

    context.DisableFragmentation();
    context.CacheEvents(
        std::max(configuration.input_pipeline_buffers, configuration.reduction_pipeline_buffers) /*raw estimation  */);

    double par_total_ms = 0.0, total_ms = 0.0;

    for (size_t rep = 0; rep < FLAGS_repetitions; ++rep)
    {
        Stopwatch psw(true);

        groute::Segment<Edge> all_edges = groute::Segment<Edge>(&context.host_edges[0], context.nedges, context.nedges, 0);
        cc::EdgePartitioner partitioner(ngpus, context.nvtxs, all_edges, configuration.vertex_partitioning);

        auto reduction_policy = FLAGS_tree_topology
            ? groute::router::Policy::CreateTreeReductionPolicy(ngpus)
            : groute::router::Policy::CreateOneWayReductionPolicy(ngpus);

        groute::router::Router<Edge> input_router(context, std::make_shared<cc::EdgeScatterPolicy>(ngpus));
        groute::router::Router<int> reduction_router(context, reduction_policy);

        groute::router::ISender<Edge>* host_sender = input_router.GetSender(groute::Device::Host);
        groute::router::IReceiver<int>* host_receiver = reduction_router.GetReceiver(groute::Device::Host); // TODO

        IntervalRangeMarker iter_rng(context.nedges, "begin");

        for (auto& edge_partition : partitioner.edge_partitions)
        {
            host_sender->Send(edge_partition, groute::Event());
        }
        host_sender->Shutdown();

        psw.stop();
        par_total_ms += psw.ms();

        std::vector< std::unique_ptr<cc::Problem> > problems;
        std::vector< std::unique_ptr<cc::Solver> > solvers;
        std::vector<std::thread> workers(ngpus);

        dim3 block_dims(MASYNC_BS, 1, 1);

        for (size_t i = 0; i < ngpus; ++i)
        {
            problems.emplace_back(new cc::Problem(context, partitioner.parents_partitions[i], i, block_dims));
            solvers.emplace_back(new cc::Solver(context, *problems.back()));

            solvers[i]->edges_in = groute::Link<Edge>(input_router, i, configuration.edges_chunk_size, configuration.input_pipeline_buffers);

            solvers[i]->reduction_in = groute::Link<component_t>(reduction_router, i, configuration.parents_chunk_size, configuration.reduction_pipeline_buffers);
            solvers[i]->reduction_out = groute::Link<component_t>(i, reduction_router);
        }

        for (size_t i = 0; i < ngpus; ++i)
        {
            // Sync the first copy operations (exclude from timing)
            solvers[i]->edges_in.Sync();
        }

        groute::internal::Barrier barrier(ngpus + 1); // barrier for accurate timing  

        for (size_t i = 0; i < ngpus; ++i)
        {
            // Run workers  
            std::thread worker(
                [&configuration, &barrier](cc::Solver& solver)
            {
                barrier.Sync();
                barrier.Sync();
                solver.Solve(configuration);
            },
                std::ref(*solvers[i]));

            workers[i] = std::move(worker);
        }

        barrier.Sync();
        Stopwatch sw(true); // all threads are running, start timing
        barrier.Sync();

        for (size_t i = 0; i < ngpus; ++i)
        {
            // Join threads  
            workers[i].join();
        }

        sw.stop();
        total_ms += sw.ms();

        // output is received from the drain device (by topology)  
        auto seg
            = host_receiver
                ->Receive(groute::Buffer<int>(&context.host_parents[0], context.nvtxs), groute::Event())
                .get();
        seg.Sync();
    }

    if (FLAGS_verbose) printf("\nPartitioning (CPU): %f ms.", par_total_ms / FLAGS_repetitions);
    printf("\nCC (Async): %f ms. <filter>\n\n", total_ms / FLAGS_repetitions);

    return CheckComponents(context.host_parents, context.nvtxs);
}

bool TestCCAsyncMulti(int ngpus)
{
    return RunCCMAsyncAtomic(ngpus);
}
