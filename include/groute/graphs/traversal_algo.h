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

#ifndef __GROUTE_GRAPHS_TRAVERSAL_ALGO_H
#define __GROUTE_GRAPHS_TRAVERSAL_ALGO_H

#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cassert>
#include <sstream>

#include <gflags/gflags.h>

#include <groute/event_pool.h>
#include <groute/distributed_worklist.h>
#include <groute/fused_distributed_worklist.h>
#include <groute/graphs/csr_graph.h>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>


DECLARE_string(graphfile);
DECLARE_bool(ggr);
DECLARE_bool(verbose);
DECLARE_int32(repetitions);
DECLARE_bool(gen_graph);
DECLARE_int32(gen_nnodes);
DECLARE_int32(gen_factor);
DECLARE_int32(gen_method);
DECLARE_bool(pn);
DECLARE_uint64(wl_alloc_abs);
DECLARE_double(wl_alloc_factor);
DECLARE_double(pipe_alloc_factor);
DECLARE_int32(pipe_alloc_size);
DECLARE_double(pipe_size_factor);
DECLARE_int32(pipe_size);
DECLARE_uint64(work_subseg);
DECLARE_int32(fragment_size);
DECLARE_int32(cached_events);
DECLARE_int32(block_size);
DECLARE_bool(iteration_fusion);
DECLARE_int32(prio_delta);
DECLARE_bool(count_work);
DECLARE_bool(stats);
DECLARE_bool(warp_append);
DECLARE_bool(debug_print);
DECLARE_bool(high_priority_receive);

DECLARE_bool(cta_np);

DECLARE_bool(gen_weights);
DECLARE_int32(gen_weight_range);

DECLARE_string(output);
DECLARE_bool(check);

using std::min;
using std::max;

inline void KernelSizing(dim3& grid_dims, dim3& block_dims, uint32_t work_size)
{
    // TODO (later): automatic kernel sizing   

    dim3 bd(FLAGS_block_size, 1, 1);
    dim3 gd(round_up(work_size, bd.x), 1, 1);

    grid_dims = gd;
    block_dims = bd;
}

namespace groute {
namespace graphs {

    namespace traversal
    {
        /*
        * @brief The global context for any traversal solver  
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
                    case 1: // no intersection chain 
                        {
                            groute::graphs::host::NoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    case 2: // chain 
                        {
                            groute::graphs::host::ChainGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    case 3: // full cliques no intersection 
                        {
                            groute::graphs::host::CliquesNoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                        break;
                    default:
                        {
                            // generates a simple random graph with 'nnodes' nodes and maximum 'gen_factor' neighbors
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
        * @brief A raw template for running multi-GPUs traversal solvers
        */
        template<typename Algo, typename Problem, typename Solver, typename SplitOps, typename TLocal, typename TRemote, typename ...TGraphDatum>
        struct __MultiRunner__
        {
            bool operator() (int ngpus, TGraphDatum&... args)
            {
                Context<Algo> context(ngpus);

                context.CacheEvents(FLAGS_cached_events);
                if (FLAGS_fragment_size > 0) {
                    context.EnableFragmentation(FLAGS_fragment_size);
                }

                if (FLAGS_verbose) {
                    printf("\n\nContext status (before):");
                    context.PrintStatus();
                    printf("\n\n");
                }

                groute::graphs::multi::CSRGraphAllocator
                    dev_graph_allocator(context, context.host_graph, ngpus);

                dev_graph_allocator.AllocateDatumObjects(args...);
                
                // currently not use, follow the code into the DistributedWorklistPeer classes
                size_t max_work_size = (context.host_graph.nedges / ngpus) * FLAGS_wl_alloc_factor; // (edges / ngpus) is the best approximation we have   
                if (FLAGS_wl_alloc_abs > 0)
                    max_work_size = FLAGS_wl_alloc_abs;


                size_t num_exch_buffs = (FLAGS_pipe_size <= 0)
                    ? ngpus*FLAGS_pipe_size_factor
                    : FLAGS_pipe_size;
                size_t max_exch_size = (FLAGS_pipe_alloc_size <= 0)
                    ? max((size_t)(context.host_graph.nnodes * FLAGS_pipe_alloc_factor), (size_t)1)
                    : FLAGS_pipe_alloc_size;

                groute::router::Router<TRemote> worklist_router(
                    context, groute::router::Policy::CreateRingPolicy(ngpus));

                groute::DistributedWorklist<TLocal, TRemote> distributed_worklist(context, worklist_router, ngpus);

                std::vector< std::unique_ptr< groute::IDistributedWorklistPeer<TLocal, TRemote> > > worklist_peers;
                std::vector< std::thread > dev_threads;

                // Prepare flags  
                groute::DistributedWorklistFlags flags = groute::DW_NoFlags;
                if (FLAGS_warp_append) flags = (groute::DistributedWorklistFlags)(flags | groute::DW_WarpAppend);
                if (FLAGS_debug_print) flags = (groute::DistributedWorklistFlags)(flags | groute::DW_DebugPrint);
                if (FLAGS_high_priority_receive) flags = (groute::DistributedWorklistFlags)(flags | groute::DW_HighPriorityReceive);

                std::vector< std::unique_ptr<Problem> > problems;
                std::vector< std::unique_ptr<Solver> > solvers;

                for (size_t i = 0; i < ngpus; ++i)
                {
                    // Set device for possible internal allocations  
                    context.SetDevice(i);

                    worklist_peers.push_back(
                        distributed_worklist.CreatePeer(
                        i,
                        SplitOps(dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...),
                        max_work_size, max_exch_size, num_exch_buffs,
                        flags));

                    // Allocating problems and solvers here to allow internal device allocations  
                    problems.push_back(groute::make_unique<Problem>(dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...));
                    solvers.push_back(groute::make_unique<Solver>(std::ref(context), std::ref(*problems[i])));
                }
                
                context.SyncAllDevices(); // allocations are on default streams, syncing all devices 

                std::vector<std::thread> workers;
                groute::internal::Barrier barrier(ngpus + 1);

                for (size_t ii = 0; ii < ngpus; ++ii)
                {
                    auto dev_func = [&](size_t i)
                    {
                        context.SetDevice(i);
                        groute::Stream stream = context.CreateStream(i);

                        auto& worklist_peer = worklist_peers[i];

                        Problem& problem = *problems[i];
                        Solver& solver = *solvers[i];

                        problem.Init(stream);
                        stream.Sync();

                        barrier.Sync(); // signal to host
                        barrier.Sync(); // receive signal from host

                        solver.Solve(context, i, distributed_worklist, worklist_peer.get(), stream);

                        barrier.Sync(); // signal to host
                    };

                    workers.push_back(std::thread(dev_func, ii));
                }

                barrier.Sync(); // wait for devices to init 

                Algo::Init(context, dev_graph_allocator, worklist_router, distributed_worklist); // init from host

                Stopwatch sw(true); // all threads are running, start timing
                
                IntervalRangeMarker algo_rng(context.nedges, "begin");

                barrier.Sync(); // signal to devices  

                barrier.Sync(); // wait for devices to end  

                algo_rng.Stop();
                sw.stop();

                if (FLAGS_repetitions > 1)
                    printf("\nWarning: ignoring repetitions flag, running just one repetition (not implemented)\n");

                printf("\n%s: %f ms. <filter>\n\n", Algo::Name(), sw.ms() / FLAGS_repetitions);

                for (size_t i = 0; i < ngpus; ++i)
                {
                    // Join workers  
                    workers[i].join();
                }

                if (FLAGS_verbose) {
                    printf("\n\nContext status (after):");
                    context.PrintStatus();
                    printf("\n\n");
                }

                // Gather
                auto gathered_output = Algo::Gather(dev_graph_allocator, args...);
                
                if (FLAGS_output.length() != 0)
                    Algo::Output(FLAGS_output.c_str(), gathered_output);

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

        
        /*
        * @brief A raw template for running multi-GPUs traversal solvers
        */
        template<typename Algo, typename Problem, typename Solver, typename SplitOps, typename TLocal, typename TRemote, typename ...TGraphDatum>
        struct __MultiRunner__Opt__
        {
            bool operator() (int ngpus, TGraphDatum&... args)
            {
                Context<Algo> context(ngpus);

                context.CacheEvents(FLAGS_cached_events);
                if (FLAGS_fragment_size > 0) {
                    context.EnableFragmentation(FLAGS_fragment_size);
                }

                if (FLAGS_verbose) {
                    printf("\n\nContext status (before):");
                    context.PrintStatus();
                    printf("\n\n");
                }

                groute::graphs::multi::CSRGraphAllocator
                    dev_graph_allocator(context, context.host_graph, ngpus);

                dev_graph_allocator.AllocateDatumObjects(args...);

                // currently not use, follow the code into the DistributedWorklistPeer classes
                size_t max_work_size = (context.host_graph.nedges / ngpus) * FLAGS_wl_alloc_factor; // (edges / ngpus) is the best approximation we have   
                if (FLAGS_wl_alloc_abs > 0)
                    max_work_size = FLAGS_wl_alloc_abs;


                size_t num_exch_buffs = (FLAGS_pipe_size <= 0)
                    ? ngpus*FLAGS_pipe_size_factor
                    : FLAGS_pipe_size;
                size_t max_exch_size = (FLAGS_pipe_alloc_size <= 0)
                    ? max((size_t)(context.host_graph.nnodes * FLAGS_pipe_alloc_factor), (size_t)1)
                    : FLAGS_pipe_alloc_size;

                groute::router::Router<TRemote> worklist_router(
                    context, groute::router::Policy::CreateRingPolicy(ngpus));

                groute::opt::DistributedWorklist<TLocal, TRemote, SplitOps> distributed_worklist(context, worklist_router, ngpus, FLAGS_prio_delta);

                std::vector< std::shared_ptr< groute::opt::IDistributedWorklistPeer<TLocal, TRemote> > > worklist_peers;
                std::vector< std::thread > dev_threads;

                // Prepare flags  
                groute::opt::DistributedWorklistFlags flags = groute::opt::DW_NoFlags;
                if (FLAGS_high_priority_receive) flags = (groute::opt::DistributedWorklistFlags)(flags | groute::opt::DW_HighPriorityReceive);

                std::vector< std::unique_ptr<Problem> > problems;
                std::vector< std::unique_ptr<Solver> > solvers;

                for (size_t i = 0; i < ngpus; ++i)
                {
                    // Set device for possible internal allocations  
                    context.SetDevice(i);

                    worklist_peers.push_back(
                        distributed_worklist.CreatePeer(
                        i,
                        SplitOps(dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...),
                        max_work_size, max_exch_size, num_exch_buffs,
                        flags));

                    // Allocating problems and solvers here to allow internal device allocations  
                    problems.push_back(groute::make_unique<Problem>(dev_graph_allocator.GetDeviceObjects()[i], args.GetDeviceObjects()[i]...));
                    solvers.push_back(groute::make_unique<Solver>(std::ref(context), std::ref(*problems[i])));
                }
                
                context.SyncAllDevices(); // allocations are on default streams, syncing all devices 

                std::vector<std::thread> workers;
                groute::internal::Barrier barrier(ngpus + 1);

                for (size_t ii = 0; ii < ngpus; ++ii)
                {
                    auto dev_func = [&](size_t i)
                    {
                        context.SetDevice(i);
                        groute::Stream stream = context.CreateStream(i);

                        auto& worklist_peer = worklist_peers[i];

                        Problem& problem = *problems[i];
                        Solver& solver = *solvers[i];

                        problem.Init(stream);
                        stream.Sync();

                        barrier.Sync(); // signal to host
                        barrier.Sync(); // receive signal from host

                        solver.Solve(context, i, distributed_worklist, worklist_peer.get(), stream);

                        barrier.Sync(); // signal to host
                    };

                    workers.push_back(std::thread(dev_func, ii));
                }

                barrier.Sync(); // wait for devices to init 

                Algo::Init(context, dev_graph_allocator, worklist_router, distributed_worklist); // init from host

                Stopwatch sw(true); // all threads are running, start timing
                
                IntervalRangeMarker algo_rng(context.nedges, "begin");

                barrier.Sync(); // signal to devices  

                barrier.Sync(); // wait for devices to end  

                algo_rng.Stop();
                sw.stop();

                if (FLAGS_repetitions > 1)
                    printf("\nWarning: ignoring repetitions flag, running just one repetition (not implemented)\n");

                printf("\n%s: %f ms. <filter>\n\n", Algo::Name(), sw.ms() / FLAGS_repetitions);

                for (size_t i = 0; i < ngpus; ++i)
                {
                    // Join workers  
                    workers[i].join();
                }

                if (FLAGS_verbose) {
                    printf("\n\nContext status (after):");
                    context.PrintStatus();
                    printf("\n\n");
                }

                // Gather
                auto gathered_output = Algo::Gather(dev_graph_allocator, args...);
                
                if (FLAGS_output.length() != 0)
                    Algo::Output(FLAGS_output.c_str(), gathered_output);

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

        /*
        * @brief A generic multi-GPU solver used for classic BFS + SSSP
        */
        template<typename Algo, typename Problem, typename TLocal, typename TRemote>
        struct __GenericMultiSolver__
        {
            Problem& m_problem;

    public:
        __GenericMultiSolver__(groute::Context& context, Problem& problem) : m_problem(problem) { }

        void Solve(
            groute::Context& context,
            groute::device_t dev,
            groute::DistributedWorklist<TLocal, TRemote>& distributed_worklist,
            groute::IDistributedWorklistPeer<TLocal, TRemote>* worklist_peer,
            groute::Stream& stream)
            {
                auto& input_worklist = worklist_peer->GetLocalInputWorklist();
                auto& temp_worklist = worklist_peer->GetTempWorklist(); // local output worklist

                while (distributed_worklist.HasWork())
                {
                    auto input_segs = worklist_peer->GetLocalWork(stream);
                    size_t new_work = 0, performed_work = 0;

                    if (input_segs.empty()) continue;

                    // If circular buffer passed the end, 2 buffers will be given: [s,end) and [0,e)
                    for (auto input_seg : input_segs)
                    {
                        auto subseg = input_seg;

                        if (FLAGS_warp_append) {
                            m_problem.template Relax <groute::Worklist<TLocal>, true>(subseg, temp_worklist, stream);
                        }
                        else {
                            m_problem.template Relax <groute::Worklist<TLocal>, false>(subseg, temp_worklist, stream);
                        }

                        input_worklist.PopItemsAsync(subseg.GetSegmentSize(), stream.cuda_stream);
                        performed_work += subseg.GetSegmentSize();
                    }

                    auto output_seg = temp_worklist.ToSeg(stream);
                    new_work += output_seg.GetSegmentSize(); // add the new work 
                    worklist_peer->PerformSplitSend(output_seg, stream); // call split-send

                    temp_worklist.ResetAsync(stream.cuda_stream); // reset the temp output worklist  

#ifndef NDEBUG  
                    if (FLAGS_debug_print)
                    {
                        std::unique_lock<std::mutex> guard(distributed_worklist.log_gate);
                        printf("\n\n\tDevice: %d\n%s->Input: ", (int)dev, Algo::Name());
                        input_worklist.PrintOffsetsDebug(stream);
                        printf("\n%s->Output: ", Algo::Name());
                        temp_worklist.PrintOffsetsDebug(stream);
                    }
#endif

                    // report work
                    distributed_worklist.ReportWork(
                        (int)new_work,
                        (int)performed_work,
                        Algo::Name(), dev
                        );
                }
            }
        };

        template<typename Algo, typename Problem, typename...TGraphDatum>
        struct __SingleRunner__
        {
            bool operator() (TGraphDatum&... args) // single GPU test
            {
                Context<Algo> context(1);

                single::CSRGraphAllocator
                    dev_graph_allocator(context.host_graph);
                
                context.SetDevice(0);
                
                dev_graph_allocator.AllocateDatumObjects(args...);

                context.SyncDevice(0); // graph allocations are on default streams, must sync device 

                Problem problem(dev_graph_allocator.DeviceObject(), args.DeviceObject()...);

                size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
                if (FLAGS_wl_alloc_abs > 0)
                    max_work_size = FLAGS_wl_alloc_abs;

                groute::Stream stream;

                groute::Worklist<index_t> wl1(max_work_size), wl2(max_work_size);
                
                wl1.ResetAsync(stream.cuda_stream);
                wl2.ResetAsync(stream.cuda_stream);
                stream.Sync();

                Stopwatch sw(true);
                IntervalRangeMarker algo_rng(context.host_graph.nedges, "begin");

                groute::Worklist<index_t>* in_wl = &wl1, *out_wl = &wl2;

                problem.Init(*in_wl, stream);

                groute::Segment<index_t> work_seg;
                work_seg = in_wl->ToSeg(stream);

                while (work_seg.GetSegmentSize() > 0)
                {
                    if (FLAGS_warp_append) {
                        problem.template Relax <groute::Worklist<index_t>, true>(work_seg, *out_wl, stream);
                    }
                    else {
                        problem.template Relax <groute::Worklist<index_t>, false>(work_seg, *out_wl, stream);
                    }

                    in_wl->ResetAsync(stream.cuda_stream);
                    std::swap(in_wl, out_wl);
                    work_seg = in_wl->ToSeg(stream);
                }

                algo_rng.Stop();
                sw.stop();

                if (FLAGS_repetitions > 1)
                    printf("\nWarning: ignoring repetitions flag, running just one repetition (not implemented)\n");

                printf("\n%s: %f ms. <filter>\n\n", Algo::Name(), sw.ms() / FLAGS_repetitions);

                // Gather
                auto gathered_output = Algo::Gather(dev_graph_allocator, args...);
                
                if (FLAGS_output.length() != 0)
                    Algo::Output(FLAGS_output.c_str(), gathered_output);

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
}

namespace
{
    typedef int mark_t;

    template<typename SourceDatum, typename TPacked>
    __global__ void PackHalosDataKernel(
        SourceDatum source_datum,
        groute::graphs::dev::GraphDatum<index_t> halos_datum,
        groute::graphs::dev::GraphDatum<mark_t> halos_marks,
        groute::dev::CircularWorklist<TPacked> remote_worklist)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (uint32_t i = 0 + tid; i < halos_datum.size; i += nthreads)
        {
            index_t halo_node = halos_datum[i]; // promised to be unique
            if (halos_marks[halo_node] == 1)
            {
                remote_worklist.append_warp(TPacked(halo_node, source_datum[halo_node]));
                halos_marks[halo_node] = 0;
            }
        }
    }

    struct WorkTargetWorklist
    {
    private:
        groute::dev::Worklist<index_t> m_worklist;

    public:
        WorkTargetWorklist(groute::Worklist<index_t>& worklist) : m_worklist(worklist.DeviceObject()) { }

        template<typename UnusedParam>
        __device__ __forceinline__ void append_work(const UnusedParam& graph, index_t work)
        {
            m_worklist.append_warp(work);
        }

        __device__ __forceinline__ void append_work(index_t work)
        {
            m_worklist.append_warp(work);
        }
    };

    struct WorkTargetSplitMark
    {
    private:
        groute::dev::Worklist<index_t> m_local_worklist;
        groute::graphs::dev::GraphDatum<mark_t> m_remote_marks;
        groute::dev::Counter m_remote_counter;

    public:
        WorkTargetSplitMark(
            groute::Worklist<index_t>& local_worklist,
            groute::graphs::dev::GraphDatum<mark_t> remote_marks,
            groute::Counter& remote_counter) :
            m_local_worklist(local_worklist.DeviceObject()), m_remote_marks(remote_marks), m_remote_counter(remote_counter.DeviceObject())
        {

        }

        template<typename TGraph>
        __device__ __forceinline__ void append_work(const TGraph& graph, index_t work)
        {
            if (graph.owns(work))
            {
                m_local_worklist.append_warp(work);
            }
            else
            {
                if (m_remote_marks[work] == 0)
                {
                    m_remote_marks[work] = 1; // mark
                    m_remote_counter.add_one_warp();
                }
            }
        }
    };

    struct WorkTargetMark
    {
    private:
        groute::graphs::dev::GraphDatum<mark_t> m_remote_marks;
        groute::dev::Counter m_remote_counter;

    public:
        WorkTargetMark(
            groute::graphs::dev::GraphDatum<mark_t> remote_marks,
            groute::Counter& remote_counter) :
            m_remote_marks(remote_marks), m_remote_counter(remote_counter.DeviceObject())
        {

        }

        __device__ __forceinline__ void append_work(index_t work)
        {
            if (m_remote_marks[work] == 0)
            {
                m_remote_marks[work] = 1; // mark
                m_remote_counter.add_one_warp();
            }
        }
    };
}

#endif // __GROUTE_GRAPHS_TRAVERSAL_ALGO_H
