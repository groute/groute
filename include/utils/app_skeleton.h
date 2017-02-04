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
#ifndef APP_SKELETON_H
#define APP_SKELETON_H

#include <cstdio>
#include <cuda_runtime.h>
#include <gflags/gflags.h>

#include <iostream>

#include <utils/utils.h>
#include <utils/interactor.h>

#ifndef _WIN32
#define gflags google
#endif

#ifndef NDEBUG
#define RUN_ALL true
#else
#define RUN_ALL false
#endif


DEFINE_bool(interactive, false, "Run an interactive session");
DEFINE_string(cmdfile, "", "A file with commands to execute");

DEFINE_int32(num_gpus, 2, "Override number of GPUs (or negative to use the amount of available GPUs)");
DEFINE_int32(startwith, 1, "Start with a specific number of GPUs");

DEFINE_string(output, "", "File to store output to");
DEFINE_bool(check, false, "Check results");

/* variants */
DEFINE_bool(all, false, "Run all variants");
DEFINE_bool(single, false, "Run the single GPU variant");
DEFINE_bool(async_multi, true, "Run the async multigpu variant");
DEFINE_bool(opt, true, "Run the optimized (prio + fusion) async multigpu variant");


// Here as a workaround for now
#ifndef NDEBUG
#define VERBOSE true
#define REPETITIONS_DEFAULT 1
#else
#define VERBOSE false
#define REPETITIONS_DEFAULT 1
#endif
DEFINE_string(graphfile, "", "A file with a graph in Dimacs 10 format");
DEFINE_bool(ggr, true, "Graph file is a Galois binary GR file");
DEFINE_bool(verbose, VERBOSE, "Verbose prints");
DEFINE_int32(repetitions, REPETITIONS_DEFAULT, "Repetitions of GPU tests");
DEFINE_bool(gen_graph, false, "Generate a random graph");
DEFINE_int32(gen_nnodes, 100000, "Number of nodes for random graph generation");
DEFINE_int32(gen_factor, 10, "A factor number for graph generation");
DEFINE_int32(gen_method, 0, "Select the requested graph generation method: \n\t0: Random graph \n\t1: Two-way chain graph without segment intersection \n\t2: Two-way chain graph with intersection \n\t3: Full cliques per device without segment intersection");
DEFINE_uint64(wl_alloc_abs, 0, "Absolute size for local worklists (if not zero, overrides --wl_alloc_factor");
DEFINE_double(wl_alloc_factor, 0.2, "Local worklists will allocate '(nedges / ngpus)' times this factor");
DEFINE_double(pipe_alloc_factor, 0.05, "Each socket pipeline buffer will allocate 'nnodes' times this factor");
DEFINE_int32(pipe_alloc_size, -1, "Each socket pipeline buffer will allocate 'pipe_alloc_size' items per buffer");
DEFINE_double(pipe_size_factor, 4, "Each socket pipeline will allocate 'ngpus' times this factor buffers");
DEFINE_int32(pipe_size, -1, "Each socket pipeline will allocate 'pipe_size' buffers");
DEFINE_uint64(work_subseg, 0, "A parameter for breaking the actual workload of the algorithm into sub-segments");
DEFINE_int32(fragment_size, -1, "Fragment size for all memcpy operations");
DEFINE_int32(cached_events, 8, "Number of events to cache in each event pool (per device)");
DEFINE_int32(block_size, 256, "Block size for traversal kernels");
DEFINE_bool(iteration_fusion, true, "Fuse multiple iterations (FusedWork kernel performs one iteration each launch if this is false)");
DEFINE_int32(prio_delta, 10, "The soft priority delta");
DEFINE_bool(count_work, false, "Count the work-items performed by each individual GPU");
DEFINE_bool(stats, false, "Print graph statistics and exit");
DEFINE_bool(warp_append, true, "Use warp aggregated operations for worklist append's");
DEFINE_bool(debug_print, false, "Print detailed debug info");
DEFINE_bool(high_priority_receive, true, "Use a high priority stream for split receive");

DEFINE_bool(cta_np, true, "Use nested parallelism withing traversal kernels");

DEFINE_double(wl_alloc_factor_local, 0.2, "Worklist allocation factor: local worklist");
DEFINE_double(wl_alloc_factor_in, 0.4, "Worklist allocation factor: incoming worklist");
DEFINE_double(wl_alloc_factor_out, 0.2, "Worklist allocation factor: outgoing worklist");
DEFINE_double(wl_alloc_factor_pass, 0.2, "Worklist allocation factor: pass-through worklist");

DEFINE_bool(gen_weights, false, "Generate edge weights if missing in graph input");
DEFINE_int32(gen_weight_range, 100, "The range to generate edge weights from (coordinate this parameter with nf-delta if running sssp-nf)");

#ifdef HAVE_METIS
DEFINE_bool(pn, true, "Partition the input graph using METIS (requires a symmetric graph)");
#else
DEFINE_bool(pn, false, "[BINARY NOT BUILT WITH METIS] Partition the input graph using METIS (requires a symmetric graph)");
#endif

DECLARE_bool(stats);

template<typename App>
struct Skeleton
{
    int operator() (int argc, char **argv)
    {
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        int exit = 0;

        if (FLAGS_stats)
        {
            // Just run anything and return
            App::Single();
            return 0;
        }

        if (!(FLAGS_cmdfile == ""))
        {
            FileInteractor file_interactor(FLAGS_cmdfile);
            std::cout << std::endl << "Starting a command file "<< App::Name() << " session" << std::endl;
            RunInteractor(file_interactor);
        }

        else if (FLAGS_interactive)
        {
            ConsoleInteractor console_interactor;
            std::cout << std::endl << "Starting an interactive "<< App::Name() << " session" << std::endl;
            RunInteractor(console_interactor);
        }

        else {
            NoInteractor no_interactor;
            RunInteractor(no_interactor);
        }

        App::Cleanup();

        return exit;
    }

    int RunInteractor(IInteractor& interactor)
    {
        int exit = 0;

        if (interactor.RunFirst()) exit = Run(); // run the first round

        while (true)
        {
            gflags::FlagSaver fs; // This saves the flags state and restores all values on destruction
            std::string cmd;

            if (!interactor.GetNextCommand(cmd)) break;
            cmd.insert(0, " "); 
            cmd.insert(0, App::Name()); // insert any string to emulate the process name usually passed on argv

            int argc; char **argv;
            stringToArgcArgv(cmd, &argc, &argv);
            gflags::ParseCommandLineFlags(&argc, &argv, false);
            freeArgcArgv(&argc, &argv);

            exit = Run();
        }

        return exit;
    }

    int Run()
    {
        if (FLAGS_all)
        {
            FLAGS_single = true;
            FLAGS_async_multi = true;
        }

        int num_actual_gpus = FLAGS_num_gpus;
        if (num_actual_gpus <= 0)
        {
            if (cudaGetDeviceCount(&num_actual_gpus) != cudaSuccess)
            {
                printf("Error %d when getting devices (is CUDA enabled?)\n", num_actual_gpus);
                return 1;
            }
        }

        if (FLAGS_startwith > num_actual_gpus || FLAGS_startwith <= 0)
        {
            printf("Starting with invalid amount of GPUs (Requested: %d, available: %d)\n",
                FLAGS_startwith, num_actual_gpus);
            return 2;
        }

        bool overall = true;

        if (num_actual_gpus > 1)
        {
            printf("Running %s with %d GPUs, starting with %d GPUs\n", App::NameUpper(), num_actual_gpus,
                FLAGS_startwith);
        }

        if (!(FLAGS_single || FLAGS_async_multi))
        {
            printf("ERROR: You must specify a %s variant (-single, -async_multi)\n", App::NameUpper());
            return 1;
        }

        if ((FLAGS_startwith > 1) && !(FLAGS_async_multi)) {
            printf("ERROR: Only -async_multi runs on multiple gpus\n");
            return 1;
        }

        if (FLAGS_startwith == 1)
        {
            printf("\nTesting single GPU %s\n", App::NameUpper());
            printf("--------------------\n\n");

            if (FLAGS_single) overall &= App::Single();
        }

        if (FLAGS_async_multi)
        {
            for (int G = FLAGS_startwith; G <= num_actual_gpus; ++G)
            {
                printf("Testing with %d GPUs\n", G);
                printf("--------------------\n\n");

                overall &= App::AsyncMulti(G);
            }
        }

        printf("Overall: Test %s\n", overall ? "passed" : "FAILED");
        return 0;
    }
};



#endif // APP_SKELETON_H
