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
#include <cstdio>
#include <cuda_runtime.h>
#include <gflags/gflags.h>

#include <iostream>

#include <utils/utils.h>
#include <utils/interactor.h>

#ifndef _WIN32
#define gflags google
#endif


bool TestCCAsyncMulti(int ngpus);

void CleanupGraphs();

DEFINE_bool(interactive, false, "Run an interactive session");
DEFINE_string(cmdfile, "", "A file with commands to execute");

DEFINE_int32(num_gpus, 2, "Override number of GPUs (or negative to use the amount of available GPUs)");
DEFINE_int32(startwith, 1, "Start with a specific number of GPUs");

DEFINE_bool(run_cc_async, true, "Run CC over Multi GPUs with Async CUDA");


int RunInteractor(IInteractor& interactor);
int Run();

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    int exit = 0;

    if (!(FLAGS_cmdfile == ""))
    {
        FileInteractor file_interactor(FLAGS_cmdfile);
        std::cout << std::endl << "Starting a command file cc session" << std::endl;
        RunInteractor(file_interactor);
    }

    else if (FLAGS_interactive)
    {
        ConsoleInteractor console_interactor;
        std::cout << std::endl << "Starting an interactive cc session" << std::endl;
        RunInteractor(console_interactor);
    }

    else {
        NoInteractor no_interactor;
        RunInteractor(no_interactor);
    }

    CleanupGraphs();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return exit;
}

int RunInteractor(IInteractor& interactor)
{
    int exit = 0;

    if(interactor.RunFirst()) exit = Run(); // run the first round

    while (true)
    {
        gflags::FlagSaver fs; // This saves the flags state and restores all values on destruction
        std::string cmd;

        if (!interactor.GetNextCommand(cmd)) break;
        cmd.insert(0, "cc "); // insert any string to emulate the process name usually passed on argv

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
        printf("Running CC with %d GPUs, starting with %d GPUs\n", num_actual_gpus,
            FLAGS_startwith);
    }

    if(!(FLAGS_run_cc_async))
    {
      printf("ERROR: You must specify a CC variant (-run* flags, use -h for a complete list)\n");
      return 1;
    }

    if((FLAGS_startwith > 1) && !(FLAGS_run_cc_async )) {
      printf("ERROR: Only -run_cc_async runs on multiple gpus\n");
      return 1;
    }

    for (int G = FLAGS_startwith; G <= num_actual_gpus; ++G)
    {
        printf("Testing with %d GPUs\n", G);
        printf("--------------------\n\n");

        if (FLAGS_run_cc_async)         overall &= TestCCAsyncMulti(G);
    }

    printf("Overall: Test %s\n", overall ? "passed" : "FAILED");
    return 0;
}
