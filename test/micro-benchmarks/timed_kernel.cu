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
#include <chrono>
#include <cstdio>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#define REPETITIONS 100
#define MS_TIME 13

__global__ void Timed(unsigned long long clocks)
{
    unsigned long long target = clock64() + clocks;
    while(clock64() < target);
}

TEST(Microbenchmarks, TimedKernel)
{
    int dev = 0;
    cudaDeviceProp props;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);

    float actual_time = ((float)MS_TIME * 1000.0f) *
        ((float)props.clockRate / 1024.0f);
    
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPETITIONS; ++i)
    {
        Timed<<<500, 32>>>((unsigned long long)actual_time);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / REPETITIONS;

    printf("Kernel length: %f ms\n", mstime);
    
    ASSERT_LE(fabs(mstime - MS_TIME), 1.0f)
        << "The kernel took "
        << mstime << " ms instead of " << MS_TIME;
}

