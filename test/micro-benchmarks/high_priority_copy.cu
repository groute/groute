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

#include <utils/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/event_pool.h>


typedef utils::SharedArray<int> SharedArray;
typedef utils::SharedValue<int> SharedValue;


TEST(Microbenchmarks, HighPriorityCopy)
{
    cudaSetDevice(0);
    
    groute::Stream stream;
    groute::Stream highprio_stream(groute::SP_High);

    SharedArray arr(100 * 1024 * 1024); // 100 MB array
    SharedValue val;

    groute::internal::Barrier bar(2);

    std::chrono::system_clock::time_point t1, t2, t3; 

    std::thread highprio_sync([&val, &highprio_stream, &bar, &t2] 
    {
        cudaSetDevice(0);

        bar.Sync();
        bar.Sync();

        val.D2HAsync(highprio_stream.cuda_stream);
        highprio_stream.Sync();
        t2 = std::chrono::high_resolution_clock::now();
    });
    
    bar.Sync();

    t1 = std::chrono::high_resolution_clock::now();

    arr.D2HAsync(stream.cuda_stream);

    bar.Sync();

    stream.Sync();
    t3 = std::chrono::high_resolution_clock::now();

    highprio_sync.join();

    double ms_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count() / 1000.0;
    double ms_flag_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    std::cout 
        << "Copy time (100 MB) " << ms_copy_time << " ms.\n"
        << "Copy time (4B Flag) " << ms_flag_time << " ms.\n";
}

