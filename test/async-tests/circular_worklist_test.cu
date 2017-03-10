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
#include <algorithm>

#include <gtest/gtest.h>
#include <groute/internal/cuda_utils.h>

#include "cuda_gtest_utils.h"
#include "test_common.h"

#include <groute/event_pool.h>
#include <groute/device/queue.cuh>

#include <utils/cuda_utils.h>

typedef utils::SharedArray<int> SharedArray;
typedef utils::SharedValue<int> SharedValue;



__global__ void ConsumeKernel(const int* work, int work_size, int* sum)
{
    int tid = GTID;

    if (tid < work_size)
    {
        atomicAdd(sum, work[tid]);
    }
}

template<bool Prepend = false, bool Warp = true>
__global__ void ProduceKernel(groute::dev::CircularWorklist<int> worklist, const int* work, int work_size)
{
    int tid = GTID;

    if (tid < work_size)
    {
        if (Prepend) {
            if (Warp) {
                worklist.prepend_warp(work[tid]);
            }
            else {
                worklist.prepend(work[tid]);
            }
        }
        else {
            if (Warp) {
                worklist.append_warp(work[tid]);
            }
            else {
                worklist.append(work[tid]);
            }
        }
    }
}

void TestAppendPrependPop(int nappend, int nprepend, int wl_alloc_factor = 1, int chunk_size_factor = 100)
{
    srand(static_cast <unsigned> (22522));

    int worklist_capacity = (nappend + nprepend) / wl_alloc_factor;
    int max_chunk_size = std::max((nappend + nprepend) / chunk_size_factor, 1);

    CUASSERT_NOERR(cudaSetDevice(0));

    SharedArray append_input(nappend);
    SharedArray prepend_input(nprepend);
    SharedValue sum;

    int regression_sum = 0;

    for (int i = 0; i < nappend; ++i) {
        regression_sum += (append_input.host_vec[i] = (rand() % 100));
    }
    for (int i = 0; i < nprepend; ++i) {
        regression_sum += (prepend_input.host_vec[i] = (rand() % 100));
    }

    append_input.H2D();
    prepend_input.H2D();

    groute::CircularWorklist<int> circular_worklist(worklist_capacity);

    // sync objects  
    std::mutex mutex;
    std::condition_variable cv;
    bool exit = false;
    groute::Event signal;

    groute::Stream producer_stream(0, groute::SP_Default);
    circular_worklist.ResetAsync(producer_stream.cuda_stream); // init 

    producer_stream.Sync(); // sync

    std::thread worker([&]()
    {
        groute::Stream alternating_stream(0, groute::SP_Default);

        srand(static_cast <unsigned> (22422));
        int pos = 0;

        while (true)
        {
            // prepend some work
            if (pos < nprepend)
            {
                int chunk = std::min(nprepend - pos, rand() % (max_chunk_size - 1) + 1);

                dim3 block_dims(32, 1, 1);
                dim3 grid_dims(round_up(chunk, block_dims.x), 1, 1);

                ProduceKernel <true> <<< grid_dims, block_dims, 0, alternating_stream.cuda_stream >>> (
                    circular_worklist.DeviceObject(), prepend_input.dev_ptr + pos, chunk);

                pos += chunk;
            }

            // check if worklist has work
            auto segs = circular_worklist.ToSegs(alternating_stream);

            if (segs.empty())
            {
                // wait for append producer  
                {
                    std::unique_lock<std::mutex> guard(mutex);
                    signal.Wait(alternating_stream.cuda_stream);
                    segs = circular_worklist.ToSegs(alternating_stream);

                    while (segs.empty())
                    {
                        if (exit) break;
                        cv.wait(guard);
                        signal.Wait(alternating_stream.cuda_stream);
                        segs = circular_worklist.ToSegs(alternating_stream);
                    }
                }
            }

            if (segs.empty()) break;

            // do and pop the work
            for (auto& seg : segs)
            {
                dim3 block_dims(32, 1, 1);
                dim3 grid_dims(round_up(seg.GetSegmentSize(), block_dims.x), 1, 1);
                ConsumeKernel <<< grid_dims, block_dims, 0, alternating_stream.cuda_stream >>> (
                    seg.GetSegmentPtr(), seg.GetSegmentSize(), sum.dev_ptr);

                circular_worklist.PopItemsAsync(seg.GetSegmentSize(), alternating_stream.cuda_stream);
            }
        }

        alternating_stream.Sync();
    });

    int pos = 0;

    while (pos < nappend)
    {
        int chunk = std::min(nappend - pos, rand() % (max_chunk_size));

        dim3 producer_block_dims(32, 1, 1);
        dim3 producer_grid_dims(round_up(chunk, producer_block_dims.x), 1, 1);

        ProduceKernel << < producer_grid_dims, producer_block_dims, 0, producer_stream.cuda_stream >> > (
            circular_worklist.DeviceObject(), append_input.dev_ptr + pos, chunk);

        circular_worklist.SyncAppendAllocAsync(producer_stream.cuda_stream);

        auto ev = groute::Event::Record(producer_stream.cuda_stream);

        {
            std::lock_guard<std::mutex> lock(mutex);
            signal = ev;
            cv.notify_one();
        }

        pos += chunk;
    }

    producer_stream.Sync();

    {
        std::lock_guard<std::mutex> lock(mutex);
        exit = true;
        cv.notify_one();
    }

    worker.join();

    int output_sum = sum.get_val_D2H();

    ASSERT_EQ(regression_sum, output_sum);
}

TEST(CircularWorklist, ProducerConsumer)
{
    TestAppendPrependPop(2048, 0);
    TestAppendPrependPop(2048, 32);
    TestAppendPrependPop(0, 2048);
    TestAppendPrependPop(32, 2048);
    TestAppendPrependPop(2048, 2048);
    TestAppendPrependPop(2048 * 1024, 2048 * 2048);
    TestAppendPrependPop(2048 * 2048, 2048 * 1024);
    TestAppendPrependPop(2048 * 2048, 2048 * 2048);
}