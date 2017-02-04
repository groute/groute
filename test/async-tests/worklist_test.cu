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
#include <gtest/gtest.h>

#include "cuda_gtest_utils.h"
#include "test_common.h"

#include <vector>
#include <algorithm>
#include <thread>
#include <memory>

#include <utils/app_skeleton.h>

#include <groute/event_pool.h>
#include <groute/distributed_worklist.h>

#include <utils/cuda_utils.h>


__global__ void CountKernel(int *buffer, int count, int offset, int *bins)
{
    int tid = GTID;
    if (tid < count)
    {
        atomicAdd(bins + buffer[tid] - offset, 1);
    }
}


std::vector<int64_t> multi_split(std::vector<int>& items, int segs, int seg_size)
{
    std::vector<int64_t> split_indexes(segs+1);

    auto beg = items.begin();
    for (int i = 0; i < segs; ++i)
    {
        split_indexes[i] = beg - items.begin();

        auto split = std::partition(
            beg, items.end(),
            [seg_size, i](const int& item)
        {
            return (item / seg_size) == i;
        }
        );

        beg = split;
    }

    split_indexes[segs] = items.size();

    return split_indexes;
}


typedef utils::SharedArray<int> Array;
typedef utils::SharedValue<int> WorkCounter;


struct Splitter
{
    int split_val;

    Splitter(int split_val) : split_val(split_val) { }

    __device__ __host__ bool operator()(int val) const
    {
        return val < split_val;
    }
};


namespace histogram
{
    struct SplitOps
    {
        __device__ __forceinline__ groute::SplitFlags on_receive(int work)
        {
            return ((work / m_seg_size) == m_seg_index)
                ? groute::SF_Take
                : groute::SF_Pass;
        }

        __device__ __forceinline__ groute::SplitFlags on_send(int work)
        {
            return ((work / m_seg_size) == m_seg_index)
                ? groute::SF_Take
                : groute::SF_Pass;
        }
    
        __device__ __forceinline__ int pack(int work)
        {
            return work;
        }

        __device__ __forceinline__ int unpack(int work)
        {
            return work;
        }
        
        __device__ __host__ SplitOps(int split_seg_index, int split_seg_size)
            : m_seg_index(split_seg_index), m_seg_size(split_seg_size)
        {
        }

    private:
        int m_seg_index;
        int m_seg_size;
    };
}


void TestHistogramWorklist(int ngpus, size_t histo_size, size_t work_size)
{
    size_t histo_seg_size = histo_size / ngpus;
    histo_size = histo_seg_size * ngpus;

    ASSERT_GT(histo_seg_size, 0);

    size_t max_work_size = work_size; // round_up(work_size, (size_t)ngpus);
    size_t num_exch_buffs = 4 * ngpus;
    size_t max_exch_size = work_size; // round_up(max_work_size, num_exch_buffs);

    groute::Context context(ngpus);

    //
    // Input routing  
    //
    groute::router::Router<int> input_router(context, groute::router::Policy::CreateScatterPolicy(groute::Device::Host, range(ngpus)));    
    groute::router::ISender<int>* input_sender = input_router.GetSender(groute::Device::Host); 

    std::vector< std::unique_ptr< groute::router::IPipelinedReceiver<int> > > input_receivers;

    for (size_t i = 0; i < ngpus; ++i)
    {
        auto receiver = input_router.CreatePipelinedReceiver(i, max_work_size, 1);
        input_receivers.push_back(std::move(receiver));
    }

    srand(static_cast <unsigned> (22522));
    std::vector<int> host_worklist;

    for (int ii = 0, count = work_size; ii < count; ++ii)
    {
        host_worklist.push_back((rand()*round_up(histo_size, RAND_MAX)) % histo_size);
    }

    input_sender->Send(groute::Segment<int>(&host_worklist[0], host_worklist.size()), groute::Event());
    input_sender->Shutdown();
    //
    //
    //

    groute::router::Router<int> exchange_router(
        context, groute::router::Policy::CreateRingPolicy(ngpus));
    
    groute::DistributedWorklist<int, int> distributed_worklist(context, exchange_router, ngpus);

    std::vector< std::unique_ptr< groute::IDistributedWorklistPeer<int, int> > > worklist_peers;
    std::vector< std::thread > dev_threads;

    std::vector<int*> dev_segs(ngpus);

    for (size_t i = 0; i < ngpus; ++i)
    {
        context.SetDevice(i);

        CUASSERT_NOERR(cudaMalloc(&dev_segs[i], histo_seg_size * sizeof(int)));
        CUASSERT_NOERR(cudaMemset(dev_segs[i], 0, histo_seg_size * sizeof(int)));

        worklist_peers.push_back(
            distributed_worklist.CreatePeer(i, histogram::SplitOps(i, histo_seg_size), max_work_size, max_exch_size, num_exch_buffs));
    }

    std::vector<std::thread> workers;
    groute::internal::Barrier barrier(ngpus);

    for (size_t i = 0; i < ngpus; ++i)
    {
        std::thread worker([&, i]()
        {
            context.SetDevice(i);
            groute::Stream stream = context.CreateStream(i);

            auto& worklist_peer = worklist_peers[i];

            auto input_fut = input_receivers[i]->Receive();
            auto input_seg = input_fut.get();

            distributed_worklist.ReportWork(input_seg.GetSegmentSize());

            barrier.Sync();

            //
            // Start processing  
            //

            auto& input_worklist = worklist_peer->GetLocalInputWorklist(); 

            input_seg.Wait(stream.cuda_stream);
            worklist_peer->PerformSplitSend(input_seg, stream);

            while (true)
            {
                auto input_segs = worklist_peer->GetLocalWork(stream);
                size_t total_segs_size = 0;

                if (input_segs.empty()) break;

                for (auto seg : input_segs)
                {
                    dim3 count_block_dims(32, 1, 1);
                    dim3 count_grid_dims(round_up(seg.GetSegmentSize(), count_block_dims.x), 1, 1);

                    CountKernel <<< count_grid_dims, count_block_dims, 0, stream.cuda_stream >>>
                        (seg.GetSegmentPtr(), seg.GetSegmentSize(), i*histo_seg_size, dev_segs[i]);

                    input_worklist.PopItemsAsync(seg.GetSegmentSize(), stream.cuda_stream);
                    total_segs_size += seg.GetSegmentSize();
                }
                
                // report work
                distributed_worklist.ReportWork(- (int)total_segs_size);
            }

            stream.Sync();
        });

        workers.push_back(std::move(worker));
    }

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Join workers  
        workers[i].join();
    }

    std::vector<int> regression_segs(histo_seg_size*ngpus, 0);
    std::vector<int> host_segs(histo_seg_size*ngpus);

    for (auto it : host_worklist)
    {
        ++regression_segs[it];
    }

    for (size_t i = 0; i < ngpus; ++i)
    {
        context.SetDevice(i);
        CUASSERT_NOERR(cudaMemcpy(&host_segs[i*histo_seg_size], dev_segs[i], histo_seg_size * sizeof(int), cudaMemcpyDeviceToHost));
    }

    int over_errors = 0, miss_errors = 0;
    std::vector<int> over_error_indices, miss_error_indices;

    for (int i = 0; i < histo_size; ++i)
    {
        int hv = host_segs[i];
        int rv = regression_segs[i];

        if (hv > rv)
        {
            ++over_errors;
            over_error_indices.push_back(i);
        }

        else if (hv < rv)
        {
            ++miss_errors;
            miss_error_indices.push_back(i);
        }
    }

    ASSERT_EQ(0, over_errors + miss_errors);

    for (size_t i = 0; i < ngpus; ++i)
    {
        CUASSERT_NOERR(cudaFree(dev_segs[i]));
    }
}


TEST(Worklist, Ring_2)
{
    TestHistogramWorklist(2, 1024, 4096);
    TestHistogramWorklist(2, 1024, 20000);
    TestHistogramWorklist(2, 10000, 4096);
    TestHistogramWorklist(2, 10000, 200000);
}

TEST(Worklist, Ring_4)
{
    TestHistogramWorklist(4, 2048, 4096);
    TestHistogramWorklist(4, 2048, 20000);
    TestHistogramWorklist(4, 10000, 4096);
    TestHistogramWorklist(4, 10000, 200000);
}

TEST(Worklist, Ring_8)
{
    TestHistogramWorklist(8, 1024, 4096);
    TestHistogramWorklist(8, 1024, 20000);
    TestHistogramWorklist(8, 10000, 4096);
    TestHistogramWorklist(8, 10000, 200000);
}

TEST(Worklist, Ring_N)
{
    TestHistogramWorklist(3, 10000, 20000);
    TestHistogramWorklist(4, 10000, 20000);
    TestHistogramWorklist(5, 10000, 20000);
    TestHistogramWorklist(15, 10000, 20000);
    TestHistogramWorklist(27, 10000, 20000);
}
