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

#include <groute/event_pool.h>
#include <groute/dwl/distributed_worklist.cuh>
#include <groute/dwl/workers.cuh>

#include <utils/cuda_utils.h>


namespace histogram
{
    struct CountWork
    {
        template<
            typename WorkSource, typename WorkTarget>
            __device__ static void work(
            const WorkSource& work_source, WorkTarget& work_target,
            int offset, int *bins
            )
        {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;

            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
            {
                atomicAdd(bins + work_source.get_work(i) - offset, 1);
            }
        }
    };

    struct DWCallbacks
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
        
        __device__ __forceinline__ bool should_defer(int work, const int& global_threshold)
        {
            return false;
        }
        
        __device__ __host__ DWCallbacks(int split_seg_index, int split_seg_size)
            : m_seg_index(split_seg_index), m_seg_size(split_seg_size)
        {
        }

        DWCallbacks() : m_seg_index(-1), m_seg_size(-1) { }

    private:
        int m_seg_index;
        int m_seg_size;
    };
}

template<typename TWorker>
void TestHistogramWorklist(int ngpus, size_t histo_size, size_t work_size)
{
    size_t histo_seg_size = histo_size / ngpus;
    histo_size = histo_seg_size * ngpus;

    ASSERT_GT(histo_seg_size, 0);

    size_t num_exch_buffs = 4 * ngpus;
    size_t exch_packet_size = work_size; 

    groute::Context context(ngpus);
    context.configuration.verbose = false;
    context.configuration.trace = false;

    groute::Endpoint host = groute::Endpoint::HostEndpoint(0);

    srand(static_cast <unsigned> (22522));
    std::vector<int> initial_work;
    for (size_t ii = 0, count = work_size; ii < count; ++ii)
    {
        initial_work.push_back((rand()*round_up(histo_size, RAND_MAX)) % histo_size);
    }

    std::vector<int*> dev_segs(ngpus);

    for (int i = 0; i < ngpus; ++i)
    {
        context.SetDevice(i);

        CUASSERT_NOERR(cudaMalloc(&dev_segs[i], histo_seg_size * sizeof(int)));
        CUASSERT_NOERR(cudaMemset(dev_segs[i], 0, histo_seg_size * sizeof(int)));
    }

    // Prepare DistributedWorklist parameters
    groute::EndpointList worker_endpoints = groute::Endpoint::Range(ngpus);
    std::map<groute::Endpoint, histogram::DWCallbacks> callbacks;
    for (int i = 0; i < ngpus; ++i)
    {
        callbacks[worker_endpoints[i]] = histogram::DWCallbacks(i, histo_seg_size);
    }

    groute::DistributedWorklist<int, int, histogram::DWCallbacks, TWorker> 
        distributed_worklist(context, { host }, worker_endpoints, callbacks, exch_packet_size, num_exch_buffs, 0);

    distributed_worklist.ReportInitialWork(initial_work.size(), host);
    distributed_worklist
        .GetLink(host)
        .Send(groute::Segment<int>(&initial_work[0], initial_work.size()), groute::Event());

    std::vector<std::thread> workers;
    groute::internal::Barrier barrier(ngpus);

    for (int i = 0; i < ngpus; ++i)
    {
        std::thread worker([&, i]()
        {
            context.SetDevice(i);
            groute::Stream stream = context.CreateStream(i);

            barrier.Sync();

            //
            // Start processing  
            //

            // Loop over the work until convergence  
            distributed_worklist.Work(i, stream, i*histo_seg_size, dev_segs[i]);

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

    for (auto it : initial_work)
    {
        ++regression_segs[it];
    }

    for (int i = 0; i < ngpus; ++i)
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

    
typedef groute::FusedWorker<true, int, int, int, histogram::DWCallbacks, histogram::CountWork, int, int*> FusedWorkerType;
typedef groute::Worker<int, int, histogram::DWCallbacks, histogram::CountWork, int, int*> WorkerType;


TEST(DWL, Worker_2)
{
    TestHistogramWorklist<WorkerType>(2, 1024, 4096);
    TestHistogramWorklist<WorkerType>(2, 1024, 20000);
    TestHistogramWorklist<WorkerType>(2, 10000, 4096);
    TestHistogramWorklist<WorkerType>(2, 10000, 200000);
}

TEST(DWL, Worker_4)
{
    TestHistogramWorklist<WorkerType>(4, 2048, 4096);
    TestHistogramWorklist<WorkerType>(4, 2048, 20000);
    TestHistogramWorklist<WorkerType>(4, 10000, 4096);
    TestHistogramWorklist<WorkerType>(4, 10000, 200000);
}

TEST(DWL, Worker_8)
{
    TestHistogramWorklist<WorkerType>(8, 1024, 4096);
    TestHistogramWorklist<WorkerType>(8, 1024, 20000);
    TestHistogramWorklist<WorkerType>(8, 10000, 4096);
    TestHistogramWorklist<WorkerType>(8, 10000, 200000);
}

TEST(DWL, Worker_N)
{
    TestHistogramWorklist<WorkerType>(3, 10000, 20000);
    TestHistogramWorklist<WorkerType>(4, 10000, 20000);
    TestHistogramWorklist<WorkerType>(5, 10000, 20000);
    TestHistogramWorklist<WorkerType>(15, 10000, 20000);
    TestHistogramWorklist<WorkerType>(27, 10000, 20000);
}


TEST(DWL, FusedWorker_2)
{
    TestHistogramWorklist<FusedWorkerType>(2, 1024, 4096);
    TestHistogramWorklist<FusedWorkerType>(2, 1024, 20000);
    TestHistogramWorklist<FusedWorkerType>(2, 10000, 4096);
    TestHistogramWorklist<FusedWorkerType>(2, 10000, 200000);
}

TEST(DWL, FusedWorker_4)
{
    TestHistogramWorklist<FusedWorkerType>(4, 2048, 4096);
    TestHistogramWorklist<FusedWorkerType>(4, 2048, 20000);
    TestHistogramWorklist<FusedWorkerType>(4, 10000, 4096);
    TestHistogramWorklist<FusedWorkerType>(4, 10000, 200000);
}

TEST(DWL, FusedWorker_8)
{
    TestHistogramWorklist<FusedWorkerType>(8, 1024, 4096);
    TestHistogramWorklist<FusedWorkerType>(8, 1024, 20000);
    TestHistogramWorklist<FusedWorkerType>(8, 10000, 4096);
    TestHistogramWorklist<FusedWorkerType>(8, 10000, 200000);
}

TEST(DWL, FusedWorker_N)
{
    TestHistogramWorklist<FusedWorkerType>(3, 10000, 20000);
    TestHistogramWorklist<FusedWorkerType>(4, 10000, 20000);
    TestHistogramWorklist<FusedWorkerType>(5, 10000, 20000);
    TestHistogramWorklist<FusedWorkerType>(15, 10000, 20000);
    TestHistogramWorklist<FusedWorkerType>(27, 10000, 20000);
}