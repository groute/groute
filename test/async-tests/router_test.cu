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
#include <cmath>
#include <gtest/gtest.h>

#include "cuda_gtest_utils.h"
#include "test_common.h"

#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/event_pool.h>
#include <groute/groute.h>

__global__ void AddKernel(int *buffer, int count, int add)
{
    int tid = GTID;
    if (tid < count)
    {
        buffer[tid] += add;
    }
}

__global__ void SumKernel(int *buffer, int count, int *sum)
{
    int tid = GTID;
    if (tid < count)
    {
        atomicAdd(sum, buffer[tid]);
    }
}

__global__ void HistogramKernel(int *buffer, int count, int *bins)
{
    int tid = GTID;
    if (tid < count)
    {
        atomicAdd(bins + buffer[tid], 1);
    }
}

__global__ void MergeBinsKernel(int *bins1, int *bins2, int bins_count)
{
    int tid = GTID;
    if (tid < bins_count)
    {
        bins1[tid] += bins2[tid];
    }
}

void FragmentedCopy(size_t buffer_size, size_t mtu)
{
    CUASSERT_NOERR(cudaSetDevice(0));

    std::vector<int> host(buffer_size);
    for (size_t i = 0; i < host.size(); ++i)
    {
        host[i] = i;
    }

    // The event manager
    groute::Context context;

    int *dev;
    CUASSERT_NOERR(cudaMalloc(&dev, host.size() * sizeof(int)));

    cudaStream_t stream;
    CUASSERT_NOERR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    dim3 block_dims(512, 1, 1);
    dim3 grid_dims(round_up(host.size(), block_dims.x), 1, 1);

    auto copy = std::make_shared<groute::MemcpyWork>(context.GetEventPool(0), mtu);

    copy->src_dev_id = groute::Device::Host;
    copy->src_buffer = &host[0];
    copy->copy_bytes = host.size() * sizeof(int);

    copy->dst_dev_id = 0;
    copy->dst_buffer = dev;
    copy->dst_size = host.size() * sizeof(int);

    std::promise<groute::Event> promise;
    std::future<groute::Event> future = promise.get_future();

    copy->completion_callback = 
        [&promise](size_t bytes, const groute::Event& ev)
        {
            promise.set_value(ev);
        };

    groute::MemcpyWorker bus(0);
    bus.Enqueue(copy);

    auto res = future.get(); // blocking until callback is called  
    res.Wait(stream);

    AddKernel <<< grid_dims, block_dims, 0, stream >>> (dev, host.size(), 1);

    CUASSERT_NOERR(
        cudaMemcpyAsync(
        &host[0], dev, host.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUASSERT_NOERR(cudaStreamSynchronize(stream));

    int errors = 0;
    for (size_t i = 0; i < host.size(); ++i)
    {
        if (host[i] != i + 1) ++errors;
    }

    ASSERT_EQ(0, errors);

    CUASSERT_NOERR(cudaStreamDestroy(stream));
    CUASSERT_NOERR(cudaFree(dev));
}

void H2DevsRouting(int ngpus, int buffer_size, int chunk_size, int fragment_size)
{
    groute::pinned_vector<int> host_buffer(buffer_size);
    for (size_t i = 0; i < host_buffer.size(); ++i)
    {
        host_buffer[i] = i % 100;
    }

    groute::Context context(ngpus);

    if (fragment_size > 0)
        context.EnableFragmentation(fragment_size);

    groute::router::Router<int> router(context, 
        groute::router::Policy::CreateScatterPolicy(groute::Device::Host, range(ngpus)));

    groute::router::ISender<int>* host_sender = router.GetSender(groute::Device::Host); 

    std::vector<int*> dev_sums(ngpus);
    std::vector< std::unique_ptr< groute::router::IPipelinedReceiver<int> > > dev_receivers;
    std::vector< std::thread > dev_threads;

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Init dev sums
        context.SetDevice(i);
        CUASSERT_NOERR(cudaMalloc(&dev_sums[i], sizeof(int)));
        CUASSERT_NOERR(cudaMemset(dev_sums[i], 0, sizeof(int)));

        // Init dev receivers  
        auto receiver = router.CreatePipelinedReceiver(i, chunk_size, 3);
        dev_receivers.push_back(std::move(receiver));
    }

    host_sender->Send(groute::Segment<int>(&host_buffer[0], buffer_size, buffer_size, 0), groute::Event());
    host_sender->Shutdown();

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Sync (for pre loading case)
        dev_receivers[i]->Sync();

        // Run threads  
        std::thread dev_worker([&, i]()
        {
            context.SetDevice(i);
            groute::Stream stream = context.CreateStream(i);

            while (true)
            {
                auto fut = dev_receivers[i]->Receive();
                auto seg = fut.get();
                if (seg.Empty()) break;

                dim3 block_dims(64, 1, 1);
                dim3 grid_dims(round_up(seg.GetSegmentSize(), block_dims.x), 1, 1);

                // queue a wait on stream
                seg.Wait(stream.cuda_stream);

                SumKernel <<< grid_dims, block_dims, 0, stream.cuda_stream >>>
                    (seg.GetSegmentPtr(), seg.GetSegmentSize(), dev_sums[i]);

                dev_receivers[i]->ReleaseBuffer(seg, context.RecordEvent(i, stream.cuda_stream));
            }

            stream.Sync();
        });

        dev_threads.push_back(std::move(dev_worker));
    }

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Join threads  
        dev_threads[i].join();
    }

    int aggregated_sum = 0;
    for (size_t i = 0; i < ngpus; ++i)
    {
        int ss;
        CUASSERT_NOERR(cudaMemcpy(&ss, dev_sums[i], sizeof(int), cudaMemcpyDeviceToHost));
        CUASSERT_NOERR(cudaFree(dev_sums[i]));
        aggregated_sum += ss;
    }

    int sum = 0;
    for (size_t i = 0; i < host_buffer.size(); ++i)
    {
        sum += host_buffer[i];
    }

    ASSERT_EQ(sum, aggregated_sum);
}

void P2PDevsRouting(int ngpus, int buffer_size, int chunk_size, int fragment_size)
{
    int bins = 1000;

    // Building an histogram with inter-device reduction.
    // Each device builds its local histogram from dynamic host chunks
    // and then distributes the histogram to peers. Peer devices merge the segments from 
    // the remote histogram into their local histogram.  
    // Based on correct topology, the aggregated histogram should converge on a single device (0)

    groute::pinned_vector<int> host_input(buffer_size);
    std::vector<int> expected_bins(bins);
    srand(static_cast <unsigned> (2222));

    for (size_t i = 0; i < host_input.size(); ++i)
    {
        host_input[i] = rand() % bins;
        ++expected_bins[host_input[i]];
    }

    groute::Context context(ngpus);

    if (fragment_size > 0)
        context.EnableFragmentation(fragment_size);

    groute::router::Router<int> input_router(context, groute::router::Policy::CreateScatterPolicy(groute::Device::Host, range(ngpus)));
    groute::router::Router<int> reduction_router(context, groute::router::Policy::CreateOneWayReductionPolicy(ngpus));
    
    groute::router::ISender<int>* host_sender = input_router.GetSender(groute::Device::Host); 
    groute::router::IReceiver<int>* host_receiver = reduction_router.GetReceiver(groute::Device::Host); // TODO

    std::vector<int*> dev_bins(ngpus);
    std::vector< groute::Link<int> > input_links;
    std::vector< groute::Link<int> > reduction_in_links;
    std::vector< groute::Link<int> > reduction_out_links;
    std::vector< std::thread > dev_threads;

    for (size_t i = 0; i < ngpus; ++i)
    {
        context.SetDevice(i);
        CUASSERT_NOERR(cudaMalloc(&dev_bins[i], bins * sizeof(int)));
        CUASSERT_NOERR(cudaMemset(dev_bins[i], 0, bins * sizeof(int)));

        input_links.push_back(groute::Link<int>(input_router, i, chunk_size, 4));

        reduction_in_links.push_back(groute::Link<int>(reduction_router, i, chunk_size, 3));
        reduction_out_links.push_back(groute::Link<int>(i, reduction_router));
    }

    host_sender->Send(groute::Segment<int>(&host_input[0], buffer_size, buffer_size, 0), groute::Event());
    host_sender->Shutdown();

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Run threads  
        std::thread dev_worker([&, i]()
        {
            context.SetDevice(i);
            groute::Stream stream = context.CreateStream(i);

            // First, add all input segments into the local histogram  
            while (true)
            {
                auto fut = input_links[i].Receive();
                auto seg = fut.get();
                if (seg.Empty()) break;

                dim3 block_dims(64, 1, 1);
                dim3 grid_dims(round_up(seg.GetSegmentSize(), block_dims.x), 1, 1);

                // queue a wait on stream
                seg.Wait(stream.cuda_stream);

                HistogramKernel <<< grid_dims, block_dims, 0, stream.cuda_stream >>>
                    (seg.GetSegmentPtr(), seg.GetSegmentSize(), dev_bins[i]);

                input_links[i].ReleaseBuffer(seg, context.RecordEvent(i, stream.cuda_stream));
            }

            auto fut = reduction_in_links[i].Receive();

            // Merge histogram segments
            while (true)
            {
                auto seg = fut.get();
                if (seg.Empty()) break;

                dim3 block_dims(64, 1, 1);
                dim3 grid_dims(round_up(seg.GetSegmentSize(), block_dims.x), 1, 1);

                // queue a wait on stream
                seg.Wait(stream.cuda_stream);

                MergeBinsKernel <<< grid_dims, block_dims, 0, stream.cuda_stream >>>
                    (dev_bins[i] + seg.GetSegmentOffset(), seg.GetSegmentPtr(), seg.GetSegmentSize());

                reduction_in_links[i].ReleaseBuffer(seg, context.RecordEvent(i, stream.cuda_stream));
                fut = reduction_in_links[i].Receive();
            }

            // Distribute the local segment to peers 
            reduction_out_links[i].Send(
                groute::Segment<int>(dev_bins[i], bins, bins, 0), 
                context.RecordEvent(i, stream.cuda_stream));
            reduction_out_links[i].Shutdown();

            stream.Sync();
        });

        dev_threads.push_back(std::move(dev_worker));
    }

    for (size_t i = 0; i < ngpus; ++i)
    {
        // Join threads  
        dev_threads[i].join();
    }

    std::vector<int> host_bins(bins);
    host_receiver->Receive(groute::Buffer<int>(host_bins.data(), bins), groute::Event()).wait();

    int errors = 0;
    for (size_t i = 0; i < bins; ++i)
    {
        if (host_bins[i] != expected_bins[i]) ++errors;
    }

    ASSERT_EQ(0, errors);

    for (size_t i = 0; i < ngpus; ++i)
    {
        CUASSERT_NOERR(cudaFree(dev_bins[i]));
    }
}

TEST(Async, FragmentedCopy)
{
    FragmentedCopy(256, 1024);
    FragmentedCopy(1024, 1024);
    FragmentedCopy(256 * 1024, 1024);
    FragmentedCopy(256 * 1024, 256 * 1024);
}

TEST(Async, H2DevsRouter)
{
    H2DevsRouting(1, 256 * 1024, 1024, -1);
    H2DevsRouting(2, 256 * 1024, 1024, -1);
    H2DevsRouting(2, 256 * 1024, 1024, 64);
    H2DevsRouting(3, 256 * 1024, 1000, 115);
    H2DevsRouting(4, 256 * 1024, 1024, -1);
}

TEST(Async, P2PDevsRouter)
{
    P2PDevsRouting(2, 256 * 1024, 1024, -1);
    P2PDevsRouting(2, 256 * 1024, 1024, 64);
    P2PDevsRouting(3, 256 * 1024, 1000, 115);
    P2PDevsRouting(4, 256 * 1024, 1024, -1);
}