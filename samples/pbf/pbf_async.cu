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
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>

#include <cuda_runtime.h>
#include <cub/util_ptx.cuh>

#include <gflags/gflags.h>

#include <groute/internal/worker.h> // for Barrier
#include <groute/groute.h>

#include <utils/markers.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>

DEFINE_string(input, "", "Input file to use");
DEFINE_string(output, "", "Save output to file");
DEFINE_string(dumpinput, "", "Dump generated input to another file");
DEFINE_string(dumpoutput, "", "Dump correct output to another file");

DEFINE_int32(seed, -1, "Random seed, or -1 for randomly-generated seed");
DEFINE_double(desired_ratio, 0.5, "Desired ratio for predicate generation");
DEFINE_uint64(size, 25 * 1024 * 1024, "Number of elements to filter");
DEFINE_uint64(chunksize, 1*1024*1024, "Chunk size (in bytes)");

DEFINE_uint64(pipeline, 9, "Number of receive operations to pipeline");

DEFINE_bool(regression, true, "Check values for correctness");
DEFINE_uint64(repetitions, 100, "Number of trials to average over");

DEFINE_uint64(num_gpus, 0, "Maximal number of GPUs to use");
DEFINE_uint64(startwith, 1, "Minimal number of GPUs to use");

DEFINE_bool(all, false, "Run all versions");
DEFINE_bool(single, false, "Run single-GPU version");
DEFINE_bool(async_multi, true, "Run async multi-GPU version");

///////////////////////////////////////////////////////////////////////
#define GTID (blockIdx.x * blockDim.x + threadIdx.x)

template<typename T>
struct ComplexPredicate
{
    static __host__ __device__ __forceinline__ bool Test(const T& value)
    {
      double a = exp(value);
      double b = sin(value);
      double c = cos(a * b) / (cos((b+3.0) / (a + 2.0)) + 4.0);

      double d = sin(c);
      return (double(value) >= d);
    }
};

template<typename T, typename Pred>
__global__ void Filter(const T * __restrict__ in, int in_size,
                       T * __restrict__ out, int * __restrict__ out_size)
{
    int tid = GTID;
    if (tid >= in_size) return;

    T value = in[tid];
    if (Pred::Test(value))
    {
        // Warp-aggregated filter
        int lanemask = __ballot_sync(__activemask(), 1);
        int leader = __ffs(lanemask) - 1;
        int thread_offset = __popc(lanemask & ((1 << (threadIdx.x & 31)) - 1));

        int ptr_offset;
        if ((threadIdx.x & 31) == leader)
            ptr_offset = atomicAdd(out_size, __popc(lanemask));
        ptr_offset = cub::ShuffleIndex<32>(ptr_offset, leader, __activemask());
        // End of warp-aggregated filter

        out[ptr_offset + thread_offset] = value;
    }
}

__global__ void GetItemCount(const int * __restrict__ in, volatile int * __restrict__ out)
{
    *out = *in;
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool RegressionTest(const std::vector<T>& out, const std::vector<T>& regression)
{
    if (out.size() != regression.size())
    {
        printf("ERROR: Output size mismatch (%d != %d)\n", out.size(), regression.size());
        return false;
    }

    int errctr = 0;
    std::vector<T> sorted_out = out;
    std::sort(sorted_out.begin(), sorted_out.end());
    for (size_t i = 0; i < regression.size(); ++i)
    {
        if (sorted_out[i] != regression[i]) // Can do bitwise floating point comparison since values are not changed
        {
            //printf("ERROR when comparing value at index %d: %f != %f\n", i, out[i], regression[i]);
            ++errctr;
        }
    }

    if (errctr > 0)
    {
        printf("ERROR in %d/%d values\n", errctr, regression.size());
        return false;
    }

    printf("Comparison OK (%llu outputs)\n", regression.size());
    return true;
}

template <typename T>
bool RunPBFSingleGPU(const std::vector<T>& in, const std::vector<T>& regression,
                     std::vector<T>& out)
{
    printf("Running PBF (Single GPU)\n");
    printf("========================\n");

    int sz = static_cast<int>(in.size());

    printf("Input items: %d\n", sz);
    
    dim3 block_dims (256);
    dim3 grid_dims (round_up(sz, block_dims.x));

    void *d_in = nullptr, *d_out = nullptr, *d_outsz = nullptr;

    cudaEvent_t ev;
    GROUTE_CUDA_CHECK(cudaEventCreate(&ev));
    GROUTE_CUDA_CHECK(cudaMalloc(&d_in, sizeof(T) * sz));
    GROUTE_CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * sz));
    GROUTE_CUDA_CHECK(cudaMalloc(&d_outsz, sizeof(int)));

    GROUTE_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(T) * sz));

    groute::pinned_vector<T> pinned_out;
    pinned_out.reserve(in.size());

    int outsz = 0;


    // Measure overall time
    Stopwatch sw (false);
    {
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        sw.start();

        for (uint64_t i = 0; i < FLAGS_repetitions; ++i)
        {
            IntervalRangeMarker rng(in.size(), "begin");

            GROUTE_CUDA_CHECK(cudaMemcpyAsync(d_in, in.data(), sizeof(T) * sz, cudaMemcpyHostToDevice));
            GROUTE_CUDA_CHECK(cudaMemsetAsync(d_outsz, 0, sizeof(int)));
            Marker::MarkWorkitems(in.size(), "Filter");

            Filter<T, ComplexPredicate<T>><<<grid_dims, block_dims>>>((const T*)d_in, 
                                                                    sz, 
                                                                    (T *)d_out, 
                                                                    (int *)d_outsz);
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(&outsz, d_outsz, sizeof(int), cudaMemcpyDeviceToHost));
            GROUTE_CUDA_CHECK(cudaEventRecord(ev));
            GROUTE_CUDA_CHECK(cudaEventSynchronize(ev));

            pinned_out.resize(outsz);
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(pinned_out.data(), d_out, sizeof(T) * outsz, cudaMemcpyDeviceToHost));
            GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        }
        sw.stop();
    }

    printf("PBF: %f ms. <filter>\n", sw.ms() / FLAGS_repetitions);

    // Free GPU memory
    GROUTE_CUDA_CHECK(cudaEventDestroy(ev));
    GROUTE_CUDA_CHECK(cudaFree(d_in));
    GROUTE_CUDA_CHECK(cudaFree(d_out));
    GROUTE_CUDA_CHECK(cudaFree(d_outsz));

    out.clear();
    out.insert(out.begin(), pinned_out.begin(), pinned_out.end());
    
    // Verify output
    if (FLAGS_regression)
    {
        return RegressionTest(out, regression);
    }

    return true;
}


template <typename T>
bool RunPBFConfiguration(int ngpus, const std::vector<T>& in, const std::vector<T>& regression,
                         std::vector<T>& out)
{
    printf("\nRunning PBF with %d GPUs\n", ngpus);
    printf("========================\n");

    int sz = static_cast<int>(in.size());

    printf("Input items: %d\n", sz);

    groute::Context ctx(ngpus);

    auto gpu_work = [&](groute::device_t device, size_t maxout,
                        groute::router::Router<T>& scatter,
                        groute::router::Router<T>& gather,
                        groute::internal::Barrier& barrier) {
        groute::Stream stm(ctx.GetDevId(device));
        groute::Link<T> sock_in(scatter, device, maxout, FLAGS_pipeline);
        groute::Link<T> sock_out(device, gather, maxout, FLAGS_pipeline);

        dim3 block_dims(256);

        int *d_outsz = nullptr;

        // Initialize buffers
        GROUTE_CUDA_CHECK(cudaMalloc(&d_outsz, sizeof(int)));
        GROUTE_CUDA_CHECK(cudaMemset(d_outsz, 0, sizeof(int)));

        size_t total_input = 0;
        size_t total_processed = 0;
        int *outsz = nullptr;
        int *d_h_outsz = nullptr;
        GROUTE_CUDA_CHECK(cudaMallocHost(&outsz, sizeof(int)));
        GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&d_h_outsz, outsz, 0));

        // For timing
        GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        barrier.Sync();

        RangeMarker range(true, "Total GPU work");

        // Work thread
        while (true) 
        {
            groute::router::PendingSegment<T> seg = sock_in.Receive().get();
            if (seg.Empty()) break;
            groute::Segment<T> outseg = sock_out.GetSendBuffer();

            total_input += seg.GetSegmentSize();
            seg.Wait(stm.cuda_stream);    

            Marker::MarkWorkitems(seg.GetSegmentSize(), "Filter");

            dim3 grid_dims(round_up(seg.GetSegmentSize(), block_dims.x));
            cudaMemsetAsync(d_outsz, 0, sizeof(int), stm.cuda_stream);
            Filter<T, ComplexPredicate<T>> <<<grid_dims, block_dims, 0, 
                                               stm.cuda_stream>>>(seg.GetSegmentPtr(),
                                                                  seg.GetSegmentSize(),
                                                                  outseg.GetSegmentPtr(), 
                                                                  d_outsz);
            GetItemCount<<<1,1,0,stm.cuda_stream>>>(d_outsz, d_h_outsz);
            groute::Event ev = ctx.RecordEvent(device, stm.cuda_stream);

            ev.Sync(); // To obtain outsz

            total_processed += *outsz;

            auto sendevf = sock_out.Send(groute::Segment<T>(outseg.GetSegmentPtr(), *outsz), ev);

            sock_out.ReleaseSendBuffer(outseg, sendevf.get());
            sock_in.ReleaseBuffer(seg, groute::Event());
        }

        sock_out.Shutdown();
        
        range.Stop();
        barrier.Sync();

        printf("GPU%d: inputs: %llu, outputs: %llu\n", device, total_input, total_processed);

        cudaFree(d_outsz);
        cudaFreeHost(outsz);
    };
    ////////////////////////////////////////

    groute::router::Router<T> scatter(ctx, 
        groute::router::Policy::CreateScatterPolicy(groute::Device::Host, range(ngpus)));
    groute::router::Router<T> gather(ctx,
        groute::router::Policy::CreateGatherPolicy(groute::Device::Host, range(ngpus)));
    size_t chunksize = FLAGS_chunksize;

    groute::Link<T> dist(groute::Device::Host, scatter, chunksize, 1);
    groute::Link<T> collect(gather, groute::Device::Host, chunksize, 2 * ngpus);

    groute::internal::Barrier bar(ngpus + 1);




    // Start GPU work threads
    for (groute::device_t dev = 0; dev < ngpus; ++dev)
    {   
        std::thread tdev(gpu_work, dev,
                         chunksize, std::ref(scatter), 
                         std::ref(gather), std::ref(bar));
        tdev.detach();
    }

    // Convert to pinned memory
    groute::pinned_vector<T> pinned_in (in.size()), pinned_out;
    pinned_in.insert(pinned_in.begin(), in.begin(), in.end());

    // Inflate output
    size_t offset = 0;
    pinned_out.resize(in.size());


    // Measure overall time
    Stopwatch sw (false);
    {
        bar.Sync();
        sw.start();

        // CPU Controller
        IntervalRangeMarker cpurange(in.size(), "begin");

        // Distribute work
        dist.Send(groute::Segment<T>(pinned_in.data(), in.size()), groute::Event());
        dist.Shutdown();
        
        // Aggregate segments one by one
        while (true)
        {
            groute::router::PendingSegment<T> seg = collect.Receive().get();
            if (seg.Empty()) break;

            seg.Sync();

            // Copy data and release buffer
            memcpy(&pinned_out[offset], seg.GetSegmentPtr(), seg.GetSegmentSize() * sizeof(T));

            offset += seg.GetSegmentSize();


            collect.ReleaseBuffer(seg, groute::Event());
        }
        /////////////////

        bar.Sync();
        cpurange.Stop();
        sw.stop();
    }

    printf("PBF: %f ms. <filter>\n", sw.ms());

    out.clear();
    out.insert(out.begin(), pinned_out.begin(), pinned_out.begin() + offset);
    
    // Verify output
    if (FLAGS_regression)
    {
        return RegressionTest(out, regression);
    }

    return true;
}

template <typename T>
bool ReadFile(const std::string& fname, std::vector<T>& result)
{
    FILE *fp = fopen(fname.c_str(), "rb");
    if (fp)
    {
        // ASCII version
        /*
        float val;
        while (fscanf(fp, "%f", &val) > 0)
            result.push_back(val);
        */
        // Binary version
        fseek(fp, 0L, SEEK_END);
        long len = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        result.resize(len / sizeof(T));

        fread(result.data(), sizeof(T), result.size(), fp); 
        
        fclose(fp);
    }
    else
    {
        printf("ERROR opening input file %s\n", fname.c_str());
        return false;
    }

    return true;
}

template <typename T>
bool WriteFile(const std::string& fname, const std::vector<T>& data)
{
    FILE *fp = fopen(fname.c_str(), "wb");
    if (fp)
    {
        // ASCII
        /*
        for (size_t i = 0; i < data.size(); ++i)
            fprintf(fp, "%f\n", data[i]);
        */

        // Binary
        fwrite(data.data(), sizeof(T), data.size(), fp);
        
        fclose(fp);
       
        return true;
    }
    return false;
}

bool RunPBF()
{
    bool result = true;

    int ndevs = 0;
    GROUTE_CUDA_CHECK(cudaGetDeviceCount(&ndevs));
    if (ndevs == 0)
    {
        printf("ERROR: No GPUs found\n");
        return false;
    }
    
    if (FLAGS_num_gpus == 0)
        FLAGS_num_gpus = ndevs;

    // Flag checks
    if (FLAGS_all)
    {
        FLAGS_async_multi = true;
        FLAGS_single = true;
    }


    if (FLAGS_startwith > FLAGS_num_gpus)
    {
        printf("ERROR: --startwith must be in the range of [1,%llu]\n", FLAGS_num_gpus);
        return false;
    }

    std::vector<float> inputs, regression, outputs;
    
    // Input from external file
    if (FLAGS_input.length() > 0)
    {
        if (!ReadFile(FLAGS_input, inputs))
            return false;
    }
    else // Input generation
    {
        inputs.resize(FLAGS_size);        
        
        // Random
        std::random_device rd;
        std::mt19937 gen(FLAGS_seed < 0 ? rd() : FLAGS_seed);

        // Set range according to desired ratio
        float maxval = (float)FLAGS_desired_ratio;
        std::uniform_real_distribution<float> dis(maxval - 1.0f, maxval);

        // Create random values
        for (size_t i = 0; i < FLAGS_size; ++i)
            inputs[i] = dis(gen);

        if (FLAGS_dumpinput.length() > 0)
            WriteFile(FLAGS_dumpinput, inputs);
    }

    // Apply predicate to generate regression data
    if (FLAGS_regression)
    {
        regression.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i)
            if (ComplexPredicate<float>::Test(inputs[i])) 
                regression.push_back(inputs[i]);
        std::sort(regression.begin(), regression.end());
        if (FLAGS_dumpoutput.length() > 0)
            WriteFile(FLAGS_dumpoutput, regression);
    }

    // Running single-GPU configuration
    if (FLAGS_single)
        result &= RunPBFSingleGPU<float>(inputs, regression, outputs);

    // Running for each GPU configuration
    if (FLAGS_async_multi)
    {
        for (uint64_t devs = FLAGS_startwith; devs <= FLAGS_num_gpus; ++devs)
            result &= RunPBFConfiguration<float>(static_cast<int>(devs), inputs, regression, outputs);
    }
    
    // Save last output to file
    if (FLAGS_output.length() > 0)
    {
        std::sort(outputs.begin(), outputs.end());
        WriteFile(FLAGS_output, outputs);
    }

    return result;
}
