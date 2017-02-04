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
#ifndef __CC_COMMON_H
#define __CC_COMMON_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>

#include <groute/groute.h>
#include <groute/internal/cuda_utils.h>
#include <groute/graphs/coo_graph.h>

#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include "cc_kernels.cuh"
#include "cc_config.h"
#include "cc_context.h"
#include "cc_partitioner.h"

namespace cc
{
    struct Problem // a per device cc problem
    {
        const Context& context;

        int dev;
        dim3 block_dims;
        size_t compressed_size;

        Partition partition;
        groute::Segment<int> parents;

        cudaStream_t    compute_stream;
        cudaEvent_t     sync_event;

        Problem(const Problem& other) = delete;
        Problem(Problem&& other) = delete;

        Problem(
            const Context& context, 
            const Partition& partition, int dev, dim3 block_dims) :
            context(context), partition(partition), 
            dev(dev), block_dims(block_dims), compressed_size(0)
        {
            size_t parents_ss; // The parent segment size
            size_t parents_so; // The parent segment offset

            parents_ss = partition.parents_segment.GetSegmentSize();
            parents_so = partition.parents_segment.GetSegmentOffset();

            int* parents_ptr;

            context.SetDevice(dev);
            GROUTE_CUDA_CHECK(cudaMalloc(&parents_ptr, parents_ss * sizeof(int)));
            GROUTE_CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
            GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));

            parents = groute::Segment<int>(parents_ptr, context.nvtxs, parents_ss, parents_so);
        }

        ~Problem()
        {
            context.SetDevice(dev);
            GROUTE_CUDA_CHECK(cudaFree(parents.GetSegmentPtr()));
            GROUTE_CUDA_CHECK(cudaStreamDestroy(compute_stream));
            GROUTE_CUDA_CHECK(cudaEventDestroy(sync_event));
        }

        void Init() const
        {
            dim3 grid_dims(round_up(parents.GetSegmentSize(), block_dims.x), 1, 1);

            InitParents <<< grid_dims, block_dims, 0, compute_stream >>> (
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), parents.GetSegmentSize(), parents.GetSegmentOffset()));
        }

        template<bool R1 = false>
        void Work(const groute::router::PendingSegment<Edge>& edge_seg)
        {
            dim3 grid_dims(round_up(edge_seg.GetSegmentSize(), block_dims.x), 1, 1);

            edge_seg.Wait(compute_stream); // queue a wait on the compute stream  

            Marker::MarkWorkitems(edge_seg.GetSegmentSize(), "Hook");

            Hook <groute::graphs::dev::EdgeList, R1> <<< grid_dims, block_dims, 0, compute_stream >>>(
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), parents.GetSegmentSize(), parents.GetSegmentOffset()),
                groute::graphs::dev::EdgeList(edge_seg.GetSegmentPtr(), edge_seg.GetSegmentSize())
                );
            compressed_size = 0;
        }

        void WorkAtomic(const groute::router::PendingSegment<Edge>& edge_seg)
        {
            dim3 grid_dims(round_up(edge_seg.GetSegmentSize(), block_dims.x), 1, 1);
            
            edge_seg.Wait(compute_stream); // queue a wait on the compute stream  

            Marker::MarkWorkitems(edge_seg.GetSegmentSize(), "HookHighToLowAtomic");

            HookHighToLowAtomic <groute::graphs::dev::EdgeList> <<< grid_dims, block_dims, 0, compute_stream >>>(
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), parents.GetSegmentSize(), parents.GetSegmentOffset()),
                groute::graphs::dev::EdgeList(edge_seg.GetSegmentPtr(), edge_seg.GetSegmentSize())
                );
            compressed_size = 0;
        }

        void Merge(const groute::router::PendingSegment<int>& merge_seg)
        {
            assert(merge_seg.GetSegmentOffset() >= parents.GetSegmentOffset());
            assert(merge_seg.GetSegmentOffset() - parents.GetSegmentOffset() + 
                merge_seg.GetSegmentSize() <= parents.GetSegmentSize()); 

            dim3 grid_dims(round_up(merge_seg.GetSegmentSize(), block_dims.x), 1, 1);

            merge_seg.Wait(compute_stream); // queue a wait on the compute stream  

            Marker::MarkWorkitems(merge_seg.GetSegmentSize(), "HookHighToLowAtomic");

            // Note the Tree graph and the vertex grid dims
            HookHighToLowAtomic <groute::graphs::dev::Tree> <<< grid_dims, block_dims, 0, compute_stream >>> (
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), parents.GetSegmentSize(), parents.GetSegmentOffset()),
                groute::graphs::dev::Tree(merge_seg.GetSegmentPtr(), merge_seg.GetSegmentSize(), merge_seg.GetSegmentOffset())
                );
            compressed_size = 0;
        }

        void Compress()
        {
            dim3 grid_dims(round_up(parents.GetSegmentSize(), block_dims.x), 1, 1);

            Marker::MarkWorkitems(parents.GetSegmentSize(), "MultiJumpCompress");

            MultiJumpCompress <<< grid_dims, block_dims, 0, compute_stream >>> (
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), parents.GetSegmentSize(), parents.GetSegmentOffset()));

            compressed_size = parents.GetSegmentSize(); // keep the compress size 
        }

        void CompressLocal()
        {
            size_t size = partition.local_upper_bound - parents.GetSegmentOffset();

            assert(size <= parents.GetSegmentSize());
            assert(size > 0);

            dim3 grid_dims(round_up(size, block_dims.x), 1, 1);

            Marker::MarkWorkitems(size, "MultiJumpCompress");


            MultiJumpCompress <<< grid_dims, block_dims, 0, compute_stream >>> (
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), size, parents.GetSegmentOffset()));

            compressed_size = size; // keep the compress size 
        }

        void Compress(const groute::router::PendingSegment<int>& merged_seg)
        {
            // makes the decision what part to compress after a merge  
            // compresses from begining of local parents and up to covering the merged segment  

            assert(merged_seg.GetSegmentOffset() >= parents.GetSegmentOffset());

            size_t offset_diff = merged_seg.GetSegmentOffset() - parents.GetSegmentOffset();
            size_t size = offset_diff + merged_seg.GetSegmentSize();

            assert(size <= parents.GetSegmentSize());

            dim3 grid_dims(round_up(size, block_dims.x), 1, 1);

            Marker::MarkWorkitems(size, "MultiJumpCompress");


            MultiJumpCompress <<< grid_dims, block_dims, 0, compute_stream >>> (
                groute::graphs::dev::Irregular<int>(parents.GetSegmentPtr(), size, parents.GetSegmentOffset()));

            compressed_size = size; // keep the compress size 
        }

        void Finish()
        {
            if (compressed_size == parents.GetSegmentSize()) return;
            Compress(); // make sure the entire local segment is compressed   
        }

        groute::Event Record() const
        {
            return context.RecordEvent(dev, compute_stream);
        }

        void Sync() const
        {
            GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, compute_stream));
            GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
        }
    };

    struct Solver // a per device cc solver  
    {
        const Context& context;
        Problem& problem;

        groute::Link<Edge> edges_in;
        groute::Link<component_t> reduction_in;
        groute::Link<component_t> reduction_out;

        Solver(const Solver& other) = delete;
        Solver(Solver&& other) = delete;

        Solver(const Context& context, Problem& problem) :
            context(context), problem(problem)
        {
        }

        void Solve(const Configuration& configuration)
        {
            context.SetDevice(problem.dev);

            // Init
            problem.Init();
            
            auto input_fut = edges_in.Receive();
            auto reduce_fut = reduction_in.Receive();

            groute::router::PendingSegment<int> merge_seg;
            groute::router::PendingSegment<Edge> input_seg;

            for (int i = 0;; ++i)
            {
                input_seg = input_fut.get();
                if (input_seg.Empty()) break;

                for (int ii = 0; ii < configuration.nonatomic_rounds; ++ii) // non atomic rounds  
                {
                    IntervalRangeMarker iter_rng(input_seg.GetSegmentSize(), "CC iteration (non-atomic)");

                    if (i == 0 && ii == 0)  problem.Work<true>(input_seg); // round 1
                    else                    problem.Work<false>(input_seg);

                    problem.CompressLocal(); 
                        // for non atomic rounds we must compress any how  
                        // compress only up to local bounds, everything else will get compressed later   
                }

                IntervalRangeMarker iter_rng(input_seg.GetSegmentSize(), "CC iteration (atomic)");


                // Work atomic
                problem.WorkAtomic(input_seg);

                edges_in.ReleaseBuffer(input_seg, problem.Record()); // dismiss depends on the recorded event  
                input_fut = edges_in.Receive();

                problem.Compress();   

                // Sync CPU thread every odd segment to obtain   
                // dynamic load balancing for input segments    
                // NOTE: figure out a better approach  
                if(configuration.load_balancing && (i%2) == 1) problem.Sync(); 
            }

            for (int i = 0;; ++i)
            {
                merge_seg = reduce_fut.get();
                if (merge_seg.Empty()) break;
                IntervalRangeMarker iter_rng(merge_seg.GetSegmentSize(), "CC merge iteration");


                // Merge
                problem.Merge(merge_seg);

                reduction_in.ReleaseBuffer(merge_seg, problem.Record()); // dismiss depends on the recorded event  
                reduce_fut = reduction_in.Receive();
            }

            // Makes sure the entire local segment is compressed  
            problem.Finish();

            // Send the final distribution to peers
            reduction_out.Send(problem.parents, problem.Record());
            reduction_out.Shutdown();

            problem.Sync();
        }
    };
}

#endif // __CC_COMMON_H
