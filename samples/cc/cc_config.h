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
#ifndef __CC_CONFIG_H
#define __CC_CONFIG_H


#include <vector>
#include <algorithm>

#include <groute/common.h>

#include <utils/utils.h>

namespace cc
{
    struct Configuration
    {
        bool vertex_partitioning;
        bool load_balancing;

        size_t edges_chunk_size;    // number of edges in each input chunk 
        size_t parents_chunk_size;  // number of parent entries at each merge chunk

        size_t input_pipeline_buffers;    // number of input pipeline buffers per device
        size_t reduction_pipeline_buffers; // number of reduction pipeline buffers per device

        int nonatomic_rounds;   // number of non atomic rounds per input segment 
        double compute_latency_ratio; // Hint for the (compute_time/memory_latency) ratio

        void print() const
        {
            printf("\nInput edges chunk size: %lu\n", (unsigned long) edges_chunk_size);
            printf("Merge parents chunk size: %lu\n", (unsigned long) parents_chunk_size);
            printf("Input pipeline buffers count: %lu\n", (unsigned long) input_pipeline_buffers);
            printf("Reduction pipeline buffers count: %lu\n", (unsigned long) reduction_pipeline_buffers);
            printf("Non-atomic rounds per segment: %d\n", nonatomic_rounds);
            printf("The compute_time/memory_latency hint: %f\n", compute_latency_ratio);
            printf("Performing vertex partitioning: %s\n", vertex_partitioning ? "Yes" : "No");
            printf("Load balancing: %s\n", load_balancing ? "Yes" : "No");
        }
    };

    inline void BuildConfiguration(
        int ngpus, int nedges, int nvtxs,
        size_t edge_chunks, size_t parent_chunks,
        size_t edges_chunk_size, size_t parents_chunk_size,
        size_t input_pipeline_buffers, size_t reduction_pipeline_buffers, 
        int nonatomic_rounds, bool vertex_partitioning,
        Configuration& configuration)
    {
        configuration.vertex_partitioning = vertex_partitioning;
        configuration.compute_latency_ratio = 0.0;
        configuration.nonatomic_rounds = nonatomic_rounds < 0 ? 0 : nonatomic_rounds;

        if (edges_chunk_size == 0 && edge_chunks == 0) edge_chunks = 1; // just in case
        if (parents_chunk_size == 0 && parent_chunks == 0) parent_chunks = 1; // just in case

        if (edge_chunks > 0)
        {
            edges_chunk_size = round_up((size_t)nedges, edge_chunks); // chunks param wins
            printf("\n%lu segments, %d non-atomic rounds <filter>\n", (unsigned long) edge_chunks, configuration.nonatomic_rounds);
        } 
        if (parent_chunks > 0) parents_chunk_size = round_up((size_t)nvtxs, parent_chunks); // chunks param wins

        configuration.edges_chunk_size = std::min(edges_chunk_size, (size_t)nedges);
        configuration.parents_chunk_size = std::min(parents_chunk_size, (size_t) nvtxs);

        configuration.input_pipeline_buffers = (input_pipeline_buffers == 0)
            ? round_up(round_up(nedges, ngpus), configuration.edges_chunk_size)
            : std::min(input_pipeline_buffers, round_up(round_up(nedges, ngpus), configuration.edges_chunk_size));

        configuration.reduction_pipeline_buffers = (reduction_pipeline_buffers == 0) 
            ? round_up(nvtxs, configuration.parents_chunk_size) 
            : std::min(reduction_pipeline_buffers, round_up(nvtxs, configuration.parents_chunk_size));

        configuration.load_balancing = 
            ngpus > 1 && 
            configuration.edges_chunk_size*configuration.input_pipeline_buffers*ngpus < nedges;
    }

    inline void BuildConfigurationAuto(
        int ngpus, int nedges, int nvtxs, 
        double compute_latency_ratio, int threshold_degree,
        int nonatomic_rounds,
        Configuration& configuration)
    {
        // TODO: define a performance model, currently this represents only the extreme cases  

        // round to between [0, 1], if compute time is larger then latency (ratio > 1) we do not care
        // and two input pipeline buffers should do  
        configuration.compute_latency_ratio = std::max(0.0, std::min(1.0, compute_latency_ratio));

        // approximation of max number of edges per device
        size_t edges_mpd = std::min((size_t)(round_up((size_t)nedges, ngpus) * (3.0/2)), (size_t)nedges); 

        if (nedges / nvtxs < threshold_degree) // low degree graphs like road maps
        {
            size_t merge_chunks = 8; // TODO: maybe chunk size should be the constant factor and not this relative split   

            configuration.vertex_partitioning = true;
            configuration.edges_chunk_size = edges_mpd; // take the max size as we want to use a single segment 
            configuration.parents_chunk_size = round_up((size_t)nvtxs, merge_chunks); // we have a large array to merge, let's break it up

            configuration.input_pipeline_buffers = 1; // a single large segment of edges  
            configuration.reduction_pipeline_buffers = std::max(merge_chunks, (size_t)2); // (chunks/2) buffers should do since we are memory bound here

            configuration.nonatomic_rounds = nonatomic_rounds < 0 ? 1 : nonatomic_rounds; // one non atomic rounds for the single segment  
        }

        else // high degree graphs like kronecker graphs
        {
            size_t merge_chunks = 2;
            
            configuration.vertex_partitioning = false;

            configuration.edges_chunk_size = nvtxs; // approximate the number of edges per chunk to be close to the number of nodes
            configuration.parents_chunk_size =  round_up((size_t)nvtxs, merge_chunks); // entire parent array is expected to be small

             // just enough buffers to preload most of the data but still allow dynamic behavior for a small number of buffers  
            configuration.input_pipeline_buffers = 
                std::max(
                    (size_t) (round_up(edges_mpd, configuration.edges_chunk_size) * (1-configuration.compute_latency_ratio)), 
                    (size_t) 2);  
            configuration.reduction_pipeline_buffers = std::max(merge_chunks/2, (size_t)2);

            configuration.nonatomic_rounds = nonatomic_rounds < 0 ? 0 : nonatomic_rounds;
        }

        configuration.load_balancing = 
            ngpus > 1 && 
            configuration.edges_chunk_size*configuration.input_pipeline_buffers*ngpus < nedges;
    }
}

#endif // __CC_CONFIG_H
