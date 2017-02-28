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
#ifndef __CC_LOAD_H
#define __CC_LOAD_H

#include <groute/internal/cuda_utils.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/graphs/coo_graph.h>
#include <utils/utils.h>
#include <random>

typedef groute::graphs::Edge Edge;
typedef int component_t;

inline bool CheckComponents(std::vector<int> &host_parents, size_t count)
{
    int components = 0, errors = 0;
    for (int i = 0; i < count; i++) {
        if (host_parents[i] == i) {
            components++;
        }
        else 
        {
            int depth = 1;

            int p = host_parents[i];
            int pp = host_parents[p];

            while (p != pp) // parent is expected to be a root
            {
                ++depth; // track the depth

                if (pp > p)
                {
                    printf("Critical error: encountered a low to high pointer (p=%d, pp=%d, nvtxs=%d)\n", 
                        p, pp, (int)host_parents.size());
                    break;
                }

                p = pp;
                pp = host_parents[p];
            }

            if (depth > 1) {
                printf("Warning: component is not a complete star (i=%d, depth=%d, nvtxs=%d)\n", 
                    i, depth, (int)host_parents.size());
                ++errors;
            }
        }
    }

    printf("\nComponents: %d\n\n", components);
    if (errors > 0)
        printf("Errors: %d\n\n", errors);

    return errors == 0;
}

inline bool SampleBidirectionalEdges(graph_t* graph)
{
    bool duplicated = true;
    
    std::default_random_engine gen;
    std::uniform_int_distribution<uint32_t> distribution(0, graph->nvtxs-1);
    
    // Randomly check if the graph actually has duplicated bidirectional edges
    for (int i = 0; i < 100 && duplicated; ++i)
    {
        uint32_t u = distribution(gen); // Choose a random 'u' and make sure 100 of it's neighbor edges are duplicated  

        for (uint32_t ui = graph->xadj[u], un = std::min(graph->xadj[u + 1], ui + 100 /*up to 100 neighbors*/); ui < un; ui++)
        {
            uint32_t v = graph->adjncy[ui];

            // Go over neighbors of 'v' now 
            bool found = false;
            for (uint32_t vi = graph->xadj[v], vn = graph->xadj[v + 1]; vi < vn; vi++)
            {
                if (vi == u) 
                {
                    found = true; // Reverse edge was found  
                    break;
                }
            }

            if (found == false)
            {
                duplicated = false;
                break;
            }
        }
    }

    return duplicated;
}

template<typename EdgeInsertIterator>
void LoadGraph(EdgeInsertIterator insert_iterator, uint32_t* nvtxs, uint32_t* nedges, graph_t* graph, bool duplicated) 
{
    uint32_t nedges_csr = graph->nedges, nvtxs_csr = graph->nvtxs, edge_counter = 0;

    for (uint32_t u = 0; u < nvtxs_csr; u++)
    {
        for (uint32_t ui = graph->xadj[u], un = graph->xadj[u+1]; ui < un; ui++)
        {
            uint32_t v = graph->adjncy[ui];

            if (duplicated && u > v) continue; // Take only one of the two [u, v] [v, u] edges

            edge_counter++;
            *(insert_iterator++) = Edge(u, v);
        }
    }

    *nvtxs = nvtxs_csr;
    *nedges = edge_counter;
}

inline void LoadGraph(std::vector<Edge>& out_edges, uint32_t* nvtxs, uint32_t* nedges, graph_t* graph, bool undirected)
{
    bool duplicated = undirected ? SampleBidirectionalEdges(graph) : false;
    out_edges.reserve(duplicated ? graph->nedges / 2 : graph->nedges); 
    LoadGraph(std::back_inserter(out_edges), nvtxs, nedges, graph, duplicated);
}

inline void LoadGraph(groute::pinned_vector<Edge>& out_edges, uint32_t* nvtxs, uint32_t* nedges, graph_t* graph, bool undirected)
{
    bool duplicated = undirected ? SampleBidirectionalEdges(graph) : false;
    out_edges.reserve(duplicated ? graph->nedges / 2 : graph->nedges); 
    LoadGraph(std::back_inserter(out_edges), nvtxs, nedges, graph, duplicated);
}

#endif // __CC_LOAD_H
