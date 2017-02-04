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

inline void LoadGraph(
    std::vector<Edge>& host_edges,
    uint32_t* nvtxs, uint32_t* nedges, graph_t* graph,
    bool ugtv, bool removeBidirectionalEdges)
{
    uint32_t i, j, u, v, a, b;
    Edge e;

    i = graph->nedges;
    j = graph->nvtxs;

    host_edges.resize(i);

    if (host_edges.size() != i){
        printf("Insufficient memory, data lost");
        exit(0);
    }

    uint32_t edc = 0;

    for (a = 0; a < j; a++){
        uint32_t n = graph->xadj[a + 1];
        for (b = graph->xadj[a]; b < n; b++){

            u = a;
            v = graph->adjncy[b];

            if (removeBidirectionalEdges && ((ugtv && u < v) || (!ugtv && u > v)))
                continue; // take only one of the two [u, v] [v, u] edges

            e.u = u;
            e.v = v;
            host_edges[edc++] = e;
        }
    }

    *nvtxs = j;
    *nedges = edc;

    host_edges.resize(edc);
}

inline void LoadGraph(
    groute::pinned_vector<Edge>& host_edges,
    uint32_t* nvtxs, uint32_t* nedges, graph_t* graph,
    bool ugtv, bool removeBidirectionalEdges)
{
    uint32_t i, j, u, v, a, b;
    Edge e;

    i = graph->nedges;
    j = graph->nvtxs;

    host_edges.resize(i);

    if (host_edges.size() != i){
        printf("Insufficient memory, data lost");
        exit(0);
    }

    uint32_t edc = 0;

    for (a = 0; a < j; a++){
        uint32_t n = graph->xadj[a + 1];
        for (b = graph->xadj[a]; b < n; b++){

            u = a;
            v = graph->adjncy[b];

            if (removeBidirectionalEdges && ((ugtv && u < v) || (!ugtv && u > v)))
                continue; // take only one of the two [u, v] [v, u] edges

            e.u = u;
            e.v = v;
            host_edges[edc++] = e;
        }
    }

    *nvtxs = j;
    *nedges = edc;

    host_edges.resize(edc);
}

#endif // __CC_LOAD_H
