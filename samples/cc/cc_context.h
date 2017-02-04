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
#ifndef __CC_CONTEXT_H
#define __CC_CONTEXT_H

#include <vector>
#include <map>
#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>

#include <utils/parser.h>
#include <utils/utils.h>

#include <groute/context.h>

#include "cc_load.h"


#include <gflags/gflags.h>


#ifndef NDEBUG
#define VERBOSE true
#define REPETITIONS_DEFAULT 1
#else
#define VERBOSE false
#define REPETITIONS_DEFAULT 1
#endif

DEFINE_string(graphfile, "", "A file with a graph in Dimacs 10 format");
DEFINE_bool(ggr, true, "Graph file is a Galois binary GR file");
DEFINE_bool(verbose, VERBOSE, "Verbose prints");
DEFINE_int32(repetitions, REPETITIONS_DEFAULT, "Repetitions of GPU tests");
DEFINE_bool(undirected, true, "The input graph is undirected (one of each bidirectional edges will be removed)");

namespace cc
{
    class Context : public groute::Context // the global context for the cc problem solving  
    {
    public:
        groute::pinned_vector<groute::graphs::Edge>   host_edges;
        std::vector<int>            host_parents;

        int ngpus;
        unsigned int nvtxs, nedges;
        
        Context(std::string &graphfile, bool ggr, bool verbose, int ngpus) : 
            groute::Context(ngpus), ngpus(ngpus)
        {
            graph_t *graph;

            if (graphfile == "") {
                printf("A Graph File must be provided\n");
                exit(0);
            }

            printf("\nLoading graph %s (%d)\n", graphfile.substr(graphfile.find_last_of('\\') + 1).c_str(), ggr);
            graph = GetCachedGraph(graphfile, ggr);

            if (graph->nvtxs == 0) {
                printf("Empty graph!\n");
                exit(0);
            }

            printf("\n----- Running CC Async -----\n\n");

            if (FLAGS_verbose)
            {
                if (!FLAGS_undirected)  printf("undirected=false, expecting a directed (not symmetric) graph, keeping all edges\n");
                else                    printf("undirected=true,  expecting an undirected (symmetric) graph, removing bidirectional edges\n");
            }

            LoadGraph(host_edges, &nvtxs, &nedges, graph, false, FLAGS_undirected);
            host_parents.resize(nvtxs);

            if (verbose) {
                printf("The graph has %d vertices, and %d edges (average degree: %f)\n", nvtxs, nedges, (float)nedges / nvtxs);
            }
        }
    };
}

#endif // __CC_CONTEXT_H
