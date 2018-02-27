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

#include "pr_common.h"
#include <utils/stopwatch.h>
#include <float.h>
#include <unordered_set>

DEFINE_int32(top_ranks, 10, "The number of top ranks to compare for PR regression");
DEFINE_bool(print_ranks, false, "Write out ranks to output");
DEFINE_bool(norm, false, "Normalize PR output ranks (L1)");
DECLARE_bool(verbose);


std::vector<rank_t> PageRankHost(groute::graphs::host::CSRGraph& graph)
{
    Stopwatch sw(true);

    std::vector<rank_t> residual(graph.nnodes, 0.0);
    std::vector<rank_t> ranks(graph.nnodes, 1.0 - ALPHA);

    for (index_t node = 0; node < graph.nnodes; ++node)
    {
        index_t
            begin_edge = graph.begin_edge(node),
            end_edge = graph.end_edge(node),
            out_degree = end_edge - begin_edge;

        if (out_degree == 0) continue;

        rank_t update = ((1.0 - ALPHA) * ALPHA) / out_degree;

        for (index_t edge = begin_edge; edge < end_edge; ++edge)
        {
            index_t dest = graph.edge_dest(edge);
            residual[dest] += update;
        }
    }

    std::queue<index_t> wl1, wl2;
    std::queue<index_t>* in_wl = &wl1, *out_wl = &wl2;

    for (index_t node = 0; node < graph.nnodes; ++node)
    {
        in_wl->push(node);
    }

    int iteration = 0;

    while (!in_wl->empty())
    {
        while (!in_wl->empty())
        {
            index_t node = in_wl->front();
            in_wl->pop();

            rank_t res = residual[node];
            ranks[node] += res;
            residual[node] = 0;

            index_t
                begin_edge = graph.begin_edge(node),
                end_edge = graph.end_edge(node),
                out_degree = end_edge - begin_edge;

            if (out_degree == 0) continue;

            rank_t update = res * ALPHA / out_degree;

            for (index_t edge = begin_edge; edge < end_edge; ++edge)
            {
                index_t dest = graph.edge_dest(edge);
                rank_t prev = residual[dest];
                residual[dest] += update;

                if (prev + update > EPSILON && prev < EPSILON)
                {
                    out_wl->push(dest);
                }
            }
        }

        ++iteration;
        std::swap(in_wl, out_wl);
    }

    sw.stop();

    if (FLAGS_verbose)
    {
        printf("\nPR Host: %f ms. \n", sw.ms());
        printf("PR Host converged after %d iterations \n\n", iteration);
    }

    return ranks;
}

int PageRankCheckErrors(std::vector<rank_t>& ranks, std::vector<rank_t>& regression)
{
    if (ranks.size() != regression.size()) {
        return std::abs((int64_t)ranks.size() - (int64_t)regression.size());
    }

    if (FLAGS_norm) // L1 normalization  
    {
        rank_t ranks_sum = 0.0, regression_sum = 0.0;
        for (auto val : ranks) ranks_sum += val;
        for (auto val : regression) regression_sum += val;
        for (auto& val : ranks) val /= ranks_sum;
        for (auto& val : regression) val /= regression_sum;
    }

    struct pr_pair {
            index_t node;
            rank_t rank;
            pr_pair(index_t node, rank_t rank) : node(node), rank(rank) { }
            pr_pair() : node(-1), rank(-1) { }
            inline bool operator< (const pr_pair& rhs) const {
                return rank < rhs.rank;
            }
        };

    std::vector<pr_pair> ranks_pairs(ranks.size());
    std::vector<pr_pair> regression_pairs(ranks.size());

    for (size_t i = 0; i < ranks.size(); i++)
    {
        ranks_pairs[i] = pr_pair(i, ranks[i]);
        regression_pairs[i] = pr_pair(i, regression[i]);
    }

    std::stable_sort(ranks_pairs.rbegin(), ranks_pairs.rend());
    std::stable_sort(regression_pairs.rbegin(), regression_pairs.rend()); // reversed sort, we want the top ranks

    int top = std::min((size_t)FLAGS_top_ranks, ranks.size());

    float mean_diff = 0.0f;
    int num_diffs = 0, missing_nodes = 0;

    std::unordered_set<index_t> top_nodes;
    for (int i = 0, t = std::min((size_t)(top*1.01), ranks.size()); i < t; ++i)
    {
        top_nodes.insert(regression_pairs[i].node);
    }

    for (int i = 0; i < top; ++i)
    {
        float diff = ranks_pairs[i].rank - regression_pairs[i].rank;
        if (top_nodes.find(ranks_pairs[i].node) == top_nodes.end())
        {
            missing_nodes++; // <-- nodes may switch locations in the rank because of small diffs as well
        }
        else if (fabs(1.0f - (ranks_pairs[i].rank / regression_pairs[i].rank)) > 1e-2)
        {
            if (FLAGS_verbose)
                printf("Difference in index %d: %f != %f\n", i, ranks_pairs[i].rank, regression_pairs[i].rank);
            num_diffs++;
        }
        mean_diff += fabs(diff);
    }
    mean_diff /= top;

    bool res = num_diffs + missing_nodes == 0 && mean_diff <= 1e-2;
    if (!res || FLAGS_verbose)
    {
        printf("[regression]\t\t\t[result]\n");
        for (int i = 0; i < 10; ++i)
        {
            printf("(%d, %f)\t\t(%d, %f)\n", 
                regression_pairs[i].node, regression_pairs[i].rank, ranks_pairs[i].node, ranks_pairs[i].rank);
        }
        printf("\nSummary: %d/%d large differences, %d/%d missing nodes, total mean diff: %f\n\n", num_diffs, (int)top, missing_nodes, (int)top, mean_diff);
    }

    return res ? 0 : num_diffs + missing_nodes;
}

int PageRankOutput(const char *file, const std::vector<rank_t>& ranks)
{
    FILE *f;
    f = fopen(file, "w");

    if (f) {
        struct pr_value {
            index_t node;
            rank_t rank;
            inline bool operator< (const pr_value& rhs) const {
                return rank < rhs.rank;
            }
        } *pr;

        pr = (struct pr_value *) calloc(ranks.size(), sizeof(struct pr_value));

        if (!pr) {
            fprintf(stderr, "PageRankOutput: Failed to allocate memory!");
            return 0;
        }

        rank_t sum = 0;
        for (int i = 0; i < ranks.size(); i++) {
            pr[i].node = i;
            pr[i].rank = ranks[i];
            sum += ranks[i];
        }

        fprintf(stderr, "Sorting by rank ...\n");
        std::stable_sort(pr, pr + ranks.size());
        fprintf(stderr, "Writing to file ...\n");

        fprintf(f, "ALPHA %*e EPSILON %*e\n", FLT_DIG, ALPHA, FLT_DIG, EPSILON);
        fprintf(f, "RANKS 1--%d of %d\n", FLAGS_top_ranks, (int)ranks.size());
        for (int i = 1; i <= FLAGS_top_ranks; i++) {
            if (!FLAGS_print_ranks)
                fprintf(f, "%d %d\n", i, pr[ranks.size() - i].node);
            else
                fprintf(f, "%d %d %*e\n", i, pr[ranks.size() - i].node, FLT_DIG, pr[ranks.size() - i].rank / sum);
        }

        free(pr);
        return 1;
    }
    else {
        fprintf(stderr, "Could not open '%s' for writing\n", file);
        return 0;
    }
}
