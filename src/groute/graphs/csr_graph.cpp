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
#ifdef HAVE_METIS
#include <metis.h>
#endif

#include <unordered_set>
#include <groute/graphs/csr_graph.h>

namespace groute {
namespace graphs {

    namespace multi
    {
        MetisPartitioner::MetisPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (int i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (int i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (int i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, calling METIS\n", (int)IDXTYPEWIDTH);
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         //
                NULL,                         //
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  //
                &nparts,                      //
                NULL,                         //
                NULL,                         //
                NULL,                         //
                &edgecut,                     //
                &partition_table[0]);         //

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void MetisPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

        std::vector<index_t> GetUniqueHalos(
            const index_t* edge_dst,
            index_t seg_snode, index_t seg_nnodes,
            index_t seg_sedge, index_t seg_nedges, int& halos_counter)
        {
            std::unordered_set<index_t> halos_set;
            halos_counter = 0;

            for (int i = 0; i < seg_nedges; ++i)
            {
                index_t dest = edge_dst[seg_sedge + i];
                if (dest < seg_snode || dest >= (seg_snode + seg_nnodes)) // an halo
                {
                    ++halos_counter;
                    halos_set.insert(dest);
                }
            }

            std::vector<index_t> halos_vec(halos_set.size());
            std::copy(halos_set.begin(), halos_set.end(), halos_vec.begin());

            return halos_vec;
        }
    }
}
}
