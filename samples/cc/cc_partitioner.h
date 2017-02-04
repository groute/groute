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
#ifndef __CC_PARTITIONER_H
#define __CC_PARTITIONER_H


#include <vector>
#include <algorithm>

#include <utils/utils.h>

#include "cc_context.h"
#include "cc_config.h"

#include <groute/groute.h>

struct vertex_bounds
{
    int global_lower;   // the global lower bound
    int local_upper;    // the upper bound for local vtxs 
    int cross_upper;    // the upper bound for cross vtxs

    vertex_bounds(int lower, int local_upper, int cross_upper) :
        global_lower(lower), local_upper(local_upper), cross_upper(cross_upper)
    {
    }
};

struct vertex_partition
{
    vertex_bounds bounds;   // vertex bounds for this partition  
    int index;              // the starting index in the edges array  

    vertex_partition(const vertex_bounds& bounds, int index) :
        bounds(bounds), index(index)
    {
    }
};

template<typename TRandEdgeIt>
void hierarchic_partition(
    TRandEdgeIt first,                               /*the first global iterator */
    TRandEdgeIt begin, TRandEdgeIt end,               /*the local iterator bounds */
    vertex_bounds bounds,                                          /*the vertex bounds for this sub partition  */
    unsigned int npartitions, std::vector<vertex_partition>& partitions     /*the output partitions */
    )
{
    if (npartitions == 0) return;
    if (npartitions == 1)
    {
        partitions.push_back(vertex_partition(bounds, begin - first)); // push the partition info
        return;
    }

    unsigned int ps = next_power_2(npartitions) / 2; // the partitions split e.g. 4->2, 8->4, 5->4, 13->8
    float psr = (float)ps / npartitions; // the partition split ratio

    int vlrs = bounds.local_upper - bounds.global_lower; // the vertex local range size
    int bounds_split = bounds.global_lower + vlrs*psr;
    int local_upper = bounds.local_upper;

    TRandEdgeIt mid = std::partition(
        begin, end,
        [bounds_split, local_upper](const Edge& e)
    {
        // we partition all edges where both u, v are in the range [bounds_split, local_upper)  
        // to the right part, and all other edges to the left part

        return !(
            (bounds_split <= e.u && e.u < local_upper) &&
            (bounds_split <= e.v && e.v < local_upper));
    }
    );

    hierarchic_partition(
        first,
        begin, mid,
        vertex_bounds(bounds.global_lower, bounds_split, bounds.cross_upper),
        ps, partitions);

    hierarchic_partition(
        first,
        mid, end,
        vertex_bounds(bounds_split, bounds.local_upper, bounds.local_upper),
        npartitions - ps, partitions);
}

template<typename TRandEdgeIt>
std::vector<vertex_partition> hierarchic_partition(
    TRandEdgeIt begin, TRandEdgeIt end,           /*the iterator work bounds */
    int nvtxs, int npartitions                 /*the number of vetices and partitions */
    )
{
    std::vector<vertex_partition> partitions;

    hierarchic_partition(
        begin,
        begin, end,
        vertex_bounds(0, nvtxs, nvtxs),
        npartitions, partitions);

    return std::move(partitions);
}

namespace cc
{
    struct Partition
    {
        groute::Segment<int> parents_segment;
        int local_upper_bound;

        Partition(const groute::Segment<int>& parents_seg, int local_upper_bound)
            : parents_segment(parents_seg), local_upper_bound(local_upper_bound)
        {
        }
    };

    struct EdgesMetadata
    {
        std::vector<groute::device_t> dst_devs;
    };

    class EdgePartitioner
    {
    private:
        int m_ngpus;
        int m_nvtxs;
        bool m_vertex_partitioning;

        std::vector< std::unique_ptr<EdgesMetadata> > m_metadata_ptrs;

    public:
        std::vector<groute::Segment<Edge>> edge_partitions;
        std::vector<Partition> parents_partitions;

        EdgePartitioner(int ngpus, int nvtxs, const groute::Segment<Edge>& all_edges, bool vertex_partitioning = false) :
            m_ngpus(ngpus), m_nvtxs(nvtxs), m_vertex_partitioning(vertex_partitioning)
        {
            assert(groute::Device::IsHost(src_dev));

            if (m_vertex_partitioning)
            {
                m_metadata_ptrs.reserve(m_ngpus);
                edge_partitions.reserve(m_ngpus);
                parents_partitions.reserve(m_ngpus);

                int npartitions = m_ngpus;
                int nedges = all_edges.GetSegmentSize();
                Edge* ptr = all_edges.GetSegmentPtr();

                std::vector<vertex_partition> partitions
                    = hierarchic_partition(
                        ptr, ptr + nedges,
                        m_nvtxs, npartitions);

                assert(partitions.size() == npartitions);

                for (int i = 0; i < m_ngpus; i++)
                {
                    int start = partitions[i].index;
                    int end = (i == npartitions - 1) ? nedges : partitions[i + 1].index;

                    m_metadata_ptrs.emplace_back(groute::make_unique<EdgesMetadata>());
                    m_metadata_ptrs[i]->dst_devs = { i };
                    edge_partitions.push_back(groute::Segment<Edge>(ptr + start, nedges, end - start, start, (EdgesMetadata*) m_metadata_ptrs[i].get() /*metadata field*/));

                    int lower = partitions[i].bounds.global_lower;
                    int upper = partitions[i].bounds.cross_upper;
                    int local_upper = partitions[i].bounds.local_upper;

                    auto parents
                        = groute::Segment<int>(nullptr, m_nvtxs, upper - lower, lower);
                    parents_partitions.emplace_back(std::ref(parents), local_upper);
                }
            }

            else
            {
                // else, the simple one-to-all scatter 

                // all parents allocated at all devices
                auto parents = groute::Segment<int>(nullptr, m_nvtxs, m_nvtxs, 0);

                for (int i = 0; i < m_ngpus; i++)
                {
                    parents_partitions.emplace_back(std::ref(parents), m_nvtxs);
                }

                edge_partitions.push_back(all_edges);
            }
        }
    };

    class EdgeScatterPolicy : public groute::router::IPolicy
    {
    private:
        groute::RoutingTable m_topology;

    public:
        std::vector<Partition> parents_partitions;

        EdgeScatterPolicy(int ngpus)
        {
            m_topology[groute::Device::Host] = range(ngpus);
        }

        groute::RoutingTable GetRoutingTable() override
        {
            return m_topology;
        }

        groute::router::Route GetRoute(groute::device_t src_dev, void* message_metadata) override
        {
            assert(groute::Device::IsHost(src_dev));
            groute::router::Route route;

            if (message_metadata == nullptr)
            {
                route.dst_devs = m_topology[groute::Device::Host];
            }

            else
            {
                route.dst_devs = ((EdgesMetadata*)message_metadata)->dst_devs; // expecting the void* metadata to contain an EdgesMetadata structure  
                route.strategy = groute::router::Priority;
            }

            return route;
        }
    };
}

#endif // __CC_PARTITIONER_H
