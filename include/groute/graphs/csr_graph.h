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

#ifndef __GROUTE_GRAPHS_CSR_GRAPH_H
#define __GROUTE_GRAPHS_CSR_GRAPH_H

#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include <gflags/gflags_declare.h>
#include <groute/context.h>

DECLARE_bool(verbose);
DECLARE_bool(pn);

typedef uint32_t index_t;


namespace groute {
namespace graphs {

    typedef uint32_t NoData; // NoData type for template defaults  

    struct CSRGraphBase
    {
        index_t nnodes, nedges;
        index_t *row_start;
        index_t *edge_dst;

        CSRGraphBase(index_t nnodes, index_t nedges) :
            nnodes(nnodes), nedges(nedges), row_start(nullptr), edge_dst(nullptr) { }

        CSRGraphBase() :
            nnodes(0), nedges(0), row_start(nullptr), edge_dst(nullptr) { }
    };

    namespace host
    {
        /*
        * @brief A host graph object
        */
        struct CSRGraph : public CSRGraphBase
        {
            std::vector<index_t> row_start_vec; // the vectors are not always in use (see Bind)
            std::vector<index_t> edge_dst_vec;

            index_t* edge_weights;
            index_t* node_weights;

            std::vector<index_t> edge_weights_vec; 

            CSRGraph(index_t nnodes, index_t nedges) :
                CSRGraphBase(nnodes, nedges), row_start_vec(nnodes + 1), edge_dst_vec(nedges),
                edge_weights(nullptr), node_weights(nullptr)
            {
                row_start = &row_start_vec[0];
                edge_dst = &edge_dst_vec[0];
            }

            CSRGraph() : edge_weights(nullptr), node_weights(nullptr) { }

            ~CSRGraph()
            {
            }

            void Move(index_t nnodes, index_t nedges,
                std::vector<index_t>& row_start, std::vector<index_t>& edge_dst)
            {
                this->nnodes = nnodes;
                this->nedges = nedges;

                this->row_start_vec = std::move(row_start);
                this->edge_dst_vec = std::move(edge_dst);

                this->row_start = &this->row_start_vec[0];
                this->edge_dst = &this->edge_dst_vec[0];
            }

            void MoveWeights(std::vector<index_t>& edge_weights)
            {
                this->edge_weights_vec = std::move(edge_weights);
                this->edge_weights = &this->edge_weights_vec[0];
            }

            void AllocWeights()
            {
                this->edge_weights_vec.resize(nedges);
                this->edge_weights = &this->edge_weights_vec[0];
            }

            void Bind(index_t nnodes, index_t nedges,
                index_t* row_start, index_t* edge_dst,
                index_t* edge_weights, index_t* node_weights)
            {
                this->nnodes = nnodes;
                this->nedges = nedges;

                this->row_start_vec.clear();
                this->edge_dst_vec.clear();

                this->row_start = row_start;
                this->edge_dst = edge_dst;

                this->edge_weights = edge_weights;
                this->node_weights = node_weights;
            }

            index_t max_degree() const
            {
                index_t max_degree = 0;
                for (index_t node = 0; node < nnodes; node++)
                {
                    max_degree = std::max(max_degree, end_edge(node) - begin_edge(node));
                }
                return max_degree;
            }

            index_t begin_edge(index_t node) const
            {
                return row_start[node];
            }

            index_t end_edge(index_t node) const
            {
                return row_start[node + 1];
            }

            index_t edge_dest(index_t edge) const
            {
                return edge_dst[edge];
            }
        };
        
        /*
        * @brief A host graph generator (CSR)
        * @note The generated graph is asymmetric and may have duplicated edges but no self loops
        */
        class CSRGraphGenerator
        {
        private:
            index_t m_nnodes;
            int m_gen_factor;

            std::default_random_engine m_generator;
            std::uniform_int_distribution<int> m_nneighbors_distribution;
            std::uniform_int_distribution<index_t> m_node_distribution;

            int GenNeighborsNum(index_t node) { return m_nneighbors_distribution(m_generator); }
            index_t GenNeighbor(index_t node, std::set<index_t>& neighbors)
            {
                index_t neighbor;
                do
                {
                    neighbor = m_node_distribution(m_generator);
                } while (neighbor == node || neighbors.find(neighbor) != neighbors.end());

                neighbors.insert(neighbor);
                return neighbor;
            }

        public:
            CSRGraphGenerator(index_t nnodes, int gen_factor) :
                m_nnodes(nnodes), m_gen_factor(gen_factor), m_nneighbors_distribution(1, gen_factor), m_node_distribution(0, nnodes - 1)
            {
                assert(nnodes > 1);
                assert(gen_factor >= 1);
            }

            void Gen(CSRGraph& graph)
            {
                std::vector<index_t> row_start(m_nnodes + 1, 0);
                std::vector<index_t> edge_dst;
                edge_dst.reserve(m_nnodes * m_gen_factor); // approximation  

                for (index_t node = 0; node < m_nnodes; ++node)
                {
                    row_start[node] = edge_dst.size();
                    int nneighbors = GenNeighborsNum(node);
                    std::set<index_t> neighbors;
                    for (int i = 0; i < nneighbors; ++i)
                    {
                        edge_dst.push_back(GenNeighbor(node, neighbors));
                    }
                }

                index_t nedges = edge_dst.size();
                row_start[m_nnodes] = nedges;

                edge_dst.shrink_to_fit(); // 

                graph.Move(m_nnodes, nedges, row_start, edge_dst);
            }
        };


        class NoIntersectionGraphGenerator
        {
        private:
            int m_ngpus;
            index_t m_nnodes;
            int m_gen_factor;

        public:
            NoIntersectionGraphGenerator(int ngpus, index_t nnodes, int gen_factor) :
                m_ngpus(ngpus), m_nnodes((nnodes/ngpus)*ngpus /*round*/), m_gen_factor(gen_factor)
            {
                assert(nnodes >= ngpus);
                assert(gen_factor >= 1);
            }

            void Gen(CSRGraph& graph)
            {
                // Builds a simple two-way chain with no intersection between segments   

                std::vector<index_t> row_start(m_nnodes + 1, 0);
                std::vector<index_t> edge_dst;

                edge_dst.reserve(m_nnodes * 2);
                index_t seg_nnodes = m_nnodes / m_ngpus;

                for (index_t node = 0; node < m_nnodes; ++node)
                {
                    index_t seg_idx = node / seg_nnodes;
                    index_t seg_snode = seg_idx * seg_nnodes;

                    row_start[node] = edge_dst.size();
                    
                    if (node >= seg_snode + 1) edge_dst.push_back(node-1);
                    if (node + 1 < seg_snode + seg_nnodes) edge_dst.push_back(node+1);
                }

                index_t nedges = edge_dst.size();
                row_start[m_nnodes] = nedges;

                edge_dst.shrink_to_fit(); // 

                graph.Move(m_nnodes, nedges, row_start, edge_dst);
            }
        };


        class ChainGraphGenerator
        {
        private:
            int m_ngpus;
            index_t m_nnodes;
            int m_gen_factor;

        public:
            ChainGraphGenerator(int ngpus, index_t nnodes, int gen_factor) :
                m_ngpus(ngpus), m_nnodes((nnodes/ngpus)*ngpus /*round*/), m_gen_factor(gen_factor)
            {
                assert(nnodes >= ngpus);
                assert(gen_factor >= 1);
            }

            void Gen(CSRGraph& graph)
            {
                std::vector<index_t> row_start(m_nnodes + 1, 0);
                std::vector<index_t> edge_dst;

                edge_dst.reserve(m_nnodes * 2);
                
                for (index_t node = 0; node < m_nnodes; ++node)
                {
                    row_start[node] = edge_dst.size();
                    
                    if (node >= 1) edge_dst.push_back(node-1);
                    if (node + 1 < m_nnodes) edge_dst.push_back(node+1);
                }

                index_t nedges = edge_dst.size();
                row_start[m_nnodes] = nedges;

                edge_dst.shrink_to_fit(); // 

                graph.Move(m_nnodes, nedges, row_start, edge_dst);
            }
        };


        class CliquesNoIntersectionGraphGenerator
        {
        private:
            int m_ngpus;
            index_t m_nnodes;
            int m_gen_factor;

        public:
            CliquesNoIntersectionGraphGenerator(int ngpus, index_t nnodes, int gen_factor) :
                m_ngpus(ngpus), m_nnodes((nnodes/ngpus)*ngpus /*round*/), m_gen_factor(gen_factor)
            {
                assert(nnodes >= ngpus);
                assert(gen_factor >= 1);
            }

            void Gen(CSRGraph& graph)
            {
                std::vector<index_t> row_start(m_nnodes + 1, 0);
                std::vector<index_t> edge_dst;
                
                index_t seg_nnodes = m_nnodes / m_ngpus;
                edge_dst.reserve(m_nnodes * seg_nnodes);
                
                for (index_t node = 0; node < m_nnodes; ++node)
                {
                    index_t seg_idx = node / seg_nnodes;
                    index_t seg_snode = seg_idx * seg_nnodes;

                    row_start[node] = edge_dst.size();
                    for (int i = 0; i < seg_nnodes; ++i)
                    {
                        if (seg_snode + i == node) continue;
                        edge_dst.push_back(seg_snode+i);
                    }
                }

                index_t nedges = edge_dst.size();
                row_start[m_nnodes] = nedges;

                edge_dst.shrink_to_fit(); // 

                graph.Move(m_nnodes, nedges, row_start, edge_dst);
            }
        };
    }

    namespace dev // device objects 
    {
        /*
        * @brief A multi-GPU graph segment object (represents a segment allocated at one GPU)
        */
        struct CSRGraphSeg : public CSRGraphBase
        {
            int seg_idx, nsegs;

            index_t nodes_offset, edges_offset;
            index_t nnodes_local, nedges_local;

            CSRGraphSeg() :
                seg_idx(-1), nsegs(-1),
                nodes_offset(0), edges_offset(0), nnodes_local(0), nedges_local(0) { }

            __device__ __host__ __forceinline__ bool owns(index_t node) const
            {
                assert(node >= 0 && node < nnodes);
                return node >= nodes_offset && node < (nodes_offset + nnodes_local);
            }

            __host__ __device__ __forceinline__ index_t owned_start_node() const
            {
                return nodes_offset;
            }
            
            __host__ __device__ __forceinline__ index_t owned_nnodes() const
            {
                return nnodes_local;
            }

            __host__ __device__ __forceinline__ index_t global_nnodes() const
            {
                return nnodes;
            }

            __device__ __forceinline__ index_t begin_edge(index_t node) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(row_start + node - nodes_offset);
#else
                return row_start[node - nodes_offset];
#endif
            }

            __device__ __forceinline__ index_t end_edge(index_t node) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(row_start + node + 1 - nodes_offset);
#else
                return row_start[node + 1 - nodes_offset];
#endif
            }

            __device__ __forceinline__ index_t edge_dest(index_t edge) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(edge_dst + edge - edges_offset);
#else
                return edge_dst[edge - edges_offset];
#endif
            }
        };

        /*
        * @brief A single GPU graph object (a complete graph allocated at one GPU)
        */
        struct CSRGraph : public CSRGraphBase
        {
            CSRGraph() { }

            __device__ __host__ __forceinline__ bool owns(index_t node) const
            {
                assert(node >= 0 && node < nnodes);
                return true;
            }

            __host__ __device__ __forceinline__ index_t owned_start_node() const
            {
                return 0;
            }
            
            __host__ __device__ __forceinline__ index_t owned_nnodes() const
            {
                return nnodes;
            }

            __host__ __device__ __forceinline__ index_t global_nnodes() const
            {
                return nnodes;
            }

            __device__ __forceinline__ index_t begin_edge(index_t node) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(row_start + node);
#else
                return row_start[node];
#endif
            }

            __device__ __forceinline__ index_t end_edge(index_t node) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(row_start + node + 1);
#else
                return row_start[node + 1];
#endif
            }

            __device__ __forceinline__ index_t edge_dest(index_t edge) const
            {
#if __CUDA_ARCH__ >= 320
                return __ldg(edge_dst + edge);
#else
                return edge_dst[edge];
#endif
            }
        };

        template<typename T>
        struct GraphDatumSeg
        {
            T* data_ptr;
            index_t offset;
            index_t size;

            GraphDatumSeg() : data_ptr(nullptr), offset(0), size(0) { }
            GraphDatumSeg(T* data_ptr, index_t offset, index_t size) : data_ptr(data_ptr), offset(offset), size(size) { }

            __device__ __forceinline__ T get_item(index_t idx) const
            {
                assert(idx >= offset && idx < offset + size);
                return data_ptr[idx - offset];
            }

            __device__ __forceinline__ T& operator[](index_t idx)
            {
                return data_ptr[idx - offset];
            }

            __device__ __forceinline__ T* get_item_ptr(index_t idx) const
            {
                assert(idx >= offset && idx < offset + size);
                return data_ptr + (idx - offset);
            }

            __device__ __forceinline__ void set_item(index_t idx, const T& item) const
            {
                assert(idx >= offset && idx < offset + size);
                data_ptr[idx - offset] = item;
            }
        };

        template<typename T>
        struct GraphDatum
        {
            T* data_ptr;
            index_t size;

            GraphDatum() : data_ptr(nullptr), size(0) { }
            GraphDatum(T* data_ptr, index_t size) : data_ptr(data_ptr), size(size) { }

            __device__ __forceinline__ T get_item(index_t idx) const
            {
                assert(idx >= 0 && idx < size);
                return data_ptr[idx];
            }

            __device__ __forceinline__ T& operator[](index_t idx)
            {
                return data_ptr[idx];
            }

            __device__ __forceinline__ T* get_item_ptr(index_t idx) const
            {
                assert(idx >= 0 && idx < size);
                return data_ptr + (idx);
            }

            __device__ __forceinline__ void set_item(index_t idx, const T& item) const
            {
                assert(idx >= 0 && idx < size);
                data_ptr[idx] = item;
            }
        };
    }

    namespace multi
    {
        struct GraphPartitioner
        {
            virtual ~GraphPartitioner() { }

            virtual host::CSRGraph& GetOriginGraph() = 0;
            virtual host::CSRGraph& GetPartitionedGraph() = 0;

            virtual void GetSegIndices(
                int seg_idx,
                index_t& seg_snode, index_t& seg_nnodes,
                index_t& seg_sedge, index_t& seg_nedges) const = 0;

            virtual bool NeedsReverseLookup() = 0;
            virtual std::function<index_t(index_t)> GetReverseLookupFunc() = 0;
        };

        class RandomPartitioner : public GraphPartitioner
        {
            host::CSRGraph& m_origin_graph;
            int m_nsegs;

        public:
            RandomPartitioner(host::CSRGraph& origin_graph, int nsegs) : m_origin_graph(origin_graph), m_nsegs(nsegs)
            {
                assert(nsegs >= 1);
            }
            
            host::CSRGraph& GetOriginGraph() override
            {
                return m_origin_graph;
            }
            
            host::CSRGraph& GetPartitionedGraph() override
            {
                return m_origin_graph;
            }

            void GetSegIndices(
                int seg_idx,
                index_t& seg_snode, index_t& seg_nnodes,
                index_t& seg_sedge, index_t& seg_nedges) const override
            {
                index_t seg_enode, seg_eedge;

                seg_nnodes = round_up(m_origin_graph.nnodes, m_nsegs);                      // general nodes seg size
                seg_snode = seg_nnodes * seg_idx;                                           // start node
                seg_nnodes = std::min(m_origin_graph.nnodes - seg_snode, seg_nnodes);       // fix for last seg case
                seg_enode = seg_snode + seg_nnodes;                                         // end node
                seg_sedge = m_origin_graph.row_start[seg_snode];                            // start edge
                seg_eedge = m_origin_graph.row_start[seg_enode];                            // end edge
                seg_nedges = seg_eedge - seg_sedge;  
            }
            
            bool NeedsReverseLookup() override { return false; }

            std::function<index_t(index_t)> GetReverseLookupFunc() override
            {
                return [](index_t idx) { return idx; }; // Just the identity func
            }
        };

        class MetisPartitioner : public GraphPartitioner
        {
            host::CSRGraph& m_origin_graph;
            host::CSRGraph m_partitioned_graph;
            std::vector<index_t> m_reverse_lookup;
            std::vector<index_t> m_seg_offsets;
            int m_nsegs;

        public:
            MetisPartitioner(host::CSRGraph& origin_graph, int nsegs);
            
            host::CSRGraph& GetOriginGraph() override { return m_origin_graph; }
            host::CSRGraph& GetPartitionedGraph() override { return m_partitioned_graph; }

            void GetSegIndices(
                int seg_idx,
                index_t& seg_snode, index_t& seg_nnodes,
                index_t& seg_sedge, index_t& seg_nedges) const override;
            
            bool NeedsReverseLookup() override { return true; }
            std::function<index_t(index_t)> GetReverseLookupFunc() override;
        };

        /*
        * @brief A multi-GPU graph segment allocator (allocates a graph segment over each GPU)
        */
        struct CSRGraphAllocator
        {
            typedef dev::CSRGraphSeg DeviceObjectType;

        private:
            groute::Context& m_context;
            std::unique_ptr<GraphPartitioner> m_partitioner;

            int m_ngpus;

            std::vector<dev::CSRGraphSeg> m_dev_segs;

        public:
            CSRGraphAllocator(groute::Context& context, host::CSRGraph& host_graph, int ngpus) :
                m_context(context), m_ngpus(ngpus)
            {
                m_partitioner = FLAGS_pn && (m_ngpus > 1)
                    ? (std::unique_ptr<GraphPartitioner>) std::unique_ptr<MetisPartitioner>(new MetisPartitioner(host_graph, ngpus))
                    : (std::unique_ptr<GraphPartitioner>) std::unique_ptr<RandomPartitioner>(new RandomPartitioner(host_graph, ngpus));

                m_dev_segs.resize(m_ngpus);

                for (int i = 0; i < m_ngpus; i++)
                {
                    m_context.SetDevice(i);
                    AllocateDevSeg(m_ngpus, i, m_dev_segs[i]);
                }
            }

            ~CSRGraphAllocator()
            {
                for (auto& seg : m_dev_segs)
                {
                    DeallocateDevSeg(seg);
                }
            }

            GraphPartitioner* GetGraphPartitioner() const
            {
                return m_partitioner.get();
            }

            const std::vector<dev::CSRGraphSeg>& GetDeviceObjects() const
            {
                return m_dev_segs;
            }

            void AllocateDatumObjects() { }

            template<typename TFirstGraphDatum, typename...TGraphDatum>
            void AllocateDatumObjects(TFirstGraphDatum& first_datum, TGraphDatum&... more_data)
            {
                AllocateDatum(first_datum);
                AllocateDatumObjects(more_data...);
            }

            template<typename TGraphDatum>
            void AllocateDatum(TGraphDatum& graph_datum)
            {
                graph_datum.PrepareAllocate(m_ngpus, m_partitioner.get());

                for (int i = 0; i < m_ngpus; i++)
                {
                    index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                    m_partitioner->GetSegIndices(i, seg_snode, seg_nnodes, seg_sedge, seg_nedges);

                    m_context.SetDevice(i);
                    graph_datum.AllocateSeg(i, m_partitioner->GetPartitionedGraph().nnodes, m_partitioner->GetPartitionedGraph().nedges, seg_snode, seg_nnodes, seg_sedge, seg_nedges);
                }
            }

            template<typename TGraphDatum>
            void GatherDatum(TGraphDatum& graph_datum)
            {
                graph_datum.PrepareGather(m_ngpus, m_partitioner->GetPartitionedGraph());

                for (int i = 0; i < m_ngpus; i++)
                {
                    index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                    m_partitioner->GetSegIndices(i, seg_snode, seg_nnodes, seg_sedge, seg_nedges);
                    
                    m_context.SetDevice(i);
                    graph_datum.GatherSeg(i, m_partitioner->GetPartitionedGraph().nnodes, m_partitioner->GetPartitionedGraph().nedges, seg_snode, seg_nnodes, seg_sedge, seg_nedges);
                }

                if (m_partitioner->NeedsReverseLookup())
                    graph_datum.FinishGather(m_partitioner->GetReverseLookupFunc());
            }

        private:
            void AllocateDevSeg(int nsegs, int seg_idx, dev::CSRGraphSeg& graph_seg) const
            {
                graph_seg.seg_idx = seg_idx;
                graph_seg.nsegs = nsegs;

                index_t seg_snode, seg_nnodes, seg_sedge, seg_nedges;
                m_partitioner->GetSegIndices(seg_idx, seg_snode, seg_nnodes, seg_sedge, seg_nedges);

                graph_seg.nnodes = m_partitioner->GetPartitionedGraph().nnodes;
                graph_seg.nedges = m_partitioner->GetPartitionedGraph().nedges;
                graph_seg.nodes_offset = seg_snode;
                graph_seg.edges_offset = seg_sedge;
                graph_seg.nnodes_local = seg_nnodes;
                graph_seg.nedges_local = seg_nedges;

                GROUTE_CUDA_CHECK(cudaMalloc(&graph_seg.row_start, (seg_nnodes + 1) * sizeof(index_t))); // malloc and copy +1 for the row_start's extra cell
                GROUTE_CUDA_CHECK(cudaMemcpy(graph_seg.row_start, m_partitioner->GetPartitionedGraph().row_start + seg_snode, (seg_nnodes + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

                GROUTE_CUDA_CHECK(cudaMalloc(&graph_seg.edge_dst, seg_nedges * sizeof(index_t)));
                GROUTE_CUDA_CHECK(cudaMemcpy(graph_seg.edge_dst, m_partitioner->GetPartitionedGraph().edge_dst + seg_sedge, seg_nedges * sizeof(index_t), cudaMemcpyHostToDevice));
            }

            void DeallocateDevSeg(dev::CSRGraphSeg& graph_seg) const
            {
                GROUTE_CUDA_CHECK(cudaFree(graph_seg.row_start));
                GROUTE_CUDA_CHECK(cudaFree(graph_seg.edge_dst));

                graph_seg.row_start = nullptr;
                graph_seg.edge_dst = nullptr;

                graph_seg.seg_idx = 0;
                graph_seg.nsegs = 0;

                graph_seg.nnodes = 0;
                graph_seg.nedges = 0;
                graph_seg.nodes_offset = 0;
                graph_seg.edges_offset = 0;
                graph_seg.nnodes_local = 0;
                graph_seg.nedges_local = 0;
            }
        };

        template<typename T>
        class EdgeInputDatum
        {
        public:
            typedef dev::GraphDatumSeg < T > DeviceObjectType; // edges are scattered, so we need seg objects 

        private:
            T* m_origin_data;
            T* m_partitioned_data;

            std::vector<T> m_ones;

            std::vector<DeviceObjectType> m_dev_segs;

        public:
            EdgeInputDatum() : m_origin_data(nullptr), m_partitioned_data(nullptr) { }

            EdgeInputDatum(CSRGraphAllocator& graph_allocator)
            {
                // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~EdgeInputDatum()
            {
                DeallocateDevSegs();
            }

            void PrepareAllocate(int nsegs, GraphPartitioner* partitioner)
            {
                DeallocateDevSegs();

                if (partitioner->GetOriginGraph().edge_weights == nullptr ||
                    partitioner->GetPartitionedGraph().edge_weights == nullptr) 
                {
                    assert(partitioner->GetOriginGraph().edge_weights == nullptr);
                    assert(partitioner->GetPartitionedGraph().edge_weights == nullptr);

                    printf("\nWarning: Expecting edge weights, falling back to all one's weights (use gen_weights and gen_weight_range).\n\n");
                   
                    m_ones = std::vector<T>(partitioner->GetOriginGraph().nedges, 1);
                    m_origin_data = m_ones.data();  // since data is all one's we can use the same for both origin and partitioned
                    m_partitioned_data = m_ones.data();
                }

                else
                {
                    m_origin_data = partitioner->GetOriginGraph().edge_weights; // Bind to edge_weights from the original graph
                    m_partitioned_data = partitioner->GetPartitionedGraph().edge_weights; // Bind to edge_weights from the partitioned graph
                }

                m_dev_segs.resize(nsegs);
            }

            void AllocateSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, seg_nedges * sizeof(T)));
                GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_segs[seg_idx].data_ptr, m_partitioned_data + seg_sedge, seg_nedges * sizeof(T), cudaMemcpyHostToDevice));
                
                m_dev_segs[seg_idx].offset = seg_sedge;
                m_dev_segs[seg_idx].size = seg_nedges;
            }

            T* GetHostDataPtr() { return m_origin_data; }

            const std::vector< DeviceObjectType >& GetDeviceObjects() const
            {
                return m_dev_segs;
            }

        protected:
            void DeallocateDevSegs()
            {
                for (auto& seg : m_dev_segs)
                {
                    DeallocateDevSeg(seg);
                }
                m_dev_segs.clear();
            }

            void DeallocateDevSeg(DeviceObjectType& datum_seg)
            {
                GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                datum_seg.data_ptr = nullptr;
                datum_seg.offset = 0;
                datum_seg.size = 0;
            }
        };

        std::vector<index_t> GetUniqueHalos(
            const index_t* edge_dst,
            index_t seg_snode, index_t seg_nnodes,
            index_t seg_sedge, index_t seg_nedges, int& halos_counter);

        class HalosDatum
        {        
        public:
            typedef dev::GraphDatum < index_t > DeviceObjectType; 

        private:
            const host::CSRGraph* m_host_graph;
            std::vector<DeviceObjectType> m_dev_segs;

        public:
            HalosDatum() : m_host_graph(nullptr) { }

            HalosDatum(CSRGraphAllocator& graph_allocator) : m_host_graph(nullptr)
            {
                // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~HalosDatum()
            {
                DeallocateDevSegs();
            }

            void PrepareAllocate(int nsegs, GraphPartitioner* partitioner)
            {
                m_host_graph = &partitioner->GetPartitionedGraph(); // cache the graph instance

                DeallocateDevSegs();
                m_dev_segs.resize(nsegs);
            }

            void AllocateSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                assert(m_host_graph);

                int halos_counter = 0;
                std::vector<index_t> halos_vec 
                    = GetUniqueHalos(m_host_graph->edge_dst, seg_snode, seg_nnodes, seg_sedge, seg_nedges, halos_counter);
                
                if (FLAGS_verbose)
                {
                    printf(
                        "Halo stats -> seg: %d, seg nodes: %d, seg edges: %d, halos: %d, unique halos: %llu\n", 
                        seg_idx, seg_nnodes, seg_nedges, halos_counter, halos_vec.size());
                }

                if (halos_vec.size() == 0)
                {
                    m_dev_segs[seg_idx].data_ptr = nullptr;
                }
                else
                {
                    GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, halos_vec.size() * sizeof(index_t)));
                    GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_segs[seg_idx].data_ptr, &halos_vec[0], halos_vec.size() * sizeof(index_t), cudaMemcpyHostToDevice));
                }

                m_dev_segs[seg_idx].size = halos_vec.size();
            }

            const std::vector< DeviceObjectType >& GetDeviceObjects() const
            {
                return m_dev_segs;
            }

        protected:
            void DeallocateDevSegs()
            {
                for (auto& seg : m_dev_segs)
                {
                    DeallocateDevSeg(seg);
                }
                m_dev_segs.clear();
            }

            void DeallocateDevSeg(DeviceObjectType& datum_seg)
            {
                GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                datum_seg.data_ptr = nullptr;
                datum_seg.size = 0;
            }
        };

        /*
        * @brief A node data array with global allocation for each device
        * and ownership over owned nodes data
        * Date is gathered to host from the owned segment of each device
        */
        template<typename T>
        class NodeOutputGlobalDatum
        {
        public:
            typedef dev::GraphDatum < T > DeviceObjectType; 
                // no need for dev::GraphDatumSeg because data is allocated globally for each device

        private:
            std::vector<T> m_host_data;
            std::vector<DeviceObjectType> m_dev_segs;

        public:
            NodeOutputGlobalDatum() { }

            NodeOutputGlobalDatum(CSRGraphAllocator& graph_allocator)
            {
                // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~NodeOutputGlobalDatum()
            {
                DeallocateDevSegs();
            }

            void PrepareAllocate(int nsegs, GraphPartitioner* partitioner)
            {
                DeallocateDevSegs();
                m_dev_segs.resize(nsegs);
            }

            void AllocateSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, nnodes * sizeof(T)));
                m_dev_segs[seg_idx].size = nnodes;
            }

            void PrepareGather(int nsegs, const host::CSRGraph& host_graph)
            {
                m_host_data.resize(host_graph.nnodes);
            }

            void GatherSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                DeviceObjectType& datum_seg = m_dev_segs[seg_idx];

                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &m_host_data[seg_snode], datum_seg.data_ptr + seg_snode,
                    seg_nnodes * sizeof(T), cudaMemcpyDeviceToHost));
            }

            void FinishGather(const std::function<index_t(index_t)>& reverse_lookup)
            {
                std::vector<T> temp_data(m_host_data.size());

                for (int i = 0; i < m_host_data.size(); i++)
                {
                    temp_data[i] = m_host_data[reverse_lookup(i)];
                }

                m_host_data = std::move(temp_data);
            }

            const std::vector<T>& GetHostData() { return m_host_data; }

            const std::vector< DeviceObjectType >& GetDeviceObjects() const
            {
                return m_dev_segs;
            }

        protected:
            void DeallocateDevSegs()
            {
                for (auto& seg : m_dev_segs)
                {
                    DeallocateDevSeg(seg);
                }
                m_dev_segs.clear();
            }

            void DeallocateDevSeg(DeviceObjectType& datum_seg)
            {
                GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                datum_seg.data_ptr = nullptr;
                datum_seg.size = 0;
            }
        };

        /*
        * @brief A node data array with local allocation for each device
        * Each device can read/write only from/to its local nodes
        * Date is gathered to host from each segment of each device
        */
        template<typename T>
        class NodeOutputLocalDatum
        {
        public:
            typedef dev::GraphDatumSeg < T > DeviceObjectType;

        private:
            std::vector<T> m_host_data;
            std::vector<DeviceObjectType> m_dev_segs;

        public:
            NodeOutputLocalDatum() { }

            NodeOutputLocalDatum(CSRGraphAllocator& graph_allocator)
            {
                // Will call this->PrepareAllocate and this->AllocateSeg with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~NodeOutputLocalDatum()
            {
                DeallocateDevSegs();
            }

            void PrepareAllocate(int nsegs, GraphPartitioner* partitioner)
            {
                DeallocateDevSegs();
                m_dev_segs.resize(nsegs);
            }

            void AllocateSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_segs[seg_idx].data_ptr, seg_nnodes * sizeof(T)));

                m_dev_segs[seg_idx].offset = seg_snode;
                m_dev_segs[seg_idx].size = seg_nnodes;
            }

            void PrepareGather(int nsegs, const host::CSRGraph& host_graph)
            {
                m_host_data.resize(host_graph.nnodes);
            }

            void GatherSeg(int seg_idx,
                index_t nnodes, index_t nedges,
                index_t seg_snode, index_t seg_nnodes,
                index_t seg_sedge, index_t seg_nedges)
            {
                DeviceObjectType& datum_seg = m_dev_segs[seg_idx];

                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &m_host_data[seg_snode], datum_seg.data_ptr,
                    seg_nnodes * sizeof(T), cudaMemcpyDeviceToHost));
            }

            void FinishGather(const std::function<index_t(index_t)>& reverse_lookup)
            {
                std::vector<T> temp_data(m_host_data.size());

                for (int i = 0; i < m_host_data.size(); i++)
                {
                    temp_data[i] = m_host_data[reverse_lookup(i)];
                }

                m_host_data = std::move(temp_data);
            }

            const std::vector<T>& GetHostData() { return m_host_data; }

            const std::vector< DeviceObjectType >& GetDeviceObjects() const
            {
                return m_dev_segs;
            }

        protected:
            void DeallocateDevSegs()
            {
                for (auto& seg : m_dev_segs)
                {
                    DeallocateDevSeg(seg);
                }
                m_dev_segs.clear();
            }

            void DeallocateDevSeg(DeviceObjectType& datum_seg)
            {
                GROUTE_CUDA_CHECK(cudaFree(datum_seg.data_ptr));

                datum_seg.data_ptr = nullptr;
                datum_seg.offset = 0;
                datum_seg.size = 0;
            }
        };
    }

    namespace single
    {
        /*
        * @brief A single GPU graph allocator (allocates a complete mirror graph at one GPU)
        */
        struct CSRGraphAllocator
        {
            typedef dev::CSRGraph DeviceObjectType;

        private:
            host::CSRGraph& m_origin_graph;
            dev::CSRGraph m_dev_mirror;

        public:
            CSRGraphAllocator(host::CSRGraph& host_graph) :
                m_origin_graph(host_graph)
            {
                AllocateDevMirror();
            }

            ~CSRGraphAllocator()
            {
                DeallocateDevMirror();
            }

            const dev::CSRGraph& DeviceObject() const
            {
                return m_dev_mirror;
            }

            void AllocateDatumObjects() { }

            template<typename TFirstGraphDatum, typename...TGraphDatum>
            void AllocateDatumObjects(TFirstGraphDatum& first_datum, TGraphDatum&... more_data)
            {
                AllocateDatum(first_datum);
                AllocateDatumObjects(more_data...);
            }

            template<typename TGraphDatum>
            void AllocateDatum(TGraphDatum& graph_datum)
            {
                graph_datum.Allocate(m_origin_graph);
            }

            template<typename TGraphDatum>
            void GatherDatum(TGraphDatum& graph_datum)
            {
                graph_datum.Gather(m_origin_graph);
            }

        private:
            void AllocateDevMirror()
            {
                index_t nnodes, nedges;

                m_dev_mirror.nnodes = nnodes = m_origin_graph.nnodes;
                m_dev_mirror.nedges = nedges = m_origin_graph.nedges;

                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.row_start, (nnodes + 1) * sizeof(index_t))); // malloc and copy +1 for the row_start's extra cell
                GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.row_start, m_origin_graph.row_start, (nnodes + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_mirror.edge_dst, nedges * sizeof(index_t)));
                GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_mirror.edge_dst, m_origin_graph.edge_dst, nedges * sizeof(index_t), cudaMemcpyHostToDevice));
            }

            void DeallocateDevMirror()
            {
                GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.row_start));
                GROUTE_CUDA_CHECK(cudaFree(m_dev_mirror.edge_dst));

                m_dev_mirror.row_start = nullptr;
                m_dev_mirror.edge_dst = nullptr;
            }
        };

        template<typename T>
        class EdgeInputDatum
        {
        public:
            typedef dev::GraphDatum < T > DeviceObjectType;
               
        private:
            T* m_edge_data;
            std::vector<T> m_ones;

            DeviceObjectType m_dev_datum;

        public:
            EdgeInputDatum() { }

            EdgeInputDatum(CSRGraphAllocator& graph_allocator)
            {
                // Will call this->SetNumSegs and this->AllocateDevSeg with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~EdgeInputDatum()
            {
                Deallocate();
            }

            void Deallocate()
            {
                GROUTE_CUDA_CHECK(cudaFree(m_dev_datum.data_ptr));

                m_dev_datum.data_ptr = nullptr;
                m_dev_datum.size = 0;
            }

            void Allocate(const host::CSRGraph& host_graph)
            {
                Deallocate();

                if (host_graph.edge_weights == nullptr) 
                {
                    printf("\nWarning: Expecting edge weights, falling back to all one's weights (use gen_weights and gen_weight_range).\n\n");
                   
                    m_ones = std::vector<T>(host_graph.nedges, 1);
                    m_edge_data = m_ones.data(); 
                }

                else
                {
                    m_edge_data = host_graph.edge_weights; // Bind to edge_weights from the original graph
                }

                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_datum.data_ptr, host_graph.nedges * sizeof(T)));
                m_dev_datum.size = host_graph.nedges;

                GROUTE_CUDA_CHECK(cudaMemcpy(m_dev_datum.data_ptr, m_edge_data, host_graph.nedges * sizeof(T), cudaMemcpyHostToDevice));
            }

            T* GetHostDataPtr() { return m_edge_data; }

            const DeviceObjectType& DeviceObject() const
            {
                return m_dev_datum;
            }
        };

        template<typename T>
        class NodeOutputDatum
        {
        public:
            typedef dev::GraphDatum < T > DeviceObjectType;

        private:
            std::vector<T> m_host_data;
            DeviceObjectType m_dev_datum;

        public:
            NodeOutputDatum() { }

            NodeOutputDatum(CSRGraphAllocator& graph_allocator)
            {
                // Will call this->Allocate with the correct device context 
                graph_allocator.AllocateDatum(*this);
            }

            ~NodeOutputDatum()
            {
                Deallocate();
            }

            void Deallocate()
            {
                GROUTE_CUDA_CHECK(cudaFree(m_dev_datum.data_ptr));

                m_dev_datum.data_ptr = nullptr;
                m_dev_datum.size = 0;
            }

            void Allocate(const host::CSRGraph& host_graph)
            {
                Deallocate();

                GROUTE_CUDA_CHECK(cudaMalloc(&m_dev_datum.data_ptr, host_graph.nnodes * sizeof(T)));
                m_dev_datum.size = host_graph.nnodes;
            }

            void Gather(const host::CSRGraph& host_graph)
            {
                m_host_data.resize(host_graph.nnodes);

                GROUTE_CUDA_CHECK(cudaMemcpy(
                    &m_host_data[0], m_dev_datum.data_ptr,
                    host_graph.nnodes * sizeof(T), cudaMemcpyDeviceToHost));
            }

            const std::vector<T>& GetHostData() { return m_host_data; }

            const DeviceObjectType& DeviceObject() const
            {
                return m_dev_datum;
            }
        };
    }
}
}


#endif // __GROUTE_GRAPHS_CSR_GRAPH_H
