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

#ifndef __GROUTE_GRAPHS_COO_GRAPH_H
#define __GROUTE_GRAPHS_COO_GRAPH_H

#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include <groute/graphs/common.h>

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)

namespace groute {
namespace graphs {

    namespace dev {

        class EdgeList // Graph abstraction  
        {
        private:
            const Edge* const m_ptr;
            size_t m_size;

        public:
            __host__ __device__ EdgeList(const Edge* ptr, size_t size) : m_ptr(ptr), m_size(size) { }

            __host__ const Edge* GetPtr() const   { return m_ptr; }
            __host__ size_t GetSize() const             { return m_size; }

            __device__ __forceinline__ Edge GetEdge(unsigned int eid)
            {
                return m_ptr[eid < m_size ? eid : m_size - 1]; // clamp
            }
        };

        class Tree // Graph abstraction  
        {
        private:
            const int* m_ptr;
            size_t m_size;
            size_t m_offset;

        public:
            __host__ __device__ Tree(const int* ptr, size_t size, size_t offset = 0) :
                m_ptr(ptr), m_size(size), m_offset(offset) { }

            __device__ __forceinline__ Edge GetEdge(unsigned int eid)
            {
                eid = eid < m_size ? eid : m_size - 1; // clamp
                Edge e; e.u = m_offset + eid; e.v = m_ptr[eid];
                return e;
            }
        };

        template <typename T>
        class Irregular
        {
        private:
            T* m_ptr;
            size_t m_size;
            size_t m_offset;

        public:
            __host__ __device__ Irregular(T* ptr, size_t size, size_t virtual_offset = 0) :
                m_ptr(ptr), m_size(size), m_offset(virtual_offset) { }

            __host__ __device__ Irregular() : m_ptr(nullptr), m_size(0), m_offset(0) { }

            __host__ T* GetPtr() const   { return m_ptr; }
            __host__ size_t GetSize() const    { return m_size; }

            __device__ __forceinline__ int get_wid()
            {
                return m_offset + GTID;
            }

            __device__ __forceinline__ void write(int i, T val)
            {
                m_ptr[i - m_offset] = val;
            }

            __device__ __forceinline__ T write_atomicCAS(int i, T compare, T val)
            {
                return atomicCAS(m_ptr + (i - m_offset), compare, val);
            }

            __device__ __forceinline__ T read(int i) const
            {
                return m_ptr[i - m_offset];
            }

            __device__ __forceinline__ bool has(int i)
            {
                return (i - m_offset) < m_size;
            }
        };

        class Flag
        {
        private:
            int *m_dev_flag;
            int *m_shared_flag;

        public:
            Flag(int *dev_flag) : m_dev_flag(dev_flag)
            {
            }

            typedef int SharedData;

            __device__ __forceinline__ void init(SharedData &shared_flag)
            {
                m_shared_flag = &shared_flag;
                if (threadIdx.x == 0)
                {
                    *m_shared_flag = 0;
                }
                __syncthreads();
            }

            __device__ __forceinline__ void set()
            {
                *m_shared_flag = 1;
            }

            __device__ __forceinline__ void commit()
            {
                __syncthreads();
                if (threadIdx.x == 0 && *m_shared_flag == 1)
                {
                    *m_dev_flag = 1;
                }
            }
        };

    }

    class Flag
    {
    private:
        cudaStream_t m_stream;
        std::vector<int> m_host_flag;
        int *m_dev_flag;

    public:
        __host__ Flag(cudaStream_t stream) : m_stream(stream), m_host_flag(1)
        {
            m_host_flag[0] = 1;
            GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_dev_flag, sizeof(int)));
        }

        __host__ void free()
        {
            GROUTE_CUDA_CHECK(cudaFree(m_dev_flag));
        }

        __host__ void reset()
        {
            m_host_flag[0] = 1;
        }

        __host__ void zero()
        {
            m_host_flag[0] = 0;
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_dev_flag, &m_host_flag[0], sizeof(int), cudaMemcpyHostToDevice, m_stream));
            GROUTE_CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }

        __host__ bool gather()
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(&m_host_flag[0], m_dev_flag, sizeof(int), cudaMemcpyDeviceToHost, m_stream));
            GROUTE_CUDA_CHECK(cudaStreamSynchronize(m_stream));
            return check();
        }

        __host__ bool check() { return m_host_flag[0] != 0; }

        __host__ dev::Flag get_dev_flag() { return dev::Flag(m_dev_flag); }
    };
}
}

#endif // __GROUTE_GRAPHS_COO_GRAPH_H
