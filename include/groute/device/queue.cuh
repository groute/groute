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

#ifndef __GROUTE_WORK_QUEUE_H
#define __GROUTE_WORK_QUEUE_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>

#include <new> // need this for the in-memory ctor call in the move assignment operator below  

#include <cub/util_ptx.cuh>

#include <groute/event_pool.h>

#define WARP_SIZE 32
#define DBS 256

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)


namespace groute {
    namespace dev {

        __device__ __forceinline__ void warp_active_count(int &first, int& offset, int& total) {
            unsigned int active = __ballot(1);
            total = __popc(active);
            offset = __popc(active & cub::LaneMaskLt());
            first = __ffs(active) - 1;
        }

        //
        // worklist classes (device):
        //

        template<typename T>
        class Worklist
        {
        public:
            T* m_data;
            uint32_t* m_count;
            uint32_t m_capacity;

            __host__ __device__ Worklist(T* data, uint32_t* count, uint32_t capacity) : 
                m_data(data), m_count(count), m_capacity(capacity) { }

            __device__ __forceinline__ void append(const T& item) const
            {
                uint32_t allocation = atomicAdd(m_count, 1); // just a naive atomic add
                m_data[allocation] = item;
            }

            __device__ void append_warp(const T& item) const
            {
                int first, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(first, offset, total);

                if (offset == 0) {
                    allocation = atomicAdd((uint32_t *)m_count, total);
                    assert(allocation + total <= m_capacity);
                }
    
                allocation = cub::ShuffleIndex(allocation, first);
                m_data[allocation + offset] = item;
            }

            __device__ void append_warp(const T& item, int leader, int warp_count, int offset) const
            {
                uint32_t allocation = 0;

                if (cub::LaneId() == leader) // the leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_count, warp_count);
                    assert(allocation + warp_count <= m_capacity);
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[allocation + offset] = item;
            }

            __device__ __forceinline__ void reset() const
            {
                *m_count = 0;
            }

            __device__ __forceinline__ T read(int i) const
            {
                return m_data[i];
            }
                        
            __device__ __forceinline__ uint32_t len() const
            {
                return *m_count;
            }
        };

        template<typename T, bool POWER_OF_TWO = true>
        class CircularWorklist
        {
            T* m_data;
            volatile uint32_t *m_start, *m_end, *m_alloc_end;
            uint32_t m_capacity;

        public:
            __host__ __device__ CircularWorklist(T* data, uint32_t *start, uint32_t *end, uint32_t *alloc_end, uint32_t capacity) : 
                m_data(data), m_start(start), m_end(end), m_alloc_end(alloc_end), m_capacity((POWER_OF_TWO ? (capacity - 1) : (capacity)))
            {
                assert((capacity - 1 & capacity) == 0); // must be a power of two for handling circular overflow correctly  
            }

            __device__ __forceinline__ void reset()
            {
                *m_start = 0; 
                *m_end = 0;
                *m_alloc_end = 0;
            }

            __device__ __forceinline__ void pop_items(uint32_t items)
            {
                (*m_start) += items; 
            }

            __device__ __forceinline__ void sync_append_alloc()
            {
                *m_end = *m_alloc_end;
            }

            __device__ __forceinline__ void append(const T& item)
            {
                uint32_t allocation = atomicAdd((uint32_t *)m_alloc_end, 1);

                if (POWER_OF_TWO)
                    m_data[allocation & m_capacity] = item;
                else
                    m_data[allocation % m_capacity] = item;
            }

            __device__ void append_warp(const T& item)
            {
                int first, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(first, offset, total);

                if (offset == 0) // the leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_alloc_end, total);
                    assert((allocation + total) - *m_start < (POWER_OF_TWO ? (m_capacity + 1) : m_capacity));
                }
    
                allocation = cub::ShuffleIndex(allocation, first);
                
                if (POWER_OF_TWO)
                    m_data[(allocation + offset) & m_capacity] = item;
                else
                    m_data[(allocation + offset) % m_capacity] = item;
            }

            __device__ void append_warp(const T& item, int leader, int warp_count, int offset)
            {
                uint32_t allocation = 0;

                if (cub::LaneId() == leader) // the leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_alloc_end, warp_count);
                    assert((allocation + warp_count) - *m_start < (POWER_OF_TWO ? (m_capacity + 1) : m_capacity));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                
                if (POWER_OF_TWO)
                    m_data[(allocation + offset) & m_capacity] = item;
                else
                    m_data[(allocation + offset) % m_capacity] = item;
            }

            __device__ __forceinline__ void prepend(const T& item)
            {
                uint32_t allocation = atomicSub((uint32_t *)m_start, 1) - 1;

                if (POWER_OF_TWO)
                    m_data[allocation & m_capacity] = item;
                else
                    m_data[allocation % m_capacity] = item;
            }

            __device__ void prepend_warp(const T& item)
            {
                int first, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(first, offset, total);

                if (offset == 0) // the leader thread  
                {
                    allocation = atomicSub((uint32_t *)m_start, total) - total; // allocate 'total' items from the start
                    assert(*m_end - allocation < (POWER_OF_TWO ? (m_capacity + 1) : m_capacity));
                }
    
                allocation = cub::ShuffleIndex(allocation, first);
                
                if (POWER_OF_TWO)
                    m_data[(allocation + offset) & m_capacity] = item;
                else
                    m_data[(allocation + offset) % m_capacity] = item;
            }
                        
            __device__ void prepend_warp(const T& item, int leader, int warp_count, int offset)
            {
                uint32_t allocation = 0;

                if (cub::LaneId() == leader) // the leader thread  
                {
                    allocation = atomicSub((uint32_t *)m_start, warp_count) - warp_count; // allocate 'total' items from the start
                    assert(*m_end - allocation < (POWER_OF_TWO ? (m_capacity + 1) : m_capacity));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                
                if (POWER_OF_TWO)
                    m_data[(allocation + offset) & m_capacity] = item;
                else
                    m_data[(allocation + offset) % m_capacity] = item;
            }

            __device__ __forceinline__ T read(uint32_t i) const
            {
                if (POWER_OF_TWO)
                    return m_data[(*m_start + i) & m_capacity];
                else
                    return m_data[(*m_start + i) % m_capacity];
            }
                        
            __device__ __forceinline__ uint32_t size() const
            {
                uint32_t start = *m_start;
                uint32_t end = *m_end;

                if (POWER_OF_TWO)
                {
                    start = start & m_capacity;
                    end = end & m_capacity;
                    return end >= start ? end - start : ((m_capacity+1) - start + end); // normal and circular cases
                }
                else
                {
                    start = start % m_capacity;
                    end = end % m_capacity;
                    return end >= start ? end - start : (m_capacity - start + end); // normal and circular cases
                }
            }

            __device__ __forceinline__ uint32_t get_alloc_count_and_sync() const
            {
                uint32_t end = *m_end;
                uint32_t alloc_end = *m_alloc_end;

                uint32_t count;

                if (POWER_OF_TWO)
                {
                    end = end & m_capacity;
                    alloc_end = alloc_end & m_capacity;
                    count = alloc_end >= end ? alloc_end - end : ((m_capacity+1) - end + alloc_end); // normal and circular cases
                }
                else
                {
                    end = end % m_capacity;
                    alloc_end = alloc_end % m_capacity;
                    count = alloc_end >= end ? alloc_end - end : (m_capacity - end + alloc_end); // normal and circular cases
                }
                
                // sync alloc
                *m_end = *m_alloc_end;
                return count;
            }

            __device__ __forceinline__ uint32_t get_start() const
            {
                return *m_start;
            }

            __device__ __forceinline__ uint32_t get_start_diff(uint32_t prev_start) const
            {
                return prev_start - *m_start;
            }
        };
    }


    // 
    // worklist control kernels:  
    //

    template<typename T>
    __global__ void WorklistReset(dev::Worklist<T> worklist)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            worklist.reset();
    }

    static __global__ void ResetCounters(uint32_t* counters, uint32_t num_counters)
    {
        if (TID_1D < num_counters)
            counters[TID_1D] = 0;
    }

    template<typename T>
    __global__ void WorklistAppendItem(dev::Worklist<T> worklist, T item)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            worklist.append(item);
    }

    template<typename T>
    __global__ void CircularWorklistReset(dev::CircularWorklist<T> worklist)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            worklist.reset();
    }

    template<typename T>
    __global__ void CircularWorklistPopItems(dev::CircularWorklist<T> worklist, uint32_t items)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            worklist.pop_items(items);
    }

    template<typename T>
    __global__ void CircularWorklistAppendItem(dev::CircularWorklist<T> worklist, T item)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            worklist.append(item);
            worklist.sync_append_alloc();
        }
    }

    template<typename T>
    __global__ void CircularWorklistSyncAppendAlloc(dev::CircularWorklist<T> worklist)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            worklist.sync_append_alloc();
    }

    // 
    // worklist control classes (host):  
    //

    template<typename T>
    class Worklist
    {
        enum { WS = 32 };

        //
        // device buffer / counters 
        //
        T* m_data;
        bool m_mem_owner;

        uint32_t *m_counters;
        uint32_t m_capacity;
        uint32_t *m_host_count;

        int32_t m_current_slot;
    
    public:
        Worklist(uint32_t capacity = 0) : m_data(nullptr), m_mem_owner(true), m_counters(nullptr), m_capacity(capacity), m_current_slot(-1)
        {
            Alloc();
        }

        Worklist(T* mem_buffer, uint32_t mem_size) : m_data(mem_buffer), m_mem_owner(false), m_counters(nullptr), m_capacity(mem_size), m_current_slot(-1)
        {
            Alloc();
        }

        Worklist(const Worklist& other) = delete;

        Worklist(Worklist&& other)
        {
            *this = std::move(other);
        }

    private:
        Worklist& operator=(const Worklist& other) = default;

    public:
        Worklist& operator=(Worklist&& other)
        {
            *this = other;              // First copy all fields  
            new (&other) Worklist(0);   // Clear up other

            return (*this);
        }

        ~Worklist()
        {
            Free();
        }

        typedef dev::Worklist<T> DeviceObjectType;
    
    private:
        void Alloc()
        {
            if (m_capacity == 0) return;

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(T) * m_capacity));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_counters, WS * sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_count, sizeof(uint32_t)));
        }
    
        void Free()
        {
            if (m_capacity == 0) return;

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaFree(m_data));
            GROUTE_CUDA_CHECK(cudaFree(m_counters));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_count));
        }
    
    public:
        DeviceObjectType DeviceObject() const
        {
            assert(m_current_slot >= 0 && m_current_slot < WS);
            return dev::Worklist<T>(m_data, m_counters + m_current_slot, m_capacity);
        }

        void ResetAsync(cudaStream_t stream)
        {
            m_current_slot = (m_current_slot + 1) % WS;
            if (m_current_slot == 0)
            {
                ResetCounters <<< 1, WS, 0, stream >>>(m_counters, WS);
            }
        }

        void ResetAsync(Stream& stream)
        {
            ResetAsync(stream.cuda_stream);
        }

        void AppendItemAsync(cudaStream_t stream, const T& item) const
        {
            WorklistAppendItem <<<1, 1, 0, stream >>>(DeviceObject(), item);
        }

        T* GetDataPtr() const { return m_data; }
        
        uint32_t GetLength(const Stream& stream) const
        {
            assert(m_current_slot >= 0 && m_current_slot < WS);

            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_count, m_counters + m_current_slot, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            stream.Sync();
    
            if(*m_host_count > m_capacity)
            {
                printf(
                    "\n\nCritical Warning: worklist has overflowed, please allocate more memory \n\t\[endpoint: %d, name: %s, instance id: %d, \n\t capacity: %d, overflow: %d] \nExiting \n\n", 
                    (Endpoint::identity_type)0, "", -1, m_capacity, *m_host_count - m_capacity);
                exit(1);
            }

            return *m_host_count;
        }

        void PrintOffsetsDebug(const Stream& stream) const
        {
            printf("\nWorklist (Debug): count: %u (capacity: %u)", 
                GetLength(stream), m_capacity);
        }

        Segment<T> ToSeg(const Stream& stream) const
        {
            return Segment<T>(GetDataPtr(), GetLength(stream));
        }
    };

    class MemoryMonitor
    {
        struct Entry
        {
            
        };

        std::string m_app, m_dataset;

        std::atomic<int> m_id_gen;

        MemoryMonitor() : m_id_gen(0) { }

        static MemoryMonitor& Instance()
        {
            static MemoryMonitor monitor;
            return monitor;
        }

    public:
        static void Init(const std::string& app, const std::string& dataset)
        {
            Instance().m_app = app;
            Instance().m_dataset = dataset;
        }

        static int Register()
        {
            return Instance().m_id_gen++;
        }
    };
    
    template<typename T>
    class CircularWorklist
    {
        //
        // device buffer / counters 
        //
        
        T* m_data;
        bool m_mem_owner;

        uint32_t *m_start, *m_end, *m_alloc_end;

        // Host buffers  
        uint32_t *m_host_start, *m_host_end, *m_host_alloc_end;
        
        uint32_t m_capacity;

        int m_instance_id;
        mutable uint32_t m_max_usage;

        Endpoint m_endpoint;
        const char* m_name;
    
    public:
        CircularWorklist(uint32_t capacity = 0, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(nullptr), m_mem_owner(true), 
            m_start(nullptr), m_end(nullptr), m_alloc_end(nullptr), 
            m_host_start(nullptr), m_host_end(nullptr), m_host_alloc_end(nullptr), 
            m_capacity(capacity == 0 ? 0 : next_power_2(capacity)),
            m_endpoint(endpoint), m_name(name),
            m_instance_id(0), m_max_usage(0)
        {
            Alloc();
        }

        CircularWorklist(T *mem_buffer, uint32_t mem_size, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(mem_buffer), m_mem_owner(false), 
            m_start(nullptr), m_end(nullptr), m_alloc_end(nullptr), 
            m_host_start(nullptr), m_host_end(nullptr), m_host_alloc_end(nullptr), 
            m_capacity(mem_size),
            m_endpoint(endpoint), m_name(name),
            m_instance_id(0), m_max_usage(0)
        {
            Alloc();
        }

        CircularWorklist(const CircularWorklist& other) = delete;

        CircularWorklist(CircularWorklist&& other)
        {
            *this = std::move(other);
        }

    private:
        CircularWorklist& operator=(const CircularWorklist& other) = default;

    public:
        CircularWorklist& operator=(CircularWorklist&& other)
        {
            *this = other;                      // First copy all fields 
            new (&other) CircularWorklist(0);   // Clear up other

            return (*this);
        }

        ~CircularWorklist()
        {
            Free();
        }
        
        typedef dev::CircularWorklist<T> DeviceObjectType;
    
    private:
        void Alloc()
        {
            if (m_capacity == 0) return;

            assert((m_capacity - 1 & m_capacity) == 0);

            m_instance_id = MemoryMonitor::Register();
    
            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(T) * m_capacity));

            GROUTE_CUDA_CHECK(cudaMalloc(&m_start, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_end, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_alloc_end, sizeof(uint32_t)));

            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_start, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_end, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_alloc_end, sizeof(uint32_t)));
        }
    
        void Free()
        {
            if (m_capacity == 0) return;

            //printf("\nCircular worklist usage stats (instance id: %d, capacity: %d, max_usage: %d)\n", 
            //        m_instance_id, m_capacity, m_max_usage);

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaFree(m_data));

            GROUTE_CUDA_CHECK(cudaFree(m_start));
            GROUTE_CUDA_CHECK(cudaFree(m_end));
            GROUTE_CUDA_CHECK(cudaFree(m_alloc_end));

            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_start));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_end));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_alloc_end));
        }       

        void GetRealBounds(uint32_t& start, uint32_t& end, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_start, m_start, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();

            start = *m_host_start;
            end = *m_host_end;

            assert(end - start < m_capacity);

            if (end - start >= m_capacity)
            {
                printf(
                    "\n\nCritical Warning: circular worklist has overflowed, please allocate more memory \n\t\[endpoint: %d, name: %s, instance id: %d, \n\t start: %d, end: %d, capacity: %d, overflow: %d] \nExiting \n\n", 
                    (Endpoint::identity_type)m_endpoint, m_name, m_instance_id, start, end, m_capacity, (end - start) - m_capacity);
                exit(1);
            }

            m_max_usage = std::max(m_max_usage, end - start);
        }
        
        void GetBounds(uint32_t& start, uint32_t& end, uint32_t& size, const Stream& stream) const
        {
            GetRealBounds(start, end, stream);

            start = start % m_capacity;
            end = end % m_capacity;

            size = end >= start ? end - start : (m_capacity - start + end); // normal and circular cases
        }

        void GetAllocCount(uint32_t& former_end, uint32_t& alloc_end, uint32_t& count, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_alloc_end, m_alloc_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();

            former_end = *m_host_end;
            alloc_end = *m_host_alloc_end;

            former_end = former_end % m_capacity;
            alloc_end = alloc_end % m_capacity;

            count = alloc_end >= former_end ? alloc_end - former_end : (m_capacity - former_end + alloc_end); // normal and circular cases
        }
    
    public:
        
        struct Bounds
        {
            uint32_t start, end;

            Bounds(uint32_t start, uint32_t end) : start(start), end(end) { }
            Bounds() : start(0), end(0) { }

            int GetLength() const { return end - start; } // Works also if numbers over/under flow
            Bounds Exclude(Bounds other) const { return Bounds(other.end, end); }
        };

        Bounds GetBounds(const Stream& stream)
        {
            Bounds bounds;
            GetRealBounds(bounds.start, bounds.end, stream);
            return bounds;
        }

        DeviceObjectType DeviceObject() const
        {
            return dev::CircularWorklist<T>(m_data, m_start, m_end, m_alloc_end, m_capacity);
        }

        void ResetAsync(cudaStream_t stream) const
        {
            CircularWorklistReset<<<1, 1, 0, stream >>>(DeviceObject());
        }

        void SyncAppendAllocAsync(cudaStream_t stream) const 
        {
            CircularWorklistSyncAppendAlloc<<<1, 1, 0, stream >>>(DeviceObject());
        }

        void AppendItemAsync(cudaStream_t stream, const T& item) const
        {
            CircularWorklistAppendItem <<<1, 1, 0, stream >>>(DeviceObject(), item);
        }

        void PopItemsAsync(uint32_t items, cudaStream_t stream) const 
        {
            if (items == 0) return;

            CircularWorklistPopItems <<<1, 1, 0, stream >>>(DeviceObject(), items);
        }

        void PopItemsAsync(uint32_t items, const Stream& stream) const 
        {
            if (items == 0) return;

            CircularWorklistPopItems <<<1, 1, 0, stream.cuda_stream >>>(DeviceObject(), items);
        }

        int GetLength(const Stream& stream) const
        {
            uint32_t start, end, size;
            GetBounds(start, end, size, stream);

            return size;
        }

        int GetSpace(const Stream& stream) const
        {
            return m_capacity - GetLength(stream);
        }

        int GetSpace(Bounds bounds) const
        {
            return m_capacity - bounds.GetLength();
        }
        
        int GetAllocCount(const Stream& stream) const
        {
            uint32_t former_end, alloc_end, count;
            GetAllocCount(former_end, alloc_end, count, stream);

            return count;
        }

        int GetAllocCountAndSync(const Stream& stream) const
        {
            uint32_t count = GetAllocCount(stream);
            SyncAppendAllocAsync(stream.cuda_stream);
            return count;
        }

        void GetOffsetsDebug(uint32_t& capacity, uint32_t& start, uint32_t& end, uint32_t& alloc_end, uint32_t& size, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_start, m_start, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_alloc_end, m_alloc_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();
            capacity = m_capacity;
            start = *m_host_start;
            end = *m_host_end;
            alloc_end = *m_host_alloc_end;
            size = end - start;
        }

        void PrintOffsetsDebug(const Stream& stream) const
        {
            uint32_t capacity, start, end, alloc_end, size;
            GetOffsetsDebug(capacity, start, end, alloc_end, size, stream);
            printf("\nCircularWorklist (Debug): start: %u, end: %u, alloc_end: %u, size: %u (capacity: %u)", 
                start, end, alloc_end, size, capacity);
        }

        std::vector< Segment<T> > GetSegs(Bounds bounds)
        {
            uint32_t start = bounds.start, end = bounds.end, size = bounds.GetLength();

            start = start % m_capacity;
            end = end % m_capacity;

            //size = end >= start ? end - start : (m_capacity - start + end); // normal and circular cases

            std::vector< Segment<T> > segs;

            if (end > start) // normal case
            {
                segs.push_back(Segment<T>(m_data + start, size));
            }
            else if (start > end)
            {
                segs.push_back(Segment<T>(m_data + start, size - end));
                if (end > 0) {
                    segs.push_back(Segment<T>(m_data, end));
                }
            }

            // else empty

            return segs;
        }

        std::vector< Segment<T> > ToSegs(const Stream& stream)
        {
            return GetSegs(GetBounds(stream));
        }
    };
}

#endif // __GROUTE_WORK_QUEUE_H
