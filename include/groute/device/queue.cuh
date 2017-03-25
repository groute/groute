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
#include <new> // Used for the in-memory ctor call in the move assignment operator below  

#include <cub/util_ptx.cuh>
#include <groute/common.h>

//
// Common device-related MACROS
//

// Default block size for system kernels  
#define GROUTE_BLOCK_THREADS 256
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

//
//
//

namespace groute {
    namespace dev {

        __device__ __forceinline__ void warp_active_count(int &first, int& offset, int& total) {
            unsigned int active = __ballot(1);
            total = __popc(active);
            offset = __popc(active & cub::LaneMaskLt());
            first = __ffs(active) - 1;
        }

        //
        // Queue classes (device):
        //

        /*
        * @brief A device-level Queue (see host controller object below for usage)
        */
        template<typename T>
        class Queue
        {
            T* m_data;
            uint32_t* m_count;
            uint32_t m_capacity;
            
        public:
            __host__ __device__ Queue(T* data, uint32_t* count, uint32_t capacity) : 
                m_data(data), m_count(count), m_capacity(capacity) { }

            __device__ __forceinline__ void append(const T& item) const
            {
                uint32_t allocation = atomicAdd(m_count, 1); // Just a naive atomic add
                m_data[allocation] = item;
            }

            __device__ void append_warp(const T& item) const
            {
                int leader, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(leader, offset, total);

                if (offset == 0) {
                    allocation = atomicAdd((uint32_t *)m_count, total);
                    assert(allocation + total <= m_capacity);
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[allocation + offset] = item;
            }

            __device__ void append_warp(const T& item, int leader, int total, int offset) const
            {
                uint32_t allocation = 0;

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_count, total);
                    assert(allocation + total <= m_capacity);
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
                        
            __device__ __forceinline__ uint32_t count() const
            {
                return *m_count;
            }

            __device__ __forceinline__ void pop(uint32_t count) const
            {
                assert(*m_count >= count);
                *m_count -= count;
            }

            __device__ __forceinline__  T* data_ptr() { return m_data; }
        };

        /*
        * @brief A device-level Producer-Consumer Queue (see host controller object below for usage and notes)
        */
        template<typename T>
        class PCQueue
        {
            T* m_data;
            volatile uint32_t *m_start, *m_end, *m_pending;
            uint32_t m_capacity_mask;

        public:
            __host__ __device__ PCQueue(T* data, uint32_t *start, uint32_t *end, uint32_t *pending, uint32_t capacity) : 
                m_data(data), m_start(start), m_end(end), m_pending(pending), m_capacity_mask((capacity - 1))
            {
                assert((capacity - 1 & capacity) == 0); // Must be a power of two for handling circular overflow correctly  
            }

            __device__ __forceinline__ void reset() const
            {
                *m_start = 0; 
                *m_end = 0;
                *m_pending = 0;
            }

            __device__ __forceinline__ void pop(uint32_t count) const
            {
                (*m_start) += count; 
            }

            __device__ __forceinline__ void append(const T& item)
            {
                uint32_t allocation = atomicAdd((uint32_t *)m_pending, 1);
                m_data[allocation & m_capacity_mask] = item;
            }

            __device__ void append_warp(const T& item)
            {
                int leader, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(leader, offset, total);

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_pending, total);
                    assert((allocation + total) - *m_start < (m_capacity_mask + 1));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[(allocation + offset) & m_capacity_mask] = item;
            }

            __device__ void append_warp(const T& item, int leader, int total, int offset)
            {
                uint32_t allocation = 0;

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicAdd((uint32_t *)m_pending, total);
                    assert((allocation + total) - *m_start < (m_capacity_mask + 1));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[(allocation + offset) & m_capacity_mask] = item;
            }

            __device__ __forceinline__ void prepend(const T& item)
            {
                uint32_t allocation = atomicSub((uint32_t *)m_start, 1) - 1;
                m_data[allocation & m_capacity_mask] = item;
            }

            __device__ void prepend_warp(const T& item)
            {
                int leader, total, offset; 
                uint32_t allocation = 0;

                warp_active_count(leader, offset, total);

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicSub((uint32_t *)m_start, total) - total; // Allocate 'total' items from the start
                    assert(*m_end - allocation < (m_capacity_mask + 1));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[(allocation + offset) & m_capacity_mask] = item;
            }
                        
            __device__ void prepend_warp(const T& item, int leader, int total, int offset)
            {
                uint32_t allocation = 0;

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicSub((uint32_t *)m_start, total) - total; // Allocate 'total' items from the start
                    assert(*m_end - allocation < (m_capacity_mask + 1));
                }
    
                allocation = cub::ShuffleIndex(allocation, leader);
                m_data[(allocation + offset) & m_capacity_mask] = item;
            }

            __device__ __forceinline__ T read(uint32_t i) const
            {
                return m_data[(*m_start + i) & m_capacity_mask];
            }
                        
            __device__ __forceinline__ uint32_t count() const
            {
                return *m_end - *m_start;
            }

            // Returns the 'count' of pending items and commits
            __device__ __forceinline__ uint32_t commit_pending() const
            {
                uint32_t count = *m_pending - *m_end;
                
                // Sync end with pending, this makes the pushed items visible to the consumer
                *m_end = *m_pending;
                return count;
            }

            __device__ __forceinline__ uint32_t get_start() const
            {
                return *m_start;
            }

            __device__ __forceinline__ uint32_t get_start_delta(uint32_t prev_start) const
            {
                return prev_start - *m_start;
            }
        };
    }


    // 
    // Queue control kernels:  
    //

    namespace queue     {
    namespace kernels   {

        template<typename T>
        __global__ void QueueReset(dev::Queue<T> queue)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
                queue.reset();
        }

        static __global__ void ResetCounters(uint32_t* counters, uint32_t num_counters)
        {
            if (TID_1D < num_counters)
                counters[TID_1D] = 0;
        }

        template<typename T>
        __global__ void QueueAppendItem(dev::Queue<T> queue, T item)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
                queue.append(item);
        }

        template<typename T>
        __global__ void PCQueueReset(dev::PCQueue<T> pcqueue)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
                pcqueue.reset();
        }

        template<typename T>
        __global__ void PCQueuePop(dev::PCQueue<T> pcqueue, uint32_t count)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
                pcqueue.pop(count);
        }

        template<typename T>
        __global__ void PCQueueAppendItem(dev::PCQueue<T> pcqueue, T item)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                pcqueue.append(item);
                pcqueue.commit_pending();
            }
        }

        template<typename T>
        __global__ void PCQueueCommitPending(dev::PCQueue<T> pcqueue)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
                pcqueue.commit_pending();
        }
    }
    }

    //
    // Queue memory monitor
    //
    class QueueMemoryMonitor
    {
        struct Entry
        {
            Endpoint endpoint;
            const char* name;
            uint32_t capacity;
            uint32_t max_usage;

            Entry() : endpoint(), name(nullptr), capacity(0), max_usage(0) { }
            Entry(Endpoint endpoint, const char* name, uint32_t capacity) : endpoint(endpoint), name(name), capacity(capacity), max_usage(0) { }
        };

        std::vector<Entry> m_entries;
        std::atomic<bool> m_exiting;

        QueueMemoryMonitor() : m_exiting(false) { }

        static QueueMemoryMonitor& Instance()
        {
            static QueueMemoryMonitor monitor; // Lazy singleton (per compilation unit)
            return monitor;
        }

        int RegisterInternal(uint32_t capacity, Endpoint endpoint, const char* name)
        {
            m_entries.push_back(Entry(endpoint, name, capacity));
            return m_entries.size() - 1;
        }

        void ReportUsageInternal(int index, uint32_t usage)
        {
            if (index < 0 || index >= m_entries.size()) return;

            // Keeping the max_usage update not thread-safe.
            // As long as we dont miss an overflow its ok

            if (usage > m_entries[index].max_usage)
                m_entries[index].max_usage = usage;

            if (usage > m_entries[index].capacity)
            {
                //
                // We got an overflow, report overall usage stats and exit 
                //

                bool exiting = m_exiting;
                if (exiting || !m_exiting.compare_exchange_strong(exiting, true)) return; // Avoid printing stats by more than one thread  

                std::map<Endpoint, std::map<std::string, std::vector<Entry>>> grouped_entries;
                size_t name_max = 0;

                for (const auto& entry : m_entries)
                {
                    std::string name(entry.name);
                    name_max = std::max(name.size(), name_max);

                    grouped_entries[entry.endpoint][name].push_back(entry); // Group by endpoint and name
                }
                
                printf("\nQueue has overflowed, dumping overall queue memory statistics: ");

                for (const auto& ep : grouped_entries)
                {
                    Endpoint endpoint = ep.first;
                    printf("\nEndpoint %d: ", (Endpoint::identity_type)endpoint);
                    for (const auto& np : ep.second)
                    {
                        std::string name = np.first;

                        for (size_t i = 0; i < np.second.size(); ++i)
                        {
                            const Entry& entry = np.second[i];

                            if(i == 0)  printf("\n\t[name: '%s'%s", name.c_str(), std::string(name_max + 3 - name.size(), ' ').c_str());
                            else        printf("\n\t[name: '%s(%llu)'%s", name.c_str(), i + 1, std::string(name_max - name.size(), ' ').c_str());

                            printf(", capacity: %10d, usage: %10d, ratio: %.3f", entry.capacity, entry.max_usage, (double)entry.max_usage/entry.capacity);

                            if (entry.max_usage > entry.capacity)   printf(", OVERFLOW: %10d]", entry.max_usage - entry.capacity);
                            else                                    printf("]");
                        }
                    }
                }

                printf("\n\nUse -wl_alloc_factor_{name} flags to reconfigure memory allocation factors");
                printf("\nExiting with code %d\n", 16);
                exit(16);
            }
        }

    public:
        static int Register(uint32_t capacity, Endpoint endpoint, const char* name)
        {
            return Instance().RegisterInternal(capacity, endpoint, name);
        }

        static void ReportUsage(int instance_id, uint32_t usage)
        {
            Instance().ReportUsageInternal(instance_id, usage);
        }
    };


    // 
    // Queue control classes (host):  
    //

    /*
    * @brief Host controller object for dev::Queue (see above)
    */
    template<typename T>
    class Queue
    {
        enum { NUM_COUNTERS = 32 }; // Number of counter slots

        // device buffer / counters 
        T* m_data;
        uint32_t *m_counters;

        // Pinned host counter
        uint32_t *m_host_count;
        
        uint32_t m_capacity;
        int32_t m_current_slot; // The currently used counter slot (see ResetAsync method)

        bool m_mem_owner;
        int m_instance_id;

        Queue& operator=(const Queue& other) = default; // For private use only
        
        void Alloc(Endpoint endpoint, const char* name)
        {
            if (m_capacity == 0) return;
            
            m_instance_id = QueueMemoryMonitor::Register(m_capacity, endpoint, name);

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(T) * m_capacity));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_counters, NUM_COUNTERS * sizeof(uint32_t)));
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
        Queue(uint32_t capacity = 0, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(nullptr), m_mem_owner(true), m_counters(nullptr), m_capacity(capacity), m_current_slot(-1), m_instance_id(-1)
        {
            Alloc(endpoint, name);
        }

        Queue(T* mem_buffer, uint32_t mem_size, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(mem_buffer), m_mem_owner(false), m_counters(nullptr), m_capacity(mem_size), m_current_slot(-1), m_instance_id(-1)
        {
            Alloc(endpoint, name);
        }

        Queue(const Queue& other) = delete;

        Queue(Queue&& other)
        {
            *this = std::move(other);
        }

        Queue& operator=(Queue&& other)
        {
            *this = other;           // First copy all fields  
            new (&other) Queue(0);   // Clear up other

            return (*this);
        }

        ~Queue()
        {
            Free();
        }

        T* GetDeviceDataPtr() const { return m_data; }

        typedef dev::Queue<T> DeviceObjectType;

        DeviceObjectType DeviceObject() const
        {
            assert(m_current_slot >= 0 && m_current_slot < NUM_COUNTERS);
            return dev::Queue<T>(m_data, m_counters + m_current_slot, m_capacity);
        }
        
        void ResetAsync(cudaStream_t stream)
        {
            //
            // We use multiple counter slots to avoid running a kernel each time a reset is required  
            //

            m_current_slot = (m_current_slot + 1) % NUM_COUNTERS;
            if (m_current_slot == 0)
            {
                queue::kernels::ResetCounters <<< 1, NUM_COUNTERS, 0, stream >>>(m_counters, NUM_COUNTERS);
            }
        }

        void AppendItemAsync(cudaStream_t stream, const T& item) const
        {
            queue::kernels::QueueAppendItem <<<1, 1, 0, stream >>>(DeviceObject(), item);
        }

        void ResetAsync(const Stream& stream)
        {
            ResetAsync(stream.cuda_stream);
        }
                
        void AppendItemAsync(const Stream& stream, const T& item) const
        {
            AppendItemAsync(stream.cuda_stream, item);
        }
        
        uint32_t GetCount(const Stream& stream) const
        {
            assert(m_current_slot >= 0 && m_current_slot < NUM_COUNTERS);

            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_count, m_counters + m_current_slot, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            stream.Sync();
            
            QueueMemoryMonitor::ReportUsage(m_instance_id, *m_host_count);

            return *m_host_count;
        }

        Segment<T> GetSeg(const Stream& stream) const
        {
            return Segment<T>(GetDeviceDataPtr(), GetCount(stream));
        }

        void PrintOffsets(const Stream& stream) const
        {
            printf("\nQueue (Debug): count: %u (capacity: %u)", 
                GetCount(stream), m_capacity);
        }
    };
    
    /*
    * @brief Host controller object for dev::PCQueue (see above)
    * @notes:
        -> The producer-consumer queue supports concurrent access from two kernels/streams running over the same device
        -> All threads in the consumer kernel can read items from the queue, but only a single thread pop's after all others are done
        -> All threads in the producer kernel can push values into the queue, but only a single thread commit's at the end, signaling to readers
        -> The consumer kernel can also have a write phase where it 'prepends' items into the queue
        -> In order to support a lock-free implementation (with only atomics), the capacity of the queue must be of power-of-two so that 
           numeric under/over-flows are correctly managed allong with the modulo test
        -> Power-of-two capacity enables the use of a capacity mask for the modulo test 
        -> The queue does not support blocking writers when it's full, instead, a memory overflow is reported and the application will terminate  
    */
    template<typename T>
    class PCQueue
    {
        // Device buffer / counters 
        T* m_data;
        uint32_t *m_start, *m_end, *m_pending;

        // Pinned host counters  
        uint32_t *m_host_start, *m_host_end, *m_host_pending;
        
        // Capacity, must be a power-of-two 
        uint32_t m_capacity;
        
        bool m_mem_owner;
        int m_instance_id;

        PCQueue& operator=(const PCQueue& other) = default; // For private use only

        void Alloc(Endpoint endpoint, const char* name)
        {
            if (m_capacity == 0) return;

            if ((m_capacity - 1 & m_capacity) != 0) throw groute::exception("PCQueue must have power-of-two capacity");

            m_instance_id = QueueMemoryMonitor::Register(m_capacity, endpoint, name);
    
            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(T) * m_capacity));

            GROUTE_CUDA_CHECK(cudaMalloc(&m_start, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_end, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_pending, sizeof(uint32_t)));

            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_start, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_end, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_pending, sizeof(uint32_t)));
        }
    
        void Free()
        {
            if (m_capacity == 0) return;

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaFree(m_data));

            GROUTE_CUDA_CHECK(cudaFree(m_start));
            GROUTE_CUDA_CHECK(cudaFree(m_end));
            GROUTE_CUDA_CHECK(cudaFree(m_pending));

            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_start));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_end));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_pending));
        }       

        void GetActualBounds(uint32_t& start, uint32_t& end, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_start, m_start, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();

            start = *m_host_start;
            end = *m_host_end;

            QueueMemoryMonitor::ReportUsage(m_instance_id, end - start);
        }
        
        void GetBounds(uint32_t& start, uint32_t& end, uint32_t& size, const Stream& stream) const
        {
            GetActualBounds(start, end, stream);

            start = start % m_capacity;
            end = end % m_capacity;

            size = end >= start ? end - start : (m_capacity - start + end); // normal and circular cases
        }

        void GetPendingCount(uint32_t& end, uint32_t& pending, uint32_t& count, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_pending, m_pending, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();

            end = *m_host_end;
            pending = *m_host_pending;

            end = end % m_capacity;
            pending = pending % m_capacity;

            count = pending >= end ? pending - end : (m_capacity - end + pending); // normal and circular cases
        }
    
    public:
        PCQueue(uint32_t capacity = 0, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(nullptr), m_mem_owner(true), 
            m_start(nullptr), m_end(nullptr), m_pending(nullptr), 
            m_host_start(nullptr), m_host_end(nullptr), m_host_pending(nullptr), 
            m_capacity(capacity == 0 ? 0 : next_power_2(capacity)),
            m_instance_id(-1)
        {
            Alloc(endpoint, name);
        }

        PCQueue(T *mem_buffer, uint32_t mem_size, Endpoint endpoint = Endpoint(), const char* name = "") : 
            m_data(mem_buffer), m_mem_owner(false), 
            m_start(nullptr), m_end(nullptr), m_pending(nullptr), 
            m_host_start(nullptr), m_host_end(nullptr), m_host_pending(nullptr), 
            m_capacity(mem_size),
            m_instance_id(-1)
        {
            Alloc(endpoint, name);
        }

        PCQueue(const PCQueue& other) = delete;

        PCQueue(PCQueue&& other)
        {
            *this = std::move(other);
        }

        PCQueue& operator=(PCQueue&& other)
        {
            *this = other;             // First copy all fields 
            new (&other) PCQueue(0);   // Clear up other

            return (*this);
        }

        ~PCQueue()
        {
            Free();
        }
        
        typedef dev::PCQueue<T> DeviceObjectType;

        DeviceObjectType DeviceObject() const
        {
            return dev::PCQueue<T>(m_data, m_start, m_end, m_pending, m_capacity);
        }

        void ResetAsync(cudaStream_t stream) const
        {
            queue::kernels::PCQueueReset<<<1, 1, 0, stream >>>(DeviceObject());
        }

        void CommitPendingAsync(cudaStream_t stream) const 
        {
            queue::kernels::PCQueueCommitPending<<<1, 1, 0, stream >>>(DeviceObject());
        }

        void AppendItemAsync(cudaStream_t stream, const T& item) const
        {
            queue::kernels::PCQueueAppendItem <<<1, 1, 0, stream >>>(DeviceObject(), item);
        }

        void PopAsync(uint32_t count, cudaStream_t stream) const 
        {
            if (count == 0) return;

            queue::kernels::PCQueuePop <<<1, 1, 0, stream >>>(DeviceObject(), count);
        }

        //
        // groute::Stream overloads  
        //
        void ResetAsync(const Stream& stream) const
        {
            ResetAsync(stream.cuda_stream);
        }

        void CommitPendingAsync(const Stream& stream) const
        {
            CommitPendingAsync(stream.cuda_stream);
        }

        void AppendItemAsync(const Stream& stream, const T& item) const
        {
            AppendItemAsync(stream.cuda_stream);
        }

        void PopAsync(uint32_t count, const Stream& stream) const 
        {
            PopAsync(count, stream.cuda_stream);
        }

        uint32_t GetCount(const Stream& stream) const
        {
            uint32_t start, end, size;
            GetBounds(start, end, size, stream);

            return size;
        }
        
        uint32_t GetPendingCount(const Stream& stream) const
        {
            uint32_t end, pending, count;
            GetPendingCount(end, pending, count, stream);

            return count;
        }

        uint32_t GetSpace(const Stream& stream) const
        {
            return m_capacity - GetCount(stream);
        }

        //
        // Helper class for queue bounds
        //
        struct Bounds
        {
            uint32_t start, end;

            Bounds(uint32_t start, uint32_t end) : start(start), end(end) { }
            Bounds() : start(0), end(0) { }

            int GetCount() const { return end - start; } // Works also if numbers over/under flow
            Bounds Exclude(Bounds other) const { return Bounds(other.end, end); }
        };

        Bounds GetBounds(const Stream& stream)
        {
            Bounds bounds;
            GetActualBounds(bounds.start, bounds.end, stream);
            return bounds;
        }

        uint32_t GetSpace(Bounds bounds) const
        {
            return m_capacity - bounds.GetCount();
        }

        std::vector< Segment<T> > GetSegs(Bounds bounds)
        {
            uint32_t start = bounds.start, end = bounds.end, size = bounds.GetCount();

            start = start % m_capacity;
            end = end % m_capacity;

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

        std::vector< Segment<T> > GetSegs(const Stream& stream)
        {
            return GetSegs(GetBounds(stream));
        }

        //
        // Debug methods
        //
        void GetOffsets(uint32_t& capacity, uint32_t& start, uint32_t& end, uint32_t& pending, uint32_t& size, const Stream& stream) const
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_start, m_start, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_end, m_end, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_pending, m_pending, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            
            stream.Sync();
            capacity = m_capacity;
            start = *m_host_start;
            end = *m_host_end;
            pending = *m_host_pending;
            size = end - start;
        }

        void PrintOffsets(const Stream& stream) const
        {
            uint32_t capacity, start, end, pending, size;
            GetOffsets(capacity, start, end, pending, size, stream);
            printf("\nPCQueue (Debug): start: %u, end: %u, pending: %u, size: %u (capacity: %u)", 
                start, end, pending, size, capacity);
        }
    };
}

#endif // __GROUTE_WORK_QUEUE_H
