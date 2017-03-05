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

#ifndef __GROUTE_DEVICE_COUNTER_H
#define __GROUTE_DEVICE_COUNTER_H

#include <groute/device/queue.cuh>

namespace groute {
    namespace dev {

        class Counter
        {
        public:
            uint32_t* m_counter;

            __host__ __device__ Counter(uint32_t* counter) : 
                m_counter(counter) { }

            __device__ __forceinline__ void add(uint32_t count)
            {
                atomicAdd(m_counter, count); 
            }

            __device__ __forceinline__ void add_one_warp()
            {
                int lanemask = __ballot(1);
                int leader = __ffs(lanemask) - 1;
                    
                if (cub::LaneId() == leader) {
                    int amount = __popc(lanemask);
                    atomicAdd(m_counter, amount);
                }
            }

            __device__ __forceinline__ void reset()
            {
                *m_counter = 0;
            }
                        
            __device__ __forceinline__ uint32_t get_count() const
            {
                return *m_counter;
            }
        };
    }

    /*
    * @brief Host lifetime manager for dev::Counter
    */
    class Counter  
    {
        enum { WS = 32 };

        //
        // device buffer / counters 
        //
        uint32_t *m_counters;
        uint32_t *m_host_counter;
        int32_t m_current_slot;
    
    public:
        Counter() : m_counters(nullptr), m_current_slot(-1)
        {
            Alloc();
        }

        Counter(const Counter& other) = delete;
        Counter(Counter&& other) = delete;

        ~Counter()
        {
            Free();
        }
        
        typedef dev::Counter DeviceObjectType;
    
    private:
        void Alloc()
        {
            GROUTE_CUDA_CHECK(cudaMalloc(&m_counters, WS * sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_counter, sizeof(uint32_t)));
        }
    
        void Free()
        {
            GROUTE_CUDA_CHECK(cudaFree(m_counters));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_counter));
        }
    
    public:
        DeviceObjectType DeviceObject() const
        {
            assert(m_current_slot >= 0 && m_current_slot < WS);
            return dev::Counter(m_counters + m_current_slot);
        }

        template<typename T>
        typename Worklist<T>::DeviceObjectType ToDeviceWorklist(T* data_ptr, uint32_t capacity) const
        {
            assert(m_current_slot >= 0 && m_current_slot < WS);
            return dev::Worklist<T>(data_ptr, m_counters + m_current_slot, capacity);
        }

        void ResetAsync(cudaStream_t stream)
        {
            m_current_slot = (m_current_slot + 1) % WS;
            if (m_current_slot == 0)
            {
                ResetCounters <<< 1, WS, 0, stream >>>(m_counters, WS);
            }
        }
        
        uint32_t GetCount(const Stream& stream) const
        {
            assert(m_current_slot >= 0 && m_current_slot < WS);

            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_counter, m_counters + m_current_slot, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream));
            stream.Sync();
    
            return *m_host_counter;
        }
    };
}

#endif // __GROUTE_DEVICE_COUNTER_H
