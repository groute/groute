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

#ifndef __GROUTE_CONTEXT_H
#define __GROUTE_CONTEXT_H

#include <initializer_list>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/common.h>
#include <groute/event_pool.h>
#include <groute/memcpy.h>
#include <groute/memory_pool.h>

namespace groute {

    /*
    * @brief The global groute context
    */
    class Context
    {
    private:
        std::map<device_t, int> m_dev_map;

        int m_fragment_size; // TODO: should this be a global configuration, or a router decision?   
        
        std::set<int> m_physical_devs; // the physical devs currently in use by this context  
        std::map<int, std::unique_ptr<EventPool> > m_event_pools;

        std::map< LaneIdentifier, std::shared_ptr<IMemcpyInvoker> > m_memcpy_invokers;
        mutable std::mutex m_mutex;

        std::map<int, std::unique_ptr<MemoryPool> > m_memory_pools;

    public: 
        void RequireMemcpyLane(int src_dev, int dst_dev)
        {
            assert(m_dev_map.find(src_dev) != m_dev_map.end());
            assert(m_dev_map.find(dst_dev) != m_dev_map.end());

            LaneIdentifier lane_identifier 
                = Lane(src_dev, dst_dev).GetIdentifier();

            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_memcpy_invokers.find(lane_identifier) == m_memcpy_invokers.end())
            {
                m_memcpy_invokers[lane_identifier] = (m_fragment_size < 0)
                    ? (std::shared_ptr<IMemcpyInvoker>) std::make_shared<MemcpyInvoker>(m_dev_map.at(lane_identifier.first)) // no fragmentation  
                    : (std::shared_ptr<IMemcpyInvoker>) std::make_shared<MemcpyWorker>(m_dev_map.at(lane_identifier.first)); // for fragmentation a dedicated worker is required  
            }
        }

        void RequireMemcpyLanes(const RoutingTable& required_routes)
        {
            for (auto& p : required_routes)
            {
                device_t src_dev = p.first;
                for (auto dst_dev : p.second)
                {
                    RequireMemcpyLane(src_dev, dst_dev);
                }
            }
        }

    private:
        void InitPhysicalDevs()
        {
            for (auto& p : m_dev_map)
            {
                if (p.second == Device::Host) continue;
                m_physical_devs.insert(p.second);
            }

            for (int physical_dev_i : m_physical_devs)
            {
                GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev_i));
                for (int physical_dev_j : m_physical_devs)
                    if (physical_dev_i != physical_dev_j)
                        cudaDeviceEnablePeerAccess(physical_dev_j, 0);
            }
        }

        void CreateEventPools()
        {            
            for (int physical_dev : m_physical_devs)
            {
                auto pool = make_unique<EventPool>(physical_dev);
                m_event_pools[physical_dev] = std::move(pool);
            }
        }

    public:
        Context() : m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            // build a simple one-to-one dev map  
            for (int physical_dev = 0; physical_dev < actual_ngpus; ++physical_dev)
            {
                m_dev_map[physical_dev] = physical_dev;
            }
            // host
            m_dev_map[Device::Host] = Device::Host;

            InitPhysicalDevs();
            CreateEventPools();
            InitMemoryPools();
        }

        Context(int ngpus) : m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            for (device_t i = 0; i < ngpus; ++i)
            {
                m_dev_map[i] = i % actual_ngpus; // The real CUDA GPU index for all virtual GPUs
            }
            // host
            m_dev_map[Device::Host] = Device::Host;
            
            InitPhysicalDevs();
            CreateEventPools();
            InitMemoryPools();
        }

        Context(const std::map<device_t, int>& dev_map) :
            m_dev_map(dev_map), m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            for (auto& p : m_dev_map)
            {
                if (Device::IsHost(p.second)) continue;
                if (p.second < 0 || p.second >= actual_ngpus)
                {
                    printf(
                        "\n\nWarning: %d is claimed to be a physical device but is not (actual_ngpus = %d), exiting.\n\n",
                        p.second, actual_ngpus);
                    exit(1);
                }
            }
            
            InitPhysicalDevs();
            CreateEventPools();
            InitMemoryPools();
        }

        std::shared_ptr<groute::MemcpyWork> QueueMemcpyWork(
            int src_dev, void* src_buffer, int dst_dev, void* dst_buffer, size_t count,
            const Event& src_ready_event, const Event& dst_ready_event,
            const MemcpyCallback& callback)
        {
            LaneIdentifier lane_identifier 
                = Lane(src_dev, dst_dev).GetIdentifier();

            int src_dev_id = m_dev_map.at(src_dev);
            int dst_dev_id = m_dev_map.at(dst_dev);

            // resolve the correct device associated with the copy stream of this lane
            // we need this in order to provide the correct event pool
            int stream_dev_id 
                = m_dev_map.at(lane_identifier.first);

            auto copy = std::make_shared<groute::MemcpyWork>(*m_event_pools[stream_dev_id], m_fragment_size);

            copy->src_dev_id = src_dev_id;
            copy->src_buffer = src_buffer;
            copy->copy_bytes = count;
            copy->src_ready_event = src_ready_event;

            copy->dst_dev_id = dst_dev_id;
            copy->dst_buffer = dst_buffer;
            copy->dst_size = count;
            copy->dst_ready_event = dst_ready_event;

            copy->completion_callback = callback;

            m_memcpy_invokers.at(lane_identifier)->InvokeCopyAsync(copy);

            return copy;
        }

        void DisableFragmentation()
        {
            assert(m_memcpy_invokers.empty());
            m_fragment_size = -1;
        } 

        void EnableFragmentation(int fragment_size)
        {
            assert(m_memcpy_invokers.empty());
            m_fragment_size = fragment_size;
        }

        void CacheEvents(size_t num_evs)
        {
            num_evs = num_evs * round_up(m_dev_map.size() - 1 /*- host*/, m_physical_devs.size()); // approximation for each real GPU  

            for (int physical_dev : m_physical_devs)
            {
                m_event_pools.at(physical_dev)->CacheEvents(num_evs);
            }
        }

        const std::map<int, int>& GetDevMap() const { return m_dev_map; }

        int GetDevId(int dev) const { return m_dev_map.at(dev); }

        void SetDevice(int dev) const
        {
            if (dev == Device::Host) return;

            int current_dev_id, requested_dev_id = m_dev_map.at(dev);
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_dev_id));

            if (current_dev_id == requested_dev_id) return;
            GROUTE_CUDA_CHECK(cudaSetDevice(requested_dev_id));
        }

        void SyncDevice(int dev) const
        {
            if (dev == Device::Host) return;

            SetDevice(dev);
            GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        }

        void SyncAllDevices() const
        {
            for (int physical_dev_i : m_physical_devs)
            {
                GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev_i));
                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        Stream CreateStream(int dev, StreamPriority priority = SP_Default) const
        {
            return Stream(m_dev_map.at(dev), priority);
        }

        Stream CreateStream(StreamPriority priority = SP_Default) const
        {
            return Stream(priority);
        }

        EventPool& GetEventPool(int dev) const
        {
            return *m_event_pools.at(m_dev_map.at(dev));
        }

        Event RecordEvent(int dev, cudaStream_t stream) const
        {
            return m_event_pools.at(m_dev_map.at(dev))->Record(stream);
        }

        Event RecordEvent(cudaStream_t stream) const
        {
            int current_physical_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));
            return m_event_pools.at(current_physical_dev)->Record(stream);
        }

        // -----------------
        
    private:
        
        

        void InitMemoryPools()
        {        
            GROUTE_CUDA_DAPI_CHECK(cuInit(0));

            for (int physical_dev : m_physical_devs)
            {
                auto pool = make_unique<MemoryPool>(physical_dev);
                m_memory_pools[physical_dev] = std::move(pool);
            }
        }
        
    public:
        void ReserveMemory(size_t membytes)
        {
            for (auto& pair : m_memory_pools)
            {
                pair.second->ReserveMemory(membytes);
            }
        }

        void ReserveFreeMemoryPercentage(float percent = 0.9f) // Default: 90% of free memory
        {            
            for (auto& pair : m_memory_pools)
            {
                pair.second->ReserveFreeMemoryPercentage(percent);
            }
        }

        void* Alloc(device_t dev, size_t size)
        {
            return m_memory_pools.at(m_dev_map.at(dev))->Alloc(size);
        }

        void* Alloc(device_t dev, double hint, size_t& size, AllocationFlags flags = AF_None)
        {
            double vpp = (double)(m_dev_map.size() - 1) / m_physical_devs.size(); // virtual per physical
            return m_memory_pools.at(m_dev_map.at(dev))->Alloc(hint / vpp, size, flags);
        }

        void* Alloc(size_t size)
        {
            int current_physical_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));
            return m_memory_pools.at(current_physical_dev)->Alloc(size);
        }

        void* Alloc(double hint, size_t& size, AllocationFlags flags = AF_None)
        {
            int current_physical_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));
            
            double vpp = (double)(m_dev_map.size() - 1) / m_physical_devs.size(); // virtual per physical
            return m_memory_pools.at(current_physical_dev)->Alloc(hint / vpp, size, flags);
        }
        
        // -----------------

        void PrintStatus() const
        {
            printf("\nDevice map:");
            for (auto& p : m_dev_map)
            {
                printf("\n\tVirtual: %d,\tPhysical: %d", p.first, p.second);
            }

            printf("\nMemcpy lanes:");
            for (auto& p : m_memcpy_invokers)
            {
                printf("\n\tDevice (virtual): %d,\tLane: %s", p.first.first, p.first.second == In ? "In" : p.first.second == Out ? "Out" : "Internal" );
            }
            printf("\nFragmentation: %d", m_fragment_size);

            printf("\nEvent pools:");
            for (auto& p : m_event_pools)
            {
                printf("\n\tDevice (physical): %d,\tCached events: %d", p.first, p.second->GetCachedEventsNum());
            }
        }
    };
}

#endif // __GROUTE_CONTEXT_H
