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
        //
        // The context provides an abstraction layer between virtual 'endpoints' and the actual physical devices in the system.
        // In addition, it provides global services, such as memory-copy lanes for queing asynchronous copy operations, and event management.
        //

        std::map<Endpoint, device_t> m_endpoint_map; // Maps from endpoints to physical devices   

        int m_fragment_size; // The fragment size determining memory copy granularity. 
                             // In some cases, such fragmentation improves responsiveness of the underlying node, by interleaving memory traffic   
        
        std::set<int> m_physical_devs; // The physical devices currently in use by this context  
        std::map<int, std::unique_ptr<EventPool> > m_event_pools;
        std::map<int, std::unique_ptr<MemoryPool> > m_memory_pools;

        std::map< LaneIdentifier, std::shared_ptr<IMemcpyInvoker> > m_memcpy_invokers; // Memory-copy workers for each lane
        mutable std::mutex m_mutex;

        void InitPhysicalDevs()
        {
            for (auto& p : m_endpoint_map)
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
        Context() : m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            // build a simple one-to-one endpoint map  
            for (int physical_dev = 0; physical_dev < actual_ngpus; ++physical_dev)
            {
                m_endpoint_map[physical_dev] = physical_dev;
            }
            // host
            m_endpoint_map[Endpoint::HostEndpoint(0)] = Device::Host;

            InitPhysicalDevs();
            CreateEventPools();
            InitMemoryPools();
        }

        Context(int ngpus) : m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            for (int i = 0; i < ngpus; ++i)
            {
                m_endpoint_map[i] = i % actual_ngpus; // The real CUDA GPU index for all virtual GPUs
            }
            // host
            m_endpoint_map[Endpoint::HostEndpoint(0)] = Device::Host;
            
            InitPhysicalDevs();
            CreateEventPools();
            InitMemoryPools();
        }

        Context(const std::map<Endpoint, int>& endpoint_map) :
            m_endpoint_map(endpoint_map), m_fragment_size(-1)
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            for (auto& p : m_endpoint_map)
            {
                if (p.first.IsHost() && p.second != Device::Host)
                {
                    printf(
                        "\n\nWarning: %d is claimed to be a host endpoint but is not mapped over Device::Host.\n\n",
                        (Endpoint::identity_type)p.first);
                    exit(1);
                }

                if (p.first.IsGPU() && p.second < 0 || p.second >= actual_ngpus)
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

        void RequireMemcpyLane(Endpoint src, Endpoint dst)
        {
            assert(m_endpoint_map.find(src) != m_endpoint_map.end());
            assert(m_endpoint_map.find(dst) != m_endpoint_map.end());

            LaneIdentifier lane_identifier = Lane(src, dst).GetIdentifier();

            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_memcpy_invokers.find(lane_identifier) == m_memcpy_invokers.end())
            {
                m_memcpy_invokers[lane_identifier] = (m_fragment_size < 0)
                    ? (std::shared_ptr<IMemcpyInvoker>) std::make_shared<MemcpyInvoker>(m_endpoint_map.at(lane_identifier.first)) // no fragmentation  
                    : (std::shared_ptr<IMemcpyInvoker>) std::make_shared<MemcpyWorker>(m_endpoint_map.at(lane_identifier.first)); // for fragmentation, a dedicated worker is required  
            }
        }

        void RequireMemcpyLanes(const RoutingTable& routing_table)
        {
            for (auto& entry : routing_table)
            {
                Endpoint src = entry.first;
                for (auto dst : entry.second)
                {
                    RequireMemcpyLane(src, dst);
                }
            }
        }

        std::shared_ptr<groute::MemcpyWork> QueueMemcpyWork(
            Endpoint src, void* src_buffer, Endpoint dst, void* dst_buffer, size_t count,
            const Event& src_ready_event, const Event& dst_ready_event,
            const MemcpyCallback& callback)
        {
            LaneIdentifier lane_identifier 
                = Lane(src, dst).GetIdentifier();

            int physical_src_dev = m_endpoint_map.at(src);
            int physical_dst_dev = m_endpoint_map.at(dst);

            // Resolve the physical device associated with the copy stream of this lane.
            // We need this in order to provide the correct event pool
            int physical_stream_dev 
                = m_endpoint_map.at(lane_identifier.first);

            auto copy = std::make_shared<groute::MemcpyWork>(*m_event_pools[physical_stream_dev], m_fragment_size);

            copy->physical_src_dev = physical_src_dev;
            copy->src_buffer = src_buffer;
            copy->copy_bytes = count;
            copy->src_ready_event = src_ready_event;

            copy->physical_dst_dev = physical_dst_dev;
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

        void CacheEvents(size_t per_endpoint)
        {
            for (int physical_dev : m_physical_devs)
            {
                int endpoints = 0; // Count the number of endpoints using the physical device  
                for (auto& p : m_endpoint_map)
                {
                    if (p.second == physical_dev) ++endpoints;
                }

                m_event_pools.at(physical_dev)->CacheEvents(per_endpoint * endpoints);
            }
        }

        const std::map<Endpoint, int>& GetEndpointMap() const { return m_endpoint_map; }

        int GetPhysicalDevice(Endpoint endpoint) const { return m_endpoint_map.at(endpoint); }

        void SetDevice(Endpoint endpoint) const
        {
            if (endpoint.IsHost()) return;

            int current_physical_dev, requested_physical_dev = m_endpoint_map.at(endpoint);
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));

            if (current_physical_dev == requested_physical_dev) return;
            GROUTE_CUDA_CHECK(cudaSetDevice(requested_physical_dev));
        }

        void SyncDevice(Endpoint endpoint) const
        {
            if (endpoint.IsHost()) return;

            SetDevice(endpoint);
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

        Stream CreateStream(Endpoint endpoint, StreamPriority priority = SP_Default) const
        {
            return Stream(m_endpoint_map.at(endpoint), priority);
        }

        Stream CreateStream(StreamPriority priority = SP_Default) const
        {
            return Stream(priority);
        }

        EventPool& GetEventPool(Endpoint endpoint) const
        {
            return *m_event_pools.at(m_endpoint_map.at(endpoint));
        }

        Event RecordEvent(Endpoint endpoint, cudaStream_t stream) const
        {
            return m_event_pools.at(m_endpoint_map.at(endpoint))->Record(stream);
        }

        Event RecordEvent(cudaStream_t stream) const
        {
            int current_physical_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));
            return m_event_pools.at(current_physical_dev)->Record(stream);
        }

        Event RecordEvent(Endpoint endpoint, const Stream& stream) const
        {
            return RecordEvent(endpoint, stream.cuda_stream);
        }

        Event RecordEvent(const Stream& stream) const
        {
            return RecordEvent(stream.cuda_stream);
        }

        // -----------------

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

        void* Alloc(Endpoint endpoint, size_t size)
        {
            return m_memory_pools.at(m_endpoint_map.at(endpoint))->Alloc(size);
        }

        void* Alloc(Endpoint endpoint, double hint, size_t& size, AllocationFlags flags = AF_None)
        {
            int physical_dev = m_endpoint_map.at(endpoint);

            int endpoints = 0; // Count the number of endpoints using the physical device  
            for (auto& p : m_endpoint_map)
            {
                if (p.second == physical_dev) ++endpoints;
            }

            return m_memory_pools.at(physical_dev)->Alloc(hint / endpoints, size, flags);
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

            int endpoints = 0; // Count the number of endpoints using the physical device  
            for (auto& p : m_endpoint_map)
            {
                if (p.second == current_physical_dev) ++endpoints;
            }
            return m_memory_pools.at(current_physical_dev)->Alloc(hint / endpoints, size, flags);
        }
        
        // -----------------

        void PrintStatus() const
        {
            printf("\nDevice map:");
            for (auto& p : m_endpoint_map)
            {
                printf("\n\tVirtual: %d,\tPhysical: %d", (Endpoint::identity_type)p.first, p.second);
            }

            printf("\nMemcpy lanes:");
            for (auto& p : m_memcpy_invokers)
            {
                printf("\n\tDevice (virtual): %d,\tLane: %s", (Endpoint::identity_type)p.first.first, p.first.second == In ? "In" : p.first.second == Out ? "Out" : "Intra" );
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
