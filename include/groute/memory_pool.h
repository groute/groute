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

#ifndef __GROUTE_MEMPOOL_H
#define __GROUTE_MEMPOOL_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>
#include <cassert>

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

namespace groute {

    enum AllocationFlags
    {
        AF_None = 0,
        AF_PO2 = 1 << 0, // Is allocation required to be in power-of-two size 
    };

    struct Memory
    {
        void* ptr;
        size_t size;

        Memory() : ptr(nullptr), size(0) { }
        Memory(void* ptr, size_t size) : ptr(ptr), size(size) { }
    };

    class MemoryPool
    {
    private:
        int m_physical_dev;
        void* m_mem;
        size_t m_size, m_offset;
        size_t m_reserved;
        float m_reserved_percent;

    public:
        // Lazy initialization, see below
        MemoryPool() : m_physical_dev(Device::Null), m_mem(nullptr), m_size(0), m_offset(0), m_reserved(0), 
            m_reserved_percent(-1.0f) { }

        // Lazy initialization, see below
        explicit MemoryPool(int physical_dev) : m_physical_dev(physical_dev), m_mem(nullptr), m_size(0), m_offset(0), 
            m_reserved(0), m_reserved_percent(-1.0f) { }

        ~MemoryPool()
        {
            if (m_size == 0) return;
            GROUTE_CUDA_CHECK(cudaFree(m_mem));
        }

        void ReserveMemory(size_t membytes)
        {
            if (m_size > 0)
            {
                printf("ERROR: Cannot reserve memory for device %d - memory already allocated\n", m_physical_dev);
                return;
            }
            if (m_reserved_percent >= 0)
            {
                printf("ERROR: Cannot reserve absolute memory for device %d - relative memory requested\n", m_physical_dev);
                return;
            }

            m_reserved = membytes;
        }
            
        void ReserveFreeMemoryPercentage(float percent = 0.9f) // Default: 90% of free memory
        {
            if (m_size > 0)
            {
                printf("ERROR: Cannot reserve memory for device %d - memory already allocated\n", m_physical_dev);
                return;
            }
            if (m_reserved > 0)
            {
                printf("ERROR: Cannot reserve relative memory for device %d - absolute memory requested\n", m_physical_dev);
                return;
            }

            m_reserved_percent = percent;
        }

    private:
        void VerifyDev() const
        {
            int actual_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&actual_dev));
            if (actual_dev != m_physical_dev)
            {
                printf("\n\nWarning: actual dev: %d, expected dev: %d, exiting.\n\n", actual_dev, m_physical_dev);
                exit(1);
            }
        }

        void VerifyInit()
        {
            if (m_size > 0) return;

            // Expected to be on the correct device context
            VerifyDev();

            size_t total, free;
            GROUTE_CUDA_DAPI_CHECK(cuMemGetInfo(&free, &total));

            size_t alloc_size;
            if (m_reserved_percent >= 0)
                alloc_size = (size_t)(free*m_reserved_percent);
            else if (m_reserved > 0)
                alloc_size = m_reserved;
            else // Size not specified, defaulting to 90% free memory
                alloc_size = (size_t)(free*0.9f);

            if (cudaMalloc(&m_mem, alloc_size) != cudaSuccess)
            {
                printf("Free memory allocation failed for device %d, retrying\n", m_physical_dev);

                // Retry once
                GROUTE_CUDA_DAPI_CHECK(cuMemGetInfo(&free, &total));
                
                if (m_reserved_percent >= 0)
                    alloc_size = (size_t)(free*m_reserved_percent);
                else if (m_reserved > 0)
                    alloc_size = m_reserved;
                else // Size not specified, defaulting to 90% free memory
                    alloc_size = (size_t)(free*0.9f);

                GROUTE_CUDA_CHECK(cudaMalloc(&m_mem, alloc_size));
            }

            //float percent = 100.0f * ((float)alloc_size / (float)free);
            //printf("Allocated memory on device %d - total: %llu, free: %llu, allocated: %llu (%d percent)\n", m_physical_dev, total, free, alloc_size, (int)percent);

            m_size = alloc_size;
            m_offset = 0;
        }

        static void Normalize(std::vector<double>& factors)
        {
            // Ignore negatives
            for (auto& f : factors) if (f < 0) f = 0;

            double sum = 0;
            for (auto f : factors) sum += f;
            if (sum == 0)
            {
                // All zeros, give an equal part to each
                for (auto& f : factors) f = 1.0 / factors.size();
                sum = 1.0;
            }

            // Normalize 
            for (auto& f : factors) f /= sum;
        }

    public:
        void* Alloc(size_t size, size_t align)
        {
            VerifyInit(); // Lazy init  

            align = std::max(sizeof(int64_t), align);

            if (size > m_size - m_offset)
            {
                printf(
                    "\n\nWarning: bad context allocation for device %d, exiting.\n\n",
                    m_physical_dev);
                exit(1);
            }

            size_t offset = m_offset;

            // Align also here, just in case
            size = (size / align) * align;

            m_offset += size;
            return (void*)((char*)m_mem + offset);
        }

        void* Alloc(double hint, size_t align, size_t& size, AllocationFlags flags)
        {
            VerifyInit(); // Lazy init  

            align = std::max(sizeof(int64_t), align);
            size = (size_t)(m_size*hint);

            if (flags & AF_PO2)
            {
                size_t p2s = next_power_2(size);
                size = p2s > size ? p2s >> 1 : p2s;
            }

            // Align 
            size = (size / align) * align;

            return Alloc(size, align);
        }

        std::vector<Memory> Alloc(double percent, size_t align, const std::vector<double>& po2_factors, const std::vector<double>& non_po2_factors)
        {
            VerifyInit(); // Lazy init
            
            align = std::max(sizeof(int64_t), align);

            size_t num_po2_allocs = po2_factors.size();
            size_t num_allocs = num_po2_allocs + non_po2_factors.size();

            // Group factors 
            std::vector<double> factors;
            for (const auto& f : po2_factors)     factors.push_back(f);
            for (const auto& f : non_po2_factors) factors.push_back(f);
            // And normalize
            Normalize(factors);

            // Calculate total memory for this Alloc call
            size_t size = (size_t)(m_size*percent);
            size = (size / align) * align; // Align 
            //And allocate
            char* mem = (char*)Alloc(size, align);
            size_t offset = 0;

            std::vector<Memory> allocs;
            allocs.reserve(num_allocs);

            for (size_t i = 0; i < num_po2_allocs; ++i)
            {
                size_t s = (size_t)(size*factors[i]);
                size_t p2s = next_power_2(s);
                s = p2s > s ? p2s >> 1 : p2s; // Round to prev power-of-two

                allocs.push_back(Memory(mem + offset, s));
                offset += s;
            }

            // Remove power-of-two factors
            factors.erase(factors.begin(), factors.begin() + num_po2_allocs);
            // And normalize again (this way we use whatever is left after power-of-two rounding's above)
            Normalize(factors);

            // Use the remaining size
            size_t remaining = size - offset;

            for (const auto& f : factors)
            {
                size_t s = (size_t)(remaining*f);
                s = (s / align) * align; // Align 
                allocs.push_back(Memory(mem + offset, s));
                offset += s;
            }

            //
            // Currently we only round down power-of-two allocations, 
            // this can be improved by iterating over possibilities and making a wiser allocation distribution
            //

            return allocs;
        }
    };

    
}

#endif // __GROUTE_MEMPOOL_H
