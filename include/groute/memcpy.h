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

#ifndef __GROUTE_MEMCPY_H
#define __GROUTE_MEMCPY_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <future>
#include <functional>
#include <cassert>

#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/common.h>
#include <groute/event_pool.h>

namespace groute {

    typedef std::function<void(size_t, const Event&)> MemcpyCallback;

    class MemcpyWork : public groute::internal::IWork
    {
    public:
        int src_dev_id;
        int dst_dev_id;

        size_t copy_bytes;
        size_t dst_size;

        const int fragment_size;

        void* src_buffer;
        void* dst_buffer;

        Event src_ready_event;
        Event dst_ready_event;

        cudaStream_t copy_stream;
        cudaEvent_t sync_event; 

        MemcpyCallback completion_callback;
        
    private:
        EventPool& m_event_pool;

        void CheckParams() const
        {
            // Verifying since parameters are expected to be provided by multiple resources  

            assert(!Device::IsNull(src_dev_id));
            assert(!Device::IsNull(dst_dev_id));

            assert(src_buffer != nullptr);
            assert(dst_buffer != nullptr);

            assert(copy_stream != nullptr);
            assert(sync_event != nullptr);

            assert(copy_bytes <= dst_size);
        }

        void CopyAsync(void *dst_buffer, const void *src_buffer, size_t count) const
        {
            if (!Device::IsHost(src_dev_id) && !Device::IsHost(dst_dev_id)) // dev to dev
            {
                GROUTE_CUDA_CHECK(
                    cudaMemcpyPeerAsync(
                    dst_buffer, dst_dev_id, src_buffer, src_dev_id, count, copy_stream));
            }

            else if (Device::IsHost(src_dev_id)) // host to dev
            {
                GROUTE_CUDA_CHECK(
                    cudaMemcpyAsync(
                    dst_buffer, src_buffer, count, cudaMemcpyHostToDevice, copy_stream));
            }

            else if (Device::IsHost(dst_dev_id)) // dev to host
            {
                GROUTE_CUDA_CHECK(
                    cudaMemcpyAsync(
                    dst_buffer, src_buffer, count, cudaMemcpyDeviceToHost, copy_stream));
            }

            else // host to host
            {
                assert(false); // TODO: std::memcpy(dst_buffer, src_buffer, count);
            }
        }
        
        void Complete(size_t bytes, const Event& ev) const
        {
            if (completion_callback) completion_callback(bytes, ev);
        }

    public:
        MemcpyWork(EventPool& event_pool, int fragment_size = -1) :
            m_event_pool(event_pool),
            src_dev_id(Device::Null), dst_dev_id(Device::Null),
            fragment_size(fragment_size), copy_bytes(0), dst_size(0),
            src_buffer(nullptr), dst_buffer(nullptr),
            copy_stream(nullptr), sync_event(nullptr), completion_callback(nullptr)
        {
#ifndef NDEBUG
            if (fragment_size < -1 || fragment_size == 0)
                throw std::invalid_argument("invalid value for fragment_size");
#endif
        }

        void operator()(groute::internal::Barrier *barrier) override
        {
#ifndef NDEBUG
            CheckParams();
#endif

            src_ready_event.Wait(copy_stream);
            dst_ready_event.Wait(copy_stream);

            if (fragment_size < 0) // No fragmentation  
            {
                CopyAsync(dst_buffer, src_buffer, copy_bytes);
            }

            else
            {
                // Fragmented Copy 

                int fragment = fragment_size < 0 ? copy_bytes : fragment_size;
                int pos = 0;
                while (pos < copy_bytes)
                {
                    void *receive = ((void*)((char*)dst_buffer + pos));
                    void *send = ((void*)((char*)src_buffer + pos));

                    CopyAsync(receive, send, (size_t)((pos + fragment) > copy_bytes ? (copy_bytes - pos) : fragment));
                    
                    pos += fragment;

                    if (pos >= copy_bytes) break; // Avoid syncing on last segment  

                    // We must sync the host thread in order to achive real fragmentation  
                    //
                    GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, copy_stream));
                    GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
                }
            }

            Complete(copy_bytes, m_event_pool.Record(copy_stream));
        }
    };

    struct IMemcpyInvoker
    {
        virtual ~IMemcpyInvoker() { }
        virtual void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) = 0;
    };

    class MemcpyInvoker : public IMemcpyInvoker
    {
    protected:
        const int m_dev_id; // the real dev id
        cudaStream_t m_copy_stream;
        cudaEvent_t m_sync_event;

    public:
        MemcpyInvoker(int dev_id) : m_dev_id(dev_id)
        {
            GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));
            GROUTE_CUDA_CHECK(cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking));
            GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&m_sync_event, cudaEventDisableTiming));
        }

        virtual ~MemcpyInvoker()
        {
            GROUTE_CUDA_CHECK(cudaStreamDestroy(m_copy_stream));
            GROUTE_CUDA_CHECK(cudaEventDestroy(m_sync_event));
        }

        void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) override
        {
            assert(memcpy_work->fragment_size == -1); // this invoker does not support fragmentation  
            
            int current_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_dev));

            if(current_dev != m_dev_id)
                GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));

            memcpy_work->copy_stream = m_copy_stream;
            memcpy_work->sync_event = m_sync_event;

            (*memcpy_work)(nullptr); // invoke

            if(current_dev != m_dev_id) // set back to the correct device
                GROUTE_CUDA_CHECK(cudaSetDevice(current_dev));
        }
    };

    class MemcpyWorker : public groute::internal::Worker < MemcpyWork >, public IMemcpyInvoker
    {
    private:
        const int m_dev_id; // the real dev id
        cudaStream_t m_copy_stream;
        cudaEvent_t m_sync_event;

    protected:
        /// Called by the worker thread on start
        void OnStart() override
        {
            GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));
            GROUTE_CUDA_CHECK(cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking));
            GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&m_sync_event, cudaEventDisableTiming));
        }

        void OnBeginWork(std::shared_ptr<MemcpyWork> work) override
        {
            work->copy_stream = m_copy_stream;
            work->sync_event = m_sync_event;
        }

    public:
        explicit MemcpyWorker(int dev_id)
            : groute::internal::Worker<MemcpyWork>(nullptr), m_dev_id(dev_id)
        {
            this->Run();
        }

        virtual ~MemcpyWorker()
        {
            GROUTE_CUDA_CHECK(cudaStreamDestroy(m_copy_stream));
            GROUTE_CUDA_CHECK(cudaEventDestroy(m_sync_event));
        }

        void InvokeCopyAsync(std::shared_ptr<MemcpyWork> memcpy_work) override
        {
            this->Enqueue(memcpy_work);
        }
    };
}

#endif // __GROUTE_MEMCPY_H
