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
#ifndef __UTILS_CUDA_UTILS_H
#define __UTILS_CUDA_UTILS_H

#include <future>
#include <queue>

#include <groute/internal/cuda_utils.h>
#include <groute/common.h>

namespace utils {

    template<typename T>
    class BlockingQueue
    {
        std::queue<T> m_queue;
        std::mutex m_mutex;
        std::condition_variable m_cv;

    public:
        void Enqueue(const T& item)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.push(item);
            m_cv.notify_one();
        }

        T Dequeue()
        {
            T item;

            { // Lock block
                std::unique_lock<std::mutex> lock(m_mutex);

                if (m_queue.empty()) {
                    // Waiting for work
                    m_cv.wait(lock, [this]() {
                        return !m_queue.empty(); });
                }

                item = m_queue.front();
                m_queue.pop();
            }

            return item;
        }

        bool TryDequeue(T& item)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_queue.empty()) return false;

            item = m_queue.front();
            m_queue.pop();
            return true;
        }

        bool Empty()
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            return (m_queue.empty());
        }
    };

    template<typename T>
    struct SharedArray
    {
        size_t buffer_size;
        std::vector<T> host_vec;
        T* dev_ptr;
        bool dev_ptr_owner;

        SharedArray(size_t buffer_size) : buffer_size(buffer_size), host_vec(buffer_size, 0), dev_ptr(nullptr), dev_ptr_owner(true)
        {
            GROUTE_CUDA_CHECK(cudaMalloc(&dev_ptr, buffer_size * sizeof(T)));
            GROUTE_CUDA_CHECK(cudaMemset(dev_ptr, 0, buffer_size * sizeof(T)));
        }

        SharedArray(T* dev_ptr, size_t buffer_size) : buffer_size(buffer_size), host_vec(buffer_size, 0), dev_ptr(dev_ptr), dev_ptr_owner(false)
        {
        }

        ~SharedArray()
        {
            if (dev_ptr_owner)
                GROUTE_CUDA_CHECK(cudaFree(dev_ptr));
        }

        T* host_ptr() { return &host_vec[0]; }

        void H2D()
        {
            if (buffer_size == 0) return;
            GROUTE_CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr(), buffer_size * sizeof(T), cudaMemcpyHostToDevice));
        }

        void D2H()
        {
            if (buffer_size == 0) return;
            GROUTE_CUDA_CHECK(cudaMemcpy(host_ptr(), dev_ptr, buffer_size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void H2DAsync(cudaStream_t stream)
        {
            if (buffer_size == 0) return;
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(dev_ptr, host_ptr(), buffer_size * sizeof(T), cudaMemcpyHostToDevice, stream));
        }

        void D2HAsync(cudaStream_t stream)
        {
            if (buffer_size == 0) return;
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(host_ptr(), dev_ptr, buffer_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    };

    template<typename T>
    struct SharedValue : public SharedArray < T >
    {
        SharedValue() : SharedArray<T>(1) { }
        SharedValue(T* dev_ptr) : SharedArray<T>(dev_ptr, 1) { }

        T& host_val() { return this->host_vec[0]; }

        void set_val_H2D(const T& item)
        {
            host_val() = item;
            this->H2D();
        }

        void set_val_H2DAsync(const T& item, cudaStream_t stream)
        {
            host_val() = item;
            this->H2DAsync(stream);
        }

        T get_val_D2H()
        {
            this->D2H();
            return host_val();
        }
    };
}

#endif // __UTILS_CUDA_UTILS_H
