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

#ifndef __GROUTE_DEVICE_SIGNAL_H
#define __GROUTE_DEVICE_SIGNAL_H

#include <groute/device/queue.cu.h>
#include <thread>

namespace groute {
    namespace dev {

        class Signal
        {
            volatile int *m_signal_ptr;
        public:

            __device__ __host__ Signal(volatile int *signal_ptr) : m_signal_ptr(signal_ptr) { }

            __device__ __forceinline__ void increase(int value)
            {
                __threadfence_system();
                *m_signal_ptr = *m_signal_ptr + value;
            }
            
            __device__ __forceinline__ void set(int value)
            {
                __threadfence_system();
                *m_signal_ptr = value;
            }

            static __device__ __forceinline__ void Increase(volatile int *signal_ptr, int value)
            {
                __threadfence_system();
                *signal_ptr = *signal_ptr + value;
            }

            static __device__ __forceinline__ void Increment(volatile int *signal_ptr)
            {
                __threadfence_system();
                *signal_ptr = *signal_ptr + 1;
            }

            static __device__ __forceinline__ void Set(volatile int *signal_ptr, int value)
            {
                __threadfence_system();
                *signal_ptr = value;
            }
        };
    }
    
    /*
    * @brief Host lifetime manager for dev::Signal
    */
    class Signal
    {
        volatile int * m_signal_host, * m_signal_dev;

    public:
        typedef dev::Signal DeviceObjectType;

        Signal() 
        {
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_signal_host, sizeof(int)));
            GROUTE_CUDA_CHECK(cudaHostGetDevicePointer(&m_signal_dev, (int*)m_signal_host, 0));
            *m_signal_host = 0;
        }

        ~Signal()
        {
            GROUTE_CUDA_CHECK(cudaFreeHost((void*)m_signal_host));
        }

        Signal(const Signal& other) = delete;
        Signal(Signal&& other) = delete;

        DeviceObjectType DeviceObject() const
        {
            return dev::Signal(m_signal_dev);
        }

        volatile int * GetDevPtr() const { return m_signal_dev; }

        int Peek() const { return *m_signal_host; }

        int WaitForSignal(int prev_signal, Stream& stream)
        {
            int signal = *m_signal_host;

            while (signal == prev_signal)
            {
                std::this_thread::yield();
                if (stream.Query()) // Means kernel is done
                {
                    signal = *m_signal_host; // Make sure to read any later signal as well
                    break;
                }

                signal = *m_signal_host;
            }
            return signal;
        }
    };
}

#endif // __GROUTE_DEVICE_SIGNAL_H
