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

#ifndef __GROUTE_COMMUNICATION_H
#define __GROUTE_COMMUNICATION_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>

#include <cuda_runtime.h>

#include <groute/context.h>


namespace groute {
    namespace router {

        /**
        * @brief Represents a pending segment of valid data
        */
        template <typename T>
        class PendingSegment : public Segment < T >
        {
        private:
            Event m_ready_event; // the event indicating data is valid  

        public:
            PendingSegment(T* segment_ptr, size_t total_size, size_t segment_size, size_t segment_offset, const Event& ready_event) :
                Segment<T>(segment_ptr, total_size, segment_size, segment_offset), m_ready_event(ready_event)
            {

            }

            PendingSegment() : Segment<T>(), m_ready_event()
            {

            }

            Event GetEvent() const { return m_ready_event; }

            void Wait(cudaStream_t stream) const
            {
                m_ready_event.Wait(stream);
            }

            void Sync() const
            {
                m_ready_event.Sync();
            }

            bool Query() const
            {
                return m_ready_event.Query();
            }
        };

        /**
        * @brief The sender should be used by data producers for distributing segments of data
        * @note The segment may get distributed to multiple consumers
        */
        template <typename T>
        struct ISender
        {
            virtual ~ISender() { }

            /// @brief Send a segment of data to any peer/s 
            virtual std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) = 0;

            /// @brief Report no more segments from this sender  
            virtual void Shutdown() = 0;


            /// @brief (for pipelined senders only)
            virtual Segment<T> GetSendBuffer() = 0;
            virtual void ReleaseSendBuffer(const Segment<T>& segment, const Event& ready_event) = 0;
        };

        /**
        * @brief A receiver for data segments
        */
        template <typename T>
        struct IReceiver
        {
            virtual ~IReceiver() { }

            /// @brief Receive a segment of data from peers into the provided buffer
            virtual std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) = 0;

            /// @brief Can this receiver still receive segments
            virtual bool Active() = 0;
        };

        /**
        * @brief A pipelined receiver which encapsulates memory buffer management
        */
        template <typename T>
        struct IPipelinedReceiver 
        {
            virtual ~IPipelinedReceiver() { }

            /// @brief Sync on all current memory operations in the pipeline
            virtual void Sync() const = 0;

            /// @brief Receive a segment of data from peers
            virtual std::shared_future< PendingSegment<T> > Receive() = 0;

            /// @brief Release the segment buffer for reuse
            virtual void ReleaseBuffer(const Segment<T>& segment, const Event& ready_event) = 0;
            
            /// @brief Receive a segment of data from peers into the provided buffer
            virtual std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) = 0;

            /// @brief Can this receiver still receive segments
            virtual bool Active() = 0;
        };

        /**
        * @brief Pipelined receiver implementation
        */
        template <typename T>
        class PipelinedReceiver : public IPipelinedReceiver < T >
        {
        private:
            IReceiver<T>* m_receiver;
            size_t m_chunk_size;
            std::vector <T*> m_dev_buffers;
            std::deque  < std::shared_future< PendingSegment<T> > > m_promised_segments;
            device_t m_dev;
            groute::Context& m_ctx;

        public:
            PipelinedReceiver(groute::Context& context, IReceiver<T>* receiver, device_t dev, size_t chunk_size, size_t num_buffers = 2) :
                m_receiver(receiver), m_chunk_size(chunk_size), m_dev_buffers(num_buffers), m_dev(dev), m_ctx(context)
            {
                context.SetDevice(dev);

                if (!m_receiver->Active()) // inactive receiver, no need for buffers
                {
                    m_chunk_size = 0;
                    m_dev_buffers.clear();
                }

                for (size_t i = 0; i < m_dev_buffers.size(); ++i)
                {
                    T *buffer;
                    if (dev != Device::Host)
                    {
                        GROUTE_CUDA_CHECK(cudaMalloc((void**)&buffer, m_chunk_size * sizeof(T)));
                    }
                    else
                    {
                        GROUTE_CUDA_CHECK(cudaMallocHost((void**)&buffer, m_chunk_size * sizeof(T)));
                    }
                    m_dev_buffers[i] = buffer;
                }

                for (size_t i = 0; i < m_dev_buffers.size(); ++i)
                {
                    m_promised_segments.push_back(m_receiver->Receive(Buffer<T>(m_dev_buffers[i], m_chunk_size), Event()));
                }
            }

            ~PipelinedReceiver()
            {
                m_ctx.SetDevice(m_dev);

                for (size_t i = 0; i < m_dev_buffers.size(); ++i)
                {
                    if (m_dev != Device::Host)
                    {
                        GROUTE_CUDA_CHECK(cudaFree(m_dev_buffers[i]));
                    }
                    else
                    {
                        GROUTE_CUDA_CHECK(cudaFreeHost(m_dev_buffers[i]));
                    }
                }
            }

            void Sync() const override
            {
                for (auto& pseg : m_promised_segments)
                {
                    pseg.get().Sync();
                }
            }

            std::shared_future< PendingSegment<T> > Receive() override
            {
                if (m_promised_segments.empty())
                {
                    if (!m_receiver->Active())
                    {
                        return groute::completed_future(PendingSegment<T>());
                    }
                    throw std::exception(); // m_promised_segments is empty (usage: Start -> Receive -> Release)
                }

                auto pseg = m_promised_segments.front();
                m_promised_segments.pop_front();
                return pseg;
            }

            void ReleaseBuffer(const Segment<T>& segment, const Event& ready_event) override
            {
                T* buffer = segment.GetSegmentPtr();
#ifndef NDEBUG
                if (std::find(m_dev_buffers.begin(), m_dev_buffers.end(), buffer) == m_dev_buffers.end())
                    throw std::exception(); // unrecognized buffer
#endif
                m_promised_segments.push_back(m_receiver->Receive(Buffer<T>(buffer, m_chunk_size), ready_event));
            }

            std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) override
            {
                return m_receiver->Receive(dst_buffer, ready_event);
            }

            bool Active() override
            {
                return m_receiver->Active();
            }
        };
    }
}

#endif // __GROUTE_COMMUNICATION_H
