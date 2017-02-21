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

    /**
    * @brief Represents a pending buffer
    */
    template <typename T>
    class PendingBuffer : public Buffer < T >
    {
    private:
        Event m_ready_event; // The event indicating buffer is ready for use  

    public:
        PendingBuffer(T* ptr, size_t size, const Event& ready_event) :
            Buffer<T>(ptr, size), m_ready_event(ready_event)
        {

        }
        
        PendingBuffer() : Buffer<T>(), m_ready_event()
        {

        }

        Event GetEvent() const { return m_ready_event; }

        void Wait(cudaStream_t stream) const
        {
            m_ready_event.Wait(stream);
        }

        void Wait(const Stream& stream) const
        {
            m_ready_event.Wait(stream.cuda_stream);
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
    * @brief Represents a pending segment of valid data
    */
    template <typename T>
    class PendingSegment : public Segment < T >
    {
    private:
        Event m_ready_event; // The event indicating data is valid  

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

        void Wait(const Stream& stream) const
        {
            m_ready_event.Wait(stream.cuda_stream);
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
    };

    /**
    * @brief A sender with pipeline support for send buffers
    */
    template <typename T>
    struct IPipelinedSender
    {
        virtual ~IPipelinedSender() { }

        /// @brief Send a segment of data to any peer/s 
        virtual std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) = 0;

        /// @brief Report no more segments from this sender  
        virtual void Shutdown() = 0;

        /// @brief Wait for an available send buffer
        virtual PendingBuffer<T> GetSendBuffer() = 0;

        /// @brief Sends data and pipelines the buffer
        /// @note Sent buffers are queued for reuse through GetSendBuffer
        virtual void PipelinedSend(const Segment<T>& segment, const Event& ready_event) = 0;
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
        
        /// @brief Receive a segment of data from peers into the provided buffer
        virtual std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) = 0;

        /// @brief Can this receiver still receive segments
        virtual bool Active() = 0;

        /// @brief Receive a segment of data from peers
        virtual std::shared_future< PendingSegment<T> > PipelinedReceive() = 0;

        /// @brief Release the segment buffer for reuse
        virtual void ReleaseReceiveBuffer(T* buffer, const Event& ready_event) = 0;

        /// @brief Sync on all current memory operations in the pipeline
        virtual void PipelineSync() const = 0;
    };

    template <typename T>
    class Pipeline
    {
    protected:
        size_t m_chunk_size;
        std::vector <T*> m_endpoint_buffers;
        Endpoint m_endpoint;
        Context& m_ctx;
    
        Pipeline(Context& context, Endpoint endpoint, size_t chunk_size, size_t num_buffers) :
            m_chunk_size(chunk_size), m_endpoint_buffers(num_buffers), m_endpoint(endpoint), m_ctx(context)
        {
            context.SetDevice(endpoint);
    
            for (size_t i = 0; i < m_endpoint_buffers.size(); ++i)
            {
                T *buffer;
                if (!endpoint.IsHost())
                {
                    GROUTE_CUDA_CHECK(cudaMalloc((void**)&buffer, m_chunk_size * sizeof(T)));
                }
                else
                {
                    GROUTE_CUDA_CHECK(cudaMallocHost((void**)&buffer, m_chunk_size * sizeof(T)));
                }
                m_endpoint_buffers[i] = buffer;
            }
        }
    
        virtual ~Pipeline()
        {
            m_ctx.SetDevice(m_endpoint);
    
            for (size_t i = 0; i < m_endpoint_buffers.size(); ++i)
            {
                if (!m_endpoint.IsHost())
                {
                    GROUTE_CUDA_CHECK(cudaFree(m_endpoint_buffers[i]));
                }
                else
                {
                    GROUTE_CUDA_CHECK(cudaFreeHost(m_endpoint_buffers[i]));
                }
            }
        }
    };
    
    /**
    * @brief Pipelined receiver implementation
    */
    template <typename T>
    class PipelinedReceiver : public IPipelinedReceiver <T>, protected Pipeline<T>
    {
    private:
        IReceiver<T>* m_receiver;
        std::deque  < std::shared_future< PendingSegment<T> > > m_promised_segments;
    
    public:
        PipelinedReceiver(Context& context, IReceiver<T>* receiver, Endpoint endpoint, size_t chunk_size, size_t num_buffers) :
            Pipeline<T>(context, endpoint, chunk_size, num_buffers), m_receiver(receiver)
        {
            for (size_t i = 0; i < this->m_endpoint_buffers.size(); ++i)
            {
                m_promised_segments.push_back(m_receiver->Receive(Buffer<T>(this->m_endpoint_buffers[i], this->m_chunk_size), Event()));
            }
        }
    
        void PipelineSync() const override
        {
            for (auto& pseg : m_promised_segments)
            {
                pseg.get().Sync();
            }
        }
    
        std::shared_future< PendingSegment<T> > PipelinedReceive() override
        {
            if (m_promised_segments.empty())
            {
                printf("\n\nWarning: No pipeline buffers available (usage: PipelinedReceive -> ReleaseReceiveBuffer)\n\n");
                throw std::exception("No pipeline buffers"); 
            }
    
            auto pseg = m_promised_segments.front();
            m_promised_segments.pop_front();
            return pseg;
        }
    
        void ReleaseReceiveBuffer(T* buffer, const Event& ready_event) override
        {
#ifndef NDEBUG
            if (std::find(this->m_endpoint_buffers.begin(), this->m_endpoint_buffers.end(), buffer) == this->m_endpoint_buffers.end())
                throw std::exception("Unrecognized buffer in pipelined receiver"); 
#endif
            m_promised_segments.push_back(m_receiver->Receive(Buffer<T>(buffer, this->m_chunk_size), ready_event));
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

    /**
    * @brief Pipelined sender implementation
    */
    template <typename T>
    class PipelinedSender : public IPipelinedSender<T>, protected Pipeline<T>
    {
    private:
        ISender<T>* m_sender;

        struct BufferFuture
        {
            std::shared_future<Event> future;
            T* ptr;

            BufferFuture(std::shared_future< Event > future, T* ptr) : future(future), ptr(ptr) { }
            BufferFuture() : future(), ptr(nullptr) { }
        };
        std::deque<BufferFuture> m_promised_buffers;  

    public:
        PipelinedSender(Context& context, ISender<T>* sender, Endpoint endpoint, size_t chunk_size, size_t num_buffers) :
            Pipeline<T>(context, endpoint, chunk_size, num_buffers), m_sender(sender)
        {
            for (size_t i = 0; i < this->m_endpoint_buffers.size(); ++i)
            {
                m_promised_buffers.push_back(
                    BufferFuture(groute::completed_future(Event()), this->m_endpoint_buffers[i]));
            }
        }

        PendingBuffer<T> GetSendBuffer() override
        {
            if (m_promised_buffers.empty())
            {
                printf("\n\nWarning: No pipeline buffers available (usage: GetSendBuffer -> PipelinedSend)\n\n");
                throw std::exception("No pipeline buffers"); 
            }

            auto buff = m_promised_buffers.front();
            m_promised_buffers.pop_front();

            return PendingBuffer<T>(buff.ptr, this->m_chunk_size, buff.future.get() /*block on event future*/);
            // Note: assumes send buffers are handled by FIFO order in router, which is true unless segments have metadata attached
        }

        void PipelinedSend(const Segment<T>& segment, const Event& ready_event) override
        {
            T* ptr = segment.GetSegmentPtr();
#ifndef NDEBUG
            if (std::find(this->m_endpoint_buffers.begin(), this->m_endpoint_buffers.end(), ptr) == this->m_endpoint_buffers.end())
                throw std::exception("Unrecognized buffer in pipelined receiver"); 
#endif
            auto event_fut = m_sender->Send(segment, ready_event);
            m_promised_buffers.push_back(BufferFuture(event_fut, ptr));
        }

        std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) override
        {
            return m_sender->Send(segment, ready_event);
        }

        void Shutdown() override
        {
            m_sender->Shutdown();
        }
    };
}

#endif // __GROUTE_COMMUNICATION_H
