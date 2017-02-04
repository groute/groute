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

#ifndef __GROUTE_EVPOOL_H
#define __GROUTE_EVPOOL_H

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

#include <groute/common.h>

namespace groute {

    struct IEvent
    {
        virtual ~IEvent() { }

        virtual void Wait(cudaStream_t stream) const = 0;
        virtual void Sync() const = 0;
        virtual bool Query() const = 0;
    };

    class EventHolder : public IEvent
    {
    private:
        const cudaEvent_t cuda_event;
        std::function<void(cudaEvent_t)> m_releaser;

    public:
        EventHolder(cudaEvent_t cuda_event, const std::function<void(cudaEvent_t)>& releaser) : 
            cuda_event(cuda_event), m_releaser(releaser)
        {
        }

        ~EventHolder()
        {
            m_releaser(cuda_event);
        }

        void Wait(cudaStream_t stream) const override 
        {
            GROUTE_CUDA_CHECK(cudaStreamWaitEvent(stream, cuda_event, 0));
        }

        void Sync() const override 
        {
            GROUTE_CUDA_CHECK(cudaEventSynchronize(cuda_event));
        }

        bool Query() const override 
        {
            return cudaEventQuery(cuda_event) == cudaSuccess;
        }
    };

    class Event
    {
    private:
        std::shared_ptr<IEvent> m_internal_event;

    public:
        Event(cudaEvent_t cuda_event, const std::function<void(cudaEvent_t)>& releaser)
        {
            assert(cuda_event != nullptr);
            assert(releaser != nullptr);

            m_internal_event = std::make_shared<EventHolder>(cuda_event, releaser);
        }

        Event() : m_internal_event(nullptr) // dummy event
        {
        }
        
        Event(const Event& other) : m_internal_event(other.m_internal_event)
        {
            
        }

        Event(Event&& other) : m_internal_event(std::move(other.m_internal_event))
        {
        }

        Event& operator=(Event&& other) 
        {
            this->m_internal_event = std::move(other.m_internal_event);
            return *this;
        }

        Event& operator=(const Event& other)
        {
            m_internal_event = other.m_internal_event;
            return *this;
        }

        Event(std::shared_ptr<IEvent> internal_event) : m_internal_event(internal_event)
        {
        }

        static Event Record(cudaStream_t stream)
        {
            cudaEvent_t cuda_event;
            GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming));
            GROUTE_CUDA_CHECK(cudaEventRecord(cuda_event, stream));

            return Event(
                cuda_event, 
                [](cudaEvent_t ev) // releaser, called on destruction event
                {
                    GROUTE_CUDA_CHECK(cudaEventDestroy(ev));
                }
            );
        }

        void Wait(cudaStream_t stream) const
        {
            if (m_internal_event == nullptr) return; // dummy event
            m_internal_event->Wait(stream);
        }

        void Sync() const
        {
            if (m_internal_event == nullptr) return; // dummy event
            m_internal_event->Sync();
        }

        bool Query() const
        {
            if (m_internal_event == nullptr) return true; // dummy event
            return m_internal_event->Query();
        }
    };

    class EventGroup : public IEvent
    {
    private:
        std::vector<Event> m_internal_events;

    public:
        EventGroup()
        {
        }

        ~EventGroup()
        {
        }

        static Event Create(const std::vector<Event>& evs)
        {
            auto ev_group = std::make_shared<EventGroup>();
            ev_group->m_internal_events = evs;
            return Event(ev_group);
        }

        void Add(Event ev)
        {
            m_internal_events.push_back(ev);
        }

        void Merge(EventGroup& other)
        {
            for (auto& ev : other.m_internal_events)
            {
                m_internal_events.push_back(std::move(ev));
            }

            other.m_internal_events.clear();
        }

        void Wait(cudaStream_t stream) const override 
        {
            for (auto& ev : m_internal_events)
            {
                ev.Wait(stream);
            }
        }

        void Sync() const override 
        {
            for (auto& ev : m_internal_events)
            {
                ev.Sync();
            }
        }

        bool Query() const override 
        {
            for (auto& ev : m_internal_events)
            {
                if (!ev.Query()) return false;
            }
            return true;
        }
    };

    /**
    * @brief A simple helper class for aggregating events with some pre known count until setting a promise
    */
    class AggregatedEventPromise
    {
    private:
        std::promise<Event> m_promise;
        std::shared_future<Event> m_shared_future;
        std::shared_ptr<EventGroup> m_ev_group;
    
        int m_reporters_count;
        mutable std::mutex m_mutex;
    
        void Complete()
        {
            assert(!is_ready(m_shared_future));
            m_promise.set_value(Event(m_ev_group));
        }
    
    public:
        AggregatedEventPromise(int reporters_count = 0) : m_reporters_count(reporters_count)
        {
            m_ev_group = std::make_shared<EventGroup>();
            m_shared_future = m_promise.get_future();
        }
    
        ~AggregatedEventPromise()
        {
            assert(is_ready(m_shared_future));
        }
    
        std::shared_future<Event> GetFuture() const
        {
            return m_shared_future;
        }
    
        void SetReportersCount(int reporters_count)
        {
            m_reporters_count = reporters_count;
        }
    
        void Report(EventGroup& group)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_ev_group->Merge(group);
            if (--m_reporters_count == 0) Complete();
        }
    };

    /**
    * @brief Event-pool for managing and recycling per device cuda events
    */
    class EventPool
    {
    private:
        std::vector<cudaEvent_t> m_events;
        std::deque<cudaEvent_t> m_pool;

        int m_dev_id; // the real device id
        mutable std::mutex m_mutex;

        void VerifyDev() const
        {
#ifndef NDEBUG
            int actual_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&actual_dev));
            if(actual_dev != m_dev_id)
            {
                printf("\nWarning: actual dev: %d, expected dev: %d\n", actual_dev, m_dev_id);
            }
#endif
        }

        cudaEvent_t Create() const
        {
            VerifyDev();

            cudaEvent_t ev;
            GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            return ev;
        }

        void Destroy(cudaEvent_t ev) const
        {
            GROUTE_CUDA_CHECK(cudaEventDestroy(ev));
        }

    public:
        EventPool(int dev_id, size_t cached_evs = 0) : 
            m_dev_id(dev_id)
        {
            m_events.reserve(cached_evs);
            CacheEvents(cached_evs);
        }

        void CacheEvents(size_t cached_evs)
        {
            GROUTE_CUDA_CHECK(cudaSetDevice(m_dev_id));

            std::lock_guard<std::mutex> guard(m_mutex);

            for (size_t i = m_events.size(); i < cached_evs; i++)
            {
                m_events.push_back(Create());
                m_pool.push_back(m_events[i]);
            }
        }

        int GetCachedEventsNum() const
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            return m_events.size();
        }

        ~EventPool()
        {
            for (auto ev : m_events)
            {
                Destroy(ev);
            }
        }

        Event Record(cudaStream_t stream)
        {
            cudaEvent_t cuda_event;

            { // guard block
                std::lock_guard<std::mutex> guard(m_mutex);

                if (m_pool.empty())
                {
                    m_events.push_back(cuda_event = Create());
                }
                else
                {
                    cuda_event = m_pool.front();
                    m_pool.pop_front();
                }
            }
            
            VerifyDev();

            // Record the event on the provided stream
            // The stream must be associated with the same device as the event
            GROUTE_CUDA_CHECK(cudaEventRecord(cuda_event, stream));

            return Event(
                cuda_event, 
                [this](cudaEvent_t ev) // releaser, called on destruction of internal EventHolder
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_pool.push_back(ev);
                }
            );
        }
    };

    enum StreamPriority
    {
        SP_Default, SP_High, SP_Low
    };

    class Stream
    {
    public:
        cudaStream_t    cuda_stream;
        cudaEvent_t     sync_event;

        Stream(int dev_id, StreamPriority priority = SP_Default)
        {
            GROUTE_CUDA_CHECK(cudaSetDevice(dev_id));
            Init(priority);
        }

        Stream(StreamPriority priority = SP_Default)
        {
            Init(priority);
        }

        void Init(StreamPriority priority)
        {
            if (priority == SP_Default)
            {
                GROUTE_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
                GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));
            }

            else
            {
                int leastPriority, greatestPriority;
                cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority); // range: [*greatestPriority, *leastPriority]

                GROUTE_CUDA_CHECK(cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, priority == SP_High ? greatestPriority : leastPriority));
                GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));
            }
        }

        Stream(const Stream& other) = delete;

        Stream(Stream&& other) : cuda_stream(other.cuda_stream), sync_event(other.sync_event)
        {
            other.cuda_stream = nullptr;
            other.sync_event = nullptr;
        }

        Stream& operator=(const Stream& other) = delete;

        Stream& operator=(Stream&& other) 
        {
            this->cuda_stream = other.cuda_stream;
            this->sync_event = other.sync_event;

            other.cuda_stream = nullptr;
            other.sync_event = nullptr;

            return *this;
        }

        ~Stream()
        {
            if(cuda_stream != nullptr) GROUTE_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
            if(sync_event != nullptr) GROUTE_CUDA_CHECK(cudaEventDestroy(sync_event));
        }

        void Sync() const
        {
            GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, cuda_stream));
            GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
        }
    };
}

#endif // __GROUTE_EVPOOL_H
