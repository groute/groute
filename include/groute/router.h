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

#ifndef __GROUTE_ROUTER_H
#define __GROUTE_ROUTER_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>

#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/context.h>
#include <groute/communication.h>


namespace groute {
    namespace router {

        /**
        * @brief Represents a segment receive operation from a single source to a single destination
        */
        template <typename T>
        class ReceiveOperation
        {
        private:
            std::promise< PendingSegment<T> > m_promise;
            std::shared_future< PendingSegment<T> > m_shared_future;

            Endpoint m_src_endpoint;
            Endpoint m_dst_endpoint;

            Segment<T> m_src_segment; // The source is a segment of some valid data
            Buffer<T> m_dst_buffer; // The destination is any memory buffer with enough space
            Event m_dst_ready_event;

        public:
            ReceiveOperation(Endpoint dst_endpoint, const Buffer<T>& dst_buffer, const Event& dst_ready_event) :
                m_src_endpoint(), m_dst_endpoint(dst_endpoint), m_dst_buffer(dst_buffer), m_dst_ready_event(dst_ready_event)
            {
                m_shared_future = m_promise.get_future(); // get the future and (implicitly) cast to a shared future  
            }

            ReceiveOperation() :
                m_src_endpoint(), m_dst_endpoint(), m_dst_buffer(nullptr, 0)
            {
                m_shared_future = m_promise.get_future(); // get the future and (implicitly) cast to a shared future  
            }

            ~ReceiveOperation()
            {
                assert(is_ready(m_shared_future));
            }

            std::shared_future< PendingSegment<T> > GetFuture() const
            {
                return m_shared_future;
            }

            Endpoint GetSrcEndpoint() const { return m_src_endpoint; }
            Endpoint GetDstEndpoint() const { return m_dst_endpoint; }

            void SetSrcSegment(Endpoint src_endpoint, const Segment<T>& src_segment)
            {
                m_src_endpoint = src_endpoint;
                m_src_segment = Segment<T>(src_segment);
            }

            const Segment<T>& GetSrcSegment() const
            {
                return m_src_segment;
            }

            void SetDstBuffer(Endpoint dst_endpoint, const Buffer<T>& dst_buffer)
            {
                m_dst_endpoint = dst_endpoint;
                m_dst_buffer = Buffer<T>(dst_buffer);
            }

            const Buffer<T>& GetDstBuffer() const
            {
                return m_dst_buffer;
            }

            Event GetDstReadyEvent() const
            {
                return m_dst_ready_event;
            }

            void Complete(Event ready_event)
            {
                assert(!is_ready(m_shared_future));

                m_promise.set_value(
                    PendingSegment<T>(
                    m_dst_buffer.GetPtr(), m_src_segment.GetTotalSize(),
                    m_src_segment.GetSegmentSize(), m_src_segment.GetSegmentOffset(),
                    ready_event));
            }

            void Cancel()
            {
                assert(!is_ready(m_shared_future));

                m_promise.set_value(PendingSegment<T>(m_dst_buffer.GetPtr(), 0, 0, 0, Event()));
            }
        };

        /**
        * @brief Represents a segment send operation from a single source to multiple destinations
        *           each segment send reports to an AggregatedEventPromise on completion
        */
        template <typename T>
        class SendOperation
        {
        private:
            std::shared_ptr<AggregatedEventPromise> m_aggregated_event;
            std::shared_ptr<EventGroup> m_ev_group;

            Endpoint m_src_endpoint;
            Segment<T> m_src_segment;
            Event m_src_ready_event;

            size_t m_pos;
            size_t m_progress;
            mutable std::mutex m_mutex;

            void Complete()
            {
                m_aggregated_event->Report(*m_ev_group);
            }

        public:
            SendOperation(
                std::shared_ptr<AggregatedEventPromise> aggregated_event,
                Endpoint src_endpoint, const Segment<T>& src_segment, const Event& src_ready_event) :
                m_aggregated_event(aggregated_event),
                m_src_endpoint(src_endpoint), m_src_segment(src_segment), 
                m_src_ready_event(src_ready_event), m_pos(0), m_progress(0)
            {
                m_ev_group = std::make_shared<EventGroup>();
            }

            ~SendOperation()
            {
            }

            const Segment<T>& GetSrcSegment() const
            {
                return m_src_segment;
            }

            Event GetSrcReadyEvent() const
            {
                return m_src_ready_event;
            }

            Endpoint GetSrcEndpoint() const
            {
                return m_src_endpoint;
            }

            Segment<T> OccupySubSegment(size_t max_size)
            {
                size_t total_src_size = m_src_segment.GetSegmentSize();

                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_pos >= total_src_size) return Segment<T>();

                size_t ss_size = (m_pos + max_size) > total_src_size ? (total_src_size - m_pos) : max_size;
                size_t ss_pos = m_pos;

                m_pos += ss_size;

                return m_src_segment.GetSubSegment(ss_pos, ss_size);
            }

            bool AssignReceiveOperation(std::shared_ptr< ReceiveOperation<T> > receive_op)
            {
                auto src_ss = OccupySubSegment(receive_op->GetDstBuffer().GetSize());
                if (src_ss.GetSegmentSize() == 0) return false;

                receive_op->SetSrcSegment(m_src_endpoint, src_ss);
                return true;
            }

            void ReportProgress(size_t progress, const Event& ready_event)
            {
                std::lock_guard<std::mutex> lock(m_mutex); // use another mutex ?..
                if (m_progress >= m_src_segment.GetSegmentSize()) return;
                m_progress += progress;
                m_ev_group->Add(ready_event);
                if (m_progress >= m_src_segment.GetSegmentSize()) Complete();
            }
        };


        enum RouteStrategy
        {
            Availability, Priority, Broadcast,
        };

        struct Route
        {
            EndpointList dst_endpoints; // order matters if strategy is Priority  
            RouteStrategy strategy;

            Route(RouteStrategy strategy = Availability) : strategy(strategy) { }

            Route(const std::vector<Endpoint>& dst_endpoints, RouteStrategy strategy = Availability) :
                dst_endpoints(dst_endpoints), strategy(strategy)
            {
            }
        };

        struct IPolicy
        {
            virtual ~IPolicy() { }

            virtual RoutingTable GetRoutingTable() = 0; // TODO: we can avoid this
            virtual Route GetRoute(Endpoint src, void* message_metadata) = 0;
        };

        struct IRouterBase // an untyped base interface for the Router
        {
            virtual ~IRouterBase() { }
            virtual void Shutdown() = 0;
        };

        template <typename T>
        struct IRouter : IRouterBase
        {
            virtual ~IRouter() { }

            virtual ISender<T>* GetSender(Endpoint endpoint, size_t chunk_size, size_t num_chunks) = 0;
            virtual IReceiver<T>* GetReceiver(Endpoint endpoint) = 0;

            virtual std::unique_ptr< IPipelinedReceiver<T> > CreatePipelinedReceiver(Endpoint endpoint, size_t chunk_size, size_t num_buffers) = 0;
        };

        /**
        * @brief The focal point between multiple senders and receivers.
        *        The router routs data from a sender into one/many receivers
        */
        template <typename T>
        class Router : public IRouter < T >
        {
        private:
            Context& m_context;

            std::shared_ptr<IPolicy> m_policy;
            RoutingTable m_possible_routes;

            class InactiveReceiver;
            class InactiveSender;
            class Receiver;
            class Sender;

            std::map<Endpoint, std::unique_ptr<Receiver> > m_receivers;
            std::map<Endpoint, std::unique_ptr<Sender> > m_senders;

            std::map<Endpoint, std::unique_ptr<InactiveReceiver> > m_inactive_receivers;
            std::map<Endpoint, std::unique_ptr<InactiveSender> > m_inactive_senders;
            std::mutex m_inactive_mutex;

            class InactiveReceiver : public IReceiver < T >
            {
            public:
                std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) override
                {
                    return groute::completed_future(PendingSegment<T>());
                }

                bool Active() override { return false; }
            };

            class InactiveSender : public ISender < T >
            {
            public:
                InactiveSender()  { }
                ~InactiveSender() { }

                std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) override
                {
                    throw std::exception(); // cannot send segments on an inactive sender
                }

                void Shutdown() override
                {
                }

                Segment<T> GetSendBuffer() override { return Segment<T>();  }
                void ReleaseSendBuffer(const Segment<T>& segment, const Event& ready_event) override {}
            };

            class Receiver : public IReceiver < T >
            {
            private:
                Router<T>& m_router;
                const Endpoint m_endpoint;

                std::set<Endpoint> m_possible_senders;

                std::mutex m_mutex; 

                std::deque < std::shared_ptr< SendOperation<T> > > m_send_queue;
                std::deque < std::shared_ptr< ReceiveOperation<T> > > m_receive_queue;

            public:
                Receiver(Router<T>& router, Endpoint endpoint) : m_router(router), m_endpoint(endpoint) { }
                ~Receiver() { }

                std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) override
                {
                    auto receive_op =
                        std::make_shared< ReceiveOperation<T> >(m_endpoint, dst_buffer, ready_event);

                    QueueReceiveOp(receive_op);
                    if (!Assign())
                    {
                        CheckPossibleSenders();
                    }

                    return receive_op->GetFuture();
                }

                bool Active() override
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    return (!m_send_queue.empty() || !m_possible_senders.empty());
                }

                void QueueSendOp(const std::shared_ptr< SendOperation<T> >& send_op)
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_send_queue.push_back(send_op);
                }

                void QueueReceiveOp(const std::shared_ptr< ReceiveOperation<T> >& receive_op)
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_receive_queue.push_back(receive_op);
                }

                /*
                * @brief Tries to perform a single 'assignment' between a receive operation and a send operation (both at queues front)
                */
                bool Assign()
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    if (m_send_queue.empty() || m_receive_queue.empty()) return false;

                    auto send_op = m_send_queue.front();
                    auto receive_op = m_receive_queue.front();

                    while (true)
                    {
                        if (send_op->AssignReceiveOperation(receive_op))
                        {
                            m_router.QueueMemcpyWork(send_op, receive_op);
                            m_receive_queue.pop_front();
                            return true;
                        }

                        m_send_queue.pop_front();
                        if (m_send_queue.empty()) return false;

                        send_op = m_send_queue.front();
                    }
                }

                void CheckPossibleSenders()
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    if (m_send_queue.empty() && m_possible_senders.empty())
                    {
                        for (auto& receive_op : m_receive_queue)
                        {
                            receive_op->Cancel();
                        }
                        m_receive_queue.clear();
                    }
                }

                void AddPossibleSender(Endpoint endpoint)
                {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_possible_senders.insert(endpoint);
                }

                void RemovePossibleSender(Endpoint endpoint)
                {
                    {
                        std::lock_guard<std::mutex> guard(m_mutex);
                        m_possible_senders.erase(endpoint);
                    }
                    CheckPossibleSenders();
                }

                void ClearPossibleSenders()
                {
                    {
                        std::lock_guard<std::mutex> guard(m_mutex);
                        m_possible_senders.clear();
                    }
                    CheckPossibleSenders();
                }
            };

            struct SegmentPool
            {
                Endpoint m_endpoint;
                std::deque<T*> m_buffers;
                std::deque<T*> m_buffers_in_use;
                int m_chunksize;

                std::mutex m_lock, m_destructo_lock;

                SegmentPool(const Context& ctx, Endpoint endpoint, int chunksize, int numchunks) :
                    m_endpoint(endpoint), m_chunksize(chunksize)
                {
                    ctx.SetDevice(endpoint);
                    for (int i = 0; i < numchunks; ++i)
                    {
                        T * buff;

                        if (endpoint.IsHost())
                            GROUTE_CUDA_CHECK(cudaMallocHost(&buff, chunksize*sizeof(T)));
                        else
                            GROUTE_CUDA_CHECK(cudaMalloc(&buff, chunksize*sizeof(T)));

                        m_buffers.push_back(buff);
                    }
                }

                ~SegmentPool()
                {
                    std::lock_guard<std::mutex> guard2(m_destructo_lock);
                    std::lock_guard<std::mutex> guard(m_lock);
                    
                    if (m_buffers_in_use.size() > 0)
                    {
                        printf("ERROR: SOME (%llu) BUFFERS ARE STILL IN USE, NOT DEALLOCATING\n",
                               m_buffers_in_use.size());
                    }
                    for (T* buff : m_buffers)
                    {
                        if (m_endpoint.IsHost())
                            GROUTE_CUDA_CHECK(cudaFreeHost(buff));
                        else
                            GROUTE_CUDA_CHECK(cudaFree(buff));
                    }

                    m_buffers.clear();
                    m_buffers_in_use.clear();
                }

                Segment<T> GetBuffer()
                {
                    std::lock_guard<std::mutex> guard(m_lock);

                    if (m_buffers.empty())
                        return Segment<T>();

                    T* buff = m_buffers.front();
                    m_buffers.pop_front();
                    m_buffers_in_use.push_back(buff);
                    return Segment<T>(buff, m_chunksize);
                }

                void ReleaseBuffer(T* buff)
                {
                    std::lock_guard<std::mutex> guard(m_lock);

                    for (auto iter = m_buffers_in_use.begin(); iter != m_buffers_in_use.end(); ++iter)
                    {
                        if (*iter == buff)
                        {
                            m_buffers_in_use.erase(iter);
                            m_buffers.push_back(buff);
                            return;
                        }
                    }
                    printf("ERROR: NO SUCH BUFFER EXISTS\n");
                }

                void ReleaseBufferEvent(T* buff, int physical_dev, const Event& ev)
                {
                    std::lock_guard<std::mutex> guard(m_destructo_lock);

                    if (physical_dev >= 0)
                        GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev));
                    ev.Sync();

                    ReleaseBuffer(buff);
                }
            };

            class Sender : public ISender < T >
            {
            private:
                Router<T>& m_router;
                const Endpoint m_endpoint;
                bool m_shutdown;

                SegmentPool m_segpool;

            public:
                Sender(Router<T>& router, Endpoint endpoint, int chunksize = 0, int numchunks = 0) : 
                    m_router(router), m_endpoint(endpoint), m_shutdown(false), m_segpool(router.m_context, endpoint, chunksize, numchunks) { }
                ~Sender() { }

                std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) override
                {
                    if (m_shutdown)
                    {
                        //throw std::exception();
                        return groute::completed_future(Event()); // Fail gracefully. TODO: should be configurable  
                    }

                    if (segment.Empty()) return groute::completed_future(Event());

                    return QueueSendOps(segment, ready_event)
                        ->GetFuture();
                }

                void Shutdown() override
                {
                    m_shutdown = true;

                    if (m_router.m_possible_routes.find(m_endpoint) == m_router.m_possible_routes.end()) return;

                    for (auto dst_endpoint : m_router.m_possible_routes.at(m_endpoint))
                    {
                        m_router.m_receivers.at(dst_endpoint)
                            ->RemovePossibleSender(m_endpoint);
                    }
                }

                Segment<T> GetSendBuffer() override 
                { 
                    return m_segpool.GetBuffer();
                }

                void ReleaseSendBuffer(const Segment<T>& segment, const Event& ready_event) override 
                {
                    int physical_dev = m_router.m_context.GetPhysicalDevice(m_endpoint);

                    std::thread asyncrelease([](SegmentPool& segpool, int pd, Segment<T> seg, Event ev) {
                        segpool.ReleaseBufferEvent(seg.GetSegmentPtr(), pd, ev);
                    }, std::ref(m_segpool), physical_dev, segment, ready_event);
                    asyncrelease.detach();
                }

                void AssertRoute(Endpoint src_endpoint, const Route& route) const
                {
                    for (Endpoint dst_endpoint : route.dst_endpoints)
                    {
                        if (m_router.m_possible_routes.find(src_endpoint) == m_router.m_possible_routes.end()) {
                            printf(
                                "\n\nWarning: %d was not configured to be a possible route source by the current policy, please fix the policy implementation\n\n", 
                                (Endpoint::identity_type)src_endpoint);
                            exit(1);
                        }

                        bool possible = false;
                        for (Endpoint possible_dst_endpoint : m_router.m_possible_routes.at(src_endpoint))
                        {
                            if (possible_dst_endpoint == dst_endpoint)
                            {
                                possible = true;
                                break;
                            }
                        }

                        if (!possible)
                        {
                            printf(
                                "\n\nWarning: (%d -> %d) was not configured to be a possible route by the current policy, please fix the policy implementation\n\n", 
                                (Endpoint::identity_type)src_endpoint, (Endpoint::identity_type)dst_endpoint);
                            exit(1);
                        }
                    }
                }

                std::shared_ptr<AggregatedEventPromise> QueueSendOps(const Segment<T>& segment, const Event& ready_event)
                {
                    auto aggregated_event = std::make_shared<AggregatedEventPromise>();  

                    Route route = m_router.m_policy->GetRoute(m_endpoint, segment.metadata);
#ifndef NDEBUG
                    AssertRoute(m_endpoint, route);
#endif
                    switch (route.strategy)
                    {
                    case Availability:
                        aggregated_event->SetReportersCount(1);
                        AvailabilityMultiplexing(segment, route, ready_event, aggregated_event);
                        break;

                    case Priority:
                        aggregated_event->SetReportersCount(1);
                        PriorityMultiplexing(segment, route, ready_event, aggregated_event);
                        break;

                    case Broadcast:
                        aggregated_event->SetReportersCount((int)route.dst_endpoints.size());
                        BroadcastMultiplexing(segment, route, ready_event, aggregated_event);
                        break;
                    }

                    return aggregated_event;
                }

                // TODO: Refactor Multiplexer object

                void AvailabilityMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& aggregated_event)
                {
                    auto send_op = std::make_shared< SendOperation<T> >(aggregated_event, m_endpoint, segment, ready_event);

                    for (auto dst_endpoint : route.dst_endpoints)
                    {
                        m_router.m_receivers.at(dst_endpoint)
                            ->QueueSendOp(send_op);
                    }

                    bool assigning = true;
                    while (assigning)
                    {
                        assigning = false;
                        for (auto dst_endpoint : route.dst_endpoints)
                        {
                            if (m_router.m_receivers.at(dst_endpoint) // go on at rounds, give an equal chance to all receivers  
                                ->Assign())
                            {
                                assigning = true;
                            }
                        }
                    }
                }

                void PriorityMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& aggregated_event)
                {
                    auto send_op = std::make_shared< SendOperation<T> >(aggregated_event, m_endpoint, segment, ready_event);

                    for (auto dst_endpoint : route.dst_endpoints) // go over receivers by priority  
                    {
                        m_router.m_receivers.at(dst_endpoint)
                            ->QueueSendOp(send_op);

                        // let the prioritized receiver occupy as much as he can from the send_op  
                        while (m_router.m_receivers.at(dst_endpoint)
                            ->Assign());
                    }
                }

                void BroadcastMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& aggregated_event)
                {
                    for (auto dst_endpoint : route.dst_endpoints)
                    {
                        // create a send_op per receiver (broadcasting)  
                        auto send_op = std::make_shared< SendOperation<T> >(aggregated_event, m_endpoint, segment, ready_event);

                        m_router.m_receivers.at(dst_endpoint)
                            ->QueueSendOp(send_op);

                        while (m_router.m_receivers.at(dst_endpoint)
                            ->Assign());
                    }
                }
            };

            void QueueMemcpyWork(std::shared_ptr< SendOperation<T> > send_op, std::shared_ptr< ReceiveOperation<T> > receive_op)
            {
                m_context.QueueMemcpyWork(
                    receive_op->GetSrcEndpoint(), receive_op->GetSrcSegment().GetSegmentPtr(),
                    receive_op->GetDstEndpoint(), receive_op->GetDstBuffer().GetPtr(),
                    receive_op->GetSrcSegment().GetSegmentSize() * sizeof(T),
                    send_op->GetSrcReadyEvent(), receive_op->GetDstReadyEvent(),
                    [send_op, receive_op](size_t bytes, const Event& ready_event) // captures both shared pointers  
                {
                    receive_op->Complete(ready_event);
                    send_op->ReportProgress(bytes / sizeof(T), ready_event);
                }
                );
            }

        public:
            Router(Context& context, const std::shared_ptr<IPolicy>& policy) :
                m_context(context), m_policy(policy), m_possible_routes(policy->GetRoutingTable())
            {
                context.RequireMemcpyLanes(m_possible_routes);

                std::set<Endpoint> dst_endpoints;

                for (auto& p : m_possible_routes)
                {
                    Endpoint src_endpoint = p.first;
                    m_senders[src_endpoint] = groute::make_unique<Sender>(*this, src_endpoint);

                    // add all dst endpoints to the set
                    dst_endpoints.insert(std::begin(p.second), std::end(p.second));
                }

                // create a receiver for each dst endpoint
                for (auto dst_endpoint : dst_endpoints)
                {
                    m_receivers[dst_endpoint] = groute::make_unique<Receiver>(*this, dst_endpoint);
                }

                for (auto& p : m_possible_routes)
                {
                    Endpoint src_endpoint = p.first;
                    for (auto dst_endpoint : p.second)
                    {
                        m_receivers[dst_endpoint]->AddPossibleSender(src_endpoint);
                    }
                }
            }

            void Shutdown() override
            {
                for (auto& p : m_senders)
                {
                    p.second->Shutdown();
                }
            }

            ISender<T>* GetSender(Endpoint endpoint, size_t chunk_size = 0, size_t num_chunks = 0) override
            {
                if (m_senders.find(endpoint) == m_senders.end())
                    // no active sender, this means this endpoint was not registered as a sender by the policy
                    // just create an inactive one
                {
                    std::lock_guard<std::mutex> guard(m_inactive_mutex);
                    if (m_inactive_senders.find(endpoint) == m_inactive_senders.end())
                    {
                        m_inactive_senders[endpoint] = groute::make_unique<InactiveSender>();
                    }
                    return m_inactive_senders.at(endpoint).get();
                }

                // Create pipelined sender
                if (num_chunks > 0)
                    m_senders[endpoint] = groute::make_unique<Sender>(*this, endpoint, chunk_size, num_chunks);


                return m_senders.at(endpoint).get();
            }

            IReceiver<T>* GetReceiver(Endpoint endpoint) override
            {
                if (m_receivers.find(endpoint) == m_receivers.end())
                    // no active receiver, this means this endpoint was not registered as a receiver by the policy
                    // just create an inactive one
                {
                    std::lock_guard<std::mutex> guard(m_inactive_mutex);
                    if (m_inactive_receivers.find(endpoint) == m_inactive_receivers.end())
                    {
                        m_inactive_receivers[endpoint] = groute::make_unique<InactiveReceiver>();
                    }
                    return m_inactive_receivers.at(endpoint).get();
                }

                return m_receivers.at(endpoint).get();
            }

            std::unique_ptr< IPipelinedReceiver<T> > CreatePipelinedReceiver(Endpoint endpoint, size_t chunk_size, size_t num_buffers) override
            {
                return groute::make_unique<PipelinedReceiver<T>>(m_context, GetReceiver(endpoint), endpoint, chunk_size, num_buffers);
            }
        };
    }
}

#endif // __GROUTE_ROUTER_H
