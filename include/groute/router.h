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
    namespace internal {

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
                std::shared_ptr<AggregatedEventPromise> agg_event,
                Endpoint src_endpoint, const Segment<T>& src_segment, const Event& src_ready_event) :
                m_aggregated_event(agg_event),
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

    }
    
    template <typename T>
    class Link;

    enum RouteStrategy
    {
        Availability, Priority, Broadcast,
    };
    
    struct Route
    {
        EndpointList dst_endpoints; // order matters if strategy is Priority  
        RouteStrategy strategy;
    
        Route(RouteStrategy strategy = Availability) : strategy(strategy) { }
    
        Route(const EndpointList& dst_endpoints, RouteStrategy strategy = Availability) :
            dst_endpoints(dst_endpoints), strategy(strategy)
        {
        }
    };
    
    struct IPolicy
    {
        virtual ~IPolicy() { }
        virtual Route GetRoute(Endpoint src, const EndpointList& router_dst, void* message_metadata) const = 0;
    };

    typedef std::function<Route(Endpoint src, const EndpointList& router_dst, void* message_metadata)> PolicyFunc;

    class PolicyFuncObject : public IPolicy
    {
        PolicyFunc m_func;
    public:
        PolicyFuncObject(const PolicyFunc& func) : m_func(func) { }
        Route GetRoute(Endpoint src, const EndpointList& router_dst, void* message_metadata) const override 
        { 
            return m_func(src, router_dst, message_metadata); 
        }
    };
    
    struct IRouter // An untyped base interface for the Router
    {
        virtual ~IRouter() { }
        virtual void Shutdown() = 0;
    };
    
    /**
    * @brief The focal point between multiple senders and receivers.
    *        The router routs data from a sender into one/many receivers
    */
    template <typename T>
    class Router : public IRouter
    {
        friend class Link <T>;
    
        Context& m_context;
    
        std::shared_ptr<IPolicy> m_policy;
        RoutingTable m_possible_routes;
        EndpointList m_router_dst;

        int m_num_inputs, m_num_outputs;
        std::mutex m_initialization_mutex;

        volatile bool m_finalized;
    
        class Receiver;
        class Sender;
    
        std::map<Endpoint, std::unique_ptr<Receiver> > m_receivers;
        std::map<Endpoint, std::unique_ptr<Sender> > m_senders;
    
        class Receiver : public IReceiver < T >
        {
        private:
            Router<T>& m_router;
            const Endpoint m_endpoint;
    
            std::set<Endpoint> m_possible_senders;
    
            std::mutex m_mutex; 
    
            std::deque < std::shared_ptr< internal::SendOperation<T> > > m_send_queue;
            std::deque < std::shared_ptr< internal::ReceiveOperation<T> > > m_receive_queue;
    
        public:
            Receiver(Router<T>& router, Endpoint endpoint) : m_router(router), m_endpoint(endpoint) { }
            ~Receiver() { }
    
            std::shared_future< PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) override
            {
                auto receive_op =
                    std::make_shared< internal::ReceiveOperation<T> >(m_endpoint, dst_buffer, ready_event);
    
                QueueReceiveOp(receive_op);
                if (!Assign())
                {
                    CheckPossibleSenders();
                }
    
                return receive_op->GetFuture();
            }
    
            bool Active() override
            {
                if (!m_router.m_finalized) return true; // Routing schema is not finalized and senders are unknown yet, hence we must assume this receiver is Active 

                std::lock_guard<std::mutex> guard(m_mutex);
                return (!m_send_queue.empty() || !m_possible_senders.empty());
            }
    
            void QueueSendOp(const std::shared_ptr< internal::SendOperation<T> >& send_op)
            {
                std::lock_guard<std::mutex> guard(m_mutex);
                m_send_queue.push_back(send_op);
            }
    
            void QueueReceiveOp(const std::shared_ptr< internal::ReceiveOperation<T> >& receive_op)
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
                if (!m_router.m_finalized) return;

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
    
        class Sender : public ISender < T >
        {
        private:
            Router<T>& m_router;
            const Endpoint m_endpoint;
            volatile bool m_shutdown;
    
        public:
            Sender(Router<T>& router, Endpoint endpoint) : 
                m_router(router), m_endpoint(endpoint), m_shutdown(false) { }
            ~Sender() { }
    
            std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) override
            {
                m_router.AssertFinalized();

                if (m_shutdown)
                {
                    printf("\n\nWarning: Sender was Shutdown and is now used again for Send\n\n");
                    throw std::exception("Sender was Shutdown"); 
                }
    
                if (segment.Empty()) return groute::completed_future(Event());
    
                return QueueSendOps(segment, ready_event)
                    ->GetFuture();
            }
    
            void Shutdown() override
            {
                m_router.AssertFinalized();
                m_shutdown = true;
    
                if (m_router.m_possible_routes.find(m_endpoint) == m_router.m_possible_routes.end()) return;
    
                for (auto dst_endpoint : m_router.m_possible_routes.at(m_endpoint))
                {
                    m_router.m_receivers.at(dst_endpoint)
                        ->RemovePossibleSender(m_endpoint);
                }
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
                auto agg_event = std::make_shared<AggregatedEventPromise>();  
    
                Route route = m_router.m_policy->GetRoute(m_endpoint, m_router.m_router_dst, segment.metadata);
#ifndef NDEBUG
                AssertRoute(m_endpoint, route);
#endif
                switch (route.strategy)
                {
                case Availability:
                    agg_event->SetReportersCount(1);
                    AvailabilityMultiplexing(segment, route, ready_event, agg_event);
                    break;
    
                case Priority:
                    agg_event->SetReportersCount(1);
                    PriorityMultiplexing(segment, route, ready_event, agg_event);
                    break;
    
                case Broadcast:
                    agg_event->SetReportersCount((int)route.dst_endpoints.size());
                    BroadcastMultiplexing(segment, route, ready_event, agg_event);
                    break;
                }
    
                return agg_event;
            }
    
            // TODO: Refactor Multiplexer object
    
            void AvailabilityMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& agg_event)
            {
                auto send_op = std::make_shared< internal::SendOperation<T> >(agg_event, m_endpoint, segment, ready_event);
    
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
    
            void PriorityMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& agg_event)
            {
                auto send_op = std::make_shared< internal::SendOperation<T> >(agg_event, m_endpoint, segment, ready_event);
    
                for (auto dst_endpoint : route.dst_endpoints) // go over receivers by priority  
                {
                    m_router.m_receivers.at(dst_endpoint)
                        ->QueueSendOp(send_op);
    
                    // let the prioritized receiver occupy as much as he can from the send_op  
                    while (m_router.m_receivers.at(dst_endpoint)
                        ->Assign());
                }
            }
    
            void BroadcastMultiplexing(const Segment<T>& segment, const Route& route, const Event& ready_event, const std::shared_ptr<AggregatedEventPromise>& agg_event)
            {
                for (auto dst_endpoint : route.dst_endpoints)
                {
                    // create a send_op per receiver (broadcasting)  
                    auto send_op = std::make_shared< internal::SendOperation<T> >(agg_event, m_endpoint, segment, ready_event);
    
                    m_router.m_receivers.at(dst_endpoint)
                        ->QueueSendOp(send_op);
    
                    while (m_router.m_receivers.at(dst_endpoint)
                        ->Assign());
                }
            }
        };
    
        void QueueMemcpyWork(std::shared_ptr< internal::SendOperation<T> > send_op, std::shared_ptr< internal::ReceiveOperation<T> > receive_op)
        {
            m_context.QueueMemcpyWork(
                receive_op->GetSrcEndpoint(), receive_op->GetSrcSegment().GetSegmentPtr(),
                receive_op->GetDstEndpoint(), receive_op->GetDstBuffer().GetPtr(),
                receive_op->GetSrcSegment().GetSegmentSize() * sizeof(T),
                send_op->GetSrcReadyEvent(), receive_op->GetDstReadyEvent(),
                [send_op, receive_op](size_t bytes, const Event& ready_event) // callback, captures both shared pointers  
                {
                    receive_op->Complete(ready_event);
                    send_op->ReportProgress(bytes / sizeof(T), ready_event);
                }
            );
        }
    
    public:
        Router(Context& context, const std::shared_ptr<IPolicy>& policy, int num_inputs, int num_outputs) :
            m_context(context), m_policy(policy), m_num_inputs(num_inputs), m_num_outputs(num_outputs), m_finalized(false)
        {
        }

        Router(Context& context, const PolicyFunc& policy, int num_inputs, int num_outputs) :
            m_context(context), m_policy(std::make_shared<PolicyFuncObject>(policy)), m_num_inputs(num_inputs), m_num_outputs(num_outputs), m_finalized(false)
        {
        }
    
        void Shutdown() override
        {
            for (auto& p : m_senders)
            {
                p.second->Shutdown();
            }
        }
    
    private:

        void AssertNotFinalized() const
        {
            if (m_finalized) 
            {
                printf("\n\nWarning: Router is finalized, cannot add more links at this point\n\n");
                throw std::exception("Router already finalized"); 
            }
        }

        void AssertFinalized() const
        {
            if (!m_finalized) 
            {
                printf("\n\nWarning: Router is not finalized yet, cannot send data through links\n\n");
                throw std::exception("Router not finalized yet");
            }
        }

        /// @brief Finalizes the routing schema for this router
        void Finalize()
        {
            m_possible_routes.clear();
            m_router_dst.clear();
            m_router_dst.reserve(m_num_outputs);

            for (auto& receiver : m_receivers)
            {
                m_router_dst.push_back(receiver.first);
            }

            for (auto& sender : m_senders)
            {
                Endpoint src = sender.first;

                // When passing nullptr as metadata, the policy is expected to return all possible routes between src -> router_dst
                auto route = m_policy->GetRoute(src, m_router_dst, nullptr); 

                // Keep the info in a table for future assertions and Shutdown management
                m_possible_routes[src] = route.dst_endpoints; 

                for (Endpoint dst : route.dst_endpoints)
                {
                    if (m_receivers.find(dst) == m_receivers.end()) 
                    {
                        printf("\n\nWarning: Policy specified a destination endpoint which was not registered to the router through any link\n\n");
                        throw std::exception("Destination not found");
                    }

                    m_context.RequireMemcpyLane(src, dst);
                    m_receivers[dst]->AddPossibleSender(src);
                }
            }

            m_finalized = true;

            for (auto& receiver : m_receivers)
            {
                receiver.second->CheckPossibleSenders();
            }
        }

        void TryFinalize()
        {
            int registered_inputs = m_senders.size();
            int registered_outputs = m_receivers.size();

            if (m_num_inputs == registered_inputs && m_num_outputs == registered_outputs) 
                // All expected senders and receivers are registered, finalize
            {
                Finalize();
            }
        }
    
        //
        // Internal router API, used by friend class Link
        //
    
        ISender<T>* GetSender(Endpoint endpoint, size_t chunk_size = 0, size_t num_chunks = 0)
        {
            std::lock_guard<std::mutex> guard(m_initialization_mutex);
            if (m_senders.find(endpoint) == m_senders.end())
            {
                AssertNotFinalized();
                m_senders[endpoint] = groute::make_unique<Sender>(*this, endpoint);
                TryFinalize();
            }

            return m_senders.at(endpoint).get();
        }
    
        IReceiver<T>* GetReceiver(Endpoint endpoint) 
        {
            std::lock_guard<std::mutex> guard(m_initialization_mutex);
            if (m_receivers.find(endpoint) == m_receivers.end())
            {
                AssertNotFinalized();
                m_receivers[endpoint] = groute::make_unique<Receiver>(*this, endpoint); 
                TryFinalize();
            }
    
            return m_receivers.at(endpoint).get();
        }
    
        std::unique_ptr< IPipelinedReceiver<T> > CreatePipelinedReceiver(Endpoint endpoint, size_t chunk_size, size_t num_buffers)
        {
            return groute::make_unique<PipelinedReceiver<T>>(m_context, GetReceiver(endpoint), endpoint, chunk_size, num_buffers);
        }

        std::unique_ptr< IPipelinedSender<T> > CreatePipelinedSender(Endpoint endpoint, size_t chunk_size, size_t num_buffers)
        {
            return groute::make_unique<PipelinedSender<T>>(m_context, GetSender(endpoint), endpoint, chunk_size, num_buffers);
        }
    };
}

#endif // __GROUTE_ROUTER_H
