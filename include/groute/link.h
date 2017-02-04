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

#ifndef __GROUTE_LINK_H
#define __GROUTE_LINK_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>

#include <cuda_runtime.h>

#include <groute/communication.h>
#include <groute/router.h>
#include <groute/policy.h>


namespace groute {
    
    template <typename T>
    class Link : public router::IPipelinedReceiver <T>, public router::ISender <T>
    {
    private:
        router::ISender<T>* m_sender;
        std::shared_ptr<router::IPipelinedReceiver<T>> m_receiver;
        std::shared_ptr<router::Router<T>> m_p2p_router; // Used only for direct links between endpoints
    
    public:
        Link() : m_sender(nullptr), m_receiver(nullptr), m_p2p_router(nullptr)
        {
        }

        Link(Endpoint from, router::IRouter<T>& to, size_t packet_size = 0, size_t num_buffers = 0) : m_sender(nullptr), m_receiver(nullptr), m_p2p_router(nullptr)
        {
            m_sender = to.GetSender(from, packet_size, num_buffers);
        }
    
        Link(router::IRouter<T>& from, Endpoint to, size_t packet_size, size_t num_buffers) : m_sender(nullptr), m_receiver(nullptr), m_p2p_router(nullptr)
        {
            m_receiver = from.CreatePipelinedReceiver(to, packet_size, num_buffers);
        }
    
        Link(Context& context, Endpoint from, Endpoint to, size_t receive_packet_size, size_t receive_num_buffers, size_t send_packet_size = 0, size_t send_num_buffers = 0) : 
            m_sender(nullptr), m_receiver(nullptr), m_p2p_router(nullptr)
        {
            m_p2p_router = std::make_shared<router::Router<T>>(context, router::Policy::CreateP2PPolicy(from, to)); // must keep a reference to the router
            
            m_sender = m_p2p_router->GetSender(from, send_packet_size, send_num_buffers); 
            m_receiver = m_p2p_router->CreatePipelinedReceiver(to, receive_packet_size, receive_num_buffers);
        }
    
        router::ISender<T>* GetSender() const
        {
            assert(m_sender);
            return m_sender;
        }
    
        router::IPipelinedReceiver<T>* GetReceiver() const
        {
            assert(m_receiver.get());
            return m_receiver.get();
        }
    
        void Sync() const override
        {
            GetReceiver()->Sync();
        }
    
        std::shared_future< router::PendingSegment<T> > Receive() override
        {
            return GetReceiver()->Receive();
        }
    
        void ReleaseBuffer(const Segment<T>& segment, const Event& ready_event) override
        {
            GetReceiver()->ReleaseBuffer(segment, ready_event);
        }
        
        std::shared_future< router::PendingSegment<T> > Receive(const Buffer<T>& dst_buffer, const Event& ready_event) override
        {
            return GetReceiver()->Receive(dst_buffer, ready_event);
        }
    
        bool Active() override
        {
            return GetReceiver()->Active();
        }
    
        std::shared_future<Event> Send(const Segment<T>& segment, const Event& ready_event) override
        {
            return GetSender()->Send(segment, ready_event);
        }
    
        Segment<T> GetSendBuffer() override
        {
            return GetSender()->GetSendBuffer();
        }
    
        void ReleaseSendBuffer(const Segment<T>& segment, const Event& ready_event) override
        {
            GetSender()->ReleaseSendBuffer(segment, ready_event);
        }
    
        void Shutdown() override
        {
            GetSender()->Shutdown();
        }
    };
}

#endif // __GROUTE_LINK_H
