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

#ifndef __GROUTE_FUSED_DISTRIBUTED_WORKLIST_H
#define __GROUTE_FUSED_DISTRIBUTED_WORKLIST_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>

#include <gflags/gflags_declare.h>

#include <groute/event_pool.h>
#include <groute/context.h>
#include <groute/worklist.h>
#include <groute/groute.h>

DECLARE_bool(verbose);
DECLARE_bool(count_work);


__device__ __forceinline__ void SignalHostFlag(volatile int *signal_ptr, int value)
{
    {
        __threadfence_system();
        *signal_ptr = value;
    }
}

__device__ __forceinline__ void IncreaseHostFlag(volatile int *signal_ptr, int value)
{
    {
        __threadfence_system();
        *signal_ptr = *signal_ptr + value;
    }
}


namespace groute {
    namespace opt {

        enum Signal
        {
            SIGNAL_EXIT = -1,
            SIGNAL_DATA = 1, // NOTE: One or more
        };

        /*
        Bitmap flags for split kernels
        */
        enum SplitFlags
        {
            SF_None = 0,
            SF_Take = 1 << 0,
            SF_Pass = 1 << 1,
            //SF_HighPrio = 1 << 2,
        };

        /*
        template<typename TUnpacked, typename TPacked>
        struct SplitOps // an example for the required format
        {
        SplitFlags on_receive(const TPacked& data);
        SplitFlags on_send(const TUnpacked& data);

        TPacked pack(const TUnpacked& data);
        TUnpacked unpack(const TPacked& data);
        };
        */

        template<typename TLocal, typename TRemote, typename SplitOps>
        __global__ void SplitSendKernel(
            SplitOps split_ops, TLocal* work_ptr, uint32_t work_size,
            dev::CircularWorklist<TLocal> local_work, dev::CircularWorklist<TRemote> remote_work)
        {
            int tid = TID_1D;
            if (tid < work_size)
            {
                TLocal work = work_ptr[tid];
                SplitFlags flags = split_ops.on_send(work);

                // no filter counter here

                if (flags & SF_Take)
                {
                    local_work.prepend_warp(work); // notice the prepend  
                }

                if (flags & SF_Pass)
                {
                    // pack data
                    TRemote packed = split_ops.pack(work);
                    remote_work.append_warp(packed); // notice the append  
                }
            }
        }

        template<typename TLocal, typename TRemote, typename SplitOps>
        __global__ void SplitReceiveKernel(
            SplitOps split_ops,
            TRemote* work_ptr, uint32_t work_size,
            dev::CircularWorklist<TLocal> local_work,
            dev::CircularWorklist<TRemote> remote_work,
            dev::Counter filter_counter
            )
        {
            int tid = TID_1D;
            if (tid < work_size)
            {
                TRemote work = work_ptr[tid];
                SplitFlags flags = split_ops.on_receive(work);

                int filter_mask = __ballot(flags == SF_None ? 1 : 0);
                int take_mask = __ballot(flags & SF_Take ? 1 : 0);
                int pass_mask = __ballot(flags & SF_Pass ? 1 : 0);
                // never inline the masks into the conditional branching below  
                // although it may work. The compiler should optimize this anyhow, 
                // but this avoids it from unifying the __ballot's 

                if (flags == SF_None)
                {
                    int filter_leader = __ffs(filter_mask) - 1;
                    if (lane_id() == filter_leader)
                        filter_counter.add(__popc(filter_mask));
                }
                else
                {
                    if (flags & SF_Take)
                    {
                        int take_leader = __ffs(take_mask) - 1;
                        int thread_offset = __popc(take_mask & ((1 << lane_id()) - 1));
                        local_work.append_warp(split_ops.unpack(work), take_leader, __popc(take_mask), thread_offset);
                    }

                    if (flags & SF_Pass)
                        // pass on to another endpoint
                    {
                        int pass_leader = __ffs(pass_mask) - 1;
                        int thread_offset = __popc(pass_mask & ((1 << lane_id()) - 1));
                        remote_work.append_warp(work, pass_leader, __popc(pass_mask), thread_offset);
                    }
                }
            }
        }

        /*
        Just a bunch of configuration bitmap flags for the distributed worklist
        */
        enum DistributedWorklistFlags
        {
            DW_NoFlags = 0,
            DW_WarpAppend = 1 << 0,
            DW_DebugPrint = 1 << 1,
            DW_HighPriorityReceive = 1 << 2
        };

        struct IDistributedWorklist
        {
            virtual ~IDistributedWorklist() { }

            virtual void ReportHighPrioWork(int new_work, int performed_work, const char* caller, Endpoint endpoint, bool initial = false) = 0;
            virtual void ReportLowPrioWork(int new_work, int performed_work, const char* caller, Endpoint endpoint) = 0;

            virtual int GetCurrentPrio() = 0;

            virtual bool HasWork() const = 0;

            virtual void ReportPeerTermination() = 0;
            virtual bool HasActivePeers() = 0;
        };

        template<typename TLocal, typename TRemote>
        struct IDistributedWorklistPeer
        {
            virtual ~IDistributedWorklistPeer() { }

            /// The LocalInputWorklist, exposed for customized usage  
            virtual CircularWorklist<TLocal>& GetLocalInputWorklist() = 0;

            /// The RemoteOutputWorklist, exposed for customized usage 
            virtual CircularWorklist<TRemote>& GetRemoteOutputWorklist() = 0;

            virtual void SendWork() = 0;

            /// A blocking call for local work segments 
            virtual std::vector< Segment<TLocal> > GetLocalWork(Stream& stream) = 0;

            virtual std::vector< Segment<TLocal> > WaitForPrioOrWork(int current_prio, Stream& stream) = 0;

            /// A non-blocking call for local work segments
            virtual std::vector< Segment<TLocal> > PeekLocalWork(Stream& stream) = 0;
        };

        template<typename TLocal, typename TRemote, typename SplitOps>
        class DistributedWorklistPeer : public IDistributedWorklistPeer < TLocal, TRemote >
        {
        protected:
            Endpoint m_endpoint, m_ngpus;

        private:

            Context& m_context;
            IDistributedWorklist& m_distributed_worklist;

            size_t m_chunk_size;

            SplitOps m_split_ops;
            DistributedWorklistFlags m_flags;
            Counter m_filter_counter;

            CircularWorklist < TLocal >
                m_receive_worklist; // From split-receive  

            CircularWorklist<TRemote>
                m_send_worklist, // From local work (split-send)
                m_pass_worklist; // From previous endpoint on the ring (split-receive), passing on  

            std::thread m_receive_thread;
            std::thread m_send_thread;

            // Receive sync objects   
            std::mutex m_receive_mutex;
            std::condition_variable m_receive_cv;
            bool m_receive_work = false;
            Event m_receive_work_event;

            // Exit:
            volatile bool m_exit = false;
            volatile int m_current_priority;
            
            Link<TRemote> m_link_in, m_link_out;

            void SplitReceive(
                const Segment<TRemote>& received_work, Stream& stream)
            {
                m_filter_counter.ResetAsync(stream.cuda_stream);

                dim3 block_dims(DBS, 1, 1);
                dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1, 1);

                SplitReceiveKernel <TLocal, TRemote, SplitOps> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_split_ops,
                    received_work.GetSegmentPtr(), received_work.GetSegmentSize(),
                    m_receive_worklist.DeviceObject(),
                    m_pass_worklist.DeviceObject(),
                    m_filter_counter.DeviceObject()
                    );
                
                int filtered_work = (int)m_filter_counter.GetCount(stream);

                if (FLAGS_verbose)
                {
                    int take_counter = m_receive_worklist.GetAllocCountAndSync(stream);
                    int pass_counter = m_pass_worklist.GetAllocCountAndSync(stream);

                    printf("%d - split-rcv, take: %d, filter: %d, pass: %d\n", (Endpoint::identity_type)m_endpoint, take_counter, filtered_work, pass_counter);
                }
                else
                {
                    m_receive_worklist.SyncAppendAllocAsync(stream.cuda_stream);
                    m_pass_worklist.SyncAppendAllocAsync(stream.cuda_stream);
                }

                m_distributed_worklist.ReportHighPrioWork(
                    0,
                    filtered_work, // All sent work is reported as high priority, so we report filtering as high-prio work done 
                    "SplitReceive", m_endpoint
                    );
            }

            void SplitSend(
                const groute::Segment<TLocal>& sent_work, groute::Stream& stream)
            {
                dim3 block_dims(DBS, 1, 1);
                dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

                SplitSendKernel <TLocal, TRemote, SplitOps> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                    m_split_ops,
                    sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
                    m_receive_worklist.DeviceObject(), m_send_worklist.DeviceObject()
                    );

                m_send_worklist.SyncAppendAllocAsync(stream.cuda_stream);

                // Split-send does no filtering, no need to update distributed worklist with work
            }

            struct Pop
            {
                std::shared_future< Event > future;
                size_t size;

                Pop(std::shared_future< Event > future, size_t size) : future(future), size(size) { }
                Pop() : future(), size(0) { }
            };

            void ReceiveLoop()
            {
                m_context.SetDevice(m_endpoint);
                Stream stream = m_context.CreateStream(m_endpoint, (m_flags & DW_HighPriorityReceive) ? SP_High : SP_Default);

                std::deque<Pop> pops; // Queue for managing efficient memory releases  
                auto bounds = m_pass_worklist.GetBounds(stream);

                while (true)
                {
                    auto fut = m_link_in.PipelinedReceive();
                    auto seg = fut.get();
                    if (seg.Empty()) break; 

                    int pop_count = 0;
                    while (!pops.empty() && groute::is_ready(pops.front().future)) // Loop over 'ready' (future, Event) pairs
                    {
                        auto pop = pops.front();
                        auto e = pop.future.get(); // Future is ready, so no blocking here 
                        if (!e.Query()) break; // We avoid waiting on either future or Event, unless we must for space reasons. See while below

                        // Pop only if Event is ready as well. 
                        pop_count += pop.size;
                        pops.pop_front(); 
                    }

                    int space = m_pass_worklist.GetSpace(bounds);
                    int work = seg.GetSegmentSize(); // Maximum 'pass' output for split-receive is input work size 

                    while (pop_count + space < work) // Loop over future + Event until we have space (one iteration should usually give space)
                    {
                        assert(!pops.empty()); // Cannot be empty unless pass worklist is too small

                        auto pop = pops.front();
                        auto e = pop.future.get(); // Block on future 
                        e.Wait(stream); // Wait on event
                        pop_count += pop.size;
                        pops.pop_front(); 

                    }

                    m_pass_worklist.PopItemsAsync(pop_count, stream);

                    // Queue a segment-wait on stream
                    seg.Wait(stream);
                    SplitReceive(seg, stream);

                    auto ev = m_context.RecordEvent(m_endpoint, stream);

                    { // Signal to worker
                        std::lock_guard<std::mutex> guard(m_receive_mutex);
                        m_receive_work = true;
                        m_receive_work_event = ev;
                        m_receive_cv.notify_one();
                    }
                    
                    // Release receive buffer for pipelining  
                    m_link_in.ReleaseReceiveBuffer(seg.GetSegmentPtr(), ev);

                    auto current = m_pass_worklist.GetBounds(stream);
                    auto exclude = current.Exclude(bounds);
                    bounds = current; // Keep for next round  

                    std::vector< Segment<TRemote> > segs = m_pass_worklist.GetSegs(exclude); 

                    for (auto& s : segs)
                    {
                        for (auto& ss : s.Split(m_chunk_size)) // We want good granularity for optimal memory management 
                        {
                            auto event_fut = m_link_out.Send(ss, Event());
                            pops.push_back(Pop(std::move(event_fut), ss.GetSegmentSize()));
                        }
                    }
                }

                // Signal all streaming threads to exit
                std::lock_guard<std::mutex> guard(m_receive_mutex);
                std::lock_guard<std::mutex> guard2(m_pop_mutex);
                m_exit = true;
                m_receive_cv.notify_one();
                m_pop_cv.notify_one(); // Here for now
            }

            // Temp
            Stream m_send_stream;
            typename CircularWorklist<TRemote>::Bounds m_send_bounds;
            std::deque<Pop> m_send_pops;
            
            std::mutex m_pop_mutex;
            std::condition_variable m_pop_cv;

        public:
            void SendWork() override // Temp: Called by a single thread (worker) 
            {
                auto current = m_send_worklist.GetBounds(m_send_stream);
                auto exclude = current.Exclude(m_send_bounds);
                m_send_bounds = current; // Keep for next round  

                std::vector< Segment<TRemote> > segs = m_send_worklist.GetSegs(exclude); 

                for (auto& s : segs)
                {
                    for (auto& ss : s.Split(m_chunk_size)) // We want good granularity for optimal memory management 
                    {
                        auto event_fut = m_link_out.Send(ss, Event());

                        std::lock_guard<std::mutex> guard(m_pop_mutex);
                        m_send_pops.push_back(Pop(std::move(event_fut), ss.GetSegmentSize()));
                        m_pop_cv.notify_one();
                    }
                }
            }

        private:
            void PopLoop() // Pops items ASAP from circular 'send' queue
            {
                m_context.SetDevice(m_endpoint);
                Stream stream = m_context.CreateStream(m_endpoint);

                while (true)
                {
                    Pop pop;

                    { // Lock block
                        std::unique_lock<std::mutex> guard(m_pop_mutex);

                        if (m_send_pops.empty()) {

                            if (m_exit) break;

                            // Waiting for (work|exit)
                            m_pop_cv.wait(guard, [this]() {
                                return m_exit || !m_send_pops.empty(); });
                            
                            if (m_exit) break;
                        }

                        pop = m_send_pops.front();
                        m_send_pops.pop_front(); 
                    }
                    
                    auto e = pop.future.get(); 
                    e.Wait(stream); 
                    m_send_worklist.PopItemsAsync(pop.size, stream);
                }
            }

        public:
            DistributedWorklistPeer(
                Context& context, Router<TRemote>& router,
                IDistributedWorklist& distributed_worklist, int current_priority, const SplitOps& split_ops, DistributedWorklistFlags flags,
                Endpoint endpoint, int ngpus, size_t max_work_size, size_t chunk_size, size_t num_buffers)
                :
                m_context(context), m_endpoint(endpoint), m_ngpus(ngpus), m_distributed_worklist(distributed_worklist),
                m_chunk_size(chunk_size), m_current_priority(current_priority), m_split_ops(split_ops), m_flags(flags),
                m_link_in(router, endpoint, chunk_size, num_buffers), 
                m_link_out(endpoint, router)
            {
                void* mem_buffer;
                size_t mem_size;

                mem_buffer = context.Alloc(0.3, mem_size, AF_PO2);
                m_receive_worklist = groute::CircularWorklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal));

                mem_buffer = context.Alloc(0.15, mem_size, AF_PO2);
                m_send_worklist = groute::CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote));

                mem_buffer = context.Alloc(0.15, mem_size, AF_PO2);
                m_pass_worklist = groute::CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote)); // TODO: should be relative to chunk_size and num_buffers

                m_receive_worklist.ResetAsync((cudaStream_t)0);

                m_send_worklist.ResetAsync((cudaStream_t)0);
                m_pass_worklist.ResetAsync((cudaStream_t)0);

                GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0)); // just in case

                // Temp
                m_send_stream = m_context.CreateStream(m_endpoint);
                m_send_bounds = m_pass_worklist.GetBounds(m_send_stream);

                m_receive_thread = std::thread([this]() { ReceiveLoop(); });
                m_send_thread = std::thread([this]() { PopLoop(); });
            }

            ~DistributedWorklistPeer()
            {
                m_receive_thread.join();
                m_send_thread.join();
            }

            CircularWorklist<TLocal>& GetLocalInputWorklist() override { return m_receive_worklist; }

            CircularWorklist<TRemote>& GetRemoteOutputWorklist() override { return m_send_worklist; }

            std::vector< Segment<TLocal> > GetLocalWork(Stream& stream) override
            {
                auto segs = m_receive_worklist.ToSegs(stream);

                while (segs.empty())
                {
                    Event work_ev;

                    {
                        std::unique_lock<std::mutex> guard(m_receive_mutex);

                        while (true)
                        {
                            if (m_exit) break;

                            if (m_receive_work)
                            {
                                m_receive_work = false;
                                work_ev = std::move(m_receive_work_event);
                                break;
                            }

                            m_receive_cv.wait(guard);
                        }
                    }

                    if (m_exit) return segs;

                    work_ev.Wait(stream.cuda_stream);
                    segs = m_receive_worklist.ToSegs(stream);
                }

                return segs;
            }

            std::vector< Segment<TLocal> > WaitForPrioOrWork(int current_prio, Stream& stream) override
            {
                auto segs = m_receive_worklist.ToSegs(stream);

                while (segs.empty())
                {
                    Event work_ev;

                    {
                        std::unique_lock<std::mutex> guard(m_receive_mutex);

                        while (true)
                        {
                            if (current_prio < m_current_priority) break;
                            if (m_exit) break;

                            if (m_receive_work)
                            {
                                m_receive_work = false;
                                work_ev = std::move(m_receive_work_event);
                                break;
                            }

                            m_receive_cv.wait(guard);
                        }
                    }

                    if (current_prio < m_current_priority) return segs;
                    if (m_exit) return segs;

                    work_ev.Wait(stream.cuda_stream);
                    segs = m_receive_worklist.ToSegs(stream);
                }

                return segs;
            }

            std::vector< Segment<TLocal> > PeekLocalWork(Stream& stream) override
            {
                return m_receive_worklist.ToSegs(stream);
            }

            void AdvanceLowPrio(int current_priority)
            {
                std::lock_guard<std::mutex> guard(m_receive_mutex);
                m_current_priority = current_priority;

                m_receive_work = true;
                m_receive_work_event = Event();
                m_receive_cv.notify_one();
            }
        };

        template<typename TLocal, typename TRemote, typename SplitOps>
        class DistributedWorklist : public IDistributedWorklist
        {
        private:
            typedef DistributedWorklistPeer<TLocal, TRemote, SplitOps> PeerType;

            Context& m_context;
            Router<TRemote>& m_router;
            int m_ngpus;

            std::vector< std::shared_ptr<PeerType> > m_peers;

            std::atomic<int> m_active_peers_counter;

            std::atomic<int> m_high_prio_work;
            std::atomic<int> m_low_prio_work;

            const int m_priority_delta;
            volatile int m_current_priority;

            // Work-item counter
            std::atomic<unsigned int> m_reported_work;
            std::vector<unsigned int> m_ctr;
        public:
            unsigned int GetCurrentWorkCount(Endpoint endpoint)
            {
                return m_ctr[(Endpoint::identity_type)endpoint + 1];
            }

        public:
            std::mutex log_gate;

        public:

            DistributedWorklist(Context& context, Router<TRemote>& router, int ngpus, int priority_delta) :
                m_context(context), m_router(router), m_ngpus(ngpus), m_active_peers_counter(ngpus),
                m_high_prio_work(0), m_low_prio_work(0), m_priority_delta(priority_delta), m_current_priority(priority_delta), m_reported_work(0)
            {
                if (FLAGS_count_work)
                {
                    m_ctr.resize(ngpus + 1, 0);
                }
            }

            virtual ~DistributedWorklist()
            {
                if (FLAGS_count_work && FLAGS_verbose)
                {
                    printf("Work performed by each GPU:\n");
                    for (size_t i = 1; i < m_ctr.size(); ++i)
                        printf("  GPU %llu: %d witems\n", i, m_ctr[i]);
                    int repwork = m_reported_work;
                    printf("Total witems: %d\n", repwork);
                }
            }

            std::shared_ptr< IDistributedWorklistPeer<TLocal, TRemote> > CreatePeer(
                Endpoint endpoint, const SplitOps& split_ops,
                size_t max_work_size, size_t max_exch_size, size_t exch_buffs, DistributedWorklistFlags flags = (DistributedWorklistFlags)(DW_WarpAppend | DW_HighPriorityReceive))
            {
                m_context.SetDevice(endpoint);
                auto peer = std::make_shared< PeerType >(
                    m_context, m_router, *this, (int)m_current_priority, split_ops, flags, endpoint, m_ngpus, max_work_size, max_exch_size, exch_buffs);
                m_peers.push_back(peer);
                return peer;
            }

            void ReportPeerTermination() override // currently unused 
            {
                if (--m_active_peers_counter == 0)
                {
                    m_router.Shutdown();
                }
            }

            void ReportHighPrioWork(int new_work, int performed_work, const char* caller, Endpoint endpoint, bool initialwork = false) override
            {
                int work = new_work - performed_work;

                if (FLAGS_count_work)
                {
                    m_reported_work += performed_work;
                    m_ctr[(Endpoint::identity_type)endpoint + 1] += performed_work;
                }

                if (work == 0) return;

                if (!initialwork && (m_high_prio_work + m_low_prio_work) == 0) {
                    printf("Warning: seems like a BUG in the distributed worklist\n");
                    return;
                }

                int high_prio_work = (m_high_prio_work += work);

                if (high_prio_work == 0)
                {
                    if (m_low_prio_work == 0)
                    {
                        if(FLAGS_verbose) printf("Distributed Worklist Shutting Down: %s\n", caller);
                        m_router.Shutdown();
                    }

                    else
                    {
                        m_high_prio_work = (int)m_low_prio_work;
                        m_low_prio_work = 0;

                        m_current_priority += m_priority_delta;
                        for (auto& peer : m_peers)
                        {
                            peer->AdvanceLowPrio(m_current_priority);
                        }
                    }
                }
            }

            void ReportLowPrioWork(int new_work, int performed_work, const char* caller, Endpoint endpoint) override
            {
                int work = new_work - performed_work;

                if (FLAGS_count_work)
                {
                    m_reported_work += performed_work;
                    m_ctr[(Endpoint::identity_type)endpoint + 1] += performed_work;
                }

                if (work == 0) return;

                if ((m_high_prio_work + m_low_prio_work) == 0) {
                    printf("Warning: seems like a BUG in the distributed worklist\n");
                    return;
                }


                int low_prio_work = (m_low_prio_work += work);

                //if (low_prio_work == 0)
                //{
                //
                //}
            }

            int GetCurrentPrio() override
            {
                return m_current_priority;
            }

            bool HasWork() const override
            {
                return (m_high_prio_work + m_low_prio_work) > 0;
            }

            bool HasActivePeers() override
            {
                return m_active_peers_counter > 0;
            }
        };
    }
}

#endif // __GROUTE_FUSED_DISTRIBUTED_WORKLIST_H
