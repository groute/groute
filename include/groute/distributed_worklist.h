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

#ifndef __GROUTE_DISTRIBUTED_WORKLIST_H
#define __GROUTE_DISTRIBUTED_WORKLIST_H

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

DECLARE_double(wl_alloc_factor_local);
DECLARE_double(wl_alloc_factor_in);
DECLARE_double(wl_alloc_factor_out);
DECLARE_double(wl_alloc_factor_pass);

namespace groute {

    /*
    Bitmap flags for split kernels
    */
    enum SplitFlags
    {
        SF_None = 0,
        SF_Take = 1 << 0,
        SF_Pass = 1 << 1,
    };

    /*
    template<typename TUnpacked, typename TPacked>
    struct DWCallbacks // an example for the required format
    {
        SplitFlags on_receive(const TPacked& data);
        SplitFlags on_send(const TUnpacked& data);

        TPacked pack(const TUnpacked& data);
        TUnpacked unpack(const TPacked& data);

        bool should_defer(TUnpacked work, TPrio global_threshold)
    };
    */

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    __global__ void SplitSendKernel(
        DWCallbacks callbacks, TLocal* work_ptr, uint32_t work_size,
        dev::CircularWorklist<TLocal> local_work, dev::CircularWorklist<TRemote> remote_work)
    {
        int tid = TID_1D;
        if (tid < work_size)
        {
            TLocal work = work_ptr[tid];
            SplitFlags flags = callbacks.on_send(work);

            // no filter counter here

            if (flags & SF_Take)
            {
                local_work.prepend_warp(work); // notice the prepend  
            }

            if (flags & SF_Pass)
            {
                // pack data
                TRemote packed = callbacks.pack(work);
                remote_work.append_warp(packed); // notice the append  
            }
        }
    }

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    __global__ void SplitReceiveKernel(
        DWCallbacks callbacks,
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
            SplitFlags flags = callbacks.on_receive(work);

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
                    local_work.append_warp(callbacks.unpack(work), take_leader, __popc(take_mask), thread_offset);
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

    struct IDistributedWorklist
    {
        virtual ~IDistributedWorklist() { }

        virtual void ReportWork(int new_work, int performed_work, const char* caller, Endpoint endpoint, bool initial = false) = 0;
        virtual void ReportDeferredWork(int new_work, int performed_work, const char* caller, Endpoint endpoint) = 0;

        virtual int GetPriorityThreshold() = 0;
        virtual bool HasWork() const = 0;
        
        std::mutex log_gate;
    };

    template<typename TLocal, typename TRemote>
    struct IDistributedWorklistPeer
    {
        virtual ~IDistributedWorklistPeer() { }

        /// Get a local workspace 
        virtual Worklist<TLocal>& GetLocalWorkspace(int i) = 0;

        /// The LocalInputWorklist, acts as a device-level 'link' for local input   
        virtual CircularWorklist<TLocal>& GetLocalInputWorklist() = 0;

        /// The RemoteOutputWorklist, acts as a device-level 'link' for remote output 
        virtual CircularWorklist<TRemote>& GetRemoteOutputWorklist() = 0;

        /// Wait for local work or for priority threshold change
        virtual std::vector< Segment<TLocal> > WaitForLocalWork(Stream& stream, int priority_threshold = 0) = 0;

        /// Signal that work was pushed into the RemoteOutputWorklist
        virtual void SignalRemoteWork(const Event& ev) = 0;

        /// Perform split-send, local work will be prepended into the LocalInputWorklist
        /// and remote work will be appended into the RemoteOutputWorklist (+SignalRemoteWork)  
        virtual void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) = 0;
    };

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    class DistributedWorklist;

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    class DistributedWorklistPeer : public IDistributedWorklistPeer < TLocal, TRemote >
    {
        friend class DistributedWorklist < TLocal, TRemote, DWCallbacks > ;

        Context& m_context;
        IDistributedWorklist& m_distributed_worklist;

        Endpoint m_endpoint;
        size_t m_chunk_size, m_num_buffers, m_num_workspaces;
        DWCallbacks m_device_callbacks;

        Counter m_filter_counter; // split-receive filter counter
        CircularWorklist<TLocal> m_receive_worklist; // From split-receive  
        CircularWorklist<TRemote>   m_send_worklist, // From local work (split-send)
                                    m_pass_worklist; // From previous endpoint on the ring (split-receive), passing on  

        std::vector<Worklist<TLocal>> m_local_workspaces; // Local workspaces for efficient work scheduling   

        std::thread m_receive_thread;
        std::thread m_pop_thread;

        // Internal class for managing circular-queue pops
        struct Pop
        {
            std::shared_future< Event > future;
            size_t size;

            Pop(std::shared_future< Event > future, size_t size) : future(future), size(size) { }
            Pop() : future(), size(0) { }
        };

        // Receive sync objects   
        std::mutex m_receive_mutex;
        std::condition_variable m_receive_cv;
        bool m_receive_work = false;
        Event m_receive_work_event;

        // Send-Pop sync objects
        Stream m_send_stream;
        typename CircularWorklist<TRemote>::Bounds m_send_bounds;
        std::deque<Pop> m_send_pops;

        std::mutex m_pop_mutex;
        std::condition_variable m_pop_cv;

        // Exit:
        volatile bool m_exit = false;
        volatile int m_current_threshold;

        Link<TRemote> m_link_in, m_link_out;

        void SplitReceive(
            const Segment<TRemote>& received_work, Stream& stream)
        {
            m_filter_counter.ResetAsync(stream.cuda_stream);

            dim3 block_dims(DBS, 1, 1);
            dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1, 1);

            SplitReceiveKernel <TLocal, TRemote, DWCallbacks> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_device_callbacks,
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

            m_distributed_worklist.ReportWork(
                0,
                filtered_work, // All sent work is reported as high priority, so we report filtering as high-prio work done 
                "SplitReceive", m_endpoint
                );
        }

        void SplitSend(
            const Segment<TLocal>& sent_work, Stream& stream)
        {
            dim3 block_dims(DBS, 1, 1);
            dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

            SplitSendKernel <TLocal, TRemote, DWCallbacks> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_device_callbacks,
                sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
                m_receive_worklist.DeviceObject(), m_send_worklist.DeviceObject()
                );

            m_send_worklist.SyncAppendAllocAsync(stream.cuda_stream);

            // Split-send does no filtering, no need to update distributed worklist with work
        }

        void ReceiveLoop()
        {
            m_context.SetDevice(m_endpoint);
            Stream stream = m_context.CreateStream(m_endpoint, SP_High);

            std::deque<Pop> pops; // Queue for managing efficient memory releases  
            auto bounds = m_pass_worklist.GetBounds(stream);

            while (true)
            {
                auto fut = m_link_in.PipelinedReceive();
                auto seg = fut.get();
                if (seg.Empty()) break;

                //{
                //    std::lock_guard<std::mutex> guard(m_distributed_worklist.log_gate);
                //    printf("Receive (%d) | got: %llu\n", (Endpoint::identity_type)m_endpoint, seg.GetSegmentSize());
                //}

                size_t pop_count = 0;
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
                
                //if (pop_count > 0)
                //{
                //    std::lock_guard<std::mutex> guard(m_distributed_worklist.log_gate);
                //    printf("Receive (%d) | pop: %llu\n", (Endpoint::identity_type)m_endpoint, pop_count);
                //}
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

                //{
                //    std::lock_guard<std::mutex> guard(m_distributed_worklist.log_gate);
                //    printf("Send (%d) | pop: %llu\n", (Endpoint::identity_type)m_endpoint, pop.size);
                //}
            }
        }
        
        void InitWorklists()
        {
            void* mem_buffer;
            size_t mem_size;

            mem_buffer = m_context.Alloc(FLAGS_wl_alloc_factor_in, mem_size, AF_PO2);
            m_receive_worklist = CircularWorklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal), m_endpoint, "receive");

            mem_buffer = m_context.Alloc(FLAGS_wl_alloc_factor_out, mem_size, AF_PO2);
            m_send_worklist = CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote), m_endpoint, "send");

            mem_buffer = m_context.Alloc(FLAGS_wl_alloc_factor_pass, mem_size, AF_PO2);
            m_pass_worklist = CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote), m_endpoint, "pass"); // TODO: should be relative to chunk_size and num_buffers

            for (size_t i = 0; i < m_num_workspaces; i++)
            {
                mem_buffer = m_context.Alloc(FLAGS_wl_alloc_factor_local / m_num_workspaces, mem_size);
                m_local_workspaces.push_back(Worklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal)));
            }

            m_receive_worklist.ResetAsync((cudaStream_t)0);
            m_send_worklist.ResetAsync((cudaStream_t)0);
            m_pass_worklist.ResetAsync((cudaStream_t)0);

            for (size_t i = 0; i < m_num_workspaces; i++)
            {
                m_local_workspaces[i].ResetAsync((cudaStream_t)0);
            }

            GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0)); // just in case

            m_send_stream = m_context.CreateStream(m_endpoint);
            m_send_bounds = m_pass_worklist.GetBounds(m_send_stream);

            m_receive_thread = std::thread([this]() { ReceiveLoop(); });
            m_pop_thread = std::thread([this]() { PopLoop(); });
        }
    public:
        DistributedWorklistPeer(
            Context& context, Router<TRemote>& router, IDistributedWorklist& distributed_worklist,
            int priority_threshold, const DWCallbacks& device_callbacks,
            Endpoint endpoint, size_t chunk_size, size_t num_buffers, size_t num_workspaces)
            :
            m_context(context), m_endpoint(endpoint), m_distributed_worklist(distributed_worklist),
            m_chunk_size(chunk_size), m_num_buffers(num_buffers), m_num_workspaces(num_workspaces),
            m_current_threshold(priority_threshold), m_device_callbacks(device_callbacks),
            m_link_in(router, endpoint, chunk_size, num_buffers),
            m_link_out(endpoint, router)
        {
        }

        ~DistributedWorklistPeer()
        {
            m_receive_thread.join();
            m_pop_thread.join();
        }

        Worklist<TLocal>& GetLocalWorkspace(int i) override { return m_local_workspaces[i]; }

        CircularWorklist<TLocal>& GetLocalInputWorklist() override { return m_receive_worklist; }

        CircularWorklist<TRemote>& GetRemoteOutputWorklist() override { return m_send_worklist; }

        std::vector< Segment<TLocal> > WaitForLocalWork(Stream& stream, int priority_threshold = 0) override
        {
            auto segs = m_receive_worklist.ToSegs(stream);

            while (segs.empty())
            {
                Event work_ev;

                {
                    std::unique_lock<std::mutex> guard(m_receive_mutex);

                    while (true)
                    {
                        if (priority_threshold > 0 && priority_threshold < m_current_threshold) break;
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

                if (priority_threshold > 0 && priority_threshold < m_current_threshold) return segs;
                if (m_exit) return segs;

                work_ev.Wait(stream.cuda_stream);
                segs = m_receive_worklist.ToSegs(stream);
            }

            return segs;
        }

        void SignalRemoteWork(const Event& ev) override
        {
            // This method is called by a single thread (worker) 

            ev.Wait(m_send_stream);

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
        
        void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) override
        {
            if (split_work.Empty()) return;

            SplitSend(split_work, stream);
            SignalRemoteWork(m_context.RecordEvent(m_endpoint, stream));
        }

        void AdvancePriorityThreshold(int priority_threshold)
        {
            std::lock_guard<std::mutex> guard(m_receive_mutex);
            m_current_threshold = priority_threshold;

            m_receive_work = true;
            m_receive_work_event = Event();
            m_receive_cv.notify_one();
        }
    };

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    class DistributedWorklist : public IDistributedWorklist
    {
    private:
        typedef DistributedWorklistPeer<TLocal, TRemote, DWCallbacks> PeerType;

        Context& m_context;
        Router<TRemote> m_router;

        EndpointList m_workers; // Worker endpoints  
        std::map<Endpoint, std::unique_ptr<PeerType> > m_peers; // DW peer per worker
        std::map<Endpoint, Link<TRemote> > m_links; // Link per source

        std::atomic<int> m_current_work_counter;
        std::atomic<int> m_deferred_work_counter;

        const int m_priority_delta;
        volatile int m_current_threshold;

        // Work-item counter
        std::atomic<uint32_t> m_total_work;
        std::map<Endpoint, std::atomic<uint32_t>> m_endpoint_work;

        // TODO: std::thread m_watchdog; 
    public:

        DistributedWorklist(
            Context& context, const EndpointList& sources, const EndpointList& workers, const std::map<Endpoint, DWCallbacks>& callbacks, 
            size_t chunk_size, size_t num_buffers, int priority_delta = 0) :
            m_context(context), m_router(context, Policy::CreateRingPolicy(workers), (int)(sources.size() + workers.size()), (int)workers.size()), 
            m_workers(workers), m_current_work_counter(0), m_deferred_work_counter(0), m_priority_delta(priority_delta), 
            m_current_threshold(priority_delta == 0 ? INT32_MAX : priority_delta), m_total_work(0)
        {
            if (workers.size() != callbacks.size()) throw std::exception("DW parameter mismatch (workers <-> callbacks)");

            if (FLAGS_verbose)
            {
                printf(
                    "DW configuration: chunk: %llu, buffers: %llu, priority -> [delta: %d, initial threshold: %d]\n", 
                    chunk_size, num_buffers, priority_delta, m_current_threshold);
            }

            for (Endpoint source : sources)
            {
                m_endpoint_work[source] = 0; // Also sources report work
                m_links[source] = Link<TRemote>(source, m_router);
            }

            for (Endpoint worker : m_workers)
            {
                m_endpoint_work[worker] = 0;
                m_context.SetDevice(worker);

                if (callbacks.find(worker) == callbacks.end()) throw std::exception("DW: missing DWCallbacks for worker");

                auto peer = groute::make_unique< PeerType >(
                    m_context, m_router, *this, (int)m_current_threshold, callbacks.at(worker), worker, chunk_size, num_buffers, priority_delta == 0 ? 1 : 2);
                m_peers[worker] = std::move(peer);
            }

            // Second phase: for allocating available memory for local work-queues after link allocation   
            for (Endpoint worker : m_workers)
            {
                m_peers[worker]->InitWorklists();
            }
        }

        virtual ~DistributedWorklist()
        {
            if (FLAGS_count_work && FLAGS_verbose)
            {
                printf("Work performed by each GPU:\n");
                for (auto& p : m_endpoint_work)
                    printf("  GPU %d: %d work-items\n", (Endpoint::identity_type)p.first, (uint32_t)p.second);
                int repwork = m_total_work;
                printf("Total work-items: %d\n", repwork);
            }
        }

        Link<TRemote>& GetLink(Endpoint source)
        {
            if (m_links.find(source) == m_links.end()) throw std::exception("Endpoint not registered as a source");
            return m_links[source];
        }

        IDistributedWorklistPeer<TLocal, TRemote>* GetPeer(Endpoint worker)
        {
            if (m_peers.find(worker) == m_peers.end()) throw std::exception("Endpoint not registered as a worker");
            return m_peers[worker].get();
        }

        uint32_t GetCurrentWorkCount(Endpoint endpoint)
        {
            return m_endpoint_work[endpoint];
        }

        void ReportWork(int new_work, int performed_work, const char* caller, Endpoint endpoint, bool initial = false) override
        {
            int work = new_work - performed_work;

            if (FLAGS_count_work)
            {
                uint32_t w = (new_work < 0 ? -new_work : 0) + (performed_work > 0 ? performed_work : 0);
                m_total_work += w;
                m_endpoint_work[endpoint] += w;
            }

            if (work == 0) return;

            if (!initial && (m_current_work_counter + m_deferred_work_counter) == 0) {
                printf("Warning: work is reported after reaching zero work-items\n");
                return;
            }

            int high_prio_work = (m_current_work_counter += work);

            if (high_prio_work == 0)
            {
                if (m_deferred_work_counter == 0)
                {
                    if (FLAGS_verbose) printf("Distributed Worklist Shutting Down successfully: %s\n", caller);
                    m_router.Shutdown();
                }

                else
                {
                    m_current_work_counter = (int)m_deferred_work_counter;
                    m_deferred_work_counter = 0;

                    m_current_threshold += m_priority_delta;
                    for (auto& p : m_peers)
                    {
                        p.second->AdvancePriorityThreshold(m_current_threshold);
                    }
                }
            }
        }

        void ReportDeferredWork(int new_work, int performed_work, const char* caller, Endpoint endpoint) override
        {
            int work = new_work - performed_work;

            if (FLAGS_count_work)
            {
                uint32_t w = (new_work < 0 ? -new_work : 0) + (performed_work > 0 ? performed_work : 0);
                m_total_work += w;
                m_endpoint_work[endpoint] += w;
            }

            if (work == 0) return;

            if ((m_current_work_counter + m_deferred_work_counter) == 0) {
                printf("Warning: work is reported after reaching zero work-items\n");
                return;
            }

            m_deferred_work_counter += work;
        }

        int GetPriorityThreshold() override
        {
            return m_current_threshold;
        }

        bool HasWork() const override
        {
            return (m_current_work_counter + m_deferred_work_counter) > 0;
        }
    };
}

#endif // __GROUTE_DISTRIBUTED_WORKLIST_H
