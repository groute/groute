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

#include <groute/groute.h>

#include <groute/device/queue.cuh>
#include <groute/device/counter.cuh>

#include <groute/dwl/split_kernels.cuh>

namespace groute {

    struct DistributedWorklistConfiguration
    {
        bool count_work;
        double 
            alloc_factor_in, alloc_factor_out, 
            alloc_factor_pass, alloc_factor_local;
        int fused_chunk_size;

        DistributedWorklistConfiguration() : // Default configuration  
            count_work(false), 
            alloc_factor_in(0.4), alloc_factor_out(0.2), alloc_factor_pass(0.2), alloc_factor_local(0.2), fused_chunk_size(std::numeric_limits<int>::max())
        { }
    };

    template<typename TLocal, typename TRemote>
    struct IDistributedWorklist
    {
        IDistributedWorklist(DistributedWorklistConfiguration configuration = DistributedWorklistConfiguration()) : configuration(configuration) { }
        virtual ~IDistributedWorklist() { }

        virtual void ReportInitialWork(int initial_work, Endpoint endpoint, const char* caller = "") = 0;
        virtual void ReportWork(int new_work, int performed_work, Endpoint endpoint, const char* caller = "") = 0;
        virtual void ReportDeferredWork(int new_work, int performed_work, Endpoint endpoint, const char* caller = "") = 0;

        virtual uint32_t GetCurrentWorkCount(Endpoint endpoint) = 0;
        virtual int GetPriorityThreshold() = 0;
        virtual bool HasWork() const = 0;
        
        virtual Link<TRemote>& GetLink(Endpoint source) = 0;

        DistributedWorklistConfiguration configuration;
        std::mutex log_gate; // Using this may effect performance  
    };

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    struct IDistributedWorklistPeer
    {
        virtual ~IDistributedWorklistPeer() { }

        /// Get a queue for local work 
        virtual Queue<TLocal>& GetLocalQueue(int i) = 0;

        // Get device callbacks (see split_kernels.cuh for contract) 
        virtual const DWCallbacks& GetDeviceCallbacks() = 0;

        /// The RemoteInputQueue, acts as a device-level 'link' for remote input   
        virtual PCQueue<TLocal>& GetRemoteInputQueue() = 0;

        /// The RemoteOutputQueue, acts as a device-level 'link' for remote output 
        virtual PCQueue<TRemote>& GetRemoteOutputQueue() = 0;

        /// Wait for input work or for priority threshold change
        virtual std::vector< Segment<TLocal> > WaitForInputWork(Stream& stream, int priority_threshold = 0) = 0;

        /// Send work from RemoteOutputQueue to router
        virtual void SendRemoteWork(const Event& ev) = 0;

        /// Perform split-send, local work will be prepended into the RemoteInputQueue
        /// and remote work will be appended into the RemoteOutputQueue (+SendRemoteWork)  
        virtual void SplitSend(Segment<TLocal>& split_work, Stream& stream) = 0;
    };

    template<typename TLocal, typename TRemote, typename DWCallbacks>
    class DistributedWorklistPeer : public IDistributedWorklistPeer < TLocal, TRemote, DWCallbacks>
    {
        template<typename TL, typename TR, typename DWC, typename TW>
        friend class DistributedWorklist;

        Context& m_context;
        IDistributedWorklist < TLocal, TRemote >& m_distributed_worklist;

        Endpoint m_endpoint;
        size_t m_chunk_size; // Chunk size used for router links
        DWCallbacks m_device_callbacks;

        Counter m_filter_counter; // split-receive filter counter
        PCQueue<TLocal>     m_receive_queue; // From split-receive  
        PCQueue<TRemote>    m_send_queue, // From local work (split-send)
                            m_pass_queue; // From previous endpoint on the ring (split-receive), passing on  

        std::vector<Queue<TLocal>> m_local_queues; // Local queues used by workers   

        std::thread m_receive_thread;
        std::thread m_pop_thread;

        // Internal class for managing producer-consumer-queue pops
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
        typename PCQueue<TRemote>::Bounds m_send_bounds;
        std::deque<Pop> m_send_pops;

        std::mutex m_pop_mutex;
        std::condition_variable m_pop_cv;

        // Exit:
        volatile bool m_exit = false;
        volatile int m_current_threshold;

        Link<TRemote> m_link_in, m_link_out;

        void InvokeSplitReceive(
            const Segment<TRemote>& received_work, Stream& stream)
        {
            m_filter_counter.ResetAsync(stream.cuda_stream);

            dim3 block_dims(GROUTE_BLOCK_THREADS, 1, 1);
            dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1, 1);

            SplitReceiveKernel <TLocal, TRemote, DWCallbacks> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_device_callbacks,
                received_work.GetSegmentPtr(), received_work.GetSegmentSize(),
                m_receive_queue.DeviceObject(),
                m_pass_queue.DeviceObject(),
                m_filter_counter.DeviceObject()
                );

            int filtered_work = (int)m_filter_counter.GetCount(stream);

            if (m_context.configuration.trace)
            {
                uint32_t take_counter = m_receive_queue.GetPendingCount(stream);
                uint32_t pass_counter = m_pass_queue.GetPendingCount(stream);

                printf("%d - split-rcv, take: %d, filter: %d, pass: %d\n", (Endpoint::identity_type)m_endpoint, take_counter, filtered_work, pass_counter);
            }

            m_receive_queue.CommitPendingAsync(stream);
            m_pass_queue.CommitPendingAsync(stream);

            m_distributed_worklist.ReportWork(
                0,
                filtered_work, // All sent work is reported as high priority, so we report filtering as high-prio work done 
                m_endpoint, 
                "SplitReceive" 
                );
        }

        void InvokeSplitSend(
            const Segment<TLocal>& sent_work, Stream& stream)
        {
            dim3 block_dims(GROUTE_BLOCK_THREADS, 1, 1);
            dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

            SplitSendKernel <TLocal, TRemote, DWCallbacks> << < grid_dims, block_dims, 0, stream.cuda_stream >> >(
                m_device_callbacks,
                sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
                m_receive_queue.DeviceObject(), m_send_queue.DeviceObject()
                );

            m_send_queue.CommitPendingAsync(stream);

            // Split-send does no filtering, no need to update distributed worklist with work
        }

        void ReceiveLoop()
        {
            m_context.SetDevice(m_endpoint);
            Stream stream = m_context.CreateStream(m_endpoint, SP_High);

            std::deque<Pop> pops; // Queue for managing efficient memory releases  
            auto bounds = m_pass_queue.GetBounds(stream);

            while (true)
            {
                auto fut = m_link_in.PipelinedReceive();
                auto seg = fut.get();
                if (seg.Empty()) break;

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

                size_t space = m_pass_queue.GetSpace(bounds);
                size_t work = seg.GetSegmentSize(); // Maximum 'pass' output for split-receive is input work size 

                while (pop_count + space < work) // Loop over future + Event until we have space (one iteration should usually give space)
                {
                    assert(!pops.empty()); // Cannot be empty unless pass worklist is too small

                    auto pop = pops.front();
                    auto e = pop.future.get(); // Block on future 
                    e.Wait(stream); // Wait on event
                    pop_count += pop.size;
                    pops.pop_front();

                }
                
                m_pass_queue.PopAsync(pop_count, stream);

                // Queue a segment-wait on stream
                seg.Wait(stream);
                InvokeSplitReceive(seg, stream);

                auto ev = m_context.RecordEvent(m_endpoint, stream);

                { // Signal to worker
                    std::lock_guard<std::mutex> guard(m_receive_mutex);
                    m_receive_work = true;
                    m_receive_work_event = ev;
                    m_receive_cv.notify_one();
                }

                // Release receive buffer for pipelining  
                m_link_in.ReleaseReceiveBuffer(seg.GetSegmentPtr(), ev);

                auto current = m_pass_queue.GetBounds(stream);
                auto exclude = current.Exclude(bounds);
                bounds = current; // Keep for next round  

                std::vector< Segment<TRemote> > segs = m_pass_queue.GetSegs(exclude);

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
            Stream stream = m_context.CreateStream(m_endpoint, SP_High);

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
                m_send_queue.PopAsync(pop.size, stream);
            }
        }

        //
        // Internal peer API, used by DistributedWorklist (friend class)
        //
        
        void AllocateLinks(Router<TRemote>& router, size_t num_buffers)
        {
            m_link_in = groute::Link<TRemote>(router, m_endpoint, m_chunk_size, num_buffers);
            m_link_out = groute::Link<TRemote>(m_endpoint, router);
        }

        void AllocateQueues(size_t num_local_queues)
        {
            std::vector<double> po2_factors = {
                m_distributed_worklist.configuration.alloc_factor_in,
                m_distributed_worklist.configuration.alloc_factor_out,
                m_distributed_worklist.configuration.alloc_factor_pass };

            std::vector<double> non_po2_factors;
            for (size_t i = 0; i < num_local_queues; i++) 
                non_po2_factors.push_back(m_distributed_worklist.configuration.alloc_factor_local / num_local_queues);

            std::vector<Memory> allocations
                = m_context.Alloc(m_endpoint, std::max(sizeof(TLocal), sizeof(TRemote)), po2_factors, non_po2_factors);

            assert(allocations.size() == po2_factors.size() + non_po2_factors.size());

            m_receive_queue = PCQueue<TLocal>((TLocal*)allocations[0].ptr, allocations[0].size / sizeof(TLocal), m_endpoint, "in");
            m_send_queue = PCQueue<TRemote>((TRemote*)allocations[1].ptr, allocations[1].size / sizeof(TRemote), m_endpoint, "out");
            m_pass_queue = PCQueue<TRemote>((TRemote*)allocations[2].ptr, allocations[2].size / sizeof(TRemote), m_endpoint, "pass"); 

            for (size_t i = 0; i < num_local_queues; i++)
            {
                m_local_queues.push_back(
                    Queue<TLocal>((TLocal*)allocations[3 + i].ptr, allocations[3 + i].size / sizeof(TLocal), m_endpoint, "local"));
            }

            m_receive_queue.ResetAsync((cudaStream_t)0);
            m_send_queue.ResetAsync((cudaStream_t)0);
            m_pass_queue.ResetAsync((cudaStream_t)0);

            for (size_t i = 0; i < num_local_queues; i++)
            {
                m_local_queues[i].ResetAsync((cudaStream_t)0);
            }

            GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)0)); // Just in case
        }

        void Run()
        {
            m_send_stream = m_context.CreateStream(m_endpoint);
            m_send_bounds = m_pass_queue.GetBounds(m_send_stream);

            m_receive_thread = std::thread([this]() { ReceiveLoop(); });
            m_pop_thread = std::thread([this]() { PopLoop(); });
        }
        
        void AdvancePriorityThreshold(int priority_threshold)
        {
            std::lock_guard<std::mutex> guard(m_receive_mutex);
            m_current_threshold = priority_threshold;

            m_receive_work = true;
            m_receive_work_event = Event();
            m_receive_cv.notify_one();
        }
    public:
        DistributedWorklistPeer(
            Context& context, IDistributedWorklist < TLocal, TRemote >& distributed_worklist,
            int priority_threshold, const DWCallbacks& device_callbacks, Endpoint endpoint, size_t chunk_size)
            :
            m_context(context), m_distributed_worklist(distributed_worklist), m_endpoint(endpoint),
            m_chunk_size(chunk_size), m_device_callbacks(device_callbacks), m_current_threshold(priority_threshold)
        {
        }

        ~DistributedWorklistPeer()
        {
            m_receive_thread.join();
            m_pop_thread.join();
        }

        Queue<TLocal>& GetLocalQueue(int i) override { return m_local_queues[i]; }

        const DWCallbacks& GetDeviceCallbacks() override { return m_device_callbacks; }

        PCQueue<TLocal>& GetRemoteInputQueue() override { return m_receive_queue; }

        PCQueue<TRemote>& GetRemoteOutputQueue() override { return m_send_queue; }

        std::vector< Segment<TLocal> > WaitForInputWork(Stream& stream, int priority_threshold = 0) override
        {
            auto segs = m_receive_queue.GetSegs(stream);

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
                segs = m_receive_queue.GetSegs(stream);
            }

            return segs;
        }

        void SendRemoteWork(const Event& ev) override
        {
            //
            // This method should be called by a single thread (worker) 
            //

            ev.Wait(m_send_stream);

            auto current = m_send_queue.GetBounds(m_send_stream);
            auto exclude = current.Exclude(m_send_bounds);
            m_send_bounds = current; // Keep for next round  

            std::vector< Segment<TRemote> > segs = m_send_queue.GetSegs(exclude);

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
        
        void SplitSend(Segment<TLocal>& split_work, Stream& stream) override
        {
            if (split_work.Empty()) return;

            InvokeSplitSend(split_work, stream);
            SendRemoteWork(m_context.RecordEvent(m_endpoint, stream));
        }
    };

    template<typename TLocal, typename TRemote, typename DWCallbacks, typename TWorker>
    class DistributedWorklist : public IDistributedWorklist < TLocal, TRemote >
    {
    private:
        typedef DistributedWorklistPeer<TLocal, TRemote, DWCallbacks> PeerType;
        typedef TWorker WorkerType;

        Context& m_context;
        Router<TRemote> m_router;

        EndpointList m_source_endpoints, m_work_endpoints; // Source and Work endpoints  
        std::map<Endpoint, std::unique_ptr<PeerType> > m_peers; // DW peer per work endpoint
        std::map<Endpoint, std::unique_ptr<WorkerType> > m_workers; // Worker per work endpoint

        std::map<Endpoint, Link<TRemote> > m_links; // Link per source endpoint  

        std::atomic<int> m_current_work_counter;
        std::atomic<int> m_deferred_work_counter;

        const int m_priority_delta;
        volatile int m_current_threshold;

        // Work-item counter
        std::atomic<uint32_t> m_total_work;
        std::map<Endpoint, std::atomic<uint32_t>> m_endpoint_work;

        volatile bool m_started, m_shotdown;
        volatile int m_report_time; // Last report time (seconds from start)
        
        std::chrono::high_resolution_clock::time_point m_start_time;
        std::thread m_watchdog; 
        std::mutex m_watchdog_mutex;
        std::condition_variable m_watchdog_cv;

        void RunWatchdog()
        {
            const int max_seconds = 5; // Max seconds allowed with no report activity  

            while (!m_shotdown)
            {
                { // Lock block
                    std::unique_lock<std::mutex> guard(m_watchdog_mutex);

                    m_watchdog_cv.wait_for(
                        guard, std::chrono::seconds(max_seconds/2), [this]() { return m_shotdown; });
                }
                    
                if (!m_started) continue;
                if (m_shotdown) break;

                auto report = m_report_time;
                auto current 
                    = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - m_start_time).count();

                if (current - report > max_seconds)
                {
                    //
                    // We encountered a possible deadlock, report and exit
                    //

                    printf("\nDistributed Worklist seems to be deadlocked. This is usually do to insufficient memory allocated for 'pass' queues");
                    printf("\nExiting with code %d\n", 15);
                    exit(15);
                }
            }
        }

        void ExitWatchdog()
        {
            if (!m_shotdown)
            {
                m_shotdown = true;
                std::lock_guard<std::mutex> guard(m_watchdog_mutex);
                m_watchdog_cv.notify_one();
            }
        }

        void MarkReportTime()
        {
            m_report_time 
                = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - m_start_time).count();
        }

    public:

        DistributedWorklist(
            Context& context, const EndpointList& sources, const EndpointList& workers, const std::map<Endpoint, DWCallbacks>& callbacks, 
            size_t chunk_size, size_t num_buffers, int priority_delta = 0, DistributedWorklistConfiguration configuration = DistributedWorklistConfiguration()) :
            IDistributedWorklist<TLocal, TRemote>(configuration),
            m_context(context), m_router(context, Policy::CreateRingPolicy(sources, workers), (int)(sources.size() + workers.size()), (int)workers.size()), 
            m_source_endpoints(sources), m_work_endpoints(workers), m_current_work_counter(0), m_deferred_work_counter(0), m_priority_delta(WorkerType::soft_prio ? priority_delta : 0), 
            m_current_threshold(priority_delta == 0 || WorkerType::soft_prio == false ? INT32_MAX : priority_delta), m_total_work(0),
            m_started(false), m_shotdown(false)
        {
            if (workers.size() != callbacks.size()) throw groute::exception("DWL parameter mismatch (workers <-> callbacks)");

            if (context.configuration.verbose)
            {
                printf(
                    "\nDistributed Worklist configuration: \n\tchunk: %llu, buffers: %llu, priority delta: %d, initial threshold: %d\n", 
                    chunk_size, num_buffers, priority_delta, m_current_threshold);
            }

            if (priority_delta > 0 && !WorkerType::soft_prio)
            {
                printf("Note: the Worker type used for the DWL does not support soft priority scheduling\n");
            }

            for (Endpoint source : sources)
            {
                m_endpoint_work[source] = 0; // Also sources report work
                m_links[source] = Link<TRemote>(source, m_router);
            }

            for (Endpoint worker : m_work_endpoints)
            {
                if (callbacks.find(worker) == callbacks.end()) throw groute::exception("DWL: missing DWCallbacks for worker endpoint");

                m_endpoint_work[worker] = 0;
                m_context.SetDevice(worker);

                m_peers[worker] = groute::make_unique< PeerType >(
                    m_context, *this, (int)m_current_threshold, callbacks.at(worker), worker, chunk_size);
                m_peers[worker]->AllocateLinks(m_router, num_buffers);
                m_workers[worker] = groute::make_unique< WorkerType >(m_context, worker);
            }

            if (context.configuration.verbose) printf("Distributed Worklist starting to run \n");

            // Second phase: reserving available memory for local work-queues after links allocation   
            m_context.ReserveFreeMemoryPercentage(0.9);
            for (Endpoint worker : m_work_endpoints)
            {
                m_context.SetDevice(worker);
                m_peers[worker]->AllocateQueues(WorkerType::num_local_queues);
                m_peers[worker]->Run();
            }

            m_start_time = std::chrono::high_resolution_clock::now();
            m_watchdog = std::thread([this]() { RunWatchdog(); });
        }

        virtual ~DistributedWorklist()
        {
            if (this->configuration.count_work && m_context.configuration.verbose)
            {
                printf("Work performed by each GPU:\n");
                for (auto& p : m_endpoint_work)
                    printf("  GPU %d: %d work-items\n", (Endpoint::identity_type)p.first, (uint32_t)p.second);
                int repwork = m_total_work;
                printf("Total work-items: %d\n", repwork);
            }

            ExitWatchdog();
            m_watchdog.join();
        }

        Link<TRemote>& GetLink(Endpoint source) override
        {
            if (m_links.find(source) == m_links.end()) throw groute::exception("Endpoint not registered as a source");
            return m_links[source];
        }

        IDistributedWorklistPeer<TLocal, TRemote, DWCallbacks>* GetPeer(Endpoint endpoint)
        {
            if (m_peers.find(endpoint) == m_peers.end()) throw groute::exception("Endpoint not registered as a worker");
            return m_peers[endpoint].get();
        }

        WorkerType* GetWorker(Endpoint endpoint)
        {
            if (m_workers.find(endpoint) == m_workers.end()) throw groute::exception("Endpoint not registered as a worker");
            return m_workers[endpoint].get();
        }

        template<typename... WorkArgs>
        void Work(Endpoint endpoint, Stream& stream, const WorkArgs&... args)
        {
            GetWorker(endpoint)->Work(*this, GetPeer(endpoint), stream, args...); 
        }

        uint32_t GetCurrentWorkCount(Endpoint endpoint) override
        {
            return m_endpoint_work[endpoint];
        }

        void ReportInitialWork(int initial_work, Endpoint endpoint, const char* caller = "") override
        {
            m_started = true; // Signal work has started 
            ReportWork(initial_work, 0, endpoint, caller, true);
        }

        void ReportWork(int new_work, int performed_work, Endpoint endpoint, const char* caller = "") override
        {
            ReportWork(new_work, performed_work, endpoint, caller, false);
        }

        void ReportWork(int new_work, int performed_work, Endpoint endpoint, const char* caller, bool initial)
        {
            MarkReportTime();

            int work = new_work - performed_work;

            if (this->configuration.count_work)
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
                    if (m_context.configuration.verbose) printf("Distributed Worklist shutting down successfully (%s)\n", caller);

                    ExitWatchdog();
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

        void ReportDeferredWork(int new_work, int performed_work, Endpoint endpoint, const char* caller = "") override
        {
            MarkReportTime();

            int work = new_work - performed_work;

            if (this->configuration.count_work)
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
