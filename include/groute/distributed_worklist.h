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

#include <groute/internal/cuda_utils.h>
#include <groute/internal/worker.h>
#include <groute/internal/pinned_allocation.h>

#include <groute/event_pool.h>
#include <groute/context.h>
#include <groute/worklist.h>
#include <groute/groute.h>

#include <gflags/gflags.h>

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
    struct SplitOps // an example for the required format
    {
        SplitFlags on_receive(const TPacked& data);
        SplitFlags on_send(const TUnpacked& data);
    
        TPacked pack(const TUnpacked& data);
        TUnpacked unpack(const TPacked& data);
    };
    */
    
    template<typename TLocal, typename TRemote, typename SplitOps, bool WarpAppend = true>
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
                if (WarpAppend) {
                    local_work.prepend_warp(work); // notice the prepend  
                }
                else {
                    local_work.prepend(work);
                }
            }

            if (flags & SF_Pass)
            {
                // pack data
                TRemote packed = split_ops.pack(work);
    
                if (WarpAppend) {
                    remote_work.append_warp(packed); // notice the append    
                }
                else {
                    remote_work.append(packed);
                }
            }
        }
    }

    template<typename TLocal, typename TRemote, typename SplitOps, bool WarpAppend = true>
    __global__ void SplitReceiveKernel(
        SplitOps split_ops, TRemote* work_ptr, uint32_t work_size,
        dev::CircularWorklist<TLocal> local_work, dev::CircularWorklist<TRemote> remote_work, dev::Counter filter_counter)
    {
        int tid = TID_1D;
        if (tid < work_size)
        {
            TRemote work = work_ptr[tid];
            SplitFlags flags = split_ops.on_receive(work);

            if (WarpAppend)
            {
                int filter_mask = __ballot(flags == SF_None ? 1 : 0);
                int take_mask = __ballot(flags & SF_Take ? 1 : 0);
                int pass_mask = __ballot(flags & SF_Pass ? 1 : 0);
                    // never inline the masks into the conditional branching below  
                    // although it may work. The compiler should optimize this anyhow, 
                    // but this avoids him from unifying the __ballot's 

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
                        // pass on to another device
                    {
                        int pass_leader = __ffs(pass_mask) - 1;
                        int thread_offset = __popc(pass_mask & ((1 << lane_id()) - 1));
                        remote_work.append_warp(work, pass_leader, __popc(pass_mask), thread_offset);
                    }
                }
            }

            else // templated, no warp operations  
            {
                if (flags == SF_None)
                {
                    filter_counter.add(1);    
                }

                else
                {
                    if (flags & SF_Take)
                    {
                        local_work.append(split_ops.unpack(work));
                    }
                    if (flags & SF_Pass)
                    {
                        // belongs to another device
                        remote_work.append(work);
                    }
                }
            }
        }
    }

    /*
    Just a bunch of configuration bitmap flags for the distributed worklist
    */
    enum DistributedWorklistFlags
    {
        DW_NoFlags             = 0,
        DW_WarpAppend          = 1 << 0,
        DW_DebugPrint          = 1 << 1,
        DW_HighPriorityReceive = 1 << 2
    };

    struct IDistributedWorklist
    {
        virtual ~IDistributedWorklist() { }

        virtual void ReportWork(int new_work, int performed_work, const char* caller, device_t dev) = 0;        
        virtual void ReportWork(int work) = 0;
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

        /// A temp worklist for user-code, just allocated correctly, not used internally
        virtual Worklist<TLocal>& GetTempWorklist() = 0;

        /// A blocking call for local work segments 
        virtual std::vector< Segment<TLocal> > GetLocalWork(Stream& stream) = 0;

        /// Perform split-send, local work will be prepended into the LocalInputWorklist
        /// and remote work will be appended into the RemoteOutputWorklist (+signal to send thread)  
        virtual void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) = 0; 

        /// Signal that work was pushed into the RemoteOutputWorklist
        virtual void SignalRemoteWork(const Event& ev) = 0; 
    };

    template<typename TLocal, typename TRemote, typename SplitOps>
    class DistributedWorklistPeer : public IDistributedWorklistPeer<TLocal, TRemote>
    {
    protected:
        int m_dev, m_ngpus;

    private:
        Context& m_context;
        IDistributedWorklist& m_distributed_worklist;
        
        SplitOps m_split_ops;
        DistributedWorklistFlags m_flags;
        Counter m_filter_counter;

        CircularWorklist<TLocal> m_local_input_worklist;
        Worklist<TLocal> m_temp_worklist;

        CircularWorklist<TRemote> 
            m_send_remote_output_worklist, // From local work (split-send)
            m_pass_remote_output_worklist; // From previous device on the ring (split-receive), passing on  

        std::thread m_receive_thread;
        std::thread m_send_thread;

        // Sync objects  
        //
        // Receive:  
        std::mutex m_receive_mutex;
        std::condition_variable m_receive_cv;
        bool m_receive_work = false;
        Event m_receive_work_event;
        //
        // Send (wait any)
        std::mutex m_send_mutex;
        std::condition_variable m_send_cv;
        //
        //  Send-remote: (split-send)
        bool m_send_remote_work = false;
        Event m_send_remote_work_event;
        //
        // Pass-remote: (split-receive)  
        bool m_pass_remote_work = false;
        Event m_pass_remote_work_event;
        //
        // Exit:
        volatile bool m_exit = false;

        Link<TRemote> m_link_in, m_link_out;
        
        void SplitReceive(
            const groute::Segment<TRemote>& received_work,
            groute::CircularWorklist<TLocal>& local_work,
            groute::CircularWorklist<TRemote>& remote_work, groute::Stream& stream)
        {
            m_filter_counter.ResetAsync(stream.cuda_stream);

            dim3 block_dims(DBS, 1, 1);
            dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1, 1);

            if (m_flags & DW_WarpAppend) 
            {
                SplitReceiveKernel <TLocal, TRemote, SplitOps, true> <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_split_ops,
                    received_work.GetSegmentPtr(), received_work.GetSegmentSize(),
                    local_work.DeviceObject(), remote_work.DeviceObject(), m_filter_counter.DeviceObject()
                    );
            }
            else
            {
                SplitReceiveKernel <TLocal, TRemote, SplitOps, false> <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_split_ops,
                    received_work.GetSegmentPtr(), received_work.GetSegmentSize(),
                    local_work.DeviceObject(), remote_work.DeviceObject(), m_filter_counter.DeviceObject()
                    );
            }

            local_work.SyncAppendAllocAsync(stream.cuda_stream);
            remote_work.SyncAppendAllocAsync(stream.cuda_stream);

#ifndef NDEBUG
            if (m_flags & DW_DebugPrint) // debug prints  
            {
                printf("\n\n\tDevice: %d\nSplitReceive->Local work: ", m_dev);
                local_work.PrintOffsetsDebug(stream);
                printf("\nSplitReceive->Remote work: ");
                remote_work.PrintOffsetsDebug(stream);
            }
#endif

            // Report work
            // TODO (later): Try to avoid copies to host
            m_distributed_worklist.ReportWork(
                (int)received_work.GetSegmentSize() - (int)m_filter_counter.GetCount(stream),
                (int)received_work.GetSegmentSize(),
                "Filter", m_dev
                );
        }

        void SplitSend(
            const groute::Segment<TLocal>& sent_work,
            groute::CircularWorklist<TLocal>& local_work,
            groute::CircularWorklist<TRemote>& remote_work, groute::Stream& stream)
        {
            dim3 block_dims(DBS, 1, 1);
            dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

            if (m_flags & DW_WarpAppend) 
            {
                SplitSendKernel <TLocal, TRemote, SplitOps, true> <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_split_ops,
                    sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
                    local_work.DeviceObject(), remote_work.DeviceObject()
                    );
            }
            else
            {
                SplitSendKernel <TLocal, TRemote, SplitOps, false> <<< grid_dims, block_dims, 0, stream.cuda_stream >>>(
                    m_split_ops,
                    sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
                    local_work.DeviceObject(), remote_work.DeviceObject()
                    );
            }

            remote_work.SyncAppendAllocAsync(stream.cuda_stream);

#ifndef NDEBUG
            if (m_flags & DW_DebugPrint) // debug prints  
            {
                printf("\n\n\tDevice: %d\nSplitSend->Local work: ", m_dev);
                local_work.PrintOffsetsDebug(stream);
                printf("\nSplitSend->Remote work: ");
                remote_work.PrintOffsetsDebug(stream);
            }
#endif

            // split-send does no filtering, no need to update distributed worklist with work
        }

        void ReceiveLoop()
        {
            m_context.SetDevice(m_dev);
            Stream stream = m_context.CreateStream(m_dev, (m_flags & DW_HighPriorityReceive) ? SP_High : SP_Default);

            while (true)
            {
                auto fut = m_link_in.Receive();
                auto seg = fut.get();
                if (seg.Empty()) break;

                // queue a wait on stream
                seg.Wait(stream.cuda_stream);
                SplitReceive(seg, m_local_input_worklist, m_pass_remote_output_worklist, stream);

                Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);

                m_link_in.ReleaseBuffer(seg, split_ev);

                // Signal
                {
                    std::lock_guard<std::mutex> guard(m_send_mutex);
                    m_pass_remote_work = true;
                    m_pass_remote_work_event = split_ev;
                    m_send_cv.notify_one();
                }

                {
                    std::lock_guard<std::mutex> guard(m_receive_mutex);
                    m_receive_work = true;
                    m_receive_work_event = split_ev;
                    m_receive_cv.notify_one();
                }
            }

            stream.Sync();
                            
            // Signal exit
            {
                std::lock_guard<std::mutex> guard(m_send_mutex);
                m_exit = true;
                m_send_cv.notify_one();
            }

            {
                std::lock_guard<std::mutex> guard(m_receive_mutex);
                m_exit = true;
                m_receive_cv.notify_one();
            }
        }

        void SendLoop()
        {
            m_context.SetDevice(m_dev);
            Stream stream = m_context.CreateStream(m_dev);

            int source = 0;

            while (true)
            {
                CircularWorklist<TRemote>* worklist; 
                Event work_ev;

                {
                    std::unique_lock<std::mutex> guard(m_send_mutex);

                    while (true)
                    {
                        if (m_exit) break;

                        if (source == 0) // we alternate source for giving each worklist a fair chance  
                        {
                            if (m_pass_remote_work) // we first check the pass list at this round   
                            {
                                m_pass_remote_work = false;
                                work_ev = std::move(m_pass_remote_work_event);
                                worklist = &m_pass_remote_output_worklist;
                                break;
                            }

                            if (m_send_remote_work)
                            {
                                m_send_remote_work = false;
                                work_ev = std::move(m_send_remote_work_event);
                                worklist = &m_send_remote_output_worklist;
                                break;
                            }
                        }

                        else
                        {
                            if (m_send_remote_work) // we first check the send list at this round  
                            {
                                m_send_remote_work = false;
                                work_ev = std::move(m_send_remote_work_event);
                                worklist = &m_send_remote_output_worklist;
                                break;
                            }

                            if (m_pass_remote_work)
                            {
                                m_pass_remote_work = false;
                                work_ev = std::move(m_pass_remote_work_event);
                                worklist = &m_pass_remote_output_worklist;
                                break;
                            }
                        }

                        m_send_cv.wait(guard);
                    }
                }

                if (m_exit) break;

                source = 1 - source;

                work_ev.Wait(stream.cuda_stream);
                std::vector< Segment<TRemote> > output_segs = worklist->ToSegs(stream);

                for (auto output_seg : output_segs)
                {
                    auto ev = m_link_out.Send(output_seg, Event()).get();
                    ev.Wait(stream.cuda_stream);
                    worklist->PopItemsAsync(output_seg.GetSegmentSize(), stream.cuda_stream);
                }
            }
        }

    public:
        DistributedWorklistPeer(
            Context& context, router::IRouter<TRemote>& router, 
            IDistributedWorklist& distributed_worklist, const SplitOps& split_ops, DistributedWorklistFlags flags,
            device_t dev, int ngpus, size_t max_work_size, size_t max_exch_size, size_t exch_buffs) 
            :
            m_context(context), m_dev(dev), m_ngpus(ngpus), m_distributed_worklist(distributed_worklist), 
            m_split_ops(split_ops), m_flags(flags),
            m_link_in(router, dev, max_exch_size, exch_buffs), 
            m_link_out(dev, router)
        {
            void* mem_buffer;
            size_t mem_size;

            mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_in, mem_size, AF_PO2);
            m_local_input_worklist = groute::CircularWorklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal));

            mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_local, mem_size);
            m_temp_worklist = groute::Worklist<TLocal>((TLocal*)mem_buffer, mem_size / sizeof(TLocal));

            mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_out, mem_size, AF_PO2);
            m_send_remote_output_worklist = groute::CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote));
            
            mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_pass, mem_size, AF_PO2);
            m_pass_remote_output_worklist = groute::CircularWorklist<TRemote>((TRemote*)mem_buffer, mem_size / sizeof(TRemote));

            m_local_input_worklist.ResetAsync((cudaStream_t) 0);
            m_temp_worklist.ResetAsync((cudaStream_t) 0);
            
            m_send_remote_output_worklist.ResetAsync((cudaStream_t) 0); 
            m_pass_remote_output_worklist.ResetAsync((cudaStream_t) 0);

            GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) 0)); // just in case
            
            m_receive_thread = std::thread([this]() { ReceiveLoop(); });
            m_send_thread = std::thread([this]() { SendLoop(); });
        }

        ~DistributedWorklistPeer()
        {
            m_receive_thread.join();
            m_send_thread.join();
        }

        CircularWorklist<TLocal>& GetLocalInputWorklist()   override { return m_local_input_worklist; }
        CircularWorklist<TRemote>& GetRemoteOutputWorklist() override { return m_send_remote_output_worklist; }
        Worklist<TLocal>& GetTempWorklist()                 override { return m_temp_worklist; }

        std::vector< Segment<TLocal> > GetLocalWork(Stream& stream) override
        {
            auto segs = m_local_input_worklist.ToSegs(stream);

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
                segs = m_local_input_worklist.ToSegs(stream);
            }

            return segs;
        }

        void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) override
        {
            if (split_work.Empty()) return;

            SplitSend(split_work, m_local_input_worklist, m_send_remote_output_worklist, stream);
            Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
            SignalRemoteWork(split_ev);
        }

        void SignalRemoteWork(const Event& ev) override
        {
            // Signal
            std::lock_guard<std::mutex> guard(m_send_mutex);
            m_send_remote_work = true;
            m_send_remote_work_event = ev;
            m_send_cv.notify_one();
        }
    };

    template<typename TLocal, typename TRemote>
    class DistributedWorklist : public IDistributedWorklist
    {
    private:
        Context& m_context;
        router::IRouter<TRemote>& m_router;
        int m_ngpus;

        std::atomic<int> m_active_peers_counter;
        std::atomic<int> m_work_counter;

        // Workitem counter
        std::atomic<unsigned int> m_reported_work;
        std::vector<unsigned int> m_ctr;
    public:
        unsigned int GetCurrentWorkCount(device_t dev)
        {
            return m_ctr[dev + 1];
        }
        
    public:
        std::mutex log_gate;

    public:

        DistributedWorklist(Context& context, router::IRouter<TRemote>& router, int ngpus) :
        m_context(context), m_router(router), m_ngpus(ngpus), m_work_counter(0), m_active_peers_counter(ngpus), m_reported_work(0)
        {
            if (false)
            {
                m_ctr.resize(ngpus+1, 0);
            }
        }

        virtual ~DistributedWorklist()
        {
            if (false)
            {
                printf("Work performed by each GPU:\n");
                for (size_t i = 1; i < m_ctr.size(); ++i)
                    printf("  GPU %llu: %lu witems\n", i, m_ctr[i]);
                int repwork = m_reported_work;
                printf("Total witems: %lu\n", repwork);
            }
        }

        template<typename SplitOps>
        std::unique_ptr< IDistributedWorklistPeer<TLocal, TRemote> > CreatePeer(
            device_t dev, const SplitOps& split_ops, 
            size_t max_work_size, size_t max_exch_size, size_t exch_buffs, DistributedWorklistFlags flags = (DistributedWorklistFlags)(DW_WarpAppend | DW_HighPriorityReceive))
        {
            m_context.SetDevice(dev);
            return groute::make_unique< DistributedWorklistPeer<TLocal, TRemote, SplitOps> >(
                m_context, m_router, *this, split_ops, flags, dev, m_ngpus, max_work_size, max_exch_size, exch_buffs);
        }

        void ReportPeerTermination()
        {
            if (--m_active_peers_counter == 0)
            {
                m_router.Shutdown();
            }
        }

        void ReportWork(int new_work, int performed_work, const char* caller, device_t dev) override
        {
            int work = new_work - performed_work;

            if (false)
            {
                m_reported_work += performed_work;
                m_ctr[dev + 1] += performed_work;
            }
            
            if (work == 0) return;

            int current_work = (m_work_counter += work);

            //{
            //    std::lock_guard<std::mutex> lock(log_gate);

            //    std::cout 
            //        << std::endl 
            //        << '[' << std::this_thread::get_id() << ",\t" << caller << ']' 
            //        << "\tNew: " << new_work
            //        << ",\tPerformed: " << performed_work
            //        << ",\tCurrent: " << current_work;
            //}

            if (current_work == 0)
            {
                m_router.Shutdown();
            }
        }

        void ReportWork(int work) override
        {
            if (work == 0) return;

            int current_work = (m_work_counter += work);

            //{
            //    std::lock_guard<std::mutex> lock(log_gate);

            //    std::cout 
            //        << std::endl 
            //        << '[' << std::this_thread::get_id() << ']' 
            //        << "\t\tWork: " << work
            //        << ",\t\tCurrent: " << current_work;
            //}

            if (current_work == 0)
            {
                m_router.Shutdown();
            }
        }

        bool HasWork() const override
        {
            return m_work_counter > 0;
        }
        
        bool HasActivePeers() override
        {
            return m_active_peers_counter > 0;
        }
    };
}

#endif // __GROUTE_DISTRIBUTED_WORKLIST_H
