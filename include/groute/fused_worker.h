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

#ifndef __GROUTE_FUSED_WORKER_CUH
#define __GROUTE_FUSED_WORKER_CUH

#include <vector>

#include <cub/grid/grid_barrier.cuh>

#include <groute/worklist.h>

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)

namespace groute
{

    // Three different kernel fusion working schemes: Never stop, Run N times, 
    // and Run for N cycles

    struct NeverStop
    {
        __device__ __forceinline__ bool stop() { return false; }
    };

    template<int N = 1>
    struct RunNTimes
    {
        int m_run;
        __device__ RunNTimes() : m_run(0) {}
        __device__ __forceinline__ bool stop()
        {
            if (m_run >= N) return true;
            ++m_run;
            return false;
        }
    };

    template<int CYCLES>
    struct RunFor
    {
        uint64_t m_target;
        __device__ RunFor() { m_target = clock64() + CYCLES; }
        __device__ __forceinline__ bool stop()
        {
            return (clock64() > m_target);
        }
    };


    template <typename T, typename SplitOps, typename PrioT>
    __device__ void SplitHighLow(SplitOps& ops, PrioT global_priority,
                                 groute::dev::CircularWorklist<T>& remote_in,
                                 uint32_t work_size,
                                 groute::dev::Worklist<T>& high_wl,
                                 groute::dev::Worklist<T>& low_wl)
    {
        // ballot/warp-aggregated
        // Returns number of processed items

        // Expecting a function in split ops with the following signature:  
        //      bool is_high_prio(T work, PrioT global_prio)

        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
        {
            T work = remote_in.read(i); // takes care of circularity  

            bool is_high_prio = ops.is_high_prio(work, global_priority);

            int high_mask = __ballot(is_high_prio ? 1 : 0);
            int low_mask = __ballot(is_high_prio ? 0 : 1);

            if (is_high_prio)
            {
                int high_leader = __ffs(high_mask) - 1;
                int thread_offset = __popc(high_mask & ((1 << lane_id()) - 1));
                high_wl.append_warp(work, high_leader, __popc(high_mask), thread_offset);
            }
            else
            {
                int low_leader = __ffs(low_mask) - 1;
                int thread_offset = __popc(low_mask & ((1 << lane_id()) - 1));
                low_wl.append_warp(work, low_leader, __popc(low_mask), thread_offset);
            }
        }
    }

    template <typename StoppingCondition, typename LocalT, typename RemoteT, typename PrioT,
        typename SplitOps, typename Work, typename... WorkArgs>
        __global__ void FusedWork(groute::dev::Worklist<LocalT>           high_wl,
                                  groute::dev::Worklist<LocalT>           low_wl,
                                  groute::dev::CircularWorklist<LocalT>   remote_in,
                                  groute::dev::CircularWorklist<RemoteT>  remote_out,
                                  int               chunk_size,
                                  PrioT             global_prio,
                                  volatile int*     host_high_work_counter,
                                  volatile int*     host_low_work_counter,
                                  uint32_t*         g_work_size,
                                  volatile int *    send_signal_ptr,
                                  cub::GridBarrier  gbar,
                                  SplitOps          ops,
                                  WorkArgs...       args)
    {
        // Keep one SM free (use N-1 blocks where N is the number of SMs)
        StoppingCondition cond;

        // Message transmission variables
        int new_high_work = 0, performed_high_work = 0;
        uint32_t prev_start;

        int prev_low_work, prev_high_work;
        if (GTID == 0)
        {
            prev_low_work = low_wl.len();
            prev_high_work = high_wl.len();
        }

        while (!cond.stop())
        {
            if (GTID == 0)
            {
                *g_work_size = remote_in.size(); // we must decide on work size globally  
            }

            gbar.Sync();

            uint32_t work_size = *g_work_size; //  broadcast to all threads   
            SplitHighLow<LocalT, SplitOps>(ops, global_prio, remote_in, work_size, high_wl, low_wl);

            gbar.Sync();

            if (GTID == 0)
            {
                new_high_work += (int)high_wl.len();
                performed_high_work += (int)work_size;

                remote_in.pop_items(work_size);
                prev_start = remote_in.get_start();
            }

            if (high_wl.len() == 0)
                break;

            // This work also fills remote_in
            for (int chunk = 0; chunk < high_wl.len(); chunk += chunk_size)
            {
                gbar.Sync();

                // Perform work chunk
                int cur_chunk = min(high_wl.len() - chunk, chunk_size);
                Work::work(groute::dev::WorkSourceArray<LocalT>(high_wl.m_data + chunk, cur_chunk),
                    remote_in,  // prepending local work here 
                    remote_out, // appending remote work here
                    args...
                    );

                gbar.Sync();

                // Transmit work
                if (GTID == 0)
                {
                    uint32_t remote_work_count = remote_out.get_alloc_count_and_sync();
                    if (remote_work_count > 0) IncreaseHostFlag(send_signal_ptr, remote_work_count);
                }
            }

            // Count total performed work
            if (GTID == 0)
            {
                new_high_work += remote_in.get_start_diff(prev_start);
                performed_high_work += high_wl.len();

                high_wl.reset();
            }
        }

        if (GTID == 0)
        {
            __threadfence();
            // Report work
            *host_high_work_counter = new_high_work - performed_high_work - prev_high_work;
            *host_low_work_counter = low_wl.len() - prev_low_work;
        }
    }

    template <typename RemoteT>
    __global__ void RemoteSignal(groute::dev::CircularWorklist<RemoteT> remote_out,
                                 volatile int * send_signal_ptr)
    {
		// Transmit work
        if (GTID == 0)
        {
            uint32_t remote_work_count = remote_out.get_alloc_count_and_sync();
            if (remote_work_count > 0) IncreaseHostFlag(send_signal_ptr, remote_work_count);
        }
    }
}

#endif //  __GROUTE_FUSED_WORKER_CUH
