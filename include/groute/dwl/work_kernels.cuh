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

#ifndef __GROUTE_WORK_KERNELS_CUH
#define __GROUTE_WORK_KERNELS_CUH

#include <vector>

#include <cub/grid/grid_barrier.cuh>

#include <groute/device/signal.cuh>
#include <groute/device/queue.cuh>

#include <groute/dwl/work_source.cuh>
#include <groute/dwl/work_target.cuh>

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

    template <typename T, typename DWCallbacks, typename TPrio>
    __device__ void SplitDeferredWork(DWCallbacks& callbacks, TPrio priority_threshold,
                                 dev::CircularWorklist<T>& remote_input,
                                 uint32_t work_size,
                                 dev::Worklist<T>& immediate_worklist,
                                 dev::Worklist<T>& deferred_worklist)
    {
        // ballot/warp-aggregated
        // Returns number of processed items

        // Expecting a function in callbacks with the following signature:  
        //      bool should_defer(T work, TPrio global_threshold)  

        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
        {
            T work = remote_input.read(i); // takes care of circularity  

            bool is_immediate_work = !callbacks.should_defer(work, priority_threshold);

            int immediate_mask = __ballot(is_immediate_work ? 1 : 0);
            int deferred_mask = __ballot(is_immediate_work ? 0 : 1);

            if (is_immediate_work)
            {
                int leader = __ffs(immediate_mask) - 1;
                int thread_offset = __popc(immediate_mask & ((1 << cub::LaneId()) - 1));
                immediate_worklist.append_warp(work, leader, __popc(immediate_mask), thread_offset);
            }
            else
            {
                int leader = __ffs(deferred_mask) - 1;
                int thread_offset = __popc(deferred_mask & ((1 << cub::LaneId()) - 1));
                deferred_worklist.append_warp(work, leader, __popc(deferred_mask), thread_offset);
            }
        }
    }

    template <
        typename StoppingCondition, typename TLocal, typename TRemote, typename TPrio,
        typename DWCallbacks, typename Work, typename... WorkArgs>
    __global__ void FusedWorkKernel(
                                  dev::Worklist<TLocal>           immediate_worklist,
                                  dev::Worklist<TLocal>           deferred_worklist,
                                  dev::CircularWorklist<TLocal>   remote_input,
                                  dev::CircularWorklist<TRemote>  remote_output,
                                  int               chunk_size,
                                  TPrio             priority_threshold,
                                  volatile int*     host_current_work_counter,
                                  volatile int*     host_deferred_work_counter,
                                  uint32_t*         grid_work_size,
                                  volatile int*     remote_work_signal,
                                  cub::GridBarrier  grid_barrier,
                                  DWCallbacks       callbacks,
                                  WorkArgs...       args)
    {
        // Keeping one SM free (use N-1 blocks where N is the number of SMs)
        StoppingCondition cond;

        // The work target for Work::work
        dev::WorkTargetSplitSend<TLocal, TRemote, DWCallbacks> work_target(remote_input, remote_output, callbacks);

        // Message transmission variables
        int new_immediate_work = 0, performed_immediate_work = 0;
        uint32_t prev_start;

        int prev_deferred_work, prev_immediate_work;
        if (TID_1D == 0)
        {
            prev_deferred_work = deferred_worklist.len();
            prev_immediate_work = immediate_worklist.len();
        }

        while (!cond.stop())
        {
            if (TID_1D == 0)
            {
                *grid_work_size = remote_input.size(); // we must decide on work size globally  
            }

            grid_barrier.Sync();

            uint32_t work_size = *grid_work_size; //  broadcast to all threads   
            SplitDeferredWork<TLocal, DWCallbacks>(callbacks, priority_threshold, remote_input, work_size, immediate_worklist, deferred_worklist);

            grid_barrier.Sync();

            if (TID_1D == 0)
            {
                new_immediate_work += (int)immediate_worklist.len();
                performed_immediate_work += (int)work_size;

                remote_input.pop_items(work_size);
                prev_start = remote_input.get_start();
            }

            if (immediate_worklist.len() == 0)
                break;

            // This work also fills remote_input
            for (int chunk = 0; chunk < immediate_worklist.len(); chunk += chunk_size)
            {
                grid_barrier.Sync();

                // Perform work chunk
                int cur_chunk = min(immediate_worklist.len() - chunk, chunk_size);

                Work::work(
                    dev::WorkSourceArray<TLocal>(immediate_worklist.m_data + chunk, cur_chunk),
                    work_target,
                    args...
                    );

                grid_barrier.Sync();

                // Transmit work
                if (TID_1D == 0)
                {
                    uint32_t remote_work_count = remote_output.get_alloc_count_and_sync();
                    if (remote_work_count > 0) dev::Signal::Increase(remote_work_signal, remote_work_count);
                }
            }

            // Count total performed work
            if (TID_1D == 0)
            {
                new_immediate_work += remote_input.get_start_diff(prev_start);
                performed_immediate_work += immediate_worklist.len();

                immediate_worklist.reset();
            }
        }

        if (TID_1D == 0)
        {
            __threadfence();
            // Report work
            *host_current_work_counter = new_immediate_work - performed_immediate_work - prev_immediate_work;
            *host_deferred_work_counter = deferred_worklist.len() - prev_deferred_work;
        }
    }


    template <
            typename WorkSource, typename TLocal, typename TRemote, 
            typename DWCallbacks, typename Work, typename... WorkArgs>
    __global__ void WorkKernel(WorkSource work_source, dev::Worklist<TLocal> output_worklist,
                                  DWCallbacks       callbacks,
                                  WorkArgs...       args)
    {
        dev::WorkTargetWorklist<TLocal, TRemote, DWCallbacks> work_target(output_worklist, callbacks);
        Work::work(work_source, work_target, args...);
    }
}

#endif //  __GROUTE_WORK_KERNELS_CUH
