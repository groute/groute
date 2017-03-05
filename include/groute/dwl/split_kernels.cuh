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

#ifndef __GROUTE_SPLIT_KERNELS_CUH
#define __GROUTE_SPLIT_KERNELS_CUH

#include <vector>

#include <groute/device/queue.cuh>
#include <groute/device/signal.cuh>
#include <groute/device/counter.cuh>

namespace groute
{
        
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
                if (cub::LaneId() == filter_leader)
                    filter_counter.add(__popc(filter_mask));
            }
            else
            {
                if (flags & SF_Take)
                {
                    int take_leader = __ffs(take_mask) - 1;
                    int thread_offset = __popc(take_mask & ((1 << cub::LaneId()) - 1));
                    local_work.append_warp(callbacks.unpack(work), take_leader, __popc(take_mask), thread_offset);
                }

                if (flags & SF_Pass)
                    // pass on to another endpoint
                {
                    int pass_leader = __ffs(pass_mask) - 1;
                    int thread_offset = __popc(pass_mask & ((1 << cub::LaneId()) - 1));
                    remote_work.append_warp(work, pass_leader, __popc(pass_mask), thread_offset);
                }
            }
        }
    }
}

#endif //  __GROUTE_SPLIT_KERNELS_CUH
