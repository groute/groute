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

#ifndef __GROUTE_WORK_TARGET_H
#define __GROUTE_WORK_TARGET_H

#include <groute/worklist/work_queue.cu.h>
#include <groute/worklist/distributed_worklist.cu.h>

namespace groute {
    namespace dev {
          

        /*
        //
        // WorkTarget classes (device):
        //
        API:
        template<typename TLocal, typename TRemote>
        struct WorkTarget
        {
            __device__ void append_work(const TLocal& work) { ... }
            __device__ void append_work(const TRemote& work) { ... }
        };
        */

        template<typename TLocal, typename TRemote, typename DWCallbacks>
        struct WorkTargetWorklist
        {
        private:
            dev::Worklist<TLocal>& m_worklist;
            DWCallbacks& m_callbacks;

        public:
            __device__ __forceinline__  WorkTargetWorklist(dev::Worklist<TLocal>& worklist, DWCallbacks& callbacks) : m_worklist(worklist), m_callbacks(callbacks) { }

            __device__ __forceinline__ void append_work(const TLocal& work)
            {
                m_worklist.append_warp(work);
            }

            __device__ __forceinline__ void append_work(const TRemote& work)
            {
                m_worklist.append_warp(m_callbacks.unpack(work));
            }
        };

        template<typename T, typename DWCallbacks>
        struct WorkTargetWorklist < T, T, DWCallbacks >
        {
        private:
            dev::Worklist<T>& m_worklist;

        public:
            __device__ __forceinline__  WorkTargetWorklist(dev::Worklist<T>& worklist, DWCallbacks& callbacks) : m_worklist(worklist) { }

            __device__ __forceinline__ void append_work(const T& work)
            {
                m_worklist.append_warp(work);
            }
        };

        template<typename TLocal, typename TRemote, typename DWCallbacks>
        struct WorkTargetSplit
        {
        private:
            dev::CircularWorklist<TLocal>& m_remote_input;
            dev::CircularWorklist<TRemote>& m_remote_output;
            DWCallbacks& m_callbacks;

        public:
            __device__ __forceinline__  WorkTargetSplit(dev::CircularWorklist<TLocal>& remote_input, dev::CircularWorklist<TRemote>& remote_output, DWCallbacks& callbacks) :
                m_remote_input(remote_input), m_remote_output(remote_output), m_callbacks(callbacks) { }

            __device__ __forceinline__ void append_work(const TLocal& unpacked)
            {
                SplitFlags flags = m_callbacks.on_send(unpacked);
                if (flags & SF_Take)
                {
                    m_remote_input.prepend_warp(unpacked); // prepending to input 
                }

                if (flags & SF_Pass)
                {
                    // pack data
                    TRemote packed = m_callbacks.pack(unpacked);
                    m_remote_output.append_warp(packed); // appending  
                }
            }

            __device__ __forceinline__ void append_work(const TRemote& packed)
            {
                // unpack data
                TLocal unpacked = m_callbacks.unpack(packed);
                SplitFlags flags = m_callbacks.on_send(unpacked);
                if (flags & SF_Take)
                {
                    m_remote_input.prepend_warp(unpacked); // prepending to input 
                }

                if (flags & SF_Pass)
                {
                    m_remote_output.append_warp(packed); // appending  
                }
            }
        };

        template<typename T, typename DWCallbacks>
        struct WorkTargetSplit < T, T, DWCallbacks >
        {
        private:
            dev::CircularWorklist<T>& m_remote_input;
            dev::CircularWorklist<T>& m_remote_output;
            DWCallbacks& m_callbacks;

        public:
            __device__ __forceinline__  WorkTargetSplit(dev::CircularWorklist<T>& remote_input, dev::CircularWorklist<T>& remote_output, DWCallbacks& callbacks) :
                m_remote_input(remote_input), m_remote_output(remote_output), m_callbacks(callbacks) { }

            __device__ __forceinline__ void append_work(const T& unpacked)
            {
                SplitFlags flags = m_callbacks.on_send(unpacked);
                if (flags & SF_Take)
                {
                    m_remote_input.prepend_warp(unpacked); // prepending to input 
                }

                if (flags & SF_Pass)
                {
                    // pack data
                    T packed = m_callbacks.pack(unpacked);
                    m_remote_output.append_warp(packed); // appending  
                }
            }
        };
    }
}

#endif // __GROUTE_WORK_TARGET_H
