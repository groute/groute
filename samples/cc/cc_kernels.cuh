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
#ifndef __CC_KERNELS_H
#define __CC_KERNELS_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>

typedef groute::graphs::Edge Edge;

static __global__ void InitParents(groute::graphs::dev::Irregular<int> parents)
{
    int tid = parents.get_wid();

    if (!parents.has(tid))
        return;

    parents.write(tid, tid);
}

template<typename Graph>
__global__ void HookHighToLow(
    groute::graphs::dev::Irregular<int>     parents,
    Graph                    edges,
    groute::graphs::dev::Flag          flag
    )
{
    __shared__ typename decltype(flag)::SharedData flag_sdata;

    flag.init(flag_sdata);

    Edge e = edges.GetEdge(GTID);

    if (e.u != e.v)
    {
        int p_u = parents.read(e.u);
        int p_v = parents.read(e.v);

        if (p_u != p_v)
        {
            int high = p_u > p_v ? p_u : p_v;
            int low = p_u + p_v - high;

            flag.set();      // signal work was done
            parents.write(high, low);    // hook
        }
    }

    flag.commit();
}


template<typename Graph, bool R1 = false>
__global__ void Hook(
    groute::graphs::dev::Irregular<int>     parents,
    Graph                    edges
    )
{
    Edge e = edges.GetEdge(GTID);

    if (!R1) // at R1 we know parents contains self pointers  
    {
        e.u = parents.read(e.u);
        e.v = parents.read(e.v);
    }

    int high = e.u > e.v ? e.u : e.v;
    int low = e.u + e.v - high;

    parents.write(high, low);    // hook
}


template<typename Graph>
__global__ void HookHighToLowAtomic(
    groute::graphs::dev::Irregular<int>     parents,
    Graph                    edges
    )
{
    Edge e = edges.GetEdge(GTID);

    if (e.u != e.v)
    {
        int p_u = parents.read(e.u);
        int p_v = parents.read(e.v);

        while (p_u != p_v)
        {
            int high = p_u > p_v ? p_u : p_v;
            int low = p_u + p_v - high;

            int prev = parents.write_atomicCAS(high, high, low);

            if (prev == high || prev == low) {
                break;
            }

            p_u = parents.read(prev);
            p_v = parents.read(low);
        }
    }
}

static __global__ void MultiJumpCompress(groute::graphs::dev::Irregular<int> parents)
{
    int tid = parents.get_wid();

    if (!parents.has(tid))
        return;

    int p, pp;

    p = parents.read(tid);
    pp = parents.read(p);

    while (p != pp)
    {
        parents.write(tid, pp);

        p = pp;
        pp = parents.read(p);
    }
}

#endif // __CC_KERNELS_H
