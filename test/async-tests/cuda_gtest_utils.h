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
#ifndef __CUDA_GTEST_UTILS_H_
#define __CUDA_GTEST_UTILS_H_

#include <cuda_runtime.h>
#include <gtest/gtest.h>

//////////////////////////////////////////////////////////////////////////////
// CUDA assertions

#define CUASSERT_NOERR(expr) ASSERT_EQ((expr), cudaSuccess)

#define CUEXPECT_NOERR(expr) EXPECT_EQ((expr), cudaSuccess)

//////////////////////////////////////////////////////////////////////////////
// Converts values to printable integers

template <typename T>
__host__ __device__ inline int ToPrintable(const T& val)
{
    return (int)val;
}

template <>
__host__ __device__ inline int ToPrintable<float4>(const float4& val)
{
    return (int)val.x;
}

template <>
__host__ __device__ inline int ToPrintable<int3>(const int3& val)
{
    return (int)val.x;
}

//////////////////////////////////////////////////////////////////////////////
// Initializes values for various types

template <typename T>
__host__ __device__  inline T Initialize(int value)
{
    return T(value);
}

template <>
__host__ __device__ inline int3 Initialize<int3>(int value)
{
    return make_int3(value, value, value);
}

template <>
__host__ __device__ inline float4 Initialize<float4>(int value)
{
    return make_float4((float)value, (float)value, (float)value, (float)value);
}

__host__ __device__ inline bool operator==(const int3& a, const int3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__ inline int3& operator+=(int3& lhs, const int3& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__host__ __device__ inline float4& operator+=(float4& lhs, const float4& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

#endif // __CUDA_GTEST_UTILS_H_
