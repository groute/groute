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

#ifndef __GROUTE_CUDA_UTILS_HPP_
#define __GROUTE_CUDA_UTILS_HPP_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace groute
{
    static void HandleError(const char *file, int line, cudaError_t err)
    {
        printf("ERROR in %s:%d: %s (%d)\n", file, line, 
               cudaGetErrorString(err), err);
        exit(1);
    }

    static void HandleError(const char *file, int line, CUresult err)
    {
        const char *err_str;
        cuGetErrorString(err, &err_str);

        printf("ERROR in %s:%d: %s (%d)\n", file, line, 
               err_str == nullptr ? "UNKNOWN ERROR VALUE" : err_str, err);
        exit(1);
    }

    // CUDA assertions
#define GROUTE_CUDA_CHECK(err) do {                                  \
    cudaError_t errr = (err);                                        \
    if(errr != cudaSuccess)                                          \
    {                                                                \
        ::groute::HandleError(__FILE__, __LINE__, errr);             \
    }                                                                \
} while(0)

#define GROUTE_CUDA_DAPI_CHECK(err) do {                             \
    CUresult errr = (err);                                           \
    if(errr != ::CUDA_SUCCESS)                                       \
    {                                                                \
        ::groute::HandleError(__FILE__, __LINE__, errr);             \
    }                                                                \
} while(0)

}  // namespace groute

#endif  // __GROUTE_CUDA_UTILS_HPP_
