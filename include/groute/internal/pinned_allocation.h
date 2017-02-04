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

#ifndef __GROUTE_PINNED_ALLOCATION_H
#define __GROUTE_PINNED_ALLOCATION_H

namespace groute {
    template <typename T, unsigned int FLAGS = 0U>
    class pinned_allocator : public std::allocator<T>
    {
    public:
        typedef size_t size_type;
        typedef T* pointer;
        typedef const T* const_pointer;

        template<typename _Other>
        struct rebind
        {
            typedef pinned_allocator<_Other> other;
        };

        pointer allocate(size_type n, const void *hint = nullptr)
        {
            if (n == 0)
                return nullptr;

            pointer p;
            cudaError_t err = cudaMallocHost(&p, n * sizeof(T), FLAGS);
            if (err == cudaSuccess)
                return p;
            return nullptr;
        }

        void deallocate(pointer p, size_type n)
        {
            cudaFreeHost(p);
            return;
        }

        pinned_allocator() throw() : std::allocator<T>() { }
        pinned_allocator(const pinned_allocator &a) throw() 
            : std::allocator<T>(a) { }

        template <class U>
        pinned_allocator(const pinned_allocator<U> &a) throw() 
            : std::allocator<T>(a) { }

        ~pinned_allocator() throw() { }
    };

    template<typename T, unsigned int FLAGS = 0U>
    using pinned_vector = std::vector< T, pinned_allocator<T, FLAGS> >;

} // namespace groute

#endif // __GROUTE_PINNED_ALLOCATION_H
