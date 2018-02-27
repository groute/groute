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

#ifndef __GROUTE_WORK_SOURCE_H
#define __GROUTE_WORK_SOURCE_H

namespace groute {
    namespace dev {

        /*
        //
        // WorkSource classes (device):
        //
        API:
        struct WorkSource
        {
            __device__ uint32_t get_size() const { return ...; }
            __device__ T get_work(uint32_t i) const { return ...; }
        };
        */


        /*
        * @brief A work source based on a device array and size
        */
        template<typename T>
        struct WorkSourceArray
        {
        private:
            T* work_ptr;
            uint32_t work_size;

        public:
            __host__ __device__ WorkSourceArray(T* work_ptr, uint32_t work_size) :
                work_ptr(work_ptr), work_size(work_size) { }

            __device__ __forceinline__ T get_work(uint32_t i) const { return work_ptr[i]; }
            __host__ __device__ __forceinline__ uint32_t get_size() const { return work_size; }
        };

        /*
        * @brief A work source based on two device arrays + sizes
        */
        template<typename T>
        struct WorkSourceTwoArrays
        {
        private:
            T* work_ptr1, *work_ptr2;
            uint32_t work_size1, work_size2;

        public:
            WorkSourceTwoArrays(T* work_ptr1, uint32_t work_size1, T* work_ptr2, uint32_t work_size2) :
                work_ptr1(work_ptr1), work_size1(work_size1), work_ptr2(work_ptr2), work_size2(work_size2) { }

            __device__ __forceinline__ T get_work(uint32_t i) { return i < work_size1 ? work_ptr1[i] : work_ptr2[i-work_size1]; }
            __host__ __device__ __forceinline__ uint32_t get_size() const { return work_size1 + work_size2; }
        };

        /*
        * @brief A work source based on a device array and a device counter
        */
        template<typename T>
        struct WorkSourceCounter
        {
        private:
            T* work_ptr;
            uint32_t* work_counter;

        public:
            WorkSourceCounter(T* work_ptr, uint32_t* work_counter) :
                work_ptr(work_ptr), work_counter(work_counter) { }

            __device__ __forceinline__ T get_work(uint32_t i) { return work_ptr[i]; }
            __device__ __forceinline__ uint32_t get_size() const { return *work_counter; }
        };

        /*
        * @brief A work source based on a discrete T range [range, range+size)
        */
        template<typename T>
        struct WorkSourceRange
        {
        private:
            T m_range_start;
            uint32_t m_range_size;

        public:
            __host__ __device__ WorkSourceRange(T range_start, uint32_t range_size) :
                m_range_start(range_start), m_range_size(range_size) { }

            __device__ __forceinline__ T get_work(uint32_t i) const { return (T)(m_range_start + i); }
            __host__ __device__ __forceinline__ uint32_t get_size() const { return m_range_size; }
        };
    }
}

#endif // __GROUTE_WORK_SOURCE_H
