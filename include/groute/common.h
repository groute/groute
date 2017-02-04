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

#ifndef __GROUTE_COMMON_H
#define __GROUTE_COMMON_H

#include <map>
#include <future>
#include <vector>
#include <climits>
#include <type_traits>


namespace {
    static inline __host__ __device__ size_t round_up(
        size_t numerator, size_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    // Adapted from http://stackoverflow.com/questions/466204/rounding-up-to-nearest-power-of-2
    template <typename UnsignedType>
    UnsignedType next_power_2(UnsignedType v) {
        static_assert(std::is_unsigned<UnsignedType>::value, "Only works for unsigned types");
        --v;
        for (int i = 1; i < sizeof(v) * CHAR_BIT; i *= 2) {
            v |= v >> i;
        }
        return ++v;
    }

    inline std::vector<int> range(int count, int from = 0)
    {
        std::vector<int> vec(count);
        for (size_t i = 0; i < count; i++)
        {
            vec[i] = from + i;
        }

        return std::move(vec);
    }
}



namespace groute {

    /**
    * @brief Device related metadata  
    */
    class Device
    {
    public:
        Device() = delete;

        enum : int
        {
            Null = INT32_MIN,
            Host = -1
        };

        static bool IsHost(int dev) { return dev == Host; }
        static bool IsNull(int dev) { return dev < Host; }
    };

    typedef int device_t;

    typedef device_t Endpoint; 

    typedef std::map<device_t, std::vector<device_t>> RoutingTable;

    enum LaneType
    {
        In, Out, Internal
    };

    typedef std::pair<device_t, LaneType> LaneIdentifier;

    struct Lane
    {
        device_t src_dev;
        device_t dst_dev;

        Lane() : src_dev(Device::Null), dst_dev(Device::Null) { }
        Lane(device_t src_dev, device_t dst_dev) : src_dev(src_dev), dst_dev(dst_dev) { }

        LaneIdentifier GetIdentifier() const
        {
            assert(!Device::IsNull(src_dev));
            assert(!Device::IsNull(dst_dev));

            device_t dev;
            LaneType type;

            if (src_dev == dst_dev) // internal
            {
                dev = src_dev;
                type = Internal;
            }

            else if (Device::IsHost(src_dev)) // host -> gpu
            {
                dev = dst_dev;
                type = In;
            }

            else // gpu -> host / peers
            {
                dev = src_dev;
                type = Out;
            }

            return std::make_pair(dev, type);
        }
    };

    /**
    * @brief Represents a raw memory buffer on some device
    */
    template <typename T>
    class Buffer
    {
    private:
        T* const m_ptr;
        const size_t m_size;

    public:
        Buffer(T* ptr, size_t size) : m_ptr(ptr), m_size(size) { }
        T* GetPtr() const   { return m_ptr; }
        size_t GetSize() const    { return m_size; }
    };

    /**
    * @brief Represents a segment copy of valid data (for some known datum)
    */
    template <typename T>
    class Segment
    {
    private:
        T* m_segment_ptr;
        size_t m_total_size;
        size_t m_segment_size;
        size_t m_segment_offset;

    public:
        Segment(T* segment_ptr, size_t total_size, size_t segment_size, size_t segment_offset, void* metadata = nullptr) :
            m_segment_ptr(segment_ptr), m_total_size(total_size),
            m_segment_size(segment_size), m_segment_offset(segment_offset), metadata(metadata)
        {

        }

        Segment(T* segment_ptr, size_t total_size, void* metadata = nullptr) :
            m_segment_ptr(segment_ptr), m_total_size(total_size),
            m_segment_size(total_size), m_segment_offset(0), metadata(metadata)
        {

        }

        Segment() : m_segment_ptr(nullptr), m_total_size(0), m_segment_size(0), m_segment_offset(0), metadata(nullptr)
        {

        }

        void* metadata; // a metadata field for user customization  

        /// @brief is the segment empty
        bool Empty() const { return m_segment_size == 0; }

        /// @brief a pointer to segment start
        T* GetSegmentPtr() const   { return m_segment_ptr; }

        /// @brief The total size of the source datum
        size_t GetTotalSize() const    { return m_total_size; }

        /// @brief The size of the segment  
        size_t GetSegmentSize() const    { return m_segment_size; }

        /// @brief The offset within the original buffer   
        size_t GetSegmentOffset() const   { return m_segment_offset; }

        /// @brief Builds a sub-segment out of this segment 
        Segment<T> GetSubSegment(size_t relative_offset, size_t sub_segment_size) const
        {
#ifndef NDEBUG
            if (relative_offset > m_segment_size ||
                sub_segment_size > m_segment_size - relative_offset)
                throw std::exception(); // out of segment range
#endif
            return Segment<T>(
                m_segment_ptr + relative_offset, m_total_size, sub_segment_size, m_segment_offset + relative_offset, metadata);
        }

        Segment<T> GetFirstSubSegment(size_t max_subseg_size) const
        {
            if (max_subseg_size == 0) return { *this };

            return GetSubSegment(0, (size_t)((max_subseg_size) > m_segment_size ? (m_segment_size) : max_subseg_size));
        }

        std::vector< Segment<T> > Split(size_t max_subseg_size) const
        {
            if (max_subseg_size == 0) return { *this };

            std::vector< Segment<T> > subsegs;
            subsegs.reserve(round_up(m_segment_size, max_subseg_size));

            size_t pos = 0;

            while (pos < m_segment_size)
            {
                subsegs.push_back(
                    GetSubSegment(pos, (size_t)((pos + max_subseg_size) > m_segment_size ? (m_segment_size - pos) : max_subseg_size)));

                pos += max_subseg_size;
            }

            return subsegs;
        }
    };

    
    template<typename Future>
    bool is_ready(const Future& f)
    {
#ifdef WIN32
        return f._Is_ready();
#else
        return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
#endif
    }

    template<typename T>
    std::shared_future<T> completed_future(const T& val)
    {
        std::promise<T> prom;
        std::shared_future<T> fut = prom.get_future();
        prom.set_value(val);
        return fut;
    }

    // workaround for C++11
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    
}

#endif // __GROUTE_COMMON_H
