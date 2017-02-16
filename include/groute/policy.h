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

#ifndef __GROUTE_POLICY_H
#define __GROUTE_POLICY_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>

#include <groute/router.h>


namespace groute {
    namespace router {

        /**
        * @brief A general purpose Policy object based on a routing table 
        */
        class Policy : public IPolicy
        {
        private:
            RoutingTable m_routing_table;
            RouteStrategy m_strategy;

        public:
            Policy(const RoutingTable& routing_table, RouteStrategy strategy = Availability) : m_routing_table(routing_table), m_strategy(strategy)
            {
            }

            RoutingTable GetRoutingTable() override
            {
                return m_routing_table;
            }

            Route GetRoute(Endpoint src, void* message_metadata) override
            {
                assert(m_routing_table.find(src) != m_routing_table.end());

                return Route(m_routing_table.at(src), m_strategy);
            }

            static std::shared_ptr<IPolicy> CreateBroadcastPolicy(Endpoint src, const EndpointList& dst_endpoints)
            {
                RoutingTable topology;
                topology[src] = dst_endpoints;
                return std::make_shared<Policy>(topology, Broadcast);
            }

            static std::shared_ptr<IPolicy> CreateScatterPolicy(Endpoint src, const EndpointList& dst_endpoints)
            {
                RoutingTable topology;
                topology[src] = dst_endpoints;
                return std::make_shared<Policy>(topology, Availability);
            }

            static std::shared_ptr<IPolicy> CreateP2PPolicy(Endpoint src, Endpoint dst)
            {
                RoutingTable topology;
                topology[src] = { dst };
                return std::make_shared<Policy>(topology, Availability);
            }

            static std::shared_ptr<IPolicy> CreateGatherPolicy(Endpoint dst, const EndpointList& src_endpoints)
            {
                RoutingTable topology;
                for (const auto& src : src_endpoints)
                    topology[src] = { dst };
                return std::make_shared<Policy>(topology, Availability);
            }

            static std::shared_ptr<IPolicy> CreateOneWayReductionPolicy(int ndevs)
            {
                assert(ndevs > 0);

                // Each device N can send to devices [0...N-1]

                RoutingTable routing_table;

                for (Endpoint::identity_type i = 0; i < ndevs; i++)
                {
                    routing_table[i] = Endpoint::Range(i);
                }
                routing_table[0].push_back(Endpoint::HostEndpoint());

                return std::make_shared<Policy>(routing_table, Availability);
            }

            static std::shared_ptr<IPolicy> CreateTreeReductionPolicy(int ndevs)
            {
                assert(ndevs > 0);

                RoutingTable routing_table;


                // 0
                // 1 -> 0
                // 2 -> 0
                // 3 -> 2
                // 4 -> 0
                // 5 -> 4
                // 6 -> 4
                // 7 -> 6
                // ..

                unsigned int p = next_power_2((unsigned int)ndevs) / 2;
                unsigned int stride = 1;

                while (p > 0)
                {
                    for (unsigned int i = 0; i < p; i++)
                    {
                        int to = stride*(2 * i);
                        int from = stride*(2 * i + 1);

                        from = std::min(ndevs - 1, from);
                        if (from <= to) continue;

                        routing_table[from].push_back(to);
                    }

                    p /= 2;
                    stride *= 2;
                }

                // add host as a receiver for the drain device
                routing_table[0].push_back(Endpoint::HostEndpoint());

                return std::make_shared<Policy>(routing_table, Availability);
            }

            static std::shared_ptr<IPolicy> CreateRingPolicy(int ndevs)
            {
                assert(ndevs > 0);

                RoutingTable routing_table;

                for (device_t i = 0; i < ndevs; i++)
                {
                    routing_table[i] = { (i + 1) % ndevs };
                }

                // Instead of pushing to GPU 0, we push tasks to the first available device,
                // this is beneficial for the case where the first device is already utilized
                // with a prior task.
                routing_table[Endpoint::HostEndpoint()] = Endpoint::Range(ndevs); // For initial work from host

                return std::make_shared<Policy>(routing_table, Availability);
            }
        };
    }
}

#endif // __GROUTE_POLICY_H
