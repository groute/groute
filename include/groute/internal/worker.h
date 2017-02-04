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

#ifndef __GROUTE_WORKER_H
#define __GROUTE_WORKER_H

#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <memory>

namespace groute {
    namespace internal {

        /**
        * @brief Represents a CPU thread barrier
        * @note The barrier automatically resets after all threads are synced
        */
        class Barrier
        {
        private:
            std::mutex m_mutex;
            std::condition_variable m_cv;

            size_t m_count;
            const size_t m_initial;

            enum State : unsigned char {
                Up, Down
            };
            State m_state;

        public:
            explicit Barrier(std::size_t count) : m_count{ count },
                m_initial{ count }, m_state{ State::Down } { }

            /// Blocks until all N threads reach here
            void Sync()
            {
                std::unique_lock<std::mutex> lock{ m_mutex };

                if (m_state == State::Down)
                {
                    // Counting down the number of syncing threads
                    if (--m_count == 0) {
                        m_state = State::Up;
                        m_cv.notify_all();
                    }
                    else {
                        m_cv.wait(lock, [this] {
                            return m_state == State::Up; });
                    }
                }

                else // (m_state == State::Up)
                {
                    // Counting back up for Auto reset
                    if (++m_count == m_initial) {
                        m_state = State::Down;
                        m_cv.notify_all();
                    }
                    else {
                        m_cv.wait(lock, [this] {
                            return m_state == State::Down; });
                    }
                }
            }
        };

        struct IWork
        {
            virtual ~IWork() { }
            virtual void operator()(Barrier *barrier) = 0;
        };

        struct EmptyWork : public IWork
        {
            virtual ~EmptyWork() {}
            virtual void operator()(Barrier *barrier) override
            {
                barrier->Sync();
                barrier->Sync();
            }
        };

        /**
        * @brief Represents a worker thread looping over queued Work
        * @note Assuming only a single thread is interacting with this object
        */
        template<class Work = IWork>
        class Worker
        {
        private:
            /// The queue for sending work from the Owner thread
            std::queue< std::shared_ptr<Work> > m_queue;

            /// The mutex for syncing all operations
            std::mutex m_mutex;

            /// A condition variable for signaling operations from Owner thread
            /// to Worker thread
            /// (work, exit)
            std::condition_variable m_send_cv;

            /// A condition variable for signaling that all work in the queue 
            /// was performed
            std::condition_variable m_sync_cv;

            /// A barrier for syncing with other Workers
            std::shared_ptr<Barrier> m_barrier;

            /// The worker thread
            std::thread m_workerThread;

            bool m_exit;

            void Loop()
            {
                OnStart();

                while (true) {
                    std::shared_ptr<Work> work;

                    { // Lock block
                        std::unique_lock<std::mutex> lock(m_mutex);

                        // We loop over the queue even if exit is true
                        if (m_queue.empty()) {
                            // Signaling that all Work is done
                            m_sync_cv.notify_all();

                            // Queue is empty and we were signaled to exit. We 
                            // quit
                            if (m_exit) break;

                            // Waiting for (work|exit)
                            m_send_cv.wait(lock, [this]() {
                                return m_exit || !m_queue.empty(); });

                            // Exit was signaled
                            if (m_exit && m_queue.empty()) break;
                        }

                        work = m_queue.front();
                    }

                    // Performing the work out of the Lock block
                    OnBeginWork(work);
                    (*work)(m_barrier.get());

                    { // Lock block
                        std::unique_lock<std::mutex> lock(m_mutex);

                        // Remove the work only after the invocation has been performed
                        // (causes Sync() to wait for final job to complete, not start)
                        m_queue.pop();
                    }
                }
            }
        protected:
            typedef Work WorkType;

            /// Called by the worker thread on start
            virtual void OnStart() { }

            /// Called by the worker thread before beginning
            virtual void OnBeginWork(std::shared_ptr<WorkType> work) { }

        public:
            explicit Worker(std::shared_ptr<Barrier> barrier)
                : m_barrier(barrier), m_workerThread(), m_exit(false)
            {

            }

            virtual ~Worker()
            {
                Exit();
            }

            void Run()
            {
                m_workerThread = std::thread([this]() { Loop(); });
            }

            /// @brief Exit the Worker thread 
            /// @note Worker finishes all the current Work in the queue before 
            /// exiting
            /// @note Caller thread joins the Worker thread until it exists 
            void Exit()
            {
                { // Lock block
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_exit) return;

                    m_exit = true;
                    m_send_cv.notify_one();
                }

                // We must unlock before trying to join..
                m_workerThread.join();
            }

            virtual void Enqueue(std::shared_ptr<Work> work)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_queue.push(work);
                m_send_cv.notify_one();
            }

            void Sync()
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_sync_cv.wait(lock, [this]() { return m_queue.empty(); });
            }
        };

    } // namespace multi

} // namespace groute

#endif // __GROUTE_WORKER_H
