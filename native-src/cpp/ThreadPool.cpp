//
// Created by ghima on 12-11-2025.
//
#include "ThreadPool.h"

namespace ip {
    ThreadPool::ThreadPool(uint32_t size) {
        for (int i = 0; i < size; i++) {
            m_threads.emplace_back([this]() -> void {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> u_lock{mutex_};
                        m_cv.wait(u_lock, [this]() -> bool {
                            return m_stop || !this->m_tasks.empty();
                        });
                        if (m_tasks.empty() && m_stop) return;
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    task();

                }
            });
        }
    }


    ThreadPool::~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock_{mutex_};
            m_stop = true;
        }
        m_cv.notify_all();

        for (std::thread &thread: m_threads) {
            if (thread.joinable()) thread.join();
        }
    }

    void ThreadPool::joinAll() {
        {
            std::lock_guard<std::mutex> lock_{mutex_};
            m_stop = true;
        }
        m_cv.notify_all();

        for (std::thread &thread: m_threads) {
            if (thread.joinable()) thread.join();
        }
    }
}