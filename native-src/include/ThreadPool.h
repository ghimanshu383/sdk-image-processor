//
// Created by ghima on 12-11-2025.
//

#ifndef OSFEATURENDKDEMO_THREADPOOL_H
#define OSFEATURENDKDEMO_THREADPOOL_H

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace ip {
    class ThreadPool {
    private:
        std::vector<std::thread> m_threads{};
        std::queue<std::function<void()>> m_tasks{};
        std::condition_variable m_cv{};
        std::mutex mutex_;
        bool m_stop = false;

    public:
        explicit ThreadPool(uint32_t size);

        template<typename T>
        void enqueue_task(T &&task) {
            {
                std::lock_guard<std::mutex> lock{mutex_};
                m_tasks.emplace(std::forward<T>(task));
            }
            m_cv.notify_one();
        }

        ~ThreadPool();
        void joinAll();
    };
}
#endif //OSFEATURENDKDEMO_THREADPOOL_H
