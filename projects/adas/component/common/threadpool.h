#ifndef ADAS_COMMON_THREADPOOL_H_
#define ADAS_COMMON_THREADPOOL_H_

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "cyber/base/bounded_queue.h"

using namespace apollo::cyber::base;

namespace watrix
{
    namespace projects
    {
        namespace adas
        {

            class ThreadPool
            {
            public:
                explicit ThreadPool(std::size_t thread_num, std::size_t max_task_num = 30);
                explicit ThreadPool(std::size_t thread_num, std::function<void*()> init_func, std::size_t max_task_num = 30);

                template <typename F, typename... Args>
                auto Enqueue(F &&f, Args &&... args)
                    -> std::future<typename std::result_of<F(Args...)>::type>;

                ~ThreadPool();
                std::map<std::thread::id,void *> ptr_map_;
            private:
                std::vector<std::thread> workers_;
                BoundedQueue<std::function<void()>> task_queue_;
                std::atomic_bool stop_;
                std::function<void*()> init_func_;
                std::mutex init_mutex_;
            };

            inline ThreadPool::ThreadPool(std::size_t thread_num, std::size_t max_task_num)
                : stop_(false)
            {
                if (!task_queue_.Init(max_task_num, new BlockWaitStrategy()))
                {
                    throw std::runtime_error("Task queue init failed.");
                }
                for (size_t i = 0; i < thread_num; ++i)
                {
                    workers_.emplace_back([this] {
                        while (!stop_)
                        {
                            std::function<void()> task;
                            if (task_queue_.WaitDequeue(&task))
                            {
                                task();
                            }
                        }
                    });
                }
            }

            inline ThreadPool::ThreadPool(std::size_t thread_num, std::function<void*()> init_func, std::size_t max_task_num)
                : stop_(false)
            {
                init_func_ = init_func;
                if (!task_queue_.Init(max_task_num, new BlockWaitStrategy()))
                {
                    throw std::runtime_error("Task queue init failed.");
                }

                for (size_t i = 0; i < thread_num; ++i)
                {
                    workers_.emplace_back([this,i] {
                        // init_mutex_.lock();
                        this->ptr_map_[std::this_thread::get_id()] = this->init_func_();
                        // init_mutex_.unlock();

                        while (!stop_)
                        {
                            std::function<void()> task;
                            if (task_queue_.WaitDequeue(&task))
                            {
                                task();
                            }
                        }
                    });
                }
            }

            // before using the return value, you should check value.valid()
            template <typename F, typename... Args>
            auto ThreadPool::Enqueue(F &&f, Args &&... args)
                -> std::future<typename std::result_of<F(Args...)>::type>
            {
                using return_type = typename std::result_of<F(Args...)>::type;

                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...));

                std::future<return_type> res = task->get_future();

                // don't allow enqueueing after stopping the pool
                if (stop_)
                {
                    return std::future<return_type>();
                }
                task_queue_.Enqueue([task]() { (*task)(); });
                return res;
            }

            // the destructor joins all threads
            inline ThreadPool::~ThreadPool()
            {
                if (stop_.exchange(true))
                {
                    return;
                }
                task_queue_.BreakAllWait();
                for (std::thread &worker : workers_)
                {
                    worker.join();
                }
                for(auto & ptr :ptr_map_ )
                {
                    delete ptr.second;
                }
                ptr_map_.clear();

            }

        } // namespace adas
    }     // namespace projects
} // namespace watrix

#endif