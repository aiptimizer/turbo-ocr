#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "turbo_ocr/common/errors.h"

namespace turbo_ocr::server {

/// Thread pool for offloading blocking HTTP handler work from Drogon's
/// event-loop threads.
///
/// submit() is non-blocking: it always enqueues and returns immediately.
/// Backpressure is handled downstream — the PipelineDispatcher rejects
/// with PoolExhaustedError when the GPU queue is full, and the
/// run_with_error_handling wrapper converts that to a 503 response.
///
/// Queue depth is bounded (default 8192) as a safety net against memory
/// exhaustion.  When full, submit() throws PoolExhaustedError.
class WorkPool {
public:
  explicit WorkPool(int num_threads, size_t max_depth = 8192)
      : max_depth_(max_depth) {
    workers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] {
              return stop_.load(std::memory_order_acquire) || !queue_.empty();
            });
            if (queue_.empty()) {
              if (stop_.load(std::memory_order_acquire)) return;
              continue;
            }
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  }

  ~WorkPool() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  WorkPool(const WorkPool &) = delete;
  WorkPool &operator=(const WorkPool &) = delete;

  void submit(std::function<void()> fn) {
    {
      std::lock_guard lock(mutex_);
      if (queue_.size() >= max_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (work queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push(std::move(fn));
    }
    cv_.notify_one();
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  size_t max_depth_;
};

} // namespace turbo_ocr::server
