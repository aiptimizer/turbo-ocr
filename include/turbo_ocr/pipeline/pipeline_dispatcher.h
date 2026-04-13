#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "turbo_ocr/pipeline/gpu_pipeline_pool.h"

namespace turbo_ocr::pipeline {

/// Work-queue dispatcher that keeps GPU pipelines permanently busy.
///
/// Each worker thread owns one GpuPipelineEntry and pulls tasks from a
/// shared FIFO queue.  Unlike the acquire/release PipelinePool pattern,
/// the GPU never idles waiting for HTTP round-trip overhead — while one
/// request's response is being serialised and sent, the worker is already
/// processing the next queued image.
class PipelineDispatcher {
public:
  explicit PipelineDispatcher(std::vector<std::unique_ptr<GpuPipelineEntry>> entries) {
    size_t n = entries.size();
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      workers_.emplace_back([this, entry = std::move(entries[i])]() mutable {
        while (true) {
          WorkFn work;
          {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return stop_.load(std::memory_order_acquire) || !queue_.empty(); });
            if (queue_.empty()) {
              if (stop_.load(std::memory_order_acquire)) return;
              continue;
            }
            work = std::move(queue_.front());
            queue_.pop();
          }
          work(*entry);
        }
      });
    }
  }

  ~PipelineDispatcher() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  PipelineDispatcher(const PipelineDispatcher &) = delete;
  PipelineDispatcher &operator=(const PipelineDispatcher &) = delete;

  /// Submit work that runs on a GPU worker thread.  Returns a future
  /// whose value is whatever the callable returns.
  ///
  /// The callable signature must be:  R fn(GpuPipelineEntry &)
  template <typename F>
  auto submit(F &&fn) -> std::future<std::invoke_result_t<F, GpuPipelineEntry &>> {
    using R = std::invoke_result_t<F, GpuPipelineEntry &>;
    auto task = std::make_shared<std::packaged_task<R(GpuPipelineEntry &)>>(
        std::forward<F>(fn));
    auto future = task->get_future();
    {
      std::unique_lock lock(mutex_);
      if (queue_.size() >= max_queue_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (GPU queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push([task = std::move(task)](GpuPipelineEntry &e) { (*task)(e); });
    }
    cv_.notify_one();
    return future;
  }

  [[nodiscard]] size_t worker_count() const noexcept { return workers_.size(); }

private:
  using WorkFn = std::function<void(GpuPipelineEntry &)>;

  std::queue<WorkFn> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  size_t max_queue_depth_ = 4096;
};

/// Factory: create, init, warmup GPU pipelines and wrap in a dispatcher.
[[nodiscard]] inline std::unique_ptr<PipelineDispatcher> make_pipeline_dispatcher(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "") {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Dispatcher] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    auto pipeline = std::make_unique<OcrPipeline>();
    if (!pipeline->init(det_model, rec_model, rec_dict, cls_model)) {
      std::cerr << std::format("[Dispatcher] Failed to init GPU pipeline {} of {}", i, pool_size) << '\n';
      continue;
    }
    if (!layout_model.empty()) {
      if (!pipeline->load_layout_model(layout_model)) {
        throw turbo_ocr::ModelLoadError(std::format(
            "[Dispatcher] Failed to load layout model for pipeline {} of {}",
            i, pool_size));
      }
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    entries.push_back(std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(
        std::format("[Dispatcher] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Warming up {} pipelines...", entries.size()) << '\n';
  for (auto &e : entries) {
    e->pipeline->warmup_gpu(e->stream);
  }
  std::cout << "Pipeline warmup complete." << '\n';

  return std::make_unique<PipelineDispatcher>(std::move(entries));
}

} // namespace turbo_ocr::pipeline
