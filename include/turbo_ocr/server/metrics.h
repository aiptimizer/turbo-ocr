#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>

#ifndef USE_CPU_ONLY
#include <cuda_runtime_api.h>
#endif
#include <drogon/HttpAppFramework.h>

namespace turbo_ocr::server {

/// Prometheus-compatible metrics with zero external dependencies.
/// Thread-safe via atomics (counters) and a mutex (histogram buckets).
class Metrics {
public:
  // ── Route index (used as label dimension) ──────────────────────────────

  enum Route : int {
    kOcr = 0,
    kOcrRaw,
    kOcrBatch,
    kOcrPixels,
    kOcrPdf,
    kHealth,
    kRouteCount
  };

  static constexpr const char *route_name(Route r) {
    constexpr const char *names[] = {
        "/ocr", "/ocr/raw", "/ocr/batch", "/ocr/pixels", "/ocr/pdf", "/health"};
    return names[r];
  }

  static Route route_from_path(std::string_view path) {
    if (path == "/ocr")        return kOcr;
    if (path == "/ocr/raw")    return kOcrRaw;
    if (path == "/ocr/batch")  return kOcrBatch;
    if (path == "/ocr/pixels") return kOcrPixels;
    if (path == "/ocr/pdf")    return kOcrPdf;
    return kHealth;
  }

  // ── Recording ──────────────────────────────────────────────────────────

  void record_request(Route route, int http_status, double duration_s) {
    auto &r = routes_[route];
    if (http_status >= 200 && http_status < 300)
      r.ok.fetch_add(1, std::memory_order_relaxed);
    else if (http_status >= 400 && http_status < 500)
      r.client_err.fetch_add(1, std::memory_order_relaxed);
    else if (http_status >= 500)
      r.server_err.fetch_add(1, std::memory_order_relaxed);

    // Histogram: increment the first bucket that fits (serializer accumulates).
    // Values above all buckets only appear in _count/_sum (+Inf bucket).
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (duration_s <= kBuckets[i]) {
        r.hist_buckets[i].fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
    r.hist_sum.fetch_add(
        static_cast<uint64_t>(duration_s * 1e6), std::memory_order_relaxed);
    r.hist_count.fetch_add(1, std::memory_order_relaxed);
  }

  void record_request_size(size_t bytes) {
    request_bytes_total_.fetch_add(bytes, std::memory_order_relaxed);
    request_count_sized_.fetch_add(1, std::memory_order_relaxed);
  }

  void record_pool_exhaustion() {
    pool_exhaustions_.fetch_add(1, std::memory_order_relaxed);
  }

  void set_pool_size(int n) {
    pool_size_.store(n, std::memory_order_relaxed);
  }

  void set_gpu_vram_used_bytes(size_t bytes) {
    gpu_vram_used_.store(bytes, std::memory_order_relaxed);
  }

  void set_gpu_vram_total_bytes(size_t bytes) {
    gpu_vram_total_.store(bytes, std::memory_order_relaxed);
  }

  // ── Prometheus text exposition ─────────────────────────────────────────

  [[nodiscard]] std::string serialize() const {
    std::string out;
    out.reserve(4096);

    // requests_total
    out += "# HELP turbo_ocr_requests_total Total HTTP requests by route and status.\n";
    out += "# TYPE turbo_ocr_requests_total counter\n";
    for (int i = 0; i < kRouteCount; ++i) {
      auto &r = routes_[i];
      auto name = route_name(static_cast<Route>(i));
      append_counter(out, "turbo_ocr_requests_total", name, "2xx",
                     r.ok.load(std::memory_order_relaxed));
      append_counter(out, "turbo_ocr_requests_total", name, "4xx",
                     r.client_err.load(std::memory_order_relaxed));
      append_counter(out, "turbo_ocr_requests_total", name, "5xx",
                     r.server_err.load(std::memory_order_relaxed));
    }

    // request_duration_seconds (histogram)
    out += "# HELP turbo_ocr_request_duration_seconds Request latency histogram.\n";
    out += "# TYPE turbo_ocr_request_duration_seconds histogram\n";
    for (int i = 0; i < kRouteCount; ++i) {
      if (i == kHealth) continue;  // skip health from histogram
      auto &r = routes_[i];
      auto name = route_name(static_cast<Route>(i));
      uint64_t cumulative = 0;
      for (size_t b = 0; b < kNumBuckets; ++b) {
        cumulative += r.hist_buckets[b].load(std::memory_order_relaxed);
        char le_buf[32];
        std::snprintf(le_buf, sizeof(le_buf), "%.3f", kBuckets[b]);
        append_histogram_bucket(out, name, le_buf, cumulative);
      }
      uint64_t count = r.hist_count.load(std::memory_order_relaxed);
      append_histogram_bucket(out, name, "+Inf", count);
      double sum = static_cast<double>(
          r.hist_sum.load(std::memory_order_relaxed)) / 1e6;
      append_histogram_summary(out, name, sum, count);
    }

    // pool_exhaustions_total
    out += "# HELP turbo_ocr_pool_exhaustions_total Times pipeline pool was full (503).\n";
    out += "# TYPE turbo_ocr_pool_exhaustions_total counter\n";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_exhaustions_total %" PRIu64 "\n",
                  pool_exhaustions_.load(std::memory_order_relaxed));
    out += buf;

    // pool_size
    out += "# HELP turbo_ocr_pipeline_pool_size Number of pipeline slots.\n";
    out += "# TYPE turbo_ocr_pipeline_pool_size gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pipeline_pool_size %d\n",
                  pool_size_.load(std::memory_order_relaxed));
    out += buf;

    // GPU VRAM
    size_t vram_used = gpu_vram_used_.load(std::memory_order_relaxed);
    size_t vram_total = gpu_vram_total_.load(std::memory_order_relaxed);
    if (vram_total > 0) {
      out += "# HELP turbo_ocr_gpu_vram_used_bytes GPU memory currently in use.\n";
      out += "# TYPE turbo_ocr_gpu_vram_used_bytes gauge\n";
      std::snprintf(buf, sizeof(buf), "turbo_ocr_gpu_vram_used_bytes %zu\n", vram_used);
      out += buf;
      out += "# HELP turbo_ocr_gpu_vram_total_bytes Total GPU memory.\n";
      out += "# TYPE turbo_ocr_gpu_vram_total_bytes gauge\n";
      std::snprintf(buf, sizeof(buf), "turbo_ocr_gpu_vram_total_bytes %zu\n", vram_total);
      out += buf;
    }

    // Request body sizes
    out += "# HELP turbo_ocr_request_bytes_total Total request body bytes received.\n";
    out += "# TYPE turbo_ocr_request_bytes_total counter\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_request_bytes_total %" PRIu64 "\n",
                  request_bytes_total_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_request_body_avg_bytes Average request body size.\n";
    out += "# TYPE turbo_ocr_request_body_avg_bytes gauge\n";
    uint64_t cnt = request_count_sized_.load(std::memory_order_relaxed);
    uint64_t total_bytes = request_bytes_total_.load(std::memory_order_relaxed);
    double avg = cnt > 0 ? static_cast<double>(total_bytes) / cnt : 0.0;
    std::snprintf(buf, sizeof(buf), "turbo_ocr_request_body_avg_bytes %.0f\n", avg);
    out += buf;

    return out;
  }

  // ── Singleton ──────────────────────────────────────────────────────────

  static Metrics &instance() {
    static Metrics m;
    return m;
  }

private:
  // Histogram bucket boundaries (seconds)
  static constexpr size_t kNumBuckets = 9;
  static constexpr double kBuckets[kNumBuckets] = {
      0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0};

  struct RouteMetrics {
    std::atomic<uint64_t> ok{0};
    std::atomic<uint64_t> client_err{0};
    std::atomic<uint64_t> server_err{0};
    std::array<std::atomic<uint64_t>, kNumBuckets> hist_buckets{};
    std::atomic<uint64_t> hist_sum{0};   // microseconds
    std::atomic<uint64_t> hist_count{0};
  };

  std::array<RouteMetrics, kRouteCount> routes_{};
  std::atomic<uint64_t> pool_exhaustions_{0};
  std::atomic<int> pool_size_{0};
  std::atomic<size_t> gpu_vram_used_{0};
  std::atomic<size_t> gpu_vram_total_{0};
  std::atomic<uint64_t> request_bytes_total_{0};
  std::atomic<uint64_t> request_count_sized_{0};

  // ── Formatting helpers ─────────────────────────────────────────────────

  static void append_counter(std::string &out, const char *metric,
                             const char *route, const char *status,
                             uint64_t val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s{route=\"%s\",status=\"%s\"} %" PRIu64 "\n",
                  metric, route, status, val);
    out += buf;
  }

  static void append_histogram_bucket(std::string &out, const char *route,
                                       const char *le, uint64_t cumulative) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_bucket{route=\"%s\",le=\"%s\"} %" PRIu64 "\n",
        route, le, cumulative);
    out += buf;
  }

  static void append_histogram_summary(std::string &out, const char *route,
                                        double sum, uint64_t count) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_sum{route=\"%s\"} %.6f\n",
        route, sum);
    out += buf;
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_count{route=\"%s\"} %" PRIu64 "\n",
        route, count);
    out += buf;
  }
};

/// Register the /metrics endpoint and automatic per-request recording.
/// Call this BEFORE drogon::app().run().
inline void register_metrics_route() {
  // Endpoint — update GPU VRAM on each scrape (cheap syscall)
  drogon::app().registerHandler(
      "/metrics",
      [](const drogon::HttpRequestPtr &,
         std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
#ifndef USE_CPU_ONLY
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == 0) {
          Metrics::instance().set_gpu_vram_used_bytes(total_mem - free_mem);
          Metrics::instance().set_gpu_vram_total_bytes(total_mem);
        }
#endif
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setBody(Metrics::instance().serialize());
        resp->setContentTypeString(
            "text/plain; version=0.0.4; charset=utf-8");
        callback(resp);
      },
      {drogon::Get});
}

} // namespace turbo_ocr::server
