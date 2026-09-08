#include "turbo_ocr/server/bootstrap/host_allocator.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <fstream>
#include <charconv>
#include <string_view>
#include <thread>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#if defined(__GLIBC__)
#include <sched.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>  // mallopt, malloc_trim
#endif

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/host_image_pool.h"

namespace turbo_ocr::server::bootstrap {

namespace {

// int mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen)
using MallctlFn = int (*)(const char *, void *, size_t *, void *, size_t);

// jemalloc's MALLCTL_ARENAS_ALL: the arena index that addresses every arena
// in "arena.<i>.<cmd>" names. Spelled out here so the server does not need
// jemalloc's headers at build time (it is normally only preloaded).
constexpr const char *kDecayAllArenas = "arena.4096.decay";

MallctlFn find_mallctl() noexcept {
#if defined(_WIN32)
  return nullptr;
#else
  // Both spellings: unprefixed is the distro default, je_ the upstream one.
  for (const char *name : {"mallctl", "je_mallctl"}) {
    if (void *sym = dlsym(RTLD_DEFAULT, name))
      return reinterpret_cast<MallctlFn>(sym);
  }
  return nullptr;
#endif
}

MallctlFn mallctl_fn() noexcept {
  static const MallctlFn fn = find_mallctl();
  return fn;
}

} // namespace

HostAllocator detect_host_allocator() noexcept {
  if (mallctl_fn()) return HostAllocator::Jemalloc;
#if defined(__GLIBC__)
  return HostAllocator::Glibc;
#else
  return HostAllocator::Unknown;
#endif
}

const char *host_allocator_name(HostAllocator a) noexcept {
  switch (a) {
    case HostAllocator::Glibc: return "glibc";
    case HostAllocator::Jemalloc: return "jemalloc";
    case HostAllocator::Unknown: break;
  }
  return "unknown";
}

bool release_idle_host_memory(HostAllocator a) noexcept {
  switch (a) {
    case HostAllocator::Jemalloc:
      if (MallctlFn fn = mallctl_fn())
        return fn(kDecayAllArenas, nullptr, nullptr, nullptr, 0) == 0;
      return false;
    case HostAllocator::Glibc:
#if defined(__GLIBC__)
      (void)malloc_trim(0);  // returns whether anything was released; either way the call succeeded
      return true;
#else
      return false;
#endif
    case HostAllocator::Unknown: break;
  }
  return false;
}

void install_host_image_pool(size_t slots, size_t max_block_bytes,
                             decode::BlockMemory memory) {
  if (env::env_present("TURBO_OCR_DISABLE_HOST_IMAGE_POOL")) {
    TOCR_LOG_INFO("Host image pool disabled by TURBO_OCR_DISABLE_HOST_IMAGE_POOL");
    return;
  }
  auto &pool = decode::HostImagePool::install_default(slots, max_block_bytes, memory);
  TOCR_LOG_INFO("Host image pool installed", "slots", slots,
                "memory", std::string_view(pool.memory_name()),
                "max_block_mb", static_cast<int>(max_block_bytes >> 20),
                "budget_mb", static_cast<int>((slots * max_block_bytes) >> 20),
                "threshold_kb", static_cast<int>(pool.threshold_bytes() >> 10));
}

int glibc_arena_cap(int cores) noexcept { return std::max(8, cores); }

// CPUs this process may actually use: the affinity mask (a cpuset) capped by
// the cgroup CPU quota (`docker --cpus`, Kubernetes limits), never below 1.
// Neither alone is right: a quota leaves the mask untouched and a cpuset
// leaves the quota at "max".
int cpus_from_cgroup_cpu_max(std::string_view cpu_max) noexcept {
  // cgroup v2 cpu.max: "<quota> <period>" or "max <period>"; v1 callers pass
  // the same shape assembled from cfs_quota_us / cfs_period_us.
  const auto sp = cpu_max.find(' ');
  if (sp == std::string_view::npos) return 0;
  const std::string_view quota = cpu_max.substr(0, sp);
  const std::string_view period = cpu_max.substr(sp + 1);
  if (quota == "max") return 0;
  long q = 0, per = 0;
  auto rq = std::from_chars(quota.data(), quota.data() + quota.size(), q);
  auto rp = std::from_chars(period.data(), period.data() + period.size(), per);
  if (rq.ec != std::errc{} || rp.ec != std::errc{} || q <= 0 || per <= 0) return 0;
  return static_cast<int>((q + per - 1) / per);  // ceil: 1.5 CPUs of quota needs 2 arenas' worth
}

namespace {

int cgroup_quota_cpus() noexcept {
#if defined(__linux__)
  std::string text;
  auto read = [&](const char *path) {
    std::ifstream in(path);
    return in && std::getline(in, text);
  };
  if (read("/sys/fs/cgroup/cpu.max")) return cpus_from_cgroup_cpu_max(text);  // v2
  std::string quota, period;                                                   // v1
  if (read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")) {
    quota = text;
    if (read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")) period = text;
    if (!quota.empty() && !period.empty() && quota != "-1")
      return cpus_from_cgroup_cpu_max(quota + " " + period);
  }
#endif
  return 0;
}

}  // namespace

int usable_cpus() noexcept {
  int n = static_cast<int>(std::thread::hardware_concurrency());
#if defined(__GLIBC__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0) n = CPU_COUNT(&set);
#endif
  if (const int quota = cgroup_quota_cpus(); quota > 0 && quota < n) n = quota;
  return std::max(1, n);
}

// The operator's own arena setting wins over ours: glibc reads MALLOC_ARENA_MAX
// and glibc.malloc.arena_max from GLIBC_TUNABLES before main(), and a
// deployment that set either did so deliberately. Only a value glibc itself
// would accept counts (a positive integer); a typo does not silently lift the cap.
bool operator_set_arena_max() noexcept {
  auto positive_int = [](std::string_view v) {
    long n = 0;
    auto r = std::from_chars(v.data(), v.data() + v.size(), n);
    return r.ec == std::errc{} && r.ptr == v.data() + v.size() && n > 0;
  };
  if (const char *e = std::getenv("MALLOC_ARENA_MAX"); e && positive_int(e))  // pre-commit-allow-getenv
    return true;
  const char *tunables = std::getenv("GLIBC_TUNABLES");  // pre-commit-allow-getenv
  if (!tunables) return false;
  std::string_view t(tunables);
  static constexpr std::string_view key = "glibc.malloc.arena_max=";
  const auto pos = t.find(key);
  if (pos == std::string_view::npos) return false;
  std::string_view value = t.substr(pos + key.size());
  if (const auto colon = value.find(':'); colon != std::string_view::npos) value = value.substr(0, colon);
  return positive_int(value);
}

HostAllocator tune_host_allocator(int reaper_period_s) {
  const HostAllocator a = detect_host_allocator();

  // One glibc setting: cap the arena count at the number of CPUs this process
  // may use (affinity and cgroup quota; at least 8). Each arena keeps its own high-water mark, and with
  // the default of 8 per core a server with a few hundred threads can leave
  // that many peaks resident. The cap costs no system calls, and an operator's
  // own setting is left alone.
  //
  // No mmap or trim thresholds. 3.5.3 to 3.5.5 forced a 1 MB mmap threshold
  // and a 4 MB trim threshold; that turned every multi-megabyte tensor of a
  // CPU inference into an mmap/munmap pair with fresh page faults, cheap on
  // bare metal and ruinous inside a VM, where each munmap is a TLB shootdown
  // across every vCPU (#34: 6x slower under WSL2). Page-sized image buffers
  // come from the host image pool and never touch malloc, and the reaper
  // below returns freed pages of every arena to the OS.
  int arena_max = 0;  // 0: not set here (not glibc, or the operator set one)
#if defined(__GLIBC__)
  if (a == HostAllocator::Glibc && !operator_set_arena_max()) {
    arena_max = glibc_arena_cap(usable_cpus());
    mallopt(M_ARENA_MAX, arena_max);
  }
#endif

  const bool reaper = reaper_period_s > 0 && a != HostAllocator::Unknown &&
                      !env::env_present("TURBO_OCR_DISABLE_MALLOC_REAPER");
  if (arena_max > 0)
    TOCR_LOG_INFO("Host allocator", "allocator", std::string_view(host_allocator_name(a)),
                  "arena_max", arena_max,
                  "idle_memory_reaper", reaper, "period_s", reaper ? reaper_period_s : 0);
  else
    TOCR_LOG_INFO("Host allocator", "allocator", std::string_view(host_allocator_name(a)),
                  "idle_memory_reaper", reaper, "period_s", reaper ? reaper_period_s : 0);
  if (reaper) {
    std::thread([a, reaper_period_s] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(reaper_period_s));
        (void)release_idle_host_memory(a);
      }
    }).detach();
  }
  return a;
}

} // namespace turbo_ocr::server::bootstrap
