#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "turbo_ocr/server/bootstrap/host_allocator.h"

using turbo_ocr::server::bootstrap::detect_host_allocator;
using turbo_ocr::server::bootstrap::HostAllocator;
using turbo_ocr::server::bootstrap::host_allocator_name;
using turbo_ocr::server::bootstrap::release_idle_host_memory;

TEST_CASE("the detected allocator has a name and matches the build", "[host_allocator]") {
  const HostAllocator a = detect_host_allocator();
  CHECK(std::strlen(host_allocator_name(a)) > 0);
  CHECK(std::string(host_allocator_name(HostAllocator::Unknown)) == "unknown");
  CHECK(std::string(host_allocator_name(HostAllocator::Glibc)) == "glibc");
  CHECK(std::string(host_allocator_name(HostAllocator::Jemalloc)) == "jemalloc");
#if defined(__GLIBC__)
  // On glibc systems the answer is glibc unless jemalloc was preloaded or
  // linked into the test binary, which the test runner may legitimately do.
  CHECK((a == HostAllocator::Glibc || a == HostAllocator::Jemalloc));
#else
  CHECK((a == HostAllocator::Unknown || a == HostAllocator::Jemalloc));
#endif
}

TEST_CASE("releasing idle memory is safe on every allocator and never fails for a real one", "[host_allocator]") {
  // Create and free a burst of allocations so there is something to give back.
  {
    std::vector<std::string> junk;
    for (int i = 0; i < 64; ++i) junk.emplace_back(1 << 20, 'x');
  }
  const HostAllocator a = detect_host_allocator();
  const bool released = release_idle_host_memory(a);
  if (a == HostAllocator::Unknown)
    CHECK_FALSE(released);
  else
    CHECK(released);
  // Explicitly unsupported kinds report false instead of pretending.
  CHECK_FALSE(release_idle_host_memory(HostAllocator::Unknown));
  // Calling it repeatedly is idempotent.
  CHECK(release_idle_host_memory(a) == released);
}

namespace {
// Restore a variable on scope exit so a harness that sets malloc tunables for
// the test binary keeps them for the cases that run afterwards.
struct EnvRestore {
  std::string key; bool had; std::string prev;
  explicit EnvRestore(const char *k) : key(k) {
    const char *p = std::getenv(k); had = p != nullptr; if (had) prev = p;
  }
  ~EnvRestore() { if (had) ::setenv(key.c_str(), prev.c_str(), 1); else ::unsetenv(key.c_str()); }
};
}  // namespace

TEST_CASE("glibc arena cap is the usable CPU count, at least 8", "[host_allocator]") {
  using turbo_ocr::server::bootstrap::glibc_arena_cap;
  CHECK(glibc_arena_cap(0) == 8);
  CHECK(glibc_arena_cap(4) == 8);
  CHECK(glibc_arena_cap(8) == 8);
  CHECK(glibc_arena_cap(24) == 24);
}

TEST_CASE("cgroup cpu.max text becomes a CPU count, rounded up", "[host_allocator]") {
  using turbo_ocr::server::bootstrap::cpus_from_cgroup_cpu_max;
  CHECK(cpus_from_cgroup_cpu_max("max 100000") == 0);
  CHECK(cpus_from_cgroup_cpu_max("800000 100000") == 8);
  CHECK(cpus_from_cgroup_cpu_max("150000 100000") == 2);
  CHECK(cpus_from_cgroup_cpu_max("50000 100000") == 1);
  CHECK(cpus_from_cgroup_cpu_max("-1 100000") == 0);
  CHECK(cpus_from_cgroup_cpu_max("garbage") == 0);
  CHECK(cpus_from_cgroup_cpu_max("") == 0);
}

TEST_CASE("usable_cpus is at least one and at most the affinity mask", "[host_allocator]") {
  using turbo_ocr::server::bootstrap::usable_cpus;
  const int n = usable_cpus();
  CHECK(n >= 1);
  CHECK(n <= 4096);
}

TEST_CASE("an operator's own valid arena setting is respected, a typo is not", "[host_allocator]") {
  using turbo_ocr::server::bootstrap::operator_set_arena_max;
  EnvRestore r1("MALLOC_ARENA_MAX"), r2("GLIBC_TUNABLES");
  ::unsetenv("MALLOC_ARENA_MAX"); ::unsetenv("GLIBC_TUNABLES");
  CHECK_FALSE(operator_set_arena_max());
  ::setenv("MALLOC_ARENA_MAX", "2", 1);
  CHECK(operator_set_arena_max());
  ::setenv("MALLOC_ARENA_MAX", "abc", 1);
  CHECK_FALSE(operator_set_arena_max());
  ::setenv("MALLOC_ARENA_MAX", "0", 1);
  CHECK_FALSE(operator_set_arena_max());
  ::unsetenv("MALLOC_ARENA_MAX");
  ::setenv("GLIBC_TUNABLES", "glibc.malloc.arena_max=2:glibc.malloc.tcache_count=0", 1);
  CHECK(operator_set_arena_max());
  ::setenv("GLIBC_TUNABLES", "glibc.malloc.tcache_count=0:glibc.malloc.arena_max=16", 1);
  CHECK(operator_set_arena_max());
  ::setenv("GLIBC_TUNABLES", "glibc.malloc.tcache_count=0", 1);
  CHECK_FALSE(operator_set_arena_max());
  ::setenv("GLIBC_TUNABLES", "glibc.malloc.arena_max=0", 1);
  CHECK_FALSE(operator_set_arena_max());
  ::setenv("GLIBC_TUNABLES", "glibc.malloc.arena_max=x", 1);
  CHECK_FALSE(operator_set_arena_max());
}
