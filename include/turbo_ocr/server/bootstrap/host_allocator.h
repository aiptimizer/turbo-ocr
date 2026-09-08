#pragma once

// Host-memory containment for the long-running server process.
//
// Request handling allocates large, short-lived host buffers (a decoded page
// is tens to hundreds of MB; the base64 routes hold the encoded text as well)
// on whichever work-pool thread took the request. Every general-purpose
// allocator keeps such freed memory in per-thread/per-arena free lists and
// returns it to the OS only on its own schedule:
//
//  - glibc parks freed blocks in per-arena free lists and auto-raises its mmap
//    threshold as large blocks are freed, so later large allocations come from
//    the arena instead of being mapped and unmapped. That is accepted on
//    purpose: forcing fixed thresholds instead (3.5.3 to 3.5.5) made every
//    multi-megabyte tensor an mmap/munmap pair, which inside a VM became a TLB
//    shootdown storm (#34). Containment comes from three things: the arena
//    count is capped (each arena keeps its own high-water mark), page-sized
//    image buffers live in the host image pool outside malloc, and a periodic
//    malloc_trim(0) returns the already-free pages of every arena.
//  - jemalloc (commonly LD_PRELOADed on hosts that share memory with other
//    services) keeps ~4 arenas per CPU, and an arena releases its dirty pages
//    only when ITS decay clock advances, which happens on allocation activity
//    on that arena or through jemalloc's optional background thread. With
//    many work threads spread over many arenas, an arena that goes quiet keeps
//    its peak for as long as the process lives, so RSS plateaus inside a
//    burst, never comes down between bursts, and the floor ratchets up as new
//    peaks land on arenas that had not seen them. Periodically advancing every
//    arena's decay ("arena.<all>.decay") releases what the operator's decay
//    settings say is releasable; it never forces a purge and never touches
//    live memory.
//
// Neither call reclaims live buffers, so the reaper cannot mask a real leak;
// it only removes retention. Cadence is low (seconds) so the cost is noise.
// The allocator is detected at run time, which is what makes an LD_PRELOADed
// jemalloc work without a rebuild.

#include <cstddef>
#include <string_view>

#include "turbo_ocr/decode/host_image_pool.h"

namespace turbo_ocr::server::bootstrap {

enum class HostAllocator {
  Unknown,   // neither glibc nor jemalloc detected (musl, macOS, ...): nothing to do
  Glibc,     // glibc malloc: arena cap + periodic malloc_trim
  Jemalloc,  // jemalloc (linked or LD_PRELOADed): mallctl arena decay
};

// Detects the allocator serving malloc in this process. jemalloc is
// recognised by its exported mallctl symbol, so a preloaded copy counts.
[[nodiscard]] HostAllocator detect_host_allocator() noexcept;

[[nodiscard]] const char *host_allocator_name(HostAllocator a) noexcept;

// Asks the allocator to hand already-freed memory back to the OS: malloc_trim
// on glibc, decay on every arena on jemalloc. Returns false when the
// allocator offers no such call (Unknown) or the call failed. Safe to call
// from any thread at any time.
bool release_idle_host_memory(HostAllocator a) noexcept;

// Installs the process-wide pinned host image pool (decode/host_image_pool.h)
// as OpenCV's default allocator and logs its budget: `slots` reusable page
// buffers, each growing to at most `max_block_bytes` (the MAX_IMAGE_PIXELS_MP
// budget). TURBO_OCR_DISABLE_HOST_IMAGE_POOL=1 leaves OpenCV's allocator in
// place (troubleshooting only).
void install_host_image_pool(size_t slots, size_t max_block_bytes,
                             decode::BlockMemory memory);

// Process-wide setup, once at startup: detects the allocator and starts the
// low-frequency reaper thread that calls release_idle_host_memory() every
// `reaper_period_s` seconds (0 disables it, as does
// TURBO_OCR_DISABLE_MALLOC_REAPER=1 in the environment). On glibc it also caps
// the arena count; no mmap/trim thresholds (see the note in the implementation).
// Returns the detected allocator for the startup log.
HostAllocator tune_host_allocator(int reaper_period_s = 5);

// The glibc arena cap tune_host_allocator() applies: the core count, at least 8.
// Each arena keeps its own high-water mark; the default of 8 per core would let
// a server with a few hundred threads leave that many peaks resident.
[[nodiscard]] int glibc_arena_cap(int cores) noexcept;

// CPUs this process may actually use: the affinity mask capped by the cgroup
// CPU quota (docker --cpus, Kubernetes limits), never below 1.
[[nodiscard]] int usable_cpus() noexcept;

// cgroup v2 cpu.max text ("<quota> <period>" or "max <period>") to a CPU count,
// rounded up; 0 when there is no quota or the text is not a quota.
[[nodiscard]] int cpus_from_cgroup_cpu_max(std::string_view cpu_max) noexcept;

// True when the operator set a valid arena count themselves (a positive integer
// in MALLOC_ARENA_MAX, or glibc.malloc.arena_max=<n> in GLIBC_TUNABLES);
// tune_host_allocator() then leaves it.
[[nodiscard]] bool operator_set_arena_max() noexcept;

} // namespace turbo_ocr::server::bootstrap
