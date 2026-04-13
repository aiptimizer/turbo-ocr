#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace turbo_ocr::server {

/// Read an environment variable with a fallback default.
[[nodiscard]] inline std::string env_or(const char *name,
                                        std::string_view def) {
  if (const char *v = std::getenv(name))
    return std::string(v);
  return std::string(def);
}

/// Check if an environment variable equals "1".
[[nodiscard]] inline bool env_enabled(const char *name) noexcept {
  const char *v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

/// Read an integer env var with bounds validation.
/// Returns def if not set or invalid. Clamps to [min_val, max_val].
[[nodiscard]] inline int env_int(const char *name, int def,
                                  int min_val = 1, int max_val = 65535) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  char *end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') return def;  // not a valid integer
  if (val < min_val) return min_val;
  if (val > max_val) return max_val;
  return static_cast<int>(val);
}

} // namespace turbo_ocr::server
