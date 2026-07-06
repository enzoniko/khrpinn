// khrpinn/bench.hpp — timing + statistics utilities for the latency harness.
#pragma once

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

namespace khrpinn {

inline uint64_t now_ns() {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

struct BenchStats {
    double median = 0, p10 = 0, p90 = 0, min = 0, mean = 0;
};

inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

inline BenchStats bench_stats(const std::vector<double>& samples) {
    BenchStats s;
    if (samples.empty()) return s;
    s.median = percentile(samples, 0.5);
    s.p10 = percentile(samples, 0.10);
    s.p90 = percentile(samples, 0.90);
    s.min = *std::min_element(samples.begin(), samples.end());
    double sum = 0;
    for (double x : samples) sum += x;
    s.mean = sum / samples.size();
    return s;
}

inline std::string host_arch() {
#if defined(__unix__) || defined(__APPLE__)
    struct utsname u;
    if (uname(&u) == 0) return std::string(u.machine);
#endif
    return "unknown";
}

inline const char* compiler_id() {
#if defined(__clang__)
    return "clang-" __clang_version__;
#elif defined(__GNUC__)
    return "gcc-" __VERSION__;
#else
    return "unknown";
#endif
}

}  // namespace khrpinn
