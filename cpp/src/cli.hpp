// Tiny shared CLI helpers for the khrpinn tools (flag parsing, precision).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "khrpinn/quant.hpp"

namespace cli {

struct Args {
    int argc;
    char** argv;
    const char* get(const char* flag, const char* dflt) const {
        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
        return dflt;
    }
    double getd(const char* flag, double dflt) const {
        const char* s = get(flag, nullptr);
        return s ? std::atof(s) : dflt;
    }
    long getl(const char* flag, long dflt) const {
        const char* s = get(flag, nullptr);
        return s ? std::atol(s) : dflt;
    }
    bool has(const char* flag) const {
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], flag) == 0) return true;
        return false;
    }
    const char* positional(int idx) const {  // idx-th non-flag arg (0-based)
        int n = 0;
        for (int i = 1; i < argc; ++i) {
            if (argv[i][0] == '-') { ++i; continue; }  // skip flag + value
            if (n == idx) return argv[i];
            ++n;
        }
        return nullptr;
    }
};

// precision string -> bits (0 = fp32)
inline int parse_precision(const std::string& p) {
    if (p == "f32" || p == "fp32") return 0;
    if (p == "i16" || p == "int16") return 16;
    if (p == "i8" || p == "int8") return 8;
    if (p == "i6" || p == "int6") return 6;
    if (p == "i4" || p == "int4") return 4;
    std::fprintf(stderr, "unknown precision '%s' (f32|i16|i8|i6|i4)\n", p.c_str());
    std::exit(2);
}

inline const char* precision_name(int bits) {
    switch (bits) {
        case 0: return "f32";
        case 16: return "i16";
        case 8: return "i8";
        case 6: return "i6";
        case 4: return "i4";
    }
    return "?";
}

// Extract "slug" (basename without extension) from a model path.
inline std::string slug_of(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    std::string base = s == std::string::npos ? path : path.substr(s + 1);
    size_t d = base.rfind('.');
    return d == std::string::npos ? base : base.substr(0, d);
}

// Deterministic xorshift32 float in [-1, 1] (benchmark input generation).
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    float uniform() { return (float)(next() >> 8) * (2.0f / 16777216.0f) - 1.0f; }
};

}  // namespace cli
