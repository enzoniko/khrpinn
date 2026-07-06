// Minimal zero-dependency test harness (TEST registration + CHECK macros).
// Compiles unmodified under qemu-riscv64. Non-zero exit code on any failure.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace ktest {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int& failures() { static int f = 0; return f; }
inline int& checks() { static int c = 0; return c; }
inline const char*& current() { static const char* n = ""; return n; }

inline void fail(const char* file, int line, const std::string& msg) {
    ++failures();
    std::fprintf(stderr, "  FAIL [%s] %s:%d: %s\n", current(), file, line, msg.c_str());
}

#define TEST(name)                                                     \
    static void ktest_fn_##name();                                     \
    static ktest::Registrar ktest_reg_##name(#name, ktest_fn_##name);  \
    static void ktest_fn_##name()

#define CHECK(cond)                                                    \
    do {                                                               \
        ++ktest::checks();                                             \
        if (!(cond)) ktest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_MSG(cond, msg)                                           \
    do {                                                               \
        ++ktest::checks();                                             \
        if (!(cond)) ktest::fail(__FILE__, __LINE__, std::string("CHECK(" #cond "): ") + (msg)); \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                         \
    do {                                                               \
        ++ktest::checks();                                             \
        double _a = (a), _b = (b), _t = (tol);                         \
        if (!(std::fabs(_a - _b) <= _t)) {                             \
            char _buf[256];                                            \
            std::snprintf(_buf, sizeof(_buf),                          \
                          "CHECK_CLOSE(%s=%.9g, %s=%.9g, tol=%.3g) diff=%.3g", \
                          #a, _a, #b, _b, _t, std::fabs(_a - _b));     \
            ktest::fail(__FILE__, __LINE__, _buf);                     \
        }                                                              \
    } while (0)

#define CHECK_THROWS(expr)                                             \
    do {                                                               \
        ++ktest::checks();                                             \
        bool _threw = false;                                           \
        try { (void)(expr); } catch (...) { _threw = true; }           \
        if (!_threw) ktest::fail(__FILE__, __LINE__, "CHECK_THROWS(" #expr ") did not throw"); \
    } while (0)

inline int run_all() {
    int ran = 0;
    for (auto& c : registry()) {
        current() = c.name;
        int before = failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            fail(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
        }
        std::printf("%-52s %s\n", c.name, failures() == before ? "PASS" : "FAIL");
        ++ran;
    }
    std::printf("\n%d tests, %d checks, %d failures\n", ran, checks(), failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace ktest
