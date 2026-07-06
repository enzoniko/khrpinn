// B-spline basis unit tests (no goldens needed).
#include <cmath>
#include <vector>

#include "khrpinn/kan.hpp"
#include "test.hpp"

using khrpinn::eval_bases;

// Build a uniform extended knot vector like kan.py's initial grid:
// (G+2k+1) knots covering [lo, hi] with k extension knots each side.
static std::vector<float> uniform_grid(uint32_t G, uint32_t k, float lo, float hi) {
    std::vector<float> g(G + 2 * k + 1);
    float h = (hi - lo) / G;
    for (uint32_t j = 0; j < g.size(); ++j)
        g[j] = lo + h * ((int)j - (int)k);
    return g;
}

TEST(spline_partition_of_unity_uniform) {
    const uint32_t G = 5, k = 3;
    auto g = uniform_grid(G, k, -1.0f, 1.0f);
    std::vector<float> bas(G + 2 * k);
    for (float x = -0.999f; x < 1.0f; x += 0.0173f) {
        eval_bases(g.data(), G, k, x, bas.data());
        float sum = 0;
        for (uint32_t j = 0; j < G + k; ++j) sum += bas[j];
        CHECK_CLOSE(sum, 1.0f, 1e-5);
    }
}

TEST(spline_partition_of_unity_nonuniform) {
    // Perturbed knots (like a training-adapted grid) must still sum to 1
    // strictly inside the core interval.
    const uint32_t G = 5, k = 3;
    auto g = uniform_grid(G, k, -1.0f, 1.0f);
    float bump = 0.03f;
    for (size_t j = 0; j < g.size(); ++j) { g[j] += bump; bump = -bump; }
    // core interval is [g[k], g[k+G])
    std::vector<float> bas(G + 2 * k);
    for (float x = g[k] + 1e-3f; x < g[k + G] - 1e-3f; x += 0.0137f) {
        eval_bases(g.data(), G, k, x, bas.data());
        float sum = 0;
        for (uint32_t j = 0; j < G + k; ++j) sum += bas[j];
        CHECK_CLOSE(sum, 1.0f, 1e-4);
    }
}

TEST(spline_hat_function_order1) {
    // k=1 bases are hat functions: B_j peaks at 1 on knot j+1, linear sides.
    const uint32_t G = 4, k = 1;
    auto g = uniform_grid(G, k, 0.0f, 4.0f);  // knots -1,0,1,...,5
    std::vector<float> bas(G + 2 * k);
    // x = 1.0 sits exactly on a knot: hat centered there = 1, neighbors 0.
    eval_bases(g.data(), G, k, 1.0f, bas.data());
    CHECK_CLOSE(bas[1], 1.0f, 1e-6);  // hat over [0, 2] centered at 1
    CHECK_CLOSE(bas[0], 0.0f, 1e-6);
    CHECK_CLOSE(bas[2], 0.0f, 1e-6);
    // x = 1.5: two hats at 0.5 each.
    eval_bases(g.data(), G, k, 1.5f, bas.data());
    CHECK_CLOSE(bas[1], 0.5f, 1e-6);
    CHECK_CLOSE(bas[2], 0.5f, 1e-6);
}

TEST(spline_outside_grid_is_zero) {
    const uint32_t G = 5, k = 3;
    auto g = uniform_grid(G, k, -1.0f, 1.0f);
    std::vector<float> bas(G + 2 * k);
    for (float x : {-5.0f, -2.3f, 2.3f, 5.0f, 1e6f}) {
        eval_bases(g.data(), G, k, x, bas.data());
        for (uint32_t j = 0; j < G + k; ++j) CHECK_CLOSE(bas[j], 0.0f, 0.0);
    }
    // Right edge of the EXTENDED grid is exclusive (x < g[m]), torch behavior.
    eval_bases(g.data(), G, k, g[G + 2 * k], bas.data());
    for (uint32_t j = 0; j < G + k; ++j) CHECK_CLOSE(bas[j], 0.0f, 0.0);
}

TEST(spline_nonnegative_and_bounded) {
    const uint32_t G = 3, k = 3;
    auto g = uniform_grid(G, k, -1.0f, 1.0f);
    std::vector<float> bas(G + 2 * k);
    for (float x = -1.6f; x < 1.6f; x += 0.011f) {
        eval_bases(g.data(), G, k, x, bas.data());
        for (uint32_t j = 0; j < G + k; ++j) {
            CHECK(bas[j] >= -1e-7f);
            CHECK(bas[j] <= 1.0f + 1e-6f);
        }
    }
}
