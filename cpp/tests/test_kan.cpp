// KAN forward unit tests (structure-level; exact numerics pinned by goldens).
#include <cmath>

#include "khrpinn/kan.hpp"
#include "khrpinn/mlp.hpp"
#include "test.hpp"

using namespace khrpinn;

static KanLayer make_layer(uint32_t in, uint32_t out, uint32_t G, uint32_t k) {
    KanLayer l;
    l.in = in; l.out = out; l.G = G; l.k = k;
    l.grid.resize((size_t)in * l.n_knots());
    float h = 2.0f / G;
    for (uint32_t i = 0; i < in; ++i)
        for (uint32_t j = 0; j < l.n_knots(); ++j)
            l.grid[(size_t)i * l.n_knots() + j] = -1.0f + h * ((int)j - (int)k);
    l.base_w.assign((size_t)out * in, 0.0f);
    l.spline_w.assign((size_t)out * in * l.n_bases(), 0.0f);
    return l;
}

TEST(silu_matches_double_reference) {
    for (float x : {-4.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 4.0f}) {
        double ref = (double)x / (1.0 + std::exp(-(double)x));
        CHECK_CLOSE(silu(x), ref, 1e-6);
    }
    CHECK_CLOSE(silu(0.0f), 0.0f, 0.0);
}

TEST(kan_zero_spline_weights_is_base_path_only) {
    Model m;
    m.arch = Arch::KAN;
    m.n_layers = 1;
    KanLayer l = make_layer(2, 1, 5, 3);
    l.base_w = {2.0f, -1.0f};
    m.kan = {l};
    Workspace ws(m);
    float x[2] = {0.5f, -0.25f};
    float expect = 2.0f * silu(0.5f) - 1.0f * silu(-0.25f);
    CHECK_CLOSE(kan_forward(m, ws, x), expect, 1e-6);
}

TEST(kan_single_spline_coeff_reproduces_basis) {
    Model m;
    m.arch = Arch::KAN;
    m.n_layers = 1;
    KanLayer l = make_layer(2, 1, 5, 3);
    l.spline_w[3] = 1.0f;  // coeff (out 0, feature 0, basis 3)
    m.kan = {l};
    Workspace ws(m);
    float x[2] = {0.37f, 0.0f};
    // Direct basis evaluation for feature 0
    float bas[5 + 2 * 3];
    eval_bases(l.grid.data(), l.G, l.k, x[0], bas);
    CHECK_CLOSE(kan_forward(m, ws, x), bas[3], 1e-7);
}

TEST(kan_out_of_grid_input_uses_base_path_only) {
    Model m;
    m.arch = Arch::KAN;
    m.n_layers = 1;
    KanLayer l = make_layer(2, 1, 5, 3);
    l.base_w = {1.0f, 1.0f};
    for (auto& w : l.spline_w) w = 0.7f;  // nonzero spline weights everywhere
    m.kan = {l};
    Workspace ws(m);
    float x[2] = {5.0f, -5.0f};  // beyond the extended grid: bases all zero
    float expect = silu(5.0f) + silu(-5.0f);
    CHECK_CLOSE(kan_forward(m, ws, x), expect, 1e-6);
}

TEST(kan_recip_basis_path_matches_division_path) {
    // The bench-only reciprocal path must agree with the golden-pinned
    // division path to float rounding (~1 ulp per op).
    KanLayer l = make_layer(2, 1, 5, 3);
    // Perturb knots (non-uniform, like a trained grid) and set random-ish weights
    float bump = 0.017f;
    for (auto& g : l.grid) { g += bump; bump = -bump * 0.9f; }
    l.precompute_recips();
    for (size_t i = 0; i < l.spline_w.size(); ++i)
        l.spline_w[i] = 0.3f * (float)((i * 7) % 11) - 1.0f;
    l.base_w = {0.5f, -0.7f};
    Model m;
    m.arch = Arch::KAN;
    m.n_layers = 1;
    m.kan = {l};
    Workspace ws(m);
    for (float x = -1.4f; x <= 1.4f; x += 0.0231f) {
        for (float v = -1.3f; v <= 1.3f; v += 0.117f) {
            float s[2] = {x, v};
            float a = kan_forward(m, ws, s, false);
            float b = kan_forward(m, ws, s, true);
            CHECK_CLOSE(a, b, 1e-5);
        }
    }
}

TEST(kan_two_layer_composition) {
    // Composition sanity: layer1 sees layer0 output. With layer0 = base-only
    // identity-ish weights and layer1 base-only, result = silu chain.
    Model m;
    m.arch = Arch::KAN;
    m.n_layers = 2;
    KanLayer l0 = make_layer(2, 2, 5, 3);
    l0.base_w = {1.0f, 0.0f, 0.0f, 1.0f};
    KanLayer l1 = make_layer(2, 1, 5, 3);
    l1.base_w = {1.0f, 1.0f};
    m.kan = {l0, l1};
    Workspace ws(m);
    float x[2] = {0.4f, -0.6f};
    float h0 = silu(0.4f), h1 = silu(-0.6f);
    float expect = silu(h0) + silu(h1);
    CHECK_CLOSE(kan_forward(m, ws, x), expect, 1e-6);
}
