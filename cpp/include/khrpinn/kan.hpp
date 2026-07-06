// khrpinn/kan.hpp — scalar FP32 B-spline KAN forward.
//
// Numerics mirror python/iclr/kan.py (and python/common/npref.py) op-for-op:
// Cox-de Boor recursion with explicit divisions on the (possibly non-uniform,
// training-adapted) knot vector; inputs outside the extended grid produce
// all-zero bases so only the SiLU base path contributes (torch behavior).
// No allocation in the hot path: callers pass a Workspace.
#pragma once

#include <cmath>
#include <vector>

#include "model.hpp"

namespace khrpinn {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Evaluate the G+k spline bases for scalar input x on one feature's knot row
// g[0 .. G+2k]. bas must hold G+2k floats; on return the first G+k are valid.
inline void eval_bases(const float* g, uint32_t G, uint32_t k, float x, float* bas) {
    const uint32_t m = G + 2 * k;  // order-0 basis count
    for (uint32_t j = 0; j < m; ++j)
        bas[j] = (x >= g[j] && x < g[j + 1]) ? 1.0f : 0.0f;
    for (uint32_t r = 1; r <= k; ++r) {
        const uint32_t n = m - r;
        for (uint32_t j = 0; j < n; ++j) {
            // In-place: bas[j+1] is still the previous-order value here.
            const float left = (x - g[j]) / (g[j + r] - g[j]) * bas[j];
            const float right = (g[j + r + 1] - x) / (g[j + r + 1] - g[j + 1]) * bas[j + 1];
            bas[j] = left + right;
        }
    }
}

struct Workspace {
    std::vector<float> act_a, act_b;  // activation ping-pong buffers
    std::vector<float> silu_buf;      // per-layer SiLU(x) values
    std::vector<float> bases;         // [in * (G+2k)] basis values, stride n_bases0

    explicit Workspace(const Model& m) {
        const uint32_t w = m.max_width();
        act_a.assign(w, 0.0f);
        act_b.assign(w, 0.0f);
        silu_buf.assign(w, 0.0f);
        bases.assign(m.max_bases_buf(), 0.0f);
    }
};

// One KANLinear layer: y[out] = base_w * SiLU(x) + spline_w * B(x).
inline void kan_layer_forward(const KanLayer& L, const float* x, float* y,
                              float* silu_buf, float* bases_buf) {
    const uint32_t nb = L.n_bases();
    const uint32_t stride = L.n_bases0();
    for (uint32_t i = 0; i < L.in; ++i) {
        silu_buf[i] = silu(x[i]);
        eval_bases(L.grid.data() + (size_t)i * L.n_knots(), L.G, L.k, x[i],
                   bases_buf + (size_t)i * stride);
    }
    for (uint32_t u = 0; u < L.out; ++u) {
        float acc = 0.0f;
        const float* bw = L.base_w.data() + (size_t)u * L.in;
        for (uint32_t i = 0; i < L.in; ++i) acc += bw[i] * silu_buf[i];
        for (uint32_t i = 0; i < L.in; ++i) {
            const float* B = bases_buf + (size_t)i * stride;
            const float* w = L.spline_w.data() + ((size_t)u * L.in + i) * nb;
            for (uint32_t j = 0; j < nb; ++j) acc += w[j] * B[j];
        }
        y[u] = acc;
    }
}

// Full KAN residual branch. x: input_dim floats -> returns scalar output.
inline float kan_forward(const Model& m, Workspace& ws, const float* x_in) {
    const float* x = x_in;
    float* cur = ws.act_a.data();
    float* nxt = ws.act_b.data();
    for (size_t li = 0; li < m.kan.size(); ++li) {
        const KanLayer& L = m.kan[li];
        kan_layer_forward(L, li == 0 ? x : cur, nxt, ws.silu_buf.data(), ws.bases.data());
        std::swap(cur, nxt);
    }
    return cur[0];
}

}  // namespace khrpinn
