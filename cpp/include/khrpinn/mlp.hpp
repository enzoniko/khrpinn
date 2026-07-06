// khrpinn/mlp.hpp — scalar FP32 MLP forward (Linear/ReLU chain, torch layout).
#pragma once

#include <algorithm>

#include "kan.hpp"  // Workspace
#include "model.hpp"

namespace khrpinn {

inline void mlp_layer_forward(const MlpLayer& L, const float* x, float* y) {
    for (uint32_t u = 0; u < L.out; ++u) {
        float acc = L.b[u];
        const float* w = L.W.data() + (size_t)u * L.in;
        for (uint32_t i = 0; i < L.in; ++i) acc += w[i] * x[i];
        y[u] = L.has_relu ? std::max(acc, 0.0f) : acc;
    }
}

inline float mlp_forward(const Model& m, Workspace& ws, const float* x_in) {
    const float* x = x_in;
    float* cur = ws.act_a.data();
    float* nxt = ws.act_b.data();
    for (size_t li = 0; li < m.mlp.size(); ++li) {
        mlp_layer_forward(m.mlp[li], li == 0 ? x : cur, nxt);
        std::swap(cur, nxt);
    }
    return cur[0];
}

// Architecture dispatch for the FP32 residual branch.
inline float residual_forward(const Model& m, Workspace& ws, const float* x) {
    return m.arch == Arch::MLP ? mlp_forward(m, ws, x) : kan_forward(m, ws, x);
}

}  // namespace khrpinn
