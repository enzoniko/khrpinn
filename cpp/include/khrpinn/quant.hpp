// khrpinn/quant.hpp — post-training static quantization, simulated-integer
// execution. This is the C++ twin of python/common/quant_sim.py; both follow
// the same formulas (see that module's docstring for the full spec):
//   weights: symmetric per-tensor, scale = max|w|/qmax
//   activations/state: affine per-tensor from calibrated ranges (widened to 0)
//   rounding: round-half-away-from-zero = sign(x) * floor(|x| + 0.5), float32
//   execution: intN storage semantics via fake-quant (dequantize -> float32 op
//   -> requantize at each tensor boundary). MLP bias stays float (lossless
//   int32 bias in a real pipeline).
// The B-spline bin search consumes the DEQUANTIZED (truncated) input — the
// bin-shift mechanism under test in Hypothesis 3.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "kan.hpp"
#include "mlp.hpp"
#include "model.hpp"

namespace khrpinn {

inline void int_range(int bits, int& qmin, int& qmax) {
    switch (bits) {
        case 16: qmin = -32768; qmax = 32767; return;
        case 8:  qmin = -128;   qmax = 127;   return;
        case 6:  qmin = -32;    qmax = 31;    return;
        case 4:  qmin = -8;     qmax = 7;     return;
        default: throw std::runtime_error("unsupported bit-width");
    }
}

inline float round_half_away(float x) {
    const float r = std::floor(std::fabs(x) + 0.5f);
    return x < 0.0f ? -r : r;
}

struct QParams {
    float scale = 1.0f;
    float zp = 0.0f;  // kept in float: all arithmetic is float32 in fake-quant
    float qmin = 0.0f, qmax = 0.0f;
    bool enabled = false;
};

inline QParams affine_qparams(Range r, int bits) {
    int qmin, qmax;
    int_range(bits, qmin, qmax);
    float vmin = r.min < 0.0f ? r.min : 0.0f;  // widen to include exact zero
    float vmax = r.max > 0.0f ? r.max : 0.0f;
    float scale = (vmax - vmin) / (float)(qmax - qmin);
    if (scale < 1e-12f) scale = 1e-12f;
    QParams q;
    q.scale = scale;
    float zp = round_half_away((float)qmin - vmin / scale);
    if (zp < (float)qmin) zp = (float)qmin;
    if (zp > (float)qmax) zp = (float)qmax;
    q.zp = zp;
    q.qmin = (float)qmin;
    q.qmax = (float)qmax;
    q.enabled = true;
    return q;
}

inline QParams symmetric_qparams(float absmax, int bits) {
    int qmin, qmax;
    int_range(bits, qmin, qmax);
    float scale = absmax / (float)qmax;
    if (scale < 1e-12f) scale = 1e-12f;
    QParams q;
    q.scale = scale;
    q.zp = 0.0f;
    q.qmin = (float)qmin;
    q.qmax = (float)qmax;
    q.enabled = true;
    return q;
}

// Quantize-dequantize one value (identity if the quantizer is disabled).
inline float fake_quant(float x, const QParams& q) {
    if (!q.enabled) return x;
    float t = round_half_away(x / q.scale) + q.zp;
    if (t < q.qmin) t = q.qmin;
    if (t > q.qmax) t = q.qmax;
    return q.scale * (t - q.zp);
}

inline void fake_quant_array(float* x, size_t n, const QParams& q) {
    if (!q.enabled) return;
    for (size_t i = 0; i < n; ++i) x[i] = fake_quant(x[i], q);
}

struct QuantConfig {
    int bits = 8;
    bool q_input = true;       // quantize the branch input (state)
    bool q_weights = true;     // quantize all weight tensors
    bool q_acts = true;        // quantize SiLU/basis/inter-layer activations
    bool q_state_loop = true;  // requantize the recurrent state every step
};

inline QuantConfig qmode_config(const std::string& mode, int bits) {
    QuantConfig c;
    int qmin, qmax;
    int_range(bits, qmin, qmax);  // validate bit-width eagerly
    c.bits = bits;
    if (mode == "full")               { c.q_input = true;  c.q_weights = true;  c.q_acts = true;  c.q_state_loop = true;  }
    else if (mode == "residual_only") { c.q_input = true;  c.q_weights = true;  c.q_acts = true;  c.q_state_loop = false; }
    else if (mode == "input_only")    { c.q_input = true;  c.q_weights = false; c.q_acts = false; c.q_state_loop = false; }
    else if (mode == "weights_only")  { c.q_input = false; c.q_weights = true;  c.q_acts = false; c.q_state_loop = false; }
    else if (mode == "acts_only")     { c.q_input = false; c.q_weights = false; c.q_acts = true;  c.q_state_loop = false; }
    else throw std::runtime_error("unknown qmode: " + mode);
    return c;
}

namespace detail {
inline float absmax(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) { float a = std::fabs(x); if (a > m) m = a; }
    return m;
}
inline void fq_weights(std::vector<float>& w, int bits) {
    QParams q = symmetric_qparams(absmax(w), bits);
    for (float& x : w) {
        float t = round_half_away(x / q.scale);
        if (t < q.qmin) t = q.qmin;
        if (t > q.qmax) t = q.qmax;
        x = q.scale * t;
    }
}
}  // namespace detail

// A model prepared for simulated-integer execution: weights fake-quantized
// (if configured) and activation quantizers derived from calibration ranges.
struct QuantModel {
    Model m;  // deep copy; weights possibly fake-quantized
    QuantConfig cfg;
    QParams in_q;                        // branch input / recurrent state
    std::vector<QParams> mlp_act;        // per MLP layer
    std::vector<QParams> kan_silu, kan_basis, kan_act;  // per KAN layer

    QuantModel(const Model& src, const QuantConfig& c) : m(src), cfg(c) {
        if (cfg.q_weights) {
            for (auto& l : m.mlp) detail::fq_weights(l.W, cfg.bits);
            for (auto& l : m.kan) {
                detail::fq_weights(l.base_w, cfg.bits);
                detail::fq_weights(l.spline_w, cfg.bits);
            }
        }
        in_q = cfg.q_input ? affine_qparams(m.in_r, cfg.bits) : QParams{};
        for (auto& l : m.mlp)
            mlp_act.push_back(cfg.q_acts ? affine_qparams(l.act, cfg.bits) : QParams{});
        for (auto& l : m.kan) {
            kan_silu.push_back(cfg.q_acts ? affine_qparams(l.silu_r, cfg.bits) : QParams{});
            kan_basis.push_back(cfg.q_acts ? affine_qparams(l.basis_r, cfg.bits) : QParams{});
            kan_act.push_back(cfg.q_acts ? affine_qparams(l.act, cfg.bits) : QParams{});
        }
    }
};

// Quantized KAN layer (mirrors quant_sim.branch_forward_quant's KAN branch).
inline void kan_layer_forward_quant(const KanLayer& L, const float* x, float* y,
                                    float* silu_buf, float* bases_buf,
                                    const QParams& silu_q, const QParams& basis_q,
                                    const QParams& act_q) {
    const uint32_t nb = L.n_bases();
    const uint32_t stride = L.n_bases0();
    for (uint32_t i = 0; i < L.in; ++i) {
        silu_buf[i] = fake_quant(silu(x[i]), silu_q);
        float* B = bases_buf + (size_t)i * stride;
        eval_bases(L.grid.data() + (size_t)i * L.n_knots(), L.G, L.k, x[i], B);
        fake_quant_array(B, nb, basis_q);
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
        y[u] = fake_quant(acc, act_q);
    }
}

inline void mlp_layer_forward_quant(const MlpLayer& L, const float* x, float* y,
                                    const QParams& act_q) {
    mlp_layer_forward(L, x, y);
    fake_quant_array(y, L.out, act_q);
}

// Quantized residual branch. Input x[input_dim] is NOT modified.
inline float residual_forward_quant(const QuantModel& qm, Workspace& ws, const float* x_in) {
    const Model& m = qm.m;
    float* cur = ws.act_a.data();
    float* nxt = ws.act_b.data();
    const uint32_t din = m.input_dim();
    for (uint32_t i = 0; i < din; ++i) cur[i] = fake_quant(x_in[i], qm.in_q);
    if (m.arch == Arch::MLP) {
        for (size_t li = 0; li < m.mlp.size(); ++li) {
            mlp_layer_forward_quant(m.mlp[li], cur, nxt, qm.mlp_act[li]);
            std::swap(cur, nxt);
        }
    } else {
        for (size_t li = 0; li < m.kan.size(); ++li) {
            kan_layer_forward_quant(m.kan[li], cur, nxt, ws.silu_buf.data(),
                                    ws.bases.data(), qm.kan_silu[li],
                                    qm.kan_basis[li], qm.kan_act[li]);
            std::swap(cur, nxt);
        }
    }
    return cur[0];
}

}  // namespace khrpinn
