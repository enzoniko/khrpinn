// khrpinn/hrpinn.hpp — the hard-constrained recurrent cell (FP32 and
// simulated-quantized). Mirrors common/hrpinn.py HRPINNCell.forward exactly:
//   denormalize -> known physics + learned residual -> explicit Euler ->
//   renormalize. States are 2-vectors [x, v] in the normalized domain.
#pragma once

#include "mlp.hpp"
#include "quant.hpp"
#include "systems.hpp"

namespace khrpinn {

// One FP32 cell step. s_in and s_out may alias.
inline void hrpinn_step(const Model& m, Workspace& ws, const float* s_in, float* s_out) {
    const float x = (s_in[0] + 1.0f) * m.norm_range[0] / 2.0f + m.norm_min[0];
    const float v = (s_in[1] + 1.0f) * m.norm_range[1] / 2.0f + m.norm_min[1];
    const float dv_known = known_accel(m.system, x, v);
    const float dv_res = residual_forward(m, ws, s_in);
    const float dv = dv_known + dv_res;
    const float x_next = x + m.dt * v;
    const float v_next = v + m.dt * dv;
    s_out[0] = 2.0f * (x_next - m.norm_min[0]) / m.norm_range[0] - 1.0f;
    s_out[1] = 2.0f * (v_next - m.norm_min[1]) / m.norm_range[1] - 1.0f;
}

// One quantized cell step. With cfg.q_state_loop the recurrent state is
// requantized to the state grid every step (deployment-faithful "full" mode);
// otherwise only the residual branch sees quantization ("residual_only").
inline void hrpinn_step_quant(const QuantModel& qm, Workspace& ws,
                              const float* s_in, float* s_out) {
    const Model& m = qm.m;
    float s[2] = {s_in[0], s_in[1]};
    if (qm.cfg.q_state_loop) {  // truncated state feeds physics AND branch
        s[0] = fake_quant(s[0], qm.in_q);
        s[1] = fake_quant(s[1], qm.in_q);
    }
    const float x = (s[0] + 1.0f) * m.norm_range[0] / 2.0f + m.norm_min[0];
    const float v = (s[1] + 1.0f) * m.norm_range[1] / 2.0f + m.norm_min[1];
    const float dv_known = known_accel(m.system, x, v);
    const float dv_res = residual_forward_quant(qm, ws, s);
    const float dv = dv_known + dv_res;
    const float x_next = x + m.dt * v;
    const float v_next = v + m.dt * dv;
    s_out[0] = 2.0f * (x_next - m.norm_min[0]) / m.norm_range[0] - 1.0f;
    s_out[1] = 2.0f * (v_next - m.norm_min[1]) / m.norm_range[1] - 1.0f;
    if (qm.cfg.q_state_loop) {
        s_out[0] = fake_quant(s_out[0], qm.in_q);
        s_out[1] = fake_quant(s_out[1], qm.in_q);
    }
}

// Normalize a physical state into the model's normalized domain and back.
inline void normalize_state(const Model& m, float xp, float vp, float* s) {
    s[0] = 2.0f * (xp - m.norm_min[0]) / m.norm_range[0] - 1.0f;
    s[1] = 2.0f * (vp - m.norm_min[1]) / m.norm_range[1] - 1.0f;
}
inline void denormalize_state(const Model& m, const float* s, float& xp, float& vp) {
    xp = (s[0] + 1.0f) * m.norm_range[0] / 2.0f + m.norm_min[0];
    vp = (s[1] + 1.0f) * m.norm_range[1] / 2.0f + m.norm_min[1];
}

}  // namespace khrpinn
