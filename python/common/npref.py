"""NumPy reference forward pass for exported .khrp models.

This is the executable specification of the C++ FP32 engine: the same folded
weights, the same Cox-de Boor recursion as python/iclr/kan.py, float32 end to end.
Used for (a) validating exports against the torch model, (b) PTQ calibration,
(c) as the substrate of the simulated-quantization reference (quant_sim.py).

Model dict ("kmodel") layout:
  arch: 'mlp' | 'kan'
  system: 'Duffing' | 'VdP'
  dt: float
  norm_min, norm_range: float32[2]
  in_range: (min, max)                      # branch-input calibration range
  layers: list of dicts:
    mlp: type,in,out,has_relu, W float32[out,in], b float32[out], act_range
    kan: type,in,out,G,k, grid float32[in, G+2k+1], base_w float32[out,in],
         spline_w float32[out,in,G+k]  (spline_scaler pre-folded),
         silu_range, basis_range, act_range
"""

import numpy as np

F32 = np.float32


def silu(x):
    x = x.astype(F32)
    return (x / (F32(1.0) + np.exp(-x))).astype(F32)


def b_splines_np(x, grid, k):
    """Cox-de Boor bases, mirroring KANLinear.b_splines exactly.

    x: (N, in) float32; grid: (in, G+2k+1) float32 -> (N, in, G+k) float32.
    Inputs outside the extended grid yield all-zero bases (torch behavior).
    """
    x = x.astype(F32)[..., None]  # (N, in, 1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(F32)
    for r in range(1, k + 1):
        left = (x - grid[:, : -(r + 1)]) / (grid[:, r:-1] - grid[:, : -(r + 1)]) * bases[:, :, :-1]
        right = (grid[:, r + 1:] - x) / (grid[:, r + 1:] - grid[:, 1:(-r)]) * bases[:, :, 1:]
        bases = (left + right).astype(F32)
    return bases


def kan_layer_forward(layer, x, return_internals=False):
    """One KANLinear layer with folded spline weights."""
    s = silu(x)                                     # (N, in)
    base_out = s @ layer['base_w'].T.astype(F32)    # (N, out)
    B = b_splines_np(x, layer['grid'], layer['k'])  # (N, in, G+k)
    n = x.shape[0]
    spline_out = B.reshape(n, -1) @ layer['spline_w'].reshape(layer['out'], -1).T
    y = (base_out + spline_out).astype(F32)
    if return_internals:
        return y, s, B
    return y


def mlp_layer_forward(layer, x):
    y = (x @ layer['W'].T.astype(F32) + layer['b'].astype(F32)).astype(F32)
    if layer['has_relu']:
        y = np.maximum(y, F32(0.0))
    return y


def forward(kmodel, x):
    """Residual-branch forward. x: (N, 2) float32 normalized states -> (N, 1)."""
    x = np.asarray(x, dtype=F32)
    for layer in kmodel['layers']:
        if layer['type'] == 'mlp':
            x = mlp_layer_forward(layer, x)
        else:
            x = kan_layer_forward(layer, x)
    return x


def hrpinn_step(kmodel, known_accel_fn, s_norm):
    """One HRPINN cell step in float32 (mirrors HRPINNCell.forward and cpp hrpinn.hpp).

    s_norm: (N, 2) normalized state.
    """
    s_norm = np.asarray(s_norm, dtype=F32)
    nmin = kmodel['norm_min'].astype(F32)
    nrng = kmodel['norm_range'].astype(F32)
    state = ((s_norm + F32(1.0)) * nrng / F32(2.0) + nmin).astype(F32)
    x, v = state[:, 0:1], state[:, 1:2]
    dv_known = known_accel_fn(x, v).astype(F32)
    dv_res = forward(kmodel, s_norm)
    dv = (dv_known + dv_res).astype(F32)
    dt = F32(kmodel['dt'])
    x_next = (x + dt * v).astype(F32)
    v_next = (v + dt * dv).astype(F32)
    nxt = np.concatenate([x_next, v_next], axis=1)
    return (F32(2.0) * (nxt - nmin) / nrng - F32(1.0)).astype(F32)


def collect_calibration(kmodel, states):
    """Populate in_range / silu_range / basis_range / act_range from data.

    states: (N, 2) float32 normalized training states. Mutates kmodel in place.
    """
    x = np.asarray(states, dtype=F32)
    kmodel['in_range'] = (float(x.min()), float(x.max()))
    for layer in kmodel['layers']:
        if layer['type'] == 'mlp':
            y = mlp_layer_forward(layer, x)
            layer['act_range'] = (float(y.min()), float(y.max()))
        else:
            y, s, B = kan_layer_forward(layer, x, return_internals=True)
            layer['silu_range'] = (float(s.min()), float(s.max()))
            layer['basis_range'] = (float(B.min()), float(B.max()))
            layer['act_range'] = (float(y.min()), float(y.max()))
        x = y
    return kmodel


def param_count(kmodel):
    """Torch-equivalent parameter count (counts the pre-fold spline_scaler too)."""
    n = 0
    for layer in kmodel['layers']:
        if layer['type'] == 'mlp':
            n += layer['W'].size + layer['b'].size
        else:
            # base_w + spline_w + spline_scaler (folded at export but a real
            # parameter of the trained torch model)
            n += layer['base_w'].size + layer['spline_w'].size + layer['base_w'].size
    return n


def flops_estimate(kmodel):
    """Analytical scalar-op estimate per single inference (documented model).

    MACs count as 2 ops. SiLU counted as 5 ops + 1 exp (exp reported separately).
    B-spline basis: indicator pass ~2*(G+2k) cmps, then for r=1..k about
    4*(G+2k-r) mul/div/add per feature.
    Returns dict {flops, exps}.
    """
    flops = 0
    exps = 0
    for layer in kmodel['layers']:
        if layer['type'] == 'mlp':
            flops += 2 * layer['W'].size + layer['b'].size  # GEMV + bias
            if layer['has_relu']:
                flops += layer['out']
        else:
            G, k, cin, cout = layer['G'], layer['k'], layer['in'], layer['out']
            flops += 2 * cin * cout            # base GEMV
            flops += 5 * cin                   # SiLU arithmetic (excl. exp)
            exps += cin
            basis_ops = 2 * (G + 2 * k)        # indicator pass
            for r in range(1, k + 1):
                basis_ops += 6 * (G + 2 * k - r)  # 2 divs, 2 muls, 2 add/sub per entry
            flops += cin * basis_ops
            flops += 2 * cin * cout * (G + k)  # spline GEMV
    return {'flops': int(flops), 'exps': int(exps)}
