"""Reference post-training static quantization (simulated integer execution).

This module is the executable SPEC for cpp/include/khrpinn/quant.hpp and the
*_quant.hpp forwards. The C++ implementation must follow the exact same formulas;
tests compare outputs within 1 quantization LSB (float accumulation order differs
between numpy BLAS and the scalar C++ loops, which fake-quant absorbs except at
rounding boundaries).

Scheme ("baseline PTQ without hardware co-design", see plan section 3.4):
  - weights: symmetric per-tensor, scale = max|w| / qmax, zero-point 0
  - activations/state: affine per-tensor from calibrated [min, max] (range is
    always widened to include 0 so zero is exactly representable)
  - execution: values stored as intN; at each tensor boundary dequantize ->
    float32 math -> requantize ("fake quant"). MLP bias stays float32
    (real integer pipelines keep bias in int32 at scale s_in*s_w: lossless).
  - KAN: the B-spline bin search and basis recursion consume the DEQUANTIZED
    (i.e. truncated) input — this is the bin-shift mechanism of Hypothesis 3.
    SiLU outputs and basis values are fake-quantized before their GEMVs.

Rounding rule (both languages): round-half-away-from-zero implemented as
  sign(x) * floor(|x| + 0.5)  in float32.
"""

import numpy as np

from . import npref
from .npref import F32, silu, b_splines_np

INT_RANGES = {16: (-32768, 32767), 8: (-128, 127), 6: (-32, 31), 4: (-8, 7)}

QMODES = ('full', 'residual_only', 'input_only', 'weights_only', 'acts_only')


def round_half_away(x):
    x = np.asarray(x, dtype=F32)
    return (np.sign(x) * np.floor(np.abs(x) + F32(0.5))).astype(F32)


class AffineQuant:
    """Affine per-tensor quantizer for activations/state."""

    def __init__(self, vmin, vmax, bits):
        qmin, qmax = INT_RANGES[bits]
        vmin = min(F32(vmin), F32(0.0))
        vmax = max(F32(vmax), F32(0.0))
        scale = F32(F32(vmax) - F32(vmin)) / F32(qmax - qmin)
        if scale < F32(1e-12):
            scale = F32(1e-12)
        self.qmin, self.qmax = qmin, qmax
        self.scale = F32(scale)
        zp = float(round_half_away(F32(qmin) - F32(vmin) / self.scale))
        self.zp = int(np.clip(zp, qmin, qmax))

    def quantize(self, x):
        q = round_half_away(np.asarray(x, dtype=F32) / self.scale) + F32(self.zp)
        return np.clip(q, self.qmin, self.qmax).astype(np.int32)

    def dequantize(self, q):
        return (self.scale * (q.astype(F32) - F32(self.zp))).astype(F32)

    def fake(self, x):
        return self.dequantize(self.quantize(x))


class SymmetricQuant:
    """Symmetric per-tensor quantizer for weights."""

    def __init__(self, w, bits):
        qmin, qmax = INT_RANGES[bits]
        amax = F32(np.max(np.abs(w))) if w.size else F32(0.0)
        scale = amax / F32(qmax)
        if scale < F32(1e-12):
            scale = F32(1e-12)
        self.qmin, self.qmax = qmin, qmax
        self.scale = F32(scale)

    def quantize(self, w):
        q = round_half_away(np.asarray(w, dtype=F32) / self.scale)
        return np.clip(q, self.qmin, self.qmax).astype(np.int32)

    def dequantize(self, q):
        return (self.scale * q.astype(F32)).astype(F32)

    def fake(self, w):
        return self.dequantize(self.quantize(w))


def fake_quant_weights(kmodel, bits):
    """Return a deep-copied kmodel whose weight tensors are fake-quantized."""
    import copy
    m = copy.deepcopy(kmodel)
    for layer in m['layers']:
        if layer['type'] == 'mlp':
            layer['W'] = SymmetricQuant(layer['W'], bits).fake(layer['W'])
            # bias intentionally not quantized (int32 bias is lossless)
        else:
            layer['base_w'] = SymmetricQuant(layer['base_w'], bits).fake(layer['base_w'])
            layer['spline_w'] = SymmetricQuant(layer['spline_w'], bits).fake(layer['spline_w'])
    return m


def branch_forward_quant(kmodel, x, bits, quant_input=True, quant_weights=True,
                         quant_acts=True):
    """Quantized residual-branch forward. x: (N,2) float32 -> (N,1) float32.

    The returned values are dequantized floats (what the integrator consumes).
    """
    m = fake_quant_weights(kmodel, bits) if quant_weights else kmodel
    x = np.asarray(x, dtype=F32)
    if quant_input:
        x = AffineQuant(*m['in_range'], bits).fake(x)

    for layer in m['layers']:
        if layer['type'] == 'mlp':
            y = npref.mlp_layer_forward(layer, x)
            if quant_acts:
                y = AffineQuant(*layer['act_range'], bits).fake(y)
        else:
            s = silu(x)
            if quant_acts:
                s = AffineQuant(*layer['silu_range'], bits).fake(s)
            B = b_splines_np(x, layer['grid'], layer['k'])
            if quant_acts:
                B = AffineQuant(*layer['basis_range'], bits).fake(B)
            n = x.shape[0]
            base_out = s @ layer['base_w'].T.astype(F32)
            spline_out = B.reshape(n, -1) @ layer['spline_w'].reshape(layer['out'], -1).T
            y = (base_out + spline_out).astype(F32)
            if quant_acts:
                y = AffineQuant(*layer['act_range'], bits).fake(y)
        x = y
    return x


def qmode_flags(qmode):
    """Map a qmode name to (quant_input, quant_weights, quant_acts, quant_state_loop)."""
    if qmode == 'full':
        return True, True, True, True
    if qmode == 'residual_only':
        return True, True, True, False
    if qmode == 'input_only':
        return True, False, False, False
    if qmode == 'weights_only':
        return False, True, False, False
    if qmode == 'acts_only':
        return False, False, True, False
    raise ValueError(f"unknown qmode: {qmode}")
