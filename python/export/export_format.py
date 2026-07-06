"""Writer/reader for the .khrp binary model format (v1, little-endian).

Layout (all little-endian; f32 = IEEE-754 binary32):

  char[4] magic = "KHRP"
  u32     version = 1
  u8      arch      (0 = MLP, 1 = KAN)
  u8      system    (0 = Duffing, 1 = VdP)
  u16     n_layers
  f32     dt
  f32[2]  norm_min
  f32[2]  norm_range
  f32     in_min, in_max                 # branch-input calibration range
  then per layer:
    MLP: u32 in | u32 out | u8 has_relu | f32 W[out*in] (row-major, torch layout)
         | f32 b[out] | f32 act_min | f32 act_max
    KAN: u32 in | u32 out | u32 G | u32 k
         | f32 grid[in * (G+2k+1)]      (row-major, per-input-feature knots)
         | f32 base_w[out*in]
         | f32 spline_w[out*in*(G+k)]   (spline_scaler pre-folded)
         | f32 silu_min | f32 silu_max
         | f32 basis_min | f32 basis_max
         | f32 act_min | f32 act_max

The JSON sidecar manifest (same basename, .json) carries human-readable
metadata and is never read by the C++ engine.
"""

import json
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from common.systems import SYSTEM_IDS  # noqa: E402

MAGIC = b"KHRP"
VERSION = 1
ARCH_IDS = {'mlp': 0, 'kan': 1}
ARCH_NAMES = {0: 'mlp', 1: 'kan'}
SYSTEM_NAMES = {v: k for k, v in SYSTEM_IDS.items()}

F32 = np.float32


def kmodel_from_torch(residual_network, system_name, normalizer, dt=0.05):
    """Build a kmodel dict (see common/npref.py) from a trained torch model.

    Folds spline_scaler into spline_weight. Calibration ranges are left as
    placeholders; call common.npref.collect_calibration afterwards.
    """
    import torch

    layers = []
    if hasattr(residual_network, 'kan'):
        arch = 'kan'
        for kl in residual_network.kan.layers:
            with torch.no_grad():
                folded = kl.scaled_spline_weight.detach().cpu().numpy().astype(F32)
            layers.append({
                'type': 'kan',
                'in': kl.in_features,
                'out': kl.out_features,
                'G': kl.grid_size,
                'k': kl.spline_order,
                'grid': kl.grid.detach().cpu().numpy().astype(F32),
                'base_w': kl.base_weight.detach().cpu().numpy().astype(F32),
                'spline_w': folded,
                'silu_range': (0.0, 0.0),
                'basis_range': (0.0, 1.0),
                'act_range': (0.0, 0.0),
            })
    else:
        arch = 'mlp'
        import torch.nn as nn
        mods = list(residual_network.net)
        i = 0
        while i < len(mods):
            lin = mods[i]
            assert isinstance(lin, nn.Linear)
            has_relu = (i + 1 < len(mods)) and isinstance(mods[i + 1], nn.ReLU)
            layers.append({
                'type': 'mlp',
                'in': lin.in_features,
                'out': lin.out_features,
                'has_relu': has_relu,
                'W': lin.weight.detach().cpu().numpy().astype(F32),
                'b': lin.bias.detach().cpu().numpy().astype(F32),
                'act_range': (0.0, 0.0),
            })
            i += 2 if has_relu else 1

    return {
        'arch': arch,
        'system': system_name,
        'dt': float(dt),
        'norm_min': np.asarray(normalizer.min_vals, dtype=F32),
        'norm_range': np.asarray(normalizer.range, dtype=F32),
        'in_range': (-1.0, 1.0),
        'layers': layers,
    }


def write_khrp(kmodel, path):
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<BBH', ARCH_IDS[kmodel['arch']],
                            SYSTEM_IDS[kmodel['system']], len(kmodel['layers'])))
        f.write(struct.pack('<f', kmodel['dt']))
        f.write(np.asarray(kmodel['norm_min'], dtype='<f4').tobytes())
        f.write(np.asarray(kmodel['norm_range'], dtype='<f4').tobytes())
        f.write(struct.pack('<ff', *[float(v) for v in kmodel['in_range']]))
        for layer in kmodel['layers']:
            if layer['type'] == 'mlp':
                f.write(struct.pack('<IIB', layer['in'], layer['out'],
                                    1 if layer['has_relu'] else 0))
                f.write(np.ascontiguousarray(layer['W'], dtype='<f4').tobytes())
                f.write(np.ascontiguousarray(layer['b'], dtype='<f4').tobytes())
                f.write(struct.pack('<ff', *[float(v) for v in layer['act_range']]))
            else:
                f.write(struct.pack('<IIII', layer['in'], layer['out'],
                                    layer['G'], layer['k']))
                f.write(np.ascontiguousarray(layer['grid'], dtype='<f4').tobytes())
                f.write(np.ascontiguousarray(layer['base_w'], dtype='<f4').tobytes())
                f.write(np.ascontiguousarray(layer['spline_w'], dtype='<f4').tobytes())
                f.write(struct.pack('<ff', *[float(v) for v in layer['silu_range']]))
                f.write(struct.pack('<ff', *[float(v) for v in layer['basis_range']]))
                f.write(struct.pack('<ff', *[float(v) for v in layer['act_range']]))


def read_khrp(path):
    """Read a .khrp file back into a kmodel dict (numpy)."""
    with open(path, 'rb') as f:
        raw = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from('<' + fmt, raw, off)
        off += struct.calcsize('<' + fmt)
        return vals

    def take_f32(n, shape):
        nonlocal off
        arr = np.frombuffer(raw, dtype='<f4', count=n, offset=off).reshape(shape).copy()
        off += 4 * n
        return arr

    magic = raw[:4]
    off = 4
    assert magic == MAGIC, f"bad magic in {path}"
    (version,) = take('I')
    assert version == VERSION
    arch_id, system_id, n_layers = take('BBH')
    (dt,) = take('f')
    norm_min = take_f32(2, (2,))
    norm_range = take_f32(2, (2,))
    in_min, in_max = take('ff')

    arch = ARCH_NAMES[arch_id]
    layers = []
    for _ in range(n_layers):
        if arch == 'mlp':
            cin, cout, has_relu = take('IIB')
            W = take_f32(cout * cin, (cout, cin))
            b = take_f32(cout, (cout,))
            amin, amax = take('ff')
            layers.append({'type': 'mlp', 'in': cin, 'out': cout,
                           'has_relu': bool(has_relu), 'W': W, 'b': b,
                           'act_range': (amin, amax)})
        else:
            cin, cout, G, k = take('IIII')
            grid = take_f32(cin * (G + 2 * k + 1), (cin, G + 2 * k + 1))
            base_w = take_f32(cout * cin, (cout, cin))
            spline_w = take_f32(cout * cin * (G + k), (cout, cin, G + k))
            smin, smax = take('ff')
            bmin, bmax = take('ff')
            amin, amax = take('ff')
            layers.append({'type': 'kan', 'in': cin, 'out': cout, 'G': G, 'k': k,
                           'grid': grid, 'base_w': base_w, 'spline_w': spline_w,
                           'silu_range': (smin, smax), 'basis_range': (bmin, bmax),
                           'act_range': (amin, amax)})
    assert off == len(raw), f"trailing bytes in {path}: {len(raw) - off}"

    return {'arch': arch, 'system': SYSTEM_NAMES[system_id], 'dt': dt,
            'norm_min': norm_min, 'norm_range': norm_range,
            'in_range': (in_min, in_max), 'layers': layers}


def write_manifest(path, **fields):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, 'w') as f:
        json.dump(fields, f, indent=2, default=default)
        f.write('\n')
