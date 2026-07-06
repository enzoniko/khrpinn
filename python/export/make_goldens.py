"""Generate golden test vectors for the C++ engine from the exported models.

Per model (data/goldens/<slug>/):
  residual_io.bin   u32 N | N x (f32 x, f32 v, f32 out)      torch FP32 branch
  step_io.bin       u32 N | N x (f32 s0x, s0v, s1x, s1v)     torch HRPINN step
  quant_io_b8.bin   u32 N | N x (f32 x, f32 v, f32 out)      quant_sim full-quant branch
  quant_io_b16.bin  same, 16-bit
  traj_f32.csv      ic,step,x,v (normalized)                 500-step torch rollout, 3 ICs
  golden_manifest.json  tolerances + traj validity flags

Global (data/goldens/):
  ics.csv                     20 dependability ICs (physical), seed 7
  truth_duffing.csv/.vdp.csv  high-accuracy scipy reference trajectory (RK4 unit test)

Inputs are in the NORMALIZED branch domain. The input set stresses: uniform
coverage, near-knot boundaries (KAN bin edges), out-of-grid extrapolation, and
on-trajectory states.
"""

import glob
import json
import os
import struct
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from common.systems import DuffingOscillator, VanDerPolOscillator  # noqa: E402
from common.models import MLPResidual, KANResidual  # noqa: E402
from common.hrpinn import HRPINNCell  # noqa: E402
from common import npref, quant_sim  # noqa: E402
from export_format import read_khrp  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MODELS_DIR = os.path.join(REPO, 'models')
GOLD_DIR = os.path.join(REPO, 'data', 'goldens')

TRAJ_ICS = [(1.0, 0.0), (-1.5, 1.0), (0.5, -1.2)]  # physical
TRAJ_STEPS = 500


class NormShim:
    def __init__(self, min_vals, range_vals):
        self.min_vals = np.asarray(min_vals, dtype=np.float64)
        self.range = np.asarray(range_vals, dtype=np.float64)

    def normalize(self, data):
        return 2 * (data - self.min_vals) / self.range - 1


def get_sys(name):
    return DuffingOscillator() if name.lower().startswith('duf') else VanDerPolOscillator()


def build_inputs(kmodel, cell, seed=123):
    rng = np.random.RandomState(seed)
    parts = [rng.uniform(-1, 1, size=(128, 2))]

    # Near-knot inputs (KAN bin boundaries); uniform filler for MLP
    if kmodel['arch'] == 'kan':
        layer0 = kmodel['layers'][0]
        k = layer0['k']
        knots_x = layer0['grid'][0][k:-k]  # interior knots, feature 0
        knots_v = layer0['grid'][1][k:-k]
        near = []
        i = 0
        while len(near) < 64:
            kx = knots_x[i % len(knots_x)]
            kv = knots_v[(i // len(knots_x)) % len(knots_v)]
            eps = 1e-4 if i % 2 == 0 else -1e-4
            near.append([kx + eps, kv - eps])
            i += 1
        parts.append(np.array(near))
    else:
        parts.append(rng.uniform(-1, 1, size=(64, 2)))

    # Out-of-grid extrapolation inputs
    mag = rng.uniform(1.2, 2.0, size=(32, 2))
    sgn = rng.choice([-1.0, 1.0], size=(32, 2))
    parts.append(mag * sgn)

    # On-trajectory states (rollout from first traj IC)
    s = torch.tensor([norm_ic(kmodel, TRAJ_ICS[0])], dtype=torch.float32)
    traj_states = []
    with torch.no_grad():
        for i in range(32 * 5):
            s = cell(s)
            if i % 5 == 0:
                traj_states.append(s.numpy()[0].copy())
    ts = np.array(traj_states)
    ts = np.nan_to_num(ts, nan=0.0, posinf=2.0, neginf=-2.0)  # untrained models may blow up
    ts = np.clip(ts, -2.0, 2.0)
    parts.append(ts)

    return np.vstack(parts).astype(np.float32)


def norm_ic(kmodel, ic):
    nmin, nrng = kmodel['norm_min'].astype(np.float64), kmodel['norm_range'].astype(np.float64)
    return (2 * (np.asarray(ic, dtype=np.float64) - nmin) / nrng - 1).astype(np.float32)


def write_io_bin(path, xs, outs):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(xs)))
        rec = np.column_stack([xs[:, 0], xs[:, 1], outs[:, 0]]).astype('<f4')
        f.write(rec.tobytes())


def write_step_bin(path, s0, s1):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(s0)))
        rec = np.column_stack([s0, s1]).astype('<f4')
        f.write(rec.tobytes())


def rebuild_torch(pt_path):
    blob = torch.load(pt_path, map_location='cpu', weights_only=False)
    entry = blob['entry']
    if entry['arch'] == 'kan':
        model = KANResidual(**entry['model_kwargs'])
    else:
        model = MLPResidual(**entry['model_kwargs'])
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, blob


def process_model(khrp_path):
    slug = os.path.splitext(os.path.basename(khrp_path))[0]
    out_dir = os.path.join(GOLD_DIR, slug)
    os.makedirs(out_dir, exist_ok=True)

    kmodel = read_khrp(khrp_path)
    model, blob = rebuild_torch(os.path.join(MODELS_DIR, 'pt', slug + '.pt'))
    system = get_sys(kmodel['system'])
    norm = NormShim(blob['norm_min'], blob['norm_range'])
    cell = HRPINNCell(system, model, norm, dt=kmodel['dt'])

    # --- residual_io ---
    xs = build_inputs(kmodel, cell)
    with torch.no_grad():
        outs = model(torch.from_numpy(xs)).numpy().astype(np.float32)
    write_io_bin(os.path.join(out_dir, 'residual_io.bin'), xs, outs)

    # Export sanity: numpy reference vs torch on the same inputs
    y_np = npref.forward(kmodel, xs)
    denom = max(1e-6, float(np.max(np.abs(outs))))
    np_torch_rel = float(np.max(np.abs(y_np - outs))) / denom

    # --- step_io ---
    rng = np.random.RandomState(321)
    s0 = rng.uniform(-1, 1, size=(64, 2)).astype(np.float32)
    with torch.no_grad():
        s1 = cell(torch.from_numpy(s0)).numpy().astype(np.float32)
    write_step_bin(os.path.join(out_dir, 'step_io.bin'), s0, s1)

    # --- trajectory golden ---
    traj_rows = []
    traj_valid = []
    for ic_idx, ic in enumerate(TRAJ_ICS):
        s = torch.tensor([norm_ic(kmodel, ic)], dtype=torch.float32)
        states = [s.numpy()[0].copy()]
        with torch.no_grad():
            for _ in range(TRAJ_STEPS):
                s = cell(s)
                states.append(s.numpy()[0].copy())
        states = np.array(states)
        finite = bool(np.all(np.isfinite(states)) and np.max(np.abs(states)) < 3.0)
        traj_valid.append(finite)
        for step, st in enumerate(states):
            traj_rows.append(f"{ic_idx},{step},{float(st[0])!r},{float(st[1])!r}")
    with open(os.path.join(out_dir, 'traj_f32.csv'), 'w') as f:
        f.write("ic,step,x,v\n")
        f.write("\n".join(traj_rows) + "\n")

    # --- quantized branch goldens (full branch quantization) ---
    for bits in (8, 16):
        yq = quant_sim.branch_forward_quant(kmodel, xs, bits,
                                            quant_input=True, quant_weights=True,
                                            quant_acts=True)
        write_io_bin(os.path.join(out_dir, f'quant_io_b{bits}.bin'), xs, yq)

    # Output LSB (scale of the last layer act quantizer) for the 1-LSB test tolerance
    last = kmodel['layers'][-1]
    lsb8 = quant_sim.AffineQuant(*last['act_range'], 8).scale
    lsb16 = quant_sim.AffineQuant(*last['act_range'], 16).scale

    with open(os.path.join(out_dir, 'golden_manifest.json'), 'w') as f:
        json.dump({
            'slug': slug, 'n_residual': len(xs), 'n_step': len(s0),
            'traj_steps': TRAJ_STEPS, 'traj_ics': TRAJ_ICS, 'traj_valid': traj_valid,
            'np_torch_max_rel_err': np_torch_rel,
            'quant_out_lsb_b8': float(lsb8), 'quant_out_lsb_b16': float(lsb16),
            'trained_ok': bool(blob.get('trained_ok', True)),
        }, f, indent=2)

    status = "OK" if np_torch_rel < 1e-4 else "WARN(np/torch mismatch)"
    print(f"[{slug}] goldens written ({status}, np/torch rel={np_torch_rel:.2e}, "
          f"traj_valid={traj_valid})", flush=True)
    return np_torch_rel < 1e-4


def write_globals():
    os.makedirs(GOLD_DIR, exist_ok=True)
    # Dependability initial conditions (physical), seed 7
    rng = np.random.RandomState(7)
    ics = rng.uniform(-1.5, 1.5, size=(20, 2))
    with open(os.path.join(GOLD_DIR, 'ics.csv'), 'w') as f:
        f.write("ic,x0,v0\n")
        for i, (x0, v0) in enumerate(ics):
            f.write(f"{i},{float(x0)!r},{float(v0)!r}\n")

    # High-accuracy truth trajectories for the C++ RK4 unit test
    for system in (DuffingOscillator(), VanDerPolOscillator()):
        def rhs(t, y):
            return system.full_dynamics(y, t)
        t_eval = np.arange(0, 100.0 + 0.05, 0.05)
        sol = solve_ivp(rhs, (0, 100.0), [1.0, 0.0], t_eval=t_eval,
                        rtol=1e-10, atol=1e-12, method='RK45')
        name = 'duffing' if system.name() == 'Duffing' else 'vdp'
        with open(os.path.join(GOLD_DIR, f'truth_{name}.csv'), 'w') as f:
            f.write("t,x,v\n")
            for t, x, v in zip(sol.t, sol.y[0], sol.y[1]):
                f.write(f"{float(t)!r},{float(x)!r},{float(v)!r}\n")
    print("Global goldens written (ics.csv, truth_*.csv)", flush=True)


def main():
    write_globals()
    ok = True
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, '*.khrp')))
    if not paths:
        print("No .khrp models found — run train_and_export.py first", file=sys.stderr)
        sys.exit(1)
    for p in paths:
        ok &= process_model(p)
    print("MAKE-GOLDENS-" + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
