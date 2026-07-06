"""Parity proof: python/common reproduces the frozen ICLR scripts exactly.

Runs one seed of KAN Very Small + MLP Tiny on Duffing through BOTH the frozen
kan_hrpinn_paper_bptt.py functions and the common package, asserting identical
TestMSE / Discovery R^2 (same RNG consumption order => bitwise-identical runs).

Run:  python python/tests/test_parity.py   (takes a couple of minutes)
"""

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, '..')
ICLR = os.path.join(PY, 'iclr')
sys.path.insert(0, PY)
sys.path.insert(0, ICLR)

from common.systems import DuffingOscillator  # noqa: E402
from common.data import generate_training_data_bptt  # noqa: E402
from common.models import KANResidual, MLPResidual  # noqa: E402
from common.training import train_model_bptt  # noqa: E402
from common.metrics import compute_discovery_metrics  # noqa: E402


def load_frozen():
    spec = importlib.util.spec_from_file_location(
        'frozen_bptt', os.path.join(ICLR, 'kan_hrpinn_paper_bptt.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_common(arch, seed, epochs):
    system = DuffingOscillator()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, normalizer, dt = generate_training_data_bptt(
        system, seed=seed, device='cpu')
    if arch == 'kan':
        model = KANResidual(layers_hidden=[2, 4, 1], scale_base=0.1,
                            scale_spline=1.0, grid_size=5)
        tk = {'grid_update_freq': 5, 'early_stop_patience': 50,
              'reg_entropy': 0.0, 'reg_activation': 0.0}
    else:
        model = MLPResidual(input_dim=2, hidden_dims=[8, 8], output_dim=1)
        tk = {'grid_update_freq': 0, 'early_stop_patience': 50,
              'reg_entropy': 0.0, 'reg_activation': 0.0}
    model, _, test_losses = train_model_bptt(
        system, model, train_loader, test_loader, normalizer, dt,
        epochs=epochs, lr=1e-2, device='cpu', verbose=False, **tk)
    _, r2, _, _ = compute_discovery_metrics(model, system, normalizer)
    return test_losses[-1], r2


def run_frozen(frozen, arch, seed, epochs):
    system = frozen.DuffingOscillator()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, normalizer, dt = frozen.generate_training_data(
        system, seed=seed, device='cpu')
    if arch == 'kan':
        model = frozen.KANResidual(layers_hidden=[2, 4, 1], scale_base=0.1,
                                   scale_spline=1.0)
        tk = {'grid_update_freq': 5, 'early_stop_patience': 50,
              'reg_entropy': 0.0, 'reg_activation': 0.0}
    else:
        model = frozen.MLPResidual(input_dim=2, hidden_dims=[8, 8], output_dim=1)
        tk = {'grid_update_freq': 0, 'early_stop_patience': 50,
              'reg_entropy': 0.0, 'reg_activation': 0.0}
    model, _, test_losses = frozen.train_model(
        system, model, train_loader, test_loader, normalizer, dt,
        epochs=epochs, lr=1e-2, device='cpu', verbose=False, **tk)
    _, r2, _, _ = frozen.compute_discovery_metrics(model, system, normalizer)
    return test_losses[-1], r2


def main():
    # Single-threaded for bitwise determinism: the KAN grid-update path calls
    # torch.linalg.lstsq, whose threaded LAPACK reductions are not
    # run-to-run deterministic at >1 thread.
    torch.set_num_threads(1)
    epochs = int(os.environ.get('PARITY_EPOCHS', '10'))
    frozen = load_frozen()
    ok = True
    for arch in ('kan', 'mlp'):
        mse_c, r2_c = run_common(arch, seed=3, epochs=epochs)
        mse_f, r2_f = run_frozen(frozen, arch, seed=3, epochs=epochs)
        exact = np.isclose(mse_c, mse_f, rtol=0, atol=0) and np.isclose(r2_c, r2_f, rtol=0, atol=0)
        close = np.isclose(mse_c, mse_f, rtol=1e-6) and np.isclose(r2_c, r2_f, rtol=1e-6)
        verdict = "IDENTICAL" if exact else ("CLOSE(fp-nondet)" if close else "MISMATCH")
        print(f"parity[{arch}]: common (mse={mse_c:.10f}, r2={r2_c:.10f}) vs "
              f"frozen (mse={mse_f:.10f}, r2={r2_f:.10f}) -> {verdict}")
        ok &= bool(exact or close)
    print("PARITY-" + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
