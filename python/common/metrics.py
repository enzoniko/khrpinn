"""Discovery metrics + symbolic extraction (verbatim from kan_hrpinn_paper_bptt.py).

NOTE (documented discrepancy): the ICLR paper text says 100x100 over [-2.5, 2.5],
but every frozen script uses 50x50 over [-2, 2]. The SBESC paper must describe the
code's actual behavior; all new results keep 50x50 over [-2, 2] for consistency
with the archived ICLR numbers.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DISCOVERY_GRID = np.linspace(-2, 2, 50)


def extract_symbolic_term(model, system, normalizer, x_range=None, v_range=None, device='cpu'):
    """Extract symbolic term by linear regression on candidate features."""
    x_range = DISCOVERY_GRID if x_range is None else x_range
    v_range = DISCOVERY_GRID if v_range is None else v_range
    model.eval()

    candidates = {
        'x': lambda x, v: x,
        'v': lambda x, v: v,
        'x^3': lambda x, v: x**3,
        'xv': lambda x, v: x * v,
        'x^2v': lambda x, v: x**2 * v,
    }

    X_grid, V_grid = np.meshgrid(x_range, v_range)
    learned_residual = np.zeros_like(X_grid)

    min_vals = torch.tensor(normalizer.min_vals, dtype=torch.float32, device=device)
    range_vals = torch.tensor(normalizer.range, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                state = torch.tensor([[x, v]], dtype=torch.float32, device=device)
                state_norm = 2 * (state - min_vals) / range_vals - 1
                residual = model(state_norm).cpu().numpy()[0, 0]
                learned_residual[j, i] = residual

    x_flat = X_grid.flatten()
    v_flat = V_grid.flatten()
    residual_flat = learned_residual.flatten()

    features = []
    feature_names = []
    for name, func in candidates.items():
        feat = func(x_flat, v_flat)
        features.append(feat)
        feature_names.append(name)

    X_features = np.column_stack(features)

    reg = LinearRegression()
    reg.fit(X_features, residual_flat)

    coefs = np.abs(reg.coef_)
    best_idx = np.argmax(coefs)
    best_candidate = feature_names[best_idx]
    best_coef = reg.coef_[best_idx]

    y_pred_single = reg.coef_[best_idx] * X_features[:, best_idx]
    r2 = r2_score(residual_flat, y_pred_single)

    return best_candidate, best_coef, r2


def compute_discovery_metrics(model, system, normalizer, x_range=None, v_range=None, device='cpu'):
    """Compute discovery MSE and R^2 on the 50x50 [-2,2] grid."""
    x_range = DISCOVERY_GRID if x_range is None else x_range
    v_range = DISCOVERY_GRID if v_range is None else v_range
    model.eval()

    X_grid, V_grid = np.meshgrid(x_range, v_range)
    learned_residual = np.zeros_like(X_grid)
    true_residual = np.zeros_like(X_grid)

    min_vals = torch.tensor(normalizer.min_vals, dtype=torch.float32, device=device)
    range_vals = torch.tensor(normalizer.range, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, x in enumerate(x_range):
            for j, v in enumerate(v_range):
                state = torch.tensor([[x, v]], dtype=torch.float32, device=device)
                state_norm = 2 * (state - min_vals) / range_vals - 1
                residual = model(state_norm).cpu().numpy()[0, 0]
                learned_residual[j, i] = residual
                true_residual[j, i] = system.residual_true(np.array([x, v]))

    mse = np.mean((learned_residual - true_residual) ** 2)
    ss_res = np.sum((true_residual - learned_residual) ** 2)
    ss_tot = np.sum((true_residual - np.mean(true_residual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return mse, r2, learned_residual, true_residual


def discovery_r2_from_surface(learned_residual, system, x_range=None, v_range=None):
    """R^2 of an externally computed residual surface (e.g., from the C++ engine).

    learned_residual: array [len(v_range), len(x_range)] indexed [j, i] like above.
    Uses the exact same formula as compute_discovery_metrics.
    """
    x_range = DISCOVERY_GRID if x_range is None else x_range
    v_range = DISCOVERY_GRID if v_range is None else v_range
    true_residual = np.zeros_like(learned_residual)
    for i, x in enumerate(x_range):
        for j, v in enumerate(v_range):
            true_residual[j, i] = system.residual_true(np.array([x, v]))
    ss_res = np.sum((true_residual - learned_residual) ** 2)
    ss_tot = np.sum((true_residual - np.mean(true_residual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
