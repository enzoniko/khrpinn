"""Train the SBESC model matrix and export .khrp models + manifests.

Matrix (20 models):
  - 16 BPTT models: {Duffing, VdP} x {KAN VS/S/W/D, MLP T/S/M/L}, exactly the
    ICLR Table 2 (BPTT) recipe from kan_hrpinn_paper_bptt.py.
  - 4 TF models: {Duffing, VdP} x {KAN Config A, KAN Config F}, exactly the
    ICLR Table 1 recipe from kan_hrpinn_paper_config_ablation.py.

For each model: run seeds 1..N (default 5), pick the MEDIAN seed by Discovery R^2
(representative, not cherry-picked), export it. All per-seed metrics go to
models/training_metrics.csv and into the manifest.

Usage:
  python train_and_export.py --n-seeds 5 --jobs 5 --out ../../models
  python train_and_export.py --fast            # 3 epochs / 2 seeds smoke test
  python train_and_export.py --only duffing_kan_very_small
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from common.systems import DuffingOscillator, VanDerPolOscillator  # noqa: E402
from common.data import generate_training_data_tf, generate_training_data_bptt, training_states  # noqa: E402
from common.models import MLPResidual, KANResidual  # noqa: E402
from common.training import train_model_tf, train_model_bptt  # noqa: E402
from common.metrics import compute_discovery_metrics, extract_symbolic_term  # noqa: E402
from common import npref  # noqa: E402
from export_format import kmodel_from_torch, write_khrp, write_manifest  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

BPTT_TRAIN_KAN = {'grid_update_freq': 5, 'early_stop_patience': 50,
                  'reg_entropy': 0.0, 'reg_activation': 0.0}
BPTT_TRAIN_MLP = {'grid_update_freq': 0, 'early_stop_patience': 50,
                  'reg_entropy': 0.0, 'reg_activation': 0.0}


def build_matrix():
    entries = []
    kan_sizes = {
        'kan_very_small': [2, 4, 1],
        'kan_small': [2, 8, 1],
        'kan_wide': [2, 16, 1],
        'kan_deep': [2, 8, 8, 1],
    }
    mlp_sizes = {
        'mlp_tiny': [8, 8],
        'mlp_small': [16, 16],
        'mlp_medium': [32, 32],
        'mlp_large': [64, 64],
    }
    for sysname in ('duffing', 'vdp'):
        for cfg, layers in kan_sizes.items():
            entries.append({
                'slug': f'{sysname}_{cfg}', 'system': sysname, 'arch': 'kan',
                'config_name': cfg.replace('kan_', 'KAN ').replace('_', ' ').title(),
                'recipe': 'bptt',
                'model_kwargs': {'layers_hidden': layers, 'scale_base': 0.1,
                                 'scale_spline': 1.0, 'grid_size': 5},
                'train_kwargs': dict(BPTT_TRAIN_KAN),
            })
        for cfg, hidden in mlp_sizes.items():
            entries.append({
                'slug': f'{sysname}_{cfg}', 'system': sysname, 'arch': 'mlp',
                'config_name': cfg.replace('mlp_', 'MLP ').replace('_', ' ').title(),
                'recipe': 'bptt',
                'model_kwargs': {'input_dim': 2, 'hidden_dims': hidden, 'output_dim': 1},
                'train_kwargs': dict(BPTT_TRAIN_MLP),
            })
        # ICLR Table 1 continuity: Config A (Baseline) and Config F (Coarse-Grid), TF recipe
        entries.append({
            'slug': f'{sysname}_kan_configA', 'system': sysname, 'arch': 'kan',
            'config_name': 'KAN Config A (Baseline, TF)', 'recipe': 'tf',
            'model_kwargs': {'layers_hidden': [2, 8, 1], 'scale_base': 1.0,
                             'scale_spline': 1.0, 'grid_size': 5},
            'train_kwargs': {'grid_update_freq': 20, 'early_stop_patience': 20,
                             'reg_entropy': 0.0, 'reg_activation': 0.0},
        })
        entries.append({
            'slug': f'{sysname}_kan_configF', 'system': sysname, 'arch': 'kan',
            'config_name': 'KAN Config F (Coarse-Grid, TF)', 'recipe': 'tf',
            'model_kwargs': {'layers_hidden': [2, 8, 1], 'scale_base': 0.1,
                             'scale_spline': 1.0, 'grid_size': 3},
            'train_kwargs': {'grid_update_freq': 5, 'early_stop_patience': 50,
                             'reg_entropy': 0.0, 'reg_activation': 0.0},
        })
    return entries


def get_sys(name):
    return DuffingOscillator() if name == 'duffing' else VanDerPolOscillator()


def run_single(entry, seed, epochs):
    """Mirror of the frozen scripts' run_single_experiment (same RNG order)."""
    system = get_sys(entry['system'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        if entry['recipe'] == 'bptt':
            train_loader, test_loader, normalizer, dt = generate_training_data_bptt(
                system, seed=seed, device='cpu')
        else:
            train_loader, test_loader, normalizer, dt = generate_training_data_tf(
                system, seed=seed, device='cpu')

        if entry['arch'] == 'kan':
            model = KANResidual(**entry['model_kwargs'])
        else:
            model = MLPResidual(**entry['model_kwargs'])

        train_fn = train_model_bptt if entry['recipe'] == 'bptt' else train_model_tf
        model, train_losses, test_losses = train_fn(
            system, model, train_loader, test_loader, normalizer, dt,
            epochs=epochs, lr=1e-2, device='cpu', verbose=False,
            **entry['train_kwargs'])

        test_mse = test_losses[-1]
        _, discovery_r2, _, _ = compute_discovery_metrics(model, system, normalizer)
        cand, coef, sym_r2 = extract_symbolic_term(model, system, normalizer)
        if np.isnan(discovery_r2) or np.isinf(discovery_r2):
            discovery_r2 = 0.0
        return {
            'seed': seed, 'success': True, 'test_mse': float(test_mse),
            'discovery_r2': float(discovery_r2), 'symbolic_term': cand,
            'symbolic_coef': float(coef), 'symbolic_r2': float(sym_r2),
            'params': sum(p.numel() for p in model.parameters()),
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'norm_min': normalizer.min_vals.copy(),
            'norm_range': normalizer.range.copy(),
        }
    except Exception as e:  # matches frozen script's failure handling
        return {'seed': seed, 'success': False, 'test_mse': float('nan'),
                'discovery_r2': 0.0, 'symbolic_term': 'N/A', 'symbolic_coef': 0.0,
                'symbolic_r2': 0.0, 'params': 0, 'state_dict': None,
                'norm_min': None, 'norm_range': None, 'error': str(e)}


def process_entry(args):
    entry, n_seeds, epochs, out_dir = args
    torch.set_num_threads(2)
    t0 = time.time()
    results = [run_single(entry, seed, epochs) for seed in range(1, n_seeds + 1)]

    ok = [r for r in results if r['success']]
    if ok:
        ok_sorted = sorted(ok, key=lambda r: r['discovery_r2'])
        chosen = ok_sorted[len(ok_sorted) // 2]  # median by discovery R^2
        trained_ok = True
    else:
        # Export a fresh-init model so latency/energy experiments still have a
        # subject; dependability analysis must skip it (trained_ok=false).
        chosen = None
        trained_ok = False

    system = get_sys(entry['system'])
    if chosen is not None:
        if entry['arch'] == 'kan':
            model = KANResidual(**entry['model_kwargs'])
        else:
            model = MLPResidual(**entry['model_kwargs'])
        model.load_state_dict(chosen['state_dict'])

        class _N:  # minimal normalizer shim
            pass
        norm = _N()
        norm.min_vals = chosen['norm_min']
        norm.range = chosen['norm_range']
        chosen_seed = chosen['seed']
    else:
        torch.manual_seed(1)
        np.random.seed(1)
        if entry['recipe'] == 'bptt':
            _, _, norm, _ = generate_training_data_bptt(system, seed=1, device='cpu')
        else:
            _, _, norm, _ = generate_training_data_tf(system, seed=1, device='cpu')
        if entry['arch'] == 'kan':
            model = KANResidual(**entry['model_kwargs'])
        else:
            model = MLPResidual(**entry['model_kwargs'])
        chosen_seed = 1

    model.eval()

    # Build kmodel, verify folding against torch, calibrate, export.
    kmodel = kmodel_from_torch(model, system.name(), norm, dt=0.05)
    rng = np.random.RandomState(9)
    xs = rng.uniform(-1, 1, size=(128, 2)).astype(np.float32)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(xs)).numpy()
    y_np = npref.forward(kmodel, xs)
    fold_err = float(np.max(np.abs(y_torch - y_np)))
    denom = max(1e-6, float(np.max(np.abs(y_torch))))
    assert fold_err / denom < 1e-4, \
        f"{entry['slug']}: numpy/export forward mismatch {fold_err} (rel {fold_err/denom})"

    cal_states, _ = training_states(system, seed=chosen_seed)
    npref.collect_calibration(kmodel, cal_states)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pt'), exist_ok=True)
    khrp_path = os.path.join(out_dir, entry['slug'] + '.khrp')
    write_khrp(kmodel, khrp_path)
    torch.save({'state_dict': model.state_dict(), 'entry': entry,
                'norm_min': norm.min_vals, 'norm_range': norm.range,
                'chosen_seed': chosen_seed, 'trained_ok': trained_ok},
               os.path.join(out_dir, 'pt', entry['slug'] + '.pt'))

    seed_metrics = [{k: r[k] for k in ('seed', 'success', 'test_mse', 'discovery_r2',
                                       'symbolic_term', 'symbolic_coef', 'symbolic_r2')}
                    for r in results]
    flops = npref.flops_estimate(kmodel)
    write_manifest(
        os.path.join(out_dir, entry['slug'] + '.json'),
        slug=entry['slug'], system=system.name(), arch=entry['arch'],
        config_name=entry['config_name'], recipe=entry['recipe'],
        model_kwargs=entry['model_kwargs'], train_kwargs=entry['train_kwargs'],
        epochs=epochs, n_seeds=n_seeds, chosen_seed=chosen_seed,
        trained_ok=trained_ok, params=npref.param_count(kmodel),
        flops_per_inference=flops['flops'], exps_per_inference=flops['exps'],
        weight_bytes_f32=sum(l['W'].size + l['b'].size if l['type'] == 'mlp'
                             else l['base_w'].size + l['spline_w'].size + l['grid'].size
                             for l in kmodel['layers']) * 4,
        seed_metrics=seed_metrics,
        fold_check_max_abs_err=fold_err,
        in_range=kmodel['in_range'],
        calibration={f"layer{i}": {k: layer[k] for k in
                                   (('silu_range', 'basis_range', 'act_range')
                                    if layer['type'] == 'kan' else ('act_range',))}
                     for i, layer in enumerate(kmodel['layers'])},
    )

    elapsed = time.time() - t0
    rows = []
    for r in results:
        rows.append([entry['slug'], system.name(), entry['arch'], entry['config_name'],
                     entry['recipe'], r['seed'], r['params'], r['test_mse'],
                     r['discovery_r2'], r['symbolic_term'], r['symbolic_coef'],
                     r['symbolic_r2'], r['success'], r['seed'] == chosen_seed])
    r2s = ", ".join("%.3f" % r['discovery_r2'] for r in results)
    print(f"[{entry['slug']}] done in {elapsed:.1f}s; chosen seed {chosen_seed} "
          f"(trained_ok={trained_ok}) r2=[{r2s}]", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(REPO, 'models'))
    ap.add_argument('--only', default=None, help='substring filter on slug')
    ap.add_argument('--fast', action='store_true', help='2 seeds x 3 epochs smoke test')
    args = ap.parse_args()

    n_seeds, epochs = args.n_seeds, args.epochs
    if args.fast:
        n_seeds, epochs = 2, 3

    entries = build_matrix()
    if args.only:
        entries = [e for e in entries if args.only in e['slug']]
    print(f"Training {len(entries)} models x {n_seeds} seeds x {epochs} epochs "
          f"({args.jobs} parallel jobs)", flush=True)

    work = [(e, n_seeds, epochs, args.out) for e in entries]
    if args.jobs > 1:
        with get_context('spawn').Pool(args.jobs) as pool:
            all_rows = pool.map(process_entry, work)
    else:
        all_rows = [process_entry(w) for w in work]

    csv_path = os.path.join(args.out, 'training_metrics.csv')
    write_header = not os.path.exists(csv_path) or args.only is None
    mode = 'w' if write_header else 'a'
    with open(csv_path, mode, newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['slug', 'system', 'arch', 'config_name', 'recipe', 'seed',
                        'params', 'test_mse', 'discovery_r2', 'symbolic_term',
                        'symbolic_coef', 'symbolic_r2', 'success', 'chosen'])
        for rows in all_rows:
            w.writerows(rows)
    print(f"Wrote {csv_path}")
    print("TRAIN-AND-EXPORT-OK")


if __name__ == '__main__':
    main()
