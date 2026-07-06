"""Aggregate latency benchmark CSVs (PC + board) into paper-ready tables.

Reads results/latency_pc/latency.csv and results/latency_board/latency.csv
(if present), collapses the interleaved rounds (median of per-run medians),
joins manifests (params, flops, chosen-seed Discovery R^2), and emits:
  - results/latency_<host>/aggregated.csv
  - markdown tables (returned; written by build_summary.py)
"""

import os

import pandas as pd

from common_analysis import RESULTS, load_manifests, md_table, read_csv_optional, model_sort_key


def aggregate_one(csv_path, manifests):
    df = read_csv_optional(csv_path)
    if df is None:
        return None, None
    rows = []
    for (model, mode), g in df.groupby(['model', 'mode']):
        man = manifests.get(model, {})
        rows.append({
            'model': model,
            'system': man.get('system', '?'),
            'arch': man.get('arch', '?'),
            'params': man.get('params', 0),
            'flops_est': man.get('flops_per_inference', 0),
            'exps_est': man.get('exps_per_inference', 0),
            'mode': mode,
            'ns_median': g['ns_median'].median(),
            'ns_p10': g['ns_p10'].median(),
            'ns_p90': g['ns_p90'].median(),
            'ns_min': g['ns_min'].min(),
            'runs': len(g),
            'chosen_r2': man.get('chosen_r2', float('nan')),
        })
    agg = pd.DataFrame(rows).sort_values('model', key=lambda s: s.map(model_sort_key))
    return df, agg


def kan_mlp_ratios(agg):
    """Latency ratios at comparable parameter count (the H1 headline)."""
    pairs = [('kan_very_small', 'mlp_tiny'), ('kan_small', 'mlp_small'),
             ('kan_wide', 'mlp_small'), ('kan_deep', 'mlp_medium')]
    rows = []
    for system in ('Duffing', 'VdP'):
        sysslug = 'duffing' if system == 'Duffing' else 'vdp'
        for kan, mlp in pairs:
            k = agg[(agg.model == f'{sysslug}_{kan}') & (agg['mode'] == 'residual')]
            m = agg[(agg.model == f'{sysslug}_{mlp}') & (agg['mode'] == 'residual')]
            if len(k) and len(m):
                rows.append({
                    'system': system, 'kan': kan, 'mlp': mlp,
                    'kan_params': int(k.params.iloc[0]), 'mlp_params': int(m.params.iloc[0]),
                    'kan_ns': k.ns_median.iloc[0], 'mlp_ns': m.ns_median.iloc[0],
                    'latency_ratio': k.ns_median.iloc[0] / m.ns_median.iloc[0],
                })
    return pd.DataFrame(rows)


def run(host='pc'):
    manifests = load_manifests()
    csv_path = os.path.join(RESULTS, f'latency_{host}', 'latency.csv')
    _, agg = aggregate_one(csv_path, manifests)
    if agg is None:
        return None
    agg.to_csv(os.path.join(RESULTS, f'latency_{host}', 'aggregated.csv'), index=False)

    md = [f"### Latency ({host.upper()}, FP32, ns per call; median over interleaved rounds)\n"]
    view = agg[['model', 'arch', 'params', 'flops_est', 'exps_est', 'mode',
                'ns_median', 'ns_p10', 'ns_p90']]
    md.append(md_table(view, floatfmt='{:.1f}'))
    ratios = kan_mlp_ratios(agg)
    if len(ratios):
        md.append("\n### KAN/MLP residual-latency ratios at comparable parameter count (H1)\n")
        md.append(md_table(ratios, floatfmt='{:.3g}'))
    return '\n'.join(md)


if __name__ == '__main__':
    for host in ('pc', 'board'):
        out = run(host)
        print(out if out else f"(no latency_{host} results yet)")
