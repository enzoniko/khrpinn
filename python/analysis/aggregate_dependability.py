"""Aggregate the dependability experiments (E4 rollout summaries + curves,
E5 quantized discovery surfaces) into tables and degradation figures.

Only models with trained_ok=true in their manifest enter the tables (untrained
fallback exports are latency subjects only).
"""

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import FIGURES, RESULTS, load_manifests, md_table, read_csv_optional

from common.metrics import discovery_r2_from_surface  # noqa: E402  (path set in common_analysis)
from common.systems import get_system  # noqa: E402

DEP = os.path.join(RESULTS, 'dependability')
PRECS = ['f32', 'i16', 'i8', 'i6', 'i4']


def summary_tables(manifests):
    df = read_csv_optional(os.path.join(DEP, 'summaries.csv'))
    if df is None:
        return None, None
    ok = {s for s, m in manifests.items() if m.get('trained_ok')}
    df = df[df.model.isin(ok)].copy()
    df['arch'] = df.model.map(lambda s: manifests[s]['arch'])
    df['system'] = df.model.map(lambda s: manifests[s]['system'])
    steps = int(df.steps_requested.max())
    # divergence_step == -1 means "never" -> censor at steps+1 for medians
    df['div_step_c'] = df.divergence_step.where(df.divergence_step >= 0, steps + 1)

    full = df[(df.qmode == 'full') & (df.ref == 'f32')]
    rows = []
    for (model, prec), g in full.groupby(['model', 'precision']):
        rows.append({
            'model': model, 'arch': g.arch.iloc[0], 'system': g.system.iloc[0],
            'precision': prec,
            'diverged_frac': g.diverged.mean(),
            'median_div_step': g.div_step_c.median(),
            'median_mse': g.mse.median(),
            'max_err_median': g.max_err.median(),
        })
    tab = pd.DataFrame(rows)
    tab['prec_order'] = tab.precision.map({p: i for i, p in enumerate(PRECS)})
    tab = tab.sort_values(['system', 'arch', 'model', 'prec_order']).drop(columns='prec_order')
    tab.to_csv(os.path.join(DEP, 'aggregated_full.csv'), index=False)

    # Attribution view (i8): which quantization point drives the error?
    att = df[(df.precision == 'i8') & (df.ref == 'f32')]
    arows = []
    for (model, qmode), g in att.groupby(['model', 'qmode']):
        arows.append({'model': model, 'arch': g.arch.iloc[0], 'qmode': qmode,
                      'diverged_frac': g.diverged.mean(),
                      'median_mse': g.mse.median()})
    atab = pd.DataFrame(arows).sort_values(['model', 'qmode'])
    atab.to_csv(os.path.join(DEP, 'aggregated_attribution_i8.csv'), index=False)
    return tab, atab


def discovery_table(manifests):
    files = sorted(glob.glob(os.path.join(DEP, 'discovery', '*.csv')))
    if not files:
        return None
    rows = []
    for path in files:
        mobj = re.match(r'(.+)_(f32|i16|i8|i6|i4)_(\w+)\.csv',
                        os.path.basename(path))
        if not mobj:
            continue
        slug, prec, qmode = mobj.groups()
        man = manifests.get(slug)
        if not man or not man.get('trained_ok'):
            continue
        df = pd.read_csv(path)
        n = int(df.i.max()) + 1
        surface = np.zeros((n, n))
        surface[df.j.values, df.i.values] = df.residual.values  # [j, i] like metrics.py
        r2 = discovery_r2_from_surface(surface, get_system(man['system']))
        rows.append({'model': slug, 'arch': man['arch'], 'system': man['system'],
                     'precision': prec, 'qmode': qmode, 'discovery_r2': r2})
    tab = pd.DataFrame(rows)
    piv = tab.pivot_table(index=['system', 'arch', 'model'], columns='precision',
                          values='discovery_r2').reset_index()
    order = [c for c in ['system', 'arch', 'model'] + PRECS if c in piv.columns]
    piv = piv[order]
    piv.to_csv(os.path.join(DEP, 'discovery_r2_by_precision.csv'), index=False)
    return piv


def degradation_figures(manifests):
    """Per-system error-growth curves (IC 0, qmode=full, ref=f32)."""
    os.makedirs(FIGURES, exist_ok=True)
    made = []
    for system in ('duffing', 'vdp'):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), sharey=True)
        for ax, prec in zip(axes, ['i16', 'i8', 'i6', 'i4']):
            for path in sorted(glob.glob(os.path.join(DEP, 'curves', f'{system}_*_{prec}_full.csv'))):
                slug = os.path.basename(path).replace(f'_{prec}_full.csv', '')
                man = manifests.get(slug)
                if not man or not man.get('trained_ok'):
                    continue
                df = pd.read_csv(path)
                color = 'tab:red' if man['arch'] == 'kan' else 'tab:blue'
                ax.semilogy(df.step, df.err_norm.clip(lower=1e-9), color=color,
                            alpha=0.7, lw=1.2, label=slug.replace(f'{system}_', ''))
            ax.set_title(f'{system} — {prec} (qmode=full)')
            ax.set_xlabel('step (dt=0.05s)')
            ax.grid(alpha=0.3)
            ax.axhline(0.1, color='k', ls='--', lw=0.8)
        axes[0].set_ylabel('normalized L2 error vs FP32 rollout')
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(fontsize=7, loc='lower right')
        fig.suptitle(f'{system.upper()}: trajectory error growth under quantization '
                     f'(KAN=red, MLP=blue; dashed = divergence threshold)')
        fig.tight_layout()
        out = os.path.join(FIGURES, f'degradation_{system}.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        made.append(out)
    return made


def run():
    manifests = load_manifests()
    tab, atab = summary_tables(manifests)
    if tab is None:
        return None
    piv = discovery_table(manifests)
    figs = degradation_figures(manifests)

    md = ["### Dependability (E4): closed-loop divergence vs precision "
          "(qmode=full, ref=f32, 20 ICs, 2000 steps)\n"]
    md.append(md_table(tab, floatfmt='{:.4g}'))
    md.append("\n### Attribution at INT8 (E4b): which quantization point hurts?\n")
    md.append(md_table(atab, floatfmt='{:.4g}'))
    if piv is not None:
        md.append("\n### Quantized Discovery R^2 (E5) — the ICLR metric through the quantized engine\n")
        md.append(md_table(piv, floatfmt='{:.4f}'))
    for f in figs:
        md.append(f"\n![degradation]({os.path.relpath(f, RESULTS)})")
    return '\n'.join(md)


if __name__ == '__main__':
    out = run()
    print(out if out else "(no dependability results yet)")
