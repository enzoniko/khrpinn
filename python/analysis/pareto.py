"""Accuracy–latency Pareto figure: chosen-seed Discovery R^2 (the deployed
model's quality) vs residual-branch latency, per system. Preempts the
"KAN needs fewer params for equal accuracy" counter-argument by showing the
whole frontier rather than equal-parameter pairs.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common_analysis import FIGURES, RESULTS, load_manifests, read_csv_optional


def run(host='pc'):
    agg = read_csv_optional(os.path.join(RESULTS, f'latency_{host}', 'aggregated.csv'))
    if agg is None:
        return None
    manifests = load_manifests()
    res = agg[agg['mode'] == 'residual'].copy()
    res['trained_ok'] = res.model.map(lambda s: manifests.get(s, {}).get('trained_ok', False))
    res = res[res.trained_ok]

    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, system in zip(axes, ('Duffing', 'VdP')):
        g = res[res.system == system]
        for arch, color, marker in (('kan', 'tab:red', 'o'), ('mlp', 'tab:blue', 's')):
            gg = g[g.arch == arch]
            ax.scatter(gg.ns_median, gg.chosen_r2.clip(lower=-0.1),
                       s=20 + gg.params / 12.0, c=color, marker=marker,
                       alpha=0.8, label=arch.upper())
            for _, r in gg.iterrows():
                ax.annotate(r.model.split('_', 1)[1], (r.ns_median, min(1.0, max(-0.1, r.chosen_r2))),
                            fontsize=7, xytext=(4, 4), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_xlabel(f'residual-branch latency on {host.upper()} (ns, median)')
        ax.set_ylabel('Discovery $R^2$ (exported median seed)')
        ax.set_title(system)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('Accuracy vs latency Pareto (marker size ~ parameter count)')
    fig.tight_layout()
    out = os.path.join(FIGURES, f'pareto_{host}.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == '__main__':
    for host in ('pc', 'board'):
        out = run(host)
        print(out if out else f"(no latency_{host} aggregate yet)")
