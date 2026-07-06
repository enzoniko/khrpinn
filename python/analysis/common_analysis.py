"""Shared helpers for the analysis/aggregation scripts."""

import glob
import json
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RESULTS = os.path.join(REPO, 'results')
MODELS = os.path.join(REPO, 'models')
FIGURES = os.path.join(RESULTS, 'figures')

sys.path.insert(0, os.path.join(REPO, 'python'))


def load_manifests():
    """slug -> manifest dict, plus derived fields (chosen-seed metrics)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(MODELS, '*.json'))):
        with open(p) as f:
            man = json.load(f)
        chosen = next((s for s in man['seed_metrics'] if s['seed'] == man['chosen_seed']),
                      None)
        man['chosen_r2'] = chosen['discovery_r2'] if chosen else float('nan')
        man['chosen_test_mse'] = chosen['test_mse'] if chosen else float('nan')
        out[man['slug']] = man
    return out


def read_csv_optional(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def md_table(df, floatfmt='{:.3g}'):
    """Render a DataFrame as a GitHub markdown table (no external deps)."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def model_sort_key(slug):
    order = ['kan_very_small', 'kan_small', 'kan_wide', 'kan_deep',
             'kan_configA', 'kan_configF',
             'mlp_tiny', 'mlp_small', 'mlp_medium', 'mlp_large']
    for i, o in enumerate(order):
        if slug.endswith(o):
            return (0 if slug.startswith('duffing') else 1, i)
    return (2, 0)
