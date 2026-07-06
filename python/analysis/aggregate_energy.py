"""Aggregate the manual energy session (E3): join the USB power meter log
(energy_log.csv, filled by hand from energy_log_template.csv) with the
benchmark's energy-loop rows (energy_runs.csv) and compute uJ/step.
"""

import os

import pandas as pd

from common_analysis import RESULTS, load_manifests, md_table, read_csv_optional

# KHRPINN_ENERGY_DIR overrides the source dir (used by the dry-run validation
# against results/energy_board_simulated; real sessions use energy_board).
NRG = os.environ.get('KHRPINN_ENERGY_DIR', os.path.join(RESULTS, 'energy_board'))
RUN_COLS = ['host', 'compiler', 'arch', 'precision', 'qmode', 'system', 'model',
            'params', 'mode', 'loop_seconds', 'steps', 'elapsed_s',
            'steps_per_sec', 'checksum']


def run():
    log = read_csv_optional(os.path.join(NRG, 'energy_log.csv'))
    runs_path = os.path.join(NRG, 'energy_runs.csv')
    if log is None or not os.path.exists(runs_path):
        return None
    runs = pd.read_csv(runs_path, names=RUN_COLS, header=None)
    runs = runs[runs['mode'] == 'energy']
    log = log.dropna(subset=['p_idle_w', 'p_load_w'])
    if not len(log):
        return None

    manifests = load_manifests()
    rows = []
    for _, r in log.iterrows():
        g = runs[runs.model == r.model]
        if not len(g):
            continue
        sps = g.steps_per_sec.astype(float).median()
        uj = (float(r.p_load_w) - float(r.p_idle_w)) / sps * 1e6
        man = manifests.get(r.model, {})
        rows.append({'model': r.model, 'arch': man.get('arch', '?'),
                     'params': man.get('params', 0),
                     'p_idle_w': float(r.p_idle_w), 'p_load_w': float(r.p_load_w),
                     'delta_w': float(r.p_load_w) - float(r.p_idle_w),
                     'steps_per_sec': sps, 'uJ_per_step': uj})
    if not rows:
        return None
    tab = pd.DataFrame(rows).sort_values('model')
    tab.to_csv(os.path.join(NRG, 'aggregated.csv'), index=False)
    md = ["### Energy per HRPINN step (E3, VisionFive 2, FP32, differential USB meter)\n",
          md_table(tab, floatfmt='{:.4g}')]
    return '\n'.join(md)


if __name__ == '__main__':
    out = run()
    print(out if out else "(no energy results yet — manual board session pending; "
                          "see scripts/energy_protocol.md)")
