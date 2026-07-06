"""Regenerate results/SUMMARY.md — the results index.

Runs every aggregator over whatever raw results exist and writes one markdown
document with every table the SBESC paper needs, each traceable to raw CSVs.

Usage: python python/analysis/build_summary.py
"""

import datetime
import os
import subprocess

import pandas as pd

import aggregate_dependability
import aggregate_energy
import aggregate_latency
import pareto
from common_analysis import MODELS, RESULTS, load_manifests, md_table


def training_section(manifests):
    rows = []
    for slug, m in sorted(manifests.items()):
        rows.append({
            'model': slug, 'arch': m['arch'], 'system': m['system'],
            'recipe': m['recipe'], 'params': m['params'],
            'chosen_seed': m['chosen_seed'], 'trained_ok': m['trained_ok'],
            'discovery_r2': m['chosen_r2'], 'test_mse': m['chosen_test_mse'],
            'flops/inf': m['flops_per_inference'], 'exp/inf': m['exps_per_inference'],
        })
    return md_table(pd.DataFrame(rows), floatfmt='{:.4g}')


def memory_section(manifests):
    rows = []
    for slug, m in sorted(manifests.items()):
        wb = m.get('weight_bytes_f32', 0)
        rows.append({'model': slug, 'params': m['params'],
                     'bytes_f32': wb, 'bytes_i16': wb // 2, 'bytes_i8': wb // 4})
    return md_table(pd.DataFrame(rows))


def git_describe():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=RESULTS, text=True).strip()
    except Exception:
        return 'unknown'


def main():
    manifests = load_manifests()
    parts = []
    parts.append(f"# SBESC 2026 results summary\n")
    parts.append(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
                 f"at commit `{git_describe()}` by python/analysis/build_summary.py. "
                 f"Every number traces to a raw CSV under results/.\n")
    parts.append("Units: latency in **ns per call**; energy in **uJ per step**; "
                 "errors in the **normalized state domain** ([-1,1]^2); "
                 "Discovery R^2 on the frozen ICLR 50x50 grid over [-2,2]^2.\n")

    parts.append("## Exported models (median-of-5-seeds, BPTT recipe unless noted)\n")
    parts.append("Raw: `models/training_metrics.csv`, per-model `models/*.json`.\n")
    parts.append(training_section(manifests))

    for host in ('pc', 'board'):
        md = aggregate_latency.run(host)
        if md:
            parts.append(f"\n## Latency — {host.upper()} (E1)\n")
            parts.append(f"Raw: `results/latency_{host}/latency.csv`.\n")
            parts.append(md)
            fig = pareto.run(host)
            if fig:
                parts.append(f"\n![pareto]({os.path.relpath(fig, RESULTS)})")
        else:
            parts.append(f"\n## Latency — {host.upper()} (E1)\n\n*(pending — "
                         f"run scripts/run_bench_{'pc.sh' if host == 'pc' else 'board.sh (on the board)'})*")

    perf = os.path.join(RESULTS, 'perf_pc', 'perf.csv')
    parts.append("\n## Microarchitectural counters (E2)\n")
    found_perf = False
    for host in ('pc', 'board'):
        p = os.path.join(RESULTS, f'perf_{host}', 'perf.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            if len(df):
                parts.append(f"\nRaw: `results/perf_{host}/perf.csv`.\n")
                parts.append(md_table(df))
                found_perf = True
    if not found_perf:
        parts.append("*(pending — perf unavailable under WSL2; board run is authoritative. "
                     "Analytical per-inference op counts are in the model table above.)*")

    md = aggregate_energy.run()
    parts.append("\n## Energy (E3)\n")
    if md:
        parts.append("Raw: `results/energy_board/energy_log.csv` + `energy_runs.csv`.\n")
        parts.append(md)
    else:
        parts.append("*(pending — manual board session; see scripts/energy_protocol.md)*")

    md = aggregate_dependability.run()
    parts.append("\n## Quantization dependability (E4/E5)\n")
    if md:
        parts.append("Raw: `results/dependability/summaries.csv`, `curves/`, `discovery/`.\n")
        parts.append(md)
    else:
        parts.append("*(pending — run scripts/run_dependability.sh)*")

    parts.append("\n## Memory footprint (E6)\n")
    parts.append(memory_section(manifests))

    parts.append("""
## Claims checklist (hypothesis -> evidence)

| Hypothesis | Evidence | Where |
|---|---|---|
| H1: KAN has higher per-step latency than MLP at comparable params | `latency_ratio` column | latency tables, `results/latency_board/aggregated.csv` |
| H1 mechanism: transcendental + basis-recursion overhead | `flops_est`, `exps_est`; branch/cache counters | model table; `results/perf_*/perf.csv` |
| H2: KAN consumes more uJ/step | `uJ_per_step` column | `results/energy_board/aggregated.csv` |
| H3: KAN degrades worse under low-bit quantization in the loop | `diverged_frac`, `median_div_step` by precision | `results/dependability/aggregated_full.csv` |
| H3 mechanism attribution | i8 input/weights/acts-only rows | `results/dependability/aggregated_attribution_i8.csv` |
| Discovery quality vs precision (ICLR metric bridge) | quantized Discovery R^2 | `results/dependability/discovery_r2_by_precision.csv` |
| Accuracy-latency trade-off (anti-strawman) | Pareto figures | `results/figures/pareto_*.png` |
""")

    out = os.path.join(RESULTS, 'SUMMARY.md')
    with open(out, 'w') as f:
        f.write('\n'.join(parts) + '\n')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
