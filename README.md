# khrpinn

Research code for comparing B-spline Kolmogorov–Arnold networks (KANs) and ReLU
multilayer perceptrons (MLPs) as learned residual functions in recurrent
physics-informed models. The repository contains oscillator identification
experiments, a scalar C++17 inference engine, model exports, reference outputs,
and evaluation scripts.

The identification study is described in [Empirical Stability Analysis of
Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed
Discovery](https://arxiv.org/abs/2602.09988). The original experiment scripts and
recorded results are retained under `python/iclr/` and `results/iclr/`.

## Repository layout

| Directory | Contents |
|---|---|
| `python/iclr/` | Original identification and configuration-ablation experiments |
| `python/common/` | Shared dynamics, training, metrics, and simulated quantization |
| `python/export/` | Training, model export, calibration, and reference-vector generation |
| `python/analysis/` | Aggregation and plots from recorded results |
| `python/tests/` | Checks against the original Python implementations |
| `cpp/` | C++17 inference tools and unit/reference tests |
| `models/` | Exported models, selection metrics, and calibration manifests |
| `data/goldens/` | Reference inputs, outputs, and trajectories |
| `scripts/` | Experiment runners and measurement protocols |
| `results/` | Raw measurements, aggregates, and `SUMMARY.md` |
| `deploy/` | RISC-V deployment files |
| `paper/` | Earlier SBESC manuscript draft and bibliography |
| `docs/validation.md` | Numerical conventions, validation, and evaluation limits |

## Build and test

The C++ engine requires a C++17 compiler and Make. Its reference tests use the
committed models and vectors and can run without retraining.

```sh
make -C cpp -j8 all
make -C cpp test
```

Python training and analysis use NumPy, SciPy, pandas, Matplotlib, scikit-learn,
and PyTorch. To regenerate model exports and reference vectors:

```sh
python python/export/train_and_export.py --n-seeds 5 --jobs 10
python python/export/make_goldens.py
```

Training is optional when using the committed exports. It can take approximately
an hour with the configuration above; KAN grid updates can produce small
floating-point differences across linear-algebra runtimes.

## Evaluation

```sh
sh scripts/run_bench_pc.sh
sh scripts/run_bench_recip.sh
sh scripts/run_perf_pc.sh
sh scripts/run_dependability.sh
python python/analysis/build_summary.py
```

The reciprocal-basis benchmark measures the spline implementation with
precomputed knot reciprocals. Performance counters are optional when `perf` is
unavailable. See `results/SUMMARY.md` for recorded results and raw CSV locations.

For a StarFive VisionFive 2, the cross-build and reference checks are:

```sh
make -C cpp riscv test-riscv deploy
```

Follow `scripts/board_runbook.md` and `scripts/energy_protocol.md` for board
measurements. QEMU checks numerical behavior; it does not measure board timing
or power. `scripts/simulate_energy_session.sh` tests the aggregation pipeline
with synthetic meter readings stored separately in `results/energy_board_simulated/`.
Those readings are not hardware energy measurements.

## Interpreting the results

- Exported checkpoints are selected by median discovery R² among five seeds.
  Manifests record the seed metrics and `trained_ok` status.
- Discovery R² is evaluated on a 50×50 physical grid over `[-2,2]²`. This is the
  grid implemented in the original code; an earlier manuscript described a
  different grid.
- The quantization reference uses floating-point arithmetic to simulate rounding
  and clipping. It measures prediction sensitivity, not the speed of an integer
  inference kernel. Timing and board-energy experiments use FP32 inference.
- State errors use the per-coordinate normalization stored with each model.
- `paper/sbesc2026.tex` is an earlier draft with pending board-data placeholders;
  it is not a final published record. Measured and simulated outputs remain
  distinguishable in the result directories.

Numerical tolerances, calibration effects, and reproducibility details are in
`docs/validation.md`.
