# khrpinn — KAN vs MLP residual branches in HRPINN: algorithmic + embedded evaluation

Codebase for two papers:

1. **ICLR 2026 workshop (non-archival, frozen):** "Empirical Stability Analysis of
   Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed
   Discovery" — scripts in `python/iclr/`, results in `results/iclr/`. **Do not modify.**
2. **SBESC 2026 (this work):** embedded evaluation of the same models on RISC-V —
   scalar C++ inference engine, latency (H1), energy (H2), and quantization
   dependability (H3). 

## Layout

```
python/iclr/        frozen ICLR scripts (archival record)
python/common/      single-source physics/training/metrics + quant_sim (PTQ spec)
python/export/      train the 20-model matrix, export .khrp + goldens
python/analysis/    aggregate results -> results/SUMMARY.md
python/tests/       parity test (common == frozen scripts, bitwise)
cpp/include/khrpinn zero-dependency C++17 engine (KAN, MLP, HRPINN cell, PTQ)
cpp/src/            CLI tools: infer, verify, bench, traj, discovery
cpp/tests/          unit + golden tests (vs PyTorch / quant_sim)
models/             exported .khrp models + JSON manifests (committed)
data/goldens/       golden vectors + reference trajectories (committed)
scripts/            experiment runners, board runbook, energy protocol
results/            all experiment outputs; SUMMARY.md is the master index
deploy/             RISC-V tarball (built by `make -C cpp deploy`)
paper/              SBESC 2026 draft (sbesc2026.tex; red \tbd{} = pending board data)
```

## Quickstart (PC, WSL Ubuntu)

```sh
# 1. Train + export the 20-model matrix (~1 h with 10 jobs), then goldens
python python/export/train_and_export.py --n-seeds 5 --jobs 10
python python/export/make_goldens.py

# 2. Build + test the C++ engine (all tests must pass)
make -C cpp -j8 all && make -C cpp test

# 3. PC experiments
sh scripts/run_bench_pc.sh          # E1 latency
sh scripts/run_bench_recip.sh       # E1b anti-strawman: reciprocal KAN basis path
sh scripts/run_perf_pc.sh           # E2 counters (skips if perf unavailable)
sh scripts/run_dependability.sh     # E4+E5 quantization dependability
sh scripts/simulate_energy_session.sh  # H2 pipeline dry-run (synthetic meter data,
                                       # quarantined in results/energy_board_simulated/)

# 4. Aggregate everything
python python/analysis/build_summary.py   # -> results/SUMMARY.md
```

## RISC-V (StarFive VisionFive 2)

```sh
make -C cpp riscv test-riscv deploy   # cross-build, qemu-verify, tarball
# then follow scripts/board_runbook.md (target: <= 2 h manual session)
```

## Key invariants

- The C++ FP32 engine reproduces PyTorch within 1e-4 relative (golden tests);
  the quantized engine matches `python/common/quant_sim.py` within 1 LSB.
- All training recipes are verbatim ICLR (proof: `python/tests/test_parity.py`).
- Latency/energy are measured at FP32 only — the quantized path is a
  *simulated* integer pipeline (fake-quant) for dependability, not a fast kernel.
- Exported models are the **median seed of 5** by Discovery R^2 (representative,
  not cherry-picked); `trained_ok=false` manifests mark fallback exports that
  must be excluded from accuracy/dependability claims.
- Discovery R^2 everywhere uses the frozen ICLR definition: 50x50 grid over
  [-2,2]^2 (note: the ICLR paper TEXT says 100x100/[-2.5,2.5]; the code never did —
  the SBESC paper must describe the code's actual behavior).
