# VisionFive 2 board session runbook (manual, target ≤ 2 hours)

Everything below is prepared and tested on PC + qemu before the session; the
board session is measurement-only.

## 0. Prepare on the PC (done once, before the session)

```sh
cd ~/ICLR2026/khrpinn
make -C cpp riscv          # static RV64GC binaries
make -C cpp test-riscv     # full test suite under qemu — must be green
make -C cpp deploy         # -> deploy/khrpinn_riscv.tar.gz (bins+models+goldens+scripts)
```

## 1. Deploy to the board (~5 min)

```sh
scp deploy/khrpinn_riscv.tar.gz user@visionfive2:
ssh user@visionfive2
tar xzf khrpinn_riscv.tar.gz && cd khrpinn_riscv
```

Binaries are fully static — no libraries needed on the minimal image.

## 2. Verification + latency suite (~20-30 min, unattended)

```sh
sh run_bench_board.sh
```

This first verifies EVERY model against its golden vectors on the board
(aborts if anything mismatches — never benchmark unverified binaries), then
runs the interleaved FP32 latency suite, then perf counters if perf exists.
Results in `board_results/`.

> The whole energy pipeline (loop binary, `energy_runs.csv` format,
> `energy_log.csv` format, μJ/step aggregation) was validated end-to-end in a
> dry run — see `scripts/simulate_energy_session.sh` and
> `results/energy_board_simulated/`. Your session only produces the same two
> CSVs with real numbers, into `results/energy_board/`.

## 3. Energy session (~45-60 min, attended — USB power meter)

Follow `energy_protocol.md` step by step:
1. idle baseline 60 s → note W;
2. per model: `bin/khrpinn_bench models/<slug>.khrp --mode step --loop-seconds 120 >> board_results/energy_runs.csv`
   while reading the meter; fill `energy_log.csv` (copy from
   `energy_log_template.csv`) as you go;
3. re-measure idle + first model at the end (drift check).

## 4. Optional on-board dependability spot-check (~5 min)

The dependability experiment is deterministic (same fake-quant float ops), so
PC results are authoritative; spot-check two runs to demonstrate bit-equality:

```sh
bin/khrpinn_traj models/duffing_kan_very_small.khrp --x0 1.0 --v0 0.0 \
    --precision i8 --qmode full --ref f32 --summary > board_results/spot_kan_i8.csv
bin/khrpinn_traj models/duffing_mlp_tiny.khrp --x0 1.0 --v0 0.0 \
    --precision i8 --qmode full --ref f32 --summary > board_results/spot_mlp_i8.csv
```

(Compare against the same rows in `results/dependability/summaries.csv` on the
PC; MLP rows must match bit-for-bit, KAN rows may differ microscopically via
libm expf but summaries should agree to ~6 significant digits.)

## 5. Bring results home

```sh
exit
scp -r user@visionfive2:khrpinn_riscv/board_results .
cp board_results/latency_board.csv  results/latency_board/latency.csv
cp board_results/perf_board.csv     results/perf_board/perf.csv   # if produced
cp board_results/energy_runs.csv board_results/energy_log.csv results/energy_board/
python python/analysis/build_summary.py   # regenerates results/SUMMARY.md
```

## Troubleshooting

- `run_bench_board.sh` verify failure → check `board_results/verify_*.log`;
  a uniform tiny FP32 drift on KAN models only usually means a libm expf
  difference (acceptable: raise tolerance note), anything on MLP models or
  quantized rows is a real bug — stop.
- Meter W jumping around → make sure ethernet is idle, HDMI unplugged, and the
  governor is `performance` (no DVFS transitions).
- If the board image lacks `perf`, skip it — instruction-count deltas are
  already covered analytically (flops_est/exps_est columns in latency.csv).
