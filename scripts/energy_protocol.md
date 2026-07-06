# Energy measurement protocol (E3) — VisionFive 2 + inline USB power meter

**Instrument:** an inline USB-C power meter between the PSU and the board's
USB-C power input, reading average V·A (W). Any meter with a stable averaged W
readout works; if yours integrates mWh, the mWh method below is more accurate.

**Model of measurement:** energy per inference step is computed differentially:

```
uJ_per_step = (P_load_W - P_idle_W) / steps_per_sec * 1e6
```

`steps_per_sec` comes from the benchmark's energy-loop CSV row; `P_idle` /
`P_load` come from the meter. This subtracts the board's static power (SoC
idle, DRAM refresh, PHYs) and attributes only the incremental compute power to
the model — the quantity that differs between KAN and MLP.

## Preconditions (do once per session)

1. Fresh boot, let the board settle 2 minutes. No ethernet traffic during
   measurements (use a serial console, or connect via SSH but stay idle).
2. Pin the CPU governor: `run_bench_board.sh` attempts this; verify with
   `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` → `performance`.
3. Run the latency suite first (`sh run_bench_board.sh`) — it also verifies
   all models against goldens, which you want BEFORE spending meter time.

## Per-session steps

1. **Idle baseline (2x):** with nothing running, watch the meter for 60 s and
   record the average W as `p_idle_w`. Repeat at the end of the session; if the
   two differ by more than ~3%, redo the session (thermal or background drift).
2. **Per model** (do all 20, or at minimum the 16 BPTT models):

   ```sh
   bin/khrpinn_bench models/<slug>.khrp --mode step --loop-seconds 120 >> board_results/energy_runs.csv
   ```

   While it runs, record the meter's average W (or note mWh at start/end).
   Immediately fill one row of `energy_log.csv` (template provided):
   `board,model,precision,p_idle_w,p_load_w,meter,notes`.
3. **Interleave a repeat:** re-measure the first model at the end; the two
   `p_load_w` readings should agree within ~3% (thermal sanity check).

## Notes

- 120 s per model keeps meter averaging error small and the whole session
  ≈ 45 min for 20 models. `--loop-seconds` can be raised if your meter
  averages slowly.
- Energy is measured at FP32 (the H2 comparison). The quantized engine is a
  fake-quant simulation, so its power draw is not representative of a true
  integer pipeline — do not measure it.
- `aggregate_energy.py` joins `energy_log.csv` with `energy_runs.csv` (per-model
  steps_per_sec) and computes `uJ_per_step` — you never compute it by hand.
- If your meter reports accumulated mWh: record `mwh_start`/`mwh_end` in the
  notes column instead; `uJ_per_step = ((mwh_end-mwh_start)*3600/elapsed_s - P_idle) / steps_per_sec * 1e6`.
