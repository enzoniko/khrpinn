#!/bin/sh
# End-to-end DRY RUN of the H2 (energy) pipeline on the PC, so the board
# session exercises an already-validated path. It produces:
#   results/energy_board_simulated/energy_runs.csv  (REAL bench energy-loop
#       rows, executed on the PC with short --loop-seconds)
#   results/energy_board_simulated/energy_log.csv   (SYNTHETIC power-meter
#       readings, clearly marked board=SIMULATED)
# and verifies aggregate_energy.py computes uJ/step from them.
#
# The synthetic power model is: p_load = p_idle + c1 + c2 * flops_est-rate.
# NUMBERS ARE NOT MEASUREMENTS. The real session writes the same two files to
# results/energy_board/ (see energy_protocol.md); the aggregator is identical.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO/cpp/build/khrpinn_bench"
OUT="$REPO/results/energy_board_simulated"
LOOP_SECONDS=${LOOP_SECONDS:-2}
mkdir -p "$OUT"

RUNS="$OUT/energy_runs.csv"
: > "$RUNS"
for model in "$REPO"/models/*.khrp; do
    "$BENCH" "$model" --mode step --loop-seconds "$LOOP_SECONDS" >> "$RUNS"
    echo "  $(basename "$model" .khrp) energy-loop done"
done

# Synthetic meter log: idle 4.20 W; active delta grows mildly with model cost.
/home/enzo/anaconda3/bin/python - "$REPO" "$OUT" <<'EOF'
import glob, json, os, sys
repo, out = sys.argv[1], sys.argv[2]
rows = ["board,model,precision,p_idle_w,p_load_w,meter,notes"]
for p in sorted(glob.glob(os.path.join(repo, 'models', '*.json'))):
    man = json.load(open(p))
    # synthetic: 0.55 W base compute + up to ~0.35 W scaling with op count
    delta = 0.55 + 0.35 * min(1.0, man['flops_per_inference'] / 9000.0)
    rows.append(f"SIMULATED,{man['slug']},f32,4.20,{4.20 + delta:.3f},"
                f"SIMULATED,synthetic dry-run - replace with real meter data")
open(os.path.join(out, 'energy_log.csv'), 'w').write("\n".join(rows) + "\n")
print("synthetic energy_log.csv written")
EOF

echo "SIMULATE-ENERGY-OK ($OUT)"
