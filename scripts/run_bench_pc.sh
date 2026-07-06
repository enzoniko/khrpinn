#!/bin/sh
# E1 (PC): FP32 latency of all exported models, residual-branch and full-step.
# Interleaved rounds (3 x 10 reps) so slow drift affects all models equally.
# Output: results/latency_pc/latency.csv + meta.json
#
# NOTE: latency is measured at FP32 only. The quantized engine simulates intN
# semantics with float arithmetic (fake-quant), so its latency is NOT the
# latency of a true integer pipeline and would be misleading to report.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO/cpp/build/khrpinn_bench"
OUT="$REPO/results/latency_pc"
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-10}
ITERS=${ITERS:-100000}

[ -x "$BENCH" ] || { echo "build first: make -C cpp all" >&2; exit 1; }
mkdir -p "$OUT"

PIN=""
if command -v taskset >/dev/null 2>&1; then PIN="taskset -c 0"; fi

CSV="$OUT/latency.csv"
$PIN "$BENCH" --header > "$CSV"

for round in $(seq 1 "$ROUNDS"); do
    echo "== round $round/$ROUNDS =="
    for model in "$REPO"/models/*.khrp; do
        for mode in residual step; do
            $PIN "$BENCH" "$model" --mode "$mode" --precision f32 \
                --reps "$REPS" --iters "$ITERS" >> "$CSV"
        done
        echo "  $(basename "$model" .khrp) done"
    done
done

cat > "$OUT/meta.json" <<EOF
{
  "experiment": "latency_pc",
  "date": "$(date -Iseconds)",
  "host": "$(uname -srm)",
  "cpu": "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //' | sed 's/"//g')",
  "compiler_flags": "-std=c++17 -O2 -ffp-contract=off (see cpp/Makefile)",
  "rounds": $ROUNDS, "reps_per_round": $REPS, "iters_per_rep": $ITERS,
  "pinning": "$PIN",
  "notes": "WSL2/PC numbers contextualize the board results; the RISC-V board run is authoritative for the paper."
}
EOF
echo "Wrote $CSV"
echo "BENCH-PC-OK"
