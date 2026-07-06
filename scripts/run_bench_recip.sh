#!/bin/sh
# Anti-strawman check: measure KAN residual latency with the deployment-
# optimized precomputed-reciprocal basis path and compare against the default
# division path. Output: results/latency_pc/latency_recip.csv (kept separate
# from latency.csv so the main aggregation is not polluted).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO/cpp/build/khrpinn_bench"
OUT="$REPO/results/latency_pc"
mkdir -p "$OUT"
CSV="$OUT/latency_recip.csv"
PIN=""
if command -v taskset >/dev/null 2>&1; then PIN="taskset -c 0"; fi

$PIN "$BENCH" --header > "$CSV"
for round in 1 2 3; do
    for model in "$REPO"/models/*kan*.khrp; do
        $PIN "$BENCH" "$model" --mode residual --precision f32 \
            --reps 10 --iters 100000 >> "$CSV"
        $PIN "$BENCH" "$model" --mode residual --precision f32 --recip \
            --reps 10 --iters 100000 >> "$CSV"
    done
done
echo "Wrote $CSV"
echo "BENCH-RECIP-OK"
