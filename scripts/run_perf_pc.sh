#!/bin/sh
# E2 (PC): hardware counters per model (instructions, branches, cache misses)
# around a fixed-iteration FP32 residual benchmark. Degrades gracefully when
# perf or its counters are unavailable (typical under WSL2) — the board run is
# the authoritative source for the paper's microarchitectural narrative.
# Output: results/perf_pc/perf.csv
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO/cpp/build/khrpinn_bench"
OUT="$REPO/results/perf_pc"
ITERS=${ITERS:-2000000}

if ! command -v perf >/dev/null 2>&1; then
    echo "perf not available — skipping (board run covers counters)" >&2
    exit 0
fi
mkdir -p "$OUT"
CSV="$OUT/perf.csv"
echo "host,model,precision,iters,instructions,cycles,branches,branch_misses,cache_references,cache_misses,task_clock_ms" > "$CSV"

EVENTS="instructions,cycles,branches,branch-misses,cache-references,cache-misses,task-clock"
for model in "$REPO"/models/*.khrp; do
    slug=$(basename "$model" .khrp)
    raw=$(perf stat -x, -e "$EVENTS" \
          "$BENCH" "$model" --mode residual --precision f32 \
          --reps 1 --iters "$ITERS" --warmup 10000 2>&1 >/dev/null) || {
        echo "perf failed for $slug — counters likely unsupported here" >&2
        continue
    }
    get() { echo "$raw" | awk -F, -v ev="$1" '$3==ev {print $1; exit}'; }
    echo "$(uname -m),$slug,f32,$ITERS,$(get instructions),$(get cycles),$(get branches),$(get branch-misses),$(get cache-references),$(get cache-misses),$(get task-clock)" >> "$CSV"
    echo "  $slug done"
done
echo "Wrote $CSV"
echo "PERF-PC-OK"
