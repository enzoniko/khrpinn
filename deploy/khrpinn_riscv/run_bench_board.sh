#!/bin/sh
# Board-side experiment runner for the StarFive VisionFive 2 (RV64GC).
# Run FROM the untarred deploy directory (bin/, models/, goldens/ siblings):
#     sh run_bench_board.sh
# Results land in board_results/ — scp the whole directory back to the PC.
#
# Sequence: (1) golden verification of every model ON the board (catches
# silent miscompiles/endianness/libm surprises), (2) FP32 latency suite,
# (3) optional perf counters, (4) prints energy-loop instructions.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/bin"
OUT="$HERE/board_results"
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-10}
ITERS=${ITERS:-20000}   # U74 is ~20-50x slower than a desktop core

mkdir -p "$OUT"
echo "khrpinn board run on: $(uname -srm)"
echo "cpu: $(grep -m1 -E 'model name|uarch' /proc/cpuinfo || true)"

# Best-effort: pin frequency (needs root; ignore failures)
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -w "$gov" ] && echo performance > "$gov" 2>/dev/null || true
done

echo "== 1/3 golden verification (pre-flight) =="
FAILED=0
for model in "$HERE"/models/*.khrp; do
    slug=$(basename "$model" .khrp)
    if ! "$BIN/khrpinn_verify" "$model" "$HERE/goldens/$slug" > "$OUT/verify_$slug.log" 2>&1; then
        echo "  VERIFY FAILED: $slug (see board_results/verify_$slug.log)"
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "verification failures — DO NOT trust benchmark numbers; investigate first" >&2
    exit 1
fi
echo "  all models verified OK"

echo "== 2/3 FP32 latency suite =="
CSV="$OUT/latency_board.csv"
"$BIN/khrpinn_bench" --header > "$CSV"
for round in $(seq 1 "$ROUNDS"); do
    echo "  round $round/$ROUNDS"
    for model in "$HERE"/models/*.khrp; do
        for mode in residual step; do
            "$BIN/khrpinn_bench" "$model" --mode "$mode" --precision f32 \
                --reps "$REPS" --iters "$ITERS" >> "$CSV"
        done
    done
done
echo "  wrote $CSV"

echo "== 3/3 perf counters (optional) =="
if command -v perf >/dev/null 2>&1; then
    PCSV="$OUT/perf_board.csv"
    echo "host,model,precision,iters,instructions,cycles,branches,branch_misses,cache_references,cache_misses,task_clock_ms" > "$PCSV"
    for model in "$HERE"/models/*.khrp; do
        slug=$(basename "$model" .khrp)
        raw=$(perf stat -x, -e instructions,cycles,branches,branch-misses,cache-references,cache-misses,task-clock \
              "$BIN/khrpinn_bench" "$model" --mode residual --precision f32 \
              --reps 1 --iters 200000 --warmup 5000 2>&1 >/dev/null) || continue
        get() { echo "$raw" | awk -F, -v ev="$1" '$3==ev {print $1; exit}'; }
        echo "$(uname -m),$slug,f32,200000,$(get instructions),$(get cycles),$(get branches),$(get branch-misses),$(get cache-references),$(get cache-misses),$(get task-clock)" >> "$PCSV"
    done
    echo "  wrote $PCSV"
else
    echo "  perf not present on this image — skipped"
fi

{
    echo "board: $(uname -srm)"
    echo "date: $(date -Iseconds)"
    echo "rounds: $ROUNDS reps: $REPS iters: $ITERS"
    cat /proc/cpuinfo
} > "$OUT/board_info.txt"

echo ""
echo "BENCH-BOARD-OK — next: energy measurement (see energy_protocol.md)."
echo "Per-model energy loop example:"
echo "  $BIN/khrpinn_bench models/duffing_mlp_tiny.khrp --mode step --loop-seconds 120 >> $OUT/energy_runs.csv"
echo "Record the USB power meter average W during each run into energy_log.csv."
