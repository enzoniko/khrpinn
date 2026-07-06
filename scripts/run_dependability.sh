#!/bin/sh
# E4 + E5: quantization dependability experiments (PC; bit-identical on RV64).
#
# E4 (closed-loop rollouts, 2000 steps = 100 s):
#   - summaries: all 20 ICs x {f32,i16,i8,i6,i4} x {full, residual_only}
#                + attribution set {input_only, weights_only, acts_only} @ i8
#                -> results/dependability/summaries.csv
#   - curves:    IC 0 only, per-step error rows for plotting
#                -> results/dependability/curves/<slug>_<prec>_<qmode>.csv
#   - truth ref: f32 run against the RK4-integrated true dynamics (model quality)
#
# E5 (quantized discovery surfaces, 50x50 grid):
#   -> results/dependability/discovery/<slug>_<prec>_<qmode>.csv
#
# Only models with trained_ok=true are meaningful for E4/E5; the aggregator
# filters on the manifest flag, this script runs everything (cheap).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRAJ="$REPO/cpp/build/khrpinn_traj"
DISC="$REPO/cpp/build/khrpinn_discovery"
ICS="$REPO/data/goldens/ics.csv"
OUT="$REPO/results/dependability"
STEPS=${STEPS:-2000}

[ -x "$TRAJ" ] || { echo "build first: make -C cpp all" >&2; exit 1; }
[ -f "$ICS" ] || { echo "missing $ICS — run make_goldens.py" >&2; exit 1; }
mkdir -p "$OUT/curves" "$OUT/discovery"

SUM="$OUT/summaries.csv"
echo "model,ic,precision,qmode,ref,steps_requested,steps_run,mse,max_err,divergence_step,diverged,finite" > "$SUM"

run_summary() {  # model_path slug ic x0 v0 precision qmode ref
    "$TRAJ" "$1" --ic "$3" --x0 "$4" --v0 "$5" --steps "$STEPS" \
        --precision "$6" --qmode "$7" --ref "$8" --summary | tail -n 1 | \
        sed "s/^/$2,/" >> "$SUM"
}

for model in "$REPO"/models/*.khrp; do
    slug=$(basename "$model" .khrp)
    echo "== $slug =="
    # Curves for IC 0 (plots): all precisions, qmode=full, ref=f32
    x0=$(awk -F, 'NR==2 {print $2}' "$ICS")
    v0=$(awk -F, 'NR==2 {print $3}' "$ICS")
    for prec in f32 i16 i8 i6 i4; do
        "$TRAJ" "$model" --ic 0 --x0 "$x0" --v0 "$v0" --steps "$STEPS" \
            --precision "$prec" --qmode full --ref f32 \
            > "$OUT/curves/${slug}_${prec}_full.csv"
    done

    # Summaries over all ICs
    tail -n +2 "$ICS" | while IFS=, read -r ic x0 v0; do
        # truth reference at f32 (model quality baseline)
        run_summary "$model" "$slug" "$ic" "$x0" "$v0" f32 full truth
        for prec in f32 i16 i8 i6 i4; do
            run_summary "$model" "$slug" "$ic" "$x0" "$v0" "$prec" full f32
            run_summary "$model" "$slug" "$ic" "$x0" "$v0" "$prec" residual_only f32
        done
        for qmode in input_only weights_only acts_only; do
            run_summary "$model" "$slug" "$ic" "$x0" "$v0" i8 "$qmode" f32
        done
    done

    # E5: discovery surfaces
    for prec in f32 i16 i8; do
        "$DISC" "$model" --precision "$prec" --qmode full \
            > "$OUT/discovery/${slug}_${prec}_full.csv"
    done
done

cat > "$OUT/meta.json" <<EOF
{
  "experiment": "dependability",
  "date": "$(date -Iseconds)",
  "host": "$(uname -srm)",
  "steps": $STEPS,
  "divergence_threshold_normalized_l2": 0.1,
  "ics": "data/goldens/ics.csv (20 ICs, uniform [-1.5,1.5]^2, seed 7)",
  "notes": "err_norm is L2 in the normalized state domain; ref=f32 isolates quantization effects, ref=truth measures total model error."
}
EOF
echo "DEPENDABILITY-OK"
