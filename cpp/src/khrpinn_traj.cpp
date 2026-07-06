// khrpinn_traj — closed-loop HRPINN rollout at a chosen precision, with a
// per-step error column against a reference (FP32 model rollout or the true
// dynamics integrated with double-precision RK4). Dependability experiment E4.
//
// Usage:
//   khrpinn_traj model.khrp --x0 <phys> --v0 <phys> [--steps 2000]
//     [--precision f32|i16|i8|i6|i4] [--qmode full|residual_only|...]
//     [--ref f32|truth|none] [--ic <index>] [--summary]
//
// Default CSV (per step):
//   ic,step,t,x,v,x_ref,v_ref,err_norm
// x/v are PHYSICAL coordinates; err_norm is the L2 error in the NORMALIZED
// state domain (the controller's own units). Rollout stops early if the state
// becomes non-finite.
//
// --summary emits a single aggregated row instead (used for the 20-IC sweep):
//   ic,precision,qmode,ref,steps_requested,steps_run,mse,max_err,
//   divergence_step,diverged,finite
// where divergence_step is the first step with err_norm > 0.1 (-1 if never)
// and diverged = err exceeded threshold OR the state went non-finite.
#include <cmath>
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "khrpinn/hrpinn.hpp"
#include "khrpinn/quant.hpp"
#include "khrpinn/systems.hpp"

using namespace khrpinn;

static constexpr float DIVERGENCE_THRESHOLD = 0.1f;  // normalized L2

int main(int argc, char** argv) {
    cli::Args args{argc, argv};
    const char* mpath = args.positional(0);
    if (!mpath) {
        std::fprintf(stderr,
                     "usage: khrpinn_traj model.khrp --x0 X --v0 V [--steps N] "
                     "[--precision ...] [--qmode ...] [--ref f32|truth|none] "
                     "[--ic idx] [--summary]\n");
        return 2;
    }
    double x0 = args.getd("--x0", 1.0);
    double v0 = args.getd("--v0", 0.0);
    long steps = args.getl("--steps", 2000);
    long ic = args.getl("--ic", 0);
    int bits = cli::parse_precision(args.get("--precision", "f32"));
    std::string qmode = args.get("--qmode", "full");
    std::string ref = args.get("--ref", "f32");
    bool summary = args.has("--summary");

    Model m = load_model(mpath);
    Workspace ws(m), ws_ref(m);
    QuantModel* qm = bits != 0 ? new QuantModel(m, qmode_config(qmode, bits)) : nullptr;

    float s[2], s_ref[2];
    normalize_state(m, (float)x0, (float)v0, s);
    s_ref[0] = s[0];
    s_ref[1] = s[1];
    // In "full" mode the initial state is quantized too (sensor path).
    if (qm && qm->cfg.q_state_loop) {
        s[0] = fake_quant(s[0], qm->in_q);
        s[1] = fake_quant(s[1], qm->in_q);
    }
    double xt = x0, vt = v0;  // truth integrator state (double, physical)

    if (!summary) std::printf("ic,step,t,x,v,x_ref,v_ref,err_norm\n");

    double sum_sq = 0.0, max_err = 0.0;
    long divergence_step = -1, steps_run = 0;
    bool finite = true;

    for (long t = 0; t <= steps; ++t) {
        float xr = 0, vr = 0, err = 0;
        if (ref == "f32") {
            denormalize_state(m, s_ref, xr, vr);
            float dx = s[0] - s_ref[0], dv = s[1] - s_ref[1];
            err = std::sqrt(dx * dx + dv * dv);
        } else if (ref == "truth") {
            xr = (float)xt;
            vr = (float)vt;
            float tr[2];
            normalize_state(m, xr, vr, tr);
            float dx = s[0] - tr[0], dv = s[1] - tr[1];
            err = std::sqrt(dx * dx + dv * dv);
        }
        if (!summary) {
            float xp, vp;
            denormalize_state(m, s, xp, vp);
            std::printf("%ld,%ld,%.4f,%.7g,%.7g,%.7g,%.7g,%.7g\n", ic, t,
                        (double)t * m.dt, xp, vp, xr, vr, err);
        }
        if (std::isfinite(err)) {
            sum_sq += (double)err * err;
            if (err > max_err) max_err = err;
            if (divergence_step < 0 && err > DIVERGENCE_THRESHOLD) divergence_step = t;
        } else if (divergence_step < 0) {
            divergence_step = t;
        }
        steps_run = t;
        if (!std::isfinite(s[0]) || !std::isfinite(s[1])) { finite = false; break; }
        if (t == steps) break;
        if (qm) hrpinn_step_quant(*qm, ws, s, s);
        else hrpinn_step(m, ws, s, s);
        if (ref == "f32") hrpinn_step(m, ws_ref, s_ref, s_ref);
        else if (ref == "truth") rk4_advance(m.system, m.dt, 10, xt, vt);
    }

    if (summary) {
        bool diverged = (divergence_step >= 0) || !finite;
        std::printf("ic,precision,qmode,ref,steps_requested,steps_run,mse,max_err,"
                    "divergence_step,diverged,finite\n");
        std::printf("%ld,%s,%s,%s,%ld,%ld,%.9g,%.9g,%ld,%d,%d\n", ic,
                    cli::precision_name(bits), qmode.c_str(), ref.c_str(), steps,
                    steps_run, sum_sq / (double)(steps_run + 1), max_err,
                    divergence_step, diverged ? 1 : 0, finite ? 1 : 0);
    }
    delete qm;
    return 0;
}
