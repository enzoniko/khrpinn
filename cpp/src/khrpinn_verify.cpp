// khrpinn_verify — verify one model against its golden vectors (the on-board
// pre-flight check before benchmarking; same checks as the test suite).
//
// Usage: khrpinn_verify model.khrp goldens_dir/<slug>
// Exit 0 on pass, 1 on any mismatch.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "khrpinn/hrpinn.hpp"
#include "khrpinn/quant.hpp"

// Reuse the same binary readers as the test suite.
#include "../tests/golden_io.hpp"

using namespace khrpinn;

static bool report(const char* what, double diff, double tol) {
    bool ok = diff <= tol;
    std::printf("  %-24s max_diff=%.3e tol=%.3e  %s\n", what, diff, tol,
                ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    cli::Args args{argc, argv};
    const char* mpath = args.positional(0);
    const char* gdir = args.positional(1);
    if (!mpath || !gdir) {
        std::fprintf(stderr, "usage: khrpinn_verify model.khrp goldens_dir/<slug>\n");
        return 2;
    }
    Model m = load_model(mpath);
    Workspace ws(m);
    std::string g(gdir);
    bool ok = true;
    std::printf("verify %s (%s/%s, %zu params)\n", cli::slug_of(mpath).c_str(),
                arch_name(m.arch), system_name(m.system), m.param_count());

    {   // FP32 residual
        auto recs = gio::read_bin<gio::IoRec>(g + "/residual_io.bin");
        double diff = 0, scale = 0;
        for (const auto& r : recs) {
            float x[2] = {r.x, r.v};
            diff = std::max(diff, (double)std::fabs(residual_forward(m, ws, x) - r.out));
            scale = std::max(scale, (double)std::fabs(r.out));
        }
        ok &= report("residual f32", diff, std::max(1e-6, 1e-4 * scale));
    }
    {   // FP32 step
        auto recs = gio::read_bin<gio::StepRec>(g + "/step_io.bin");
        double diff = 0;
        for (const auto& r : recs) {
            float s[2] = {r.s0x, r.s0v}, out[2];
            hrpinn_step(m, ws, s, out);
            diff = std::max(diff, (double)std::fabs(out[0] - r.s1x));
            diff = std::max(diff, (double)std::fabs(out[1] - r.s1v));
        }
        ok &= report("hrpinn step f32", diff, 3e-5);
    }
    for (int bits : {8, 16}) {  // quantized branch
        QuantConfig cfg;
        cfg.bits = bits;
        cfg.q_input = cfg.q_weights = cfg.q_acts = true;
        cfg.q_state_loop = false;
        QuantModel qm(m, cfg);
        auto recs = gio::read_bin<gio::IoRec>(g + "/quant_io_b" + std::to_string(bits) + ".bin");
        Range last_act = m.arch == Arch::MLP ? m.mlp.back().act : m.kan.back().act;
        double lsb = affine_qparams(last_act, bits).scale;
        double diff = 0;
        for (const auto& r : recs) {
            float x[2] = {r.x, r.v};
            diff = std::max(diff, (double)std::fabs(residual_forward_quant(qm, ws, x) - r.out));
        }
        char label[32];
        std::snprintf(label, sizeof(label), "residual int%d", bits);
        ok &= report(label, diff, 1.5 * lsb);
    }
    std::printf("%s\n", ok ? "VERIFY-OK" : "VERIFY-FAILED");
    return ok ? 0 : 1;
}
