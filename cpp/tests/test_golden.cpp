// Golden-vector tests: the C++ engine must reproduce PyTorch (FP32, tolerance)
// and quant_sim.py (quantized, <= ~1 LSB) on every exported model.
#include <cmath>
#include <string>

#include "golden_io.hpp"
#include "khrpinn/hrpinn.hpp"
#include "khrpinn/mlp.hpp"
#include "khrpinn/quant.hpp"
#include "test.hpp"

using namespace khrpinn;

static double file_scale(const std::vector<gio::IoRec>& recs) {
    double s = 0;
    for (const auto& r : recs) s = std::max(s, (double)std::fabs(r.out));
    return s;
}

TEST(golden_residual_fp32) {
    auto slugs = gio::list_model_slugs();
    CHECK(!slugs.empty());
    for (const auto& slug : slugs) {
        Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
        Workspace ws(m);
        auto recs = gio::read_bin<gio::IoRec>(gio::goldens_dir() + "/" + slug + "/residual_io.bin");
        CHECK(recs.size() >= 128);
        double max_diff = 0, scale = file_scale(recs);
        for (const auto& r : recs) {
            float x[2] = {r.x, r.v};
            max_diff = std::max(max_diff, (double)std::fabs(residual_forward(m, ws, x) - r.out));
        }
        double tol = std::max(1e-6, 1e-4 * scale);
        CHECK_MSG(max_diff <= tol,
                  slug + ": fp32 residual max_diff=" + std::to_string(max_diff) +
                      " tol=" + std::to_string(tol));
    }
}

TEST(golden_hrpinn_step_fp32) {
    for (const auto& slug : gio::list_model_slugs()) {
        Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
        Workspace ws(m);
        auto recs = gio::read_bin<gio::StepRec>(gio::goldens_dir() + "/" + slug + "/step_io.bin");
        CHECK(recs.size() >= 32);
        double max_diff = 0;
        for (const auto& r : recs) {
            float s[2] = {r.s0x, r.s0v}, out[2];
            hrpinn_step(m, ws, s, out);
            max_diff = std::max(max_diff, (double)std::fabs(out[0] - r.s1x));
            max_diff = std::max(max_diff, (double)std::fabs(out[1] - r.s1v));
        }
        CHECK_MSG(max_diff <= 3e-5,
                  slug + ": step max_diff=" + std::to_string(max_diff));
    }
}

static void check_quant_golden(const std::string& slug, int bits) {
    Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
    QuantConfig cfg;  // branch-only full quantization, matches quant_sim goldens
    cfg.bits = bits;
    cfg.q_input = cfg.q_weights = cfg.q_acts = true;
    cfg.q_state_loop = false;
    QuantModel qm(m, cfg);
    Workspace ws(m);
    auto recs = gio::read_bin<gio::IoRec>(
        gio::goldens_dir() + "/" + slug + "/quant_io_b" + std::to_string(bits) + ".bin");
    // Tolerance: a few output LSBs. numpy (BLAS) and the scalar C++ loops
    // accumulate in different orders; a value landing within ~1e-6 of a
    // rounding boundary flips by 1 LSB, and an inner-layer flip propagates
    // through the next layer's weights. INT8's coarse grid absorbs this
    // (1.5 LSB suffices); INT16's fine grid needs ~4 LSB for 2-3 layer models.
    Range last_act = m.arch == Arch::MLP ? m.mlp.back().act : m.kan.back().act;
    double lsb = affine_qparams(last_act, bits).scale;
    double lsb_mult = bits >= 16 ? 4.0 : 1.5;
    double max_diff = 0;
    for (const auto& r : recs) {
        float x[2] = {r.x, r.v};
        max_diff = std::max(max_diff, (double)std::fabs(residual_forward_quant(qm, ws, x) - r.out));
    }
    CHECK_MSG(max_diff <= lsb_mult * lsb,
              slug + " b" + std::to_string(bits) + ": quant max_diff=" +
                  std::to_string(max_diff) + " lsb=" + std::to_string(lsb));
}

TEST(golden_quant_int8) {
    for (const auto& slug : gio::list_model_slugs()) check_quant_golden(slug, 8);
}

TEST(golden_quant_int16) {
    for (const auto& slug : gio::list_model_slugs()) check_quant_golden(slug, 16);
}

TEST(golden_trajectory_fp32) {
    int tested = 0, skipped = 0;
    for (const auto& slug : gio::list_model_slugs()) {
        Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
        Workspace ws(m);
        auto trajs = gio::read_traj_csv(gio::goldens_dir() + "/" + slug + "/traj_f32.csv");
        CHECK(!trajs.empty());
        for (const auto& tr : trajs) {
            // Only compare trajectories the Python reference itself kept
            // finite and bounded (unstable models diverge chaotically and are
            // not comparable step-by-step).
            bool valid = true;
            for (const auto& s : tr)
                if (!std::isfinite(s.first) || !std::isfinite(s.second) ||
                    std::fabs(s.first) > 3.0f || std::fabs(s.second) > 3.0f)
                    valid = false;
            if (!valid) { ++skipped; continue; }
            float s[2] = {tr[0].first, tr[0].second};
            double max_diff = 0;
            for (size_t t = 1; t < tr.size(); ++t) {
                hrpinn_step(m, ws, s, s);
                max_diff = std::max(max_diff, (double)std::fabs(s[0] - tr[t].first));
                max_diff = std::max(max_diff, (double)std::fabs(s[1] - tr[t].second));
            }
            CHECK_MSG(max_diff <= 5e-3,
                      slug + ": traj max_diff=" + std::to_string(max_diff));
            ++tested;
        }
    }
    std::printf("  (trajectories compared: %d, skipped-unstable: %d)\n", tested, skipped);
    CHECK(tested > 0);
}

TEST(quant_int16_close_to_fp32_for_mlp_on_manifold) {
    // Sanity: ON THE OPERATING MANIFOLD (states visited by the model's own
    // rollout — where the PTQ activation ranges were calibrated), the INT16
    // MLP branch must sit within a few dozen output LSBs of FP32. Off-manifold
    // inputs legitimately saturate the calibrated activation quantizers, so
    // random [-1,1]^2 states are NOT a valid closeness probe (that clipping is
    // part of what experiment E4 measures).
    for (const auto& slug : gio::list_model_slugs()) {
        if (slug.find("mlp") == std::string::npos) continue;
        Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
        QuantModel qm(m, qmode_config("residual_only", 16));
        Workspace ws(m);
        double lsb = affine_qparams(m.mlp.back().act, 16).scale;
        std::vector<double> diffs;
        float s[2];
        normalize_state(m, 1.0f, 0.0f, s);  // training-like IC
        for (int t = 0; t < 400; ++t) {
            if (std::fabs(s[0]) <= 1.0f && std::fabs(s[1]) <= 1.0f)
                diffs.push_back(std::fabs(residual_forward_quant(qm, ws, s) -
                                          residual_forward(m, ws, s)));
            hrpinn_step(m, ws, s, s);
        }
        CHECK(diffs.size() > 100);
        // p95 bound: rare activation-range clipping (states slightly outside
        // the calibration envelope) may spike individual diffs — that clipping
        // is measured properly by experiment E4, not asserted away here.
        std::sort(diffs.begin(), diffs.end());
        double p95 = diffs[(size_t)(0.95 * (diffs.size() - 1))];
        CHECK_MSG(p95 <= 30 * lsb,
                  slug + ": on-manifold int16-vs-fp32 p95=" + std::to_string(p95) +
                      " (30 lsb=" + std::to_string(30 * lsb) + ")");
    }
}
