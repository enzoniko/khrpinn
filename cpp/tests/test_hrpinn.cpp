// HRPINN cell + reference RK4 integrator tests.
#include <cmath>

#include "golden_io.hpp"
#include "khrpinn/hrpinn.hpp"
#include "test.hpp"

using namespace khrpinn;

// A model whose residual branch outputs exactly 0 and whose normalizer is the
// identity map (norm_min=-1, norm_range=2  =>  normalized == physical).
static Model zero_residual_model(System sys) {
    Model m;
    m.arch = Arch::MLP;
    m.system = sys;
    m.n_layers = 1;
    m.dt = 0.05f;
    m.norm_min[0] = m.norm_min[1] = -1.0f;
    m.norm_range[0] = m.norm_range[1] = 2.0f;
    MlpLayer l;
    l.in = 2; l.out = 1; l.has_relu = false;
    l.W = {0.0f, 0.0f};
    l.b = {0.0f};
    m.mlp = {l};
    return m;
}

TEST(hrpinn_step_is_euler_on_known_physics) {
    for (System sys : {System::Duffing, System::VdP}) {
        Model m = zero_residual_model(sys);
        Workspace ws(m);
        float s[2] = {0.4f, -0.3f};  // == physical (identity normalizer)
        float out[2];
        hrpinn_step(m, ws, s, out);
        float expect_x = 0.4f + 0.05f * (-0.3f);
        float expect_v = -0.3f + 0.05f * known_accel(sys, 0.4f, -0.3f);
        CHECK_CLOSE(out[0], expect_x, 1e-6);
        CHECK_CLOSE(out[1], expect_v, 1e-6);
    }
}

TEST(normalize_denormalize_roundtrip) {
    Model m = zero_residual_model(System::Duffing);
    m.norm_min[0] = -2.13f; m.norm_min[1] = -1.7f;
    m.norm_range[0] = 4.2f; m.norm_range[1] = 3.9f;
    float s[2];
    normalize_state(m, 0.77f, -0.31f, s);
    float xp, vp;
    denormalize_state(m, s, xp, vp);
    CHECK_CLOSE(xp, 0.77f, 1e-6);
    CHECK_CLOSE(vp, -0.31f, 1e-6);
}

TEST(rk4_matches_scipy_truth_duffing) {
    auto rows = gio::read_truth_csv(gio::goldens_dir() + "/truth_duffing.csv");
    CHECK(rows.size() > 400);
    double x = 1.0, v = 0.0;
    double max_err = 0.0;
    // Compare over the first 20 s (400 steps of dt=0.05, 10 RK4 substeps each).
    for (size_t i = 1; i <= 400 && i < rows.size(); ++i) {
        rk4_advance(System::Duffing, 0.05, 10, x, v);
        max_err = std::max(max_err, std::fabs(x - rows[i][1]));
        max_err = std::max(max_err, std::fabs(v - rows[i][2]));
    }
    CHECK_MSG(max_err < 1e-6, "duffing rk4 max_err=" + std::to_string(max_err));
}

TEST(rk4_matches_scipy_truth_vdp) {
    auto rows = gio::read_truth_csv(gio::goldens_dir() + "/truth_vdp.csv");
    CHECK(rows.size() > 400);
    double x = 1.0, v = 0.0;
    double max_err = 0.0;
    for (size_t i = 1; i <= 400 && i < rows.size(); ++i) {
        rk4_advance(System::VdP, 0.05, 10, x, v);
        max_err = std::max(max_err, std::fabs(x - rows[i][1]));
        max_err = std::max(max_err, std::fabs(v - rows[i][2]));
    }
    CHECK_MSG(max_err < 1e-5, "vdp rk4 max_err=" + std::to_string(max_err));
}

TEST(true_residual_definitions) {
    // Residual + known must equal the full dynamics acceleration.
    for (System sys : {System::Duffing, System::VdP}) {
        for (float x = -2.0f; x <= 2.0f; x += 0.37f) {
            for (float v = -2.0f; v <= 2.0f; v += 0.41f) {
                double dx, dv;
                full_dynamics(sys, x, v, dx, dv);
                float sum = known_accel(sys, x, v) + true_residual(sys, x, v);
                CHECK_CLOSE(sum, dv, 1e-5);
            }
        }
    }
}
