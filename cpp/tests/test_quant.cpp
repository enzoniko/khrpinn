// Quantization primitive tests (cross-language spec pinned by quant goldens).
#include <cmath>

#include "khrpinn/quant.hpp"
#include "test.hpp"

using namespace khrpinn;

TEST(round_half_away_from_zero) {
    CHECK_CLOSE(round_half_away(0.5f), 1.0f, 0.0);
    CHECK_CLOSE(round_half_away(-0.5f), -1.0f, 0.0);
    CHECK_CLOSE(round_half_away(1.5f), 2.0f, 0.0);
    CHECK_CLOSE(round_half_away(2.5f), 3.0f, 0.0);   // NOT banker's rounding
    CHECK_CLOSE(round_half_away(-2.5f), -3.0f, 0.0);
    CHECK_CLOSE(round_half_away(0.49f), 0.0f, 0.0);
    CHECK_CLOSE(round_half_away(0.0f), 0.0f, 0.0);
}

TEST(affine_quant_zero_is_exact) {
    for (int bits : {4, 6, 8, 16}) {
        QParams q = affine_qparams({-1.0f, 1.0f}, bits);
        CHECK_CLOSE(fake_quant(0.0f, q), 0.0f, 0.0);
        // asymmetric range still represents zero exactly
        QParams q2 = affine_qparams({-0.278f, 0.9313f}, bits);
        CHECK_CLOSE(fake_quant(0.0f, q2), 0.0f, 0.0);
        // positive-only range widened to include zero
        QParams q3 = affine_qparams({0.2f, 1.7f}, bits);
        CHECK_CLOSE(fake_quant(0.0f, q3), 0.0f, 0.0);
    }
}

TEST(affine_quant_error_bound_and_idempotence) {
    QParams q = affine_qparams({-1.0f, 1.0f}, 8);
    for (float x = -1.0f; x <= 1.0f; x += 0.00137f) {
        float fx = fake_quant(x, q);
        CHECK(std::fabs(fx - x) <= q.scale * 0.5f + 1e-7f);
        CHECK_CLOSE(fake_quant(fx, q), fx, 0.0);  // idempotent
    }
}

TEST(affine_quant_clamps_out_of_range) {
    QParams q = affine_qparams({-1.0f, 1.0f}, 8);
    float top = fake_quant(10.0f, q);
    float bot = fake_quant(-10.0f, q);
    CHECK(top <= 1.0f + q.scale);
    CHECK(bot >= -1.0f - q.scale);
    CHECK_CLOSE(fake_quant(5.0f, q), top, 0.0);   // saturated
    CHECK_CLOSE(fake_quant(-5.0f, q), bot, 0.0);
}

TEST(affine_quant_monotone) {
    QParams q = affine_qparams({-0.7f, 1.3f}, 6);
    float prev = fake_quant(-1.0f, q);
    for (float x = -1.0f; x <= 1.5f; x += 0.003f) {
        float fx = fake_quant(x, q);
        CHECK(fx >= prev - 1e-9f);
        prev = fx;
    }
}

TEST(symmetric_weight_quant) {
    QParams q = symmetric_qparams(1.0f, 8);
    CHECK_CLOSE(q.zp, 0.0f, 0.0);
    // extremes map to +-127 * scale ~= +-1
    float t = round_half_away(1.0f / q.scale) * q.scale;
    CHECK_CLOSE(t, 1.0f, 1e-6);
    // int16 has much finer resolution
    QParams q16 = symmetric_qparams(1.0f, 16);
    CHECK(q16.scale < q.scale / 200.0f);
}

TEST(quant_config_modes) {
    QuantConfig full = qmode_config("full", 8);
    CHECK(full.q_input && full.q_weights && full.q_acts && full.q_state_loop);
    QuantConfig ro = qmode_config("residual_only", 8);
    CHECK(ro.q_input && ro.q_weights && ro.q_acts && !ro.q_state_loop);
    QuantConfig io = qmode_config("input_only", 8);
    CHECK(io.q_input && !io.q_weights && !io.q_acts && !io.q_state_loop);
    QuantConfig wo = qmode_config("weights_only", 8);
    CHECK(!wo.q_input && wo.q_weights && !wo.q_acts && !wo.q_state_loop);
    QuantConfig ao = qmode_config("acts_only", 8);
    CHECK(!ao.q_input && !ao.q_weights && ao.q_acts && !ao.q_state_loop);
    CHECK_THROWS(qmode_config("bogus", 8));
    CHECK_THROWS(qmode_config("full", 7));
}
