// MLP forward unit tests against hand-computed values.
#include "khrpinn/mlp.hpp"
#include "test.hpp"

using namespace khrpinn;

static Model tiny_mlp() {
    Model m;
    m.arch = Arch::MLP;
    m.n_layers = 2;
    MlpLayer l0;
    l0.in = 2; l0.out = 2; l0.has_relu = true;
    l0.W = {1.0f, -2.0f,    // row 0
            0.5f, 0.25f};   // row 1
    l0.b = {0.1f, -0.1f};
    MlpLayer l1;
    l1.in = 2; l1.out = 1; l1.has_relu = false;
    l1.W = {3.0f, -1.0f};
    l1.b = {0.5f};
    m.mlp = {l0, l1};
    return m;
}

TEST(mlp_hand_computed_forward) {
    Model m = tiny_mlp();
    Workspace ws(m);
    // x = (1, 0.5): h = relu([1-1+0.1, 0.5+0.125-0.1]) = [0.1, 0.525]
    // y = 3*0.1 - 0.525 + 0.5 = 0.275
    float x[2] = {1.0f, 0.5f};
    CHECK_CLOSE(mlp_forward(m, ws, x), 0.275f, 1e-6);
    // ReLU clamps: x = (-1, 0): h = relu([-1+0.1, -0.5-0.1]) = [0, 0]
    // y = 0.5
    float x2[2] = {-1.0f, 0.0f};
    CHECK_CLOSE(mlp_forward(m, ws, x2), 0.5f, 1e-6);
}

TEST(mlp_dispatch_matches) {
    Model m = tiny_mlp();
    Workspace ws(m);
    float x[2] = {0.3f, -0.7f};
    CHECK_CLOSE(residual_forward(m, ws, x), mlp_forward(m, ws, x), 0.0);
}
