// khrpinn_infer — debug tool: run the residual branch (and one HRPINN step)
// on a single input state.
//
// Usage: khrpinn_infer model.khrp <x_norm> <v_norm> [--precision f32|i16|i8|i6|i4]
//        [--qmode full|residual_only|input_only|weights_only|acts_only]
#include <cstdio>
#include <cstdlib>

#include "cli.hpp"
#include "khrpinn/hrpinn.hpp"

using namespace khrpinn;

int main(int argc, char** argv) {
    cli::Args args{argc, argv};
    const char* path = args.positional(0);
    const char* xs = args.positional(1);
    const char* vs = args.positional(2);
    if (!path || !xs || !vs) {
        std::fprintf(stderr,
                     "usage: khrpinn_infer model.khrp <x_norm> <v_norm> "
                     "[--precision f32|i16|i8|i6|i4] [--qmode full|...]\n");
        return 2;
    }
    int bits = cli::parse_precision(args.get("--precision", "f32"));
    std::string qmode = args.get("--qmode", "full");

    Model m = load_model(path);
    Workspace ws(m);
    float s[2] = {(float)std::atof(xs), (float)std::atof(vs)};

    long flops = 0, exps = 0;
    m.flops_estimate(flops, exps);
    std::printf("model=%s arch=%s system=%s params=%zu flops=%ld exps=%ld\n",
                cli::slug_of(path).c_str(), arch_name(m.arch), system_name(m.system),
                m.param_count(), flops, exps);

    float out[2];
    if (bits == 0) {
        float r = residual_forward(m, ws, s);
        hrpinn_step(m, ws, s, out);
        std::printf("residual(f32) = %.9g\n", r);
        std::printf("step(f32)     = [%.9g, %.9g]\n", out[0], out[1]);
    } else {
        QuantModel qm(m, qmode_config(qmode, bits));
        float r = residual_forward_quant(qm, ws, s);
        hrpinn_step_quant(qm, ws, s, out);
        std::printf("residual(%s,%s) = %.9g\n", cli::precision_name(bits), qmode.c_str(), r);
        std::printf("step(%s,%s)     = [%.9g, %.9g]\n", cli::precision_name(bits),
                    qmode.c_str(), out[0], out[1]);
    }
    return 0;
}
