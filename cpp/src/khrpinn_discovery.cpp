// khrpinn_discovery — residual surface on the ICLR discovery grid (50x50 over
// physical [-2,2]^2), at a chosen precision. Experiment E5: quantized
// Discovery R^2 (the R^2 itself is computed by python/analysis, using the
// exact frozen-metric formula).
//
// Usage: khrpinn_discovery model.khrp [--precision ...] [--qmode ...] [--n 50]
// CSV: i,j,x,v,residual   (i indexes x, j indexes v — matches metrics.py)
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "khrpinn/hrpinn.hpp"
#include "khrpinn/mlp.hpp"
#include "khrpinn/quant.hpp"

using namespace khrpinn;

int main(int argc, char** argv) {
    cli::Args args{argc, argv};
    const char* mpath = args.positional(0);
    if (!mpath) {
        std::fprintf(stderr,
                     "usage: khrpinn_discovery model.khrp [--precision f32|i16|i8|i6|i4] "
                     "[--qmode full|...] [--n 50]\n");
        return 2;
    }
    int bits = cli::parse_precision(args.get("--precision", "f32"));
    std::string qmode = args.get("--qmode", "full");
    long n = args.getl("--n", 50);

    Model m = load_model(mpath);
    Workspace ws(m);
    QuantModel* qm = bits != 0 ? new QuantModel(m, qmode_config(qmode, bits)) : nullptr;

    std::printf("i,j,x,v,residual\n");
    for (long i = 0; i < n; ++i) {
        // np.linspace(-2, 2, n)
        float x = -2.0f + 4.0f * (float)i / (float)(n - 1);
        for (long j = 0; j < n; ++j) {
            float v = -2.0f + 4.0f * (float)j / (float)(n - 1);
            float s[2];
            normalize_state(m, x, v, s);
            float r = qm ? residual_forward_quant(*qm, ws, s)
                         : residual_forward(m, ws, s);
            std::printf("%ld,%ld,%.7g,%.7g,%.9g\n", i, j, x, v, r);
        }
    }
    delete qm;
    return 0;
}
