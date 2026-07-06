// khrpinn_bench — latency / energy-loop benchmark for the residual branch and
// the full HRPINN step. Single-threaded, allocation-free hot loop, checksum
// sink against dead-code elimination, inputs cycle a 1024-state buffer.
//
// Latency mode (default): --reps 30 --iters 100000 --warmup 10000
//   Emits one CSV row:
//   host,compiler,arch,precision,qmode,system,model,params,flops_est,exps_est,
//   weight_bytes,mode,reps,iters_per_rep,ns_median,ns_p10,ns_p90,ns_min,checksum
//
// Energy mode: --loop-seconds S   (run step loop for S wall seconds)
//   Emits:
//   host,compiler,arch,precision,qmode,system,model,params,mode,loop_seconds,
//   steps,elapsed_s,steps_per_sec,checksum
//
// --header prints the CSV header for the chosen mode and exits.
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "khrpinn/bench.hpp"
#include "khrpinn/hrpinn.hpp"
#include "khrpinn/quant.hpp"

using namespace khrpinn;

static const char* LAT_HEADER =
    "host,compiler,arch,precision,qmode,system,model,params,flops_est,exps_est,"
    "weight_bytes,mode,reps,iters_per_rep,ns_median,ns_p10,ns_p90,ns_min,checksum";
static const char* NRG_HEADER =
    "host,compiler,arch,precision,qmode,system,model,params,mode,loop_seconds,"
    "steps,elapsed_s,steps_per_sec,checksum";

int main(int argc, char** argv) {
    cli::Args args{argc, argv};
    std::string mode = args.get("--mode", "residual");  // residual | step
    double loop_seconds = args.getd("--loop-seconds", 0.0);
    if (args.has("--header")) {
        std::printf("%s\n", loop_seconds > 0 ? NRG_HEADER : LAT_HEADER);
        return 0;
    }
    const char* mpath = args.positional(0);
    if (!mpath || (mode != "residual" && mode != "step")) {
        std::fprintf(stderr,
                     "usage: khrpinn_bench model.khrp [--mode residual|step] "
                     "[--precision f32|i16|i8|i6|i4] [--qmode full|...] "
                     "[--reps N] [--iters N] [--warmup N] [--loop-seconds S] [--header]\n");
        return 2;
    }
    int bits = cli::parse_precision(args.get("--precision", "f32"));
    std::string qmode = args.get("--qmode", "full");
    long reps = args.getl("--reps", 30);
    long iters = args.getl("--iters", 100000);
    long warmup = args.getl("--warmup", 10000);
    // --recip: FP32 KAN with precomputed knot-difference reciprocals instead
    // of divisions (deployment-optimized basis path; anti-strawman variant).
    const bool recip = args.has("--recip");
    if (recip) qmode = "recip_bases";  // distinguishes rows in the CSV

    Model m = load_model(mpath);
    Workspace ws(m);
    QuantModel* qm = nullptr;
    if (bits != 0) qm = new QuantModel(m, qmode_config(qmode, bits));

    // 1024 deterministic input states in [-1,1]^2 (realistic branch domain).
    constexpr int NBUF = 1024;
    std::vector<float> buf(2 * NBUF);
    cli::Rng rng(0xC0FFEEu);
    for (auto& x : buf) x = rng.uniform();

    double checksum = 0.0;
    long idx = 0;
    auto one_iter = [&]() {
        const float* s = &buf[2 * (idx & (NBUF - 1))];
        ++idx;
        if (mode == "residual") {
            if (bits != 0) checksum += residual_forward_quant(*qm, ws, s);
            else if (recip && m.arch == Arch::KAN) checksum += kan_forward(m, ws, s, true);
            else checksum += residual_forward(m, ws, s);
        } else {
            float out[2];
            if (bits == 0) hrpinn_step(m, ws, s, out);
            else hrpinn_step_quant(*qm, ws, s, out);
            checksum += out[0] + out[1];
        }
    };

    for (long i = 0; i < warmup; ++i) one_iter();

    long flops = 0, exps = 0;
    m.flops_estimate(flops, exps);
    std::string host = host_arch();

    if (loop_seconds > 0) {  // energy mode
        const uint64_t t0 = now_ns();
        const uint64_t budget = (uint64_t)(loop_seconds * 1e9);
        long steps = 0;
        while (now_ns() - t0 < budget) {
            for (int i = 0; i < 2000; ++i) one_iter();  // check clock coarsely
            steps += 2000;
        }
        const double elapsed = (double)(now_ns() - t0) / 1e9;
        std::printf("%s,%s,%s,%s,%s,%s,%s,%zu,energy,%g,%ld,%.6f,%.1f,%.6g\n",
                    host.c_str(), compiler_id(), arch_name(m.arch),
                    cli::precision_name(bits), qmode.c_str(), system_name(m.system),
                    cli::slug_of(mpath).c_str(), m.param_count(), loop_seconds,
                    steps, elapsed, steps / elapsed, checksum);
        delete qm;
        return 0;
    }

    std::vector<double> ns_per_iter;
    ns_per_iter.reserve(reps);
    for (long r = 0; r < reps; ++r) {
        const uint64_t t0 = now_ns();
        for (long i = 0; i < iters; ++i) one_iter();
        const uint64_t t1 = now_ns();
        ns_per_iter.push_back((double)(t1 - t0) / (double)iters);
    }
    BenchStats st = bench_stats(ns_per_iter);
    std::printf("%s,%s,%s,%s,%s,%s,%s,%zu,%ld,%ld,%zu,%s,%ld,%ld,%.2f,%.2f,%.2f,%.2f,%.6g\n",
                host.c_str(), compiler_id(), arch_name(m.arch),
                cli::precision_name(bits), qmode.c_str(), system_name(m.system),
                cli::slug_of(mpath).c_str(), m.param_count(), flops, exps,
                m.weight_bytes(32), mode.c_str(), reps, iters,
                st.median, st.p10, st.p90, st.min, checksum);
    delete qm;
    return 0;
}
