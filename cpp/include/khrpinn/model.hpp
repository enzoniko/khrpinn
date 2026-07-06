// khrpinn/model.hpp — .khrp model container + loader (format v1, little-endian).
// Zero dependencies beyond the C++17 standard library. See
// python/export/export_format.py for the authoritative format description.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace khrpinn {

enum class Arch : uint8_t { MLP = 0, KAN = 1 };
enum class System : uint8_t { Duffing = 0, VdP = 1 };

struct Range {
    float min = 0.0f, max = 0.0f;
};

struct MlpLayer {
    uint32_t in = 0, out = 0;
    bool has_relu = false;
    std::vector<float> W;  // [out*in], row-major (torch nn.Linear layout)
    std::vector<float> b;  // [out]
    Range act;             // calibrated post-activation output range
};

struct KanLayer {
    uint32_t in = 0, out = 0, G = 0, k = 0;
    std::vector<float> grid;      // [in * (G+2k+1)], per-input-feature knots
    std::vector<float> base_w;    // [out*in]
    std::vector<float> spline_w;  // [out*in*(G+k)], spline_scaler pre-folded
    // Precomputed knot-difference reciprocals for the optimized basis path
    // (bench --recip): per feature, for r=1..k, R_r[j] = 1/(g[j+r]-g[j]),
    // j = 0..(G+2k-r). Computed at load; the DEFAULT path uses divisions to
    // match kan.py numerics op-for-op (golden-pinned).
    std::vector<float> recip;     // [in * recip_stride()]
    Range silu_r, basis_r, act;

    uint32_t n_knots() const { return G + 2 * k + 1; }
    uint32_t n_bases() const { return G + k; }        // bases after recursion
    uint32_t n_bases0() const { return G + 2 * k; }   // order-0 (scratch size)
    uint32_t recip_stride() const {                   // sum over r of (G+2k-r+1)
        uint32_t s = 0;
        for (uint32_t r = 1; r <= k; ++r) s += G + 2 * k - r + 1;
        return s;
    }
    void precompute_recips() {
        const uint32_t m = n_bases0();
        recip.assign((size_t)in * recip_stride(), 0.0f);
        for (uint32_t i = 0; i < in; ++i) {
            const float* g = grid.data() + (size_t)i * n_knots();
            float* R = recip.data() + (size_t)i * recip_stride();
            size_t off = 0;
            for (uint32_t r = 1; r <= k; ++r) {
                for (uint32_t j = 0; j + r <= m; ++j)
                    R[off + j] = 1.0f / (g[j + r] - g[j]);
                off += m - r + 1;
            }
        }
    }
};

struct Model {
    Arch arch = Arch::MLP;
    System system = System::Duffing;
    uint16_t n_layers = 0;
    float dt = 0.05f;
    float norm_min[2] = {0, 0};
    float norm_range[2] = {1, 1};
    Range in_r;  // branch-input (state) calibration range

    std::vector<MlpLayer> mlp;
    std::vector<KanLayer> kan;

    uint32_t input_dim() const {
        if (arch == Arch::MLP) return mlp.front().in;
        return kan.front().in;
    }
    uint32_t max_width() const {
        uint32_t w = 2;
        for (const auto& l : mlp) { w = std::max(w, std::max(l.in, l.out)); }
        for (const auto& l : kan) { w = std::max(w, std::max(l.in, l.out)); }
        return w;
    }
    uint32_t max_bases_buf() const {  // widest (in * n_bases0) among KAN layers
        uint32_t w = 1;
        for (const auto& l : kan) w = std::max(w, l.in * l.n_bases0());
        return w;
    }
    size_t param_count() const {  // matches torch (incl. pre-fold spline_scaler)
        size_t n = 0;
        for (const auto& l : mlp) n += l.W.size() + l.b.size();
        for (const auto& l : kan) n += 2 * l.base_w.size() + l.spline_w.size();
        return n;
    }
    size_t weight_bytes(int bits_per_weight = 32) const {
        size_t n = 0;
        for (const auto& l : mlp) n += l.W.size() + l.b.size();
        for (const auto& l : kan) n += l.base_w.size() + l.spline_w.size() + l.grid.size();
        return n * (size_t)bits_per_weight / 8;
    }
    // Analytical scalar-op estimate per inference; mirrors npref.flops_estimate.
    void flops_estimate(long& flops, long& exps) const {
        flops = 0; exps = 0;
        for (const auto& l : mlp) {
            flops += 2L * (long)l.W.size() + (long)l.b.size();
            if (l.has_relu) flops += l.out;
        }
        for (const auto& l : kan) {
            flops += 2L * l.in * l.out;
            flops += 5L * l.in;
            exps += l.in;
            long basis_ops = 2L * (l.G + 2 * l.k);
            for (uint32_t r = 1; r <= l.k; ++r) basis_ops += 6L * (l.G + 2 * l.k - r);
            flops += (long)l.in * basis_ops;
            flops += 2L * l.in * l.out * l.n_bases();
        }
    }
};

namespace detail {

struct Reader {
    FILE* f;
    std::string path;
    explicit Reader(const std::string& p) : path(p) {
        f = std::fopen(p.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open model file: " + p);
    }
    ~Reader() { if (f) std::fclose(f); }
    void read(void* dst, size_t n) {
        if (std::fread(dst, 1, n, f) != n)
            throw std::runtime_error("truncated model file: " + path);
    }
    template <typename T> T scalar() { T v; read(&v, sizeof(T)); return v; }
    void floats(std::vector<float>& v, size_t n) { v.resize(n); read(v.data(), 4 * n); }
    Range range() { Range r; r.min = scalar<float>(); r.max = scalar<float>(); return r; }
    bool at_eof() {
        int c = std::fgetc(f);
        if (c == EOF) return true;
        std::ungetc(c, f);
        return false;
    }
};

}  // namespace detail

inline Model load_model(const std::string& path) {
    detail::Reader r(path);
    char magic[4];
    r.read(magic, 4);
    if (std::memcmp(magic, "KHRP", 4) != 0)
        throw std::runtime_error("bad magic in " + path);
    uint32_t version = r.scalar<uint32_t>();
    if (version != 1) throw std::runtime_error("unsupported .khrp version");

    Model m;
    uint8_t arch = r.scalar<uint8_t>();
    uint8_t sys = r.scalar<uint8_t>();
    if (arch > 1 || sys > 1) throw std::runtime_error("bad arch/system id in " + path);
    m.arch = static_cast<Arch>(arch);
    m.system = static_cast<System>(sys);
    m.n_layers = r.scalar<uint16_t>();
    if (m.n_layers == 0 || m.n_layers > 64) throw std::runtime_error("bad layer count");
    m.dt = r.scalar<float>();
    r.read(m.norm_min, 8);
    r.read(m.norm_range, 8);
    m.in_r = r.range();

    for (uint16_t i = 0; i < m.n_layers; ++i) {
        if (m.arch == Arch::MLP) {
            MlpLayer l;
            l.in = r.scalar<uint32_t>();
            l.out = r.scalar<uint32_t>();
            if (l.in == 0 || l.out == 0 || l.in > 4096 || l.out > 4096)
                throw std::runtime_error("bad MLP layer dims");
            l.has_relu = r.scalar<uint8_t>() != 0;
            r.floats(l.W, (size_t)l.out * l.in);
            r.floats(l.b, l.out);
            l.act = r.range();
            m.mlp.push_back(std::move(l));
        } else {
            KanLayer l;
            l.in = r.scalar<uint32_t>();
            l.out = r.scalar<uint32_t>();
            l.G = r.scalar<uint32_t>();
            l.k = r.scalar<uint32_t>();
            if (l.in == 0 || l.out == 0 || l.G == 0 || l.G > 64 || l.k == 0 || l.k > 8)
                throw std::runtime_error("bad KAN layer dims");
            r.floats(l.grid, (size_t)l.in * l.n_knots());
            r.floats(l.base_w, (size_t)l.out * l.in);
            r.floats(l.spline_w, (size_t)l.out * l.in * l.n_bases());
            l.silu_r = r.range();
            l.basis_r = r.range();
            l.act = r.range();
            l.precompute_recips();
            m.kan.push_back(std::move(l));
        }
    }
    if (!r.at_eof()) throw std::runtime_error("trailing bytes in " + path);

    // Chain consistency
    uint32_t prev = 2;
    for (const auto& l : m.mlp) {
        if (l.in != prev) throw std::runtime_error("layer dim chain mismatch");
        prev = l.out;
    }
    for (const auto& l : m.kan) {
        if (l.in != prev) throw std::runtime_error("layer dim chain mismatch");
        prev = l.out;
    }
    if (prev != 1) throw std::runtime_error("model output dim must be 1");
    return m;
}

inline const char* arch_name(Arch a) { return a == Arch::MLP ? "mlp" : "kan"; }
inline const char* system_name(System s) { return s == System::Duffing ? "duffing" : "vdp"; }

}  // namespace khrpinn
