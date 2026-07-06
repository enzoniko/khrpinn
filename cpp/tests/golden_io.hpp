// Shared helpers for golden-vector tests: binary/CSV readers + model discovery.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gio {

// Golden IO record: (x, v) input -> out. Files: u32 N | N x 3 f32.
struct IoRec { float x, v, out; };
// Step record: s0 -> s1. Files: u32 N | N x 4 f32.
struct StepRec { float s0x, s0v, s1x, s1v; };

inline std::string models_dir() {
    const char* e = std::getenv("KHRPINN_MODELS");
    return e ? e : "models";
}
inline std::string goldens_dir() {
    const char* e = std::getenv("KHRPINN_GOLDENS");
    return e ? e : "data/goldens";
}

inline std::vector<std::string> list_model_slugs() {
    std::vector<std::string> slugs;
    namespace fs = std::filesystem;
    if (!fs::exists(models_dir())) return slugs;
    for (const auto& e : fs::directory_iterator(models_dir())) {
        if (e.path().extension() == ".khrp")
            slugs.push_back(e.path().stem().string());
    }
    std::sort(slugs.begin(), slugs.end());
    return slugs;
}

template <typename Rec>
inline std::vector<Rec> read_bin(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("missing golden: " + path);
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) { std::fclose(f); throw std::runtime_error("bad golden header"); }
    std::vector<Rec> recs(n);
    size_t got = std::fread(recs.data(), sizeof(Rec), n, f);
    std::fclose(f);
    if (got != n) throw std::runtime_error("truncated golden: " + path);
    return recs;
}

// traj_f32.csv: header "ic,step,x,v"; returns per-IC vectors of (x, v).
inline std::vector<std::vector<std::pair<float, float>>> read_traj_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("missing golden: " + path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<std::vector<std::pair<float, float>>> tra;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        size_t ic = std::stoul(tok);
        std::getline(ss, tok, ',');  // step (implicit by order)
        std::getline(ss, tok, ',');
        float x = std::stof(tok);
        std::getline(ss, tok, ',');
        float v = std::stof(tok);
        if (tra.size() <= ic) tra.resize(ic + 1);
        tra[ic].push_back({x, v});
    }
    return tra;
}

// truth_*.csv: header "t,x,v"; returns (t, x, v) rows (double precision).
inline std::vector<std::array<double, 3>> read_truth_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("missing golden: " + path);
    std::string line;
    std::getline(in, line);
    std::vector<std::array<double, 3>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::array<double, 3> r;
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, tok, ',');
            r[i] = std::stod(tok);
        }
        rows.push_back(r);
    }
    return rows;
}

}  // namespace gio
