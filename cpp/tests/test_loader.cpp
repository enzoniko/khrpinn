// Model loader robustness tests.
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "golden_io.hpp"
#include "khrpinn/model.hpp"
#include "test.hpp"

using namespace khrpinn;
namespace fs = std::filesystem;

static std::string temp_file(const char* name, const std::string& content) {
    fs::path p = fs::temp_directory_path() / name;
    std::ofstream out(p, std::ios::binary);
    out.write(content.data(), (std::streamsize)content.size());
    out.close();
    return p.string();
}

TEST(loader_rejects_bad_magic) {
    std::string p = temp_file("khrp_badmagic.bin", std::string("XXXX") + std::string(64, '\0'));
    CHECK_THROWS(load_model(p));
    std::remove(p.c_str());
}

TEST(loader_rejects_truncated) {
    std::string p = temp_file("khrp_trunc.bin", std::string("KHRP\x01\x00\x00\x00", 8));
    CHECK_THROWS(load_model(p));
    std::remove(p.c_str());
}

TEST(loader_rejects_missing_file) {
    CHECK_THROWS(load_model("/nonexistent/path/model.khrp"));
}

TEST(loader_loads_all_exported_models) {
    auto slugs = gio::list_model_slugs();
    CHECK_MSG(!slugs.empty(), "no models found in " + gio::models_dir() +
                                  " — run train_and_export.py + make_goldens.py");
    for (const auto& slug : slugs) {
        Model m = load_model(gio::models_dir() + "/" + slug + ".khrp");
        CHECK(m.input_dim() == 2);
        CHECK(m.param_count() > 0);
        CHECK(m.dt > 0.0f && m.dt < 1.0f);
        CHECK(m.norm_range[0] > 0.0f && m.norm_range[1] > 0.0f);
        // arch encoded in the slug must match the file
        bool is_kan = slug.find("kan") != std::string::npos;
        CHECK_MSG((m.arch == Arch::KAN) == is_kan, "arch/slug mismatch for " + slug);
        bool is_vdp = slug.rfind("vdp", 0) == 0;
        CHECK_MSG((m.system == System::VdP) == is_vdp, "system/slug mismatch for " + slug);
        // Rejecting a truncated copy of a real model exercises deep validation.
        long flops = 0, exps = 0;
        m.flops_estimate(flops, exps);
        CHECK(flops > 0);
        CHECK((m.arch == Arch::KAN) == (exps > 0));
    }
}
