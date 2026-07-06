# khrpinn C++ engine — native + RISC-V builds, tests, tools.
#
#   make            build native tools + tests (build/)
#   make test       run the full test suite (needs models/ + data/goldens/)
#   make test-o3    same at -O3 (catches FP-contraction surprises)
#   make riscv      static RV64GC cross-build (build/riscv/)
#   make test-riscv run the RISC-V test binary under qemu-riscv64
#   make deploy     tarball for the VisionFive 2 (deploy/khrpinn_riscv.tar.gz)
#
# -ffp-contract=off keeps FMA contraction from perturbing the golden numerics
# (identical op-by-op float32 rounding on x86-64 and RV64GC).

CXX        ?= g++
CXXSTD      = -std=c++17
WARN        = -Wall -Wextra -Werror
FPFLAGS     = -ffp-contract=off
OPT        ?= -O2
CXXFLAGS    = $(CXXSTD) $(OPT) $(WARN) $(FPFLAGS) -Iinclude
LDFLAGS     =

RISCV_CXX   ?= riscv64-linux-gnu-g++
RISCV_FLAGS  = $(CXXSTD) -O2 $(WARN) $(FPFLAGS) -Iinclude -march=rv64gc -mabi=lp64d -static

BUILD       = build
RBUILD      = build/riscv
REPO        = ..

TOOLS       = khrpinn_infer khrpinn_verify khrpinn_bench khrpinn_traj khrpinn_discovery
TEST_SRCS   = tests/test_main.cpp tests/test_spline.cpp tests/test_mlp.cpp \
              tests/test_kan.cpp tests/test_quant.cpp tests/test_hrpinn.cpp \
              tests/test_loader.cpp tests/test_golden.cpp
HDRS        = $(wildcard include/khrpinn/*.hpp) tests/test.hpp tests/golden_io.hpp src/cli.hpp

NATIVE_BINS = $(addprefix $(BUILD)/,$(TOOLS)) $(BUILD)/khrpinn_tests
RISCV_BINS  = $(addprefix $(RBUILD)/,$(TOOLS)) $(RBUILD)/khrpinn_tests

.PHONY: all test test-o3 riscv test-riscv deploy clean

all: $(NATIVE_BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(RBUILD):
	mkdir -p $(RBUILD)

$(BUILD)/khrpinn_%: src/khrpinn_%.cpp $(HDRS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BUILD)/khrpinn_tests: $(TEST_SRCS) $(HDRS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRCS) $(LDFLAGS)

test: $(BUILD)/khrpinn_tests
	cd $(REPO) && ./cpp/$(BUILD)/khrpinn_tests

test-o3:
	$(MAKE) OPT=-O3 BUILD=build/o3 build/o3/khrpinn_tests
	cd $(REPO) && ./cpp/build/o3/khrpinn_tests

$(RBUILD)/khrpinn_%: src/khrpinn_%.cpp $(HDRS) | $(RBUILD)
	$(RISCV_CXX) $(RISCV_FLAGS) -o $@ $<

$(RBUILD)/khrpinn_tests: $(TEST_SRCS) $(HDRS) | $(RBUILD)
	$(RISCV_CXX) $(RISCV_FLAGS) -o $@ $(TEST_SRCS)

riscv: $(RISCV_BINS)

test-riscv: $(RBUILD)/khrpinn_tests
	cd $(REPO) && qemu-riscv64 ./cpp/$(RBUILD)/khrpinn_tests

deploy: riscv
	mkdir -p $(REPO)/deploy/khrpinn_riscv/bin
	cp $(RISCV_BINS) $(REPO)/deploy/khrpinn_riscv/bin/
	cp -r $(REPO)/models $(REPO)/deploy/khrpinn_riscv/models
	rm -rf $(REPO)/deploy/khrpinn_riscv/models/pt
	cp -r $(REPO)/data/goldens $(REPO)/deploy/khrpinn_riscv/goldens
	cp $(REPO)/scripts/run_bench_board.sh $(REPO)/deploy/khrpinn_riscv/
	cp $(REPO)/scripts/energy_protocol.md $(REPO)/deploy/khrpinn_riscv/ 2>/dev/null || true
	cp $(REPO)/scripts/energy_log_template.csv $(REPO)/deploy/khrpinn_riscv/ 2>/dev/null || true
	cd $(REPO)/deploy && tar czf khrpinn_riscv.tar.gz khrpinn_riscv
	@echo "deploy/khrpinn_riscv.tar.gz ready"

clean:
	rm -rf $(BUILD)
