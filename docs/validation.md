# Validation and numerical conventions

## Reference implementations

`python/iclr/` retains the original experiment scripts. `python/tests/test_parity.py`
compares the shared implementations with those scripts using matching seeds and
training recipes. KAN grid adaptation uses `torch.linalg.lstsq`; threaded LAPACK
reductions can produce small differences between runs. The parity test pins a
single thread and uses the tolerance stated in the test for KAN outputs.

Exports fold the learned spline scaler into spline coefficients. The exporter
checks the NumPy forward implementation against the trained PyTorch model before
writing a model. Reference-vector tests then compare the C++ residual functions,
integration steps, and trajectories against the exported model's reference data.

```sh
make -C cpp test
make -C cpp test-o3
make -C cpp test-riscv
```

The last command requires the RISC-V cross-toolchain and QEMU. A successful QEMU
test is a correctness check, not a board-performance measurement. Run the
deployment verifier on the board before collecting measurements.

## Floating-point and simulated quantization

Builds disable floating-point contraction with `-ffp-contract=off`. NumPy/BLAS
and scalar C++ accumulate sums in different orders, so FP32 outputs can differ
slightly. The tests document tolerances for residuals, integration steps, and
rollouts; these are separate comparisons with different error propagation.

The quantized reference rounds and clips intermediate values while retaining
floating-point computations. Rounding-boundary differences can propagate through
successive layers. INT16 golden tests consequently use a wider tolerance in LSBs
than INT8 tests. Consult the executable tests for the enforced tolerances; the
standalone deployment verifier may enforce a different check.

Calibration ranges come from training trajectories. Inputs outside those ranges
can clip, including at 16 bits, and may produce larger errors than an evaluation
restricted to the training domain. The evaluation distinguishes rounding of
inputs, weights, activations, the residual branch, and the integrated state.
The selected mode is recorded with each result.

The reported INT16/INT8 memory estimates scale stored arrays, including KAN knots.
An implementation that retains higher-precision knots would have a different
memory footprint. Original PyTorch checkpoint files are not committed; regenerating
them requires rerunning training.

## Timing and board measurements

The C++ benchmark uses contiguous scalar loops, checksum sinks, warmup, repeated
measurements, and a fixed input buffer. Step-mode timing uses precomputed states
rather than an autoregressive rollout. The reciprocal KAN variant precomputes
knot-difference reciprocals; its measured results are stored separately from the
division-based variant.

KAN timing can vary across trained models of the same architecture. Recorded
measurements should therefore be interpreted per model, compiler, and platform;
candidate explanations for such differences are not established causal findings.

The board protocol records idle and active readings separately and checks idle
drift. The energy-pipeline example under `results/energy_board_simulated/` uses
synthetic meter data and is kept separate from actual board measurements. A
simulated pipeline run does not establish hardware energy consumption.
