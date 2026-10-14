# Native timing propagation on CPU and CUDA

The placement timing fast loop dispatches through the existing device setting.
For a CPU placement, use `gpu=0` and `timing_propagation_device="cpu"`.
With `production_fast_loop=true`, the existing `timing_lut_2d_native_op="auto"`
now permits native LUT queries on both devices. No extra CPU enable switch is
required. Rebuild the native extensions before using this implementation.

The canonical timing-driven placement, STA-only, standalone `diff_sizing`,
and ECC `diff_opt` buffering profiles enable the fast loop and automatic native
LUT dispatch. The legacy `diff_buffering` step is an alias of `diff_opt`.
Standalone GR/RRR sizing keeps its separate RC configuration. Its size/VT cell
surrogates retain their differentiable path; fixed-master cell fusion is used
only when surrogate evaluation is disabled.

From the ECC workspace root, rebuild the editable native package under Nix:

```bash
nix develop
uv sync --no-build-isolation-package ecc-dreamplace \
  --no-build-isolation-package ecc-tools-bin --reinstall-package ecc-dreamplace
source .venv/bin/activate
```

Run focused tests from the DreamPlace checkout with that environment. The
device fixtures exercise CPU and CUDA and skip CUDA when unavailable:

```bash
python -m pytest -q \
  tests/ops/timing_propagation/test_lut_entry_indexed.py \
  tests/ops/timing_propagation/test_cell_aat_cuda.py \
  tests/ops/timing_propagation/test_net_aat_cuda.py \
  tests/ops/timing_propagation/test_timing_lanes_cuda.py \
  tests/ops/net_subgraph_timing/test_candidate_native_cpu.py \
  tests/ops/net_subgraph_timing/test_candidate_cuda_gradients.py
```

The common autograd modules dispatch to `lut_entry_2d_cpp` or
`lut_entry_2d_cuda`. They cover indexed scalar/1D/2D LUT queries, fixed-master
cell arrival/slew propagation, net arrival/slew propagation, four-lane
aggregation, hard/smooth scatter derivatives, buffer size interpolation and
buffer device LUT queries. Trainable LUT tables and cell surrogates retain
the existing differentiable reference paths. These operators provide the
first-order gradients used by GP and discrete sizing/buffering.

CPU forward loops distribute independent arcs across OpenMP workers. Cell/net
backward and four-lane aggregation give each destination pin one owner and
retain stable arc order. Candidate tree backward gives each independent net
one owner, checks forest ownership before entering OpenMP, and seeds repeated
sinks in input order. Both devices use common per-row arithmetic headers.
Small workloads run without spawning an OpenMP team.

`CandidateNativeAutogradFunction` dispatches the existing candidate forward
and the new CPU/CUDA reverse pass. The CPU reverse pass also supports the
shared four-strength probe used during B windows. GP retains frozen candidate
activations and differentiates through current edge R/C, sink capacitance,
buffer LUT coordinates and placement coordinates. Candidate positions keep
the existing segment split ratios. CPU candidate forward still uses the
existing C++ tree sweep; the CUDA-specific fixed-BSU fused tree forward is
not part of this CPU port.

Native autograd boundaries retain original tensor version counters, support
non-contiguous inputs, and preserve unused output gradient ownership. Fully
written CPU/CUDA allocations avoid a redundant diagnostic fill without
changing Torch's global deterministic memory-fill setting. Callers of those
allocation helpers must write every returned element.

Validation covers independent reference values and all first-order input
gradients, finite differences, hard ties, NaN/Inf adjoints, empty ranges,
appended Steiner nodes, table mutation and deterministic repeated reductions.
CPU-specific candidate tests also cover an optional empty wire-C vector and
sink timing relative to the input driver when a root candidate is present.

The 2026-10-14 BM64 qualification used a standalone CPU PyTorch 2.11 runtime,
8 threads, the same frozen placement inputs and S5B1 profile as the preceding
CUDA experiment. The complete native run converged in 413 GP iterations,
materialized 39 buffers and passed final legalization. Across all 90 ordinary
GP steps after the first S5B1 window, excluding the five S/B trigger steps,
full-step wall time averaged 310.816 ms (median 201.113 ms). This includes
Nesterov repeated objective evaluations and normal topology refresh.

On actual states with 13 and 39 buffers, native versus unfused-reference
position-gradient relative L2 errors were 4.717e-6 and 1.958e-6. Objective,
WNS and TNS were exactly equal at both states. Source, frozen runtime,
build commands, tests, full logs, final DEF/Verilog and timing summaries are
archived under `/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc-native-cpu-qualified-20261014.tt6z9_mh/`. The latest binary repeat averaged 408.339 ms
(median 255.921 ms), with the same final DEF/Verilog hashes.
