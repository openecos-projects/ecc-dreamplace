# Routability Merge Verification

Date: 2026-08-12

This record covers the local, non-published result on
`routability-driven-placement`. It does not authorize or imply a push.

## History

- Repository: `git@gitee.com:ieda-ipd/ecc-dreamplace.git`
- Architecture baseline: `main` at `390a18c3af9adbc9aea3fda88b9d3636af2e3d5d`
- Routability source baseline: `experiment/ggr-lshape-al` at `05d92c93f707223558b7cd4656258f106916942e`
- Merge base: `8e18e1b2d49b427a4e4ea39070d005b2721751e4`
- Merge commit: `1a95944b4086a2b26fb4db1f747182c8ae373da8`
  - first parent: `390a18c3` (`main`)
  - second parent: `8e18e1b2` (local experiment lineage containing the source baseline)
- Safety ref: `codex/pre-routability-main-merge-20260812` -> `05d92c93f707223558b7cd4656258f106916942e`
- Final local branch: `routability-driven-placement`
- Final child tip: `857c3de2db70d181f842932a5b67831bef6619db`

Follow-up commits after the merge are `9cb66fa0` (ECC API integration),
`241b5586` (compatibility hardening), `c07da3e5` (legacy EGR boundary
clarification and contract test), `4bbab9a0` (verification record),
`2cf70d6b` (duplicate-log cleanup), `92a81089` (tip record), and `e48ce8ae`
(final tip alignment).

## Implementation and dependency boundary

`MacroPlaceDB` uses the ECC-injected `ECCToolsModule` and current five-argument
`pydb` contract. Placement write-back uses the ECC DB/DEF/TCL APIs. The legacy
`adjust_nctugr_area_flag` name remains accepted for configuration compatibility,
but resolves to the ECC/iRT EGR route map and logs that NCTUgr is not invoked.
Modularity inflation explicitly rejects that legacy key because it requires the
GPUGR route source.

The tracked source tree no longer contains `dreamplace/ops/nctugr_binary`,
`dreamplace/ops/place_io`, `thirdparty/NCTUgr.ICCAD2012`, `thirdparty/OpenTimer`,
`dreamplace/Timer.py`, or `dreamplace/ops/timing`. Active Python source has no
`nctugr_binary`, `PlaceIOFunction`, `place_io import`, `IEDAIO`, or
`IEDADesign` references. Remaining hits are compatibility names, explicit
unsupported timing text, ordinary profiling timers, or fake modules in tests.

## Build and focused tests

- CMake build/install: `cmake --build .../ecc-dreamplace/build` and install;
  `98/98` native targets completed successfully (subsequent test invocations
  reported `ninja: no work to do`). Python 3.11 extensions are present for
  Steiner topology, routability compaction/scoring, and the other touched ops.
- `python -m compileall -q dreamplace`: passed.
- `git diff --check`: passed.
- Source-aware contract test:
  `PYTHONPATH=. .venv/bin/python dreamplace/unittest_legacy_route_map_source.py -v`:
  `3 passed`.
- L-shape/Steiner/inflation focused set using
  `pytest --import-mode=importlib`: `45 passed, 1 warning`.
- `unittest_params_l_shape_preset.py`: `9 passed`.
- `unittest_m2_pg_rail_blockage_flag.py`: `17 passed`.
- `unittest/ops/egr_padding_flow_unittest.py`: `1 passed` after restoring the production
  lifecycle `standard legalization -> EGR map/padding -> padded legalization ->
  restore`; the fixture now supplies the current `timing_opt_flag` contract.
- Final combined child focused set including EGR padding and legacy route
  source: `68 passed, 2 warnings, 1 subtest passed`.

The warning in the 45-test set is the expected CPU-only CUDA initialization
warning on a host without a CUDA driver.

The two additional warnings in the final combined set are the same expected
CPU-only CUDA initialization warning plus a PyTorch tensor-copy warning in the
existing enhanced-inflation helper.

## ECC smoke runs

Both commands were rerun after the EGR padding lifecycle fix with the editable ECC entry point and `--only place
--force --json`:

```text
.venv/bin/ecc run --workspace /home/zhaoxueyan/ecc-routability-plain-1786528350 --only place --force --json
.venv/bin/ecc run --workspace /home/zhaoxueyan/ecc-routability-route-1786528350 --only place --force --json
```

The plain workspace used `routability_opt_flag=0`; the route workspace used
`routability_opt_flag=1`, `l_shape_routability_flag=1`,
`adjust_nctugr_area_flag=1`, `adjust_gpugr_area_flag=0`, and
`timing_opt_flag=0`. Both used a five-iteration global placement stage.

Both runs exited with business failure status `1`, not signal `139`. Persisted
states after the rerun were:

```text
place: Incomplete
run placement: Incomplete
save data: Unstart
analysis: Unstart
```

The placement log reports overflow `0.7437971234` at iteration `5`, skips
legalization/detail placement, and logs `placement failed`. This is a safe
failure-path result, not a successful placement claim.

The routability workspace initialized ECC `pydb`, ran the ECC/iRT EGR route
map, and generated these artifacts under
`place_dreamplace/feature/` (also retained in the attempt record):

- `egr_congestion_map/place_egr_{horizontal,vertical,union}_overflow.{csv,png}`
- `margin_map/place_{horizontal,vertical,union}_margin.{csv,png}`
- `density_map/` maps
- `place.map.json` and `place.step.json`

No `SIGSEGV`, NCTUgr binary lookup, or `place_io` import failure occurred.

The rerun preserved the same persisted subflow states and exit status: both
commands returned `1`, with `run placement=Incomplete`, `save data=Unstart`,
and `analysis=Unstart`. The plain log again reported overflow `0.744` and
`placement failed`; the routability workspace again produced the EGR maps,
margin maps, density maps, and `place.map.json` listed above.

## Standalone fixture and timing negative check

The plan's historical command was attempted:

```text
.venv/bin/python dreamplace/Placer.py test/simple.json
```

The current `test/simple.json` still points to
`benchmarks/simple/simple.aux`, but that fixture is absent from the tracked
tree. In addition, the current `dreamplace/Placer.py` exposes an ECC engine API
without a standalone `__main__` runner, so this command only refreshed the
editable native install and did not execute placement. This is recorded as a
pre-existing standalone Bookshelf/toolchain limitation; the ECC `ics55_gcd`
workspace smoke above is the architecture-valid replacement.

Timing compatibility was checked with:

```text
.venv/bin/python -c 'from types import SimpleNamespace; from dreamplace.Placer import PlacementEngine; e=PlacementEngine.__new__(PlacementEngine); e.params=SimpleNamespace(timing_opt_flag=1); e.place()'
```

It exits `1` with exactly:
`RuntimeError: timing_opt_flag is no longer supported because OpenTimer integration has been removed`.

## Remote boundary

At verification time `origin/main` was `390a18c3` and
`origin/experiment/ggr-lshape-al` was `05d92c93`; `origin/routability-driven-placement`
did not exist. No push was performed.

## Built-in Xplace GPUGR verification (2026-08-26)

The GPUGR orchestration now lives in `dreamplace/ops/gpugr/` and has no runtime
import or fallback to `tools.iEDA.module.gpugr`. Xplace is a nested submodule.
The local Xplace commit `5632aaf55f08b46835aba5e9a2fe694999edadc8`
builds on `b74548412f9896ef9963dce5cfbc4584b0422bb9` and records nested pybind11
`f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8` for the Torch 2.11 pybind ABI.
The submodule is found automatically; `ECC_XPLACE_ROOT` is an optional override.

The build script was rerun using the parent ECC `.venv`. It detected
`torch 2.11.0+cpu`, configured `XPLACE_ENABLE_CUDA=OFF`, retained the CPU-compatible
GPUGR, IOParser, Flute, route-DP, and timer extensions, and removed stale CUDA
extensions from an older build. The loader imported IOParser, GPUGR, Flute,
`calc_gr_wl_via`, and `estimate_num_shorts`; `gpugr.cuda_enabled()` was false.
This verifies CPU_PR only, not the CUDA backend.

Focused tests passed after the final source changes:

```text
unittest_gpugr_backend_select.py       6 passed
unittest_xplace_parser_cache.py        2 passed
unittest_iopin_density_weight.py      13 passed
unittest_params_l_shape_preset.py      9 passed
unittest_shared_precondition.py        6 passed
```

The final physical rerun used:

```text
.venv/bin/ecc run --workspace /home/zhaoxueyan/gpugr-debug-lshape-fixed \
  --only place --force --json
```

It returned status `success`, executed only `place`, and persisted
`place=Success` with runtime `0:1:2`. The GGR L-shape topology pack was loaded
for nine placement updates. The final metrics recorded CPU_PR, a `19x16` route
grid, `rrr_iters=0`, `parser_cache_hit=true`, `parser_cache_active=true`, and
`native_issue_line_count=0`. The final DreamPlace placement overflow was
`0.0986458957`; this is a placement stopping metric, not a GPUGR routing-overflow
or QoR claim. The GPUGR route/map/metrics artifacts and the ECC EGR
`route.guide`, per-layer overflow maps, `place.map.json`, DEF, Verilog, and GDS
outputs were non-empty.

Two retained verification workspaces also record successful complete place
steps: `/home/zhaoxueyan/gpugr-debug-c1-fixed` for the control path and
`/home/zhaoxueyan/gpugr-debug-area-fixed` for `adjust_gpugr_area_flag=1`. The
area run resolved `backend=auto` to CPU_PR and recorded no native issue lines.

The earlier signal 139 was outside the Xplace backend. Union feature evaluation
ran the default planar EGR stage, then attempted to summarize missing egr3D
per-layer overflow maps. The empty input reached an unconditional top-one
access in `CongestionEval::evalAvgOverflow`. ECC-Tools now selects `egr3D` for
the union feature path and returns the unavailable sentinel for empty overflow
data. `eval_map_layout_writer_test` and `eval_congestion_metrics_test` passed,
and the complete ECC place analysis no longer crashes.

`xplace_backend.py` remains above the repository's normal module-size guideline.
The repository owner explicitly waived decomposition for this integration, so
no file split is required for this delivery. No push was performed.
