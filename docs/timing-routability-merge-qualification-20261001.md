# Timing/routability merge qualification

The selected main is `4a4b3df7`; the timing starting point is `2d894908`.
Merge commit `90863235` preserves both parents. This CPU qualification uses
Torch 2.11.0+cpu, Python 3.11.15 and C++ ABI 1.

Canonical integration report:
`ecc/docs/qualification/ecc-dreamplace-main-merge-20261001.md`.
Local evidence:
`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_dreamplace_merge_20261001.5RL63n`.
The latter contains build/install logs, effective configs, native hashes,
DEF/Verilog, observer reports and command status. Historical outputs retain
their actual source version; they are not relabelled as runs of later commits.

## Production owners

- M2 export options: `ops/placeio_ecc/export_options.py`; initial export and
  refresh share the effective options.
- Native DEF: existing `ops/placeio_common/physical_mutation.py` writer.
- Full-rebuild route invalidation: `ops/routability/gpugr_context.py`;
  parser keys include topology generation.
- L-shape objective gradient: `ops/routability/l_shape_gradient.py`;
  reset remains with `l_shape_telemetry.py`.
- Macro pin-halo restoration: `ops/macro_overlap/macro_pin_geometry.py`.
- L-shape plotting: `l_shape_segment_plots.py`, `l_shape_density_plots.py`,
  `l_shape_source_plots.py`. No forwarding functions remain in the operator.
- Macro-only coordinate synchronization: `MacroPlaceDB.write_placement_back`
  uses the native frozen-candidate API; Xplace reparses its selectively written
  DEF. The dense parser cache cannot preserve excluded instances.
- Xplace plotting/filler dependency shims exist only during helper import and
  restore process modules in `finally`.

## Bounded qualification entry points

`tests/ops/gpugr/native_refresh_route.py` takes `--manifest`, `--backend`,
`--rc-tcl`, `--output`. On the clocked GCD fixture it runs three native routes:
initial, master-only mutation/full refresh, buffer mutation/full refresh.
Each uses CPU PR MT, RRR=0 and two threads. It also checks native Steiner
ordinary/deterministic forward/backward.

`native_routability_loop.py` uses the same manifest and RC arguments.
It observes real optimizer updates, one inflation and physical-size restoration;
its explicit ten-step fixture configuration does not change production defaults.

`native_route_geometry_audit.py` takes `--manifest`, `--routes`, `--output`.
It reparses retained route DEFs without rerouting. Component masters, integer
coordinates and orientations must match exactly; signal connectivity must match
the corresponding native snapshot. The changed master's pin bbox is checked
against the old/new LEFs with 1 DBU quantization tolerance.
Physical-only taps/endcaps remain in geometry checks.

`native_macro_only_handoff.py` takes the same manifest/RC/output arguments.
It proposes moves for all GCD standard cells and verifies the real ECC native
selective API leaves excluded coordinates, orientations and statuses unchanged.
GCD has no movable macros; this proves excluded-cell preservation, not macro
movement or OpenROAD macro-only support.

Xplace uses a union bbox pin center; placement PyDB uses a rectangle-center
average. The audit records their differences instead of asserting model parity.

The area-mode density diagnostics also remove three upstream calls to an
undefined negative-scale helper. Planar/H-V payload tests validate calibrated
residual/overflow output; optimization formulas are unchanged.

## Limits

BM64 ECC/OpenROAD joint and continuation were qualified through normal
`ecc run`, with nonempty optimizer-generated actions. Outputs were not legalized.
Native ECC STA without fresh SPEF remains diagnostic; native RAT is still used
and placement RC is recalculated by DreamPlace LEF/FLUTE.
`supports_committed_refresh=false`,
`sta_state_status=native_rebuilt_without_fresh_rc` and
`atomic_across_sizing_and_buffer=false` remain unchanged.

This is neither CUDA qualification nor OpenROAD-native GP/repair_timing QoR
parity. OpenROAD has no native M2 rail export and no qualified macro-only
selective-writeback API. Its ordinary route/joint paths were qualified.
