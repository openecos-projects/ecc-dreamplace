# `dreamplace/flows` Layered Architecture

`dreamplace/Placer.py` (`PlacementEngine`) is the programmatic and executable entry point; `dreamplace/placer_cli.py` owns standalone CLI argument parsing and `run_optimization_flow`.

## Subpackage Layers & Dependency Rules

`dreamplace/flows/` is organized into 6 domain subpackages with a strict unidirectional dependency DAG (enforced by `tests/flows/test_flow_layer_boundaries.py`):

```text
config  <-  gp  <-  discrete  <-  {s5b1, post_gp}  <-  baselines
```

1. **`config/`** (`profiles.py`, `specs.py`, `resolver.py`): Flow parameter schemas, default profiles, and `apply_flow_defaults` / `resolve_flow_config`. Leaf package within `flows/`.
2. **`gp/`** (`l_shape_iteration.py`, `topology_refresh.py`, `timing_carriers.py`, `hold_timing.py`, `metrics.py`, `timing_artifacts.py`, `continuation.py`): Global placement iteration hooks, timing carriers, metric/artifact writers, and warm-restart continuation state.
3. **`discrete/`** (`common.py`, `owner.py`, `sizing_filter.py`, `buffering_filter.py`, `buffer_commit.py`): Shared discrete Sizing/Buffering optimization engine (`DiscreteTimingOwner`), incremental timing acceptance gates, and direct ECC buffer commit.
4. **`s5b1/`** (`inflation_s5b1.py`, `committed_topology.py`): Default production milestone co-optimization (`InflationS5B1`: S5 discrete sizing + B1 3-round 2n-subnet `candidate` buffering) and terminal materialization (`finalize_inflation_s5b1`).
5. **`post_gp/`** (`rc_sizing.py`, `diff_sizing.py`, `diff_opt.py`): Post-placement standalone electrical flows (`DIFF_SIZING` with GR/Placement/SPEF parasitics, and optional `DIFF_OPT`).
6. **`baselines/`** (`joint_coordinator.py`, `joint_runtime.py`, `segment_joint.py`, `segment_buffering.py`, `sizing_dynamics.py`, `real_size_transition.py`, `gate_projection.py`, `alternating_discrete.py`, `midplace_repair.py`, `repair_buffering.py`): Retained comparison and historical co-optimization flows. Production subpackages (`config`, `gp`, `discrete`, `s5b1`, `post_gp`) must never import `baselines`, and `dreamplace/ops/` must never import `dreamplace/flows/`.
