# Gate Projection Op

This directory hosts the first standalone projection operator for continuous
`size_var + vt_var` states. The current version is intentionally Python-only
and strategy-light: it focuses on the contract and extension points instead of
hard-coding one gate sizing method into the optimization flow.

## Scope

The operator is meant to run after a continuous optimization step and map each
instance back to a legal library cell candidate.

Current responsibilities:

- enumerate legal candidates from `inst_main_id` and library metadata
- expose pluggable scorer and validator interfaces
- expose a pluggable resolver interface instead of forcing plain `argmin`
- return a structured projection result instead of a bare offset tensor

Current non-goals:

- no integration into the differentiable placement gradient loop
- no consensus ADMM logic
- no committed timing-sensitivity backend yet

## Main Objects

- `MainIdCandidateProvider`
  - enumerates per-instance candidates from `main_id_2_cell_id_start`
  - preserves per-instance legality constraints from `inst_size_lower`,
    `inst_size_upper`, and `inst_vt_mask`
- `ProjectionScorer`
  - pluggable scoring interface
  - future timing-aware or sensitivity-guided strategies should replace this
    layer, not the whole operator
- `ProjectionValidator`
  - checks whether the selected discrete mapping is legal and structurally sane
- `ProjectionResolver`
  - selects the final candidate from the scored candidate set
  - built-ins include plain argmin and a stable current-cell tie-break resolver
- `GateProjectionOp`
  - orchestrates candidate enumeration, scoring, resolution, and
    validation
  - can emit standalone post-optimization artifacts without coupling the
    operator to a specific sizing strategy

## Post-Optimization Artifacts

The operator is now connected to a standalone post-optimization/report path in
`NonLinearPlace`. When sizing tensors exist, AutoDMP projects sizeable
instances once at the end of the run and emits:

- `<design>_projection.jsonl`
- `<design>_projection.csv`
- `<design>_projection_summary.json`

These artifacts live under `params.result_dir` and are intentionally separate
from the iterative metrics traces. The summary file carries validation issues,
resolver metadata, and aggregate change counts so later scorers can reuse the
same report contract.
It now also records discrete leakage accounting via:

- `total_current_leakage`
- `total_projected_leakage`
- `total_leakage_delta`
- `num_leakage_improved_cells`

and mirrors the same information under `validation.metrics.leakage`.

When pin-to-libpin metadata exists, the operator also emits:

- `<design>_projection_pin_offsets.jsonl`
- `<design>_projection_pin_offsets.csv`
- `<design>_projection_pin_offsets_summary.json`
- `<design>_projection_pin_offsets_runtime_summary.json`

The first three files describe the post-projection target pin-offset state
implied by the selected discrete cell. The runtime summary records the
subsequent in-memory rewrite of `pin_offset_x/y` that `NonLinearPlace` now
applies after projection so later reporting and legalization-consistency checks
see the projected pin offsets instead of the pre-projection ones.

## Default Behavior

The default scorer is `DistanceProjectionScorer`. It is only a placeholder
baseline and scores candidates with:

- size distance
- VT mismatch
- a tiny offset tie-breaker
- optional externally supplied candidate terms

This means the operator already supports future external signals such as:

- local timing penalty
- area penalty
- density penalty
- leakage penalty
- sensitivity-guided candidate scores

through `ProjectionContext(candidate_terms=..., term_weights=...)`.

There is also a `TimingAwareProjectionScorer` template that reserves stable
names for future strategy terms:

- `timing_penalty`
- `area_penalty`
- `density_penalty`
- `leakage_penalty`
- `sensitivity_penalty`

## Extension Path

The intended extension path is:

1. keep `MainIdCandidateProvider` stable as the library legality layer
2. swap in a richer scorer, e.g. `SensitivityGuidedScorer`
3. replace or extend the resolver if plain argmin becomes too weak for repair
   or stability constraints
4. keep validation independent so all strategies share the same legality checks

## Basic Usage

```python
from dreamplace.ops.gate_projection.gate_projection import GateProjectionOp

projection_op = GateProjectionOp(data_collections)
result = projection_op()

print(result.projected_libcell_offset)
print(result.projection_cost)
print(result.resolution_metadata)
print(result.validation.is_valid)

paths = projection_op.write_artifacts(
    result,
    result_dir="/tmp/projection",
    design_name="toy",
    metadata={"artifact_scope": "post_optimization"},
)
print(paths["summary"])
```

Custom scorers should subclass `ProjectionScorer` and return a
`ProjectionScore(total_score=..., terms=...)`.
