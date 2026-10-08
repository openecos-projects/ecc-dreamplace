# AutoDMP Flow Entry Boundary

`dreamplace/Placer.py` remains the executable entry point for AutoDMP runs.
`dreamplace/placer_cli.py` owns the canonical CLI argument surface and effective
parameter construction for placement, diff-sizing, buffering, joint, and
physical-ECO flows.

Flow-level inference and recommended defaults live in `flow_config.py`. For
example, the canonical diff-sizing entry is `--flow-kind sizing`; its defaults
are intentionally defined here instead of in external experiment wrappers.

The canonical buffering entries are:

```text
--flow-kind buffering
--flow-kind physical_eco
--buffering-mode {segment,candidate}
```

`buffering` runs the current relaxed buffering analysis without enabling real
OpenROAD coordinate commit by default. `physical_eco` inherits the same
buffering profile and enables the legal-state real-commit loop.

The public buffering algorithm choice is `--buffering-mode`. `segment` is the
default production-oriented profile: optimize segment repeater counts and
project them to equal-spaced buffer insertions. `candidate` keeps the explicit
candidate-level relaxed/projection path available for comparison and debug.

Historical research-specific buffering switches are no longer registered in
the public CLI. Existing internal `buffering_*` fields can remain in flow
defaults and the config adapter while lower-level runner code is migrated. New
flow code should read the canonical buffering config rather than adding more
direct `buffering_*` accesses.

The top-level `baseline/run_workspace_flow1.py` script is a legacy experiment
wrapper. It may still be useful for historical comparisons, but new canonical
flow defaults and new public entry behavior should not be added there.
