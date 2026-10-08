# ICCAD24 Regressions

This benchmark family uses the local mirror at
`references/benchmarks/iccad24_benchmark`.

Experiments:

```text
diff_sizing/   Canonical full-design diff-sizing smoke/regression.
sta_compare/   OpenROAD STA versus AutoDMP STA-only timing comparison.
buffering_reasonable_mode/   Canonical buffering TNS improvement smoke.
```

Each experiment directory owns its entrypoint, parameter notes, and checked-in
report notes. Generated logs are written under `logs/regression/iccad24/`.
