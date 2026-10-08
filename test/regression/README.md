# AutoDMP Regression Entrypoints

Regression entrypoints are organized by benchmark first, then experiment:

```text
test/regression/
  common/
    common.sh
  iccad24/
    diff_sizing/
      run.sh
      params/
      reports/
    sta_compare/
      run.py
      params/
      reports/
    buffering_reasonable_mode/
      run.py
      params/
      reports/
```

`params/` keeps experiment-specific parameter presets or notes. `reports/`
keeps small checked-in report templates or summary notes. Large generated logs
and CSV/JSON outputs still go under the repository-level `logs/regression/`
tree and are not checked in.

Scripts use the canonical AutoDMP entrypoint, `dreamplace/Placer.py`, instead
of legacy wrappers under the repository-level `baseline/` directory.

## ICCAD24

The ICCAD24 mirror is read from `references/benchmarks/iccad24_benchmark`.
Scripts pass benchmark file paths explicitly because copied parameter JSONs may
still contain absolute paths from the original workspace mirror.

Diff-sizing:

```bash
DRY_RUN=1 ITERATIONS=2 bash AiEDA/third_party/AutoDMP/test/regression/iccad24/diff_sizing/run.sh NV_NVDLA_partition_m
bash AiEDA/third_party/AutoDMP/test/regression/iccad24/diff_sizing/run.sh NV_NVDLA_partition_m
```

STA compare:

```bash
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python AiEDA/third_party/AutoDMP/test/regression/iccad24/sta_compare/run.py --cases NV_NVDLA_partition_m --dry-run
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python AiEDA/third_party/AutoDMP/test/regression/iccad24/sta_compare/run.py --cases NV_NVDLA_partition_m --skip-autodmp
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python AiEDA/third_party/AutoDMP/test/regression/iccad24/sta_compare/run.py
```

The default OpenROAD parasitics mode is `placement`, which is faster and closer
to the current AutoDMP timing model. AutoDMP uses `Placer.py --flow-kind sta`,
which builds the timing database, writes timing artifacts, and exits without
running placement or sizing optimization. Use `--openroad-parasitics
global_routing` when a routed OpenROAD reference is required.

Buffering reasonable mode:

```bash
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python AiEDA/third_party/AutoDMP/test/regression/iccad24/buffering_reasonable_mode/run.py --dry-run
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python AiEDA/third_party/AutoDMP/test/regression/iccad24/buffering_reasonable_mode/run.py --case NV_NVDLA_partition_m
```

This gate uses `Placer.py --flow-kind buffering --buffering-mode segment` and
fails unless Python-side setup TNS improves (`delta_tns_ns > 0`).

`iccad25` and `ispd26` benchmark data may exist under `references/benchmarks`,
but they are not yet converted to AutoDMP workspace-compatible params.
