#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../common/common.sh"

usage() {
  cat <<'EOF'
Usage:
  run.sh <case>

Environment:
  PYTHON_BIN=/path/to/python      Python interpreter. Defaults to PlaceOPT.
  DRY_RUN=1                       Print effective Placer config and exit.
  ITERATIONS=N                    Override sizing iterations. Default is 50.
  GPU_ID=N                        Select CUDA_VISIBLE_DEVICES for this run.
  REGRESSION_LOG_ROOT=/path       Log root. Defaults to <repo>/logs/regression.

Examples:
  DRY_RUN=1 ITERATIONS=2 bash test/regression/iccad24/diff_sizing/run.sh NV_NVDLA_partition_m
  bash test/regression/iccad24/diff_sizing/run.sh aes_256
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

benchmark="iccad24"
case_name="$1"
iterations="${ITERATIONS:-50}"

iccad24_root="${BENCHMARK_ROOT}/iccad24_benchmark"
case_dir="${iccad24_root}/design/${case_name}"
workspace_dir="${case_dir}/workspace"
params_json="${workspace_dir}/config/dreamplace_config/param.json"
asap7_dir="${iccad24_root}/ASAP7"

require_dir "$case_dir"
require_dir "$workspace_dir"
require_file "$params_json"
require_file "$case_dir/${case_name}.def"
require_file "$case_dir/${case_name}.v"
require_file "$case_dir/${case_name}.sdc"
require_file "$asap7_dir/setRC.tcl"
require_file "$asap7_dir/lef/asap7_tech_1x_201209.lef"
require_file "$asap7_dir/lef/asap7sc7p5t_27_R_1x_201211.lef"

log_dir="${REGRESSION_LOG_ROOT}/${benchmark}/diff_sizing/${case_name}/$(timestamp)"
mkdir -p "$log_dir"

cmd=(
  "$PYTHON_BIN"
  "$PLACER_PY"
  "$params_json"
  --flow-kind sizing
  --place-io-engine openroad
  --workspace "$workspace_dir"
  --result-dir "$log_dir/result"
  --base-design-name "$case_name"
  --output-def "$log_dir/result/${case_name}_diff_sizing_final.def"
  --iterations "$iterations"
  --def-input "$case_dir/${case_name}.def"
  --verilog-input "$case_dir/${case_name}.v"
  # Canonical diff-sizing regression uses the official ICCAD24 original SDC.
  # Optimizer-compatible SDCs are diagnostic variants and must not be selected here.
  --sdc "$case_dir/${case_name}.sdc"
  --rc-tcl "$asap7_dir/setRC.tcl"
  --tech-lef "$asap7_dir/lef/asap7_tech_1x_201209.lef"
  --lef "$asap7_dir/lef/asap7sc7p5t_27_R_1x_201211.lef"
  --lef "$asap7_dir/lef/sram_asap7_16x256_1rw.lef"
  --lef "$asap7_dir/lef/sram_asap7_32x256_1rw.lef"
  --lef "$asap7_dir/lef/sram_asap7_64x256_1rw.lef"
  --lef "$asap7_dir/lef/sram_asap7_64x64_1rw.lef"
  --lib "$asap7_dir/lib/asap7sc7p5t_AO_RVT_FF_nldm_201020.lib"
  --lib "$asap7_dir/lib/asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib"
  --lib "$asap7_dir/lib/asap7sc7p5t_OA_RVT_FF_nldm_201020.lib"
  --lib "$asap7_dir/lib/asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib"
  --lib "$asap7_dir/lib/asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib"
  --lib "$asap7_dir/lib/sram_asap7_16x256_1rw.lib"
  --lib "$asap7_dir/lib/sram_asap7_32x256_1rw.lib"
  --lib "$asap7_dir/lib/sram_asap7_64x256_1rw.lib"
  --lib "$asap7_dir/lib/sram_asap7_64x64_1rw.lib"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run-config)
fi

{
  echo "benchmark: $benchmark"
  echo "case: $case_name"
  echo "log_dir: $log_dir"
  echo "command:"
  printf ' %q' "${cmd[@]}"
  echo
} | tee "$log_dir/command.txt"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

"${cmd[@]}" 2>&1 | tee "$log_dir/run.log"
