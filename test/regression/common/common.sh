#!/usr/bin/env bash

set -euo pipefail

REGRESSION_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_DIR="$(cd "${REGRESSION_COMMON_DIR}/.." && pwd)"
AUTODMP_ROOT="$(cd "${REGRESSION_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${REGRESSION_DIR}/../../../../.." && pwd)"
BENCHMARK_ROOT="${REPO_ROOT}/references/benchmarks"
PLACER_PY="${AUTODMP_ROOT}/dreamplace/Placer.py"
PYTHON_BIN="${PYTHON_BIN:-/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python}"
REGRESSION_LOG_ROOT="${REGRESSION_LOG_ROOT:-${REPO_ROOT}/logs/regression}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "required directory not found: $path"
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}
