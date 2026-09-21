#!/usr/bin/env bash
# Repeated medium-fixture gate for cpu_pr vs cpu_pr_mt on the darjeeling
# placed DEF. Runs one discarded warm-up, then REPEATS fresh processes per
# configuration under a fixed CPU affinity and single-threaded BLAS env, and
# finally aggregates median/p95 into gate_summary.json.
#
# Usage: bash scripts/benchmark_gpugr_medium_gate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMPLACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECC_ROOT="$(cd "${DREAMPLACE_ROOT}/../../.." && pwd)"
PYTHON_BIN="${ECC_VENV_PYTHON:-${ECC_ROOT}/.venv/bin/python}"

FIXTURE_CONFIG="${MEDIUM_CONFIG:-/nfs/share/home/zhaoxueyan/dataset_gj_2609/darjeeling/config/dreamplace_ecc.json}"
FIXTURE_DEF="${MEDIUM_DEF:-/nfs/share/home/zhaoxueyan/dataset_gj_2609/darjeeling/place_dreamplace/data/pl/gpugr_l_shape/gpugr_8nj18xkr/darjeeling_gpugr.def}"
OUT_ROOT="${MEDIUM_OUT_ROOT:-/nfs/share/home/zhaoxueyan/dataset_gj_2609/darjeeling_cpu_pr_mt_medium_gate_20260921}"
DESIGN_NAME="darjeeling"
ROUTE_XSIZE=512
ROUTE_YSIZE=512
BOTTOM_LAYER=MET2
TOP_LAYER=MET5
REPEATS="${REPEATS:-3}"
AFFINITY="${AFFINITY:-0-63}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one() {
    local backend="$1" threads="$2" tag="$3"
    echo "== gate run: ${tag} (backend=${backend} threads=${threads})"
    taskset -c "${AFFINITY}" "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_gpugr_cpu_backend.py" \
        --config "${FIXTURE_CONFIG}" \
        --input-def "${FIXTURE_DEF}" \
        --output-dir "${OUT_ROOT}/${tag}/results" \
        --design-name "${DESIGN_NAME}" \
        --backend "${backend}" \
        --threads "${threads}" \
        --route-xsize "${ROUTE_XSIZE}" \
        --route-ysize "${ROUTE_YSIZE}" \
        --bottom-routing-layer "${BOTTOM_LAYER}" \
        --top-routing-layer "${TOP_LAYER}" \
        --benchmark "${DESIGN_NAME}_${backend}_t${threads}_${tag}"
}

# One discarded warm-up so page cache / FLUTE LUT state is not attributed to
# the first measured process.
run_one cpu_pr_mt 8 warmup

for spec in "cpu_pr 1" "cpu_pr_mt 1" "cpu_pr_mt 2" "cpu_pr_mt 8" "cpu_pr_mt 64"; do
    set -- ${spec}
    for rep in $(seq 1 "${REPEATS}"); do
        run_one "$1" "$2" "$1_t$2_rep${rep}"
    done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_gpugr_benchmark_gate.py" \
    --root "${OUT_ROOT}" \
    --output "${OUT_ROOT}/gate_summary.json"
echo "== gate summary: ${OUT_ROOT}/gate_summary.json"
