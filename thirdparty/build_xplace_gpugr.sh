#!/usr/bin/env bash
# Build the Xplace gpugr native extensions against the ECC venv torch.
#
# Rebuild boundary: any torch / Python ABI change in the ECC venv (torch
# version, Python version) requires re-running this script. The produced
# extensions hard-link the venv's libtorch via rpath; verify with ldd.
#
# What it does:
#   1. Aligns the nested thirdparty/pybind11 submodule of xplace to v3.0.1,
#      matching the pybind11 bundled with torch 2.11 (xplace pins 2.14-dev,
#      whose pybind11 internals are ABI-incompatible with torch 2.11).
#      The vendored Xplace commit records this nested submodule version.
#   2. Configures an out-of-source build with the ECC venv Python and the venv
#      torch CMAKE_PREFIX_PATH. Set XPLACE_ENABLE_CUDA=OFF for a CPU-only build;
#      the default follows whether the active torch build includes CUDA. Set
#      XPLACE_INSTALL=OFF to validate an isolated build without replacing the
#      active runtime extensions in cpp_to_py/cpybin.
#   3. Builds the extension set supported by the active torch. GPUGR only needs
#      gpugr, io_parser and flute_cpp, but xplace's Python side
#      (src/__init__.py) imports every extension at package import time, so a
#      partial build cannot satisfy `from src import Flute`. A CUDA-enabled
#      build provides both cuda and cpu_pr; a CPU torch build provides cpu_pr.
#   4. Installs into xplace's runtime lib dir cpp_to_py/cpybin (the location
#      upstream cpp_to_py/__init__.py imports from).
#
# Build artifacts (thirdparty/xplace-build/, cpp_to_py/cpybin/) are git-ignored
# and must never be committed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMPLACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ECC_ROOT="$(cd "${DREAMPLACE_ROOT}/../../.." && pwd)"
XPLACE_ROOT="${DREAMPLACE_ROOT}/thirdparty/xplace"
CPYBIN_DIR="${XPLACE_RUNTIME_DIR:-${XPLACE_ROOT}/cpp_to_py/cpybin}"

# Pinned toolchain: ECC venv interpreter (torch 2.11, Python 3.11).
PYTHON_BIN="${ECC_VENV_PYTHON:-${ECC_ROOT}/.venv/bin/python}"
PYBIND11_REQUIRED_TAG="v3.0.1"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "error: ECC venv python not found at ${PYTHON_BIN}" >&2
    echo "       set ECC_VENV_PYTHON to override" >&2
    exit 1
fi

TORCH_PREFIX="$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
TORCH_CUDA_ENABLED="$("${PYTHON_BIN}" -c 'import torch; print("ON" if torch.backends.cuda.is_built() else "OFF")')"
XPLACE_CUDA_REQUEST="${XPLACE_ENABLE_CUDA:-AUTO}"
XPLACE_CUDA_REQUEST="${XPLACE_CUDA_REQUEST^^}"
XPLACE_INSTALL_REQUEST="${XPLACE_INSTALL:-ON}"
XPLACE_INSTALL_REQUEST="${XPLACE_INSTALL_REQUEST^^}"
case "${XPLACE_CUDA_REQUEST}" in
    AUTO)
        XPLACE_CUDA_ENABLED="${TORCH_CUDA_ENABLED}"
        ;;
    ON)
        if [[ "${TORCH_CUDA_ENABLED}" != "ON" ]]; then
            echo "error: XPLACE_ENABLE_CUDA=ON requires a CUDA-enabled torch build" >&2
            exit 1
        fi
        XPLACE_CUDA_ENABLED="ON"
        ;;
    OFF)
        XPLACE_CUDA_ENABLED="OFF"
        ;;
    *)
        echo "error: XPLACE_ENABLE_CUDA must be AUTO, ON, or OFF; got ${XPLACE_CUDA_REQUEST}" >&2
        exit 1
        ;;
esac
case "${XPLACE_INSTALL_REQUEST}" in
    ON|OFF) ;;
    *)
        echo "error: XPLACE_INSTALL must be ON or OFF; got ${XPLACE_INSTALL_REQUEST}" >&2
        exit 1
        ;;
esac
if [[ "${XPLACE_CUDA_ENABLED}" == "ON" ]]; then
    BUILD_DIR="${XPLACE_BUILD_DIR:-${DREAMPLACE_ROOT}/thirdparty/xplace-build}"
else
    BUILD_DIR="${XPLACE_BUILD_DIR:-${XPLACE_ROOT}/build_cpu_pr}"
fi
PYTHON_EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
# xplace's cmake passes ${PYTHON_INCLUDE_DIRS} into the static libs (ggr,
# xplace_common); pybind11 3.x no longer sets that legacy variable, so pin it
# explicitly to the venv interpreter's headers.
PYTHON_INCLUDE_DIR="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
echo "== python:  ${PYTHON_BIN}"
echo "== torch:   $("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)') (prefix ${TORCH_PREFIX})"
echo "== torch cuda:  ${TORCH_CUDA_ENABLED}"
echo "== xplace cuda: ${XPLACE_CUDA_ENABLED}"
echo "== install:     ${XPLACE_INSTALL_REQUEST}"
echo "== build dir:   ${BUILD_DIR}"

# 1. pybind11 alignment with the venv torch.
PYBIND11_DIR="${XPLACE_ROOT}/thirdparty/pybind11"
if [[ ! -f "${PYBIND11_DIR}/include/pybind11/detail/common.h" ]]; then
    git -C "${XPLACE_ROOT}" submodule update --init thirdparty/pybind11
fi
PYBIND11_VERSION_HEADER="${PYBIND11_DIR}/include/pybind11/detail/common.h"
pybind11_major="$(sed -n 's/#define PYBIND11_VERSION_MAJOR //p' "${PYBIND11_VERSION_HEADER}")"
pybind11_minor="$(sed -n 's/#define PYBIND11_VERSION_MINOR //p' "${PYBIND11_VERSION_HEADER}")"
pybind11_patch="$(sed -n 's/#define PYBIND11_VERSION_PATCH //p' "${PYBIND11_VERSION_HEADER}")"
current_pybind11="${pybind11_major}.${pybind11_minor}.${pybind11_patch}"
if [[ "${current_pybind11}" != "3.0.1" ]]; then
    echo "== aligning pybind11 (${current_pybind11} -> ${PYBIND11_REQUIRED_TAG}) to match torch-bundled pybind11 3.0.1"
    git -C "${PYBIND11_DIR}" fetch --depth 1 origin tag "${PYBIND11_REQUIRED_TAG}"
    git -C "${PYBIND11_DIR}" checkout "${PYBIND11_REQUIRED_TAG}"
fi

# 2. Configure.
cmake_args=(
    -S "${XPLACE_ROOT}" -B "${BUILD_DIR}" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DPYTHON_EXECUTABLE="${PYTHON_BIN}"
    -DPYTHON_INCLUDE_DIRS="${PYTHON_INCLUDE_DIR}"
    -DPython_EXECUTABLE="${PYTHON_BIN}"
    -DPython3_EXECUTABLE="${PYTHON_BIN}"
    -DCMAKE_PREFIX_PATH="${TORCH_PREFIX}"
    -DCMAKE_CXX_ABI=1
    -DXPLACE_ENABLE_CUDA="${XPLACE_CUDA_ENABLED}"
    -DXPLACE_LIB_DIR="${CPYBIN_DIR}"
)
if [[ "${XPLACE_CUDA_ENABLED}" == "ON" ]]; then
    cmake_args+=(
        -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda-12.8/bin/nvcc}"
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
    )
fi
cmake "${cmake_args[@]}"

# 3. Build all extension targets (see note 3 above).
cmake --build "${BUILD_DIR}" -j "${JOBS}"

if [[ "${XPLACE_INSTALL_REQUEST}" == "ON" ]]; then
    # 4. Install into cpp_to_py/cpybin.
    cmake --install "${BUILD_DIR}" --config Release

    if [[ "${XPLACE_CUDA_ENABLED}" == "OFF" ]]; then
        rm -f \
            "${CPYBIN_DIR}/density_map_cuda${PYTHON_EXT_SUFFIX}" \
            "${CPYBIN_DIR}/dct_cuda${PYTHON_EXT_SUFFIX}" \
            "${CPYBIN_DIR}/gpudp${PYTHON_EXT_SUFFIX}" \
            "${CPYBIN_DIR}/hpwl_cuda${PYTHON_EXT_SUFFIX}" \
            "${CPYBIN_DIR}/wa_wirelength_hpwl_cuda${PYTHON_EXT_SUFFIX}" \
            "${CPYBIN_DIR}/wirelength_timing_cuda${PYTHON_EXT_SUFFIX}"
    fi
    # Drop any stale legacy cugr extension left by older builds.
    rm -f "${CPYBIN_DIR}/cugr${PYTHON_EXT_SUFFIX}"

    echo "== installed extensions:"
    ls -1 "${CPYBIN_DIR}/"
else
    echo "== build complete; install skipped"
fi
