#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
openroad_src_dir="${OPENROAD_SOURCE_DIR:-${script_dir}/OpenROAD}"
build_dir_name="${OPENROAD_BUILD_DIR:-build-shared-abi1}"

if [[ "${build_dir_name}" = /* ]]; then
  build_dir="${build_dir_name}"
else
  build_dir="${openroad_src_dir}/${build_dir_name}"
fi

build_type="${OPENROAD_BUILD_TYPE:-Release}"
thread_count="${OPENROAD_BUILD_JOBS:-$(nproc)}"
compiler_c="${OPENROAD_CC:-/usr/bin/gcc-10}"
compiler_cxx="${OPENROAD_CXX:-/usr/bin/g++-10}"
python_executable="${OPENROAD_PYTHON_EXECUTABLE:-}"
openroad_build_shared="${OPENROAD_BUILD_SHARED:-ON}"
if [[ -n "${OPENROAD_INSTALL_PREFIX:-}" ]]; then
  install_prefix="${OPENROAD_INSTALL_PREFIX}"
elif [[ "${openroad_build_shared}" == "ON" ]]; then
  install_prefix="${openroad_src_dir}/install-shared-abi1"
else
  install_prefix=""
fi
run_install="${OPENROAD_RUN_INSTALL:-AUTO}"

extra_c_flags="${OPENROAD_EXTRA_C_FLAGS:--D_GLIBCXX_USE_CXX11_ABI=1}"
extra_cxx_flags="${OPENROAD_EXTRA_CXX_FLAGS:--D_GLIBCXX_USE_CXX11_ABI=1}"
extra_cmake_args="${OPENROAD_EXTRA_CMAKE_ARGS:-}"
cmake_generator="${OPENROAD_CMAKE_GENERATOR:-}"

if [[ ! -d "${openroad_src_dir}" ]]; then
  echo "OpenROAD source dir not found: ${openroad_src_dir}" >&2
  exit 1
fi

mkdir -p "${build_dir}"

cmake_args=(
  -S "${openroad_src_dir}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE="${build_type}"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DOPENROAD_BUILD_SHARED="${openroad_build_shared}"
  -DCMAKE_C_COMPILER="${compiler_c}"
  -DCMAKE_CXX_COMPILER="${compiler_cxx}"
  -DCMAKE_C_FLAGS="${extra_c_flags}"
  -DCMAKE_CXX_FLAGS="${extra_cxx_flags}"
  -DENABLE_TESTS=OFF
  -DBUILD_PYTHON=OFF
  -DUSE_SYSTEM_BOOST=ON
)

if [[ -n "${cmake_generator}" ]]; then
  cmake_args+=(-G "${cmake_generator}")
fi

if [[ -n "${python_executable}" ]]; then
  cmake_args+=(-DPython3_EXECUTABLE="${python_executable}")
fi

if [[ -n "${install_prefix}" ]]; then
  cmake_args+=(-DCMAKE_INSTALL_PREFIX="${install_prefix}")
fi

if [[ -n "${extra_cmake_args}" ]]; then
  # shellcheck disable=SC2206
  extra_cmake_args_array=(${extra_cmake_args})
  cmake_args+=("${extra_cmake_args_array[@]}")
fi

echo "[build_openroad] source: ${openroad_src_dir}"
echo "[build_openroad] build : ${build_dir}"
echo "[build_openroad] type  : ${build_type}"
echo "[build_openroad] jobs  : ${thread_count}"
echo "[build_openroad] shared: ${openroad_build_shared}"
echo "[build_openroad] C ABI : ${extra_c_flags}"
echo "[build_openroad] CXX ABI: ${extra_cxx_flags}"
if [[ -n "${install_prefix}" ]]; then
  echo "[build_openroad] install prefix: ${install_prefix}"
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target openroad -j"${thread_count}"

if [[ "${run_install}" == "ON" || ( "${run_install}" == "AUTO" && -n "${install_prefix}" ) ]]; then
  cmake --install "${build_dir}"
fi

echo "[build_openroad] done"
if [[ "${openroad_build_shared}" == "ON" ]]; then
  echo "[build_openroad] library: ${build_dir}/src/libopenroad.so"
else
  echo "[build_openroad] library: ${build_dir}/src/libopenroad.a"
fi
