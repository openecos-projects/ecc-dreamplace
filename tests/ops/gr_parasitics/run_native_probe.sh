#!/usr/bin/env bash
# Source-build qualification only; never install into a frozen runtime.
set -euo pipefail
task_gr_script=$(readlink -f "${BASH_SOURCE[0]}")
task_gr_dreamplace=$(cd "$(dirname "$task_gr_script")/../../.." && pwd)
task_gr_ecc=$(cd "$task_gr_dreamplace/../../.." && pwd)
if [[ -z "${IN_NIX_SHELL:-}" ]]; then
    exec nix develop "$task_gr_ecc" --command bash "$task_gr_script" "$@"
fi
task_gr_xplace="$task_gr_dreamplace/thirdparty/xplace"
task_gr_build="$task_gr_xplace/build_cpu_pr"
task_gr_libc=$(<"$NIX_CC/nix-support/orig-libc")
task_gr_stdlib=$(dirname "$(g++ -print-file-name=libstdc++.so.6)")
task_gr_library_path="$task_gr_libc/lib:$task_gr_stdlib"
task_gr_library_path+=":$task_gr_build/cpp_to_py/common:$task_gr_build/thirdparty/flute"
task_gr_library_path+=":/usr/local/lib:/usr/lib/x86_64-linux-gnu"
export PYTHONPATH="$task_gr_ecc/.venv/lib/python3.11/site-packages"
exec "$task_gr_libc/lib/ld-linux-x86-64.so.2" --library-path "$task_gr_library_path" \
    "$task_gr_ecc/.venv/bin/python" -S \
    "$(dirname "$task_gr_script")/native_route_probe.py" --xplace "$task_gr_xplace" "$@"
