# ecc-dreamplace

ECC-branded DREAMPlace placement engine

## Build

### Prerequisites

- Linux x86_64
- Bazel 8+
- Python 3.11 + [uv](https://docs.astral.sh/uv/)
- System packages: `cmake ninja-build build-essential pkg-config libboost-all-dev libcairo2-dev libgflags-dev libgoogle-glog-dev flex libfl-dev bison libeigen3-dev libgtest-dev`

### Dev Setup

```bash
# Setup Python environment
uv sync --frozen --all-groups --python 3.11
source .venv/bin/activate

# Build and install ecc-dreamplace
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=your_install_path \
  -DPYTHON_EXECUTABLE=$(which python)
make -j`nproc`
make install
```

### Build Wheel

ecc-dreamplace is packaged as a Python wheel for the [ECOS Studio](https://github.com/openecos-projects/ecos-studio) silicon design platform.

```bash
bazel run //:build_dreamplace_wheel
```

Output: `dist/wheel/repaired/ecc_dreamplace-*.whl` (CMake compile -> auditwheel repair -> smoke test).

## Release

See [docs/release.md](docs/release.md).
