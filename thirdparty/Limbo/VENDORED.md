# Vendored Limbo Subset

This directory is an in-tree copy of the Limbo sources DreamPlace needs. The
`thirdparty/Limbo` submodule was removed because its recorded commit
(`d0fa19cb0005add2fdbfe8a2f093bd80c966600c`) does not exist in the upstream
repository and clones could not fetch it. This mirrors how the ecc line vendors
Limbo instead of using a submodule.

- Upstream repository: <https://github.com/limbo018/Limbo.git>
- Upstream revision: `6c115bfb1d4dca3a92044d505ee00456aac2bd45` (master)
- Limbo license: [LICENSE](LICENSE)

## Retained

- `limbo/geometry`, `limbo/math`, `limbo/preprocessor`, `limbo/string` -
  headers used by `dreamplace/ops/place_io`, `draw_place` and the timing ops.
- `limbo/parsers` (lef, def, verilog, bookshelf, gdsii, lp) and its third-party
  support (`lefdef`, `gzstream`, `flex`, `libdivide`) - the standalone
  LEF/DEF/Verilog/Bookshelf/GDS parsing this line still performs.
- `limbo/programoptions` - command line option handling used by those drivers.
- `limbo/solvers` with the LEMON core (`limbo/thirdparty/lemon/lemon`,
  `limbo/thirdparty/lemon/cmake`) - `DualMinCostFlow` for macro legalization.
- `limbo/bibtex`, `limbo/containers`, `limbo/makeutils`, `cmake/` and the root
  `CMakeLists.txt` - required unconditionally by Limbo's own build.

## Omitted

- `limbo/algorithms` and `limbo/thirdparty/dlx` - not referenced by any vendored
  header; `thirdparty/CMakeLists.txt` therefore configures `ALGORITHMS=OFF`.
- `limbo/thirdparty/Csdp` and `limbo/thirdparty/OpenBLAS` - only built when
  `OPENBLAS=ON`, which DreamPlace configures off.
- `thirdparty/lemon/{doc,demo,test,tools,scripts,contrib}`, `test/` and `docs/` -
  never configured when Limbo is built as a subproject (its top-level
  `CMakeLists.txt` only adds them when it is the root project) or when
  `ENABLE_TEST`/`GENERATE_DOCS` are off, as DreamPlace sets them.

Compared with the ecc line's vendored copy this subset is larger: it also keeps
the parsing, geometry, program-option and string components that this line's
`place_io` and `draw_place` C++ still compile against. The ecc copy omits them
because the ecc placement database comes from iEDA instead.

DreamPlace requires bison >= 3.3, flex, zlib and Boost (graph, regex) to build
this tree, as upstream Limbo does.