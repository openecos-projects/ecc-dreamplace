# ECC Vendored Limbo Subset

This directory contains the source subset of Limbo required by ECC's
DreamPlace macro legalization implementation.

- Upstream repository: <https://github.com/limbo018/Limbo.git>
- Upstream revision: `4cf5d2cd407570b0c39c6fb766b0740423b864bf`
- Limbo license: [LICENSE](LICENSE)
- Retained Limbo components: `DualMinCostFlow`, its model and preprocessor
  headers, and the LP parser still inherited by `Solvers.h`
- Retained third-party component: [LEMON](https://lemon.cs.elte.hu/) 1.3.1 core
- LEMON license: [limbo/thirdparty/lemon/LICENSE](limbo/thirdparty/lemon/LICENSE)

Standalone LEF, DEF, Verilog, Bookshelf, and GDS parsing; program options;
OpenBLAS/Csdp; and the upstream documentation and test trees are intentionally
omitted.
ECC supplies the placement database through iEDA, and final GDS generation is
owned by the downstream iEDA flow.
