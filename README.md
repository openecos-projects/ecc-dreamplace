# ecc-dreamplace

`ecc-dreamplace` is an ECC-integrated placement engine based on
[DREAMPlace](https://github.com/limbo018/DREAMPlace). It keeps the
GPU/CPU analytical placement foundation from DREAMPlace and extends it for the
ECC physical-design data flow, differentiable timing analysis, timing-aware
net weighting, and ECC early-global-routing driven routability optimization.

This repository is packaged as a Python wheel for the
[ECOS Studio](https://github.com/openecos-projects/ecos-studio) silicon design
platform. The original upstream DREAMPlace README is preserved in
[README_DREAMPlace.md](README_DREAMPlace.md), and the inherited AutoDMP notes
are preserved in [README_AutoDMP.md](README_AutoDMP.md).

## What Is New Compared with DREAMPlace?

### ECC Data-Flow Integration

`ecc-dreamplace` can run placement directly from the ECC data flow instead of
requiring a standalone DREAMPlace benchmark conversion path.

- `dreamplace/Placer.py` exposes `Placer.setup_rawdb(ecc_module)` to initialize
  DREAMPlace from an ECC module.
- `dreamplace/macroPlaceDB.py` builds the Python placement database from ECC
  data via `ecc_module.pydb(...)`.
- Placement results can be written back through ECC with
  `ecc_module.write_placement_back(...)` / `ecc_module.def_save(...)`.

### PyTorch-Based Differentiable STA

The repository adds a PyTorch-based static timing analysis path that supports
both forward timing evaluation and backward timing-gradient computation.

The timing path includes:

- Steiner topology construction in `dreamplace/ops/steiner_topo/`.
- Elmore-delay net modeling in `dreamplace/ops/rc_timing/`.
- Timing-graph propagation in `dreamplace/ops/timing_propagation/`.
- Integration with the placement objective in `dreamplace/PlaceObj.py`.

With `with_sta` enabled, the placer builds timing propagation and Elmore-delay
operators during placement initialization. The timing objective computes
Steiner topology, net delay, slew/load propagation, and WNS/TNS-style timing
metrics in the PyTorch computation graph.

### Timing-Driven Placement and Net Weighting

The repository implements timing-aware controls commonly used in recent
timing-driven placement flows, including net-level and pin-to-pin weighting.
The main configuration knobs are:

| Parameter | Purpose |
| --- | --- |
| `enable_net_weighting` | Enable timing-aware net weighting during global placement. |
| `net_weighting_scheme` | Select the net-weighting scheme, currently configured for options such as `adam` and `lilith`. |
| `max_net_weight` | Cap timing-driven net weights, or use `inf` for no cap. |
| `pin2pin_net_weighting` | Enable pin-to-pin timing weighting. |
| `pin2pin_weight` | Base multiplier for pin-to-pin timing weights. |
| `timing_eval_flag` | Enable timing evaluation reporting. |
| `risa_weights` | Use RISA-style weighted smooth HPWL to improve correlation with routed/Steiner wirelength. |

Note: `timing_opt_flag` is a legacy DREAMPlace/OpenTimer flag in this fork.
It is intentionally marked as unsupported because the old OpenTimer integration
has been removed. Use the ECC-integrated STA path controlled by `with_sta`,
`differentiable_timing_obj`, and the net-weighting parameters above.

### ECC EGR-Based Routability Inflation

`ecc-dreamplace` supports routability-driven cell inflation using ECC/iRT early
global routing feedback.

- `dreamplace/ops/irt_egr/` wraps the ECC/iRT early global routing path.
- `dreamplace/PlaceObj.py` builds `irt_egr_congestion_map_op` when routability
  optimization is enabled.
- `dreamplace/ops/adjust_node_area/` inflates movable-cell areas from route and
  pin utilization maps.

Relevant configuration knobs include:

| Parameter | Purpose |
| --- | --- |
| `routability_opt_flag` | Enable routability-driven global placement. |
| `adjust_nctugr_area_flag` | Use the integrated EGR/NCTUgr-style route map for route-area adjustment. |
| `adjust_rudy_area_flag` | Use RUDY-style route utilization for route-area adjustment. |
| `route_num_bins_x`, `route_num_bins_y` | Routing-utilization grid resolution. |
| `max_route_opt_adjust_rate` | Maximum route-driven area inflation rate. |
| `route_opt_adjust_exponent` | Exponent applied to the route utilization map before inflation. |
| `route_area_adjust_stop_ratio` | Stop threshold for route-area inflation. |

## Build

### Prerequisites

- Linux x86_64
- Python 3.11 + [uv](https://docs.astral.sh/uv/)
- Optional: Nix, for entering the repository development shell before sync
- System packages:
  `cmake ninja-build build-essential pkg-config libboost-all-dev libcairo2-dev libgflags-dev libgoogle-glog-dev flex libfl-dev bison libeigen3-dev libgtest-dev`

### Dev Setup

```bash
# If Nix is available, enter the dev shell first.
nix develop

# Sync the editable development environment.
uv sync --no-build-isolation-package ecc-dreamplace --verbose
source .venv/bin/activate
```

If Nix is not available, skip `nix develop` and run the `uv sync` command in the
normal shell after installing the system packages above.

The package uses scikit-build editable rebuilds. Source edits are picked up on
the next import, and native extensions rebuild automatically when needed.

### Build Package

```bash
uv build
```

Output:

```text
dist/ecc_dreamplace-*
```

The uv build runs the package build defined by `pyproject.toml`.

## Repository Pointers

| Path | Description |
| --- | --- |
| `dreamplace/Placer.py` | Top-level placer interface and ECC raw database setup. |
| `dreamplace/macroPlaceDB.py` | ECC-backed placement database construction and write-back. |
| `dreamplace/PlaceObj.py` | Placement objective, differentiable timing integration, and routing-inflation ops. |
| `dreamplace/ops/steiner_topo/` | Steiner topology operator. |
| `dreamplace/ops/rc_timing/` | Elmore-delay and RC timing operators. |
| `dreamplace/ops/timing_propagation/` | Timing-graph propagation operator. |
| `dreamplace/ops/irt_egr/` | ECC/iRT early-global-routing congestion-map wrapper. |
| `dreamplace/ops/adjust_node_area/` | Route/pin utilization driven cell-area inflation. |
| `dreamplace/params.json` | Full parameter schema and defaults. |
| `docs/release.md` | Release workflow. |

## Release

Releases are triggered by a version-bump PR and are published as GitHub release
wheels. See [docs/release.md](docs/release.md).

## References

If you use this repository, please also cite the relevant upstream and related
works:

- Y. Lin, S. Dhar, W. Li, H. Ren, B. Khailany, and D. Z. Pan,
  "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI
  Placement," DAC 2019.
  [[NVIDIA Research](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement)]
- P. Liao, D. Guo, Z. Guo, S. Liu, Y. Lin, and B. Yu,
  "DREAMPlace 4.0: Timing-Driven Placement With Momentum-Based Net Weighting
  and Lagrangian-Based Refinement," IEEE TCAD, 2023.
  [[DOI: 10.1109/TCAD.2023.3240132](https://doi.org/10.1109/TCAD.2023.3240132)]
- Z. Guo and Y. Lin,
  "Differentiable-Timing-Driven Global Placement," DAC 2022.
  [[DOI: 10.1145/3489517.3530486](https://doi.org/10.1145/3489517.3530486)]
  [[PDF](https://guozz.cn/publication/tdpdac-22/tdpdac-22.pdf)]
- Y. Shi, S. Xu, S. Kai, X. Lin, K. Xue, M. Yuan, and C. Qian,
  "Timing-Driven Global Placement by Efficient Critical Path Extraction,"
  DATE 2025.
  [[DOI: 10.23919/DATE64628.2025.10993273](https://doi.org/10.23919/DATE64628.2025.10993273)]
  [[PDF](https://www.lamda.nju.edu.cn/qianc/DATE_25_TDP_final.pdf)]
  [[Code](https://github.com/lamda-bbo/Efficient-TDP)]
- A. Agnesina, P. Rajvanshi, T. Yang, G. Pradipta, A. Jiao, B. Keller,
  B. Khailany, and H. Ren,
  "AutoDMP: Automated DREAMPlace-based Macro Placement," ISPD 2023.
  [[NVIDIA Research](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)]
- iEDA project,
  "iEDA: An Open-Source Intelligent Physical Implementation Toolkit and
  Library," 2023.
  [[arXiv](https://arxiv.org/abs/2308.01857)]

## Contact

For questions about this ECC-integrated DREAMPlace fork, contact:

- Xueyan Zhao: <zhaoxueyan21b@ict.ac.cn>
