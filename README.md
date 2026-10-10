# ecc-dreamplace

`ecc-dreamplace` is an ECC-integrated placement engine based on
[DREAMPlace](https://github.com/limbo018/DREAMPlace). It keeps the
GPU/CPU analytical placement foundation from DREAMPlace and extends it for the
ECC physical-design data flow, differentiable timing analysis, timing-aware
net weighting, and ECC early-global-routing driven routability optimization.

Built upon the GPU-accelerated global placer [DREAMPlace](https://doi.org/10.1109/TCAD.2020.3003843) and detailed placer [ABCDPlace](https://doi.org/10.1109/TCAD.2020.2971531),
AutoDMP adds simultaneous macro and standard cell placement enhancements.

* Simultaneous Macro and Standard Cell Placement Animations

| MemPool Group | Ariane |
| -------- | ----------- |
| ![MemPool Group](images/mempool.gif) | ![Ariane](images/ariane.gif) |

# Publications

* Anthony Agnesina, Puranjay Rajvanshi, Tian Yang, Geraldo Pradipta, Austin Jiao, Ben Keller, Brucek Khailany, and Haoxing Ren, 
  "**AutoDMP: Automated DREAMPlace-based Macro Placement**", 
  International Symposium on Physical Design (ISPD), Virtual Event, Mar 26-29, 2023 ([preprint](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)) ([blog](https://developer.nvidia.com/blog/autodmp-optimizes-macro-placement-for-chip-design-with-ai-and-gpus/))

This repository is packaged as a Python wheel for the
[ECOS Studio](https://github.com/openecos-projects/ecos-studio) silicon design
platform. Upstream documentation is available in the
[DREAMPlace repository](https://github.com/limbo018/DREAMPlace) and
[AutoDMP repository](https://github.com/NVlabs/AutoDMP).

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

Default CMake and Python package builds use the ECC backend and do not require
OpenROAD. The OpenROAD submodule and its superbuild have been removed. The
optional OpenROAD adapter can be built against an existing external installation
with `-DDREAMPLACE_USE_EXTERNAL_OPENROAD=ON` and
`-DOPENROAD_EXTERNAL_INSTALL_PREFIX=/path/to/openroad/install`.

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

ECC timing now requires the schema-v2 native snapshot from the matching
`ecc-tools` revision. Rebuild/reinstall both packages together. The consumer
validates per-edge qualification for endpoints and setup/recovery checks at
initial import and after sizing/buffer refresh; unconstrained endpoints retain
physical slew/cap coverage but do not contribute WNS/TNS. Unsupported max path
exceptions and clock groups fail explicitly. This placement model does not
implement full latch borrowing, hold or CPPR.

The sizing rounds inside placement timing-optimization windows have one
coefficient policy parameter, `timing_opt_coefficients`. The default fixes
WNS/TNS/cap/slew at 500/5/1/1 with outer timing weight 1. The following
configuration matches the defaults:

```json
"timing_opt_coefficients": {
  "mode": "fixed", "wns": 500, "tns": 5, "slew": 1, "cap": 1
}
```

Fixed mode requires all four finite, nonnegative values and uses outer
`timing_grad_balance_weight` 1.0, making those values the effective weights.
Inherit mode retains placement's live coefficients and outer weight. The
original placement coefficients and outer weight are restored when sizing
exits, including on failure. This policy
applies to GP sizing windows; standalone `diff_sizing` uses its own configuration.
Each window's `sizing.coefficients` report records the mode and effective values.
The fixed preset may remain in the object when switching `mode` to `inherit`;
inherit mode always uses the live placement values.

`timing_coeff_growth_factor` controls the WNS/TNS coefficient multiplier at
each GP density-weight update. The default `1.0` freezes the coefficients;
`1.01` grows them, and values between zero and one decay them.
The multiplier must be positive and finite. Slew/cap weights and the outer norm
weight retain their own settings. Inherit-mode sizing windows read the resulting
live coefficients; fixed-mode windows use their configured values. Standalone
`size_only` sizing, including `diff_sizing` S50, skips this growth schedule.

`timing_grad_balance_target_ratio` defaults to 0.2 for direct-loss placement.
At the first active timing step, the timing and wirelength gradient L1 norms
determine the outer timing weight, which is then reused. Set the ratio to 0.0
to disable balancing and retain weight 1.0. It is a gradient-ratio target,
not a multiplier on WNS/TNS coefficients. Standalone `size_only` sizing does
not initialize coordinate gradient balancing. ECC exposes the ratio through
`place.timing_grad_balance_target_ratio`; explicit values override the default.

`timing_aggregation_mode` selects AAT/RAT propagation aggregation: `smooth`
(default) uses LSE, while `hard` uses max/min. LSE uses the positive temperature
`timing_aggregation_tau_ps` (default 2.0 ps). Smaller temperatures approach
hard max/min. ECC exposes both through `place.*` parameters. Endpoint WNS
still uses hard min, and TNS still sums negative slacks.

`overflow_reference_mode` selects the area used to normalize overflow. The
default `initial` freezes the GP-entry movable area. `ordinary` preserves the
existing PR behavior: use the published native-plus-virtual movable area when
available, otherwise use `placedb.total_movable_node_area`. Both exclude
fillers. `ordinary` follows the existing publication points; it does not
recompute all area state on every iteration. The normalized overflow also
feeds gamma and overflow-based scheduling.

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
| `adjust_nctugr_area_flag` | Use the ECC/iRT EGR route map for route-area adjustment. The legacy parameter name is retained for configuration compatibility. |
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
  `cmake ninja-build build-essential pkg-config libcairo2-dev libgflags-dev libgoogle-glog-dev flex libfl-dev bison libeigen3-dev libgtest-dev`
- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

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

# Physical Design Flow

The physical design flow requires RTL, Python, and Tcl files from the [TILOS-MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) repository. Only the codes that we have added and modified are provided in [scripts](scripts). 

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
