# AutoDMP: Automated DREAMPlace-based Macro Placement

Built upon the GPU-accelerated global placer [DREAMPlace](https://doi.org/10.1109/TCAD.2020.3003843) and detailed placer [ABCDPlace](https://doi.org/10.1109/TCAD.2020.2971531),
AutoDMP adds simultaneous macro and standard cell placement enhancements. (The upstream multi-objective Bayesian optimization tuner was removed in this branch; see below.)

* Simultaneous Macro and Standard Cell Placement Animations

| MemPool Group | Ariane |
| -------- | ----------- |
| ![MemPool Group](../images/mempool.gif) | ![Ariane](../images/ariane.gif) |

# Publications

* Anthony Agnesina, Puranjay Rajvanshi, Tian Yang, Geraldo Pradipta, Austin Jiao, Ben Keller, Brucek Khailany, and Haoxing Ren, 
  "**AutoDMP: Automated DREAMPlace-based Macro Placement**", 
  International Symposium on Physical Design (ISPD), Virtual Event, Mar 26-29, 2023 ([preprint](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)) ([blog](https://developer.nvidia.com/blog/autodmp-optimizes-macro-placement-for-chip-design-with-ai-and-gpus/))

# Dependency 

- [DREAMPlace](https://github.com/limbo018/DREAMPlace)
    - Commit b8f87eec1f4ddab3ad50bbd43cc5f4ccb0072892 
    - Other versions may also work, but not tested

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

# How to Build

You can build in two ways:
- Build without Docker by following the instructions of the DREAMPlace build at [README_DREAMPlace.md](README_DREAMPlace.md). The build system auto-detects PyTorch configuration via `import torch`; you can also pass `TORCH_INSTALL_PREFIX`, `TORCH_VERSION`, and `TORCH_ENABLE_CUDA` as CMake cache entries to override when `import torch` is not available during configuration.
- Use the provided Dockerfile to build an image with the required library dependencies.

# Multi-Objective Bayesian Optimization

The upstream AutoDMP tuner (`tuner/`, `hpbandster/`, `scripts/genFlow.py`) and its
MOBOHB-based parameter search were removed from this branch: placement flows are
driven directly through `dreamplace/placer_cli.py` and `dreamplace/flows/`. The NVDLA
benchmark data the tuner examples used is still available under
`test/nvdla_nangate45_51/`.

# Physical Design Flow

The physical design flow requires RTL, Python, and Tcl files from the [TILOS-MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) repository. Only the codes that we have added and modified are provided in [scripts](../scripts). 
