"""Backend-neutral GPUGR request and result contracts.

The runtime intentionally keeps dictionaries at the public boundary because
older DreamPlace callers consume them directly.  These aliases document the
stable shape without forcing a broad migration to dataclass objects.
"""

from typing import Any, Mapping, Protocol, Sequence, TypedDict

import torch


class GPUGRRequest(TypedDict, total=False):
    input_def: str
    out_dir: str
    design_name: str
    design_identity: str
    benchmark: str
    gpu: int
    threads: int
    route_xsize: int
    route_ysize: int
    rrr_iters: int
    skip_m1_route: bool
    include_route_entries: bool
    include_topology_pack: bool
    include_l_shape_topology_pack: bool
    backend: str
    bottom_routing_layer: str
    top_routing_layer: str
    require_clean_source: bool


class GPUGRResult(TypedDict, total=False):
    metrics: Mapping[str, Any]
    native_stats: Mapping[str, Any]
    maps: Mapping[str, torch.Tensor]
    route_entries: Sequence[Mapping[str, Any]]
    artifact_paths: Mapping[str, str]
    same_net_topology_cache: Mapping[str, Any]
    same_net_topology_stats: Mapping[str, Any]
    l_shape_topology_pack: Mapping[str, Any]


class GPUGRBackend(Protocol):
    """Minimal operator surface shared by the CUDA and CPU pattern backends."""

    def run_gpugr(self, **kwargs: Any) -> GPUGRResult:
        ...
