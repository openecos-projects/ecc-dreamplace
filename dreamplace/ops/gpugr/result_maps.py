"""Shared congestion-map normalization and validation helpers."""

import math

import torch


def aggregate_layer_maps(
    demand_map: torch.Tensor,
    capacity_map: torch.Tensor,
    layer_ids: torch.Tensor,
) -> torch.Tensor:
    """Aggregate selected routing layers using total demand/total capacity."""

    if layer_ids.numel() == 0:
        return torch.zeros_like(demand_map[0])
    demand = demand_map.index_select(0, layer_ids).sum(dim=0)
    capacity = capacity_map.index_select(0, layer_ids).sum(dim=0)
    floor = torch.finfo(capacity.dtype).eps
    return torch.where(
        capacity > floor,
        demand / capacity.clamp(min=floor),
        torch.zeros_like(demand),
    )


def summarize_congestion_tensor(tensor: torch.Tensor, overflow_threshold: float) -> dict:
    """Return JSON-safe summary statistics for one congestion tensor."""

    flat = tensor.detach().reshape(-1).to(dtype=torch.float32)
    if flat.numel() <= 0:
        return {
            "max": 0.0,
            "mean": 0.0,
            "top1pct_mean": 0.0,
            "overflow_bin_ratio": 0.0,
        }
    topk = max(1, int(math.ceil(flat.numel() * 0.01)))
    top_vals = torch.topk(flat, k=topk).values
    return {
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "top1pct_mean": float(top_vals.mean().item()),
        "overflow_bin_ratio": float(
            (flat > float(overflow_threshold)).to(dtype=torch.float32).mean().item()
        ),
    }


def validate_map_planes(maps: dict, expected_layers: int = None) -> None:
    """Reject missing, malformed, or non-finite normalized map planes."""

    required = (
        "dmd_map",
        "wire_demand_map",
        "via_demand_map",
        "fix_usage_map",
        "mov_usage_map",
        "capacity_map",
    )
    missing = [name for name in required if name not in maps]
    if missing:
        raise RuntimeError("GPUGR result is missing map planes: " + ", ".join(missing))

    reference = maps[required[0]]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
        raise RuntimeError("GPUGR demand map must be a 3-D tensor")
    if expected_layers is not None and reference.shape[0] != int(expected_layers):
        raise RuntimeError(
            f"GPUGR demand map has {reference.shape[0]} layers; expected {int(expected_layers)}"
        )
    for name in required:
        value = maps[name]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise RuntimeError(f"GPUGR map plane {name} does not match demand-map shape")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"GPUGR map plane {name} contains NaN or Inf")
        if (value < 0).any():
            raise RuntimeError(f"GPUGR map plane {name} contains negative values")


def validate_map_contract(maps: dict, expected_layers: int = None) -> None:
    """Validate both layer-indexed and aggregated congestion map planes.

    The legacy validator intentionally checks only the six mandatory 3-D
    planes because CUDA callers expose a wider, historically inconsistent map
    vocabulary.  This validator holds every backend to a stricter result
    contract: every exported tensor must be finite and non-negative,
    ``cg_map_*`` aggregates are 2-D, and all other map planes retain the
    layer dimension.  Keeping this check explicit prevents accidental
    broadcasting from turning an orientation or layer projection error into
    a plausible-looking congestion result.
    """

    validate_map_planes(maps, expected_layers=expected_layers)
    reference = maps["dmd_map"]
    layer_shape = tuple(reference.shape)
    aggregate_shape = tuple(reference.shape[1:])
    for name, value in maps.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"GPUGR map plane {name} is not a tensor")
        expected_shape = aggregate_shape if name.startswith("cg_map_") else layer_shape
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"GPUGR map plane {name} has shape {tuple(value.shape)}; "
                f"expected {expected_shape}"
            )
        if value.ndim not in (2, 3):
            raise RuntimeError(f"GPUGR map plane {name} must be 2-D or 3-D")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"GPUGR map plane {name} contains NaN or Inf")
        if (value < 0).any():
            raise RuntimeError(f"GPUGR map plane {name} contains negative values")
