"""One native validation/packing boundary for immutable GR RC trees.

Native indices are int32. Pin scatter/gather indices are int64. RC values use
microns, ohms and picofarads; the existing RC solver consequently returns ps.
The route graph is never installed in the placement or routability topology.
"""

from dataclasses import dataclass

import numpy as np
import torch

from . import gr_parasitics_cpp
from .rc_parameters import RCParameters


@dataclass(frozen=True)
class PinMapping:
    pin_names: dict[str, int]
    net_names: dict[str, int]
    pin_net: torch.Tensor
    driver_pin: torch.Tensor
    eligible_nets: torch.Tensor


@dataclass(frozen=True)
class RouteIdentity:
    input_sha256: str
    geometry_sha256: str
    generation: int


@dataclass(frozen=True)
class RouteSnapshot:
    identity: RouteIdentity
    rc_parameters: RCParameters
    pin_to_vertex: torch.Tensor
    pin_grid_vertex: torch.Tensor
    valid_pin_mask: torch.Tensor
    valid_pin_ids: torch.Tensor
    valid_vertex_ids: torch.Tensor
    vertex_net: torch.Tensor
    vertex_layer: torch.Tensor
    vertex_xy_um: torch.Tensor
    parent: torch.Tensor
    incoming_resistance: torch.Tensor
    wire_cap: torch.Tensor
    child_start: torch.Tensor
    child_vertex: torch.Tensor
    topo_order: torch.Tensor
    net_topo_start: torch.Tensor
    root_vertex: torch.Tensor
    net_ids: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_resistance: torch.Tensor
    edge_capacitance: torch.Tensor
    filtered_net_count: int
    model_report: dict


def _indices(value):
    if value.device.type != "cpu" or value.ndim != 1:
        raise ValueError("GR pin mapping indices must be flat CPU tensors")
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError("GR pin mapping indices must be integers")
    if value.numel() and (int(value.min()) < -(2**31) or int(value.max()) >= 2**31):
        raise ValueError("GR pin mapping exceeds int32 capacity")
    return value.to(dtype=torch.int32).contiguous().numpy()


def prepare_snapshot(
    route_pack,
    rc_parameters: RCParameters,
    pin_mapping: PinMapping,
    identity: RouteIdentity,
    *,
    dtype=torch.float32,
):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("GR RC supports CPU float32/float64")
    if tuple(route_pack["layer_names"]) != rc_parameters.layer_names:
        raise ValueError("GR route/LEF routing layer order mismatch")
    eligible = pin_mapping.eligible_nets
    if eligible.device.type != "cpu" or eligible.ndim != 1 or eligible.dtype != torch.bool:
        raise ValueError("GR eligible-net mask must be a flat CPU bool tensor")
    # Native exclusions carry no route geometry. Keep those pins on the
    # estimator's existing fallback path rather than requiring RC trees.
    excluded = [
        pin_mapping.net_names[name]
        for name, status in zip(route_pack["net_names"], route_pack["net_status"], strict=True)
        if status == 4 and name in pin_mapping.net_names
    ]
    eligible = eligible.clone()
    eligible[excluded] = False
    packed = gr_parasitics_cpp.prepare_tree(
        route_pack,
        pin_mapping.pin_names,
        pin_mapping.net_names,
        _indices(pin_mapping.pin_net),
        _indices(pin_mapping.driver_pin),
        eligible.contiguous().numpy().astype(np.int32),
        rc_parameters.resistance_per_um,
        rc_parameters.capacitance_per_um,
        rc_parameters.via_resistance,
    )
    floating = {
        "vertex_xy_um",
        "incoming_resistance",
        "wire_cap",
        "edge_resistance",
        "edge_capacitance",
    }
    tensors = {
        name: torch.from_numpy(value).to(dtype=dtype if name in floating else torch.int32)
        for name, value in packed.items()
        if isinstance(value, np.ndarray)
    }
    pin_vertex = tensors["pin_to_vertex"].to(torch.int64)
    valid = pin_vertex >= 0
    valid_ids = valid.nonzero().flatten()
    tensors["pin_to_vertex"] = pin_vertex
    return RouteSnapshot(
        identity=identity,
        rc_parameters=rc_parameters,
        valid_pin_mask=valid,
        valid_pin_ids=valid_ids,
        valid_vertex_ids=pin_vertex[valid_ids],
        filtered_net_count=int(packed["filtered_net_count"]),
        model_report=packed["model_report"],
        **tensors,
    )
