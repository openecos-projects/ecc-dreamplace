from dataclasses import dataclass

import torch


@dataclass
class InflationGeometryBackup:
    node_size_x: torch.Tensor
    node_size_y: torch.Tensor
    pin_offset_x: torch.Tensor
    pin_offset_y: torch.Tensor


def _is_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def is_enabled(params):
    return _is_enabled(
        getattr(params, "legalize_before_each_inflation_flag", 0)
    )


def validate_params(params):
    if not is_enabled(params):
        return
    if not _is_enabled(getattr(params, "routability_opt_flag", 0)):
        raise ValueError(
            "legalize_before_each_inflation_flag requires routability_opt_flag=1"
        )
    if not _is_enabled(getattr(params, "legalize_flag", 0)):
        raise ValueError(
            "legalize_before_each_inflation_flag requires legalize_flag=1"
        )
    if _is_enabled(getattr(params, "enhanced_inflation_flag", 0)):
        raise ValueError(
            "legalize_before_each_inflation_flag supports ordinary inflation only; "
            "set enhanced_inflation_flag=0"
        )


def legacy_trigger_threshold(params):
    if is_enabled(params):
        return float(getattr(params, "stop_overflow", 0.1))
    return float(getattr(params, "node_area_adjust_overflow", 0.15))


def should_trigger_legacy_inflation(
    params,
    num_area_adjust,
    max_area_adjust_rounds,
    overflow,
    trigger_enhanced_inflation=False,
):
    if trigger_enhanced_inflation:
        return False
    if int(num_area_adjust) >= int(max_area_adjust_rounds):
        return False
    if isinstance(overflow, torch.Tensor):
        overflow = overflow.detach().reshape(-1)[0].item()
    return float(overflow) < legacy_trigger_threshold(params)


def _apply_geometry_preserving_centers(pos, data_collections, geometry):
    num_nodes = int(data_collections.node_size_x.numel())
    if pos.numel() < num_nodes * 2:
        raise ValueError("position tensor is too small for the node geometry")

    with torch.no_grad():
        center_x = pos.data[:num_nodes] + data_collections.node_size_x * 0.5
        center_y = (
            pos.data[num_nodes : num_nodes * 2]
            + data_collections.node_size_y * 0.5
        )
        data_collections.node_size_x.copy_(geometry.node_size_x)
        data_collections.node_size_y.copy_(geometry.node_size_y)
        pos.data[:num_nodes].copy_(
            center_x - data_collections.node_size_x * 0.5
        )
        pos.data[num_nodes : num_nodes * 2].copy_(
            center_y - data_collections.node_size_y * 0.5
        )
        data_collections.pin_offset_x.copy_(geometry.pin_offset_x)
        data_collections.pin_offset_y.copy_(geometry.pin_offset_y)


def use_physical_geometry(pos, data_collections):
    backup = InflationGeometryBackup(
        node_size_x=data_collections.node_size_x.detach().clone(),
        node_size_y=data_collections.node_size_y.detach().clone(),
        pin_offset_x=data_collections.pin_offset_x.detach().clone(),
        pin_offset_y=data_collections.pin_offset_y.detach().clone(),
    )
    physical_geometry = InflationGeometryBackup(
        node_size_x=data_collections.original_node_size_x,
        node_size_y=data_collections.original_node_size_y,
        pin_offset_x=data_collections.original_pin_offset_x,
        pin_offset_y=data_collections.original_pin_offset_y,
    )
    _apply_geometry_preserving_centers(pos, data_collections, physical_geometry)
    return backup


def restore_inflated_geometry(pos, data_collections, backup):
    _apply_geometry_preserving_centers(pos, data_collections, backup)
