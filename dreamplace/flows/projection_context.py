"""Read-only timing, slew/cap and endpoint terms for ECC gate projection.

The runtime data, timing operator views and position device are explicit.
Endpoint callbacks read report/graph metadata. Candidate selection, runtime
mutation, native STA and artifact scheduling are owned by the caller.
"""

from dataclasses import dataclass
from dreamplace.flows.timing_artifacts import normalize_pin_name
from typing import Callable

import torch

from dreamplace.ops.gate_projection.gate_projection import ProjectionContext
from dreamplace.ops.size_interpolated_pin.sizing_limit_utils import (
    compute_current_pin2libpin_flat_ids,
    compute_size_interpolated_pin_properties,
)


@dataclass
class ProjectionScoringContext:
    data_collections: object
    op_collections: object
    pos: torch.Tensor
    build_endpoint_payload: Callable
    collect_cone_nodes: Callable


def current_pin_limit_context(data_collections):
    if data_collections is None:
        return None

    required_names = (
        "pin2node_map",
        "inst_main_id",
        "inst_libcell_offset",
        "main_id_2_cell_id_start",
        "cell_id_2_libpin_id_start",
        "pin_2_libpin_offset",
        "flat_lib_pin_cap",
        "flat_lib_pin_slew_limit",
        "flat_lib_pin_cap_limit",
    )
    if any(getattr(data_collections, name, None) is None for name in required_names):
        return None

    pin2libpin_flat_ids, inst_pins_mask, _ = compute_current_pin2libpin_flat_ids(
        data_collections
    )
    if pin2libpin_flat_ids is None:
        return None
    pin_cap_base = torch.zeros(
        pin2libpin_flat_ids.numel(),
        device=pin2libpin_flat_ids.device,
        dtype=torch.float32,
    )
    if torch.any(inst_pins_mask):
        pin_cap_base[inst_pins_mask] = data_collections.flat_lib_pin_cap[
            pin2libpin_flat_ids[inst_pins_mask]
        ].float()

    output_pin_mask = inst_pins_mask & (pin_cap_base == 0)
    slew_limits, cap_limits = compute_size_interpolated_pin_properties(
        data_collections,
        [
            data_collections.flat_lib_pin_slew_limit,
            data_collections.flat_lib_pin_cap_limit,
        ],
    )
    if slew_limits is None or cap_limits is None:
        return None
    return {
        "inst_pins_mask": inst_pins_mask,
        "output_pin_mask": output_pin_mask,
        "slew_limits": slew_limits.float(),
        "cap_limits": cap_limits.float(),
    }


def prepare_design_pin_values(values, expected_len):
    if values is None or not torch.is_tensor(values):
        return None
    flat_values = values.reshape(-1).float()
    if flat_values.numel() < expected_len:
        return None
    if flat_values.numel() > expected_len:
        flat_values = flat_values[:expected_len]
    return flat_values


def aggregate_violation_per_instance(
    context,
    rise_values,
    fall_values,
    pin_mask,
    limits,
    inst_ids,
):
    device = context.pos.device
    zero_result = torch.zeros(inst_ids.numel(), device=device, dtype=torch.float32)
    if pin_mask is None or limits is None:
        return zero_result

    expected_len = pin_mask.numel()
    rise_values = prepare_design_pin_values(rise_values, expected_len)
    fall_values = prepare_design_pin_values(fall_values, expected_len)
    if rise_values is None or fall_values is None:
        return zero_result

    pin_mask = pin_mask.to(device=device)
    limits = limits.to(device=device, dtype=torch.float32)
    rise_values = rise_values.to(device=device)
    fall_values = fall_values.to(device=device)
    valid_mask = pin_mask & torch.isfinite(limits) & (limits > 0)
    if not torch.any(valid_mask):
        return zero_result

    rise_violation = torch.clamp(rise_values[valid_mask] - limits[valid_mask], min=0.0)
    fall_violation = torch.clamp(fall_values[valid_mask] - limits[valid_mask], min=0.0)
    total_violation = torch.maximum(rise_violation, fall_violation)
    if not torch.any(total_violation > 0):
        return zero_result

    valid_pin_ids = torch.where(valid_mask)[0]
    node_ids = (
        context.data_collections.pin2node_map[valid_pin_ids].long().to(device=device)
    )
    per_node = torch.zeros(
        context.data_collections.inst_main_id.numel(),
        device=device,
        dtype=torch.float32,
    )
    per_node.scatter_add_(0, node_ids, total_violation)
    selected = per_node[inst_ids.long()]
    positive = selected[selected > 0]
    if positive.numel() > 0:
        scale = max(float(positive.mean().item()), 1e-6)
        selected = selected / scale
    return selected


def compute_local_timing_scores(context):
    pin_slack = getattr(context.data_collections, "pin_slack", None)
    if pin_slack is None or not torch.is_tensor(pin_slack):
        return torch.zeros(
            context.data_collections.inst_main_id.numel(),
            device=context.pos.device,
            dtype=torch.float32,
        )

    pin_nodes = context.data_collections.pin2node_map.long()
    pin_slack = pin_slack.to(context.pos.device).float()
    node_slack = torch.full(
        (context.data_collections.inst_main_id.numel(),),
        float("inf"),
        device=context.pos.device,
        dtype=torch.float32,
    )
    if hasattr(node_slack, "scatter_reduce_"):
        node_slack.scatter_reduce_(
            0, pin_nodes, pin_slack, reduce="amin", include_self=True
        )
    else:
        for pin_idx in range(pin_nodes.numel()):
            node_id = int(pin_nodes[pin_idx].item())
            node_slack[node_id] = torch.minimum(node_slack[node_id], pin_slack[pin_idx])
    local_scores = torch.clamp(-node_slack, min=0.0)
    local_scores = torch.where(
        torch.isfinite(local_scores), local_scores, torch.zeros_like(local_scores)
    )
    return local_scores


def projection_objective_profile(params):
    return getattr(params, "projection_objective_profile", "current")


def projection_profile_term_weights(params):
    profile = projection_objective_profile(params)
    if profile == "slew_cap_heavy":
        return {
            "timing_penalty": 1.0,
            "slew_penalty": 2.0,
            "cap_penalty": 2.0,
            "area_penalty": 0.02,
            "density_penalty": 0.02,
            "leakage_penalty": 0.05,
        }
    if profile == "setup_endpoint_focused":
        return {
            "timing_penalty": 1.0,
            "area_penalty": 0.03,
            "density_penalty": 0.03,
            "leakage_penalty": 0.08,
            "endpoint_focus_penalty": 2.0,
        }
    return {
        "timing_penalty": 1.0,
        "area_penalty": 0.05,
        "density_penalty": 0.05,
        "leakage_penalty": 0.1,
    }


def endpoint_focus_penalty(
    params,
    inst_ids,
    local_timing_scores,
    downsize_penalty,
    *,
    build_endpoint_payload,
    collect_cone_nodes,
):
    focus_mask = torch.zeros(
        inst_ids.numel(),
        device=local_timing_scores.device,
        dtype=local_timing_scores.dtype,
    )
    if params is None:
        return focus_mask.unsqueeze(1) * downsize_penalty, focus_mask

    _, top_endpoints = build_endpoint_payload(params, "pre_projection")
    cone_node_ids, _ = collect_cone_nodes({"pre_projection": top_endpoints})
    if cone_node_ids:
        focus_values = [
            1.0 if int(inst_id.item()) in cone_node_ids else 0.0 for inst_id in inst_ids
        ]
        focus_mask = torch.as_tensor(
            focus_values,
            device=local_timing_scores.device,
            dtype=local_timing_scores.dtype,
        )
    focus_score = torch.where(
        focus_mask > 0,
        local_timing_scores + 1.0,
        torch.zeros_like(local_timing_scores),
    )
    return focus_score.unsqueeze(1) * downsize_penalty, focus_mask


def build_projection_context(
    context,
    projection_op,
    inst_ids,
    size_var,
    vt_var,
    params=None,
    stage_timing_summary=None,
    candidates=None,
):
    if candidates is None:
        candidates = projection_op.candidate_provider.enumerate(
            context.data_collections,
            inst_ids,
        )
    request = projection_op._build_request(candidates, size_var=size_var, vt_var=vt_var)
    local_timing_scores = compute_local_timing_scores(context)[inst_ids]
    term_weights = projection_profile_term_weights(params)

    continuous_size = request.continuous_sizes.unsqueeze(1)
    downsize_penalty = torch.clamp(
        continuous_size - candidates.candidate_sizes, min=0.0
    )
    area_penalty = torch.clamp(candidates.candidate_sizes - continuous_size, min=0.0)
    continuous_total_size = float(request.continuous_sizes.clamp(min=0).sum().item())
    density_penalty = area_penalty / max(continuous_total_size, 1.0)
    current_leakage = request.current_leakage.unsqueeze(1).clamp(min=0.0)
    leakage_reference = max(
        float(request.current_leakage.clamp(min=0).mean().item()), 1.0
    )
    leakage_penalty = candidates.candidate_leakages.clamp(min=0.0) / leakage_reference
    endpoint_focus_mask = torch.zeros(
        inst_ids.numel(),
        device=local_timing_scores.device,
        dtype=local_timing_scores.dtype,
    )

    candidate_terms = {
        "timing_penalty": local_timing_scores.unsqueeze(1) * downsize_penalty,
        "area_penalty": area_penalty,
        "density_penalty": density_penalty,
        "leakage_penalty": leakage_penalty,
    }
    pin_limit_context = current_pin_limit_context(context.data_collections)
    timing_op = getattr(context.op_collections, "timing_propagation_op", None)
    if "slew_penalty" in term_weights:
        slew_overflow = aggregate_violation_per_instance(
            context,
            None if timing_op is None else getattr(timing_op, "pin_rtran", None),
            None if timing_op is None else getattr(timing_op, "pin_ftran", None),
            None
            if pin_limit_context is None
            else pin_limit_context.get("inst_pins_mask"),
            None if pin_limit_context is None else pin_limit_context.get("slew_limits"),
            inst_ids,
        )
        candidate_terms["slew_penalty"] = slew_overflow.unsqueeze(1) * downsize_penalty
    if "cap_penalty" in term_weights:
        cap_overflow = aggregate_violation_per_instance(
            context,
            None if timing_op is None else getattr(timing_op, "pin_net_cap_rise", None),
            None if timing_op is None else getattr(timing_op, "pin_net_cap_fall", None),
            None
            if pin_limit_context is None
            else pin_limit_context.get("output_pin_mask"),
            None if pin_limit_context is None else pin_limit_context.get("cap_limits"),
            inst_ids,
        )
        candidate_terms["cap_penalty"] = cap_overflow.unsqueeze(1) * downsize_penalty
    if "endpoint_focus_penalty" in term_weights:
        focus_penalty, endpoint_focus_mask = endpoint_focus_penalty(
            params,
            inst_ids,
            local_timing_scores,
            downsize_penalty,
            build_endpoint_payload=context.build_endpoint_payload,
            collect_cone_nodes=context.collect_cone_nodes,
        )
        candidate_terms["endpoint_focus_penalty"] = focus_penalty

    return ProjectionContext(
        candidate_terms=candidate_terms,
        term_weights=term_weights,
        metadata={
            "local_timing_scores": local_timing_scores.detach().cpu(),
            "stage_timing_summary": stage_timing_summary or {},
            "continuous_total_size": continuous_total_size,
            "current_total_leakage": float(
                request.current_leakage.clamp(min=0).sum().item()
            ),
            "mean_current_leakage": float(current_leakage.mean().item())
            if current_leakage.numel()
            else 0.0,
            "projection_objective_profile": projection_objective_profile(params),
            "endpoint_focus_instance_count": int(endpoint_focus_mask.sum().item()),
        },
    )

def reverse_reachable_pin_ids(seed_pin_ids, reverse_offsets, reverse_edges, num_pins):
    if num_pins <= 0:
        return set()
    visited = set()
    stack = [int(pin_id) for pin_id in seed_pin_ids]
    while stack:
        pin_id = stack.pop()
        if pin_id in visited or pin_id < 0 or pin_id >= num_pins:
            continue
        visited.add(pin_id)
        start = int(reverse_offsets[pin_id].item())
        end = int(reverse_offsets[pin_id + 1].item())
        for edge_idx in range(start, end):
            predecessor = int(reverse_edges[edge_idx].item())
            if predecessor not in visited:
                stack.append(predecessor)
    return visited


@dataclass(frozen=True)
class CriticalConeContext:
    data_collections: object
    placedb: object

def endpoint_pin_name_to_id_map(context):
    placedb = getattr(context, "placedb", None)
    pin_names = getattr(placedb, "pin_names", None)
    if pin_names is None:
        return {}
    return {
        normalize_pin_name(pin_name): pin_id
        for pin_id, pin_name in enumerate(pin_names)
    }


def collect_critical_endpoint_cone_nodes(context, top_critical_endpoints):
    data_collections = getattr(context, "data_collections", None)
    pin2node_map = getattr(data_collections, "pin2node_map", None)
    reverse_offsets = getattr(data_collections, "flat_pin_to_graph_start_reverse", None)
    reverse_edges = getattr(data_collections, "flat_pin_to_graph_reverse", None)
    if (
        pin2node_map is None
        or reverse_offsets is None
        or reverse_edges is None
    ):
        return set(), {}

    pin2node_map = torch.as_tensor(pin2node_map).detach().cpu().to(torch.int64)
    reverse_offsets = torch.as_tensor(reverse_offsets).detach().cpu().to(torch.int64)
    reverse_edges = torch.as_tensor(reverse_edges).detach().cpu().to(torch.int64)
    if pin2node_map.numel() == 0 or reverse_offsets.numel() < 2:
        return set(), {}

    name_to_pin_id = endpoint_pin_name_to_id_map(context)
    num_pins = min(int(pin2node_map.numel()), int(reverse_offsets.numel()) - 1)
    cone_node_ids = set()
    stage_hits = {}

    for stage_name, rows in (top_critical_endpoints or {}).items():
        seed_pin_ids = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            pin_name = normalize_pin_name(row.get("pin_name"))
            if pin_name is None:
                continue
            pin_id = name_to_pin_id.get(pin_name)
            if pin_id is not None:
                seed_pin_ids.append(pin_id)
        if not seed_pin_ids:
            continue

        visited_pin_ids = reverse_reachable_pin_ids(
            seed_pin_ids,
            reverse_offsets,
            reverse_edges,
            num_pins,
        )
        for pin_id in visited_pin_ids:
            node_id = int(pin2node_map[pin_id].item())
            cone_node_ids.add(node_id)
            stage_hits.setdefault(node_id, set()).add(stage_name)

    return cone_node_ids, {
        node_id: sorted(stage_names)
        for node_id, stage_names in stage_hits.items()
    }
