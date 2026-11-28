"""Projection frame assembly and file-only artifacts for ECC sizing/VT.

Frame assembly clones runtime views into explicit candidate/global index spaces;
it does not install the projection or refresh STA. Writers receive a computed
projection result and frame. The placement flow decides policy and write timing.
"""


import json
import os
from dataclasses import dataclass
from typing import Callable
import torch
from dreamplace.ops.gate_projection.gate_projection import (
    ProjectionFrame,
    ProjectionTensorMetadata,
)
from dreamplace.flows.gp.timing_artifacts import normalize_pin_name
from dreamplace.ops.gate_projection.gate_projection import ProjectionContext
from dreamplace.ops.size_interpolated_pin.sizing_limit_utils import (
    compute_current_pin2libpin_flat_ids,
    compute_size_interpolated_pin_properties,
)
import logging
import time
from dataclasses import replace
from dreamplace.flows.gp.metrics import timing_scalar


@dataclass
class ProjectionFrameContext:
    data_collections: object
    op_collections: object
    changed_inst_mask: Callable
    record_bytes: Callable
    digest_tensors: Callable


def projected_vt_distribution(vt_tensor, legal_inst_ids, projected_vt):
    if vt_tensor is None:
        return None
    projected_vt_tensor = vt_tensor.detach().clone().float()
    if projected_vt_tensor.dim() == 2 and projected_vt_tensor.size(1) > 0:
        vt_one_hot = torch.nn.functional.one_hot(
            projected_vt.long().to(projected_vt_tensor.device),
            num_classes=projected_vt_tensor.size(1),
        ).to(projected_vt_tensor.dtype)
        projected_vt_tensor[legal_inst_ids] = vt_one_hot
        return projected_vt_tensor
    if projected_vt_tensor.dim() == 1:
        projected_vt_tensor[legal_inst_ids] = projected_vt.float().to(
            projected_vt_tensor.device
        )
        return projected_vt_tensor
    return projected_vt_tensor


def projection_changed_inst_mask(projection_result):
    if projection_result is None:
        return None

    legal_mask = getattr(projection_result, "has_legal_candidate", None)
    if legal_mask is None:
        return None
    legal_mask = legal_mask.bool()
    if legal_mask.numel() == 0:
        return legal_mask

    delta_terms = []
    for current_name, projected_name in (
        ("current_cell_id", "projected_cell_id"),
        ("current_libcell_offset", "projected_libcell_offset"),
        ("current_vt", "projected_vt"),
    ):
        current_value = getattr(projection_result, current_name, None)
        projected_value = getattr(projection_result, projected_name, None)
        if current_value is None or projected_value is None:
            continue
        delta_terms.append(current_value.long() != projected_value.long())

    current_size = getattr(projection_result, "current_size", None)
    projected_size = getattr(projection_result, "projected_size", None)
    if current_size is not None and projected_size is not None:
        delta_terms.append(
            ~torch.isclose(
                current_size.float(),
                projected_size.float(),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    if not delta_terms:
        return torch.zeros_like(legal_mask, dtype=torch.bool)

    changed_mask = torch.zeros_like(legal_mask, dtype=torch.bool)
    for term in delta_terms:
        changed_mask |= term.bool()
    return legal_mask & changed_mask


def projection_artifact_policy(params, stage_timing_summary=None):
    policy = (
        str(
            getattr(params, "real_size_transition_artifact_policy", "summary_only")
            or "summary_only"
        )
        .strip()
        .lower()
    )
    if policy not in {"summary_only", "full"}:
        raise ValueError(
            "unsupported real_size_transition_artifact_policy: "
            f"{policy!r}; expected summary_only or full"
        )
    scope = str((stage_timing_summary or {}).get("artifact_scope", ""))
    if scope == "real_size_warmup_transition":
        return policy
    return "full"


def write_projection_summary_artifact(
    projection_op,
    params,
    projection_result,
    metadata=None,
    *,
    projection_frame,
):
    summary = projection_op.build_artifact_summary(
        projection_result,
        metadata=metadata,
        pin_offset_artifacts=None,
    )
    if (
        projection_frame is not None
        and projection_frame.projection_result is projection_result
    ):
        summary.update(
            {
                "state_digest": projection_frame.state_digest,
                "num_changed_instances": int(projection_frame.changed_inst_ids.numel()),
                "num_changed_pins": int(projection_frame.changed_pin_ids.numel()),
                "topology_generation": int(projection_frame.topology_generation_before),
                "timing_model_generation": int(
                    projection_frame.timing_model_generation_before
                ),
                "tensor_contract": [
                    {
                        "name": item.name,
                        "shape": item.shape,
                        "dtype": item.dtype,
                        "device": item.device,
                        "index_space": item.index_space,
                    }
                    for item in projection_frame.tensor_contract
                ],
            }
        )
    summary_path = os.path.join(
        params.result_dir,
        f"{params.design_name()}_real_size_transition_projection_summary.json",
    )
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return {"summary": summary_path}


def build_projection_frame(context, params, projection_result):
    """Materialize one projection result in candidate and global index spaces."""
    if projection_result is None:
        raise RuntimeError("cannot build a projection frame without a result")
    data_collections = context.data_collections
    inst_cell_id = getattr(data_collections, "inst_cell_id", None)
    inst_libcell_offset = getattr(data_collections, "inst_libcell_offset", None)
    if inst_cell_id is None or inst_libcell_offset is None:
        raise RuntimeError(
            "projection frame requires global cell-id and libcell-offset state"
        )

    result_inst_ids = projection_result.inst_ids.long()
    legal_mask = projection_result.has_legal_candidate.bool()
    if result_inst_ids.dim() != 1 or legal_mask.dim() != 1:
        raise ValueError("projection frame requires one-dimensional row tensors")
    if legal_mask.numel() != result_inst_ids.numel():
        raise ValueError("projection legal mask does not match instance rows")
    for name in (
        "current_cell_id",
        "current_libcell_offset",
        "current_vt",
        "current_size",
        "projected_cell_id",
        "projected_libcell_offset",
        "projected_vt",
        "projected_size",
    ):
        value = getattr(projection_result, name, None)
        if value is None or value.numel() != result_inst_ids.numel():
            raise ValueError(f"projection field {name} does not match instance rows")
    changed_candidate_mask = context.changed_inst_mask(projection_result)
    if changed_candidate_mask is None:
        changed_candidate_mask = torch.zeros_like(legal_mask, dtype=torch.bool)

    node_device = inst_cell_id.device
    num_nodes = int(inst_cell_id.numel())
    if result_inst_ids.numel() > 0:
        if bool(torch.any(result_inst_ids < 0).item()) or bool(
            torch.any(result_inst_ids >= num_nodes).item()
        ):
            raise ValueError("projection frame contains an invalid instance id")
        if int(torch.unique(result_inst_ids).numel()) != int(result_inst_ids.numel()):
            raise ValueError("projection frame contains duplicate instance ids")
    legal_inst_ids = result_inst_ids[legal_mask].to(device=node_device)
    changed_inst_ids = result_inst_ids[changed_candidate_mask].to(device=node_device)
    if changed_inst_ids.numel() > 0:
        if bool(torch.any(changed_inst_ids < 0).item()) or bool(
            torch.any(changed_inst_ids >= num_nodes).item()
        ):
            raise ValueError("projection frame contains an invalid changed instance id")
    changed_node_mask = torch.zeros(
        num_nodes,
        device=node_device,
        dtype=torch.bool,
    )
    if changed_inst_ids.numel() > 0:
        changed_node_mask[changed_inst_ids] = True

    projected_cell_id_global = inst_cell_id.detach().clone().long()
    projected_libcell_offset_global = inst_libcell_offset.detach().clone().long()
    if legal_inst_ids.numel() > 0:
        projected_cell_id_global[legal_inst_ids] = (
            projection_result.projected_cell_id[legal_mask]
            .detach()
            .to(device=node_device, dtype=torch.long)
        )
        projected_libcell_offset_global[legal_inst_ids] = (
            projection_result.projected_libcell_offset[legal_mask]
            .detach()
            .to(device=node_device, dtype=torch.long)
        )

    current_size = data_collections.get_size_var()
    if current_size is None:
        raise RuntimeError("projection frame requires a live continuous size state")
    projected_size_global = current_size.detach().clone()
    if legal_inst_ids.numel() > 0:
        projected_size_global[legal_inst_ids.to(projected_size_global.device)] = (
            projection_result.projected_size[legal_mask]
            .detach()
            .to(device=projected_size_global.device, dtype=projected_size_global.dtype)
        )

    current_vt = data_collections.get_vt_var()
    projected_vt_global = (
        projected_vt_distribution(
            current_vt,
            legal_inst_ids.to(device=current_vt.device)
            if current_vt is not None
            else legal_inst_ids,
            projection_result.projected_vt[legal_mask].detach(),
        )
        if current_vt is not None
        else None
    )

    projected_inst_size_init = getattr(data_collections, "inst_size_init", None)
    if projected_inst_size_init is not None:
        projected_inst_size_init = projected_inst_size_init.detach().clone()
        if legal_inst_ids.numel() > 0:
            projected_inst_size_init[
                legal_inst_ids.to(projected_inst_size_init.device)
            ] = (
                projection_result.projected_size[legal_mask]
                .detach()
                .to(
                    device=projected_inst_size_init.device,
                    dtype=projected_inst_size_init.dtype,
                )
            )

    projected_node_size_x = getattr(data_collections, "node_size_x", None)
    projected_node_size_y = getattr(data_collections, "node_size_y", None)
    projected_node_area = getattr(data_collections, "node_areas", None)
    if projected_node_size_x is not None:
        projected_node_size_x = projected_node_size_x.detach().clone()
    if projected_node_size_y is not None:
        projected_node_size_y = projected_node_size_y.detach().clone()
    if projected_node_area is not None:
        projected_node_area = projected_node_area.detach().clone()
    flat_width = getattr(data_collections, "flat_libcell_width", None)
    flat_height = getattr(data_collections, "flat_libcell_height", None)
    if (
        legal_inst_ids.numel() > 0
        and projected_node_size_x is not None
        and projected_node_size_y is not None
        and projected_node_area is not None
        and flat_width is not None
        and flat_height is not None
    ):
        projected_cell_ids = projected_cell_id_global[legal_inst_ids]
        width = flat_width[projected_cell_ids.to(flat_width.device)].to(
            device=projected_node_size_x.device,
            dtype=projected_node_size_x.dtype,
        )
        height = flat_height[projected_cell_ids.to(flat_height.device)].to(
            device=projected_node_size_y.device,
            dtype=projected_node_size_y.dtype,
        )
        padding_x = float(getattr(params, "cell_padding_x", 0.0) or 0.0)
        if padding_x > 0.0:
            width = width + 2.0 * padding_x
        projected_node_size_x[legal_inst_ids.to(projected_node_size_x.device)] = width
        projected_node_size_y[legal_inst_ids.to(projected_node_size_y.device)] = height
        projected_node_area[legal_inst_ids.to(projected_node_area.device)] = width.to(
            device=projected_node_area.device, dtype=projected_node_area.dtype
        ) * height.to(
            device=projected_node_area.device, dtype=projected_node_area.dtype
        )

    projection_op = getattr(context.op_collections, "gate_projection_op", None)
    if projection_op is None or not callable(
        getattr(projection_op, "compute_projected_pin_offsets", None)
    ):
        raise RuntimeError(
            "projection frame requires the gate projection pin-offset op"
        )
    pin_offset_result = projection_op.compute_projected_pin_offsets(
        projection_result,
        row_mask=changed_candidate_mask,
    )
    has_projected_pin = pin_offset_result["has_projected_pin_offset"].bool()
    changed_pin_ids = pin_offset_result["pin_ids"][has_projected_pin].long()
    projected_pin_offset_x = pin_offset_result["projected_pin_offset_x"][
        has_projected_pin
    ].detach()
    projected_pin_offset_y = pin_offset_result["projected_pin_offset_y"][
        has_projected_pin
    ].detach()

    projected_original_pin_offset_x = getattr(
        data_collections, "original_pin_offset_x", None
    )
    projected_original_pin_offset_y = getattr(
        data_collections, "original_pin_offset_y", None
    )
    if projected_original_pin_offset_x is not None:
        projected_original_pin_offset_x = (
            projected_original_pin_offset_x.detach().clone()
        )
        projected_original_pin_offset_x[
            changed_pin_ids.to(projected_original_pin_offset_x.device)
        ] = projected_pin_offset_x.to(
            device=projected_original_pin_offset_x.device,
            dtype=projected_original_pin_offset_x.dtype,
        )
    if projected_original_pin_offset_y is not None:
        projected_original_pin_offset_y = (
            projected_original_pin_offset_y.detach().clone()
        )
        projected_original_pin_offset_y[
            changed_pin_ids.to(projected_original_pin_offset_y.device)
        ] = projected_pin_offset_y.to(
            device=projected_original_pin_offset_y.device,
            dtype=projected_original_pin_offset_y.dtype,
        )

    digest_inputs = (
        ("changed_inst_ids", changed_inst_ids),
        ("projected_cell_id", projected_cell_id_global[changed_inst_ids]),
        ("projected_libcell_offset", projected_libcell_offset_global[changed_inst_ids]),
        (
            "projected_size",
            projected_size_global[changed_inst_ids.to(projected_size_global.device)],
        ),
        ("changed_pin_ids", changed_pin_ids),
        ("projected_pin_offset_x", projected_pin_offset_x),
        ("projected_pin_offset_y", projected_pin_offset_y),
    )
    context.record_bytes(
        "device_to_host_bytes",
        *(value for _name, value in digest_inputs if value.device.type != "cpu"),
    )
    digest = context.digest_tensors(digest_inputs)

    def tensor_metadata(name, value, index_space):
        if value is None:
            return ProjectionTensorMetadata(
                name=name,
                shape=None,
                dtype=None,
                device=None,
                index_space=index_space,
            )
        return ProjectionTensorMetadata(
            name=name,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            index_space=index_space,
        )

    tensor_contract = tuple(
        tensor_metadata(name, value, index_space)
        for name, value, index_space in (
            (
                "projection_result.inst_ids",
                projection_result.inst_ids,
                "candidate_row_to_global_node_id",
            ),
            (
                "changed_candidate_mask",
                changed_candidate_mask,
                "candidate_row",
            ),
            (
                "changed_inst_ids",
                changed_inst_ids,
                "changed_instance_row_to_global_node_id",
            ),
            ("changed_node_mask", changed_node_mask, "global_node"),
            (
                "changed_pin_ids",
                changed_pin_ids,
                "changed_pin_row_to_global_pin_id",
            ),
            (
                "projected_pin_offset_x",
                projected_pin_offset_x,
                "changed_pin_row",
            ),
            (
                "projected_pin_offset_y",
                projected_pin_offset_y,
                "changed_pin_row",
            ),
            ("projected_node_size_x", projected_node_size_x, "global_node"),
            ("projected_node_size_y", projected_node_size_y, "global_node"),
            ("projected_node_area", projected_node_area, "global_node"),
            ("projected_size_global", projected_size_global, "global_node"),
            ("projected_vt_global", projected_vt_global, "global_node"),
            (
                "projected_cell_id_global",
                projected_cell_id_global,
                "global_node",
            ),
            (
                "projected_libcell_offset_global",
                projected_libcell_offset_global,
                "global_node",
            ),
            (
                "projected_inst_size_init",
                projected_inst_size_init,
                "global_node",
            ),
            (
                "projected_original_pin_offset_x",
                projected_original_pin_offset_x,
                "global_pin",
            ),
            (
                "projected_original_pin_offset_y",
                projected_original_pin_offset_y,
                "global_pin",
            ),
        )
    )
    topo_op = getattr(context.op_collections, "steiner_topo_op", None)
    return ProjectionFrame(
        projection_result=projection_result,
        changed_candidate_mask=changed_candidate_mask,
        changed_inst_ids=changed_inst_ids,
        changed_node_mask=changed_node_mask,
        changed_pin_ids=changed_pin_ids,
        projected_pin_offset_x=projected_pin_offset_x,
        projected_pin_offset_y=projected_pin_offset_y,
        projected_node_size_x=projected_node_size_x,
        projected_node_size_y=projected_node_size_y,
        projected_node_area=projected_node_area,
        projected_size_global=projected_size_global,
        projected_vt_global=projected_vt_global,
        projected_cell_id_global=projected_cell_id_global,
        projected_libcell_offset_global=projected_libcell_offset_global,
        projected_inst_size_init=projected_inst_size_init,
        projected_original_pin_offset_x=projected_original_pin_offset_x,
        projected_original_pin_offset_y=projected_original_pin_offset_y,
        state_digest=digest,
        topology_generation_before=int(getattr(topo_op, "topology_generation", 0) or 0),
        topology_generation_after=None,
        timing_model_generation_before=int(
            getattr(data_collections, "timing_model_generation", 0) or 0
        ),
        timing_model_generation_after=None,
        tensor_contract=tensor_contract,
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


def apply_projected_runtime_refresh(engine, params, placedb):
    projection_frame = getattr(engine, "last_projection_frame", None)
    projection_result = getattr(engine, "last_projection_result", None)
    if projection_frame is not None:
        if projection_result is not None and projection_frame.projection_result is not projection_result:
            projection_frame = None
        else:
            projection_result = projection_frame.projection_result
    if projection_result is None:
        summary = {
            "runtime_refresh_applied": False,
            "num_candidate_instances": 0,
            "num_changed_instances": 0,
            "num_runtime_updated_instances": 0,
            "num_runtime_updated_node_geometries": 0,
            "num_runtime_updated_pins": 0,
            "runtime_geometry_refresh_applied": False,
            "runtime_pin_offset_consistency_applied": False,
            "runtime_topology_rebuilt": False,
            "metadata": {
                "artifact_scope": "post_projection_runtime",
                "placement_sizing_mode": engine._placement_sizing_mode(params),
            },
        }
        engine._write_projection_runtime_refresh_summary(params, summary)
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_runtime_refresh_summary = summary
        engine.last_projection_runtime_refresh_summary_path = getattr(
            engine,
            "last_projection_runtime_refresh_summary_path",
            None,
        )
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        return summary

    continuous_size_sync = engine._sync_projected_real_size(projection_result)

    changed_mask = (
        projection_frame.changed_candidate_mask
        if projection_frame is not None
        else engine._projection_changed_inst_mask(projection_result)
    )
    if changed_mask is None:
        summary = {
            "runtime_refresh_applied": False,
            "num_candidate_instances": int(
                getattr(projection_result, "inst_ids", torch.empty(0)).numel()
            ),
            "num_changed_instances": 0,
            "num_runtime_updated_instances": 0,
            "num_runtime_updated_node_geometries": 0,
            "num_runtime_updated_pins": 0,
            "runtime_geometry_refresh_applied": False,
            "runtime_pin_offset_consistency_applied": False,
            "runtime_topology_rebuilt": False,
            "continuous_size_sync": continuous_size_sync,
            "metadata": {
                "artifact_scope": "post_projection_runtime",
                "placement_sizing_mode": engine._placement_sizing_mode(params),
            },
        }
        engine._write_projection_runtime_refresh_summary(params, summary)
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_runtime_refresh_summary = summary
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        return summary

    changed_inst_ids = (
        projection_frame.changed_inst_ids
        if projection_frame is not None
        else projection_result.inst_ids[changed_mask].long()
    )
    summary = {
        "runtime_refresh_applied": False,
        "num_candidate_instances": int(projection_result.inst_ids.numel()),
        "num_changed_instances": int(changed_inst_ids.numel()),
        "num_runtime_updated_instances": 0,
        "num_runtime_updated_node_geometries": 0,
        "num_runtime_updated_pins": 0,
        "runtime_geometry_refresh_applied": False,
        "runtime_pin_offset_consistency_applied": False,
        "runtime_topology_rebuilt": False,
        "continuous_size_sync": continuous_size_sync,
        "metadata": {
            "artifact_scope": "post_projection_runtime",
            "placement_sizing_mode": engine._placement_sizing_mode(params),
        },
    }

    if changed_inst_ids.numel() == 0:
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        engine._write_projection_runtime_refresh_summary(params, summary)
        return summary

    defer_placedb_sync = engine._defer_real_size_transition_placedb_sync()
    changed_inst_ids_cpu = (
        None if defer_placedb_sync else changed_inst_ids.detach().cpu().numpy()
    )
    if projection_frame is not None:
        projected_cell_ids = projection_frame.projected_cell_id_global[
            changed_inst_ids.to(projection_frame.projected_cell_id_global.device)
        ].long()
        projected_offsets = projection_frame.projected_libcell_offset_global[
            changed_inst_ids.to(
                projection_frame.projected_libcell_offset_global.device
            )
        ].long()
    else:
        projected_cell_ids = projection_result.projected_cell_id[changed_mask].long()
        projected_offsets = projection_result.projected_libcell_offset[changed_mask].long()

    runtime_cell_apply_started_at = time.perf_counter()
    with torch.no_grad():
        if getattr(engine.data_collections, "inst_cell_id", None) is not None:
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.inst_cell_id.device
            )
            projected_cell_ids_device = projected_cell_ids.to(
                engine.data_collections.inst_cell_id.device,
                dtype=engine.data_collections.inst_cell_id.dtype,
            )
            engine.data_collections.inst_cell_id[changed_inst_ids_device] = projected_cell_ids_device
        if getattr(engine.data_collections, "inst_libcell_offset", None) is not None:
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.inst_libcell_offset.device
            )
            projected_offsets_device = projected_offsets.to(
                engine.data_collections.inst_libcell_offset.device,
                dtype=engine.data_collections.inst_libcell_offset.dtype,
            )
            engine.data_collections.inst_libcell_offset[changed_inst_ids_device] = projected_offsets_device
        if (
            projection_frame is not None
            and projection_frame.projected_vt_global is not None
            and getattr(engine.data_collections, "vt_logits", None) is not None
        ):
            vt_logits = engine.data_collections.vt_logits
            vt_ids = changed_inst_ids.to(device=vt_logits.device)
            projected_vt_distribution = projection_frame.projected_vt_global[
                changed_inst_ids.to(
                    projection_frame.projected_vt_global.device
                )
            ].to(device=vt_logits.device, dtype=vt_logits.dtype)
            projected_vt_logits = torch.log(
                projected_vt_distribution.clamp(min=1e-6)
            )
            vt_logits[vt_ids] = projected_vt_logits
            summary_vt_updated = int(vt_ids.numel())
        else:
            summary_vt_updated = 0

    if not defer_placedb_sync and getattr(placedb, "inst_cell_id", None) is not None:
        placedb.inst_cell_id[changed_inst_ids_cpu] = projected_cell_ids.detach().cpu().numpy()
    if not defer_placedb_sync and getattr(placedb, "inst_libcell_offset", None) is not None:
        placedb.inst_libcell_offset[changed_inst_ids_cpu] = projected_offsets.detach().cpu().numpy()
    if not defer_placedb_sync and getattr(placedb, "inst_cell_id", None) is not None and getattr(placedb, "flat_libcell_names", None) is not None:
        preview = []
        for idx in range(min(10, len(placedb.inst_cell_id))):
            cell_id = int(placedb.inst_cell_id[idx])
            if 0 <= cell_id < len(placedb.flat_libcell_names):
                preview.append((cell_id, str(placedb.flat_libcell_names[cell_id])))
        logging.info(
            "projection runtime refresh preview placedb_id=%s first_cells=%s",
            id(placedb),
            preview,
        )

    flat_libcell_width = getattr(engine.data_collections, "flat_libcell_width", None)
    flat_libcell_height = getattr(engine.data_collections, "flat_libcell_height", None)
    if (
        projection_frame is not None
        and projection_frame.projected_node_size_x is not None
        and projection_frame.projected_node_size_y is not None
    ):
        frame_ids = changed_inst_ids.to(
            projection_frame.projected_node_size_x.device
        )
        projected_width = projection_frame.projected_node_size_x[frame_ids]
        projected_height = projection_frame.projected_node_size_y[
            changed_inst_ids.to(projection_frame.projected_node_size_y.device)
        ]

        with torch.no_grad():
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.node_size_x.device
            )
            engine.data_collections.node_size_x[changed_inst_ids_device] = projected_width.to(
                device=engine.data_collections.node_size_x.device,
                dtype=engine.data_collections.node_size_x.dtype,
            )
            engine.data_collections.node_size_y[changed_inst_ids_device] = projected_height.to(
                device=engine.data_collections.node_size_y.device,
                dtype=engine.data_collections.node_size_y.dtype,
            )
            if getattr(engine.data_collections, "original_node_size_x", None) is not None:
                engine.data_collections.original_node_size_x[changed_inst_ids_device] = projected_width.to(
                    device=engine.data_collections.original_node_size_x.device,
                    dtype=engine.data_collections.original_node_size_x.dtype,
                )
                engine.data_collections.original_node_size_y[changed_inst_ids_device] = projected_height.to(
                    device=engine.data_collections.original_node_size_y.device,
                    dtype=engine.data_collections.original_node_size_y.dtype,
                )
            if getattr(engine.data_collections, "node_areas", None) is not None:
                engine.data_collections.node_areas[changed_inst_ids_device] = (
                    projected_width.to(
                        device=engine.data_collections.node_areas.device,
                        dtype=engine.data_collections.node_areas.dtype,
                    )
                    * projected_height.to(
                        device=engine.data_collections.node_areas.device,
                        dtype=engine.data_collections.node_areas.dtype,
                    )
                )
        summary["runtime_geometry_refresh_applied"] = True
        summary["num_runtime_updated_node_geometries"] = int(changed_inst_ids.numel())
        if projection_frame.projected_inst_size_init is not None:
            with torch.no_grad():
                init_ids = changed_inst_ids.to(
                    projection_frame.projected_inst_size_init.device
                )
                engine.data_collections.inst_size_init[init_ids] = (
                    projection_frame.projected_inst_size_init[init_ids].to(
                        device=engine.data_collections.inst_size_init.device,
                        dtype=engine.data_collections.inst_size_init.dtype,
                    )
                )
        if (
            not defer_placedb_sync
            and placedb is not None
            and getattr(placedb, "node_size_x", None) is not None
        ):
            changed_cpu = changed_inst_ids.detach().cpu().numpy()
            placedb.node_size_x[changed_cpu] = projected_width.detach().cpu().numpy()
            placedb.node_size_y[changed_cpu] = projected_height.detach().cpu().numpy()
            if getattr(placedb, "inst_size_init", None) is not None:
                init_values = projection_frame.projected_inst_size_init[
                    changed_inst_ids.to(
                        projection_frame.projected_inst_size_init.device
                    )
                ]
                placedb.inst_size_init[changed_cpu] = init_values.detach().cpu().numpy()
    elif flat_libcell_width is not None and flat_libcell_height is not None:
        projected_width = flat_libcell_width[projected_cell_ids].to(
            engine.data_collections.node_size_x.device,
            dtype=engine.data_collections.node_size_x.dtype,
        )
        projected_height = flat_libcell_height[projected_cell_ids].to(
            engine.data_collections.node_size_y.device,
            dtype=engine.data_collections.node_size_y.dtype,
        )

        with torch.no_grad():
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.node_size_x.device
            )
            engine.data_collections.node_size_x[changed_inst_ids_device] = projected_width
            engine.data_collections.node_size_y[changed_inst_ids_device] = projected_height
            if getattr(engine.data_collections, "original_node_size_x", None) is not None:
                original_changed_inst_ids_device = changed_inst_ids.to(
                    engine.data_collections.original_node_size_x.device
                )
                engine.data_collections.original_node_size_x[original_changed_inst_ids_device] = projected_width
                engine.data_collections.original_node_size_y[original_changed_inst_ids_device] = projected_height
            if getattr(engine.data_collections, "node_areas", None) is not None:
                projected_area = projected_width * projected_height
                area_changed_inst_ids_device = changed_inst_ids.to(
                    engine.data_collections.node_areas.device
                )
                engine.data_collections.node_areas[area_changed_inst_ids_device] = projected_area

        if not defer_placedb_sync and getattr(placedb, "node_size_x", None) is not None:
            placedb.node_size_x[changed_inst_ids_cpu] = projected_width.detach().cpu().numpy()
        if not defer_placedb_sync and getattr(placedb, "node_size_y", None) is not None:
            placedb.node_size_y[changed_inst_ids_cpu] = projected_height.detach().cpu().numpy()

        summary["runtime_geometry_refresh_applied"] = True
        summary["num_runtime_updated_node_geometries"] = int(changed_inst_ids.numel())

    engine._record_real_size_transition_stage(
        "runtime_cell_apply_ms",
        runtime_cell_apply_started_at,
    )
    summary["num_runtime_updated_vt_instances"] = summary_vt_updated

    pin_summary = engine._apply_projected_pin_offset_consistency(
        params,
        placedb,
        projection_result=projection_result,
        changed_inst_ids=changed_inst_ids,
        projection_frame=projection_frame,
    )
    if isinstance(pin_summary, dict):
        summary["runtime_pin_offset_consistency_applied"] = bool(
            pin_summary.get("runtime_pin_offset_consistency_applied", False)
        )
        summary["num_runtime_updated_pins"] = int(
            pin_summary.get("num_runtime_updated_pins", 0)
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )

    if (
        getattr(engine.op_collections, "steiner_topo_op", None) is not None
        and getattr(engine.op_collections, "pin_pos_op", None) is not None
        and hasattr(engine, "pos")
    ):
        topology = engine._refresh_live_timing_topology(engine.pos[0].data)
        update_kind = str(topology.get("topology_update_kind", "unknown"))
        summary["runtime_topology_refresh_kind"] = update_kind
        summary["runtime_topology_rebuilt"] = update_kind == "topology_rebuild"
        summary["runtime_topology_geometry_refreshed"] = update_kind != "gr_snapshot_unchanged"

    data_collections = engine.data_collections
    data_collections.runtime_cell_state_generation = int(
        getattr(data_collections, "runtime_cell_state_generation", 0) or 0
    ) + 1
    timing_model_refresh_started_at = time.perf_counter()
    data_collections.timing_model_generation = int(
        getattr(data_collections, "timing_model_generation", 0) or 0
    ) + 1
    model = getattr(engine, "last_global_place_model", None)
    virtual_density_cache_invalidated = False
    if model is not None:
        model._timing_geometry_cache = None
        virtual_density_view = getattr(model, "_virtual_cell_density_view", None)
        density_ops = getattr(virtual_density_view, "_density_ops", None)
        if isinstance(density_ops, dict):
            density_ops.clear()
            virtual_density_cache_invalidated = True
    engine._record_real_size_transition_stage(
        "timing_model_refresh_ms",
        timing_model_refresh_started_at,
    )

    summary["runtime_refresh_applied"] = True
    summary["num_runtime_updated_instances"] = int(changed_inst_ids.numel())
    summary["runtime_cell_state_generation"] = int(
        getattr(data_collections, "runtime_cell_state_generation", 0) or 0
    )
    summary["timing_model_generation"] = int(
        getattr(data_collections, "timing_model_generation", 0) or 0
    )
    if projection_frame is not None:
        engine.last_projection_frame = replace(
            projection_frame,
            topology_generation_after=int(
                getattr(
                    getattr(engine.op_collections, "steiner_topo_op", None),
                    "topology_generation",
                    projection_frame.topology_generation_before,
                )
                or projection_frame.topology_generation_before
            ),
            timing_model_generation_after=int(
                getattr(data_collections, "timing_model_generation", 0) or 0
            ),
        )
        summary["projection_frame_state_digest"] = engine.last_projection_frame.state_digest
        summary["projection_frame_topology_generation_before"] = (
            projection_frame.topology_generation_before
        )
        summary["projection_frame_topology_generation_after"] = (
            engine.last_projection_frame.topology_generation_after
        )
        summary["projection_frame_timing_model_generation_before"] = (
            projection_frame.timing_model_generation_before
        )
        summary["projection_frame_timing_model_generation_after"] = (
            engine.last_projection_frame.timing_model_generation_after
        )
    summary["timing_cache_invalidated"] = model is not None
    summary["virtual_density_cache_invalidated"] = virtual_density_cache_invalidated
    engine._write_projection_runtime_refresh_summary(params, summary)
    logging.info(
        "applied projected runtime refresh to %d changed instances (%d pins, geometry=%s, topology=%s)",
        summary["num_runtime_updated_instances"],
        summary["num_runtime_updated_pins"],
        summary["runtime_geometry_refresh_applied"],
        summary["runtime_topology_rebuilt"],
    )
    return summary


@dataclass
class ProjectionTimingSnapshotContext:
    """Runtime views and owners temporarily used for projected-state timing."""

    data_collections: object
    op_collections: object
    model: object
    pos: torch.Tensor | None
    projection_result: object
    capture_timing_metrics: Callable
    restore_timing_metrics: Callable
    enrich_record: Callable


def compute_post_projection_timing_snapshot(
    context,
    params,
    place_record,
    legalization_record,
    projection_summary,
):
    projection_result = context.projection_result
    model = context.model
    data_collections = context.data_collections
    if projection_result is None or model is None or data_collections is None:
        return None

    size_var_getter = getattr(data_collections, "get_size_var", None)
    vt_var_getter = getattr(data_collections, "get_vt_var", None)
    if size_var_getter is None or vt_var_getter is None:
        return None

    current_size = size_var_getter()
    current_vt = vt_var_getter()
    if current_size is None or current_vt is None:
        return None

    legal_mask = projection_result.has_legal_candidate.bool()
    if projection_result.inst_ids.numel() == 0 or not torch.any(legal_mask):
        return None

    projected_size = current_size.detach().clone().float()
    legal_inst_ids = projection_result.inst_ids[legal_mask].long()
    projected_size[legal_inst_ids] = (
        projection_result.projected_size[legal_mask]
        .detach()
        .to(
            projected_size.device,
            dtype=projected_size.dtype,
        )
    )
    projected_vt = projected_vt_distribution(
        current_vt,
        legal_inst_ids,
        projection_result.projected_vt[legal_mask].detach(),
    )

    original_size_var_getter = data_collections.get_size_var
    original_vt_var_getter = data_collections.get_vt_var
    had_inst_libcell_offset = hasattr(data_collections, "inst_libcell_offset")
    had_inst_cell_id = hasattr(data_collections, "inst_cell_id")
    original_inst_libcell_offset = getattr(
        data_collections, "inst_libcell_offset", None
    )
    original_inst_cell_id = getattr(data_collections, "inst_cell_id", None)
    projected_inst_libcell_offset = None
    projected_inst_cell_id = None
    if original_inst_libcell_offset is not None:
        projected_inst_libcell_offset = (
            original_inst_libcell_offset.detach().clone().long()
        )
        projected_inst_libcell_offset[legal_inst_ids] = (
            projection_result.projected_libcell_offset[legal_mask]
            .detach()
            .to(projected_inst_libcell_offset.device)
        )
    if original_inst_cell_id is not None:
        projected_inst_cell_id = original_inst_cell_id.detach().clone().long()
        projected_inst_cell_id[legal_inst_ids] = (
            projection_result.projected_cell_id[legal_mask]
            .detach()
            .to(projected_inst_cell_id.device)
        )

    timing_op = getattr(context.op_collections, "timing_propagation_op", None)
    original_last_timing_metrics = context.capture_timing_metrics()
    original_timing_scalars = {}
    if timing_op is not None:
        for name in (
            "last_total_slew_violation",
            "last_total_cap_violation",
            "last_total_leakage",
            "last_runtime_seconds",
        ):
            original_timing_scalars[name] = getattr(timing_op, name, None)
    try:
        data_collections.get_size_var = lambda: projected_size
        data_collections.get_vt_var = lambda: projected_vt
        if projected_inst_libcell_offset is not None:
            data_collections.inst_libcell_offset = projected_inst_libcell_offset
        if projected_inst_cell_id is not None:
            data_collections.inst_cell_id = projected_inst_cell_id

        if (
            getattr(context.op_collections, "steiner_topo_op", None) is not None
            and getattr(context.op_collections, "pin_pos_op", None) is not None
            and context.pos is not None
        ):
            pin_pos_for_topology = (
                context.op_collections.pin_pos_op(context.pos)
                .detach()
                .cpu()
                .contiguous()
            )
            (
                context.data_collections.net_flat_topo_sort,
                context.data_collections.net_flat_topo_sort_start,
                context.data_collections.pin_fa,
                context.data_collections.flat_pin_to,
                context.data_collections.flat_pin_to_start,
                context.data_collections.flat_pin_from,
            ) = context.op_collections.steiner_topo_op.rebuild_tree(
                pin_pos_for_topology
            )

        pos = context.pos
        wns, tns, ws, ts = model.timing_obj(pos)

        leakage = projection_summary.get("total_projected_leakage")
        if leakage is None and timing_op is not None:
            leakage = getattr(timing_op, "last_total_leakage", None)
        record = context.enrich_record(
            {
                "wns": timing_scalar(wns),
                "tns": timing_scalar(tns),
                "timing_objective": timing_scalar(ws),
                "slew_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_slew_violation", None),
                "cap_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_cap_violation", None),
                "leakage": leakage,
                "stage_note": "projected_discrete_state_retimed_before_legalization",
            }
        )
        return {
            "record": record,
            "endpoint_payload": {
                "source": "backend_endpoint_debug",
                "available": False,
                "summary_path": None,
                "csv_path": None,
                "wns": None,
                "tns": None,
                "num_negative_endpoints": None,
                "python_wns": None,
                "python_tns": None,
            },
            "top_endpoints": [],
        }
    except Exception:
        logging.exception("failed to compute dedicated post-projection timing snapshot")
        return None
    finally:
        data_collections.get_size_var = original_size_var_getter
        data_collections.get_vt_var = original_vt_var_getter
        if had_inst_libcell_offset:
            data_collections.inst_libcell_offset = original_inst_libcell_offset
        if had_inst_cell_id:
            data_collections.inst_cell_id = original_inst_cell_id
        if original_last_timing_metrics is not None:
            context.restore_timing_metrics(original_last_timing_metrics)
        if timing_op is not None:
            for name, value in original_timing_scalars.items():
                setattr(timing_op, name, value)


@dataclass(frozen=True)
class ProjectedPinContext:
    data_collections: object
    projection_op: object
    projection_result: object
    write_summary: Callable
    record_stage: Callable
    defer_placedb_sync: Callable

def apply_projected_pin_offset_consistency(
    context,
    params,
    placedb,
    projection_result=None,
    changed_inst_ids=None,
    projection_frame=None,
):
    if projection_result is None:
        projection_result = getattr(context, "projection_result", None)
    projection_op = context.projection_op
    if projection_result is None or projection_op is None:
        return context.write_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
    if not callable(getattr(projection_op, "compute_projected_pin_offsets", None)):
        raise TypeError(
            "gate_projection_op must provide compute_projected_pin_offsets(projection_result)"
        )

    if projection_frame is not None:
        pin_apply_started_at = time.perf_counter()
        pin_ids = projection_frame.changed_pin_ids.long()
        projected_x = projection_frame.projected_pin_offset_x
        projected_y = projection_frame.projected_pin_offset_y
        if pin_ids.numel() == 0:
            context.record_stage(
                "runtime_pin_offset_apply_ms",
                pin_apply_started_at,
            )
            return context.write_summary(
                params,
                runtime_pin_offset_consistency_applied=False,
                num_runtime_updated_pins=0,
                num_runtime_updated_instances=0,
            )
        pin_device = context.data_collections.pin_offset_x.device
        pin_ids_device = pin_ids.to(device=pin_device)
        projected_x = projected_x.to(
            device=pin_device,
            dtype=context.data_collections.pin_offset_x.dtype,
        )
        projected_y = projected_y.to(
            device=pin_device,
            dtype=context.data_collections.pin_offset_y.dtype,
        )
        if bool(torch.any(pin_ids_device < 0).item()) or bool(
            torch.any(pin_ids_device >= context.data_collections.pin_offset_x.numel()).item()
        ):
            raise RuntimeError("projection frame contains an invalid changed pin id")
        with torch.no_grad():
            context.data_collections.pin_offset_x[pin_ids_device] = projected_x
            context.data_collections.pin_offset_y[pin_ids_device] = projected_y
            if getattr(context.data_collections, "original_pin_offset_x", None) is not None:
                context.data_collections.original_pin_offset_x[pin_ids_device] = projected_x.to(
                    device=context.data_collections.original_pin_offset_x.device,
                    dtype=context.data_collections.original_pin_offset_x.dtype,
                )
                context.data_collections.original_pin_offset_y[pin_ids_device] = projected_y.to(
                    device=context.data_collections.original_pin_offset_y.device,
                    dtype=context.data_collections.original_pin_offset_y.dtype,
                )
        if not context.defer_placedb_sync() and placedb is not None:
            pin_ids_cpu = pin_ids_device.detach().cpu().numpy()
            placedb.pin_offset_x[pin_ids_cpu] = projected_x.detach().cpu().numpy()
            placedb.pin_offset_y[pin_ids_cpu] = projected_y.detach().cpu().numpy()
        summary = context.write_summary(
            params,
            runtime_pin_offset_consistency_applied=True,
            num_runtime_updated_pins=int(pin_ids_device.numel()),
            num_runtime_updated_instances=int(projection_frame.changed_inst_ids.numel()),
        )
        context.record_stage(
            "runtime_pin_offset_apply_ms",
            pin_apply_started_at,
        )
        return summary

    pin_compute_started_at = time.perf_counter()
    pin_offset_result = projection_op.compute_projected_pin_offsets(projection_result)
    context.record_stage(
        "pin_offset_compute_ms",
        pin_compute_started_at,
    )
    pin_apply_started_at = time.perf_counter()
    has_projected = pin_offset_result["has_projected_pin_offset"].bool()
    if changed_inst_ids is not None:
        changed_inst_ids = changed_inst_ids.long().to(
            pin_offset_result["inst_ids"].device
        )
        if changed_inst_ids.numel() == 0:
            has_projected = torch.zeros_like(has_projected, dtype=torch.bool)
        else:
            pin_inst_ids = pin_offset_result["inst_ids"].long()
            num_nodes = int(getattr(context.data_collections, "inst_cell_id").numel())
            if bool(torch.any(pin_inst_ids < 0).item()) or bool(
                torch.any(pin_inst_ids >= num_nodes).item()
            ):
                raise RuntimeError(
                    "projected pin instance IDs are outside runtime node state"
                )
            changed_node_mask = torch.zeros(
                num_nodes,
                dtype=torch.bool,
                device=pin_inst_ids.device,
            )
            changed_node_mask[changed_inst_ids] = True
            has_projected &= changed_node_mask[pin_inst_ids]
    if not has_projected.any():
        context.record_stage(
            "runtime_pin_offset_apply_ms",
            pin_apply_started_at,
        )
        return context.write_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )

    pin_ids = pin_offset_result["pin_ids"][has_projected].long()
    projected_x = pin_offset_result["projected_pin_offset_x"][has_projected].to(
        context.data_collections.pin_offset_x.device,
        dtype=context.data_collections.pin_offset_x.dtype,
    )
    projected_y = pin_offset_result["projected_pin_offset_y"][has_projected].to(
        context.data_collections.pin_offset_y.device,
        dtype=context.data_collections.pin_offset_y.dtype,
    )

    pin_ids_device = pin_ids.to(context.data_collections.pin_offset_x.device)
    with torch.no_grad():
        context.data_collections.pin_offset_x[pin_ids_device] = projected_x
        context.data_collections.pin_offset_y[pin_ids_device] = projected_y

    if not context.defer_placedb_sync():
        pin_ids_cpu = pin_ids.detach().cpu().numpy()
        projected_x_cpu = projected_x.detach().cpu().numpy()
        projected_y_cpu = projected_y.detach().cpu().numpy()
        placedb.pin_offset_x[pin_ids_cpu] = projected_x_cpu
        placedb.pin_offset_y[pin_ids_cpu] = projected_y_cpu

    if getattr(context.data_collections, "original_pin_offset_x", None) is not None:
        with torch.no_grad():
            context.data_collections.original_pin_offset_x[pin_ids_device] = projected_x
            context.data_collections.original_pin_offset_y[pin_ids_device] = projected_y

    summary = context.write_summary(
        params,
        runtime_pin_offset_consistency_applied=True,
        num_runtime_updated_pins=int(pin_ids.numel()),
        num_runtime_updated_instances=int(
            torch.unique(pin_offset_result["inst_ids"][has_projected]).numel()
        ),
    )
    context.record_stage(
        "runtime_pin_offset_apply_ms",
        pin_apply_started_at,
    )
    logging.info(
        "applied projected pin-offset consistency to %d pins across %d instances",
        summary["num_runtime_updated_pins"],
        summary["num_runtime_updated_instances"],
    )
    return summary
