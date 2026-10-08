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
