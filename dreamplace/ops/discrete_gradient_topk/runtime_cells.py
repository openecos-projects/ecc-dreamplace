"""Synchronize discrete size/VT actions with runtime geometry and timing."""

import numpy as np
import torch

from .vt_commit import DiscreteVtCommit
from .runtime_timing_arcs import sync_timing_arc_lut_rows


def sync_runtime_cells(params, summary, *, data_collections, placedb, timing_op, model, direct_joint, geometry_window=None):
    if not isinstance(summary, dict):
        return {
            "runtime_cell_state_synced": False,
            "runtime_synced_instances": 0,
            "runtime_sync_reason": "summary_unavailable",
        }
    inst_ids = summary.get("applied_instance_ids") or []
    cell_ids = summary.get("applied_cell_ids") or []
    if not inst_ids or not cell_ids or len(inst_ids) != len(cell_ids):
        return {
            "runtime_cell_state_synced": False,
            "runtime_synced_instances": 0,
            "runtime_sync_reason": "no_applied_cells",
        }

    if data_collections is None:
        return {
            "runtime_cell_state_synced": False,
            "runtime_synced_instances": 0,
            "runtime_sync_reason": "data_collections_unavailable",
        }

    if direct_joint:
        if len(getattr(placedb, "regions", ()) or ()) > 0:
            raise RuntimeError(
                "milestone sizing density refresh does not support fence regions"
            )
        if bool(getattr(params, "routability_opt_flag", False)):
            raise RuntimeError(
                "milestone sizing density refresh does not support routability area adjustment"
            )

    target_device = (
        data_collections.inst_cell_id.device
        if getattr(data_collections, "inst_cell_id", None) is not None
        else None
    )
    inst_tensor = torch.as_tensor(inst_ids, dtype=torch.long, device=target_device)
    cell_tensor = torch.as_tensor(cell_ids, dtype=torch.long, device=target_device)
    synced_instances = int(inst_tensor.numel())
    if int(torch.unique(inst_tensor).numel()) != synced_instances:
        raise RuntimeError("discrete sizing selected an instance more than once")

    vt_commit = DiscreteVtCommit.from_summary(
        data_collections,
        cell_tensor,
        summary,
    )

    node_size_x = getattr(data_collections, "node_size_x", None)
    node_size_y = getattr(data_collections, "node_size_y", None)
    node_areas = getattr(data_collections, "node_areas", None)
    flat_width = getattr(data_collections, "flat_libcell_width", None)
    flat_height = getattr(data_collections, "flat_libcell_height", None)
    geometry_available = all(
        value is not None
        for value in (node_size_x, node_size_y, node_areas, flat_width, flat_height)
    )
    if direct_joint and not geometry_available:
        raise RuntimeError(
            "milestone sizing requires exported library width/height geometry"
        )

    num_movable_nodes = int(getattr(placedb, "num_movable_nodes", 0) or 0)
    old_movable_area = (
        None
        if node_areas is None or geometry_window is not None
        else float(
            node_areas[:num_movable_nodes]
            .detach()
            .double()
            .sum()
            .cpu()
            .item()
        )
    )
    changed_pin_ids = None
    offsets = None
    with torch.no_grad():
        if getattr(data_collections, "inst_cell_id", None) is not None:
            data_collections.inst_cell_id[inst_tensor] = cell_tensor.to(
                data_collections.inst_cell_id.device,
                dtype=data_collections.inst_cell_id.dtype,
            )

        vt_commit.apply_data_collections(inst_tensor, cell_tensor)

        if (
            getattr(data_collections, "inst_libcell_offset", None) is not None
            and getattr(data_collections, "main_id_2_cell_id_start", None) is not None
            and getattr(data_collections, "flat_libcell_info", None) is not None
        ):
            flat_info = data_collections.flat_libcell_info
            main_ids = flat_info[cell_tensor.to(flat_info.device), 1].long()
            starts = data_collections.main_id_2_cell_id_start[main_ids].long()
            offsets = cell_tensor.to(starts.device) - starts
            data_collections.inst_libcell_offset[inst_tensor.to(data_collections.inst_libcell_offset.device)] = offsets.to(
                data_collections.inst_libcell_offset.device,
                dtype=data_collections.inst_libcell_offset.dtype,
            )
        else:
            offsets = None

        if geometry_available:
            width_values = flat_width[
                cell_tensor.to(device=flat_width.device)
            ].to(device=node_size_x.device, dtype=node_size_x.dtype)
            height_values = flat_height[
                cell_tensor.to(device=flat_height.device)
            ].to(device=node_size_y.device, dtype=node_size_y.dtype)
            padding_x = float(getattr(params, "cell_padding_x", 0.0) or 0.0)
            if padding_x > 0.0:
                width_values = width_values + 2.0 * padding_x
            node_indices_x = inst_tensor.to(device=node_size_x.device)
            node_indices_y = inst_tensor.to(device=node_size_y.device)
            node_size_x[node_indices_x] = width_values
            node_size_y[node_indices_y] = height_values
            node_areas[inst_tensor.to(device=node_areas.device)] = (
                width_values.to(device=node_areas.device, dtype=node_areas.dtype)
                * height_values.to(device=node_areas.device, dtype=node_areas.dtype)
            )
            if geometry_window is not None:
                geometry_window.apply_sizes(node_indices_x, width_values, height_values)

            inst_size_init = getattr(data_collections, "inst_size_init", None)
            flat_info = getattr(data_collections, "flat_libcell_info", None)
            if inst_size_init is not None and flat_info is not None:
                exact_sizes = flat_info[
                    cell_tensor.to(device=flat_info.device), 2
                ].to(device=inst_size_init.device, dtype=inst_size_init.dtype)
                inst_size_init[inst_tensor.to(device=inst_size_init.device)] = exact_sizes

            pin2node = getattr(data_collections, "pin2node_map", None)
            pin_lib_offset = getattr(
                data_collections,
                "pin_2_libpin_offset",
                None,
            )
            if pin2node is not None and pin_lib_offset is not None:
                pin_nodes = pin2node.long()
                valid_node = (pin_nodes >= 0) & (
                    pin_nodes < int(data_collections.inst_cell_id.numel())
                )
                safe_pin_nodes = pin_nodes.clamp(
                    min=0,
                    max=max(int(data_collections.inst_cell_id.numel()) - 1, 0),
                )
                selected_nodes = torch.zeros(
                    int(data_collections.inst_cell_id.numel()),
                    dtype=torch.bool,
                    device=pin_nodes.device,
                )
                selected_nodes[
                    inst_tensor.to(device=selected_nodes.device)
                ] = True
                changed_pin_mask = (
                    valid_node
                    & selected_nodes[safe_pin_nodes]
                    & (pin_lib_offset.to(device=pin_nodes.device) >= 0)
                )
                changed_pin_ids = torch.nonzero(
                    changed_pin_mask,
                    as_tuple=False,
                ).flatten()
                if changed_pin_ids.numel() > 0:
                    changed_nodes = pin_nodes[changed_pin_ids]
                    changed_cells = data_collections.inst_cell_id[
                        changed_nodes.to(device=data_collections.inst_cell_id.device)
                    ].long()
                    starts = data_collections.cell_id_2_libpin_id_start[
                        changed_cells.to(
                            device=data_collections.cell_id_2_libpin_id_start.device
                        )
                    ].long()
                    libpin_ids = starts + pin_lib_offset[
                        changed_pin_ids.to(device=pin_lib_offset.device)
                    ].long().to(device=starts.device)
                    flat_pin_x = data_collections.flat_lib_pin_offset_x
                    flat_pin_y = data_collections.flat_lib_pin_offset_y
                    if not bool(
                        (
                            (libpin_ids >= 0)
                            & (libpin_ids < int(flat_pin_x.numel()))
                            & (libpin_ids < int(flat_pin_y.numel()))
                        ).all()
                    ):
                        raise RuntimeError(
                            "resized instance pin maps outside library pin geometry"
                        )
                    pin_x = flat_pin_x[
                        libpin_ids.to(device=flat_pin_x.device)
                    ].to(
                        device=data_collections.pin_offset_x.device,
                        dtype=data_collections.pin_offset_x.dtype,
                    )
                    pin_y = flat_pin_y[
                        libpin_ids.to(device=flat_pin_y.device)
                    ].to(
                        device=data_collections.pin_offset_y.device,
                        dtype=data_collections.pin_offset_y.dtype,
                    )
                    if padding_x > 0.0:
                        pin_x = pin_x + padding_x
                    data_collections.pin_offset_x[
                        changed_pin_ids.to(
                            device=data_collections.pin_offset_x.device
                        )
                    ] = pin_x
                    data_collections.pin_offset_y[
                        changed_pin_ids.to(
                            device=data_collections.pin_offset_y.device
                        )
                    ] = pin_y
                    if geometry_window is not None:
                        geometry_window.apply_pins(changed_pin_ids)

        data_collections.runtime_cell_state_generation = int(
            getattr(data_collections, "runtime_cell_state_generation", 0) or 0
        ) + 1

    timing_arc_sync = sync_timing_arc_lut_rows(data_collections, placedb, timing_op, inst_tensor)

    if placedb is not None:
        inst_cpu = inst_tensor.detach().cpu().numpy()
        cell_cpu = cell_tensor.detach().cpu().numpy()
        if getattr(placedb, "inst_cell_id", None) is not None:
            placedb.inst_cell_id[inst_cpu] = cell_cpu
        vt_commit.apply_placedb(placedb, inst_tensor)
        if getattr(placedb, "inst_libcell_offset", None) is not None:
            offset_cpu = None
            if "offsets" in locals() and offsets is not None:
                offset_cpu = offsets.detach().cpu().numpy()
            elif getattr(data_collections, "main_id_2_cell_id_start", None) is not None:
                starts_cpu = (
                    data_collections.main_id_2_cell_id_start.detach().cpu().numpy()
                )
                flat_info_cpu = data_collections.flat_libcell_info.detach().cpu().numpy()
                offset_cpu = []
                for cell_id in cell_cpu:
                    main_id = int(round(float(flat_info_cpu[int(cell_id), 1])))
                    offset_cpu.append(int(cell_id) - int(starts_cpu[main_id]))
                offset_cpu = np.asarray(offset_cpu, dtype=placedb.inst_libcell_offset.dtype)
            if offset_cpu is not None:
                placedb.inst_libcell_offset[inst_cpu] = offset_cpu

        if geometry_available:
            width_cpu = node_size_x[inst_tensor.to(node_size_x.device)].detach().cpu().numpy()
            height_cpu = node_size_y[inst_tensor.to(node_size_y.device)].detach().cpu().numpy()
            placedb.node_size_x[inst_cpu] = width_cpu
            placedb.node_size_y[inst_cpu] = height_cpu
            if getattr(placedb, "inst_size_init", None) is not None:
                placedb.inst_size_init[inst_cpu] = (
                    data_collections.inst_size_init[
                        inst_tensor.to(data_collections.inst_size_init.device)
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
            if changed_pin_ids is not None and changed_pin_ids.numel() > 0:
                pin_cpu = changed_pin_ids.detach().cpu().numpy()
                placedb.pin_offset_x[pin_cpu] = (
                    data_collections.pin_offset_x[
                        changed_pin_ids.to(data_collections.pin_offset_x.device)
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                placedb.pin_offset_y[pin_cpu] = (
                    data_collections.pin_offset_y[
                        changed_pin_ids.to(data_collections.pin_offset_y.device)
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

    new_movable_area = (
        None
        if node_areas is None or geometry_window is not None
        else float(
            node_areas[:num_movable_nodes]
            .detach()
            .double()
            .sum()
            .cpu()
            .item()
        )
    )
    area_delta = (
        0.0
        if old_movable_area is None or new_movable_area is None
        else new_movable_area - old_movable_area
    )
    if placedb is not None and new_movable_area is not None:
        placedb.total_movable_node_area = new_movable_area

    size_var_getter = getattr(data_collections, "get_size_var", None)
    size_var = size_var_getter() if callable(size_var_getter) else None
    applied_sizes = summary.get("applied_sizes") or []
    size_gap = 0.0
    if size_var is not None and applied_sizes:
        expected_sizes = torch.as_tensor(
            applied_sizes,
            dtype=size_var.dtype,
            device=size_var.device,
        )
        actual_sizes = size_var[inst_tensor.to(device=size_var.device)]
        size_gap = float((actual_sizes - expected_sizes).abs().max().cpu().item())
        if size_gap > 1.0e-4:
            raise RuntimeError(
                "discrete sizing logits do not match selected legal sizes"
            )

    vt_gap = vt_commit.probability_gap(inst_tensor)

    virtual_density_cache_invalidated = False
    if model is not None:
        model._timing_geometry_cache = None
        virtual_density_view = getattr(model, "_virtual_cell_density_view", None)
        density_ops = getattr(virtual_density_view, "_density_ops", None)
        if isinstance(density_ops, dict):
            density_ops.clear()
            virtual_density_cache_invalidated = True

    return {
        "runtime_cell_state_synced": True,
        "runtime_synced_instances": synced_instances,
        "runtime_sync_reason": "applied",
        "runtime_cell_state_generation": int(
            getattr(data_collections, "runtime_cell_state_generation", 0) or 0
        ),
        "virtual_density_cache_invalidated": virtual_density_cache_invalidated,
        "timing_cache_generation": int(
            getattr(data_collections, "runtime_cell_state_generation", 0) or 0
        ),
        "runtime_geometry_synced": bool(geometry_available),
        "runtime_pin_geometry_synced": int(
            0 if changed_pin_ids is None else changed_pin_ids.numel()
        ),
        "runtime_pin_cap_view": (
            "dynamic_from_inst_libcell_offset"
            if all(
                getattr(data_collections, name, None) is not None
                for name in (
                    "inst_libcell_offset",
                    "inst_main_id",
                    "pin2node_map",
                    "pin_2_libpin_offset",
                    "main_id_2_cell_id_start",
                    "cell_id_2_libpin_id_start",
                    "flat_lib_pin_cap",
                )
            )
            else "unavailable"
        ),
        "runtime_timing_arc_lut_sync": timing_arc_sync,
        "movable_area_before_internal": old_movable_area,
        "movable_area_after_internal": new_movable_area,
        "area_delta_internal": float(area_delta),
        "density_visible_area_delta_internal": float(area_delta),
        "legal_size_max_abs_gap": float(size_gap),
        "legal_vt_max_abs_gap": float(vt_gap),
        "timing_cache_invalidated": model is not None,
    }
