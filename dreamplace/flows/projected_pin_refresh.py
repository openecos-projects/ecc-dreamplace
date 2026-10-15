"""Apply projected pin geometry to the existing runtime and PyDB views.

Call-scoped data and projection operator; original/current offsets stay aliased
to their owners. Artifact callbacks only observe the completed application.
"""
from dataclasses import dataclass
from typing import Callable
import time
import logging
import torch

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
