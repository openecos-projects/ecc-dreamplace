"""Refreshed segment z-gradient and its finite/generation boundary."""
from dataclasses import dataclass
from typing import Callable
import os
import json
import torch

@dataclass(frozen=True)
class SegmentGradientContext:
    joint_coordinator: object
    data_collections: object
    digest: Callable
    profile_enabled: Callable
    profile_clock: Callable
    refresh_topology: Callable

def compute_refreshed_segment_buffer_gradient(
    context,
    *,
    params,
    model,
    pos,
    event,
    refresh_topology=True,
    frame="post_placement_post_sizing_x_next",
    expected_topology_generation=None,
):
    coordinator = context.joint_coordinator
    state = coordinator.buffering_lane._current_state()
    z_param = state.z_param
    if not coordinator.segment_route_b_enabled:
        return {
            "status": "disabled",
            "reason": "joint_segment_route_b_disabled",
            "event_id": int(event["event_id"]),
            "state_digest": context.digest(
                (("pos", pos), ("z_param", z_param))
            ),
        }

    if z_param.grad is not None:
        z_param.grad = None
    size_logits = getattr(context.data_collections, "size_logits", None)
    if size_logits is not None and size_logits.grad is not None:
        size_logits.grad = None
    pos_before = pos.detach().clone()
    profile_enabled = context.profile_enabled(params)

    def profile_clock():
        return context.profile_clock(
            profile_enabled,
            pos,
        )

    started_at = profile_clock()
    if refresh_topology:
        context.refresh_topology(pos)
    timing_started_at = profile_clock()
    wns, tns, ws, ts = model.timing_obj(pos)
    timing_finished_at = profile_clock()
    timing_loss = model._timing_loss(wns, tns, ws, ts)
    loss_finished_at = profile_clock()
    z_grad = torch.autograd.grad(
        timing_loss,
        z_param,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    autograd_finished_at = profile_clock()
    if z_grad is None or not bool(torch.isfinite(z_grad).all()):
        finite = (
            torch.zeros_like(z_param, dtype=torch.bool)
            if z_grad is None
            else torch.isfinite(z_grad)
        )
        bad_indices = torch.nonzero(~finite, as_tuple=False).flatten()
        sample_indices = bad_indices[:32]
        sample_rows = []
        for index in sample_indices.detach().cpu().tolist():
            row = {"state_index": int(index)}
            for key, values in (
                ("segment_id", getattr(state, "segment_ids", None)),
                ("net_id", getattr(state, "segment_net_id", None)),
                ("parent_node_id", getattr(state, "parent_node_id", None)),
                ("child_node_id", getattr(state, "child_node_id", None)),
                ("z", getattr(state, "z_param", None)),
                ("bsu_index", getattr(state, "bsu_index_param", None)),
            ):
                if torch.is_tensor(values) and int(index) < int(values.numel()):
                    value = values[int(index)].detach().cpu().item()
                    row[key] = float(value) if values.dtype.is_floating_point else int(value)
            if z_grad is not None:
                row["gradient"] = float(z_grad[int(index)].detach().cpu().item())
            for key, attr in (
                ("per_size_input_cap", "buffer_segment_count_per_size_input_cap"),
                ("per_size_delay", "buffer_segment_count_per_size_delay"),
                (
                    "per_size_output_slew",
                    "buffer_segment_count_per_size_output_slew",
                ),
            ):
                values = getattr(context.data_collections, attr, None)
                if torch.is_tensor(values) and int(values.shape[0]) > 0:
                    row_index = 0 if int(values.shape[0]) == 1 else int(index)
                    if row_index >= int(values.shape[0]):
                        continue
                    tensor_row = values[row_index].detach().float().cpu()
                    row[key] = [float(value) for value in tensor_row.tolist()]
                    row[f"{key}_finite"] = bool(torch.isfinite(tensor_row).all())
            if hasattr(state, "segment_record"):
                record = state.segment_record(
                    int(index), materialization_kind="selected"
                )
                for key in (
                    "parent_x_dbu",
                    "parent_y_dbu",
                    "child_x_dbu",
                    "child_y_dbu",
                ):
                    if record.get(key) is not None:
                        row[key] = int(record[key])
                edge_rc = dict(record.get("edge_rc", {}) or {})
                if edge_rc:
                    row["edge_resistance"] = float(edge_rc.get("r", 0.0))
                    row["edge_capacitance"] = float(edge_rc.get("c", 0.0))
            sample_rows.append(row)
        failure = {
            "artifact": "segment_count_nonfinite_gradient_failure",
            "artifact_version": 2,
            "event_id": int(event["event_id"]),
            "frame": str(frame),
            "gradient_count": int(z_param.numel()),
            "nonfinite_count": int(bad_indices.numel()),
            "samples": sample_rows,
        }
        failure_path = os.path.join(
            str(params.result_dir),
            f"{params.design_name()}_segment_count_nonfinite_gradient.json",
        )
        os.makedirs(os.path.dirname(os.path.abspath(failure_path)), exist_ok=True)
        with open(failure_path, "w", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2, sort_keys=True)
            stream.write("\n")
        raise RuntimeError(
            "refreshed segment z gradient is missing or non-finite: "
            f"count={int(bad_indices.numel())}, artifact={failure_path}"
        )
    if not torch.equal(pos.detach(), pos_before):
        raise RuntimeError("refreshed segment timing pass changed placement")
    z_param.grad = z_grad.detach().clone()
    model._timing_geometry_cache = None
    expected_generation = expected_topology_generation
    if expected_generation is None:
        expected_generation = (
            coordinator.segment_milestones.frozen_topology_generation
        )
    topology = context.data_collections.buffering_timing_topology
    actual_generation = int(topology.get("topology_generation", 0) or 0)
    if actual_generation != int(expected_generation):
        raise RuntimeError("refreshed segment timing changed frozen topology generation")
    state_digest = context.digest(
        (
            ("pos", pos),
            ("size_logits", size_logits),
            ("inst_cell_id", context.data_collections.inst_cell_id),
            ("z_param", z_param),
        )
    )
    finished_at = profile_clock()
    result = {
        "status": "ready",
        "event_id": int(event["event_id"]),
        "iteration": int(event["iteration"]),
        "frame": str(frame),
        "state_digest": state_digest,
        "topology_generation": actual_generation,
        "runtime_cell_state_generation": int(
            getattr(context.data_collections, "runtime_cell_state_generation", 0)
            or 0
        ),
        "wns": float(wns),
        "tns": float(tns),
        "timing_loss": float(timing_loss.detach().cpu().item()),
        "gradient_norm": float(z_grad.detach().float().norm().cpu().item()),
        "gradient_nonzero_count": int((z_grad != 0.0).sum().cpu().item()),
        "runtime_ms": float((finished_at - started_at) * 1000.0),
    }
    if profile_enabled:
        result["runtime_profile"] = {
            "enabled": True,
            "synchronized": bool(pos.is_cuda),
            "timing_obj_ms": float(
                (timing_finished_at - timing_started_at) * 1000.0
            ),
            "timing_loss_ms": float(
                (loss_finished_at - timing_finished_at) * 1000.0
            ),
            "autograd_ms": float(
                (autograd_finished_at - loss_finished_at) * 1000.0
            ),
            "post_gradient_validation_ms": float(
                (finished_at - autograd_finished_at) * 1000.0
            ),
            "total_ms": float((finished_at - started_at) * 1000.0),
        }
    return result


