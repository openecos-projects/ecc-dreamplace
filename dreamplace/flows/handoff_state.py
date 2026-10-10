"""State values used when a topology mutation requires continuation.

This module owns the value-level handoff policy and schedule snapshot rules.
Database mutation and placer reconstruction stay with their existing owners.
"""

import copy
import inspect
import logging
from numbers import Integral

import numpy as np
import torch


def resolve_restart_policy(params):
    policy = getattr(params, "handoff_restart_policy", None)
    if policy is None or policy == "":
        return "cold"
    if policy in {"cold", "warm_schedule"}:
        return policy
    raise RuntimeError("unknown handoff_restart_policy %s" % policy)


def clone_restart_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def capture_warm_schedule_state(model):
    state = {}
    if model is None:
        return state
    for field_name in (
        "density_weight",
        "density_weight_u",
        "density_weight_step_size",
    ):
        if hasattr(model, field_name):
            state[field_name] = clone_restart_value(getattr(model, field_name))
    return state


def _shape_tuple(value):
    shape = getattr(value, "shape", None)
    return () if shape is None else tuple(shape)


def apply_warm_schedule_state(model, state):
    report = {"applied": [], "skipped": []}
    if model is None or not state:
        return report
    for field_name in (
        "density_weight",
        "density_weight_u",
        "density_weight_step_size",
    ):
        if field_name not in state:
            continue
        if not hasattr(model, field_name):
            report["skipped"].append(field_name)
            continue
        target_value = getattr(model, field_name)
        source_value = state[field_name]
        if _shape_tuple(target_value) != _shape_tuple(source_value):
            report["skipped"].append(field_name)
            continue
        if isinstance(target_value, torch.Tensor) and isinstance(
            source_value, torch.Tensor
        ):
            with torch.no_grad():
                target_value.copy_(source_value.to(target_value.device))
        elif isinstance(target_value, torch.Tensor):
            report["skipped"].append(field_name)
            continue
        elif isinstance(target_value, np.ndarray):
            if not isinstance(source_value, np.ndarray):
                report["skipped"].append(field_name)
                continue
            setattr(model, field_name, clone_restart_value(source_value))
        else:
            setattr(model, field_name, clone_restart_value(source_value))
        report["applied"].append(field_name)
    return report


def restart_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.numpy().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (Integral, float)):
        return float(value)
    return value


def restart_probe_payload(policy, phase, handoff_seq, metric):
    hpwl = restart_scalar(getattr(metric, "hpwl", None))
    overflow = restart_scalar(getattr(metric, "overflow", None))
    if isinstance(overflow, list):
        overflow = max(overflow) if overflow else None
    density_weight = restart_scalar(getattr(metric, "density_weight", None))
    return {
        "restart_policy": policy,
        "phase": phase,
        "handoff_seq": handoff_seq,
        "hpwl": hpwl,
        "overflow": overflow,
        "density_weight": density_weight,
    }


def build_continuation_seed(
    placedb, handoff_result, pos, handoff_seq, *, pos_to_numpy, decode_node_name
):
    pos = pos_to_numpy(pos)
    source_counts = handoff_result.get("_source_placement_counts", {}) or {}

    def _count(name, fallback_attr, default=None):
        if name in source_counts:
            return int(source_counts[name])
        try:
            return int(getattr(placedb, fallback_attr))
        except (AttributeError, TypeError, ValueError):
            if default is not None:
                return default
            raise

    num_nodes = _count("num_nodes", "num_nodes")
    num_movable_nodes = _count("num_movable_nodes", "num_movable_nodes")
    num_physical_nodes = _count("num_physical_nodes", "num_physical_nodes")
    num_filler_nodes = _count("num_filler_nodes", "num_filler_nodes", 0)
    filler_end = min(num_nodes, num_physical_nodes + num_filler_nodes)
    if "movable_node_names" not in source_counts:
        source_counts = dict(source_counts)
        try:
            node_names = getattr(placedb, "node_names")
            source_counts["movable_node_names"] = [
                decode_node_name(name) for name in list(node_names)[:num_movable_nodes]
            ]
        except (AttributeError, TypeError):
            pass
    if pos.size < num_nodes * 2:
        raise RuntimeError(
            "live placement position length %d is smaller than expected %d"
            % (pos.size, num_nodes * 2)
        )

    session = getattr(placedb, "handoff_session", None)
    seed = {
        "movable_x": pos[0:num_movable_nodes].tolist(),
        "movable_y": pos[num_nodes : num_nodes + num_movable_nodes].tolist(),
        "filler_x": pos[num_physical_nodes:filler_end].tolist(),
        "filler_y": pos[
            num_nodes + num_physical_nodes : num_nodes + filler_end
        ].tolist(),
        "source_handoff_seq": handoff_result.get("handoff_seq", handoff_seq),
        "source_snapshot_fingerprint": handoff_result.get(
            "pre_snapshot_fingerprint",
            handoff_result.get("post_snapshot_fingerprint"),
        ),
        "source_topology_epoch": getattr(session, "topology_epoch", None),
    }
    if "movable_node_names" in source_counts:
        seed["movable_node_names"] = list(source_counts["movable_node_names"])[
            :num_movable_nodes
        ]
    if hasattr(placedb, "overflow_reference_area"):
        seed["overflow_reference_area"] = placedb.overflow_reference_area
    return seed


def restart_after_topology_sync(
    params,
    placedb,
    handoff_result,
    pos=None,
    handoff_seq=None,
    model=None,
    *,
    prepare_restart_state,
    create_placer,
):
    sync_contract = handoff_result.get("sync_contract", {}) or {}
    mutation_kind = sync_contract.get("mutation_kind") or handoff_result.get(
        "mutation_kind"
    )
    if mutation_kind in ("no_mutation", "placement_only"):
        placedb.pending_continuation_seed = None
        placedb.pending_warm_schedule_state = None
        return None

    requires_runtimedb_rebuild = bool(
        sync_contract.get(
            "requires_runtimedb_rebuild",
            handoff_result.get("requires_runtimedb_rebuild", False),
        )
    )
    if not requires_runtimedb_rebuild:
        placedb.pending_continuation_seed = None
        placedb.pending_warm_schedule_state = None
        return None

    logging.info(
        "Restart DreamPlace runtimedb after OpenROAD topology sync: old=%s new=%s",
        sync_contract.get("old_counts"),
        sync_contract.get("new_counts"),
    )

    restart_state = prepare_restart_state(
        params,
        placedb,
        handoff_result,
        pos=pos,
        handoff_seq=handoff_seq,
        model=model,
    )
    logging.info("handoff restart_state=%s", restart_state)
    return create_placer(params, placedb)


def call_restart_after_topology_sync(
    restart_fn,
    params,
    placedb,
    handoff_result,
    pos=None,
    handoff_seq=None,
    model=None,
):
    signature = inspect.signature(restart_fn)
    accepts_model = "model" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {"pos": pos, "handoff_seq": handoff_seq}
    if accepts_model:
        kwargs["model"] = model
    return restart_fn(params, placedb, handoff_result, **kwargs)
