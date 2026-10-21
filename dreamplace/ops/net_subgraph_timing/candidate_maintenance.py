"""Native incremental maintenance for candidate buffer geometry."""

import importlib

import torch


class CandidateGeometryRefreshError(RuntimeError):
    """A dirty candidate row could not be refreshed in the current topology."""


def _native():
    return importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")


def _cpu_integer(values, *, name):
    if values is None:
        return torch.empty(0, dtype=torch.long)
    if torch.is_tensor(values):
        tensor = values.detach().to(device="cpu", dtype=torch.long)
        return tensor.reshape(-1).contiguous()
    return torch.as_tensor(values, dtype=torch.long).reshape(-1).contiguous()


def _cpu_floating(values, *, name):
    if torch.is_tensor(values):
        tensor = values.detach().to(device="cpu")
        if tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.to(dtype=torch.float64)
    else:
        tensor = torch.as_tensor(values, dtype=torch.float64)
    return tensor.reshape(-1).contiguous()


def compute_candidate_dirty_nets(
    *,
    changed_instance_ids,
    moved_instance_ids,
    accepted_net_ids,
    candidate_net_ids,
    flat_node2pin_map,
    flat_node2pin_start_map,
    pin2net_map,
):
    """Map changed cells and accepted buffer nets to dirty candidate nets."""

    return _native().compute_candidate_dirty_nets(
        _cpu_integer(changed_instance_ids, name="changed_instance_ids"),
        _cpu_integer(moved_instance_ids, name="moved_instance_ids"),
        _cpu_integer(accepted_net_ids, name="accepted_net_ids"),
        _cpu_integer(candidate_net_ids, name="candidate_net_ids"),
        _cpu_integer(flat_node2pin_map, name="flat_node2pin_map"),
        _cpu_integer(flat_node2pin_start_map, name="flat_node2pin_start_map"),
        _cpu_integer(pin2net_map, name="pin2net_map"),
    )


def active_candidate_net_ids(state):
    """Return the candidate nets with at least one accepted activation."""

    candidate_net_id = getattr(state, "candidate_net_id", None)
    activation = getattr(state, "activation_param", None)
    if candidate_net_id is None or activation is None:
        return []
    activation = activation.detach()
    active = activation > 0 if activation.ndim == 1 else activation.sum(dim=1) > 0
    if active.numel() != candidate_net_id.numel():
        raise CandidateGeometryRefreshError(
            "candidate activation and net domains have different lengths"
        )
    if not bool(active.any().item()):
        return []
    return (
        torch.unique(candidate_net_id.detach()[active], sorted=True)
        .to(device="cpu", dtype=torch.long)
        .tolist()
    )


def collect_candidate_dirty_nets(
    *,
    state,
    lane,
    placedb,
    data,
    changed_instance_ids=(),
    accepted_net_ids=(),
):
    """Collect native dirty candidate nets shared by placement and diff-opt."""

    if state is None or not hasattr(state, "activation_param"):
        return None, {"status": "skipped", "reason": "non_candidate_state"}
    changed = sorted({int(value) for value in changed_instance_ids or ()})
    accepted = sorted({int(value) for value in accepted_net_ids or ()})
    if not changed and not accepted:
        return [], {
            "status": "ok",
            "reason": "no_pending_changes",
            "counts": {
                "changed_instance_count": 0,
                "accepted_net_count": 0,
                "dirty_net_count": 0,
            },
        }

    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    if isinstance(payload, dict) and payload.get("candidate_net_id") is not None:
        candidate_net_ids = payload["candidate_net_id"]
    elif hasattr(state, "candidate_net_id"):
        candidate_net_ids = state.candidate_net_id
    else:
        candidate_net_ids = None

    def all_candidate_net_ids():
        if candidate_net_ids is None:
            return None
        return torch.unique(
            torch.as_tensor(candidate_net_ids).detach().cpu().long(),
            sorted=True,
        ).tolist()

    owners = (placedb, data)

    def first_value(*names):
        for owner in owners:
            if owner is None:
                continue
            for name in names:
                value = getattr(owner, name, None)
                if value is not None:
                    return value
        return None

    flat_node2pin = first_value("flat_node2pin_map")
    flat_start = first_value("flat_node2pin_start_map")
    pin2net = first_value("pin2net_map")
    if flat_node2pin is None or flat_start is None or pin2net is None:
        dirty_net_ids = all_candidate_net_ids()
        return dirty_net_ids, {
            "status": "conservative_all" if dirty_net_ids is not None else "fallback",
            "reason": "missing_incident_pin_csr",
            "dirty_net_ids": list(dirty_net_ids or ()),
            "pending_dirty_instance_count": len(changed),
            "pending_dirty_net_count": len(accepted),
        }

    try:
        result = compute_candidate_dirty_nets(
            changed_instance_ids=changed,
            moved_instance_ids=(),
            accepted_net_ids=accepted,
            candidate_net_ids=candidate_net_ids,
            flat_node2pin_map=flat_node2pin,
            flat_node2pin_start_map=flat_start,
            pin2net_map=pin2net,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        dirty_net_ids = all_candidate_net_ids()
        return dirty_net_ids, {
            "status": "conservative_all" if dirty_net_ids is not None else "fallback",
            "reason": str(exc),
            "dirty_net_ids": list(dirty_net_ids or ()),
            "pending_dirty_instance_count": len(changed),
            "pending_dirty_net_count": len(accepted),
        }
    dirty_net_ids = result["dirty_net_ids"].detach().cpu().tolist()
    return dirty_net_ids, {
        "status": "ok",
        "dirty_net_ids": list(dirty_net_ids),
        "counts": {key: int(value) for key, value in dict(result.get("counts", {}) or {}).items()},
        "metadata": dict(result.get("metadata", {}) or {}),
    }


def refresh_candidate_geometry_rows(
    *,
    payload,
    dirty_net_ids,
    node_x_dbu,
    node_y_dbu,
    all_candidates=False,
):
    """Refresh packed candidate coordinates without rebuilding its topology."""

    required = (
        "candidate_net_id",
        "candidate_parent_node_id",
        "candidate_child_node_id",
        "candidate_tree_node_id",
        "candidate_is_segment",
        "candidate_split_ratio",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise CandidateGeometryRefreshError(
            "candidate payload lacks native geometry columns: " + ", ".join(missing)
        )
    result = _native().refresh_candidate_geometry(
        _cpu_integer(dirty_net_ids, name="dirty_net_ids"),
        _cpu_integer(payload["candidate_net_id"], name="candidate_net_id"),
        _cpu_integer(
            payload["candidate_parent_node_id"],
            name="candidate_parent_node_id",
        ),
        _cpu_integer(
            payload["candidate_child_node_id"],
            name="candidate_child_node_id",
        ),
        _cpu_integer(
            payload["candidate_tree_node_id"],
            name="candidate_tree_node_id",
        ),
        _cpu_integer(
            payload["candidate_is_segment"],
            name="candidate_is_segment",
        ),
        _cpu_floating(
            payload["candidate_split_ratio"],
            name="candidate_split_ratio",
        ),
        _cpu_floating(node_x_dbu, name="node_x_dbu"),
        _cpu_floating(node_y_dbu, name="node_y_dbu"),
        bool(all_candidates),
    )
    invalid = result["invalid_candidate_rows"].tolist()
    if invalid:
        raise CandidateGeometryRefreshError(
            "candidate geometry refresh found invalid dirty rows: "
            f"count={len(invalid)} sample={invalid[:8]}"
        )
    return result


def apply_candidate_geometry_rows(records, result, *, dbu):
    """Apply native per-row geometry fields to candidate records."""

    dbu = float(dbu)
    if not dbu > 0.0:
        raise CandidateGeometryRefreshError(
            f"candidate geometry requires a positive dbu, got {dbu}"
        )
    updated = list(records)
    rows = result["candidate_rows"].tolist()
    x_values = result["x_dbu"].tolist()
    y_values = result["y_dbu"].tolist()
    segment_lengths = result["segment_length_dbu"].tolist()
    upstream_lengths = result["upstream_length_dbu"].tolist()
    downstream_lengths = result["downstream_length_dbu"].tolist()
    for row, x_dbu, y_dbu, length, upstream, downstream in zip(
        rows,
        x_values,
        y_values,
        segment_lengths,
        upstream_lengths,
        downstream_lengths,
        strict=True,
    ):
        if row < 0 or row >= len(updated):
            raise CandidateGeometryRefreshError(
                f"native candidate row is outside state domain: {row}"
            )
        record = dict(updated[row])
        x_round = int(round(x_dbu))
        y_round = int(round(y_dbu))
        record["x_dbu"] = x_round
        record["y_dbu"] = y_round
        record["candidate_location_x_dbu"] = x_round
        record["candidate_location_y_dbu"] = y_round
        record["x_um"] = x_dbu / dbu
        record["y_um"] = y_dbu / dbu
        if (
            record.get("parent_node_id") is not None
            and len(record.get("child_node_ids", ()) or ()) == 1
        ):
            record["segment_length_dbu"] = int(round(length))
            record["upstream_length_dbu"] = float(upstream)
            record["downstream_length_dbu"] = float(downstream)
        record["candidate_geometry_status"] = "refreshed_native_dirty_net"
        updated[row] = record
    return tuple(updated)


__all__ = [
    "CandidateGeometryRefreshError",
    "active_candidate_net_ids",
    "apply_candidate_geometry_rows",
    "collect_candidate_dirty_nets",
    "compute_candidate_dirty_nets",
    "refresh_candidate_geometry_rows",
]
