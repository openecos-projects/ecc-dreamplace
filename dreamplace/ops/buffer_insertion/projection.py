from dataclasses import dataclass, field
import math
import time


@dataclass(frozen=True)
class BufferingProjectionResult:
    mode: str
    selected_actions: tuple = ()
    projected_buffer_count: int = 0
    affected_nets: tuple = ()
    validation: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _actions_from_virtual_state(virtual_state):
    if virtual_state is None:
        return ()
    entries = getattr(virtual_state, "entries", {}) or {}
    actions = []
    for action_id, payload in entries.items():
        record = dict(payload)
        record.setdefault("action_id", action_id)
        actions.append(record)
    return tuple(actions)


def _affected_nets(actions):
    nets = []
    for action in actions:
        net_id = action.get("net_id", action.get("affected_net_id"))
        if net_id is not None:
            nets.append(int(net_id))
    return tuple(sorted(set(nets)))


def _actions_from_virtual_state_with_legal_table(virtual_state, legal_table):
    if virtual_state is None or not legal_table:
        return None
    from dreamplace.ops.physical_action import build_commit_action_candidates

    artifact = build_commit_action_candidates(
        virtual_state,
        legal_table=legal_table,
    )
    return tuple(dict(action) for action in artifact.get("actions", ()) or ())


def _actions_from_entries_with_legal_table(entries, legal_table):
    if not legal_table:
        return None
    from dreamplace.ops.physical_action import build_commit_action_candidates_from_entries

    artifact = build_commit_action_candidates_from_entries(
        entries,
        legal_table=legal_table,
    )
    return tuple(dict(action) for action in artifact.get("actions", ()) or ())


def empty_buffering_projection(mode="unknown", reason="empty"):
    return BufferingProjectionResult(
        mode=str(mode),
        validation={"status": "empty", "reason": str(reason)},
    )


def _sink_pin_ids_for_net(net):
    rc_tree = dict(net.get("rc_tree", {}) or {})
    for key in ("sink_pin_ids", "sink_nodes"):
        values = rc_tree.get(key)
        if values:
            return sorted({int(value) for value in list(values or [])})
    values = net.get("sink_pin_ids", net.get("sink_nodes", None))
    if values:
        return sorted({int(value) for value in list(values or [])})
    driver_pin_id = net.get("driver_pin_id")
    net_pin_ids = net.get("net_pin_ids")
    if driver_pin_id is None or not net_pin_ids:
        return []
    return sorted(
        {
            int(pin_id)
            for pin_id in list(net_pin_ids or [])
            if int(pin_id) != int(driver_pin_id)
        }
    )


def _pin_slack_values(data_collections):
    if data_collections is None:
        return None
    pin_slack = getattr(data_collections, "pin_slack", None)
    if pin_slack is None:
        return None
    try:
        import torch

        if not torch.is_tensor(pin_slack):
            pin_slack = torch.as_tensor(pin_slack)
        pin_slack = pin_slack.detach().flatten().cpu()
        if pin_slack.numel() == 0:
            return pin_slack
        return pin_slack
    except (TypeError, ValueError, RuntimeError):
        return None


def _overlay_current_sink_slack(nets, data_collections):
    slack_values = _pin_slack_values(data_collections)
    if slack_values is None:
        return nets, {
            "criticality_overlay_status": "skipped",
            "criticality_overlay_reason": "missing_pin_slack",
        }
    if nets is None:
        return nets, {
            "criticality_overlay_status": "skipped",
            "criticality_overlay_reason": "missing_nets",
        }

    updated_nets = []
    net_count = 0
    sink_count = 0
    updated_sink_count = 0
    negative_sink_count = 0
    missing_pin_count = 0
    nonfinite_slack_count = 0
    for net in list(nets or []):
        net_count += 1
        sink_pin_ids = _sink_pin_ids_for_net(net)
        sink_count += len(sink_pin_ids)
        if not sink_pin_ids:
            updated_nets.append(net)
            continue

        local_sink_slack = dict(net.get("sink_slack_by_pin") or {})
        local_npath = dict(net.get("npath_by_pin") or {})
        changed = False
        for pin_id in sink_pin_ids:
            pin_id = int(pin_id)
            if pin_id < 0 or pin_id >= int(slack_values.numel()):
                missing_pin_count += 1
                continue
            slack = float(slack_values[pin_id].item())
            if not math.isfinite(slack):
                nonfinite_slack_count += 1
                continue
            local_sink_slack[pin_id] = slack
            local_npath.setdefault(pin_id, 1)
            updated_sink_count += 1
            if slack < 0.0:
                negative_sink_count += 1
            changed = True
        if changed:
            item = dict(net)
            item["sink_slack_by_pin"] = {
                int(pin_id): float(slack)
                for pin_id, slack in local_sink_slack.items()
            }
            item["npath_by_pin"] = {
                int(pin_id): int(npath) for pin_id, npath in local_npath.items()
            }
            item["criticality_source"] = "current_pin_slack_projection_overlay"
            updated_nets.append(item)
        else:
            updated_nets.append(net)

    status = "ok" if updated_sink_count > 0 else "skipped"
    reason = None if updated_sink_count > 0 else "no_sink_slack_overlay_entries"
    summary = {
        "criticality_overlay_status": status,
        "criticality_overlay_source": "current_pin_slack",
        "criticality_overlay_net_count": int(net_count),
        "criticality_overlay_sink_count": int(sink_count),
        "criticality_overlay_updated_sink_count": int(updated_sink_count),
        "criticality_overlay_negative_sink_count": int(negative_sink_count),
        "criticality_overlay_missing_pin_count": int(missing_pin_count),
        "criticality_overlay_nonfinite_slack_count": int(nonfinite_slack_count),
    }
    if reason is not None:
        summary["criticality_overlay_reason"] = reason
    return tuple(updated_nets), summary


def _with_projection_metadata(result, extra_metadata):
    extra_metadata = dict(extra_metadata or {})
    if not extra_metadata:
        return result
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(extra_metadata)
    object.__setattr__(result, "metadata", metadata)
    return result


def project_candidate_buffer_state(
    state,
    *,
    top_k=None,
    threshold=None,
    mode="candidate",
    legal_table=None,
):
    if state is None:
        return empty_buffering_projection(mode=mode, reason="missing_state")
    from .optimization_state import (
        project_buffer_optimization_state,
        project_discrete_candidate_state,
    )

    is_discrete = hasattr(state, "activation_param")
    if is_discrete:
        if top_k is not None or threshold is not None:
            raise ValueError("discrete candidate projection does not allow filtering")
        virtual_state = project_discrete_candidate_state(state)
    else:
        virtual_state = project_buffer_optimization_state(
            state,
            top_k=top_k,
            threshold=threshold,
        )
    actions = _actions_from_virtual_state_with_legal_table(
        virtual_state,
        legal_table,
    )
    action_model = "coordinate_commit_actions" if actions is not None else "virtual_state_entries"
    if actions is None:
        actions = _actions_from_virtual_state(virtual_state)
    projected_candidate_ids = tuple(
        int(action.get("candidate_id", action.get("action_id", -1)))
        for action in actions
    )
    active_candidate_ids = (
        tuple(
            sorted(
                int(value)
                for value in state.candidate_ids[
                    state.activation_param.detach() == 1.0
                ].detach().cpu().tolist()
            )
        )
        if is_discrete
        else None
    )
    validation = {"status": "ok"}
    if is_discrete:
        validation.update(
            {
                "active_candidate_ids": active_candidate_ids,
                "projected_candidate_ids": projected_candidate_ids,
                "exact_active_candidate_match": (
                    projected_candidate_ids == active_candidate_ids
                ),
            }
        )
        if projected_candidate_ids != active_candidate_ids:
            raise ValueError("discrete candidate projection identity mismatch")
    return BufferingProjectionResult(
        mode=str(mode),
        selected_actions=actions,
        projected_buffer_count=len(actions),
        affected_nets=_affected_nets(actions),
        validation=validation,
        metadata={
            "projection_source": (
                "discrete_candidate_state"
                if is_discrete
                else "candidate_buffer_optimization_state"
            ),
            "action_model": action_model,
            "candidate_strategy": (
                "discrete_net_gradient" if is_discrete else "continuous"
            ),
        },
    )


def project_segment_count_buffer_state(
    state,
    *,
    nets,
    candidate_id_start=0,
    mode="segment",
    legal_table=None,
    top_k=None,
    min_z_to_insert=0.5,
    require_setup_criticality=False,
    live_projection_context=None,
):
    if state is None:
        return empty_buffering_projection(mode=mode, reason="missing_state")
    if nets is None:
        return empty_buffering_projection(mode=mode, reason="missing_nets")
    from .segment_count_projection import project_segment_count_state_to_candidates

    projection_start = time.perf_counter()
    artifact = project_segment_count_state_to_candidates(
        nets,
        state,
        candidate_id_start=candidate_id_start,
        top_k=top_k,
        min_z_to_insert=min_z_to_insert,
        require_setup_criticality=require_setup_criticality,
        live_projection_context=live_projection_context,
    )
    projection_wall_ms = float((time.perf_counter() - projection_start) * 1000.0)
    actions = None
    materialize_wall_ms = 0.0
    if legal_table:
        materialize_start = time.perf_counter()
        actions = _actions_from_entries_with_legal_table(
            artifact.get("candidates", ()) or (),
            legal_table,
        )
        materialize_wall_ms = float(
            (time.perf_counter() - materialize_start) * 1000.0
        )
    action_model = "coordinate_commit_actions" if actions is not None else "batch_projected_candidates"
    if actions is None:
        actions = tuple(dict(candidate) for candidate in artifact.get("candidates", ()) or ())
    return BufferingProjectionResult(
        mode=str(mode),
        selected_actions=actions,
        projected_buffer_count=len(actions),
        affected_nets=_affected_nets(actions),
        validation={
            "status": str(artifact.get("status", "ok")),
            "skipped_missing_net_count": int(
                artifact.get("skipped_missing_net_count", 0) or 0
            ),
        },
        metadata={
            "projection_source": "segment_count_state",
            "action_model": action_model,
            "projection_wall_ms": projection_wall_ms,
            "action_materialize_wall_ms": materialize_wall_ms,
            "projection_policy": artifact.get("projection_policy"),
            "min_z_to_insert": artifact.get("min_z_to_insert"),
            "require_setup_criticality": bool(
                artifact.get("require_setup_criticality", False)
            ),
            "projection_order_policy": artifact.get("projection_order_policy"),
            "segment_count": int(artifact.get("segment_count", 0) or 0),
            "z_value_distribution": dict(
                artifact.get("z_value_distribution", {}) or {}
            ),
            "projected_candidate_count": int(
                artifact.get("projected_candidate_count", len(actions)) or 0
            ),
            "projected_candidate_count_before_topk": int(
                artifact.get(
                    "projected_candidate_count_before_topk",
                    artifact.get("projected_candidate_count", len(actions)),
                )
                or 0
            ),
            "projected_candidate_count_after_criticality_filter": int(
                artifact.get(
                    "projected_candidate_count_after_criticality_filter",
                    artifact.get("projected_candidate_count", len(actions)),
                )
                or 0
            ),
            "criticality_filtered_candidate_count": int(
                artifact.get("criticality_filtered_candidate_count", 0) or 0
            ),
            "top_candidate_diagnostics": tuple(
                dict(item)
                for item in list(artifact.get("top_candidate_diagnostics", ()) or ())
            ),
            "selected_candidate_diagnostics": tuple(
                dict(item)
                for item in list(
                    artifact.get("selected_candidate_diagnostics", ()) or ()
                )
            ),
            "projection_provenance": dict(
                artifact.get("projection_provenance", {}) or {}
            ),
            "top_k": artifact.get("top_k"),
            "local_ranking_used": bool(artifact.get("local_ranking_used", False)),
            "segment_shared_bsu": bool(artifact.get("segment_shared_bsu", True)),
            "full_row_materialization_count": int(
                getattr(state, "summary", {}).get("full_row_materialization_count", 0)
            ),
            "terminal_row_materialization_count": int(
                getattr(state, "summary", {}).get("terminal_row_materialization_count", 0)
            ),
        },
    )


def project_buffering_lane_state(
    lane,
    *,
    top_k=None,
    threshold=None,
    live_projection_context=None,
):
    mode = str(getattr(lane.config, "mode", "unknown"))
    state = getattr(lane, "buffer_optimization_state", None)
    if state is None and getattr(lane, "data_collections", None) is not None:
        state = getattr(lane.data_collections, "buffer_optimization_state", None)
    if state is None:
        return empty_buffering_projection(mode=mode, reason="missing_state")
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    if payload is None and getattr(lane, "data_collections", None) is not None:
        payload = getattr(lane.data_collections, "buffer_relaxed_timing_payload", None)
    metadata = dict((payload or {}).get("metadata", {}) or {}) if isinstance(payload, dict) else {}
    legal_table = (payload or {}).get("buffer_legal_table") if isinstance(payload, dict) else None
    if metadata.get("state_kind") == "segment_count" or hasattr(state, "z_param"):
        nets = (payload or {}).get("nets") if isinstance(payload, dict) else None
        min_z_to_insert = getattr(
            lane.config,
            "segment_projection_min_z_to_insert",
            0.5,
        )
        if getattr(state, "is_tensor_backed", False) and not nets:
            active_indices = (
                state.z_value().detach() >= float(min_z_to_insert)
            ).nonzero(as_tuple=False).flatten()
            selected_net_ids = state.segment_net_ids_for_indices(active_indices)
            nets = state.net_records(selected_net_ids)
        nets, criticality_overlay = _overlay_current_sink_slack(
            nets,
            getattr(lane, "data_collections", None),
        )
        result = project_segment_count_buffer_state(
            state,
            nets=nets,
            mode=mode,
            legal_table=legal_table,
            top_k=top_k,
            min_z_to_insert=min_z_to_insert,
            require_setup_criticality=getattr(
                lane.config,
                "segment_projection_require_setup_criticality",
                False,
            ),
            live_projection_context=live_projection_context,
        )
        return _with_projection_metadata(result, criticality_overlay)
    return project_candidate_buffer_state(
        state,
        top_k=top_k,
        threshold=threshold,
        mode=mode,
        legal_table=legal_table,
    )
