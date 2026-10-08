COORDINATE_BACKEND_NAME = "coordinate_or_branch_specific_buffer_insert"

REQUIRED_ACTION_FIELDS = (
    "action_kind",
    "net_name",
    "buffer_main_type_index",
    "bsu",
    "buffer_master_id",
    "buffer_master_name",
    "candidate_location_x_dbu",
    "candidate_location_y_dbu",
    "driver_pin_name",
    "load_pin_name",
)

REQUIRED_BRIDGE_STEPS = (
    "resolve_original_net_and_driver_load_partition",
    "create_buffer_instance_at_candidate_coordinate",
    "connect_buffer_input_to_upstream_net",
    "create_or_resolve_downstream_net",
    "disconnect_downstream_load_pins_from_original_net",
    "connect_downstream_load_pins_to_buffer_output_net",
    "legalize_or_mark_incremental_legalization_required",
    "invalidate_or_rebuild_parasitics_and_timing",
    "sync_openroad_pydb_runtime_views",
    "run_openroad_opensta_before_after_accept_reject",
)

OPENROAD_SOURCE_EVIDENCE = (
    {
        "path": "src/rsz/src/SplitLoadMove.cc",
        "evidence": "uses makeBuffer, connectPin, dbITerm::disconnect, and dbITerm::connect to split selected loads",
    },
    {
        "path": "src/rsz/src/Rebuffer.cc",
        "evidence": "exports a buffered tree by creating buffer instances, creating downstream nets, and reconnecting loads",
    },
    {
        "path": "src/rsz/src/Resizer.cc",
        "evidence": "Resizer::makeBuffer creates and places a buffer instance but does not itself split the original net",
    },
    {
        "path": "src/rsz/src/Resizer.i",
        "evidence": "SWIG exposes repair_net_cmd, but no simple candidate-coordinate insert_buffer command was found",
    },
)


def _has_value(action, field):
    if field not in action:
        return False
    value = action.get(field)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _is_nonnegative_int(value):
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _is_integer_value(value):
    if isinstance(value, bool):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric.is_integer()


def _coordinate_present(action):
    return _has_value(action, "candidate_location_x_dbu") and _has_value(
        action,
        "candidate_location_y_dbu",
    )


def _commit_int(action, key, default):
    try:
        return int(action.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _commit_float(action, key, default):
    try:
        return float(action.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _topology_order_coordinate_actions(actions):
    """Order same-net segment actions before mutating their shared topology."""
    annotated = []
    for input_order, action in enumerate(actions):
        item = dict(action)
        item["commit_input_order"] = int(input_order)
        annotated.append(item)

    def sort_key(action):
        input_order = _commit_int(action, "commit_input_order", 0)
        tree_depth = _commit_int(action, "segment_tree_depth", -1)
        has_segment_topology = tree_depth >= 0
        downstream_pin_count = len(set(action.get("downstream_pin_names") or ()))
        return (
            str(action.get("net_name", "")),
            0 if has_segment_topology else 1,
            tree_depth if has_segment_topology else -downstream_pin_count,
            _commit_int(action, "segment_parent_node_id", -1),
            _commit_int(action, "segment_child_node_id", -1),
            _commit_int(action, "segment_split_index", 0),
            _commit_float(action, "segment_split_ratio", 0.0),
            input_order,
        )

    annotated.sort(key=sort_key)
    for commit_order, action in enumerate(annotated):
        action["commit_topology_order"] = int(commit_order)
    return annotated


def summarize_coordinate_action_readiness(action):
    action = dict(action or {})
    missing_fields = [field for field in REQUIRED_ACTION_FIELDS if not _has_value(action, field)]
    blockers = []
    warnings = []
    if action.get("action_kind") != "buffer_insert":
        blockers.append("action_kind_must_be_buffer_insert")
    if not _is_nonnegative_int(action.get("buffer_main_type_index", -1)):
        blockers.append("missing_buffer_main_type_index")
    if not _is_nonnegative_int(action.get("bsu", -1)):
        blockers.append("missing_bsu")
    if not _is_nonnegative_int(action.get("buffer_master_id", -1)):
        blockers.append("missing_buffer_master_id")
    if not _coordinate_present(action):
        blockers.append("missing_candidate_coordinate")
    if not str(action.get("net_name", "")).strip():
        blockers.append("missing_net_name")
    if "bu" in action and not _is_integer_value(action.get("bu")):
        blockers.append("fractional_bu_must_not_reach_coordinate_backend")
    if "bsu_index_param" in action or "bsu_index_value" in action:
        blockers.append("continuous_bsu_index_must_not_reach_coordinate_backend")

    downstream_pin_names = list(action.get("downstream_pin_names") or [])
    if downstream_pin_names:
        partition_status = "explicit_downstream_partition"
    elif str(action.get("load_pin_name", "")).strip():
        partition_status = "singleton_load_partition_seed"
        warnings.append("explicit_downstream_partition_missing")
    else:
        partition_status = "missing_downstream_partition"
        blockers.append("missing_downstream_partition")

    payload_ready = not blockers and not missing_fields
    return {
        "action_id": int(action.get("action_id", -1)),
        "net_name": str(action.get("net_name", "")),
        "payload_ready": payload_ready,
        "payload_status": "ready_minimal_coordinate_seed" if payload_ready else "blocked",
        "partition_status": partition_status,
        "missing_fields": missing_fields,
        "blockers": blockers,
        "warnings": warnings,
        "buffer_main_type_index": int(action.get("buffer_main_type_index", -1))
        if _is_nonnegative_int(action.get("buffer_main_type_index", -1))
        else -1,
        "bsu": int(action.get("bsu", -1)) if _is_nonnegative_int(action.get("bsu", -1)) else -1,
        "candidate_location_x_dbu": float(action.get("candidate_location_x_dbu", 0.0) or 0.0),
        "candidate_location_y_dbu": float(action.get("candidate_location_y_dbu", 0.0) or 0.0),
    }


def build_coordinate_backend_contract(actions=None, *, openroad_source_root=None):
    actions = list(actions or [])
    action_summaries = [summarize_coordinate_action_readiness(action) for action in actions]
    payload_ready_count = sum(1 for item in action_summaries if item["payload_ready"])
    explicit_partition_count = sum(
        1
        for item in action_summaries
        if item["partition_status"] == "explicit_downstream_partition"
    )
    has_actions = bool(actions)
    implementation_status = "not_implemented"
    status = (
        "contract_ready_backend_not_implemented"
        if has_actions and payload_ready_count == len(actions)
        else "blocked_missing_action_payload"
        if has_actions
        else "blocked_no_actions"
    )
    return {
        "artifact": "buffering_coordinate_backend_contract",
        "artifact_version": 1,
        "backend": COORDINATE_BACKEND_NAME,
        "status": status,
        "implementation_status": implementation_status,
        "qor_evidence": False,
        "openroad_source_root": str(openroad_source_root or ""),
        "required_action_fields": list(REQUIRED_ACTION_FIELDS),
        "required_bridge_steps": list(REQUIRED_BRIDGE_STEPS),
        "openroad_source_evidence": list(OPENROAD_SOURCE_EVIDENCE),
        "action_count": len(actions),
        "payload_ready_count": payload_ready_count,
        "explicit_downstream_partition_count": explicit_partition_count,
        "singleton_seed_count": sum(
            1
            for item in action_summaries
            if item["partition_status"] == "singleton_load_partition_seed"
        ),
        "unsupported_realization_reasons": [
            "coordinate_backend_bridge_not_implemented",
            "requires_cxx_openroad_db_net_split_and_pin_reconnect",
            "requires_openroad_opensta_before_after_verification",
        ],
        "action_summaries": action_summaries,
    }


def _metric_value(metrics, name):
    metrics = dict(metrics or {})
    value = metrics.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_delta(result, before_metrics, after_metrics, name):
    explicit = _metric_value(result, f"actual_delta_{name}")
    if explicit is not None:
        return explicit
    before = _metric_value(before_metrics, name)
    after = _metric_value(after_metrics, name)
    if before is None or after is None:
        return None
    return after - before


def _first_metric_value(metrics, names):
    for name in names:
        value = _metric_value(metrics, name)
        if value is not None:
            return value
    return None


def _result_metric_delta(result, before_metrics, after_metrics, *, result_names, metric_names):
    for name in result_names:
        value = _metric_value(result, name)
        if value is not None:
            return value
    before = _first_metric_value(before_metrics, metric_names)
    after = _first_metric_value(after_metrics, metric_names)
    if before is None or after is None:
        return None
    return after - before


def _dict_or_empty(value):
    return dict(value) if isinstance(value, dict) else {}


def _first_reason(*values):
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _result_reject_reason(result):
    for field in ("reject_reason", "error", "skip_reason"):
        value = result.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _db_commit_status(result):
    explicit = result.get("db_commit_status")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    backend_status = str(result.get("backend_status", ""))
    status = str(result.get("status", ""))
    if backend_status and backend_status != "ok":
        return "backend_failed_not_committed"
    if status == "accepted":
        if bool(result.get("committed_to_source_design", False)):
            return "committed_to_source_design"
        if bool(result.get("disposable_design", True)):
            return "committed_to_disposable_design"
        return "accepted_commit_scope_unknown"
    if status == "rejected":
        return "rejected_after_measured_commit"
    if status == "skipped":
        return "not_attempted"
    if status == "failed":
        return "failed_not_committed"
    return "unknown"


def build_measured_coordinate_action_result(
    action,
    realization_result=None,
    *,
    disposable_design_path=None,
):
    action = dict(action or {})
    result = dict(realization_result or {})
    before_metrics = dict(result.get("before_metrics") or {})
    after_metrics = dict(result.get("after_metrics") or {})
    actual_delta_wns = _metric_delta(result, before_metrics, after_metrics, "wns")
    actual_delta_tns = _metric_delta(result, before_metrics, after_metrics, "tns")
    pre_slew_violation_count = _first_metric_value(
        before_metrics,
        ("slew_violation_count", "slew_vio_count", "max_slew_violation_count"),
    )
    post_slew_violation_count = _first_metric_value(
        after_metrics,
        ("slew_violation_count", "slew_vio_count", "max_slew_violation_count"),
    )
    actual_delta_slew_violation_count = _result_metric_delta(
        result,
        before_metrics,
        after_metrics,
        result_names=(
            "actual_delta_slew_violation_count",
            "slew_violation_count_delta",
        ),
        metric_names=(
            "slew_violation_count",
            "slew_vio_count",
            "max_slew_violation_count",
        ),
    )
    pre_cap_violation_count = _first_metric_value(
        before_metrics,
        ("cap_violation_count", "cap_vio_count", "max_cap_violation_count"),
    )
    post_cap_violation_count = _first_metric_value(
        after_metrics,
        ("cap_violation_count", "cap_vio_count", "max_cap_violation_count"),
    )
    actual_delta_cap_violation_count = _result_metric_delta(
        result,
        before_metrics,
        after_metrics,
        result_names=(
            "actual_delta_cap_violation_count",
            "cap_violation_count_delta",
        ),
        metric_names=(
            "cap_violation_count",
            "cap_vio_count",
            "max_cap_violation_count",
        ),
    )
    hard_commit_diagnostic = _dict_or_empty(result.get("hard_commit_diagnostic"))
    hard_commit_topology_delta = _dict_or_empty(
        result.get(
            "hard_commit_topology_delta",
            hard_commit_diagnostic.get("topology_delta", {}),
        )
    )
    hard_commit_load_partition = _dict_or_empty(
        result.get(
            "hard_commit_load_partition",
            hard_commit_diagnostic.get("load_partition", {}),
        )
    )
    hard_commit_cell_arc_delay_samples = _dict_or_empty(
        result.get(
            "hard_commit_cell_arc_delay_samples",
            hard_commit_diagnostic.get("cell_arc_delay_samples", {}),
        )
    )
    return {
        "artifact": "buffering_coordinate_measured_action_result",
        "artifact_version": 1,
        "action_id": int(action.get("action_id", result.get("action_id", -1))),
        "candidate_id": int(
            action.get(
                "candidate_id",
                result.get(
                    "candidate_id",
                    action.get("action_id", result.get("action_id", -1)),
                ),
            )
        ),
        "segment_id": action.get("segment_id", result.get("segment_id")),
        "segment_parent_node_id": action.get("segment_parent_node_id"),
        "segment_child_node_id": action.get("segment_child_node_id"),
        "segment_tree_depth": action.get("segment_tree_depth"),
        "segment_split_index": action.get("segment_split_index"),
        "segment_split_count_on_edge": action.get("segment_split_count_on_edge"),
        "segment_split_ratio": action.get("segment_split_ratio"),
        "z_value": action.get("z_value"),
        "bsu_value": action.get("bsu_value"),
        "projected_bsu_index": action.get("projected_bsu_index"),
        "projected_repeater_count": action.get("projected_repeater_count"),
        "projection_policy": action.get("projection_policy"),
        "net_name": str(action.get("net_name", result.get("net_name", ""))),
        "driver_pin_name": str(
            action.get("driver_pin_name", result.get("driver_pin_name", ""))
        ),
        "load_pin_name": str(
            action.get("load_pin_name", result.get("load_pin_name", ""))
        ),
        "downstream_pin_names": list(action.get("downstream_pin_names") or []),
        "commit_input_order": action.get("commit_input_order"),
        "commit_topology_order": action.get("commit_topology_order"),
        "source_net_name": result.get("source_net_name", ""),
        "resolved_target_net_name": result.get("resolved_target_net_name", ""),
        "target_net_remapped": result.get("target_net_remapped"),
        "resolved_driver_pin_name": result.get("resolved_driver_pin_name", ""),
        "driver_pin_remapped": result.get("driver_pin_remapped"),
        "requested_downstream_pin_count": result.get("requested_downstream_pin_count"),
        "moved_load_pin_names": list(result.get("moved_load_pin_names") or []),
        "unsupported_reasons": list(result.get("unsupported_reasons") or []),
        "error": str(result.get("error", "") or ""),
        "candidate_location_x_dbu": action.get(
            "candidate_location_x_dbu",
            result.get("candidate_location_x_dbu"),
        ),
        "candidate_location_y_dbu": action.get(
            "candidate_location_y_dbu",
            result.get("candidate_location_y_dbu"),
        ),
        "candidate_location_x_dbu_analytical": action.get(
            "candidate_location_x_dbu_analytical"
        ),
        "candidate_location_y_dbu_analytical": action.get(
            "candidate_location_y_dbu_analytical"
        ),
        "segment_parent_x_dbu": action.get("segment_parent_x_dbu"),
        "segment_parent_y_dbu": action.get("segment_parent_y_dbu"),
        "segment_child_x_dbu": action.get("segment_child_x_dbu"),
        "segment_child_y_dbu": action.get("segment_child_y_dbu"),
        "coordinate_source": action.get("coordinate_source", ""),
        "placement_snapshot_identity": action.get(
            "placement_snapshot_identity", ""
        ),
        "placement_snapshot_iteration": action.get(
            "placement_snapshot_iteration"
        ),
        "topology_generation": action.get("topology_generation"),
        "frozen_topology_generation": action.get(
            "frozen_topology_generation"
        ),
        "rectilinear_path_policy": action.get("rectilinear_path_policy", ""),
        "action_signature": result.get("action_signature")
        or action.get("action_signature"),
        "status": str(result.get("status", "unknown")),
        "backend_status": str(result.get("backend_status", result.get("status", ""))),
        "accept_policy": str(result.get("accept_policy", "")),
        "balanced_acceptance_score": result.get("balanced_acceptance_score"),
        "balanced_acceptance_min_score": result.get("balanced_acceptance_min_score"),
        "balanced_acceptance_breakdown": dict(
            result.get("balanced_acceptance_breakdown") or {}
        ),
        "pre_wns": _metric_value(before_metrics, "wns"),
        "pre_tns": _metric_value(before_metrics, "tns"),
        "post_wns": _metric_value(after_metrics, "wns"),
        "post_tns": _metric_value(after_metrics, "tns"),
        "actual_delta_wns": actual_delta_wns,
        "actual_delta_tns": actual_delta_tns,
        "pre_slew_violation_count": pre_slew_violation_count,
        "post_slew_violation_count": post_slew_violation_count,
        "actual_delta_slew_violation_count": actual_delta_slew_violation_count,
        "pre_cap_violation_count": pre_cap_violation_count,
        "post_cap_violation_count": post_cap_violation_count,
        "actual_delta_cap_violation_count": actual_delta_cap_violation_count,
        "accepted": bool(result.get("accepted", False))
        or str(result.get("status", "")) == "accepted",
        "reject_reason": _result_reject_reason(result),
        "db_commit_status": _db_commit_status(result),
        "rollback_status": str(result.get("rollback_status", "not_attempted")),
        "disposable_design_path": str(
            result.get("disposable_design_path", disposable_design_path or "")
        ),
        "realization_backend": result.get("realization_backend"),
        "implementation_scope": result.get("implementation_scope"),
        "inserted_buffer_name": result.get("inserted_buffer_name"),
        "downstream_net_name": result.get("downstream_net_name"),
        "hard_commit_diagnostic": hard_commit_diagnostic,
        "hard_commit_diagnostic_status": str(
            result.get(
                "hard_commit_diagnostic_status",
                hard_commit_diagnostic.get("status", "missing"),
            )
        ),
        "hard_commit_topology_delta": hard_commit_topology_delta,
        "hard_commit_load_partition": hard_commit_load_partition,
        "hard_commit_cell_arc_delay_samples": hard_commit_cell_arc_delay_samples,
        "measured_before_after": (
            _metric_value(before_metrics, "wns") is not None
            and _metric_value(after_metrics, "wns") is not None
            and _metric_value(before_metrics, "tns") is not None
            and _metric_value(after_metrics, "tns") is not None
        ),
        "measurement_source": str(
            result.get(
                "measurement_source",
                action.get("measurement_source", "commit"),
            )
        ),
        "measurement_role": str(
            result.get(
                "measurement_role",
                action.get("measurement_role", "strategy_coordinate_commit"),
            )
        ),
        "diagnostic_probe_context_policy": result.get(
            "diagnostic_probe_context_policy",
            action.get("diagnostic_probe_context_policy"),
        ),
        "proxy_predicted_delta_obj": action.get("predicted_delta_obj"),
        "proxy_predicted_delta_tns": action.get("predicted_delta_tns"),
        "proxy_predicted_delta_wns": action.get("predicted_delta_wns"),
        "proxy_selection_score": action.get("selection_score"),
        "measured_label_probe_rank_source": result.get(
            "measured_label_probe_rank_source",
            action.get("measured_label_probe_rank_source"),
        ),
        "measured_label_probe_rank_sources": list(
            result.get(
                "measured_label_probe_rank_sources",
                action.get("measured_label_probe_rank_sources", []),
            )
            or []
        ),
        "measured_label_probe_rank_value": result.get(
            "measured_label_probe_rank_value",
            action.get("measured_label_probe_rank_value"),
        ),
        "measured_label_probe_rank_values": dict(
            result.get(
                "measured_label_probe_rank_values",
                action.get("measured_label_probe_rank_values", {}),
            )
            or {}
        ),
        "measured_label_probe_rank_direction": result.get(
            "measured_label_probe_rank_direction",
            action.get("measured_label_probe_rank_direction"),
        ),
        "measured_label_probe_rank_directions": dict(
            result.get(
                "measured_label_probe_rank_directions",
                action.get("measured_label_probe_rank_directions", {}),
            )
            or {}
        ),
    }


def build_coordinate_measured_results_artifact(
    actions,
    realization_results,
    *,
    disposable_design_path=None,
):
    actions = list(actions or [])
    realization_results = list(realization_results or [])
    measured_results = []
    for index, action in enumerate(actions):
        result = realization_results[index] if index < len(realization_results) else {}
        measured_results.append(
            build_measured_coordinate_action_result(
                action,
                result,
                disposable_design_path=disposable_design_path,
            )
        )
    tns_improved_count = sum(
        1
        for result in measured_results
        if result["actual_delta_tns"] is not None and result["actual_delta_tns"] > 0
    )
    tns_regressed_count = sum(
        1
        for result in measured_results
        if result["actual_delta_tns"] is not None and result["actual_delta_tns"] < 0
    )
    slew_improved_count = sum(
        1
        for result in measured_results
        if result["actual_delta_slew_violation_count"] is not None
        and result["actual_delta_slew_violation_count"] < 0
    )
    slew_regressed_count = sum(
        1
        for result in measured_results
        if result["actual_delta_slew_violation_count"] is not None
        and result["actual_delta_slew_violation_count"] > 0
    )
    cap_improved_count = sum(
        1
        for result in measured_results
        if result["actual_delta_cap_violation_count"] is not None
        and result["actual_delta_cap_violation_count"] < 0
    )
    cap_regressed_count = sum(
        1
        for result in measured_results
        if result["actual_delta_cap_violation_count"] is not None
        and result["actual_delta_cap_violation_count"] > 0
    )
    accepted_slew_delta_sum = sum(
        result["actual_delta_slew_violation_count"]
        for result in measured_results
        if result["accepted"] and result["actual_delta_slew_violation_count"] is not None
    )
    accepted_cap_delta_sum = sum(
        result["actual_delta_cap_violation_count"]
        for result in measured_results
        if result["accepted"] and result["actual_delta_cap_violation_count"] is not None
    )
    accepted_action_count = sum(1 for result in measured_results if result["accepted"])
    accepted_tns_delta_sum = sum(
        result["actual_delta_tns"]
        for result in measured_results
        if result["accepted"] and result["actual_delta_tns"] is not None
    )
    accepted_tns_regressed_count = sum(
        1
        for result in measured_results
        if result["accepted"]
        and result["actual_delta_tns"] is not None
        and result["actual_delta_tns"] < 0
    )
    accepted_tns_improved_count = sum(
        1
        for result in measured_results
        if result["accepted"]
        and result["actual_delta_tns"] is not None
        and result["actual_delta_tns"] > 0
    )
    measured_qor = any(
        result["accepted"] and result["actual_delta_tns"] is not None
        for result in measured_results
    )
    if not accepted_action_count:
        qor_status = "no_accepted_actions"
    elif not measured_qor:
        qor_status = "missing_committed_tns"
    elif accepted_tns_delta_sum > 0.0 and accepted_tns_regressed_count == 0:
        qor_status = "pass"
    elif accepted_tns_delta_sum <= 0.0:
        qor_status = "fail_tns_nonpositive"
    else:
        qor_status = "mixed"
    return {
        "artifact": "buffering_coordinate_commit_results",
        "artifact_version": 1,
        "status": "ok",
        "coordinate_commit_attempt_count": len(measured_results),
        "measured_before_after_count": sum(
            1 for result in measured_results if result["measured_before_after"]
        ),
        "accepted_action_count": accepted_action_count,
        "tns_improved_action_count": tns_improved_count,
        "tns_regressed_action_count": tns_regressed_count,
        "accepted_tns_delta_sum": accepted_tns_delta_sum,
        "accepted_tns_improved_action_count": accepted_tns_improved_count,
        "qor_status": qor_status,
        "qor_pass": qor_status == "pass",
        "slew_violation_improved_action_count": slew_improved_count,
        "slew_violation_regressed_action_count": slew_regressed_count,
        "cap_violation_improved_action_count": cap_improved_count,
        "cap_violation_regressed_action_count": cap_regressed_count,
        "accepted_slew_violation_delta_sum": accepted_slew_delta_sum,
        "accepted_cap_violation_delta_sum": accepted_cap_delta_sum,
        "accepted_tns_regressed_action_count": accepted_tns_regressed_count,
        "rejected_action_count": sum(
            1 for result in measured_results if result["status"] == "rejected"
        ),
        "failed_action_count": sum(
            1 for result in measured_results if result["status"] == "failed"
        ),
        "skipped_action_count": sum(
            1 for result in measured_results if result["status"] == "skipped"
        ),
        "disposable_design_path": str(disposable_design_path or ""),
        "results": measured_results,
    }


def build_coordinate_batch_measured_results_artifact(
    actions,
    realization_results,
    *,
    before_metrics=None,
    after_metrics=None,
    disposable_design_path=None,
):
    artifact = build_coordinate_measured_results_artifact(
        actions,
        realization_results,
        disposable_design_path=disposable_design_path,
    )
    before_metrics = dict(before_metrics or {})
    after_metrics = dict(after_metrics or {})
    actual_delta_wns = (
        None
        if _metric_value(before_metrics, "wns") is None
        or _metric_value(after_metrics, "wns") is None
        else _metric_value(after_metrics, "wns") - _metric_value(before_metrics, "wns")
    )
    actual_delta_tns = (
        None
        if _metric_value(before_metrics, "tns") is None
        or _metric_value(after_metrics, "tns") is None
        else _metric_value(after_metrics, "tns") - _metric_value(before_metrics, "tns")
    )
    if actual_delta_wns is None or actual_delta_tns is None:
        qor_status = "missing_committed_tns"
    elif actual_delta_wns < 0.0:
        qor_status = "fail_wns_regression"
    elif actual_delta_tns <= 0.0:
        qor_status = "fail_tns_nonpositive"
    else:
        qor_status = "pass"
    accepted_count = int(artifact.get("accepted_action_count", 0) or 0)
    measured_before_after = actual_delta_wns is not None and actual_delta_tns is not None
    for result in artifact["results"]:
        result["measurement_source"] = "batch_commit"
        result["measurement_role"] = "batch_coordinate_commit_action"
        result["measured_before_after"] = False
        result["pre_wns"] = None
        result["pre_tns"] = None
        result["post_wns"] = None
        result["post_tns"] = None
        result["actual_delta_wns"] = None
        result["actual_delta_tns"] = None
    artifact.update(
        {
            "coordinate_commit_measurement_mode": "single_batch_before_after",
            "per_action_sta_loop": False,
            "measured_before_after_count": 1 if measured_before_after else 0,
            "batch_before_metrics": before_metrics,
            "batch_after_metrics": after_metrics,
            "pre_wns": _metric_value(before_metrics, "wns"),
            "pre_tns": _metric_value(before_metrics, "tns"),
            "post_wns": _metric_value(after_metrics, "wns"),
            "post_tns": _metric_value(after_metrics, "tns"),
            "actual_delta_wns": actual_delta_wns,
            "actual_delta_tns": actual_delta_tns,
            "tns_improved_action_count": 0,
            "tns_regressed_action_count": 0,
            "accepted_tns_regressed_action_count": (
                accepted_count
                if actual_delta_tns is not None and actual_delta_tns < 0.0
                else 0
            ),
            "accepted_tns_delta_sum": actual_delta_tns
            if actual_delta_tns is not None
            else 0.0,
            "accepted_tns_improved_action_count": (
                accepted_count
                if actual_delta_tns is not None and actual_delta_tns > 0.0
                else 0
            ),
            "qor_status": qor_status,
            "qor_pass": qor_status == "pass",
        }
    )
    return artifact


def _write_commit_artifact(output_dir, artifact):
    if not output_dir:
        return ""
    from pathlib import Path
    import json

    path = Path(output_dir) / "buffering_coordinate_commit_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path)


def _openroad_bridge_from_placedb(placedb):
    if placedb is None:
        return None
    return getattr(placedb, "openroad_bridge", None) or getattr(placedb, "rawdb", None)


def _query_batch_timing_metrics(bridge):
    if bridge is None:
        raise RuntimeError("missing OpenROAD bridge")
    if hasattr(bridge, "query_diff_guided_batch_timing_metrics"):
        return dict(bridge.query_diff_guided_batch_timing_metrics() or {})
    from dreamplace.ops.placeio_openroad.place_io import PlaceIOFunction

    return dict(PlaceIOFunction.query_diff_guided_batch_timing_metrics(bridge) or {})


def _run_post_batch_timing_update(bridge):
    if bridge is None:
        return {
            "status": "failed",
            "reason": "missing OpenROAD bridge",
            "commands": [],
        }
    if not hasattr(bridge, "eval_tcl_string"):
        return {
            "status": "skipped",
            "reason": "bridge_missing_eval_tcl_string",
            "commands": [],
        }
    commands = ["estimate_parasitics -placement"]
    errors = []
    for command in commands:
        try:
            bridge.eval_tcl_string(command)
        except Exception as exc:  # pragma: no cover - real OpenROAD diagnostic path
            errors.append({"command": command, "error": str(exc)})
    return {
        "status": "ok" if not errors else "partial",
        "commands": commands,
        "errors": errors,
    }


def _commit_summary_from_artifact(artifact, *, commit_enabled, artifact_path=""):
    results = list(artifact.get("results", []) or [])
    measured_results = [
        result for result in results if result.get("measured_before_after")
    ]
    first_measured = measured_results[0] if measured_results else {}
    last_measured = measured_results[-1] if measured_results else first_measured
    before_wns = artifact.get("pre_wns", first_measured.get("pre_wns"))
    before_tns = artifact.get("pre_tns", first_measured.get("pre_tns"))
    after_wns = artifact.get("post_wns", last_measured.get("post_wns"))
    after_tns = artifact.get("post_tns", last_measured.get("post_tns"))
    delta_wns = artifact.get("actual_delta_wns")
    if delta_wns is None and before_wns is not None and after_wns is not None:
        delta_wns = float(after_wns) - float(before_wns)
    elif delta_wns is None:
        delta_wns = first_measured.get("actual_delta_wns")
    delta_tns = artifact.get("actual_delta_tns")
    if delta_tns is None and before_tns is not None and after_tns is not None:
        delta_tns = float(after_tns) - float(before_tns)
    elif delta_tns is None:
        delta_tns = first_measured.get("actual_delta_tns")
    artifact_status = str(artifact.get("status", ""))
    status = "failed" if artifact_status == "failed" else "accepted"
    if artifact.get("accepted_action_count", 0) <= 0:
        status = "rejected" if artifact.get("rejected_action_count", 0) > 0 else "failed"
    if artifact.get("failed_action_count", 0) > 0:
        status = "failed"
    reason = _first_reason(
        artifact.get("reason"),
        artifact.get("reject_reason"),
        artifact.get("error"),
        *(result.get("reject_reason") or result.get("error") for result in results),
    )
    transaction_status = status
    qor_status = str(artifact.get("qor_status", "missing_committed_tns"))
    qor_pass = bool(artifact.get("qor_pass", False))
    if status != "accepted":
        qor_status = "not_applicable_no_accepted_commit"
        qor_pass = False
    return {
        "status": status,
        "transaction_status": transaction_status,
        "qor_status": qor_status,
        "qor_pass": qor_pass,
        "reason": reason,
        "commit_enabled": bool(commit_enabled),
        "action_count": int(artifact.get("coordinate_commit_attempt_count", len(results)) or 0),
        "attempted_action_count": int(artifact.get("coordinate_commit_attempt_count", len(results)) or 0),
        "accepted_action_count": int(artifact.get("accepted_action_count", 0) or 0),
        "rejected_action_count": int(artifact.get("rejected_action_count", 0) or 0),
        "failed_action_count": int(artifact.get("failed_action_count", 0) or 0),
        "skipped_action_count": int(artifact.get("skipped_action_count", 0) or 0),
        "before_wns": before_wns,
        "before_tns": before_tns,
        "after_wns": after_wns,
        "after_tns": after_tns,
        "delta_wns": delta_wns,
        "delta_tns": delta_tns,
        "accepted_tns_delta_sum": artifact.get("accepted_tns_delta_sum"),
        "accepted_tns_improved_action_count": int(
            artifact.get("accepted_tns_improved_action_count", 0) or 0
        ),
        "accepted_tns_regressed_action_count": int(
            artifact.get("accepted_tns_regressed_action_count", 0) or 0
        ),
        "backend_status": artifact.get("backend_status", ""),
        "artifact_path": artifact_path,
    }


def commit_coordinate_buffer_actions(
    actions,
    *,
    placedb=None,
    openroad_bridge=None,
    output_dir=None,
    realization_config=None,
):
    actions = [dict(action) for action in list(actions or [])]
    if not actions:
        return {
            "status": "skipped",
            "reason": "no_projected_buffer_actions",
            "commit_enabled": True,
            "action_count": 0,
            "attempted_action_count": 0,
            "accepted_action_count": 0,
            "rejected_action_count": 0,
            "failed_action_count": 0,
        }

    actions = _topology_order_coordinate_actions(actions)

    readiness = [summarize_coordinate_action_readiness(action) for action in actions]
    blocked = [item for item in readiness if not item["payload_ready"]]
    if blocked:
        artifact = {
            "artifact": "buffering_coordinate_commit_results",
            "artifact_version": 1,
            "status": "blocked",
            "reason": "blocked_missing_action_payload",
            "coordinate_commit_attempt_count": 0,
            "accepted_action_count": 0,
            "rejected_action_count": 0,
            "failed_action_count": 0,
            "skipped_action_count": len(actions),
            "action_summaries": readiness,
            "results": [],
        }
        artifact_path = _write_commit_artifact(output_dir, artifact)
        summary = _commit_summary_from_artifact(
            artifact,
            commit_enabled=True,
            artifact_path=artifact_path,
        )
        summary.update(
            {
                "status": "blocked",
                "reason": "blocked_missing_action_payload",
                "action_count": len(actions),
                "blocked_action_count": len(blocked),
                "attempted_action_count": 0,
                "skipped_action_count": len(actions),
            }
        )
        return summary

    bridge = openroad_bridge or _openroad_bridge_from_placedb(placedb)
    if bridge is None or not hasattr(bridge, "run_coordinate_buffer_insert"):
        return {
            "status": "failed",
            "reason": "missing_openroad_coordinate_buffer_bridge",
            "commit_enabled": True,
            "action_count": len(actions),
            "attempted_action_count": 0,
            "accepted_action_count": 0,
            "rejected_action_count": 0,
            "failed_action_count": len(actions),
        }

    from dreamplace.ops.physical_action import realize_coordinate_buffer_action

    config = {
        "enable_realization": True,
        "allow_in_place_without_rollback": True,
        "allow_singleton_load_partition": True,
        "load_partition_mode": "branch_downstream",
    }
    config.update(dict(realization_config or {}))
    per_action_sta_loop = bool(config.pop("per_action_sta_loop", False))
    if per_action_sta_loop:
        realization_results = [
            realize_coordinate_buffer_action(bridge, action, config)
            for action in actions
        ]
        artifact = build_coordinate_measured_results_artifact(
            actions,
            realization_results,
        )
        artifact["coordinate_commit_measurement_mode"] = "per_action_before_after"
        artifact["per_action_sta_loop"] = True
    else:
        try:
            before_metrics = _query_batch_timing_metrics(bridge)
        except Exception as exc:
            artifact = {
                "artifact": "buffering_coordinate_commit_results",
                "artifact_version": 1,
                "status": "failed",
                "reason": "batch_before_metrics_failed",
                "error": str(exc),
                "coordinate_commit_measurement_mode": "single_batch_before_after",
                "per_action_sta_loop": False,
                "coordinate_commit_attempt_count": len(actions),
                "accepted_action_count": 0,
                "rejected_action_count": 0,
                "failed_action_count": len(actions),
                "skipped_action_count": 0,
                "results": [],
            }
            artifact_path = _write_commit_artifact(output_dir, artifact)
            summary = _commit_summary_from_artifact(
                artifact,
                commit_enabled=True,
                artifact_path=artifact_path,
            )
            summary["action_count"] = len(actions)
            return summary

        deferred_config = dict(config)
        deferred_config["defer_timing_update"] = True
        realization_results = []
        for index, action in enumerate(actions):
            if index > 0 and index % 100 == 0:
                print(
                    "[buffering] batch coordinate commit applied %d/%d actions"
                    % (index, len(actions)),
                    flush=True,
                )
            realization_results.append(
                realize_coordinate_buffer_action(bridge, action, deferred_config)
            )

        post_batch_timing_update = _run_post_batch_timing_update(bridge)
        try:
            after_metrics = _query_batch_timing_metrics(bridge)
        except Exception as exc:
            after_metrics = {"status": "failed", "error": str(exc)}

        artifact = build_coordinate_batch_measured_results_artifact(
            actions,
            realization_results,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
        )
        artifact["post_batch_timing_update"] = post_batch_timing_update
        artifact["batch_before_metrics_status"] = str(before_metrics.get("status", ""))
        artifact["batch_after_metrics_status"] = str(after_metrics.get("status", ""))
        if str(after_metrics.get("status", "")) == "failed":
            artifact["status"] = "failed"
            artifact["backend_status"] = "failed"
            artifact["reason"] = "batch_after_metrics_failed"
            artifact["error"] = str(after_metrics.get("error", ""))
    artifact["backend_status"] = (
        "ok"
        if artifact.get("failed_action_count", 0) == 0
        else "failed"
    )
    artifact_path = _write_commit_artifact(output_dir, artifact)
    summary = _commit_summary_from_artifact(
        artifact,
        commit_enabled=True,
        artifact_path=artifact_path,
    )
    summary["action_count"] = len(actions)
    return summary
