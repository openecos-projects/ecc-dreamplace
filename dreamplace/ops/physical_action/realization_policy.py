def _as_bool(config, name, default=False):
    if not config or name not in config:
        return bool(default)
    return bool(config.get(name))


def _as_float(config, name, default=0.0):
    if not config or name not in config:
        return float(default)
    try:
        return float(config.get(name))
    except (TypeError, ValueError):
        return float(default)


def _as_str(config, name, default=""):
    if not config or name not in config:
        return str(default)
    value = config.get(name)
    if value is None:
        return str(default)
    return str(value)


def _has_config_value(config, name):
    return bool(config) and config.get(name) is not None


def _is_qor_evidence_result_class(result_class):
    return str(result_class) == "buffering_selected_realization"


def _requires_incremental_legalization(legalization_status):
    return str(legalization_status or "") in {
        "placed_at_candidate_requires_later_legalization",
        "pre_legalization_coordinate_insert",
    }


def _base_result(action, *, status):
    result_class = (
        "debug_realization_smoke"
        if bool(action.get("debug_for_realization_smoke", False))
        else str(action.get("result_class", "buffering_selected_realization"))
    )
    return {
        "artifact": "buffering_one_net_realization_policy_result",
        "artifact_version": 1,
        "status": status,
        "action_id": int(action.get("action_id", -1)),
        "action_kind": str(action.get("action_kind", "")),
        "net_name": str(action.get("net_name", "")),
        "result_class": result_class,
        "selected_action_is_predicted_improving": bool(
            action.get(
                "selected_action_is_predicted_improving",
                result_class == "buffering_selected_realization",
            )
        ),
        "accept_policy": "reject_wns_regression",
        "qor_evidence": _is_qor_evidence_result_class(result_class),
        "accepted": False,
        "reject_reason": "",
        "db_commit_status": "not_attempted",
        "committed_to_source_design": False,
        "rollback_status": "not_attempted",
    }


def _metric(metrics, name):
    metrics = dict(metrics or {})
    value = metrics.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attach_measured_timing_fields(result):
    before_metrics = dict(result.get("before_metrics") or {})
    after_metrics = dict(result.get("after_metrics") or {})
    result["pre_wns"] = _metric(before_metrics, "wns")
    result["pre_tns"] = _metric(before_metrics, "tns")
    result["post_wns"] = _metric(after_metrics, "wns")
    result["post_tns"] = _metric(after_metrics, "tns")
    result["measured_before_after"] = (
        result["pre_wns"] is not None
        and result["pre_tns"] is not None
        and result["post_wns"] is not None
        and result["post_tns"] is not None
    )
    return result


def _balanced_acceptance_score(result, config):
    setup_delta = float(result.get("actual_delta_tns") or 0.0)
    slew_delta = float(result.get("slew_violation_count_delta") or 0.0)
    cap_delta = float(result.get("cap_violation_count_delta") or 0.0)
    buffer_delta = float(result.get("inserted_buffer_count_delta") or 0.0)
    slew_reduction = -slew_delta
    cap_reduction = -cap_delta
    score = (
        _as_float(config, "balanced_setup_weight", 1.0) * setup_delta
        + _as_float(config, "balanced_slew_weight", 0.0) * slew_reduction
        + _as_float(config, "balanced_cap_weight", 0.0) * cap_reduction
        - _as_float(config, "balanced_buffer_weight", 0.0) * buffer_delta
    )
    return float(score), {
        "setup_tns_delta": setup_delta,
        "slew_violation_delta": slew_delta,
        "slew_violation_reduction": slew_reduction,
        "cap_violation_delta": cap_delta,
        "cap_violation_reduction": cap_reduction,
        "inserted_buffer_count_delta": buffer_delta,
        "setup_weight": _as_float(config, "balanced_setup_weight", 1.0),
        "slew_weight": _as_float(config, "balanced_slew_weight", 0.0),
        "cap_weight": _as_float(config, "balanced_cap_weight", 0.0),
        "buffer_weight": _as_float(config, "balanced_buffer_weight", 0.0),
    }


def _apply_coordinate_accept_policy(result, config):
    max_wns_degradation = _as_float(config, "max_wns_degradation", 0.0)
    if result["actual_delta_wns"] < -max_wns_degradation:
        result["status"] = "rejected"
        result["reject_reason"] = "wns_regression"
        return result

    accept_policy = _as_str(config, "accept_policy", "")
    if accept_policy == "balanced_setup_slew_cap":
        score, breakdown = _balanced_acceptance_score(result, config)
        result["accept_policy"] = "balanced_setup_slew_cap"
        result["balanced_acceptance_score"] = score
        result["balanced_acceptance_min_score"] = _as_float(
            config,
            "balanced_min_score",
            0.0,
        )
        result["balanced_acceptance_breakdown"] = breakdown
        if score < result["balanced_acceptance_min_score"]:
            result["status"] = "rejected"
            result["reject_reason"] = "balanced_score_below_threshold"
            return result
        result["status"] = "accepted"
        result["accepted"] = True
        result["committed_to_source_design"] = not _as_bool(
            config,
            "disposable_design",
            False,
        )
        return result

    if _has_config_value(config, "max_tns_degradation"):
        result["accept_policy"] = "reject_wns_tns_regression"
        max_tns_degradation = _as_float(config, "max_tns_degradation", 0.0)
        if result["actual_delta_tns"] < -max_tns_degradation:
            result["status"] = "rejected"
            result["reject_reason"] = "tns_regression"
            return result

    result["status"] = "accepted"
    result["accepted"] = True
    result["committed_to_source_design"] = not _as_bool(
        config,
        "disposable_design",
        False,
    )
    return result


def realize_one_net_buffer_action(openroad_bridge, action, config=None):
    action = dict(action or {})
    config = dict(config or {})
    result = _base_result(action, status="skipped")

    if not _as_bool(config, "enable_realization", False):
        result["skip_reason"] = "realization_disabled"
        return result

    net_name = str(action.get("net_name", "")).strip()
    if not net_name:
        result["status"] = "failed"
        result["error"] = "one-net realization requires action.net_name"
        return result
    if openroad_bridge is None or not hasattr(openroad_bridge, "run_one_net_buffer"):
        result["status"] = "failed"
        result["error"] = "OpenROAD bridge missing run_one_net_buffer"
        return result

    disposable_design = _as_bool(config, "disposable_design", False)
    allow_in_place_without_rollback = _as_bool(
        config,
        "allow_in_place_without_rollback",
        False,
    )
    if not disposable_design and not allow_in_place_without_rollback:
        result["status"] = "skipped"
        result["skip_reason"] = "rollback_or_disposable_design_required"
        return result

    backend_config = {
        "max_wire_length": _as_float(config, "max_wire_length", 0.0),
        "slew_margin": _as_float(config, "slew_margin", 0.0),
        "cap_margin": _as_float(config, "cap_margin", 0.0),
    }
    backend_result = openroad_bridge.run_one_net_buffer(net_name, backend_config)
    backend_result = dict(backend_result or {})
    result["backend_result"] = backend_result
    result["backend_status"] = str(backend_result.get("status", "failed"))
    result["before_metrics"] = backend_result.get("before_metrics")
    result["after_metrics"] = backend_result.get("after_metrics")
    result["actual_delta_wns"] = float(backend_result.get("actual_delta_wns") or 0.0)
    result["actual_delta_tns"] = float(backend_result.get("actual_delta_tns") or 0.0)
    result["setup_violation_count_delta"] = int(
        backend_result.get("setup_violation_count_delta") or 0
    )
    result["inserted_buffer_count_delta"] = int(
        backend_result.get("inserted_buffer_count_delta") or 0
    )
    result["instance_count_delta"] = int(backend_result.get("instance_count_delta") or 0)
    result["net_count_delta"] = int(backend_result.get("net_count_delta") or 0)
    result["command"] = backend_result.get("command")
    result["realization_backend"] = backend_result.get("realization_backend")
    result["disposable_design"] = disposable_design
    result["rollback_status"] = (
        "not_needed_disposable_design"
        if disposable_design
        else "not_available_in_place_allowed"
    )
    result["db_commit_status"] = (
        "committed_to_disposable_design"
        if disposable_design
        else "committed_to_source_design"
    )
    _attach_measured_timing_fields(result)

    if result["backend_status"] != "ok":
        result["status"] = "failed"
        result["error"] = backend_result.get("error", "backend_failed")
        result["db_commit_status"] = "backend_failed_not_committed"
        return result

    return _apply_coordinate_accept_policy(result, config)


def realize_coordinate_buffer_action(openroad_bridge, action, config=None):
    action = dict(action or {})
    config = dict(config or {})
    result = _base_result(action, status="skipped")

    if not _as_bool(config, "enable_realization", False):
        result["skip_reason"] = "realization_disabled"
        return result

    if openroad_bridge is None or not hasattr(openroad_bridge, "run_coordinate_buffer_insert"):
        result["status"] = "failed"
        result["error"] = "OpenROAD bridge missing run_coordinate_buffer_insert"
        return result

    disposable_design = _as_bool(config, "disposable_design", False)
    allow_in_place_without_rollback = _as_bool(
        config,
        "allow_in_place_without_rollback",
        False,
    )
    if not disposable_design and not allow_in_place_without_rollback:
        result["status"] = "skipped"
        result["skip_reason"] = "rollback_or_disposable_design_required"
        return result

    backend_config = {
        "allow_singleton_load_partition": _as_bool(
            config,
            "allow_singleton_load_partition",
            False,
        ),
        "load_partition_mode": _as_str(config, "load_partition_mode", "singleton"),
        "defer_timing_update": _as_bool(config, "defer_timing_update", False),
    }
    backend_result = openroad_bridge.run_coordinate_buffer_insert(action, backend_config)
    backend_result = dict(backend_result or {})
    result["backend_result"] = backend_result
    result["backend_status"] = str(backend_result.get("status", "failed"))
    result["before_metrics"] = backend_result.get("before_metrics")
    result["after_metrics"] = backend_result.get("after_metrics")
    result["actual_delta_wns"] = float(backend_result.get("actual_delta_wns") or 0.0)
    result["actual_delta_tns"] = float(backend_result.get("actual_delta_tns") or 0.0)
    result["setup_violation_count_delta"] = int(
        backend_result.get("setup_violation_count_delta") or 0
    )
    result["slew_violation_count_delta"] = float(
        backend_result.get("slew_violation_count_delta") or 0.0
    )
    result["cap_violation_count_delta"] = float(
        backend_result.get("cap_violation_count_delta") or 0.0
    )
    result["inserted_buffer_count_delta"] = int(
        backend_result.get("inserted_buffer_count_delta") or 0
    )
    result["instance_count_delta"] = int(backend_result.get("instance_count_delta") or 0)
    result["net_count_delta"] = int(backend_result.get("net_count_delta") or 0)
    result["realization_backend"] = backend_result.get("realization_backend")
    result["implementation_scope"] = backend_result.get("implementation_scope")
    result["source_net_name"] = backend_result.get("source_net_name", "")
    result["resolved_target_net_name"] = backend_result.get(
        "resolved_target_net_name",
        "",
    )
    result["target_net_remapped"] = backend_result.get("target_net_remapped")
    result["resolved_driver_pin_name"] = backend_result.get(
        "resolved_driver_pin_name",
        "",
    )
    result["driver_pin_remapped"] = backend_result.get("driver_pin_remapped")
    result["inserted_buffer_name"] = backend_result.get("inserted_buffer_name")
    result["downstream_net_name"] = backend_result.get("downstream_net_name")
    result["moved_load_count"] = int(backend_result.get("moved_load_count") or 0)
    result["requested_downstream_pin_count"] = int(
        backend_result.get("requested_downstream_pin_count") or 0
    )
    result["moved_load_pin_names"] = list(backend_result.get("moved_load_pin_names") or [])
    result["hard_commit_diagnostic"] = dict(
        backend_result.get("hard_commit_diagnostic") or {}
    )
    result["hard_commit_diagnostic_status"] = str(
        result["hard_commit_diagnostic"].get("status", "missing")
    )
    result["hard_commit_topology_delta"] = dict(
        backend_result.get(
            "hard_commit_topology_delta",
            result["hard_commit_diagnostic"].get("topology_delta", {}),
        )
        or {}
    )
    result["hard_commit_load_partition"] = dict(
        backend_result.get(
            "hard_commit_load_partition",
            result["hard_commit_diagnostic"].get("load_partition", {}),
        )
        or {}
    )
    result["hard_commit_cell_arc_delay_samples"] = dict(
        backend_result.get(
            "hard_commit_cell_arc_delay_samples",
            result["hard_commit_diagnostic"].get("cell_arc_delay_samples", {}),
        )
        or {}
    )
    result["legalization_status"] = backend_result.get(
        "legalization_status",
        "unknown_coordinate_legalization_status",
    )
    result["requires_incremental_legalization"] = _requires_incremental_legalization(
        result["legalization_status"]
    )
    result["post_legalization_status"] = (
        "not_run" if result["requires_incremental_legalization"] else "not_required_or_unknown"
    )
    result["disposable_design"] = disposable_design
    result["rollback_status"] = (
        "not_needed_disposable_design"
        if disposable_design
        else "not_available_in_place_allowed"
    )
    result["db_commit_status"] = (
        "committed_to_disposable_design"
        if disposable_design
        else "committed_to_source_design"
    )
    _attach_measured_timing_fields(result)

    if result["backend_status"] != "ok":
        result["status"] = "failed"
        result["error"] = backend_result.get("error", "backend_failed")
        result["db_commit_status"] = "backend_failed_not_committed"
        return result

    return _apply_coordinate_accept_policy(result, config)
