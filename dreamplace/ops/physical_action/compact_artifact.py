def _require_index(values, index, field_name):
    if index < 0 or index >= len(values):
        raise ValueError(f"{field_name} missing value for bsu={index}")
    return values[index]


def _optional_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_OPTIONAL_PROVENANCE_FIELDS = {
    "segment_id": int,
    "z_value": float,
    "bsu_value": float,
    "projected_bsu_index": int,
    "projected_repeater_count": int,
    "projection_policy": str,
    "bsu_sharing": str,
    "max_repeater_count": int,
    "segment_split_index": int,
    "coordinate_action_source": str,
    "selected_action_source": str,
    "selection_result_class": str,
    "result_class": str,
}


def _copy_optional_provenance(action, entry):
    for key, converter in _OPTIONAL_PROVENANCE_FIELDS.items():
        if key not in entry or entry.get(key) is None:
            continue
        action[key] = converter(entry.get(key))
    if "source_flags" in entry:
        action["source_flags"] = [
            str(value) for value in list(entry.get("source_flags", []) or [])
        ]


def build_backend_probe(*, executable_by_compact_batch_v1=False):
    compact_status = (
        "available"
        if executable_by_compact_batch_v1
        else "unsupported_non_sizing_transaction_mutator"
    )
    return {
        "preferred_backend": "openroad_one_net",
        "compact_batch": {
            "status": compact_status,
            "supports_buffer_insert": bool(executable_by_compact_batch_v1),
            "evidence": "ActionKind::kBufferInsert parses, but current transaction mutator rejects non-sizing actions",
        },
        "openroad_one_net": {
            "status": "available_scoped_repair",
            "supports_buffer_insert": True,
            "diagnostic_only": True,
            "evidence": "runOneNetBuffer executes OpenROAD rsz::repair_net_cmd and returns before/after OpenSTA metrics; OpenROAD chooses repair topology/locations, so this is not a Du candidate-coordinate commit backend",
        },
        "coordinate_or_branch_specific": {
            "status": "required_not_implemented",
            "supports_buffer_insert": False,
            "diagnostic_only": False,
            "evidence": "Required for fair Du QoR: create buffer at candidate coordinate, split net, reconnect selected load partition, legalize/sync, and verify with OpenROAD/OpenSTA",
        },
    }


def build_realization_boundary(*, actions, executable_by_compact_batch_v1=False):
    blockers = []
    if not actions:
        blockers.append("no_candidate_actions")
    if not executable_by_compact_batch_v1:
        blockers.append("compact_batch_buffer_insert_not_supported")
    if actions and any(not str(action.get("net_name", "")).strip() for action in actions):
        blockers.append("one_net_openroad_bridge_requires_net_name")
    realization_ready = bool(actions) and "one_net_openroad_bridge_requires_net_name" not in blockers

    return {
        "realization_ready": realization_ready,
        "realization_mode": "ready_for_openroad_one_net" if realization_ready else "artifact_only",
        "supported_realization_backends": ["openroad_one_net"] if realization_ready else [],
        "unsupported_realization_reasons": blockers,
        "backend_probe": build_backend_probe(
            executable_by_compact_batch_v1=executable_by_compact_batch_v1,
        ),
        "requires_openroad_opensta_verification": bool(actions),
        "rollback_required": bool(actions),
    }


def build_commit_action_candidates(
    virtual_state,
    *,
    legal_table,
    executable_by_compact_batch_v1=False,
):
    legal_cell_ids = [int(value) for value in legal_table.get("legal_cell_ids", [])]
    legal_master_names = [str(value) for value in legal_table.get("legal_master_names", [])]
    legal_size_values = [float(value) for value in legal_table.get("legal_size_values", [])]
    if not legal_cell_ids or len(legal_cell_ids) != len(legal_master_names):
        raise ValueError("legal cell ids and master names are required")

    actions = []
    for entry in virtual_state.entries.values():
        if int(entry.get("bu", 0)) != 1:
            continue
        bsu = int(entry.get("bsu", -1))
        buffer_master_id = _require_index(legal_cell_ids, bsu, "legal_cell_ids")
        buffer_master_name = _require_index(legal_master_names, bsu, "legal_master_names")
        legal_size_value = _require_index(legal_size_values, bsu, "legal_size_values")
        x_dbu = float(entry.get("x_dbu", entry.get("candidate_location_x", 0.0)))
        y_dbu = float(entry.get("y_dbu", entry.get("candidate_location_y", 0.0)))
        candidate_id = int(entry["candidate_id"])
        action = {
            "action_id": candidate_id,
            "candidate_id": candidate_id,
            "action_kind": "buffer_insert",
            "affected_net_id": int(entry.get("net_id", entry.get("affected_net_id", -1))),
            "affected_net_name": str(entry.get("net_name", "")),
            "net_name": str(entry.get("net_name", "")),
            "driver_pin_id": int(entry.get("driver_pin_id", -1)),
            "driver_pin_name": str(entry.get("driver_pin_name", "")),
            "load_pin_id": int(entry.get("load_pin_id", -1)),
            "load_pin_name": str(entry.get("load_pin_name", "")),
            "downstream_pin_ids": [
                int(value) for value in list(entry.get("downstream_pin_ids", []) or [])
            ],
            "downstream_pin_names": [
                str(value) for value in list(entry.get("downstream_pin_names", []) or [])
            ],
            "downstream_pin_count": int(entry.get("downstream_pin_count", 0) or 0),
            "downstream_sink_count": int(entry.get("downstream_sink_count", 0) or 0),
            "total_sink_count": int(entry.get("total_sink_count", 0) or 0),
            "segment_id": (
                int(entry["segment_id"])
                if entry.get("segment_id") is not None
                else None
            ),
            "segment_parent_node_id": (
                int(entry["segment_parent_node_id"])
                if entry.get("segment_parent_node_id") is not None
                else None
            ),
            "segment_child_node_id": (
                int(entry["segment_child_node_id"])
                if entry.get("segment_child_node_id") is not None
                else None
            ),
            "segment_tree_depth": (
                int(entry["segment_tree_depth"])
                if entry.get("segment_tree_depth") is not None
                else None
            ),
            "segment_split_index": (
                int(entry["segment_split_index"])
                if entry.get("segment_split_index") is not None
                else None
            ),
            "segment_split_count_on_edge": (
                int(entry["segment_split_count_on_edge"])
                if entry.get("segment_split_count_on_edge") is not None
                else None
            ),
            "segment_split_ratio": (
                float(entry["segment_split_ratio"])
                if entry.get("segment_split_ratio") is not None
                else None
            ),
            "downstream_sink_cap_sum": float(
                entry.get("downstream_sink_cap_sum", 0.0) or 0.0
            ),
            "total_sink_cap_sum": float(entry.get("total_sink_cap_sum", 0.0) or 0.0),
            "moved_branch_sink_cap_sum": float(
                entry.get("moved_branch_sink_cap_sum", 0.0) or 0.0
            ),
            "upstream_sibling_sink_cap_sum": float(
                entry.get("upstream_sibling_sink_cap_sum", 0.0) or 0.0
            ),
            "buffer_input_cap_minus_moved_branch_cap": float(
                entry.get("buffer_input_cap_minus_moved_branch_cap", 0.0) or 0.0
            ),
            "buffer_input_cap_to_moved_branch_cap_ratio": (
                _optional_float(entry.get("buffer_input_cap_to_moved_branch_cap_ratio"))
            ),
            "downstream_sink_cap_ratio": float(
                entry.get("downstream_sink_cap_ratio", 0.0) or 0.0
            ),
            "downstream_sink_cap_mean": float(
                entry.get("downstream_sink_cap_mean", 0.0) or 0.0
            ),
            "downstream_sink_cap_max": float(
                entry.get("downstream_sink_cap_max", 0.0) or 0.0
            ),
            "downstream_negative_slack_sink_count": int(
                entry.get("downstream_negative_slack_sink_count", 0) or 0
            ),
            "total_negative_slack_sink_count": int(
                entry.get("total_negative_slack_sink_count", 0) or 0
            ),
            "upstream_sibling_negative_slack_sink_count": int(
                entry.get("upstream_sibling_negative_slack_sink_count", 0) or 0
            ),
            "downstream_worst_sink_slack": float(
                entry.get("downstream_worst_sink_slack", 0.0) or 0.0
            ),
            "total_worst_sink_slack": float(
                entry.get("total_worst_sink_slack", 0.0) or 0.0
            ),
            "downstream_sink_criticality_sum": float(
                entry.get("downstream_sink_criticality_sum", 0.0) or 0.0
            ),
            "total_sink_criticality_sum": float(
                entry.get("total_sink_criticality_sum", 0.0) or 0.0
            ),
            "moved_branch_criticality_sum": float(
                entry.get("moved_branch_criticality_sum", 0.0) or 0.0
            ),
            "upstream_sibling_criticality_sum": float(
                entry.get("upstream_sibling_criticality_sum", 0.0) or 0.0
            ),
            "downstream_sink_criticality_ratio": float(
                entry.get("downstream_sink_criticality_ratio", 0.0) or 0.0
            ),
            "downstream_sink_criticality_mean": float(
                entry.get("downstream_sink_criticality_mean", 0.0) or 0.0
            ),
            "downstream_sink_criticality_max": float(
                entry.get("downstream_sink_criticality_max", 0.0) or 0.0
            ),
            "moved_branch_has_setup_criticality": bool(
                entry.get("moved_branch_has_setup_criticality", False)
            ),
            "sibling_has_setup_criticality": bool(
                entry.get("sibling_has_setup_criticality", False)
            ),
            "moved_noncritical_branch_with_critical_sibling": bool(
                entry.get("moved_noncritical_branch_with_critical_sibling", False)
            ),
            "buffer_input_cap_exceeds_moved_branch_cap": bool(
                entry.get("buffer_input_cap_exceeds_moved_branch_cap", False)
            ),
            "noncritical_moved_branch_replaced_by_larger_buffer_input_cap": bool(
                entry.get(
                    "noncritical_moved_branch_replaced_by_larger_buffer_input_cap",
                    False,
                )
            ),
            "load_partition_source": str(entry.get("load_partition_source", "")),
            "parent_node_id": (
                int(entry["parent_node_id"]) if entry.get("parent_node_id") is not None else None
            ),
            "child_node_ids": [
                int(value) for value in list(entry.get("child_node_ids", []) or [])
            ],
            "bsu": bsu,
            "bsu_selection_mode": str(entry.get("bsu_selection_mode", "fixed_bsu")),
            "bsu_candidate_count": int(entry.get("bsu_candidate_count", 1) or 1),
            "bsu_candidate_evaluations": [
                dict(value)
                for value in list(entry.get("bsu_candidate_evaluations", []) or [])
            ],
            "buffer_main_type_index": int(entry.get("buffer_main_type_index", -1)),
            "buffer_cell_id": buffer_master_id,
            "buffer_master_id": buffer_master_id,
            "buffer_master_name": buffer_master_name,
            "candidate_location_x": x_dbu,
            "candidate_location_y": y_dbu,
            "candidate_location_x_dbu": x_dbu,
            "candidate_location_y_dbu": y_dbu,
            "candidate_location_x_dbu_analytical": _optional_float(
                entry.get("candidate_location_x_dbu_analytical")
            ),
            "candidate_location_y_dbu_analytical": _optional_float(
                entry.get("candidate_location_y_dbu_analytical")
            ),
            "segment_parent_x_dbu": _optional_float(
                entry.get("segment_parent_x_dbu")
            ),
            "segment_parent_y_dbu": _optional_float(
                entry.get("segment_parent_y_dbu")
            ),
            "segment_child_x_dbu": _optional_float(
                entry.get("segment_child_x_dbu")
            ),
            "segment_child_y_dbu": _optional_float(
                entry.get("segment_child_y_dbu")
            ),
            "coordinate_source": str(entry.get("coordinate_source", "")),
            "placement_snapshot_identity": str(
                entry.get("placement_snapshot_identity", "")
            ),
            "placement_snapshot_iteration": entry.get(
                "placement_snapshot_iteration"
            ),
            "topology_generation": entry.get("topology_generation"),
            "frozen_topology_generation": entry.get(
                "frozen_topology_generation"
            ),
            "rectilinear_path_policy": str(
                entry.get("rectilinear_path_policy", "")
            ),
            "candidate_location_x_um": float(entry.get("x_um", 0.0)),
            "candidate_location_y_um": float(entry.get("y_um", 0.0)),
            "current_size_idx": -1,
            "target_size_idx": bsu,
            "seed_size_idx": bsu,
            "legal_cell_id_candidates": legal_cell_ids,
            "legal_master_candidates": legal_master_names,
            "legal_timing_coordinate_candidates": legal_size_values,
            "legal_size_value": float(legal_size_value),
            "scoring_node_id": int(entry.get("scoring_node_id", -1)),
            "scoring_node_source": str(entry.get("scoring_node_source", "")),
            "scoring_location_model": str(entry.get("scoring_location_model", "")),
            "scoring_location_is_exact": bool(entry.get("scoring_location_is_exact", False)),
            "candidate_input_slew": float(entry.get("candidate_input_slew", 0.0)),
            "candidate_output_cap": float(entry.get("candidate_output_cap", 0.0)),
            "buffer_input_cap": float(entry.get("buffer_input_cap", 0.0)),
            "buffer_delay": float(entry.get("buffer_delay", 0.0)),
            "buffer_output_slew": float(entry.get("buffer_output_slew", 0.0)),
            "delay_source": str(entry.get("delay_source", "")),
            "transition_source": str(entry.get("transition_source", "")),
            "delay_lut_status": dict(entry.get("delay_lut_status", {}) or {}),
            "transition_lut_status": dict(entry.get("transition_lut_status", {}) or {}),
            "baseline_obj": float(entry.get("baseline_obj", 0.0)),
            "trial_obj": float(entry.get("trial_obj", 0.0)),
            "gradient_chain": dict(entry.get("gradient_chain", {}) or {}),
            "grad_bu": _optional_float(entry.get("grad_bu")),
            "grad_bu_source": str(entry.get("grad_bu_source", "missing_grad_bu")),
            "grad_bu_is_true_objective_gradient": bool(
                entry.get("grad_bu_is_true_objective_gradient", False)
            ),
            "grad_bu_action_scope": str(entry.get("grad_bu_action_scope", "")),
            "grad_bu_is_action_aligned": bool(
                entry.get("grad_bu_is_action_aligned", False)
            ),
            "selection_signal": str(entry.get("selection_signal", "")),
            "autograd_grad_bu": _optional_float(entry.get("autograd_grad_bu")),
            "autograd_bu_logit_grad": _optional_float(
                entry.get("autograd_bu_logit_grad")
            ),
            "native_finite_difference_delta_obj": _optional_float(
                entry.get("native_finite_difference_delta_obj")
            ),
            "toy_rc_finite_difference_delta_obj": _optional_float(
                entry.get("toy_rc_finite_difference_delta_obj")
            ),
            "predicted_delta_obj": float(entry.get("predicted_delta_obj", 0.0)),
            "predicted_delta_tns": float(entry.get("predicted_delta_tns", 0.0)),
            "predicted_delta_wns": float(entry.get("predicted_delta_wns", 0.0)),
            "physical_delta_obj": float(
                entry.get("physical_delta_obj", entry.get("predicted_delta_obj", 0.0))
            ),
            "baseline_sink_slew_obj": float(entry.get("baseline_sink_slew_obj", 0.0)),
            "trial_sink_slew_obj": float(entry.get("trial_sink_slew_obj", 0.0)),
            "physical_slew_delta_obj": float(entry.get("physical_slew_delta_obj", 0.0)),
            "baseline_sink_load_obj": float(entry.get("baseline_sink_load_obj", 0.0)),
            "trial_sink_load_obj": float(entry.get("trial_sink_load_obj", 0.0)),
            "physical_load_delta_obj": float(entry.get("physical_load_delta_obj", 0.0)),
            "predicted_slew_violation_reduction": float(
                entry.get("predicted_slew_violation_reduction", 0.0)
            ),
            "predicted_cap_violation_reduction": float(
                entry.get("predicted_cap_violation_reduction", 0.0)
            ),
            "raw_setup_criticality_reduction": float(
                entry.get("raw_setup_criticality_reduction", 0.0)
            ),
            "predicted_setup_criticality_reduction": float(
                entry.get("predicted_setup_criticality_reduction", 0.0)
            ),
            "setup_pressure_guard_applied": bool(
                entry.get("setup_pressure_guard_applied", False)
            ),
            "setup_pressure_guard_reason": str(
                entry.get("setup_pressure_guard_reason", "")
            ),
            "violator_pressure_reward": float(entry.get("violator_pressure_reward", 0.0)),
            "objective_uses_violator_pressure": bool(
                entry.get("objective_uses_violator_pressure", False)
            ),
            "branch_pressure_mode": str(entry.get("branch_pressure_mode", "none")),
            "branch_pressure_applied": bool(entry.get("branch_pressure_applied", False)),
            "branch_downstream_violator_pin_count": float(
                entry.get("branch_downstream_violator_pin_count", 0.0)
            ),
            "branch_downstream_violator_pin_names": [
                str(value)
                for value in list(entry.get("branch_downstream_violator_pin_names", []) or [])
            ],
            "branch_violator_coverage_ratio": float(
                entry.get("branch_violator_coverage_ratio", 1.0)
            ),
            "branch_downstream_sink_cap_sum": float(
                entry.get("branch_downstream_sink_cap_sum", 0.0) or 0.0
            ),
            "branch_total_sink_cap_sum": float(
                entry.get("branch_total_sink_cap_sum", 0.0) or 0.0
            ),
            "branch_downstream_sink_cap_ratio": float(
                entry.get("branch_downstream_sink_cap_ratio", 0.0) or 0.0
            ),
            "branch_load_cap_equivalent_pin_count": float(
                entry.get("branch_load_cap_equivalent_pin_count", 0.0) or 0.0
            ),
            "branch_downstream_load_cap_bonus": float(
                entry.get("branch_downstream_load_cap_bonus", 0.0) or 0.0
            ),
            "effective_violator_net_priority": float(
                entry.get("effective_violator_net_priority", entry.get("violator_net_priority", 0.0))
            ),
            "predicted_improvement": float(entry.get("predicted_improvement", -float(entry.get("predicted_delta_obj", 0.0)))),
            "selection_score": float(entry.get("selection_score", entry.get("predicted_improvement", 0.0))),
            "violator_net_priority": float(entry.get("violator_net_priority", 0.0)),
            "violator_pin_count": int(entry.get("violator_pin_count", 0)),
            "violator_worst_slack": float(entry.get("violator_worst_slack", 0.0)),
            "violator_check_types": list(entry.get("violator_check_types", [])),
            "violator_pins": [str(value) for value in list(entry.get("violator_pins", []) or [])],
            "estimator_source": str(entry.get("estimator_source", "buffering_virtual")),
            "source_score": float(
                entry.get(
                    "selection_score",
                    entry.get(
                        "predicted_improvement",
                        -float(entry.get("predicted_delta_obj", 0.0)),
                    ),
                )
            ),
            "source_sink_slack": float(entry.get("source_sink_slack", 0.0)),
            "source_sink_npath": int(entry.get("source_sink_npath", 1)),
            "source_sink_criticality": float(entry.get("source_sink_criticality", 0.0)),
            "result_class": str(entry.get("result_class", "buffering_selected_realization")),
            "selection_result_class": str(
                entry.get(
                    "selection_result_class",
                    entry.get("result_class", "buffering_selected_realization"),
                )
            ),
            "selected_action_source": str(entry.get("selected_action_source", "")),
            "selected_action_is_predicted_improving": bool(
                entry.get(
                    "selected_action_is_predicted_improving",
                    float(entry.get("predicted_delta_obj", 0.0)) < 0.0,
                )
            ),
            "debug_for_realization_smoke": bool(entry.get("debug_for_realization_smoke", False)),
        }
        _copy_optional_provenance(action, entry)
        if entry.get("debug_reason"):
            action["debug_reason"] = str(entry.get("debug_reason"))
        actions.append(action)

    artifact = {
        "artifact": "buffering_commit_action_candidates",
        "artifact_version": 1,
        "schema_compatible_with_compact_action": True,
        "executable_by_compact_batch_v1": bool(executable_by_compact_batch_v1),
        "committed_buffer_count": int(virtual_state.committed_buffer_count),
        "virtual_buffer_count": int(virtual_state.virtual_buffer_count),
        "candidate_commit_action_count": len(actions),
        "actions": actions,
    }
    artifact["realization_boundary"] = build_realization_boundary(
        actions=actions,
        executable_by_compact_batch_v1=executable_by_compact_batch_v1,
    )
    return artifact


def build_commit_action_candidates_from_entries(
    entries,
    *,
    legal_table,
    executable_by_compact_batch_v1=False,
):
    class _EntryBackedVirtualState:
        def __init__(self, source_entries):
            self.entries = {}
            for fallback_id, source in enumerate(list(source_entries or [])):
                entry = dict(source)
                candidate_id = int(entry.get("candidate_id", fallback_id))
                bsu = int(entry.get("bsu", entry.get("projected_bsu_index", -1)))
                entry["candidate_id"] = candidate_id
                entry["bu"] = int(entry.get("bu", 1))
                entry["bsu"] = bsu
                self.entries[candidate_id] = entry
            self.committed_buffer_count = 0

        @property
        def virtual_buffer_count(self):
            return sum(
                1 for entry in self.entries.values() if int(entry.get("bu", 0)) == 1
            )

    return build_commit_action_candidates(
        _EntryBackedVirtualState(entries),
        legal_table=legal_table,
        executable_by_compact_batch_v1=executable_by_compact_batch_v1,
    )
