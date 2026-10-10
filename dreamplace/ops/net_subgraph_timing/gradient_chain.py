import torch

from .tensor_builder import build_net_subgraph_timing_inputs
from .net_subgraph_timing import net_subgraph_forward_native


def _default_lut_status(source):
    return {
        "source": str(source),
        "input_slew_clamped": False,
        "input_slew_clamp": "none",
        "input_slew_axis_min": None,
        "input_slew_axis_max": None,
        "output_cap_clamped": False,
        "output_cap_clamp": "none",
        "output_cap_axis_min": None,
        "output_cap_axis_max": None,
    }


def _surrogate_lut_status(surrogate, key):
    status = dict(surrogate.get(key, {}))
    if not status:
        return _default_lut_status("buffer_surrogate")
    default = _default_lut_status(status.get("source", "buffer_surrogate"))
    default.update(status)
    return default


def _candidate_net_id(candidate, net_ids):
    value = candidate.get("net_id", candidate.get("affected_net_id"))
    if value is not None:
        return int(value), None
    if len(net_ids) == 1:
        return int(net_ids[0]), None
    return None, "missing_candidate_net_id"


def _candidate_scoring_node(candidate):
    tree_node_id = candidate.get("tree_node_id")
    if tree_node_id is not None:
        return int(tree_node_id), "tree_node"
    child_node_ids = candidate.get("child_node_ids") or []
    if len(child_node_ids) == 1:
        return int(child_node_ids[0]), "segment_child_node"
    return None, None


def _candidate_segment_edge(candidate):
    if candidate.get("tree_node_id") is not None:
        return None
    parent = candidate.get("parent_node_id")
    child_node_ids = candidate.get("child_node_ids") or []
    if parent is None or len(child_node_ids) != 1:
        return None
    return int(parent), int(child_node_ids[0])


def _scoring_location_model(scoring_node_source):
    if scoring_node_source == "tree_node":
        return "exact_tree_node", True
    if scoring_node_source == "segment_edge_split":
        return "exact_segment_split", True
    if scoring_node_source == "segment_child_node":
        return "segment_child_node_proxy", False
    return "unknown", False


def _native_sink_objective(result, net_id):
    return _native_sink_max(result, net_id, "sink_arrival")


def _native_sink_max(result, net_id, key):
    sink_net_index = result.get("sink_net_index")
    values = result.get(key)
    if values is None or values.numel() == 0:
        return 0.0
    if sink_net_index is None or sink_net_index.numel() == 0:
        return float(values.max().item())
    mask = sink_net_index == int(net_id)
    if not bool(mask.any().item()):
        return 0.0
    return float(values[mask].max().item())


def _native_inputs_only(inputs):
    return {key: inputs[key] for key in inputs["native_input_keys"]}


def _sink_physical_delta_metrics(baseline, trial, net_id):
    baseline_slew = _native_sink_max(baseline, net_id, "sink_slew")
    trial_slew = _native_sink_max(trial, net_id, "sink_slew")
    baseline_load = _native_sink_max(baseline, net_id, "sink_load")
    trial_load = _native_sink_max(trial, net_id, "sink_load")
    return {
        "baseline_sink_slew_obj": baseline_slew,
        "trial_sink_slew_obj": trial_slew,
        "physical_slew_delta_obj": float(trial_slew) - float(baseline_slew),
        "baseline_sink_load_obj": baseline_load,
        "trial_sink_load_obj": trial_load,
        "physical_load_delta_obj": float(trial_load) - float(baseline_load),
    }


def _bsu_candidate_values(bsu, bsu_candidates):
    raw_values = [bsu] if bsu_candidates is None else list(bsu_candidates)
    values = []
    seen = set()
    for value in raw_values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        values.append(int(bsu))
    return values


def _score_bsu_trials(
    *,
    nets,
    candidate,
    net_id,
    bsu_values,
    buffer_surrogate,
    candidate_input_slew,
    candidate_output_cap,
    baseline_result,
    baseline_obj,
    node_id=None,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    dtype=None,
    device=None,
):
    evaluations = []
    for bsu_value in bsu_values:
        surrogate = dict(
            buffer_surrogate(
                candidate,
                bsu=int(bsu_value),
                input_slew=candidate_input_slew,
                output_cap=candidate_output_cap,
            )
        )
        trial_candidate = dict(candidate)
        trial_candidate.update(
            {
                "bu": 1.0,
                "buffer_input_cap": float(surrogate["buffer_input_cap"]),
                "buffer_delay": float(surrogate["buffer_delay"]),
                "buffer_output_slew": float(surrogate["buffer_output_slew"]),
            }
        )
        if node_id is not None:
            trial_candidate["node_id"] = int(node_id)
        trial_inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[trial_candidate],
            driver_arrival_by_net=driver_arrival_by_net,
            driver_slew_by_net=driver_slew_by_net,
            dtype=dtype,
            device=device,
        )
        trial = net_subgraph_forward_native(**_native_inputs_only(trial_inputs))
        trial_obj = _native_sink_objective(trial, net_id)
        delta = float(trial_obj) - float(baseline_obj)
        sink_delta_metrics = _sink_physical_delta_metrics(
            baseline_result,
            trial,
            net_id,
        )
        evaluations.append(
            {
                "bsu": int(bsu_value),
                "buffer_input_cap": float(surrogate["buffer_input_cap"]),
                "buffer_delay": float(surrogate["buffer_delay"]),
                "buffer_output_slew": float(surrogate["buffer_output_slew"]),
                "delay_source": str(surrogate.get("delay_source", "buffer_surrogate")),
                "transition_source": str(
                    surrogate.get("transition_source", "buffer_surrogate")
                ),
                "delay_lut_status": _surrogate_lut_status(
                    surrogate,
                    "delay_lut_status",
                ),
                "transition_lut_status": _surrogate_lut_status(
                    surrogate,
                    "transition_lut_status",
                ),
                "baseline_obj": float(baseline_obj),
                "trial_obj": float(trial_obj),
                "physical_delta_obj": delta,
                **sink_delta_metrics,
                "predicted_delta_obj": delta,
                "predicted_improvement": -delta,
                "selection_score": -delta,
            }
        )
    selected = dict(
        min(
            evaluations,
            key=lambda item: (float(item["predicted_delta_obj"]), int(item["bsu"])),
        )
    )
    selected["_all_evaluations"] = [dict(evaluation) for evaluation in evaluations]
    return selected


def _can_exact_segment_split(net, candidate):
    segment = _candidate_segment_edge(candidate)
    if net is None or segment is None:
        return False
    parent, child = segment
    coordinates = net.get("coordinates") or {}
    return (
        parent in coordinates
        and child in coordinates
        and "x_dbu" in candidate
        and "y_dbu" in candidate
    )


def score_candidates_by_native_net_subgraph_delta(
    nets,
    candidates,
    *,
    bsu,
    bsu_candidates=None,
    buffer_surrogate,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    dtype=None,
    device=None,
    estimator_source="native_net_subgraph_candidate_coordinate_finite_difference",
):
    """Score candidate insertion deltas through the native net-subgraph op.

    This is the explicit buffer-insertion gradient-chain bridge:
    baseline native timing -> candidate timing coordinate -> buffer surrogate
    tensors -> trial native timing -> objective delta. It is finite-difference
    scoring for now, not relaxed-b_u autograd/backward.
    """
    if dtype is None:
        dtype = torch.float32
    candidates = list(candidates or [])
    baseline_inputs = build_net_subgraph_timing_inputs(
        nets,
        candidates=[],
        driver_arrival_by_net=driver_arrival_by_net,
        driver_slew_by_net=driver_slew_by_net,
        dtype=dtype,
        device=device,
    )
    baseline = net_subgraph_forward_native(**_native_inputs_only(baseline_inputs))
    original_to_compact = baseline_inputs["metadata"]["original_node_to_compact"]
    original_to_compact_by_net = baseline_inputs["metadata"].get(
        "original_node_to_compact_by_net", {}
    )
    net_ids = [int(net_id) for net_id in baseline_inputs["metadata"].get("net_ids", [])]
    net_by_id = {
        int(net_id): net
        for net_id, net in zip(net_ids, nets)
    }
    bsu_values = _bsu_candidate_values(bsu, bsu_candidates)
    bsu_selection_mode = "fixed_bsu" if len(bsu_values) == 1 else "best_native_delta"
    scored = []

    for candidate in candidates:
        item = dict(candidate)
        segment_edge = _candidate_segment_edge(item)
        scoring_node_id, scoring_node_source = _candidate_scoring_node(item)
        if scoring_node_id is None and segment_edge is None:
            item.update(
                {
                    "score_status": "unsupported",
                    "unsupported_reasons": ["missing_scoring_node_id"],
                    "predicted_delta_obj": 0.0,
                    "predicted_improvement": 0.0,
                    "estimator_source": estimator_source,
                }
            )
            scored.append(item)
            continue
        net_id, net_id_error = _candidate_net_id(item, net_ids)
        if net_id_error is not None:
            item.update(
                {
                    "score_status": "unsupported",
                    "unsupported_reasons": [net_id_error],
                    "predicted_delta_obj": 0.0,
                    "predicted_improvement": 0.0,
                    "estimator_source": estimator_source,
                }
            )
            scored.append(item)
            continue
        if _can_exact_segment_split(net_by_id.get(net_id), item):
            coordinate_candidate = dict(item)
            coordinate_candidate.update(
                {
                    "bu": 0.0,
                    "buffer_input_cap": 0.0,
                    "buffer_delay": 0.0,
                    "buffer_output_slew": 0.0,
                }
            )
            coordinate_inputs = build_net_subgraph_timing_inputs(
                nets,
                candidates=[coordinate_candidate],
                driver_arrival_by_net=driver_arrival_by_net,
                driver_slew_by_net=driver_slew_by_net,
                dtype=dtype,
                device=device,
            )
            coordinate_baseline = net_subgraph_forward_native(
                **_native_inputs_only(coordinate_inputs)
            )
            compact_node = int(coordinate_inputs["candidate_node_id"][0].item())
            split_specs = coordinate_inputs["metadata"].get("synthetic_segment_splits", [])
            split_spec = dict(split_specs[0]) if split_specs else {}
            scoring_node_id = int(split_spec.get("synthetic_node_id", compact_node))
            scoring_node_source = "segment_edge_split"
            scoring_location_model, scoring_location_is_exact = _scoring_location_model(
                scoring_node_source
            )
            candidate_input_slew = float(coordinate_baseline["slew_in"][compact_node].item())
            candidate_output_cap = float(coordinate_baseline["lout"][compact_node].item())
            unsplit_baseline_obj = _native_sink_objective(baseline, net_id)
            baseline_obj = _native_sink_objective(coordinate_baseline, net_id)
            selected_trial = _score_bsu_trials(
                nets=nets,
                candidate=item,
                net_id=net_id,
                bsu_values=bsu_values,
                buffer_surrogate=buffer_surrogate,
                candidate_input_slew=candidate_input_slew,
                candidate_output_cap=candidate_output_cap,
                baseline_result=coordinate_baseline,
                baseline_obj=baseline_obj,
                driver_arrival_by_net=driver_arrival_by_net,
                driver_slew_by_net=driver_slew_by_net,
                dtype=dtype,
                device=device,
            )
            item.update(
                {
                    "score_status": "ok",
                    "bsu": int(selected_trial["bsu"]),
                    "bsu_selection_mode": bsu_selection_mode,
                    "bsu_candidate_count": len(bsu_values),
                    "bsu_candidate_evaluations": [
                        dict(evaluation)
                        for evaluation in sorted(
                            selected_trial.get("_all_evaluations", []),
                            key=lambda evaluation: int(evaluation["bsu"]),
                        )
                    ],
                    "scoring_node_id": int(scoring_node_id),
                    "scoring_node_source": scoring_node_source,
                    "scoring_location_model": scoring_location_model,
                    "scoring_location_is_exact": bool(scoring_location_is_exact),
                    "candidate_input_slew": candidate_input_slew,
                    "candidate_output_cap": candidate_output_cap,
                    "unsplit_baseline_obj": unsplit_baseline_obj,
                    "baseline_obj": float(baseline_obj),
                    "estimator_source": estimator_source,
                    "grad_bu": float(selected_trial["physical_delta_obj"]),
                    "grad_bu_source": "native_finite_difference",
                    "grad_bu_is_true_objective_gradient": False,
                    "selection_signal": "native_finite_difference_delta_obj",
                    "native_finite_difference_delta_obj": float(
                        selected_trial["physical_delta_obj"]
                    ),
                    "autograd_grad_bu": None,
                    "gradient_chain": {
                        "baseline_provider": "net_subgraph_timing_cpp",
                        "coordinate_source": "split_baseline_native_slew_in_and_lout",
                        "surrogate_provider": "buffer_surrogate",
                        "trial_provider": "net_subgraph_timing_cpp",
                        "objective": "max_sink_arrival_delta",
                        "slew_objective": "max_sink_slew_delta",
                        "load_objective": "max_sink_load_delta",
                        "baseline_objective_source": "split_baseline_same_topology",
                        "location_model": scoring_location_model,
                        "location_is_exact": bool(scoring_location_is_exact),
                        "split_parent_node_id": int(split_spec.get("parent_node_id", segment_edge[0])),
                        "split_child_node_id": int(split_spec.get("child_node_id", segment_edge[1])),
                        "split_ratio": float(split_spec.get("split_ratio", 0.0)),
                    },
                }
            )
            item.update(
                {
                    key: value
                    for key, value in selected_trial.items()
                    if key != "_all_evaluations"
                }
            )
            item["bsu_candidate_evaluations"] = [
                dict(evaluation)
                for evaluation in sorted(
                    selected_trial.get("_all_evaluations", [selected_trial]),
                    key=lambda evaluation: int(evaluation["bsu"]),
                )
            ]
            scored.append(item)
            continue
        node_map = original_to_compact_by_net.get(net_id, original_to_compact)
        if int(scoring_node_id) not in node_map:
            item.update(
                {
                    "score_status": "unsupported",
                    "unsupported_reasons": ["scoring_node_not_in_net_subgraph"],
                    "predicted_delta_obj": 0.0,
                    "predicted_improvement": 0.0,
                    "estimator_source": estimator_source,
                }
            )
            scored.append(item)
            continue

        compact_node = int(node_map[int(scoring_node_id)])
        scoring_location_model, scoring_location_is_exact = _scoring_location_model(
            scoring_node_source
        )
        candidate_input_slew = float(baseline["slew_in"][compact_node].item())
        candidate_output_cap = float(baseline["lout"][compact_node].item())
        baseline_obj = _native_sink_objective(baseline, net_id)
        selected_trial = _score_bsu_trials(
            nets=nets,
            candidate=item,
            net_id=net_id,
            bsu_values=bsu_values,
            buffer_surrogate=buffer_surrogate,
            candidate_input_slew=candidate_input_slew,
            candidate_output_cap=candidate_output_cap,
            baseline_result=baseline,
            baseline_obj=baseline_obj,
            node_id=scoring_node_id,
            driver_arrival_by_net=driver_arrival_by_net,
            driver_slew_by_net=driver_slew_by_net,
            dtype=dtype,
            device=device,
        )
        item.update(
            {
                "score_status": "ok",
                "bsu": int(selected_trial["bsu"]),
                "bsu_selection_mode": bsu_selection_mode,
                "bsu_candidate_count": len(bsu_values),
                "scoring_node_id": int(scoring_node_id),
                "scoring_node_source": scoring_node_source,
                "scoring_location_model": scoring_location_model,
                "scoring_location_is_exact": bool(scoring_location_is_exact),
                "candidate_input_slew": candidate_input_slew,
                "candidate_output_cap": candidate_output_cap,
                "baseline_obj": float(baseline_obj),
                "estimator_source": estimator_source,
                "grad_bu": float(selected_trial["physical_delta_obj"]),
                "grad_bu_source": "native_finite_difference",
                "grad_bu_is_true_objective_gradient": False,
                "selection_signal": "native_finite_difference_delta_obj",
                "native_finite_difference_delta_obj": float(
                    selected_trial["physical_delta_obj"]
                ),
                "autograd_grad_bu": None,
                "gradient_chain": {
                    "baseline_provider": "net_subgraph_timing_cpp",
                    "coordinate_source": "baseline_native_slew_in_and_lout",
                    "surrogate_provider": "buffer_surrogate",
                    "trial_provider": "net_subgraph_timing_cpp",
                    "objective": "max_sink_arrival_delta",
                    "slew_objective": "max_sink_slew_delta",
                    "load_objective": "max_sink_load_delta",
                    "location_model": scoring_location_model,
                    "location_is_exact": bool(scoring_location_is_exact),
                },
            }
        )
        item.update(
            {
                key: value
                for key, value in selected_trial.items()
                if key != "_all_evaluations"
            }
        )
        item["bsu_candidate_evaluations"] = [
            dict(evaluation)
            for evaluation in sorted(
                selected_trial.get("_all_evaluations", [selected_trial]),
                key=lambda evaluation: int(evaluation["bsu"]),
            )
        ]
        scored.append(item)
    return scored
