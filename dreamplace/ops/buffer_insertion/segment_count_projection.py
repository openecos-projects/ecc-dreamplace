import torch

from dreamplace.ops.net_subgraph_timing.segment_repeater_transfer import (
    build_equal_spaced_segment_candidates,
)
from dreamplace.ops.physical_action import VirtualBufferState


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


def _net_by_id(nets):
    return {_net_id(net, index): net for index, net in enumerate(list(nets or []))}


def _children_by_node(net):
    rc_tree = dict(net.get("rc_tree", {}) or {})
    return {
        int(node): [int(child) for child in list(children or [])]
        for node, children in (rc_tree.get("children_by_node", {}) or {}).items()
    }


def _tree_node_depths(net):
    rc_tree = dict(net.get("rc_tree", {}) or {})
    root = rc_tree.get("root_node_id", net.get("driver_pin_id"))
    if root is None:
        return {}
    children = _children_by_node(net)
    depths = {int(root): 0}
    stack = [int(root)]
    while stack:
        parent = stack.pop()
        child_depth = int(depths[parent]) + 1
        for child in children.get(parent, []):
            child = int(child)
            if child in depths:
                continue
            depths[child] = child_depth
            stack.append(child)
    return depths


def _sink_node_ids(net):
    rc_tree = dict(net.get("rc_tree", {}) or {})
    for key in ("sink_pin_ids", "sink_nodes"):
        values = rc_tree.get(key)
        if values:
            return sorted({int(value) for value in list(values or [])})
    values = net.get("sink_pin_ids", net.get("sink_nodes", []))
    return sorted({int(value) for value in list(values or [])})


def _pin_name(net, pin_id):
    pin_id = int(pin_id)
    pin_names = dict(net.get("pin_name_by_id", {}) or {})
    value = pin_names.get(pin_id)
    if value is None:
        value = pin_names.get(str(pin_id))
    if value is not None:
        return str(value)
    if pin_id == int(net.get("driver_pin_id", -1)):
        return str(net.get("driver_pin_name", ""))
    if pin_id == int(net.get("load_pin_id", -1)):
        return str(net.get("load_pin_name", ""))
    return ""


def _downstream_sink_ids(net, child_node_ids):
    children = _children_by_node(net)
    sinks = set(_sink_node_ids(net))
    if not sinks:
        return []
    stack = [int(value) for value in list(child_node_ids or [])]
    visited = set()
    downstream = []
    while stack:
        node_id = int(stack.pop())
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in sinks:
            downstream.append(node_id)
        stack.extend(children.get(node_id, []))
    return sorted(set(downstream))


def _sink_timing_values(net, pin_ids):
    slack_by_pin = {
        int(pin_id): float(slack)
        for pin_id, slack in dict(net.get("sink_slack_by_pin") or {}).items()
    }
    npath_by_pin = {
        int(pin_id): int(npath)
        for pin_id, npath in dict(net.get("npath_by_pin") or {}).items()
    }
    values = []
    for pin_id in list(pin_ids or []):
        pin_id = int(pin_id)
        slack = float(slack_by_pin.get(pin_id, 0.0))
        npath = max(1, int(npath_by_pin.get(pin_id, 1)))
        criticality = max(0.0, -slack) * float(npath)
        values.append((slack, npath, criticality))
    return values


def _timing_context(values):
    values = list(values or [])
    if not values:
        return {
            "negative_slack_sink_count": 0,
            "worst_sink_slack": 0.0,
            "sink_criticality_sum": 0.0,
            "sink_criticality_mean": 0.0,
            "sink_criticality_max": 0.0,
        }
    criticalities = [float(item[2]) for item in values]
    return {
        "negative_slack_sink_count": sum(
            1 for slack, _, _ in values if float(slack) < 0.0
        ),
        "worst_sink_slack": min(float(slack) for slack, _, _ in values),
        "sink_criticality_sum": sum(criticalities),
        "sink_criticality_mean": sum(criticalities) / float(len(criticalities)),
        "sink_criticality_max": max(criticalities),
    }


def _attach_timing_context(item, net, downstream_pin_ids):
    all_sink_ids = _sink_node_ids(net)
    downstream_ids = [int(pin_id) for pin_id in list(downstream_pin_ids or [])]
    downstream = _timing_context(_sink_timing_values(net, downstream_ids))
    total = _timing_context(_sink_timing_values(net, all_sink_ids))
    downstream_criticality = float(downstream["sink_criticality_sum"])
    total_criticality = float(total["sink_criticality_sum"])
    upstream_sibling_criticality = max(0.0, total_criticality - downstream_criticality)
    upstream_sibling_negative_slack_count = max(
        0,
        int(total["negative_slack_sink_count"])
        - int(downstream["negative_slack_sink_count"]),
    )

    source_values = _sink_timing_values(net, downstream_ids[:1])
    if source_values:
        source_slack, source_npath, source_criticality = source_values[0]
    else:
        source_slack, source_npath, source_criticality = 0.0, 1, 0.0

    item.update(
        {
            "source_sink_slack": float(source_slack),
            "source_sink_npath": int(source_npath),
            "source_sink_criticality": float(source_criticality),
            "downstream_negative_slack_sink_count": int(
                downstream["negative_slack_sink_count"]
            ),
            "total_negative_slack_sink_count": int(
                total["negative_slack_sink_count"]
            ),
            "upstream_sibling_negative_slack_sink_count": int(
                upstream_sibling_negative_slack_count
            ),
            "downstream_worst_sink_slack": float(downstream["worst_sink_slack"]),
            "total_worst_sink_slack": float(total["worst_sink_slack"]),
            "downstream_sink_criticality_sum": downstream_criticality,
            "total_sink_criticality_sum": total_criticality,
            "moved_branch_criticality_sum": downstream_criticality,
            "upstream_sibling_criticality_sum": upstream_sibling_criticality,
            "downstream_sink_criticality_ratio": (
                downstream_criticality / total_criticality
                if total_criticality > 0.0
                else 0.0
            ),
            "downstream_sink_criticality_mean": float(
                downstream["sink_criticality_mean"]
            ),
            "downstream_sink_criticality_max": float(
                downstream["sink_criticality_max"]
            ),
            "moved_branch_has_setup_criticality": downstream_criticality > 0.0,
            "sibling_has_setup_criticality": upstream_sibling_criticality > 0.0,
            "moved_noncritical_branch_with_critical_sibling": (
                downstream_criticality <= 0.0
                and upstream_sibling_criticality > 0.0
            ),
        }
    )
    return item


def _attach_downstream_context(candidate, net):
    item = dict(candidate)
    downstream_pin_ids = [
        int(value) for value in list(item.get("downstream_pin_ids", []) or [])
    ]
    if not downstream_pin_ids:
        downstream_pin_ids = _downstream_sink_ids(net, item.get("child_node_ids", []))

    if downstream_pin_ids and not item.get("load_pin_id"):
        item["load_pin_id"] = int(downstream_pin_ids[0])
    if downstream_pin_ids and not item.get("load_pin_name"):
        item["load_pin_name"] = _pin_name(net, downstream_pin_ids[0])

    downstream_pin_names = [
        str(value)
        for value in list(item.get("downstream_pin_names", []) or [])
        if str(value)
    ]
    if not downstream_pin_names:
        downstream_pin_names = [
            name
            for name in (_pin_name(net, pin_id) for pin_id in downstream_pin_ids)
            if name
        ]

    item["downstream_pin_ids"] = downstream_pin_ids
    item["downstream_pin_names"] = downstream_pin_names
    item["downstream_pin_count"] = len(downstream_pin_ids)
    item.setdefault("downstream_sink_count", len(downstream_pin_ids))
    item.setdefault("total_sink_count", len(_sink_node_ids(net)))
    item.setdefault(
        "load_partition_source",
        "segment_count_child_subtree" if downstream_pin_ids else "missing_downstream_subtree",
    )
    item = _attach_timing_context(item, net, downstream_pin_ids)
    return item


def _round_clip(value, low, high):
    if torch.is_tensor(value):
        value = float(value.detach().cpu().item())
    rounded = int(round(float(value)))
    return max(int(low), min(int(high), rounded))


def _project_repeater_count(value, low, high, min_z_to_insert):
    if torch.is_tensor(value):
        value = float(value.detach().cpu().item())
    value = float(value)
    min_z_to_insert = float(min_z_to_insert)
    if value <= 0.0 or value < min_z_to_insert:
        return 0
    return max(int(low), min(int(high), max(1, int(round(value)))))


def _value_distribution(values, min_z_to_insert):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    count = int(values.numel())
    if count == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "sum": 0.0,
            "ge_min_insert_count": 0,
            "ge_0p5_count": 0,
            "ge_1p0_count": 0,
        }
    return {
        "count": count,
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "sum": float(values.sum().item()),
        "ge_min_insert_count": int(
            (values >= float(min_z_to_insert)).sum().item()
        ),
        "ge_0p5_count": int((values >= 0.5).sum().item()),
        "ge_1p0_count": int((values >= 1.0).sum().item()),
    }


def _projection_selection_score(candidate):
    z_value = float(candidate.get("z_value", 0.0) or 0.0)
    criticality = float(candidate.get("downstream_sink_criticality_sum", 0.0) or 0.0)
    return z_value * criticality


def _sort_projected_candidates(candidates):
    enriched = []
    for insertion_order, candidate in enumerate(list(candidates or [])):
        item = dict(candidate)
        item["projection_insertion_order"] = int(insertion_order)
        item["projection_selection_score"] = float(_projection_selection_score(item))
        enriched.append(item)
    enriched.sort(
        key=lambda item: (
            float(item.get("projection_selection_score", 0.0) or 0.0),
            float(item.get("z_value", 0.0) or 0.0),
            float(item.get("downstream_sink_criticality_sum", 0.0) or 0.0),
            -int(item.get("projection_insertion_order", 0) or 0),
        ),
        reverse=True,
    )
    for order, item in enumerate(enriched):
        item["projection_order"] = int(order)
    return enriched


def _candidate_diagnostic_snapshot(candidate):
    item = {}
    for key in (
        "candidate_id",
        "action_id",
        "segment_id",
        "net_id",
        "net_name",
        "driver_pin_name",
        "load_pin_name",
        "candidate_location_x_dbu",
        "candidate_location_y_dbu",
        "x_dbu",
        "y_dbu",
        "z_value",
        "bsu_value",
        "projected_bsu_index",
        "projected_repeater_count",
        "projection_selection_score",
        "projection_order",
        "projection_insertion_order",
        "downstream_sink_criticality_sum",
        "downstream_sink_criticality_ratio",
        "downstream_sink_criticality_mean",
        "downstream_sink_criticality_max",
        "downstream_negative_slack_sink_count",
        "total_negative_slack_sink_count",
        "downstream_worst_sink_slack",
        "total_worst_sink_slack",
        "source_sink_slack",
        "source_sink_npath",
        "source_sink_criticality",
        "predicted_delta_tns",
        "predicted_delta_wns",
        "predicted_delta_obj",
        "predicted_improvement",
        "selection_score",
        "measured_label_probe_rank_source",
        "measured_label_probe_rank_value",
    ):
        if key in candidate:
            item[key] = candidate.get(key)
    if "downstream_pin_names" in candidate:
        item["downstream_pin_names"] = list(candidate.get("downstream_pin_names") or [])
    if "downstream_pin_ids" in candidate:
        item["downstream_pin_ids"] = [
            int(value) for value in list(candidate.get("downstream_pin_ids") or [])
        ]
    return item


def _candidate_diagnostic_snapshots(candidates, limit=16):
    limit = max(0, int(limit))
    return [
        _candidate_diagnostic_snapshot(candidate)
        for candidate in list(candidates or [])[:limit]
    ]


def _forced_z_prediction_for_segment(segment_state, segment_id):
    predictions = getattr(
        segment_state,
        "segment_forced_z_prediction_by_segment_id",
        None,
    )
    if not isinstance(predictions, dict):
        return None
    prediction = predictions.get(int(segment_id))
    if prediction is None:
        prediction = predictions.get(str(int(segment_id)))
    return dict(prediction) if isinstance(prediction, dict) else None


def _normalize_live_projection_context(context):
    if context is None:
        return None
    if not isinstance(context, dict):
        raise TypeError("live projection context must be a dictionary")
    coordinate_source = str(context.get("coordinate_source", "") or "")
    if coordinate_source != "final_live_geometry":
        raise ValueError(
            "live projection coordinate_source must be final_live_geometry"
        )
    snapshot_identity = str(
        context.get("placement_snapshot_identity", "") or ""
    )
    if not snapshot_identity:
        raise ValueError("live projection requires placement snapshot identity")

    node_x_dbu = context.get("node_x_dbu")
    node_y_dbu = context.get("node_y_dbu")
    if not torch.is_tensor(node_x_dbu) or not torch.is_tensor(node_y_dbu):
        raise TypeError("live projection node coordinates must be torch tensors")
    if (
        node_x_dbu.ndim != 1
        or node_y_dbu.ndim != 1
        or node_x_dbu.shape != node_y_dbu.shape
    ):
        raise ValueError("live projection node coordinates must be matching 1-D tensors")
    if node_x_dbu.device != node_y_dbu.device:
        raise ValueError("live projection node coordinates must share a device")
    if not node_x_dbu.is_floating_point() or not node_y_dbu.is_floating_point():
        raise TypeError("live projection node coordinates must be floating point")
    if not bool(torch.isfinite(node_x_dbu).all()) or not bool(
        torch.isfinite(node_y_dbu).all()
    ):
        raise ValueError("live projection node coordinates must be finite")

    topology_generation = int(context.get("topology_generation", 0) or 0)
    frozen_generation = int(
        context.get("frozen_topology_generation", 0) or 0
    )
    expected_generation = int(
        context.get("expected_frozen_topology_generation", 0) or 0
    )
    if topology_generation <= 0 or frozen_generation <= 0:
        raise ValueError("live projection requires a positive frozen topology generation")
    if topology_generation != frozen_generation:
        raise ValueError("live projection topology generation is not frozen generation")
    if expected_generation and frozen_generation != expected_generation:
        raise ValueError("live projection frozen topology generation mismatch")
    path_policy = str(context.get("rectilinear_path_policy", "x_then_y") or "")
    if path_policy != "x_then_y":
        raise ValueError("live projection requires x_then_y rectilinear path policy")

    return {
        "coordinate_source": coordinate_source,
        "placement_snapshot_identity": snapshot_identity,
        "placement_snapshot_iteration": context.get(
            "placement_snapshot_iteration"
        ),
        "topology_generation": topology_generation,
        "frozen_topology_generation": frozen_generation,
        "rectilinear_path_policy": path_policy,
        "node_x_dbu": node_x_dbu.detach(),
        "node_y_dbu": node_y_dbu.detach(),
    }


def _live_segment_projection_net(net, segment, context):
    parent_node_id = int(segment["parent_node_id"])
    child_node_id = int(segment["child_node_id"])
    node_count = int(context["node_x_dbu"].numel())
    for node_id in (parent_node_id, child_node_id):
        if node_id < 0 or node_id >= node_count:
            raise ValueError(
                "live projection segment node is outside final geometry domain"
            )
    node_x_dbu = context["node_x_dbu"]
    node_y_dbu = context["node_y_dbu"]
    parent_coordinate = (
        float(node_x_dbu[parent_node_id].cpu().item()),
        float(node_y_dbu[parent_node_id].cpu().item()),
    )
    child_coordinate = (
        float(node_x_dbu[child_node_id].cpu().item()),
        float(node_y_dbu[child_node_id].cpu().item()),
    )
    projection_net = dict(net)
    coordinates_dbu = dict(net.get("coordinates_dbu", {}) or {})
    coordinates_dbu[parent_node_id] = parent_coordinate
    coordinates_dbu[child_node_id] = child_coordinate
    projection_net["coordinates_dbu"] = coordinates_dbu
    return projection_net, parent_coordinate, child_coordinate


def _rectilinear_equal_spaced_coordinate(parent, child, ratio):
    px, py = float(parent[0]), float(parent[1])
    cx, cy = float(child[0]), float(child[1])
    dx = cx - px
    dy = cy - py
    x_length = abs(dx)
    y_length = abs(dy)
    total_length = x_length + y_length
    if total_length <= 0.0:
        return px, py
    distance = min(1.0, max(0.0, float(ratio))) * total_length
    if distance <= x_length or y_length <= 0.0:
        direction = 0.0 if x_length <= 0.0 else dx / x_length
        return px + direction * min(distance, x_length), py
    direction = dy / y_length
    return cx, py + direction * (distance - x_length)


def project_segment_count_state_to_candidates(
    nets,
    segment_state,
    *,
    candidate_id_start=0,
    projection_policy="round_clip_optimized_global_state",
    top_k=None,
    min_z_to_insert=0.5,
    require_setup_criticality=False,
    live_projection_context=None,
):
    live_context = _normalize_live_projection_context(live_projection_context)
    nets_by_id = _net_by_id(nets)
    z_value = segment_state.z_value().detach()
    bsu_value = segment_state.bsu_index().detach()
    max_repeater_count = int(segment_state.max_repeater_count)
    max_bsu_index = int(segment_state.legal_buffer_count) - 1
    candidates = []
    skipped_missing_net = 0
    next_candidate_id = int(candidate_id_start)
    tree_depths_by_net_id = {}

    if getattr(segment_state, "is_tensor_backed", False):
        active_index = torch.nonzero(
            z_value >= float(min_z_to_insert),
            as_tuple=False,
        ).flatten()
        active_indices = [int(value) for value in active_index.detach().cpu().tolist()]
        active_segments = segment_state.segment_records(
            active_indices,
            materialization_kind="terminal",
        )
        active_z = z_value.index_select(0, active_index).detach().cpu()
        active_bsu = bsu_value.index_select(0, active_index).detach().cpu()
        segment_items = zip(active_indices, active_segments, active_z, active_bsu)
    else:
        segment_items = (
            (row_index, segment, z_value[row_index], bsu_value[row_index])
            for row_index, segment in enumerate(segment_state.segment_rows)
        )

    for row_index, segment, row_z_value, row_bsu_value in segment_items:
        net_id = int(segment["net_id"])
        net = nets_by_id.get(net_id)
        if net is None:
            skipped_missing_net += 1
            continue
        repeater_count = _project_repeater_count(
            row_z_value,
            0,
            max_repeater_count,
            min_z_to_insert,
        )
        if repeater_count <= 0:
            continue
        bsu_idx = _round_clip(row_bsu_value, 0, max_bsu_index)
        parent_node_id = int(segment["parent_node_id"])
        child_node_id = int(segment["child_node_id"])
        if net_id not in tree_depths_by_net_id:
            tree_depths_by_net_id[net_id] = _tree_node_depths(net)
        tree_depth = int(tree_depths_by_net_id[net_id].get(parent_node_id, -1))
        projection_net = net
        parent_coordinate = None
        child_coordinate = None
        if live_context is not None:
            projection_net, parent_coordinate, child_coordinate = (
                _live_segment_projection_net(net, segment, live_context)
            )
        segment_candidates = build_equal_spaced_segment_candidates(
            projection_net,
            parent_node_id=parent_node_id,
            child_node_id=child_node_id,
            repeater_count=repeater_count,
            bsu=bsu_idx,
            buffer_main_type_index=int(segment_state.buffer_main_type_index),
            candidate_id_start=next_candidate_id,
            source_flags=(
                "segment_count_projection",
                "equal_spaced_segment_count",
            ),
            bu=1.0,
            coordinate_key=(
                "coordinates_dbu"
                if projection_net.get("coordinates_dbu")
                else "coordinates"
            ),
        )
        for candidate in segment_candidates:
            for key in (
                "net_name",
                "driver_pin_id",
                "driver_pin_name",
                "load_pin_id",
                "load_pin_name",
                "downstream_pin_ids",
                "downstream_pin_names",
                "downstream_pin_count",
                "downstream_sink_count",
                "total_sink_count",
            ):
                if key in net:
                    candidate[key] = net[key]
            candidate = _attach_downstream_context(candidate, net)
            if live_context is not None:
                ratio = float(candidate["segment_split_ratio"])
                analytical_x_dbu, analytical_y_dbu = (
                    _rectilinear_equal_spaced_coordinate(
                        parent_coordinate,
                        child_coordinate,
                        ratio,
                    )
                )
                candidate.update(
                    {
                        "x_dbu": int(round(analytical_x_dbu)),
                        "y_dbu": int(round(analytical_y_dbu)),
                        "candidate_location_x_dbu_analytical": analytical_x_dbu,
                        "candidate_location_y_dbu_analytical": analytical_y_dbu,
                        "segment_parent_x_dbu": float(parent_coordinate[0]),
                        "segment_parent_y_dbu": float(parent_coordinate[1]),
                        "segment_child_x_dbu": float(child_coordinate[0]),
                        "segment_child_y_dbu": float(child_coordinate[1]),
                        "coordinate_source": live_context["coordinate_source"],
                        "placement_snapshot_identity": live_context[
                            "placement_snapshot_identity"
                        ],
                        "placement_snapshot_iteration": live_context[
                            "placement_snapshot_iteration"
                        ],
                        "topology_generation": live_context[
                            "topology_generation"
                        ],
                        "frozen_topology_generation": live_context[
                            "frozen_topology_generation"
                        ],
                        "rectilinear_path_policy": live_context[
                            "rectilinear_path_policy"
                        ],
                    }
                )
            forced_prediction = _forced_z_prediction_for_segment(
                segment_state,
                int(segment["segment_id"]),
            )
            candidate.update(
                {
                    "segment_id": int(segment["segment_id"]),
                    "segment_parent_node_id": parent_node_id,
                    "segment_child_node_id": child_node_id,
                    "segment_tree_depth": tree_depth,
                    "z_value": float(row_z_value.item()),
                    "bsu_value": float(row_bsu_value.item()),
                    "projected_bsu_index": int(bsu_idx),
                    "projected_repeater_count": int(repeater_count),
                    "projection_policy": str(projection_policy),
                    "bsu_sharing": "per_segment",
                    "max_repeater_count": int(max_repeater_count),
                    "coordinate_action_source": "segment_count_projection",
                    "selected_action_source": "segment_count_projection",
                    "selection_result_class": "segment_count_projection",
                    "result_class": "segment_count_projection",
                }
            )
            if forced_prediction is not None:
                predicted_delta_tns = float(
                    forced_prediction.get("predicted_delta_tns_vs_z0", 0.0) or 0.0
                )
                predicted_delta_loss = float(
                    forced_prediction.get("predicted_delta_loss_vs_z0", 0.0) or 0.0
                )
                candidate.update(
                    {
                        "predicted_delta_tns": predicted_delta_tns,
                        "predicted_delta_wns": float(
                            forced_prediction.get("predicted_delta_wns_vs_z0", 0.0)
                            or 0.0
                        ),
                        "predicted_delta_obj": predicted_delta_loss,
                        "predicted_improvement": predicted_delta_tns,
                        "selection_score": predicted_delta_tns,
                        "measured_label_probe_rank_source": (
                            "forced_z_single_segment_pysta_delta_tns"
                        ),
                        "measured_label_probe_rank_value": predicted_delta_tns,
                        "measured_label_probe_rank_direction": "larger_is_better",
                    }
                )
            candidates.append(candidate)
        next_candidate_id += len(segment_candidates)

    all_projected_candidate_count = len(candidates)
    if bool(require_setup_criticality):
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("downstream_sink_criticality_sum", 0.0) or 0.0)
            > 0.0
        ]
    criticality_filtered_candidate_count = len(candidates)
    candidates = _sort_projected_candidates(candidates)
    top_candidate_diagnostics = _candidate_diagnostic_snapshots(candidates)
    if top_k is not None:
        candidates = candidates[: max(0, int(top_k))]
    selected_candidate_diagnostics = _candidate_diagnostic_snapshots(candidates)

    projection_provenance = None
    if live_context is not None:
        projection_provenance = {
            key: live_context[key]
            for key in (
                "coordinate_source",
                "placement_snapshot_identity",
                "placement_snapshot_iteration",
                "topology_generation",
                "frozen_topology_generation",
                "rectilinear_path_policy",
            )
        }
        projection_provenance["coordinate_rounding"] = "round_to_nearest_dbu"
        projection_provenance[
            "analytical_coordinate_fields"
        ] = [
            "candidate_location_x_dbu_analytical",
            "candidate_location_y_dbu_analytical",
        ]

    return {
        "artifact": "segment_count_projection_candidates",
        "artifact_version": 1,
        "status": "ok",
        "projection_policy": str(projection_policy),
        "min_z_to_insert": float(min_z_to_insert),
        "projection_order_policy": "z_times_downstream_sink_criticality_desc",
        "require_setup_criticality": bool(require_setup_criticality),
        "local_ranking_used": False,
        "segment_shared_bsu": True,
        "top_k": None if top_k is None else int(top_k),
        "max_repeater_count": int(max_repeater_count),
        "segment_count": int(
            getattr(segment_state, "num_segments", len(segment_state.segment_rows or ()))
        ),
        "z_value_distribution": _value_distribution(z_value, min_z_to_insert),
        "projected_candidate_count": int(len(candidates)),
        "projected_candidate_count_before_topk": int(all_projected_candidate_count),
        "projected_candidate_count_after_criticality_filter": int(
            criticality_filtered_candidate_count
        ),
        "criticality_filtered_candidate_count": int(
            all_projected_candidate_count - criticality_filtered_candidate_count
        ),
        "top_candidate_diagnostics": top_candidate_diagnostics,
        "selected_candidate_diagnostics": selected_candidate_diagnostics,
        "skipped_missing_net_count": int(skipped_missing_net),
        "projection_provenance": projection_provenance,
        "candidates": candidates,
    }


def segment_count_projection_to_virtual_state(projection_artifact):
    virtual_state = VirtualBufferState()
    for candidate in list((projection_artifact or {}).get("candidates", []) or []):
        virtual_state.insert(candidate, bsu=int(candidate["projected_bsu_index"]))
    return virtual_state
