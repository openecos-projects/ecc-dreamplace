import math
from collections import defaultdict, deque

from .timing_rooted_tree_view import build_timing_rooted_tree_view


def _path_to_root(parent_by_node, sink_pin_id):
    path = []
    cur = int(sink_pin_id)
    while cur is not None:
        path.append(cur)
        cur = parent_by_node.get(cur)
    path.reverse()
    return path


def _tree_status(node_ids, edges, root):
    adjacency = defaultdict(list)
    for src, dst in edges:
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))
    visited = set()
    parent = {int(root): None}
    queue = deque([int(root)])
    has_cycle = False
    while queue:
        node = queue.popleft()
        visited.add(node)
        for nxt in adjacency.get(node, []):
            if parent.get(node) == nxt:
                continue
            if nxt in visited:
                has_cycle = True
                continue
            if nxt in parent:
                has_cycle = True
                continue
            parent[nxt] = node
            queue.append(nxt)
    return set(node_ids).issubset(visited), has_cycle


def _wirelength(edges, coordinates):
    total = 0.0
    for src, dst in edges or []:
        src_coord = coordinates.get(int(src))
        dst_coord = coordinates.get(int(dst))
        if src_coord is None or dst_coord is None:
            continue
        total += abs(float(src_coord[0]) - float(dst_coord[0]))
        total += abs(float(src_coord[1]) - float(dst_coord[1]))
    return total


def _fallback_artifact(tree_view, *, status, skip_reason, topology_source):
    return {
        "artifact": "buffering_rebranching_summary",
        "artifact_version": 1,
        "status": status,
        "rebranching_status": "skipped_with_reason",
        "topology_source": topology_source,
        "rebranching_enabled": True,
        "rebranching_sink_criticality_source": "slack_times_npath",
        "rebranching_candidate_sink_count": len(tree_view.get("sink_pin_ids", []) or []),
        "rebranched_sink_count": 0,
        "rebranching_ratio": 0.0,
        "rebranching_max_sink_fraction": 0.2,
        "rebranching_changed_edge_count": 0,
        "rebranching_wirelength_delta": 0.0,
        "rebranching_skip_reason": skip_reason,
        "candidate_set_skeleton_source": f"fallback_{topology_source}",
        "unsupported_reasons": list(tree_view.get("unsupported_reasons", [])),
        "net_summaries": [],
    }


def build_rebranching_dry_run(
    tree_view,
    *,
    sink_slack_by_pin,
    npath_by_pin,
    max_rebranched_sink_ratio=0.2,
    max_wirelength_delta_ratio=None,
):
    topology_source = str(tree_view.get("topology_source", "timing_rooted_steiner"))
    if tree_view.get("status") != "ok":
        return _fallback_artifact(
            tree_view,
            status="unsupported",
            skip_reason=";".join(tree_view.get("unsupported_reasons", []))
            or "unsupported_tree",
            topology_source="unsupported",
        )

    parent_by_node = dict(tree_view.get("parent_by_node", {}))
    root = int(tree_view["root_node_id"])
    sink_slack_by_pin = {int(pin): float(value) for pin, value in (sink_slack_by_pin or {}).items()}
    npath_by_pin = {int(pin): int(value) for pin, value in (npath_by_pin or {}).items()}
    missing_criticality_pins = [
        int(sink_pin_id)
        for sink_pin_id in tree_view.get("sink_pin_ids", [])
        if int(sink_pin_id) not in sink_slack_by_pin
        or int(sink_pin_id) not in npath_by_pin
    ]
    if missing_criticality_pins:
        artifact = _fallback_artifact(
            tree_view,
            status="skipped_with_reason",
            skip_reason=(
                "missing_sink_slack_or_npath:"
                + ",".join(str(pin_id) for pin_id in missing_criticality_pins)
            ),
            topology_source=topology_source,
        )
        artifact["net_id"] = int(tree_view["net_id"])
        artifact["root_node_id"] = root
        artifact["sink_count"] = len(tree_view.get("sink_pin_ids", []) or [])
        artifact["max_rebranched_sink_ratio"] = float(max_rebranched_sink_ratio)
        artifact["rebranching_max_sink_fraction"] = float(max_rebranched_sink_ratio)
        artifact["rebranch_limit"] = 0
        artifact["sink_criticality"] = []
        artifact["rebranched_sinks"] = []
        artifact["original_edges"] = list(tree_view.get("edge_pairs", []))
        artifact["rebranched_edges"] = list(tree_view.get("edge_pairs", []))
        artifact["is_connected"] = bool(tree_view.get("is_connected", False))
        artifact["has_cycle"] = bool(tree_view.get("has_cycle", False))
        return artifact

    sink_records = []
    max_criticality = 0.0
    for sink_pin_id in tree_view.get("sink_pin_ids", []):
        sink_pin_id = int(sink_pin_id)
        slack = float(sink_slack_by_pin[sink_pin_id])
        npath = max(1, int(npath_by_pin[sink_pin_id]))
        criticality = max(0.0, -slack) * npath
        max_criticality = max(max_criticality, criticality)
        path = _path_to_root(parent_by_node, sink_pin_id)
        depth = max(0, len(path) - 1)
        sink_records.append(
            {
                "pin_id": sink_pin_id,
                "sink_pin_id": sink_pin_id,
                "slack": slack,
                "npath": npath,
                "criticality": criticality,
                "depth": depth,
                "path_from_root": path,
            }
        )

    for record in sink_records:
        c_hat = 0.0 if max_criticality <= 0.0 else record["criticality"] / max_criticality
        depth = int(record["depth"])
        record["C_hat"] = c_hat
        record["ki"] = min(depth, int(math.ceil(c_hat * depth)))
        record["normalized_criticality"] = c_hat
        record["jump_level"] = record["ki"]

    sorted_records = sorted(
        sink_records,
        key=lambda item: (-float(item["criticality"]), int(item["sink_pin_id"])),
    )
    rebranch_limit = int(math.floor(float(max_rebranched_sink_ratio) * len(sink_records)))
    rebranch_limit = max(0, min(len(sink_records), rebranch_limit))

    new_parent = dict(parent_by_node)
    rebranched = []
    for record in sorted_records[:rebranch_limit]:
        if record["criticality"] <= 0.0 or record["ki"] <= 0:
            continue
        path = record["path_from_root"]
        sink_pin_id = int(record["sink_pin_id"])
        old_parent = new_parent.get(sink_pin_id)
        new_parent_index = max(0, len(path) - 1 - int(record["ki"]))
        proposed_parent = int(path[new_parent_index])
        if proposed_parent == sink_pin_id or proposed_parent == old_parent:
            continue
        new_parent[sink_pin_id] = proposed_parent
        rebranched.append(
            {
                "pin_id": sink_pin_id,
                "sink_pin_id": sink_pin_id,
                "old_parent_node_id": old_parent,
                "new_parent_node_id": proposed_parent,
                "slack": record["slack"],
                "npath": record["npath"],
                "criticality": record["criticality"],
                "C_hat": record["C_hat"],
                "ki": record["ki"],
                "normalized_criticality": record["normalized_criticality"],
                "jump_level": record["jump_level"],
            }
        )

    rebranched_edges = [
        (int(parent), int(node))
        for node, parent in sorted(new_parent.items())
        if parent is not None
    ]
    is_connected, has_cycle = _tree_status(tree_view.get("node_ids", []), rebranched_edges, root)
    status = "ok" if is_connected and not has_cycle else "failed"
    original_wirelength = _wirelength(
        tree_view.get("edge_pairs", []),
        tree_view.get("coordinates", {}) or {},
    )
    rebranched_wirelength = _wirelength(
        rebranched_edges,
        tree_view.get("coordinates", {}) or {},
    )
    wirelength_delta = rebranched_wirelength - original_wirelength
    wirelength_guard_triggered = False
    if rebranched and max_wirelength_delta_ratio is not None:
        max_delta = max(0.0, float(max_wirelength_delta_ratio)) * max(
            original_wirelength,
            1.0,
        )
        if wirelength_delta > max_delta:
            wirelength_guard_triggered = True
            rebranched = []
            rebranched_edges = list(tree_view.get("edge_pairs", []))

    candidate_set_skeleton_source = (
        "timing_aware_rebranched_skeleton"
        if status == "ok" and rebranched and not wirelength_guard_triggered
        else "fallback_timing_rooted_steiner"
    )
    rebranching_status = (
        "ok"
        if status == "ok" and rebranched and not wirelength_guard_triggered
        else "skipped_with_reason"
    )
    skip_reason = ""
    if wirelength_guard_triggered:
        skip_reason = "wirelength_delta_exceeds_limit"
    elif not rebranched:
        skip_reason = "no_positive_rebranching_move"
    return {
        "artifact": "buffering_rebranching_summary",
        "artifact_version": 1,
        "status": status,
        "rebranching_status": rebranching_status,
        "topology_source": topology_source,
        "rebranching_enabled": True,
        "rebranching_sink_criticality_source": "slack_times_npath",
        "net_id": int(tree_view["net_id"]),
        "root_node_id": root,
        "sink_count": len(sink_records),
        "max_rebranched_sink_ratio": float(max_rebranched_sink_ratio),
        "rebranching_max_sink_fraction": float(max_rebranched_sink_ratio),
        "rebranch_limit": rebranch_limit,
        "rebranching_candidate_sink_count": len(sink_records),
        "rebranched_sink_count": len(rebranched),
        "rebranching_ratio": (
            float(len(rebranched)) / float(len(sink_records))
            if sink_records
            else 0.0
        ),
        "rebranching_changed_edge_count": len(rebranched),
        "rebranching_wirelength_delta": wirelength_delta,
        "rebranching_max_wirelength_delta_ratio": (
            None
            if max_wirelength_delta_ratio is None
            else float(max_wirelength_delta_ratio)
        ),
        "wirelength_guard_triggered": wirelength_guard_triggered,
        "rebranching_skip_reason": skip_reason,
        "candidate_set_skeleton_source": candidate_set_skeleton_source,
        "sink_criticality": sink_records,
        "rebranched_sinks": rebranched,
        "original_edges": list(tree_view.get("edge_pairs", [])),
        "rebranched_edges": rebranched_edges,
        "is_connected": is_connected,
        "has_cycle": has_cycle,
    }


def build_timing_aware_skeleton(
    tree_view,
    *,
    sink_slack_by_pin,
    npath_by_pin,
    max_sink_fraction=0.2,
    max_wirelength_delta_ratio=None,
):
    summary = build_rebranching_dry_run(
        tree_view,
        sink_slack_by_pin=sink_slack_by_pin,
        npath_by_pin=npath_by_pin,
        max_rebranched_sink_ratio=max_sink_fraction,
        max_wirelength_delta_ratio=max_wirelength_delta_ratio,
    )
    if summary.get("candidate_set_skeleton_source") != "timing_aware_rebranched_skeleton":
        return dict(tree_view), summary

    skeleton = build_timing_rooted_tree_view(
        net_id=tree_view["net_id"],
        net_pin_ids=[
            int(tree_view["root_node_id"]),
            *[int(pin_id) for pin_id in tree_view.get("sink_pin_ids", [])],
        ],
        driver_pin_id=tree_view["root_node_id"],
        undirected_edges=summary.get("rebranched_edges", []),
        coordinates=tree_view.get("coordinates", {}),
        flat_first_pin_id=tree_view.get("flat_first_pin_id"),
    )
    skeleton["topology_source"] = "timing_aware_rebranched_skeleton"
    summary["rebranched_skeleton_status"] = skeleton.get("status")
    return skeleton, summary
