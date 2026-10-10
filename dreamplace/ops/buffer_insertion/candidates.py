import math


def _distance_dbu(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _interpolate(a, b, ratio):
    return (
        int(round(float(a[0]) + (float(b[0]) - float(a[0])) * ratio)),
        int(round(float(a[1]) + (float(b[1]) - float(a[1])) * ratio)),
    )


def _is_endpoint_coordinate(point, start, end):
    return tuple(point) == tuple(start) or tuple(point) == tuple(end)


def _merge_candidate(candidate_by_xy, candidate):
    key = (candidate["x_dbu"], candidate["y_dbu"])
    existing = candidate_by_xy.get(key)
    if existing is None:
        candidate["source_flags"] = sorted(set(candidate["source_flags"]))
        candidate_by_xy[key] = candidate
        return True
    existing["source_flags"] = sorted(set(existing["source_flags"]) | set(candidate["source_flags"]))
    if existing.get("tree_node_id") is None and candidate.get("tree_node_id") is not None:
        existing["tree_node_id"] = candidate["tree_node_id"]
    return False


def _rebranch_related_nodes(rebranching_summary):
    related = {}
    for record in (rebranching_summary or {}).get("rebranched_sinks", []) or []:
        sink_pin_id = int(record.get("sink_pin_id", record.get("pin_id", -1)))
        if sink_pin_id < 0:
            continue
        for field in ("sink_pin_id", "old_parent_node_id", "new_parent_node_id"):
            node_id = record.get(field)
            if node_id is None:
                continue
            related.setdefault(int(node_id), set()).add(sink_pin_id)
    return related


def _candidate_rebranch_sink_pin_ids(candidate, related_nodes):
    sink_pin_ids = set()
    node_ids = []
    if candidate.get("tree_node_id") is not None:
        node_ids.append(candidate.get("tree_node_id"))
    if candidate.get("parent_node_id") is not None:
        node_ids.append(candidate.get("parent_node_id"))
    node_ids.extend(candidate.get("child_node_ids", []) or [])
    for node_id in node_ids:
        sink_pin_ids.update(related_nodes.get(int(node_id), set()))
    return sorted(int(pin_id) for pin_id in sink_pin_ids)


def _candidate_source(source_flags, *, is_rebranch_related):
    if is_rebranch_related:
        return "rebranch_related"
    if "tree_node" in source_flags:
        return "original_tree_node"
    return "segment_sample"


def _children_by_node_from_edges(node_ids, directed_edges):
    children_by_node = {int(node_id): [] for node_id in node_ids}
    parent_by_node = {}
    for parent, child in directed_edges:
        parent = int(parent)
        child = int(child)
        children_by_node.setdefault(parent, []).append(child)
        children_by_node.setdefault(child, [])
        parent_by_node[child] = parent
    return {
        int(node_id): sorted(int(child) for child in children)
        for node_id, children in children_by_node.items()
    }, parent_by_node


def _candidate_sampling_tree_view(tree_view, rebranching_summary):
    if (
        not rebranching_summary
        or rebranching_summary.get("candidate_set_skeleton_source")
        != "timing_aware_rebranched_skeleton"
        or not rebranching_summary.get("rebranched_edges")
    ):
        return tree_view

    sampled_tree = dict(tree_view)
    directed_edges = [
        (int(parent), int(child))
        for parent, child in rebranching_summary.get("rebranched_edges", [])
    ]
    children_by_node, parent_by_node = _children_by_node_from_edges(
        sampled_tree.get("node_ids", []),
        directed_edges,
    )
    root_node_id = int(sampled_tree.get("root_node_id"))
    parent_by_node[root_node_id] = None
    sampled_tree["edge_pairs"] = directed_edges
    sampled_tree["children_by_node"] = children_by_node
    sampled_tree["parent_by_node"] = parent_by_node
    sampled_tree["candidate_set_skeleton_source"] = "timing_aware_rebranched_skeleton"
    return sampled_tree


def _rooted_edge_pairs(tree_view):
    children_by_node = tree_view.get("children_by_node") or {}
    edges = []
    for parent in sorted(int(node_id) for node_id in children_by_node):
        children = children_by_node.get(parent)
        if children is None:
            children = children_by_node.get(str(parent), [])
        for child in sorted(int(value) for value in children or []):
            edges.append((parent, child))
    return edges


def sample_buffer_candidates(
    tree_view,
    *,
    buffer_main_type_index,
    dbu,
    max_candidates_per_segment=3,
    rebranching_summary=None,
    include_tree_node_candidates=False,
    candidate_generation_policy="segment_only",
    return_summary=False,
):
    if tree_view.get("status") != "ok":
        return []
    if candidate_generation_policy != "segment_only":
        raise ValueError(f"unknown candidate_generation_policy: {candidate_generation_policy}")
    max_candidates_per_segment = int(max_candidates_per_segment)
    if max_candidates_per_segment <= 0:
        raise ValueError("max_candidates_per_segment must be positive")
    tree_view = _candidate_sampling_tree_view(tree_view, rebranching_summary)
    coordinates = tree_view.get("coordinates") or {}

    candidate_by_xy = {}
    total_tree_edge_count = 0
    coordinate_supported_edge_count = 0
    nonzero_tree_edge_count = 0
    edge_with_candidate_count = 0
    candidate_attempt_count = 0
    rounded_endpoint_skip_count = 0
    coordinate_dedup_count = 0
    net_id = int(tree_view["net_id"])
    if include_tree_node_candidates:
        for node_id in tree_view.get("node_ids", []):
            if node_id not in coordinates:
                continue
            x_dbu, y_dbu = coordinates[node_id]
            _merge_candidate(
                candidate_by_xy,
                {
                    "candidate_id": None,
                    "net_id": net_id,
                    "tree_node_id": int(node_id),
                    "parent_node_id": tree_view.get("parent_by_node", {}).get(node_id),
                    "child_node_ids": list(tree_view.get("children_by_node", {}).get(node_id, [])),
                    "x_dbu": int(x_dbu),
                    "y_dbu": int(y_dbu),
                    "x_um": float(x_dbu) / float(dbu),
                    "y_um": float(y_dbu) / float(dbu),
                    "candidate_generation_mode": "equal_count",
                    "max_candidates_per_segment": max_candidates_per_segment,
                    "segment_length_dbu": 0,
                    "segment_split_index": None,
                    "segment_split_count_on_edge": None,
                    "segment_split_ratio": None,
                    "source_flags": ["tree_node"],
                    "buffer_main_type_index": int(buffer_main_type_index),
                },
            )

    for parent, child in _rooted_edge_pairs(tree_view):
        total_tree_edge_count += 1
        if parent not in coordinates or child not in coordinates:
            continue
        coordinate_supported_edge_count += 1
        start = coordinates[parent]
        end = coordinates[child]
        length_dbu = _distance_dbu(start, end)
        if length_dbu <= 0.0:
            continue
        nonzero_tree_edge_count += 1
        edge_has_candidate = False
        for split_index in range(1, max_candidates_per_segment + 1):
            candidate_attempt_count += 1
            split_ratio = split_index / float(max_candidates_per_segment + 1)
            x_dbu, y_dbu = _interpolate(start, end, split_ratio)
            if _is_endpoint_coordinate((x_dbu, y_dbu), start, end):
                rounded_endpoint_skip_count += 1
                continue
            edge_has_candidate = True
            inserted = _merge_candidate(
                candidate_by_xy,
                {
                    "candidate_id": None,
                    "net_id": net_id,
                    "tree_node_id": None,
                    "parent_node_id": int(parent),
                    "child_node_ids": [int(child)],
                    "x_dbu": x_dbu,
                    "y_dbu": y_dbu,
                    "x_um": float(x_dbu) / float(dbu),
                    "y_um": float(y_dbu) / float(dbu),
                    "candidate_generation_mode": "equal_count",
                    "max_candidates_per_segment": max_candidates_per_segment,
                    "segment_length_dbu": int(round(length_dbu)),
                    "segment_split_index": split_index,
                    "segment_split_count_on_edge": max_candidates_per_segment,
                    "segment_split_ratio": float(split_ratio),
                    "source_flags": ["segment_sample"],
                    "buffer_main_type_index": int(buffer_main_type_index),
                },
            )
            if not inserted:
                coordinate_dedup_count += 1
        if edge_has_candidate:
            edge_with_candidate_count += 1

    candidates = [
        candidate_by_xy[key]
        for key in sorted(candidate_by_xy)
    ]
    related_nodes = _rebranch_related_nodes(rebranching_summary)
    for candidate_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = candidate_id
        rebranch_sink_pin_ids = _candidate_rebranch_sink_pin_ids(
            candidate,
            related_nodes,
        )
        candidate["rebranch_sink_pin_ids"] = rebranch_sink_pin_ids
        candidate["is_rebranch_related"] = bool(rebranch_sink_pin_ids)
        candidate["candidate_source"] = _candidate_source(
            candidate.get("source_flags", []),
            is_rebranch_related=candidate["is_rebranch_related"],
        )
    if not return_summary:
        return candidates
    return candidates, {
        "total_tree_edge_count": total_tree_edge_count,
        "coordinate_supported_edge_count": coordinate_supported_edge_count,
        "nonzero_tree_edge_count": nonzero_tree_edge_count,
        "edge_with_candidate_count": edge_with_candidate_count,
        "edge_without_candidate_count": (
            nonzero_tree_edge_count - edge_with_candidate_count
        ),
        "candidate_attempt_count": candidate_attempt_count,
        "rounded_endpoint_skip_count": rounded_endpoint_skip_count,
        "coordinate_dedup_count": coordinate_dedup_count,
        "candidate_count": len(candidates),
    }
