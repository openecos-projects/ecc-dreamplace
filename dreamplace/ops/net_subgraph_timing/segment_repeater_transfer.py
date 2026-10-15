import torch

from .net_subgraph_timing import net_subgraph_forward_native
from .tensor_builder import build_net_subgraph_timing_inputs


def _net_id(net, fallback=0):
    return int(net.get("net_id", fallback))


def _coordinates_by_node(net, *, coordinate_key="coordinates"):
    return {int(node): tuple(value) for node, value in (net.get(coordinate_key) or {}).items()}


def _children_by_node(net):
    rc_tree = net.get("rc_tree", {})
    return {
        int(node): [int(child) for child in children]
        for node, children in rc_tree.get("children_by_node", {}).items()
    }


def _find_net(nets, net_id):
    target = int(net_id)
    for index, net in enumerate(nets):
        if _net_id(net, index) == target:
            return net
    raise ValueError(f"net_id {target} is not present in selected net records")


def _validate_segment(
    net,
    parent_node_id,
    child_node_id,
    *,
    require_coordinates,
    coordinate_key="coordinates",
):
    parent = int(parent_node_id)
    child = int(child_node_id)
    children = _children_by_node(net).get(parent, [])
    if child not in children:
        raise ValueError(
            f"segment edge ({parent}, {child}) is not in selected net topology"
        )
    coordinates = _coordinates_by_node(net, coordinate_key=coordinate_key)
    if require_coordinates and (parent not in coordinates or child not in coordinates):
        raise ValueError(
            f"segment edge ({parent}, {child}) requires parent and child coordinate records"
        )
    return parent, child, coordinates


def _native_inputs_only(inputs):
    return {key: inputs[key] for key in inputs["native_input_keys"]}


def _validate_repeater_count(repeater_count):
    try:
        count = int(repeater_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("repeater_count must be a non-negative integer") from exc
    if isinstance(repeater_count, bool) or count != repeater_count:
        raise ValueError("repeater_count must be a non-negative integer")
    if count < 0:
        raise ValueError("repeater_count must be a non-negative integer")
    return count


def _compact_nodes_for_segment(inputs, net_id, parent_node_id, child_node_id):
    net_map = inputs["metadata"].get("original_node_to_compact_by_net", {}).get(int(net_id))
    if net_map is None:
        raise ValueError(f"net_id {int(net_id)} is missing from compact node metadata")
    parent = int(parent_node_id)
    child = int(child_node_id)
    if parent not in net_map or child not in net_map:
        raise ValueError(
            f"segment edge ({parent}, {child}) is missing compact node mapping"
        )
    candidate_compact_nodes = [
        int(node)
        for node in inputs["candidate_node_id"].detach().cpu().tolist()
    ]
    return {
        "parent_compact": int(net_map[parent]),
        "child_compact": int(net_map[child]),
        "candidate_compact_nodes": candidate_compact_nodes,
        "first_synthetic_compact": candidate_compact_nodes[0] if candidate_compact_nodes else None,
    }


def build_equal_spaced_segment_candidates(
    net,
    *,
    parent_node_id,
    child_node_id,
    repeater_count,
    bsu=None,
    buffer_main_type_index=None,
    candidate_id_start=0,
    source_flags=("equal_spaced_segment_count",),
    bu=0.0,
    coordinate_key="coordinates",
):
    """Create candidate records for equally spaced repeaters on one tree edge."""
    repeater_count = _validate_repeater_count(repeater_count)

    parent, child, coordinates = _validate_segment(
        net,
        parent_node_id,
        child_node_id,
        require_coordinates=repeater_count > 0,
        coordinate_key=coordinate_key,
    )
    if repeater_count == 0:
        return []

    px, py = coordinates[parent]
    cx, cy = coordinates[child]
    flags = list(source_flags or ())
    candidates = []
    for index in range(repeater_count):
        ratio = float(index + 1) / float(repeater_count + 1)
        candidate = {
            "candidate_id": int(candidate_id_start) + index,
            "net_id": _net_id(net),
            "tree_node_id": None,
            "parent_node_id": parent,
            "child_node_ids": [child],
            "x_dbu": int(round(float(px) + (float(cx) - float(px)) * ratio)),
            "y_dbu": int(round(float(py) + (float(cy) - float(py)) * ratio)),
            "bu": float(bu),
            "segment_split_index": index,
            "segment_split_count_on_edge": repeater_count,
            "segment_split_ratio": ratio,
            "source_flags": list(flags),
            "buffer_main_type_index": buffer_main_type_index,
        }
        if bsu is not None:
            candidate["bsu"] = int(bsu)
        candidates.append(candidate)
    return candidates


def evaluate_expanded_segment_reference(
    nets,
    *,
    net_id,
    parent_node_id,
    child_node_id,
    repeater_count,
    bsu=None,
    buffer_surrogate=None,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    dtype=torch.float32,
    device=None,
    forward_fn=None,
    candidate_id_start=0,
    buffer_main_type_index=None,
):
    """Evaluate a segment repeater transfer through expanded synthetic nodes."""
    if dtype is None:
        dtype = torch.float32
    nets = list(nets or [])
    forward_fn = net_subgraph_forward_native if forward_fn is None else forward_fn
    net = _find_net(nets, net_id)
    repeater_count = _validate_repeater_count(repeater_count)
    _validate_segment(
        net,
        parent_node_id,
        child_node_id,
        require_coordinates=repeater_count > 0,
    )
    if repeater_count > 0 and bsu is None:
        raise ValueError("bsu is required when repeater_count is greater than zero")
    if repeater_count > 0 and buffer_surrogate is None:
        raise ValueError(
            "buffer_surrogate is required when repeater_count is greater than zero"
        )

    coordinate_candidates = build_equal_spaced_segment_candidates(
        net,
        parent_node_id=parent_node_id,
        child_node_id=child_node_id,
        repeater_count=repeater_count,
        bsu=bsu,
        buffer_main_type_index=buffer_main_type_index,
        candidate_id_start=candidate_id_start,
        bu=0.0,
    )
    for candidate in coordinate_candidates:
        candidate.update(
            {
                "buffer_input_cap": 0.0,
                "buffer_delay": 0.0,
                "buffer_output_slew": 0.0,
            }
        )

    coordinate_inputs = build_net_subgraph_timing_inputs(
        nets,
        candidates=coordinate_candidates,
        driver_arrival_by_net=driver_arrival_by_net,
        driver_slew_by_net=driver_slew_by_net,
        dtype=dtype,
        device=device,
    )
    coordinate_result = forward_fn(**_native_inputs_only(coordinate_inputs))
    coordinate_node_ids = _compact_nodes_for_segment(
        coordinate_inputs,
        net_id,
        parent_node_id,
        child_node_id,
    )

    candidate_input_slew = []
    candidate_lout = []
    surrogate_results = []
    expanded_candidates = []
    for index, candidate in enumerate(coordinate_candidates):
        compact_node = coordinate_node_ids["candidate_compact_nodes"][index]
        input_slew = float(coordinate_result["slew_in"][compact_node].item())
        output_cap = float(coordinate_result["lout"][compact_node].item())
        surrogate = dict(
            buffer_surrogate(
                candidate,
                bsu=int(bsu),
                input_slew=input_slew,
                output_cap=output_cap,
            )
        )
        candidate_input_slew.append(input_slew)
        candidate_lout.append(output_cap)
        surrogate_results.append(dict(surrogate))
        expanded = dict(candidate)
        expanded.update(
            {
                "bu": 1.0,
                "buffer_input_cap": float(surrogate["buffer_input_cap"]),
                "buffer_delay": float(surrogate["buffer_delay"]),
                "buffer_output_slew": float(surrogate["buffer_output_slew"]),
                "candidate_input_slew": input_slew,
                "candidate_output_cap": output_cap,
            }
        )
        expanded_candidates.append(expanded)

    expanded_inputs = build_net_subgraph_timing_inputs(
        nets,
        candidates=expanded_candidates,
        driver_arrival_by_net=driver_arrival_by_net,
        driver_slew_by_net=driver_slew_by_net,
        dtype=dtype,
        device=device,
    )
    expanded_result = forward_fn(**_native_inputs_only(expanded_inputs))
    expanded_node_ids = _compact_nodes_for_segment(
        expanded_inputs,
        net_id,
        parent_node_id,
        child_node_id,
    )

    parent_compact = expanded_node_ids["parent_compact"]
    child_compact = expanded_node_ids["child_compact"]
    delay = (
        expanded_result["arrival_in"][child_compact]
        - expanded_result["arrival_out"][parent_compact]
    )
    output_slew = expanded_result["slew_in"][child_compact]
    first_synthetic = expanded_node_ids["first_synthetic_compact"]
    if first_synthetic is None:
        upstream_visible_input_cap = expanded_result["lin"][child_compact]
    else:
        upstream_visible_input_cap = expanded_result["lin"][int(first_synthetic)]

    diagnostics = {
        "net_id": int(net_id),
        "parent_node_id": int(parent_node_id),
        "child_node_id": int(child_node_id),
        "repeater_count": int(repeater_count),
        "bsu": None if bsu is None else int(bsu),
        "coordinate_candidates": [dict(candidate) for candidate in coordinate_candidates],
        "expanded_candidates": [dict(candidate) for candidate in expanded_candidates],
        "coordinate_inputs": coordinate_inputs,
        "coordinate_result": coordinate_result,
        "coordinate_node_ids": coordinate_node_ids,
        "coordinate_candidate_input_slew": list(candidate_input_slew),
        "coordinate_candidate_lout": list(candidate_lout),
        "coordinate_candidate_output_cap": list(candidate_lout),
        "expanded_inputs": expanded_inputs,
        "expanded_result": expanded_result,
        "expanded_node_ids": expanded_node_ids,
        "surrogate_results": surrogate_results,
        "split_ratios": [
            float(candidate["segment_split_ratio"])
            for candidate in coordinate_candidates
        ],
    }
    return {
        "delay": delay,
        "output_slew": output_slew,
        "upstream_visible_input_cap": upstream_visible_input_cap,
        "diagnostics": diagnostics,
    }


def segment_repeater_transfer(*args, **kwargs):
    """Public MVP wrapper around the expanded segment reference."""
    return evaluate_expanded_segment_reference(*args, **kwargs)
