import importlib
import time

import torch


def _cpu_contiguous(value, *, dtype=None):
    tensor = torch.as_tensor(value).detach()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device="cpu").contiguous()


def _cpu_floating_contiguous(value):
    tensor = torch.as_tensor(value).detach()
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device="cpu").contiguous()


def _candidate_int(candidate, *names, default=-1):
    for name in names:
        value = candidate.get(name)
        if value is not None:
            return int(value)
    return int(default)


def _candidate_inputs(candidates):
    net_id = []
    parent_node_id = []
    child_node_id = []
    tree_node_id = []
    synthetic_node_id = []
    is_segment = []
    split_ratio = []
    for index, candidate in enumerate(candidates or []):
        original_node = _candidate_int(
            candidate,
            "node_id",
            "candidate_node_id",
            "tree_node_id",
            default=-1,
        )
        has_original_node = original_node >= 0
        children = list(candidate.get("child_node_ids", ()) or ())
        parent = candidate.get("parent_node_id")
        segment = not has_original_node and parent is not None and len(children) == 1
        if segment:
            ratio = candidate.get("segment_split_ratio")
            if ratio is None:
                raise ValueError(
                    "segment candidate is missing segment_split_ratio: "
                    f"candidate_index={index}"
                )
        else:
            ratio = 0.0
        net_id.append(_candidate_int(candidate, "net_id", "affected_net_id"))
        parent_node_id.append(int(parent) if segment else -1)
        child_node_id.append(int(children[0]) if segment else -1)
        tree_node_id.append(original_node if not segment else -1)
        synthetic_node_id.append(
            int(candidate.get("synthetic_node_id", -1 - index)) if segment else -1
        )
        is_segment.append(1 if segment else 0)
        split_ratio.append(float(ratio))
    return {
        "candidate_net_id": torch.as_tensor(net_id, dtype=torch.long).contiguous(),
        "candidate_parent_node_id": torch.as_tensor(
            parent_node_id, dtype=torch.long
        ).contiguous(),
        "candidate_child_node_id": torch.as_tensor(
            child_node_id, dtype=torch.long
        ).contiguous(),
        "candidate_tree_node_id": torch.as_tensor(
            tree_node_id, dtype=torch.long
        ).contiguous(),
        "candidate_synthetic_node_id": torch.as_tensor(
            synthetic_node_id, dtype=torch.long
        ).contiguous(),
        "candidate_is_segment": torch.as_tensor(
            is_segment, dtype=torch.long
        ).contiguous(),
        "candidate_split_ratio": torch.as_tensor(
            split_ratio, dtype=torch.float64
        ).contiguous(),
    }


def pack_candidate_topology(
    *,
    flat_net2pin,
    flat_net2pin_start,
    net2driver,
    pin2node,
    net_flat_topo_sort,
    net_flat_topo_sort_start,
    pin_fa,
    node_x,
    node_y,
    node_x_dbu,
    node_y_dbu,
    candidates,
    pin_capacitance,
    num_movable_nodes,
    num_terminals,
    dbu,
    scale_factor,
    r_unit,
    c_unit,
    num_threads=None,
):
    """Pack the candidate expanded topology without Python tree reconstruction."""

    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    candidate_values = _candidate_inputs(candidates)
    native_inputs = (
        _cpu_contiguous(flat_net2pin),
        _cpu_contiguous(flat_net2pin_start),
        _cpu_contiguous(net2driver),
        _cpu_contiguous(pin2node),
        _cpu_contiguous(net_flat_topo_sort),
        _cpu_contiguous(net_flat_topo_sort_start),
        _cpu_contiguous(pin_fa),
        _cpu_floating_contiguous(node_x),
        _cpu_floating_contiguous(node_y),
        _cpu_floating_contiguous(node_x_dbu),
        _cpu_floating_contiguous(node_y_dbu),
        candidate_values["candidate_net_id"],
        candidate_values["candidate_parent_node_id"],
        candidate_values["candidate_child_node_id"],
        candidate_values["candidate_tree_node_id"],
        candidate_values["candidate_synthetic_node_id"],
        candidate_values["candidate_is_segment"],
        candidate_values["candidate_split_ratio"],
        _cpu_floating_contiguous(pin_capacitance),
    )
    thread_count = torch.get_num_threads() if num_threads is None else int(num_threads)
    call_started_at = time.perf_counter()
    result = native.pack_candidate_topology(
        *native_inputs,
        int(num_movable_nodes),
        int(num_terminals),
        float(dbu),
        float(scale_factor),
        float(r_unit),
        float(c_unit),
        thread_count,
    )
    native_pybind_call_ms = (time.perf_counter() - call_started_at) * 1000.0
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "native_pybind_call_ms": float(native_pybind_call_ms),
            "candidate_count": int(len(candidates or [])),
            "candidate_input_materialize_ms": float(
                metadata.get("candidate_input_materialize_ms", 0.0) or 0.0
            ),
        }
    )
    result["metadata"] = metadata
    prepared = result["prepared_timing_inputs"]
    prepared["metadata"] = metadata
    prepared["native_input_keys"] = (
        "net_flat_topo_sort",
        "net_flat_topo_sort_start",
        "pin_fa",
        "flat_pin_to_start",
        "flat_pin_to",
        "edge_resistance",
        "node_capacitance",
        "edge_capacitance",
        "driver_arrival",
        "driver_slew",
        "candidate_node_id",
        "candidate_bu",
        "buffer_input_cap",
        "buffer_delay",
        "buffer_output_slew",
        "sink_node_id",
        "sink_net_index",
    )
    return result


def select_packed_candidate_inputs_native(
    prepared,
    *,
    active_net_ids,
    num_threads=None,
):
    """Select one timing level from a native-packed candidate topology."""

    native = importlib.import_module(f"{__package__}.net_subgraph_timing_cpp")
    if torch.is_tensor(active_net_ids):
        active_net_ids = active_net_ids.detach().to(
            device="cpu", dtype=torch.long
        ).contiguous()
    else:
        active_net_ids = torch.as_tensor(
            list(active_net_ids or ()), dtype=torch.long
        ).contiguous()
    thread_count = torch.get_num_threads() if num_threads is None else int(num_threads)
    call_started_at = time.perf_counter()
    selected = native.select_packed_candidate_inputs(
        prepared,
        active_net_ids,
        thread_count,
    )
    native_pybind_call_ms = (time.perf_counter() - call_started_at) * 1000.0
    metadata = dict(selected.get("metadata", {}) or {})
    native_cpp_accounted_ms = sum(
        float(metadata.get(name, 0.0) or 0.0)
        for name in (
            "native_active_view_match_count_ms",
            "native_active_view_prefix_sum_ms",
            "native_active_view_allocate_ms",
            "native_active_view_fill_ms",
        )
    )
    metadata.update(
        {
            "native_pybind_call_ms": float(native_pybind_call_ms),
            "native_return_attach_ms": max(
                0.0, float(native_pybind_call_ms) - native_cpp_accounted_ms
            ),
        }
    )
    selected["metadata"] = metadata
    return selected
