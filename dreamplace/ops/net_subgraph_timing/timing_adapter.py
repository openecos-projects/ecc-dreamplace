from dataclasses import replace

import torch

from .net_subgraph_timing import net_subgraph_forward_native, net_subgraph_forward_relaxed
from .tensor_builder import NATIVE_INPUT_KEYS, build_net_subgraph_timing_inputs

SINK_FILTER_ALL_NET_SINKS = "all_net_sinks"
SINK_FILTER_CANDIDATE_DOWNSTREAM = "candidate_downstream"


def _dynamic_net_arc_semantics():
    return {
        "schema_name": "buffering_dynamic_net_arc_semantics",
        "schema_version": 1,
        "pin_net_delay_semantics": "relative_net_arc_delay",
        "pin_net_impulse_semantics": "relative_net_arc_slew_square_delta",
        "pin_net_cap_semantics": "sink_load_with_optional_driver_root_lin_overlay",
        "includes_driver_arrival": False,
        "includes_buffer_intrinsic_delay": True,
        "includes_upstream_wire_delay": True,
        "includes_downstream_wire_delay": True,
        "timing_op_relative_net_arc_delay_calibrated": True,
        "timing_op_contract_status": "relative_net_arc_contract",
    }


def _as_long_index(index, *, device):
    return torch.as_tensor(index, device=device).long()


def _validate_unique_sink_pins(sink_pin_index):
    if sink_pin_index.numel() <= 1:
        return
    unique_count = int(torch.unique(sink_pin_index.detach().cpu()).numel())
    if unique_count != int(sink_pin_index.numel()):
        raise ValueError("duplicate sink_pin_id is not allowed")


def build_timing_pin_net_inputs(
    net_subgraph_result,
    *,
    sink_pin_id,
    num_pins,
    driver_pin_id=None,
    driver_net_cap=None,
    fill_value=0.0,
    allow_duplicate_sink_pins=False,
):
    """Map net-subgraph sink outputs onto TimingPropagation pin-domain inputs.

    TimingPropagation consumes pin-indexed net delay / impulse / cap tensors.
    The dynamic net-subgraph op emits compact sink outputs, so this adapter is
    the narrow contract between the two modules before full timing integration.
    """
    sink_arrival = net_subgraph_result["sink_arrival"]
    sink_net_delay = net_subgraph_result["sink_net_delay"]
    sink_slew = net_subgraph_result["sink_slew"]
    sink_net_impulse = net_subgraph_result["sink_net_impulse"]
    sink_load = net_subgraph_result["sink_load"]
    device = sink_arrival.device
    dtype = sink_arrival.dtype
    sink_pin_index = _as_long_index(sink_pin_id, device=device)

    if not allow_duplicate_sink_pins:
        _validate_unique_sink_pins(sink_pin_index)
    if sink_pin_index.numel() != sink_arrival.numel():
        raise ValueError("sink_pin_id length must match sink output length")
    if sink_net_delay.numel() != sink_arrival.numel():
        raise ValueError("sink_net_delay length must match sink output length")
    if sink_net_impulse.numel() != sink_arrival.numel():
        raise ValueError("sink_net_impulse length must match sink output length")
    if sink_slew.numel() != sink_arrival.numel() or sink_load.numel() != sink_arrival.numel():
        raise ValueError("sink output tensors must have matching lengths")
    if sink_pin_index.numel() > 0:
        if int(torch.min(sink_pin_index).item()) < 0:
            raise ValueError("sink_pin_id contains negative pin index")
        if int(torch.max(sink_pin_index).item()) >= int(num_pins):
            raise ValueError("sink_pin_id is out of pin-domain range")

    def scatter(values):
        out = torch.full(
            (int(num_pins),),
            fill_value,
            device=device,
            dtype=dtype,
        )
        out[sink_pin_index] = values.to(device=device, dtype=dtype)
        return out

    pin_delay = scatter(sink_net_delay)
    pin_impulse = scatter(sink_net_impulse)
    pin_cap = scatter(sink_load)
    pin_arrival = scatter(sink_arrival)
    pin_slew = scatter(sink_slew)
    if driver_pin_id is not None and driver_net_cap is not None:
        driver_pin_index = _as_long_index(driver_pin_id, device=device)
        driver_cap_values = driver_net_cap.to(device=device, dtype=dtype)
        if driver_pin_index.numel() != driver_cap_values.numel():
            raise ValueError("driver_pin_id length must match driver_net_cap length")
        if driver_pin_index.numel() > 0:
            if int(torch.min(driver_pin_index).item()) < 0:
                raise ValueError("driver_pin_id contains negative pin index")
            if int(torch.max(driver_pin_index).item()) >= int(num_pins):
                raise ValueError("driver_pin_id is out of pin-domain range")
            pin_cap[driver_pin_index] = driver_cap_values
    result = {
        "pin_net_delays": {
            "rise": pin_delay,
            "fall": pin_delay.clone(),
        },
        "pin_net_impulses": {
            "rise": pin_impulse,
            "fall": pin_impulse.clone(),
        },
        "pin_net_caps": {
            "rise": pin_cap,
            "fall": pin_cap.clone(),
        },
        "pin_net_arrivals": {
            "rise": pin_arrival,
            "fall": pin_arrival.clone(),
        },
        "pin_net_slews": {
            "rise": pin_slew,
            "fall": pin_slew.clone(),
        },
        "sink_pin_id": sink_pin_index.to(dtype=torch.as_tensor(sink_pin_id).dtype),
        "net_arc_semantics": _dynamic_net_arc_semantics(),
    }
    if "sink_net_index" in net_subgraph_result:
        result["sink_net_index"] = net_subgraph_result["sink_net_index"].to(device=device)
    if driver_pin_id is not None:
        result["driver_pin_id"] = _as_long_index(driver_pin_id, device=device).to(
            dtype=torch.as_tensor(driver_pin_id).dtype
        )
    return result


def _native_inputs_only(inputs):
    return {key: inputs[key] for key in inputs["native_input_keys"]}


def _candidate_net_id(candidate):
    value = candidate.get("net_id", candidate.get("affected_net_id"))
    return int(value) if value is not None else None


def _candidate_downstream_pin_ids_by_net(candidates):
    result = {}
    for candidate in candidates:
        net_id = _candidate_net_id(candidate)
        if net_id is None:
            continue
        downstream = candidate.get("downstream_pin_ids")
        if downstream is None:
            downstream = candidate.get("downstream_sink_ids")
        if downstream is None:
            continue
        pins = result.setdefault(int(net_id), set())
        pins.update(int(value) for value in list(downstream or []))
    return {
        int(net_id): sorted(int(pin_id) for pin_id in pin_ids)
        for net_id, pin_ids in result.items()
    }


def _apply_sink_filter_to_inputs(inputs, candidates, *, sink_filter_mode):
    mode = str(sink_filter_mode or SINK_FILTER_ALL_NET_SINKS)
    if mode not in {SINK_FILTER_ALL_NET_SINKS, SINK_FILTER_CANDIDATE_DOWNSTREAM}:
        raise ValueError(f"unsupported sink_filter_mode: {mode}")

    sink_pin_ids = [int(value) for value in inputs["sink_pin_id"].detach().cpu().tolist()]
    sink_net_ids = [int(value) for value in inputs["sink_net_index"].detach().cpu().tolist()]
    metadata = {
        "schema_name": "buffering_relaxed_payload_sink_filter",
        "schema_version": 1,
        "mode": mode,
        "status": "all_net_sinks",
        "all_net_sink_count": int(len(sink_pin_ids)),
        "filtered_sink_count": int(len(sink_pin_ids)),
        "candidate_downstream_sink_count": 0,
        "candidate_downstream_pin_ids_by_net": {},
    }
    if mode == SINK_FILTER_ALL_NET_SINKS:
        return inputs, metadata

    downstream_by_net = _candidate_downstream_pin_ids_by_net(candidates)
    metadata["candidate_downstream_pin_ids_by_net"] = {
        int(net_id): list(pin_ids) for net_id, pin_ids in downstream_by_net.items()
    }
    metadata["candidate_downstream_sink_count"] = int(
        sum(len(pin_ids) for pin_ids in downstream_by_net.values())
    )
    if not downstream_by_net:
        metadata["status"] = "candidate_downstream_missing_fallback_all"
        return inputs, metadata

    downstream_sets = {
        int(net_id): {int(pin_id) for pin_id in pin_ids}
        for net_id, pin_ids in downstream_by_net.items()
    }
    keep_indices = []
    fallback_net_ids = set()
    for index, (pin_id, net_id) in enumerate(zip(sink_pin_ids, sink_net_ids)):
        downstream_pins = downstream_sets.get(int(net_id))
        if downstream_pins is None:
            fallback_net_ids.add(int(net_id))
            keep_indices.append(index)
        elif int(pin_id) in downstream_pins:
            keep_indices.append(index)

    if not keep_indices:
        metadata["status"] = "candidate_downstream_no_match_fallback_all"
        metadata["filtered_sink_count"] = int(len(sink_pin_ids))
        return inputs, metadata

    if len(keep_indices) == len(sink_pin_ids):
        metadata["status"] = (
            "candidate_downstream_with_unfiltered_sinks"
            if fallback_net_ids
            else "candidate_downstream_matches_all_sinks"
        )
        return inputs, metadata

    keep_index = torch.as_tensor(
        keep_indices,
        dtype=torch.long,
        device=inputs["sink_pin_id"].device,
    )
    filtered = dict(inputs)
    filtered["sink_pin_id"] = inputs["sink_pin_id"].index_select(0, keep_index)
    filtered["sink_node_id"] = inputs["sink_node_id"].index_select(0, keep_index)
    filtered["sink_net_index"] = inputs["sink_net_index"].index_select(0, keep_index)
    metadata["status"] = "filtered"
    metadata["filtered_sink_count"] = int(len(keep_indices))
    if fallback_net_ids:
        metadata["fallback_all_sink_net_ids"] = sorted(fallback_net_ids)
    return filtered, metadata


def _driver_pin_cap_inputs(inputs, net_subgraph_result):
    metadata = dict(inputs.get("metadata", {}) or {})
    compact_node_to_original = list(metadata.get("compact_node_to_original", []))
    topo = inputs["net_flat_topo_sort"].detach().cpu().tolist()
    topo_start = inputs["net_flat_topo_sort_start"].detach().cpu().tolist()
    driver_pin_ids = []
    root_compact_ids = []
    for net_index in range(max(0, len(topo_start) - 1)):
        if int(topo_start[net_index]) == int(topo_start[net_index + 1]):
            continue
        root_compact = int(topo[int(topo_start[net_index])])
        if root_compact < 0 or root_compact >= len(compact_node_to_original):
            continue
        driver_pin_ids.append(int(compact_node_to_original[root_compact]))
        root_compact_ids.append(root_compact)
    if not driver_pin_ids:
        return None, None
    root_index = torch.as_tensor(
        root_compact_ids,
        dtype=torch.long,
        device=net_subgraph_result["lin"].device,
    )
    return (
        torch.as_tensor(
            driver_pin_ids,
            dtype=torch.int32,
            device=net_subgraph_result["lin"].device,
        ),
        net_subgraph_result["lin"][root_index],
    )


def build_dynamic_net_arc_inputs_from_net_subgraphs(
    nets,
    *,
    candidates=None,
    num_pins,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    default_driver_arrival=0.0,
    default_driver_slew=0.0,
    fill_value=0.0,
    allow_duplicate_sink_pins=False,
    dtype=None,
    device=None,
):
    """Build TimingPropagation dynamic net-arc tensors through native net timing.

    This is the scheduling adapter for the fixed-state buffer-aware net arc:
    Du net records + current virtual buffer state -> native net_subgraph_timing
    -> pin-domain delay/slew/cap dictionaries consumed by TimingPropagation.
    """
    if dtype is None:
        dtype = torch.float32
    inputs = build_net_subgraph_timing_inputs(
        nets,
        candidates=candidates,
        driver_arrival_by_net=driver_arrival_by_net,
        driver_slew_by_net=driver_slew_by_net,
        default_driver_arrival=default_driver_arrival,
        default_driver_slew=default_driver_slew,
        dtype=dtype,
        device=device,
    )
    net_result = net_subgraph_forward_native(**_native_inputs_only(inputs))
    driver_pin_id, driver_net_cap = _driver_pin_cap_inputs(inputs, net_result)
    adapted = build_timing_pin_net_inputs(
        net_result,
        sink_pin_id=inputs["sink_pin_id"],
        num_pins=num_pins,
        driver_pin_id=driver_pin_id,
        driver_net_cap=driver_net_cap,
        fill_value=fill_value,
        allow_duplicate_sink_pins=allow_duplicate_sink_pins,
    )
    adapted["source"] = "net_subgraph_timing_cpp"
    adapted["net_subgraph_timing_mode"] = "native_fixed_state_forward"
    adapted["metadata"] = dict(inputs.get("metadata", {}))
    adapted["metadata"]["net_subgraph_invocation_timing"] = (
        "pre_timing_propagation_overlay"
    )
    adapted["metadata"]["uses_current_propagated_driver_slew"] = False
    adapted["metadata"]["net_arc_semantics"] = dict(adapted["net_arc_semantics"])
    if driver_pin_id is not None:
        adapted["metadata"]["driver_pin_ids"] = [
            int(value) for value in driver_pin_id.detach().cpu().tolist()
        ]
        adapted["metadata"]["driver_net_cap_source"] = "net_subgraph_root_lin"
    return adapted


def build_relaxed_buffer_dynamic_net_arc_inputs(
    *,
    buffer_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    sink_pin_id,
    num_pins,
    fill_value=0.0,
    allow_duplicate_sink_pins=False,
    coordinate_source="current_relaxed",
    **net_subgraph_kwargs,
):
    payload_metadata = net_subgraph_kwargs.pop("metadata", None)
    for name in (
        "net_ids",
        "net_topo_start",
        "net_edge_start",
        "net_sink_start",
        "edge_start",
        "candidate_net_id",
    ):
        net_subgraph_kwargs.pop(name, None)
    candidate_node_id = net_subgraph_kwargs.pop("candidate_node_id", None)
    if candidate_node_id is not None:
        buffer_state = replace(
            buffer_state,
            candidate_node_id=torch.as_tensor(
                candidate_node_id,
                device=buffer_state.candidate_node_id.device,
                dtype=buffer_state.candidate_node_id.dtype,
            ),
        )
    net_result = net_subgraph_forward_relaxed(
        buffer_state=buffer_state,
        per_size_input_cap=per_size_input_cap,
        per_size_delay=per_size_delay,
        per_size_output_slew=per_size_output_slew,
        coordinate_source=coordinate_source,
        **net_subgraph_kwargs,
    )
    driver_pin_id = None
    driver_net_cap = None
    if isinstance(payload_metadata, dict):
        driver_pin_id, driver_net_cap = _driver_pin_cap_inputs(
            {"metadata": payload_metadata, **net_subgraph_kwargs},
            net_result,
        )
    adapted = build_timing_pin_net_inputs(
        net_result,
        sink_pin_id=sink_pin_id,
        num_pins=num_pins,
        driver_pin_id=driver_pin_id,
        driver_net_cap=driver_net_cap,
        fill_value=fill_value,
        allow_duplicate_sink_pins=allow_duplicate_sink_pins,
    )
    buffer_model = net_result.get("relaxed_buffer_model", {})
    bsu_status = dict(buffer_model.get("bsu_index_status", {}))
    if coordinate_source is not None and "coordinate_source" not in bsu_status:
        bsu_status["coordinate_source"] = str(coordinate_source)
    adapted["source"] = "net_subgraph_timing_relaxed_buffer"
    adapted["net_subgraph_timing_mode"] = "relaxed_buffer_autograd"
    adapted["gradient_chain_level"] = "toy_global_timing_autograd"
    adapted["buffer_coordinate_source"] = str(
        bsu_status.get("coordinate_source", coordinate_source)
    )
    adapted["metadata"] = {
        "candidate_count": int(buffer_state.candidate_ids.numel()),
        "legal_buffer_count": int(buffer_state.legal_buffer_count),
        "buffer_main_type_index": int(buffer_state.buffer_main_type_index),
        "bsu_index_status": bsu_status,
        "net_subgraph_invocation_timing": "pre_timing_propagation_overlay",
        "uses_current_propagated_driver_slew": False,
    }
    if payload_metadata is not None:
        adapted["metadata"]["payload_metadata"] = dict(payload_metadata)
    adapted["metadata"]["net_arc_semantics"] = dict(adapted["net_arc_semantics"])
    if driver_pin_id is not None:
        adapted["metadata"]["driver_pin_ids"] = [
            int(value) for value in driver_pin_id.detach().cpu().tolist()
        ]
        adapted["metadata"]["driver_net_cap_source"] = "net_subgraph_root_lin"
    return adapted


def build_relaxed_buffer_timing_payload(
    nets,
    candidates,
    *,
    buffer_state,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    num_pins=None,
    driver_arrival_by_net=None,
    driver_slew_by_net=None,
    default_driver_arrival=0.0,
    default_driver_slew=0.0,
    fill_value=0.0,
    coordinate_source="current_relaxed",
    sink_filter_mode=SINK_FILTER_ALL_NET_SINKS,
    dtype=None,
    device=None,
    compact_runtime_metadata=False,
    prebuilt_inputs=None,
):
    candidates = list(candidates or [])
    if int(buffer_state.candidate_ids.numel()) != len(candidates):
        raise ValueError("buffer_state candidate count must match candidates")
    if dtype is None:
        dtype = torch.float32
    if prebuilt_inputs is None:
        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=candidates,
            driver_arrival_by_net=driver_arrival_by_net,
            driver_slew_by_net=driver_slew_by_net,
            default_driver_arrival=default_driver_arrival,
            default_driver_slew=default_driver_slew,
            dtype=dtype,
            device=device,
            compact_runtime_metadata=compact_runtime_metadata,
        )
    else:
        inputs = dict(prebuilt_inputs)
        metadata = dict(inputs.get("metadata", {}) or {})
        net_ids = [int(value) for value in metadata.get("net_ids", ())]
        input_device = inputs["net_flat_topo_sort"].device
        value_dtype = torch.as_tensor(inputs["node_capacitance"]).dtype
        inputs["driver_arrival"] = torch.as_tensor(
            [
                float((driver_arrival_by_net or {}).get(net_id, default_driver_arrival))
                for net_id in net_ids
            ],
            dtype=value_dtype,
            device=input_device,
        )
        inputs["driver_slew"] = torch.as_tensor(
            [
                float((driver_slew_by_net or {}).get(net_id, default_driver_slew))
                for net_id in net_ids
            ],
            dtype=value_dtype,
            device=input_device,
        )
        inputs.setdefault("native_input_keys", NATIVE_INPUT_KEYS)
        inputs["metadata"] = metadata
    inputs, sink_filter_metadata = _apply_sink_filter_to_inputs(
        inputs,
        candidates,
        sink_filter_mode=sink_filter_mode,
    )
    payload = {
        "per_size_input_cap": per_size_input_cap,
        "per_size_delay": per_size_delay,
        "per_size_output_slew": per_size_output_slew,
        "sink_pin_id": inputs["sink_pin_id"],
        "fill_value": fill_value,
        "net_flat_topo_sort": inputs["net_flat_topo_sort"],
        "net_flat_topo_sort_start": inputs["net_flat_topo_sort_start"],
        "pin_fa": inputs["pin_fa"],
        "flat_pin_to_start": inputs["flat_pin_to_start"],
        "flat_pin_to": inputs["flat_pin_to"],
        "edge_resistance": inputs["edge_resistance"],
        "node_capacitance": inputs["node_capacitance"],
        "edge_capacitance": inputs["edge_capacitance"],
        "driver_arrival": inputs["driver_arrival"],
        "driver_slew": inputs["driver_slew"],
        "candidate_node_id": inputs["candidate_node_id"],
        "sink_node_id": inputs["sink_node_id"],
        "sink_net_index": inputs["sink_net_index"],
        "coordinate_source": coordinate_source,
        "metadata": dict(inputs.get("metadata", {})),
    }
    for name in (
        "net_ids",
        "net_topo_start",
        "net_edge_start",
        "net_sink_start",
        "edge_start",
        "candidate_net_id",
    ):
        if name in inputs:
            payload[name] = inputs[name]
    payload["metadata"]["sink_filter"] = sink_filter_metadata
    if num_pins is not None:
        payload["num_pins"] = int(num_pins)
    return payload
