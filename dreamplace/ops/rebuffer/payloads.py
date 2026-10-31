"""Array payloads shared by the native rebuffer planners.

``placedb.pydb`` stays the only data source: this module copies the STA and
Steiner arrays into contiguous numpy buffers and nothing else.
"""

import numpy as np

from .steiner_payload import build_openroad_steiner_payload
from .timing_model_payload import timing_pin_slack


def _numpy(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def timing_payload(pydb, timing_model, *, path_context=None):
    if path_context:
        return _timing_payload(pydb, timing_model, path_context=path_context)
    # Every driver payload of one snapshot reads the same timing state. Each
    # forward (full or affected) replaces these tensors, so their identity is
    # the state version; the cache holds them so ids cannot be reused. Callers
    # that edit an array (load slack override) copy it first.
    model = timing_model.model
    sources = (
        pydb,
        model,
        timing_model.loads["rise"],
        getattr(model, "pin_rtran_live", None),
        getattr(model, "pin_slack", None),
        getattr(model, "pin_slack_live_snapshot", None),
        getattr(model, "last_endpoint_slack_tensor", None),
        getattr(model, "pin_rAAT_live_snapshot", None),
    )
    cache = getattr(timing_model, "_timing_payload_cache", None)
    if cache is not None and all(a is b for a, b in zip(cache[0], sources)):
        return dict(cache[1])
    payload = _timing_payload(pydb, timing_model)
    timing_model._timing_payload_cache = (sources, payload)
    return dict(payload)


def _timing_payload(pydb, timing_model, *, path_context=None):
    payload = {
        "pin_slew_rise": _numpy(timing_model.model.pin_rtran_live),
        "pin_slew_fall": _numpy(timing_model.model.pin_ftran_live),
        "pin_load": _numpy(timing_model.loads["rise"])[: len(pydb.pin_names)],
        "pin_cap": _numpy(timing_model.lib_pin_caps),
        "pin_max_slew": _numpy(timing_model.pin_max_slew),
        "pin_max_cap": _numpy(timing_model.pin_max_cap),
        "pin_slack": _numpy(timing_pin_slack(timing_model.model))[: len(pydb.pin_names)],
        "clock_pins": _numpy(getattr(pydb, "clock_pins", ()), np.int32),
    }
    for field, snapshot in (
        ("pin_arrival_rise", "pin_rAAT_live_snapshot"),
        ("pin_arrival_fall", "pin_fAAT_live_snapshot"),
        ("pin_required_rise", "pin_rRAT_live_snapshot"),
        ("pin_required_fall", "pin_fRAT_live_snapshot"),
    ):
        values = getattr(timing_model.model, snapshot, None)
        if values is not None:
            payload[field] = _numpy(values)[: len(pydb.pin_names)]
    if path_context:
        for key in (
            "path_driver_input_pin",
            "path_driver_input_transition",
            "path_driver_output_transition",
            "path_driver_lib_arc_id",
        ):
            if key in path_context:
                payload[key] = int(path_context[key])
        for key in (
            "path_arc_delays",
            "path_arc_local_ids",
            "path_transitions",
        ):
            if key in path_context:
                payload[key] = np.ascontiguousarray(path_context[key])
        for key in (
            "path_driver_input_arrival",
            "path_driver_output_arrival",
        ):
            if key in path_context:
                payload[key] = float(path_context[key])
        if "path_driver_arrival_valid" in path_context:
            payload["path_driver_arrival_valid"] = bool(path_context["path_driver_arrival_valid"])
    return payload


def steiner_arrays(timing_model, steiner_view=None):
    """Return the active Steiner CSR and coordinates as contiguous arrays.

    One repair snapshot exports a current tree per path driver; all of them
    read the same Steiner state. Convert it once per source object instead of
    once per driver.
    """
    if steiner_view is None:
        steiner = timing_model.steiner
        owner = steiner
        source = "active_timing_propagation_steiner"
        fields = {
            "net_flat_topo_sort_start": steiner.net_flat_topo_sort_start,
            "net_flat_topo_sort": steiner.net_flat_topo_sort,
            "pin_fa": steiner.pin_fa,
            "node_x": timing_model.new_x,
            "node_y": timing_model.new_y,
        }
    else:
        owner = steiner_view
        source = "active_timing_propagation_patch_steiner"
        fields = {name: steiner_view[name] for name in (
            "net_flat_topo_sort_start", "net_flat_topo_sort", "pin_fa", "node_x", "node_y"
        )}
    cache = getattr(timing_model, "_steiner_array_cache", None)
    identity = tuple(id(value) for value in fields.values())
    if cache is not None and cache[0] is owner and cache[1] == identity:
        return cache[2]
    arrays = {
        "source": source,
        "starts": _numpy(fields["net_flat_topo_sort_start"], np.int32),
        "topo": _numpy(fields["net_flat_topo_sort"], np.int32),
        "pin_fa": _numpy(fields["pin_fa"], np.int32),
        "node_x": _numpy(fields["node_x"]),
        "node_y": _numpy(fields["node_y"]),
    }
    # Hold the source values so their ids cannot be reused while cached.
    timing_model._steiner_array_cache = (owner, identity, arrays, tuple(fields.values()))
    return arrays


def current_tree_topology_payload(pydb, timing_model, net_ids, *, steiner_view=None):
    """Copy requested CSR slices from the active TP Steiner tree."""
    arrays = steiner_arrays(timing_model, steiner_view)
    starts, topo = arrays["starts"], arrays["topo"]
    slices = [topo[int(starts[net]) : int(starts[net + 1])] for net in net_ids]
    nodes = np.concatenate(slices) if slices else np.empty(0, dtype=np.int32)
    return {
        "status": "ok",
        "topology_source": arrays["source"],
        "steiner_node_base": len(pydb.pin_names),
        "net_ids": list(net_ids),
        "net_flat_topo_sort_start": np.cumsum([0, *map(len, slices)], dtype=np.int32),
        "net_flat_topo_sort": nodes,
        "pin_fa": arrays["pin_fa"],
        # Dense coordinate arrays; native import keeps only ``nodes``.
        "node_x": arrays["node_x"],
        "node_y": arrays["node_y"],
    }


def steiner_payload(pydb, timing_model, *, provider="ecc", r_unit=0.914):
    if provider == "openroad_stt":
        return build_openroad_steiner_payload(pydb, r_unit=r_unit, dbu=getattr(pydb, "dbu", 0))
    if provider != "ecc":
        return {
            "status": "unsupported",
            "unsupported_reasons": [f"unknown_steiner_provider:{provider}"],
        }
    steiner = timing_model.steiner
    return {
        "status": "ok",
        "steiner_node_base": int(len(pydb.pin_names)),
        "net_ids": list(range(len(pydb.net_names))),
        "net_flat_topo_sort_start": _numpy(steiner.net_flat_topo_sort_start, np.int32),
        "net_flat_topo_sort": _numpy(steiner.net_flat_topo_sort, np.int32),
        "pin_fa": _numpy(steiner.pin_fa, np.int32),
        "flat_pin_to_start": _numpy(steiner.flat_pin_to_start, np.int32),
        "flat_pin_to": _numpy(steiner.flat_pin_to, np.int32),
        "node_x": _numpy(timing_model.new_x),
        "node_y": _numpy(timing_model.new_y),
        "edge_resistance": _numpy(timing_model.rc.flat_pin_to_res),
        "node_capacitance": _numpy(timing_model.rc.net_cap),
    }


__all__ = ["current_tree_topology_payload", "steiner_payload", "timing_payload"]
