from dataclasses import dataclass, field
import os
import time

import torch

from dreamplace.ops.steiner_topo import (
    build_edge_rc_by_net_from_rc_timing_topology,
    build_timing_rooted_tree_view,
)
from dreamplace.ops.net_subgraph_timing import build_relaxed_buffer_timing_payload

from .buffer_library import build_buffer_library_from_metadata
from .buffer_selection import resolve_fixed_buffer_selection
from .net_eligibility import clock_net_ids
from .candidates import sample_buffer_candidates
from .contract import BufferFamilyContract
from .optimization_state import (
    build_buffer_optimization_state,
    build_discrete_candidate_state,
)
from .segment_count_state import (
    build_packed_segment_count_state,
    build_segment_count_state,
)


@dataclass(frozen=True)
class BufferingStateBuildResult:
    status: str
    mode: str
    buffer_optimization_state: object = None
    buffer_relaxed_timing_payload: object = None
    candidate_count: int = 0
    net_count: int = 0
    reason: str | None = None
    metadata: dict = field(default_factory=dict)


def _get_data_value(data_collections, *names):
    for name in names:
        if hasattr(data_collections, name):
            return getattr(data_collections, name)
    return None


def _buffer_main_type_index(data_collections, params):
    for owner in (data_collections, params):
        value = getattr(owner, "buffer_main_type_index", None)
        if value is not None:
            return int(value)
    return None


def _infer_buffer_main_type_index(candidates):
    values = {
        int(candidate["buffer_main_type_index"])
        for candidate in candidates or []
        if candidate.get("buffer_main_type_index") is not None
    }
    if len(values) == 1:
        return next(iter(values))
    return None


def _pos_device(data_collections):
    pos = _get_data_value(data_collections, "pos")
    if isinstance(pos, (list, tuple)) and pos:
        return pos[0].device
    if torch.is_tensor(pos):
        return pos.device
    return None


def _buffer_timing_device(model, data_collections):
    """Prefer the active TimingPropagation device over a stale position view."""
    timing_op = getattr(
        getattr(model, "op_collections", None),
        "timing_propagation_op",
        None,
    )
    device = getattr(timing_op, "device", None)
    if device is not None:
        return torch.device(device)
    return _pos_device(data_collections)


def _metadata(data_collections, params):
    for owner in (data_collections, params):
        for name in ("buffering_metadata", "metadata", "pydb"):
            value = getattr(owner, name, None)
            if value is not None:
                return value
    return None


def _pydb(data_collections, params):
    for owner in (data_collections, params):
        value = getattr(owner, "pydb", None)
        if value is not None:
            return value
        value = getattr(owner, "rawdb", None)
        if value is not None:
            return value
    return None


def _live_timing_topology(data_collections):
    topology = getattr(data_collections, "buffering_timing_topology", None)
    if topology is not None:
        return topology
    required = (
        "net_flat_topo_sort",
        "net_flat_topo_sort_start",
        "pin_fa",
        "flat_pin_to",
        "flat_pin_from",
    )
    if any(getattr(data_collections, name, None) is None for name in required):
        return None
    node_x = getattr(data_collections, "buffering_timing_topology_node_x", None)
    node_y = getattr(data_collections, "buffering_timing_topology_node_y", None)
    if node_x is None or node_y is None:
        return None
    return {
        "topology_source": "live_timing_topology",
        "net_flat_topo_sort": getattr(data_collections, "net_flat_topo_sort"),
        "net_flat_topo_sort_start": getattr(data_collections, "net_flat_topo_sort_start"),
        "pin_fa": getattr(data_collections, "pin_fa"),
        "flat_pin_to": getattr(data_collections, "flat_pin_to"),
        "flat_pin_from": getattr(data_collections, "flat_pin_from"),
        "node_x": node_x,
        "node_y": node_y,
    }


def _build_edge_rc_from_live_topology(data_collections, params, topology):
    if topology is None:
        return None, None
    dbu = getattr(data_collections, "buffering_timing_topology_dbu", None)
    if dbu is None:
        pydb = _pydb(data_collections, params)
        dbu = getattr(pydb, "dbu", None) if pydb is not None else None
    if dbu is None:
        return None, {"status": "skipped", "reason": "missing_dbu"}
    r_unit = getattr(data_collections, "buffering_timing_topology_r_unit", None)
    c_unit = getattr(data_collections, "buffering_timing_topology_c_unit", None)
    scale_factor = getattr(
        data_collections,
        "buffering_timing_topology_scale_factor",
        getattr(params, "scale_factor", 1.0),
    )
    pydb = _pydb(data_collections, params)
    if pydb is not None:
        if r_unit is None:
            r_unit = getattr(pydb, "r_unit", None)
        if c_unit is None:
            c_unit = getattr(pydb, "c_unit", None)
    if r_unit is None or c_unit is None:
        return None, {"status": "skipped", "reason": "missing_rc_unit"}
    edge_rc_by_net, summary = build_edge_rc_by_net_from_rc_timing_topology(
        topology,
        dbu=dbu,
        scale_factor=scale_factor,
        r_unit=r_unit,
        c_unit=c_unit,
    )
    summary = dict(summary)
    summary["status"] = "ok"
    summary["topology_source"] = topology.get("topology_source", "live_timing_topology")
    return edge_rc_by_net, summary


def _current_timing_pin_cap_by_pin_id(model, data_collections):
    pin_rcap_base, source = _current_timing_pin_cap_tensor(model, data_collections)
    if pin_rcap_base is None:
        return None, None
    values = pin_rcap_base.cpu().tolist()
    return {
        int(pin_id): float(cap)
        for pin_id, cap in enumerate(values)
    }, source


def _current_timing_pin_cap_tensor(model, data_collections):
    pin_caps_op = getattr(model, "pin_caps_op", None)
    inst_libcell_offset = getattr(data_collections, "inst_libcell_offset", None)
    if pin_caps_op is None or inst_libcell_offset is None:
        return None, None
    try:
        _pin_cap_base, pin_rcap_base, _pin_fcap_base = pin_caps_op(
            inst_libcell_offset,
            data_collections,
        )
    except Exception:
        return None, None
    if not torch.is_tensor(pin_rcap_base):
        return None, None
    return pin_rcap_base.detach(), "current_pin_caps_op_rise_base"


def build_native_segment_count_packing_for_model(
    model,
    params,
    *,
    max_repeater_count,
    num_threads=None,
):
    """Build native Route-B topology tensors from existing flat data owners."""

    started_at = time.perf_counter()
    data_collections = getattr(model, "data_collections", None)
    if data_collections is None:
        return {"status": "unsupported", "reason": "missing_data_collections"}
    topology = _live_timing_topology(data_collections)
    if topology is None:
        return {"status": "unsupported", "reason": "missing_live_timing_topology"}

    placedb = getattr(model, "placedb", None)
    pydb = _pydb(data_collections, params)
    owners = tuple(owner for owner in (placedb, data_collections, pydb) if owner is not None)

    def first_value(*names):
        for owner in owners:
            for name in names:
                value = getattr(owner, name, None)
                if value is not None:
                    return value
        return None

    required = {
        "flat_net2pin": first_value("flat_net2pin_map"),
        "flat_net2pin_start": first_value("flat_net2pin_start_map"),
        "net2driver": first_value("net2driver_pin_map"),
        "pin2node": first_value("pin2node_map"),
        "net_flat_topo_sort": topology.get("net_flat_topo_sort"),
        "net_flat_topo_sort_start": topology.get("net_flat_topo_sort_start"),
        "pin_fa": topology.get("pin_fa"),
        "node_x": topology.get("node_x"),
        "node_y": topology.get("node_y"),
        "node_x_dbu": topology.get("node_x_dbu"),
        "node_y_dbu": topology.get("node_y_dbu"),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        return {
            "status": "unsupported",
            "reason": "missing_native_packer_inputs",
            "missing_inputs": missing,
        }

    pin_cap_started_at = time.perf_counter()
    pin_capacitance, pin_cap_source = _current_timing_pin_cap_tensor(
        model,
        data_collections,
    )
    pin_cap_eval_ms = (time.perf_counter() - pin_cap_started_at) * 1000.0
    if pin_capacitance is None:
        return {"status": "unsupported", "reason": "missing_current_pin_capacitance"}

    dbu = getattr(data_collections, "buffering_timing_topology_dbu", None)
    r_unit = getattr(data_collections, "buffering_timing_topology_r_unit", None)
    c_unit = getattr(data_collections, "buffering_timing_topology_c_unit", None)
    scale_factor = getattr(
        data_collections,
        "buffering_timing_topology_scale_factor",
        getattr(params, "scale_factor", None),
    )
    if dbu is None:
        dbu = first_value("dbu")
    if r_unit is None:
        r_unit = first_value("r_unit")
    if c_unit is None:
        c_unit = first_value("c_unit")
    missing_scalars = sorted(
        name
        for name, value in {
            "dbu": dbu,
            "scale_factor": scale_factor,
            "r_unit": r_unit,
            "c_unit": c_unit,
        }.items()
        if value is None
    )
    if missing_scalars:
        return {
            "status": "unsupported",
            "reason": "missing_native_packer_scalars",
            "missing_inputs": missing_scalars,
        }

    num_movable_nodes = first_value("num_movable_nodes")
    num_terminals = first_value("num_terminals")
    if num_movable_nodes is None or num_terminals is None:
        return {
            "status": "unsupported",
            "reason": "missing_native_packer_node_counts",
        }

    from dreamplace.ops.net_subgraph_timing.segment_count_native_packer import (
        pack_segment_count_topology,
    )

    excluded_clocks = sorted(clock_net_ids(*owners))
    if excluded_clocks:
        # The packer already excludes nets without a driver. Only its input
        # view changes; the original topology and ideal-clock STA stay intact.
        drivers = torch.as_tensor(required["net2driver"]).clone()
        drivers[excluded_clocks] = -1
        required["net2driver"] = drivers
    packed = pack_segment_count_topology(
        **required,
        pin_capacitance=pin_capacitance,
        num_movable_nodes=int(num_movable_nodes),
        num_terminals=int(num_terminals),
        dbu=float(dbu),
        scale_factor=float(scale_factor),
        r_unit=float(r_unit),
        c_unit=float(c_unit),
        max_repeater_count=int(max_repeater_count),
        num_threads=num_threads,
    )
    metadata = packed["metadata"]
    metadata["excluded_clock_net_ids"] = excluded_clocks
    metadata["native_pin_cap_eval_ms"] = float(pin_cap_eval_ms)
    metadata["native_total_ms"] = float((time.perf_counter() - started_at) * 1000.0)
    metadata["pin_cap_source"] = str(pin_cap_source)
    metadata["flat_input_owner"] = type(owners[0]).__name__ if owners else None
    return {
        "status": "ok",
        "reason": None,
        "packed": packed,
        "metadata": metadata,
    }


def build_native_candidate_packing_for_model(
    model,
    params,
    candidates,
    *,
    num_threads=None,
):
    """Build candidate-expanded timing tensors from flat live topology owners."""

    started_at = time.perf_counter()
    data_collections = getattr(model, "data_collections", None)
    if data_collections is None:
        return {"status": "unsupported", "reason": "missing_data_collections"}
    topology = _live_timing_topology(data_collections)
    if topology is None:
        return {"status": "unsupported", "reason": "missing_live_timing_topology"}

    placedb = getattr(model, "placedb", None)
    pydb = _pydb(data_collections, params)
    owners = tuple(owner for owner in (placedb, data_collections, pydb) if owner is not None)

    def first_value(*names):
        for owner in owners:
            for name in names:
                value = getattr(owner, name, None)
                if value is not None:
                    return value
        return None

    required = {
        "flat_net2pin": first_value("flat_net2pin_map"),
        "flat_net2pin_start": first_value("flat_net2pin_start_map"),
        "net2driver": first_value("net2driver_pin_map"),
        "pin2node": first_value("pin2node_map"),
        "net_flat_topo_sort": topology.get("net_flat_topo_sort"),
        "net_flat_topo_sort_start": topology.get("net_flat_topo_sort_start"),
        "pin_fa": topology.get("pin_fa"),
        "node_x": topology.get("node_x"),
        "node_y": topology.get("node_y"),
        "node_x_dbu": topology.get("node_x_dbu"),
        "node_y_dbu": topology.get("node_y_dbu"),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        return {
            "status": "unsupported",
            "reason": "missing_native_candidate_packer_inputs",
            "missing_inputs": missing,
        }

    pin_capacitance, pin_cap_source = _current_timing_pin_cap_tensor(
        model,
        data_collections,
    )
    if pin_capacitance is None:
        return {"status": "unsupported", "reason": "missing_current_pin_capacitance"}

    dbu = getattr(data_collections, "buffering_timing_topology_dbu", None)
    r_unit = getattr(data_collections, "buffering_timing_topology_r_unit", None)
    c_unit = getattr(data_collections, "buffering_timing_topology_c_unit", None)
    scale_factor = getattr(
        data_collections,
        "buffering_timing_topology_scale_factor",
        getattr(params, "scale_factor", None),
    )
    if dbu is None:
        dbu = first_value("dbu")
    if r_unit is None:
        r_unit = first_value("r_unit")
    if c_unit is None:
        c_unit = first_value("c_unit")
    missing_scalars = sorted(
        name
        for name, value in {
            "dbu": dbu,
            "scale_factor": scale_factor,
            "r_unit": r_unit,
            "c_unit": c_unit,
        }.items()
        if value is None
    )
    if missing_scalars:
        return {
            "status": "unsupported",
            "reason": "missing_native_candidate_packer_scalars",
            "missing_inputs": missing_scalars,
        }

    num_movable_nodes = first_value("num_movable_nodes")
    num_terminals = first_value("num_terminals")
    if num_movable_nodes is None or num_terminals is None:
        return {
            "status": "unsupported",
            "reason": "missing_native_candidate_packer_node_counts",
        }

    from dreamplace.ops.net_subgraph_timing.candidate_native_packer import (
        pack_candidate_topology,
    )

    packed = pack_candidate_topology(
        **required,
        candidates=candidates,
        pin_capacitance=pin_capacitance,
        num_movable_nodes=int(num_movable_nodes),
        num_terminals=int(num_terminals),
        dbu=float(dbu),
        scale_factor=float(scale_factor),
        r_unit=float(r_unit),
        c_unit=float(c_unit),
        num_threads=num_threads,
    )
    metadata = dict(packed["metadata"])
    metadata.update(
        {
            "native_total_ms": float((time.perf_counter() - started_at) * 1000.0),
            "pin_cap_source": str(pin_cap_source),
            "flat_input_owner": type(owners[0]).__name__ if owners else None,
        }
    )
    packed["metadata"] = metadata
    return {
        "status": "ok",
        "reason": None,
        "packed": packed,
        "metadata": metadata,
    }


def _native_selected_net_record_resolver(model, packed_result):
    placedb = getattr(model, "placedb", None)
    if placedb is None:
        return None
    flat_net2pin = getattr(placedb, "flat_net2pin_map", None)
    flat_net2pin_start = getattr(placedb, "flat_net2pin_start_map", None)
    net_names = getattr(placedb, "net_names", None)
    pin_names = getattr(placedb, "pin_names", None)
    if any(
        value is None
        for value in (flat_net2pin, flat_net2pin_start, net_names, pin_names)
    ):
        return None

    prepared = packed_result["prepared_timing_inputs"]
    geometry = packed_result["packed_segment_geometry"]
    prepared_net_ids = prepared["net_ids"].to(device="cpu", dtype=torch.long)
    affected_net_ids = geometry["net_ids"].to(device="cpu", dtype=torch.long)

    def decoded_name(values, index):
        if index < 0 or index >= len(values):
            return ""
        value = values[index]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def resolver(requested_net_ids):
        records = []
        for net_id in sorted(set(int(value) for value in requested_net_ids)):
            prepared_position = int(torch.searchsorted(prepared_net_ids, net_id).item())
            affected_position = int(torch.searchsorted(affected_net_ids, net_id).item())
            if (
                prepared_position >= int(prepared_net_ids.numel())
                or int(prepared_net_ids[prepared_position]) != net_id
                or affected_position >= int(affected_net_ids.numel())
                or int(affected_net_ids[affected_position]) != net_id
            ):
                continue
            node_begin = int(prepared["net_topo_start"][prepared_position])
            node_end = int(prepared["net_topo_start"][prepared_position + 1])
            edge_begin = int(prepared["net_edge_start"][prepared_position])
            edge_end = int(prepared["net_edge_start"][prepared_position + 1])
            sink_begin = int(prepared["net_sink_start"][prepared_position])
            sink_end = int(prepared["net_sink_start"][prepared_position + 1])
            segment_begin = int(geometry["net_segment_start"][affected_position])
            segment_end = int(geometry["net_segment_start"][affected_position + 1])

            topo_nodes = prepared["flat_topo_node_id"][node_begin:node_end].tolist()
            parent_nodes = prepared["edge_parent_node_id"][edge_begin:edge_end].tolist()
            child_nodes = prepared["edge_child_node_id"][edge_begin:edge_end].tolist()
            edge_r = prepared["edge_resistance"][edge_begin:edge_end].tolist()
            edge_c = prepared["edge_capacitance"][edge_begin:edge_end].tolist()
            children_by_node = {int(node_id): [] for node_id in topo_nodes}
            edge_rc = {}
            for parent, child, resistance, capacitance in zip(
                parent_nodes,
                child_nodes,
                edge_r,
                edge_c,
            ):
                children_by_node.setdefault(int(parent), []).append(int(child))
                children_by_node.setdefault(int(child), [])
                edge_rc[(int(parent), int(child))] = {
                    "r": float(resistance),
                    "c": float(capacitance),
                }

            coordinates_dbu = {}
            for offset in range(segment_begin, segment_end):
                parent = int(geometry["parent_node_id"][offset])
                child = int(geometry["child_node_id"][offset])
                coordinates_dbu[parent] = (
                    int(geometry["parent_x_dbu"][offset]),
                    int(geometry["parent_y_dbu"][offset]),
                )
                coordinates_dbu[child] = (
                    int(geometry["child_x_dbu"][offset]),
                    int(geometry["child_y_dbu"][offset]),
                )

            pin_begin = int(flat_net2pin_start[net_id])
            pin_end = int(flat_net2pin_start[net_id + 1])
            net_pin_ids = [int(value) for value in flat_net2pin[pin_begin:pin_end]]
            driver_pin_id = int(prepared["driver_pin_id"][prepared_position])
            sink_pin_ids = [
                int(value)
                for value in prepared["sink_node_id"][sink_begin:sink_end].tolist()
            ]
            pin_name_by_id = {
                int(pin_id): decoded_name(pin_names, int(pin_id))
                for pin_id in net_pin_ids
            }
            record = {
                "net_id": net_id,
                "net_name": decoded_name(net_names, net_id),
                "net_pin_ids": net_pin_ids,
                "driver_pin_id": driver_pin_id,
                "driver_pin_name": decoded_name(pin_names, driver_pin_id),
                "pin_name_by_id": pin_name_by_id,
                "sink_pin_ids": sink_pin_ids,
                "coordinates": dict(coordinates_dbu),
                "coordinates_dbu": coordinates_dbu,
                "rc_tree": {
                    "root_node_id": driver_pin_id,
                    "children_by_node": children_by_node,
                    "sink_nodes": sink_pin_ids,
                    "sink_pin_ids": sink_pin_ids,
                    "edge_rc": edge_rc,
                },
            }
            records.append(record)
        return records

    return resolver


def _build_buffer_library_from_data(data_collections, params, buffer_library, metadata):
    if buffer_library:
        return buffer_library, {
            "source": "data_collections_buffer_library",
            "status": "ok",
            "entry_count": len(buffer_library),
        }
    if metadata is None:
        return None, {"status": "skipped", "reason": "missing_metadata"}
    contract = BufferFamilyContract.from_buffering_params(
        metadata,
        params,
    ).build_artifact()
    if contract.get("status") != "ok":
        return None, {
            "status": "skipped",
            "reason": "buffer_family_contract_not_ok",
            "contract": contract,
        }
    library, summary = build_buffer_library_from_metadata(metadata, contract)
    summary["buffer_main_type_index"] = contract["buffer_main_type_index"]
    return library, summary


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _candidate_generation_policy(params):
    return str(getattr(params, "buffering_candidate_policy", "segment_only"))


def _candidate_original_node(candidate):
    for key in ("node_id", "candidate_node_id", "tree_node_id"):
        value = candidate.get(key)
        if value is not None:
            return int(value)
    return None


def _is_segment_candidate(candidate):
    if _candidate_original_node(candidate) is not None:
        return False
    parent = candidate.get("parent_node_id")
    children = candidate.get("child_node_ids") or []
    return parent is not None and len(children) == 1


def _normalize_candidate(candidate, *, synthetic_node_id=None):
    item = dict(candidate)
    original_node = _candidate_original_node(item)
    if original_node is not None:
        item.setdefault("node_id", original_node)
        item.setdefault("candidate_node_id", original_node)
    elif _is_segment_candidate(item) and item.get("synthetic_node_id") is None:
        if synthetic_node_id is None:
            raise ValueError("segment candidate requires synthetic_node_id")
        item["synthetic_node_id"] = int(synthetic_node_id)
    return item


def _bounded_nets(nets, *, net_cap):
    result = list(nets or [])
    if net_cap is None:
        return result
    return result[: max(0, int(net_cap))]


def _bounded_candidates(candidates, *, net_ids, candidate_cap):
    net_ids = {int(net_id) for net_id in net_ids}
    result = []
    next_synthetic_node_id = -1
    for candidate in candidates or []:
        if int(candidate.get("net_id", -1)) not in net_ids:
            continue
        item = _normalize_candidate(
            candidate,
            synthetic_node_id=next_synthetic_node_id,
        )
        if item.get("synthetic_node_id") is not None:
            next_synthetic_node_id = min(
                next_synthetic_node_id,
                int(item["synthetic_node_id"]),
            ) - 1
        if _candidate_original_node(item) is None and not _is_segment_candidate(item):
            continue
        if item.get("x_dbu") is None or item.get("y_dbu") is None:
            continue
        result.append(item)
    if candidate_cap is not None:
        result = result[: max(0, int(candidate_cap))]
    return result


def _per_size_tensors(candidates, buffer_library, *, dtype, shared_rows=False):
    legal_bsu = sorted(int(key) for key in buffer_library)
    if len(legal_bsu) < 2:
        raise ValueError("buffering state requires at least two legal buffer sizes")

    def table(field):
        row = torch.tensor(
            [[float(buffer_library[bsu][field]) for bsu in legal_bsu]],
            dtype=dtype,
        )
        if shared_rows:
            return row.expand(len(candidates), -1)
        return row.expand(len(candidates), -1).contiguous()

    return {
        "legal_bsu": legal_bsu,
        "per_size_input_cap": table("input_cap"),
        "per_size_delay": table("delay"),
        "per_size_output_slew": table("output_slew"),
    }


def _pin_name(net, pin_id, pin_name_by_id=None):
    pin_id = int(pin_id)
    pin_names = (
        dict(net.get("pin_name_by_id", {}) or {})
        if pin_name_by_id is None
        else pin_name_by_id
    )
    value = pin_names.get(pin_id)
    if value is None:
        value = pin_names.get(str(pin_id))
    if value is not None:
        return str(value)
    if pin_id == int(net.get("driver_pin_id", -1)):
        return str(net.get("driver_pin_name", ""))
    return ""


def _candidate_net_context(tree_view, net):
    return {
        "sink_pin_ids": {
            int(value) for value in list(tree_view.get("sink_pin_ids", []) or [])
        },
        "children_by_node": {
            int(node): tuple(int(child) for child in list(children or []))
            for node, children in dict(
                tree_view.get("children_by_node", {}) or {}
            ).items()
        },
        "pin_name_by_id": {
            int(pin_id): str(name)
            for pin_id, name in dict(net.get("pin_name_by_id", {}) or {}).items()
        },
        "downstream_pin_ids_by_seeds": {},
        "downstream_pin_names_by_ids": {},
    }


def _downstream_sink_pin_ids(context, candidate):
    sink_pin_ids = context["sink_pin_ids"]
    if not sink_pin_ids:
        return ()
    children_by_node = context["children_by_node"]
    seeds = [int(value) for value in list(candidate.get("child_node_ids", []) or [])]
    if not seeds and candidate.get("tree_node_id") is not None:
        seeds = [int(candidate["tree_node_id"])]
    if not seeds:
        seeds = tuple()
    else:
        seeds = tuple(sorted(set(seeds)))
    cached = context["downstream_pin_ids_by_seeds"].get(seeds)
    if cached is not None:
        return cached
    if not seeds:
        downstream = tuple(sorted(sink_pin_ids))
        context["downstream_pin_ids_by_seeds"][seeds] = downstream
        return downstream
    stack = list(seeds)
    visited = set()
    downstream = []
    while stack:
        node_id = int(stack.pop())
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in sink_pin_ids:
            downstream.append(node_id)
        stack.extend(children_by_node.get(node_id, []))
    downstream = tuple(sorted(set(downstream)))
    context["downstream_pin_ids_by_seeds"][seeds] = downstream
    return downstream


def _buffer_legal_table(buffer_library, legal_bsu):
    return {
        "legal_cell_ids": [
            int(buffer_library[int(bsu)].get("cell_id", int(bsu)))
            for bsu in legal_bsu
        ],
        "legal_master_names": [
            str(buffer_library[int(bsu)].get("master_name", ""))
            for bsu in legal_bsu
        ],
        "legal_size_values": [
            float(buffer_library[int(bsu)].get("legal_size_value", int(bsu)))
            for bsu in legal_bsu
        ],
    }


def _contextual_candidate(
    candidate,
    *,
    tree_view,
    net,
    context,
    global_candidate_id,
):
    item = dict(candidate)
    item["local_candidate_id"] = int(item["candidate_id"])
    item["candidate_id"] = int(global_candidate_id)
    item["driver_pin_id"] = int(tree_view.get("root_node_id", net.get("driver_pin_id", -1)))
    item["driver_pin_name"] = str(net.get("driver_pin_name", "")) or _pin_name(
        net,
        item["driver_pin_id"],
        context["pin_name_by_id"],
    )
    item["net_name"] = str(net.get("net_name", ""))
    downstream_pin_ids = _downstream_sink_pin_ids(context, item)
    item.setdefault("downstream_pin_ids", downstream_pin_ids)
    downstream_pin_names = context["downstream_pin_names_by_ids"].get(
        downstream_pin_ids
    )
    if downstream_pin_names is None:
        downstream_pin_names = tuple(
            name
            for name in (
                _pin_name(net, pin_id, context["pin_name_by_id"])
                for pin_id in downstream_pin_ids
            )
            if name
        )
        context["downstream_pin_names_by_ids"][downstream_pin_ids] = (
            downstream_pin_names
        )
    item.setdefault(
        "downstream_pin_names",
        downstream_pin_names,
    )
    item.setdefault(
        "downstream_pin_count",
        len(item.get("downstream_pin_ids", []) or []),
    )
    item.setdefault(
        "load_pin_id",
        item["downstream_pin_ids"][0] if item["downstream_pin_ids"] else None,
    )
    if item.get("load_pin_id") is not None:
        item.setdefault(
            "load_pin_name",
            _pin_name(net, item["load_pin_id"], context["pin_name_by_id"]),
        )
    item.setdefault("load_partition_source", "timing_rooted_tree_subtree")
    return item


def _generate_candidates_from_nets(
    nets,
    *,
    params,
    data_collections,
    buffer_main_type_index,
    max_candidates_per_segment,
):
    dbu = int(getattr(data_collections, "dbu", getattr(params, "dbu", 1000)) or 1000)
    candidates = []
    global_candidate_id = 0
    supported_tree_net_count = 0
    candidate_bearing_net_count = 0
    generation_totals = {
        "total_tree_edge_count": 0,
        "coordinate_supported_edge_count": 0,
        "nonzero_tree_edge_count": 0,
        "edge_with_candidate_count": 0,
        "edge_without_candidate_count": 0,
        "candidate_attempt_count": 0,
        "rounded_endpoint_skip_count": 0,
        "coordinate_dedup_count": 0,
    }
    for net in nets or []:
        tree_view = build_timing_rooted_tree_view(
            net_id=net["net_id"],
            net_pin_ids=net.get("net_pin_ids", []),
            driver_pin_id=net["driver_pin_id"],
            undirected_edges=net.get("undirected_edges", []),
            coordinates=net.get("coordinates_dbu", net.get("coordinates", {})),
            flat_first_pin_id=net.get("flat_first_pin_id"),
        )
        if tree_view.get("status") != "ok":
            continue
        supported_tree_net_count += 1
        local_candidates, local_summary = sample_buffer_candidates(
            tree_view,
            buffer_main_type_index=buffer_main_type_index,
            dbu=dbu,
            max_candidates_per_segment=max_candidates_per_segment,
            include_tree_node_candidates=_as_bool(
                getattr(params, "buffering_include_tree_node_candidates", 0)
            ),
            candidate_generation_policy=_candidate_generation_policy(params),
            return_summary=True,
        )
        for name in generation_totals:
            generation_totals[name] += int(local_summary[name])
        if local_candidates:
            candidate_bearing_net_count += 1
        context = _candidate_net_context(tree_view, net)
        for candidate in local_candidates:
            candidates.append(
                _contextual_candidate(
                    candidate,
                    tree_view=tree_view,
                    net=net,
                    context=context,
                    global_candidate_id=global_candidate_id,
                )
            )
            global_candidate_id += 1
    generation = {
        "candidate_generation_mode": "equal_count",
        "max_candidates_per_segment": int(max_candidates_per_segment),
        "supported_tree_net_count": int(supported_tree_net_count),
        "candidate_bearing_net_count": int(candidate_bearing_net_count),
        "candidate_net_coverage_ratio": (
            float(candidate_bearing_net_count) / float(supported_tree_net_count)
            if supported_tree_net_count
            else 0.0
        ),
        **generation_totals,
    }
    return candidates, generation


def _valid_buffer_main_type_index(value):
    if value is None:
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _input_diagnostics(common, *, candidates, include_candidates):
    buffer_library = common.get("buffer_library")
    buffer_main_type_index = common.get("buffer_main_type_index")
    buffer_library_summary = dict(common.get("buffer_library_summary") or {})
    contract = buffer_library_summary.get("contract")
    contract = dict(contract) if isinstance(contract, dict) else {}
    missing_inputs = []
    if not common.get("nets"):
        missing_inputs.append("nets")
    if include_candidates and not candidates:
        missing_inputs.append("candidates")
    if not buffer_library:
        missing_inputs.append("buffer_library")
    if not _valid_buffer_main_type_index(buffer_main_type_index):
        missing_inputs.append("buffer_main_type_index")
    return {
        "input_diagnostics": {
            "missing_inputs": missing_inputs,
            "net_count": int(len(common.get("nets") or [])),
            "candidate_count": int(len(candidates or [])),
            "buffer_library_size": int(len(buffer_library or {})),
            "buffer_main_type_index": (
                None
                if buffer_main_type_index is None
                else int(buffer_main_type_index)
            ),
            "net_source": str(common.get("net_source") or ""),
            "buffer_library_status": str(
                buffer_library_summary.get("status") or ""
            ),
            "buffer_library_reason": str(
                buffer_library_summary.get("reason") or ""
            ),
            "buffer_family_contract_status": str(contract.get("status") or ""),
            "buffer_family_contract_reasons": list(
                contract.get("unsupported_reasons") or []
            ),
        }
    }


def _build_result_skip(mode, reason, *, metadata=None, candidate_count=0, net_count=0):
    return BufferingStateBuildResult(
        status="skipped",
        mode=str(mode),
        candidate_count=int(candidate_count),
        net_count=int(net_count),
        reason=str(reason),
        metadata=dict(metadata or {}),
    )


def _record_runtime_stage(runtime_profile, name, started_at):
    if runtime_profile is not None:
        runtime_profile[name] = (time.perf_counter() - started_at) * 1000.0


def _resolve_common_inputs(
    data_collections,
    params,
    model=None,
    mode=None,
    runtime_profile=None,
):
    runtime_stage_started_at = time.perf_counter()
    nets = _get_data_value(data_collections, "buffering_nets", "buffer_nets")
    _record_runtime_stage(
        runtime_profile,
        "common_existing_nets_lookup_ms",
        runtime_stage_started_at,
    )
    net_source = "data_collections"
    runtime_stage_started_at = time.perf_counter()
    metadata = _metadata(data_collections, params)
    _record_runtime_stage(
        runtime_profile,
        "common_metadata_lookup_ms",
        runtime_stage_started_at,
    )
    edge_rc_summary = None
    if not nets:
        runtime_stage_started_at = time.perf_counter()
        pydb = _pydb(data_collections, params)
        _record_runtime_stage(
            runtime_profile,
            "common_pydb_lookup_ms",
            runtime_stage_started_at,
        )
        if pydb is not None:
            from .real_design_adapter import build_buffering_nets_from_pydb

            runtime_stage_started_at = time.perf_counter()
            topology = (
                _live_timing_topology(data_collections)
                if str(mode) in ("candidate", "segment")
                else None
            )
            _record_runtime_stage(
                runtime_profile,
                "common_live_topology_payload_ms",
                runtime_stage_started_at,
            )
            runtime_stage_started_at = time.perf_counter()
            edge_rc_by_net, edge_rc_summary = _build_edge_rc_from_live_topology(
                data_collections,
                params,
                topology,
            )
            _record_runtime_stage(
                runtime_profile,
                "common_edge_rc_build_ms",
                runtime_stage_started_at,
            )
            runtime_stage_started_at = time.perf_counter()
            pin_cap_by_pin_id, pin_cap_source = _current_timing_pin_cap_by_pin_id(
                model,
                data_collections,
            )
            _record_runtime_stage(
                runtime_profile,
                "common_pin_cap_build_ms",
                runtime_stage_started_at,
            )
            runtime_stage_started_at = time.perf_counter()
            if topology is not None and edge_rc_by_net:
                nets, adapter_summary = build_buffering_nets_from_pydb(
                    pydb,
                    topology=topology,
                    edge_rc_by_net=edge_rc_by_net,
                    pin_cap_by_pin_id=pin_cap_by_pin_id,
                    pin_cap_source=pin_cap_source,
                    runtime_profile=runtime_profile,
                )
                net_source = "live_timing_topology"
            else:
                nets, adapter_summary = build_buffering_nets_from_pydb(
                    pydb,
                    pin_cap_by_pin_id=pin_cap_by_pin_id,
                    pin_cap_source=pin_cap_source,
                    runtime_profile=runtime_profile,
                )
                net_source = "pydb"
            _record_runtime_stage(
                runtime_profile,
                "common_build_buffering_nets_ms",
                runtime_stage_started_at,
            )
            if metadata is None:
                metadata = pydb
        else:
            adapter_summary = None
    else:
        adapter_summary = None
    runtime_stage_started_at = time.perf_counter()
    candidates = _get_data_value(
        data_collections,
        "buffering_candidate_rows",
        "buffering_candidates",
        "buffer_candidates",
    )
    _record_runtime_stage(
        runtime_profile,
        "common_candidate_rows_lookup_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_library = _get_data_value(
        data_collections,
        "buffering_buffer_library",
        "buffer_library",
    )
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_library_lookup_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_main_type_index = _buffer_main_type_index(data_collections, params)
    if buffer_main_type_index is None:
        buffer_main_type_index = _infer_buffer_main_type_index(candidates)
    if buffer_main_type_index is None and metadata is not None:
        buffer_main_type_index = getattr(metadata, "buffer_main_type_index", None)
        if buffer_main_type_index is not None:
            buffer_main_type_index = int(buffer_main_type_index)
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_main_type_resolve_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_library, buffer_library_summary = _build_buffer_library_from_data(
        data_collections,
        params,
        buffer_library,
        metadata,
    )
    if "buffer_main_type_index" in buffer_library_summary:
        buffer_main_type_index = buffer_library_summary["buffer_main_type_index"]
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_library_contract_ms",
        runtime_stage_started_at,
    )
    return {
        "nets": nets,
        "net_source": net_source,
        "metadata": metadata,
        "adapter_summary": adapter_summary,
        "edge_rc_summary": edge_rc_summary,
        "candidates": candidates,
        "buffer_library": buffer_library,
        "buffer_main_type_index": buffer_main_type_index,
        "buffer_library_summary": buffer_library_summary,
    }


def _resolve_native_segment_common_inputs(data_collections, params, runtime_profile=None):
    runtime_stage_started_at = time.perf_counter()
    metadata = _metadata(data_collections, params)
    _record_runtime_stage(
        runtime_profile,
        "common_metadata_lookup_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_library = _get_data_value(
        data_collections,
        "buffering_buffer_library",
        "buffer_library",
    )
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_library_lookup_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_main_type_index = _buffer_main_type_index(data_collections, params)
    if buffer_main_type_index is None and metadata is not None:
        value = getattr(metadata, "buffer_main_type_index", None)
        if value is not None:
            buffer_main_type_index = int(value)
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_main_type_resolve_ms",
        runtime_stage_started_at,
    )
    runtime_stage_started_at = time.perf_counter()
    buffer_library, buffer_library_summary = _build_buffer_library_from_data(
        data_collections,
        params,
        buffer_library,
        metadata,
    )
    _record_runtime_stage(
        runtime_profile,
        "common_buffer_library_contract_ms",
        runtime_stage_started_at,
    )
    return {
        "nets": (),
        "net_source": "native_live_timing_topology",
        "metadata": metadata,
        "adapter_summary": None,
        "edge_rc_summary": None,
        "candidates": None,
        "buffer_library": buffer_library,
        "buffer_main_type_index": buffer_main_type_index,
        "buffer_library_summary": buffer_library_summary,
    }


def _native_segment_builder_fallback_reason(config, params):
    if str(getattr(config, "mode", "")) != "segment":
        return "mode_is_not_segment"
    if str(getattr(config, "segment_strategy", "")) != "discrete_net_gradient":
        return "strategy_is_not_discrete_net_gradient"
    timing_backend = str(
        getattr(params, "buffering_segment_count_timing_backend", "") or ""
    )
    if not timing_backend.startswith("cpp_cuda_"):
        return "timing_backend_is_not_cpp_cuda"
    if bool(getattr(config, "segment_capacity_enabled", False)):
        return "segment_capacity_enabled"
    blocked_environment = sorted(
        name
        for name, value in os.environ.items()
        if value
        and name.startswith("AIMP_BUFFERING_SEGMENT_")
        and ("PROBE" in name or "TARGET_NET_IDS" in name)
    )
    if blocked_environment:
        return "debug_or_bounded_environment:" + ",".join(blocked_environment)
    return None


def _build_candidate_buffering_state_for_model(
    *,
    model,
    mode,
    data_collections,
    params,
    config,
    common,
):
    nets = common["nets"]
    candidates = common["candidates"]
    buffer_library = common["buffer_library"]
    buffer_main_type_index = common["buffer_main_type_index"]
    adapter_summary = common["adapter_summary"]
    net_source = common["net_source"]
    buffer_library_summary = common["buffer_library_summary"]
    candidate_source = "model_attached_candidates"
    generation = None
    if not candidates and nets and buffer_library and buffer_main_type_index is not None:
        candidates, generation = _generate_candidates_from_nets(
            nets,
            params=params,
            data_collections=data_collections,
            buffer_main_type_index=buffer_main_type_index,
            max_candidates_per_segment=config.max_repeaters_per_segment,
        )
        candidate_source = "generated_from_nets"
    diagnostics = _input_diagnostics(
        common,
        candidates=candidates,
        include_candidates=True,
    )
    if diagnostics["input_diagnostics"]["missing_inputs"]:
        return _build_result_skip(
            mode,
            "missing_buffering_inputs",
            metadata=diagnostics,
            candidate_count=len(candidates or []),
            net_count=len(nets or []),
        )

    dtype = torch.float32
    selected_nets = _bounded_nets(nets, net_cap=None)
    net_ids = [int(net.get("net_id", index)) for index, net in enumerate(selected_nets)]
    selected_candidates = _bounded_candidates(
        candidates,
        net_ids=net_ids,
        candidate_cap=None,
    )
    if not selected_nets or not selected_candidates:
        return _build_result_skip(mode, "no_supported_candidates")
    selected_candidate_count = len(selected_candidates)
    selected_net_count = len(selected_nets)

    candidate_strategy = str(
        getattr(config, "candidate_strategy", "continuous") or "continuous"
    )
    per_size = _per_size_tensors(
        selected_candidates,
        buffer_library,
        dtype=dtype,
        shared_rows=True,
    )
    fixed_bsu_index, fixed_buffer_master, fixed_buffer_source = (
        resolve_fixed_buffer_selection(params, config, buffer_library)
    )
    state_device = _buffer_timing_device(model, data_collections)
    if candidate_strategy == "discrete_net_gradient":
        if fixed_bsu_index is None:
            raise ValueError(
                "candidate discrete_net_gradient requires a fixed buffer"
            )
        state = build_discrete_candidate_state(
            selected_candidates,
            buffer_main_type_index=buffer_main_type_index,
            legal_buffer_count=len(per_size["legal_bsu"]),
            fixed_bsu_index=fixed_bsu_index,
            dtype=dtype,
            device=state_device,
        )
    else:
        state = build_buffer_optimization_state(
            selected_candidates,
            buffer_main_type_index=buffer_main_type_index,
            legal_buffer_count=len(per_size["legal_bsu"]),
            initial_bu_logit=float(
                getattr(params, "buffering_initial_bu_logit", 0.0)
            ),
            initial_bsu_index=float(
                getattr(params, "buffering_initial_bsu_index", 0.5)
            ),
            dtype=dtype,
            device=state_device,
        )
    native_packing = None
    native_packing_reason = None
    try:
        native_packing = build_native_candidate_packing_for_model(
            model,
            params,
            selected_candidates,
        )
    except ImportError as exc:
        native_packing = None
        native_packing_reason = "native_extension_unavailable:" + str(exc)
    if native_packing is not None and native_packing.get("status") == "ok":
        native_prepared = native_packing["packed"]["prepared_timing_inputs"]
        native_candidate_node_id = native_prepared["candidate_node_id"]
        if int(native_candidate_node_id.numel()) != selected_candidate_count:
            raise ValueError(
                "native candidate packer returned an inconsistent candidate count"
            )
        state.candidate_node_id = native_candidate_node_id.to(
            device=state.candidate_node_id.device,
            dtype=state.candidate_node_id.dtype,
        )
    elif native_packing is not None:
        native_packing_reason = str(
            native_packing.get("reason") or "native_candidate_packer_unavailable"
        )
        native_packing = None
    try:
        payload = build_relaxed_buffer_timing_payload(
            selected_nets,
            selected_candidates,
            buffer_state=state,
            per_size_input_cap=per_size["per_size_input_cap"],
            per_size_delay=per_size["per_size_delay"],
            per_size_output_slew=per_size["per_size_output_slew"],
            default_driver_arrival=float(
                getattr(params, "buffering_default_driver_arrival", 0.0)
            ),
            default_driver_slew=float(
                getattr(params, "buffering_default_driver_slew", 0.1)
            ),
            coordinate_source=str(
                getattr(params, "buffering_coordinate_source", "fixed_canonical_builder")
            ),
            dtype=dtype,
            compact_runtime_metadata=True,
            prebuilt_inputs=(
                native_packing["packed"]["prepared_timing_inputs"]
                if native_packing is not None
                else None
            ),
        )
        sink_pin_id = payload["sink_pin_id"]
        if sink_pin_id.numel():
            payload["num_pins"] = int(sink_pin_id.max().cpu().item()) + 1
        metadata = dict(payload.pop("metadata", {}) or {})
        omitted_runtime_metadata = {}
        for field in (
            "candidate_ids",
            "candidate_net_id",
            "compact_node_to_net_id",
            "original_node_to_compact",
            "original_node_to_compact_by_net",
            "synthetic_segment_splits",
        ):
            value = metadata.pop(field, None)
            if value is not None:
                omitted_runtime_metadata[field] = (
                    len(value) if hasattr(value, "__len__") else None
                )
    finally:
        selected_candidates = None
        candidates = None
    metadata.update(
        {
            "state_kind": "candidate",
            "state_source": "data_collections_buffering_inputs",
            "candidate_source": candidate_source,
            "buffer_library_source": buffer_library_summary.get("source"),
            "net_source": net_source,
            "candidate_count": int(selected_candidate_count),
            "net_count": int(selected_net_count),
            "legal_size_count": int(len(per_size["legal_bsu"])),
            "candidate_strategy": candidate_strategy,
            "candidate_topology_builder_backend_requested": "native_auto",
            "candidate_topology_builder_backend_used": (
                "native_cpp" if native_packing is not None else "python"
            ),
            "candidate_topology_builder_fallback_reason": native_packing_reason,
            "fixed_bsu_index": (
                fixed_bsu_index
                if candidate_strategy == "discrete_net_gradient"
                else None
            ),
            "fixed_buffer_master": (
                fixed_buffer_master
                if candidate_strategy == "discrete_net_gradient"
                else ""
            ),
            "fixed_buffer_selection_source": (
                fixed_buffer_source
                if candidate_strategy == "discrete_net_gradient"
                else "optimized"
            ),
            "omitted_runtime_metadata": omitted_runtime_metadata,
            "python_gc_suspended_until_projection": False,
        }
    )
    payload["buffer_legal_table"] = _buffer_legal_table(
        buffer_library,
        per_size["legal_bsu"],
    )
    if adapter_summary is not None:
        metadata["net_adapter_status"] = adapter_summary.get("status")
        metadata["net_adapter_built_net_count"] = adapter_summary.get("built_net_count")
        metadata["net_adapter_rc_source"] = adapter_summary.get("rc_source")
    if common.get("edge_rc_summary") is not None:
        metadata["edge_rc_source"] = common["edge_rc_summary"].get("rc_source")
        metadata["edge_rc_map_net_count"] = common["edge_rc_summary"].get("edge_rc_map_net_count")
        metadata["edge_rc_dbu"] = common["edge_rc_summary"].get("dbu")
        metadata["edge_rc_scale_factor"] = common["edge_rc_summary"].get("scale_factor")
        metadata["edge_rc_length_denominator"] = common["edge_rc_summary"].get("length_denominator")
    if generation is not None:
        metadata.update(generation)
    if native_packing is not None:
        native_metadata = dict(native_packing["metadata"])
        metadata["candidate_native_packer_metadata"] = native_metadata
    payload["metadata"] = metadata
    return BufferingStateBuildResult(
        status="ok",
        mode=mode,
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        candidate_count=selected_candidate_count,
        net_count=selected_net_count,
        metadata=metadata,
    )


def _build_segment_count_buffering_state_for_model(
    *,
    model,
    mode,
    data_collections,
    params,
    config,
    common,
    runtime_profile=None,
    native_packing=None,
    native_fallback_reason=None,
):
    runtime_profile_enabled = bool(
        getattr(params, "buffering_runtime_profile", False)
    )
    if runtime_profile is None and runtime_profile_enabled:
        runtime_profile = {}

    def record_runtime_stage(name, started_at):
        _record_runtime_stage(runtime_profile, name, started_at)

    nets = common["nets"]
    buffer_library = common["buffer_library"]
    buffer_main_type_index = common["buffer_main_type_index"]
    adapter_summary = common["adapter_summary"]
    net_source = common["net_source"]
    buffer_library_summary = common["buffer_library_summary"]
    diagnostics = _input_diagnostics(common, candidates=None, include_candidates=False)
    missing_inputs = list(diagnostics["input_diagnostics"]["missing_inputs"])
    if native_packing is not None:
        missing_inputs = [name for name in missing_inputs if name != "nets"]
    if missing_inputs:
        return _build_result_skip(
            mode,
            "missing_buffering_inputs",
            metadata=diagnostics,
            net_count=len(nets or []),
        )

    dtype = torch.float32
    selected_nets = _bounded_nets(nets, net_cap=None)
    if native_packing is None and not selected_nets:
        return _build_result_skip(mode, "no_supported_segments")
    runtime_stage_started_at = time.perf_counter()
    placeholder_candidates = [
        {"candidate_id": 0, "buffer_main_type_index": int(buffer_main_type_index)}
    ]
    per_size_template = _per_size_tensors(placeholder_candidates, buffer_library, dtype=dtype)
    record_runtime_stage(
        "segment_size_template_build_ms",
        runtime_stage_started_at,
    )
    fixed_bsu_index, fixed_buffer_master, fixed_buffer_source = (
        resolve_fixed_buffer_selection(params, config, buffer_library)
    )
    initial_bsu_index = (
        float(fixed_bsu_index)
        if fixed_bsu_index is not None
        else float(getattr(params, "buffering_initial_bsu_index", 0.5))
    )
    runtime_stage_started_at = time.perf_counter()
    state_builder_kwargs = {
        "buffer_main_type_index": buffer_main_type_index,
        "legal_buffer_count": len(per_size_template["legal_bsu"]),
        "max_repeater_count": int(
            getattr(params, "buffering_max_repeaters_per_segment", 3)
        ),
        "initial_z": float(getattr(
            params, "buffering_segment_count_z_init",
            0.0 if getattr(config, "segment_strategy", "continuous") == "discrete_net_gradient"
            else 0.05,
        )),
        "initial_bsu_index": initial_bsu_index,
        "fixed_bsu_index": fixed_bsu_index,
        "dtype": dtype,
        "device": _buffer_timing_device(model, data_collections),
    }
    state = (
        build_segment_count_state(
            selected_nets,
            retained_edges=getattr(data_collections, "buffer_retained_segment_edges", ()),
            **state_builder_kwargs,
        )
        if native_packing is None
        else build_packed_segment_count_state(
            native_packing["packed"],
            net_record_resolver=native_packing["net_record_resolver"],
            **state_builder_kwargs,
        )
    )
    record_runtime_stage("segment_state_build_ms", runtime_stage_started_at)
    segment_count = int(state.z_param.numel())
    if segment_count <= 0:
        return _build_result_skip(mode, "no_supported_segments")
    runtime_stage_started_at = time.perf_counter()
    timing_backend = str(
        getattr(
            params,
            "buffering_segment_count_timing_backend",
            "prepared_python",
        )
        or "prepared_python"
    )
    shared_size_table = timing_backend.startswith("cpp_cuda_")
    if shared_size_table:
        # Segment rows share one Liberty-derived table.  The CUDA provider
        # expands this row on-device for gather semantics without materializing
        # a repeated [segment_count, legal_buffer_count] CPU/GPU tensor.
        per_size = per_size_template
    else:
        segment_candidates = [
            {
                "candidate_id": index,
                "buffer_main_type_index": int(buffer_main_type_index),
            }
            for index in range(segment_count)
        ]
        per_size = _per_size_tensors(segment_candidates, buffer_library, dtype=dtype)
    record_runtime_stage("per_segment_size_table_build_ms", runtime_stage_started_at)

    runtime_stage_started_at = time.perf_counter()
    metadata = {
        "state_kind": "segment_count",
        "state_source": "data_collections_buffering_inputs",
        "buffer_library_source": buffer_library_summary.get("source"),
        "net_source": net_source,
        "segment_count": segment_count,
        "net_count": int(
            len(selected_nets)
            if native_packing is None
            else native_packing["metadata"]["valid_net_count"]
        ),
        "affected_net_count": int(state.summary.get("affected_net_count", 0)),
        "legal_size_count": int(len(per_size["legal_bsu"])),
        "full_design_scope": bool(state.summary.get("full_design_scope", False)),
        "max_repeater_count": int(state.max_repeater_count),
        "fixed_bsu_index": fixed_bsu_index,
        "fixed_buffer_master": fixed_buffer_master,
        "fixed_buffer_selection_source": fixed_buffer_source,
        "buffer_size_optimization": fixed_bsu_index is None,
        "segment_size_table_layout": (
            "shared_library_row" if shared_size_table else "dense_per_segment"
        ),
        "segment_state_builder_backend_requested": "native_auto",
        "segment_state_builder_backend_used": (
            "native_cpp" if native_packing is not None else "python"
        ),
        "segment_state_builder_fallback_reason": native_fallback_reason,
        "native_packer_thread_count": (
            int(native_packing["metadata"].get("thread_count", 0))
            if native_packing is not None
            else 0
        ),
    }
    if native_packing is not None:
        native_metadata = dict(native_packing["metadata"])
        if runtime_profile is not None:
            runtime_profile.update({
                "native_input_export_ms": float(
                    native_metadata.get("native_input_export_ms", 0.0)
                )
                + float(native_metadata.get("native_pin_cap_eval_ms", 0.0)),
                "native_validate_count_ms": float(
                    native_metadata.get("validate_count_ms", 0.0)
                ),
                "native_prefix_sum_ms": float(
                    native_metadata.get("prefix_sum_ms", 0.0)
                ),
                "native_allocate_ms": float(native_metadata.get("allocate_ms", 0.0)),
                "native_fill_ms": float(native_metadata.get("fill_ms", 0.0)),
                "native_return_attach_ms": float(
                    native_metadata.get("native_return_attach_ms", 0.0)
                ),
                "native_packer_total_ms": float(
                    native_metadata.get("native_total_ms", 0.0)
                ),
            })
    if runtime_profile is not None:
        metadata["state_build_runtime_profile"] = runtime_profile
    if adapter_summary is not None:
        metadata["net_adapter_status"] = adapter_summary.get("status")
        metadata["net_adapter_built_net_count"] = adapter_summary.get("built_net_count")
        metadata["net_adapter_rc_source"] = adapter_summary.get("rc_source")
    if common.get("edge_rc_summary") is not None:
        metadata["edge_rc_source"] = common["edge_rc_summary"].get("rc_source")
        metadata["edge_rc_map_net_count"] = common["edge_rc_summary"].get("edge_rc_map_net_count")
        metadata["edge_rc_dbu"] = common["edge_rc_summary"].get("dbu")
        metadata["edge_rc_scale_factor"] = common["edge_rc_summary"].get("scale_factor")
        metadata["edge_rc_length_denominator"] = common["edge_rc_summary"].get("length_denominator")
    record_runtime_stage("segment_metadata_build_ms", runtime_stage_started_at)
    runtime_stage_started_at = time.perf_counter()
    payload = {
        "metadata": dict(metadata),
        "nets": tuple(selected_nets),
        "prepared_timing_inputs": state.prepared_timing_inputs,
        "per_size_input_cap": per_size["per_size_input_cap"],
        "per_size_delay": per_size["per_size_delay"],
        "per_size_output_slew": per_size["per_size_output_slew"],
        "buffer_legal_table": _buffer_legal_table(
            buffer_library,
            per_size["legal_bsu"],
        ),
    }
    record_runtime_stage("segment_relaxed_timing_payload_build_ms", runtime_stage_started_at)
    if runtime_profile is not None:
        state_build_stage_names = (
            "segment_size_template_build_ms",
            "segment_state_build_ms",
            "per_segment_size_table_build_ms",
            "segment_metadata_build_ms",
            "segment_relaxed_timing_payload_build_ms",
        )
        state_build_total_ms = sum(
            float(runtime_profile.get(name, 0.0) or 0.0)
            for name in state_build_stage_names
        )
        runtime_profile["segment_state_build_accounted_ms"] = state_build_total_ms
        runtime_profile["segment_size_table_elements"] = int(
            sum(int(table.numel()) for table in (
                per_size["per_size_input_cap"],
                per_size["per_size_delay"],
                per_size["per_size_output_slew"],
            ))
        )
        runtime_profile["segment_size_table_bytes"] = int(
            sum(int(table.numel() * table.element_size()) for table in (
                per_size["per_size_input_cap"],
                per_size["per_size_delay"],
                per_size["per_size_output_slew"],
            ))
        )
        runtime_profile["segment_logical_size_table_elements"] = int(
            segment_count * len(per_size["legal_bsu"]) * 3
        )
        runtime_profile["segment_rc_tree_node_count"] = int(
            native_packing["metadata"]["node_count"]
            if native_packing is not None
            else sum(
                len((net.get("rc_tree") or {}).get("children_by_node", {}) or {})
                for net in selected_nets
            )
        )
        runtime_profile["segment_rc_tree_edge_count"] = int(
            native_packing["metadata"]["edge_count"]
            if native_packing is not None
            else sum(
                len((net.get("rc_tree") or {}).get("edge_rc", {}) or {})
                for net in selected_nets
            )
        )
    return BufferingStateBuildResult(
        status="ok",
        mode=mode,
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        candidate_count=0,
        net_count=int(metadata["net_count"]),
        metadata=metadata,
    )


def build_buffering_state_for_model(model, params, config):
    """Build canonical relaxed buffer state from model-attached buffer inputs."""

    mode = str(getattr(config, "mode", getattr(params, "buffering_mode", "candidate")))
    data_collections = getattr(model, "data_collections", None)
    if data_collections is None:
        return _build_result_skip(mode, "missing_data_collections")

    runtime_profile_enabled = bool(
        getattr(params, "buffering_runtime_profile", False)
    )
    runtime_profile = {} if runtime_profile_enabled else None
    runtime_stage_started_at = time.perf_counter()
    native_packing = None
    native_fallback_reason = None
    if mode == "segment":
        native_fallback_reason = _native_segment_builder_fallback_reason(config, params)
        if native_fallback_reason is None:
            common = _resolve_native_segment_common_inputs(
                data_collections,
                params,
                runtime_profile=runtime_profile,
            )
            native_packing = build_native_segment_count_packing_for_model(
                model,
                params,
                max_repeater_count=int(config.max_repeaters_per_segment),
            )
            if native_packing.get("status") == "ok":
                native_packing["net_record_resolver"] = (
                    _native_selected_net_record_resolver(
                        model,
                        native_packing["packed"],
                    )
                )
                if native_packing["net_record_resolver"] is None:
                    native_packing = {
                        "status": "unsupported",
                        "reason": "missing_selected_net_record_owner",
                    }
            if native_packing.get("status") != "ok":
                native_fallback_reason = "native_input_unavailable:" + str(
                    native_packing.get("reason") or "unknown"
                )
                native_packing = None
                common = _resolve_common_inputs(
                    data_collections,
                    params,
                    model=model,
                    mode=mode,
                    runtime_profile=runtime_profile,
                )
        else:
            common = _resolve_common_inputs(
                data_collections,
                params,
                model=model,
                mode=mode,
                runtime_profile=runtime_profile,
            )
    else:
        common = _resolve_common_inputs(
            data_collections,
            params,
            model=model,
            mode=mode,
            runtime_profile=runtime_profile,
        )
    common_inputs_ms = (time.perf_counter() - runtime_stage_started_at) * 1000.0
    if mode == "segment":
        result = _build_segment_count_buffering_state_for_model(
            model=model,
            mode=mode,
            data_collections=data_collections,
            params=params,
            config=config,
            common=common,
            runtime_profile=runtime_profile,
            native_packing=native_packing,
            native_fallback_reason=native_fallback_reason,
        )
        if runtime_profile_enabled:
            runtime_profile["common_inputs_build_ms"] = common_inputs_ms
            common_stage_names = (
                "common_existing_nets_lookup_ms",
                "common_metadata_lookup_ms",
                "common_pydb_lookup_ms",
                "common_live_topology_payload_ms",
                "common_edge_rc_build_ms",
                "common_pin_cap_build_ms",
                "common_build_buffering_nets_ms",
                "common_candidate_rows_lookup_ms",
                "common_buffer_library_lookup_ms",
                "common_buffer_main_type_resolve_ms",
                "common_buffer_library_contract_ms",
                "native_packer_total_ms",
            )
            common_accounted_ms = sum(
                float(runtime_profile.get(name, 0.0) or 0.0)
                for name in common_stage_names
            )
            runtime_profile["common_inputs_accounted_ms"] = common_accounted_ms
            runtime_profile["common_inputs_unattributed_ms"] = max(
                0.0,
                float(common_inputs_ms) - common_accounted_ms,
            )
            result.metadata["state_build_runtime_profile"] = runtime_profile
            payload = result.buffer_relaxed_timing_payload
            if isinstance(payload, dict):
                payload_metadata = payload.setdefault("metadata", {})
                payload_metadata["state_build_runtime_profile"] = runtime_profile
        return result
    return _build_candidate_buffering_state_for_model(
        model=model,
        mode=mode,
        data_collections=data_collections,
        params=params,
        config=config,
        common=common,
    )
