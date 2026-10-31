"""Session-owned multi-slot timing patch for the Python TimingPropagation path.

The native rebuffer patch owns graph identity, active partitions, RC edges, and
the affected schedule. This module converts those native arrays to the
existing DreamPlace tensors and invokes ``TimingPropagation.forward_affected``.
Cross-net parent deltas remain a one-new-net fast path; same-net chains use the
multi-slot graph and retain a current partition payload for native planning.
"""

import copy
import itertools
from types import SimpleNamespace

import numpy as np
import torch

from dreamplace.ops.rebuffer.timing_model_payload import TimingSnapshot, timing_pin_slack

from .incremental_timing import IncrementalTimingState
from .timing_carrier import CommittedTimingPyDB


class _PyDBOverlayView:
    """Writable view over a committed carrier without invoking ``copy.copy``."""

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)


def _tensor(value, *, dtype):
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def _expanded(base, pin_count, vertex_count, slot_count):
    """Map base RC vertices into the multi-slot synthetic pin domain."""

    result = torch.zeros(
        vertex_count + 2 * int(slot_count), dtype=base.dtype, device=base.device
    )
    result[:pin_count] = base[:pin_count]
    if vertex_count > pin_count:
        result[pin_count + 2 * int(slot_count) :] = base[pin_count:vertex_count]
    return result


def _patch_rc(owner, patch, action):
    from dreamplace.ops.rebuffer import rebuffer_cpp

    pin_count = len(owner.pydb.pin_names)
    vertex_count = int(owner.new_x.numel())
    slot_count = int(
        patch.get("buffer_count", len(patch.get("slot_pin_ids", ())) // 2)
    )
    rc = patch["rc"]
    native = rebuffer_cpp.evaluate_incremental_rc(
        rc,
        np.ascontiguousarray(np.asarray(owner.lib_pin_caps, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(owner.lib_pin_rcaps, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(owner.lib_pin_fcaps, dtype=np.float64)),
        patch["buffer_caps"],
        pin_count,
    )
    affected_global = np.asarray(native["affected_nodes"], dtype=np.int64)
    affected = torch.from_numpy(affected_global)
    output = {"loads": {}, "delays": {}, "impulses": {}}
    native_modes = {"rise": "rise", "fall": "fall"}
    for mode, native_mode in native_modes.items():
        for field in output:
            base = getattr(owner, field)[mode]
            merged = _expanded(base, pin_count, vertex_count, slot_count)
            merged[affected] = torch.as_tensor(
                np.asarray(native[f"{field}_{native_mode}"], dtype=np.float64)
            )
            output[field][mode] = merged
    output["generic"] = {
        "loads": torch.as_tensor(np.asarray(native["loads_generic"], dtype=np.float64)),
        "delays": torch.as_tensor(np.asarray(native["delays_generic"], dtype=np.float64)),
        "impulses": torch.as_tensor(np.asarray(native["impulses_generic"], dtype=np.float64)),
    }
    return output


def _values(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _attach_slot_timing_metadata(current, owner, patch):
    """Extend planner timing limits for the synthetic buffer pin lanes."""

    caps = patch["buffer_caps"]
    slots = caps.get("slots", ())
    if not slots:
        slots = [caps]
    base_caps = np.asarray(_values(owner.lib_pin_caps), dtype=np.float64)
    base_slew = np.asarray(_values(owner.pin_max_slew), dtype=np.float64)
    base_max_cap = np.asarray(_values(owner.pin_max_cap), dtype=np.float64)
    extra_caps = []
    extra_slew = []
    extra_max_cap = []
    for slot in slots:
        extra_caps.extend(
            [float(slot["input"][0]), float(slot["output"][0])]
        )
        input_limits = list(slot.get("input_limits", (0.0, 0.0)))
        output_limits = list(slot.get("output_limits", (0.0, 0.0)))
        extra_slew.extend(
            [float(input_limits[0]), float(output_limits[0])]
        )
        extra_max_cap.extend(
            [float(input_limits[1]), float(output_limits[1])]
        )
    current.lib_pin_caps = np.concatenate(
        [base_caps, np.asarray(extra_caps, dtype=np.float64)]
    )
    current.pin_max_slew = np.concatenate(
        [base_slew, np.asarray(extra_slew, dtype=np.float64)]
    )
    current.pin_max_cap = np.concatenate(
        [base_max_cap, np.asarray(extra_max_cap, dtype=np.float64)]
    )


def _compact_current_pydb(base_pydb, graph, actions, slot):
    """Attach physical metadata to a compact native timing graph view.

    The persistent timing patch intentionally returns only graph and RC arrays.
    The rebuffer planner still needs the placement design contract to build a
    current native ``DesignView``. Reuse immutable metadata from the base
    ``pydb`` and append two timing pins plus one synthetic node per buffer.
    This carrier is for planning only; it must never be used as a committed
    carrier predecessor because its synthetic node is slot-owned.
    """

    if hasattr(base_pydb, "node_names") and not graph.get("pin_names"):
        if hasattr(base_pydb, "_metadata") and hasattr(base_pydb, "_physical"):
            current = _PyDBOverlayView(base_pydb)
        else:
            current = copy.copy(base_pydb)
    else:
        current = SimpleNamespace(**graph)
    base_pin_names = [str(value) for value in base_pydb.pin_names]
    base_net_names = [str(value) for value in base_pydb.net_names]
    base_node_names = [str(value) for value in base_pydb.node_names]
    libcell_names = [str(value) for value in base_pydb.flat_libcell_names]
    libcell_info = list(base_pydb.flat_libcell_info)
    main_id_starts = [int(value) for value in base_pydb.main_id_2_cell_id_start]
    base_pin_count = len(base_pin_names)
    base_node_count = int(getattr(base_pydb, "num_nodes", len(base_node_names)))
    current_pin_count = int(graph["num_pins"])
    current_net_count = len(graph["net2driver_pin_map"])
    extra_pin_count = current_pin_count - base_pin_count
    if extra_pin_count < 0:
        raise ValueError("compact timing graph removed base pins")
    slot_count = extra_pin_count // 2
    if extra_pin_count != 2 * slot_count:
        raise ValueError("compact timing graph has an incomplete buffer slot")
    actions = list(actions) if not isinstance(actions, dict) else [actions]
    if len(actions) < slot_count:
        raise ValueError("compact timing graph is missing buffer metadata")
    slot_actions = actions[-slot_count:]

    drivers = [int(value) for value in _values(graph["net2driver_pin_map"])]
    net_starts = [int(value) for value in _values(graph["net_flat_arcs_start"])]
    net_arcs = np.asarray(graph["net_flat_arcs"], dtype=np.int64).reshape(-1, 2)
    flat_net2pin = []
    flat_net2pin_start = [0]
    for net_id, driver in enumerate(drivers):
        begin, end = net_starts[net_id], net_starts[net_id + 1]
        pins = [driver]
        pins.extend(int(net_arcs[index, 1]) for index in range(begin, end))
        flat_net2pin.extend(pins)
        flat_net2pin_start.append(len(flat_net2pin))

    current.pin_names = base_pin_names + [
        name
        for index in range(slot_count)
        for name in (
            f"__rebuffer_slot{slot}_{index}:A",
            f"__rebuffer_slot{slot}_{index}:Y",
        )
    ]
    current.net_names = base_net_names + [
        f"__rebuffer_slot{slot}_{index}_net"
        for index in range(max(0, current_net_count - len(base_net_names)))
    ]
    current.node_names = base_node_names + [
        f"__rebuffer_slot{slot}_{index}" for index in range(slot_count)
    ]
    current.node_master_names = _values(getattr(base_pydb, "node_master_names", ())) + [
        str(action.get("buffer_master_name", ""))
        for action in slot_actions
    ]
    synthetic_main_ids = []
    synthetic_offsets = []
    for action in slot_actions:
        master = str(action.get("buffer_master_name", ""))
        cell_index = libcell_names.index(master)
        info = libcell_info[cell_index]
        main_id = int(info[1])
        synthetic_main_ids.append(main_id)
        synthetic_offsets.append(cell_index - main_id_starts[main_id])
    current.num_pins = current_pin_count
    current.num_nets = current_net_count
    current.num_nodes = base_node_count + slot_count
    current.flat_net2pin_map = flat_net2pin
    current.flat_net2pin_start_map = flat_net2pin_start
    current.net2driver_pin_map = drivers
    current.pin2net_map = _values(graph["pin2net_map"])
    current.net_flat_arcs_start = graph["net_flat_arcs_start"]
    current.net_flat_arcs = graph["net_flat_arcs"]
    current.pin2node_map = _values(getattr(base_pydb, "pin2node_map", ())) + [
        base_node_count + index
        for index in range(slot_count)
        for _ in (0, 1)
    ]
    current.pin_offset_x = _values(getattr(base_pydb, "pin_offset_x", ())) + [
        0.0
    ] * extra_pin_count
    current.pin_offset_y = _values(getattr(base_pydb, "pin_offset_y", ())) + [
        0.0
    ] * extra_pin_count
    current.pin_direct = _values(getattr(base_pydb, "pin_direct", ())) + [
        "R0"
    ] * extra_pin_count
    current.node_x = _values(getattr(base_pydb, "node_x", ())) + [
        float(action.get("candidate_location_x_dbu", 0.0))
        for action in slot_actions
    ]
    current.node_y = _values(getattr(base_pydb, "node_y", ())) + [
        float(action.get("candidate_location_y_dbu", 0.0))
        for action in slot_actions
    ]
    current.node_size_x = _values(getattr(base_pydb, "node_size_x", ())) + [
        0.0
    ] * slot_count
    current.node_size_y = _values(getattr(base_pydb, "node_size_y", ())) + [
        0.0
    ] * slot_count
    if hasattr(base_pydb, "node_is_buffer"):
        current.node_is_buffer = _values(base_pydb.node_is_buffer) + [True] * slot_count
    if hasattr(base_pydb, "node_is_fixed"):
        current.node_is_fixed = _values(base_pydb.node_is_fixed) + [False] * slot_count
    current.inst_main_id = _values(getattr(base_pydb, "inst_main_id", ())) + synthetic_main_ids
    current.inst_libcell_offset = _values(
        getattr(base_pydb, "inst_libcell_offset", ())
    ) + synthetic_offsets
    current.pin_2_libpin_offset = _values(
        getattr(base_pydb, "pin_2_libpin_offset", ())
    ) + [0, 1] * slot_count
    current.flat_inst_arcs_by_level = graph["flat_inst_arcs_by_level"]
    current.flat_inst_arcs_by_level_start = graph["flat_inst_arcs_by_level_start"]
    current.pin_pred_start = graph["pin_pred_start"]
    current.pin_pred_pin = graph["pin_pred_pin"]
    current.pin_pred_arc_id = graph["pin_pred_arc_id"]
    current._compact_timing_graph = True
    return current


def _persistent_steiner_payload(owner, base_pydb, graph, patch, actions, state):
    """Build the current partition contract from the incremental RC patch.

    ``export_overlay_carrier`` also emits this information, but it walks and
    materializes every physical net in the design.  A persistent same-net
    slot already owns the complete current RC forest in ``patch``; reuse that
    forest and immutable base Steiner vertices to keep provisional journal
    evaluations local to the active topology.
    """

    pin_count = len(base_pydb.pin_names)
    base_vertex_count = int(owner.new_x.numel())
    buffer_count = int(patch.get("buffer_count", 0))
    vertex_count = int(patch.get("slot_vertex_count", base_vertex_count + 2 * buffer_count))
    base_net_count = len(base_pydb.net_names)
    child_net_base = base_net_count
    state_net_id = int(state.net_id)
    vertex_shift = 2 * buffer_count

    def values(value):
        return list(_values(value))

    def map_base_node(node):
        node = int(node)
        return node if node < pin_count else node + vertex_shift

    base_topo_start = values(owner.steiner.net_flat_topo_sort_start)
    base_topo = values(owner.steiner.net_flat_topo_sort)
    base_parent = values(owner.steiner.pin_fa)
    patch_rc = patch["rc"]
    patch_topo_start = values(patch_rc["topo_start"])
    patch_topo = values(patch_rc["topo"])
    patch_parent = values(patch_rc["parent"])
    patch_edge_resistance = values(patch_rc["edge_resistance"])
    patch_edge_capacitance = values(patch_rc["edge_capacitance"])
    base_edge_resistance = values(owner.rc.flat_pin_to_res)

    from dreamplace.ops.rebuffer import rebuffer_cpp

    state_view = rebuffer_cpp.incremental_topology_view(state)
    buffers = list(state_view.get("buffers", ()) or ())
    buffer_slot_by_node = {}
    for slot, buffer in enumerate(buffers):
        buffer_slot_by_node[int(buffer["input_node"])] = slot
        buffer_slot_by_node[int(buffer["output_node"])] = slot
    state_coordinates = state_view.get("coordinates", {}) or {}

    def map_state_node(node):
        node = int(node)
        slot = buffer_slot_by_node.get(node)
        if slot is not None:
            return pin_count + 2 * int(slot) + (
                0 if node == int(buffers[slot]["input_node"]) else 1
            )
        return map_base_node(node)

    state_node_by_vertex = [-1] * vertex_count
    state_net_by_vertex = [-1] * vertex_count
    for raw_node in state_coordinates:
        node = int(raw_node)
        mapped = map_state_node(node)
        if 0 <= mapped < vertex_count:
            state_node_by_vertex[mapped] = node
            state_net_by_vertex[mapped] = state_net_id

    state_provenance = []
    for partition in range(buffer_count + 1):
        begin = int(patch_topo_start[partition])
        end = int(patch_topo_start[partition + 1])
        partition_id = (
            state_net_id if partition == 0 else child_net_base + partition - 1
        )
        for current_vertex in patch_topo[begin:end]:
            current_vertex = int(current_vertex)
            if not (0 <= current_vertex < vertex_count):
                continue
            state_node = state_node_by_vertex[current_vertex]
            if state_node < 0:
                continue
            state_provenance.append(
                {
                    "partition_id": partition_id,
                    "vertex_id": current_vertex,
                    "state_net_id": state_net_id,
                    "state_node_id": state_node,
                }
            )

    topologies = []
    base_node_slices = []
    for net_id in range(base_net_count):
        if net_id == state_net_id:
            begin = int(patch_topo_start[0])
            end = int(patch_topo_start[1])
            topologies.append([int(node) for node in patch_topo[begin:end]])
            base_node_slices.append(None)
            continue
        begin = int(base_topo_start[net_id])
        end = int(base_topo_start[net_id + 1])
        original = [int(node) for node in base_topo[begin:end]]
        topologies.append([map_base_node(node) for node in original])
        base_node_slices.append(original)
    for partition in range(1, buffer_count + 1):
        begin = int(patch_topo_start[partition])
        end = int(patch_topo_start[partition + 1])
        topologies.append([int(node) for node in patch_topo[begin:end]])

    parent = [-1] * vertex_count
    children = [[] for _ in range(vertex_count)]
    edge_resistance = [0.0] * vertex_count
    node_capacitance = [0.0] * vertex_count

    def add_edge(parent_node, child_node, resistance, capacitance):
        parent_node = int(parent_node)
        child_node = int(child_node)
        if parent_node < 0 or child_node < 0 or child_node >= vertex_count:
            return
        if parent[child_node] not in (-1, parent_node):
            raise ValueError("persistent Steiner vertex has multiple parents")
        if parent[child_node] == parent_node:
            return
        parent[child_node] = parent_node
        children[parent_node].append(child_node)
        edge_resistance[child_node] = float(resistance)
        half_cap = 0.5 * float(capacitance)
        node_capacitance[parent_node] += half_cap
        node_capacitance[child_node] += half_cap

    for net_id, original_nodes in enumerate(base_node_slices):
        if original_nodes is None:
            continue
        original_set = set(original_nodes)
        for original_child in original_nodes:
            original_parent = int(base_parent[original_child])
            if original_parent < 0 or original_parent not in original_set:
                continue
            resistance = (
                float(base_edge_resistance[original_child])
                if original_child < len(base_edge_resistance)
                else 0.0
            )
            capacitance = resistance / float(owner.r_unit) * float(owner.c_unit)
            add_edge(
                map_base_node(original_parent),
                map_base_node(original_child),
                resistance,
                capacitance,
            )

    for node in range(vertex_count):
        patch_parent_node = int(patch_parent[node]) if node < len(patch_parent) else -1
        if patch_parent_node < 0 or patch_parent_node == node:
            continue
        add_edge(
            patch_parent_node,
            node,
            patch_edge_resistance[node] if node < len(patch_edge_resistance) else 0.0,
            patch_edge_capacitance[node] if node < len(patch_edge_capacitance) else 0.0,
        )

    for row in children:
        row.sort()
    flat_pin_to = [child for row in children for child in row]
    flat_pin_to_start = [0]
    child_offset = 0
    for row in children:
        child_offset += len(row)
        flat_pin_to_start.append(child_offset)
    flat_pin_from = [
        parent_node
        for parent_node, row in enumerate(children)
        for _child in row
    ]

    node_x = [0.0] * vertex_count
    node_y = [0.0] * vertex_count
    base_x = values(owner.new_x)
    base_y = values(owner.new_y)
    for original_node in range(min(base_vertex_count, len(base_x), len(base_y))):
        mapped = map_base_node(original_node)
        if mapped < vertex_count:
            node_x[mapped] = float(base_x[original_node])
            node_y[mapped] = float(base_y[original_node])
    slot_actions = list(actions)[-buffer_count:] if buffer_count else []
    for slot, action in enumerate(slot_actions):
        pin = pin_count + 2 * slot
        if pin >= vertex_count:
            continue
        node_x[pin] = node_x[pin + 1] = float(
            action.get("candidate_location_x_dbu", action.get("x_dbu", 0.0))
        )
        node_y[pin] = node_y[pin + 1] = float(
            action.get("candidate_location_y_dbu", action.get("y_dbu", 0.0))
        )

    return {
        "status": "ok",
        "steiner_node_base": pin_count + 2 * buffer_count,
        "net_ids": list(range(base_net_count))
        + [child_net_base + index for index in range(buffer_count)],
        "net_flat_topo_sort_start": [0, *itertools.accumulate(len(row) for row in topologies)],
        "net_flat_topo_sort": [node for row in topologies for node in row],
        "pin_fa": parent,
        "flat_pin_to_start": flat_pin_to_start,
        "flat_pin_to": flat_pin_to,
        "flat_pin_from": flat_pin_from,
        "node_x": node_x,
        "node_y": node_y,
        "edge_resistance": edge_resistance,
        "node_capacitance": node_capacitance,
        "state_node_by_vertex": state_node_by_vertex,
        "state_net_by_vertex": state_net_by_vertex,
        "state_provenance": state_provenance,
    }


class _TimingView:
    def __init__(self, pydb, model, delays, loads, impulses, owner):
        self.pydb = pydb
        self.model = model
        self.delays = delays
        self.loads = loads
        self.impulses = impulses
        self._owner = owner
        for name in ("lib_pin_caps", "pin_max_slew", "pin_max_cap"):
            setattr(self, name, getattr(owner, name))

    def _snapshot_from_outputs(self, wns, tns, worst):
        slack = timing_pin_slack(self.model).cpu().numpy()
        endpoint_ids = self.model.end_points.detach().cpu().numpy()
        endpoint_slack = slack[endpoint_ids]
        return TimingSnapshot(
            wns_ps=float(wns),
            tns_ps=float(tns),
            worst_slack_ps=float(worst),
            endpoint_count=int(np.count_nonzero(np.isfinite(endpoint_slack))),
            violating_endpoint_count=int(np.count_nonzero(endpoint_slack < 0.0)),
        )


class PersistentTimingRuntime:
    """Reuse one materialized native graph for a cumulative same-net state."""

    def __init__(self, owner, first_runtime, *, slot=0):
        self.owner = owner
        self.base_pydb = owner.pydb
        # Cached carrier graphs are immutable checkpoints. The reusable slot
        # owns its operator so later alternatives cannot overwrite them.
        self.slot_model = copy.deepcopy(first_runtime.model.model)
        self.slot = int(slot)
        self.current = None
        self.payload = first_runtime.payload

    def evaluate(self, overlay):
        if not overlay.actions:
            raise ValueError("persistent timing patch requires an active action")
        net_names = {str(action["net_name"]) for action in overlay.actions}
        if len(net_names) != 1 or len(overlay.native_topologies) != 1:
            raise ValueError("persistent timing patch supports one same-net state")
        from dreamplace.ops.rebuffer import rebuffer_cpp

        actions = list(overlay.actions)
        action = actions[-1]
        state = overlay.native_topologies.get(str(action["net_name"]))
        if state is None:
            raise ValueError("persistent timing slot is missing the active net state")
        import time

        total_started = time.perf_counter()
        action_payload = {
            **action,
            "downstream_pin_ids": [
                int(self.owner._pin_id(name))
                for name in action.get("downstream_pin_names", ())
            ],
        }
        patch = rebuffer_cpp.build_incremental_timing_patch(
            self.base_pydb,
            state,
            action_payload,
            int(self.owner.new_x.numel()),
            self.slot,
        )
        patch_build_done = time.perf_counter()
        graph = dict(patch["graph"])
        buffer_count = int(patch.get("buffer_count", 0))
        current_pydb = _compact_current_pydb(
            self.base_pydb, graph, actions, self.slot
        )
        model = self.slot_model
        model.pin_net = _tensor(graph["pin2net_map"], dtype=torch.int64)
        model.net_flat_arcs_start = _tensor(graph["net_flat_arcs_start"], dtype=torch.int64)
        model.net_flat_arcs = _tensor(graph["net_flat_arcs"], dtype=torch.int64)
        model.net2driver_pin_map = _tensor(graph["net2driver_pin_map"], dtype=torch.int64)
        model.flat_inst_arcs_by_level = _tensor(
            graph["flat_inst_arcs_by_level"], dtype=torch.int64
        )
        model.flat_inst_arcs_by_level_start = _tensor(
            graph["flat_inst_arcs_by_level_start"], dtype=torch.int64
        )
        model.pin_pred_start = _tensor(graph["pin_pred_start"], dtype=torch.int64)
        model.pin_pred_pin = _tensor(graph["pin_pred_pin"], dtype=torch.int64)
        model.pin_pred_arc_id = _tensor(graph["pin_pred_arc_id"], dtype=torch.int64)
        model.num_pins = int(graph.get("num_pins", model.pin_net.numel()))
        model._sparse_runtime = None
        model._traversal_pruner = None
        model.critical_path_topology_epoch = int(
            getattr(model, "critical_path_topology_epoch", 0)
        ) + 1
        # This persistent slot mutates the shared TimingPropagation graph
        # without going through RebufferTimingModel._forward().  Invalidate
        # the owner's native path details before the next planner pass.
        invalidate_paths = getattr(self.owner, "invalidate_native_setup_path_cache", None)
        if callable(invalidate_paths):
            invalidate_paths()

        rc = _patch_rc(self.owner, patch, action_payload)
        rc_done = time.perf_counter()
        current = _TimingView(
            current_pydb,
            model,
            rc["delays"],
            rc["loads"],
            rc["impulses"],
            self.owner,
        )
        _attach_slot_timing_metadata(current, self.owner, patch)
        current.changed_pins = np.asarray(patch["changed_pin_ids"], dtype=np.int32)
        current.native_schedule = dict(patch.get("schedule", {}))
        # The compact timing graph plus the incremental RC forest already
        # contains the current same-net topology.  Build only the native
        # partition Steiner contract here; exporting a full physical carrier
        # for every provisional mutation made the OpenROAD journal path
        # rebuild all design nets before each Python affected propagation.
        current_payload = _persistent_steiner_payload(
            self.owner,
            self.base_pydb,
            graph,
            patch,
            actions,
            state,
        )
        current.pydb = current_pydb
        previous = _TimingView(
            self.base_pydb,
            self.owner._base_operator,
            self.owner.delays,
            self.owner.loads,
            self.owner.impulses,
            self.owner,
        )
        started = __import__("time").perf_counter()
        incremental_state = IncrementalTimingState(previous)
        snapshot = incremental_state.evaluate_affected(current)
        timing_done = time.perf_counter()
        model.last_incremental_profile["persistent_patch_ms"] = (
            timing_done - started
        ) * 1000.0
        model.last_incremental_profile["patch_build_ms"] = (
            patch_build_done - total_started
        ) * 1000.0
        model.last_incremental_profile["patch_rc_ms"] = (
            rc_done - patch_build_done
        ) * 1000.0
        model.last_incremental_profile["persistent_total_ms"] = (
            timing_done - total_started
        ) * 1000.0
        self.current = current
        self.payload = {
            "steiner": current_payload,
            "original_net_by_net": list(range(len(self.base_pydb.net_names)))
            + [int(state.net_id)] * buffer_count,
            "state_node_by_vertex": current_payload["state_node_by_vertex"],
            "state_net_by_vertex": current_payload["state_net_by_vertex"],
            "state_provenance": current_payload["state_provenance"],
            "graph": graph,
            "patch": patch,
        }
        return snapshot

    def evaluate_delta(self, parent_runtime, overlay, action, state):
        """Evaluate one new cross-net action on an accepted full parent.

        The parent carrier already contains the accepted actions.  The native
        one-slot patch is therefore used relative to that carrier, rather than
        relative to the immutable S0 carrier.  This is intentionally limited
        to a state with one buffer on the candidate net; same-net child chains
        still use the canonical full overlay path until their node mapping is
        extended.
        """

        from dreamplace.ops.rebuffer import rebuffer_cpp

        owner = parent_runtime.model
        action_payload = {
            **action,
            "downstream_pin_ids": [
                int(owner._pin_id(name))
                for name in action.get("downstream_pin_names", ())
            ],
        }
        patch = rebuffer_cpp.build_incremental_timing_patch(
            owner.pydb,
            state,
            action_payload,
            int(owner.new_x.numel()),
            0,
        )
        graph = dict(patch["graph"])
        current_pydb = _compact_current_pydb(owner.pydb, graph, action, 0)
        model = copy.copy(owner.model)
        model.pin_net = _tensor(graph["pin2net_map"], dtype=torch.int64)
        model.net_flat_arcs_start = _tensor(
            graph["net_flat_arcs_start"], dtype=torch.int64
        )
        model.net_flat_arcs = _tensor(graph["net_flat_arcs"], dtype=torch.int64)
        model.net2driver_pin_map = _tensor(
            graph["net2driver_pin_map"], dtype=torch.int64
        )
        model.flat_inst_arcs_by_level = _tensor(
            graph["flat_inst_arcs_by_level"], dtype=torch.int64
        )
        model.flat_inst_arcs_by_level_start = _tensor(
            graph["flat_inst_arcs_by_level_start"], dtype=torch.int64
        )
        model.pin_pred_start = _tensor(graph["pin_pred_start"], dtype=torch.int64)
        model.pin_pred_pin = _tensor(graph["pin_pred_pin"], dtype=torch.int64)
        model.pin_pred_arc_id = _tensor(graph["pin_pred_arc_id"], dtype=torch.int64)
        model.num_pins = int(graph.get("num_pins", model.pin_net.numel()))
        model._sparse_runtime = None
        model._traversal_pruner = None
        model.critical_path_topology_epoch = int(
            getattr(model, "critical_path_topology_epoch", 0)
        ) + 1

        invalidate_paths = getattr(owner, "invalidate_native_setup_path_cache", None)
        if callable(invalidate_paths):
            invalidate_paths()
        rc = _patch_rc(owner, patch, action_payload)
        current = _TimingView(
            current_pydb,
            model,
            rc["delays"],
            rc["loads"],
            rc["impulses"],
            owner,
        )
        _attach_slot_timing_metadata(current, owner, patch)
        current.changed_pins = np.asarray(patch["changed_pin_ids"], dtype=np.int32)
        current.native_schedule = dict(patch.get("schedule", {}))
        previous = _TimingView(
            owner.pydb,
            owner.model,
            owner.delays,
            owner.loads,
            owner.impulses,
            owner,
        )
        snapshot = IncrementalTimingState(previous).evaluate_affected(current)
        self.current = current
        # The compact graph is sufficient for the incremental Python timing
        # slot, but it is not a complete native planner contract.  The next
        # queue pass needs the current partition Steiner view and the
        # partition-to-original-net map to project a candidate back through
        # the live overlay.  Keep those fields alongside the patch so a
        # cross-net parent delta remains incremental on the next pass instead
        # of forcing ``force_full_overlay_runtime()``.
        from dreamplace.ops.rebuffer.payloads import steiner_payload

        carrier_payload = rebuffer_cpp.export_overlay_carrier(
            self.base_pydb,
            steiner_payload(self.base_pydb, self.owner),
            [
                overlay.native_topologies[name]
                for name in sorted(overlay.native_topologies)
            ],
            list(overlay.actions),
            self.owner.r_unit,
            self.owner.c_unit,
        )
        self.payload = {
            **carrier_payload,
            "graph": graph,
            "patch": patch,
        }
        return snapshot, current


__all__ = ["PersistentTimingRuntime"]
