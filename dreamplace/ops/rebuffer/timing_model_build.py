"""Materialize the existing Steiner, RC, and Python timing operators from PyPlaceDB.

This boundary only transports arrays and Liberty/SDC qualification. Native
DreamPlace operators own tree construction, RC, and timing propagation.
"""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from .timing_model_payload import _list, _pad_luts, _pin_pair_arc_mapping


_LUT_PREFIXES = ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check")
_LUT_FIELDS = ("values", "trans_table", "cap_table", "dim")
# Padded Liberty LUTs keyed by the identity of their source arrays. Every
# carrier of a design reads these arrays from the same metadata PyPlaceDB,
# so one padding serves all rebuilt timing models; the cached sources keep
# the ids from being reused while the entry is alive.
_ARCS_INFO_CACHE = {}


def _arcs_info(pydb, arcs_info_type, luts_info_type):
    sources = tuple(
        getattr(pydb, f"{prefix}_flat_luts_{field}")
        for prefix in _LUT_PREFIXES
        for field in _LUT_FIELDS
    )
    key = tuple(id(source) for source in sources)
    cached = _ARCS_INFO_CACHE.get(key)
    if cached is not None and all(a is b for a, b in zip(cached[0], sources)):
        return cached[1]
    infos = {}
    for index, prefix in enumerate(_LUT_PREFIXES):
        values, trans, cap, dims = _pad_luts(*(_list(item) for item in sources[4 * index : 4 * index + 4]))
        infos[f"{prefix}_luts"] = luts_info_type(values, trans, cap, dims)
    arcs_info = arcs_info_type(**infos)
    _ARCS_INFO_CACHE.clear()
    _ARCS_INFO_CACHE[key] = (sources, arcs_info)
    return arcs_info


def build_timing_model(owner):
    from dreamplace.ops.placeio_ecc.timing_schema import read_qualification_arrays
    from dreamplace.ops.rc_timing.rc_timing import RCTiming
    from dreamplace.ops.rebuffer import rebuffer_cpp
    from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo
    from dreamplace.ops.timing_propagation.timing_propagation import (
        ARCS_INFO,
        LUTS_INFO,
        TimingPropagation,
    )

    pydb = owner.pydb
    previous_operator = (
        None if owner._incremental_state is None else owner._incremental_state.operator
    )
    # Name lookups dominate overlay evaluation: a repair pass resolves
    # every downstream pin and net by name.  Build the normalized maps
    # once instead of scanning PyPlaceDB's 40k+ names per action.
    owner._pin_name_to_id = {}
    for index, name in enumerate(pydb.pin_names):
        normalized = str(name).replace("/", ":").replace("\\", ":")
        owner._pin_name_to_id.setdefault(normalized, int(index))
    owner._net_name_to_id = {}
    for index, name in enumerate(pydb.net_names):
        owner._net_name_to_id.setdefault(str(name), int(index))
    owner._net_record_cache = {}
    if owner._incremental_state is None:
        arcs_info = _arcs_info(pydb, ARCS_INFO, LUTS_INFO)
    else:
        arcs_info = owner._incremental_state.arcs_info

    reusable = {
        "inrdelays",
        "infdelays",
        "inrtrans",
        "inftrans",
        "outcaps",
        "start_points",
        "end_points",
        "clock_pins",
        "FF_ids",
        "clk_pin_rtran",
        "clk_pin_ftran",
        "endpoints_rRAT",
        "endpoints_fRAT",
        "endpoints_constraint_arcs",
        "endpoints_timing_check_arcs",
    }

    def tensor(name, dtype=torch.float64):
        if previous_operator is not None and name in reusable:
            cached = getattr(previous_operator, name, None)
            if torch.is_tensor(cached):
                current = np.asarray(getattr(pydb, name))
                if cached.numel() == current.size and cached.dtype == dtype:
                    return cached
        return torch.as_tensor(np.asarray(getattr(pydb, name)), dtype=dtype)

    def rows(name, dtype=torch.float64):
        if previous_operator is not None and name in reusable:
            cached = getattr(previous_operator, name, None)
            if torch.is_tensor(cached):
                current = np.asarray(getattr(pydb, name))
                if cached.numel() == current.size and cached.dtype == dtype:
                    return cached
        values = np.asarray(getattr(pydb, name))
        if values.size == 0:
            values = values.reshape(0, 8)
        return torch.as_tensor(values, dtype=dtype)

    pin_pair_keys, pin_pair_start, pin_pair_indices = _pin_pair_arc_mapping(pydb)
    qualification = {
        name: torch.as_tensor(values, dtype=torch.bool)
        for name, values in read_qualification_arrays(pydb).items()
        if name.endswith("_valid")
    }
    timing_kwargs = dict(
        inrdelays=tensor("inrdelays"),
        infdelays=tensor("infdelays"),
        inrtrans=tensor("inrtrans"),
        inftrans=tensor("inftrans"),
        outcaps=tensor("outcaps"),
        pin_net=tensor("pin2net_map", torch.int64),
        start_points=tensor("start_points", torch.int64),
        end_points=tensor("end_points", torch.int64),
        clock_pins=tensor("clock_pins", torch.int64),
        FF_ids=tensor("FF_ids", torch.int64),
        clk_pin_rtran=tensor("clk_pin_rtran"),
        clk_pin_ftran=tensor("clk_pin_ftran"),
        net_flat_arcs_start=tensor("net_flat_arcs_start", torch.int64),
        net_flat_arcs=tensor("net_flat_arcs", torch.int64),
        net2driver_pin_map=tensor("net2driver_pin_map", torch.int64),
        arcs_info=arcs_info,
        inst_flat_arcs_start=tensor("inst_flat_arcs_start", torch.int64),
        inst_flat_arcs=tensor("inst_flat_arcs", torch.int64),
        endpoints_constraint_arcs=rows("endpoints_constraint_arcs", torch.int64),
        flat_inst_arcs_by_level=rows("flat_inst_arcs_by_level", torch.int64),
        flat_inst_arcs_by_level_start=tensor("flat_inst_arcs_by_level_start", torch.int64),
        endpoints_rRAT=tensor("endpoints_rRAT"),
        endpoints_fRAT=tensor("endpoints_fRAT"),
        flat_pin_to_graph=tensor("flat_pin_to_graph", torch.int64),
        flat_pin_to_graph_start=tensor("flat_pin_to_graph_start", torch.int64),
        flat_pin_to_graph_reverse=tensor("flat_pin_to_graph_reverse", torch.int64),
        flat_pin_to_graph_start_reverse=tensor("flat_pin_to_graph_start_reverse", torch.int64),
        endpoints_timing_check_arcs=rows("endpoints_timing_check_arcs", torch.int64),
        pin_pred_start=tensor("pin_pred_start", torch.int64),
        pin_pred_pin=tensor("pin_pred_pin", torch.int64),
        pin_pred_arc_id=tensor("pin_pred_arc_id", torch.int64),
        pin_succ_start=tensor("pin_succ_start", torch.int64),
        pin_succ_pin=tensor("pin_succ_pin", torch.int64),
        pin_succ_arc_id=tensor("pin_succ_arc_id", torch.int64),
        arc_level_start=tensor("arc_level_start", torch.int64),
        arc_src_pin=tensor("arc_src_pin", torch.int64),
        arc_dst_pin=tensor("arc_dst_pin", torch.int64),
        arc_inst_id=tensor("arc_inst_id", torch.int64),
        endpoint_pin_ids=tensor("endpoint_pin_ids", torch.int64),
        start_pin_ids=tensor("start_pin_ids", torch.int64),
        pin_to_inst_id=tensor("pin_to_inst_id", torch.int64),
        pin_to_node_id=tensor("pin_to_node_id", torch.int64),
        inst_topo_start=tensor("inst_topo_start", torch.int64),
        inst_topo_ids=tensor("inst_topo_ids", torch.int64),
        pin2node_map=tensor("pin2node_map", torch.int64),
        inst_main_id=tensor("inst_main_id", torch.int64),
        timing_propagation_device="cpu",
        timing_aggregation_mode="hard",
        pin_pair_arc_keys=pin_pair_keys,
        flat_pin_pair_arc_start=pin_pair_start,
        flat_pin_pair_arc_indices=pin_pair_indices,
        # The buffer DP needs per-pin required times.  The production
        # fast loop skips full RAT propagation and leaves every internal
        # pin at the 2e8 sentinel, which makes the DP blind to critical
        # nets.
        production_fast_loop=False,
        sparse_timing_view=owner.sparse_timing_view,
        **qualification,
    )
    current_arc_count = int(timing_kwargs["flat_inst_arcs_by_level"].shape[0])
    previous_arc_count = (
        None
        if previous_operator is None
        else int(previous_operator.flat_inst_arcs_by_level.shape[0])
    )
    # Removing a session buffer changes the scheduled arc domain and the
    # cached per-arc delay vector at the same time.  Keep that rollback path
    # on the canonical constructor until the cache has a persistent arc-slot
    # journal; affected AAT/RAT still uses the previous Python timing state.
    can_rebind = (
        previous_operator is not None
        and previous_arc_count is not None
        and current_arc_count >= previous_arc_count
    )
    if owner._incremental_state is None or not can_rebind:
        owner.model = TimingPropagation(**timing_kwargs)
        # The carrier withholds stale native endpoint slews (timing_carrier.py),
        # so recovery checks keep the exported required-time anchor.
        owner.model.recovery_check_uses_live_slew = True
        if owner._incremental_state is not None and not can_rebind:
            # The previous delay cache is indexed by a different scheduled
            # arc domain. Do one canonical full forward for rollback; using
            # the affected adapter here would silently permute cell delays.
            owner._incremental_state = None
    else:
        # A topology patch changes graph tensors and pin count, but not the
        # Python TimingPropagation implementation, device contract, or the
        # immutable Liberty/LUT operator.  Rebind the existing operator
        # instead of paying its constructor and LUT/device setup cost for
        # every candidate.
        previous = owner._incremental_state.operator
        owner.model = copy.copy(previous)
        for name, value in timing_kwargs.items():
            setattr(owner.model, name, value)
        owner.model.num_pins = timing_kwargs["pin_net"].shape[0]
        owner.model.sparse_timing_view = owner.sparse_timing_view
        owner.model._sparse_runtime = None
        owner.model.dtype = timing_kwargs["inrdelays"].dtype
        endpoint_valid = timing_kwargs.get("endpoints_max_valid")
        owner.model.endpoint_max_valid_by_pin = None
        if endpoint_valid is not None:
            owner.model.endpoint_max_valid_by_pin = torch.zeros(
                (owner.model.num_pins, 2), dtype=torch.bool, device=owner.model.device
            )
            owner.model.endpoint_max_valid_by_pin[
                owner.model.end_points.long()
            ] = endpoint_valid
        owner.model._materialize_timing_device_tensors()
        owner.model.last_device_contract = owner.model._build_device_contract(
            {
                "inrdelays": owner.model.inrdelays,
                "infdelays": owner.model.infdelays,
                "pin_net": owner.model.pin_net,
                "start_points": owner.model.start_points,
                "end_points": owner.model.end_points,
                "flat_inst_arcs_by_level": owner.model.flat_inst_arcs_by_level,
                "flat_inst_arcs_by_level_start": owner.model.flat_inst_arcs_by_level_start,
            },
            scope="incremental_rebind",
        )
        owner.model._assert_device_contract(owner.model.last_device_contract)
        owner.model._cell_aat_static_cache = {}
        owner.model._cell_aat_static_cache_hits = 0
        owner.model._cell_aat_static_cache_misses = 0
        owner.model._traversal_pruner = None
        owner.model._traversal_pruner_topology_source = None
        owner.model._setup_critical_path_extractor = None
        owner.model._setup_critical_path_extractor_epoch = None

    pin2node = np.asarray(pydb.pin2node_map, dtype=np.int64)
    pin_x = np.asarray(pydb.node_x, dtype=np.float64)[pin2node]
    pin_y = np.asarray(pydb.node_y, dtype=np.float64)[pin2node]
    if owner.use_pin_offset:
        pin_x = pin_x + np.asarray(pydb.pin_offset_x, dtype=np.float64)
        pin_y = pin_y + np.asarray(pydb.pin_offset_y, dtype=np.float64)
    position = torch.from_numpy(np.concatenate([pin_x, pin_y]))
    if owner.steiner_view is None:
        owner.steiner = SteinerTopo(
            torch.as_tensor(np.asarray(pydb.flat_net2pin_map), dtype=torch.int32),
            torch.as_tensor(np.asarray(pydb.flat_net2pin_start_map), dtype=torch.int32),
            ignore_net_degree=len(pydb.pin_names),
        )
        owner.steiner.rebuild_tree(position)
        owner.new_x, owner.new_y = owner.steiner(position)
    else:
        # Native partition export already owns every split and RC vertex.
        # Reuse it directly; rebuilding FLUTE here would discard that state.
        view = owner.steiner_view
        owner.steiner = SimpleNamespace(
            **{
                name: torch.as_tensor(np.asarray(view[name]), dtype=torch.int32)
                for name in (
                    "net_flat_topo_sort",
                    "net_flat_topo_sort_start",
                    "pin_fa",
                    "flat_pin_to_start",
                    "flat_pin_to",
                    "flat_pin_from",
                )
            }
        )
        owner.new_x = torch.as_tensor(view["node_x"], dtype=torch.float64)
        owner.new_y = torch.as_tensor(view["node_y"], dtype=torch.float64)

    capacitance = rebuffer_cpp.timing_pin_capacitance(pydb)
    caps, rcaps, fcaps = (capacitance[name] for name in ("caps", "rcaps", "fcaps"))
    owner.lib_pin_caps = caps
    owner.lib_pin_rcaps = rcaps
    owner.lib_pin_fcaps = fcaps
    owner.pin_max_cap = capacitance["max_cap"]
    owner.pin_max_slew = capacitance["max_slew"]
    owner.rc = RCTiming(
        r_unit=owner.r_unit,
        c_unit=owner.c_unit,
        scale_factor=owner.scale_factor,
        dbu=owner.dbu,
    )
    (
        owner.pin_caps,
        owner.loads,
        owner.delays,
        owner.ldelays,
        owner.betas,
        owner.impulses,
    ) = owner.rc(
        owner.new_x,
        owner.new_y,
        owner.steiner.net_flat_topo_sort,
        owner.steiner.net_flat_topo_sort_start,
        owner.steiner.pin_fa,
        owner.steiner.flat_pin_to_start,
        owner.steiner.flat_pin_to,
        owner.steiner.flat_pin_from,
        torch.from_numpy(caps),
        torch.from_numpy(rcaps),
        torch.from_numpy(fcaps),
    )
    owner._baseline = (
        owner._forward()
        if owner._incremental_state is None
        else owner._incremental_state.evaluate_affected(owner)
    )
    # Only the new state survives. Retaining its previous owner here would
    # keep every graph/model in a long campaign alive through a reference chain.
    owner._incremental_state = None
