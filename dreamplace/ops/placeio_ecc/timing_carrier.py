"""Committed physical PyPlaceDB plus existing Liberty/SDC timing metadata.

Graph construction and index remapping are native. This module transports
those arrays and exports physical geometry without initializing ECC STA.
"""

from dataclasses import asdict

from .export_options import PyDbExportOptions
from .timing_schema import validate_timing_schema

STATIC_ROUTABILITY_FIELDS = frozenset(
    {
        "num_routing_grids_x",
        "num_routing_grids_y",
        "routing_grid_xl",
        "routing_grid_yl",
        "routing_grid_xh",
        "routing_grid_yh",
        "unit_horizontal_capacities",
        "unit_vertical_capacities",
        "min_wire_widths",
        "min_wire_spacings",
        "initial_horizontal_demand_map",
        "initial_vertical_demand_map",
    }
)
PHYSICAL_FIELDS = frozenset(
    {
        "num_nodes",
        "num_terminals",
        "num_terminal_NIs",
        "native_mutation_epoch",
        "native_state_dirty",
        "node_names",
        "node_name2id_map",
        "node_master_names",
        "node_x",
        "node_y",
        "node_size_x",
        "node_size_y",
        "node_orient",
        "node2orig_node_map",
        "node_is_hard_macro",
        "macro_writeback_candidate",
        "clock_net_names",
        "dbu",
        "xl",
        "yl",
        "xh",
        "yh",
        "row_height",
        "site_width",
        "rows",
        "regions",
        "flat_region_boxes",
        "flat_region_boxes_start",
        "node2fence_region_map",
        "m2_pg_rail_blockage_rects",
        "m2_pg_rail_boxes",
        "m2_pg_rail_density_boxes",
        "total_fixed_node_area",
        "total_space_area",
        "getCongestionMap",
        "write_macro_placement_back",
    }
)


def require_carrier_schema(pydb):
    """Reject unsupported sparse state semantics before a native mutation."""
    # Schema 3 adds the sparse constraint view on top of the same per-pin,
    # per-arc and per-endpoint rows the carrier rebuild consumes, so both
    # versions are valid rebuild bases.  Richer SDCs (multi-clock, path
    # exceptions, clock groups) only exist in schema 3.
    version = int(pydb.timing_schema_version)
    if version not in (2, 3):
        raise ValueError(
            f"committed timing carrier requires schema 2 or 3, got {version}"
        )


class CommittedTimingPyDB:
    """A fresh native graph view retaining the original library/SDC owner."""

    def __init__(
        self,
        base_pydb,
        physical_pydb,
        *,
        previous_pydb=None,
        validate_schema=True,
        recovery_rat_is_capture_deadline=False,
        preserve_sparse=True,
    ):
        from dreamplace.ops.rebuffer import rebuffer_cpp

        from .sparse_timing_carrier import (
            _old_to_new_pin_map,
            rebuild_sparse_view_dict,
            remap_hold_carrier_metadata,
        )
        from .sparse_timing_schema import read_sparse_timing_view

        require_carrier_schema(base_pydb)
        self.recovery_rat_is_capture_deadline = bool(
            recovery_rat_is_capture_deadline
            or getattr(base_pydb, "recovery_rat_is_capture_deadline", False)
        )
        self._metadata = (
            base_pydb._metadata if isinstance(base_pydb, CommittedTimingPyDB) else base_pydb
        )
        self.repair_library_source = getattr(base_pydb, "repair_library_source", None)
        self.repair_library_metadata = getattr(base_pydb, "repair_library_metadata", None)
        self._physical = physical_pydb
        # Liberty/SDC qualification and endpoint rows belong to the immutable
        # metadata owner.  Reusing a previously committed carrier as the C++
        # graph rebuild base is unsafe after an insert/remove round trip: its
        # graph indices describe a transient physical topology.  Always
        # rebuild against the original metadata carrier and the current
        # physical export.
        rebuild_base = self._metadata
        if previous_pydb is None:
            payload = rebuffer_cpp.rebuild_timing_carrier(rebuild_base, physical_pydb)
        else:
            payload = rebuffer_cpp.rebuild_timing_carrier_incremental(
                rebuild_base, previous_pydb, physical_pydb
            )
        self.__dict__.update(payload)
        meta_clocks = [str(n) for n in getattr(self._metadata, "clock_net_names", ()) or ()]
        phys_clocks = [str(n) for n in getattr(self._physical, "clock_net_names", ()) or ()]
        self.clock_net_names = list(dict.fromkeys(meta_clocks + phys_clocks))

        old_to_new_pin = None
        if hasattr(rebuild_base, "hold_contexts") or hasattr(rebuild_base, "hold_launches"):
            old_to_new_pin = _old_to_new_pin_map(rebuild_base, self)
            remap_hold_carrier_metadata(rebuild_base, self, old_to_new_pin)

        if (
            preserve_sparse
            and int(getattr(rebuild_base, "timing_schema_version", 2)) == 3
            and getattr(rebuild_base, "timing_sparse", None)
        ):
            if old_to_new_pin is None:
                old_to_new_pin = _old_to_new_pin_map(rebuild_base, self)
            self.timing_sparse = rebuild_sparse_view_dict(
                rebuild_base,
                self,
                rebuild_base.timing_sparse,
                equivalent_arc_lut=_equivalent_arc_lut,
                old_to_new_pin=old_to_new_pin,
            )
            if getattr(rebuild_base, "timing_min_sparse", None):
                self.timing_min_sparse = rebuild_sparse_view_dict(
                    rebuild_base,
                    self,
                    rebuild_base.timing_min_sparse,
                    equivalent_arc_lut=_equivalent_arc_lut,
                    old_to_new_pin=old_to_new_pin,
                )
            self.timing_schema_version = 3
            self.sparse_timing_view = (
                validate_timing_schema(self)
                if validate_schema
                else read_sparse_timing_view(self)
            )
        else:
            self.timing_schema_version = 2
            self.sparse_timing_view = None
            if validate_schema:
                validate_timing_schema(rebuild_base)

    def __getattr__(self, name):
        if name.startswith("backend_endpoint_"):
            # Native STA endpoint results belong to the old graph. Retaining
            # them would mislabel stale data as a post-commit reference.
            return []
        if name in STATIC_ROUTABILITY_FIELDS:
            meta_caps = getattr(self._metadata, "unit_horizontal_capacities", None)
            if meta_caps is not None and len(meta_caps) > 0:
                return getattr(self._metadata, name)
            return getattr(self._physical, name)
        source = self._physical if name in PHYSICAL_FIELDS else self._metadata
        return getattr(source, name)

    def apply_buffer_actions(self, actions, digest):
        return self._physical.apply_buffer_actions(actions, digest)

    def apply_sizing(self, node_ids, masters):
        return self._physical.apply_sizing(node_ids, masters)


def sync_carrier_positions(raw_db, node_x, node_y):
    """Sync written-back placement coordinates into raw_db.pydb in-place."""
    import numpy as np

    pydb = getattr(raw_db, "pydb", None)
    if pydb is None or bool(getattr(pydb, "native_state_dirty", False)):
        return False
    physical = getattr(pydb, "_physical", None)
    native_target = physical if physical is not None else pydb
    if type(native_target).__name__ == "PyPlaceDB" and hasattr(
        native_target, "sync_placement_from_idb"
    ):
        native_target.sync_placement_from_idb()
        if isinstance(pydb, CommittedTimingPyDB):
            pydb.__dict__["pin_offset_x"] = [float(v) for v in native_target.pin_offset_x]
            pydb.__dict__["pin_offset_y"] = [float(v) for v in native_target.pin_offset_y]
            pydb.__dict__["pin_direct"] = [str(v) for v in native_target.pin_direct]
    else:
        x_arr = np.trunc(np.asarray(node_x, dtype=np.float64)).astype(np.int32, copy=False)
        y_arr = np.trunc(np.asarray(node_y, dtype=np.float64)).astype(np.int32, copy=False)
        for target in (pydb, physical):
            if target is None:
                continue
            tx = getattr(target, "node_x", None)
            ty = getattr(target, "node_y", None)
            if isinstance(tx, np.ndarray) and isinstance(ty, np.ndarray):
                count = min(len(x_arr), len(y_arr), len(tx), len(ty))
                tx[:count] = x_arr[:count]
                ty[:count] = y_arr[:count]
            elif isinstance(tx, list) and isinstance(ty, list):
                count = min(len(x_arr), len(y_arr), len(tx), len(ty))
                tx[:count] = x_arr[:count].tolist()
                ty[:count] = y_arr[:count].tolist()
    try:
        raw_db._carrier_positions_synced = True
    except AttributeError:
        pass
    return True


def _equivalent_arc_lut(lib_arcs, arc_start, new_cell_id, old_lut, old_offset, cache=None):
    if cache is not None:
        key = (new_cell_id, old_lut, old_offset)
        cached = cache.get(key)
        if cached is not None:
            return cached
    begin = int(arc_start[new_cell_id])
    end = int(arc_start[new_cell_id + 1])
    if begin <= old_lut < end:
        if cache is not None:
            cache[(new_cell_id, old_lut, old_offset)] = old_lut
        return old_lut
    prev = lib_arcs[old_lut]
    sig = (str(prev[0]), str(prev[1]), int(prev[4]), int(prev[5]))
    rank = 0
    if old_offset >= 0:
        for lut in range(old_lut - old_offset, old_lut):
            cur = lib_arcs[lut]
            if (str(cur[0]), str(cur[1]), int(cur[4]), int(cur[5])) == sig:
                rank += 1
    matches = [
        lut
        for lut in range(begin, end)
        if (
            str(lib_arcs[lut][0]),
            str(lib_arcs[lut][1]),
            int(lib_arcs[lut][4]),
            int(lib_arcs[lut][5]),
        )
        == sig
    ]
    if not matches or rank >= len(matches):
        raise ValueError(f"current master {new_cell_id} does not contain timing arc {old_lut}")
    res = matches[rank]
    if cache is not None:
        cache[(new_cell_id, old_lut, old_offset)] = res
    return res


def apply_inplace_carrier_sizing(raw_db, resizes, *, profile=None):
    """Update CommittedTimingPyDB in-place for size-only commits without full IDB export."""
    from .place_io import PlaceIOFunction
    from .repair_profile import measure_phase

    old_pydb = getattr(raw_db, "pydb", None)
    if old_pydb is None or not resizes:
        return None
    physical = old_pydb._physical if isinstance(old_pydb, CommittedTimingPyDB) else old_pydb
    metadata = old_pydb._metadata if isinstance(old_pydb, CommittedTimingPyDB) else old_pydb
    is_native_phys = type(physical).__name__ == "PyPlaceDB"
    if not is_native_phys and not isinstance(old_pydb, CommittedTimingPyDB):
        return None

    preserve_sparse = (
        getattr(raw_db, "sparse_timing_view", None) is not None
        or int(getattr(old_pydb, "timing_schema_version", 2)) == 3
    )

    with measure_phase(profile, "timing_carrier_rebuild_ms"):
        if not isinstance(old_pydb, CommittedTimingPyDB):
            carrier = CommittedTimingPyDB(
                old_pydb, physical, preserve_sparse=preserve_sparse
            )
        else:
            carrier = old_pydb
            lib_names = list(getattr(metadata, "flat_libcell_names", ()))
            name_to_cell = {str(name): idx for idx, name in enumerate(lib_names)}
            main_start = getattr(metadata, "main_id_2_cell_id_start")
            libpin_start = getattr(metadata, "cell_id_2_libpin_id_start")
            libpin_names = getattr(metadata, "flat_lib_pin_names")
            arc_start = getattr(metadata, "cell_id_2_arc_id_start")
            lib_arcs = getattr(metadata, "flat_libarc_info")
            phys_pin_x = getattr(physical, "pin_offset_x")
            phys_pin_y = getattr(physical, "pin_offset_y")
            lib_pin_x = None if is_native_phys else getattr(metadata, "flat_lib_pin_offset_x", None)
            lib_pin_y = None if is_native_phys else getattr(metadata, "flat_lib_pin_offset_y", None)
            lib_w = None if is_native_phys else getattr(metadata, "flat_libcell_width", None)
            lib_h = None if is_native_phys else getattr(metadata, "flat_libcell_height", None)
            phys_masters = None if is_native_phys else getattr(physical, "node_master_names", None)
            node_names = getattr(carrier, "node_names")
            pin_names = getattr(carrier, "pin_names")
            node2pin_map = getattr(carrier, "node2pin_map")
            inst_arcs = getattr(carrier, "inst_flat_arcs")
            inst_arcs_start = getattr(carrier, "inst_flat_arcs_start")
            scheduled = getattr(carrier, "flat_inst_arcs_by_level")
            arc_cell = getattr(carrier, "arc_libcell_id")
            arc_lut = getattr(carrier, "arc_libarc_id")
            arc_sense = getattr(carrier, "arc_sense")
            arc_type = getattr(carrier, "arc_type")
            arc_offset = getattr(carrier, "arc_offset")

            resized_map = {}
            updated_arc_indices = []
            updated_arc_rows = []
            port_offset_cache = {}
            lut_cache = {}
            info_cache = {}
            for item in resizes:
                node_id = int(item["node_id"])
                cell_id = item.get("new_cell_id")
                if cell_id is None:
                    cell_id = item.get("target_cell_index")
                if cell_id is None:
                    cell_id = name_to_cell[str(item["new_master"])]
                cell_id = int(cell_id)
                resized_map[node_id] = cell_id
                main_id = int(carrier.inst_main_id[node_id])
                carrier.inst_libcell_offset[node_id] = cell_id - int(main_start[main_id])
                if not is_native_phys:
                    if phys_masters is not None and node_id < len(phys_masters):
                        phys_masters[node_id] = str(lib_names[cell_id])
                    if lib_w is not None and hasattr(physical, "node_size_x"):
                        physical.node_size_x[node_id] = float(lib_w[cell_id])
                    if lib_h is not None and hasattr(physical, "node_size_y"):
                        physical.node_size_y[node_id] = float(lib_h[cell_id])

                prefix = f"{node_names[node_id]}:"
                p_begin = int(libpin_start[cell_id])
                port_to_offset = port_offset_cache.get(cell_id)
                if port_to_offset is None:
                    p_end = int(libpin_start[cell_id + 1])
                    port_to_offset = {
                        str(libpin_names[idx]): idx - p_begin for idx in range(p_begin, p_end)
                    }
                    port_offset_cache[cell_id] = port_to_offset
                for pin_id in node2pin_map[node_id]:
                    pin_id = int(pin_id)
                    pin_name = str(pin_names[pin_id])
                    lib_offset = None
                    if pin_name.startswith(prefix):
                        port = pin_name[len(prefix) :]
                        if port in port_to_offset:
                            lib_offset = port_to_offset[port]
                            carrier.pin_2_libpin_offset[pin_id] = lib_offset
                    if is_native_phys:
                        carrier.pin_offset_x[pin_id] = float(phys_pin_x[pin_id])
                        carrier.pin_offset_y[pin_id] = float(phys_pin_y[pin_id])
                    elif lib_offset is not None and lib_pin_x is not None and lib_pin_y is not None:
                        lib_pin_idx = p_begin + lib_offset
                        carrier.pin_offset_x[pin_id] = float(lib_pin_x[lib_pin_idx])
                        carrier.pin_offset_y[pin_id] = float(lib_pin_y[lib_pin_idx])
                    else:
                        carrier.pin_offset_x[pin_id] = float(phys_pin_x[pin_id])
                        carrier.pin_offset_y[pin_id] = float(phys_pin_y[pin_id])

                a_begin = int(inst_arcs_start[node_id])
                a_end = int(inst_arcs_start[node_id + 1])
                c_arc_begin = int(arc_start[cell_id])
                for pos in range(a_begin, a_end):
                    arc_idx = int(inst_arcs[pos])
                    row = list(scheduled[arc_idx])
                    new_lut = _equivalent_arc_lut(
                        lib_arcs, arc_start, cell_id, int(row[3]), int(row[6]), lut_cache
                    )
                    info = info_cache.get(new_lut)
                    if info is None:
                        raw_info = lib_arcs[new_lut]
                        info = (int(raw_info[4]), int(raw_info[5]))
                        info_cache[new_lut] = info
                    row[2] = cell_id
                    row[3] = new_lut
                    row[4] = info[0]
                    row[5] = info[1]
                    row[6] = new_lut - c_arc_begin
                    scheduled[arc_idx] = row
                    arc_cell[arc_idx] = cell_id
                    arc_lut[arc_idx] = new_lut
                    arc_sense[arc_idx] = info[0]
                    arc_type[arc_idx] = info[1]
                    arc_offset[arc_idx] = row[6]
                    updated_arc_indices.append(arc_idx)
                    updated_arc_rows.append(row)

            pin2node = getattr(carrier, "pin2node_map")
            for field in ("endpoints_constraint_arcs", "endpoints_timing_check_arcs"):
                checks = getattr(carrier, field, None)
                if not checks:
                    continue
                for idx, row in enumerate(checks):
                    sink_node = int(pin2node[int(row[1])])
                    if sink_node not in resized_map:
                        continue
                    cell_id = resized_map[sink_node]
                    row_list = list(row)
                    new_lut = _equivalent_arc_lut(
                        lib_arcs, arc_start, cell_id, int(row_list[3]), int(row_list[7]), lut_cache
                    )
                    row_list[2] = cell_id
                    row_list[3] = new_lut
                    row_list[7] = new_lut - int(arc_start[cell_id])
                    checks[idx] = row_list

            graph_index = carrier.__dict__.get("timing_graph_index")
            if graph_index is not None and updated_arc_indices:
                if hasattr(graph_index, "update_cells"):
                    graph_index.update_cells(updated_arc_indices, updated_arc_rows)
                else:
                    carrier.__dict__.pop("timing_graph_index", None)

        try:
            physical.native_state_dirty = False
        except AttributeError:
            pass

    raw_db.pydb = carrier
    raw_db.sparse_timing_view = (
        getattr(carrier, "sparse_timing_view", None) if preserve_sparse else None
    )
    raw_db.refresh_generation += 1
    PlaceIOFunction._promote(
        raw_db,
        supports_committed_refresh=True,
        sta_reference_role="dreamplace_timing_propagation",
        sta_state_status="python_timing_carrier_rebuilt",
    )
    return carrier, {
        "status": "ok",
        "refresh_path": "inplace_carrier_sizing",
        "generation": raw_db.refresh_generation,
        "timing_evaluator": "dreamplace_timing_propagation",
        "ecc_sta_run": False,
        "external_sta_run": False,
        "pydb_replaced": carrier is not old_pydb,
        "pydb_counts": {
            "nodes": int(carrier.num_nodes),
            "pins": len(carrier.pin_names),
            "nets": len(carrier.net_names),
        },
    }


def refresh_python_timing_carrier(raw_db, *, profile=None):
    """Read committed IDB connectivity and rebuild graph metadata in C++."""

    from .place_io import PlaceIOFunction
    from .repair_profile import measure_phase

    old_pydb = raw_db.pydb
    options = getattr(raw_db, "export_options", PyDbExportOptions())
    native = getattr(raw_db.module, "ecc", raw_db.module)
    meta_source = old_pydb._metadata if isinstance(old_pydb, CommittedTimingPyDB) else old_pydb
    has_cached_routability = (
        len(getattr(meta_source, "unit_horizontal_capacities", ()) or ()) > 0
    )
    export_routability = False if has_cached_routability else raw_db.with_routability
    with measure_phase(profile, "physical_pydb_export_ms"):
        physical = native.pydb(
            raw_db.dm_inst,
            raw_db.route_num_bins_x,
            raw_db.route_num_bins_y,
            export_routability,
            with_sta=False,
            include_clock_nets=True,
            **asdict(options),
        )
    preserve_sparse = (
        getattr(raw_db, "sparse_timing_view", None) is not None
        or int(getattr(old_pydb, "timing_schema_version", 2)) == 3
    )
    with measure_phase(profile, "timing_carrier_rebuild_ms"):
        carrier = CommittedTimingPyDB(
            old_pydb, physical, preserve_sparse=preserve_sparse
        )
    raw_db.pydb = carrier
    raw_db.sparse_timing_view = (
        getattr(carrier, "sparse_timing_view", None) if preserve_sparse else None
    )
    raw_db.refresh_generation += 1
    PlaceIOFunction._promote(
        raw_db,
        supports_committed_refresh=True,
        sta_reference_role="dreamplace_timing_propagation",
        sta_state_status="python_timing_carrier_rebuilt",
    )
    return carrier, {
        "status": "ok",
        "refresh_path": "physical_pydb_native_graph_rebuild",
        "generation": raw_db.refresh_generation,
        "timing_evaluator": "dreamplace_timing_propagation",
        "ecc_sta_run": False,
        "external_sta_run": False,
        "pydb_replaced": carrier is not old_pydb,
        "pydb_counts": {
            "nodes": int(carrier.num_nodes),
            "pins": len(carrier.pin_names),
            "nets": len(carrier.net_names),
        },
    }


__all__ = [
    "CommittedTimingPyDB",
    "apply_inplace_carrier_sizing",
    "refresh_python_timing_carrier",
    "sync_carrier_positions",
]
