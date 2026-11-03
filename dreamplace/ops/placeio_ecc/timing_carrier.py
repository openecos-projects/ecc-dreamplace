"""Committed physical PyPlaceDB plus existing Liberty/SDC timing metadata.

Graph construction and index remapping are native. This module transports
those arrays and exports physical geometry without initializing ECC STA.
"""

from dataclasses import asdict

from .export_options import PyDbExportOptions
from .timing_schema import validate_timing_schema

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
        "routing_grid_xl",
        "routing_grid_yl",
        "routing_grid_xh",
        "routing_grid_yh",
        "num_routing_grids_x",
        "num_routing_grids_y",
        "unit_horizontal_capacities",
        "unit_vertical_capacities",
        "min_wire_widths",
        "min_wire_spacings",
        "initial_horizontal_demand_map",
        "initial_vertical_demand_map",
        "getCongestionMap",
        "write_macro_placement_back",
    }
)


def require_carrier_schema(pydb):
    """Reject unsupported sparse state semantics before a native mutation."""
    if int(pydb.timing_schema_version) != 2:
        raise ValueError("committed timing carrier currently requires schema 2")


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
    ):
        from dreamplace.ops.rebuffer import rebuffer_cpp

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
        if validate_schema:
            validate_timing_schema(self)

    def __getattr__(self, name):
        if name.startswith("backend_endpoint_"):
            # Native STA endpoint results belong to the old graph. Retaining
            # them would mislabel stale data as a post-commit reference.
            return []
        source = self._physical if name in PHYSICAL_FIELDS else self._metadata
        return getattr(source, name)

    def apply_buffer_actions(self, actions, digest):
        return self._physical.apply_buffer_actions(actions, digest)

    def apply_sizing(self, node_ids, masters):
        return self._physical.apply_sizing(node_ids, masters)


def refresh_python_timing_carrier(raw_db, *, profile=None):
    """Read committed IDB connectivity and rebuild graph metadata in C++."""

    from .place_io import PlaceIOFunction
    from .repair_profile import measure_phase

    old_pydb = raw_db.pydb
    options = getattr(raw_db, "export_options", PyDbExportOptions())
    native = getattr(raw_db.module, "ecc", raw_db.module)
    with measure_phase(profile, "physical_pydb_export_ms"):
        physical = native.pydb(
            raw_db.dm_inst,
            raw_db.route_num_bins_x,
            raw_db.route_num_bins_y,
            raw_db.with_routability,
            with_sta=False,
            include_clock_nets=True,
            **asdict(options),
        )
    with measure_phase(profile, "timing_carrier_rebuild_ms"):
        carrier = CommittedTimingPyDB(old_pydb, physical)
    raw_db.pydb = carrier
    raw_db.sparse_timing_view = None
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


__all__ = ["CommittedTimingPyDB", "refresh_python_timing_carrier"]
