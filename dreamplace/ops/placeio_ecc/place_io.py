from dataclasses import asdict, dataclass, field, replace

from dreamplace.ops.placeio_common.backend_contract import ecc_backend_caps

from .backend_caps import DEFAULT_BACKEND_CAPS
from .export_options import PyDbExportOptions, resolve_flag
from .sparse_timing_schema import SparseTimingView
from .timing_schema import validate_timing_schema


@dataclass
class ECCPlaceIOBackend:
    """Native ecc-tools database plus the Python wrapper that owns it."""

    module: object
    workspace: str
    dm_inst: object
    pydb: object
    backend_caps: object = DEFAULT_BACKEND_CAPS
    timing_enabled: bool = False
    timing_inputs: dict = field(default_factory=dict)
    route_num_bins_x: int = 512
    route_num_bins_y: int = 512
    with_routability: bool = False
    refresh_generation: int = 0
    export_options: PyDbExportOptions = PyDbExportOptions()
    sparse_timing_view: SparseTimingView | None = None


def _module_from_data_manager(data_manager):
    module = getattr(data_manager, "ecc_module", data_manager)
    required = ("get_dmInst_ptr", "pydb", "write_placement_back", "def_save")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError("ecc backend requires ECCToolsModule methods: " + ", ".join(missing))
    return module


class PlaceIOFunction:
    @staticmethod
    def _promote(raw_db, **changes):
        raw_db.backend_caps = replace(raw_db.backend_caps, **changes)

    @staticmethod
    def read(params, data_manager):
        module = _module_from_data_manager(data_manager)
        workspace = str(getattr(data_manager, "dir_workspace", ""))
        with_sta = bool(getattr(params, "with_sta", False))
        route_num_bins_x = int(getattr(params, "route_num_bins_x", 512) or 512)
        route_num_bins_y = int(getattr(params, "route_num_bins_y", 512) or 512)
        with_routability = bool(getattr(params, "routability_opt_flag", False))
        export_options = PyDbExportOptions.from_params(params)
        timing_inputs = dict(getattr(params, "design_inputs", {}) or {})
        timing_summary = {}
        python_timing = with_sta and (
            resolve_flag(params, "repair_buffering_enable")
            or resolve_flag(params, "repair_buffering_midplace_enable")
        )
        if with_sta:
            method = "prepare_placement_metadata" if python_timing else "prepare_place_timing"
            prepare_timing = getattr(module, method, None)
            if not callable(prepare_timing):
                raise RuntimeError(f"ECC-Tools timing export requires {method}")
            timing_summary = dict(prepare_timing(timing_inputs) or {})
        # The metadata export can only express a single ideal clock without path
        # exceptions.  When the SDC exceeds that model the preparer falls back
        # to the full native constraint export, which must not clear the
        # endpoint arrays and must run the complete snapshot compiler.
        timing_metadata_only = python_timing and bool(
            timing_summary.get("placement_metadata_supported", True)
        )
        dm_inst = module.get_dmInst_ptr()
        pydb = module.pydb(
            dm_inst,
            route_num_bins_x,
            route_num_bins_y,
            with_routability,
            with_sta=with_sta,
            **({"timing_metadata_only": True} if timing_metadata_only else {}),
            **asdict(export_options),
        )
        # Qualify the native export before the carrier rebuild replaces the
        # runtime arc schedule; the sparse base identities describe the export,
        # not the committed carrier.
        sparse_timing_view = validate_timing_schema(pydb) if with_sta else None
        if python_timing:
            from .library_metadata import attach_repair_library_source
            from .timing_carrier import CommittedTimingPyDB

            pydb = CommittedTimingPyDB(pydb, pydb, recovery_rat_is_capture_deadline=True)
            attach_repair_library_source(pydb, params, timing_inputs)
        raw_db = ECCPlaceIOBackend(
            module=module,
            workspace=workspace,
            dm_inst=dm_inst,
            pydb=pydb,
            backend_caps=ecc_backend_caps(
                timing=with_sta,
                parasitics_initialization=timing_summary.get("parasitics_initialization"),
            ),
            timing_enabled=with_sta,
            timing_inputs=timing_inputs,
            route_num_bins_x=route_num_bins_x,
            route_num_bins_y=route_num_bins_y,
            with_routability=with_routability,
            export_options=export_options,
            sparse_timing_view=sparse_timing_view,
        )
        if python_timing:
            PlaceIOFunction._promote(
                raw_db,
                sta_reference_role="dreamplace_timing_propagation",
                sta_state_status=(
                    "python_timing_metadata_ready"
                    if timing_metadata_only
                    else "python_timing_native_constraints"
                ),
            )
        return raw_db

    @staticmethod
    def pydb(raw_db):
        return raw_db.pydb

    @staticmethod
    def dm_inst(raw_db):
        return raw_db.dm_inst

    @staticmethod
    def backend_caps(raw_db=None):
        if isinstance(raw_db, ECCPlaceIOBackend):
            return raw_db.backend_caps.to_dict()
        return DEFAULT_BACKEND_CAPS.to_dict()

    @staticmethod
    def apply(raw_db, node_x, node_y):
        raw_db.module.write_placement_back(raw_db.dm_inst, node_x, node_y)

    @staticmethod
    def write_def(raw_db, filename):
        return raw_db.module.def_save(filename)

    @staticmethod
    def write_tcl(raw_db, filename):
        return raw_db.module.tcl_save(filename)

    @staticmethod
    def write_verilog(raw_db, filename):
        return raw_db.module.verilog_save(filename)

    @staticmethod
    def apply_sizing(raw_db, cell_ids, cell_master_names, *, current_cell_ids=None):
        if not raw_db.timing_enabled:
            raise RuntimeError("ECC backend does not support sizing writeback")
        pydb = raw_db.pydb
        if len(cell_ids) != len(pydb.inst_main_id):
            raise ValueError("sizing target count does not match ECC PyPlaceDB nodes")
        if current_cell_ids is not None and len(current_cell_ids) != len(cell_ids):
            raise ValueError("sizing source count does not match target nodes")
        node_ids = []
        target_names = []
        cross_family_sources = []
        cross_family_targets = []
        for node_id, target_id in enumerate(cell_ids):
            target_id = int(target_id)
            main_id = int(pydb.inst_main_id[node_id])
            if main_id < 0:
                if target_id >= 0:
                    raise ValueError(f"node {node_id} has no Liberty family")
                continue
            start = int(pydb.main_id_2_cell_id_start[main_id])
            end = int(pydb.main_id_2_cell_id_start[main_id + 1])
            source_id = start + int(pydb.inst_libcell_offset[node_id])
            current_id = source_id if current_cell_ids is None else int(current_cell_ids[node_id])
            if target_id == current_id:
                continue
            if target_id < 0 or target_id >= len(cell_master_names):
                raise ValueError(f"node {node_id} sizing target is outside its Liberty family")
            if not start <= target_id < end:
                if current_id != source_id:
                    raise ValueError("cross-family buffer sizing requires the current PyDB source")
                cross_family_sources.append(source_id)
                cross_family_targets.append(target_id)
            node_ids.append(node_id)
            target_names.append(cell_master_names[target_id])
        if not node_ids:
            return {"ok": True, "accepted_count": 0, "requested_count": 0, "rejected_count": 0}
        if cross_family_sources:
            from .library_metadata import validate_buffer_resize_pairs

            validate_buffer_resize_pairs(
                pydb, cross_family_sources, cross_family_targets, cell_master_names
            )
        try:
            summary = dict(pydb.apply_sizing(node_ids, target_names) or {})
        except AttributeError as exc:
            raise RuntimeError("ECC native PyPlaceDB has no sizing transaction") from exc
        summary["requested_count"] = len(node_ids)
        if bool(summary.get("ok")) and int(summary.get("accepted_count", 0) or 0) == len(node_ids):
            PlaceIOFunction._promote(raw_db, supports_apply_sizing=True)
        return summary

    @staticmethod
    def apply_buffer_actions(raw_db, actions, action_digest):
        if not raw_db.timing_enabled:
            raise RuntimeError("ECC backend does not support buffer commit")
        try:
            summary = dict(raw_db.pydb.apply_buffer_actions(list(actions), action_digest))
        except AttributeError as exc:
            raise RuntimeError("ECC native PyPlaceDB has no buffer transaction") from exc
        for kind in ("accepted", "rejected", "failed"):
            summary[f"{kind}_action_count"] = int(summary[f"{kind}_count"])
        if (
            str(summary.get("status") or "") in {"accepted", "ok"}
            and int(summary.get("accepted_count", 0) or 0) > 0
        ):
            PlaceIOFunction._promote(raw_db, supports_buffer_commit=True)
        return summary

    @staticmethod
    def apply_repair_design(raw_db, plan, *, refresh=True, profile=None):
        """Apply one native ``repair_design`` plan.

        Resizes are committed first through the existing batch sizing
        transaction, then buffer inserts through ``apply_buffer_actions``.
        The caller may defer ``refresh`` when it wants to evaluate a fresh DEF
        reload instead of the in-memory re-export path.
        """

        if not raw_db.timing_enabled:
            raise RuntimeError("ECC backend does not support repair_design commit")
        from .repair_profile import measure_phase
        from .timing_carrier import require_carrier_schema

        require_carrier_schema(raw_db.pydb)
        pydb = raw_db.pydb
        resizes = list(plan.get("resizes", ()) or ())
        actions = list(plan.get("actions", ()) or ())
        sizing_summary = {"ok": True, "accepted_count": 0, "requested_count": 0}
        if resizes:
            cell_ids = []
            for node_id, main_id in enumerate(pydb.inst_main_id):
                main_id = int(main_id)
                if main_id < 0:
                    cell_ids.append(-1)
                    continue
                cell_ids.append(
                    int(pydb.main_id_2_cell_id_start[main_id])
                    + int(pydb.inst_libcell_offset[node_id])
                )
            for resize in resizes:
                cell_ids[int(resize["node_id"])] = int(resize["target_cell_index"])
            with measure_phase(profile, "native_sizing_commit_ms"):
                sizing_summary = PlaceIOFunction.apply_sizing(
                    raw_db,
                    cell_ids,
                    list(pydb.flat_libcell_names),
                )
            if profile is not None:
                profile.commits["sizing"] = dict(sizing_summary)
            if not bool(sizing_summary.get("ok")) or int(
                sizing_summary.get("accepted_count", 0)
            ) != len(resizes):
                return {
                    "status": "resize_rejected",
                    "sizing": dict(sizing_summary),
                    "buffer": {"status": "skipped"},
                    "refresh": {"status": "skipped"},
                }

        buffer_summary = {"status": "skipped", "accepted_count": 0}
        if actions:
            from .buffer_session import action_digest

            with measure_phase(profile, "native_buffer_commit_ms"):
                buffer_summary = dict(
                    PlaceIOFunction.apply_buffer_actions(
                        raw_db,
                        actions,
                        action_digest(actions),
                    )
                    or {}
                )
            if profile is not None:
                profile.commits["buffer"] = buffer_summary
            if str(buffer_summary.get("status") or "") not in {"accepted", "ok"}:
                return {
                    "status": "buffer_rejected",
                    "sizing": dict(sizing_summary),
                    "buffer": buffer_summary,
                    "refresh": {"status": "skipped"},
                }

        refresh_summary = {"status": "skipped"}
        if refresh:
            from .timing_carrier import refresh_python_timing_carrier

            new_pydb, refresh_summary = refresh_python_timing_carrier(raw_db)
            raw_db.pydb = new_pydb
        return {
            "status": "committed",
            "sizing": dict(sizing_summary),
            "buffer": buffer_summary,
            "refresh": dict(refresh_summary or {}),
            "requires_refresh": bool(refresh is False),
        }

    @staticmethod
    def refresh(raw_db, *, refresh_mode="full_rebuild", rebuild_mode="full_rebuild"):
        from .timing_carrier import CommittedTimingPyDB, refresh_python_timing_carrier

        if isinstance(raw_db.pydb, CommittedTimingPyDB):
            if refresh_mode != "full_rebuild" or rebuild_mode != "full_rebuild":
                raise ValueError("Python timing carrier refresh requires full_rebuild")
            return refresh_python_timing_carrier(raw_db)

        from .refresh import RefreshController

        return RefreshController.refresh(
            raw_db,
            refresh_mode=refresh_mode,
            rebuild_mode=rebuild_mode,
        )
