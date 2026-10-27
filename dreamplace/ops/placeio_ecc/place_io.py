from dataclasses import asdict, dataclass, field, replace

from dreamplace.ops.placeio_common.backend_contract import ecc_backend_caps

from .backend_caps import DEFAULT_BACKEND_CAPS
from .export_options import PyDbExportOptions
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
    hold_enabled: bool = False
    timing_inputs: dict = field(default_factory=dict)
    route_num_bins_x: int = 512
    route_num_bins_y: int = 512
    with_routability: bool = False
    refresh_generation: int = 0
    export_options: PyDbExportOptions = PyDbExportOptions()
    sparse_timing_view: SparseTimingView | None = None
    refresh_invalidated_reason: str | None = None


def _module_from_data_manager(data_manager):
    module = getattr(data_manager, "ecc_module", data_manager)
    required = ("get_dmInst_ptr", "pydb", "write_placement_back", "def_save")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError(
            "ecc backend requires ECCToolsModule methods: " + ", ".join(missing)
        )
    return module


class PlaceIOFunction:
    @staticmethod
    def _promote(raw_db, **changes):
        raw_db.backend_caps = replace(raw_db.backend_caps, **changes)

    @staticmethod
    def _invalidate_timing(raw_db, reason):
        """Make a failed native refresh unusable to timing consumers.

        The native refresh destroys the old STA session before rebuilding it.
        Keeping the old PyPlaceDB after that point would let a caller run a
        forward/backward against a model whose native owner no longer exists.
        Geometry ownership remains available, but timing must be rebuilt from
        scratch before it can be enabled again.
        """
        destroy = getattr(raw_db.module, "destroy_place_timing", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass
        raw_db.timing_enabled = False
        raw_db.pydb = None
        raw_db.sparse_timing_view = None
        raw_db.refresh_invalidated_reason = str(reason)
        PlaceIOFunction._promote(
            raw_db,
            has_sta=False,
            supports_sta_pydb=False,
            supports_apply_sizing=False,
            supports_buffer_commit=False,
            supports_committed_refresh=False,
            sta_state_status="invalidated_refresh_failure",
        )

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
        if with_sta:
            prepare_timing = getattr(module, "prepare_place_timing", None)
            if not callable(prepare_timing):
                raise RuntimeError(
                    "ECC-Tools timing export requires prepare_place_timing"
                )
            timing_summary = dict(prepare_timing(timing_inputs) or {})
        timing_model_only = timing_summary.get("mode") == "model_only"
        dm_inst = module.get_dmInst_ptr()
        pydb = module.pydb(
            dm_inst,
            route_num_bins_x,
            route_num_bins_y,
            with_routability,
            with_sta=with_sta,
            timing_model_only=timing_model_only,
            **asdict(export_options),
        )
        sparse_timing_view = validate_timing_schema(pydb) if with_sta else None
        if bool(getattr(params, "timing_hold_enabled", False)):
            from .hold_schema import read_hold_timing_schema

            # Validate the independent min payload before any placement model
            # is constructed. A setup-only PyPlaceDB must never be accepted as
            # a hold-capable runtime by accident.
            hold_data = read_hold_timing_schema(pydb)
            timing_summary["hold_schema_version"] = 2
            timing_summary["hold_context_count"] = len(hold_data.contexts)
        return ECCPlaceIOBackend(
            module=module,
            workspace=workspace,
            dm_inst=dm_inst,
            pydb=pydb,
            backend_caps=ecc_backend_caps(
                timing=with_sta,
                parasitics_initialization=timing_summary.get(
                    "parasitics_initialization"
                ),
            ),
            timing_enabled=with_sta,
            hold_enabled=bool(getattr(params, "timing_hold_enabled", False)),
            timing_inputs=timing_inputs,
            route_num_bins_x=route_num_bins_x,
            route_num_bins_y=route_num_bins_y,
            with_routability=with_routability,
            export_options=export_options,
            sparse_timing_view=sparse_timing_view,
        )

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
        for node_id, target_id in enumerate(cell_ids):
            target_id = int(target_id)
            main_id = int(pydb.inst_main_id[node_id])
            if main_id < 0:
                if target_id >= 0:
                    raise ValueError(f"node {node_id} has no Liberty family")
                continue
            start = int(pydb.main_id_2_cell_id_start[main_id])
            end = int(pydb.main_id_2_cell_id_start[main_id + 1])
            current_id = (
                start + int(pydb.inst_libcell_offset[node_id])
                if current_cell_ids is None else int(current_cell_ids[node_id])
            )
            if target_id == current_id:
                continue
            if not start <= target_id < end or target_id >= len(cell_master_names):
                raise ValueError(f"node {node_id} sizing target is outside its Liberty family")
            node_ids.append(node_id)
            target_names.append(cell_master_names[target_id])
        if not node_ids:
            return {"ok": True, "accepted_count": 0, "requested_count": 0, "rejected_count": 0}
        try:
            summary = dict(pydb.apply_sizing(node_ids, target_names) or {})
        except AttributeError as exc:
            raise RuntimeError("ECC native PyPlaceDB has no sizing transaction") from exc
        summary["requested_count"] = len(node_ids)
        if (
            bool(summary.get("ok"))
            and int(summary.get("accepted_count", 0) or 0) == len(node_ids)
        ):
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
    def refresh(raw_db, *, refresh_mode="full_rebuild", rebuild_mode="full_rebuild"):
        from .refresh import RefreshController

        return RefreshController.refresh(
            raw_db,
            refresh_mode=refresh_mode,
            rebuild_mode=rebuild_mode,
        )
