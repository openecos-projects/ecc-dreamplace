"""Full native ECC timing refresh after a committed database mutation."""

import os
from dataclasses import asdict

from .place_io import PlaceIOFunction
from .timing_schema import validate_timing_schema


def _export_pydb(raw_db):
    dm_inst = raw_db.module.get_dmInst_ptr()
    new_pydb = raw_db.module.pydb(
        dm_inst,
        raw_db.route_num_bins_x,
        raw_db.route_num_bins_y,
        raw_db.with_routability,
        with_sta=True,
        **asdict(raw_db.export_options),
    )
    return dm_inst, new_pydb


def _reload_from_committed_def(raw_db):
    """Rebuild STA from a committed DEF when in-memory refresh duplicates endpoints."""

    module = raw_db.module
    native = getattr(module, "ecc", module)
    if not callable(getattr(module, "refresh_place_timing", None)):
        raise RuntimeError("ECC module has no refresh_place_timing for DEF fallback")
    workspace = str(raw_db.workspace or ".")
    os.makedirs(workspace, exist_ok=True)
    def_path = os.path.join(workspace, "committed_refresh.def")
    module.def_save(def_path)
    destroy_sta = getattr(native, "destroy_sta", None)
    if callable(destroy_sta):
        destroy_sta()
    if native.def_init(def_path) is not True:
        raise RuntimeError(f"ECC failed to reload committed DEF: {def_path}")
    # ``refresh_place_timing`` destroys the active STA first.  The caller
    # already destroyed it before re-reading the DEF, so clear the wrapper's
    # bookkeeping to avoid a second destroy on an uninitialized instance.
    if hasattr(module, "_place_timing_active"):
        module._place_timing_active = False
    summary = dict(module.refresh_place_timing(raw_db.timing_inputs) or {})
    dm_inst, new_pydb = _export_pydb(raw_db)
    return new_pydb, dm_inst, summary, def_path


class RefreshController:
    """Own the invalidate/rebuild/new-PyPlaceDB boundary for ECC."""

    @staticmethod
    def refresh(raw_db, *, refresh_mode="full_rebuild", rebuild_mode="full_rebuild"):
        if refresh_mode != "full_rebuild" or rebuild_mode != "full_rebuild":
            raise ValueError(
                "ECC timing refresh only supports refresh_mode=rebuild_mode=full_rebuild"
            )
        if not raw_db.timing_enabled:
            raise RuntimeError("ECC timing refresh requires an active timing PyPlaceDB")

        old_pydb = raw_db.pydb
        try:
            summary = dict(raw_db.module.refresh_place_timing(raw_db.timing_inputs) or {})
            dm_inst, new_pydb = _export_pydb(raw_db)
            sparse_timing_view = validate_timing_schema(new_pydb)
        except RuntimeError as error:
            if "endpoint identities are invalid" not in str(error):
                raise
            new_pydb, dm_inst, summary, def_path = _reload_from_committed_def(raw_db)
            sparse_timing_view = validate_timing_schema(new_pydb)
            summary["refresh_path"] = "committed_def_reload"
            summary["committed_def_path"] = def_path
        else:
            summary["refresh_path"] = "native_in_memory"

        raw_db.dm_inst = dm_inst
        raw_db.pydb = new_pydb
        raw_db.sparse_timing_view = sparse_timing_view
        raw_db.refresh_generation += 1
        fresh_rc = dict(summary.get("rcx") or {}).get("status") == "ok"
        PlaceIOFunction._promote(
            raw_db,
            supports_committed_refresh=fresh_rc,
            sta_state_status=(
                "validated_native_full_rebuild" if fresh_rc else "native_rebuilt_without_fresh_rc"
            ),
        )
        summary.update(
            {
                "status": "ok",
                "refresh_mode": refresh_mode,
                "rebuild_mode": rebuild_mode,
                "generation": raw_db.refresh_generation,
                "old_pydb_id": id(old_pydb),
                "new_pydb_id": id(new_pydb),
                "pydb_replaced": old_pydb is not new_pydb,
                "timing_schema_version": int(new_pydb.timing_schema_version),
                "backend_capabilities": raw_db.backend_caps.to_dict(),
                "pydb_counts": {
                    "nodes": int(getattr(new_pydb, "num_nodes", 0) or 0),
                    "pins": len(getattr(new_pydb, "pin_names", ())),
                    "nets": len(getattr(new_pydb, "net_names", ())),
                },
            }
        )
        return new_pydb, summary
