"""Full native ECC timing refresh after a committed database mutation."""

from dataclasses import asdict, replace

from .place_io import PlaceIOFunction
from .timing_schema import validate_timing_schema


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
        raw_db.timing_enabled = False
        try:
            summary = dict(raw_db.module.refresh_place_timing(raw_db.timing_inputs) or {})
            dm_inst = raw_db.module.get_dmInst_ptr()
            new_pydb = raw_db.module.pydb(
                dm_inst,
                raw_db.route_num_bins_x,
                raw_db.route_num_bins_y,
                raw_db.with_routability,
                with_sta=True,
                timing_model_only=str(
                    summary.get("mode") or raw_db.timing_inputs.get("timing_mode", "model_only")
                )
                == "model_only",
                **asdict(raw_db.export_options),
            )
            sparse_timing_view = validate_timing_schema(new_pydb)
            if raw_db.hold_enabled:
                from .hold_schema import read_hold_timing_schema

                read_hold_timing_schema(new_pydb)
            fresh_rc = dict(summary.get("rcx") or {}).get("status") == "ok"
            capabilities = replace(
                raw_db.backend_caps,
                supports_committed_refresh=fresh_rc,
                sta_state_status=(
                    "validated_native_full_rebuild"
                    if fresh_rc
                    else "native_rebuilt_without_fresh_rc"
                ),
            )
            generation = raw_db.refresh_generation + 1
            summary.update(
                {
                    "status": "ok",
                    "refresh_mode": refresh_mode,
                    "rebuild_mode": rebuild_mode,
                    "generation": generation,
                    "old_pydb_id": id(old_pydb),
                    "new_pydb_id": id(new_pydb),
                    "pydb_replaced": old_pydb is not new_pydb,
                    "timing_schema_version": int(new_pydb.timing_schema_version),
                    "backend_capabilities": capabilities.to_dict(),
                    "pydb_counts": {
                        "nodes": int(getattr(new_pydb, "num_nodes", 0) or 0),
                        "pins": len(getattr(new_pydb, "pin_names", ())),
                        "nets": len(getattr(new_pydb, "net_names", ())),
                    },
                }
            )
        except Exception as error:
            PlaceIOFunction._invalidate_timing(raw_db, f"refresh failed: {error}")
            raise

        raw_db.dm_inst = dm_inst
        raw_db.pydb = new_pydb
        raw_db.sparse_timing_view = sparse_timing_view
        raw_db.refresh_generation = generation
        raw_db.backend_caps = capabilities
        raw_db.timing_enabled = True
        return new_pydb, summary
