"""Full native ECC timing refresh after a committed database mutation."""

from dataclasses import asdict

from .place_io import PlaceIOFunction


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
        summary = dict(
            raw_db.module.refresh_place_timing(raw_db.timing_inputs) or {}
        )
        dm_inst = raw_db.module.get_dmInst_ptr()
        new_pydb = raw_db.module.pydb(
            dm_inst,
            raw_db.route_num_bins_x,
            raw_db.route_num_bins_y,
            raw_db.with_routability,
            with_sta=True,
            **asdict(raw_db.export_options),
        )
        if int(getattr(new_pydb, "timing_schema_version", 0) or 0) != 1:
            raise RuntimeError("ECC timing refresh produced an invalid timing PyPlaceDB")

        raw_db.dm_inst = dm_inst
        raw_db.pydb = new_pydb
        raw_db.refresh_generation += 1
        fresh_rc = dict(summary.get("rcx") or {}).get("status") == "ok"
        PlaceIOFunction._promote(
            raw_db,
            supports_committed_refresh=fresh_rc,
            sta_state_status=(
                "validated_native_full_rebuild"
                if fresh_rc else "native_rebuilt_without_fresh_rc"
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
