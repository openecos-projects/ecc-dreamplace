"""Capability-aware topology-mutation backends for buffering requests."""

import os


def write_native_def(placedb, filename):
    """Write the current native DB, shared by commits and route handoffs."""
    backend = _capabilities(placedb).get("backend")
    if backend == "openroad":
        return placedb.openroad_bridge.write_def(str(filename))
    if backend == "ecc":
        from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction

        return PlaceIOFunction.write_def(placedb.rawdb, str(filename))
    raise RuntimeError(f"Native DEF export is unsupported for backend {backend!r}")


def _capabilities(placedb):
    caps = dict(getattr(placedb, "backend_caps", {}) or {})
    if not caps:
        params = getattr(placedb, "params", None)
        backend = getattr(params, "place_io_engine", None)
        if backend:
            caps["backend"] = str(backend)
    return caps


def _unsupported_result(request, caps, capability):
    backend = str(caps.get("backend") or "unknown")
    return {
        "status": "unsupported",
        "reason": "unsupported_backend_capability",
        "backend": backend,
        "required_capability": capability,
        "backend_capabilities": dict(caps),
        "commit_enabled": bool(getattr(request, "commit_enabled", False)),
        "action_count": len(getattr(request, "actions", ()) or ()),
        "attempted_action_count": 0,
        "accepted_action_count": 0,
        "rejected_action_count": 0,
        "failed_action_count": 0,
        "topology_mutated": False,
        "refresh_required": False,
    }


class OpenRoadBufferMutationBackend:
    def __init__(self, placedb, caps):
        self.placedb = placedb
        self.caps = dict(caps)

    def capabilities(self):
        return dict(self.caps)

    def commit_buffers(self, request, *, output_dir=None):
        if not bool(self.caps.get("supports_buffer_commit", False)):
            return _unsupported_result(
                request,
                self.caps,
                "supports_buffer_commit",
            )
        from dreamplace.ops.buffer_insertion.buffering_lane import _action_digest

        if _action_digest(request.actions) != request.action_digest:
            raise ValueError("buffer commit request actions do not match action digest")
        if not bool(getattr(request, "commit_enabled", False)):
            return {
                "status": "disabled",
                "reason": "commit_not_requested",
                "backend": "openroad",
                "backend_capabilities": dict(self.caps),
                "commit_enabled": False,
                "action_count": len(getattr(request, "actions", ()) or ()),
                "attempted_action_count": 0,
                "accepted_action_count": 0,
                "rejected_action_count": 0,
                "failed_action_count": 0,
            }
        if bool(getattr(request, "is_noop", False)):
            return {
                "status": "skipped",
                "reason": "no_projected_buffer_actions",
                "backend": "openroad",
                "backend_capabilities": dict(self.caps),
                "commit_enabled": True,
                "action_count": 0,
                "attempted_action_count": 0,
                "accepted_action_count": 0,
                "rejected_action_count": 0,
                "failed_action_count": 0,
            }

        from dreamplace.ops.buffer_insertion.coordinate_backend import (
            commit_coordinate_buffer_actions,
        )

        summary = commit_coordinate_buffer_actions(
            request.actions,
            placedb=self.placedb,
            output_dir=output_dir,
        )
        summary = dict(summary)
        summary["backend"] = "openroad"
        summary["backend_capabilities"] = dict(self.caps)
        if (
            str(summary.get("status") or "") == "accepted"
            and int(summary.get("accepted_action_count", 0) or 0) > 0
        ):
            summary["refresh_required"] = True
            summary["refresh_mode"] = request.refresh_mode
            summary["rebuild_mode"] = request.rebuild_mode
            summary["continuation_policy"] = "rebuild_placer_outer_loop"
            bridge = getattr(self.placedb, "openroad_bridge", None)
            if request.committed_def_path and bridge is not None and hasattr(
                bridge, "write_def"
            ):
                try:
                    os.makedirs(
                        os.path.dirname(
                            os.path.abspath(request.committed_def_path)
                        ),
                        exist_ok=True,
                    )
                    write_native_def(self.placedb, request.committed_def_path)
                    summary["committed_def_path"] = request.committed_def_path
                except Exception as exc:  # pragma: no cover - diagnostic export path
                    summary["committed_def_error"] = str(exc)
        return summary

    def refresh(self, *, refresh_mode, rebuild_mode):
        if not bool(self.caps.get("supports_committed_refresh", False)):
            return _unsupported_result(
                None,
                self.caps,
                "supports_committed_refresh",
            )
        return self.placedb.refresh_from_openroad_bridge(
            refresh_mode=refresh_mode,
            rebuild_mode=rebuild_mode,
        )


class ECCBufferMutationBackend:
    def __init__(self, placedb, caps):
        self.placedb = placedb
        self.caps = dict(caps)

    def capabilities(self):
        return dict(self.caps)

    def commit_buffers(self, request, *, output_dir=None):
        del output_dir
        if not bool(getattr(self.placedb.rawdb, "timing_enabled", False)):
            return _unsupported_result(request, self.caps, "supports_buffer_commit")
        from dreamplace.ops.buffer_insertion.buffering_lane import _action_digest

        if _action_digest(request.actions) != request.action_digest:
            raise ValueError("buffer commit request actions do not match action digest")
        if not bool(getattr(request, "commit_enabled", False)):
            return {
                "status": "disabled",
                "reason": "commit_not_requested",
                "backend": "ecc",
                "backend_capabilities": dict(self.caps),
                "commit_enabled": False,
                "action_count": len(getattr(request, "actions", ()) or ()),
                "attempted_action_count": 0,
                "accepted_action_count": 0,
                "rejected_action_count": 0,
                "failed_action_count": 0,
                "topology_mutated": False,
                "refresh_required": False,
            }
        if bool(getattr(request, "is_noop", False)):
            return {
                "status": "skipped",
                "reason": "no_projected_buffer_actions",
                "backend": "ecc",
                "backend_capabilities": dict(self.caps),
                "commit_enabled": True,
                "action_count": 0,
                "attempted_action_count": 0,
                "accepted_action_count": 0,
                "rejected_action_count": 0,
                "failed_action_count": 0,
                "topology_mutated": False,
                "refresh_required": False,
            }

        import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

        summary = dict(
            placeio_ecc.PlaceIOFunction.apply_buffer_actions(
                self.placedb.rawdb,
                request.actions,
                request.action_digest,
            )
            or {}
        )
        summary.update(
            {
                "backend": "ecc",
                "backend_capabilities": dict(self.caps),
            }
        )
        if (
            str(summary.get("status") or "") in {"accepted", "ok"}
            and int(summary.get("accepted_action_count", 0) or 0) > 0
        ):
            summary.update(
                {
                    "status": "accepted",
                    "refresh_required": True,
                    "refresh_mode": "full_rebuild",
                    "rebuild_mode": "full_rebuild",
                    "continuation_policy": "rebuild_placer_outer_loop",
                }
            )
            if request.committed_def_path:
                write_native_def(self.placedb, request.committed_def_path)
            if getattr(request, "committed_verilog_path", None):
                self.placedb.rawdb.module.verilog_save(request.committed_verilog_path)
            self.placedb.backend_caps = placeio_ecc.PlaceIOFunction.backend_caps(
                self.placedb.rawdb
            )
            summary["backend_capabilities"] = dict(self.placedb.backend_caps)
        return summary

    def refresh(self, *, refresh_mode, rebuild_mode):
        if not bool(getattr(self.placedb.rawdb, "timing_enabled", False)):
            return _unsupported_result(None, self.caps, "supports_committed_refresh")
        result = self.placedb.refresh_from_ecc_backend(
            refresh_mode=refresh_mode,
            rebuild_mode=rebuild_mode,
        )
        self.caps = dict(self.placedb.backend_caps)
        return result


class UnsupportedBufferMutationBackend:
    def __init__(self, placedb, caps):
        self.placedb = placedb
        self.caps = dict(caps)

    def capabilities(self):
        return dict(self.caps)

    def commit_buffers(self, request, *, output_dir=None):
        del output_dir
        return _unsupported_result(
            request,
            self.caps,
            "supports_buffer_commit",
        )

    def refresh(self, *, refresh_mode, rebuild_mode):
        del refresh_mode, rebuild_mode
        return _unsupported_result(
            None,
            self.caps,
            "supports_committed_refresh",
        )


def physical_mutation_backend_for(placedb):
    caps = _capabilities(placedb)
    if str(caps.get("backend") or "").lower() == "openroad":
        return OpenRoadBufferMutationBackend(placedb, caps)
    if str(caps.get("backend") or "").lower() == "ecc":
        return ECCBufferMutationBackend(placedb, caps)
    return UnsupportedBufferMutationBackend(placedb, caps)
