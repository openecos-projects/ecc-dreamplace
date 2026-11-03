"""OpenROAD-compatible buffer characterization over ``placedb.pydb``."""

from __future__ import annotations

from .liberty import build_liberty_view_from_pydb

ARTIFACT = "rebuffer_buffer_characterization"
ARTIFACT_VERSION = 1


def _extension():
    try:
        import dreamplace.ops.rebuffer.rebuffer_cpp as native
    except ImportError as exc:
        raise RuntimeError(
            "rebuffer_cpp is not built; buffer characterization is unavailable"
        ) from exc
    return native


def characterize_buffers_from_pydb(pydb, *, canonical_target_load=False):
    """Characterize buffers with the native OpenROAD-compatible search."""

    liberty = build_liberty_view_from_pydb(pydb)
    if liberty.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                liberty.get("unsupported_reasons", ["liberty_status_not_ok"])
            ),
            "liberty_summary": dict(liberty.get("summary", {})),
        }
    native_options = {"canonical_target_load": True} if canonical_target_load else {}
    result = dict(_extension().characterize_buffers(liberty, **native_options))
    result.update(
        {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "liberty_summary": dict(liberty.get("summary", {})),
        }
    )
    return result


__all__ = ["characterize_buffers_from_pydb"]
