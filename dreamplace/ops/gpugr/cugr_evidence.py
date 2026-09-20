"""Evidence records for one structured CUGR backend invocation."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from .evidence import terminal_status, write_evidence_record


def _existing_inputs(input_def, lefs: Sequence[str]) -> dict[str, str]:
    artifacts = {}
    if input_def:
        artifacts["input_def"] = str(input_def)
    for index, lef in enumerate(lefs):
        artifacts[f"input_lef_{index}"] = str(lef)
    return artifacts


def _metric_snapshot(metrics: Mapping | None) -> dict:
    if not metrics:
        return {}
    # Keep the record JSON-native and bounded. The complete maps/routes remain
    # separate artifacts; evidence needs the scalar routing/provenance fields.
    keys = (
        "num_overflow_nets",
        "gr_wirelength",
        "gr_num_vias",
        "gr_est_shorts",
        "elapsed_sec",
        "cugr_total_passes",
        "cugr_threads",
        "cugr_pass0_worker_count",
        "cugr_route_grid_x",
        "cugr_route_grid_y",
        "cugr_route_failed_count",
        "cugr_num_nets",
        "cugr_fixed_usage_sha256",
        "cugr_movable_usage_sha256",
        "cugr_source_commit",
        "cugr_source_dirty",
        "cugr_session_cache_hit",
        "cugr_session_mode",
        "cugr_invalid_layer_warning_count",
        "cugr_peak_rss_mb",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def write_cugr_evidence(
    *,
    result_dir,
    run_id: str,
    status: str | None = None,
    returncode: int | None = None,
    terminal_marker: bool = False,
    timed_out: bool = False,
    input_def: str = "",
    lefs: Sequence[str] = (),
    artifact_paths: Mapping | None = None,
    metrics: Mapping | None = None,
    requested_backend: str = "cugr",
    resolved_backend: str = "cugr",
    source_commit: str = "",
    source_dirty: bool | None = None,
    effective_params: Mapping | None = None,
    error: str = "",
):
    """Write one CUGR gate record and return its path.

    A caller may pass an explicit status for an exception path. Successful
    records still use ``terminal_status`` so a missing native terminal marker
    cannot be mistaken for a clean result.
    """

    if status is None:
        status = terminal_status(
            returncode=returncode,
            timed_out=timed_out,
            terminal_marker=terminal_marker,
        )
    artifacts = _existing_inputs(input_def, lefs)
    artifacts.update({str(name): str(path) for name, path in (artifact_paths or {}).items()})
    record = {
        "claim_level": "L1",
        "gate": "cugr",
        "run_id": run_id,
        "status": status,
        "command": ["in-process", "cpp_to_py.cpybin.cugr.run"],
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "effective_params": dict(effective_params or {}),
        "source": {
            "commit": source_commit,
            "dirty": source_dirty,
        },
        "process": {
            "returncode": returncode,
            "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
            "timed_out": bool(timed_out),
            "terminal_marker": bool(terminal_marker),
        },
        "metrics": _metric_snapshot(metrics),
    }
    if error:
        record["error"] = str(error)
    return write_evidence_record(
        Path(result_dir) / "evidence",
        record,
        artifact_paths=artifacts,
    )


__all__ = ["write_cugr_evidence"]
