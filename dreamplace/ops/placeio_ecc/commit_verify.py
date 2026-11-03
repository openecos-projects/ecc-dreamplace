"""Transport physical writes and Python TP observations to native verification.

C++ owns grouping, acceptance, rollback and retry decisions. Python supplies
only the existing placeio_ecc database and TimingPropagation boundaries.
"""

from .buffer_session import action_digest
from .repair_profile import measure_phase
from .timing_carrier import refresh_python_timing_carrier, require_carrier_schema

ARTIFACT = "placeio_ecc_commit_verified_buffers"
ARTIFACT_VERSION = 2


def timing_reader(pydb, *, config=None):
    """Evaluate a fresh physical timing carrier with Python TimingPropagation."""
    from ..rebuffer.timing_model import RebufferTimingModel
    from ..rebuffer.timing_profile import placement_timing_kwargs
    from .electrical_observation import electrical_observation

    resolved = dict(config or {})
    model = RebufferTimingModel(pydb, **placement_timing_kwargs(resolved))
    return {
        **model.baseline.as_dict(),
        "electrical": electrical_observation(
            model,
            max_fanout=resolved.get("max_fanout", 32),
            cap_margin=resolved.get("cap_margin", 0.0),
            slew_margin=resolved.get("slew_margin", 0.0),
        ),
    }


def commit_buffer_actions_verified(
    raw_db,
    actions,
    *,
    chunk_size=0,
    wns_margin_ps=0.0,
    tns_margin_ps=0.0,
    reader=timing_reader,
    profile=None,
):
    """Verify complete native trees, preserving chains across source nets."""
    from ..rebuffer import rebuffer_cpp
    from .place_io import PlaceIOFunction

    def apply(values):
        with measure_phase(profile, "native_buffer_commit_ms"):
            return dict(PlaceIOFunction.apply_buffer_actions(raw_db, values, action_digest(values)))

    def refresh():
        kwargs = {"profile": profile} if profile is not None else {}
        _, summary = refresh_python_timing_carrier(raw_db, **kwargs)
        return dict(summary)

    def counts():
        return {
            "nodes": len(raw_db.pydb.node_names),
            "pins": len(raw_db.pydb.pin_names),
            "nets": len(raw_db.pydb.net_names),
        }

    result = dict(
        rebuffer_cpp.run_verified_buffer_commit(
            [dict(action) for action in actions or ()],
            lambda: require_carrier_schema(raw_db.pydb),
            apply,
            refresh,
            lambda: dict(reader(raw_db.pydb)),
            counts,
            action_digest,
            chunk_size=int(chunk_size),
            wns_margin_ps=float(wns_margin_ps),
            tns_margin_ps=float(tns_margin_ps),
        )
    )
    # Preserve the historical timing_after schema for callers that compare
    # TimingSnapshot dictionaries; the native verifier keeps the richer
    # electrical observation in a sibling field and in each chunk record.
    for field in ("timing_before", "timing_after"):
        value = result.get(field)
        if isinstance(value, dict) and "electrical" in value:
            result[f"{field}_electrical"] = value.pop("electrical")
    result.update(
        artifact=ARTIFACT,
        artifact_version=ARTIFACT_VERSION,
        timing_evaluator="dreamplace_timing_propagation_after_native_commit",
        refresh_deferred=False,
        ecc_sta_run=False,
        external_sta_run=False,
    )
    return result


__all__ = ["ARTIFACT", "commit_buffer_actions_verified", "timing_reader"]
