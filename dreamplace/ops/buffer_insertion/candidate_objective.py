"""Electrical metrics and the exact candidate phase restore objective."""

from .discrete_virtual_buffer_runner import _metric_snapshot


def objective_snapshot(model, loss, metrics):
    """Return raw electrical metrics and the exact PlaceObj restore score."""

    snapshot = _metric_snapshot(loss, metrics)
    terms = dict(getattr(model, "last_timing_objective_terms", {}) or {})
    raw_terms = dict(terms.get("raw_terms", {}) or {})
    weighted_terms = dict(terms.get("weighted_terms", {}) or {})
    report_aligned = dict(terms.get("report_aligned_terms", {}) or {})
    timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
    timing_metrics = dict(getattr(model, "_diff_opt_last_timing_metrics", {}) or {})
    slew = report_aligned.get("slew_violation_ns")
    if slew is None and raw_terms.get("slew") is not None:
        slew = float(raw_terms["slew"]) / 1000.0
    if slew is None and timing_op is not None:
        slew = getattr(timing_op, "last_total_slew_violation", None)
    cap = report_aligned.get("cap_violation_pF")
    if cap is None and raw_terms.get("cap") is not None:
        cap = float(raw_terms["cap"]) / 1000.0
    if cap is None and timing_op is not None:
        cap = getattr(timing_op, "last_total_cap_violation", None)
    total_slew = (
        None if timing_op is None else getattr(timing_op, "last_total_slew_violation", None)
    )
    total_cap = None if timing_op is None else getattr(timing_op, "last_total_cap_violation", None)
    snapshot.update(
        {
            "slew_violation": None if slew is None else float(slew),
            "cap_violation": None if cap is None else float(cap),
            "total_slew_violation": (None if total_slew is None else float(total_slew)),
            "total_cap_violation": None if total_cap is None else float(total_cap),
            "wns": timing_metrics.get("wns", snapshot.get("wns")),
            "tns": timing_metrics.get("tns", snapshot.get("tns")),
            "raw_terms": raw_terms,
            "weighted_terms": weighted_terms,
            "coefficients": dict(terms.get("coefficients", {}) or {}),
            "weights": dict(terms.get("weights", {}) or {}),
            "restore_score": snapshot["loss"],
        }
    )
    return snapshot
