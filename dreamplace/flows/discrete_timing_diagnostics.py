"""Objective diagnostics for discrete timing phases."""


def objective_snapshot(model, loss):
    terms = dict(getattr(model, "last_timing_objective_terms", {}) or {})
    raw_terms = dict(terms.get("raw_terms", {}) or {})
    weighted_terms = dict(terms.get("weighted_terms", {}) or {})
    report_aligned = dict(terms.get("report_aligned_terms", {}) or {})
    timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
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
    timing_metrics = dict(getattr(model, "_diff_opt_last_timing_metrics", {}) or {})
    return {
        "score": float(terms.get("total_loss", loss.detach())),
        "wns": timing_metrics.get("wns"),
        "tns": timing_metrics.get("tns"),
        "slew_violation": None if slew is None else float(slew),
        "cap_violation": None if cap is None else float(cap),
        "total_slew_violation": (None if total_slew is None else float(total_slew)),
        "total_cap_violation": None if total_cap is None else float(total_cap),
        "raw_terms": raw_terms,
        "weighted_terms": weighted_terms,
    }


def phase_metrics(snapshot):
    """Return unweighted electrical diagnostics for an S/B phase boundary."""

    if not snapshot:
        snapshot = {}
    wns = snapshot.get("wns")
    tns = snapshot.get("tns")
    slew = snapshot.get("total_slew_violation", snapshot.get("slew_violation"))
    cap = snapshot.get("total_cap_violation")
    if cap is None and snapshot.get("cap_violation") is not None:
        cap = float(snapshot["cap_violation"])
    return {
        # PlaceObj timing_obj returns WNS/TNS in ps, and scalar violation
        # snapshots use ns and pF so they line up with external STA reports.
        "wns_ns": None if wns is None else float(wns) / 1000.0,
        "tns_ns": None if tns is None else float(tns) / 1000.0,
        "total_slew_violation_ns": None if slew is None else float(slew),
        "total_cap_violation_pf": None if cap is None else float(cap),
        "total_cap_violation_ff": None if cap is None else float(cap) * 1000.0,
    }
