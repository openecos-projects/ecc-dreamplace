import torch


def _float_or_zero(value):
    if value is None:
        return 0.0
    return float(value)


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


def _edge_cap_value(edge_rc, parent, child):
    value = (edge_rc or {}).get((int(parent), int(child)))
    if value is None:
        value = (edge_rc or {}).get((str(int(parent)), str(int(child))))
    if isinstance(value, dict):
        return _float_or_zero(value.get("c"))
    return 0.0


def _net_edge_cap_sum(net):
    rc_tree = dict(net.get("rc_tree", {}) or {})
    children_by_node = {
        int(node): [int(child) for child in children]
        for node, children in (rc_tree.get("children_by_node", {}) or {}).items()
    }
    edge_rc = rc_tree.get("edge_rc", {}) or {}
    total = 0.0
    for parent, children in children_by_node.items():
        for child in children:
            total += _edge_cap_value(edge_rc, parent, child)
    return total


def _segment_upstream_fraction(segment_row):
    fractions = (segment_row.get("fractions_by_repeater_count", {}) or {}).get("1")
    if fractions:
        return float(fractions[0])
    px = float(segment_row.get("parent_x_dbu", 0.0))
    py = float(segment_row.get("parent_y_dbu", 0.0))
    cx = float(segment_row.get("child_x_dbu", px))
    cy = float(segment_row.get("child_y_dbu", py))
    dx = cx - px
    dy = cy - py
    if dx == 0.0 and dy == 0.0:
        return 0.5
    qx = round(px + 0.5 * dx)
    qy = round(py + 0.5 * dy)
    length_sq = dx * dx + dy * dy
    ratio = ((float(qx) - px) * dx + (float(qy) - py) * dy) / length_sq
    return min(1.0, max(0.0, float(ratio)))


def zero_retained_upstream_cap(segment_count, *, dtype=torch.float64, device=None):
    return torch.zeros(int(segment_count), dtype=dtype, device=device)


def retained_cap_from_target_parent_load(
    *,
    target_parent_visible_load,
    analytic_parent_visible_load,
):
    target = torch.as_tensor(target_parent_visible_load)
    analytic = torch.as_tensor(
        analytic_parent_visible_load,
        dtype=target.dtype,
        device=target.device,
    )
    return torch.clamp(target - analytic, min=0.0)


def retained_cap_from_net_edge_cap_fraction(
    nets,
    segment_rows,
    *,
    alpha=1.0,
    dtype=torch.float64,
    device=None,
):
    edge_cap_sum_by_net = {
        _net_id(net, index): _net_edge_cap_sum(net)
        for index, net in enumerate(list(nets or []))
    }
    values = []
    for row in tuple(segment_rows or ()):
        net_id = int(row.get("net_id", -1))
        upstream_fraction = _segment_upstream_fraction(row)
        retained = float(alpha) * edge_cap_sum_by_net.get(net_id, 0.0) * upstream_fraction
        values.append(retained)
    return torch.tensor(values, dtype=dtype, device=device)


def estimate_retained_cap_candidates_from_probe_row(probe_row):
    edge_rc = dict(probe_row.get("edge_rc", {}) or {})
    edge_cap = _float_or_zero(edge_rc.get("c"))
    fractions = (probe_row.get("fractions_by_repeater_count", {}) or {}).get("1")
    upstream_fraction = float(fractions[0]) if fractions else 0.5
    local_upstream_wire_cap = edge_cap * upstream_fraction
    local_parent_half_wire_cap = 0.5 * edge_cap * upstream_fraction
    zero_z_root_cap = _float_or_zero(probe_row.get("driver_zero_z_net_cap"))
    buffer_input_cap = _float_or_zero(probe_row.get("static_buffer_input_cap"))
    downstream_load = _float_or_zero(probe_row.get("segment_downstream_load"))
    root_decomposition = dict(probe_row.get("root_load_decomposition", {}) or {})
    edge_cap_sum = _float_or_zero(root_decomposition.get("edge_cap_sum"))
    total_node_cap_sum = _float_or_zero(root_decomposition.get("total_node_cap_sum"))
    root_load_manual = _float_or_zero(
        root_decomposition.get("manual_root_load_node_plus_edge")
    )
    if root_load_manual == 0.0:
        root_load_manual = zero_z_root_cap

    return {
        "zero": 0.0,
        "local_upstream_wire_cap": local_upstream_wire_cap,
        "local_parent_half_wire_cap": local_parent_half_wire_cap,
        "edge_cap_sum_fraction": edge_cap_sum * upstream_fraction,
        "root_load_fraction": root_load_manual * upstream_fraction,
        "zero_z_minus_buffer_input": max(0.0, zero_z_root_cap - buffer_input_cap),
        "zero_z_minus_downstream_load": max(0.0, zero_z_root_cap - downstream_load),
        "node_cap_sum_fraction": total_node_cap_sum * upstream_fraction,
    }


def summarize_retained_cap_candidate_errors(
    *,
    probe_row,
    opensta_upstream_net_cap,
):
    target = float(opensta_upstream_net_cap)
    analytic_parent_load = _float_or_zero(
        probe_row.get(
            "analytic_transfer_upstream_visible_input_cap",
            probe_row.get("segment_upstream_visible_input_cap"),
        )
    )
    candidates = estimate_retained_cap_candidates_from_probe_row(probe_row)
    rows = []
    for name, retained_cap in sorted(candidates.items()):
        predicted_parent_load = analytic_parent_load + float(retained_cap)
        error = predicted_parent_load - target
        abs_error = abs(error)
        rows.append(
            {
                "name": str(name),
                "retained_cap": float(retained_cap),
                "predicted_parent_load": float(predicted_parent_load),
                "error": float(error),
                "abs_error": float(abs_error),
                "ratio_to_target": (
                    None if target == 0.0 else float(predicted_parent_load / target)
                ),
            }
        )
    rows.sort(key=lambda row: row["abs_error"])
    return {
        "segment_id": int(probe_row["global_segment_id"]),
        "net_name": str(probe_row.get("net_name", "")),
        "target_opensta_upstream_net_cap": target,
        "analytic_parent_visible_load": analytic_parent_load,
        "candidate_count": int(len(rows)),
        "best_candidate": rows[0] if rows else None,
        "candidates": rows,
        "status": "candidate_estimator_calibration",
    }


def fit_alpha_for_retained_cap_samples(samples):
    """Fit a single alpha for base retained-cap estimates.

    Each sample must provide:
      - analytic_parent_visible_load
      - target_parent_visible_load
      - base_retained_cap

    The fitted model is:
      predicted_parent_visible_load =
        analytic_parent_visible_load + alpha * base_retained_cap
    """

    rows = []
    numerator = 0.0
    denominator = 0.0
    for sample in list(samples or []):
        analytic = float(sample["analytic_parent_visible_load"])
        target = float(sample["target_parent_visible_load"])
        base = float(sample["base_retained_cap"])
        needed = target - analytic
        numerator += base * needed
        denominator += base * base
        rows.append(
            {
                "segment_id": (
                    int(sample["segment_id"])
                    if sample.get("segment_id") is not None
                    else None
                ),
                "net_name": str(sample.get("net_name", "")),
                "analytic_parent_visible_load": analytic,
                "target_parent_visible_load": target,
                "base_retained_cap": base,
                "needed_retained_cap": needed,
            }
        )
    alpha = 0.0 if denominator == 0.0 else numerator / denominator
    squared_error = 0.0
    abs_errors = []
    for row in rows:
        predicted = row["analytic_parent_visible_load"] + alpha * row["base_retained_cap"]
        error = predicted - row["target_parent_visible_load"]
        abs_error = abs(error)
        squared_error += error * error
        abs_errors.append(abs_error)
        row["alpha"] = alpha
        row["predicted_parent_visible_load"] = predicted
        row["error"] = error
        row["abs_error"] = abs_error
        row["ratio_to_target"] = (
            None
            if row["target_parent_visible_load"] == 0.0
            else predicted / row["target_parent_visible_load"]
        )
    count = len(rows)
    return {
        "status": "ok" if count else "empty",
        "sample_count": int(count),
        "alpha": float(alpha),
        "rmse": (squared_error / count) ** 0.5 if count else None,
        "mean_abs_error": sum(abs_errors) / count if count else None,
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "samples": rows,
    }


def alpha_sample_from_probe_row(
    *,
    probe_row,
    target_parent_visible_load,
    base_candidate_name="edge_cap_sum_fraction",
):
    candidates = estimate_retained_cap_candidates_from_probe_row(probe_row)
    if base_candidate_name not in candidates:
        raise ValueError(f"unknown retained-cap candidate: {base_candidate_name}")
    return {
        "segment_id": int(probe_row["global_segment_id"]),
        "net_name": str(probe_row.get("net_name", "")),
        "analytic_parent_visible_load": _float_or_zero(
            probe_row.get(
                "analytic_transfer_upstream_visible_input_cap",
                probe_row.get("segment_upstream_visible_input_cap"),
            )
        ),
        "target_parent_visible_load": float(target_parent_visible_load),
        "base_retained_cap": float(candidates[base_candidate_name]),
        "base_candidate_name": str(base_candidate_name),
    }
