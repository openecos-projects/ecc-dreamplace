from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
import torch


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _extract_numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    numeric_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric_metrics[key] = _to_float(value)
        except (TypeError, ValueError):
            continue
    return numeric_metrics


@dataclass
class InflationGeometrySnapshot:
    target_density: float
    movable_area: float
    filler_area: float
    total_place_area: float
    whitespace_area: float
    num_filler_nodes: int


@dataclass
class InflationRoundRecord:
    round_idx: int
    stage_idx: int
    iteration: int
    trigger_overflow: float
    route_map_source: str
    adjust_area_flag_before: bool
    adjust_route_area_flag_before: bool
    adjust_pin_area_flag_before: bool
    before: Optional[InflationGeometrySnapshot] = None
    after: Optional[InflationGeometrySnapshot] = None
    adjust_area_flag_after: Optional[bool] = None
    adjust_route_area_flag_after: Optional[bool] = None
    adjust_pin_area_flag_after: Optional[bool] = None
    gr_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    notes: str = ""
    position_snapshot: Optional[torch.Tensor] = None


@dataclass
class InflationState:
    enabled: bool
    controller_mode: str
    route_round_idx: int = 0
    stage_idx: int = 0
    num_area_adjust: int = 0
    target_area: Optional[float] = None
    baseline: Optional[InflationGeometrySnapshot] = None
    current_snapshot: Optional[InflationGeometrySnapshot] = None
    current_round: Optional[InflationRoundRecord] = None
    round_records: List[InflationRoundRecord] = field(default_factory=list)
    best_gr_metrics: Dict[str, float] = field(default_factory=dict)
    selected_round_idx: Optional[int] = None
    selected_metric_name: Optional[str] = None
    selected_metric_value: Optional[float] = None


def is_xplace_outer_loop_enabled(params) -> bool:
    return bool(
        getattr(params, "routability_opt_flag", False)
        and getattr(params, "xplace_style_inflation_flag", False)
    )


def get_inflation_round_limit(params) -> int:
    if is_xplace_outer_loop_enabled(params):
        return int(getattr(params, "xplace_inflation_max_rounds", getattr(params, "max_num_area_adjust", 0)))
    return int(getattr(params, "max_num_area_adjust", 0))


def capture_inflation_snapshot(data_collections, placedb) -> InflationGeometrySnapshot:
    with torch.no_grad():
        movable_area = _to_float(
            (
                data_collections.node_size_x[: placedb.num_movable_nodes]
                * data_collections.node_size_y[: placedb.num_movable_nodes]
            ).sum()
        )
        if placedb.num_filler_nodes > 0:
            filler_area = _to_float(
                (
                    data_collections.node_size_x[-placedb.num_filler_nodes :]
                    * data_collections.node_size_y[-placedb.num_filler_nodes :]
                ).sum()
            )
        else:
            filler_area = 0.0
        target_density = max(
            _to_float(getattr(data_collections, "target_density", None), default=1.0),
            1e-12,
        )
        total_place_area = (movable_area + filler_area) / target_density
        whitespace_area = total_place_area - movable_area
        return InflationGeometrySnapshot(
            target_density=target_density,
            movable_area=movable_area,
            filler_area=filler_area,
            total_place_area=total_place_area,
            whitespace_area=whitespace_area,
            num_filler_nodes=int(placedb.num_filler_nodes),
        )


def create_inflation_state(params, placedb, data_collections) -> InflationState:
    enabled = bool(getattr(params, "routability_opt_flag", False))
    controller_mode = (
        "xplace_outer_loop"
        if bool(getattr(params, "xplace_style_inflation_flag", False))
        else "legacy_area_adjust"
    )
    baseline = capture_inflation_snapshot(data_collections, placedb) if enabled else None
    target_area = (
        baseline.movable_area + baseline.filler_area
        if baseline and bool(getattr(params, "xplace_inflation_use_target_area", 1))
        else None
    )
    return InflationState(
        enabled=enabled,
        controller_mode=controller_mode,
        stage_idx=0,
        num_area_adjust=0,
        target_area=target_area,
        baseline=baseline,
        current_snapshot=baseline,
    )


def ensure_inflation_state(params, placedb, data_collections, stage_idx: int = 0) -> InflationState:
    state = getattr(data_collections, "inflation_state", None)
    if state is None:
        state = create_inflation_state(params, placedb, data_collections)
        data_collections.inflation_state = state
    state.stage_idx = int(stage_idx)
    if state.enabled:
        if state.baseline is None:
            state.baseline = capture_inflation_snapshot(data_collections, placedb)
        state.current_snapshot = capture_inflation_snapshot(data_collections, placedb)
    return state


def begin_inflation_round(
    state: InflationState,
    data_collections,
    placedb,
    pos: Optional[torch.Tensor],
    round_idx: int,
    stage_idx: int,
    iteration: int,
    overflow: Any,
    route_map_source: str,
    adjust_area_flag: bool,
    adjust_route_area_flag: bool,
    adjust_pin_area_flag: bool,
    notes: str = "",
) -> InflationRoundRecord:
    record = InflationRoundRecord(
        round_idx=int(round_idx),
        stage_idx=int(stage_idx),
        iteration=int(iteration),
        trigger_overflow=_to_float(overflow),
        route_map_source=str(route_map_source),
        adjust_area_flag_before=bool(adjust_area_flag),
        adjust_route_area_flag_before=bool(adjust_route_area_flag),
        adjust_pin_area_flag_before=bool(adjust_pin_area_flag),
        before=capture_inflation_snapshot(data_collections, placedb),
        notes=notes,
        position_snapshot=pos.detach().clone() if pos is not None else None,
    )
    state.current_round = record
    state.current_snapshot = record.before
    state.route_round_idx = max(state.route_round_idx, record.round_idx + 1)
    return record


def finish_inflation_round(
    state: InflationState,
    data_collections,
    placedb,
    adjust_area_flag: bool,
    adjust_route_area_flag: bool,
    adjust_pin_area_flag: bool,
    status: str,
    gr_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[InflationRoundRecord]:
    record = state.current_round
    if record is None:
        return None

    record.after = capture_inflation_snapshot(data_collections, placedb)
    record.adjust_area_flag_after = bool(adjust_area_flag)
    record.adjust_route_area_flag_after = bool(adjust_route_area_flag)
    record.adjust_pin_area_flag_after = bool(adjust_pin_area_flag)
    if gr_metrics:
        record.gr_metrics.update(_extract_numeric_metrics(gr_metrics))
    record.status = status
    state.current_snapshot = record.after
    if status == "applied":
        state.num_area_adjust += 1
    state.round_records.append(record)
    state.current_round = None
    return record


def _metric_aliases(metric_name: str) -> List[str]:
    aliases = {
        "est_shorts": ["est_shorts", "gr_est_shorts"],
        "gr_est_shorts": ["gr_est_shorts", "est_shorts"],
        "num_overflow_nets": ["num_overflow_nets", "overflow_nets", "num_ovfl_nets"],
        "overflow_nets": ["overflow_nets", "num_overflow_nets", "num_ovfl_nets"],
        "gr_wirelength": ["gr_wirelength", "wirelength", "gr_wl"],
        "gr_num_vias": ["gr_num_vias", "num_vias", "gr_vias"],
    }
    return aliases.get(metric_name, [metric_name])


def get_round_metric(record: InflationRoundRecord, metric_name: str) -> Optional[float]:
    for key in _metric_aliases(metric_name):
        value = record.gr_metrics.get(key)
        if value is not None:
            return float(value)
    return None


def select_best_gr_solution(state: InflationState, metric_name: str = "est_shorts") -> Optional[InflationRoundRecord]:
    if state is None:
        return None
    candidates = [
        record
        for record in state.round_records
        if record.status in ("applied", "stopped")
        and record.stage_idx == state.stage_idx
        and record.position_snapshot is not None
        and get_round_metric(record, metric_name) is not None
    ]
    if not candidates:
        return None
    best_record = min(
        candidates,
        key=lambda record: (
            get_round_metric(record, metric_name),
            record.trigger_overflow,
            record.round_idx,
        ),
    )
    state.selected_round_idx = best_record.round_idx
    state.selected_metric_name = metric_name
    state.selected_metric_value = get_round_metric(best_record, metric_name)
    return best_record


def replay_best_gr_solution(
    state: InflationState,
    pos: torch.Tensor,
    metric_name: str = "est_shorts",
) -> Optional[InflationRoundRecord]:
    best_record = select_best_gr_solution(state, metric_name=metric_name)
    if best_record is None or best_record.position_snapshot is None:
        return None
    with torch.no_grad():
        pos.data.copy_(best_record.position_snapshot.data.to(device=pos.device, dtype=pos.dtype))
    return best_record


def _detect_low_utilization(
    params,
    state: Optional[InflationState],
    placedb,
    data_collections,
    gr_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = (
        capture_inflation_snapshot(data_collections, placedb)
        if state is None or state.current_snapshot is None
        else state.current_snapshot
    )
    overflow_nets = 0.0
    if gr_metrics:
        for key in _metric_aliases("num_overflow_nets"):
            value = gr_metrics.get(key)
            if value is not None:
                overflow_nets = _to_float(value)
                break
    total_nets = max(int(getattr(placedb, "num_nets", 0)), 1)
    filler_ratio = snapshot.filler_area / max(snapshot.movable_area, 1e-12)
    overflow_ratio = overflow_nets / total_nets
    triggered = bool(
        getattr(params, "xplace_inflation_dynamic_target_density_flag", 1)
        and snapshot.filler_area > 0
        and filler_ratio > _to_float(getattr(params, "xplace_inflation_low_util_filler_ratio", 1.10), 1.10)
        and overflow_ratio > _to_float(getattr(params, "xplace_inflation_low_util_overflow_ratio", 0.04), 0.04)
    )
    return {
        "triggered": triggered,
        "snapshot": snapshot,
        "filler_ratio": filler_ratio,
        "overflow_nets": overflow_nets,
        "overflow_ratio": overflow_ratio,
        "target_density_before": _to_float(
            getattr(data_collections, "target_density", None), snapshot.target_density
        ),
    }


def get_adjust_node_area_impl(adjust_node_area_op):
    return getattr(adjust_node_area_op, "_xplace_adjust_node_area_impl", None)


def prepare_low_util_inflation(
    params,
    state: Optional[InflationState],
    placedb,
    data_collections,
    adjust_node_area_op,
    gr_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context = _detect_low_utilization(params, state, placedb, data_collections, gr_metrics)
    adjust_node_area_impl = get_adjust_node_area_impl(adjust_node_area_op)
    context["expanded_budget"] = False
    if not context["triggered"] or adjust_node_area_impl is None:
        return context

    original_total_whitespace_area = _to_float(
        getattr(adjust_node_area_impl, "total_whitespace_area", None),
        default=context["snapshot"].whitespace_area,
    )
    expanded_total_whitespace_area = max(
        original_total_whitespace_area,
        10.0 * max(context["snapshot"].whitespace_area, 0.0),
    )
    adjust_node_area_impl.total_whitespace_area = expanded_total_whitespace_area
    context["expanded_budget"] = True
    context["original_total_whitespace_area"] = original_total_whitespace_area
    context["expanded_total_whitespace_area"] = expanded_total_whitespace_area
    logging.info(
        "Low-utilization detected before adjust_node_area: filler/movable=%.3f overflow_net_ratio=%.3f #OvflNets=%.0f; expand inflation area budget from %.3E to %.3E",
        context["filler_ratio"],
        context["overflow_ratio"],
        context["overflow_nets"],
        original_total_whitespace_area,
        expanded_total_whitespace_area,
    )
    return context


def restore_low_util_inflation(adjust_node_area_op, context: Optional[Dict[str, Any]]) -> None:
    if not context or not context.get("expanded_budget"):
        return
    adjust_node_area_impl = get_adjust_node_area_impl(adjust_node_area_op)
    if adjust_node_area_impl is None:
        return
    adjust_node_area_impl.total_whitespace_area = context["original_total_whitespace_area"]


def _apply_filler_target_area(
    data_collections,
    placedb,
    pos: torch.Tensor,
    desired_filler_area: float,
) -> Dict[str, float]:
    if placedb.num_filler_nodes <= 0:
        return {
            "remaining_fillers": 0.0,
            "filler_area_after": 0.0,
        }

    filler_lhs = placedb.num_nodes - placedb.num_filler_nodes
    filler_rhs = placedb.num_nodes
    num_nodes = int(pos.numel() / 2)

    filler_size_x = data_collections.node_size_x[filler_lhs:filler_rhs]
    filler_size_y = data_collections.node_size_y[filler_lhs:filler_rhs]
    filler_center_x = pos.data[filler_lhs:filler_rhs] + filler_size_x * 0.5
    filler_center_y = pos.data[num_nodes + filler_lhs : num_nodes + filler_rhs] + filler_size_y * 0.5

    original_filler_size_x = data_collections.original_node_size_x[filler_lhs:filler_rhs]
    original_filler_size_y = data_collections.original_node_size_y[filler_lhs:filler_rhs]
    original_filler_area = original_filler_size_x * original_filler_size_y
    total_original_filler_area = _to_float(original_filler_area.sum())
    clamped_filler_area = max(0.0, min(desired_filler_area, total_original_filler_area))

    filler_size_x.zero_()
    filler_size_y.zero_()

    remaining_fillers = 0
    if clamped_filler_area > 0 and total_original_filler_area > 0:
        cumulative_filler_area = torch.cumsum(original_filler_area, dim=0)
        threshold = torch.tensor(
            clamped_filler_area,
            device=cumulative_filler_area.device,
            dtype=cumulative_filler_area.dtype,
        )
        remaining_fillers = int(
            torch.searchsorted(cumulative_filler_area, threshold, right=False).item()
        ) + 1
        remaining_fillers = min(remaining_fillers, placedb.num_filler_nodes)
        filler_size_x[:remaining_fillers].copy_(original_filler_size_x[:remaining_fillers])
        filler_size_y[:remaining_fillers].copy_(original_filler_size_y[:remaining_fillers])
        kept_area = _to_float((filler_size_x[:remaining_fillers] * filler_size_y[:remaining_fillers]).sum())
        if kept_area > clamped_filler_area + 1e-12:
            filler_scale = (clamped_filler_area / kept_area) ** 0.5
            filler_size_x[:remaining_fillers].mul_(filler_scale)
            filler_size_y[:remaining_fillers].mul_(filler_scale)

    pos.data[filler_lhs:filler_rhs].copy_(filler_center_x - filler_size_x * 0.5)
    pos.data[num_nodes + filler_lhs : num_nodes + filler_rhs].copy_(
        filler_center_y - filler_size_y * 0.5
    )

    return {
        "remaining_fillers": float(remaining_fillers),
        "filler_area_after": _to_float((filler_size_x * filler_size_y).sum()),
    }


def apply_low_util_target_density(
    params,
    state: Optional[InflationState],
    data_collections,
    placedb,
    pos: torch.Tensor,
    context: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not context or not context.get("triggered"):
        return {}

    snapshot = capture_inflation_snapshot(data_collections, placedb)
    placeable_area = (
        state.baseline.total_place_area
        if state is not None and state.baseline is not None
        else snapshot.total_place_area
    )
    current_target_density = _to_float(getattr(data_collections, "target_density", None), snapshot.target_density)
    target_density_before = float(context.get("target_density_before", current_target_density))
    target_density_floor = _to_float(
        getattr(params, "xplace_inflation_target_density_floor", 0.4762), 0.4762
    )
    target_density_decay = _to_float(
        getattr(params, "xplace_inflation_target_density_decay", 0.85), 0.85
    )
    reduced_target_density = max(target_density_floor, target_density_decay * target_density_before)
    if reduced_target_density >= current_target_density - 1e-12:
        return {}

    desired_total_area = min(reduced_target_density * placeable_area, placeable_area)
    if state is not None and state.target_area is not None:
        desired_total_area = min(desired_total_area, float(state.target_area))
    desired_total_area = max(desired_total_area, snapshot.movable_area)
    desired_filler_area = min(
        max(desired_total_area - snapshot.movable_area, 0.0),
        snapshot.filler_area,
    )

    filler_result = _apply_filler_target_area(
        data_collections,
        placedb,
        pos,
        desired_filler_area=desired_filler_area,
    )
    actual_total_area = snapshot.movable_area + filler_result["filler_area_after"]
    actual_target_density = actual_total_area / max(placeable_area, 1e-12)
    with torch.no_grad():
        data_collections.target_density.fill_(actual_target_density)

    if state is not None:
        state.target_area = actual_total_area
        state.current_snapshot = capture_inflation_snapshot(data_collections, placedb)

    logging.info(
        "Low-utilization target-density adjustment: filler/movable=%.3f overflow_net_ratio=%.3f target_density %.6f -> %.6f target_area=%.3E filler_area=%.3E remaining_fillers=%d",
        context["filler_ratio"],
        context["overflow_ratio"],
        target_density_before,
        actual_target_density,
        actual_total_area,
        filler_result["filler_area_after"],
        int(filler_result["remaining_fillers"]),
    )
    return {
        "low_util_applied": 1.0,
        "low_util_filler_ratio": float(context["filler_ratio"]),
        "low_util_overflow_net_ratio": float(context["overflow_ratio"]),
        "low_util_target_density_before": float(target_density_before),
        "low_util_target_density_after": float(actual_target_density),
        "low_util_target_area": float(actual_total_area),
        "low_util_filler_area_after": float(filler_result["filler_area_after"]),
        "low_util_remaining_fillers": float(filler_result["remaining_fillers"]),
    }


def rollback_inflation_state(data_collections) -> None:
    with torch.no_grad():
        original_target_density = getattr(data_collections, "original_target_density", None)
        if original_target_density is not None:
            data_collections.target_density.copy_(original_target_density)
    state = getattr(data_collections, "inflation_state", None)
    if state is not None:
        state.current_round = None
        if state.baseline is not None:
            state.current_snapshot = state.baseline
            if state.target_area is not None:
                state.target_area = state.baseline.movable_area + state.baseline.filler_area


def should_trigger_xplace_inflation(params, num_area_adjust: int, overflow: Any) -> bool:
    return bool(
        is_xplace_outer_loop_enabled(params)
        and int(num_area_adjust) < get_inflation_round_limit(params)
        and _to_float(overflow) < _to_float(getattr(params, "node_area_adjust_overflow", 0.15), default=0.15)
    )


def run_xplace_style_inflation_round(*args, **kwargs) -> Dict[str, Any]:
    logging.info(
        "Xplace-style inflation controller skeleton is initialized, but the outer-loop execution path "
        "is not active in this PR. Falling back to the legacy area-adjust flow."
    )
    return {
        "applied": False,
        "status": "skeleton_only",
        "reason": "outer_loop_not_wired",
    }
