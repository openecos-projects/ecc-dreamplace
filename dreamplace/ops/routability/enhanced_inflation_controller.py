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
    trigger_position_snapshot: Optional[torch.Tensor] = None
    inflated_position_snapshot: Optional[torch.Tensor] = None
    geometry_backup: Optional[Dict[str, Any]] = None
    movable_area_increment_ratio: Optional[float] = None
    min_area_increment_threshold: Optional[float] = None


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
    original_model_init_density: Optional[torch.Tensor] = None
    original_density_weight_grad_precond: Optional[torch.Tensor] = None
    original_quad_penalty_coeff: Optional[torch.Tensor] = None


def is_enhanced_inflation_enabled(params) -> bool:
    return bool(
        getattr(params, "routability_opt_flag", False)
        and getattr(params, "enhanced_inflation_flag", False)
    )


def get_inflation_round_limit(params) -> int:
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


def capture_inflation_geometry_backup(
    state: Optional[InflationState],
    data_collections,
) -> Dict[str, Any]:
    target_density = getattr(data_collections, "target_density", None)
    return {
        "node_size_x": data_collections.node_size_x.detach().clone(),
        "node_size_y": data_collections.node_size_y.detach().clone(),
        "pin_offset_x": data_collections.pin_offset_x.detach().clone(),
        "pin_offset_y": data_collections.pin_offset_y.detach().clone(),
        "target_density": (
            target_density.detach().clone()
            if isinstance(target_density, torch.Tensor)
            else None
        ),
        "target_area": (
            None
            if state is None or state.target_area is None
            else float(state.target_area)
        ),
    }


def maybe_capture_model_density_state(state: Optional[InflationState], model) -> None:
    if state is None or model is None:
        return
    init_density = getattr(model, "init_density", None)
    density_weight_grad_precond = getattr(model, "density_weight_grad_precond", None)
    quad_penalty_coeff = getattr(model, "quad_penalty_coeff", None)
    if state.original_model_init_density is None and isinstance(init_density, torch.Tensor):
        state.original_model_init_density = init_density.detach().clone()
    if state.original_density_weight_grad_precond is None and isinstance(
        density_weight_grad_precond, torch.Tensor
    ):
        state.original_density_weight_grad_precond = density_weight_grad_precond.detach().clone()
    if state.original_quad_penalty_coeff is None and isinstance(quad_penalty_coeff, torch.Tensor):
        state.original_quad_penalty_coeff = quad_penalty_coeff.detach().clone()


def restore_model_density_state(state: Optional[InflationState], model) -> None:
    if state is None or model is None:
        return
    init_density = state.original_model_init_density
    density_weight_grad_precond = state.original_density_weight_grad_precond
    quad_penalty_coeff = state.original_quad_penalty_coeff
    model.init_density = None if init_density is None else init_density.detach().clone()
    model.density_weight_grad_precond = (
        None
        if density_weight_grad_precond is None
        else density_weight_grad_precond.detach().clone()
    )
    model.quad_penalty_coeff = (
        None if quad_penalty_coeff is None else quad_penalty_coeff.detach().clone()
    )


def create_inflation_state(params, placedb, data_collections) -> InflationState:
    enabled = bool(getattr(params, "routability_opt_flag", False))
    controller_mode = (
        "enhanced_inflation"
        if bool(getattr(params, "enhanced_inflation_flag", False))
        else "legacy_area_adjust"
    )
    baseline = capture_inflation_snapshot(data_collections, placedb) if enabled else None
    target_area = (
        baseline.movable_area + baseline.filler_area
        if baseline and controller_mode == "enhanced_inflation"
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
    trigger_position_snapshot = pos.detach().clone() if pos is not None else None
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
        position_snapshot=trigger_position_snapshot,
        trigger_position_snapshot=trigger_position_snapshot,
        geometry_backup=capture_inflation_geometry_backup(state, data_collections),
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
    pos: Optional[torch.Tensor] = None,
) -> Optional[InflationRoundRecord]:
    record = state.current_round
    if record is None:
        return None

    record.after = capture_inflation_snapshot(data_collections, placedb)
    if pos is not None:
        record.inflated_position_snapshot = pos.detach().clone()
    record.adjust_area_flag_after = bool(adjust_area_flag)
    record.adjust_route_area_flag_after = bool(adjust_route_area_flag)
    record.adjust_pin_area_flag_after = bool(adjust_pin_area_flag)
    if gr_metrics:
        record.gr_metrics.update(_extract_numeric_metrics(gr_metrics))
    record.status = status
    state.current_snapshot = record.after
    if status == "applied":
        state.num_area_adjust += 1
    record.geometry_backup = None
    state.round_records.append(record)
    state.current_round = None
    return record


def compute_movable_area_increment_ratio(
    before: Optional[InflationGeometrySnapshot],
    after: Optional[InflationGeometrySnapshot],
) -> Optional[float]:
    if before is None or after is None:
        return None
    return (after.movable_area - before.movable_area) / max(before.movable_area, 1e-12)


def restore_current_round_geometry(
    state: Optional[InflationState],
    data_collections,
    placedb,
    pos: torch.Tensor,
) -> bool:
    record = None if state is None else state.current_round
    backup = None if record is None else record.geometry_backup
    if backup is None:
        return False

    with torch.no_grad():
        num_nodes = int(data_collections.node_size_x.numel())
        current_center_x = pos.data[:num_nodes] + data_collections.node_size_x * 0.5
        current_center_y = (
            pos.data[num_nodes : num_nodes + num_nodes]
            + data_collections.node_size_y * 0.5
        )

        data_collections.node_size_x.copy_(backup["node_size_x"])
        data_collections.node_size_y.copy_(backup["node_size_y"])
        pos.data[:num_nodes].copy_(current_center_x - data_collections.node_size_x * 0.5)
        pos.data[num_nodes : num_nodes + num_nodes].copy_(
            current_center_y - data_collections.node_size_y * 0.5
        )
        data_collections.pin_offset_x.copy_(backup["pin_offset_x"])
        data_collections.pin_offset_y.copy_(backup["pin_offset_y"])

        target_density = backup.get("target_density")
        if isinstance(target_density, torch.Tensor):
            data_collections.target_density.copy_(target_density)

    if state is not None:
        state.target_area = backup.get("target_area")
        state.current_snapshot = capture_inflation_snapshot(data_collections, placedb)
    return True


def enforce_min_area_increment(
    params,
    state: Optional[InflationState],
    data_collections,
    placedb,
    pos: torch.Tensor,
) -> Dict[str, Any]:
    min_area_inc = _to_float(getattr(params, "enhanced_inflation_min_area_inc", 0.01), 0.01)
    if min_area_inc <= 0:
        return {
            "triggered": False,
            "movable_area_increment_ratio": None,
            "min_area_increment_threshold": min_area_inc,
            "rolled_back": False,
        }

    record = None if state is None else state.current_round
    if record is None or record.before is None:
        return {
            "triggered": False,
            "movable_area_increment_ratio": None,
            "min_area_increment_threshold": min_area_inc,
            "rolled_back": False,
        }

    attempted_after = capture_inflation_snapshot(data_collections, placedb)
    movable_area_increment_ratio = compute_movable_area_increment_ratio(
        record.before, attempted_after
    )
    record.movable_area_increment_ratio = movable_area_increment_ratio
    record.min_area_increment_threshold = min_area_inc

    result = {
        "triggered": False,
        "movable_area_increment_ratio": movable_area_increment_ratio,
        "min_area_increment_threshold": min_area_inc,
        "rolled_back": False,
    }
    if movable_area_increment_ratio is None or movable_area_increment_ratio >= min_area_inc - 1e-12:
        return result

    result["triggered"] = True
    result["rolled_back"] = restore_current_round_geometry(state, data_collections, placedb, pos)
    early_stop_note = "rejected_by_min_area_inc=%.6f<%.6f" % (movable_area_increment_ratio, min_area_inc)
    record.notes = f"{record.notes} | {early_stop_note}" if record.notes else early_stop_note
    logging.warning(
        "Too small relative area increment (%.4f < %.4f). Early terminate enhanced cell inflation round %d.",
        movable_area_increment_ratio,
        min_area_inc,
        record.round_idx,
    )
    return result


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
        if record.status == "applied"
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
    triggered = False
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
    return getattr(adjust_node_area_op, "_enhanced_adjust_node_area_impl", None)


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


def apply_low_util_target_density(
    params,
    state: Optional[InflationState],
    data_collections,
    placedb,
    pos: torch.Tensor,
    context: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    return {}


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


def should_trigger_enhanced_inflation(params, num_area_adjust: int, overflow: Any) -> bool:
    return bool(
        is_enhanced_inflation_enabled(params)
        and int(num_area_adjust) < get_inflation_round_limit(params)
        and _to_float(overflow) < _to_float(getattr(params, "node_area_adjust_overflow", 0.15), default=0.15)
    )


def get_area_adjust_flags(params) -> Dict[str, bool]:
    return {
        "adjust_area_flag": True,
        "adjust_route_area_flag": bool(
            getattr(params, "adjust_gpugr_area_flag", False)
            or getattr(params, "adjust_nctugr_area_flag", False)
            or getattr(params, "adjust_rudy_area_flag", False)
        ),
        "adjust_pin_area_flag": bool(getattr(params, "adjust_pin_area_flag", False)),
    }


def run_enhanced_inflation_round(*args, **kwargs) -> Dict[str, Any]:
    logging.info(
        "Direct helper entry for enhanced inflation is not used; the active execution path "
        "is coordinated inside NonLinearPlace."
    )
    return {
        "applied": False,
        "status": "delegated_to_non_linear_place",
        "reason": "use_non_linear_place_controller",
    }
