"""Pure ECC timing-placement gate and net-weighting policy helpers."""


import math
import torch
from dreamplace.ops.timing_net_weighting.net_weighting import (
    PIN2PIN_PAIR_RESET_INTERVAL,
)
import time
import logging
import numpy as np
from dreamplace.flows.gp.metrics import update_metric_timing, timing_scalar
from dataclasses import dataclass
from dreamplace.ops.timing_net_weighting.net_weighting import (
    update_net_weights_from_tp,
)


def placement_sizing_mode(params):
    return getattr(params, "placement_sizing_mode", "place_only")


def timing_placement_carrier(params):
    return str(
        getattr(params, "timing_placement_carrier", "direct_loss")
        or "direct_loss"
    )


def diff_tdp_enabled(params):
    if placement_sizing_mode(params) == "size_only":
        return False
    if not bool(getattr(params, "with_sta", False)):
        return False
    if not bool(getattr(params, "differentiable_timing_obj", False)):
        return False
    return bool(getattr(params, "diff_timing_driven_placement", True))


def diff_tdp_uses_direct_loss(params):
    return timing_placement_carrier(params) == "direct_loss"


def diff_tdp_uses_gradient_net_weight(params):
    return timing_placement_carrier(params) == "gradient_net_weight"


def diff_tdp_controls_direct_timing_objective(params):
    return (
        placement_sizing_mode(params) != "size_only"
        and diff_tdp_uses_direct_loss(params)
    )


def should_use_live_timing_loss(params, step_status):
    mode = placement_sizing_mode(params)
    configured = (
        mode in ("size_only", "joint")
        and params.with_sta
        and params.differentiable_timing_obj
    )
    if mode == "size_only":
        return configured
    if not diff_tdp_enabled(params):
        return configured
    if not diff_tdp_uses_direct_loss(params):
        return False
    return bool((step_status or {}).get("timing_objective_active", False))


def should_refresh_live_timing_before_objective(params, step_status):
    return should_use_live_timing_loss(params, step_status) and not diff_tdp_enabled(
        params
    )


def new_diff_tdp_gate_summary(params=None):
    return {
        "artifact": "diff_tdp_gate_summary",
        "artifact_version": 1,
        "enabled": diff_tdp_enabled(params) if params is not None else False,
        "carrier": timing_placement_carrier(params) if params is not None else None,
        "threshold": float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
        ) if params is not None else None,
        "refresh_interval": int(
            getattr(params, "timing_topology_refresh_interval", 10) or 0
        ) if params is not None else None,
        "open_count": 0,
        "closed_count": 0,
        "interval_skip_count": 0,
        "warmup_skip_count": 0,
        "invalid_overflow_count": 0,
        "disabled_interval_count": 0,
        "disabled_tdp_count": 0,
        "net_weight_reset_count": 0,
        "timing_active_step_count": 0,
        "timing_inactive_step_count": 0,
        "topology_refresh_count": 0,
        "first_timing_active_iteration": None,
        "first_timing_active_overflow": None,
        "last_gate_status": None,
        "last_reset_iteration": None,
    }


def record_diff_tdp_gate_status(summary, status):
    summary = summary if isinstance(summary, dict) else {}
    status = dict(status or {})
    summary["last_gate_status"] = status
    if status.get("timing_objective_active"):
        summary["timing_active_step_count"] = int(
            summary.get("timing_active_step_count", 0)
        ) + 1
        if summary.get("first_timing_active_iteration") is None:
            summary["first_timing_active_iteration"] = status.get("iteration")
            summary["first_timing_active_overflow"] = status.get("overflow")
    else:
        summary["timing_inactive_step_count"] = int(
            summary.get("timing_inactive_step_count", 0)
        ) + 1
    if status.get("topology_refresh_due"):
        summary["topology_refresh_count"] = int(
            summary.get("topology_refresh_count", 0)
        ) + 1
    reason = str(status.get("reason") or "")
    counter_by_reason = {
        "overflow_gate": "closed_count",
        "interval": "interval_skip_count",
        "warmup_iteration": "warmup_skip_count",
        "invalid_overflow": "invalid_overflow_count",
        "disabled_interval": "disabled_interval_count",
        "disabled_tdp": "disabled_tdp_count",
    }
    if status.get("enabled"):
        summary["open_count"] = int(summary.get("open_count", 0)) + 1
    elif reason in counter_by_reason:
        key = counter_by_reason[reason]
        summary[key] = int(summary.get(key, 0)) + 1
    return summary


def diff_tdp_gate_status(params, iteration, overflow, *, already_activated=False):
    interval = int(getattr(params, "timing_topology_refresh_interval", 10) or 0)
    threshold = float(
        getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
    )
    status = {
        "enabled": False,
        "timing_objective_active": False,
        "topology_refresh_due": False,
        "iteration": int(iteration),
        "interval": interval,
        "threshold": threshold,
        "overflow": None,
    }
    if not diff_tdp_enabled(params):
        return {**status, "reason": "disabled_tdp"}
    if getattr(params, "_midplace_objectives_pending", False):
        return {**status, "reason": "midplace_repair_pending"}
    if int(iteration) <= 0:
        return {**status, "reason": "warmup_iteration"}
    try:
        overflow_value = float(torch.as_tensor(overflow).detach().cpu().item())
    except (RuntimeError, TypeError, ValueError):
        return {**status, "reason": "invalid_overflow"}
    status["overflow"] = overflow_value
    if not already_activated and not overflow_value < threshold:
        return {**status, "reason": "overflow_gate"}
    status["timing_objective_active"] = True
    if interval <= 0:
        return {**status, "reason": "disabled_interval"}
    if int(iteration) % interval != 0:
        return {**status, "reason": "interval"}
    status["enabled"] = True
    status["topology_refresh_due"] = True
    return {**status, "reason": "ok"}


def freeze_diff_tdp_step_status(
    params,
    iteration,
    overflow,
    previous_status,
    live_timing_topology_initialized,
    *,
    already_activated=False,
):
    keep_during_inflation = bool(
        getattr(params, "timing_keep_during_inflation", False)
    )
    latched = keep_during_inflation and (
        bool(already_activated)
        or bool((previous_status or {}).get("timing_objective_active", False))
    )
    status = diff_tdp_gate_status(
        params,
        iteration,
        overflow,
        already_activated=latched,
    )
    if status["timing_objective_active"] and (
        not bool((previous_status or {}).get("timing_objective_active", False))
        or not bool(live_timing_topology_initialized)
    ):
        status.update(
            enabled=True,
            topology_refresh_due=True,
            reason="activation_topology",
        )
    return status


def reset_legacy_net_weight_gate_summary(params):
    enabled = bool(
        getattr(params, "with_sta", False)
        and getattr(params, "enable_net_weighting", False)
        and not diff_tdp_enabled(params)
        and placement_sizing_mode(params) != "size_only"
    )
    return {
        "artifact": "legacy_net_weight_gate_summary",
        "artifact_version": 1,
        "enabled": enabled,
        "scheme": str(getattr(params, "net_weighting_scheme", "")),
        "trigger_policy": "overflow",
        "threshold": float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.2)
        ),
        "update_interval": max(
            1,
            int(getattr(params, "net_weighting_update_interval", 15) or 15),
        ),
        "active_step_count": 0,
        "closed_step_count": 0,
        "update_count": 0,
        "pin2pin_pair_reset_interval": PIN2PIN_PAIR_RESET_INTERVAL,
        "pin2pin_pair_reset_count": 0,
        "pin2pin_pair_reset_update_counts": [],
        "last_pin2pin_pair_reset_before_update": None,
        "first_active_iteration": None,
        "first_active_overflow": None,
        "first_update_iteration": None,
        "first_update_overflow": None,
        "last_gate_status": None,
    }


def legacy_net_weight_gate_status(params, iteration, overflow, summary, gate_open):
    summary = summary if isinstance(summary, dict) and summary else reset_legacy_net_weight_gate_summary(params)
    status = {
        "enabled": False,
        "update_due": False,
        "iteration": int(iteration),
        "overflow": None,
        "threshold": float(summary["threshold"]),
        "update_interval": int(summary["update_interval"]),
    }
    if not summary["enabled"]:
        status["reason"] = "disabled"
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    try:
        overflow_value = float(torch.as_tensor(overflow).detach().cpu().item())
    except (RuntimeError, TypeError, ValueError):
        status["reason"] = "invalid_overflow"
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    status["overflow"] = overflow_value
    if not overflow_value < status["threshold"]:
        status["reason"] = "overflow_gate"
        summary["closed_step_count"] += 1
        summary["last_gate_status"] = dict(status)
        return status, False, summary
    first_open_step = not gate_open
    status["enabled"] = True
    status["update_due"] = bool(
        first_open_step or int(iteration) % status["update_interval"] == 0
    )
    status["reason"] = "activation" if first_open_step else (
        "interval" if status["update_due"] else "between_updates"
    )
    summary["active_step_count"] += 1
    if summary["first_active_iteration"] is None:
        summary["first_active_iteration"] = int(iteration)
        summary["first_active_overflow"] = overflow_value
    summary["last_gate_status"] = dict(status)
    return status, True, summary


def net_weighting_npaths_percent(params):
    value = getattr(params, "net_weighting_npaths", 0.0)
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        raise ValueError("net_weighting_npaths must be a percentage in [0, 100]")
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("net_weighting_npaths must be a percentage in [0, 100]")
    return value


def net_weighting_npaths_budget(params, placedb):
    percent = net_weighting_npaths_percent(params)
    if percent <= 0.0:
        return 0
    net_count = int(getattr(placedb, "num_nets", 0) or 0)
    if net_count <= 0:
        return 0
    return max(1, int(net_count * percent / 100.0))


def net_weighting_nendpoints(params, placedb=None):
    canonical = getattr(params, "net_weighting_nendpoints", None)
    legacy = getattr(params, "net_weighting_npaths", None)
    if bool(getattr(params, "_net_weighting_nendpoints_explicit", False)):
        try:
            value = int(canonical or 0)
        except (TypeError, ValueError):
            raise ValueError(
                "net_weighting_nendpoints must be a non-negative integer"
            )
        if value < 0:
            raise ValueError(
                "net_weighting_nendpoints must be a non-negative integer"
            )
        return value
    if legacy is not None:
        return net_weighting_npaths_budget(params, placedb)
    try:
        value = int(canonical or 0)
    except (TypeError, ValueError):
        raise ValueError("net_weighting_nendpoints must be a non-negative integer")
    if value < 0:
        raise ValueError("net_weighting_nendpoints must be a non-negative integer")
    return value


@dataclass
class TimingNetWeightingContext:
    """Explicit inputs and side-effect owners for one Pin2Pin update."""

    params: object
    placedb: object
    model: object
    pos: torch.Tensor
    data_collections: object
    op_collections: object
    iteration: int
    gate_status: dict
    update_reason: str
    force_clear_accumulation: bool
    profile_enabled: bool
    profile_clock: object
    prepare_timing_topology: object
    record_timing_metrics: object
    timing_scalar: object
    clear_pair_accumulation: object
    maybe_reset_pair_accumulation: object
    record_update: object
    write_artifact: object
    endpoint_budget: object
    next_update_count: object


def rebuild_pin2pin_pair_weights(context):
    """Run one timing snapshot and commit the corresponding Pin2Pin weights."""
    params = context.params
    if str(getattr(params, "net_weighting_scheme", "")).lower() != "pin2pin":
        raise RuntimeError("Pin2Pin rebuild requires net_weighting_scheme=pin2pin")
    placedb = context.placedb
    model = context.model
    pos = context.pos
    tp_op = getattr(context.op_collections, "timing_propagation_op", None)
    if tp_op is None:
        context.write_artifact(
            {
                "artifact": "legacy_net_weight_summary",
                "artifact_version": 2,
                "status": "skipped",
                "carrier": "pin2pin_pair_weight",
                "iteration": int(context.iteration),
                "reason": "timing_propagation_op_missing",
                "update_reason": str(context.update_reason),
            }
        )
        raise RuntimeError("Pin2Pin rebuild requires timing propagation op")

    profile_clock = context.profile_clock
    started_at = time.perf_counter()
    profile_started_at = profile_clock()
    nendpoints = context.endpoint_budget()
    timing_started_at = profile_clock()
    with torch.no_grad():
        context.prepare_timing_topology(pos)
        tp_op.request_critical_path_snapshot()
        try:
            timing = model.timing_obj(pos)
        except Exception:
            tp_op.clear_critical_path_snapshot_request()
            raise
        context.record_timing_metrics(*timing)
    timing_finished_at = profile_clock()
    wns, tns, ws, ts = timing
    wns_value = context.timing_scalar(wns)
    tns_value = context.timing_scalar(tns)
    timing_clean = bool(wns_value >= 0.0 and tns_value == 0.0)
    next_update_count = context.next_update_count()
    reset_before_update = False
    pair_update_started_at = profile_clock()
    try:
        if context.force_clear_accumulation or timing_clean:
            context.clear_pair_accumulation(
                placedb,
                next_update_count=next_update_count,
                reason="timing_clean" if timing_clean else context.update_reason,
            )
            reset_before_update = True
        else:
            reset_before_update = context.maybe_reset_pair_accumulation(
                placedb,
                next_update_count,
            )
        net_weight_summary = update_net_weights_from_tp(
            scheme="pin2pin",
            placedb=placedb,
            data_collections=context.data_collections,
            timing_propagation=tp_op,
            momentum_decay=params.momentum_decay_factor,
            max_net_weight=placedb.max_net_weight,
            ignore_net_degree=params.ignore_net_degree,
            nendpoints=nendpoints,
            wns=wns_value,
            pin2pin_cfg={
                "max_weight": getattr(params, "pin2pin_max_weight", float("inf")),
                "min_weight": getattr(params, "pin2pin_min_weight", 0.0),
                "accumulate": getattr(params, "pin2pin_accumulate_weight", 0.0),
            },
        )
    finally:
        tp_op.clear_critical_path_snapshot_request()
    pair_update_finished_at = profile_clock()

    pair_backend = net_weight_summary.get("backend", "cpp_openmp_transition_aware")
    attraction_backend = "cuda" if pos.is_cuda else "cpp_cpu"
    attraction_started_at = profile_clock()
    with torch.no_grad():
        pin2pin_objective = context.timing_scalar(
            model.op_collections.pin2pin_net_weight_op(pos)
        )
    attraction_finished_at = profile_clock()
    payload = context.record_update(
        params,
        placedb,
        iteration=context.iteration,
        npaths=nendpoints,
        wns=wns,
        tns=tns,
        update_ms=(time.perf_counter() - started_at) * 1000.0,
        gate_status=context.gate_status,
        pin2pin_objective=pin2pin_objective,
        pin2pin_path_pair_backend=pair_backend,
        pin2pin_attraction_backend=attraction_backend,
        pin2pin_pair_reset_before_update=reset_before_update,
        update_reason=context.update_reason,
    )
    if context.profile_enabled:
        payload["runtime_profile"] = {
            "enabled": True,
            "synchronized": bool(pos.is_cuda),
            "timing_obj_ms": float((timing_finished_at - timing_started_at) * 1000.0),
            "pair_weight_update_ms": float(
                (pair_update_finished_at - pair_update_started_at) * 1000.0
            ),
            "pin2pin_attraction_ms": float(
                (attraction_finished_at - attraction_started_at) * 1000.0
            ),
            "total_ms": float((attraction_finished_at - profile_started_at) * 1000.0),
        }
    pair_count = int(payload["pin2pin_pair_count"])
    if timing_clean:
        if pair_count != 0:
            raise RuntimeError(
                "timing-clean Pin2Pin rebuild did not install an empty pair set"
            )
        generation_status = "timing_clean"
    else:
        if wns_value < 0.0 and pair_count <= 0:
            raise RuntimeError(
                "negative-WNS Pin2Pin rebuild produced no valid extracted paths"
            )
        if pair_count <= 0:
            raise RuntimeError("non-clean Pin2Pin rebuild produced no valid pair weights")
        generation_status = "timing_violating"
    payload["generation_status"] = generation_status
    payload["timing_clean"] = timing_clean
    return payload, timing

@dataclass(frozen=True)
class LegacyWeightContext:
    data_collections: object
    device: object
    pos: object
    timing_op: object
    maybe_reset_pairs: object
    endpoint_budget: object
    path_budget: object
    path_percent: object
    prepare_topology: object
    record_update: object
    record_timing: object
    write_artifact: object
    next_update_count: object

def update_legacy_weights(context, params, placedb, model, pos, iteration,
                          cur_metric, managed_pin2pin_schedule, legacy_net_weight_gate_status):
    if (
        not managed_pin2pin_schedule
        and
        params.global_place_flag
        and legacy_net_weight_gate_status["update_due"]
    ):
        # Take the timing operator from the operator collections.
        pin2pin_scheme = (
            str(params.net_weighting_scheme).lower() == "pin2pin"
        )
        tp_op = context.timing_op
        if pin2pin_scheme:
            configured_nendpoints = context.endpoint_budget(
                params, placedb
            )
            # Pin2pin uses one global K over violating endpoint-
            # transition states; K=0 keeps every such state.
            npaths = configured_nendpoints
        else:
            # Adams/Lilith retain the legacy net/path budget and
            # ratio fallback. The endpoint budget is Pin2Pin-only.
            configured_npaths_percent = context.path_percent(
                params
            )
            if configured_npaths_percent > 0.0:
                npaths = context.path_budget(
                    params, placedb
                )
            else:
                npaths_ratio = float(
                    getattr(params, "net_weighting_npaths_ratio", 0.03)
                    or 0.03
                )
                npaths = max(1, int(placedb.num_nets * npaths_ratio))

        beg = time.time()
        if tp_op is None:
            logging.warning("timing propagation op 未初始化，跳过 net weighting 更新")
            context.write_artifact(
                params,
                {
                    "artifact": "legacy_net_weight_summary",
                    "artifact_version": 1,
                    "status": "skipped",
                    "carrier": "legacy_net_weight",
                    "iteration": int(iteration),
                    "reason": "timing_propagation_op_missing",
                },
            )
            raise RuntimeError(
                "legacy net-weighting requires timing propagation op"
            )
        else:
            with torch.no_grad():
                context.prepare_topology(pos)
                if pin2pin_scheme:
                    tp_op.request_critical_path_snapshot()
                try:
                    wns_tp, tns_tp, ws_tp, ts_tp = model.timing_obj(
                        context.pos[0].data
                    )
                except Exception:
                    if pin2pin_scheme:
                        tp_op.clear_critical_path_snapshot_request()
                    raise
                # The production timing loop avoids this CPU copy. Legacy
                # net weighting consumes it only on an actual update frame.
                if not pin2pin_scheme:
                    context.data_collections.pin_slack = tp_op.get_pin_slack()
                context.record_timing(wns_tp, tns_tp, ws_tp, ts_tp)

            if torch.is_tensor(wns_tp):
                wns_tp_value = float(wns_tp.detach().cpu().item())
            else:
                wns_tp_value = float(wns_tp)

            try:
                pin2pin_pair_reset_before_update = False
                if pin2pin_scheme:
                    pin2pin_pair_reset_before_update = (
                        context.maybe_reset_pairs(
                            placedb,
                            context.next_update_count(),
                        )
                    )
                endpoint_limit_kwargs = (
                    {"nendpoints": npaths}
                    if pin2pin_scheme
                    else {"npaths": npaths}
                )
                net_weight_summary = update_net_weights_from_tp(
                    scheme=params.net_weighting_scheme,
                    placedb=placedb,
                    data_collections=context.data_collections,
                    timing_propagation=tp_op,
                    momentum_decay=params.momentum_decay_factor,
                    max_net_weight=placedb.max_net_weight,
                    ignore_net_degree=params.ignore_net_degree,
                    **endpoint_limit_kwargs,
                    wns=wns_tp_value,
                    pin2pin_cfg={
                        "max_weight": getattr(params, "pin2pin_max_weight", float("inf")),
                        "min_weight": getattr(params, "pin2pin_min_weight", 0.0),
                        "accumulate": getattr(params, "pin2pin_accumulate_weight", 0.0),
                    },
                )
            finally:
                if pin2pin_scheme:
                    tp_op.clear_critical_path_snapshot_request()
            logging.info("max net weight: %.3f" % (
                np.max(placedb.net_weights)))
            if context.device != torch.device("cpu") and not pin2pin_scheme:
                # Copy weights from placedb.net_weights to device.
                context.data_collections.net_weights.copy_(
                    torch.from_numpy(placedb.net_weights))

            pin2pin_objective = None
            pin2pin_path_pair_backend = None
            pin2pin_attraction_backend = None
            if pin2pin_scheme:
                pin2pin_path_pair_backend = net_weight_summary.get(
                    "backend", "cpp_openmp_transition_aware"
                )
                pin2pin_attraction_backend = (
                    "cuda" if context.pos[0].is_cuda else "cpp_cpu"
                )
                with torch.no_grad():
                    pin2pin_objective = timing_scalar(
                        model.op_collections.pin2pin_net_weight_op(
                            context.pos[0].data
                        )
                    )
            net_weight_update_ms = (time.time() - beg) * 1000
            context.record_update(
                params,
                placedb,
                iteration=iteration,
                npaths=npaths,
                wns=wns_tp,
                tns=tns_tp,
                update_ms=net_weight_update_ms,
                gate_status=legacy_net_weight_gate_status,
                pin2pin_objective=pin2pin_objective,
                pin2pin_path_pair_backend=pin2pin_path_pair_backend,
                pin2pin_attraction_backend=pin2pin_attraction_backend,
                pin2pin_pair_reset_before_update=(
                    pin2pin_pair_reset_before_update
                ),
            )
            logging.info("net-weight update step %.3f ms" % net_weight_update_ms)

        update_metric_timing(cur_metric, wns_tp, tns_tp, ws_tp, context.timing_op)
        cur_metric.nvp = 1


@dataclass
class TimingGradientProjectionContext:
    """Explicit owners for one gradient-carrier update."""

    params: object
    placedb: object
    model: object
    pos: torch.Tensor
    data_collections: object
    op_collections: object
    iteration: int
    gate_status: dict
    timing_carrier: str
    record_timing_metrics: object
    write_artifact: object
    topology_prepared: bool = False


def project_timing_gradient_to_net_weights(context):
    params = context.params
    placedb = context.placedb
    model = context.model
    pos = context.pos
    op_collections = context.op_collections
    data_collections = context.data_collections
    if getattr(op_collections, "timing_propagation_op", None) is None:
        payload = {
            "status": "skipped",
            "reason": "timing_propagation_op_missing",
            "carrier": context.timing_carrier,
            "iteration": int(context.iteration),
            "gate_status": dict(context.gate_status),
        }
        context.write_artifact(payload)
        return payload, False

    old_pos_grad = None if pos.grad is None else pos.grad.detach().clone()
    old_use_timing_obj = bool(getattr(model, "use_timing_obj", False))
    if pos.grad is not None:
        pos.grad.zero_()
    try:
        started = time.time()
        if not context.topology_prepared:
            pin_pos = op_collections.pin_pos_op(pos)
            pin_pos_for_topology = pin_pos.detach().cpu().contiguous()
            (
                data_collections.net_flat_topo_sort,
                data_collections.net_flat_topo_sort_start,
                data_collections.pin_fa,
                data_collections.flat_pin_to,
                data_collections.flat_pin_to_start,
                data_collections.flat_pin_from,
            ) = op_collections.steiner_topo_op.rebuild_tree(pin_pos_for_topology)
        steiner_update_ms = (time.time() - started) * 1000.0

        started = time.time()
        model.use_timing_obj = False
        wns, tns, ws, ts = model.timing_obj(pos)
        timing_loss = model._timing_loss(wns, tns, ws, ts)
        timing_forward_ms = (time.time() - started) * 1000.0

        started = time.time()
        timing_loss.backward()
        timing_backward_ms = (time.time() - started) * 1000.0

        if pos.grad is None:
            raise RuntimeError("timing gradient projection produced no pos.grad")
        pos_grad = pos.grad.detach()
        pos_grad_float = pos_grad.float()
        pos_grad_finite_mask = torch.isfinite(pos_grad_float)
        pos_grad_nonfinite_count = int(
            torch.count_nonzero(~pos_grad_finite_mask).detach().cpu().item()
        )
        pos_grad_finite_count = int(
            torch.count_nonzero(pos_grad_finite_mask).detach().cpu().item()
        )
        if pos_grad_nonfinite_count:
            logging.warning(
                "Diff TDP gradient projection sanitized %d non-finite position-gradient entries",
                pos_grad_nonfinite_count,
            )
            pos_grad_float = torch.nan_to_num(
                pos_grad_float,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        half = int(pos_grad_float.numel() // 2)
        node_score = torch.sqrt(
            pos_grad_float[:half].pow(2) + pos_grad_float[half : half * 2].pow(2)
        )

        pin2node = data_collections.pin2node_map.long().to(node_score.device)
        valid_pin_mask = (pin2node >= 0) & (pin2node < node_score.numel())
        pin_score = torch.zeros(
            pin2node.numel(), dtype=node_score.dtype, device=node_score.device
        )
        if torch.any(valid_pin_mask):
            pin_score[valid_pin_mask] = node_score[pin2node[valid_pin_mask]]

        pin2net = data_collections.pin2net_map.long().to(pin_score.device)
        num_nets = int(
            getattr(placedb, "num_nets", data_collections.net_weights.numel())
        )
        net_score = torch.zeros(
            num_nets, dtype=pin_score.dtype, device=pin_score.device
        )
        valid_net_mask = (pin2net >= 0) & (pin2net < num_nets)
        if torch.any(valid_net_mask):
            if hasattr(net_score, "scatter_reduce_"):
                net_score.scatter_reduce_(
                    0,
                    pin2net[valid_net_mask],
                    pin_score[valid_net_mask],
                    reduce="amax",
                    include_self=True,
                )
            else:
                valid_net_ids = pin2net[valid_net_mask]
                valid_pin_scores = pin_score[valid_net_mask]
                for idx in range(valid_net_ids.numel()):
                    net_id = int(valid_net_ids[idx].item())
                    net_score[net_id] = torch.maximum(
                        net_score[net_id], valid_pin_scores[idx]
                    )

        degree_start = data_collections.flat_net2pin_start_map.long().to(
            net_score.device
        )
        if degree_start.numel() >= num_nets + 1:
            net_degree = degree_start[1 : num_nets + 1] - degree_start[:num_nets]
            net_mask = (net_degree >= 2) & (
                net_degree < int(getattr(params, "ignore_net_degree", 1000000000))
            )
        else:
            net_mask = torch.ones(
                num_nets, dtype=torch.bool, device=net_score.device
            )

        finite_mask = net_mask & torch.isfinite(net_score) & (net_score > 0)
        scale = float(getattr(params, "timing_gradient_net_weight_scale", 0.4))
        max_weight = float(getattr(params, "timing_gradient_net_weight_max", 2.0))
        if torch.any(finite_mask):
            normalizer = torch.quantile(net_score[finite_mask], 0.95).clamp_min(1e-12)
            new_weights = torch.ones(
                num_nets, dtype=net_score.dtype, device=net_score.device
            )
            new_weights[net_mask] = torch.clamp(
                1.0 + scale * net_score[net_mask] / normalizer,
                min=1.0,
                max=max_weight,
            )
        else:
            normalizer = torch.tensor(
                0.0, dtype=net_score.dtype, device=net_score.device
            )
            new_weights = torch.ones(
                num_nets, dtype=net_score.dtype, device=net_score.device
            )

        target_weights = new_weights.to(
            dtype=data_collections.net_weights.dtype,
            device=data_collections.net_weights.device,
        )
        if bool(getattr(params, "zero_clock_net_weight_flag", 0)):
            from dreamplace.ops.buffer_insertion.net_eligibility import clock_net_ids

            clk_ids = [
                int(net_id)
                for net_id in clock_net_ids(placedb, data_collections)
                if 0 <= int(net_id) < target_weights.numel()
            ]
            if clk_ids:
                target_weights[clk_ids] = 0.0
        data_collections.net_weights[: target_weights.numel()].copy_(target_weights)
        placedb.net_weights[: target_weights.numel()] = target_weights.detach().cpu().numpy().astype(
            placedb.net_weights.dtype, copy=False
        )
        projected = bool(
            torch.any(target_weights > 1.0 + 1e-6).detach().cpu().item()
        )

        top_k = min(10, num_nets)
        top_values, top_indices = torch.topk(net_score.detach(), k=top_k)
        net_names = getattr(placedb, "net_names", [])
        top_nets = []
        for rank, (net_id_tensor, score_tensor) in enumerate(
            zip(top_indices, top_values)
        ):
            net_id = int(net_id_tensor.detach().cpu().item())
            net_name = ""
            if 0 <= net_id < len(net_names):
                net_name = net_names[net_id]
                if isinstance(net_name, bytes):
                    net_name = net_name.decode("utf-8", errors="ignore")
                else:
                    net_name = str(net_name)
            top_nets.append(
                {
                    "rank": int(rank),
                    "net_id": net_id,
                    "net_name": net_name,
                    "score": float(score_tensor.detach().cpu().item()),
                    "weight": float(target_weights[net_id].detach().cpu().item()),
                }
            )

        payload = {
            "status": "updated",
            "carrier": context.timing_carrier,
            "iteration": int(context.iteration),
            "gate_status": dict(context.gate_status),
            "scale": scale,
            "max_weight": max_weight,
            "steiner_update_ms": float(steiner_update_ms),
            "timing_forward_ms": float(timing_forward_ms),
            "timing_backward_ms": float(timing_backward_ms),
            "wns": float(wns.detach().cpu().item()) if torch.is_tensor(wns) else float(wns),
            "tns": float(tns.detach().cpu().item()) if torch.is_tensor(tns) else float(tns),
            "timing_loss": float(timing_loss.detach().cpu().item()),
            "pos_grad_norm": float(pos_grad_float.norm().detach().cpu().item()),
            "pos_grad_finite_count": pos_grad_finite_count,
            "pos_grad_nonfinite_count": pos_grad_nonfinite_count,
            "net_score_max": float(net_score.max().detach().cpu().item()) if num_nets else 0.0,
            "net_score_p95": (
                float(torch.quantile(net_score[finite_mask], 0.95).detach().cpu().item())
                if torch.any(finite_mask)
                else 0.0
            ),
            "normalizer": float(normalizer.detach().cpu().item()),
            "non_unit_net_weight_count": int(
                torch.count_nonzero(target_weights > 1.0 + 1e-6).detach().cpu().item()
            ),
            "max_net_weight": float(target_weights.max().detach().cpu().item())
            if target_weights.numel()
            else 1.0,
            "top_nets": top_nets,
        }
        context.record_timing_metrics(wns, tns, ws, ts)
        context.write_artifact(payload)
        logging.info(
            "Diff TDP gradient net-weight update at iteration=%d "
            "overflow=%.6f non_unit_nets=%d max_weight=%.3f "
            "timing_loss=%.6f timing_forward_ms=%.3f timing_backward_ms=%.3f",
            int(context.iteration),
            float(context.gate_status.get("overflow") or 0.0),
            payload["non_unit_net_weight_count"],
            payload["max_net_weight"],
            payload["timing_loss"],
            payload["timing_forward_ms"],
            payload["timing_backward_ms"],
        )
        return payload, projected
    finally:
        model.use_timing_obj = old_use_timing_obj
        if old_pos_grad is None:
            pos.grad = None
        elif pos.grad is None:
            pos.grad = old_pos_grad
        else:
            pos.grad.copy_(old_pos_grad)
