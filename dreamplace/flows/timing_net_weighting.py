"""ECC Pin2Pin net-weight rebuild transaction."""

import time
import logging
import numpy as np
from dreamplace.flows.metric_state import update_metric_timing, timing_scalar
from dataclasses import dataclass

import torch

from dreamplace.ops.timing_net_weighting.net_weighting import (
    update_net_weights_from_tp,
)


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
