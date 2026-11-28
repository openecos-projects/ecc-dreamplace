"""Opt-in differentiable timing scores for native repair queue ordering.

The native repair operator still owns queue ordering, candidate generation,
acceptance, and rollback. This module only runs the existing DreamPlace
``TimingPropagation`` and reduces its position gradient to a score per active
net. Compact persistent owners without a callable Steiner topology report
``unsupported`` and keep the ordinary slack order.
"""

import copy

import numpy as np
import torch


def compare_driver_rankings(native_order, driver_scores):
    """Compare a native path order with a score order using common drivers.

    ``native_order`` is already ordered by the C++ OpenROAD-style policy.  The
    score order uses the same order as a stable tie-break, so this helper only
    measures the effect of the score and never invents a new identity order.
    """

    native = [int(driver) for driver in native_order]
    scores = {
        int(driver): float(score)
        for driver, score in dict(driver_scores or {}).items()
        if np.isfinite(float(score))
    }
    common = [driver for driver in native if driver in scores]
    if not common:
        return {
            "status": "unsupported",
            "reason": "native_gradient_ranking_has_no_common_drivers",
            "native_order": native,
            "common_driver_count": 0,
        }
    native_position = {driver: index for index, driver in enumerate(common)}
    scored = sorted(
        common,
        key=lambda driver: (-scores[driver], native_position[driver]),
    )
    scored_position = {driver: index for index, driver in enumerate(scored)}
    agreements = 0
    comparisons = 0
    for left_index, left in enumerate(common):
        for right in common[left_index + 1 :]:
            native_relation = native_position[left] < native_position[right]
            score_relation = scored_position[left] < scored_position[right]
            comparisons += 1
            agreements += int(native_relation == score_relation)
    return {
        "status": "ok",
        "native_order": native,
        "scored_order": scored,
        "driver_scores": {driver: scores[driver] for driver in common},
        "common_driver_count": len(common),
        "native_top_driver": common[0],
        "score_top_driver": scored[0],
        "top1_match": bool(common[0] == scored[0]),
        "pairwise_comparisons": comparisons,
        "pairwise_agreements": agreements,
        "pairwise_agreement": (
            float(agreements) / float(comparisons) if comparisons else 1.0
        ),
    }


def compare_path_driver_ranking(context, path, *, path_details=None, gradient_result=None):
    """Compare native C++ path ranking with the current Python TP score map."""

    from dreamplace.ops.rebuffer import rebuffer_cpp

    path = [int(pin) for pin in path]
    if not path:
        return {"status": "unsupported", "reason": "empty_path"}
    views = context.native_queue_views()
    if views.get("status") != "ok":
        return {
            "status": "unsupported",
            "reason": "native_queue_views_unavailable",
        }
    path_timing = context.native_path_context(path, path_details=path_details)
    timing = dict(views["timing"])
    timing.update(path_timing or {})
    native_order = rebuffer_cpp.rank_path_driver_pins(
        views["design"], timing, views["liberty"], path
    )
    scores = gradient_result
    if scores is None:
        scores = context.timing_gradient_scores()
    if not isinstance(scores, dict) or scores.get("status") != "ok":
        return {
            "status": "unsupported",
            "reason": "timing_gradient_unavailable",
            "native_order": [int(driver) for driver in native_order],
        }
    result = compare_driver_rankings(
        native_order,
        scores.get("driver_scores", {}),
    )
    result["path"] = path
    return result


def _pin_position(owner):
    pydb = owner.pydb
    pin2node = torch.as_tensor(
        np.asarray(pydb.pin2node_map, dtype=np.int64), dtype=torch.long
    )
    node_x = torch.as_tensor(np.asarray(pydb.node_x, dtype=np.float64))
    node_y = torch.as_tensor(np.asarray(pydb.node_y, dtype=np.float64))
    pin_x = node_x.index_select(0, pin2node)
    pin_y = node_y.index_select(0, pin2node)
    if bool(getattr(owner, "use_pin_offset", False)):
        pin_x = pin_x + torch.as_tensor(
            np.asarray(pydb.pin_offset_x, dtype=np.float64)
        )
        pin_y = pin_y + torch.as_tensor(
            np.asarray(pydb.pin_offset_y, dtype=np.float64)
        )
    return torch.cat((pin_x, pin_y), dim=0).requires_grad_(True)


def scores_from_position_gradient(position_gradient, pydb):
    """Reduce a ``[2 * pin_count]`` position gradient to net/driver scores."""

    gradient = torch.as_tensor(position_gradient, dtype=torch.float64).reshape(-1)
    pin_count = int(gradient.numel() // 2)
    if gradient.numel() != 2 * pin_count:
        raise ValueError("position gradient must contain two lanes per pin")
    pin_score = torch.sqrt(
        gradient[:pin_count].square() + gradient[pin_count:].square()
    )
    pin2net = torch.as_tensor(
        np.asarray(pydb.pin2net_map, dtype=np.int64), dtype=torch.long
    )
    if pin2net.numel() != pin_count:
        raise ValueError("pin gradient and pin2net_map have different sizes")
    net_count = len(getattr(pydb, "net_names", ()))
    if net_count <= 0:
        net_count = int(pin2net.max().item()) + 1 if pin2net.numel() else 0
    scores = torch.zeros(net_count, dtype=pin_score.dtype)
    valid = (pin2net >= 0) & (pin2net < net_count)
    if torch.any(valid):
        scores.scatter_reduce_(
            0,
            pin2net[valid],
            pin_score[valid],
            reduce="amax",
            include_self=True,
        )
    net_scores = {
        int(net): float(value)
        for net, value in enumerate(scores.detach().cpu().tolist())
        if np.isfinite(value) and value > 0.0
    }
    driver_scores = {}
    for net, driver in enumerate(
        np.asarray(getattr(pydb, "net2driver_pin_map", ()), dtype=np.int64)
    ):
        score = net_scores.get(net)
        if score is not None and int(driver) >= 0:
            driver_scores[int(driver)] = score
    return {"net_scores": net_scores, "driver_scores": driver_scores}


def compute_timing_gradient_scores(timing_model):
    """Compute one smooth Python-TP score map for the current timing state."""

    runtime = getattr(timing_model, "_overlay_runtime", None)
    owner = runtime.model if runtime is not None else timing_model
    steiner = getattr(owner, "steiner", None)
    rc = getattr(owner, "rc", None)
    operator = getattr(owner, "model", None)
    if not callable(steiner) or rc is None or operator is None:
        return {
            "status": "unsupported",
            "reason": "timing_gradient_requires_callable_steiner_owner",
        }

    try:
        position = _pin_position(owner)
        new_x, new_y = steiner(position)
        pin_caps, loads, delays, _ldelays, _betas, impulses = rc(
            new_x,
            new_y,
            steiner.net_flat_topo_sort,
            steiner.net_flat_topo_sort_start,
            steiner.pin_fa,
            steiner.flat_pin_to_start,
            steiner.flat_pin_to,
            steiner.flat_pin_from,
            torch.as_tensor(owner.lib_pin_caps),
            torch.as_tensor(owner.lib_pin_rcaps),
            torch.as_tensor(owner.lib_pin_fcaps),
        )
        del pin_caps
        probe = copy.copy(operator)
        probe.timing_aggregation_mode = "smooth"
        probe.production_fast_loop = False
        probe.timing_propagation_profile = False
        probe.log_stage_progress = False
        probe._sparse_runtime = None
        probe._critical_path_snapshot_requested = False
        probe._critical_path_snapshot = None
        with torch.enable_grad():
            _wns, tns, _worst, _total = probe(
                {"rise": delays["rise"], "fall": delays["fall"]},
                {"rise": impulses["rise"], "fall": impulses["fall"]},
                {"rise": loads["rise"], "fall": loads["fall"]},
                surrogate_mode="mixed",
            )
            loss = -tns
            if not torch.is_tensor(loss) or not loss.requires_grad:
                return {
                    "status": "unsupported",
                    "reason": "timing_gradient_is_not_differentiable",
                }
            gradient = torch.autograd.grad(loss, position, allow_unused=True)[0]
        if gradient is None or not torch.isfinite(gradient).all():
            return {"status": "unsupported", "reason": "timing_gradient_nonfinite"}
        result = scores_from_position_gradient(gradient.detach(), owner.pydb)
        result.update(
            {
                "status": "ok",
                "loss": float(loss.detach().cpu().item()),
                "pin_count": int(position.numel() // 2),
            }
        )
        return result
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
        return {"status": "unsupported", "reason": f"timing_gradient_failed:{error}"}


def native_timing_gradient_payload(timing_model, bridge, overlay):
    """Transport fresh Python-TP scores by stable driver name and native epoch.

    The native projection can expose a fixed Steiner view. Build a separate
    differentiable ECC Steiner/TP probe from its current physical carrier;
    candidate planning and acceptance retain the native owner's tree/math.
    """
    from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel

    bridge.materialize(overlay, gradient_probe=True)
    source = timing_model.active_pydb
    probe = RebufferTimingModel(
        source,
        use_pin_offset=False,
        steiner_provider="ecc",
        r_unit=timing_model.r_unit,
        c_unit=timing_model.c_unit,
    )
    with torch.enable_grad():
        result = compute_timing_gradient_scores(probe)
    if result["status"] != "ok":
        raise RuntimeError(f"native queue gradient probe failed: {result.get('reason')}")
    pin_names = source.pin_names
    scores = {
        pin_names[pin].decode() if isinstance(pin_names[pin], bytes) else str(pin_names[pin]): score
        for pin, score in result["driver_scores"].items()
    }
    return {
        "gradient_score_epoch": int(bridge.graph.epoch),
        "gradient_score_by_driver_name": scores,
        "gradient_loss": result["loss"],
        "gradient_steiner_provider": "ecc",
    }


__all__ = [
    "compare_driver_rankings",
    "compare_path_driver_ranking",
    "compute_timing_gradient_scores",
    "native_timing_gradient_payload",
    "scores_from_position_gradient",
]
