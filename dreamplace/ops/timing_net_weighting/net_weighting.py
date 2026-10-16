import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from dreamplace.ops.timing_net_weighting import timing_net_weighting_cpp as net_weighting_cpp
except ImportError:  # pragma: no cover - native extension optional during migration
    net_weighting_cpp = None


PIN2PIN_PAIR_RESET_INTERVAL = 4

__all__ = [
    "PIN2PIN_PAIR_RESET_INTERVAL",
    "clear_pin2pin_pair_accumulation",
    "pin2pin_pair_accumulation_reset_due",
    "update_net_weights_from_tp",
]


def pin2pin_pair_accumulation_reset_due(next_update_count: int) -> bool:
    """Return whether the next Pin2Pin update starts a fresh accumulation."""
    return (
        int(next_update_count) > 0
        and int(next_update_count) % PIN2PIN_PAIR_RESET_INTERVAL == 0
    )


def clear_pin2pin_pair_accumulation(placedb) -> dict:
    """Clear active pair state while retaining preallocated GPU buffers."""
    pair_mapping = getattr(placedb, "pin2pin_net_weight", None)
    length = getattr(placedb, "length", None)
    previous_pair_count = None
    if length is not None:
        try:
            length_value = length[0]
            if torch.is_tensor(length_value):
                length_value = length_value.detach().cpu().item()
            previous_pair_count = int(length_value)
        except (IndexError, TypeError, ValueError, RuntimeError):
            previous_pair_count = None
    if previous_pair_count is None:
        pair_weights = getattr(placedb, "_pin2pin_pair_weights_tensor", None)
        if torch.is_tensor(pair_weights):
            previous_pair_count = int(pair_weights.numel())
        elif pair_mapping is not None:
            previous_pair_count = len(pair_mapping)
        else:
            previous_pair_count = 0
    if pair_mapping is not None:
        pair_mapping.clear()

    placedb._pin2pin_pair_keys_tensor = torch.empty(
        (0, 2), dtype=torch.int64
    )
    placedb._pin2pin_pair_weights_tensor = torch.empty(
        (0,), dtype=torch.float64
    )
    if length is not None:
        length[0] = 0
    placedb._pin2pin_last_native_summary = {}
    return {
        "previous_pair_count": int(previous_pair_count),
        "cleared_pair_count": int(previous_pair_count),
        "preallocated_buffers_retained": True,
    }


def _ensure_numpy_attr(container, attr: str, dtype) -> np.ndarray:
    """Return the numpy view of container.attr, converting in-place when needed."""
    value = getattr(container, attr)
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    array = np.asarray(value, dtype=dtype)
    setattr(container, attr, array)
    return array


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor.detach().cpu()
    return tensor.detach()


def _compute_net_slack(pin_slack: np.ndarray,
                       flat_netpin: np.ndarray,
                       netpin_start: np.ndarray) -> np.ndarray:
    """Compute per-net slack as the minimum slack among its pins."""
    num_nets = netpin_start.size - 1
    net_slack = np.zeros(num_nets, dtype=np.float32)
    for net_id in range(num_nets):
        beg, end = netpin_start[net_id], netpin_start[net_id + 1]
        if beg >= end:
            net_slack[net_id] = 0.0
        else:
            net_slack[net_id] = float(np.min(pin_slack[flat_netpin[beg:end]]))
    return net_slack


def _select_worst_endpoints(pin_slack: np.ndarray,
                            endpoints: np.ndarray,
                            count: int) -> List[int]:
    """Return up to `count` endpoints that have negative slack.

    Only endpoints with slack < 0 are considered (violating endpoints).
    The returned endpoints are ordered by slack ascending (most negative first)
    and converted to a list of int indices.
    """
    if endpoints.size == 0:
        return []
    endpoint_slack = pin_slack[endpoints]
    # Filter to negative (violating) endpoints only
    neg_mask = endpoint_slack < 0
    if not np.any(neg_mask):
        return []
    neg_endpoints = endpoints[neg_mask]
    neg_slack = endpoint_slack[neg_mask]
    order = np.argsort(neg_slack)  # ascending: most negative first
    top_k = min(count, neg_endpoints.size)
    return neg_endpoints[order[:top_k]].astype(np.int64).tolist()


def _apply_adams(net_weights: np.ndarray,
                 net_criticality: np.ndarray,
                 degree_map: np.ndarray,
                 ignore_net_degree: int,
                 pin2net_map: np.ndarray,
                 pin_slack: np.ndarray,
                 timing_prop,
                 npaths: int,
                 max_net_weight: float) -> None:
    logging.info("apply adams net-weighting scheme (timing propagation backend)...")
    num_nets = net_weights.size

    endpoints = _to_cpu_tensor(timing_prop.end_points).numpy().astype(np.int64)
    candidate_endpoints = _select_worst_endpoints(pin_slack, endpoints, max(1, npaths))
    if not candidate_endpoints:
        logging.info("no critical endpoints selected, skip ADAMS update")
        return

    raw_paths = timing_prop.get_critical_paths(candidate_endpoints, K=1)
    paths = [[int(pin) for pin in path] for path in raw_paths if path]

    if net_weighting_cpp is not None and paths:
        net_weights_tensor = torch.from_numpy(net_weights)
        net_criticality_tensor = torch.from_numpy(net_criticality)
        degree_tensor = torch.from_numpy(degree_map.astype(np.int64, copy=False))
        pin2net_tensor = torch.from_numpy(pin2net_map.astype(np.int64, copy=False))
        try:
            net_weighting_cpp.apply_adams(
                net_weights_tensor,
                net_criticality_tensor,
                degree_tensor,
                int(ignore_net_degree),
                pin2net_tensor,
                paths,
                float(max_net_weight),
            )
            return
        except Exception:
            logging.exception("C++ ADAMS 扩展调用失败，回退至 Python 实现")

    net_critical_flag = np.zeros(num_nets, dtype=np.bool_)
    for path in paths:
        for pin in path:
            net_id = int(pin2net_map[pin])
            if 0 <= net_id < num_nets:
                net_critical_flag[net_id] = True

    for net_id in range(num_nets):
        if degree_map[net_id] > ignore_net_degree:
            continue
        net_criticality[net_id] *= 0.5
        if net_critical_flag[net_id]:
            net_criticality[net_id] += 0.5
        net_weights[net_id] *= (1.0 + net_criticality[net_id])
        if math.isfinite(max_net_weight):
            net_weights[net_id] = min(net_weights[net_id], max_net_weight)


def _apply_lilith(net_weights: np.ndarray,
                  net_criticality: np.ndarray,
                  degree_map: np.ndarray,
                  ignore_net_degree: int,
                  flat_netpin: np.ndarray,
                  netpin_start: np.ndarray,
                  pin_slack_np: np.ndarray,
                  pin_slack_tensor: torch.Tensor,
                  wns: Optional[float],
                  decay: float,
                  max_net_weight: float) -> None:
    logging.info("apply lilith net-weighting scheme (timing propagation backend)...")
    logging.info("lilith mode momentum decay factor: %.6f", decay)
    if wns is None:
        logging.warning("WNS 未提供，跳过 LILITH 更新")
        return

    if net_weighting_cpp is not None:
        net_weights_tensor = torch.from_numpy(net_weights)
        net_criticality_tensor = torch.from_numpy(net_criticality)
        degree_tensor = torch.from_numpy(degree_map.astype(np.int64, copy=False))
        flat_netpin_tensor = torch.from_numpy(flat_netpin.astype(np.int64, copy=False))
        netpin_start_tensor = torch.from_numpy(netpin_start.astype(np.int64, copy=False))
        slack_tensor = pin_slack_tensor
        if slack_tensor.dtype != net_weights_tensor.dtype:
            slack_tensor = slack_tensor.to(net_weights_tensor.dtype)
        try:
            net_weighting_cpp.apply_lilith(
                net_weights_tensor,
                net_criticality_tensor,
                degree_tensor,
                int(ignore_net_degree),
                flat_netpin_tensor,
                netpin_start_tensor,
                slack_tensor.contiguous(),
                float(wns),
                float(decay),
                float(max_net_weight),
            )
            return
        except Exception:
            logging.exception("C++ LILITH 扩展调用失败，回退至 Python 实现")

    num_nets = net_weights.size
    net_slack = _compute_net_slack(pin_slack_np, flat_netpin, netpin_start)

    for net_id in range(num_nets):
        if degree_map[net_id] > ignore_net_degree:
            continue
        if wns < 0:
            slack = net_slack[net_id]
            if slack < 0:
                nc = max(0.0, slack / wns) if wns != 0 else 0.0
            else:
                nc = 0.0
            net_criticality[net_id] = (
                math.pow(1.0 + net_criticality[net_id], decay)
                * math.pow(1.0 + nc, 1.0 - decay)
                - 1.0
            )
        net_weights[net_id] *= (1.0 + net_criticality[net_id])
        if math.isfinite(max_net_weight):
            net_weights[net_id] = min(net_weights[net_id], max_net_weight)


def _apply_pin2pin(pin2pin_net_weight: Dict[Tuple[int, int], float],
                   pin2node_map: np.ndarray,
                   pin_slack_np: np.ndarray,
                   pin_slack_tensor: torch.Tensor,
                   timing_prop,
                   wns: Optional[float],
                   nendpoints: int,
                   pin2pin_min_weight: float,
                   pin2pin_max_weight: float,
                   pin2pin_accumulate_weight: float) -> None:
    logging.info("apply reference pin2pin net-weighting scheme...")
    if wns is None or wns >= 0:
        logging.info("WNS 非负或未提供，跳过 PIN2PIN 更新")
        return

    endpoints = _to_cpu_tensor(timing_prop.end_points).numpy().astype(np.int64)
    if endpoints.size == 0:
        logging.info("没有端点，跳过 PIN2PIN 更新")
        return

    if nendpoints > 0:
        violating_endpoints = np.asarray(
            _select_worst_endpoints(pin_slack_np, endpoints, nendpoints),
            dtype=np.int64,
        )
    else:
        endpoint_slack = pin_slack_np[endpoints]
        violating_endpoints = endpoints[endpoint_slack < 0]
    if violating_endpoints.size == 0:
        logging.info("没有违例端点，跳过 PIN2PIN 更新")
        return

    raw_paths = timing_prop.get_critical_paths(violating_endpoints.tolist(), K=1)
    paths = [[int(pin) for pin in path] for path in raw_paths if path]
    logging.info(
        "pin2pin selected endpoints: policy=%s, limit=%d, selected=%d, paths=%d",
        "worst_k" if nendpoints > 0 else "all_violating",
        nendpoints,
        violating_endpoints.size,
        len(paths),
    )

    if net_weighting_cpp is not None and paths:
        pin2node_tensor = torch.from_numpy(pin2node_map.astype(np.int64, copy=False))
        slack_tensor = pin_slack_tensor
        try:
            total_pairs, unique_pairs = net_weighting_cpp.apply_pin2pin(
                paths,
                pin2node_tensor,
                slack_tensor.contiguous(),
                float(wns),
                float(pin2pin_min_weight),
                float(pin2pin_max_weight),
                float(pin2pin_accumulate_weight),
                pin2pin_net_weight,
            )
            logging.info("pin2pin updated pairs (cpp): total=%d, unique=%d", total_pairs, unique_pairs)
            return
        except Exception:
            logging.exception("C++ PIN2PIN 扩展调用失败，回退至 Python 实现")

    total_pairs = 0
    unique_pairs = 0

    for path in paths:
        if not path:
            continue
        path_slack = float(pin_slack_np[path[-1]])
        if wns == 0:
            scale = 0.0
        else:
            scale = path_slack / wns
        prev_pin = None
        prev_node = None
        for pin in path:
            pin = int(pin)
            node_id = int(pin2node_map[pin])
            if prev_pin is not None:
                if prev_node is not None and node_id == prev_node:
                    prev_pin = pin
                    prev_node = node_id
                    continue
                key = (prev_pin, pin)
                if key in pin2pin_net_weight:
                    new_weight = pin2pin_net_weight[key] + pin2pin_accumulate_weight * scale
                    pin2pin_net_weight[key] = float(min(pin2pin_max_weight, new_weight))
                    total_pairs += 1
                else:
                    pin2pin_net_weight[key] = float(pin2pin_min_weight)
                    unique_pairs += 1
            prev_pin = pin
            prev_node = node_id

    logging.info("pin2pin updated pairs: total=%d, unique=%d", total_pairs, unique_pairs)


def _apply_pin2pin_native(
    placedb,
    data_collections,
    timing_prop,
    *,
    wns: float,
    nendpoints: int,
    min_weight: float,
    max_weight: float,
    accumulate_weight: float,
):
    if net_weighting_cpp is None or not hasattr(
        net_weighting_cpp, "accumulate_pin2pin_pairs"
    ):
        raise RuntimeError(
            "native packed pin2pin aggregation is required for production flow"
        )
    if timing_prop is None or not hasattr(
        timing_prop, "extract_pin2pin_critical_paths"
    ):
        raise RuntimeError(
            "native setup-plus-recovery critical path extractor is required"
        )
    if wns is None or not math.isfinite(float(wns)) or float(wns) >= 0.0:
        return {
            "backend": "cpp_openmp_transition_aware",
            "status": "skipped_nonnegative_wns",
            "pair_count": int(getattr(placedb, "length", [0])[0]),
        }

    path_batch = timing_prop.extract_pin2pin_critical_paths(
        global_k=max(0, int(nendpoints)),
        residual_tolerance_ps=1.0e-2,
    )
    if path_batch.selected_state_count > 0 and path_batch.valid_path_count == 0:
        raise RuntimeError(
            "native critical path extractor returned no valid paths for selected states"
        )
    pair_normalization_wns = float(wns)
    if path_batch.selected_state_count > 0:
        pair_normalization_wns = float(
            path_batch.endpoint_slacks.min().detach().cpu().item()
        )
        if not math.isfinite(pair_normalization_wns) or pair_normalization_wns >= 0.0:
            raise RuntimeError(
                "selected max timing states require a finite negative normalization WNS"
            )

    existing_keys = getattr(placedb, "_pin2pin_pair_keys_tensor", None)
    existing_weights = getattr(placedb, "_pin2pin_pair_weights_tensor", None)
    if existing_keys is None:
        existing_keys = torch.empty((0, 2), dtype=torch.int64)
    if existing_weights is None:
        existing_weights = torch.empty((0,), dtype=torch.float64)
    pin2node = torch.as_tensor(
        np.asarray(placedb.pin2node_map, dtype=np.int64),
        dtype=torch.int64,
        device="cpu",
    ).contiguous()
    result = net_weighting_cpp.accumulate_pin2pin_pairs(
        path_batch.path_offsets,
        path_batch.path_pins,
        path_batch.endpoint_slacks,
        path_batch.path_valid,
        pin2node,
        existing_keys,
        existing_weights,
        pair_normalization_wns,
        float(min_weight),
        float(max_weight),
        float(accumulate_weight),
    )

    pair_count = int(result.pair_weights.numel())
    capacity = int(data_collections.weights.numel())
    if pair_count > capacity:
        raise RuntimeError(
            "pin2pin pair count exceeds preallocated capacity: "
            f"{pair_count} > {capacity}"
        )
    placedb._pin2pin_pair_keys_tensor = result.pair_keys
    placedb._pin2pin_pair_weights_tensor = result.pair_weights
    placedb.length[0] = pair_count
    if pair_count:
        data_collections.pairs[: 2 * pair_count].copy_(
            result.pair_keys.reshape(-1).to(
                device=data_collections.pairs.device,
                dtype=data_collections.pairs.dtype,
            )
        )
        data_collections.weights[:pair_count].copy_(
            result.pair_weights.to(
                device=data_collections.weights.device,
                dtype=data_collections.weights.dtype,
            )
        )

    if data_collections.pairs.device.type == "cpu":
        placedb.pin2pin_net_weight.clear()
        for key, weight in zip(
            result.pair_keys.tolist(), result.pair_weights.tolist()
        ):
            placedb.pin2pin_net_weight[(int(key[0]), int(key[1]))] = float(weight)

    extraction_stats = dict(
        getattr(timing_prop, "last_critical_path_extraction_stats", {}) or {}
    )
    state_domain = str(extraction_stats.get("state_domain", "setup_only"))
    summary = {
        "backend": "cpp_openmp_transition_aware",
        "status": "updated",
        "pair_count": pair_count,
        "raw_pair_count": int(result.raw_pair_count),
        "unique_pair_count": int(result.unique_pair_count),
        "updated_pair_count": int(result.updated_pair_count),
        "clamped_pair_count": int(result.clamped_pair_count),
        "aggregation_runtime_ms": float(result.aggregation_runtime_ms),
        "full_endpoint_wns": float(wns),
        "pair_normalization_wns": pair_normalization_wns,
        "pair_normalization_policy": (
            "selected_max_endpoint_gba_wns"
            if state_domain == "setup_plus_recovery"
            else "selected_setup_endpoint_gba_wns"
        ),
        **extraction_stats,
    }
    placedb._pin2pin_last_native_summary = summary
    logging.info(
        "native pin2pin update: states=%d valid_paths=%d pairs=%d "
        "extract_ms=%.3f aggregate_ms=%.3f",
        int(path_batch.selected_state_count),
        int(path_batch.valid_path_count),
        pair_count,
        float(path_batch.extraction_runtime_ms),
        float(result.aggregation_runtime_ms),
    )
    if path_batch.invalid_path_count:
        logging.warning(
            "native critical path extraction retained %d valid and rejected %d invalid paths",
            int(path_batch.valid_path_count),
            int(path_batch.invalid_path_count),
        )
    return summary


def update_net_weights_from_tp(
    scheme: str,
    placedb,
    data_collections,
    timing_propagation,
    *,
    momentum_decay: float,
    max_net_weight: float,
    ignore_net_degree: int,
    npaths: Optional[int] = None,
    wns: Optional[float] = None,
    pin2pin_cfg: Optional[Dict[str, float]] = None,
    nendpoints: Optional[int] = None,
) -> None:
    """Update net weights using timing propagation results."""
    if timing_propagation is None:
        logging.warning("timing propagation operator 未初始化，跳过 net weighting 更新")
        return

    scheme = (scheme or "").lower()
    if scheme == "pin2pin":
        if nendpoints is None:
            nendpoints = 0 if npaths is None else int(npaths)
        elif npaths is not None and int(npaths) != int(nendpoints):
            raise ValueError("nendpoints and legacy npaths disagree")
        nendpoints = int(nendpoints)
        if nendpoints < 0:
            raise ValueError("nendpoints must be non-negative")
        if pin2pin_cfg is None:
            raise ValueError("pin2pin 配置缺失")
        return _apply_pin2pin_native(
            placedb,
            data_collections,
            timing_propagation,
            wns=wns,
            nendpoints=nendpoints,
            min_weight=pin2pin_cfg.get("min_weight", 0.0),
            max_weight=pin2pin_cfg.get("max_weight", float("inf")),
            accumulate_weight=pin2pin_cfg.get("accumulate", 0.0),
        )

    pin_slack_tensor = data_collections.pin_slack
    if pin_slack_tensor is None:
        logging.warning("pin slack 尚未计算，跳过 net weighting 更新")
        return

    if not torch.is_tensor(pin_slack_tensor):
        pin_slack_tensor = torch.as_tensor(pin_slack_tensor)
    pin_slack_tensor_cpu = _to_cpu_tensor(pin_slack_tensor).contiguous()
    if pin_slack_tensor_cpu.dtype not in (torch.float32, torch.float64):
        pin_slack_tensor_cpu = pin_slack_tensor_cpu.to(torch.float32)
    pin_slack_np = pin_slack_tensor_cpu.numpy()

    net_weights = _ensure_numpy_attr(placedb, "net_weights", np.float32)
    net_criticality = _ensure_numpy_attr(placedb, "net_criticality", np.float32)
    netpin_start_base = _ensure_numpy_attr(placedb, "flat_net2pin_start_map", np.int32)
    degree_map = (netpin_start_base[1:] - netpin_start_base[:-1]).astype(np.int64, copy=False)
    flat_netpin = _ensure_numpy_attr(placedb, "flat_net2pin_map", np.int32).astype(np.int64, copy=False)
    netpin_start = netpin_start_base.astype(np.int64, copy=False)
    pin2net_map = _ensure_numpy_attr(placedb, "pin2net_map", np.int32).astype(np.int64, copy=False)
    pin2node_map = _ensure_numpy_attr(placedb, "pin2node_map", np.int32).astype(np.int64, copy=False)

    old_weights = net_weights.copy()

    legacy_npaths = 0 if npaths is None else int(npaths)
    if legacy_npaths < 0:
        raise ValueError("npaths must be non-negative")

    if scheme == "adams":
        _apply_adams(
            net_weights,
            net_criticality,
            degree_map,
            ignore_net_degree,
            pin2net_map,
            pin_slack_np,
            timing_propagation,
            legacy_npaths,
            max_net_weight,
        )
    elif scheme == "lilith":
        _apply_lilith(
            net_weights,
            net_criticality,
            degree_map,
            ignore_net_degree,
            flat_netpin,
            netpin_start,
            pin_slack_np,
            pin_slack_tensor_cpu,
            wns,
            momentum_decay,
            max_net_weight,
        )
    else:
        logging.warning("不支持的 net-weighting 策略: %s", scheme)
        return

    if hasattr(placedb, "net_weight_deltas") and placedb.net_weight_deltas is not None:
        delta = net_weights - old_weights
        if isinstance(placedb.net_weight_deltas, np.ndarray):
            placedb.net_weight_deltas[:] = delta
        else:
            placedb.net_weight_deltas = delta.astype(np.float32).tolist()

    logging.info("finish net-weighting update")
    return {"backend": "legacy_net_weight", "status": "updated"}
