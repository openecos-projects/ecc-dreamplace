from dataclasses import dataclass, field
import json
import logging
import os
import time

import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferingInnerLoopSummary:
    iterations: int
    final_projection: object = None
    final_projection_wall_ms: float = 0.0
    metrics_trace: tuple = field(default_factory=tuple)
    projection_trace: tuple = field(default_factory=tuple)
    periodic_integer_projection_trace: tuple = field(default_factory=tuple)
    strategy: str = "continuous"
    terminal_metrics: dict = field(default_factory=dict)
    terminal_reason: str = "completed"
    trace_path: str = ""
    trace_sha256: str = ""


def _should_project(lane, iteration, project_interval):
    if project_interval is not None:
        return int(project_interval) > 0 and (iteration + 1) % int(project_interval) == 0
    if hasattr(lane, "should_project"):
        return bool(lane.should_project(iteration))
    return False


def _project_and_refresh(lane, *, iteration, final):
    projection = lane.project(iteration=iteration, final=final)
    lane.refresh_runtime_state(projection)
    return projection


def _periodic_integer_projection_config(lane):
    config = getattr(lane, "config", None)
    interval = int(getattr(config, "segment_integer_projection_interval", 0) or 0)
    start_step = int(getattr(config, "segment_integer_projection_start_step", 0) or 0)
    return {
        "enabled": interval > 0,
        "interval": interval,
        "start_step": max(0, start_step),
        "project_bsu": bool(
            getattr(config, "segment_integer_projection_project_bsu", False)
        ),
        "reset_optimizer_state": bool(
            getattr(config, "segment_integer_projection_reset_optimizer_state", True)
        ),
    }


def _should_project_integer_state(iteration, projection_config):
    if not bool(projection_config.get("enabled")):
        return False
    step_number = int(iteration) + 1
    start_step = int(projection_config.get("start_step", 0) or 0)
    interval = int(projection_config.get("interval", 0) or 0)
    return step_number >= start_step and interval > 0 and step_number % interval == 0


def _serializable_metric(value):
    if torch.is_tensor(value):
        detached = value.detach()
        if detached.numel() == 1:
            return float(detached.cpu().item())
        return detached.cpu().tolist()
    return value


def _metrics_snapshot(loss, metrics):
    snapshot = {
        key: _serializable_metric(value)
        for key, value in dict(metrics or {}).items()
    }
    snapshot.setdefault("loss", _serializable_metric(loss))
    return snapshot


def _segment_probe_tensor_stats(tensor):
    value = tensor.detach().float().cpu()
    if value.numel() == 0:
        return {
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
        }
    positive = value > 0
    negative = value < 0
    zero = value == 0
    abs_value = value.abs()
    quantiles = torch.quantile(
        value,
        torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=value.dtype),
    )
    abs_quantiles = torch.quantile(
        abs_value,
        torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=value.dtype),
    )
    return {
        "count": int(value.numel()),
        "positive_count": int(positive.sum().item()),
        "negative_count": int(negative.sum().item()),
        "zero_count": int(zero.sum().item()),
        "positive_ratio": float(positive.float().mean().item()),
        "negative_ratio": float(negative.float().mean().item()),
        "zero_ratio": float(zero.float().mean().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "mean_abs": float(abs_value.mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
        "quantiles": {
            "q0": float(quantiles[0].item()),
            "q01": float(quantiles[1].item()),
            "q10": float(quantiles[2].item()),
            "q50": float(quantiles[3].item()),
            "q90": float(quantiles[4].item()),
            "q99": float(quantiles[5].item()),
            "q100": float(quantiles[6].item()),
        },
        "abs_quantiles": {
            "q0": float(abs_quantiles[0].item()),
            "q01": float(abs_quantiles[1].item()),
            "q10": float(abs_quantiles[2].item()),
            "q50": float(abs_quantiles[3].item()),
            "q90": float(abs_quantiles[4].item()),
            "q99": float(abs_quantiles[5].item()),
            "q100": float(abs_quantiles[6].item()),
        },
    }


def _segment_probe_rows(state, grad, *, z_before=None, max_rows=None):
    grad_cpu = grad.detach().float().cpu()
    z_cpu = state.z_param.detach().float().cpu()
    bsu_cpu = state.bsu_index_param.detach().float().cpu()
    if z_before is not None:
        z_before_cpu = z_before.detach().float().cpu()
    else:
        z_before_cpu = None
    segment_rows = tuple(getattr(state, "segment_rows", ()) or ())
    row_count = int(grad_cpu.numel())
    if max_rows is not None:
        row_count = min(row_count, int(max_rows))
    rows = []
    for index in range(row_count):
        base = dict(segment_rows[index]) if index < len(segment_rows) else {}
        grad_value = float(grad_cpu[index].item())
        row = {
            "segment_id": int(base.get("segment_id", index)),
            "net_id": int(base.get("net_id", -1)),
            "parent_node_id": int(base.get("parent_node_id", -1)),
            "child_node_id": int(base.get("child_node_id", -1)),
            "grad_z": grad_value,
            "grad_z_abs": abs(grad_value),
            "grad_z_sign": 1 if grad_value > 0 else (-1 if grad_value < 0 else 0),
            "z": float(z_cpu[index].item()),
            "bsu": float(bsu_cpu[index].item()),
        }
        if "net_name" in base:
            row["net_name"] = base["net_name"]
        if z_before_cpu is not None:
            before = float(z_before_cpu[index].item())
            delta = row["z"] - before
            row["z_before_step"] = before
            row["delta_z_after_step"] = delta
            row["delta_z_sign"] = 1 if delta > 0 else (-1 if delta < 0 else 0)
        rows.append(row)
    return rows


def _segment_probe_top_rows(state, grad, *, z_before=None, top_k=20):
    grad_cpu = grad.detach().float().cpu()
    if grad_cpu.numel() == 0:
        return {"most_negative_grad_z": [], "most_positive_grad_z": [], "largest_abs_grad_z": []}
    k = min(int(top_k), int(grad_cpu.numel()))

    def rows_for(indices):
        subset = _segment_probe_rows(state, grad, z_before=z_before)
        return [subset[int(index)] for index in indices]

    most_negative = torch.topk(-grad_cpu, k=k).indices.tolist()
    most_positive = torch.topk(grad_cpu, k=k).indices.tolist()
    largest_abs = torch.topk(grad_cpu.abs(), k=k).indices.tolist()
    return {
        "most_negative_grad_z": rows_for(most_negative),
        "most_positive_grad_z": rows_for(most_positive),
        "largest_abs_grad_z": rows_for(largest_abs),
    }


def _segment_grad_snapshot(state, *, label, loss, metrics=None, z_before=None, full_rows=True):
    grad = getattr(state.z_param, "grad", None)
    if grad is None:
        return {"label": label, "status": "missing_grad"}
    z_value = state.z_param.detach().float()
    bsu_value = state.bsu_index_param.detach().float()
    snapshot = {
        "label": label,
        "status": "ok",
        "loss": _serializable_metric(loss),
        "tns": _serializable_metric((metrics or {}).get("tns")),
        "wns": _serializable_metric((metrics or {}).get("wns")),
        "grad_z_stats": _segment_probe_tensor_stats(grad),
        "z_stats": _segment_probe_tensor_stats(z_value),
        "bsu_stats": _segment_probe_tensor_stats(bsu_value),
        "top_segments": _segment_probe_top_rows(state, grad, z_before=z_before),
    }
    if z_before is not None:
        delta = z_value.cpu() - z_before.detach().float().cpu()
        snapshot["delta_z_after_step_stats"] = _segment_probe_tensor_stats(delta)
    if full_rows:
        snapshot["segments"] = _segment_probe_rows(state, grad, z_before=z_before)
    return snapshot


def _segment_grad_probe_path():
    path = os.environ.get("AIMP_BUFFERING_SEGMENT_GRAD_PROBE_JSON")
    if not path:
        return None
    return os.path.abspath(path)


def _segment_directional_probe_path():
    path = os.environ.get("AIMP_BUFFERING_SEGMENT_DIRECTIONAL_PROBE_JSON")
    if not path:
        return None
    return os.path.abspath(path)


def _segment_forced_z_probe_path():
    path = os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_PROBE_JSON")
    if not path:
        return None
    return os.path.abspath(path)


def _device_probe_path():
    path = os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON")
    if not path:
        return None
    return os.path.abspath(path)


def _device_probe_forced_only():
    return str(
        os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_FORCED_ONLY", "")
    ).strip() in {"1", "true", "True", "yes", "on"}


class _ForcedDeviceProbeScope:
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.saved_path = None

    def __enter__(self):
        if not self.enabled:
            return
        self.saved_path = os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON")
        forced_path = os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_FORCED_JSON")
        if forced_path:
            os.environ["AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON"] = forced_path

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self.saved_path is None:
            os.environ.pop("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", None)
        else:
            os.environ["AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON"] = self.saved_path
        return False


def _disable_device_probe_for_unforced_frame():
    if not _device_probe_forced_only():
        return None
    saved_path = os.environ.pop("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", None)
    if saved_path:
        os.environ.setdefault(
            "AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_FORCED_JSON",
            saved_path,
        )
    return saved_path


def _restore_device_probe_after_unforced_frame(saved_path):
    if saved_path is not None:
        os.environ["AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON"] = saved_path


def _write_segment_grad_probe(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _loss_float(lane, model, *, device_probe=False):
    with torch.no_grad(), _ForcedDeviceProbeScope(device_probe):
        loss, metrics = lane.objective(model)
    return {
        "loss": _serializable_metric(loss),
        "tns": _serializable_metric(metrics.get("tns")),
        "wns": _serializable_metric(metrics.get("wns")),
    }


def _tensor_pin_value(tensor, pin_id):
    if not torch.is_tensor(tensor) or pin_id is None:
        return None
    pin_id = int(pin_id)
    if pin_id < 0 or pin_id >= int(tensor.numel()):
        return None
    value = tensor.detach().flatten()[pin_id]
    if not bool(torch.isfinite(value).detach().cpu().item()):
        return None
    return float(value.detach().cpu().item())


def _timing_op_pin_state(model, pin_id):
    timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
    if timing_op is None:
        return {"status": "missing_timing_op"}
    return {
        "status": "ok",
        "pin_id": None if pin_id is None else int(pin_id),
        "r_slew": _tensor_pin_value(
            getattr(timing_op, "pin_rtran_live", None),
            pin_id,
        ),
        "f_slew": _tensor_pin_value(
            getattr(timing_op, "pin_ftran_live", None),
            pin_id,
        ),
        "r_cap": _tensor_pin_value(
            getattr(timing_op, "pin_net_cap_rise_live", None),
            pin_id,
        ),
        "f_cap": _tensor_pin_value(
            getattr(timing_op, "pin_net_cap_fall_live", None),
            pin_id,
        ),
        "r_arrival": _tensor_pin_value(
            getattr(timing_op, "pin_rAAT_live_snapshot", None),
            pin_id,
        ),
        "f_arrival": _tensor_pin_value(
            getattr(timing_op, "pin_fAAT_live_snapshot", None),
            pin_id,
        ),
    }


def _driver_pin_by_net_from_lane(lane):
    payload = getattr(lane, "buffer_relaxed_timing_payload", None)
    if payload is None:
        data_collections = getattr(lane, "data_collections", None)
        payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
    if not isinstance(payload, dict):
        return {}
    result = {}
    for fallback, net in enumerate(payload.get("nets", ()) or ()):
        net_id = int(net.get("net_id", fallback))
        driver_pin = net.get("driver_pin_id")
        if driver_pin is None:
            driver_pin = (net.get("rc_tree") or {}).get("root_node_id")
        if driver_pin is not None:
            result[net_id] = int(driver_pin)
    return result


def _segment_probe_driver_pin_ids(state, indices, lane=None):
    segment_rows = tuple(getattr(state, "segment_rows", ()) or ())
    driver_pin_by_net = _driver_pin_by_net_from_lane(lane) if lane is not None else {}
    driver_pins = []
    seen = set()
    for index in [int(item) for item in indices]:
        if index < 0 or index >= len(segment_rows):
            continue
        row = dict(segment_rows[index])
        net_id = int(row.get("net_id", -1))
        pin_id = driver_pin_by_net.get(net_id)
        source = "payload_net_driver_pin"
        if pin_id is None:
            pin_id = row.get("driver_pin_id")
            source = "segment_row_driver_pin"
        if pin_id is None:
            pin_id = row.get("root_node_id")
            source = "segment_row_root_node"
        if pin_id is None:
            pin_id = row.get("parent_node_id")
            source = "segment_parent_node_fallback"
        if pin_id is None:
            continue
        pin_id = int(pin_id)
        if pin_id in seen:
            continue
        seen.add(pin_id)
        driver_pins.append(
            {
                "segment_index": int(index),
                "segment_id": int(row.get("segment_id", index)),
                "net_id": int(row.get("net_id", -1)),
                "net_name": row.get("net_name"),
                "driver_pin_id": pin_id,
                "driver_pin_source": source,
            }
        )
    return driver_pins


def _capture_driver_py_states(model, driver_pins):
    states = []
    for row in driver_pins:
        state = _timing_op_pin_state(model, row.get("driver_pin_id"))
        states.append({**row, **state})
    return states


def _segment_directional_probe(state, lane, model, base_loss, base_metrics):
    z_param = getattr(state, "z_param", None)
    bsu_param = getattr(state, "bsu_index_param", None)
    if z_param is None or bsu_param is None:
        return {"status": "missing_segment_parameters"}
    z_grad = getattr(z_param, "grad", None)
    bsu_grad = getattr(bsu_param, "grad", None)
    if z_grad is None or bsu_grad is None:
        return {"status": "missing_grad"}

    z_base = z_param.detach().clone()
    bsu_base = bsu_param.detach().clone()
    z_grad_value = z_grad.detach().clone()
    bsu_grad_value = bsu_grad.detach().clone()
    eps_values = (1e-9, 1e-8, 1e-7, 1e-6)

    def eval_direction(label, *, z_direction=None, bsu_direction=None):
        rows = []
        for eps in eps_values:
            with torch.no_grad():
                z_param.copy_(z_base)
                bsu_param.copy_(bsu_base)
                if z_direction is not None:
                    z_param.add_(z_direction, alpha=float(eps))
                if bsu_direction is not None:
                    bsu_param.add_(bsu_direction, alpha=float(eps))
                if hasattr(state, "project_"):
                    state.project_()
                elif hasattr(state, "project_bsu_index_"):
                    state.project_bsu_index_()
            measured = _loss_float(lane, model, device_probe=True)
            rows.append(
                {
                    "eps": float(eps),
                    **measured,
                    "delta_loss": float(measured["loss"] - _serializable_metric(base_loss)),
                }
            )
        return {"label": label, "rows": rows}

    try:
        return {
            "status": "ok",
            "base": {
                "loss": _serializable_metric(base_loss),
                "tns": _serializable_metric((base_metrics or {}).get("tns")),
                "wns": _serializable_metric((base_metrics or {}).get("wns")),
            },
            "grad_z_stats": _segment_probe_tensor_stats(z_grad_value),
            "grad_bsu_stats": _segment_probe_tensor_stats(bsu_grad_value),
            "directions": [
                eval_direction("minus_raw_z_grad", z_direction=-z_grad_value),
                eval_direction("minus_raw_bsu_grad", bsu_direction=-bsu_grad_value),
                eval_direction(
                    "minus_raw_z_and_bsu_grad",
                    z_direction=-z_grad_value,
                    bsu_direction=-bsu_grad_value,
                ),
                eval_direction("minus_sign_z_grad", z_direction=-torch.sign(z_grad_value)),
                eval_direction(
                    "minus_sign_z_and_bsu_grad",
                    z_direction=-torch.sign(z_grad_value),
                    bsu_direction=-torch.sign(bsu_grad_value),
                ),
            ],
        }
    finally:
        with torch.no_grad():
            z_param.copy_(z_base)
            bsu_param.copy_(bsu_base)


def _write_segment_directional_probe(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _forced_z_topks():
    raw = os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_TOPKS", "1,4,16,64")
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return tuple(values or (1, 4, 16, 64))


def _forced_z_commit_topk():
    raw = os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_COMMIT_TOPK")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _forced_z_value(state):
    raw = os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_VALUE", "1")
    try:
        value = float(str(raw).strip())
    except ValueError:
        value = 1.0
    return max(0.0, min(value, float(getattr(state, "max_repeater_count", 1))))


def _forced_z_unique_net():
    return str(
        os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_UNIQUE_NET", "")
    ).strip() in {"1", "true", "True", "yes", "on"}


def _forced_z_segment_indices(state):
    raw = os.environ.get("AIMP_BUFFERING_SEGMENT_FORCE_Z_SEGMENT_IDS", "")
    requested = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            requested.append(int(part))
        except ValueError:
            continue
    if not requested:
        return None
    requested_set = set(requested)
    matched = []
    for index, row in enumerate(tuple(getattr(state, "segment_rows", ()) or ())):
        if int(row.get("segment_id", index)) in requested_set:
            matched.append(int(index))
    return torch.as_tensor(
        matched,
        dtype=torch.long,
        device=getattr(state, "z_param").device,
    )


def _filter_indices_unique_net(state, indices):
    segment_rows = tuple(getattr(state, "segment_rows", ()) or ())
    kept = []
    seen_net_ids = set()
    for item in [int(value) for value in indices]:
        row = dict(segment_rows[item]) if item < len(segment_rows) else {}
        net_id = int(row.get("net_id", -1))
        if net_id in seen_net_ids:
            continue
        seen_net_ids.add(net_id)
        kept.append(item)
    return torch.as_tensor(
        kept,
        dtype=torch.long,
        device=getattr(state, "z_param").device,
    )


def _segment_rows_for_indices(state, indices, grad):
    rows = []
    segment_rows = tuple(getattr(state, "segment_rows", ()) or ())
    grad_cpu = grad.detach().float().cpu()
    for index in [int(item) for item in indices]:
        base = dict(segment_rows[index]) if index < len(segment_rows) else {}
        row = {
            "segment_id": int(base.get("segment_id", index)),
            "net_id": int(base.get("net_id", -1)),
            "parent_node_id": int(base.get("parent_node_id", -1)),
            "child_node_id": int(base.get("child_node_id", -1)),
            "grad_z": float(grad_cpu[index].item()) if index < grad_cpu.numel() else None,
        }
        if "net_name" in base:
            row["net_name"] = base["net_name"]
        rows.append(row)
    return rows


def _set_forced_z_selection(state, selected, value=1.0):
    with torch.no_grad():
        state.z_param.zero_()
        if selected.numel() > 0:
            state.z_param[selected] = float(value)
        state.project_()


def _forced_z_prediction_rows(state, lane, model, selected, grad, z0_loss, z0_metrics):
    predictions = {}
    rows = []
    z0_loss_value = _serializable_metric(z0_loss)
    z0_tns_value = _serializable_metric(z0_metrics.get("tns"))
    z0_wns_value = _serializable_metric(z0_metrics.get("wns"))
    for index in [int(item) for item in selected.detach().cpu().tolist()]:
        _set_forced_z_selection(
            state,
            torch.as_tensor([index], dtype=torch.long, device=state.z_param.device),
        )
        measured = _loss_float(lane, model)
        row = _segment_rows_for_indices(state, [index], grad)[0]
        row.update(
            {
                "forced_loss": measured["loss"],
                "forced_tns": measured["tns"],
                "forced_wns": measured["wns"],
                "predicted_delta_loss_vs_z0": float(measured["loss"] - z0_loss_value),
                "predicted_delta_tns_vs_z0": float(measured["tns"] - z0_tns_value),
                "predicted_delta_wns_vs_z0": float(measured["wns"] - z0_wns_value),
            }
        )
        predictions[int(row["segment_id"])] = dict(row)
        rows.append(row)
    state.segment_forced_z_prediction_by_segment_id = predictions
    return rows


def _segment_forced_z_probe(state, lane, model):
    z_param = getattr(state, "z_param", None)
    bsu_param = getattr(state, "bsu_index_param", None)
    if z_param is None or bsu_param is None:
        return {"status": "missing_segment_parameters"}

    z_base = z_param.detach().clone()
    bsu_base = bsu_param.detach().clone()
    z_grad_base = None if z_param.grad is None else z_param.grad.detach().clone()
    bsu_grad_base = None if bsu_param.grad is None else bsu_param.grad.detach().clone()

    try:
        with torch.no_grad():
            z_param.zero_()
            state.project_()
        z_param.grad = None
        bsu_param.grad = None
        saved_device_probe_path = _disable_device_probe_for_unforced_frame()
        try:
            z0_loss, z0_metrics = lane.objective(model)
        finally:
            _restore_device_probe_after_unforced_frame(saved_device_probe_path)
        grad, = torch.autograd.grad(
            z0_loss,
            z_param,
            retain_graph=False,
            create_graph=False,
        )
        grad = grad.detach().clone()
        negative_indices = torch.nonzero(grad < 0, as_tuple=False).flatten()
        if negative_indices.numel() > 0:
            order = torch.argsort(grad[negative_indices], descending=False)
            ranked_negative = negative_indices[order]
        else:
            ranked_negative = negative_indices
        if _forced_z_unique_net():
            ranked_negative = _filter_indices_unique_net(state, ranked_negative)

        top1_driver_pins = _segment_probe_driver_pin_ids(
            state,
            ranked_negative[:1].detach().cpu().tolist(),
            lane,
        )
        z0_driver_py_state = {
            "semantics": (
                "PySTA TimingPropagation live driver pin state after "
                "evaluating the all-zero segment z objective."
            ),
            "selected_driver_pin_count": int(len(top1_driver_pins)),
            "selected_driver_pins": top1_driver_pins,
            "states": _capture_driver_py_states(model, top1_driver_pins),
        }

        forced_results = []
        explicit_selected = _forced_z_segment_indices(state)
        for topk in _forced_z_topks():
            selected = (
                explicit_selected
                if explicit_selected is not None
                else ranked_negative[: min(int(topk), int(ranked_negative.numel()))]
            )
            _set_forced_z_selection(state, selected)
            measured = _loss_float(lane, model)
            selected_driver_pins = _segment_probe_driver_pin_ids(
                state,
                selected.detach().cpu().tolist(),
                lane,
            )
            forced_results.append(
                {
                    "topk_requested": int(topk),
                    "selection_policy": (
                        "explicit_segment_ids"
                        if explicit_selected is not None
                        else "most_negative_grad_z_at_z0_unique_net"
                        if _forced_z_unique_net()
                        else "most_negative_grad_z_at_z0"
                    ),
                    "selected_count": int(selected.numel()),
                    **measured,
                    "delta_loss_vs_z0": float(
                        measured["loss"] - _serializable_metric(z0_loss)
                    ),
                    "delta_tns_vs_z0": float(
                        measured["tns"] - _serializable_metric(z0_metrics.get("tns"))
                    ),
                    "selected_segments": _segment_rows_for_indices(
                        state,
                        selected.detach().cpu().tolist(),
                        grad,
                    ),
                    "driver_py_state": {
                        "semantics": (
                            "PySTA TimingPropagation live driver pin state after "
                            "evaluating this forced-z objective."
                        ),
                        "selected_driver_pin_count": int(len(selected_driver_pins)),
                        "selected_driver_pins": selected_driver_pins,
                        "states": _capture_driver_py_states(model, selected_driver_pins),
                    },
                }
            )

        return {
            "status": "ok",
            "selection_policy": (
                "most_negative_grad_z_at_z0_unique_net"
                if _forced_z_unique_net()
                else "most_negative_grad_z_at_z0"
            ),
            "z0": {
                "loss": _serializable_metric(z0_loss),
                "tns": _serializable_metric(z0_metrics.get("tns")),
                "wns": _serializable_metric(z0_metrics.get("wns")),
            },
            "z0_driver_py_state": z0_driver_py_state,
            "grad_z_stats_at_z0": _segment_probe_tensor_stats(grad),
            "negative_grad_segment_count": int(negative_indices.numel()),
            "forced_results": forced_results,
        }
    finally:
        with torch.no_grad():
            z_param.copy_(z_base)
            bsu_param.copy_(bsu_base)
        z_param.grad = None
        bsu_param.grad = None
        if z_grad_base is not None:
            z_param.grad = z_grad_base
        if bsu_grad_base is not None:
            bsu_param.grad = bsu_grad_base


def _apply_forced_z_commit_topk(state, lane, model, topk):
    z_param = getattr(state, "z_param", None)
    bsu_param = getattr(state, "bsu_index_param", None)
    if z_param is None or bsu_param is None:
        return {"status": "missing_segment_parameters"}

    z_grad_base = None if z_param.grad is None else z_param.grad.detach().clone()
    bsu_grad_base = None if bsu_param.grad is None else bsu_param.grad.detach().clone()
    z_param.grad = None
    bsu_param.grad = None
    with torch.no_grad():
        z_param.zero_()
        state.project_()
    saved_device_probe_path = _disable_device_probe_for_unforced_frame()
    try:
        z0_loss, z0_metrics = lane.objective(model)
    finally:
        _restore_device_probe_after_unforced_frame(saved_device_probe_path)
    grad, = torch.autograd.grad(
        z0_loss,
        z_param,
        retain_graph=False,
        create_graph=False,
    )
    grad = grad.detach().clone()
    negative_indices = torch.nonzero(grad < 0, as_tuple=False).flatten()
    if negative_indices.numel() > 0:
        order = torch.argsort(grad[negative_indices], descending=False)
        ranked_negative = negative_indices[order]
    else:
        ranked_negative = negative_indices
    unique_net = _forced_z_unique_net()
    if unique_net:
        ranked_negative = _filter_indices_unique_net(state, ranked_negative)
    explicit_selected = _forced_z_segment_indices(state)
    selected = (
        explicit_selected
        if explicit_selected is not None
        else ranked_negative[: min(int(topk), int(ranked_negative.numel()))]
    )
    prediction_rows = _forced_z_prediction_rows(
        state,
        lane,
        model,
        selected,
        grad,
        z0_loss,
        z0_metrics,
    )
    forced_z_value = _forced_z_value(state)
    _set_forced_z_selection(state, selected, value=forced_z_value)
    measured = _loss_float(lane, model, device_probe=True)
    z_param.grad = None
    bsu_param.grad = None
    if z_grad_base is not None:
        z_param.grad = z_grad_base
    if bsu_grad_base is not None:
        bsu_param.grad = bsu_grad_base
    return {
        "status": "ok",
        "selection_policy": "most_negative_grad_z_at_z0",
        "effective_selection_policy": (
            "explicit_segment_ids"
            if explicit_selected is not None
            else "most_negative_grad_z_at_z0_unique_net"
            if unique_net
            else "most_negative_grad_z_at_z0"
        ),
        "unique_net": bool(unique_net),
        "topk_requested": int(topk),
        "forced_z_value": float(forced_z_value),
        "selected_count": int(selected.numel()),
        "z0": {
            "loss": _serializable_metric(z0_loss),
            "tns": _serializable_metric(z0_metrics.get("tns")),
            "wns": _serializable_metric(z0_metrics.get("wns")),
        },
        "forced": measured,
        "delta_loss_vs_z0": float(measured["loss"] - _serializable_metric(z0_loss)),
        "delta_tns_vs_z0": float(
            measured["tns"] - _serializable_metric(z0_metrics.get("tns"))
        ),
        "grad_z_stats_at_z0": _segment_probe_tensor_stats(grad),
        "per_segment_predictions": prediction_rows,
        "selected_segments": _segment_rows_for_indices(
            state,
            selected.detach().cpu().tolist(),
            grad,
        ),
    }


def _write_segment_forced_z_probe(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_buffering_inner_loop(
    *,
    lane,
    model,
    optimizer,
    steps,
    project_interval=1,
):
    """Run the differentiable buffering loop without physical backend commit."""

    metrics_trace = []
    projection_trace = []
    periodic_integer_projection_trace = []
    integer_projection_config = _periodic_integer_projection_config(lane)
    if hasattr(lane, "_summary"):
        lane._summary["periodic_integer_projection_enabled"] = bool(
            integer_projection_config["enabled"]
        )
        lane._summary["periodic_integer_projection_interval"] = int(
            integer_projection_config["interval"]
        )
        lane._summary["periodic_integer_projection_start_step"] = int(
            integer_projection_config["start_step"]
        )
        lane._summary["periodic_integer_projection_project_bsu"] = bool(
            integer_projection_config["project_bsu"]
        )
        lane._summary["periodic_integer_projection_reset_optimizer_state"] = bool(
            integer_projection_config["reset_optimizer_state"]
        )
        lane._summary.setdefault("periodic_integer_projection_count", 0)
    segment_grad_probe_path = _segment_grad_probe_path()
    segment_directional_probe_path = _segment_directional_probe_path()
    segment_forced_z_probe_path = _segment_forced_z_probe_path()
    segment_grad_probe_payload = None
    for iteration in range(int(steps)):
        iteration_started_at = time.perf_counter()
        objective_started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        saved_device_probe_path = _disable_device_probe_for_unforced_frame()
        try:
            loss, metrics = lane.objective(model)
        finally:
            _restore_device_probe_after_unforced_frame(saved_device_probe_path)
        objective_forward_ms = float((time.perf_counter() - objective_started_at) * 1000.0)
        backward_started_at = time.perf_counter()
        # Buffering-only runs keep cell positions frozen, so the placement
        # topology (steiner_topo, RC graph) is built once and shared by every
        # inner-loop step; freeing it on the first backward breaks the next
        # step with "Trying to backward through the graph a second time".
        # Each step's own buffer-parameter subgraph is still rebuilt, and the
        # shared part is released when its owner drops it.
        loss.backward(retain_graph=True)
        objective_backward_ms = float((time.perf_counter() - backward_started_at) * 1000.0)
        if segment_directional_probe_path and iteration == 0:
            state = getattr(lane, "_current_state", lambda: None)()
            payload = {
                "artifact": "segment_count_directional_loss_probe",
                "artifact_version": 1,
                "iteration": int(iteration),
                "probe": _segment_directional_probe(state, lane, model, loss, metrics)
                if state is not None
                else {"status": "missing_state"},
            }
            _write_segment_directional_probe(segment_directional_probe_path, payload)
            logger.info(
                "Wrote segment directional loss probe to %s",
                segment_directional_probe_path,
            )
        if segment_forced_z_probe_path and iteration == 0:
            state = getattr(lane, "_current_state", lambda: None)()
            payload = {
                "artifact": "segment_count_force_z_objective_probe",
                "artifact_version": 1,
                "iteration": int(iteration),
                "probe": _segment_forced_z_probe(state, lane, model)
                if state is not None
                else {"status": "missing_state"},
            }
            _write_segment_forced_z_probe(segment_forced_z_probe_path, payload)
            logger.info(
                "Wrote segment forced-z objective probe to %s",
                segment_forced_z_probe_path,
            )
        if segment_grad_probe_path and iteration == 0:
            state = getattr(lane, "_current_state", lambda: None)()
            if state is not None and hasattr(state, "z_param"):
                segment_grad_probe_payload = {
                    "artifact": "segment_count_grad_z_sign_probe",
                    "artifact_version": 1,
                    "gradient_convention": (
                        "z_param.grad is d(loss)/dz for loss=-tns. "
                        "For plain gradient descent, positive grad_z lowers z; "
                        "negative grad_z raises z. Adam uses the same first-step sign."
                    ),
                    "optimizer": type(optimizer).__name__,
                    "iteration": int(iteration),
                    "initial": _segment_grad_snapshot(
                        state,
                        label="initial_before_optimizer_step",
                        loss=loss,
                        metrics=metrics,
                    ),
                }
                z_before_first_step = state.z_param.detach().clone()
            else:
                z_before_first_step = None
        else:
            z_before_first_step = None
        if hasattr(lane, "before_step"):
            lane.before_step(iteration, loss, metrics)
        optimizer.step()
        if hasattr(lane, "after_step"):
            lane.after_step(iteration, loss, metrics)
        periodic_projection = None
        if _should_project_integer_state(iteration, integer_projection_config):
            periodic_projection = lane.project_segment_count_optimizer_state_to_integer(
                project_bsu=integer_projection_config["project_bsu"],
                reset_optimizer_state=integer_projection_config[
                    "reset_optimizer_state"
                ],
                optimizer=optimizer,
                iteration=iteration,
            )
            periodic_integer_projection_trace.append(periodic_projection)
        if segment_grad_probe_payload is not None and iteration == 0:
            optimizer.zero_grad(set_to_none=True)
            saved_device_probe_path = _disable_device_probe_for_unforced_frame()
            try:
                after_loss, after_metrics = lane.objective(model)
            finally:
                _restore_device_probe_after_unforced_frame(saved_device_probe_path)
            after_loss.backward(retain_graph=True)
            state = getattr(lane, "_current_state", lambda: None)()
            if state is not None and hasattr(state, "z_param"):
                segment_grad_probe_payload["after_first_optimizer_step"] = (
                    _segment_grad_snapshot(
                        state,
                        label="after_first_optimizer_step",
                        loss=after_loss,
                        metrics=after_metrics,
                        z_before=z_before_first_step,
                    )
                )
                segment_grad_probe_payload["after_step_loss"] = _serializable_metric(
                    after_loss
                )
                segment_grad_probe_payload["after_step_tns"] = _serializable_metric(
                    after_metrics.get("tns")
                )
                segment_grad_probe_payload["after_step_wns"] = _serializable_metric(
                    after_metrics.get("wns")
                )
            _write_segment_grad_probe(
                segment_grad_probe_path,
                segment_grad_probe_payload,
            )
            logger.info(
                "Wrote segment grad-z sign probe to %s",
                segment_grad_probe_path,
            )
        objective_step_wall_ms = float((time.perf_counter() - objective_started_at) * 1000.0)
        snapshot = _metrics_snapshot(loss, metrics)
        snapshot["objective_forward_ms"] = objective_forward_ms
        snapshot["objective_backward_ms"] = objective_backward_ms
        snapshot["objective_step_wall_ms"] = objective_step_wall_ms
        if periodic_projection is not None:
            snapshot["periodic_integer_projection_status"] = periodic_projection.get(
                "status"
            )
            snapshot["periodic_integer_projection_z_nonzero_count_after"] = (
                periodic_projection.get("z_nonzero_count_after")
            )
            snapshot["periodic_integer_projection_z_sum_after"] = (
                periodic_projection.get("z_sum_after")
            )
            snapshot["periodic_integer_projection_z_max_after"] = (
                periodic_projection.get("z_max_after")
            )
        projection_refresh_wall_ms = 0.0
        if _should_project(lane, iteration, project_interval):
            projection_started_at = time.perf_counter()
            projection = _project_and_refresh(lane, iteration=iteration, final=False)
            projection_refresh_wall_ms = float(
                (time.perf_counter() - projection_started_at) * 1000.0
            )
            projection_trace.append(projection)
        snapshot["projection_refresh_wall_ms"] = projection_refresh_wall_ms
        snapshot["step_wall_ms"] = float((time.perf_counter() - iteration_started_at) * 1000.0)
        metrics_trace.append(snapshot)
        logger.info(
            "Buffering inner-loop iteration %d/%d loss=%s step_wall_ms=%.3f "
            "objective_step_wall_ms=%.3f projection_refresh_wall_ms=%.3f",
            iteration + 1,
            int(steps),
            snapshot.get("loss"),
            snapshot["step_wall_ms"],
            objective_step_wall_ms,
            projection_refresh_wall_ms,
        )

    final_projection_started_at = time.perf_counter()
    forced_z_commit_topk = _forced_z_commit_topk()
    if forced_z_commit_topk is not None:
        state = getattr(lane, "_current_state", lambda: None)()
        payload = (
            _apply_forced_z_commit_topk(state, lane, model, forced_z_commit_topk)
            if state is not None
            else {"status": "missing_state"}
        )
        if hasattr(lane, "_summary"):
            lane._summary["forced_z_commit_probe"] = payload
    final_projection = _project_and_refresh(lane, iteration=int(steps), final=True)
    final_projection_wall_ms = float(
        (time.perf_counter() - final_projection_started_at) * 1000.0
    )
    projection_trace.append(final_projection)
    logger.info(
        "Buffering inner-loop final projection wall_ms=%.3f",
        final_projection_wall_ms,
    )
    return BufferingInnerLoopSummary(
        iterations=int(steps),
        final_projection=final_projection,
        final_projection_wall_ms=final_projection_wall_ms,
        metrics_trace=tuple(metrics_trace),
        projection_trace=tuple(projection_trace),
        periodic_integer_projection_trace=tuple(periodic_integer_projection_trace),
    )
