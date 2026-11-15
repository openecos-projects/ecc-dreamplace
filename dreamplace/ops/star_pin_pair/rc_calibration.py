"""Cached Xplace-style RC calibration for the driver/sink Star model.

A trusted Steiner sample anchors the same three inexpensive corrections used
by Xplace's ``update_rc_timing`` path: a multiplicative root-load ratio, a
non-negative sink-delay compensation, and a reference sink impulse.  The
calibration is value-only; it does not claim to reproduce the Steiner model's
coordinate Jacobian.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch


_MODES = ("generic", "rise", "fall")
XPLACE_CALIBRATION_POLICY = (
    "xplace_pinload_ratio_positive_delay_compensation_reference_impulse_v3"
)
_WIRELENGTH_FLOOR_UM = 1.0e-12
_LOAD_FLOOR = 1.0e-12


def pair_wirelength_by_net(
    pair_length_um: torch.Tensor,
    pair_net_ids: torch.Tensor,
    num_nets: int,
):
    """Sum direct driver/sink Manhattan lengths in timing-net order."""
    length = torch.as_tensor(pair_length_um).reshape(-1)
    net_ids = torch.as_tensor(pair_net_ids).reshape(-1).long()
    if int(length.numel()) != int(net_ids.numel()):
        raise ValueError("pair lengths and net IDs must have equal length")
    if num_nets < 0 or (
        net_ids.numel()
        and bool(torch.any((net_ids < 0) | (net_ids >= int(num_nets))).item())
    ):
        raise ValueError("pair net ID is outside the timing-net universe")
    if net_ids.device != length.device:
        net_ids = net_ids.to(device=length.device)
    result = length.new_zeros(int(num_nets))
    if net_ids.numel():
        result = result.scatter_add(0, net_ids, length)
    return result


def hpwl_wirelength_by_net(
    *,
    position: torch.Tensor,
    pair_net_ids: torch.Tensor,
    driver_pin_ids: torch.Tensor,
    sink_pin_ids: torch.Tensor,
    num_nets: int,
    coordinate_to_um: float,
    num_pins: Optional[int] = None,
) -> torch.Tensor:
    """Compute exact Manhattan HPWL (in um) per timing net from driver/sink pairs."""
    xy = _canonical_position_xy(position, num_pins=num_pins)
    net_ids = torch.as_tensor(pair_net_ids).reshape(-1).long()
    drivers = torch.as_tensor(driver_pin_ids).reshape(-1).long()
    sinks = torch.as_tensor(sink_pin_ids).reshape(-1).long()
    if not (int(net_ids.numel()) == int(drivers.numel()) == int(sinks.numel())):
        raise ValueError("HPWL pair vectors must have equal length")
    if num_nets < 0 or (
        net_ids.numel()
        and bool(torch.any((net_ids < 0) | (net_ids >= int(num_nets))).item())
    ):
        raise ValueError("HPWL pair net ID is outside the timing-net universe")
    coord_scale = float(coordinate_to_um)
    if not math.isfinite(coord_scale) or coord_scale <= 0.0:
        raise ValueError("coordinate_to_um must be finite and positive")
    result = xy.new_zeros(int(num_nets))
    if net_ids.numel() == 0:
        return result
    dev = xy.device
    net_ids_dev = net_ids.to(device=dev)
    drv_xy = xy.index_select(0, drivers.to(device=dev))
    snk_xy = xy.index_select(0, sinks.to(device=dev))
    pair_max_xy = torch.maximum(drv_xy, snk_xy)
    pair_min_xy = torch.minimum(drv_xy, snk_xy)
    present_nets, inv = torch.unique(net_ids_dev, sorted=True, return_inverse=True)
    idx2 = inv.unsqueeze(1).expand(-1, 2)
    p_max = xy.new_empty((int(present_nets.numel()), 2)).scatter(0, idx2, drv_xy)
    p_min = p_max.clone()
    p_max.scatter_reduce_(0, idx2, pair_max_xy, reduce="amax", include_self=True)
    p_min.scatter_reduce_(0, idx2, pair_min_xy, reduce="amin", include_self=True)
    present_hpwl_um = (p_max - p_min).clamp(min=0.0).sum(dim=1) / coord_scale
    return result.scatter(0, present_nets, present_hpwl_um)


def tree_wirelength_by_net(
    *,
    edge_length_um: torch.Tensor,
    net_flat_topo_sort: torch.Tensor,
    net_flat_topo_sort_start: torch.Tensor,
    flat_pin_to: torch.Tensor,
):
    """Sum rooted Steiner edge lengths in the topology's net order."""
    edge_length = torch.as_tensor(edge_length_um).reshape(-1)
    topo = torch.as_tensor(net_flat_topo_sort).reshape(-1).long()
    starts = torch.as_tensor(net_flat_topo_sort_start).reshape(-1).long()
    edge_to = torch.as_tensor(flat_pin_to).reshape(-1).long()
    if starts.numel() < 2 or int(starts[0].item()) != 0:
        raise ValueError("Steiner topology starts must begin at zero")
    if int(starts[-1].item()) != int(topo.numel()) or bool(
        torch.any(starts[1:] < starts[:-1]).item()
    ):
        raise ValueError("Steiner topology starts are invalid")
    if int(edge_length.numel()) != int(edge_to.numel()):
        raise ValueError("Steiner edge lengths and child IDs must have equal length")
    if topo.numel() and bool(
        torch.any((topo < 0) | (topo >= int(topo.numel()))).item()
    ):
        raise ValueError("Steiner topology contains an invalid vertex")
    if edge_to.numel() and bool(
        torch.any((edge_to < 0) | (edge_to >= int(topo.numel()))).item()
    ):
        raise ValueError("Steiner edge child is outside the topology")
    num_nets = int(starts.numel() - 1)
    counts = starts[1:] - starts[:-1]
    topo_net = torch.empty(
        topo.numel(), dtype=torch.long, device=topo.device
    )
    topo_net[topo] = torch.repeat_interleave(
        torch.arange(num_nets, dtype=torch.long, device=topo.device), counts
    )
    edge_net = topo_net.index_select(0, edge_to.to(device=topo.device))
    if edge_net.device != edge_length.device:
        edge_net = edge_net.to(device=edge_length.device)
    result = edge_length.new_zeros(num_nets)
    if edge_net.numel():
        result = result.scatter_add(0, edge_net, edge_length)
    return result


def next_lower_overflow_milestone(
    overflow: float,
    *,
    activation_overflow: float = 0.5,
    overflow_step: float = 0.1,
):
    """Return the next Xplace calibration threshold below ``overflow``.

    Calibration first activates at ``activation_overflow``.  After sampling,
    the next checkpoint is the first lower threshold on the fixed decrement
    grid.  Returning ``None`` means that all positive milestones were crossed.
    """
    try:
        current = float(overflow)
        activation = float(activation_overflow)
        step = float(overflow_step)
    except (TypeError, ValueError) as error:
        raise ValueError("calibration overflow values must be finite") from error
    if not math.isfinite(current):
        raise ValueError("calibration overflow must be finite")
    if not math.isfinite(activation) or not (0.0 < activation <= 1.0):
        raise ValueError("calibration activation overflow must be in (0, 1]")
    if not math.isfinite(step) or not (0.0 < step <= activation):
        raise ValueError(
            "calibration overflow step must be positive and no larger than activation"
        )
    tolerance = max(1.0, abs(activation), abs(current)) * 1.0e-12
    threshold = activation
    while threshold >= current - tolerance:
        threshold -= step
    if threshold <= tolerance:
        return None
    return round(threshold, 12)


def _canonical_position_xy(value: torch.Tensor, *, num_pins: Optional[int] = None):
    """Return pin coordinates as [num_pins, 2] without copying devices.

    pin_pos is normally a flat [x_0..x_P-1, y_0..y_P-1] tensor.  The Star
    helper also accepts the two matrix layouts, so calibration must use the
    same convention instead of guessing pairs from flat storage.
    """
    value = torch.as_tensor(value)
    if value.ndim == 1:
        if int(value.numel()) % 2:
            raise RuntimeError("Star RC calibration position must contain 2*P values")
        inferred = int(value.numel()) // 2
        if num_pins is not None and inferred != int(num_pins):
            raise RuntimeError("Star RC calibration position pin count changed")
        return torch.stack((value[:inferred], value[inferred:]), dim=1)
    if value.ndim == 2:
        rows, cols = (int(value.shape[0]), int(value.shape[1]))
        if num_pins is not None:
            if rows == 2 and cols == int(num_pins):
                return value.transpose(0, 1)
            if rows == int(num_pins) and cols == 2:
                return value
        if cols == 2:
            return value
        if rows == 2:
            return value.transpose(0, 1)
    raise RuntimeError(
        "Star RC calibration position must be flat [x..., y...], [2,P], or [P,2]"
    )


def _position_for_pin_ids(
    value: torch.Tensor,
    pin_ids: torch.Tensor,
    *,
    num_pins: Optional[int] = None,
):
    """Gather only selected pin coordinates from any supported layout."""
    value = torch.as_tensor(value)
    ids = torch.as_tensor(pin_ids, dtype=torch.long).reshape(-1)
    if value.ndim == 1:
        if int(value.numel()) % 2:
            raise RuntimeError("Star RC calibration position must contain 2*P values")
        inferred = int(value.numel()) // 2
        if num_pins is not None and inferred != int(num_pins):
            raise RuntimeError("Star RC calibration position pin count changed")
        pin_count = inferred
        x_values = value[:pin_count]
        y_values = value[pin_count:]
    elif value.ndim == 2:
        rows, cols = (int(value.shape[0]), int(value.shape[1]))
        if num_pins is not None and rows == 2 and cols == int(num_pins):
            x_values, y_values = value[0], value[1]
            pin_count = int(num_pins)
        elif num_pins is not None and rows == int(num_pins) and cols == 2:
            x_values, y_values = value[:, 0], value[:, 1]
            pin_count = int(num_pins)
        elif cols == 2:
            x_values, y_values = value[:, 0], value[:, 1]
            pin_count = rows
        elif rows == 2:
            x_values, y_values = value[0], value[1]
            pin_count = cols
        else:
            raise RuntimeError(
                "Star RC calibration position must be flat [x..., y...], [2,P], or [P,2]"
            )
    else:
        raise RuntimeError(
            "Star RC calibration position must be flat [x..., y...], [2,P], or [P,2]"
        )
    if ids.numel() == 0:
        return x_values[:0].new_empty((0, 2))
    min_id = int(ids.min().item())
    max_id = int(ids.max().item())
    if min_id < 0 or max_id >= pin_count:
        raise RuntimeError("Star RC calibration pair pin is outside position tensor")
    ids = ids.to(device=value.device)
    return torch.stack(
        (x_values.index_select(0, ids), y_values.index_select(0, ids)), dim=1
    )


def _position_digest(
    value: torch.Tensor,
    pin_ids: Optional[torch.Tensor] = None,
    *,
    num_pins: Optional[int] = None,
) -> str:
    """Hash only the canonical coordinates used by the calibration pairs."""
    ids = None
    if pin_ids is not None:
        ids = torch.as_tensor(pin_ids, dtype=torch.long).reshape(-1)
        coordinates = _position_for_pin_ids(
            value, ids, num_pins=num_pins
        )
    else:
        coordinates = _canonical_position_xy(value, num_pins=num_pins)
    coordinates = coordinates.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(coordinates.shape)).encode("ascii"))
    digest.update(str(coordinates.dtype).encode("ascii"))
    digest.update(coordinates.view(torch.uint8).numpy().tobytes())
    if ids is not None:
        digest.update(ids.to(device="cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _pair_pin_ids(driver_pin_ids: torch.Tensor, sink_pin_ids: torch.Tensor):
    values = torch.cat((driver_pin_ids.reshape(-1), sink_pin_ids.reshape(-1)))
    if values.numel() == 0:
        return values.to(dtype=torch.long)
    return torch.unique(values.to(dtype=torch.long), sorted=True)


def calibration_policy_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error_summary(candidate: torch.Tensor, reference: torch.Tensor):
    candidate = torch.as_tensor(candidate).detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    reference = torch.as_tensor(reference).detach().reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    if candidate.numel() != reference.numel():
        raise ValueError("calibration fidelity vectors must have equal length")
    finite = torch.isfinite(candidate) & torch.isfinite(reference)
    candidate = candidate[finite]
    reference = reference[finite]
    if candidate.numel() == 0:
        return {
            "count": 0,
            "mae": None,
            "normalized_mae": None,
            "p95_abs_error": None,
            "max_abs_error": None,
            "reference_abs_mean": None,
        }
    absolute_error = torch.abs(candidate - reference)
    reference_abs_mean = torch.abs(reference).mean()
    denominator = max(float(reference_abs_mean.item()), 1.0e-12)
    return {
        "count": int(candidate.numel()),
        "mae": float(absolute_error.mean().item()),
        "normalized_mae": float(absolute_error.mean().item()) / denominator,
        "p95_abs_error": float(torch.quantile(absolute_error, 0.95).item()),
        "max_abs_error": float(absolute_error.max().item()),
        "reference_abs_mean": float(reference_abs_mean.item()),
    }


def _metric_error_summary(candidate, reference):
    candidate = float(candidate)
    reference = float(reference)
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return {
            "absolute_error": None,
            "relative_absolute_error": None,
            "reference_abs": None,
        }
    absolute_error = abs(candidate - reference)
    reference_abs = abs(reference)
    return {
        "absolute_error": absolute_error,
        "relative_absolute_error": absolute_error / max(reference_abs, 1.0e-12),
        "reference_abs": reference_abs,
    }


def calibration_fidelity_summary(
    *,
    star,
    raw_loads: Mapping[str, torch.Tensor],
    raw_delays: Mapping[str, torch.Tensor],
    raw_impulses: Mapping[str, torch.Tensor],
    calibrated_loads: Mapping[str, torch.Tensor],
    calibrated_delays: Mapping[str, torch.Tensor],
    calibrated_impulses: Mapping[str, torch.Tensor],
    teacher_loads: Mapping[str, torch.Tensor],
    teacher_delays: Mapping[str, torch.Tensor],
    teacher_impulses: Mapping[str, torch.Tensor],
    raw_slack: Optional[torch.Tensor] = None,
    calibrated_slack: Optional[torch.Tensor] = None,
    teacher_slack: Optional[torch.Tensor] = None,
    raw_endpoint_slack: Optional[torch.Tensor] = None,
    calibrated_endpoint_slack: Optional[torch.Tensor] = None,
    teacher_endpoint_slack: Optional[torch.Tensor] = None,
    raw_metrics: Optional[Mapping[str, float]] = None,
    calibrated_metrics: Optional[Mapping[str, float]] = None,
    teacher_metrics: Optional[Mapping[str, float]] = None,
):
    """Compare raw and cached Star RC against a fresh Steiner reference."""
    drivers = torch.unique(torch.as_tensor(star.driver_pin_ids).reshape(-1).long())
    sinks = torch.unique(torch.as_tensor(star.sink_pin_ids).reshape(-1).long())
    summary = {
        "contract": "raw_calibrated_fresh_steiner_v1",
        "normalization": "mae_divided_by_reference_abs_mean",
        "modes": {},
    }
    for mode in _MODES:
        fields = {}
        for field, ids, raw_values, calibrated_values, teacher_values in (
            (
                "root_load",
                drivers,
                raw_loads[mode],
                calibrated_loads[mode],
                teacher_loads[mode],
            ),
            (
                "sink_delay",
                sinks,
                raw_delays[mode],
                calibrated_delays[mode],
                teacher_delays[mode],
            ),
            (
                "sink_impulse",
                sinks,
                raw_impulses[mode],
                calibrated_impulses[mode],
                teacher_impulses[mode],
            ),
        ):
            local_raw_ids = ids.to(device=torch.as_tensor(raw_values).device)
            local_calibrated_ids = ids.to(
                device=torch.as_tensor(calibrated_values).device
            )
            local_teacher_ids = ids.to(device=torch.as_tensor(teacher_values).device)
            reference = torch.as_tensor(teacher_values).reshape(-1).index_select(
                0, local_teacher_ids
            )
            fields[field] = {
                "raw": _error_summary(
                    torch.as_tensor(raw_values).reshape(-1).index_select(
                        0, local_raw_ids
                    ),
                    reference,
                ),
                "calibrated": _error_summary(
                    torch.as_tensor(calibrated_values).reshape(-1).index_select(
                        0, local_calibrated_ids
                    ),
                    reference,
                ),
            }
        summary["modes"][mode] = fields
    slack_values = (raw_slack, calibrated_slack, teacher_slack)
    if all(value is not None for value in slack_values):
        summary["pin_slack"] = {
            "raw": _error_summary(raw_slack, teacher_slack),
            "calibrated": _error_summary(calibrated_slack, teacher_slack),
        }
    endpoint_values = (
        raw_endpoint_slack,
        calibrated_endpoint_slack,
        teacher_endpoint_slack,
    )
    if all(value is not None for value in endpoint_values):
        summary["endpoint_slack"] = {
            "raw": _error_summary(raw_endpoint_slack, teacher_endpoint_slack),
            "calibrated": _error_summary(
                calibrated_endpoint_slack, teacher_endpoint_slack
            ),
        }
    metric_values = (raw_metrics, calibrated_metrics, teacher_metrics)
    if all(isinstance(value, Mapping) for value in metric_values):
        names = tuple(
            name
            for name in ("wns", "tns", "ws", "ts")
            if name in raw_metrics
            and name in calibrated_metrics
            and name in teacher_metrics
        )
        summary["timing_metrics"] = {
            "raw": {name: float(raw_metrics[name]) for name in names},
            "calibrated": {
                name: float(calibrated_metrics[name]) for name in names
            },
            "teacher": {name: float(teacher_metrics[name]) for name in names},
        }
        summary["timing_metric_error"] = {
            name: {
                "raw": _metric_error_summary(
                    raw_metrics[name], teacher_metrics[name]
                ),
                "calibrated": _metric_error_summary(
                    calibrated_metrics[name], teacher_metrics[name]
                ),
            }
            for name in names
        }
    return summary


@dataclass(frozen=True)
class StarRCCalibrationState:
    """Immutable cached factors anchored by one trusted Steiner sample."""

    pair_identity_hash: str
    pair_topology_generation: int
    source_snapshot_generation: int
    teacher_topology_generation: int
    position_digest: str
    position_reference: torch.Tensor
    cap_digest: str
    r_unit: float
    c_unit: float
    coordinate_to_um: float
    timing_mode_identity: str
    clock_policy_digest: str
    root_load_scale: Mapping[str, torch.Tensor]
    root_load_valid: Mapping[str, torch.Tensor]
    anchor_star_wirelength_um: torch.Tensor
    anchor_teacher_wirelength_um: torch.Tensor
    delay_compensation: Mapping[str, torch.Tensor]
    reference_impulse: Mapping[str, torch.Tensor]
    calibration_policy: str = XPLACE_CALIBRATION_POLICY
    position_reference_pin_ids: Optional[torch.Tensor] = None
    # Snapshot generation is anchor provenance only.  Weight publication does
    # not invalidate an RC calibration bound to the same pair topology.
    sample_snapshot_generation: int = -1
    position_num_pins: int = -1
    anchor_iteration: int = -1
    anchor_overflow: Optional[float] = None
    anchor_hpwl_wirelength_um: Optional[torch.Tensor] = None
    anchor_steiner_to_hpwl_ratio: Optional[torch.Tensor] = None

    @classmethod
    def from_reference(
        cls,
        *,
        pair_identity_hash: str,
        pair_topology_generation: int,
        source_snapshot_generation: int,
        teacher_topology_generation: int,
        position: torch.Tensor,
        cap_digest: str,
        r_unit: float,
        c_unit: float,
        coordinate_to_um: float,
        timing_mode_identity: str = "generic,rise,fall",
        clock_policy_digest: str = "",
        pair_net_ids: torch.Tensor,
        driver_pin_ids: torch.Tensor,
        sink_pin_ids: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        star_wirelength_by_net: torch.Tensor,
        teacher_wirelength_by_net: torch.Tensor,
        star_loads: Mapping[str, torch.Tensor],
        teacher_loads: Mapping[str, torch.Tensor],
        star_delays: Mapping[str, torch.Tensor],
        teacher_delays: Mapping[str, torch.Tensor],
        teacher_impulses: Mapping[str, torch.Tensor],
        anchor_iteration: int = -1,
        anchor_overflow: Optional[float] = None,
    ) -> "StarRCCalibrationState":
        """Construct cached Xplace factors from one matched RC snapshot."""
        net_ids = torch.as_tensor(pair_net_ids).reshape(-1).long()
        drivers = torch.as_tensor(driver_pin_ids).reshape(-1).long()
        sinks = torch.as_tensor(sink_pin_ids).reshape(-1).long()
        if not (net_ids.numel() == drivers.numel() == sinks.numel()):
            raise ValueError("Star calibration pair vectors must have equal length")
        if net_ids.numel() and bool(torch.any(net_ids < 0).item()):
            raise ValueError("Star calibration net IDs must be non-negative")
        if (
            (drivers.numel() and bool(torch.any(drivers < 0).item()))
            or (sinks.numel() and bool(torch.any(sinks < 0).item()))
        ):
            raise ValueError("Star calibration pin IDs must be non-negative")
        if not isinstance(base_caps, Mapping):
            raise TypeError("base_caps must be a mapping")
        if anchor_overflow is not None and not math.isfinite(float(anchor_overflow)):
            raise ValueError("Star RC calibration anchor overflow must be finite")
        pair_pin_ids = _pair_pin_ids(drivers, sinks)
        position_num_pins = int(torch.as_tensor(base_caps["generic"]).numel())
        position_reference = _position_for_pin_ids(
            position,
            pair_pin_ids,
            num_pins=position_num_pins,
        ).detach().clone()
        root_scale: Dict[str, torch.Tensor] = {}
        root_valid: Dict[str, torch.Tensor] = {}
        delay: Dict[str, torch.Tensor] = {}
        impulse: Dict[str, torch.Tensor] = {}
        star_wirelength = torch.as_tensor(star_wirelength_by_net).reshape(-1)
        teacher_wirelength = torch.as_tensor(teacher_wirelength_by_net).reshape(-1)
        net_count = int(star_wirelength.numel())
        if int(teacher_wirelength.numel()) != net_count:
            raise ValueError("Star and Steiner wirelength vectors must have equal length")
        if net_ids.numel() and int(net_ids.max().item()) >= net_count:
            raise ValueError("Star calibration pair net is outside wirelength vectors")
        teacher_wirelength = teacher_wirelength.to(
            device=star_wirelength.device, dtype=star_wirelength.dtype
        )
        if not bool(torch.isfinite(star_wirelength).all().item()) or not bool(
            torch.isfinite(teacher_wirelength).all().item()
        ):
            raise RuntimeError("non-finite Star/Steiner calibration wirelength")
        if bool(torch.any(star_wirelength < -_WIRELENGTH_FLOOR_UM).item()) or bool(
            torch.any(teacher_wirelength < -_WIRELENGTH_FLOOR_UM).item()
        ):
            raise RuntimeError("negative Star/Steiner calibration wirelength")
        hpwl_wirelength = hpwl_wirelength_by_net(
            position=position,
            pair_net_ids=net_ids,
            driver_pin_ids=drivers,
            sink_pin_ids=sinks,
            num_nets=net_count,
            coordinate_to_um=coordinate_to_um,
            num_pins=position_num_pins,
        ).to(device=star_wirelength.device, dtype=star_wirelength.dtype)
        steiner_to_hpwl_ratio = torch.ones_like(star_wirelength)
        hpwl_valid = hpwl_wirelength > 1.0e-6
        if bool(hpwl_valid.any().item()):
            steiner_to_hpwl_ratio[hpwl_valid] = (
                teacher_wirelength[hpwl_valid] / hpwl_wirelength[hpwl_valid]
            ).clamp(min=1.0, max=4.0)
        present_nets = torch.unique(net_ids, sorted=True)

        for mode in _MODES:
            for name, values in (
                ("star_loads", star_loads),
                ("teacher_loads", teacher_loads),
                ("star_delays", star_delays),
                ("teacher_delays", teacher_delays),
                ("teacher_impulses", teacher_impulses),
            ):
                if not torch.is_tensor(values.get(mode)):
                    raise ValueError("%s requires %s tensor" % (name, mode))
            base = torch.as_tensor(base_caps[mode]).reshape(-1)
            if int(base.numel()) != position_num_pins:
                raise ValueError("calibration base-cap tensors must cover all pins")
            star_load = torch.as_tensor(star_loads[mode]).reshape(-1)
            teacher_load = torch.as_tensor(teacher_loads[mode]).reshape(-1)
            if drivers.numel() and (
                star_load.numel() <= int(drivers.max().item())
                or teacher_load.numel() <= int(drivers.max().item())
            ):
                raise ValueError("teacher load tensor does not cover star drivers")
            local_drivers = drivers.to(device=star_load.device)
            local_nets = net_ids.to(device=star_load.device)
            local_present_nets = present_nets.to(device=star_load.device)
            driver_by_net = torch.zeros(
                net_count, dtype=torch.long, device=star_load.device
            )
            if local_nets.numel():
                driver_by_net = driver_by_net.scatter(
                    0, local_nets, local_drivers
                )
            present_drivers = driver_by_net.index_select(
                0, local_present_nets
            )
            star_root_load = star_load.index_select(0, present_drivers)
            teacher_root_load = teacher_load.to(
                device=star_load.device, dtype=star_load.dtype
            ).index_select(0, present_drivers)
            if not bool(torch.isfinite(star_root_load).all().item()) or not bool(
                torch.isfinite(teacher_root_load).all().item()
            ):
                raise RuntimeError("non-finite Star/Steiner root load")
            load_valid = torch.abs(star_root_load) > _LOAD_FLOOR
            if bool(
                torch.any(
                    (~load_valid) & (torch.abs(teacher_root_load) > _LOAD_FLOOR)
                ).item()
            ):
                raise RuntimeError(
                    "Steiner root load is nonzero where Star root load is zero"
                )
            present_ratio = torch.ones_like(star_root_load)
            present_ratio[load_valid] = (
                teacher_root_load[load_valid] / star_root_load[load_valid]
            )
            if not bool(torch.isfinite(present_ratio).all().item()) or bool(
                torch.any(present_ratio < 0).item()
            ):
                raise RuntimeError("invalid Steiner/Star root-load ratio")
            ratio_by_net = star_load.new_ones(net_count)
            valid_by_net = torch.zeros(
                net_count, dtype=torch.bool, device=star_load.device
            )
            if local_present_nets.numel():
                ratio_by_net = ratio_by_net.scatter(
                    0, local_present_nets, present_ratio
                )
                valid_by_net = valid_by_net.scatter(
                    0, local_present_nets, load_valid
                )
            root_scale[mode] = ratio_by_net.detach().clone()
            root_valid[mode] = valid_by_net.detach().clone()

            star_delay = torch.as_tensor(star_delays[mode]).reshape(-1)
            teacher_delay = torch.as_tensor(teacher_delays[mode]).reshape(-1)
            teacher_impulse = torch.as_tensor(teacher_impulses[mode]).reshape(-1)
            if sinks.numel() and (
                star_delay.numel() <= int(sinks.max().item())
                or teacher_delay.numel() <= int(sinks.max().item())
                or teacher_impulse.numel() <= int(sinks.max().item())
            ):
                raise ValueError("teacher delay tensor does not cover star sinks")
            star_sinks = sinks.to(device=star_delay.device)
            teacher_sinks = sinks.to(device=teacher_delay.device)
            delay[mode] = (
                teacher_delay.index_select(0, teacher_sinks).to(
                    device=star_delay.device, dtype=star_delay.dtype
                )
                - star_delay.index_select(0, star_sinks)
            ).clamp(min=0.0).detach().clone()
            impulse[mode] = teacher_impulse.index_select(
                0, sinks.to(device=teacher_impulse.device)
            ).to(device=star_delay.device, dtype=star_delay.dtype).detach().clone()
            if not bool(torch.isfinite(delay[mode]).all().item()) or not bool(
                torch.isfinite(impulse[mode]).all().item()
            ):
                raise RuntimeError("non-finite Star RC calibration delay or impulse")
        return cls(
            pair_identity_hash=str(pair_identity_hash),
            pair_topology_generation=int(pair_topology_generation),
            source_snapshot_generation=int(source_snapshot_generation),
            teacher_topology_generation=int(teacher_topology_generation),
            position_digest=_position_digest(
                position, pair_pin_ids, num_pins=position_num_pins
            ),
            position_reference=position_reference,
            cap_digest=str(cap_digest),
            r_unit=float(r_unit),
            c_unit=float(c_unit),
            coordinate_to_um=float(coordinate_to_um),
            timing_mode_identity=str(timing_mode_identity),
            clock_policy_digest=str(clock_policy_digest),
            root_load_scale=root_scale,
            root_load_valid=root_valid,
            anchor_star_wirelength_um=star_wirelength.detach().clone(),
            anchor_teacher_wirelength_um=teacher_wirelength.detach().clone(),
            delay_compensation=delay,
            reference_impulse=impulse,
            position_reference_pin_ids=pair_pin_ids.detach().to(device="cpu").contiguous(),
            sample_snapshot_generation=int(source_snapshot_generation),
            position_num_pins=position_num_pins,
            anchor_iteration=int(anchor_iteration),
            anchor_overflow=(
                None if anchor_overflow is None else float(anchor_overflow)
            ),
            anchor_hpwl_wirelength_um=hpwl_wirelength.detach().clone(),
            anchor_steiner_to_hpwl_ratio=steiner_to_hpwl_ratio.detach().clone(),
        )

    def validate(
        self,
        *,
        star,
        position: torch.Tensor,
        cap_digest: str,
        timing_mode_identity: str = "generic,rise,fall",
        clock_policy_digest: str = "",
    ) -> None:
        if self.pair_identity_hash != star.pair_identity_hash:
            raise RuntimeError("stale Star RC calibration pair identity")
        if int(self.pair_topology_generation) != int(star.pair_topology_generation):
            raise RuntimeError("stale Star RC calibration topology generation")
        if self.cap_digest != str(cap_digest):
            raise RuntimeError("stale Star RC calibration capacitance identity")
        if self.timing_mode_identity != str(timing_mode_identity):
            raise RuntimeError("stale Star RC calibration timing mode")
        if self.clock_policy_digest != str(clock_policy_digest):
            raise RuntimeError("stale Star RC calibration clock policy")
        if self.calibration_policy != XPLACE_CALIBRATION_POLICY:
            raise RuntimeError("unsupported Star RC calibration policy")
        for name in ("r_unit", "c_unit", "coordinate_to_um"):
            if float(getattr(star, name)) != float(getattr(self, name)):
                raise RuntimeError("stale Star RC calibration RC unit: %s" % name)
        net_count = int(
            getattr(
                star,
                "num_nets",
                (
                    int(torch.as_tensor(star.pair_net_ids).max().item()) + 1
                    if int(torch.as_tensor(star.pair_net_ids).numel())
                    else 0
                ),
            )
        )
        anchor_star = torch.as_tensor(self.anchor_star_wirelength_um).reshape(-1)
        anchor_teacher = torch.as_tensor(
            self.anchor_teacher_wirelength_um
        ).reshape(-1)
        if int(anchor_star.numel()) != net_count or int(anchor_teacher.numel()) != net_count:
            raise RuntimeError("stale Star RC calibration wirelength shape")
        if not bool(torch.isfinite(anchor_star).all().item()) or not bool(
            torch.isfinite(anchor_teacher).all().item()
        ):
            raise RuntimeError("non-finite Star RC calibration anchor wirelength")
        if self.anchor_steiner_to_hpwl_ratio is not None:
            st_ratio = torch.as_tensor(self.anchor_steiner_to_hpwl_ratio).reshape(-1)
            if int(st_ratio.numel()) != net_count:
                raise RuntimeError("stale Star RC calibration Steiner/HPWL ratio shape")
            if not bool(torch.isfinite(st_ratio).all().item()) or bool(
                torch.any(st_ratio <= 0.0).item()
            ):
                raise RuntimeError("invalid Star RC calibration Steiner/HPWL ratio")
        for mode in _MODES:
            scale = torch.as_tensor(self.root_load_scale[mode]).reshape(-1)
            valid = torch.as_tensor(self.root_load_valid[mode]).reshape(-1)
            if not (int(scale.numel()) == int(valid.numel()) == net_count):
                raise RuntimeError("stale Star RC calibration root-load shape")
            if valid.dtype != torch.bool:
                raise RuntimeError("Star RC calibration valid mask must be boolean")
            if not bool(torch.isfinite(scale).all().item()):
                raise RuntimeError("non-finite Star RC calibration root-load state")
            if bool(torch.any(scale < 0).item()):
                raise RuntimeError("negative Star RC calibration root-load ratio")
        # Position is deliberately not an identity gate.  Xplace calibration
        # is useful only when the anchor factors remain active as placement
        # moves; displacement is retained as telemetry and refresh evidence.
        _canonical_position_xy(
            position,
            num_pins=(
                self.position_num_pins if int(self.position_num_pins) > 0 else None
            ),
        )

    def position_drift_summary(self, position: torch.Tensor):
        """Return displacement statistics for sparse holdout diagnostics."""
        ids = self.position_reference_pin_ids
        if ids is not None:
            ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
            if ids.numel() == 0:
                return {"max": 0.0, "p99": 0.0, "mean": 0.0}
            current = _position_for_pin_ids(
                position,
                ids,
                num_pins=(
                    self.position_num_pins if int(self.position_num_pins) > 0 else None
                ),
            ).detach()
        else:
            current = _canonical_position_xy(position).detach()
        reference = _canonical_position_xy(self.position_reference)
        if current.shape != reference.shape:
            raise RuntimeError("Star RC calibration position shape changed")
        reference = reference.to(device=current.device, dtype=torch.float32)
        current = current.to(dtype=torch.float32)
        displacement = (current - reference).norm(dim=1)
        if displacement.numel() == 0:
            return {"max": 0.0, "p99": 0.0, "mean": 0.0}
        # Bound the quantile work for very large designs.  Max and mean still
        # use every pair; p99 is a deterministic stride sample for diagnostics.
        quantile_values = displacement
        if int(displacement.numel()) > 4096:
            stride = (int(displacement.numel()) + 4095) // 4096
            quantile_values = displacement[::stride]
        return {
            "max": float(displacement.max().item()),
            "p99": float(torch.quantile(quantile_values, 0.99).item()),
            "mean": float(displacement.mean().item()),
        }

    def apply(
        self,
        *,
        star,
        position: torch.Tensor,
        cap_digest: str,
        loads: Mapping[str, torch.Tensor],
        delays: Mapping[str, torch.Tensor],
        impulses: Mapping[str, torch.Tensor],
        timing_mode_identity: str = "generic,rise,fall",
        clock_policy_digest: str = "",
    ):
        self.validate(
            star=star,
            position=position,
            cap_digest=cap_digest,
            timing_mode_identity=timing_mode_identity,
            clock_policy_digest=clock_policy_digest,
        )
        load_device = loads["generic"].device
        delay_device = delays["generic"].device
        net_ids = star.pair_net_ids.to(device=load_device)
        drivers = star.driver_pin_ids.to(device=load_device)
        sinks = star.sink_pin_ids.to(device=delay_device)
        out_loads = dict(loads)
        out_delays = dict(delays)
        out_impulses = dict(impulses)
        for mode in _MODES:
            ratio = self.root_load_scale[mode].to(
                device=loads[mode].device, dtype=loads[mode].dtype
            )
            if net_ids.numel():
                present_nets = torch.unique(net_ids, sorted=True)
                driver_by_net = torch.zeros(
                    ratio.numel(), dtype=torch.long, device=loads[mode].device
                ).scatter(0, net_ids, drivers)
                present_drivers = driver_by_net.index_select(0, present_nets)
                present_ratio = ratio.index_select(0, present_nets)
                root_load = loads[mode].index_select(0, present_drivers)
                calibrated_root = present_ratio * root_load
                root_delta = calibrated_root - root_load
            else:
                present_drivers = drivers
                root_delta = loads[mode].new_zeros(0)
            out_loads[mode] = loads[mode] + torch.zeros_like(loads[mode]).scatter(
                0, present_drivers, root_delta
            )
            delay_delta = self.delay_compensation[mode].to(
                device=delays[mode].device, dtype=delays[mode].dtype
            )
            out_delays[mode] = delays[mode] + torch.zeros_like(delays[mode]).scatter(
                0, sinks, delay_delta
            )
            ref = self.reference_impulse[mode].to(
                device=impulses[mode].device, dtype=impulses[mode].dtype
            )
            impulse_sinks = star.sink_pin_ids.to(device=impulses[mode].device)
            out_impulses[mode] = impulses[mode].clone().scatter(
                0, impulse_sinks, ref
            )
            for name, value in (
                ("load", out_loads[mode]),
                ("delay", out_delays[mode]),
                ("impulse", out_impulses[mode]),
            ):
                if not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(
                        "non-finite calibrated Star RC %s in %s mode" % (name, mode)
                    )
        return out_loads, out_delays, out_impulses


def calibration_cap_digest(base_caps: Mapping[str, torch.Tensor]) -> str:
    """Stable identity for all mode-specific library pin capacitances."""
    digest = hashlib.sha256()
    for mode in _MODES:
        value = torch.as_tensor(base_caps[mode]).detach().to(device="cpu").contiguous()
        digest.update(mode.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ClockRCCalibrationState(StarRCCalibrationState):
    """Clock-domain calibration state using the shared XPlace formulas."""

    domain: str = "clock_lcb_ff"

    def validate(self, **kwargs) -> None:
        if self.domain != "clock_lcb_ff":
            raise RuntimeError("invalid LCB->FF clock calibration domain")
        if int(self.teacher_topology_generation) <= 0:
            raise RuntimeError("LCB->FF clock calibration lacks teacher generation")
        if not self.position_digest:
            raise RuntimeError("LCB->FF clock calibration lacks coordinate snapshot")
        super().validate(**kwargs)
