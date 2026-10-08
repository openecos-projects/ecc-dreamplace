from dataclasses import dataclass, field
from typing import Any

import torch


def _as_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
        return tensor
    return torch.as_tensor(value, dtype=dtype, device=device)


def _require_finite_nonnegative(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if torch.any(value < 0):
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BufferDeviceLut:
    """Differentiable buffer device tables for local segment transfer."""

    input_cap_by_size: Any
    input_slew_axis: Any
    output_load_axis: Any
    delay_lut: Any
    output_slew_lut: Any
    source: str = "buffer_device_lut"
    arc_luts_by_size: Any = None
    phase_luts: dict[str, "BufferDeviceLut"] | None = None
    output_slew_limit_by_size: Any = None
    output_cap_limit_by_size: Any = None
    drv_limit_coverage: dict | None = None

    def tensors(self, *, dtype, device):
        input_cap_by_size = _as_tensor(
            self.input_cap_by_size,
            dtype=dtype,
            device=device,
        )
        input_slew_axis = _as_tensor(
            self.input_slew_axis,
            dtype=dtype,
            device=device,
        )
        output_load_axis = _as_tensor(
            self.output_load_axis,
            dtype=dtype,
            device=device,
        )
        delay_lut = _as_tensor(self.delay_lut, dtype=dtype, device=device)
        output_slew_lut = _as_tensor(self.output_slew_lut, dtype=dtype, device=device)

        if input_cap_by_size.ndim != 1 or input_cap_by_size.numel() == 0:
            raise ValueError("input_cap_by_size must have shape [size_count]")
        if input_slew_axis.ndim != 1 or input_slew_axis.numel() == 0:
            raise ValueError("input_slew_axis must have shape [slew_count]")
        if output_load_axis.ndim != 1 or output_load_axis.numel() == 0:
            raise ValueError("output_load_axis must have shape [load_count]")
        expected_shape = (
            int(input_cap_by_size.numel()),
            int(input_slew_axis.numel()),
            int(output_load_axis.numel()),
        )
        if tuple(delay_lut.shape) != expected_shape:
            raise ValueError("delay_lut must have shape [size_count, slew_count, load_count]")
        if tuple(output_slew_lut.shape) != expected_shape:
            raise ValueError(
                "output_slew_lut must have shape [size_count, slew_count, load_count]"
            )
        for name, tensor in (
            ("input_cap_by_size", input_cap_by_size),
            ("input_slew_axis", input_slew_axis),
            ("output_load_axis", output_load_axis),
            ("delay_lut", delay_lut),
            ("output_slew_lut", output_slew_lut),
        ):
            _require_finite_nonnegative(name, tensor)
        if input_slew_axis.numel() > 1 and torch.any(torch.diff(input_slew_axis) <= 0):
            raise ValueError("input_slew_axis must be strictly increasing")
        if output_load_axis.numel() > 1 and torch.any(torch.diff(output_load_axis) <= 0):
            raise ValueError("output_load_axis must be strictly increasing")
        result = {
            "input_cap_by_size": input_cap_by_size,
            "input_slew_axis": input_slew_axis,
            "output_load_axis": output_load_axis,
            "delay_lut": delay_lut,
            "output_slew_lut": output_slew_lut,
        }
        for name, values in (
            ("buffer_slew_limits", self.output_slew_limit_by_size),
            ("buffer_cap_limits", self.output_cap_limit_by_size),
        ):
            if values is not None:
                limits = _as_tensor(values, dtype=dtype, device=device)
                if limits.shape != input_cap_by_size.shape:
                    raise ValueError(f"{name} must match legal buffer sizes")
                _require_finite_nonnegative(name, limits)
                result[name] = limits
        return result


@dataclass(frozen=True)
class SegmentTransferInput:
    input_arrival: Any
    input_slew: Any
    downstream_load: Any
    edge_resistance: Any
    edge_capacitance: Any
    upstream_retained_capacitance: Any = 0.0
    repeater_count: int = 0
    bsu_index: Any = 0.0
    split_fractions: tuple[float, ...] | None = None
    buffer_device: BufferDeviceLut | None = None

    def validate(self):
        if isinstance(self.repeater_count, bool) or int(self.repeater_count) != self.repeater_count:
            raise ValueError("repeater_count must be an integer")
        if int(self.repeater_count) < 0:
            raise ValueError("repeater_count must be non-negative")

        dtype = torch.float64
        device = None
        for value in (
            self.input_arrival,
            self.input_slew,
            self.downstream_load,
            self.edge_resistance,
            self.edge_capacitance,
            self.upstream_retained_capacitance,
            self.bsu_index,
        ):
            if torch.is_tensor(value):
                dtype = value.dtype if value.is_floating_point() else torch.float64
                device = value.device
                break

        tensors = {
            "input_arrival": _as_tensor(self.input_arrival, dtype=dtype, device=device),
            "input_slew": _as_tensor(self.input_slew, dtype=dtype, device=device),
            "downstream_load": _as_tensor(self.downstream_load, dtype=dtype, device=device),
            "edge_resistance": _as_tensor(self.edge_resistance, dtype=dtype, device=device),
            "edge_capacitance": _as_tensor(self.edge_capacitance, dtype=dtype, device=device),
            "upstream_retained_capacitance": _as_tensor(
                self.upstream_retained_capacitance,
                dtype=dtype,
                device=device,
            ),
            "bsu_index": _as_tensor(self.bsu_index, dtype=dtype, device=device),
        }
        for name, tensor in tensors.items():
            _require_finite_nonnegative(name, tensor)
        if int(self.repeater_count) > 0 and self.buffer_device is None:
            raise ValueError("buffer_device is required when repeater_count is positive")
        if self.split_fractions is not None:
            fractions = tuple(float(value) for value in self.split_fractions)
            if len(fractions) != int(self.repeater_count) + 1:
                raise ValueError("split_fractions length must be repeater_count + 1")
            if any(value < 0.0 for value in fractions):
                raise ValueError("split_fractions must be non-negative")
            if abs(sum(fractions) - 1.0) > 1e-6:
                raise ValueError("split_fractions must sum to one")
        else:
            fractions = tuple(
                1.0 / float(int(self.repeater_count) + 1)
                for _ in range(int(self.repeater_count) + 1)
            )
        return tensors, fractions


@dataclass(frozen=True)
class SegmentTransferResult:
    upstream_visible_load: torch.Tensor
    segment_delay: torch.Tensor
    output_slew: torch.Tensor
    output_arrival: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)
