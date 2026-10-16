"""Drive-strength interpolation and diagnostics for buffer timing values."""

import torch


def _validate_per_size_tensor(name, value, *, bsu_index, check_finite=True):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=bsu_index.dtype, device=bsu_index.device)
    value = value.to(device=bsu_index.device, dtype=bsu_index.dtype)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [num_candidates, legal_buffer_count]")
    if value.shape[0] != bsu_index.numel():
        raise ValueError(f"{name} candidate dimension does not match bsu_index")
    if value.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two legal buffer sizes")
    if check_finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _clamp_status_from_bounds(
    *, low, high, raw_min, raw_max, clipped_min, clipped_max, legal_buffer_count, coordinate_source
):
    if low and high:
        clamp = "mixed"
    elif low:
        clamp = "low"
    elif high:
        clamp = "high"
    else:
        clamp = "none"
    return {
        "coordinate_source": str(coordinate_source),
        "clamped": bool(low or high),
        "clamp": clamp,
        "axis_min": 0.0,
        "axis_max": float(legal_buffer_count - 1),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "clipped_min": clipped_min,
        "clipped_max": clipped_max,
    }


def _clamp_status(raw_index, clipped_index, legal_buffer_count, coordinate_source):
    return _clamp_status_from_bounds(
        low=bool(torch.any(raw_index < 0.0).detach().cpu().item()),
        high=bool(torch.any(raw_index > float(legal_buffer_count - 1)).detach().cpu().item()),
        raw_min=float(raw_index.detach().min().cpu().item()) if raw_index.numel() else None,
        raw_max=float(raw_index.detach().max().cpu().item()) if raw_index.numel() else None,
        clipped_min=float(clipped_index.detach().min().cpu().item())
        if clipped_index.numel()
        else None,
        clipped_max=float(clipped_index.detach().max().cpu().item())
        if clipped_index.numel()
        else None,
        legal_buffer_count=legal_buffer_count,
        coordinate_source=coordinate_source,
    )


def interpolate_buffer_values_by_bsu_index(
    bsu_index,
    *,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    coordinate_source="current_relaxed",
):
    if not torch.is_tensor(bsu_index):
        bsu_index = torch.as_tensor(bsu_index, dtype=torch.float32)
    if bsu_index.ndim != 1:
        raise ValueError("bsu_index must have shape [num_candidates]")

    if bsu_index.dtype in (torch.float32, torch.float64):
        from .buffer_interpolate_native import try_buffer_interpolate_native

        native_result = try_buffer_interpolate_native(
            bsu_index,
            (per_size_input_cap, per_size_delay, per_size_output_slew),
            coordinate_source,
        )
        if native_result is not None:
            return native_result

    per_size_input_cap = _validate_per_size_tensor(
        "per_size_input_cap",
        per_size_input_cap,
        bsu_index=bsu_index,
    )
    per_size_delay = _validate_per_size_tensor(
        "per_size_delay",
        per_size_delay,
        bsu_index=bsu_index,
    )
    per_size_output_slew = _validate_per_size_tensor(
        "per_size_output_slew",
        per_size_output_slew,
        bsu_index=bsu_index,
    )
    legal_buffer_count = int(per_size_delay.shape[1])
    for name, value in (
        ("per_size_input_cap", per_size_input_cap),
        ("per_size_output_slew", per_size_output_slew),
    ):
        if int(value.shape[1]) != legal_buffer_count:
            raise ValueError(f"{name} legal-size dimension does not match per_size_delay")

    raw_index = bsu_index
    clipped_index = torch.clamp(raw_index, 0.0, float(legal_buffer_count - 1))
    lo_index = torch.floor(clipped_index).to(dtype=torch.long)
    hi_index = torch.ceil(clipped_index).to(dtype=torch.long)
    alpha = clipped_index - lo_index.to(dtype=clipped_index.dtype)

    def interpolate(table):
        lo = torch.gather(table, 1, lo_index.view(-1, 1)).view(-1)
        hi = torch.gather(table, 1, hi_index.view(-1, 1)).view(-1)
        return (1.0 - alpha) * lo + alpha * hi

    return {
        "buffer_input_cap": interpolate(per_size_input_cap),
        "buffer_delay": interpolate(per_size_delay),
        "buffer_output_slew": interpolate(per_size_output_slew),
        "bsu_index": clipped_index,
        "bsu_index_lo": lo_index,
        "bsu_index_hi": hi_index,
        "bsu_index_alpha": alpha,
        "bsu_index_status": _clamp_status(
            raw_index,
            clipped_index,
            legal_buffer_count,
            coordinate_source,
        ),
    }
