##
# @file   profile_timing.py
# @brief  Lightweight timing helpers for routability profiling.
#

import logging
import time
from contextlib import contextmanager

import torch


def l_shape_profile_enabled(params_or_enabled):
    if isinstance(params_or_enabled, bool):
        return params_or_enabled
    return bool(getattr(params_or_enabled, "l_shape_profile_flag", False))


def l_shape_log_verbose(params_or_level, default=0):
    if isinstance(params_or_level, bool):
        return 1 if params_or_level else 0
    if isinstance(params_or_level, (int, float)):
        return int(params_or_level)
    raw_value = getattr(params_or_level, "l_shape_log_verbose", default)
    if isinstance(raw_value, bool):
        return 1 if raw_value else 0
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in ("", "0", "false", "no", "off"):
            return 0
        if value in ("true", "yes", "on"):
            return 1
        try:
            return int(float(value))
        except ValueError:
            return int(default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(default)


def _sync_cuda(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _format_fields(fields):
    if not fields:
        return ""
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " " + " ".join(parts) if parts else ""


def profile_start(params_or_enabled, tensor=None):
    if not l_shape_profile_enabled(params_or_enabled):
        return None
    _sync_cuda(tensor)
    return time.perf_counter()


def profile_end(params_or_enabled, start_time, name, tensor=None, logger=None, **fields):
    if start_time is None or not l_shape_profile_enabled(params_or_enabled):
        return
    _sync_cuda(tensor)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    (logger or logging.getLogger(__name__)).info(
        "[L-shape profile] %s elapsed=%.3fms%s",
        name,
        elapsed_ms,
        _format_fields(fields),
    )


@contextmanager
def profile_scope(params_or_enabled, name, tensor=None, logger=None, **fields):
    start_time = profile_start(params_or_enabled, tensor=tensor)
    try:
        yield
    finally:
        profile_end(
            params_or_enabled,
            start_time,
            name,
            tensor=tensor,
            logger=logger,
            **fields,
        )
