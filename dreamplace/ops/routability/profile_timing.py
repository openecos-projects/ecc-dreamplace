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
