"""Skip diagnostic fills for initialized buffering outputs and restore state."""

from contextlib import contextmanager

from torch.utils import deterministic


@contextmanager
def buffer_timing_fast_path(model=None):
    timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
    static_cache_enabled = getattr(timing_op, "use_cell_aat_static_cache", None)
    empty_fill_enabled = deterministic.fill_uninitialized_memory
    try:
        # Buffer operators initialize their outputs. Keep deterministic algorithms
        # enabled while avoiding diagnostic fills before those writes.
        deterministic.fill_uninitialized_memory = False
        if static_cache_enabled is not None:
            timing_op._cell_aat_static_cache.clear()
            timing_op.use_cell_aat_static_cache = True
        yield
    finally:
        deterministic.fill_uninitialized_memory = empty_fill_enabled
        if static_cache_enabled is not None:
            timing_op.use_cell_aat_static_cache = static_cache_enabled
