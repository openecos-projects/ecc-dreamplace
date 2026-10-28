"""Device coverage and independent Torch references for native timing tests."""

from contextlib import contextmanager

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def native_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture
def native_module(native_device):
    from dreamplace.ops.timing_propagation import lut_entry_2d_op

    return lut_entry_2d_op._select_module(torch.empty(0, device=native_device))


@pytest.fixture
def native_reference():
    @contextmanager
    def reference():
        from dreamplace.ops.timing_propagation import lut_entry_2d_op

        native = lut_entry_2d_op.lut_entry_2d_cpp
        names = (
            "cell_aat_forward",
            "net_aat_forward",
            "timing_lanes_forward",
            "hard_scatter_forward",
            "smooth_scatter_forward",
            "indexed_forward",
            "buffer_indexed_forward",
            "buffer_interpolate_forward",
        )
        saved = {name: getattr(native, name) for name in names if hasattr(native, name)}
        for name in saved:
            delattr(native, name)
        try:
            yield
        finally:
            for name, function in saved.items():
                setattr(native, name, function)

    return reference


@pytest.fixture
def native_candidate_forward(native_device):
    from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import (
        net_subgraph_forward_cuda_explicit_autograd,
        net_subgraph_forward_native_explicit_autograd,
    )

    return (
        net_subgraph_forward_cuda_explicit_autograd
        if native_device == "cuda"
        else net_subgraph_forward_native_explicit_autograd
    )
