from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.buffer_timing_fast_path import buffer_timing_fast_path
from dreamplace.ops.timing_propagation.timing_propagation import (
    LUTS_INFO,
    TimingPropagation,
    set_smooth_max_enabled,
)


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_cell_mode(request):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(request.param)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _timing(device, dtype, mode):
    model = TimingPropagation.__new__(TimingPropagation)
    model.dtype = dtype
    model.use_cell_aat_static_cache = False
    model.timing_lut_2d_native_op = "on"
    model.timing_aggregation_mode = mode
    model.timing_aggregation_tau_ps = 2.0
    model._cell_modeling_enabled = lambda: False
    model.pin_net = torch.arange(5, device=device)

    def info(scale, offset):
        trans = torch.tensor([[0, 1, 2]] * 5, device=device, dtype=dtype)
        cap = trans.clone()
        trans[4, :2] = 0
        dims = torch.tensor([[2, 2], [3, 0], [0, 3], [0, 0], [2, 2]], device=device)
        values = torch.tensor(
            [[1, 2, 3, 7], [1, 2, 4, 0], [1, 3, 6, 0], [9, 0, 0, 0], [1, 2, 3, 7]],
            device=device,
            dtype=dtype,
        )
        return LUTS_INFO(values * scale + offset, trans, cap, dims)

    model.arcs_info = SimpleNamespace(
        r_delay_luts=info(1, 0),
        f_delay_luts=info(0.8, 0.2),
        r_trans_luts=info(0.02, 0.01),
        f_trans_luts=info(0.03, 0.02),
    )
    return model


def _arcs(device):
    return torch.tensor(
        [
            [0, 2, 0, 0, 1],
            [1, 2, 0, 0, -1],
            [0, 3, 0, 1, 0],
            [1, 3, 0, 2, 0],
            [0, 4, 0, 3, 1],
            [1, 4, 0, -1, -1],
        ],
        device=device,
    )


def _pins(device, dtype, steiner_nodes=0):
    values = [
        [0.3, 0.7, -100, -100, -100],
        [0.6, 0.5, -100, -100, -100],
        [0.4, 1.2, -10, -10, -10],
        [0.8, 0.3, -10, -10, -10],
        [0.1, 0.2, 0.7, 2.4, 0.3],
        [0.2, 0.4, 0.5, -0.5, 0.4],
    ]
    for caps in values[4:]:
        caps.extend([0.3] * steiner_nodes)
    return tuple(torch.tensor(v, device=device, dtype=dtype, requires_grad=True) for v in values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["hard", "smooth"])
@pytest.mark.parametrize("pair_smooth", [False, True])
@pytest.mark.parametrize("delays", [False, True])
@pytest.mark.parametrize("steiner_nodes", [0, 3])
@pytest.mark.parametrize("buffer_scope", [False, True])
def test_cell_cuda_matches_python_values_and_all_pin_gradients(
    native_device, native_reference, dtype, mode, pair_smooth, delays, steiner_nodes, buffer_scope
):
    set_smooth_max_enabled(pair_smooth)
    try:
        results = []
        for reference, device in ((True, "cpu"), (False, native_device)):
            with native_reference() if reference else nullcontext():
                model = _timing(device, dtype, mode)
                pins = _pins(device, dtype, steiner_nodes)
                with buffer_timing_fast_path() if buffer_scope else nullcontext():
                    values = model.calculate_cell_aat_level(
                        _arcs(device), *pins, use_surrogate=False, need_delay_updates=delays
                    )
                floating = [v for v in values[1:] if v is not None]
                loss = sum(torch.nan_to_num(v, neginf=0, posinf=0).sum() for v in floating)
                grads = torch.autograd.grad(loss, pins)
                results.append(
                    (
                        tuple(v.detach().cpu() if v is not None else None for v in values),
                        tuple(g.cpu() for g in grads),
                    )
                )
        tolerance = 2e-5 if dtype == torch.float32 else 1e-9
        for actual, expected in zip(results[1][0], results[0][0], strict=True):
            if actual is None:
                assert expected is None
            else:
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        for actual, expected in zip(results[1][1], results[0][1], strict=True):
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    finally:
        set_smooth_max_enabled(True)


def test_cell_cuda_uses_active_stream_and_passes_finite_differences(
    native_module,
    native_device,
):
    assert hasattr(native_module, "cell_aat_forward")
    model = _timing(native_device, torch.float64, "smooth")
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with (
        torch.cuda.stream(stream) if stream is not None else nullcontext(),
        buffer_timing_fast_path(),
    ):

        def propagate(*pins):
            return model.calculate_cell_aat_level(
                _arcs(native_device), *pins, use_surrogate=False, need_delay_updates=False
            )[1:5]

        assert torch.autograd.gradcheck(
            propagate, _pins(native_device, torch.float64), atol=1e-6, rtol=1e-3
        )
    if stream is not None:
        stream.synchronize()


@pytest.mark.parametrize(
    "arrivals",
    [
        (-float("inf"), 0.3),
        (float("inf"), 0.3),
        (-float("inf"), -float("inf")),
        (float("inf"), float("inf")),
        (float("nan"), 0.3),
        (0.3, float("nan")),
    ],
)
@pytest.mark.parametrize("pair_smooth", [False, True])
@pytest.mark.parametrize("mode", ["hard", "smooth"])
def test_cell_cuda_preserves_non_finite_pairwise_arrivals(
    native_device, native_reference, arrivals, pair_smooth, mode
):
    set_smooth_max_enabled(pair_smooth)
    try:
        results = []
        for reference, device in ((True, "cpu"), (False, native_device)):
            with native_reference() if reference else nullcontext():
                model = _timing(device, torch.float64, mode)
                pins = _pins(device, torch.float64)
                with torch.no_grad():
                    pins[0][0], pins[1][0] = arrivals
                arcs = torch.tensor([[0, 3, 0, 0, 0]], device=device)
                with buffer_timing_fast_path():
                    values = model.calculate_cell_aat_level(arcs, *pins, need_delay_updates=False)
                loss = sum(value.sum() for value in values[1:5])
                gradients = torch.autograd.grad(loss, pins)
                results.append(
                    (
                        tuple(value.detach().cpu() for value in values[1:5]),
                        tuple(gradient.cpu() for gradient in gradients),
                    )
                )
        torch.testing.assert_close(results[1], results[0], rtol=1e-9, atol=1e-9, equal_nan=True)
    finally:
        set_smooth_max_enabled(True)


def test_cell_backward_shared_pins_repeat_exactly_in_deterministic_mode(
    native_device, native_module
):
    if not torch.are_deterministic_algorithms_enabled():
        pytest.skip("requires deterministic algorithms")
    generator = torch.Generator(device=native_device).manual_seed(3000)
    groups, fanin = 512, 4
    count = groups * fanin
    arcs = torch.zeros((count, 5), dtype=torch.long, device=native_device)
    arcs[:, 0] = torch.arange(count, device=native_device) % groups
    arcs[:, 1] = torch.arange(groups, device=native_device).repeat_interleave(fanin) + groups
    arcs = arcs[torch.randperm(count, device=native_device, generator=generator)].contiguous()
    jac = torch.rand((count, 8, 6), device=native_device, generator=generator)
    gradient = torch.rand((count, 8), device=native_device, generator=generator)
    native = native_module
    reference = native.cell_aat_backward(arcs, jac, gradient, groups * 2)
    for _ in range(20):
        actual = native.cell_aat_backward(arcs, jac, gradient, groups * 2)
        assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
