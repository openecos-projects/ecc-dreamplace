"""Fused net propagation must retain CSR, slew-clamp and pin-gradient semantics."""

from contextlib import nullcontext

import pytest
import torch
from dreamplace.ops.timing_propagation.net_aat_native import _NetAat
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_net_mode(request):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(request.param)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _model(device, mode):
    model = TimingPropagation.__new__(TimingPropagation)
    model.timing_aggregation_mode = mode
    model.timing_aggregation_tau_ps = 2.0
    model.net_flat_arcs = torch.tensor(
        [[0, 3], [0, 4], [1, 4], [2, 5], [2, 6]],
        device=device,
    )
    model.net_flat_arcs_start = torch.tensor([0, 3, 5], device=device)
    return model


def _pins(device, dtype, extra_nodes=0):
    values = [
        [0.2, 0.3, 0.4, -100, -100, -100, -100, 4.2],
        [0.5, 0.2, 0.1, -100, -100, -100, -100, 3.2],
        [0.2, 0.8, 0.1, -10, -10, -10, -10, 0.4],
        [0.5, 0.3, 0.2, -10, -10, -10, -10, 0.7],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0, 0.02, 0.03, 0.04, 0.05, -0.1, 0, 0.02],
        [0, 0.03, 0.04, 0.05, 0.06, -0.1, 0, 0.03],
    ]
    for wires in values[4:]:
        wires.extend([0.1] * extra_nodes)
    return tuple(torch.tensor(v, device=device, dtype=dtype, requires_grad=True) for v in values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["hard", "smooth"])
@pytest.mark.parametrize("extra_nodes", [0, 3])
@pytest.mark.parametrize("nets", [[0, 1], [1, 0], [], [0]])
def test_net_cuda_matches_cpu_forward_and_all_eight_input_gradients(
    native_device, native_module, native_reference, dtype, mode, extra_nodes, nets
):
    assert hasattr(native_module, "net_aat_forward")
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            model = _model(device, mode)
            pins = _pins(device, dtype, extra_nodes)
            output = model.calculate_net_aat_level(
                torch.tensor(nets, device=device, dtype=torch.long), *pins
            )
            loss = sum((v * torch.arange(1, 9, dtype=dtype, device=device)).sum() for v in output)
            gradients = torch.autograd.grad(loss, pins, allow_unused=True)
            results.append(
                (
                    tuple(v.detach().cpu() for v in output),
                    tuple(
                        torch.zeros_like(p).cpu() if g is None else g.cpu()
                        for p, g in zip(pins, gradients, strict=True)
                    ),
                )
            )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(results[1], results[0], rtol=tolerance, atol=tolerance)


def test_net_cuda_obeys_nondefault_stream_and_finite_differences(
    native_device,
):
    stream = torch.cuda.Stream() if native_device == "cuda" else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        model = _model(native_device, "smooth")
        pins = _pins(native_device, torch.float64)
        nets = torch.tensor([0, 1], device=native_device)
        assert torch.autograd.gradcheck(
            lambda *values: model.calculate_net_aat_level(nets, *values),
            pins,
            atol=1e-6,
            rtol=1e-3,
        )
    if stream is not None:
        stream.synchronize()


def test_net_shared_pin_gradients_repeat_exactly(native_device):
    generator = torch.Generator(device=native_device).manual_seed(3000)
    arcs = torch.randint(0, 64, (1024, 2), device=native_device, generator=generator)
    pins = tuple(
        torch.rand(64, device=native_device, generator=generator, requires_grad=True)
        for _ in range(8)
    )
    weights = torch.rand((1024, 4), device=native_device, generator=generator)

    def evaluate():
        updates = torch.stack(_NetAat.apply(arcs, *pins, 1e-8), dim=1)
        return torch.autograd.grad((updates * weights).sum(), pins)

    reference = evaluate()
    for _ in range(20):
        assert all(torch.equal(a, b) for a, b in zip(evaluate(), reference, strict=True))


@pytest.mark.parametrize("mutation", ["arcs", "rise_slew", "fall_slew"])
def test_net_backward_rejects_inplace_topology_or_slew_changes(native_device, mutation):
    arcs = _model(native_device, "hard").net_flat_arcs.int()
    pins = _pins(native_device, torch.float64)
    output = _NetAat.apply(arcs, *pins, 1e-8)
    with torch.no_grad():
        if mutation == "arcs":
            arcs[0, 0] = 1
        else:
            pins[2 if mutation == "rise_slew" else 3][0] += 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(sum(v.sum() for v in output), pins)


@pytest.mark.parametrize("slew", [float("nan"), float("inf"), -float("inf"), 0.0])
@pytest.mark.parametrize("selected", [[0, 1], [2], [3], [0, 1, 2, 3]])
def test_net_cuda_nonfinite_and_zero_slew_preserve_reference_derivatives(
    native_device, native_reference, slew, selected
):
    results = []
    for reference, device in ((True, "cpu"), (False, native_device)):
        with native_reference() if reference else nullcontext():
            model = _model(device, "hard")
            pins = _pins(device, torch.float64)
            with torch.no_grad():
                pins[2][0] = slew
            output = model.calculate_net_aat_level(torch.tensor([0], device=device), *pins)
            gradients = torch.autograd.grad(
                sum(output[i].sum() for i in selected), pins, allow_unused=True
            )
            results.append(
                (
                    tuple(v.detach().cpu() for v in output),
                    tuple(
                        torch.zeros_like(p).cpu() if g is None else g.cpu()
                        for p, g in zip(pins, gradients, strict=True)
                    ),
                )
            )
    torch.testing.assert_close(results[1], results[0], equal_nan=True)
