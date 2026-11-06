import pytest
import torch

from sparse_fixtures import (
    inputs,
    tables,
    schedule,
)


@pytest.mark.parametrize("smooth", [False, True])
def test_domains_share_rc_but_keep_arrivals_and_count_endpoint_once(
    native_device, smooth
):
    model = schedule(native_device)
    values = inputs(native_device)
    lanes = model.propagate((), values[:2], values[2:4], values[4:], smooth=smooth)
    torch.testing.assert_close(
        lanes[0][2:], torch.tensor([15, 25], device=native_device, dtype=torch.float64)
    )
    result = model.reduce_endpoints(
        lanes, torch.zeros(4, device=native_device, dtype=torch.float64)
    )
    torch.testing.assert_close(result.tns, result.wns)
    torch.testing.assert_close(
        result.tns, torch.tensor(-3.0, device=native_device, dtype=torch.float64)
    )
    assert result.endpoint_pins.tolist() == [2]
    gradients = torch.autograd.grad(result.tns, values, allow_unused=True)
    torch.testing.assert_close(
        gradients[2],
        torch.tensor([0, 0, -1, 0], device=native_device, dtype=torch.float64),
    )


def test_endpoint_lse_is_used_by_sparse_tns_reduction(native_device):
    hard = schedule(native_device)
    smooth = schedule(native_device)
    smooth.aggregation_mode = "smooth"
    smooth.aggregation_tau_ps = 2.0
    values = inputs(native_device)

    hard_lanes = hard.propagate((), values[:2], values[2:4], values[4:])
    smooth_lanes = smooth.propagate((), values[:2], values[2:4], values[4:])
    hard_result = hard.reduce_endpoints(
        hard_lanes,
        torch.zeros(4, device=native_device, dtype=torch.float64),
    )
    smooth_result = smooth.reduce_endpoints(
        smooth_lanes,
        torch.zeros(4, device=native_device, dtype=torch.float64),
    )

    torch.testing.assert_close(
        hard_result.endpoint_slack,
        torch.tensor([-3.0], device=native_device, dtype=torch.float64),
    )
    expected = -2.0 * torch.logsumexp(
        -hard_result.check_slack / 2.0,
        dim=0,
    )
    torch.testing.assert_close(
        smooth_result.endpoint_slack,
        expected.reshape_as(smooth_result.endpoint_slack),
    )
    assert float(smooth_result.tns.detach()) < float(hard_result.tns.detach())


def test_false_path_excludes_timing_gradient_but_preserves_electrical_slew(
    native_device,
):
    model = schedule(native_device, false_path=True)
    model.reference.requires_grad_()
    values = inputs(native_device)
    lanes = model.propagate((), values[:2], values[2:4], values[4:])
    result = model.reduce_endpoints(
        lanes, torch.zeros(4, device=native_device, dtype=torch.float64)
    )
    reference_grad = torch.autograd.grad(
        result.check_slack.sum(), model.reference, retain_graph=True
    )[0]
    torch.testing.assert_close(
        reference_grad[1], torch.zeros(4, device=native_device, dtype=torch.float64)
    )
    assert lanes[2][3].item() > 0
    electrical_grad = torch.autograd.grad(lanes[2][3], model.reference)[0]
    assert electrical_grad[1, 2].item() > 0


@pytest.mark.parametrize("cell", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("adjustment", [(0, 0, 1, 1), (0.1, -0.2, 0.5, 0.75)])
def test_cell_net_and_dynamic_check_gradients_match_finite_difference(
    native_device, cell, smooth, adjustment
):
    model = schedule(native_device, cell=cell, adjustment=adjustment)
    lut = tables(native_device) if cell else ()

    def objective(*values):
        lanes = model.propagate(lut, values[:2], values[2:4], values[4:], smooth=smooth)
        check_values = torch.stack(
            tuple(
                lanes[edge + 2][state]
                for state in (len(model.state_pin) - 2, len(model.state_pin) - 1)
                for edge in range(2)
            )
        )
        result = model.reduce_endpoints(lanes, check_values)
        return result.tns

    assert torch.autograd.gradcheck(
        objective, inputs(native_device), eps=1e-5, atol=2e-5, rtol=2e-4
    )


def test_receiver_slew_conversion_does_not_change_driver_or_arrival(native_device):
    model = schedule(native_device, adjustment=(0, 0, 0.5, 0.5))
    values = inputs(native_device)
    zeros = tuple(torch.zeros_like(value) for value in values[2:])
    lanes = model.propagate((), values[:2], zeros[:2], zeros[2:])
    torch.testing.assert_close(
        lanes[0], model.reference[:, 0].new_tensor([10, 20, 10, 20])
    )
    torch.testing.assert_close(
        lanes[2], model.reference[:, 0].new_tensor([0.2, 0.3, 0.15, 0.15])
    )


def test_function_insensitive_arc_contributes_slew_without_an_arrival_path(
    native_device,
):
    model = schedule(native_device, cell=True, slew_only=True)
    model.reference.requires_grad_()
    values = inputs(native_device)
    lanes = model.propagate(tables(native_device), values[:2], values[2:4], values[4:])
    torch.testing.assert_close(lanes[0][4:], torch.full_like(lanes[0][4:], -2e8))
    assert torch.all(lanes[2][4:] > 0)
    gradient = torch.autograd.grad(lanes[2][4:].sum(), model.reference)[0]
    torch.testing.assert_close(gradient[:, :2], torch.zeros_like(gradient[:, :2]))
    assert torch.any(gradient[:, 2:] != 0)


@pytest.mark.parametrize("smooth", [False, True])
def test_sparse_cpu_cuda_state_and_gradient_parity(smooth):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    outputs = []
    for device in ("cpu", "cuda"):
        model = schedule(device, cell=True, adjustment=(0.1, -0.2, 0.5, 0.75))
        values = inputs(device)
        lanes = model.propagate(
            tables(device), values[:2], values[2:4], values[4:], smooth=smooth
        )
        checks = torch.stack(
            tuple(lanes[edge + 2][state] for state in (4, 5) for edge in (0, 1))
        )
        result = model.reduce_endpoints(lanes, checks)
        gradients = torch.autograd.grad(result.tns, values)
        outputs.append(
            tuple(
                value.detach().cpu()
                for value in (
                    *lanes,
                    result.required,
                    result.endpoint_slack,
                    *gradients,
                )
            )
        )
    for cpu, cuda in zip(*outputs):
        torch.testing.assert_close(cuda, cpu, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("smooth", [False, True])
def test_compact_inference_matches_differentiable_forward(native_device, smooth):
    model = schedule(native_device, cell=True, adjustment=(0.1, -0.2, 0.5, 0.75))
    values = inputs(native_device)
    lut = tables(native_device)
    expected = model.propagate(lut, values[:2], values[2:4], values[4:], smooth=smooth)
    with torch.no_grad():
        actual = model.propagate(
            lut, values[:2], values[2:4], values[4:], smooth=smooth
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("smooth", [False, True])
def test_compact_invalid_transition_keeps_shared_electrical_slew(native_device, smooth):
    model = schedule(native_device)
    model.valid[3, 0] = False
    model.all_transitions_valid = False
    values = inputs(native_device)
    lanes = model.propagate((), values[:2], values[2:4], values[4:], smooth=smooth)
    torch.testing.assert_close(lanes[0][3], lanes[0].new_tensor(-2e8))
    torch.testing.assert_close(
        lanes[2][2:], torch.sqrt(lanes[2].new_tensor([0.13, 0.13]))
    )
    gradient = torch.autograd.grad(lanes[0][3], values[2], retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    slew_gradient = torch.autograd.grad(lanes[2][3], values[4])[0]
    expected = torch.zeros_like(slew_gradient)
    expected[2] = 0.5 / (0.13 ** 0.5)
    torch.testing.assert_close(slew_gradient, expected)
