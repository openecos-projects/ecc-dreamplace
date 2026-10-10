"""Full routing derivatives must match the native forward energy."""

import pytest
import torch
from dreamplace.ops.dct import dct2_fft2 as dct
from dreamplace.ops.routability.l_shape_electric_potential import (
    LShapeElectricPotential,
    SegmentElectricPotentialFunction,
)


def make_fixture(*, directional=True, memory=False, gradient_mode="translation", fast_mode=False):
    capacity = torch.linspace(0.08, 0.22, 64).reshape(8, 8)
    fixed = capacity * torch.linspace(0.15, 1.4, 64).reshape(8, 8)
    zero = torch.zeros_like(capacity)
    split = {}
    if directional:
        split = dict(
            target_density_h=capacity,
            target_density_v=capacity,
            target_demand_h=zero,
            target_demand_v=zero,
            supply_original_h=capacity,
            supply_original_v=capacity,
            fix_usage_map_h=fixed,
            fix_usage_map_v=fixed,
        )
    operator = LShapeElectricPotential(
        xl=0.0,
        yl=0.0,
        xh=6.4,
        yh=8.8,
        bin_size_x=0.8,
        bin_size_y=1.1,
        num_bins_x=8,
        num_bins_y=8,
        target_density=capacity,
        target_demand=zero,
        supply_original=capacity,
        fix_usage_map=fixed,
        fast_mode=fast_mode,
        capacity_al_enable=memory,
        log_verbose=0,
        gradient_mode=gradient_mode,
        **split,
    )
    theta = torch.tensor([0.43, 3.17, 1.43, 2.37, 1.75, 0.24, 0.28, 2.35])
    weights = torch.tensor([0.65, 0.35])
    directions = torch.tensor([True, False])
    if memory:
        operator._capacity_al_rho = 0.7
        operator(
            theta[:4],
            theta[4:6],
            theta[6:],
            directions,
            weights,
            update_capacity_al_lambda=True,
            placement_iteration_id=1,
        )
        operator.lambda_h.fill_(0.12)
        operator.lambda_v.fill_(0.08)
    return operator, theta, directions, weights


def native_energy(operator, theta, directions, weights):
    return operator(theta[:4], theta[4:6], theta[6:], directions, weights)


def poisson_matrix(operator):
    forward, inverse = dct.DCT2(), dct.IDCT2()
    diagonal = operator.inv_wu2_plus_wv2.double()
    columns = []
    for column in torch.eye(64, dtype=torch.float64):
        columns.append(inverse(forward(column.reshape(8, 8)) * diagonal).clone().flatten())
    return torch.stack(columns, dim=1)


def oracle(operator, theta, directions, weights, matrix, directional):
    starts_x = torch.arange(8, dtype=theta.dtype) * operator.bin_size_x
    starts_y = torch.arange(8, dtype=theta.dtype) * operator.bin_size_y
    maps = []
    for i in range(2):
        x, y, width, height = theta[i], theta[i + 2], theta[i + 4], theta[i + 6]
        overlap_x = (
            torch.minimum(x + width, starts_x + operator.bin_size_x) - torch.maximum(x, starts_x)
        ).relu()
        overlap_y = (
            torch.minimum(y + height, starts_y + operator.bin_size_y) - torch.maximum(y, starts_y)
        ).relu()
        ratio = weights[i].double() * width * height / (width * height).clamp(min=1e-10)
        maps.append(ratio * overlap_x[:, None] * overlap_y[None, :])
    if not directional:
        maps = [sum(maps)]
    area = operator.bin_size_x * operator.bin_size_y
    capacity, fixed = operator.supply_original.double(), operator.fix_usage_map.double()
    energy = theta.new_zeros(())
    for i, demand in enumerate(maps):
        residual = (fixed + demand / area - capacity) / capacity.clamp(min=1e-6)
        if operator.capacity_al_enable:
            multiplier = (operator.lambda_h, operator.lambda_v)[i].double()
            source = (multiplier + operator._capacity_al_rho * residual).relu()
        else:
            source = residual.relu()
        source = source.flatten()
        energy = energy + source @ matrix @ source / area
    return energy, maps


@pytest.mark.parametrize("directional,memory", [(False, False), (True, False), (True, True)])
def test_native_forward_matches_independent_poisson_oracle(directional, memory):
    operator, theta, directions, weights = make_fixture(directional=directional, memory=memory)
    actual = native_energy(operator, theta, directions, weights)
    actual_maps = [SegmentElectricPotentialFunction.last_density_map.clone()]
    if directional:
        actual_maps = [
            SegmentElectricPotentialFunction.last_density_map_h.clone(),
            SegmentElectricPotentialFunction.last_density_map_v.clone(),
        ]
    matrix = poisson_matrix(operator)
    torch.testing.assert_close(matrix, matrix.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        matrix @ torch.ones(64, dtype=torch.float64),
        torch.zeros(64, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    value = theta.double().requires_grad_()
    expected, expected_maps = oracle(operator, value, directions, weights, matrix, directional)
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.stack(actual_maps).double(), torch.stack(expected_maps), atol=1e-7, rtol=1e-6
    )
    gradient = torch.autograd.grad(expected, value)[0]
    vector = torch.tensor([-0.2, 0.3, 0.7, -0.1, 0.15, -0.05, 0.08, 0.13], dtype=torch.float64)
    step = vector * 1e-5
    plus = oracle(operator, value.detach() + step, directions, weights, matrix, directional)[0]
    minus = oracle(operator, value.detach() - step, directions, weights, matrix, directional)[0]
    torch.testing.assert_close((plus - minus) / 2e-5, gradient @ vector, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("x,width", [(-0.2, 0.4), (-0.4, 0.2)])
def test_native_density_clips_empty_domain_intersections(x, width):
    operator, _, directions, _ = make_fixture()
    position = torch.tensor([x, 0.2], requires_grad=True)
    energy = operator(position, torch.tensor([width]), torch.tensor([0.2]), directions[:1])
    expected = torch.zeros(8, 8)
    expected[0, 0] = max(0, x + width) * 0.2
    torch.testing.assert_close(SegmentElectricPotentialFunction.last_density_map, expected)
    energy.backward()
    if x + width < 0:
        torch.testing.assert_close(position.grad, torch.zeros_like(position))


@pytest.mark.parametrize(
    "directional,memory,fast",
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True)],
)
def test_full_native_gradient_matches_oracle_and_survives_later_forward(directional, memory, fast):
    operator, theta, directions, weights = make_fixture(
        directional=directional,
        memory=memory,
        gradient_mode="full",
        fast_mode=fast,
    )
    theta.requires_grad_()
    energy = native_energy(operator, theta, directions, weights)
    matrix = poisson_matrix(operator)
    value = theta.detach().double().requires_grad_()
    expected, _ = oracle(operator, value, directions, weights, matrix, directional)
    gradient = torch.autograd.grad(expected, value)[0]
    native_energy(operator, theta.detach() + 0.03, directions, weights)
    (energy * 2.5).backward()
    torch.testing.assert_close(energy.double(), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(theta.grad.double(), gradient * 2.5, atol=1e-5, rtol=1e-4)
    assert operator.energy_valid
    if not directional:
        demand = SegmentElectricPotentialFunction.last_density_map
        overflow = (operator.fix_usage_map + demand / 0.88 - operator.supply_original).relu()
        torch.testing.assert_close(SegmentElectricPotentialFunction.last_overflow_map, overflow)


@pytest.mark.parametrize(
    "direction,reverse,shared",
    [(0, False, False), (1, False, False), (0, True, False), (2, False, False), (0, False, True)],
)
def test_full_geometry_gradient_matches_cell_directional_difference(direction, reverse, shared):
    from dreamplace.ops.routability.l_shape_segment import LShapeSegmentOp

    operator, _, _, _ = make_fixture(gradient_mode="full")
    builder = LShapeSegmentOp(
        wire_width_h=0.28, wire_width_v=0.24, deterministic_backward=True, differentiate_sizes=True
    )
    cells = torch.tensor([0.43, 2.18, 1.57, 3.92], requires_grad=True)
    if direction == 2:
        cells = torch.tensor([0.43, 2.18, 1.57, 1.57], requires_grad=True)
    source = torch.tensor([1 if reverse else 0])
    sink = 1 - source
    selected = torch.tensor([direction])
    if shared:
        source, sink, selected = torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([0, 1])

    def evaluate(value):
        x, y = value[:2], value[2:]
        if shared:
            # Two branches share a movable cell; the third endpoint is fixed.
            x, y = torch.cat((x, x.new_tensor([4.5]))), torch.cat((y, y.new_tensor([5.9])))
        segments = builder(x, y, source, sink, selected)
        return operator(
            segments["segment_pos"],
            segments["segment_size_x"],
            segments["segment_size_y"],
            segments["segment_is_horizontal"],
            segments["segment_weight"],
        )

    evaluate(cells).backward()
    vector = torch.tensor([-0.2, 0.3, 0.1, 0.1])
    # The straight branch remains horizontal under this transverse translation.
    plus, minus = (
        evaluate(cells.detach() + 0.002 * vector),
        evaluate(cells.detach() - 0.002 * vector),
    )
    torch.testing.assert_close((plus - minus) / 0.004, cells.grad @ vector, atol=1e-3, rtol=1e-3)


def test_full_native_boundary_ratio_and_ties():
    from dreamplace.ops.routability import segment_energy_gradient_cpp as extension

    value = torch.tensor(
        [0.0002, 0.4, -0.3, 1.0, 0.0003, 0.2, 0.2, 0.2, 5e-6, 0.4, 0.5, 0.7, 5e-6, 0.3, 0.3, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    count = 4
    weights = torch.tensor([0.7, 0.4, 0.8, 0.3], dtype=torch.float64)
    psi = torch.linspace(-0.2, 1.1, 16, dtype=torch.float64).reshape(4, 4)
    starts = torch.arange(4, dtype=torch.float64)
    objective = value.new_zeros(())
    for i in range(count):
        x, y, width, height = value[i], value[i + count], value[i + 2 * count], value[i + 3 * count]
        # Empty-overlap derivative is zero, including at a bin-boundary tie.
        ox = (torch.minimum(x + width, starts + 1) - torch.maximum(x, starts)).relu()
        oy = (torch.minimum(y + height, starts + 1) - torch.maximum(y, starts)).relu()
        ratio = weights[i] * width * height / (width * height).clamp(min=1e-10)
        objective = objective + (psi * ox[:, None] * oy[None, :]).sum() * ratio
    expected = torch.autograd.grad(objective, value)[0]
    sx, sy = value.detach()[8:12], value.detach()[12:]
    ratio = weights * sx * sy / (sx * sy).clamp(min=1e-10)
    results = extension.backward(
        value.detach()[:8].contiguous(),
        sx.contiguous(),
        sy.contiguous(),
        ratio.contiguous(),
        weights,
        psi,
        psi,
        torch.empty(0, dtype=torch.bool),
        0.0,
        0.0,
        1.0,
        1.0,
    )
    torch.testing.assert_close(torch.cat(results), expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("count", [0, 1])
def test_full_empty_and_single_direction_inputs(count):
    operator, theta, directions, weights = make_fixture(gradient_mode="full")
    position = theta[: 2 * count].clone().requires_grad_()
    sx = theta[4 : 4 + count].clone().requires_grad_()
    sy = theta[6 : 6 + count].clone().requires_grad_()
    energy = operator(position, sx, sy, directions[:count], weights[:count])
    energy.backward()
    assert all(torch.isfinite(value.grad).all() for value in (position, sx, sy))
    if count == 0:
        assert energy.item() == 0
        assert SegmentElectricPotentialFunction.last_rho_map is None
        assert SegmentElectricPotentialFunction.last_field_map_x is None


def test_full_preserves_requested_field_consumer_and_clears_unused_fields(tmp_path):
    from types import SimpleNamespace

    from dreamplace.ops.routability.l_shape_source_plots import plot_l_shape_electric_potential_map

    operator, theta, directions, weights = make_fixture(gradient_mode="full", fast_mode=True)
    operator.require_field_maps = True
    native_energy(operator, theta, directions, weights)
    expected = SegmentElectricPotentialFunction.last_field_map_x.clone()
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    plot = tmp_path / "potential.png"
    plot_l_shape_electric_potential_map(SimpleNamespace(density_op=operator), str(plot))
    assert plot.read_bytes().startswith(b"\x89PNG")
    operator.require_field_maps = False
    native_energy(operator, theta, directions, weights)
    assert SegmentElectricPotentialFunction.last_field_map_x is None
    assert SegmentElectricPotentialFunction.last_field_map_y is None
