"""Native H/V fields must remain independent through segment backward."""

import pytest
import torch
from dreamplace.ops.dct import dct2_fft2 as dct
from dreamplace.ops.routability.l_shape_electric_potential import (
    LShapeElectricPotential,
    SegmentElectricPotentialFunction,
)


def make_operator(fast_mode):
    capacity = torch.full((4, 4), 0.01)
    zero = torch.zeros_like(capacity)
    return LShapeElectricPotential(
        xl=0.0,
        yl=0.0,
        xh=4.0,
        yh=4.0,
        bin_size_x=1.0,
        bin_size_y=1.0,
        num_bins_x=4,
        num_bins_y=4,
        target_density=capacity,
        target_demand=zero,
        supply_original=capacity,
        fix_usage_map=zero,
        target_density_h=capacity,
        target_density_v=capacity,
        target_demand_h=zero,
        target_demand_v=zero,
        supply_original_h=capacity,
        supply_original_v=capacity,
        fix_usage_map_h=zero,
        fix_usage_map_v=zero,
        fast_mode=fast_mode,
        capacity_al_enable=False,
        log_verbose=0,
    )


def independent_fields(operator, rho):
    spectrum = dct.DCT2()(rho)
    return (
        dct.IDXST_IDCT()(spectrum * operator.wu_by_wu2_plus_wv2_half),
        dct.IDCT_IDXST()(spectrum * operator.wv_by_wu2_plus_wv2_half),
    )


@pytest.mark.parametrize("fast_mode", [False, True])
def test_directional_fields_and_backward_survive_later_solves(fast_mode):
    operator = make_operator(fast_mode)
    position = torch.tensor([0.2, 0.2, 2.2, 0.2], requires_grad=True)
    widths, heights = torch.tensor([1.0, 0.2]), torch.tensor([0.2, 1.0])
    directions = torch.tensor([True, False])
    energy = operator(position, widths, heights, directions)
    context = energy.grad_fn
    rho_h = SegmentElectricPotentialFunction.last_rho_map_h.clone()
    rho_v = SegmentElectricPotentialFunction.last_rho_map_v.clone()
    assert not torch.allclose(rho_h, rho_v)
    expected_h = independent_fields(operator, rho_h)
    expected_v = independent_fields(operator, rho_v)
    expected = torch.stack((*expected_h, *expected_v))

    def saved_fields():
        return torch.stack(
            (
                context.h_field_map_x,
                context.h_field_map_y,
                context.v_field_map_x,
                context.v_field_map_y,
            )
        )

    torch.testing.assert_close(saved_fields(), expected)
    operator(position.detach() + 0.3, widths, heights, directions)
    torch.testing.assert_close(saved_fields(), expected)
    energy.backward()

    reference_position = position.detach().clone().requires_grad_(requires_grad=True)
    reference_energy = operator(reference_position, widths, heights, directions)
    reference_context = reference_energy.grad_fn
    reference_context.h_field_map_x, reference_context.h_field_map_y = expected_h
    reference_context.v_field_map_x, reference_context.v_field_map_y = expected_v
    reference_energy.backward()
    torch.testing.assert_close(position.grad, reference_position.grad)
