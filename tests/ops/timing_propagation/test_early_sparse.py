"""Physical pin IDs and electrical slew groups are independent index domains."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.placeio_ecc.hold_schema import HoldContext, HoldTimingData
from dreamplace.ops.timing_propagation.early_propagation import EarlyPropagation
from sparse_fixtures import sparse_view


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("groups", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("native", [False, True])
def test_input_slew_uses_group_identity_and_preserves_wire_derivative(device, groups, native):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    view, _ = sparse_view()
    arrays = dict(view.arrays)
    arrays["state_slew_group"] = np.asarray(groups)[arrays["state_pin"]]
    arrays["context_state"] = np.array([2, 2, 3, 3], dtype=np.int64)
    view = replace(view, arrays=arrays, early=True)

    def ids(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    empty_lut = SimpleNamespace(use_2d_coeff_cache=True)
    check_lut = SimpleNamespace(valid_arc_mask=torch.zeros(0, dtype=torch.bool, device=device))
    timing = SimpleNamespace(
        num_pins=3,
        device=torch.device(device),
        dtype=torch.float64,
        timing_aggregation_mode="hard",
        timing_aggregation_tau_ps=2.0,
        flat_inst_arcs_by_level=ids([]).reshape(0, 8),
        flat_inst_arcs_by_level_start=ids([0, 0]),
        start_points=ids([0, 1]),
        end_points=ids([2]),
        endpoints_timing_check_arcs=ids([]).reshape(0, 8),
        arcs_info=SimpleNamespace(
            r_delay_luts=empty_lut, f_delay_luts=empty_lut,
            r_trans_luts=empty_lut, f_trans_luts=empty_lut,
            r_check_luts=check_lut, f_check_luts=check_lut,
        ),
        lut_entry_vectorized=lambda cell, clock, data, arc, lut: torch.zeros_like(clock),
    )
    context = HoldContext(
        start_pin=0, native_start_pin=0, launch_pin=-1, endpoint_pin=2, capture_pin=-1,
        launch_clock="sys", capture_clock="sys", path_tag="sys", source_type=1,
        check_type=0, data_transition=1, launch_transition=1, capture_transition=1,
        capture_pin_transition=1, arrival_ps=10, required_ps=5, data_slew_ps=0.2,
        launch_time_ps=0, launch_clock_arrival_ps=0, launch_clock_slew_ps=0,
        capture_edge_ps=0, capture_arrival_ps=0, capture_slew_ps=0, check_time_ps=0,
        uncertainty_ps=0, cppr_ps=0,
    )
    contexts = (
        context, replace(context, data_transition=-1),
        replace(context, start_pin=1, native_start_pin=1, launch_clock="jtag"),
        replace(context, start_pin=1, native_start_pin=1, launch_clock="jtag", data_transition=-1),
    )
    data = HoldTimingData(
        endpoint_valid=np.ones((1, 2), bool), endpoint_reason=np.zeros((1, 2), np.int32),
        check_valid=np.zeros((0, 2), bool), check_reason=np.zeros((0, 2), np.int32),
        startpoint_valid=np.ones((2, 2), bool), input_arrival=np.array([[10, 9], [20, 18]]),
        input_slew=np.array([[0.2, 0.25], [0.3, 0.35]]), contexts=contexts, launches=(),
        min_pin_caps=(np.zeros(0), np.zeros(0)), lib_arc_check_types=np.zeros(0, np.int64),
        output_caps=(np.zeros(1), np.zeros(1)), lib_arc_is_latch_data=np.zeros(0, bool),
        sparse_view=view,
    )
    zero = torch.zeros(3, dtype=torch.float64, device=device)
    impulse = zero.clone()
    impulse[2] = 9
    impulse.requires_grad_()
    result = EarlyPropagation(data, use_native=native).forward(
        timing, (zero, zero), (impulse, impulse), (zero, zero)
    )
    # Both domains reach the same physical endpoint. The faster input slew is
    # 0.2 ps; the wire contributes 9 ps^2, regardless of group numbering.
    expected = impulse.new_tensor(9.04).sqrt()
    torch.testing.assert_close(result.slews[0][2], expected)
    (gradient,) = torch.autograd.grad(result.slews[0][2], impulse)
    torch.testing.assert_close(gradient, torch.stack((zero[0], zero[1], 0.5 / expected)))
