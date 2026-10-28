"""GP coordinate/cap gradients after typed buffering, against finite differences."""

import pytest
import torch

from dreamplace.ops.buffer_insertion.drive_type_table import DriveTypeEntry, DriveTypeTable
from dreamplace.ops.buffer_insertion.optimization_state import build_typed_discrete_candidate_state
from dreamplace.ops.net_subgraph_timing.candidate_live_geometry import CandidateLiveGeometry
from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import net_subgraph_forward_relaxed
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    BufferDeviceLut,
    lookup_buffer_device_per_size,
)


def _objective(variables, active_drive, backend):
    device, dtype = variables.device, variables.dtype
    candidates = [
        {
            "candidate_id": 0,
            "net_id": 0,
            "candidate_node_id": 1,
            "buffer_main_type_index": 4,
            "parent_node_id": 0,
            "child_node_ids": [1],
            "segment_split_ratio": 0.25,
            "x_dbu": 5,
            "y_dbu": 2,
        },
        {
            "candidate_id": 1,
            "net_id": 0,
            "candidate_node_id": 4,
            "buffer_main_type_index": 4,
            "parent_node_id": 1,
            "child_node_ids": [3],
            "segment_split_ratio": 0.7,
            "x_dbu": 35,
            "y_dbu": -2,
        },
    ]
    table = DriveTypeTable(
        tuple(DriveTypeEntry(i, f"BUF{i}", i + 1, i, i) for i in range(4)),
        source="coordinate_gradient_fixture",
    )
    state = build_typed_discrete_candidate_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        device=device,
        dtype=dtype,
    )
    if active_drive is not None:
        with torch.no_grad():
            state.activation_param[0, active_drive] = 1
            state.activation_param[1, (active_drive + 1) % 4] = 1
    payload = {
        "metadata": {"compact_node_to_original": [0, -1, 1, 2, -2, 3]},
        "pin_fa": torch.tensor([-1, 0, 1, 2, 2, 4]),
        "sink_node_id": torch.tensor([3, 5]),
        "sink_pin_id": torch.tensor([2, 3]),
        "node_capacitance": torch.zeros(6),
        "candidate_node_id": torch.tensor([1, 4]),
    }
    geometry = CandidateLiveGeometry(payload, state)
    geometry.bind(
        new_x=variables[:4],
        new_y=variables[4:8],
        pin_capacitance_by_sense={"base": variables[8:]},
        r_unit=0.4,
        c_unit=0.2,
        scale_factor=0.5,
        dbu=10,
    )
    overrides = geometry.forward_overrides(
        active_view=None, device=device, dtype=dtype, sense="base"
    )
    slew_axis = variables.new_tensor([0, 20])
    cap_axis = variables.new_tensor([0, 30])
    lut = BufferDeviceLut(
        input_cap_by_size=variables.new_tensor([0.2, 0.3, 0.4, 0.5]),
        input_slew_axis=slew_axis,
        output_load_axis=cap_axis,
        delay_lut=torch.stack(
            [0.1 * (i + 1) + 0.03 * slew_axis[:, None] + 0.04 * cap_axis[None, :] for i in range(4)]
        ),
        output_slew_lut=torch.stack(
            [0.2 * (i + 1) + 0.05 * slew_axis[:, None] + 0.06 * cap_axis[None, :] for i in range(4)]
        ),
    )

    def lookup(**kwargs):
        return lookup_buffer_device_per_size(
            lut,
            input_slew=kwargs["candidate_input_slew"],
            output_load=kwargs["candidate_output_cap"],
        )

    topology = {
        "net_flat_topo_sort": [0, 1, 2, 3, 4, 5],
        "net_flat_topo_sort_start": [0, 6],
        "pin_fa": [-1, 0, 1, 2, 2, 4],
        "flat_pin_to_start": [0, 1, 2, 4, 4, 5, 5],
        "flat_pin_to": [1, 2, 3, 4, 5],
        "sink_node_id": [3, 5],
        "sink_net_index": [0, 0],
    }
    result = net_subgraph_forward_relaxed(
        buffer_state=state,
        forward_backend=backend,
        buffer_coordinate_callback=lookup,
        **{
            key: torch.tensor(value, device=device, dtype=torch.long)
            for key, value in topology.items()
        },
        **overrides,
        driver_arrival=variables.new_tensor([0.3]),
        driver_slew=variables.new_tensor([0.4]),
        per_size_input_cap=lut.input_cap_by_size.expand(2, -1),
        per_size_delay=lut.delay_lut.mean(dim=(1, 2)).expand(2, -1),
        per_size_output_slew=lut.output_slew_lut.mean(dim=(1, 2)).expand(2, -1),
    )
    arrival = result["sink_arrival"]
    return (
        2 * torch.logsumexp(arrival / 2, 0)
        + 0.2 * arrival.sum()
        + 0.05 * result["sink_slew"].square().sum()
        + 0.03 * result["lin"][0].square()
    )


@pytest.mark.parametrize("active_drive", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("backend", ["native_explicit_autograd", "cuda_explicit_autograd"])
def test_buffered_coordinates_and_pin_caps_match_reference_and_finite_difference(
    active_drive, backend
):
    if backend.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    initial = torch.tensor([0, 10, 14, 21, 0, 4, 11, -3, 0, 0, 1.2, 0.8], dtype=torch.float64)
    oracle = initial.clone().requires_grad_()
    expected_loss = _objective(oracle, active_drive, "python")
    (expected,) = torch.autograd.grad(expected_loss, oracle)
    inputs = initial.to("cuda" if backend.startswith("cuda") else "cpu").requires_grad_()
    loss = _objective(inputs, active_drive, backend)
    (actual,) = torch.autograd.grad(loss, inputs)
    torch.testing.assert_close(loss.cpu(), expected_loss, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-8, atol=1e-8)
    finite = torch.zeros_like(initial)
    for index in range(initial.numel()):
        plus, minus = initial.clone(), initial.clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        finite[index] = (
            _objective(plus, active_drive, "python") - _objective(minus, active_drive, "python")
        ) / 2e-5
    torch.testing.assert_close(actual.cpu(), finite, rtol=1e-6, atol=1e-7)
