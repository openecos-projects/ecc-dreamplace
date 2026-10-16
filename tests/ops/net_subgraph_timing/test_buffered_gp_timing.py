"""Buffered GP propagation against an explicitly expanded pi-RC circuit."""

import math
from dataclasses import replace

import pytest
import torch
from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state
from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
    SegmentCountDynamicNetProvider,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import BufferDeviceLut
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def _device():
    slew = torch.tensor([0.0, 20.0], dtype=torch.float64)
    cap = torch.tensor([0.0, 20.0], dtype=torch.float64)
    # Exact affine devices make interpolation independent of the oracle.
    return BufferDeviceLut(
        input_cap_by_size=torch.tensor([0.3, 0.6], dtype=torch.float64),
        input_slew_axis=slew,
        output_load_axis=cap,
        delay_lut=torch.stack([0.2 + 0.07 * slew[:, None] + 0.11 * cap[None, :]] * 2),
        output_slew_lut=torch.stack([0.1 + 0.23 * slew[:, None] + 0.17 * cap[None, :]] * 2),
    )


def _provider(count):
    net = {
        "net_id": 11,
        "driver_pin_id": 1,
        "coordinates": {1: (0, 0), 2: (100, 0), 3: (0, 70)},
        "rc_tree": {
            "root_node_id": 1,
            "children_by_node": {1: [2, 3], 2: [], 3: []},
            "edge_rc": {(1, 2): {"r": 2.0, "c": 0.4}, (1, 3): {"r": 1.4, "c": 0.28}},
            "node_cap": {1: 0.0, 2: 1.0, 3: 0.8},
            "sink_nodes": [2, 3],
        },
    }
    state = build_segment_count_state(
        [net],
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=3,
        initial_z=0.0,
        initial_bsu_index=0.0,
        dtype=torch.float64,
    )
    with torch.no_grad():
        for row in state.segment_rows:
            if row["child_node_id"] == 2:
                state.z_param[row["segment_id"]] = count
    provider = SegmentCountDynamicNetProvider(
        nets=[net],
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.6]], dtype=torch.float64),
        per_size_delay=torch.tensor([[0.2, 0.2]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.1]], dtype=torch.float64),
        backend="cpp_cpu_segment_transfer_explicit_autograd",
        segment_transfer_backend="segment_transfer_native",
        buffer_device_lut=_device(),
    )
    return provider


def _expanded_branch(length, sink_cap, count, *, limits=None):
    """Each wire is a pi section; each gate sees the next section's entire C."""
    r, c = length * 0.02 / (count + 1), length * 0.004 / (count + 1)
    arrival, slew = length.new_tensor(0.4), length.new_tensor(0.25)
    slew_violation, cap_violation = [], []
    for section in range(count + 1):
        downstream = sink_cap if section == count else length.new_tensor(0.3)
        delay = r * (downstream + c / 2)
        arrival = arrival + delay
        slew = torch.sqrt(slew.square() + (math.log(10) * delay).square())
        if section != count:
            gate_load = (sink_cap if section + 1 == count else 0.3) + c
            arrival = arrival + 0.2 + 0.07 * slew + 0.11 * gate_load
            slew = 0.1 + 0.23 * slew + 0.17 * gate_load
            if limits is not None:
                slew_violation.append(torch.relu(slew - limits[0]))
                cap_violation.append(torch.relu(gate_load - limits[1]))
    visible = (sink_cap if count == 0 else 0.3) + c
    return arrival, slew, visible, sink_cap + c / 2, slew_violation, cap_violation


def _reference(x, y, caps, count):
    a = _expanded_branch((x[2] - x[1]).abs() + (y[2] - y[1]).abs(), caps[2], count)
    b = _expanded_branch((x[3] - x[1]).abs() + (y[3] - y[1]).abs(), caps[3], 0)
    return {
        "sink_arrival": torch.stack([a[0], b[0]]),
        "sink_slew": torch.stack([a[1], b[1]]),
        "driver_net_cap": (caps[1] + a[2] + b[2]).reshape(1),
        "sink_load": torch.stack([a[3], b[3]]),
    }


def _evaluate(provider, x, y, caps):
    provider.bind_live_geometry(
        new_x=x,
        new_y=y,
        r_unit=0.02,
        c_unit=0.004,
        scale_factor=1.0,
        dbu=1.0,
        topology_epoch=0,
        forward_id="test",
        pin_capacitance_by_sense={"base": caps},
        require_axis_aligned=False,
    )
    try:
        return provider._run_lane(
            active_net_ids={11},
            driver_arrival_by_net={11: 0.4},
            driver_slew_by_net={11: 0.25},
        )
    finally:
        provider.release_live_geometry(forward_id="test")


def _loss(values):
    return sum(
        values[key].sum()
        for key in (
            "sink_arrival",
            "sink_slew",
            "driver_net_cap",
            "sink_load",
        )
    )


@pytest.mark.parametrize("count", [0, 1, 3])
def test_gp_buffer_chain_live_rc_cap_and_gradients(count):
    provider = _provider(count)
    provider.segment_state.z_param.requires_grad_(requires_grad=False)
    x = torch.tensor([0.0, 5.0, 106.0, 7.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 3.0, 4.0, 74.0], dtype=torch.float64, requires_grad=True)
    # Second evaluation models a sink master/VT update, without rebuilding topology.
    for sink_cap in (1.0, 2.4):
        caps = torch.tensor([0.0, 0.0, sink_cap, 0.8], dtype=torch.float64, requires_grad=True)
        actual, expected = _evaluate(provider, x, y, caps), _reference(x, y, caps, count)
        torch.testing.assert_close({key: actual[key] for key in expected}, expected)
        grad = torch.autograd.grad(_loss(actual), (x, y, caps), allow_unused=True)
        ref_grad = torch.autograd.grad(_loss(expected), (x, y, caps))
        assert all(value is not None for value in grad), "buffered cap gradient was detached"
        torch.testing.assert_close(grad, ref_grad)
        delta = torch.zeros_like(x)
        delta[2] = 1e-4
        finite_difference = (
            _loss(_evaluate(provider, x + delta, y, caps))
            - _loss(_evaluate(provider, x - delta, y, caps))
        ) / 2e-4
        torch.testing.assert_close(grad[0][2], finite_difference, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_virtual_buffer_drv_uses_same_chain_and_has_cap_coordinate_count_gradients(count):
    provider = _provider(count)
    provider.buffer_device_lut = replace(
        provider.buffer_device_lut,
        output_slew_limit_by_size=torch.tensor([0.2, 0.2], dtype=torch.float64),
        output_cap_limit_by_size=torch.tensor([0.35, 0.35], dtype=torch.float64),
    )
    x = torch.tensor([0.0, 5.0, 106.0, 7.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 3.0, 4.0, 74.0], dtype=torch.float64, requires_grad=True)
    caps = torch.tensor([0.0, 0.0, 1.0, 0.8], dtype=torch.float64, requires_grad=True)
    actual = _evaluate(provider, x, y, caps)
    reference = _expanded_branch((x[2] - x[1]) + (y[2] - y[1]), caps[2], count, limits=(0.2, 0.35))
    row = next(
        row["segment_id"]
        for row in provider.segment_state.segment_rows
        if row["child_node_id"] == 2
    )
    for key, values in zip(
        ("buffer_slew_violation", "buffer_cap_violation"), reference[4:], strict=True
    ):
        expected = torch.stack(values) if values else x.new_empty(0)
        torch.testing.assert_close(actual[key][row, :count], expected)
        assert torch.count_nonzero(actual[key][row, count:]) == 0
    loss = actual["buffer_slew_violation"].sum() + actual["buffer_cap_violation"].sum()
    if count:
        expected_loss = sum(value.sum() for value in reference[4]) + sum(reference[5])
        torch.testing.assert_close(
            torch.autograd.grad(loss, (x, y, caps), retain_graph=True),
            torch.autograd.grad(expected_loss, (x, y, caps)),
        )
    else:
        assert loss == 0
    z = provider.segment_state.z_param
    grad_z = torch.autograd.grad(loss, z)[0][row]
    # Integer count derivative uses its right adjacent discrete circuit, as B does.
    if count < provider.segment_state.max_repeater_count:
        with torch.no_grad():
            z[row] += 1e-5
        shifted = _evaluate(provider, x, y, caps)
        shifted_loss = (
            shifted["buffer_slew_violation"].sum() + shifted["buffer_cap_violation"].sum()
        )
        torch.testing.assert_close(
            grad_z, (shifted_loss - loss.detach()) / 1e-5, atol=1e-7, rtol=1e-7
        )


def test_virtual_drv_reaches_timing_aux_with_worst_phase_per_buffer():
    provider = _provider(3)
    device = replace(
        provider.buffer_device_lut,
        output_slew_limit_by_size=torch.tensor([0.2, 0.2], dtype=torch.float64),
        output_cap_limit_by_size=torch.tensor([0.35, 0.35], dtype=torch.float64),
    )
    fall = replace(device, output_slew_lut=device.output_slew_lut * 1.5)
    provider.buffer_device_lut = replace(device, phase_luts={"rise": device, "fall": fall})
    x = torch.tensor([0.0, 0.0, 100.0, 0.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 0.0, 0.0, 70.0], dtype=torch.float64, requires_grad=True)
    caps = torch.tensor([0.0, 0.0, 1.0, 0.8], dtype=torch.float64)
    zeros = torch.zeros_like(x)
    provider.begin_critical_path_net_delay_snapshot(
        base_pin_net_delay_rise=zeros, base_pin_net_delay_fall=zeros
    )
    provider.bind_live_geometry(
        new_x=x,
        new_y=y,
        r_unit=0.02,
        c_unit=0.004,
        scale_factor=1.0,
        dbu=1.0,
        topology_epoch=0,
        forward_id="aux",
        pin_capacitance_by_sense={"base": caps, "rise": caps, "fall": caps},
    )
    try:
        expected = [
            provider._run_lane(
                active_net_ids={11},
                driver_arrival_by_net={11: 0.4},
                driver_slew_by_net={11: 0.25},
                sense=sense,
            )
            for sense in ("rise", "fall")
        ]
        arrival, slew = zeros.clone(), zeros.clone()
        arrival[1], slew[1] = 0.4, 0.25
        provider.propagate_net_aat_level(
            net_ids=torch.tensor([11]),
            pin_rAAT=arrival,
            pin_fAAT=arrival,
            pin_rtran=slew,
            pin_ftran=slew,
            base_pin_net_delay_rise=zeros,
            base_pin_net_delay_fall=zeros,
            base_pin_net_impulse_rise=zeros,
            base_pin_net_impulse_fall=zeros,
            static_calculate_net_aat_level=lambda _ids, *args: args[:4],
            level_id=1,
            level_view_epoch=0,
        )
    finally:
        provider.release_live_geometry(forward_id="aux")
    timing = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(timing)
    timing.virtual_buffer_drv = provider.virtual_buffer_drv_tensors()
    timing.pin_rtran_live = timing.pin_ftran_live = zeros
    timing.pin_net_cap_rise_live = timing.pin_net_cap_fall_live = zeros
    mask = torch.zeros(4, dtype=torch.bool)
    limits = torch.ones_like(zeros)
    actual = torch.stack(
        [
            timing.get_total_slew_violation_tensor(mask, limits),
            timing.get_total_cap_violation_tensor(mask, limits),
        ]
    )
    reference = torch.stack(
        [
            torch.maximum(
                expected[0]["buffer_slew_violation"], expected[1]["buffer_slew_violation"]
            ).sum()
            / 1000,
            torch.maximum(
                expected[0]["buffer_cap_violation"], expected[1]["buffer_cap_violation"]
            ).sum(),
        ]
    )
    torch.testing.assert_close(actual, reference)
    assert bool((actual > 0).all())
    assert torch.autograd.grad(actual.sum(), x)[0][2] != 0
