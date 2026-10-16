import json

import pytest
import torch
import dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider as provider_module

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
    SegmentCountDynamicNetProvider,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import BufferDeviceLut
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


def _line_net():
    return {
        "net_id": 5,
        "net_name": "n5",
        "driver_pin_id": 1,
        "coordinates": {1: (0, 0), 2: (100, 0)},
        "rc_tree": {
            "root_node_id": 1,
            "children_by_node": {1: [2], 2: []},
            "edge_rc": {(1, 2): {"r": 10.0, "c": 0.0}},
            "node_cap": {1: 0.0, 2: 1.0},
            "sink_nodes": [2],
        },
    }


def _line_net_with_id(net_id, driver_pin_id, sink_pin_id):
    net = _line_net()
    net["net_id"] = int(net_id)
    net["driver_pin_id"] = int(driver_pin_id)
    net["coordinates"] = {
        int(driver_pin_id): (0, 0),
        int(sink_pin_id): (100, 0),
    }
    net["rc_tree"] = {
        "root_node_id": int(driver_pin_id),
        "children_by_node": {int(driver_pin_id): [int(sink_pin_id)], int(sink_pin_id): []},
        "edge_rc": {
            (int(driver_pin_id), int(sink_pin_id)): {"r": 10.0, "c": 0.0}
        },
        "node_cap": {int(driver_pin_id): 0.0, int(sink_pin_id): 1.0},
        "sink_nodes": [int(sink_pin_id)],
    }
    return net


def _branched_net():
    return {
        "net_id": 11,
        "net_name": "n11",
        "driver_pin_id": 1,
        "coordinates": {1: (0, 0), 2: (100, 0), 3: (0, 100)},
        "rc_tree": {
            "root_node_id": 1,
            "children_by_node": {1: [2, 3], 2: [], 3: []},
            "edge_rc": {
                (1, 2): {"r": 10.0, "c": 0.0},
                (1, 3): {"r": 10.0, "c": 0.0},
            },
            "node_cap": {1: 0.0, 2: 2.0, 3: 1.0},
            "sink_nodes": [2, 3],
        },
    }


def _provider():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
    )
    return provider, state


def test_segment_count_provider_exports_effective_delays_for_path_snapshot():
    provider, _state = _provider()
    base_rise = torch.tensor([0.0, 0.0, 101.0], dtype=torch.float32)
    base_fall = torch.tensor([0.0, 0.0, 202.0], dtype=torch.float32)
    provider.begin_critical_path_net_delay_snapshot(
        base_pin_net_delay_rise=base_rise,
        base_pin_net_delay_fall=base_fall,
    )
    pin_rise_aat = torch.tensor([0.0, 10.0, 0.0], dtype=torch.float32)
    pin_fall_aat = torch.tensor([0.0, 20.0, 0.0], dtype=torch.float32)
    pin_rise_slew = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    pin_fall_slew = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)

    def static_forward(
        _net_ids,
        rise_aat,
        fall_aat,
        rise_slew,
        fall_slew,
        *_args,
    ):
        return rise_aat, fall_aat, rise_slew, fall_slew

    rise_aat, fall_aat, _rise_slew, _fall_slew = (
        provider.propagate_net_aat_level(
            net_ids=torch.tensor([5], dtype=torch.long),
            pin_rAAT=pin_rise_aat,
            pin_fAAT=pin_fall_aat,
            pin_rtran=pin_rise_slew,
            pin_ftran=pin_fall_slew,
            base_pin_net_delay_rise=base_rise,
            base_pin_net_delay_fall=base_fall,
            base_pin_net_impulse_rise=torch.zeros_like(base_rise),
            base_pin_net_impulse_fall=torch.zeros_like(base_fall),
            static_calculate_net_aat_level=static_forward,
            level_id=1,
            level_view_epoch=0,
        )
    )
    snapshot_rise, snapshot_fall = provider.consume_critical_path_net_delays(
        base_pin_net_delay_rise=base_rise,
        base_pin_net_delay_fall=base_fall,
    )

    torch.testing.assert_close(snapshot_rise[2], rise_aat[2] - pin_rise_aat[1])
    torch.testing.assert_close(snapshot_fall[2], fall_aat[2] - pin_fall_aat[1])
    assert snapshot_rise[2] != base_rise[2]
    assert snapshot_fall[2] != base_fall[2]
    assert provider.metadata["critical_path_delay_overlay_sink_count"] == 2
    assert provider._critical_path_pin_net_delay_rise is None
    assert provider._critical_path_pin_net_delay_fall is None


def test_segment_count_provider_expands_shared_size_row_for_active_segments():
    provider = object.__new__(SegmentCountDynamicNetProvider)
    active_state = type(
        "ActiveSegmentState",
        (),
        {"z_param": torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))},
    )()
    table = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32)

    expanded = provider._active_size_table(table, [7, 8, 9], active_state)

    assert expanded.shape == (3, 3)
    assert expanded.stride(0) == 0
    assert torch.equal(expanded[0], table[0])
    assert torch.equal(expanded[1], table[0])
    assert torch.equal(expanded[2], table[0])


def test_segment_count_provider_live_geometry_reaches_coordinates_and_releases():
    provider, _state = _provider()
    provider.metadata["topology_epoch"] = 4
    new_x = torch.tensor(
        [0.0, 0.0, 100.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    new_y = torch.zeros(3, dtype=torch.float32, requires_grad=True)

    summary = provider.bind_live_geometry(
        new_x=new_x,
        new_y=new_y,
        r_unit=0.1,
        c_unit=0.005,
        scale_factor=1.0,
        dbu=1.0,
        topology_epoch=4,
        forward_id="forward-7",
    )
    result = provider._run_lane(
        active_net_ids={5},
        driver_arrival_by_net={5: torch.tensor(0.0)},
        driver_slew_by_net={5: torch.tensor(0.2)},
    )
    objective = (
        result["sink_arrival"].sum()
        + result["sink_slew"].sum()
        + result["sink_load"].sum()
    )
    grad_x, grad_y = torch.autograd.grad(objective, (new_x, new_y))

    assert summary["topology_epoch"] == 4
    assert provider.metadata["live_rc_local_build_count"] == 1
    assert bool(torch.isfinite(grad_x).all())
    assert bool(torch.isfinite(grad_y).all())
    assert bool((grad_x != 0.0).any())
    assert result["metadata"]["live_edge_override"]
    assert result["metadata"]["canonical_equal_spacing"]

    provider.release_live_geometry(forward_id="forward-7")
    assert provider._live_geometry_context is None
    assert not provider.metadata["live_geometry_bound"]


def test_segment_count_native_provider_live_geometry_reaches_coordinates():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.0,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend="cpp_cpu_segment_transfer_explicit_autograd",
        segment_transfer_backend="segment_transfer_native",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float32),
    )
    provider.metadata["topology_epoch"] = 1
    new_x = torch.tensor([0.0, 0.0, 100.0], requires_grad=True)
    new_y = torch.zeros_like(new_x, requires_grad=True)

    provider.bind_live_geometry(
        new_x=new_x,
        new_y=new_y,
        r_unit=0.1,
        c_unit=0.005,
        scale_factor=1.0,
        dbu=1.0,
        topology_epoch=1,
        forward_id="native-live",
    )
    result = provider._run_lane(
        active_net_ids={5},
        driver_arrival_by_net={5: torch.tensor(0.0)},
        driver_slew_by_net={5: torch.tensor(0.2)},
    )
    objective = (
        result["sink_arrival"].sum()
        + result["sink_slew"].sum()
        + result["driver_net_cap"].sum()
    )
    grad_x, grad_y = torch.autograd.grad(objective, (new_x, new_y))

    assert result["metadata"]["backend"] == "cpp_cpu_segment_transfer_explicit_autograd"
    assert result["metadata"]["canonical_equal_spacing"]
    assert bool((grad_x != 0.0).any())
    assert bool(torch.isfinite(grad_y).all())
    provider.release_live_geometry(forward_id="native-live")


def test_segment_count_provider_rejects_stale_live_geometry_identity():
    provider, _state = _provider()
    provider.metadata["topology_epoch"] = 2
    coordinates = torch.tensor([0.0, 0.0, 100.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="topology epoch"):
        provider.bind_live_geometry(
            new_x=coordinates,
            new_y=torch.zeros_like(coordinates),
            r_unit=0.1,
            c_unit=0.005,
            scale_factor=1.0,
            dbu=1.0,
            topology_epoch=3,
            forward_id="stale",
        )

    provider.bind_live_geometry(
        new_x=coordinates,
        new_y=torch.zeros_like(coordinates),
        r_unit=0.1,
        c_unit=0.005,
        scale_factor=1.0,
        dbu=1.0,
        topology_epoch=2,
        forward_id="current",
    )
    with pytest.raises(ValueError, match="forward identity"):
        provider.release_live_geometry(forward_id="stale")
    provider.release_live_geometry(forward_id="current")


def test_segment_count_provider_reuses_warm_level_view_without_reextracting_net_ids(
    monkeypatch,
):
    provider, _state = _provider()
    calls = []
    original = provider_module._as_net_id_set

    def spy(net_ids):
        calls.append(net_ids)
        return original(net_ids)

    monkeypatch.setattr(provider_module, "_as_net_id_set", spy)

    assert provider.has_dynamic_nets(
        torch.tensor([5], dtype=torch.long),
        level_id=3,
        level_view_epoch=0,
    )
    assert provider.has_dynamic_nets(
        torch.tensor([5], dtype=torch.long),
        level_id=3,
        level_view_epoch=0,
    )

    assert len(calls) == 1
    assert provider.metadata["level_view_cache_hit_count"] == 1
    assert provider.metadata["level_view_build_count"] == 1


def test_segment_count_provider_rebuilds_level_view_after_epoch_change(monkeypatch):
    provider, _state = _provider()
    calls = []
    original = provider_module._as_net_id_set

    def spy(net_ids):
        calls.append(net_ids)
        return original(net_ids)

    monkeypatch.setattr(provider_module, "_as_net_id_set", spy)

    assert provider.has_dynamic_nets(
        torch.tensor([5], dtype=torch.long),
        level_id=3,
        level_view_epoch=0,
    )
    assert provider.has_dynamic_nets(
        torch.tensor([5], dtype=torch.long),
        level_id=3,
        level_view_epoch=1,
    )

    assert len(calls) == 2
    assert provider.metadata["level_view_build_count"] == 2


def test_segment_count_provider_uses_tensorized_driver_gather_for_level_view(
    monkeypatch,
):
    provider, state = _provider()
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def fail_python_driver_dict(*args, **kwargs):
        raise AssertionError("warm level dispatch must not build a Python driver dictionary")

    monkeypatch.setattr(provider, "_driver_values", fail_python_driver_dict)

    def static_calculate_net_aat_level(curnets, *args):
        return args[:4]

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
        level_id=3,
        level_view_epoch=0,
    )

    loss = pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2]
    loss.backward()
    assert provider.metadata["tensorized_driver_gather_count"] == 4
    assert state.z_param.grad is not None


def test_packed_cuda_provider_uses_native_level_selector_without_python_driver_lookup(
    monkeypatch,
):
    from dreamplace.ops.net_subgraph_timing.segment_count_native_packer import (
        pack_segment_count_topology,
    )
    from dreamplace.ops.buffer_insertion.segment_count_state import (
        build_packed_segment_count_state,
    )
    from test_segment_count_native_packer import (
        _fixture,
    )

    values = _fixture()
    packed = pack_segment_count_topology(**values, num_threads=1)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=1.0,
        dtype=values["pin_capacitance"].dtype,
    )
    table = torch.tensor([[0.1, 0.2, 0.3]], dtype=state.z_param.dtype)
    provider = SegmentCountDynamicNetProvider(
        nets=(),
        segment_state=state,
        per_size_input_cap=table,
        per_size_delay=table,
        per_size_output_slew=table,
        backend="cpp_cuda_segment_transfer_explicit_autograd",
        profile_enabled=False,
        prepared_timing_inputs=state.prepared_timing_inputs,
    )

    monkeypatch.setattr(
        provider,
        "_packed_driver_pins",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native selector must provide driver_pin_id directly")
        ),
    )
    view = provider._level_view(
        torch.tensor([1], dtype=torch.long), level_id=3, level_view_epoch=0
    )
    active = provider._active_segment_state_from_source_rows(
        view["source_segment_row_index"]
    )
    active.z_param.sum().backward()

    assert view["native_active_view"]
    assert view["has_dynamic_nets"]
    assert provider.metadata["active_view_selector_used"] == "native"
    assert state.z_param.grad is not None
    torch.testing.assert_close(state.z_param.grad, torch.ones_like(state.z_param))

    provider.reset_runtime_metadata()
    assert not provider.metadata["active_view_selector_requested"]
    assert provider.metadata["active_view_selector_used"] == "python_reference"
    assert provider.metadata["native_active_view_selector_count"] == 0


def test_reference_python_level_dispatch_keeps_current_driver_timing(monkeypatch):
    provider, _state = _provider()
    provider = SegmentCountDynamicNetProvider(
        nets=[_line_net()],
        segment_state=provider.segment_state,
        per_size_input_cap=provider.per_size_input_cap,
        per_size_delay=provider.per_size_delay,
        per_size_output_slew=provider.per_size_output_slew,
        backend="reference_python",
    )
    observed = []
    original = provider_module.segment_count_relaxed_timing

    def spy(nets_arg, **kwargs):
        observed.append(
            (
                kwargs["driver_arrival_by_net"],
                kwargs["driver_slew_by_net"],
            )
        )
        return original(nets_arg, **kwargs)

    monkeypatch.setattr(provider_module, "segment_count_relaxed_timing", spy)
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, *args):
        return args[:4]

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
        level_id=3,
        level_view_epoch=0,
    )

    assert provider.metadata["tensorized_driver_gather_count"] == 0
    assert [float(values[0][5]) for values in observed] == [40.0, 40.0, 50.0, 50.0]
    assert [float(values[1][5]) for values in observed] == [2.0, 2.0, 3.0, 3.0]


def test_segment_count_provider_reset_releases_prior_cap_overlay_graph():
    provider, state = _provider()
    provider._last_pin_net_cap_rise_after_overlay = state.z_param.sum()
    provider._last_pin_net_cap_fall_after_overlay = state.z_param.sum()

    provider.reset_runtime_metadata()

    assert provider._last_pin_net_cap_rise_after_overlay is None
    assert provider._last_pin_net_cap_fall_after_overlay is None


def test_segment_count_provider_overlays_driver_cap_with_root_load():
    nets = [_branched_net()]
    nets[0]["coordinates"].pop(3)
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=0.5,
        initial_bsu_index=0.0,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.8, 0.8]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.1]], dtype=torch.float32),
        backend="prepared_python",
    )
    pin_caps = torch.full((4,), -1.0, dtype=torch.float32)
    pin_caps[1] = 3.0

    pin_cap_rise, pin_cap_fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )

    torch.testing.assert_close(pin_cap_rise[1], torch.tensor(2.4))
    torch.testing.assert_close(pin_cap_fall[1], torch.tensor(2.4))
    torch.testing.assert_close(pin_cap_rise[2], torch.tensor(-1.0))
    torch.testing.assert_close(pin_cap_rise[3], torch.tensor(-1.0))

    loss = pin_cap_rise[1] + pin_cap_fall[1]
    loss.backward()
    assert state.z_param.grad is not None
    assert float(state.z_param.grad[0].item()) < 0.0


def test_segment_count_provider_zero_state_preserves_static_driver_cap_gradient():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=0.0,
        initial_bsu_index=0.0,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        backend="prepared_python",
    )
    pin_caps = torch.tensor([0.0, 9.0, 0.0], dtype=torch.float32)

    rise, fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )

    torch.testing.assert_close(rise, pin_caps)
    torch.testing.assert_close(fall, pin_caps)
    (rise[1] + fall[1]).backward()
    assert state.z_param.grad is not None
    assert float(state.z_param.grad.abs().max().item()) > 0.0


def test_segment_count_provider_zero_state_preserves_static_sink_timing_gradient():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=0.0,
        initial_bsu_index=0.0,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        backend="prepared_python",
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, -1.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(
        _net_ids,
        rise_aat,
        fall_aat,
        rise_tran,
        fall_tran,
        *_args,
    ):
        rise_aat = rise_aat.clone()
        fall_aat = fall_aat.clone()
        rise_tran = rise_tran.clone()
        fall_tran = fall_tran.clone()
        rise_aat[2] = 123.0
        fall_aat[2] = 124.0
        rise_tran[2] = 7.0
        fall_tran[2] = 8.0
        return rise_aat, fall_aat, rise_tran, fall_tran

    rise_aat, fall_aat, rise_tran, fall_tran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    torch.testing.assert_close(rise_aat[2], torch.tensor(123.0))
    torch.testing.assert_close(fall_aat[2], torch.tensor(124.0))
    torch.testing.assert_close(rise_tran[2], torch.tensor(7.0))
    torch.testing.assert_close(fall_tran[2], torch.tensor(8.0))
    (rise_aat[2] + fall_aat[2] + rise_tran[2] + fall_tran[2]).backward()
    assert state.z_param.grad is not None
    assert float(state.z_param.grad.abs().max().item()) > 0.0


def test_segment_count_provider_cpp_explicit_driver_cap_overlay_has_z_gradient():
    nets = [_branched_net()]
    nets[0]["coordinates"].pop(3)
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=0.5,
        initial_bsu_index=0.0,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.8, 0.8]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.1]], dtype=torch.float32),
        backend="cpp_cpu_explicit_autograd",
    )
    pin_caps = torch.full((4,), -1.0, dtype=torch.float32)
    pin_caps[1] = 3.0

    pin_cap_rise, pin_cap_fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )

    torch.testing.assert_close(pin_cap_rise[1], torch.tensor(2.4))
    loss = pin_cap_rise[1] + pin_cap_fall[1]
    loss.backward()
    assert state.z_param.grad is not None
    assert float(state.z_param.grad[0].item()) < 0.0


def test_segment_count_provider_cuda_driver_cap_overlay_uses_cap_only(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda", torch.cuda.current_device())
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=2,
        initial_z=0.5,
        initial_bsu_index=0.0,
        dtype=torch.float64,
        device=device,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
        backend="cpp_cuda_segment_transfer_explicit_autograd",
    )

    def fail_full_transfer(*args, **kwargs):
        raise AssertionError("CUDA driver-cap overlay must use the cap-only path")

    observed_edge_capacitance = []
    original_cap_only = provider_module.segment_count_driver_cap_native_cuda_autograd

    def capture_cap_only(prepared_inputs, **kwargs):
        observed_edge_capacitance.append(prepared_inputs["edge_capacitance"])
        return original_cap_only(prepared_inputs, **kwargs)

    live_capacitance = torch.full(
        (1,),
        0.125,
        dtype=torch.float64,
        device=device,
    )
    monkeypatch.setattr(provider, "_run_lane", fail_full_transfer)
    monkeypatch.setattr(
        provider,
        "_live_edge_overrides",
        lambda _prepared: {
            "edge_resistance_override": torch.full_like(live_capacitance, 2.0),
            "edge_capacitance_override": live_capacitance,
        },
    )
    monkeypatch.setattr(
        provider_module,
        "segment_count_driver_cap_native_cuda_autograd",
        capture_cap_only,
    )
    pin_caps = torch.full((3,), -1.0, dtype=torch.float64, device=device)
    pin_cap_rise, pin_cap_fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )

    assert torch.isfinite(pin_cap_rise[1])
    assert torch.isfinite(pin_cap_fall[1])
    assert provider.metadata["driver_cap_overlay_path"] == "cuda_cap_only"
    assert provider.metadata["driver_cap_overlay_fallback_reason"] is None
    assert len(observed_edge_capacitance) == 2
    assert all(value is live_capacitance for value in observed_edge_capacitance)
    (pin_cap_rise[1] + pin_cap_fall[1]).backward()
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert torch.isfinite(state.z_param.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()


def test_segment_count_provider_cuda_cap_overlay_emits_synchronized_profile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda", torch.cuda.current_device())
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=2,
        initial_z=0.5,
        initial_bsu_index=0.0,
        dtype=torch.float64,
        device=device,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
        backend="cpp_cuda_segment_transfer_explicit_autograd",
        profile_enabled=True,
    )
    pin_caps = torch.full((3,), -1.0, dtype=torch.float64, device=device)

    provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )
    provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=pin_caps,
        pin_net_cap_fall=pin_caps,
    )

    metadata = provider.metadata
    assert metadata["driver_cap_overlay_path"] == "cuda_cap_only"
    assert metadata["driver_cap_overlay_fallback_reason"] is None
    assert metadata["driver_cap_overlay_backend"] == "cpp_cuda_segment_count_cap_autograd"
    for key in (
        "driver_cap_overlay_forward_ms",
        "driver_cap_overlay_scatter_ms",
        "driver_cap_overlay_ms",
        "gpu_static_runtime_build_wall_ms",
        "gpu_static_runtime_reuse_wall_ms",
    ):
        assert torch.isfinite(torch.tensor(metadata[key]))
        assert metadata[key] >= 0.0
    assert metadata["gpu_static_runtime_build_count"] > 0
    assert metadata["gpu_static_runtime_cache_hit_count"] > 0
    assert metadata["gpu_static_runtime_h2d_bytes"] > 0
    assert metadata["gpu_static_runtime_d2h_bytes"] == 0
    assert metadata["gpu_peak_allocated_bytes"] > 0
    assert metadata["gpu_peak_reserved_bytes"] > 0


def test_segment_count_provider_cuda_level_profile_splits_rise_and_fall_work():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda", torch.cuda.current_device())
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=2,
        initial_z=0.5,
        initial_bsu_index=0.0,
        dtype=torch.float64,
        device=device,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
        backend="cpp_cuda_segment_transfer_explicit_autograd",
        profile_enabled=True,
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float64, device=device)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float64, device=device)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64, device=device)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64, device=device)
    zeros = torch.zeros(3, dtype=torch.float64, device=device)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        wire_delay = torch.tensor(10.0, dtype=pin_rAAT.dtype)
        pin_rAAT[2] = pin_rAAT[1] + wire_delay
        pin_fAAT[2] = pin_fAAT[1] + wire_delay
        log10 = torch.log(torch.tensor(10.0, dtype=pin_rAAT.dtype))
        pin_rtran[2] = torch.sqrt(pin_rtran[1] ** 2 + (log10 * wire_delay) ** 2)
        pin_ftran[2] = torch.sqrt(pin_ftran[1] ** 2 + (log10 * wire_delay) ** 2)
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
        level_id=1,
        level_view_epoch=0,
    )

    for key in (
        "driver_timing_gather_ms",
        "rise_transfer_ms",
        "fall_transfer_ms",
        "rise_sink_scatter_ms",
        "fall_sink_scatter_ms",
        "sink_scatter_ms",
        "provider_dispatch_ms",
    ):
        assert torch.isfinite(torch.tensor(provider.metadata[key]))
        assert provider.metadata[key] >= 0.0
    assert provider.metadata["segment_state_device"] == str(device)


def test_segment_count_provider_uses_current_driver_timing_and_gradients():
    provider, state = _provider()
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        wire_delay = torch.tensor(10.0, dtype=pin_rAAT.dtype)
        log10 = torch.log(torch.tensor(10.0, dtype=pin_rAAT.dtype))
        pin_rAAT[2] = pin_rAAT[1] + wire_delay
        pin_fAAT[2] = pin_fAAT[1] + wire_delay
        pin_rtran[2] = torch.sqrt(pin_rtran[1] ** 2 + (log10 * wire_delay) ** 2)
        pin_ftran[2] = torch.sqrt(pin_ftran[1] ** 2 + (log10 * wire_delay) ** 2)
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert float(pin_rAAT[2].item()) > 40.0
    assert float(pin_fAAT[2].item()) > 50.0
    assert float(pin_rtran[2].item()) > 0.0
    assert float(pin_ftran[2].item()) > 0.0
    loss = pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2]
    loss.backward()
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert torch.isfinite(state.z_param.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.z_param.grad.abs().max().item()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max().item()) > 0.0
    assert provider.metadata["uses_current_propagated_driver_slew"] is True
    assert provider.metadata["affected_net_call_count"] == 1
    assert provider.metadata["segment_count_state_count"] == 1
    assert provider.metadata["max_repeater_count"] == 2
    assert provider.metadata["segment_shared_bsu"] is True
    assert provider.metadata["segment_count_timing_backend_used"] == "prepared_python"


def test_segment_count_provider_writes_device_probe(tmp_path, monkeypatch):
    provider, _state = _provider()
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    base_cap = torch.full((3,), -1.0, dtype=torch.float32)
    base_cap[1] = 1.0
    provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=base_cap,
        pin_net_cap_fall=base_cap,
    )
    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "segment_count_device_model_probe"
    assert len(payload["rows"]) == 2
    by_sense = {row["sense"]: row for row in payload["rows"]}
    assert by_sense["rise"]["driver_slew"] == 2.0
    assert by_sense["fall"]["driver_slew"] == 3.0
    assert by_sense["rise"]["static_buffer_delay"] == 3.0
    assert by_sense["rise"]["static_buffer_output_slew"] == pytest.approx(0.3)
    assert by_sense["rise"]["static_buffer_input_cap"] == pytest.approx(0.6)
    assert by_sense["rise"]["per_size_delay"] == [1.0, 2.0, 4.0]
    assert by_sense["rise"]["z_value"] == 1.25
    assert by_sense["rise"]["bsu_index"] == 1.5
    assert by_sense["rise"]["driver_pin_id"] == 1
    assert by_sense["rise"]["root_node_id"] == 1
    assert by_sense["rise"]["root_matches_driver_pin"] is True
    assert by_sense["rise"]["is_parent_root"] is True
    assert by_sense["rise"]["parent_compact_id"] == 0
    assert by_sense["rise"]["child_compact_id"] == 1
    assert by_sense["rise"]["root_to_parent_path_node_ids"] == [1]
    assert by_sense["rise"]["root_to_child_path_node_ids"] == [1, 2]
    assert by_sense["rise"]["root_to_parent_path"]["edge_count"] == 0
    assert by_sense["rise"]["root_to_child_path"]["edge_count"] == 1
    assert by_sense["rise"]["root_to_child_path"]["r_sum"] == 10.0
    assert by_sense["rise"]["driver_overlay_net_cap"] == pytest.approx(0.6)
    assert by_sense["rise"]["driver_zero_z_net_cap"] == pytest.approx(1.0)
    assert by_sense["rise"]["driver_pin_net_cap_rise_after_overlay"] == pytest.approx(0.6)
    assert by_sense["rise"]["driver_pin_net_cap_fall_after_overlay"] == pytest.approx(0.6)
    assert by_sense["rise"]["root_load_decomposition"]["root_node_id"] == 1
    assert by_sense["rise"]["root_load_decomposition"]["zero_z_relaxed_cap"] == pytest.approx(1.0)


def test_segment_count_provider_device_probe_can_include_analytic_transfer(tmp_path, monkeypatch):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
    )
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float64)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float64)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
    zeros = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        wire_delay = torch.tensor(10.0, dtype=pin_rAAT.dtype)
        log10 = torch.log(torch.tensor(10.0, dtype=pin_rAAT.dtype))
        pin_rAAT[2] = pin_rAAT[1] + wire_delay
        pin_fAAT[2] = pin_fAAT[1] + wire_delay
        pin_rtran[2] = torch.sqrt(pin_rtran[1] ** 2 + (log10 * wire_delay) ** 2)
        pin_ftran[2] = torch.sqrt(pin_ftran[1] ** 2 + (log10 * wire_delay) ** 2)
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    by_sense = {row["sense"]: row for row in payload["rows"]}
    assert by_sense["rise"]["analytic_transfer_source"] == "deterministic_fake_lut"
    assert "analytic_transfer_buffer_input_slew" in by_sense["rise"]
    assert "analytic_transfer_buffer_output_load" in by_sense["rise"]
    assert by_sense["rise"]["analytic_transfer_buffer_delay"] != by_sense["rise"]["static_buffer_delay"]
    assert provider.metadata["segment_transfer_device_lut_source"] == "deterministic_fake_lut"


def test_segment_count_provider_device_probe_replays_projected_repeater_count(tmp_path, monkeypatch):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=2,
        initial_z=2.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
    )
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    pins = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
        pin_ftran=torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64),
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    rise = {row["sense"]: row for row in payload["rows"]}["rise"]
    assert rise["analytic_transfer_repeater_count"] == 2
    assert len(rise["analytic_transfer_buffer_delays"]) == 2
    assert len(rise["analytic_transfer_buffer_output_slews"]) == 2


def test_segment_count_device_probe_can_replay_opensta_input_state(tmp_path, monkeypatch):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
    )
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    monkeypatch.setenv(
        "AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_INPUT_SLEW_PS",
        "8.0,9.0",
    )
    monkeypatch.setenv(
        "AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_OUTPUT_LOAD_PF",
        "4.0",
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float64)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float64)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
    zeros = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    by_sense = {row["sense"]: row for row in payload["rows"]}
    assert by_sense["rise"]["opensta_input_replay_input_slew_ps"] == 8.0
    assert by_sense["fall"]["opensta_input_replay_input_slew_ps"] == 9.0
    assert by_sense["rise"]["opensta_input_replay_output_load_pf"] == 4.0
    assert by_sense["fall"]["opensta_input_replay_output_load_pf"] == 4.0
    assert (
        by_sense["rise"]["opensta_input_replay_buffer_delay"]
        != by_sense["rise"]["analytic_transfer_buffer_delay"]
    )
    assert (
        by_sense["fall"]["opensta_input_replay_buffer_output_slew"]
        != by_sense["fall"]["analytic_transfer_buffer_output_slew"]
    )


def test_segment_count_device_probe_reports_arc_level_replay(tmp_path, monkeypatch):
    arc_rows = [
        {
            "arc_id": 10,
            "prefix": "f",
            "input_slew_axis": [0.0, 10.0],
            "output_load_axis": [0.0, 10.0],
            "delay_lut": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
            "output_slew_lut": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        },
        {
            "arc_id": 10,
            "prefix": "r",
            "input_slew_axis": [0.0, 10.0],
            "output_load_axis": [0.0, 10.0],
            "delay_lut": torch.tensor([[20.0, 30.0], [40.0, 50.0]], dtype=torch.float64),
            "output_slew_lut": torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64),
        },
    ]
    device = BufferDeviceLut(
        input_cap_by_size=torch.tensor([0.3, 0.5], dtype=torch.float64),
        input_slew_axis=torch.tensor([0.0, 10.0], dtype=torch.float64),
        output_load_axis=torch.tensor([0.0, 10.0], dtype=torch.float64),
        delay_lut=torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
            ],
            dtype=torch.float64,
        ),
        output_slew_lut=torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.3], [0.4, 0.5]],
            ],
            dtype=torch.float64,
        ),
        source="arc_diagnostic_test_lut",
        arc_luts_by_size=[arc_rows, arc_rows],
    )
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=device,
    )
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_INPUT_SLEW_PS", "5.0")
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_OUTPUT_LOAD_PF", "15.0")
    pins = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
        pin_ftran=torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64),
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    row = {item["sense"]: item for item in payload["rows"]}["rise"]
    summary = row["opensta_input_arc_replay"]
    assert summary["arc_count"] == 2
    assert summary["delay_stats"]["min"] == pytest.approx(35.0)
    assert summary["delay_stats"]["max"] == pytest.approx(45.0)
    assert summary["output_slew_stats"]["min"] == pytest.approx(3.5)
    assert summary["output_slew_stats"]["max"] == pytest.approx(4.5)
    assert sorted(summary["arcs_by_size"]) == ["0", "1"]


def _assert_segment_count_provider_can_select_transfer_backend(transfer_backend):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
        segment_transfer_backend=transfer_backend,
    )
    pins = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        wire_delay = torch.tensor(10.0, dtype=pin_rAAT.dtype)
        log10 = torch.log(torch.tensor(10.0, dtype=pin_rAAT.dtype))
        pin_rAAT[2] = pin_rAAT[1] + wire_delay
        pin_fAAT[2] = pin_fAAT[1] + wire_delay
        pin_rtran[2] = torch.sqrt(pin_rtran[1] ** 2 + (log10 * wire_delay) ** 2)
        pin_ftran[2] = torch.sqrt(pin_ftran[1] ** 2 + (log10 * wire_delay) ** 2)
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
        pin_ftran=torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64),
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert float(pin_rAAT[2].item()) > 0.0
    assert float(pin_fAAT[2].item()) > 0.0
    assert provider.metadata["segment_transfer_backend"] == transfer_backend
    assert provider.metadata["segment_transfer_forward_ms"] >= 0.0


def test_segment_count_provider_can_select_python_segment_transfer_backend():
    _assert_segment_count_provider_can_select_transfer_backend("segment_transfer_python")


def test_segment_count_provider_can_select_native_segment_transfer_backend():
    _assert_segment_count_provider_can_select_transfer_backend("segment_transfer_native")


def test_segment_count_provider_rejects_transfer_backend_for_native_backend():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    with pytest.raises(ValueError, match="prepared_python"):
        SegmentCountDynamicNetProvider(
            nets=nets,
            segment_state=state,
            per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
            per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
            backend="cpp_cpu",
            buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
            segment_transfer_backend="segment_transfer_native",
        )


def test_segment_count_provider_analytic_probe_uses_explicit_retained_upstream_cap(tmp_path, monkeypatch):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        backend="prepared_python",
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
        segment_retained_upstream_cap=torch.tensor([0.25], dtype=torch.float64),
    )
    probe_path = tmp_path / "segment_device_probe.json"
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON", str(probe_path))
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "0")
    pins = torch.zeros(3, dtype=torch.float64)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64),
        pin_ftran=torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64),
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    row = {item["sense"]: item for item in payload["rows"]}["rise"]
    assert row["analytic_transfer_upstream_retained_capacitance"] == pytest.approx(0.25)
    assert row["analytic_transfer_upstream_visible_input_cap"] > 0.25
    assert provider.metadata["segment_retained_upstream_cap_source"] == "explicit_segment_tensor"


def test_segment_count_provider_rejects_wrong_retained_upstream_cap_length():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=1,
        initial_z=1.0,
        initial_bsu_index=0.5,
    )
    with pytest.raises(ValueError, match="segment_retained_upstream_cap"):
        SegmentCountDynamicNetProvider(
            nets=nets,
            segment_state=state,
            per_size_input_cap=torch.tensor([[0.3, 0.5]], dtype=torch.float64),
            per_size_delay=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
            segment_retained_upstream_cap=torch.tensor([0.1, 0.2], dtype=torch.float64),
        )


def test_segment_count_provider_falls_back_to_static_for_unaffected_nets():
    provider, _state = _provider()
    calls = {"static": 0}
    pins = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        calls["static"] += 1
        return pin_rAAT + 1.0, pin_fAAT + 1.0, pin_rtran + 1.0, pin_ftran + 1.0

    result = provider.propagate_net_aat_level(
        net_ids=torch.tensor([9], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=pins,
        pin_ftran=pins,
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert calls["static"] == 1
    assert torch.allclose(result[0], torch.ones(3))
    assert not provider.has_dynamic_nets(torch.tensor([9], dtype=torch.long))


def test_segment_count_provider_cpp_cpu_backend_runs_forward_without_backward():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend="cpp_cpu",
    )
    pins = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        wire_delay = torch.tensor(10.0, dtype=pin_rAAT.dtype)
        log10 = torch.log(torch.tensor(10.0, dtype=pin_rAAT.dtype))
        pin_rAAT[2] = pin_rAAT[1] + wire_delay
        pin_fAAT[2] = pin_fAAT[1] + wire_delay
        pin_rtran[2] = torch.sqrt(pin_rtran[1] ** 2 + (log10 * wire_delay) ** 2)
        pin_ftran[2] = torch.sqrt(pin_ftran[1] ** 2 + (log10 * wire_delay) ** 2)
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=pins,
        pin_ftran=pins,
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert float(pin_rAAT[2].item()) > 0.0
    assert float(pin_fAAT[2].item()) > 0.0
    assert float(pin_rtran[2].item()) > 0.0
    assert float(pin_ftran[2].item()) > 0.0
    assert provider.metadata["segment_count_timing_backend_used"] == "cpp_cpu"


def _assert_segment_count_provider_gradient_backend(backend):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend=backend,
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    loss = pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2]
    loss.backward()
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert torch.isfinite(state.z_param.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.z_param.grad.abs().max().item()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max().item()) > 0.0
    assert provider.metadata["segment_count_timing_backend_used"] == backend


def test_segment_count_provider_cpp_cpu_recompute_autograd_backend_produces_gradients():
    _assert_segment_count_provider_gradient_backend("cpp_cpu_recompute_autograd")


def test_segment_count_provider_cpp_cpu_explicit_autograd_backend_produces_gradients():
    _assert_segment_count_provider_gradient_backend("cpp_cpu_explicit_autograd")


def test_segment_count_provider_cpp_cpu_segment_transfer_backend_produces_gradients():
    _assert_segment_count_provider_segment_transfer_gradient_backend(
        "cpp_cpu_segment_transfer_recompute_autograd"
    )


def test_segment_count_provider_cpp_cpu_segment_transfer_explicit_backend_produces_gradients():
    _assert_segment_count_provider_segment_transfer_gradient_backend(
        "cpp_cpu_segment_transfer_explicit_autograd"
    )


def test_segment_count_provider_segment_transfer_backend_reports_effective_native_transfer():
    provider, _state = _assert_segment_count_provider_segment_transfer_gradient_backend(
        "cpp_cpu_segment_transfer_explicit_autograd",
        segment_transfer_backend="static_size_table",
    )
    assert provider.metadata["segment_transfer_backend_requested"] == "static_size_table"
    assert provider.metadata["segment_transfer_backend"] == "segment_transfer_native"
    assert provider.metadata["segment_transfer_backend_used"] == "segment_transfer_native"


def test_segment_count_provider_cuda_segment_transfer_backend_uses_explicit_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    provider, state = _assert_segment_count_provider_segment_transfer_gradient_backend(
        "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_requested"] == (
        "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_used"] == (
        "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_fallback_reason"] is None
    assert provider.metadata["segment_count_timing_backend_fallback_is_explicit"] is False
    assert provider.metadata["segment_transfer_backend_used"] == "segment_transfer_native"
    assert state.z_param.grad is not None


def test_segment_count_provider_cuda_alias_backend_uses_explicit_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    provider, _state = _assert_segment_count_provider_segment_transfer_gradient_backend(
        "cpp_cuda_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_requested"] == (
        "cpp_cuda_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_used"] == (
        "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert provider.metadata["segment_count_timing_backend_fallback_reason"] is None
    assert provider.metadata["segment_count_timing_backend_fallback_is_explicit"] is False
    assert provider.metadata["segment_transfer_backend_used"] == "segment_transfer_native"


def _assert_segment_count_provider_segment_transfer_gradient_backend(
    backend,
    *,
    segment_transfer_backend="segment_transfer_native",
):
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend=backend,
        buffer_device_lut=default_fake_buffer_device(dtype=torch.float32),
        segment_transfer_backend=segment_transfer_backend,
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    loss = pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2]
    loss.backward()
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert torch.isfinite(state.z_param.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.z_param.grad.abs().max().item()) > 0.0
    expected_used = (
        "cpp_cuda_segment_transfer_explicit_autograd"
        if backend.startswith("cpp_cuda")
        else backend
    )
    assert provider.metadata["segment_count_timing_backend_requested"] == backend
    assert provider.metadata["segment_count_timing_backend_used"] == expected_used
    assert provider.metadata["segment_transfer_backend"] == "segment_transfer_native"
    assert provider.metadata["segment_transfer_backend_used"] == "segment_transfer_native"
    return provider, state


def test_segment_count_provider_can_disable_detailed_profile_timing():
    nets = [_line_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend="cpp_cpu_explicit_autograd",
        profile_enabled=False,
    )
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(3, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pin_rAAT,
        pin_fAAT=pin_fAAT,
        pin_rtran=pin_rtran,
        pin_ftran=pin_ftran,
        base_pin_net_delay_rise=zeros,
        base_pin_net_delay_fall=zeros,
        base_pin_net_impulse_rise=zeros,
        base_pin_net_impulse_fall=zeros,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert provider.metadata["profile_enabled"] is False
    assert provider.metadata["dynamic_provider_call_count"] == 1
    assert provider.metadata["dynamic_provider_affected_net_call_count"] == 1
    assert provider.metadata["segment_count_timing_backend_used"] == "cpp_cpu_explicit_autograd"
    assert provider.metadata["provider_dispatch_ms"] == 0.0
    assert provider.metadata["driver_timing_gather_ms"] == 0.0
    assert provider.metadata["segment_count_relaxed_timing_ms"] == 0.0
    assert provider.metadata["segment_count_timing_forward_ms"] == 0.0
    assert provider.metadata["net_subgraph_forward_relaxed_ms"] == 0.0
    assert provider.metadata["sink_scatter_ms"] == 0.0
    assert provider.metadata["driver_cap_overlay_forward_ms"] == 0.0
    assert provider.metadata["driver_cap_overlay_scatter_ms"] == 0.0
    assert provider.metadata["gpu_static_runtime_h2d_bytes"] == 0
    assert provider.metadata["gpu_static_runtime_d2h_bytes"] == 0


def test_segment_count_provider_only_runs_active_level_nets(monkeypatch):
    nets = [
        _line_net_with_id(5, 1, 2),
        _line_net_with_id(6, 3, 4),
    ]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor(
            [[0.3, 0.5, 0.7], [0.3, 0.5, 0.7]],
            dtype=torch.float32,
        ),
        per_size_delay=torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]],
            dtype=torch.float32,
        ),
        per_size_output_slew=torch.tensor(
            [[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]],
            dtype=torch.float32,
        ),
        backend="reference_python",
    )
    observed = []
    original = provider_module.segment_count_relaxed_timing

    def spy(nets_arg, **kwargs):
        observed.append(
            {
                "net_ids": [int(net["net_id"]) for net in nets_arg],
                "segment_count": int(kwargs["segment_state"].z_param.numel()),
            }
        )
        return original(nets_arg, **kwargs)

    monkeypatch.setattr(provider_module, "segment_count_relaxed_timing", spy)
    pins = torch.zeros(5, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([5], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=pins,
        pin_ftran=pins,
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert observed == [
        {"net_ids": [5], "segment_count": 1},
        {"net_ids": [5], "segment_count": 1},
        {"net_ids": [5], "segment_count": 1},
        {"net_ids": [5], "segment_count": 1},
    ]


def test_segment_count_provider_uses_empty_segment_view_for_active_nets_without_segments(monkeypatch):
    nets = [
        _line_net_with_id(5, 1, 2),
        _line_net_with_id(7, 3, 4),
    ]
    # Remove RC for net 7 so it is part of provider dispatch but has no eligible
    # segment-count variables.
    nets[1]["rc_tree"]["edge_rc"] = {}
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend="reference_python",
    )
    observed = []
    original = provider_module.segment_count_relaxed_timing

    def spy(nets_arg, **kwargs):
        observed.append(
            {
                "net_ids": [int(net["net_id"]) for net in nets_arg],
                "segment_count": int(kwargs["segment_state"].z_param.numel()),
            }
        )
        return original(nets_arg, **kwargs)

    monkeypatch.setattr(provider_module, "segment_count_relaxed_timing", spy)
    pins = torch.zeros(5, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    provider.propagate_net_aat_level(
        net_ids=torch.tensor([7], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=pins,
        pin_ftran=pins,
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert observed == [
        {"net_ids": [7], "segment_count": 0},
        {"net_ids": [7], "segment_count": 0},
        {"net_ids": [7], "segment_count": 0},
        {"net_ids": [7], "segment_count": 0},
    ]


def test_segment_count_provider_cpp_cpu_recompute_autograd_handles_active_net_without_segments():
    nets = [
        _line_net_with_id(5, 1, 2),
        _line_net_with_id(7, 3, 4),
    ]
    nets[1]["rc_tree"]["edge_rc"] = {}
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.5,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float32),
        per_size_delay=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
        backend="cpp_cpu_recompute_autograd",
    )
    pins = torch.zeros(5, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    result = provider.propagate_net_aat_level(
        net_ids=torch.tensor([7], dtype=torch.long),
        pin_rAAT=pins,
        pin_fAAT=pins,
        pin_rtran=pins,
        pin_ftran=pins,
        base_pin_net_delay_rise=pins,
        base_pin_net_delay_fall=pins,
        base_pin_net_impulse_rise=pins,
        base_pin_net_impulse_fall=pins,
        static_calculate_net_aat_level=static_calculate_net_aat_level,
    )

    assert torch.equal(result[0], pins)
    assert torch.equal(result[1], pins)
    assert provider.metadata["segment_count_timing_backend_used"] == "cpp_cpu_recompute_autograd"
