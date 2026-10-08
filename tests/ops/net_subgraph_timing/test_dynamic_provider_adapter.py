import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    build_discrete_candidate_state,
)
from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
    RelaxedBufferDynamicNetProvider,
)
from dreamplace.ops.net_subgraph_timing.timing_adapter import (
    build_relaxed_buffer_timing_payload,
)


def _build_provider():
    candidates = [
        {
            "candidate_id": 11,
            "net_id": 5,
            "node_id": 2,
            "candidate_node_id": 2,
            "buffer_main_type_index": 4,
        }
    ]
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    nets = [
        {
            "net_id": 5,
            "driver_pin_id": 1,
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 0.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.3]], dtype=torch.float32),
        driver_arrival_by_net={5: -999.0},
        driver_slew_by_net={5: 99.0},
        coordinate_source="fixed_smoke",
        num_pins=3,
    )
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
        affected_net_ids=[5],
    )
    return provider, state


def _build_two_net_provider():
    candidates = [
        {
            "candidate_id": 11,
            "net_id": 5,
            "node_id": 2,
            "candidate_node_id": 2,
            "buffer_main_type_index": 4,
        },
        {
            "candidate_id": 22,
            "net_id": 6,
            "node_id": 4,
            "candidate_node_id": 4,
            "buffer_main_type_index": 4,
        },
    ]
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    nets = [
        {
            "net_id": 5,
            "driver_pin_id": 1,
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 0.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        },
        {
            "net_id": 6,
            "driver_pin_id": 3,
            "rc_tree": {
                "root_node_id": 3,
                "children_by_node": {3: [4], 4: []},
                "edge_rc": {(3, 4): {"r": 0.0, "c": 0.0}},
                "node_cap": {3: 0.0, 4: 1.0},
                "sink_nodes": [4],
            },
        },
    ]
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor(
            [[0.2, 0.4], [0.2, 0.4]],
            dtype=torch.float32,
        ),
        per_size_delay=torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]],
            dtype=torch.float32,
        ),
        per_size_output_slew=torch.tensor(
            [[0.1, 0.3], [0.1, 0.3]],
            dtype=torch.float32,
        ),
        driver_arrival_by_net={5: -999.0, 6: -999.0},
        driver_slew_by_net={5: 99.0, 6: 99.0},
        coordinate_source="fixed_smoke",
        num_pins=5,
    )
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
    )
    return provider, state


def _build_discrete_provider(*, two_nets=False):
    candidates = [
        {
            "candidate_id": 11,
            "net_id": 5,
            "node_id": 2,
            "candidate_node_id": 2,
            "buffer_main_type_index": 4,
            "x_dbu": 100,
            "y_dbu": 0,
        }
    ]
    nets = [
        {
            "net_id": 5,
            "driver_pin_id": 1,
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 1.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    per_size_input_cap = [[0.2, 0.2]]
    per_size_delay = [[0.1, 0.1]]
    per_size_output_slew = [[0.1, 0.1]]
    if two_nets:
        candidates.append(
            {
                "candidate_id": 22,
                "net_id": 6,
                "node_id": 4,
                "candidate_node_id": 4,
                "buffer_main_type_index": 4,
                "x_dbu": 200,
                "y_dbu": 0,
            }
        )
        nets.append(
            {
                "net_id": 6,
                "driver_pin_id": 3,
                "rc_tree": {
                    "root_node_id": 3,
                    "children_by_node": {3: [4], 4: []},
                    "edge_rc": {(3, 4): {"r": 1.0, "c": 0.0}},
                    "node_cap": {3: 0.0, 4: 2.0},
                    "sink_nodes": [4],
                },
            }
        )
        per_size_input_cap.append([0.8, 0.8])
        per_size_delay.append([0.6, 0.6])
        per_size_output_slew.append([0.4, 0.4])
    state = build_discrete_candidate_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        fixed_bsu_index=0,
    )
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor(per_size_input_cap, dtype=torch.float32),
        per_size_delay=torch.tensor(per_size_delay, dtype=torch.float32),
        per_size_output_slew=torch.tensor(per_size_output_slew, dtype=torch.float32),
        coordinate_source="discrete_candidate_test",
        num_pins=5 if two_nets else 3,
    )
    return RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
    ), state


def test_relaxed_buffer_provider_consumes_current_driver_timing():
    provider, state = _build_provider()
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

    assert float(pin_rAAT[2].item()) > 40.0
    assert float(pin_fAAT[2].item()) > 50.0
    assert float(pin_rAAT[2].item()) > -999.0
    loss = pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2]
    loss.backward()
    assert state.bu_logits.grad is not None
    assert state.bsu_index_param.grad is not None
    assert float(state.bu_logits.grad.abs().max().item()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max().item()) > 0.0
    assert provider.metadata["provider_dispatch_ms"] > 0.0
    assert provider.metadata["driver_timing_gather_ms"] > 0.0
    assert provider.metadata["net_subgraph_forward_relaxed_ms"] > 0.0
    assert provider.metadata["sink_scatter_ms"] > 0.0
    assert provider.metadata["static_payload_cache_hit_count"] > 0


def test_candidate_driver_cap_overlay_tracks_virtual_buffer_state():
    provider, state = _build_discrete_provider()
    base_rise = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    base_fall = base_rise.clone()

    zero_rise, zero_fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=base_rise,
        pin_net_cap_fall=base_fall,
    )
    torch.testing.assert_close(zero_rise[1], torch.tensor(1.0))
    torch.testing.assert_close(zero_fall[1], torch.tensor(1.0))

    with torch.no_grad():
        state.activation_param.fill_(1.0)
    buffered_rise, buffered_fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=base_rise,
        pin_net_cap_fall=base_fall,
    )
    torch.testing.assert_close(buffered_rise[1], torch.tensor(0.2))
    torch.testing.assert_close(buffered_fall[1], torch.tensor(0.2))
    assert provider.metadata["driver_cap_overlay_path"] == "candidate_native_load_only"


def test_candidate_driver_cap_overlay_preserves_bu_gradient():
    provider, state = _build_provider()
    base_cap = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    rise, _fall = provider.apply_dynamic_net_cap_overlay(
        pin_net_cap_rise=base_cap,
        pin_net_cap_fall=base_cap,
    )
    rise[1].backward()

    assert state.bu_logits.grad is not None
    assert float(state.bu_logits.grad[0]) < 0.0


def test_relaxed_buffer_provider_scopes_forward_to_active_net():
    provider, state = _build_two_net_provider()
    pin_rAAT = torch.tensor([0.0, 40.0, -1.0, 80.0, -1.0], dtype=torch.float32)
    pin_fAAT = torch.tensor([0.0, 50.0, -1.0, 90.0, -1.0], dtype=torch.float32)
    pin_rtran = torch.tensor([0.0, 2.0, 0.0, 4.0, 0.0], dtype=torch.float32)
    pin_ftran = torch.tensor([0.0, 3.0, 0.0, 5.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(5, dtype=torch.float32)

    def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    pin_rAAT, pin_fAAT, _pin_rtran, _pin_ftran = provider.propagate_net_aat_level(
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
    assert float(pin_rAAT[4].item()) == -1.0
    assert float(pin_fAAT[4].item()) == -1.0
    active_view = provider._active_view({5})
    assert int(active_view["candidate_indices_cpu"].numel()) == 1
    assert int(active_view["sink_indices_cpu"].numel()) == 1
    assert active_view["static"]["sink_net_index"].tolist() == [0]
    assert active_view["sink_net_id_cpu"].tolist() == [5]
    loss = pin_rAAT[2] + pin_fAAT[2]
    loss.backward()
    assert state.bu_logits.grad is not None
    assert float(state.bu_logits.grad[0].abs().item()) > 0.0
    assert float(state.bu_logits.grad[1].abs().item()) == 0.0


def test_relaxed_buffer_provider_reports_dynamic_invocation_metadata():
    provider, _state = _build_provider()

    assert provider.has_dynamic_nets(torch.tensor([5], dtype=torch.long))
    assert not provider.has_dynamic_nets(torch.tensor([6], dtype=torch.long))
    assert provider.metadata["net_subgraph_invocation_timing"] == (
        "during_timing_propagation_net_aat"
    )
    assert provider.metadata["uses_current_propagated_driver_slew"] is True


def test_discrete_candidate_zero_state_gradient_matches_zero_to_one_direction():
    provider, state = _build_discrete_provider()
    driver_arrival = torch.tensor([0.0], dtype=torch.float32)
    driver_slew = torch.tensor([0.1], dtype=torch.float32)

    zero_result = provider._run_lane(
        driver_arrival=driver_arrival,
        driver_slew=driver_slew,
        active_net_ids={5},
    )
    zero_loss = zero_result["sink_arrival"].sum()
    grad, = torch.autograd.grad(zero_loss, state.activation_param)
    with torch.no_grad():
        state.activation_param.fill_(1.0)
    one_result = provider._run_lane(
        driver_arrival=driver_arrival,
        driver_slew=driver_slew,
        active_net_ids={5},
    )
    one_loss = one_result["sink_arrival"].sum()

    predicted_improvement = -grad[0]
    discrete_improvement = zero_loss.detach() - one_loss.detach()
    assert torch.isfinite(predicted_improvement)
    assert torch.isfinite(discrete_improvement)
    assert float(predicted_improvement) > 0.0
    assert float(discrete_improvement) > 0.0
    assert float(predicted_improvement * discrete_improvement) > 0.0


def test_discrete_candidate_active_view_preserves_global_gradient_and_lut_row_identity():
    provider, state = _build_discrete_provider(two_nets=True)
    active_view = provider._active_view({5})
    assert active_view["candidate_indices_cpu"].tolist() == [0]
    per_size = provider._per_size_tensors(
        driver_arrival=torch.tensor([0.0]),
        active_view=active_view,
    )
    torch.testing.assert_close(per_size["per_size_input_cap"], torch.tensor([[0.2, 0.2]]))
    torch.testing.assert_close(per_size["per_size_delay"], torch.tensor([[0.1, 0.1]]))

    result = provider._run_lane(
        driver_arrival=torch.tensor([0.0]),
        driver_slew=torch.tensor([0.1]),
        active_net_ids={5},
    )
    grad, = torch.autograd.grad(result["sink_arrival"].sum(), state.activation_param)
    assert float(grad[0].abs()) > 0.0
    assert float(grad[1].abs()) == 0.0


def test_native_provider_keeps_static_net_subgraph_payload_on_cpu():
    provider, _state = _build_provider()
    provider.forward_backend = "native_explicit_autograd"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    driver_arrival = torch.tensor([1.0], dtype=torch.float32, device=device)
    driver_slew = torch.tensor([0.1], dtype=torch.float32, device=device)

    kwargs = provider._forward_kwargs(
        driver_arrival=driver_arrival,
        driver_slew=driver_slew,
    )

    assert kwargs["edge_resistance"].device.type == "cpu"
    assert kwargs["node_capacitance"].device.type == "cpu"
    assert kwargs["edge_capacitance"].device.type == "cpu"
    assert kwargs["driver_arrival"].device.type == "cpu"
    assert kwargs["driver_slew"].device.type == "cpu"
