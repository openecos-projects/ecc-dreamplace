import time

import torch

import dreamplace.ops.timing_propagation.timing_propagation as timing_propagation_module
from dreamplace.ops.timing_propagation.dynamic_net_provider import DynamicNetProvider
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


class RecordingProvider(DynamicNetProvider):
    def __init__(self, handled_net_ids):
        self.handled_net_ids = {int(net_id) for net_id in handled_net_ids}
        self.calls = []

    def has_dynamic_nets(self, net_ids, *, level_id=None, level_view_epoch=None):
        return any(int(net_id) in self.handled_net_ids for net_id in net_ids.detach().cpu().tolist())

    def propagate_net_aat_level(
        self,
        *,
        net_ids,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
        base_pin_net_impulse_rise,
        base_pin_net_impulse_fall,
        static_calculate_net_aat_level,
        level_id=None,
        level_view_epoch=None,
    ):
        self.calls.append(
            {
                "net_ids": net_ids.detach().cpu().tolist(),
                "level_id": level_id,
                "level_view_epoch": level_view_epoch,
                "driver_rise_aat_at_call": float(pin_rAAT[1].detach().cpu().item()),
                "driver_rise_slew_at_call": float(pin_rtran[1].detach().cpu().item()),
            }
        )
        pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = static_calculate_net_aat_level(
            net_ids,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            base_pin_net_delay_rise,
            base_pin_net_delay_fall,
            base_pin_net_impulse_rise,
            base_pin_net_impulse_fall,
        )
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        if 1 in self.calls[-1]["net_ids"]:
            pin_rAAT[2] = pin_rAAT[1] + 7.0
            pin_fAAT[2] = pin_fAAT[1] + 8.0
            pin_rtran[2] = pin_rtran[1] + 0.7
            pin_ftran[2] = pin_ftran[1] + 0.8
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran


class CapOverlayProvider(RecordingProvider):
    def __init__(self, handled_net_ids, rise_cap, fall_cap):
        super().__init__(handled_net_ids)
        self.rise_cap = float(rise_cap)
        self.fall_cap = float(fall_cap)
        self.overlay_calls = 0

    def apply_dynamic_net_cap_overlay(
        self,
        *,
        pin_net_cap_rise,
        pin_net_cap_fall,
    ):
        self.overlay_calls += 1
        pin_net_cap_rise = pin_net_cap_rise.clone()
        pin_net_cap_fall = pin_net_cap_fall.clone()
        pin_net_cap_rise[1] = self.rise_cap
        pin_net_cap_fall[1] = self.fall_cap
        return pin_net_cap_rise, pin_net_cap_fall


class SnapshotDelayProvider(RecordingProvider):
    def __init__(self, handled_net_ids):
        super().__init__(handled_net_ids)
        self.rise_delay = None
        self.fall_delay = None

    def begin_critical_path_net_delay_snapshot(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        self.rise_delay = base_pin_net_delay_rise.detach().clone()
        self.fall_delay = base_pin_net_delay_fall.detach().clone()

    def consume_critical_path_net_delays(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        assert self.rise_delay is not None
        assert self.fall_delay is not None
        self.rise_delay[2] = 7.0
        self.fall_delay[2] = 8.0
        return self.rise_delay, self.fall_delay


def _build_minimal_forward_model():
    device = torch.device("cpu")
    dtype = torch.float32
    model = TimingPropagation.__new__(TimingPropagation)
    torch.nn.Module.__init__(model)
    model.production_fast_loop = True
    model.timing_propagation_profile = False
    model.timing_forward_count = 0
    model.device = device
    model.dtype = dtype
    model.num_pins = 3
    model.pin_net = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    model.inrdelays = torch.tensor([0.0], device=device, dtype=dtype)
    model.infdelays = torch.tensor([0.0], device=device, dtype=dtype)
    model.inrtrans = torch.tensor([1.0], device=device, dtype=dtype)
    model.inftrans = torch.tensor([1.0], device=device, dtype=dtype)
    model.start_points = torch.tensor([0], device=device, dtype=torch.long)
    model.end_points = torch.tensor([2], device=device, dtype=torch.long)
    model.endpoints_rRAT = torch.tensor([100.0], device=device, dtype=dtype)
    model.endpoints_fRAT = torch.tensor([100.0], device=device, dtype=dtype)
    model.clk_pin_rtran = torch.tensor([1.0], device=device, dtype=dtype)
    model.clk_pin_ftran = torch.tensor([1.0], device=device, dtype=dtype)
    model.flat_inst_arcs_by_level = torch.tensor(
        [[0, 1, 0, 0, 1]],
        device=device,
        dtype=torch.long,
    )
    model.flat_inst_arcs_by_level_start = torch.tensor(
        [0, 0, 1],
        device=device,
        dtype=torch.long,
    )
    model.last_critical_endpoint_pruning_stats = {}
    model.last_traversal_pruning_stats = {}

    model._begin_forward_runtime_state = lambda: None
    model._end_forward_runtime_state = lambda: None
    model._validate_runtime_indices = lambda: None
    model._resolve_surrogate_mode = lambda surrogate_mode: (False, False)
    model._build_device_contract = lambda tensors, scope: {"scope": scope}
    model._assert_device_contract = lambda contract: None
    model._empty_cell_aat_profile = lambda enabled=False: {"enabled": enabled}
    model._build_cell_aat_surrogate_support_cache = lambda *args, **kwargs: None
    model._record_cell_aat_profile_value = lambda *args, **kwargs: None
    model._format_surrogate_unsupported_top_keys = lambda *args, **kwargs: None
    model.update_critical_endpoint_pruning_state = lambda iteration: None
    model._refresh_traversal_pruning_state = lambda iteration: None
    model._try_run_cpu_cuda_parity = lambda *args, **kwargs: None
    model._build_profile_payload = lambda **kwargs: kwargs
    model._sync_profile_clock = lambda device=None: time.perf_counter()
    model._build_traversal_pruning_working_view = lambda device: (
        model.flat_inst_arcs_by_level,
        model.flat_inst_arcs_by_level_start,
        torch.arange(1, device=device, dtype=torch.long),
        False,
    )
    model._captured_store_forward_state = None
    model._store_forward_state = lambda **kwargs: setattr(
        model,
        "_captured_store_forward_state",
        kwargs,
    )

    def calculate_clk2q_aat(pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args, **kwargs):
        return torch.empty(0, device=device, dtype=torch.long), pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def calculate_net_aat_level(cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def calculate_cell_aat_level(inst_arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args, **kwargs):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        pin_rAAT[1] = 20.0
        pin_fAAT[1] = 25.0
        pin_rtran[1] = 2.0
        pin_ftran[1] = 2.5
        delays = torch.zeros(inst_arcs.shape[0], device=device, dtype=dtype)
        return (
            torch.tensor([1], device=device, dtype=torch.long),
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            delays,
            delays,
            delays,
            delays,
        )

    def calculate_setup_rat(pin_rRAT, pin_fRAT, *args):
        return pin_rRAT, pin_fRAT

    def calculate_net_rat_level(cur_endpoint, pin_rRAT, pin_fRAT, *args):
        return pin_rRAT, pin_fRAT

    def calculate_cell_rat_level(*args, **kwargs):
        raise AssertionError("provider smoke should use production fast-loop RAT skip")

    model.calculate_clk2q_aat = calculate_clk2q_aat
    model.calculate_net_aat_level = calculate_net_aat_level
    model.calculate_cell_aat_level = calculate_cell_aat_level
    model.calculate_setup_rat = calculate_setup_rat
    model.calculate_net_rat_level = calculate_net_rat_level
    model.calculate_cell_rat_level = calculate_cell_rat_level
    return model


def _build_cap_sensitive_forward_model():
    model = _build_minimal_forward_model()
    device = model.device
    dtype = model.dtype

    def calculate_cell_aat_level(inst_arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, pin_net_cap_rise, pin_net_cap_fall, *args, **kwargs):
        pin_rAAT = pin_rAAT.clone()
        pin_fAAT = pin_fAAT.clone()
        pin_rtran = pin_rtran.clone()
        pin_ftran = pin_ftran.clone()
        pin_rAAT[1] = pin_net_cap_rise[1] * 10.0
        pin_fAAT[1] = pin_net_cap_fall[1] * 10.0
        pin_rtran[1] = 2.0
        pin_ftran[1] = 2.5
        delays = torch.zeros(inst_arcs.shape[0], device=device, dtype=dtype)
        return (
            torch.tensor([1], device=device, dtype=torch.long),
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            delays,
            delays,
            delays,
            delays,
        )

    model.calculate_cell_aat_level = calculate_cell_aat_level
    return model


def test_dynamic_net_provider_protocol_requires_implementation():
    provider = DynamicNetProvider()
    net_ids = torch.tensor([1], dtype=torch.long)

    try:
        provider.has_dynamic_nets(net_ids)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("DynamicNetProvider.has_dynamic_nets must be implemented")


def test_forward_routes_net_aat_through_dynamic_provider():
    model = _build_minimal_forward_model()
    provider = RecordingProvider({1})
    zeros = torch.zeros(3, dtype=torch.float32)

    model(
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        dynamic_net_provider=provider,
    )

    captured = model._captured_store_forward_state
    assert provider.calls
    assert provider.calls[-1]["net_ids"] == [1]
    assert provider.calls[-1]["driver_rise_aat_at_call"] == 20.0
    assert provider.calls[-1]["driver_rise_slew_at_call"] == 2.0
    assert float(captured["pin_rAAT"][2].item()) == 27.0
    torch.testing.assert_close(captured["pin_rtran"][2], torch.tensor(2.7))


def test_dynamic_provider_cap_overlay_is_visible_before_cell_arc_eval():
    model = _build_cap_sensitive_forward_model()
    provider = CapOverlayProvider({1}, rise_cap=3.0, fall_cap=4.0)
    zeros = torch.zeros(3, dtype=torch.float32)

    model(
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        dynamic_net_provider=provider,
    )

    captured = model._captured_store_forward_state
    assert provider.overlay_calls == 1
    assert float(captured["pin_rAAT"][1].item()) == 30.0
    assert float(captured["pin_fAAT"][1].item()) == 40.0
    assert provider.calls[-1]["driver_rise_aat_at_call"] == 30.0


def test_critical_path_snapshot_consumes_dynamic_provider_net_delays():
    model = _build_minimal_forward_model()
    model._store_forward_state = TimingPropagation._store_forward_state.__get__(
        model,
        TimingPropagation,
    )
    model.timing_aggregation_mode = "hard"
    model.critical_path_topology_epoch = 0
    provider = SnapshotDelayProvider({1})
    zeros = torch.zeros(3, dtype=torch.float32)
    model.request_critical_path_snapshot()

    model(
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        dynamic_net_provider=provider,
    )

    snapshot = model._critical_path_snapshot
    torch.testing.assert_close(
        snapshot["pin_net_delay_rise"],
        torch.tensor([0.0, 0.0, 7.0]),
    )
    torch.testing.assert_close(
        snapshot["pin_net_delay_fall"],
        torch.tensor([0.0, 0.0, 8.0]),
    )


def test_required_time_consumes_dynamic_delay_without_path_snapshot():
    model = _build_minimal_forward_model()
    provider = SnapshotDelayProvider({1})
    zeros = torch.zeros(3, dtype=torch.float32)

    def calculate_net_rat_level(endpoints, rise, fall, delay_rise, delay_fall):
        # The fixture has one driver/sink; exercise the production RAT operator.
        model.net2driver_pin_map = torch.tensor([0, 1, 1], dtype=torch.long)
        model.timing_aggregation_mode = "hard"
        model.timing_aggregation_tau_ps = 1.0
        return TimingPropagation.calculate_net_rat_level(
            model, endpoints, rise, fall, delay_rise, delay_fall
        )

    model.calculate_net_rat_level = calculate_net_rat_level
    model(
        {"rise": zeros, "fall": zeros}, {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros}, dynamic_net_provider=provider,
    )
    state = model._captured_store_forward_state
    torch.testing.assert_close(
        torch.stack([state["pin_rRAT"][1], state["pin_fRAT"][1],
                     state["pin_rRAT"][2] - state["pin_rAAT"][2]]),
        torch.tensor([93.0, 92.0, 73.0]),
    )


def test_profile_uses_synchronized_clock_for_outer_runtime(monkeypatch):
    model = _build_minimal_forward_model()
    model.timing_propagation_profile = True
    model._empty_cell_aat_profile = lambda enabled=False: {
        "enabled": enabled,
        "top_levels": [],
        "num_levels": 0,
        "num_arcs": 0,
        "query_count": 0,
        "total_ms": 0.0,
    }
    clock = [0.0]

    def synchronized_clock(device=None):
        clock[0] += 0.001
        return clock[0]

    def fail_host_clock():
        raise AssertionError("profiled TimingPropagation must use the synchronized clock")

    model._sync_profile_clock = synchronized_clock
    monkeypatch.setattr(timing_propagation_module.time, "perf_counter", fail_host_clock)
    zeros = torch.zeros(3, dtype=torch.float32)

    model(
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        {"rise": zeros, "fall": zeros},
        dynamic_net_provider=RecordingProvider({1}),
    )

    profile = model.last_profile_payload
    assert profile["total_runtime_ms"] > 0.0
    assert profile["stage_runtime_ms"]["input_clone"] > 0.0
    assert profile["stage_runtime_ms"]["slack_finalize"] > 0.0


def test_dynamic_provider_is_mutually_exclusive_with_precomputed_overlay():
    model = _build_minimal_forward_model()
    provider = RecordingProvider({1})
    zeros = torch.zeros(3, dtype=torch.float32)
    dynamic_net_arc_inputs = {
        "pin_net_delays": {"rise": zeros, "fall": zeros},
        "pin_net_impulses": {"rise": zeros, "fall": zeros},
        "pin_net_caps": {"rise": zeros, "fall": zeros},
    }

    try:
        model(
            {"rise": zeros, "fall": zeros},
            {"rise": zeros, "fall": zeros},
            {"rise": zeros, "fall": zeros},
            dynamic_net_arc_inputs=dynamic_net_arc_inputs,
            dynamic_net_provider=provider,
        )
    except ValueError as exc:
        assert "dynamic_net_provider" in str(exc)
    else:
        raise AssertionError("dynamic_net_provider must be exclusive with precomputed overlays")
