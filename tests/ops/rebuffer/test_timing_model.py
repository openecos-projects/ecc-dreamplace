from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.ops.rebuffer.timing_model import LibertyBufferLut, RebufferTimingModel


def _view():
    # Two arcs, each with 2x2 fall/rise delay and transition tables.
    arc0 = [
        10.0,
        20.0,
        30.0,
        40.0,
        12.0,
        22.0,
        32.0,
        42.0,
        1.0,
        2.0,
        3.0,
        4.0,
        1.5,
        2.5,
        3.5,
        4.5,
    ]
    arc1 = [
        11.0,
        21.0,
        31.0,
        41.0,
        13.0,
        23.0,
        33.0,
        43.0,
        1.2,
        2.2,
        3.2,
        4.2,
        1.6,
        2.6,
        3.6,
        4.6,
    ]
    return {
        "names": ["BUF"],
        "cell_input_cap": [0.005],
        "cell_arc_start": [0, 2],
        "arc_dim": [2, 2, 2, 2],
        "arc_trans_start": [0, 2, 4],
        "arc_trans_axis": [0.01, 0.1, 0.01, 0.1],
        "arc_cap_start": [0, 2, 4],
        "arc_cap_axis": [0.001, 0.01, 0.001, 0.01],
        "arc_value_start": [0, 16, 32],
        "arc_values": arc0 + arc1,
    }


def test_liberty_buffer_lut_takes_worst_arc_and_rise_fall():
    lut = LibertyBufferLut.from_liberty_view(_view(), "BUF")

    assert lut.input_cap == pytest.approx(0.005)
    assert lut.slew_axis == pytest.approx((0.01, 0.1))
    assert lut.load_axis == pytest.approx((0.001, 0.01))
    assert lut.delay[0, 0, 0] == pytest.approx(13.0)
    assert lut.delay[0, 1, 1] == pytest.approx(43.0)
    assert lut.output_slew[0, 0, 0] == pytest.approx(1.6)
    assert lut.output_slew[0, 1, 1] == pytest.approx(4.6)


def test_liberty_buffer_lut_tensors_are_contiguous_and_sized():
    lut = LibertyBufferLut.from_liberty_view(_view(), "BUF")
    tables = lut.tensors(dtype=torch.float64, device="cpu")

    assert tables["input_cap_by_size"].shape == (1,)
    assert tables["input_slew_axis"].shape == (2,)
    assert tables["output_load_axis"].shape == (2,)
    assert tables["delay_lut"].shape == (1, 2, 2)
    assert tables["output_slew_lut"].shape == (1, 2, 2)
    assert all(value.is_contiguous() for value in tables.values())
    assert np.isfinite(tables["delay_lut"].numpy()).all()


def test_liberty_buffer_lut_rejects_unknown_master():
    with pytest.raises(ValueError, match="unknown buffer master"):
        LibertyBufferLut.from_liberty_view(_view(), "MISSING")


def test_forward_uses_subtree_loads_not_local_pin_caps():
    model = object.__new__(RebufferTimingModel)
    model.delays = {"rise": torch.tensor([1.0]), "fall": torch.tensor([2.0])}
    model.impulses = {"rise": torch.tensor([3.0]), "fall": torch.tensor([4.0])}
    model.loads = {"rise": torch.tensor([5.0]), "fall": torch.tensor([6.0])}
    model.pin_caps = {"rise": torch.tensor([0.5]), "fall": torch.tensor([0.6])}

    class _FakeTimingPropagation:
        def __init__(self):
            self.caps = None
            self.end_points = torch.tensor([0])

        def __call__(self, delays, impulses, caps, **kwargs):
            self.caps = caps
            return 0.0, 0.0, 0.0, None

        def get_pin_slack(self):
            return torch.tensor([1.0])

    fake = _FakeTimingPropagation()
    model.model = fake

    snapshot = model._forward()

    assert torch.equal(fake.caps["rise"], model.loads["rise"])
    assert torch.equal(fake.caps["fall"], model.loads["fall"])
    assert not torch.equal(fake.caps["rise"], model.pin_caps["rise"])
    assert snapshot.wns_ps == 0.0
    assert snapshot.endpoint_count == 1


def test_path_snapshot_forward_keeps_current_overlay_rc():
    owner = object.__new__(RebufferTimingModel)
    owner.delays = {"rise": torch.tensor([1.0]), "fall": torch.tensor([2.0])}
    owner.impulses = {"rise": torch.tensor([3.0]), "fall": torch.tensor([4.0])}
    owner.loads = {"rise": torch.tensor([5.0]), "fall": torch.tensor([6.0])}
    active = SimpleNamespace(
        delays={"rise": torch.tensor([11.0]), "fall": torch.tensor([12.0])},
        impulses={"rise": torch.tensor([13.0]), "fall": torch.tensor([14.0])},
        loads={"rise": torch.tensor([15.0]), "fall": torch.tensor([16.0])},
    )
    calls = []

    class Operator:
        end_points = torch.tensor([0])

        def __call__(self, delays, impulses, loads, **kwargs):
            calls.append((delays, impulses, loads))
            return -1.0, -1.0, -1.0, None

        def get_pin_slack(self):
            return torch.tensor([-1.0])

    owner.model = Operator()
    active.model = owner.model
    owner._overlay_runtime = SimpleNamespace(model=active)

    owner._forward()

    assert calls == [(active.delays, active.impulses, active.loads)]


def test_native_setup_paths_choose_worst_transition_and_keep_path_contract():
    class _Batch:
        path_offsets = torch.tensor([0, 3, 5], dtype=torch.int64)
        path_pins = torch.tensor([1, 2, 9, 4, 9], dtype=torch.int32)
        path_transitions = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
        path_arc_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
        endpoint_pins = torch.tensor([9, 9], dtype=torch.int32)
        endpoint_transitions = torch.tensor([0, 1], dtype=torch.int32)
        endpoint_slacks = torch.tensor([-5.0, -7.0])
        path_valid = torch.tensor([True, True])
        topology_epoch = 17

    class _Model:
        end_points = torch.tensor([9], dtype=torch.int32)
        critical_path_topology_epoch = 17
        pin_rAAT_live_snapshot = torch.arange(10, dtype=torch.float64)
        pin_fAAT_live_snapshot = torch.arange(10, dtype=torch.float64) + 10.0

        def __init__(self):
            self.requests = 0
            self.extractions = 0
            self.clears = 0

        def request_critical_path_snapshot(self):
            self.requests += 1

        def extract_setup_critical_paths(self, **_kwargs):
            self.extractions += 1
            return _Batch()

        def clear_critical_path_snapshot_request(self):
            self.clears += 1

        def get_pin_slack(self):
            return torch.zeros(10)

    model = object.__new__(RebufferTimingModel)
    model.model = _Model()
    model._native_setup_path_cache = None
    model._native_setup_path_stats = {
        "request_count": 0,
        "cache_hit_count": 0,
        "unsupported_count": 0,
        "failure_count": 0,
        "extraction_count": 0,
        "last_error_type": None,
        "last_extraction": None,
    }
    forward_calls = []

    def forward():
        forward_calls.append(True)
        return model._snapshot_from_outputs(0.0, 0.0, 0.0)

    model._forward = forward

    details = model.native_setup_path_details()

    # The two records share one endpoint pin; the worse rise/fall state wins.
    assert details == {
        9: {
            "path": [4, 9],
            "transitions": [1, 0],
            "arc_ids": [12],
            "endpoint_transition": 1,
            "endpoint_slack": -7.0,
            "valid": True,
            "topology_epoch": 17,
            "path_arrivals": [14.0, 9.0],
        }
    }
    assert len(details[9]["arc_ids"]) == len(details[9]["path"]) - 1
    assert forward_calls == [True]
    assert model.model.extractions == 1

    # The current-state cache is reused until a new forward invalidates it.
    assert model.native_setup_path_details() is details
    assert model.model.requests == 1
    model._snapshot_from_outputs(0.0, 0.0, 0.0)
    model.native_setup_path_details()
    assert model.model.requests == 2
    assert model.model.extractions == 2
    stats = model.native_setup_path_statistics()
    assert stats["request_count"] == 2
    assert stats["cache_hit_count"] == 1
    assert stats["extraction_count"] == 2

    # A persistent slot can change the topology without a full owner forward;
    # the epoch check must reject a stale cached path in that case.
    model.model.critical_path_topology_epoch = 18
    model.native_setup_path_details()
    assert model.model.extractions == 3


def test_native_setup_path_failure_clears_pending_request_for_fallback():
    class _Model:
        def __init__(self):
            self.clear_calls = 0

        def request_critical_path_snapshot(self):
            return None

        def extract_setup_critical_paths(self, **_kwargs):
            raise RuntimeError("native path unavailable")

        def clear_critical_path_snapshot_request(self):
            self.clear_calls += 1

    model = object.__new__(RebufferTimingModel)
    model.model = _Model()
    model._native_setup_path_cache = None
    model._native_setup_path_stats = {
        "request_count": 0,
        "cache_hit_count": 0,
        "unsupported_count": 0,
        "failure_count": 0,
        "extraction_count": 0,
        "last_error_type": None,
        "last_extraction": None,
    }
    model._forward = lambda: None

    assert model.native_setup_path_details() is None
    assert model.model.clear_calls == 1
    stats = model.native_setup_path_statistics()
    assert stats["failure_count"] == 1
    assert stats["last_error_type"] == "RuntimeError"


def test_native_setup_path_details_returns_none_when_native_api_is_unsupported():
    model = object.__new__(RebufferTimingModel)
    model.model = object()
    model._native_setup_path_cache = None
    model._native_setup_path_stats = {
        "request_count": 0,
        "cache_hit_count": 0,
        "unsupported_count": 0,
        "failure_count": 0,
        "extraction_count": 0,
        "last_error_type": None,
        "last_extraction": None,
    }

    assert model.native_setup_path_details() is None
    assert model.native_setup_path_statistics()["unsupported_count"] == 1


def test_empty_overlay_restores_live_timing_state_instead_of_only_cached_metrics():
    model = object.__new__(RebufferTimingModel)
    model.use_native_timing_graph = False
    model.use_native_timing_evaluation = False
    model.use_persistent_graph = False
    model.pydb = SimpleNamespace()
    model._base_operator = None
    model._overlay_runtime = None
    calls = []
    snapshot = SimpleNamespace(as_dict=lambda: {"wns_ps": -10.0, "tns_ps": -20.0})

    def forward():
        calls.append("full_timing_propagation")
        return snapshot

    model._forward = forward
    assert model.evaluate(model.pydb, SimpleNamespace(actions=[])) == {
        "status": "ok",
        "action_count": 0,
        "wns_ps": -10.0,
        "tns_ps": -20.0,
    }
    assert calls == ["full_timing_propagation"]


def test_native_resolver_keeps_the_explicit_dp_branch_at_a_junction():
    from dreamplace.ops.rebuffer import rebuffer_cpp

    state = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        3,
        {
            "edges": [
                {"parent": 0, "child": 1, "r": 1.0, "c": 0.1},
                {"parent": 1, "child": 2, "r": 1.0, "c": 0.1},
            ],
            "node_loads": {2: [2]},
            "coordinates": {0: [0.0, 0.0], 1: [50.0, 0.0], 2: [100.0, 0.0]},
        },
    )
    action = {
        "parent_node_id": 1,
        "child_node_id": 2,
        "x_dbu": 50,
        "y_dbu": 0,
        "downstream_pin_ids": [2],
    }
    candidate = rebuffer_cpp.resolve_overlay_split(state, action)

    assert (candidate["parent_node_id"], candidate["child_node_id"]) == (1, 2)
    assert 0.0 < candidate["split_ratio"] < 1.0
    assert "split_ratio" not in action
