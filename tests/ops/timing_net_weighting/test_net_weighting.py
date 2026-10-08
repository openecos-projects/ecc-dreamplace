import numpy as np
import pytest
import torch
from types import SimpleNamespace

from dreamplace.ops.timing_net_weighting import net_weighting
from dreamplace.flows import timing_net_weighting as timing_net_weighting_flow
from dreamplace.NonLinearPlace import NonLinearPlace


class TimingPropagationStub:
    end_points = torch.tensor([2, 3], dtype=torch.int32)

    def __init__(self):
        self.selected_endpoints = None

    def get_critical_paths(self, endpoints, K):
        assert K == 1
        self.selected_endpoints = endpoints
        return [[0, endpoint] for endpoint in endpoints]


def _apply(nendpoints, monkeypatch):
    monkeypatch.setattr(net_weighting, "net_weighting_cpp", None)
    timing_prop = TimingPropagationStub()
    weights = {}
    pin_slack = np.asarray([0.0, 0.0, -3.0, -1.0], dtype=np.float32)
    net_weighting._apply_pin2pin(
        weights,
        np.arange(4, dtype=np.int64),
        pin_slack,
        torch.from_numpy(pin_slack),
        timing_prop,
        -3.0,
        nendpoints,
        10.0,
        50.0,
        0.2,
    )
    return timing_prop.selected_endpoints


def test_pin2pin_zero_endpoint_limit_selects_all_violating_endpoints(monkeypatch):
    assert _apply(0, monkeypatch) == [2, 3]


def test_pin2pin_positive_endpoint_limit_selects_worst_k_endpoints(monkeypatch):
    assert _apply(1, monkeypatch) == [2]


def test_pin2pin_pair_accumulation_resets_before_every_fourth_update():
    due = net_weighting.pin2pin_pair_accumulation_reset_due
    assert [due(count) for count in range(1, 9)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_clear_pin2pin_pair_accumulation_keeps_capacity_buffers():
    placedb = SimpleNamespace(
        pin2pin_net_weight={},
        length=[3],
        _pin2pin_pair_weights_tensor=torch.ones(3),
        _pin2pin_last_native_summary={"status": "updated"},
    )
    result = net_weighting.clear_pin2pin_pair_accumulation(placedb)
    assert result == {
        "previous_pair_count": 3,
        "cleared_pair_count": 3,
        "preallocated_buffers_retained": True,
    }
    assert placedb.pin2pin_net_weight == {}
    assert placedb.length[0] == 0
    assert placedb._pin2pin_pair_keys_tensor.shape == (0, 2)
    assert placedb._pin2pin_pair_weights_tensor.shape == (0,)
    assert placedb._pin2pin_last_native_summary == {}


def test_non_linear_place_resets_pair_state_only_before_fourth_update():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer._legacy_net_weight_gate_summary = {
        "pin2pin_pair_reset_count": 0,
        "pin2pin_pair_reset_update_counts": [],
    }
    placedb = SimpleNamespace(
        pin2pin_net_weight={(1, 2): 10.0},
        length=[1],
    )

    assert placer._maybe_reset_pin2pin_pair_accumulation(placedb, 3) is False
    assert placedb.pin2pin_net_weight == {(1, 2): 10.0}
    assert placer._maybe_reset_pin2pin_pair_accumulation(placedb, 4) is True
    assert placedb.pin2pin_net_weight == {}
    assert placer._legacy_net_weight_gate_summary == {
        "pin2pin_pair_reset_count": 1,
        "pin2pin_pair_reset_update_counts": [4],
        "last_pin2pin_pair_reset_before_update": 4,
        "pin2pin_pair_reset_reasons": ["periodic_accumulation_reset"],
    }


def test_native_pin2pin_updates_preallocated_tensor_buffers(monkeypatch):
    from dreamplace.ops.timing_net_weighting import timing_net_weighting_cpp

    monkeypatch.setattr(
        net_weighting,
        "net_weighting_cpp",
        timing_net_weighting_cpp,
    )
    path_batch = SimpleNamespace(
        path_offsets=torch.tensor([0, 3, 6], dtype=torch.int64),
        path_pins=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        endpoint_slacks=torch.tensor([-5.0, -2.5]),
        path_valid=torch.tensor([True, True]),
        selected_state_count=2,
        valid_path_count=2,
        invalid_path_count=0,
        extraction_runtime_ms=0.1,
    )

    class NativeTimingStub:
        last_critical_path_extraction_stats = {
            "failing_state_count": 2,
            "selected_state_count": 2,
            "state_domain": "setup_plus_recovery",
            "recovery_endpoint_count": 1,
        }

        def extract_pin2pin_critical_paths(self, **kwargs):
            assert kwargs["global_k"] == 1
            return path_batch

    placedb = SimpleNamespace(
        pin2node_map=np.asarray([10, 10, 20], dtype=np.int32),
        pin2pin_net_weight={},
        length=[0],
    )
    data = SimpleNamespace(
        pairs=torch.zeros(16, dtype=torch.int32),
        weights=torch.zeros(8, dtype=torch.float32),
    )
    summary = net_weighting._apply_pin2pin_native(
        placedb,
        data,
        NativeTimingStub(),
        wns=-10.0,
        nendpoints=1,
        min_weight=10.0,
        max_weight=50.0,
        accumulate_weight=0.2,
    )
    assert summary["backend"] == "cpp_openmp_transition_aware"
    assert placedb.length[0] == 1
    assert data.pairs[:2].tolist() == [1, 2]
    assert data.weights[0].item() == pytest.approx(10.1)
    assert placedb.pin2pin_net_weight[(1, 2)] == pytest.approx(10.1)
    assert summary["full_endpoint_wns"] == -10.0
    assert summary["pair_normalization_wns"] == -5.0
    assert summary["pair_normalization_policy"] == (
        "selected_max_endpoint_gba_wns"
    )


def test_update_pin2pin_accepts_legacy_npaths_alias(monkeypatch):
    captured = {}

    def fake_apply(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "updated"}

    monkeypatch.setattr(net_weighting, "_apply_pin2pin_native", fake_apply)
    summary = net_weighting.update_net_weights_from_tp(
        scheme="pin2pin",
        placedb=SimpleNamespace(),
        data_collections=SimpleNamespace(),
        timing_propagation=object(),
        momentum_decay=0.5,
        max_net_weight=50.0,
        ignore_net_degree=100,
        npaths=7,
        wns=-1.0,
        pin2pin_cfg={"min_weight": 10.0, "max_weight": 50.0, "accumulate": 0.2},
    )

    assert summary["status"] == "updated"
    assert captured["nendpoints"] == 7


def test_update_pin2pin_canonical_endpoint_limit_reaches_native_boundary(monkeypatch):
    captured = {}

    def fake_apply(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "updated"}

    monkeypatch.setattr(net_weighting, "_apply_pin2pin_native", fake_apply)
    net_weighting.update_net_weights_from_tp(
        scheme="pin2pin",
        placedb=SimpleNamespace(),
        data_collections=SimpleNamespace(),
        timing_propagation=object(),
        momentum_decay=0.5,
        max_net_weight=50.0,
        ignore_net_degree=100,
        nendpoints=32,
        wns=-1.0,
        pin2pin_cfg={"min_weight": 10.0, "max_weight": 50.0, "accumulate": 0.2},
    )

    assert captured["nendpoints"] == 32


def test_pin2pin_rebuild_context_preserves_timing_update_sequence(monkeypatch):
    events = []

    class TimingPropagation:
        def request_critical_path_snapshot(self):
            events.append("request")

        def clear_critical_path_snapshot_request(self):
            events.append("clear")

    timing_propagation = TimingPropagation()
    params = SimpleNamespace(
        net_weighting_scheme="pin2pin",
        momentum_decay_factor=0.5,
        ignore_net_degree=100,
        pin2pin_min_weight=10.0,
        pin2pin_max_weight=50.0,
        pin2pin_accumulate_weight=0.2,
    )
    placedb = SimpleNamespace(max_net_weight=50.0)
    data_collections = SimpleNamespace()
    op_collections = SimpleNamespace(timing_propagation_op=timing_propagation)
    model = SimpleNamespace(
        timing_obj=lambda _pos: (
            torch.tensor(-1.0),
            torch.tensor(-2.0),
            None,
            None,
        ),
        op_collections=SimpleNamespace(
            pin2pin_net_weight_op=lambda _pos: torch.tensor(1.0)
        ),
    )
    pos = torch.tensor([0.0, 1.0])

    def fake_update(**kwargs):
        events.append(("update", kwargs["nendpoints"]))
        return {"backend": "test_backend"}

    monkeypatch.setattr(
        timing_net_weighting_flow,
        "update_net_weights_from_tp",
        fake_update,
    )

    def record_timing(*_timing):
        events.append("record_timing")

    def record_update(*_args, **kwargs):
        events.append("record_update")
        assert kwargs["pin2pin_path_pair_backend"] == "test_backend"
        return {"pin2pin_pair_count": 1}

    context = timing_net_weighting_flow.TimingNetWeightingContext(
        params=params,
        placedb=placedb,
        model=model,
        pos=pos,
        data_collections=data_collections,
        op_collections=op_collections,
        iteration=4,
        gate_status={"enabled": True},
        update_reason="contract",
        force_clear_accumulation=False,
        profile_enabled=False,
        profile_clock=lambda: 0.0,
        prepare_timing_topology=lambda _pos: events.append("prepare"),
        record_timing_metrics=record_timing,
        timing_scalar=lambda value: float(value),
        clear_pair_accumulation=lambda *_args, **_kwargs: events.append("reset"),
        maybe_reset_pair_accumulation=lambda *_args: False,
        record_update=record_update,
        write_artifact=lambda _payload: events.append("artifact"),
        endpoint_budget=lambda: 3,
        next_update_count=lambda: 1,
    )

    payload, timing = timing_net_weighting_flow.rebuild_pin2pin_pair_weights(context)

    assert payload["generation_status"] == "timing_violating"
    assert timing[0].item() == -1.0
    assert events == [
        "prepare",
        "request",
        "record_timing",
        ("update", 3),
        "clear",
        "record_update",
    ]


def test_pin2pin_rebuild_missing_timing_op_skips_budget_resolution():
    params = SimpleNamespace(net_weighting_scheme="pin2pin")
    artifacts = []
    context = timing_net_weighting_flow.TimingNetWeightingContext(
        params=params,
        placedb=SimpleNamespace(),
        model=SimpleNamespace(),
        pos=torch.tensor([0.0]),
        data_collections=SimpleNamespace(),
        op_collections=SimpleNamespace(timing_propagation_op=None),
        iteration=1,
        gate_status={},
        update_reason="missing-op",
        force_clear_accumulation=False,
        profile_enabled=False,
        profile_clock=lambda: 0.0,
        prepare_timing_topology=lambda _pos: None,
        record_timing_metrics=lambda *_timing: None,
        timing_scalar=float,
        clear_pair_accumulation=lambda *_args, **_kwargs: None,
        maybe_reset_pair_accumulation=lambda *_args: False,
        record_update=lambda *_args, **_kwargs: None,
        write_artifact=artifacts.append,
        endpoint_budget=lambda: (_ for _ in ()).throw(
            AssertionError("endpoint budget must not be resolved")
        ),
        next_update_count=lambda: 1,
    )

    with pytest.raises(RuntimeError, match="timing propagation op"):
        timing_net_weighting_flow.rebuild_pin2pin_pair_weights(context)
    assert artifacts[0]["status"] == "skipped"


def test_legacy_npaths_percentage_matches_efficient_tdp_net_budget():
    params = SimpleNamespace(net_weighting_npaths=3.0)
    placedb = SimpleNamespace(num_nets=80257)

    assert NonLinearPlace._net_weighting_npaths_budget(params, placedb) == 2407


def test_explicit_endpoint_budget_overrides_legacy_net_percentage():
    params = SimpleNamespace(
        net_weighting_nendpoints=32,
        net_weighting_npaths=3.0,
        _net_weighting_nendpoints_explicit=True,
    )
    placedb = SimpleNamespace(num_nets=80257)

    assert NonLinearPlace._net_weighting_nendpoints(params, placedb) == 32
