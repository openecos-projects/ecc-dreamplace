from types import SimpleNamespace

import torch
from torch import nn
import pytest

from dreamplace.ops.buffer_insertion.candidate_geometry import (
    candidate_key,
    refresh_candidate_geometry,
    remap_candidate_state,
)
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane


def _segment(candidate_id, ratio=0.5):
    return {
        "candidate_id": candidate_id,
        "net_id": 7,
        "parent_node_id": 2,
        "child_node_ids": [5],
        "segment_split_ratio": ratio,
        "x_dbu": 0,
        "y_dbu": 0,
    }


def test_segment_candidate_follows_live_endpoint_geometry():
    record = _segment(10, ratio=0.25)
    result = refresh_candidate_geometry(
        (record,),
        {
            "topology_generation": 4,
            "node_x_dbu": torch.tensor([0, 0, 100, 0, 0, 400]),
            "node_y_dbu": torch.tensor([0, 0, 100, 0, 0, 200]),
        },
    )

    refreshed = result.candidate_records[0]
    assert (refreshed["x_dbu"], refreshed["y_dbu"]) == (175, 125)
    assert refreshed["segment_length_dbu"] == 316
    assert refreshed["upstream_length_dbu"] == pytest.approx(79.05694150420949)
    assert candidate_key(refreshed) == candidate_key(record)
    assert result.candidate_generation == 4


def test_polyline_candidate_uses_normalized_arc_position():
    record = _segment(3, ratio=0.75)
    result = refresh_candidate_geometry(
        (record,),
        {
            "node_x_dbu": torch.tensor([0, 0, 100, 0, 0, 400]),
            "node_y_dbu": torch.tensor([0, 0, 100, 0, 0, 200]),
            "edge_polylines": {(7, 2, 5): ((0, 0), (100, 0), (100, 200))},
        },
    )
    refreshed = result.candidate_records[0]
    assert (refreshed["x_dbu"], refreshed["y_dbu"]) == (100, 125)


def test_remap_preserves_activation_and_old_id_by_stable_key():
    old_record = _segment(31, ratio=0.5)
    new_record = _segment(2, ratio=0.5)
    old_state = SimpleNamespace(
        candidate_records=(old_record,),
        candidate_ids=torch.tensor([31]),
        activation_param=nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
    )
    new_state = SimpleNamespace(
        candidate_records=(new_record,),
        candidate_ids=torch.tensor([2]),
        activation_param=nn.Parameter(torch.zeros(1, 4)),
    )

    summary = remap_candidate_state(old_state, new_state)

    assert summary["matched_count"] == 1
    assert torch.equal(new_state.candidate_ids, torch.tensor([31]))
    assert torch.equal(
        new_state.activation_param.detach(),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    assert new_state.candidate_records[0]["candidate_id"] == 31


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_remap_preserves_reordered_and_new_candidate_ids_on_state_device(device):
    old_state = SimpleNamespace(
        candidate_records=(_segment(31, 0.25), _segment(7, 0.5)),
        candidate_ids=torch.tensor([31, 7], device=device),
        activation_param=nn.Parameter(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], device=device),
            requires_grad=False,
        ),
        is_typed_candidate=True,
        timing_active_only=True,
    )
    new_state = SimpleNamespace(
        candidate_records=(_segment(1, 0.5), _segment(2, 0.75), _segment(3, 0.25)),
        candidate_ids=torch.tensor([1, 2, 3], device=device),
        activation_param=nn.Parameter(torch.zeros(3, 4, device=device)),
    )
    summary = remap_candidate_state(old_state, new_state)
    torch.testing.assert_close(
        (new_state.candidate_ids, new_state.activation_param),
        (
            torch.tensor([7, 32, 31], device=device),
            torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=device
            ),
        ),
    )
    assert [row["candidate_id"] for row in new_state.candidate_records] == [7, 32, 31]
    assert not new_state.activation_param.requires_grad
    assert new_state.timing_active_only
    assert summary == {
        "status": "ok",
        "matched_count": 2,
        "old_active_count": 1,
        "new_active_count": 1,
        "invalidated_count": 0,
        "candidate_state_remap_failed": [],
    }


@pytest.mark.parametrize("coordinates_available", [True, False])
def test_lane_refresh_rebuilds_payload_and_retains_virtual_activation(
    monkeypatch, coordinates_available
):
    old_record = _segment(31, ratio=0.5)
    new_record = _segment(2, ratio=0.5)
    old_state = SimpleNamespace(
        candidate_records=(old_record,),
        candidate_ids=torch.tensor([31]),
        candidate_net_id=torch.tensor([7]),
        activation_param=nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
    )
    new_state = SimpleNamespace(
        candidate_records=(new_record,),
        candidate_ids=torch.tensor([2]),
        candidate_net_id=torch.tensor([7]),
        activation_param=nn.Parameter(torch.zeros(1, 4)),
    )
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        active_strategy="discrete_net_gradient",
    )
    lane = BufferingOptimizationLane(
        config,
        buffer_optimization_state=old_state,
        buffer_relaxed_timing_payload={"metadata": {"state_kind": "candidate"}},
    )
    data = SimpleNamespace(
        buffering_timing_topology={
            "topology_generation": 8,
            "node_x_dbu": torch.tensor([0, 0, 100, 0, 0, 400]),
            "node_y_dbu": torch.tensor([0, 0, 100, 0, 0, 200]),
        },
        set_buffer_optimization_state=lambda state: setattr(
            data, "buffer_optimization_state", state
        ),
    )
    lane.data_collections = data
    if not coordinates_available:
        data.buffering_timing_topology.pop("node_x_dbu")
        data.buffering_timing_topology.pop("node_y_dbu")
    lane.model = SimpleNamespace(
        _relaxed_buffer_dynamic_net_provider=object(),
        _relaxed_buffer_dynamic_net_provider_key=object(),
        _timing_geometry_cache=object(),
        register_buffer_optimization_state=lambda state: None,
    )
    lane.params = SimpleNamespace()

    def fake_build(model, params, config):
        return SimpleNamespace(
            status="ok",
            reason=None,
            buffer_optimization_state=new_state,
            buffer_relaxed_timing_payload={"metadata": {"state_kind": "candidate"}},
            metadata={},
            candidate_count=1,
            net_count=1,
        )

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.buffering_state_builder.build_buffering_state_for_model",
        fake_build,
    )
    if not coordinates_available:
        with pytest.raises(RuntimeError, match="lost accepted virtual buffers"):
            lane.refresh_candidate_geometry(reason="routing_feedback")
        assert lane._current_state() is old_state
        torch.testing.assert_close(old_state.activation_param, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        return

    summary = lane.refresh_candidate_geometry(
        model=lane.model,
        params=lane.params,
        reason="unit_geometry_refresh",
    )

    assert summary["status"] == "ok"
    assert summary["geometry_generation"] == 1
    assert lane._current_state().candidate_ids.tolist() == [31]
    assert lane._current_state().activation_param.detach().tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert lane._current_state().candidate_records[0]["x_dbu"] == 250
    assert lane.model._relaxed_buffer_dynamic_net_provider is None


def test_active_refresh_compacts_search_and_reexpands_without_changing_timing(
    tmp_path, monkeypatch
):
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )
    from dreamplace.ops.buffer_insertion.drive_type_table import (
        EXPECTED_DRIVE_TYPE_MASTERS,
        write_case_drive_type_table,
    )
    from dreamplace.ops.net_subgraph_timing.dynamic_provider import RelaxedBufferDynamicNetProvider

    nets = [
        {
            "net_id": net_id,
            "driver_pin_id": parent,
            "net_pin_ids": [parent, child],
            "rc_tree": {
                "root_node_id": parent,
                "children_by_node": {parent: [child], child: []},
                "edge_rc": {(parent, child): {"r": 1.0, "c": 0.1}},
                "node_cap": {child: 1.0},
                "sink_nodes": [child],
            },
        }
        for net_id, parent, child in ((7, 0, 1), (9, 2, 3))
    ]
    candidates = [
        {
            "candidate_id": index,
            "net_id": net_id,
            "parent_node_id": parent,
            "child_node_ids": [child],
            "segment_split_ratio": ratio,
            "buffer_main_type_index": 7,
            "x_dbu": int(100 * ratio),
            "y_dbu": 0,
        }
        for index, (net_id, parent, child, ratio) in enumerate(
            ((7, 0, 1, 0.25), (7, 0, 1, 0.5), (7, 0, 1, 0.75), (9, 2, 3, 0.5))
        )
    ]
    library = {
        index: {
            "master_name": name,
            "cell_id": index,
            "input_cap": 0.2 + index * 0.1,
            "delay": 2.0 + index,
            "output_slew": 0.1 + index * 0.1,
        }
        for index, name in enumerate(EXPECTED_DRIVE_TYPE_MASTERS)
    }
    table = tmp_path / "types.json"
    write_case_drive_type_table(table, buffer_library=library)
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        candidate_drive_type_mode="four_strength_table",
        candidate_drive_strength_table=str(table),
    )
    data = SimpleNamespace(
        buffering_nets=nets,
        buffering_candidate_rows=candidates,
        buffering_buffer_library=library,
        pos=[nn.Parameter(torch.zeros(8))],
        buffering_timing_topology={
            "node_x_dbu": torch.tensor([0, 100, 0, 100]),
            "node_y_dbu": torch.zeros(4),
            "topology_generation": 1,
        },
    )
    model = SimpleNamespace(
        data_collections=data, register_buffer_optimization_state=lambda state: None
    )
    params = SimpleNamespace(buffering_mode="candidate")
    built = build_buffering_state_for_model(model, params, config)
    assert built.status == "ok"
    full = built.buffer_optimization_state
    full.timing_active_only = True
    with torch.no_grad():
        full.activation_param[1, 2] = 1.0
    lane = BufferingOptimizationLane(
        config,
        buffer_optimization_state=full,
        buffer_relaxed_timing_payload=built.buffer_relaxed_timing_payload,
    )
    lane.data_collections = data

    def timing(state, payload):
        provider = RelaxedBufferDynamicNetProvider(static_payload=payload, buffer_state=state)
        result = provider._run_lane(
            driver_arrival=torch.tensor([0.0]), driver_slew=torch.tensor([0.1]), active_net_ids={7}
        )
        (gradient,) = torch.autograd.grad(result["sink_arrival"].sum(), state.activation_param)
        return result, gradient

    before, full_grad = timing(full, built.buffer_relaxed_timing_payload)
    data.pydb = SimpleNamespace()

    def rebuild_nets(pydb, *, net_ids=None, **kwargs):
        assert pydb is data.pydb
        assert data.buffering_nets == ()
        selected = nets if net_ids is None else [net for net in nets if net["net_id"] in net_ids]
        return selected, {"status": "ok"}

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.real_design_adapter.build_buffering_nets_from_pydb",
        rebuild_nets,
    )
    lane.refresh_candidate_geometry(model=model, params=params, active_only=True)
    compact = lane._current_state()
    assert [candidate_key(row) for row in compact.candidate_records] == [
        candidate_key(row) for row in candidates[:3]
    ]
    assert compact.activation_param.detach().tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert compact.candidate_ids.tolist() == [0, 1, 2]
    after, compact_grad = timing(compact, lane.buffer_relaxed_timing_payload)
    for key in ("sink_arrival", "sink_slew"):
        torch.testing.assert_close(after[key], before[key])
    torch.testing.assert_close(after["lin"][0], before["lin"][0])
    torch.testing.assert_close(compact_grad, full_grad[:3])

    lane.refresh_candidate_geometry(model=model, params=params)
    expanded = lane._current_state()
    assert len(expanded.candidate_records) == 4
    active = torch.nonzero(expanded.activation_param.detach().sum(dim=1) > 0).flatten().tolist()
    assert [candidate_key(expanded.candidate_records[index]) for index in active] == [
        candidate_key(candidates[1])
    ]
    assert expanded.activation_param.detach()[active].tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_active_refresh_with_no_accepted_buffers_skips_search_builder(monkeypatch):
    state = SimpleNamespace(
        candidate_records=(_segment(1),), activation_param=nn.Parameter(torch.zeros(1, 4))
    )
    lane = BufferingOptimizationLane(
        SimpleNamespace(mode="candidate"), buffer_optimization_state=state
    )
    lane.data_collections = SimpleNamespace(buffering_timing_topology={})

    def unexpected_build(*args, **kwargs):
        raise AssertionError("ordinary GP must not build inactive search candidates")

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.buffering_state_builder.build_buffering_state_for_model",
        unexpected_build,
    )
    result = lane.refresh_candidate_geometry(
        model=SimpleNamespace(), params=SimpleNamespace(), active_only=True
    )
    assert result == {"status": "ok", "reason": "no_active_candidates", "candidate_count": 0}
