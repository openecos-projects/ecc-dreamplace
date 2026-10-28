"""Routing feedback keeps the coordinate contract used by retained candidates."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows.timing_topology import (
    TimingTopologyContext,
    refresh_live_timing_topology,
)
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.buffer_insertion.candidate_geometry import (
    refresh_candidate_geometry,
    remap_candidate_state,
)
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo


@pytest.mark.parametrize("publisher", ["routing_feedback", "timing_refresh"])
def test_publication_preserves_selected_candidate_and_updates_dbu_geometry(publisher):
    topology = SteinerTopo(
        torch.arange(3, dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        deterministic_flag=True,
    )
    context = TimingTopologyContext(
        data_collections=SimpleNamespace(),
        op_collections=SimpleNamespace(steiner_topo_op=topology, pin_pos_op=lambda p: p),
        placedb=SimpleNamespace(dbu=1000, r_unit=0.653, c_unit=0.00018889),
        params=SimpleNamespace(scale_factor=0.005, shift_factor=(100.0, 200.0)),
        segment_direct_joint_enabled=False,
        record_stage=lambda *args: None,
        record_bytes=lambda *args: None,
        refresh_callback=None,
    )
    position = torch.tensor([0.0, 1.0, 2.0, 0.0, 2.0, 0.0])
    refresh_live_timing_topology(context, position)
    record = {
        "candidate_id": 9,
        "net_id": 0,
        "parent_node_id": 0,
        "child_node_ids": [1],
        "segment_split_ratio": 0.25,
    }
    old_state = SimpleNamespace(
        candidate_records=(record,),
        candidate_ids=torch.tensor([9]),
        activation_param=torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
    )

    placer = object.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    placer.pos = torch.nn.ParameterList([torch.nn.Parameter(position + 0.5)])
    placer.last_global_place_model = object()
    placer._timing_topology_context = lambda: context
    consumed = []

    def consume(model, pos, *, iteration):
        assert model is placer.last_global_place_model and pos is placer.pos[0]
        assert iteration == 7
        live = context.data_collections.buffering_timing_topology
        refreshed = refresh_candidate_geometry(old_state, live)
        new_state = SimpleNamespace(
            candidate_records=refreshed.candidate_records,
            candidate_ids=torch.tensor([0]),
            activation_param=torch.nn.Parameter(torch.zeros(1, 4)),
        )
        remap = remap_candidate_state(
            old_state, new_state, invalidated_keys=refreshed.invalidated_candidate_keys
        )
        consumed.append((refreshed, new_state, remap))

    placer.alternating_discrete = SimpleNamespace(
        placement_step_count=7, refresh_live_geometry=consume
    )
    if publisher == "routing_feedback":
        topology.rebuild_tree(placer.pos[0].detach())
        placer._publish_live_timing_topology(
            topology.newx, topology.newy, update_kind="routing_feedback"
        )
    else:
        placer._refresh_live_timing_topology(placer.pos[0].detach())
    # Every publication must update its consumer before a probe or final STA
    # can evaluate the current tree, including calls outside the GP loop.
    assert len(consumed) == 1
    refreshed, new_state, remap = consumed[0]

    assert refreshed.invalidated_candidate_keys == ()
    assert remap["candidate_state_remap_failed"] == []
    torch.testing.assert_close(new_state.activation_param, old_state.activation_param)
    candidate = refreshed.candidate_records[0]
    assert (candidate["x_dbu"], candidate["y_dbu"], candidate["x_um"], candidate["y_um"]) == (
        250,
        400,
        0.25,
        0.4,
    )
    assert context.data_collections.buffering_timing_topology_scale_factor == 0.005


def test_publication_refreshes_retained_s5b1_candidate_payload():
    placer = object.__new__(NonLinearPlace)
    placer.alternating_discrete = None
    placer.last_global_place_model = object()
    calls = []
    placer.inflation_s5b1 = SimpleNamespace(
        refresh_candidate_geometry_after_topology=lambda model, *, reason: calls.append(
            (model, reason)
        )
    )

    topology = SimpleNamespace(topology_generation=8)
    result = placer._after_live_timing_topology_published(topology)

    assert result is topology
    assert calls == [(placer.last_global_place_model, "timing_topology_refresh")]
