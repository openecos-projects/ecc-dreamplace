"""Terminal lane replacement and publication at the segment-joint seam."""

from types import SimpleNamespace

import pytest
import torch

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.buffer_insertion import buffering_lane


def terminal_rebuild_fixture(monkeypatch, *, rebuild_advances=True):
    calls = []
    data = SimpleNamespace(buffer_segment_timing_topology_epoch=7)
    model = SimpleNamespace(
        _timing_geometry_cache="old",
        _relaxed_buffer_dynamic_net_provider="old",
        _relaxed_buffer_dynamic_net_provider_key="old",
        _virtual_cell_density_view="old",
    )

    class Lane:
        def __init__(self, config, placedb=None):
            self.config = config
            self.placedb = placedb
            self.state = SimpleNamespace(z_param=torch.nn.Parameter(torch.ones(2)))
            self.buffer_relaxed_timing_payload = {"generation": "fresh"}

        def _current_state(self):
            return self.state

        def prepare(self, *, model, params):
            calls.append("prepare")

        def attach_state(self, data_collections):
            calls.append("attach")
            data_collections.buffer_segment_count_state = self.state

        def register_model_parameters(self, model):
            calls.append("register")

        def summarize(self):
            return {"state_kind": "segment_count"}

    class Topology:
        topology_generation = 3
        frozen_topology_generation = 3

        def unfreeze_topology(self):
            calls.append("unfreeze")
            self.frozen_topology_generation = None

        def freeze_topology(self):
            calls.append("freeze")
            self.frozen_topology_generation = self.topology_generation
            return self.frozen_topology_generation

    topo = Topology()

    def refresh(pos):
        calls.append("refresh")
        if topo.frozen_topology_generation is None and rebuild_advances:
            topo.topology_generation += 1
        return {
            "topology_generation": topo.topology_generation,
            "frozen_topology_generation": topo.frozen_topology_generation,
        }

    def capture_anchors(model, pos, *, force):
        assert force
        assert model._timing_geometry_cache is None
        calls.append("anchors")
        return {"status": "created"}

    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.data_collections = data
    placer.op_collections = SimpleNamespace(
        steiner_topo_op=topo, virtual_cell_density_op="old"
    )
    old_lane = Lane(config=SimpleNamespace())
    placer.joint_coordinator = SimpleNamespace(
        buffering_lane=old_lane,
        summary={},
        _proximal_objective=SimpleNamespace(capture_anchors=capture_anchors),
    )
    placer._refresh_live_timing_topology = refresh
    monkeypatch.setattr(buffering_lane, "BufferingOptimizationLane", Lane)
    return placer, model, calls, old_lane


def test_terminal_rebuild_replaces_lane_before_refreshing_anchors(monkeypatch):
    placer, model, calls, old_lane = terminal_rebuild_fixture(monkeypatch)
    pos = torch.tensor([1.0])
    state, summary = placer._rebuild_terminal_segment_route_b_state(
        params=SimpleNamespace(), model=model, pos=pos
    )

    assert calls == [
        "unfreeze", "refresh", "freeze", "refresh",
        "prepare", "attach", "register", "anchors",
    ]
    assert torch.equal(old_lane.state.z_param, torch.ones(2))
    assert torch.equal(state.z_param, torch.zeros(2))
    assert placer.joint_coordinator.buffering_lane is placer.last_buffering_lane
    assert placer.buffer_optimization_state is state
    assert placer.data_collections.buffer_segment_count_state is state
    assert placer.buffer_relaxed_timing_payload == {"generation": "fresh"}
    assert summary["topology_generation_after"] == 4
    assert summary["frozen_topology_generation_after"] == 4
    assert model._relaxed_buffer_dynamic_net_provider is None
    assert model._virtual_cell_density_view is None
    assert placer.op_collections.virtual_cell_density_op is None
    assert torch.equal(pos, torch.tensor([1.0]))


def test_terminal_rebuild_failure_does_not_publish_a_replacement_lane(monkeypatch):
    placer, model, calls, old_lane = terminal_rebuild_fixture(
        monkeypatch, rebuild_advances=False
    )
    with pytest.raises(RuntimeError, match="did not rebuild the Steiner tree"):
        placer._rebuild_terminal_segment_route_b_state(
            params=SimpleNamespace(), model=model, pos=torch.tensor([1.0])
        )

    assert calls == ["unfreeze", "refresh"]
    assert placer.joint_coordinator.buffering_lane is old_lane
    assert "last_buffering_lane" not in placer.__dict__
    assert placer.joint_coordinator.summary == {}
