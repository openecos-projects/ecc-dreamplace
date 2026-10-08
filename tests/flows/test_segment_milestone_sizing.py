"""Segment milestone sizing frames, refreshed gradients and rebootstrap."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.PlaceObj import PlaceObj
from dreamplace.ops.buffer_insertion.discrete_virtual_scheduler import (
    schedule_net_gradient_actions,
)


def _milestone_params(**overrides):
    values = {
        "joint_segment_sizing_enabled": 1,
        "timing_objective_lane": "timing_only",
        "timing_wns_coeff": 0.01,
        "timing_tns_coeff": 0.0001,
        "cell_padding_x": 0.0,
        "routability_opt_flag": False,
        "continuous_size_dynamics_mode": "none",
        "discrete_gradient_topk_lambda_update_mode": "fixed_gradient_price",
        "discrete_gradient_topk_lambda_update_scale": 1.0,
        "discrete_gradient_topk_lambda_min": 0.0,
        "discrete_gradient_topk_lambda_max": 1.0e6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_milestone_sizing_uses_gradient_captured_before_optimizer_evaluations():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    size_logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    size_logits.grad = torch.tensor([-3.0, 2.0])
    placer.data_collections = SimpleNamespace(
        size_logits=size_logits,
        inst_cell_id=torch.tensor([0, 0], dtype=torch.long),
    )
    z_param = torch.nn.Parameter(torch.zeros(1))
    state = SimpleNamespace(z_param=z_param)
    milestones = SimpleNamespace(frozen_topology_generation=4)
    placer.joint_coordinator = SimpleNamespace(
        uses_overflow_milestone_actions=True,
        buffering_lane=SimpleNamespace(_current_state=lambda: state),
        segment_milestones=milestones,
    )
    params = _milestone_params()
    model = SimpleNamespace(size_density_area_weight=0.1)
    pos = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    event = {"event_id": 7, "iteration": 20, "milestone": 0.2}

    frame = placer._freeze_segment_milestone_sizing_frame(
        params=params,
        model=model,
        pos=pos,
        event=event,
    )
    captured_grad = frame["grad"].clone()
    size_logits.grad.copy_(torch.tensor([99.0, 99.0]))

    observed = {}

    def select_discrete_gradient_topk(**kwargs):
        observed["grad"] = kwargs["size_grad"].detach().clone()
        return (
            kwargs["size_logits"].detach().clone(),
            torch.ones_like(kwargs["size_logits"]),
            {
                "applied_instance_ids": [],
                "applied_cell_ids": [],
                "applied_sizes": [],
                "num_changed_instances": 0,
            },
            {},
        )

    placer._select_discrete_gradient_topk = select_discrete_gradient_topk
    placer._sync_discrete_gradient_topk_runtime_cells = lambda _params, _summary: {
        "runtime_cell_state_synced": False,
        "runtime_sync_reason": "no_applied_cells",
    }

    placer._apply_segment_milestone_discrete_sizing(
        params=params,
        event=event,
        sizing_frame=frame,
    )

    assert torch.equal(captured_grad, torch.tensor([-3.0, 2.0]))
    assert torch.equal(observed["grad"], captured_grad)
    assert not torch.equal(observed["grad"], size_logits.grad)


def test_nesterov_milestone_uses_size_differentiable_surrogate_objective():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    size_logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    size_logits.grad = torch.tensor([99.0, 99.0])
    placer.data_collections = SimpleNamespace(
        size_logits=size_logits,
        inst_cell_id=torch.tensor([0, 0], dtype=torch.long),
    )
    state = SimpleNamespace(z_param=torch.nn.Parameter(torch.zeros(1)))
    advanced = []
    placer.joint_coordinator = SimpleNamespace(
        uses_overflow_milestone_actions=True,
        buffering_lane=SimpleNamespace(_current_state=lambda: state),
        segment_milestones=SimpleNamespace(frozen_topology_generation=4),
        advance_segment_milestone=lambda **kwargs: advanced.append(kwargs),
    )
    pos = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    event = {"event_id": 7, "iteration": 20, "milestone": 0.2}
    calls = []

    def timing_obj(observed_pos, surrogate_mode=None):
        calls.append((observed_pos.detach().clone(), surrogate_mode))
        timing_loss = torch.sum(size_logits * torch.tensor([-3.0, 2.0]))
        zero = timing_loss * 0.0
        return timing_loss, zero, zero, zero

    frame = placer._capture_segment_milestone_sizing_frame(
        params=_milestone_params(timing_obj_profile=True),
        model=SimpleNamespace(
            size_density_area_weight=0.1,
            timing_obj=timing_obj,
            _timing_loss=lambda wns, _tns, _ws, _ts: wns,
            _continuous_density_area_penalty=lambda: None,
            _timing_geometry_cache="stale",
        ),
        pos=pos,
        event=event,
        evaluate_objective=True,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0][0], pos.detach())
    assert calls[0][1] == "surrogate_only"
    assert torch.equal(frame["grad"], torch.tensor([-3.0, 2.0]))
    assert frame["summary"]["objective_mode"] == (
        "milestone_sizing_surrogate_plus_area"
    )
    assert frame["summary"]["gradient_positive_count"] == 1
    assert frame["summary"]["gradient_negative_count"] == 1
    runtime_profile = frame["summary"]["runtime_profile"]
    assert runtime_profile["enabled"] is True
    assert runtime_profile["timing_obj_ms"] >= 0.0
    assert runtime_profile["autograd_ms"] >= 0.0
    assert advanced[0]["stage"] == "sizing_gradient_frozen"


def test_disabled_milestone_sizing_does_not_evaluate_surrogate_objective():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    size_logits = torch.nn.Parameter(torch.zeros(1))
    size_logits.grad = torch.ones(1)
    placer.data_collections = SimpleNamespace(
        size_logits=size_logits,
        inst_cell_id=torch.tensor([0], dtype=torch.long),
    )
    state = SimpleNamespace(z_param=torch.nn.Parameter(torch.zeros(1)))
    advanced = []
    placer.joint_coordinator = SimpleNamespace(
        uses_overflow_milestone_actions=True,
        buffering_lane=SimpleNamespace(_current_state=lambda: state),
        segment_milestones=SimpleNamespace(frozen_topology_generation=4),
        advance_segment_milestone=lambda **kwargs: advanced.append(kwargs),
    )

    frame = placer._capture_segment_milestone_sizing_frame(
        params=_milestone_params(joint_segment_sizing_enabled=0),
        model=SimpleNamespace(
            size_density_area_weight=0.1,
            timing_obj=lambda *_args, **_kwargs: pytest.fail(
                "disabled sizing must not evaluate timing_obj"
            ),
        ),
        pos=torch.nn.Parameter(torch.zeros(1)),
        event={"event_id": 1, "iteration": 5, "milestone": 0.3},
        evaluate_objective=True,
    )

    assert frame["grad"] is None
    assert frame["summary"]["status"] == "disabled"
    assert frame["summary"]["objective_mode"] == "disabled"
    assert size_logits.grad is None
    assert advanced[0]["stage"] == "sizing_gradient_frozen"


def test_pin2pin_net_weight_refresh_reuses_frozen_direct_joint_topology():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = SimpleNamespace(is_segment_direct_joint=True)
    placer.data_collections = SimpleNamespace()
    placer.op_collections = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            topology_frozen=True,
            rebuild_tree=lambda _pin_pos: pytest.fail(
                "frozen direct-joint topology must not be rebuilt"
            ),
        ),
        pin_pos_op=lambda _pos: pytest.fail(
            "frozen direct-joint topology must use the geometry-forward path"
        ),
    )
    observed = []
    placer._refresh_live_timing_topology = lambda pos: observed.append(pos)
    pos = torch.tensor([1.0])

    placer._prepare_legacy_net_weight_timing_topology(pos)

    assert observed == [pos]


def test_pin2pin_net_weight_refresh_rebuilds_unfrozen_topology():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = SimpleNamespace(is_segment_direct_joint=True)
    placer.data_collections = SimpleNamespace()
    pin_pos = torch.tensor([2.0])
    rebuilt = tuple(torch.tensor([index]) for index in range(6))
    observed = []

    def rebuild_tree(value):
        observed.append(value)
        return rebuilt

    placer.op_collections = SimpleNamespace(
        steiner_topo_op=SimpleNamespace(
            topology_frozen=False,
            rebuild_tree=rebuild_tree,
        ),
        pin_pos_op=lambda _pos: pin_pos,
    )

    placer._prepare_legacy_net_weight_timing_topology(torch.tensor([1.0]))

    assert observed == [pin_pos]
    assert placer.data_collections.net_flat_topo_sort is rebuilt[0]
    assert placer.data_collections.net_flat_topo_sort_start is rebuilt[1]
    assert placer.data_collections.pin_fa is rebuilt[2]
    assert placer.data_collections.flat_pin_to is rebuilt[3]
    assert placer.data_collections.flat_pin_to_start is rebuilt[4]
    assert placer.data_collections.flat_pin_from is rebuilt[5]


def test_milestone_runtime_refresh_updates_geometry_pin_and_local_density_view():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = SimpleNamespace(is_segment_direct_joint=True)
    virtual_density_view = SimpleNamespace(_density_ops={"stale": object()})
    model = SimpleNamespace(
        _timing_geometry_cache="stale",
        _virtual_cell_density_view=virtual_density_view,
    )
    placer.last_global_place_model = model

    node_size_x = torch.tensor([1.0, 1.0, 1.0])
    node_size_y = torch.tensor([1.0, 1.0, 1.0])
    node_areas = node_size_x * node_size_y
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0, 0], dtype=torch.long),
        inst_libcell_offset=torch.tensor([0, 0], dtype=torch.long),
        inst_main_id=torch.tensor([0, 0], dtype=torch.long),
        main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.long),
        flat_libcell_info=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 2.0, 0.0]]
        ),
        flat_libcell_width=torch.tensor([1.0, 2.0]),
        flat_libcell_height=torch.tensor([1.0, 1.0]),
        inst_size_init=torch.tensor([1.0, 1.0]),
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        node_areas=node_areas,
        pin2node_map=torch.tensor([0, 1], dtype=torch.long),
        pin_2_libpin_offset=torch.tensor([0, 0], dtype=torch.long),
        cell_id_2_libpin_id_start=torch.tensor([0, 1], dtype=torch.long),
        flat_lib_pin_offset_x=torch.tensor([0.2, 0.7]),
        flat_lib_pin_offset_y=torch.tensor([0.3, 0.4]),
        flat_lib_pin_cap=torch.tensor([0.1, 0.4]),
        flat_lib_pin_rcap=torch.tensor([0.1, 0.4]),
        flat_lib_pin_fcap=torch.tensor([0.1, 0.4]),
        cell_id_2_arc_id_start=torch.tensor([0, 1, 2], dtype=torch.long),
        flat_libarc_info=torch.tensor(
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
            ],
            dtype=torch.long,
        ),
        flat_inst_arcs_by_level=torch.tensor(
            [[0, 0, 0, 0, 1, 0]],
            dtype=torch.long,
        ),
        endpoints_constraint_arcs=torch.empty((0, 6), dtype=torch.long),
        pin_offset_x=torch.tensor([0.2, 0.2]),
        pin_offset_y=torch.tensor([0.3, 0.3]),
    )
    placer.data_collections = data
    timing_op = SimpleNamespace(
        flat_inst_arcs_by_level=data.flat_inst_arcs_by_level,
        endpoints_constraint_arcs=data.endpoints_constraint_arcs,
        _cell_aat_static_cache={"stale": object()},
    )
    placer.op_collections = SimpleNamespace(timing_propagation_op=timing_op)
    placer.placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_filler_nodes=1,
        regions=[],
        inst_cell_id=np.array([0, 0], dtype=np.int32),
        inst_libcell_offset=np.array([0, 0], dtype=np.int32),
        inst_size_init=np.array([1.0, 1.0], dtype=np.float32),
        node_size_x=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        node_size_y=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        pin_offset_x=np.array([0.2, 0.2], dtype=np.float32),
        pin_offset_y=np.array([0.3, 0.3], dtype=np.float32),
        flat_inst_arcs_by_level=np.array(
            [[0, 0, 0, 0, 1, 0]],
            dtype=np.int32,
        ),
        endpoints_constraint_arcs=np.empty((0, 6), dtype=np.int32),
        total_movable_node_area=2.0,
    )

    def local_density_bins():
        return torch.stack((data.node_areas[0], data.node_areas[1])).clone()

    density_before = local_density_bins()
    place_obj = PlaceObj.__new__(PlaceObj)
    pin_caps_before = place_obj._pin_caps_from_offsets(
        data.inst_libcell_offset,
        data,
    )[2]
    refresh = placer._sync_discrete_gradient_topk_runtime_cells(
        _milestone_params(),
        {
            "applied_instance_ids": [0],
            "applied_cell_ids": [1],
            "applied_sizes": [2.0],
        },
    )
    density_after = local_density_bins()
    pin_caps_after = place_obj._pin_caps_from_offsets(
        data.inst_libcell_offset,
        data,
    )[2]

    assert torch.equal(data.inst_cell_id, torch.tensor([1, 0]))
    assert torch.equal(data.inst_libcell_offset, torch.tensor([1, 0]))
    assert torch.equal(data.node_size_x, torch.tensor([2.0, 1.0, 1.0]))
    assert torch.equal(data.node_areas, torch.tensor([2.0, 1.0, 1.0]))
    assert torch.allclose(data.pin_offset_x, torch.tensor([0.7, 0.2]))
    assert torch.allclose(data.pin_offset_y, torch.tensor([0.4, 0.3]))
    assert torch.equal(density_before, torch.tensor([1.0, 1.0]))
    assert torch.equal(density_after, torch.tensor([2.0, 1.0]))
    assert torch.allclose(pin_caps_before, torch.tensor([0.1, 0.1]))
    assert torch.allclose(pin_caps_after, torch.tensor([0.4, 0.1]))
    assert torch.equal(
        data.flat_inst_arcs_by_level,
        torch.tensor([[0, 0, 1, 1, 1, 0]]),
    )
    assert np.array_equal(
        placer.placedb.flat_inst_arcs_by_level,
        np.array([[0, 0, 1, 1, 1, 0]], dtype=np.int32),
    )
    assert np.array_equal(placer.placedb.inst_cell_id, np.array([1, 0]))
    assert np.allclose(placer.placedb.node_size_x, np.array([2.0, 1.0, 1.0]))
    assert np.allclose(placer.placedb.pin_offset_x, np.array([0.7, 0.2]))
    assert refresh["area_delta_internal"] == 1.0
    assert refresh["density_visible_area_delta_internal"] == 1.0
    assert refresh["runtime_pin_geometry_synced"] == 1
    assert refresh["runtime_pin_cap_view"] == "dynamic_from_inst_libcell_offset"
    assert refresh["runtime_timing_arc_lut_sync"]["status"] == "ok"
    assert refresh["runtime_timing_arc_lut_sync"]["updated_row_count"] == 1
    assert timing_op._cell_aat_static_cache == {}
    assert refresh["runtime_cell_state_generation"] == 1
    assert refresh["timing_cache_generation"] == 1
    assert refresh["virtual_density_cache_invalidated"]
    assert model._timing_geometry_cache is None
    assert virtual_density_view._density_ops == {}


def test_runtime_refresh_commits_vt_and_leakage_state_atomically():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = None
    vt_logits = torch.nn.Parameter(
        torch.tensor([[-13.81551056, 0.0]], dtype=torch.float32)
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0], dtype=torch.long),
        flat_libcell_info=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]
        ),
        flat_libcell_leakage=torch.tensor([1.0, 4.0]),
        inst_vt_init=torch.tensor([[1.0, 0.0]]),
        inst_leakage_init=torch.tensor([1.0]),
        vt_logits=vt_logits,
        runtime_cell_state_generation=0,
    )
    data.get_vt_var = lambda: torch.softmax(data.vt_logits, dim=1)
    placer.data_collections = data
    placer.placedb = SimpleNamespace(
        num_movable_nodes=1,
        regions=[],
        inst_cell_id=np.array([0], dtype=np.int32),
        inst_vt_init=np.array([[1.0, 0.0]], dtype=np.float32),
        inst_leakage_init=np.array([1.0], dtype=np.float32),
    )
    placer._sync_runtime_timing_arc_lut_rows = lambda _inst_ids: {"status": "ok"}

    refresh = placer._sync_discrete_gradient_topk_runtime_cells(
        _milestone_params(),
        {
            "applied_instance_ids": [0],
            "applied_cell_ids": [1],
            "applied_sizes": [1.0],
            "applied_vts": [1],
        },
    )

    assert torch.equal(data.inst_cell_id, torch.tensor([1]))
    assert torch.equal(data.inst_vt_init, torch.tensor([[0.0, 1.0]]))
    assert torch.equal(data.inst_leakage_init, torch.tensor([4.0]))
    assert np.array_equal(placer.placedb.inst_cell_id, np.array([1]))
    assert np.array_equal(
        placer.placedb.inst_vt_init,
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    assert np.array_equal(
        placer.placedb.inst_leakage_init,
        np.array([4.0], dtype=np.float32),
    )
    assert refresh["legal_vt_max_abs_gap"] < 1.0e-4


def test_runtime_arc_sync_maps_smaller_target_arc_schema_by_semantic_identity():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0], dtype=torch.long),
        inst_cell_id=torch.tensor([1], dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor([0, 3, 5], dtype=torch.long),
        flat_libarc_info=torch.tensor(
            [
                [0, 1, 0, 0, -1, 0],
                [0, 1, 0, 1, -1, 0],
                [2, 1, 0, 2, 1, 0],
                [2, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, -1, 0],
            ],
            dtype=torch.long,
        ),
        flat_inst_arcs_by_level=torch.tensor(
            [
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 1, -1, 0],
                [0, 0, 0, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
        endpoints_constraint_arcs=torch.empty((0, 6), dtype=torch.long),
    )
    placer.data_collections = data
    placer.placedb = SimpleNamespace(
        flat_inst_arcs_by_level=data.flat_inst_arcs_by_level.detach().cpu().numpy(),
        endpoints_constraint_arcs=np.empty((0, 6), dtype=np.int32),
    )
    timing_op = SimpleNamespace(
        flat_inst_arcs_by_level=data.flat_inst_arcs_by_level,
        endpoints_constraint_arcs=data.endpoints_constraint_arcs,
        resolved_timing_propagation_device=torch.device("meta"),
        _cell_aat_static_cache={"stale": object()},
    )
    placer.op_collections = SimpleNamespace(timing_propagation_op=timing_op)

    summary = placer._sync_runtime_timing_arc_lut_rows([0])

    assert torch.equal(
        data.flat_inst_arcs_by_level,
        torch.tensor(
            [
                [0, 0, 1, 4, -1, 0],
                [0, 0, 1, 4, -1, 0],
                [0, 0, 1, 3, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    assert summary["status"] == "ok"
    assert summary["transition_count"] == 1
    assert summary["transitions"][0]["collapsed_source_arc_count"] == 1
    assert summary["tables"]["flat_inst_arcs_by_level"][
        "semantic_remap_row_count"
    ] == 2
    assert timing_op.flat_inst_arcs_by_level.device.type == "meta"
    assert timing_op.endpoints_constraint_arcs.device.type == "meta"
    assert timing_op._cell_aat_static_cache == {}


def test_runtime_arc_sync_rejects_target_arc_schema_expansion():
    with pytest.raises(RuntimeError, match="cannot represent the target Liberty arc schema"):
        NonLinearPlace._runtime_liberty_arc_transition_map(
            0,
            1,
            torch.tensor([0, 1, 3], dtype=torch.long),
            torch.tensor(
                [
                    [0, 1, 0, 0, -1, 0],
                    [0, 1, 1, 0, -1, 0],
                    [0, 1, 1, 1, -1, 0],
                ],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    ("placedb_overrides", "params_overrides", "message"),
    [
        ({"regions": [object()]}, {}, "does not support fence regions"),
        ({}, {"routability_opt_flag": True}, "does not support routability"),
    ],
)
def test_milestone_runtime_refresh_rejects_unsupported_derived_area_state(
    placedb_overrides,
    params_overrides,
    message,
):
    placer = NonLinearPlace.__new__(NonLinearPlace)
    placer.joint_coordinator = SimpleNamespace(is_segment_direct_joint=True)
    placer.data_collections = SimpleNamespace(
        inst_cell_id=torch.tensor([0], dtype=torch.long)
    )
    placedb_values = {
        "num_filler_nodes": 0,
        "regions": [],
    }
    placedb_values.update(placedb_overrides)
    placer.placedb = SimpleNamespace(**placedb_values)

    with pytest.raises(RuntimeError, match=message):
        placer._sync_discrete_gradient_topk_runtime_cells(
            _milestone_params(**params_overrides),
            {
                "applied_instance_ids": [0],
                "applied_cell_ids": [1],
                "applied_sizes": [2.0],
            },
        )


def test_refreshed_buffer_gradient_selects_post_sizing_best_segment():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    z_param = torch.nn.Parameter(torch.zeros(2))
    state = SimpleNamespace(z_param=z_param)
    placer.joint_coordinator = SimpleNamespace(
        segment_route_b_enabled=True,
        buffering_lane=SimpleNamespace(_current_state=lambda: state),
        segment_milestones=SimpleNamespace(frozen_topology_generation=7),
    )
    placer.data_collections = SimpleNamespace(
        size_logits=torch.nn.Parameter(torch.zeros(1)),
        inst_cell_id=torch.tensor([1], dtype=torch.long),
        runtime_cell_state_generation=1,
        buffering_timing_topology={"topology_generation": 7},
    )
    placer._refresh_live_timing_topology = lambda _pos: None
    pos = torch.nn.Parameter(torch.tensor([5.0]))

    def timing_obj(_pos):
        coefficients = torch.where(
            placer.data_collections.inst_cell_id[0] == 0,
            torch.tensor([-3.0, 1.0]),
            torch.tensor([1.0, -3.0]),
        )
        loss = torch.sum(z_param * coefficients)
        zero = loss * 0.0
        return loss, zero, zero, zero

    model = SimpleNamespace(
        timing_obj=timing_obj,
        _timing_loss=lambda wns, _tns, _ws, _ts: wns,
        _timing_geometry_cache="stale",
    )
    before = schedule_net_gradient_actions(
        segment_net_index=torch.tensor([0, 1]),
        net_ids=torch.tensor([10, 20]),
        segment_ids=torch.tensor([101, 202]),
        counts=torch.zeros(2),
        raw_grad=torch.tensor([-3.0, 1.0]),
        max_count=3,
    )
    refreshed = placer._compute_refreshed_segment_buffer_gradient(
        params=_milestone_params(),
        model=model,
        pos=pos,
        event={"event_id": 4, "iteration": 20},
        refresh_topology=False,
    )
    after = schedule_net_gradient_actions(
        segment_net_index=torch.tensor([0, 1]),
        net_ids=torch.tensor([10, 20]),
        segment_ids=torch.tensor([101, 202]),
        counts=torch.zeros(2),
        raw_grad=z_param.grad,
        max_count=3,
    )

    assert before.selected_segment_ids.tolist() == [101]
    assert torch.equal(z_param.grad, torch.tensor([1.0, -3.0]))
    assert after.selected_segment_ids.tolist() == [202]
    assert refreshed["frame"] == "post_placement_post_sizing_x_next"
    assert model._timing_geometry_cache is None


def test_completed_milestone_transaction_clears_discrete_gradients():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    size_logits = torch.nn.Parameter(torch.zeros(1))
    size_logits.grad = torch.ones(1)
    z_param = torch.nn.Parameter(torch.zeros(1))
    z_param.grad = -torch.ones(1)
    state = SimpleNamespace(z_param=z_param)

    class Coordinator:
        uses_overflow_milestone_actions = True
        buffering_lane = SimpleNamespace(_current_state=lambda: state)
        segment_milestones = SimpleNamespace(pending_event=None)

        @staticmethod
        def advance_segment_milestone(**_kwargs):
            return None

        @staticmethod
        def complete_segment_milestone(**_kwargs):
            return {"status": "consumed"}

    placer.joint_coordinator = Coordinator()
    placer.data_collections = SimpleNamespace(
        size_logits=size_logits,
        buffering_timing_topology={
            "topology_generation": 3,
            "frozen_topology_generation": 3,
        },
    )
    placer._apply_segment_milestone_discrete_sizing = lambda **_kwargs: {
        "status": "applied",
        "num_changed_instances": 1,
        "area_delta_internal": 1.0,
        "density_visible_area_delta_internal": 1.0,
        "runtime_ms": 1.0,
    }
    placer._refresh_live_timing_topology = (
        lambda _pos: placer.data_collections.buffering_timing_topology
    )
    placer._compute_refreshed_segment_buffer_gradient = lambda **_kwargs: {
        "status": "ready",
        "runtime_ms": 1.0,
    }

    placer._apply_segment_overflow_milestone_transaction(
        params=_milestone_params(),
        model=SimpleNamespace(),
        pos=torch.nn.Parameter(torch.zeros(1)),
        optimizer=None,
        event={"event_id": 1, "iteration": 4},
        sizing_frame={"grad": torch.ones(1), "summary": {}},
    )

    assert size_logits.grad is None
    assert z_param.grad is None


def test_rebootstrap_installs_pair_generation_before_nesterov_rebase():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    calls = []
    payload = {
        "generation_status": "timing_violating",
        "pin2pin_pair_count": 4,
        "update_count": 2,
        "wns": -0.5,
        "tns": -10.0,
        "pin2pin_endpoint_limit": 0,
        "pin2pin_path_pair_backend": "fixture",
        "pin2pin_backend": "cpu",
    }

    def rebuild(*_args, **_kwargs):
        calls.append("rebuild")
        return dict(payload), (-0.5, -10.0, 0.0, 0.0)

    def install(**_kwargs):
        calls.append("install")
        return {
            "generation_id": 3,
            "status": "timing_violating",
            "pair_count": 4,
        }

    placer._rebuild_pin2pin_pair_weights = rebuild
    placer._write_legacy_net_weight_artifact = lambda *_args: calls.append("write")
    placer.joint_coordinator = SimpleNamespace(
        uses_pin2pin_rebootstrap_schedule=True,
        install_pin2pin_generation=install,
    )
    optimizer = SimpleNamespace(
        rebase_objective_state=lambda **_kwargs: (
            calls.append("rebase")
            or {"status": "rebased"}
        )
    )

    result = placer._rebuild_and_install_segment_pin2pin(
        params=SimpleNamespace(
            timing_topology_enable_overflow_threshold=0.35,
            net_weighting_update_interval=15,
        ),
        placedb=SimpleNamespace(),
        model=SimpleNamespace(),
        pos=torch.nn.Parameter(torch.zeros(1)),
        optimizer=optimizer,
        iteration=8,
        overflow=0.29,
        reason="milestone_rebootstrap",
        force_clear_accumulation=True,
    )

    assert calls == ["rebuild", "install", "rebase", "write"]
    assert result["generation"]["generation_id"] == 3
    assert result["payload"]["nesterov_rebase"]["status"] == "rebased"


def test_milestone_sizing_micro_loop_refreshes_gradient_after_every_update():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    size_logits = torch.nn.Parameter(torch.zeros(1))
    placer.data_collections = SimpleNamespace(size_logits=size_logits)
    captures = []
    refreshes = []
    advances = []

    def capture(**_kwargs):
        round_index = len(captures)
        captures.append(round_index)
        return {
            "grad": torch.tensor([float(round_index + 1)]),
            "summary": {
                "gradient_digest": f"gradient-{round_index}",
                "state_digest": f"state-{round_index}",
            },
        }

    changed_counts = iter((1, 1, 0))

    def apply_sizing(**kwargs):
        changed = next(changed_counts)
        round_index = len(captures) - 1
        return {
            "status": "applied",
            "num_changed_instances": changed,
            "area_delta_internal": float(changed),
            "input_state_digest": kwargs["sizing_frame"]["summary"]["state_digest"],
            "output_state_digest": f"state-{round_index + 1}",
        }

    placer._capture_segment_milestone_sizing_frame = capture
    placer._apply_segment_milestone_discrete_sizing = apply_sizing
    placer._refresh_live_timing_topology = lambda _pos: refreshes.append(True)
    placer._compact_segment_milestone_sizing_summary = lambda value: dict(value)
    placer.joint_coordinator = SimpleNamespace(
        advance_segment_milestone=lambda **kwargs: advances.append(kwargs)
    )
    model = SimpleNamespace(_timing_geometry_cache="stale")

    summary = placer._run_segment_milestone_sizing_micro_loop(
        params=_milestone_params(joint_segment_sizing_rounds=5),
        model=model,
        pos=torch.nn.Parameter(torch.zeros(1)),
        event={"event_id": 2, "iteration": 9, "milestone": 0.30},
    )

    assert captures == [0, 1, 2]
    assert len(refreshes) == 2
    assert summary["rounds_requested"] == 5
    assert summary["rounds_completed"] == 3
    assert summary["terminal_reason"] == "stationary"
    assert [row["gradient_digest"] for row in summary["rounds"]] == [
        "gradient-0",
        "gradient-1",
        "gradient-2",
    ]
    assert advances[0]["stage"] == "sizing_micro_loop_completed"
