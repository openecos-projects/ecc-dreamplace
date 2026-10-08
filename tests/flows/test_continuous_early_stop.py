"""Best-state contracts owned by flows.continuous_early_stop."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState


def _placer():
    return NonLinearPlace.__new__(NonLinearPlace)


def test_best_state_restore_restores_committed_oscillation_history():
    placer = _placer()
    placer.params = SimpleNamespace(placement_sizing_mode="size_only")
    placer.data_collections = SimpleNamespace(
        sizing_parameterization="logits",
        size_logits=torch.zeros(1),
        inst_cell_id=torch.tensor([2]),
        inst_libcell_offset=torch.tensor([0]),
    )
    placer.placedb = SimpleNamespace(
        inst_cell_id=np.array([2]),
        inst_libcell_offset=np.array([0]),
    )
    oscillator = CellOscillationState()
    oscillator.record(phase="timing", iteration=0,
                      instance_ids=[0], previous_ids=[1], target_ids=[2])
    placer._quad_oscillation_state = oscillator
    snapshot = placer._capture_continuous_early_stop_best_state()
    oscillator.record(phase="timing", iteration=1,
                      instance_ids=[0], previous_ids=[2], target_ids=[1])
    oscillator.record(phase="timing", iteration=2,
                      instance_ids=[0], previous_ids=[1], target_ids=[2])
    assert oscillator.blocked(phase="timing", iteration=3, count=1, device="cpu")[0]

    result = placer._restore_continuous_early_stop_best_state(
        {"enabled": True, "restore_best": True, "best_state": snapshot}
    )

    assert result["restored"]
    assert oscillator.snapshot() == snapshot["quad_oscillation_state"]
    assert not oscillator.blocked(phase="timing", iteration=3, count=1, device="cpu")[0]


def test_timing_sizing_waits_for_combined_timing_loss_before_tracking_best():
    params = SimpleNamespace(
        placement_sizing_mode="size_only",
        with_sta=True,
        differentiable_timing_obj=True,
    )
    metric = SimpleNamespace(objective=4.7e-5, combined_timing_loss=None)

    loss, source = _placer()._continuous_early_stop_loss_and_source(params, metric)

    assert loss is None
    assert source is None


def test_timing_sizing_tracks_combined_timing_loss_when_available():
    params = SimpleNamespace(
        placement_sizing_mode="size_only",
        with_sta=True,
        differentiable_timing_obj=True,
    )
    metric = SimpleNamespace(objective=4.7e-5, combined_timing_loss=22.97)

    loss, source = _placer()._continuous_early_stop_loss_and_source(params, metric)

    assert loss == 22.97
    assert source == "combined_timing_loss"


def test_best_loss_uses_the_evaluated_master_snapshot_not_the_next_update():
    placer = _placer()
    placer.params = SimpleNamespace(
        placement_sizing_mode="size_only",
        early_stop_restore_best=True,
    )
    placer.data_collections = SimpleNamespace(
        sizing_parameterization="logits",
        size_logits=torch.tensor([1.0]),
        vt_logits=torch.tensor([[12.0, -12.0]]),
        inst_cell_id=torch.tensor([0]),
        inst_libcell_offset=torch.tensor([0]),
    )
    placer.placedb = SimpleNamespace(
        inst_cell_id=np.array([0]),
        inst_libcell_offset=np.array([0]),
    )
    state = placer._build_continuous_early_stop_state(placer.params)
    state["candidate_state"] = placer._capture_continuous_early_stop_best_state()
    state["candidate_state"]["capture_phase"] = "pre_optimizer_step"

    with torch.no_grad():
        placer.data_collections.inst_cell_id[0] = 1
        placer.data_collections.vt_logits[0] = torch.tensor([-12.0, 12.0])
    placer.placedb.inst_cell_id[0] = 1
    assert not placer._update_continuous_early_stop_state(state, 1.0, iteration=4)

    assert state["best_loss"] == 1.0
    assert state["best_iteration"] == 4
    assert state["best_state"]["capture_phase"] == "pre_optimizer_step"
    assert state["best_state"]["inst_cell_id"].tolist() == [0]
    assert state["best_state"]["placedb_inst_cell_id"].tolist() == [0]
    torch.testing.assert_close(state["best_state"]["vt_logits"], torch.tensor([[12., -12.]]))


def test_non_timing_flow_keeps_objective_fallback():
    params = SimpleNamespace(
        placement_sizing_mode="place_only",
        with_sta=False,
        differentiable_timing_obj=False,
    )
    metric = SimpleNamespace(objective=3.5, combined_timing_loss=None)

    loss, source = _placer()._continuous_early_stop_loss_and_source(params, metric)

    assert loss == 3.5
    assert source == "objective"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_best_master_restore_rebuilds_geometry_seen_by_legalization(device):
    placer = _placer()
    placer.params = SimpleNamespace(cell_padding_x=0.0)
    tensor = lambda values, **kwargs: torch.tensor(values, device=device, **kwargs)
    placer.data_collections = SimpleNamespace(
        sizing_parameterization="logits",
        size_logits=tensor([1.0, 2.0]),
        inst_cell_id=tensor([0, 1]),
        inst_libcell_offset=tensor([0, 1]),
        inst_size_init=tensor([1.0, 2.0]),
        main_id_2_cell_id_start=tensor([0]),
        flat_libcell_info=tensor([[0., 0., 1.], [1., 0., 2.]]),
        flat_libcell_width=tensor([1296., 540.]),
        flat_libcell_height=tensor([270., 270.]),
        node_size_x=tensor([1296., 540.]),
        node_size_y=tensor([270., 270.]),
        node_areas=tensor([1296. * 270., 540. * 270.]),
        pin2node_map=tensor([0, 1]),
        pin_2_libpin_offset=tensor([0, 0]),
        cell_id_2_libpin_id_start=tensor([0, 1]),
        flat_lib_pin_offset_x=tensor([648., 270.]),
        flat_lib_pin_offset_y=tensor([135., 135.]),
        pin_offset_x=tensor([648., 270.]),
        pin_offset_y=tensor([135., 135.]),
    )
    data = placer.data_collections
    mirrored = ("inst_cell_id", "inst_libcell_offset", "inst_size_init",
                "node_size_x", "node_size_y", "pin_offset_x", "pin_offset_y")
    placer.placedb = SimpleNamespace(num_movable_nodes=2, **{
        key: getattr(data, key).cpu().numpy().copy() for key in mirrored
    })
    # Legalization operators retain these tensor objects when they are built.
    legalization_width = data.node_size_x
    legalization_height = data.node_size_y
    state = {"enabled": True, "restore_best": True, "best_iteration": 48,
             "best_state": placer._capture_continuous_early_stop_best_state()}
    # The last iteration selects a narrower cell, updating geometry correctly.
    placer._sync_discrete_gradient_topk_runtime_cells(
        placer.params, {"applied_instance_ids": [0], "applied_cell_ids": [1]})
    assert float(legalization_width[0]) == 540.
    placer.last_global_place_model = SimpleNamespace(_timing_geometry_cache=object())
    generation = data.runtime_cell_state_generation

    result = placer._restore_continuous_early_stop_best_state(state)

    assert result["sizing_restored"]
    assert data.inst_cell_id.tolist() == [0, 1]
    assert data.node_size_x is legalization_width
    assert data.node_size_y is legalization_height
    torch.testing.assert_close(legalization_width, tensor([1296., 540.]))
    torch.testing.assert_close(data.node_areas, tensor([1296. * 270., 540. * 270.]))
    torch.testing.assert_close(data.pin_offset_x, tensor([648., 270.]))
    torch.testing.assert_close(data.inst_size_init, tensor([1., 2.]))
    for key in mirrored:
        np.testing.assert_array_equal(getattr(placer.placedb, key), getattr(data, key).cpu().numpy())
    assert placer.placedb.total_movable_node_area == 1836. * 270.
    assert data.runtime_cell_state_generation == generation + 1
    assert placer.last_global_place_model._timing_geometry_cache is None


def test_best_master_restore_rebuilds_vt_and_leakage_state():
    placer = _placer()
    placer.params = SimpleNamespace(cell_padding_x=0.0, routability_opt_flag=False)
    placer.joint_coordinator = None
    placer.last_global_place_model = None
    placer._sync_runtime_timing_arc_lut_rows = lambda _inst_ids: {"status": "ok"}
    vt_logits = torch.nn.Parameter(torch.tensor([[0.0, -13.81551056]]))
    placer.data_collections = SimpleNamespace(
        sizing_parameterization="logits",
        size_logits=torch.nn.Parameter(torch.tensor([0.0])),
        vt_logits=vt_logits,
        inst_cell_id=torch.tensor([0]),
        inst_libcell_offset=torch.tensor([0]),
        flat_libcell_info=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]
        ),
        flat_libcell_leakage=torch.tensor([1.0, 4.0]),
        inst_vt_init=torch.tensor([[1.0, 0.0]]),
        inst_leakage_init=torch.tensor([1.0]),
        runtime_cell_state_generation=0,
    )
    data = placer.data_collections
    data.get_vt_var = lambda: torch.softmax(data.vt_logits, dim=1)
    placer.placedb = SimpleNamespace(
        num_movable_nodes=1,
        regions=[],
        inst_cell_id=np.array([0], dtype=np.int32),
        inst_libcell_offset=np.array([0], dtype=np.int32),
        inst_vt_init=np.array([[1.0, 0.0]], dtype=np.float32),
        inst_leakage_init=np.array([1.0], dtype=np.float32),
    )
    state = {
        "enabled": True,
        "restore_best": True,
        "best_iteration": 4,
        "best_state": placer._capture_continuous_early_stop_best_state(),
    }

    with torch.no_grad():
        vt_logits.copy_(torch.tensor([[-13.81551056, 0.0]]))
    placer._sync_discrete_gradient_topk_runtime_cells(
        placer.params,
        {
            "applied_instance_ids": [0],
            "applied_cell_ids": [1],
            "applied_vts": [1],
        },
    )

    result = placer._restore_continuous_early_stop_best_state(state)

    assert result["sizing_restored"]
    assert data.inst_cell_id.tolist() == [0]
    assert data.inst_libcell_offset.tolist() == [0]
    assert torch.argmax(data.get_vt_var(), dim=1).tolist() == [0]
    torch.testing.assert_close(data.inst_vt_init, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(data.inst_leakage_init, torch.tensor([1.0]))
    np.testing.assert_array_equal(placer.placedb.inst_cell_id, np.array([0]))
    np.testing.assert_array_equal(placer.placedb.inst_libcell_offset, np.array([0]))
    np.testing.assert_array_equal(
        placer.placedb.inst_vt_init,
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        placer.placedb.inst_leakage_init,
        np.array([1.0], dtype=np.float32),
    )


@pytest.mark.parametrize("owner", ["segment_joint", "timing_opt"])
def test_best_restore_preserves_milestone_owned_cells_and_geometry(owner):
    placer = _placer()
    placer.params = SimpleNamespace(timing_opt_enabled=owner == "timing_opt")
    placer._segment_direct_joint_enabled = lambda: owner == "segment_joint"
    placer.data_collections = SimpleNamespace(inst_cell_id=torch.tensor([1]),
                                             node_size_x=torch.tensor([540.]))
    placer.placedb = SimpleNamespace(inst_cell_id=np.array([1]), node_size_x=np.array([540.]))
    state = {"enabled": True, "restore_best": True, "best_state": {
        "inst_cell_id": torch.tensor([0]), "placedb_inst_cell_id": np.array([0])}}
    result = placer._restore_continuous_early_stop_best_state(state)
    assert not result["sizing_restored"]
    assert placer.data_collections.inst_cell_id.item() == 1
    assert placer.data_collections.node_size_x.item() == 540.
    assert placer.placedb.inst_cell_id[0] == 1
