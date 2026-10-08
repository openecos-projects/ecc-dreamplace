"""Sizing/VT projection frame and file round-trip at the flow boundary."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.gate_projection.gate_projection import GateProjectionOp


def projection_artifact_fixture():
    size = torch.tensor([1.9, 3.0])
    vt = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    data = SimpleNamespace(
        device=torch.device("cpu"),
        inst_main_id=torch.tensor([0, 1]),
        inst_cell_id=torch.tensor([0, 2]),
        inst_libcell_offset=torch.tensor([0, 0]),
        inst_is_sizeable=torch.tensor([True, False]),
        inst_size_lower=torch.tensor([1.0, 3.0]),
        inst_size_upper=torch.tensor([2.0, 3.0]),
        inst_vt_mask=torch.tensor([[True, True], [True, False]]),
        inst_size_init=torch.tensor([1.0, 3.0]),
        main_id_2_cell_id_start=torch.tensor([0, 2, 3]),
        flat_libcell_info=torch.tensor(
            [
                [10.0, 0.0, 1.0, 0.0],
                [11.0, 0.0, 2.0, 1.0],
                [20.0, 1.0, 3.0, 0.0],
            ]
        ),
        flat_libcell_leakage=torch.tensor([3.0, 1.0, 5.0]),
        flat_libcell_width=torch.tensor([1.0, 2.0, 3.0]),
        flat_libcell_height=torch.ones(3),
        node_size_x=torch.tensor([1.0, 3.0]),
        node_size_y=torch.ones(2),
        node_areas=torch.tensor([1.0, 3.0]),
        pin2node_map=torch.tensor([0, 0, 1, 1]),
        flat_node2pin_map=torch.tensor([0, 1, 2, 3]),
        flat_node2pin_start_map=torch.tensor([0, 2, 4]),
        pin_2_libpin_offset=torch.tensor([0, 1, 0, 1]),
        cell_id_2_libpin_id_start=torch.tensor([0, 2, 4, 6]),
        flat_lib_pin_offset_x=torch.tensor([0.1, 0.2, 0.4, 0.5, 0.7, 0.8]),
        flat_lib_pin_offset_y=torch.tensor([0.2, 0.3, 0.5, 0.6, 0.8, 0.9]),
        pin_offset_x=torch.tensor([0.1, 0.2, 0.7, 0.8]),
        pin_offset_y=torch.tensor([0.2, 0.3, 0.8, 0.9]),
        timing_model_generation=9,
        get_size_var=lambda: size,
        get_vt_var=lambda: vt,
    )
    engine = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(engine)
    engine.data_collections = data
    engine.pos = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(4))])
    engine.op_collections = SimpleNamespace(
        gate_projection_op=GateProjectionOp(data),
        steiner_topo_op=SimpleNamespace(topology_generation=7),
    )
    return engine


def test_projection_pipeline_writes_size_vt_and_pin_artifacts_without_installing(
    tmp_path,
):
    engine = projection_artifact_fixture()
    before_cells = engine.data_collections.inst_cell_id.clone()
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        design_name=lambda: "unit",
        placement_sizing_mode="size_only",
        cell_padding_x=0.0,
        real_size_transition_backend="reference",
        real_size_transition_artifact_policy="full",
    )
    paths = engine._write_projection_artifacts(params)
    with open(paths["csv"], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with open(paths["summary"], encoding="utf-8") as stream:
        summary = json.load(stream)

    assert len(rows) == 1
    assert {
        key: rows[0][key]
        for key in ("inst_id", "projected_cell_id", "projected_vt", "projected_size")
    } == {
        "inst_id": "0",
        "projected_cell_id": "1",
        "projected_vt": "1",
        "projected_size": "2.0",
    }
    assert summary["num_changed_cells"] == 1
    assert summary["metadata"]["projection_backend"] == "reference"
    assert torch.equal(
        engine.last_projection_frame.projected_cell_id_global, torch.tensor([1, 2])
    )
    assert torch.equal(
        engine.last_projection_frame.changed_pin_ids, torch.tensor([0, 1])
    )
    assert engine.last_projection_frame.topology_generation_before == 7
    assert engine.last_projection_frame.timing_model_generation_before == 9
    assert torch.equal(engine.data_collections.inst_cell_id, before_cells)


@pytest.mark.parametrize("use_frame", [True, False])
def test_projected_pin_application_keeps_runtime_original_and_pydb_views_equal(tmp_path, use_frame):
    engine = projection_artifact_fixture()
    params = SimpleNamespace(
        result_dir=str(tmp_path), design_name=lambda: "unit",
        placement_sizing_mode="size_only", cell_padding_x=0.0,
        real_size_transition_backend="reference", real_size_transition_artifact_policy="full",
    )
    engine._write_projection_artifacts(params)
    data = engine.data_collections
    data.original_pin_offset_x = data.pin_offset_x.clone()
    data.original_pin_offset_y = data.pin_offset_y.clone()
    pydb = SimpleNamespace(pin_offset_x=data.pin_offset_x.numpy().copy(),
                           pin_offset_y=data.pin_offset_y.numpy().copy())
    cells_before = data.inst_cell_id.clone()
    frame = engine.last_projection_frame
    engine._apply_projected_pin_offset_consistency(
        params, pydb, changed_inst_ids=frame.changed_inst_ids,
        projection_frame=frame if use_frame else None,
    )
    expected = ([0.4, 0.5, 0.7, 0.8], [0.5, 0.6, 0.8, 0.9])
    for x, y in ((data.pin_offset_x, data.pin_offset_y),
                 (data.original_pin_offset_x, data.original_pin_offset_y),
                 (torch.from_numpy(pydb.pin_offset_x), torch.from_numpy(pydb.pin_offset_y))):
        torch.testing.assert_close(torch.stack((x, y)), torch.tensor(expected))
    assert torch.equal(data.inst_cell_id, cells_before)


def test_invalid_projection_frame_rejects_before_pin_mutation(tmp_path):
    engine = projection_artifact_fixture()
    params = SimpleNamespace(
        result_dir=str(tmp_path), design_name=lambda: "unit",
        placement_sizing_mode="size_only", cell_padding_x=0.0,
        real_size_transition_backend="reference", real_size_transition_artifact_policy="full",
    )
    engine._write_projection_artifacts(params)
    data = engine.data_collections
    before = torch.stack((data.pin_offset_x, data.pin_offset_y)).clone()
    frame = replace(engine.last_projection_frame, changed_pin_ids=torch.tensor([-1, 1]))
    with pytest.raises(RuntimeError, match="invalid changed pin id"):
        engine._apply_projected_pin_offset_consistency(params, None, projection_frame=frame)
    assert torch.equal(torch.stack((data.pin_offset_x, data.pin_offset_y)), before)


def test_projection_frame_maps_candidate_state_to_global_nodes_and_changed_pins():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0, 2, 0], dtype=torch.long),
        inst_libcell_offset=torch.tensor([0, 0, 0], dtype=torch.long),
        real_size=torch.nn.Parameter(torch.tensor([1.2, 3.0, 1.0])),
        size_logits=None,
        sizing_parameterization="real_size",
        vt_logits=torch.nn.Parameter(torch.zeros(3, 2)),
        inst_vt_mask=torch.ones(3, 2, dtype=torch.bool),
        inst_is_sizeable=torch.tensor([True, True, False]),
        inst_vt_init=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        inst_size_init=torch.tensor([1.0, 3.0, 1.0]),
        node_size_x=torch.tensor([1.0, 3.0, 1.0]),
        node_size_y=torch.ones(3),
        node_areas=torch.tensor([1.0, 3.0, 1.0]),
        original_node_size_x=torch.tensor([1.0, 3.0, 1.0]),
        original_node_size_y=torch.ones(3),
        flat_libcell_width=torch.tensor([1.0, 2.0, 3.0]),
        flat_libcell_height=torch.ones(3),
        flat_node2pin_map=torch.tensor([0, 1, 2], dtype=torch.long),
        flat_node2pin_start_map=torch.tensor([0, 2, 3, 3], dtype=torch.long),
        pin_2_libpin_offset=torch.tensor([0, 1, 0], dtype=torch.long),
        cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.long),
        flat_lib_pin_offset_x=torch.tensor([0.1, 0.2, 0.4, 0.5, 0.7, 0.8]),
        flat_lib_pin_offset_y=torch.tensor([0.2, 0.3, 0.5, 0.6, 0.8, 0.9]),
        pin_offset_x=torch.tensor([0.1, 0.2, 0.7]),
        pin_offset_y=torch.tensor([0.2, 0.3, 0.8]),
        original_pin_offset_x=torch.tensor([0.1, 0.2, 0.7]),
        original_pin_offset_y=torch.tensor([0.2, 0.3, 0.8]),
        timing_model_generation=7,
    )
    data.get_size_var = lambda: data.real_size
    data.get_vt_var = lambda: torch.softmax(data.vt_logits, dim=1)
    projection_result = SimpleNamespace(
        inst_ids=torch.tensor([0, 1], dtype=torch.long),
        has_legal_candidate=torch.tensor([True, True]),
        current_cell_id=torch.tensor([0, 2]),
        current_libcell_offset=torch.tensor([0, 0]),
        current_vt=torch.tensor([0, 0]),
        current_size=torch.tensor([1.0, 3.0]),
        projected_cell_id=torch.tensor([1, 2]),
        projected_libcell_offset=torch.tensor([1, 0]),
        projected_vt=torch.tensor([1, 0]),
        projected_size=torch.tensor([2.0, 3.0]),
    )
    projection_op = GateProjectionOp(data)
    placer.data_collections = data
    placer.op_collections = SimpleNamespace(
        gate_projection_op=projection_op,
        steiner_topo_op=SimpleNamespace(topology_generation=11),
    )
    params = SimpleNamespace(cell_padding_x=0.0)

    frame = placer._build_projection_frame(params, projection_result)

    assert torch.equal(frame.changed_candidate_mask, torch.tensor([True, False]))
    assert torch.equal(frame.changed_inst_ids, torch.tensor([0]))
    assert torch.equal(frame.changed_node_mask, torch.tensor([True, False, False]))
    assert torch.equal(frame.projected_cell_id_global, torch.tensor([1, 2, 0]))
    assert torch.equal(frame.projected_libcell_offset_global, torch.tensor([1, 0, 0]))
    assert torch.equal(frame.projected_size_global, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(frame.projected_node_size_x, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(frame.projected_node_area, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(frame.changed_pin_ids, torch.tensor([0, 1]))
    assert torch.allclose(frame.projected_pin_offset_x, torch.tensor([0.4, 0.5]))
    assert torch.allclose(frame.projected_pin_offset_y, torch.tensor([0.5, 0.6]))
    assert frame.topology_generation_before == 11
    assert frame.timing_model_generation_before == 7
    assert frame.state_digest.startswith("sha256:")
    tensor_contract = {item.name: item for item in frame.tensor_contract}
    assert tensor_contract["changed_candidate_mask"].shape == (2,)
    assert tensor_contract["changed_candidate_mask"].dtype == "torch.bool"
    assert tensor_contract["changed_candidate_mask"].device == "cpu"
    assert tensor_contract["changed_candidate_mask"].index_space == "candidate_row"
    assert tensor_contract["projected_size_global"].index_space == "global_node"
    assert tensor_contract["projected_pin_offset_x"].index_space == "changed_pin_row"

    no_change_result = SimpleNamespace(**vars(projection_result))
    no_change_result.current_size = torch.tensor([1.2, 3.0])
    no_change_result.projected_cell_id = no_change_result.current_cell_id.clone()
    no_change_result.projected_libcell_offset = (
        no_change_result.current_libcell_offset.clone()
    )
    no_change_result.projected_vt = no_change_result.current_vt.clone()
    no_change_result.projected_size = no_change_result.current_size.clone()
    no_change_frame = placer._build_projection_frame(params, no_change_result)
    assert no_change_frame.changed_inst_ids.numel() == 0
    assert no_change_frame.changed_pin_ids.numel() == 0

    duplicate_result = SimpleNamespace(**vars(projection_result))
    duplicate_result.inst_ids = torch.tensor([0, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="duplicate instance ids"):
        placer._build_projection_frame(params, duplicate_result)

    invalid_result = SimpleNamespace(**vars(projection_result))
    invalid_result.inst_ids = torch.tensor([0, 9], dtype=torch.long)
    with pytest.raises(ValueError, match="invalid instance id"):
        placer._build_projection_frame(params, invalid_result)
