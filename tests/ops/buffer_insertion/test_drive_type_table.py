import json

import pytest
import torch

from dreamplace.ops.buffer_insertion.drive_type_table import (
    DRIVE_TYPE_TABLE_SCHEMA,
    EXPECTED_DRIVE_TYPE_MASTERS,
    load_drive_type_table,
    write_case_drive_type_table,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_typed_discrete_candidate_state,
)
from dreamplace.ops.buffer_insertion.projection import project_candidate_buffer_state
from dreamplace.ops.buffer_insertion.coordinate_backend import commit_coordinate_buffer_actions


def _buffer_library():
    return {
        index: {
            "cell_id": 100 + index,
            "master_name": name,
            "input_cap": 0.1 + index,
            "delay": 1.0 + index,
            "output_slew": 0.5 + index,
            "buffer_main_type_index": 4,
        }
        for index, name in enumerate(EXPECTED_DRIVE_TYPE_MASTERS)
    }


def _candidates():
    return [
        {
            "candidate_id": 10,
            "candidate_node_id": 20,
            "net_id": 30,
            "buffer_main_type_index": 4,
            "x_dbu": 100,
            "y_dbu": 200,
            "net_name": "net30",
            "driver_pin_name": "drv/Y",
            "load_pin_name": "u0/A",
            "downstream_pin_names": ["u0/A"],
        },
        {
            "candidate_id": 11,
            "candidate_node_id": 21,
            "net_id": 30,
            "buffer_main_type_index": 4,
            "x_dbu": 300,
            "y_dbu": 200,
            "net_name": "net30",
            "driver_pin_name": "drv/Y",
            "load_pin_name": "u1/A",
            "downstream_pin_names": ["u1/A"],
        },
    ]


def _write_table(path):
    payload = {
        "schema": DRIVE_TYPE_TABLE_SCHEMA,
        "source": "unit-test",
        "units": {"drive_strength": "relative"},
        "buffers": [
            {
                "drive_type_id": index,
                "master_name": name,
                "drive_strength": strength,
                "cell_id": 100 + index,
            }
            for index, (name, strength) in enumerate(
                zip(EXPECTED_DRIVE_TYPE_MASTERS, (8, 12, 16, 20))
            )
        ],
    }
    path.write_text(json.dumps(payload))


def test_load_drive_type_table_resolves_legal_bsu_and_provenance(tmp_path):
    path = tmp_path / "drive_strength.json"
    _write_table(path)

    table = load_drive_type_table(path, buffer_library=_buffer_library())

    assert table.master_names == EXPECTED_DRIVE_TYPE_MASTERS
    assert table.bsu_indices == (0, 1, 2, 3)
    assert table.cell_ids == (100, 101, 102, 103)
    assert table.sha256
    assert table.to_metadata()["path"] == str(path)


def test_drive_type_table_rejects_wrong_master_order(tmp_path):
    path = tmp_path / "drive_strength.json"
    _write_table(path)
    payload = json.loads(path.read_text())
    payload["buffers"][0]["master_name"] = "BUFX20H7L"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="must be BUFX8H7L"):
        load_drive_type_table(path, buffer_library=_buffer_library())


def test_write_case_drive_type_table_is_explicit_and_reloadable(tmp_path):
    path = tmp_path / "generated.json"
    table = write_case_drive_type_table(path, buffer_library=_buffer_library())

    assert path.is_file()
    assert [entry.drive_strength for entry in table.entries] == [8.0, 12.0, 16.0, 20.0]


def test_typed_state_has_position_one_hot_and_type_specific_bsu(tmp_path):
    table_path = tmp_path / "drive_strength.json"
    _write_table(table_path)
    table = load_drive_type_table(table_path, buffer_library=_buffer_library())
    state = build_typed_discrete_candidate_state(
        _candidates(),
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        device="cpu",
    )

    assert state.activation_param.shape == (2, 4)
    assert state.candidate_bu().tolist() == [0.0, 0.0]
    assert state.bsu_index().tolist() == [0, 0]

    with torch.no_grad():
        state.activation_param[0, 2] = 1.0
    state.validate_binary_()
    assert state.candidate_bu().tolist() == [1.0, 0.0]
    assert state.bsu_index().tolist() == [2, 0]

    state.set_gradient_drive_type(3)
    typed_bu = state.candidate_bu()
    typed_bsu = state.bsu_index()
    assert typed_bu.tolist() == [1.0, 0.0]
    assert typed_bsu.tolist() == [2, 3]
    state.clear_gradient_drive_type()
    state.validate_binary_()


def test_typed_projection_and_commit_preserve_selected_master(tmp_path):
    table_path = tmp_path / "drive_strength.json"
    _write_table(table_path)
    table = load_drive_type_table(table_path, buffer_library=_buffer_library())
    state = build_typed_discrete_candidate_state(
        _candidates(),
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        device="cpu",
    )
    with torch.no_grad():
        state.activation_param[0, 1] = 1.0
    projection = project_candidate_buffer_state(
        state,
        legal_table={
            "legal_cell_ids": [100, 101, 102, 103],
            "legal_master_names": list(EXPECTED_DRIVE_TYPE_MASTERS),
            "legal_size_values": [0.0, 1.0, 2.0, 3.0],
        },
    )

    assert projection.validation["exact_active_action_match"]
    assert projection.selected_actions[0]["drive_type_id"] == 1
    assert projection.selected_actions[0]["buffer_master_name"] == "BUFX12H7L"
    assert projection.selected_actions[0]["drive_type_master_name"] == "BUFX12H7L"


def test_typed_projection_forwards_selected_master_to_coordinate_commit(tmp_path):
    table_path = tmp_path / "drive_strength.json"
    table = write_case_drive_type_table(table_path, buffer_library=_buffer_library())
    state = build_typed_discrete_candidate_state(
        _candidates(),
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        device="cpu",
    )
    with torch.no_grad():
        state.activation_param[0, 3] = 1.0
    projection = project_candidate_buffer_state(
        state,
        legal_table={
            "legal_cell_ids": [100, 101, 102, 103],
            "legal_master_names": list(EXPECTED_DRIVE_TYPE_MASTERS),
            "legal_size_values": [0.0, 1.0, 2.0, 3.0],
            "buffer_main_type_indices": [4, 4, 4, 4],
        },
    )

    class FakeBridge:
        def __init__(self):
            self.actions = []

        def query_diff_guided_batch_timing_metrics(self):
            if not self.actions:
                return {"status": "ok", "wns": -2.0, "tns": -20.0}
            return {"status": "ok", "wns": -2.0, "tns": -20.0}

        def run_coordinate_buffer_insert(self, action, config):
            self.actions.append((dict(action), dict(config)))
            return {
                "status": "ok",
                "inserted_buffer_count_delta": 1,
                "instance_count_delta": 1,
                "net_count_delta": 1,
            }

        def eval_tcl_string(self, _command):
            return ""

    bridge = FakeBridge()
    summary = commit_coordinate_buffer_actions(
        projection.selected_actions,
        openroad_bridge=bridge,
    )

    assert summary["status"] == "accepted"
    assert summary["accepted_action_count"] == 1
    assert bridge.actions[0][0]["drive_type_id"] == 3
    assert bridge.actions[0][0]["buffer_master_name"] == "BUFX20H7L"
    assert bridge.actions[0][0]["buffer_master_id"] == 103
