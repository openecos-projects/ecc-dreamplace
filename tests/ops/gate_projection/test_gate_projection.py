from types import SimpleNamespace

import pytest
import torch

from dreamplace.ops.gate_projection.gate_projection import (
    GateProjectionOp,
    MainIdCandidateProvider,
    VectorizedMainIdCandidateProvider,
)


def _projection_op():
    data = SimpleNamespace(
        pin_offset_x=torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
        pin_offset_y=torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0]),
        flat_node2pin_map=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        flat_node2pin_start_map=torch.tensor([0, 2, 2, 5], dtype=torch.long),
        pin_2_libpin_offset=torch.tensor([0, 1, 0, 1, 2], dtype=torch.long),
        cell_id_2_libpin_id_start=torch.tensor([0, 3, 6, 9], dtype=torch.long),
        flat_lib_pin_offset_x=torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ),
        flat_lib_pin_offset_y=torch.tensor(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
        ),
    )
    return GateProjectionOp(data), data


def _candidate_data():
    return SimpleNamespace(
        device=torch.device("cpu"),
        inst_main_id=torch.tensor([0, 1, -1, 0, 1], dtype=torch.long),
        inst_cell_id=torch.tensor([0, 2, 3, 1, -1], dtype=torch.long),
        inst_libcell_offset=torch.tensor([0, 0, 0, 1, 0], dtype=torch.long),
        inst_is_sizeable=torch.tensor([True, True, False, True, True]),
        inst_size_lower=torch.tensor([1.0, 1.0, 1.0, 1.5, 1.0]),
        inst_size_upper=torch.tensor([2.0, 3.0, 4.0, 2.0, 3.0]),
        inst_vt_mask=torch.tensor(
            [[True, True], [True, False], [True, True], [True, True], [True, True]]
        ),
        main_id_2_cell_id_start=torch.tensor([0, 2, 3], dtype=torch.long),
        flat_libcell_info=torch.tensor(
            [
                [10.0, 0.0, 1.0, 0.0],
                [10.0, 0.0, 2.0, 1.0],
                [20.0, 0.0, 3.0, 0.0],
                [30.0, 0.0, 1.0, 0.0],
            ]
        ),
        flat_libcell_leakage=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )


def test_vectorized_candidate_provider_matches_reference_provider():
    data = _candidate_data()
    inst_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    reference = MainIdCandidateProvider().enumerate(data, inst_ids)
    vectorized = VectorizedMainIdCandidateProvider().enumerate(data, inst_ids)

    for field in (
        "inst_ids",
        "main_ids",
        "current_cell_ids",
        "current_offsets",
        "candidate_cell_ids",
        "candidate_libcell_offsets",
        "candidate_name_ids",
        "candidate_sizes",
        "candidate_vts",
        "candidate_leakages",
        "candidate_mask",
        "candidate_legal_mask",
    ):
        assert torch.equal(getattr(reference, field), getattr(vectorized, field)), field


def test_vectorized_candidate_provider_rejects_invalid_sizeable_main_id():
    data = _candidate_data()
    data.inst_main_id = torch.tensor([99], dtype=torch.long)
    data.inst_cell_id = torch.tensor([0], dtype=torch.long)
    data.inst_is_sizeable = torch.tensor([True])
    data.inst_size_lower = torch.tensor([1.0])
    data.inst_size_upper = torch.tensor([2.0])
    data.inst_vt_mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="invalid main_id"):
        VectorizedMainIdCandidateProvider().enumerate(data)


def _projection_result(inst_ids, current_cells, projected_cells, legal):
    return SimpleNamespace(
        inst_ids=torch.tensor(inst_ids, dtype=torch.long),
        current_cell_id=torch.tensor(current_cells, dtype=torch.long),
        projected_cell_id=torch.tensor(projected_cells, dtype=torch.long),
        has_legal_candidate=torch.tensor(legal, dtype=torch.bool),
    )


def _reference_pin_offsets(data, result):
    pin_ids = []
    inst_ids = []
    current_cells = []
    projected_cells = []
    legal = []
    for row, inst_id in enumerate(result.inst_ids.tolist()):
        start = int(data.flat_node2pin_start_map[inst_id].item())
        end = int(data.flat_node2pin_start_map[inst_id + 1].item())
        for pin_id in data.flat_node2pin_map[start:end].tolist():
            pin_ids.append(pin_id)
            inst_ids.append(inst_id)
            current_cells.append(int(result.current_cell_id[row].item()))
            projected_cells.append(int(result.projected_cell_id[row].item()))
            legal.append(bool(result.has_legal_candidate[row].item()))

    device = result.inst_ids.device
    pin_ids = torch.tensor(pin_ids, dtype=torch.long, device=device)
    inst_ids = torch.tensor(inst_ids, dtype=torch.long, device=device)
    current_cells = torch.tensor(current_cells, dtype=torch.long, device=device)
    projected_cells = torch.tensor(projected_cells, dtype=torch.long, device=device)
    legal = torch.tensor(legal, dtype=torch.bool, device=device)
    libpin_offsets = data.pin_2_libpin_offset[pin_ids]
    projected_libpin_ids = torch.full_like(libpin_offsets, -1)
    valid = legal & (projected_cells >= 0)
    projected_libpin_ids[valid] = (
        data.cell_id_2_libpin_id_start[projected_cells[valid]]
        + libpin_offsets[valid]
    )
    has_projected = (
        valid
        & (libpin_offsets >= 0)
        & (projected_libpin_ids >= 0)
        & (projected_libpin_ids < data.flat_lib_pin_offset_x.numel())
    )
    projected_x = torch.zeros_like(data.pin_offset_x[pin_ids])
    projected_y = torch.zeros_like(data.pin_offset_y[pin_ids])
    projected_x[has_projected] = data.flat_lib_pin_offset_x[
        projected_libpin_ids[has_projected]
    ]
    projected_y[has_projected] = data.flat_lib_pin_offset_y[
        projected_libpin_ids[has_projected]
    ]
    changed = has_projected & (
        (projected_x - data.pin_offset_x[pin_ids]).abs() > 1e-6
    )
    changed |= has_projected & (
        (projected_y - data.pin_offset_y[pin_ids]).abs() > 1e-6
    )
    return {
        "inst_ids": inst_ids,
        "pin_ids": pin_ids,
        "current_cell_id": current_cells,
        "projected_cell_id": projected_cells,
        "legal_mask": legal,
        "libpin_offsets": libpin_offsets,
        "current_pin_offset_x": data.pin_offset_x[pin_ids],
        "current_pin_offset_y": data.pin_offset_y[pin_ids],
        "projected_pin_offset_x": projected_x,
        "projected_pin_offset_y": projected_y,
        "has_projected_pin_offset": has_projected,
        "changed_pin_offset": changed,
    }


def test_vectorized_pin_offset_gather_matches_reference_for_multi_pin_nodes():
    op, data = _projection_op()
    result = _projection_result(
        inst_ids=[0, 2],
        current_cells=[0, 2],
        projected_cells=[1, 3],
        legal=[True, False],
    )

    observed = op.compute_projected_pin_offsets(result)
    expected = _reference_pin_offsets(data, result)

    assert observed.keys() == expected.keys()
    for key in expected:
        assert torch.equal(observed[key], expected[key]), key


def test_vectorized_pin_offset_gather_handles_node_without_pins():
    op, _data = _projection_op()
    result = _projection_result(
        inst_ids=[1],
        current_cells=[1],
        projected_cells=[2],
        legal=[True],
    )

    observed = op.compute_projected_pin_offsets(result)

    assert observed["pin_ids"].numel() == 0
    assert observed["changed_pin_offset"].numel() == 0


def test_vectorized_pin_offset_gather_handles_empty_row_mask():
    op, _data = _projection_op()
    result = _projection_result(
        inst_ids=[0, 2],
        current_cells=[0, 2],
        projected_cells=[1, 2],
        legal=[True, True],
    )

    observed = op.compute_projected_pin_offsets(
        result,
        row_mask=torch.tensor([False, False]),
    )

    assert all(value.numel() == 0 for value in observed.values())


def test_vectorized_pin_offset_gather_rejects_invalid_libpin_mapping():
    op, data = _projection_op()
    data.pin_2_libpin_offset = torch.tensor([-1, 1, 0, 1, 2], dtype=torch.long)
    result = _projection_result(
        inst_ids=[0],
        current_cells=[0],
        projected_cells=[1],
        legal=[True],
    )

    with pytest.raises(ValueError, match="invalid libpin offset"):
        op.compute_projected_pin_offsets(result)
