from types import SimpleNamespace

import torch
from dreamplace.ops.discrete_gradient_topk.action_arcs import (
    build_action_arc_table,
)
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    QuadCandidateTable,
)


def _data(*, target_cells, pin_starts, arc_info, arcs, inst_cell_id, pin2node):
    inst_cell_id = torch.tensor(inst_cell_id, dtype=torch.long)
    num_instances = int(inst_cell_id.numel())
    return SimpleNamespace(
        flat_inst_arcs_by_level=torch.tensor(arcs, dtype=torch.long),
        inst_cell_id=inst_cell_id,
        pin2node_map=torch.tensor(pin2node, dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor(
            [index * 1 for index in range(len(arc_info) + 1)],
            dtype=torch.long,
        ),
        flat_libarc_info=torch.tensor(arc_info, dtype=torch.long),
        cell_id_2_libpin_id_start=torch.tensor(pin_starts, dtype=torch.long),
        inst_is_sizeable=torch.ones(num_instances, dtype=torch.bool),
        inst_vt_mask=torch.ones((num_instances, 2), dtype=torch.bool),
        flat_libcell_info=torch.tensor(
            [
                [0.0, 10.0, 1.0, 0.0],
                [1.0, 10.0, 2.0, 0.0],
                [2.0, 10.0, 1.0, 1.0],
                [3.0, 10.0, 2.0, 1.0],
            ]
        ),
    )


def test_all_four_actions_map_to_semantically_equal_target_arcs():
    table = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [
                [1, -1, 2, -1],
                [-1, 0, 3, 2],
                [3, -1, -1, 0],
                [-1, 2, -1, 1],
            ]
        )
    )
    data = _data(
        target_cells=table.target_cell_ids,
        pin_starts=[0, 2, 4, 6, 8],
        arc_info=[
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 3, 3, 1, 0],
        ],
        arcs=[[0, 1, 0, 0, 1], [2, 3, 1, 1, 1]],
        inst_cell_id=[0, 1],
        pin2node=[0, 0, 1, 1],
    )

    result = build_action_arc_table(data, table, action_names=ACTION_NAMES)

    assert result.valid_arcs.tolist() == [
        [True, False, True, False],
        [False, True, True, True],
    ]
    assert result.valid_instances.tolist() == result.valid_arcs.tolist()
    assert result.target_arc_ids.tolist() == [
        [1, -1, 2, -1],
        [-1, 0, 3, 2],
    ]


def test_all_arcs_rule_invalidates_the_whole_instance_action():
    table = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [
                [1, -1, 2, -1],
                [-1, 0, 3, 2],
                [3, -1, -1, 0],
                [-1, 2, -1, 1],
            ]
        )
    )
    data = _data(
        target_cells=table.target_cell_ids,
        pin_starts=[0, 2, 4, 6, 8],
        arc_info=[
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 3, 3, 1, 0],
        ],
        arcs=[[0, 1, 0, 0, 1], [2, 3, 1, 1, 1]],
        inst_cell_id=[0, 0],
        pin2node=[0, 0, 0, 0],
    )

    result = build_action_arc_table(data, table, action_names=ACTION_NAMES)

    assert result.valid_arcs[:, 0].tolist() == [True, False]
    assert result.valid_instances[:, 0].tolist() == [False, False]


def test_setup_and_hold_with_equal_pins_and_trigger_keep_their_check_class():
    candidates = QuadCandidateTable(target_cell_ids=torch.tensor([[1, -1, -1, -1]]))
    data = _data(
        target_cells=candidates.target_cell_ids, pin_starts=[0, 2, 4],
        arc_info=[[0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1],
                  [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1]],
        arcs=[[0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1]],
        inst_cell_id=[0], pin2node=[0, 0],
    )
    data.cell_id_2_arc_id_start = torch.tensor([0, 2, 4])
    data.flat_libarc_check_types = torch.tensor([1, 2, 2, 1])
    result = build_action_arc_table(data, candidates, action_names=ACTION_NAMES)
    assert result.target_arc_ids.tolist() == [[3, -1, -1, -1], [2, -1, -1, -1]]
    assert result.valid_instances.tolist() == [[True, False, False, False]]
    data.flat_libarc_check_types[2:] = torch.tensor([1, 2])
    refreshed = build_action_arc_table(data, candidates, action_names=ACTION_NAMES)
    assert refreshed.target_arc_ids.tolist() == [[2, -1, -1, -1], [3, -1, -1, -1]]


def test_pin_count_change_is_invalid_even_with_equal_arc_signature():
    table = QuadCandidateTable(
        target_cell_ids=torch.tensor([[1, -1, -1, -1], [-1, 0, -1, -1]])
    )
    data = _data(
        target_cells=table.target_cell_ids,
        pin_starts=[0, 2, 5],
        arc_info=[[0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0]],
        arcs=[[0, 1, 0, 0, 1]],
        inst_cell_id=[0],
        pin2node=[0, 0],
    )

    result = build_action_arc_table(data, table, action_names=ACTION_NAMES)

    assert result.valid_arcs[:, 0].tolist() == [False]
    assert result.summary["actions"]["size_up"]["failures"]["pin_count"] == 1


def test_target_cell_is_resolved_from_the_current_cell_not_the_instance_row():
    table = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [
                [1, -1, -1, -1],
                [-1, 0, -1, -1],
                [3, -1, -1, -1],
                [-1, 2, -1, -1],
            ]
        )
    )
    data = _data(
        target_cells=table.target_cell_ids,
        pin_starts=[0, 2, 4, 6, 8],
        arc_info=[
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 3, 3, 1, 0],
        ],
        arcs=[
            [0, 2, 2, 2, 1],
            [0, 0, 0, 0, 1],
            [0, 4, 0, 0, 1],
        ],
        inst_cell_id=[2, 0, 0],
        pin2node=[1, 1, 0, 0, 2, 2],
    )

    result = build_action_arc_table(data, table, action_names=ACTION_NAMES)

    # Instance 0 is cell 2, so its size_up target must be cell 3.  Looking up
    # target_cells[2] would incorrectly use instance 2's cell-0 target.
    assert result.target_arc_ids[:, 0].tolist() == [3, 1, 1]
    assert result.valid_arcs[:, 0].tolist() == [True, True, True]


def test_swapped_pin_names_invalidate_the_action():
    table = QuadCandidateTable(
        target_cell_ids=torch.tensor([[1, -1, -1, -1]])
    )
    data = _data(
        target_cells=table.target_cell_ids,
        pin_starts=[0, 2, 4],
        arc_info=[[0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0]],
        arcs=[[0, 1, 0, 0, 1]],
        inst_cell_id=[0],
        pin2node=[0, 0],
    )
    # The arc schema matches, so a missing name table would accept the action.
    # Swapped liberty pin order must not silently move the cap delta.
    matched = build_action_arc_table(
        data,
        table,
        action_names=ACTION_NAMES,
        pin_names=("A", "Z", "A", "Z"),
    )
    swapped = build_action_arc_table(
        data,
        table,
        action_names=ACTION_NAMES,
        pin_names=("A", "Z", "Z", "A"),
    )

    assert matched.valid_instances[:, 0].tolist() == [True]
    assert matched.summary["pin_identity_checked"] is True
    assert swapped.valid_instances[:, 0].tolist() == [False]
    assert swapped.valid_arcs[:, 0].tolist() == [False]
    assert swapped.summary["actions"]["size_up"]["failures"]["pin_identity"] == 1
