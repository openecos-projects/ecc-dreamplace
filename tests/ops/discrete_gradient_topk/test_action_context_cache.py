from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    QuadCandidateTable,
)
from dreamplace.ops.timing_propagation.discrete_actions import (
    build_discrete_action_context,
)


def _data():
    return SimpleNamespace(
        flat_inst_arcs_by_level=torch.tensor(
            [[0, 1, 0, 0, 1], [2, 3, 1, 1, 1]], dtype=torch.long
        ),
        inst_cell_id=torch.tensor([0, 1], dtype=torch.long),
        pin2node_map=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        pin2net_map=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        net2driver_pin_map=torch.tensor([1, 3], dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        flat_libarc_info=torch.tensor(
            [
                [0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 1, 0],
                [0, 1, 3, 3, 1, 0],
            ],
            dtype=torch.long,
        ),
        cell_id_2_libpin_id_start=torch.tensor(
            [0, 2, 4, 6, 8], dtype=torch.long
        ),
        pin_2_libpin_offset=torch.zeros(4, dtype=torch.long),
        inst_is_sizeable=torch.ones(2, dtype=torch.bool),
        inst_vt_mask=torch.ones((2, 2), dtype=torch.bool),
        flat_libcell_info=torch.tensor(
            [
                [0.0, 10.0, 1.0, 0.0],
                [1.0, 10.0, 2.0, 0.0],
                [2.0, 10.0, 1.0, 1.0],
                [3.0, 10.0, 2.0, 1.0],
            ]
        ),
        flat_lib_pin_rcap=torch.arange(8, dtype=torch.float32),
        flat_lib_pin_fcap=torch.arange(8, dtype=torch.float32),
    )


def test_context_cache_reuses_detached_tables_without_sharing_mutable_masks():
    data = _data()
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [[1, -1, 2, -1], [-1, 0, 3, 2], [3, -1, -1, 0], [-1, 2, -1, 1]]
        )
    )
    cache = {}
    first = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        dtype=torch.float32,
        device=torch.device("cpu"),
        drive_load_feedback=True,
        cache=cache,
    )
    pristine = first.valid_instances.clone()
    first.valid_instances.zero_()
    first.valid_arcs.zero_()

    second = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        dtype=torch.float32,
        device=torch.device("cpu"),
        drive_load_feedback=True,
        cache=cache,
    )

    assert second.mapping_summary["cache_hit"] is True
    torch.testing.assert_close(second.valid_instances, pristine)
    assert first.beta.data_ptr() != second.beta.data_ptr()

    data.inst_cell_id[0] = 1
    third = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        dtype=torch.float32,
        device=torch.device("cpu"),
        drive_load_feedback=True,
        cache=cache,
    )
    assert third.mapping_summary["cache_hit"] is False
    assert third.mapping_summary["output_buffer_reused"] is True
    assert not torch.equal(third.target_cell_ids, second.target_cell_ids)


def test_context_validation_covers_drive_load_index_devices():
    data = _data()
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [[1, -1, 2, -1], [-1, 0, 3, 2], [3, -1, -1, 0], [-1, 2, -1, 1]]
        )
    )
    context = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        dtype=torch.float32,
        device=torch.device("cpu"),
        drive_load_feedback=True,
    )
    context.drive_load_node = torch.empty(1, dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="drive_load_node device mismatch"):
        context.validate(device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_context_moves_cpu_metadata_indices_to_runtime_cuda_device():
    data = _data()
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [[1, -1, 2, -1], [-1, 0, 3, 2], [3, -1, -1, 0], [-1, 2, -1, 1]]
        )
    )
    context = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        dtype=torch.float32,
        device=torch.device("cuda"),
        drive_load_feedback=True,
    )

    for name in (
        "beta",
        "target_arc_ids",
        "target_senses",
        "valid_arcs",
        "inst_ids",
        "drive_load_rcap",
        "drive_load_fcap",
        "drive_load_node",
        "drive_load_pin_net",
        "drive_load_driver_mask",
    ):
        assert getattr(context, name).device.type == "cuda"

    cap = torch.zeros(4, device="cuda", dtype=torch.float32)
    context.apply_drive_load(cap, cap.clone())
