from types import SimpleNamespace

import numpy as np
import torch
from dreamplace.ops.discrete_gradient_topk.runtime_timing_arcs import (
    sync_timing_arc_lut_rows,
)


def _runtime_data():
    # Two Liberty cells with the same fixed arc schema and three instances.
    flat_arc_info = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    arcs = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    endpoint_arcs = torch.tensor([[0, 2, 0, 0]], dtype=torch.long)
    data = SimpleNamespace(
        pin2node_map=torch.tensor([0, 1, 2], dtype=torch.long),
        inst_cell_id=torch.tensor([1, 1, 0], dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor([0, 2, 4], dtype=torch.long),
        flat_libarc_info=flat_arc_info,
        flat_inst_arcs_by_level=arcs,
        endpoints_constraint_arcs=endpoint_arcs,
    )
    placedb = SimpleNamespace(
        flat_inst_arcs_by_level=np.zeros_like(arcs.numpy()),
        endpoints_constraint_arcs=np.zeros_like(endpoint_arcs.numpy()),
    )
    return data, placedb


def test_runtime_arc_sync_updates_only_changed_instances_and_reuses_index():
    data, placedb = _runtime_data()

    first = sync_timing_arc_lut_rows(data, placedb, None, [0])

    assert first["status"] == "ok"
    assert first["updated_row_count"] == 2
    assert first["tables"]["flat_inst_arcs_by_level"]["selected_row_count"] == 2
    assert first["tables"]["endpoints_constraint_arcs"]["selected_row_count"] == 0
    assert torch.equal(data.flat_inst_arcs_by_level[:2, 2], torch.tensor([1, 1]))
    assert torch.equal(data.flat_inst_arcs_by_level[2, 2], torch.tensor(0))
    assert set(data._runtime_timing_arc_row_index_cache) == {
        "flat_inst_arcs_by_level",
        "endpoints_constraint_arcs",
    }

    data.inst_cell_id[0] = 0
    second = sync_timing_arc_lut_rows(data, placedb, None, [0])

    assert second["status"] == "ok"
    assert second["updated_row_count"] == 2
    assert len(data._runtime_timing_arc_row_index_cache) == 2
