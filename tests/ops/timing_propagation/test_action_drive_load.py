"""Action capacitance belongs to physical pins, independently of timing arcs."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.discrete_actions import _drive_load_tables


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clock_local_ids_and_repeated_arcs_do_not_alias_physical_pin_caps(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def ids(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    def values(rows):
        return torch.tensor(rows, dtype=dtype, device=device)

    # FF masters have CK, EN, Q. Buffer masters have A, Y. Runtime clock
    # source 0 addresses clock_pins[0] = physical pin 2, while physical pin 0
    # is the other cell's A. EN has no timing arc but still loads its net.
    caps = values(
        [
            0.003,
            0.004,
            0,
            0.006,
            0.010,
            0,
            0.002,
            0.002,
            0,
            0.005,
            0.006,
            0,
            0.001,
            0.001,
            0,
            0.005,
            0,
            0.008,
            0,
            0.002,
            0,
            0.004,
            0,
            0.006,
            0,
        ]
    )
    data = SimpleNamespace(
        flat_inst_arcs_by_level=ids([[0, 4, 0, 0, 1], [0, 1, 5, 1, 1], [0, 1, 5, 1, 1]]),
        pin2node_map=ids([1, 1, 0, 0, 0]),
        inst_cell_id=ids([0, 5]),
        pin2net_map=ids([0, 1, 2, 3, 4]),
        clock_pins=ids([2]),
        cell_id_2_libpin_id_start=ids([0, 3, 6, 9, 12, 15, 17, 19, 21, 23, 25]),
        pin_2_libpin_offset=ids([0, 1, 0, 1, 2]),
        flat_lib_pin_rcap=caps,
        flat_lib_pin_fcap=2 * caps,
    )
    table = SimpleNamespace(
        inst_ids=ids([0, 1, 1]),
        target_cell_ids=ids([[1, 2, 3, 4], [6, 7, 8, 9]]),
        valid_arcs=torch.ones((3, 4), dtype=torch.bool, device=device),
        valid_instances=torch.ones((2, 4), dtype=torch.bool, device=device),
    )
    expected = values(
        [
            [0.003, -0.003, -0.001, 0.001],
            [0, 0, 0, 0],
            [0.003, -0.001, 0.002, -0.002],
            [0.006, -0.002, 0.002, -0.003],
            [0, 0, 0, 0],
        ]
    )

    for order in ([0, 1, 2], [2, 1, 0]):
        data.flat_inst_arcs_by_level = data.flat_inst_arcs_by_level[ids(order)]
        table.inst_ids = table.inst_ids[ids(order)]
        actual = _drive_load_tables(data, table, device=device, dtype=dtype)
        torch.testing.assert_close(actual, (expected, 2 * expected))
