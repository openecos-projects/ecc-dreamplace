import numpy as np
import torch

from dreamplace.ops.placeio_ecc.sparse_timing_schema import SparseTimingView
from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule


def add_transition_masks(arrays, base_cell):
    """Complete synthetic fixtures whose arcs have no exception selectors."""
    base = np.asarray(base_cell, dtype=np.int64).reshape(-1, 8)
    for name in ("cell_edges", "slew_only_cell_edges"):
        rows = base[arrays[name][:, 0]]
        masks = np.where(rows[:, 4] == 1, 9, np.where(rows[:, 4] == -1, 6, 15))
        triggers = np.where(rows[:, 5] == 1, 3, np.where(rows[:, 5] == -1, 12, 15))
        arrays[name.replace("edges", "transition_mask")] = masks & triggers
    arrays["net_transition_mask"] = np.full(len(arrays["net_edges"]), 9, dtype=np.int64)


def sparse_view(
    *, false_path=False, cell=False, adjustment=(0, 0, 1, 1), slew_only=False
):
    count = 6 if cell else 4
    arrays = {
        "pin_state_start": np.array(
            [0, 1, 2, 4, 6][: 5 if cell else 4], dtype=np.int64
        ),
        "state_pin": np.array([0, 1, 2, 2, 3, 3][:count], dtype=np.int64),
        "state_domain": np.array([0, 1, 0, 1, 0, 1][:count], dtype=np.int64),
        "state_tag": np.array([0, 1, 0, 1, 0, 1][:count], dtype=np.int64),
        "state_launch_tag": np.array([0, 1, 0, 1, 0, 1][:count], dtype=np.int64),
        "state_source_type": np.ones(count, dtype=np.int64),
        "state_role": np.zeros(count, dtype=np.int64),
        "state_level": np.array([1, 1, 2, 2, 3, 3][:count], dtype=np.int64),
        "state_slew_group": np.array([0, 1, 2, 2, 3, 3][:count], dtype=np.int64),
        "state_valid": np.ones((count, 2), dtype=np.uint8),
        "state_seed": np.array(
            [[1, 1], [1, 1]] + [[0, 0]] * (count - 2), dtype=np.uint8
        ),
        "state_reference": np.array(
            [[10, 9, 0.2, 0.25], [20, 18, 0.3, 0.35]]
            + [[-2e8, -2e8, 0, 0]] * (count - 2),
            dtype=np.float64,
        ),
        "net_edges": np.array([[2, 0, 2], [2, 1, 3]], dtype=np.int64),
        "net_transition_mask": np.full(2, 9, dtype=np.int64),
        "cell_transition_mask": np.full(2 if cell else 0, 9, dtype=np.int64),
        "slew_only_cell_transition_mask": np.empty(0, dtype=np.int64),
        "context_state": np.empty(0, dtype=np.int64),
        "net_threshold_adjustments": np.array([adjustment] * 2, dtype=np.float64),
        "cell_edges": np.array([[0, 2, 4], [0, 3, 5]], dtype=np.int64)
        if cell
        else np.empty((0, 3), dtype=np.int64),
        "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
        "check_index": np.array(
            [
                [count - 2, 0, -1, 0, -1, 0, 1],
                [count - 2, 0, -1, 1, -1, 0, 1],
                [
                    count - 1,
                    0,
                    -1,
                    0,
                    0,
                    3 if false_path else 1,
                    0 if false_path else 1,
                ],
                [
                    count - 1,
                    0,
                    -1,
                    1,
                    0,
                    3 if false_path else 1,
                    0 if false_path else 1,
                ],
            ],
            dtype=np.int64,
        ),
        "check_values": np.array([[22, -0.5, 0, 22]] * 4, dtype=np.float64),
    }
    if slew_only:
        arrays["slew_only_cell_edges"] = arrays["cell_edges"]
        arrays["slew_only_cell_transition_mask"] = arrays["cell_transition_mask"]
        arrays["cell_edges"] = np.empty((0, 3), dtype=np.int64)
        arrays["cell_transition_mask"] = np.empty(0, dtype=np.int64)
    view = SparseTimingView(
        arrays, ("sys", "jtag"), ("sys", "jtag"), (), "fixture", "fixture"
    )
    base = [[2, 3, 0, 0, 1, 0, 0, 0]] if cell else []
    return view, base


def schedule(device, **kwargs):
    view, base = sparse_view(**kwargs)
    return SparseTimingSchedule(view, base, device=device, dtype=torch.float64)


def inputs(device):
    return tuple(
        torch.tensor(value, device=device, dtype=torch.float64, requires_grad=True)
        for value in (
            [0, 0, 0.2, 0.5],
            [0, 0, 0.25, 0.55],
            [0, 0, 5, 0],
            [0, 0, 4, 0],
            [0, 0, 0.04, 0],
            [0, 0, 0.05, 0],
        )
    )


def tables(device):
    result = []
    for scale in (1.0, 0.8, 0.1, 0.12):
        result.extend(
            (
                torch.tensor([[1, 2, 3, 4]], device=device, dtype=torch.float64)
                * scale,
                torch.tensor([[0, 1]], device=device, dtype=torch.float64),
                torch.tensor([[0, 1]], device=device, dtype=torch.float64),
                torch.tensor([2], device=device),
                torch.tensor([2], device=device),
            )
        )
    return result


def production_timing(device, mode="hard"):
    from types import SimpleNamespace
    from dreamplace.ops.timing_propagation.timing_propagation import (
        LUTS_INFO,
        TimingPropagation,
    )

    view, base = sparse_view(cell=True)
    data = tables(device)
    infos = {}
    for offset, name in zip(
        range(0, 20, 5), ("r_delay", "f_delay", "r_trans", "f_trans")
    ):
        values, transitions, loads, nt, nc = data[offset : offset + 5]
        infos[f"{name}_luts"] = LUTS_INFO(
            values, transitions, loads, torch.stack((nt, nc), dim=1)
        )
    infos["r_check_luts"] = infos["r_delay_luts"]
    infos["f_check_luts"] = infos["f_delay_luts"]

    def ids(value):
        return torch.tensor(value, device=device, dtype=torch.long)

    empty = torch.empty(0, device=device, dtype=torch.float64)
    arcs = ids(base)
    return TimingPropagation(
        inrdelays=empty,
        infdelays=empty,
        inrtrans=empty,
        inftrans=empty,
        outcaps=empty,
        pin_net=ids([0, 1, 2, 3]),
        start_points=ids([0, 1]),
        end_points=ids([3]),
        clock_pins=ids([]),
        FF_ids=ids([]),
        clk_pin_rtran=empty,
        clk_pin_ftran=empty,
        net_flat_arcs_start=ids([0, 1, 2, 2, 2]),
        net_flat_arcs=ids([[0, 2], [1, 2]]),
        net2driver_pin_map=ids([0, 1, 2, 3]),
        arcs_info=SimpleNamespace(**infos),
        inst_flat_arcs_start=ids([0, 1]),
        inst_flat_arcs=arcs,
        endpoints_constraint_arcs=ids([]).reshape(0, 8),
        endpoints_timing_check_arcs=ids([]).reshape(0, 8),
        flat_inst_arcs_by_level=arcs,
        flat_inst_arcs_by_level_start=ids([0, 0, 1]),
        endpoints_rRAT=empty,
        endpoints_fRAT=empty,
        flat_pin_to_graph=ids([]),
        flat_pin_to_graph_start=ids([0] * 5),
        flat_pin_to_graph_reverse=ids([]),
        flat_pin_to_graph_start_reverse=ids([0] * 5),
        timing_metric_scope="setup_only",
        timing_aggregation_mode=mode,
        sparse_timing_view=view,
        timing_propagation_device=device,
        production_fast_loop=True,
    )


def shared_driver_timing():
    timing = production_timing("cpu")
    arrays = timing.sparse_timing_view.arrays
    arrays["state_pin"][:] = [0, 0, 2, 2, 3, 3]
    arrays["pin_state_start"][:] = [0, 2, 2, 4, 6]
    arrays["state_slew_group"][:2] = 0
    timing.pin_net[:] = torch.tensor([2, 0, 2, 3])
    timing.net_flat_arcs = torch.tensor([[0, 2]])
    timing.net_flat_arcs_start = torch.tensor([0, 0, 0, 1, 1])
    timing.net2driver_pin_map[:] = torch.tensor([0, 1, 0, 3])
    return timing


def shared_driver_net():
    return {
        "net_id": 2,
        "driver_pin_id": 0,
        "coordinates": {0: (0, 0), 2: (100, 0)},
        "rc_tree": {
            "root_node_id": 0,
            "children_by_node": {0: [2], 2: []},
            "edge_rc": {(0, 2): {"r": 10.0, "c": 0.0}},
            "node_cap": {0: 0.0, 2: 1.0},
            "sink_nodes": [2],
        },
    }


def rise_fall(pair):
    return dict(zip(("rise", "fall"), pair))


def run_production(timing, values):
    return timing(
        rise_fall(values[2:4]),
        rise_fall(values[4:]),
        rise_fall(values[:2]),
        surrogate_mode="lut_only",
    )
