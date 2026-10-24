from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.sparse_timing_schema import (
    STATE_KEYS,
    read_sparse_timing_view,
    sparse_digest,
)


def sign(view):
    arrays, metadata = view["arrays"], view["metadata"]
    view["semantic_digest"] = sparse_digest(arrays, metadata)
    view["state_key_digest"] = sparse_digest(
        {key: arrays[key] for key in STATE_KEYS},
        {"domains": metadata["domains"], "tags": metadata["tags"]},
    )


@pytest.fixture
def pydb():
    view = {
        "metadata": {
            "format_version": 1,
            "domains": ["sys", "jtag", ""],
            "tags": ["sys", "jtag"],
            "unsupported": [],
            "capabilities": ["sparse_max_v1", "native_required_affine_v1"],
        },
        "arrays": {
            "pin_reference_cap": np.zeros((3, 2), dtype=np.float64),
            "pin_state_start": np.array([0, 1, 2, 4], dtype=np.int64),
            "state_pin": np.array([0, 1, 2, 2], dtype=np.int64),
            "state_domain": np.array([0, 1, 0, 1], dtype=np.int64),
            "state_tag": np.array([0, 1, 0, 1], dtype=np.int64),
            "state_source_type": np.full(4, 2, dtype=np.int64),
            "state_slew_group": np.array([0, 1, 2, 3], dtype=np.int64),
            "state_level": np.array([0, 0, 1, 1], dtype=np.int64),
            "state_valid": np.ones((4, 2), dtype=np.uint8),
            "state_seed": np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=np.uint8),
            "state_reference": np.array(
                [[10, 10, 2, 2], [20, 20, 3, 3], [30, 30, 4, 4], [40, 40, 5, 5]],
                dtype=np.float64,
            ),
            "cell_edges": np.empty((0, 3), dtype=np.int64),
            "slew_only_cell_edges": np.empty((0, 3), dtype=np.int64),
            "net_edges": np.array([[2, 0, 2], [2, 1, 3]], dtype=np.int64),
            "net_threshold_adjustments": np.array([[0, 0, 1, 1]] * 2, dtype=np.float64),
            "check_index": np.array(
                [[2, 0, -1, 0, -1, 0, 1], [3, 0, -1, 0, 0, 3, 0]], dtype=np.int64
            ),
            "check_values": np.array(
                [[100, 0, 0, 100], [0, 0, 0, 0]], dtype=np.float64
            ),
        },
    }
    sign(view)
    return SimpleNamespace(
        timing_schema_version=3,
        timing_sparse=view,
        pin_names=["q_sys", "q_jtag", "out"],
        clock_pins=[],
        end_points=[2],
        endpoints_timing_check_arcs=[],
        flat_inst_arcs_by_level=[],
        flat_inst_arcs_by_level_start=[0, 0],
        net_flat_arcs=[[0, 2], [1, 2]],
    )


def test_reconvergent_domains_remain_distinct(pydb):
    view = read_sparse_timing_view(pydb)
    view.require_supported()
    statistics = view.statistics()
    assert (
        statistics["num_states"],
        statistics["states_per_pin_max"],
        statistics["num_legal_endpoint_candidates"],
        statistics["rule_counts"],
    ) == (4, 2, 1, {"normal": 1, "max_delay": 0, "multicycle": 0, "false_path": 1})
    assert (
        view.semantic_digest == read_sparse_timing_view(deepcopy(pydb)).semantic_digest
    )


def test_inout_role_transition_preserves_launch_tag(pydb):
    arrays = pydb.timing_sparse["arrays"]
    arrays["net_edges"] = np.array([[2, 0, 3]], dtype=np.int64)
    arrays["net_threshold_adjustments"] = np.array([[0, 0, 1, 1]], dtype=np.float64)
    arrays["state_domain"][3] = arrays["state_domain"][0]
    arrays["state_source_type"][3] = arrays["state_source_type"][0]
    pydb.timing_sparse["metadata"]["tags"] = ["load\x1fsys", "drive\x1fsys"]
    sign(pydb.timing_sparse)

    view = read_sparse_timing_view(pydb)
    view.require_supported()
    assert view.tags == ("load\x1fsys", "drive\x1fsys")


@pytest.mark.parametrize(
    "name,index,value,message",
    [
        ("pin_state_start", 2, 3, "ownership"),
        ("net_edges", (1, 1), 0, "domain"),
        ("net_edges", (0, 0), 0, "sink"),
        ("check_index", (0, 0), 99, "launch state"),
        ("check_index", (1, 6), 1, "false path"),
        ("state_seed", (2, 0), 2, "seed mask"),
        ("state_reference", (0, 0), np.nan, "nonfinite"),
    ],
)
def test_structural_corruption_is_rejected_before_materialization(
    pydb, name, index, value, message
):
    pydb.timing_sparse["arrays"][name][index] = value
    sign(pydb.timing_sparse)
    with pytest.raises(RuntimeError, match=message):
        read_sparse_timing_view(pydb)


def test_byte_change_invalidates_semantic_digest(pydb):
    pydb.timing_sparse["arrays"]["check_values"][0, 0] += 1
    with pytest.raises(RuntimeError, match="semantic digest"):
        read_sparse_timing_view(pydb)


def test_unsupported_constraints_remain_visible_and_stop_optimization(pydb):
    pydb.timing_sparse["metadata"]["unsupported"] = [
        "exception 7 needs through transitions"
    ]
    sign(pydb.timing_sparse)
    view = read_sparse_timing_view(pydb)
    with pytest.raises(RuntimeError, match="through transitions"):
        view.require_supported()
