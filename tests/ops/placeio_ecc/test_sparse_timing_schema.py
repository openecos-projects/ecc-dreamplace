from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.sparse_timing_schema import (
    STATE_KEYS,
    model_digest,
    model_rule_units,
    read_sparse_timing_view,
    sparse_digest,
)


def sign(view):
    arrays, metadata = view["arrays"], view["metadata"]
    view["semantic_digest"] = sparse_digest(arrays, metadata)
    view["model_digest"] = model_digest(arrays, metadata)
    view["state_key_digest"] = sparse_digest(
        {key: arrays[key] for key in STATE_KEYS},
        {"domains": metadata["domains"], "tags": metadata["tags"], "launch_tags": metadata["launch_tags"]},
    )


@pytest.fixture
def pydb():
    view = {
        "metadata": {
            "format_version": 2,
            "analysis": "max",
            "domains": ["sys", "jtag", ""],
            "tags": ["sys", "jtag"],
            "launch_tags": ["sys", "jtag"],
            "unsupported": [],
            "capabilities": ["sparse_max_v1", "native_required_affine_v1", "native_transition_mask_v1"],
        },
        "arrays": {
            "pin_reference_cap": np.zeros((3, 2), dtype=np.float64),
            "pin_state_start": np.array([0, 1, 2, 4], dtype=np.int64),
            "state_pin": np.array([0, 1, 2, 2], dtype=np.int64),
            "state_domain": np.array([0, 1, 0, 1], dtype=np.int64),
            "state_tag": np.array([0, 1, 0, 1], dtype=np.int64),
            "state_launch_tag": np.array([0, 1, 0, 1], dtype=np.int64),
            "state_source_type": np.full(4, 2, dtype=np.int64),
            "state_role": np.zeros(4, dtype=np.int64),
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
            "cell_transition_mask": np.empty(0, dtype=np.int64),
            "slew_only_cell_transition_mask": np.empty(0, dtype=np.int64),
            "net_transition_mask": np.full(2, 9, dtype=np.int64),
            "context_state": np.empty(0, dtype=np.int64),
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
        and view.model_digest == read_sparse_timing_view(deepcopy(pydb)).model_digest
    )


@pytest.fixture
def electrical_pydb(pydb):
    raw = pydb.timing_sparse
    raw["metadata"]["format_version"] = 3
    raw["metadata"]["tags"].append("electrical_only")
    raw["metadata"]["launch_tags"].append("electrical_only")
    a = raw["arrays"]
    pydb.pin_names.append("electrical_sink")
    pydb.net_flat_arcs.append([2, 3])
    a["pin_reference_cap"] = np.vstack((a["pin_reference_cap"], [0.0, 0.0]))
    a["pin_state_start"] = np.append(a["pin_state_start"], 5)
    for name, value in (
        ("state_pin", 3), ("state_domain", 2), ("state_tag", 2),
        ("state_launch_tag", 2), ("state_source_type", 0), ("state_role", 0),
        ("state_level", 2), ("state_slew_group", 4),
    ):
        a[name] = np.append(a[name], value)
    for name, value in (
        ("state_valid", [0, 0]), ("state_seed", [0, 0]),
        ("state_reference", [-9e7, -9e7, 0, 0]),
    ):
        a[name] = np.vstack((a[name], np.asarray(value, dtype=a[name].dtype)))
    a["net_edges"] = np.vstack((a["net_edges"], [3, 2, 4]))
    a["net_transition_mask"] = np.append(a["net_transition_mask"], 0)
    a["net_threshold_adjustments"] = np.vstack((a["net_threshold_adjustments"], [0, 0, 1, 1]))
    sign(raw)
    return pydb


def test_electrical_only_net_keeps_slew_gradient_without_a_data_path(electrical_pydb):
    import torch
    from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule

    view = read_sparse_timing_view(electrical_pydb)
    model = SparseTimingSchedule(view, [], device="cpu", dtype=torch.float64)
    model.reference.requires_grad_()
    zero = torch.zeros(4, dtype=torch.float64)
    lanes = model.propagate((), (zero, zero), (zero, zero), (zero, zero))
    torch.testing.assert_close(lanes[0][4], torch.tensor(-2e8, dtype=torch.float64))
    torch.testing.assert_close(lanes[2][4], torch.tensor(2.0, dtype=torch.float64))
    gradient = torch.autograd.grad(lanes[2][4], model.reference)[0]
    expected = torch.zeros_like(gradient)
    expected[0, 2] = 1
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("field", ["state_valid", "state_seed"])
def test_electrical_only_state_cannot_create_arrival_seed_or_transition(electrical_pydb, field):
    raw = electrical_pydb.timing_sparse
    raw["arrays"][field][-1, 0] = 1
    sign(raw)
    with pytest.raises(RuntimeError, match="electrical-only state has an arrival"):
        read_sparse_timing_view(electrical_pydb)


def test_electrical_only_state_cannot_be_a_timing_check(electrical_pydb):
    raw = electrical_pydb.timing_sparse
    raw["arrays"]["check_index"][0, 0] = 4
    sign(raw)
    with pytest.raises(RuntimeError, match="check uses an electrical-only"):
        read_sparse_timing_view(electrical_pydb)


def test_inout_role_transition_preserves_launch_tag(pydb):
    arrays = pydb.timing_sparse["arrays"]
    arrays["net_edges"] = np.array([[2, 0, 3]], dtype=np.int64)
    arrays["net_transition_mask"] = np.array([9], dtype=np.int64)
    arrays["net_threshold_adjustments"] = np.array([[0, 0, 1, 1]], dtype=np.float64)
    arrays["state_domain"][3] = arrays["state_domain"][0]
    arrays["state_source_type"][3] = arrays["state_source_type"][0]
    arrays["state_launch_tag"][3] = arrays["state_launch_tag"][0]
    pydb.timing_sparse["metadata"]["tags"] = ["load\x1fsys", "drive\x1fsys"]
    sign(pydb.timing_sparse)

    view = read_sparse_timing_view(pydb)
    view.require_supported()
    assert view.tags == ("load\x1fsys", "drive\x1fsys")


def test_exception_progress_keeps_native_launch_and_transition_ownership(pydb):
    arrays = pydb.timing_sparse["arrays"]
    arrays["net_edges"] = np.array([[2, 0, 3]], dtype=np.int64)
    arrays["net_transition_mask"] = np.array([1], dtype=np.int64)
    arrays["net_threshold_adjustments"] = np.array([[0, 0, 1, 1]], dtype=np.float64)
    arrays["state_domain"][3] = 0
    arrays["state_launch_tag"][3] = 0
    arrays["state_valid"][3, 1] = 0
    pydb.timing_sparse["metadata"]["tags"] = ["before_selector", "after_selector"]
    sign(pydb.timing_sparse)
    view = read_sparse_timing_view(pydb)
    view.require_supported()
    np.testing.assert_array_equal(view.arrays["net_transition_mask"], [1])

    arrays["net_transition_mask"][0] = 9
    sign(pydb.timing_sparse)
    with pytest.raises(RuntimeError, match="invalid state lane"):
        read_sparse_timing_view(pydb)


@pytest.mark.parametrize(
    "name,index,value,message",
    [
        ("pin_state_start", 2, 3, "ownership"),
        ("net_edges", (1, 1), 0, "domain"),
        ("state_launch_tag", 2, 1, "launch identity"),
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


def test_reference_changes_do_not_change_model_digest(pydb):
    before = read_sparse_timing_view(deepcopy(pydb))
    arrays = pydb.timing_sparse["arrays"]
    arrays["state_reference"][0, 0] += 11
    arrays["check_values"][0, 3] += 7
    sign(pydb.timing_sparse)
    after = read_sparse_timing_view(pydb)

    assert after.model_digest == before.model_digest
    assert after.semantic_digest != before.semantic_digest


def test_model_digest_ignores_sub_femtosecond_rule_roundoff(pydb):
    before = read_sparse_timing_view(deepcopy(pydb))
    pydb.timing_sparse["arrays"]["check_values"][0, 0] += 5e-13
    sign(pydb.timing_sparse)
    after = read_sparse_timing_view(pydb)

    assert after.model_digest == before.model_digest
    assert after.semantic_digest != before.semantic_digest


def test_model_rule_units_round_half_away_from_zero_like_native_llround():
    # 15299.9999999925 ps scales to exactly ...992.5; std::llround gives ...993.
    values = np.array([15299.9999999925, -2.5e-9, 1.5e-9, 0.0])

    assert model_rule_units(values).tolist() == [15299999999993, -3, 2, 0]


def test_model_rule_changes_invalidate_model_digest(pydb):
    before = read_sparse_timing_view(deepcopy(pydb))
    pydb.timing_sparse["arrays"]["check_values"][0, 0] += 1
    sign(pydb.timing_sparse)
    after = read_sparse_timing_view(pydb)

    assert after.model_digest != before.model_digest


def test_unsupported_constraints_remain_visible_and_stop_optimization(pydb):
    pydb.timing_sparse["metadata"]["unsupported"] = [
        "exception 7 needs through transitions"
    ]
    sign(pydb.timing_sparse)
    view = read_sparse_timing_view(pydb)
    with pytest.raises(RuntimeError, match="through transitions"):
        view.require_supported()
