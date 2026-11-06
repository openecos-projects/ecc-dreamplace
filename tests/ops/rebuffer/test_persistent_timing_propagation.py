"""Graph-owned sweeps must recompute results from constraints, not imported AAT/RAT."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.propagation_payload import persistent_propagation_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base, _reference_after


def _graph(owner):
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    reference = persistent_timing_state_payload(owner)
    poisoned = {
        name: values.copy() if name.startswith("pin_net_") else np.full_like(values, 987654.0)
        for name, values in reference.items()
    }
    graph.initialize_timing_state(poisoned)
    graph.initialize_propagation(persistent_propagation_payload(owner))
    return graph, reference


def _compare(graph, owner, reference=None):
    if reference is None:
        reference = persistent_timing_state_payload(owner)
    result = graph.propagate()
    view = graph.state_view()
    for name, values in reference.items():
        np.testing.assert_allclose(view[name], values, rtol=0, atol=1e-6, err_msg=name)
    assert result["metrics"] == pytest.approx(owner.baseline.as_dict(), abs=1e-6, rel=0)
    np.testing.assert_array_equal(result["endpoint_pins"], owner.model.last_endpoint_ids_tensor)
    np.testing.assert_allclose(
        result["endpoint_slacks"], owner.model.last_endpoint_slack_tensor, rtol=0, atol=1e-6
    )
    return result


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("hard_pairwise", [False, True])
def test_full_sweep_recomputes_poisoned_state_and_is_repeatable(
    monkeypatch, chain, sense, hard_pairwise
):
    monkeypatch.setenv("AIMP_FORCE_HARD_PAIRWISE_MAX", str(int(hard_pairwise)))
    base = _reference_after(_base()) if chain else _base()
    for row in base.flat_inst_arcs_by_level:
        row[4] = sense
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph, reference = _graph(owner)
    checkpoint = graph.checkpoint()

    first = _compare(graph, owner, reference)

    assert first["changed_pins"]
    assert first["coverage"]["cell_forward_rows"] == len(base.flat_inst_arcs_by_level)
    repeated = _compare(graph, owner, reference)
    assert repeated["changed_pins"] == repeated["changed_endpoints"] == []
    graph.restore(checkpoint)
    assert np.all(graph.state_view()["pin_rAAT"] == 987654.0)
    _compare(graph, owner, reference)


def test_reconvergent_and_equivalent_arcs_read_the_same_prelevel_state():
    base = _base()
    base.flat_inst_arcs_by_level = [
        [0, 2, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 1, 0, 0, 0],
        [1, 2, 1, 1, -1, 0, 0, 0],
    ]
    base.flat_inst_arcs_by_level_start = [0, 0, 3]
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph, _ = _graph(owner)

    _compare(graph, owner)


def _checks_base():
    base = _base()
    base.clock_pins = [1]  # local 0 differs from physical 1
    base.FF_ids = [2]
    base.clk_pin_rtran, base.clk_pin_ftran = [10.0], [11.0]
    base.flat_inst_arcs_by_level = [
        [0, 2, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 1, 0, 0, 0],
    ]
    base.flat_inst_arcs_by_level_start = [0, 1, 2]
    base.endpoints_constraint_arcs = np.array(
        [[0, 3, 0, 0, 1, 0, 1, 0], [0, 3, 1, 1, -1, 0, 1, 0]], dtype=np.int64
    )
    base.endpoints_constraint_max_valid = np.array([[True, False], [True, True]])
    base.endpoints_constraint_max_reason = (~base.endpoints_constraint_max_valid).astype(np.int32)
    base.endpoints_timing_check_arcs = np.array(
        [[1, 3, 0, 0, 1, -1, 3, 0], [1, 3, 1, 1, 1, 1, 3, 0]], dtype=np.int64
    )
    base.endpoints_timing_check_max_valid = np.array([[True, False], [True, True]])
    base.endpoints_timing_check_max_reason = (~base.endpoints_timing_check_max_valid).astype(
        np.int32
    )
    return base


@pytest.mark.parametrize("live_reference", [False, True])
@pytest.mark.parametrize("scope", ["setup_only", "setup_plus_recovery"])
def test_active_clock_setup_and_recovery_rows_match_python(live_reference, scope):
    owner = RebufferTimingModel(_checks_base(), use_pin_offset=False)
    owner.model.recovery_check_uses_live_slew = live_reference
    owner.model.timing_metric_scope = scope
    owner.model.backend_endpoint_rSlew = torch.tensor([5.0], dtype=torch.float64)
    owner.model.backend_endpoint_fSlew = torch.tensor([6.0], dtype=torch.float64)
    owner._baseline = owner._forward()
    graph, _ = _graph(owner)

    result = _compare(graph, owner)

    assert result["coverage"]["clk2q_rows"] == 1
    assert result["coverage"]["setup_rows"] == result["coverage"]["recovery_rows"] == 2
    assert graph.state_view()["cell_arc_rr_delays"][0] == 0


def test_metadata_capture_deadline_matches_python_with_mixed_setup_and_recovery():
    base = _checks_base()
    base.recovery_rat_is_capture_deadline = True
    owner = RebufferTimingModel(base, use_pin_offset=False)
    owner.model.timing_metric_scope = "setup_plus_recovery"
    owner._baseline = owner._forward()
    graph, _ = _graph(owner)
    result = _compare(graph, owner)
    assert result["coverage"]["recovery_rows"] == 2


@pytest.mark.parametrize("scope", ["setup_only", "setup_plus_recovery"])
def test_recovery_only_endpoint_metric_scope_and_state_qualification(scope):
    base = _checks_base()
    base.endpoints_constraint_arcs = np.empty((0, 8), dtype=np.int64)
    base.endpoints_constraint_max_valid = np.empty((0, 2), dtype=bool)
    base.endpoints_constraint_max_reason = np.empty((0, 2), dtype=np.int32)
    base.endpoints_max_valid = np.array([[False, True]])
    base.endpoints_max_reason = np.array([[1, 0]], dtype=np.int32)
    base.endpoints_timing_check_max_valid[:, 0] = False
    base.endpoints_timing_check_max_reason[:, 0] = 1
    owner = RebufferTimingModel(base, use_pin_offset=False)
    owner.model.timing_metric_scope = scope
    owner._baseline = owner._forward()
    graph, _ = _graph(owner)

    result = _compare(graph, owner)

    assert len(result["endpoint_pins"]) == (scope == "setup_plus_recovery")


def test_rc_change_and_restore_match_fresh_full_tp_without_reimporting_results():
    owner = RebufferTimingModel(_reference_after(_base()), use_pin_offset=False)
    graph, reference = _graph(owner)
    _compare(graph, owner, reference)
    saved = graph.state_view()
    checkpoint = graph.checkpoint()
    owner.loads["rise"][2] += 0.02
    owner.delays["rise"][3] += 3.0
    owner.impulses["rise"][3] += 4.0
    owner._baseline = owner._forward()
    changed = persistent_timing_state_payload(owner)
    graph.set_state({name: value for name, value in changed.items() if name.startswith("pin_net_")})

    result = _compare(graph, owner, changed)

    assert result["changed_endpoints"] == [5]
    assert result["changed_pins"]
    graph.restore(checkpoint)
    for name in reference:
        np.testing.assert_array_equal(
            graph.state_view()[name].view(np.uint64), saved[name].view(np.uint64)
        )


def test_inputs_are_owned_and_native_sweep_does_not_call_python(monkeypatch):
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    payload = persistent_propagation_payload(owner)
    graph.initialize_propagation(payload)
    payload["start_rAAT"][:] = 999.0
    payload["luts"][0][0][:] = 999.0

    def forbidden(*args, **kwargs):
        pytest.fail("native propagation called the Python timing operator")

    monkeypatch.setattr(owner.model, "calculate_cell_aat_level", forbidden)
    _compare(graph, owner)
    with pytest.raises(ValueError, match="immutable"):
        graph.initialize_propagation(payload)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unknown",
        "dtype",
        "layout",
        "pins",
        "cells",
        "clock",
        "offset",
        "lut_row",
        "lut_dims",
        "net_csr",
        "net_driver",
        "start_length",
        "mask",
        "mode",
        "schema",
    ],
)
def test_invalid_propagation_inputs_fail_before_any_state_or_owner_mutation(failure):
    owner = RebufferTimingModel(_checks_base(), use_pin_offset=False)
    payload = persistent_propagation_payload(owner)
    invalid = deepcopy(payload)
    if failure == "missing":
        del invalid["start_rAAT"]
    elif failure == "unknown":
        invalid["typo"] = 0
    elif failure == "dtype":
        invalid["clock_rtran"] = invalid["clock_rtran"].astype(np.float32)
    elif failure == "layout":
        invalid["start_rAAT"] = np.zeros(2)[::2].reshape(1, 1)
    elif failure == "pins":
        invalid["pin_names"] = list(reversed(invalid["pin_names"]))
    elif failure == "cells":
        invalid["cell_arcs"][1, 0] = 0
    elif failure == "clock":
        invalid["setup_arcs"][0, 0] = 1
    elif failure == "offset":
        invalid["cell_level_start"][-1] += 1
    elif failure == "lut_row":
        invalid["setup_arcs"][0, 3] = 99
    elif failure == "lut_dims":
        invalid["luts"][0][3][0] = 99
    elif failure == "net_csr":
        invalid["net_arc_start"][-1] += 1
    elif failure == "net_driver":
        invalid["net_drivers"][0] = 99
    elif failure == "start_length":
        invalid["start_ftran"] = np.empty(0)
    elif failure == "mask":
        invalid["setup_valid"] = np.zeros((1, 2), dtype=bool)
    elif failure == "mode":
        invalid["aggregation_mode"] = "smooth"
    else:
        # Schema 3 is a supported propagation source; only unknown versions fail.
        invalid["timing_schema_version"] = 4
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    state = persistent_timing_state_payload(owner)
    graph.initialize_timing_state(state)
    before = graph.summary()

    with pytest.raises(ValueError):
        graph.initialize_propagation(invalid)

    assert graph.summary() == before
    for name, values in state.items():
        np.testing.assert_array_equal(graph.state_view()[name], values)
    graph.initialize_propagation(payload)
    _compare(graph, owner)


def test_sweep_requires_both_original_inputs_and_canonical_state():
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    graph.initialize_propagation(persistent_propagation_payload(owner))

    with pytest.raises(ValueError, match="initialized"):
        graph.propagate()

    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    _compare(graph, owner)


@pytest.mark.parametrize("kind", ["no_cells", "no_endpoints", "missing_net_driver"])
def test_empty_domains_and_undriven_reverse_nets(kind):
    base = _base()
    if kind == "no_cells":
        base.flat_inst_arcs_by_level = []
        base.flat_inst_arcs_by_level_start = [0, 0]
    elif kind == "no_endpoints":
        base.end_points = base.endpoint_pin_ids = []
        base.endpoints_rRAT = base.endpoints_fRAT = base.outcaps = []
        base.endpoints_max_valid = np.empty((0, 2), dtype=bool)
        base.endpoints_max_reason = np.empty((0, 2), dtype=np.int32)
    else:
        base.net2driver_pin_map[1] = -1
    owner = RebufferTimingModel(base, use_pin_offset=False)
    graph, _ = _graph(owner)

    _compare(graph, owner)


@pytest.mark.parametrize(
    "field,pin,value",
    [
        ("delays", 1, np.nan),
        ("delays", 3, np.inf),
        ("impulses", 3, -np.inf),
        ("loads", 2, np.nan),
        ("loads", 2, np.inf),
    ],
)
def test_nonfinite_rc_inputs_follow_python_propagation(field, pin, value):
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    getattr(owner, field)["rise"][pin] = value
    # Direct TP returns nonfinite diagnostics. The repair snapshot API rejects
    # those metrics, so the oracle intentionally compares the operator state.
    wns, tns, worst, _ = owner.model(
        owner.delays, owner.impulses, owner.loads, surrogate_mode="lut_only"
    )
    graph, reference = _graph(owner)

    result = graph.propagate()

    for name, expected in reference.items():
        np.testing.assert_allclose(
            graph.state_view()[name], expected, rtol=0, atol=1e-6, equal_nan=True, err_msg=name
        )
    np.testing.assert_allclose(
        [result["metrics"][name] for name in ("wns_ps", "tns_ps", "worst_slack_ps")],
        [float(wns), float(tns), float(worst)],
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )


def test_graph_sweep_matches_torch_row_and_scatter_reference(native_reference):
    with native_reference():
        owner = RebufferTimingModel(_checks_base(), use_pin_offset=False)
    graph, _ = _graph(owner)

    _compare(graph, owner)
