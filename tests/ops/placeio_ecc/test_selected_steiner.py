"""Selected placement trees must feed both Python TP and native repair."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from dreamplace.ops.rebuffer.payloads import steiner_payload, timing_payload
from dreamplace.ops.rebuffer.steiner_payload import build_openroad_steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from test_timing_carrier import _base


def _with_offsets():
    base = _base()
    base.pin_offset_x = [0.0, 1000.0, 2000.0, 0.0]
    base.pin_offset_y = [0.0, 3000.0, 4000.0, 0.0]
    return base


@pytest.mark.parametrize("use_pin_offset", [False, True])
def test_selected_tree_feeds_actual_rc_tp_and_planner_without_rebuild(monkeypatch, use_pin_offset):
    base = _with_offsets()
    units = {"r_unit": 0.653, "c_unit": 1.8889e-4}
    model = RebufferTimingModel(
        base, steiner_provider="openroad_stt", use_pin_offset=use_pin_offset, **units
    )
    # These two-pin trees have independently known Manhattan lengths.
    lengths = np.array([14.0, 92.0] if use_pin_offset else [10.0, 90.0])
    expected_load = lengths * units["c_unit"] + [0.005, 0.0]
    expected_delay = units["r_unit"] * lengths * (lengths * units["c_unit"] / 2 + [0.005, 0.0])
    np.testing.assert_allclose(model.loads["rise"][[0, 2]], expected_load, rtol=1e-14)
    np.testing.assert_allclose(model.delays["rise"][[1, 3]], expected_delay, rtol=1e-14)

    def forbidden(*args, **kwargs):
        raise AssertionError("planner rebuilt the selected tree")

    monkeypatch.setattr(
        "dreamplace.ops.rebuffer.steiner_payload.build_openroad_stt_tree", forbidden
    )
    payload = steiner_payload(base, model, provider="openroad_stt", r_unit=units["r_unit"])
    np.testing.assert_array_equal(payload["node_x"], model.new_x)
    np.testing.assert_array_equal(payload["node_y"], model.new_y)
    np.testing.assert_array_equal(payload["edge_resistance"], model.rc.flat_pin_to_res)
    np.testing.assert_array_equal(payload["node_capacitance"], model.rc.net_cap)
    np.testing.assert_array_equal(timing_payload(base, model)["pin_load"], model.loads["rise"])
    assert payload["input_identity"] == model.steiner_input_identity

    frozen = RebufferTimingModel(
        base, use_pin_offset=use_pin_offset, steiner_view=model.steiner_view, **units
    )
    assert frozen.baseline == model.baseline
    for name in ("loads", "delays", "impulses"):
        for transition in ("rise", "fall"):
            assert torch.equal(getattr(frozen, name)[transition], getattr(model, name)[transition])


@pytest.mark.parametrize(
    "field",
    [
        "node_x",
        "pin_offset_x",
        "flat_net2pin_map",
        "inst_libcell_offset",
        "flat_lib_pin_rcap",
        "outcaps",
    ],
)
def test_selected_tree_rejects_stale_source(field):
    base = _with_offsets()
    model = RebufferTimingModel(base, steiner_provider="openroad_stt")
    getattr(base, field)[0] += 1
    with pytest.raises(ValueError, match="Steiner input identity"):
        steiner_payload(base, model, provider="openroad_stt", r_unit=model.r_unit)


def test_selected_tree_rejects_mismatched_provider_rc_and_pydb():
    base = _base()
    model = RebufferTimingModel(base, steiner_provider="openroad_stt", r_unit=0.653)
    with pytest.raises(ValueError, match="Steiner provider"):
        steiner_payload(base, model, provider="ecc", r_unit=0.653)
    with pytest.raises(ValueError, match="r_unit"):
        steiner_payload(base, model, provider="openroad_stt", r_unit=0.914)
    with pytest.raises(ValueError, match="PyPlaceDB"):
        steiner_payload(deepcopy(base), model, provider="openroad_stt", r_unit=0.653)
    legacy = RebufferTimingModel(base)
    with pytest.raises(ValueError, match="Steiner provider"):
        steiner_payload(base, legacy, provider="openroad_stt")


def test_openroad_tree_is_a_complete_rc_input_with_expanded_sources():
    base = _base()
    view = build_openroad_steiner_payload(base)
    np.testing.assert_array_equal(view["flat_pin_from"], [0, 2])
    frozen = RebufferTimingModel(base, steiner_view=view)
    assert frozen.baseline.endpoint_count == 1


@pytest.mark.parametrize("supplied_model", [False, True])
def test_direct_design_planner_reuses_selected_python_tp_inputs(supplied_model):
    from dreamplace.ops.rebuffer.repair_design import plan_repair_design_from_pydb

    base = _with_offsets()
    config = {"steiner_provider": "openroad_stt", "r_unit": 0.653, "c_unit": 1.8889e-4}
    model = RebufferTimingModel(base, **config)
    plan = plan_repair_design_from_pydb(
        base, model if supplied_model else None, config=None if supplied_model else config
    )
    assert plan["status"] == "ok", plan
    assert plan["baseline"] == model.baseline.as_dict()
    assert plan["steiner_input_identity"] == model.steiner_input_identity


def test_expanded_sources_cover_branching_and_coincident_terminal():
    base = _base()
    # This fixture exercises the CSR transport, independent of timing schema.
    base.pin_names = ["driver", "load0", "load1", "load2"]
    base.net_names = ["n"]
    base.pin2node_map = [0, 1, 2, 3]
    base.node_x, base.node_y = [0.0, 100.0, 100.0, 100.0], [0.0, 100.0, -100.0, -100.0]
    base.pin_offset_x = base.pin_offset_y = [0.0] * 4
    base.flat_net2pin_map, base.flat_net2pin_start_map = [0, 1, 2, 3], [0, 4]
    base.net2driver_pin_map = [0]
    view = build_openroad_steiner_payload(base)
    assert view["status"] == "ok"
    sources, sinks = np.array(view["flat_pin_from"]), np.array(view["flat_pin_to"])
    np.testing.assert_array_equal(np.array(view["pin_fa"])[sinks], sources)
    assert sorted(view["net_flat_topo_sort"]) == list(range(len(view["pin_fa"])))
    assert len(sources) == len(view["pin_fa"]) - 1
    assert np.max(np.bincount(sources)) > 1


def test_design_max_transition_payload_invalidates_when_only_fall_load_changes():
    base = _base()
    model = RebufferTimingModel(base)
    first = timing_payload(base, model, max_transition_load=True)
    np.testing.assert_array_equal(first["pin_cap"], model.lib_pin_fcaps)
    model.loads["fall"] = model.loads["fall"] + 0.1
    second = timing_payload(base, model, max_transition_load=True)
    np.testing.assert_array_equal(second["pin_load"], model.loads["fall"])
    assert np.all(second["pin_load"] > first["pin_load"])
    legacy = timing_payload(base, model)
    np.testing.assert_array_equal(legacy["pin_load"], model.loads["rise"])
