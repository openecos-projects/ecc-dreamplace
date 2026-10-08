"""An explicit GP carrier owns its timing controls, including S-only windows."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dreamplace.flows.flow_config import resolve_flow_config
from dreamplace.flows.timing_objective_policy import (
    diff_tdp_enabled,
    reset_legacy_net_weight_gate_summary,
)


@pytest.fixture
def placement_config():
    metadata = json.loads(
        (Path(__file__).resolve().parents[2] / "dreamplace/params.json").read_text()
    )
    return {name: item["default"] for name, item in metadata.items()} | {
        "flow_kind": "placement",
        "l_shape_use_ggr_topology": 0,
        "timing_opt_enabled": 1,
        "timing_opt_buffering_enabled": 0,
        "timing_opt_sizing_rounds": 10,
    }


@pytest.mark.parametrize(
    "carrier,controls",
    [
        (
            "pin2pin",
            {
                "with_sta": 1,
                "timing_eval_flag": 1,
                "diff_timing_driven_placement": 0,
                "differentiable_timing_obj": 0,
                "enable_net_weighting": 1,
                "pin2pin_net_weighting": 1,
                "net_weighting_scheme": "pin2pin",
            },
        ),
        (
            "direct_loss",
            {
                "with_sta": 1,
                "timing_eval_flag": 1,
                "diff_timing_driven_placement": 1,
                "differentiable_timing_obj": 1,
                "enable_net_weighting": 0,
                "pin2pin_net_weighting": 0,
                "net_weighting_scheme": "lilith",
            },
        ),
    ],
)
def test_carrier_overrides_old_switches_and_survives_window_defaults(
    carrier,
    controls,
    placement_config,
):
    # Leave the old switches in the opposite mode, as when switching a config.
    opposite = "direct_loss" if carrier == "pin2pin" else "pin2pin"
    previous = resolve_flow_config(
        placement_config | {"timing_placement_carrier": opposite},
        explicit_keys=("timing_placement_carrier",),
    )
    resolved = resolve_flow_config(
        previous | {"timing_placement_carrier": carrier},
        explicit_keys=("timing_placement_carrier", *controls),
    )
    assert {name: resolved[name] for name in controls} == controls
    assert (
        resolved["timing_placement_carrier"],
        resolved["timing_opt_sizing_rounds"],
        resolved["timing_opt_buffering_enabled"],
        resolved["routability_opt_flag"],
    ) == (
        carrier,
        10,
        0,
        1,
    )
    params = SimpleNamespace(**resolved)
    assert (diff_tdp_enabled(params), reset_legacy_net_weight_gate_summary(params)["enabled"]) == (
        carrier == "direct_loss",
        carrier == "pin2pin",
    )


def test_pin2pin_selection_does_not_activate_geometry_only_stage(placement_config):
    disabled = {
        "global_place_flag": 0,
        "timing_opt_enabled": 0,
        "with_sta": 0,
        "timing_eval_flag": 0,
        "diff_timing_driven_placement": 0,
        "differentiable_timing_obj": 0,
        "enable_net_weighting": 0,
        "pin2pin_net_weighting": 0,
    }
    resolved = resolve_flow_config(
        placement_config | disabled | {"timing_placement_carrier": "pin2pin"},
        explicit_keys=("timing_placement_carrier", *disabled),
    )
    assert {name: resolved[name] for name in disabled} == disabled


def test_invalid_carrier_is_rejected_before_selecting_controls(placement_config):
    with pytest.raises(ValueError, match="timing_placement_carrier must be"):
        resolve_flow_config(
            placement_config | {"timing_placement_carrier": "unknown"},
            explicit_keys=("timing_placement_carrier",),
        )


def test_legacy_timing_opt_fields_normalize_to_canonical(placement_config):
    resolved = resolve_flow_config(
        placement_config
        | {
            "timing_opt_enabled": 0,
            "inflation_s5b1_enabled": 1,
            "inflation_sizing_rounds": 10,
        },
        explicit_keys=("inflation_s5b1_enabled", "inflation_sizing_rounds"),
    )
    assert resolved["timing_opt_enabled"] == 1
    assert resolved["timing_opt_sizing_rounds"] == 10
    assert "inflation_s5b1_enabled" not in resolved
    assert "inflation_sizing_rounds" not in resolved


def test_conflicting_timing_opt_field_names_are_rejected(placement_config):
    with pytest.raises(ValueError, match="conflicting timing optimization parameters"):
        resolve_flow_config(
            placement_config
            | {"timing_opt_enabled": 1, "inflation_s5b1_enabled": 0},
            explicit_keys=("timing_opt_enabled", "inflation_s5b1_enabled"),
        )


def test_legacy_options_serialize_canonically_and_allow_internal_disable():
    from dreamplace.Params import Params

    params = Params()
    params.fromJson({"inflation_s5b1_enabled": 1, "inflation_sizing_rounds": 10})
    resolved = params.toJson()
    assert {
        key: value for key, value in resolved.items() if key.startswith("timing_opt_")
    } == {
        "timing_opt_enabled": 1,
        "timing_opt_buffering_enabled": 1,
        "timing_opt_max_windows": 5,
        "timing_opt_overflow_milestones": [],
        "timing_opt_sizing_rounds": 10,
    }
    assert "inflation_s5b1_enabled" not in resolved
    params.timing_opt_enabled = 0
    assert params.toJson()["timing_opt_enabled"] == 0
