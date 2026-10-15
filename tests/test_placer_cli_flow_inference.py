"""Schema defaults preserve implicit flows and explicit buffering selections."""

import json
from types import SimpleNamespace

import pytest
from dreamplace.flows.flow_config import FlowKind, infer_flow_kind
from dreamplace.placer_cli import build_arg_parser, build_effective_params_from_args


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, "placement"),
        ({"placement_sizing_mode": "size_only"}, "sizing"),
        ({"placement_sizing_mode": "joint"}, "joint"),
        ({"buffering_mode": "segment"}, "buffering"),
        ({"buffering_mode": "candidate"}, "buffering"),
        ({"buffering_continuous_relaxed_optimization": True}, "buffering"),
        ({"flow_kind": "placement", "buffering_mode": "segment"}, "placement"),
    ],
)
def test_json_flow_inference_uses_user_selection(tmp_path, config, expected):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(config))
    args = build_arg_parser().parse_args([str(path)])
    params, _ = build_effective_params_from_args(args)
    assert params.flow_kind == expected


@pytest.mark.parametrize("mode", ["segment", "candidate"])
def test_cli_buffering_selection_overrides_schema_default(mode):
    args = build_arg_parser().parse_args(["--buffering-mode", mode])
    params, _ = build_effective_params_from_args(args)
    assert (params.flow_kind, params.buffering_mode) == ("buffering", mode)


def test_default_cli_is_placement():
    params, _ = build_effective_params_from_args(build_arg_parser().parse_args([]))
    assert params.flow_kind == "placement"


def test_legacy_programmatic_buffering_selection():
    assert infer_flow_kind(SimpleNamespace(buffering_mode="segment")) == FlowKind.BUFFERING
