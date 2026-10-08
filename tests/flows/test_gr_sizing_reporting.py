"""GR coverage reports use the prepared snapshot after native exclusions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamplace.flows import gr_sizing


@pytest.mark.parametrize("net_ids, filtered", [([0], 0), ([], 1)])
@pytest.mark.parametrize("auto_adjust, expected_grid", [(0, (8, 8)), (1, (32, 16))])
def test_prepare_reports_snapshot_eligibility(
    tmp_path, monkeypatch, net_ids, filtered, auto_adjust, expected_grid
):
    definition = tmp_path / "input.def"
    definition.write_text("fixture")
    params = SimpleNamespace(
        lef_input=["fixture.lef"],
        result_dir=str(tmp_path),
        def_input=str(definition),
        design_name=lambda: "fixture",
        num_threads=1,
        route_num_bins_x=8,
        route_num_bins_y=8,
        auto_adjust_bins=auto_adjust,
        gpugr_backend="cpu_pr_mt",
        dtype="float32",
    )
    placedb = SimpleNamespace(
        pin_names=["U:Y", "V:A"],
        net_names=["N"],
        node_names=["U", "V"],
        pin2net_map=np.array([0, 0]),
        net2driver_pin_map=np.array([0]),
        num_bins_y=16,
        xl=0.0,
        yl=0.0,
        xh=200.0,
        yh=100.0,
    )
    for name in (
        "node_x",
        "node_y",
        "node_size_x",
        "node_size_y",
        "pin_offset_x",
        "pin_offset_y",
        "inst_libcell_offset",
    ):
        setattr(placedb, name, np.zeros(2))
    snapshot = SimpleNamespace(
        net_ids=torch.tensor(net_ids),
        filtered_net_count=filtered,
        model_report={},
        parent=torch.zeros(len(net_ids)),
        edge_parent=torch.empty(0),
        valid_pin_ids=torch.arange(2 if net_ids else 0),
        wire_cap=torch.zeros(len(net_ids)),
    )

    route_calls = []

    def route(**kwargs):
        route_calls.append(kwargs)
        (tmp_path / "gr_parasitics").mkdir()
        return {
            "timing_route_pack": {"pin_access_model": "fixture", "layer_names": ["M2"]},
            "metrics": {"parser_cache_hit": False, "gpugr_backend": "cpu_pr_mt"},
        }

    monkeypatch.setattr(gr_sizing, "clock_net_ids", lambda db: set())
    monkeypatch.setattr(gr_sizing, "XplaceGPUGR", lambda *args: SimpleNamespace(run_gpugr=route))
    monkeypatch.setattr(
        gr_sizing.RCParameters,
        "from_lefs",
        lambda *args: SimpleNamespace(sources=[], via_cap_model="fixture"),
    )
    monkeypatch.setattr(
        gr_sizing.GRParasiticsOp,
        "prepare",
        lambda *args, **kwargs: SimpleNamespace(snapshot=snapshot),
    )
    gr_sizing.prepare_gr_sizing(params, placedb)
    assert route_calls == [
        {
            "input_def": str(definition),
            "out_dir": str(tmp_path / "gr_parasitics"),
            "design_name": "fixture",
            "threads": 1,
            "route_xsize": expected_grid[0],
            "route_ysize": expected_grid[1],
            "rrr_iters": 0,
            "backend": "cpu_pr_mt",
            "keep_temp_def": True,
            "include_timing_route_pack": True,
            "profile_enabled": True,
        }
    ]
    report = placedb.gr_sizing.report
    assert {key: report[key] for key in ("eligible_nets", "filtered_nets", "covered_pins")} == {
        "eligible_nets": len(net_ids),
        "filtered_nets": filtered,
        "covered_pins": 2 if net_ids else 0,
    }
