"""Flow-level unit and native parity tests for sequential incremental STA sizing."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows import incremental_sizing_filter
from dreamplace.flows.incremental_sizing_filter import IncrementalSizingGate
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo
from dreamplace.Params import Params
from dreamplace.Placer import PlacementEngine
from gr_sizing_fixture import make_fixture

from chipcompiler.data import PDK, EccData, EccStep, OriginDesign, StepEnum, Workspace
from chipcompiler.data.workspace.timing_constraints import generate_timing_scripts
from chipcompiler.tools.ecc.module import ECCToolsModule
from chipcompiler.tools.ecc_dreamplace.module import DreamplaceModule, DreamplaceRunMode


def test_incremental_gate_ranks_by_score_and_advances_past_rejected_chunk(monkeypatch):
    """Verify score-ascending evaluation order, canonical output order, and no-op advancement."""
    observed_orders = []

    def fake_filter(
        owner,
        model,
        pos,
        geometry,
        capacity,
        base_size,
        base_vt,
        next_size,
        next_vt,
        summary,
        *,
        size_parameterization="real_size",
    ):
        ranked = sorted(
            summary["selected_scores"],
            key=lambda item: (float(item["predicted_delta_obj"]), int(item["instance_id"])),
        )
        observed_orders.append([int(item["instance_id"]) for item in ranked])
        # Reject all in round 1 ([2, 0]); accept instance 1 in round 2 ([3, 1]).
        accepted = [1] if 1 in observed_orders[-1] else []
        accepted_cells = [5] if accepted else []
        report = {
            "status": "ok",
            "input_count": len(ranked),
            "evaluated_count": len(ranked),
            "accepted_count": len(accepted),
            "rejected_area_count": 0,
            "rejected_loss_count": len(ranked) - len(accepted),
            "accepted_instance_ids": accepted,
            "accepted_cell_ids": accepted_cells,
        }
        return incremental_sizing_filter._rebuild_filtered_outputs(
            owner.data,
            base_size,
            base_vt,
            summary,
            accepted,
            accepted_cells,
            report,
            size_parameterization=size_parameterization,
        )

    monkeypatch.setattr(
        incremental_sizing_filter,
        "filter_sizing_candidates_incrementally",
        fake_filter,
    )

    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0, 0, 0, 0]),
        inst_size_lower=torch.ones(4),
        inst_size_upper=torch.full((4,), 4.0),
    )
    owner = SimpleNamespace(data=data, params=SimpleNamespace())
    gate = IncrementalSizingGate(owner, SimpleNamespace(), None, capacity=100.0)

    base_size = torch.ones(4, dtype=torch.float64)
    base_vt = torch.zeros((4, 2), dtype=torch.float64)
    valid = torch.ones((4, 4), dtype=torch.bool)
    blocked = torch.zeros(4, dtype=torch.bool)

    # Round 1: summary has instance-id order [0, 2], but instance 2 has stronger score (-9 vs -4).
    round1_summary = {
        "num_changed_instances": 2,
        "action_counts": {"size_up": 2, "size_down": 0, "vt_slowdown": 0, "vt_speedup": 0},
        "selected_scores": [
            {
                "instance_id": 0,
                "action": "size_up",
                "target_cell_id": 2,
                "target_size": 2.0,
                "target_vt": 0,
                "predicted_delta_obj": -4.0,
            },
            {
                "instance_id": 2,
                "action": "vt_speedup",
                "target_cell_id": 3,
                "target_size": 1.0,
                "target_vt": 1,
                "predicted_delta_obj": -9.0,
            },
        ],
    }
    r1_size, _, r1_out = gate.filter_round(
        torch.zeros(8), base_size, base_vt, base_size.clone(), base_vt.clone(), round1_summary
    )
    assert observed_orders[0] == [2, 0]
    assert r1_out["num_changed_instances"] == 0
    torch.testing.assert_close(r1_size, base_size)

    # Because round 1 accepted 0 moves, those exact (inst, action) pairs are masked for round 2.
    masked_valid = gate.mask_valid_actions(valid)
    assert bool(masked_valid[0, 0]) is False  # (0, size_up) masked
    assert bool(masked_valid[0, 3]) is True   # other actions on inst 0 remain open
    assert bool(masked_valid[2, 3]) is False  # (2, vt_speedup) masked
    masked_blocked = gate.mask_blocked_instances(blocked)
    assert masked_blocked.tolist() == [True, False, True, False]

    # Round 2 explores [1, 3]; instance 1 is accepted, which resets the unchanged-state mask!
    round2_summary = {
        "num_changed_instances": 2,
        "action_counts": {"size_up": 2, "size_down": 0, "vt_slowdown": 0, "vt_speedup": 0},
        "selected_scores": [
            {
                "instance_id": 1,
                "action": "size_up",
                "target_cell_id": 5,
                "target_size": 3.0,
                "target_vt": 1,
                "predicted_delta_obj": -2.0,
            },
            {
                "instance_id": 3,
                "action": "size_up",
                "target_cell_id": 6,
                "target_size": 2.0,
                "target_vt": 0,
                "predicted_delta_obj": -5.0,
            },
        ],
    }
    r2_size, _, r2_out = gate.filter_round(
        torch.zeros(8), base_size, base_vt, base_size.clone(), base_vt.clone(), round2_summary
    )
    assert observed_orders[1] == [3, 1]
    assert r2_out["applied_instance_ids"] == [1]
    assert r2_out["applied_cell_ids"] == [5]
    assert r2_size.tolist() == [1.0, 3.0, 1.0, 1.0]
    assert torch.equal(gate.mask_valid_actions(valid), valid)
    assert torch.equal(gate.mask_blocked_instances(blocked), blocked)


@pytest.mark.parametrize("timing_rc_mode", ["gr", "placement"])
def test_native_diff_sizing_incremental_commit_matches_placeobj_loss(tmp_path, timing_rc_mode):
    """Run real native diff_sizing (DFF + BUF) and verify exact C++/PlaceObj loss parity."""
    repo_root = Path(__file__).resolve().parents[5]
    fixture = make_fixture(tmp_path / "inputs")
    base_cfg = json.loads(
        (
            repo_root
            / "chipcompiler/tools/ecc_dreamplace/configs/dreamplace_ecc.json"
        ).read_text()
    )
    base_cfg.update(
        {
            "timing_sizing_incremental_commit": 1,
            "diff_sizing_timing_rc_mode": timing_rc_mode,
            "diff_sizing_optimization_flows": [
                ["quad", 3, [500.0, 5.0, 1.0, 1.0], 50.0],
            ],
            "place_io_engine": "ecc",
            "gpugr_backend": "cpu_pr_mt",
            "route_num_bins_x": 10,
            "route_num_bins_y": 10,
            "gpugr_bottom_routing_layer": "M2",
            "gpugr_top_routing_layer": "M3",
            "base_design_name": "gr_fixture",
            "num_threads": 2,
            "num_bins_x": 8,
            "num_bins_y": 8,
            "cell_padding_x": 0,
            "result_dir": str(tmp_path / "result"),
            "def_input": str(fixture / "fixture.def"),
            "verilog_input": str(fixture / "fixture.v"),
            "lef_input": [str(fixture / "fixture.lef")],
            "design_inputs": {
                "lib": [str(fixture / "fixture.lib")],
                "sdc": str(fixture / "fixture.sdc"),
                "work_dir": str(tmp_path / "native_sta"),
            },
        }
    )
    config_path = tmp_path / "dreamplace.json"
    config_path.write_text(json.dumps(base_cfg))
    workspace = Workspace(
        directory=tmp_path,
        design=OriginDesign(name="gr_fixture"),
        pdk=PDK(
            name="gr_fixture",
            tech=fixture / "fixture.lef",
            libs=[fixture / "fixture.lib"],
            sdc=fixture / "fixture.sdc",
        ),
        config={"dreamplace": config_path},
    )
    step = EccStep(
        name=StepEnum.PLACEMENT.value,
        data=EccData(steps={StepEnum.PLACEMENT.value: tmp_path / "result"}),
    )
    wrapper = DreamplaceModule(
        workspace,
        step,
        None,
        fixture / "fixture.def",
        fixture / "fixture.v",
        None,
        None,
    )
    run_params = wrapper._build_params(Params, mode=DreamplaceRunMode.DIFF_SIZING)

    module = ECCToolsModule()
    module.init_config(
        str(fixture / "db.json"), tmp_path / "native", tmp_path / "feature"
    )
    module.init_techlef(str(fixture / "fixture.lef"))
    assert module.read_def(str(fixture / "fixture.def"))

    steiner_calls = {"forward": 0, "rebuild": 0}
    orig_forward, orig_rebuild = SteinerTopo.forward, SteinerTopo.rebuild_tree

    def count_forward(op, *args, **kwargs):
        steiner_calls["forward"] += 1
        return orig_forward(op, *args, **kwargs)

    def count_rebuild(op, *args, **kwargs):
        steiner_calls["rebuild"] += 1
        return orig_rebuild(op, *args, **kwargs)

    SteinerTopo.forward, SteinerTopo.rebuild_tree = count_forward, count_rebuild
    try:
        engine = PlacementEngine(run_params)
        engine.setup_rawdb(module)
        result = engine.run()
    finally:
        SteinerTopo.forward, SteinerTopo.rebuild_tree = orig_forward, orig_rebuild
        module.close()

    assert result["status"] == "ok"
    if timing_rc_mode == "gr":
        assert steiner_calls == {"forward": 0, "rebuild": 0}

    sizing = result["diff_sizing"]["sizing"]
    rounds = sizing["rounds"]
    assert len(rounds) == 3
    assert sizing["best_loss"] < sizing["initial_loss"]

    committed_rounds = [r for r in rounds if not r.get("no_action", False)]
    assert committed_rounds, "expected at least one accepted sizing round"
    for r in committed_rounds:
        inc = r["incremental_commit"]
        assert inc["status"] == "ok"
        assert inc["accepted_count"] == r["changed_instance_count"]
        # Exact parity between C++ IncrementalSizingEngine and Python PlaceObj._timing_loss!
        assert inc["final_loss"] == pytest.approx(r["loss"], rel=1e-6, abs=1e-6)
        assert inc["final_loss"] < inc["initial_loss"]

    if len(rounds) >= 2 and rounds[1]["incremental_commit"].get("input_count", 0) > 0:
        assert rounds[1]["incremental_commit"]["cache_hit"] is True


def test_native_spef_diff_sizing_uses_propagated_post_cts_sta_and_matches_placeobj(tmp_path):
    """Verify post-route .spef.gz diff_sizing feeds post_CTS SDC + SPEF STA and keeps parity."""
    repo_root = Path(__file__).resolve().parents[5]
    fixture = make_fixture(tmp_path / "inputs")

    # Give H7L distinct leakage (so VT classification separates R vs L) and give DFFX1H7R/L
    # distinct setup constraints (80ps vs 10ps) so swapping capture FF changes check_arcs_ by 70ps!
    lib_path = fixture / "fixture.lib"
    lib_text = lib_path.read_text(encoding="utf-8")
    old_check = (
        'rise_constraint(check) {values("0.01, 0.02", "0.02, 0.03");} '
        'fall_constraint(check) {values("0.01, 0.02", "0.02, 0.03");}'
    )
    r_check = (
        'rise_constraint(check) {values("0.08, 0.10", "0.10, 0.12");} '
        'fall_constraint(check) {values("0.08, 0.10", "0.10, 0.12");}'
    )
    for name, leak in (("BUFX1H7L", 1.5), ("BUFX2H7L", 3.0), ("DFFX1H7L", 1.5), ("DFFX2H7L", 3.0)):
        size = 1 if "X1" in name else 2
        lib_text = lib_text.replace(
            f"cell({name}) {{area: {size}; cell_leakage_power: {size};",
            f"cell({name}) {{area: {size}; cell_leakage_power: {leak};",
        )
    dff_r_body = lib_text.split("cell(DFFX1H7R) {")[1].split("cell(DFFX1H7L)")[0]
    lib_path.write_text(
        lib_text.replace(
            f"cell(DFFX1H7R) {{{dff_r_body}",
            f"cell(DFFX1H7R) {{{dff_r_body.replace(old_check, r_check)}",
        ),
        encoding="utf-8",
    )

    # Add a virtual I/O clock so complete STA compiles Schema 3 SparseTimingView.
    (fixture / "fixture.sdc").write_text(
        "create_clock -name clk -period 0.8 [get_ports clk]\n"
        "create_clock -name vclk -period 0.8\n"
        "set_input_delay 0.01 -clock vclk [get_ports in]\n"
        "set_output_delay 0.01 -clock clk [get_ports out]\n"
        "set_input_transition 0.02 [get_ports in]\nset_load 0.02 [get_ports out]\n",
        encoding="utf-8",
    )

    spef_text = """*SPEF "IEEE 1481-1999"
*DESIGN "gr_fixture"
*DATE "Thu Nov 12 00:00:00 2026"
*VENDOR "ECC"
*PROGRAM "test"
*VERSION "1.0"
*DESIGN_FLOW "PIN_CAP NONE"
*DIVIDER /
*DELIMITER :
*BUS_DELIMITER []
*T_UNIT 1 NS
*C_UNIT 1 PF
*R_UNIT 1 KOHM
*L_UNIT 1 HENRY

*PORTS
clk I
in I
out O

*D_NET clk 0.06
*CONN
*P clk I
*I launch:CK I
*I capture:CK I
*CAP
1 launch:CK 0.01
2 capture:CK 0.05
*RES
1 clk launch:CK 0.1
2 clk capture:CK 2.0
*END

*D_NET in 0.01
*CONN
*P in I
*I launch:D I
*CAP
1 launch:D 0.01
*RES
1 in launch:D 0.1
*END

*D_NET n1 0.02
*CONN
*I launch:Q O
*I b1:A I
*CAP
1 launch:Q 0.01
2 b1:A 0.01
*RES
1 launch:Q b1:A 0.5
*END

*D_NET n2 0.02
*CONN
*I b1:Y O
*I b2:A I
*CAP
1 b1:Y 0.01
2 b2:A 0.01
*RES
1 b1:Y b2:A 0.5
*END

*D_NET n3 0.02
*CONN
*I b2:Y O
*I capture:D I
*CAP
1 b2:Y 0.01
2 capture:D 0.01
*RES
1 b2:Y capture:D 0.5
*END

*D_NET out 0.01
*CONN
*I capture:Q O
*P out O
*CAP
1 out 0.01
*RES
1 capture:Q out 0.1
*END
"""
    spef_gz = fixture / "fixture.spef.gz"
    with gzip.open(spef_gz, "wt", encoding="utf-8") as stream:
        stream.write(spef_text)

    base_cfg = json.loads(
        (repo_root / "chipcompiler/tools/ecc_dreamplace/configs/dreamplace_ecc.json").read_text()
    )
    base_cfg.update(
        {
            "timing_sizing_incremental_commit": 1,
            "diff_sizing_timing_rc_mode": "spef",
            "spef_input": str(spef_gz),
            "diff_sizing_optimization_flows": [
                ["vt_fast", 4, [500.0, 5.0, 1.0, 1.0], 50.0],
            ],
            "place_io_engine": "ecc",
            "base_design_name": "gr_fixture",
            "num_threads": 2,
            "num_bins_x": 8,
            "num_bins_y": 8,
            "cell_padding_x": 0,
            "result_dir": str(tmp_path / "result"),
            "def_input": str(fixture / "fixture.def"),
            "verilog_input": str(fixture / "fixture.v"),
            "lef_input": [str(fixture / "fixture.lef")],
        }
    )
    config_path = tmp_path / "dreamplace.json"
    config_path.write_text(json.dumps(base_cfg))
    (tmp_path / "origin").mkdir()
    workspace = Workspace(
        directory=tmp_path,
        design=OriginDesign(name="gr_fixture"),
        pdk=PDK(
            name="gr_fixture",
            tech=fixture / "fixture.lef",
            libs=[lib_path],
            sdc=fixture / "fixture.sdc",
        ),
        config={"dreamplace": config_path},
    )
    generate_timing_scripts(workspace)
    step = EccStep(
        name=StepEnum.DIFF_SIZING.value,
        data=EccData(steps={StepEnum.DIFF_SIZING.value: tmp_path / "result"}),
    )
    wrapper = DreamplaceModule(
        workspace,
        step,
        None,
        fixture / "fixture.def",
        fixture / "fixture.v",
        None,
        None,
    )
    run_params = wrapper._build_params(Params, mode=DreamplaceRunMode.DIFF_SIZING)
    assert (
        run_params.design_inputs["sdc"]
        == tmp_path / "origin/post_CTS/fixture.sdc.ecc.tcl"
    )
    assert run_params.design_inputs["spef"] == str(spef_gz)
    assert run_params.design_inputs["timing_mode"] == "complete"

    module = ECCToolsModule()
    module.init_config(str(fixture / "db.json"), tmp_path / "native", tmp_path / "feature")
    module.init_techlef(str(fixture / "fixture.lef"))
    assert module.read_def(str(fixture / "fixture.def"))
    try:
        engine = PlacementEngine(run_params)
        engine.setup_rawdb(module)
        # Asymmetric SPEF RC on clk (launch 0.1*0.01 vs capture 2.0*0.05) yields real clock skew!
        assert engine.placedb.rawdb.sparse_timing_view is not None
        clk_aat = list(engine.placedb.rawdb.pydb.clk_pin_r_aat)
        assert len(clk_aat) == 2 and clk_aat[0] != clk_aat[1] and max(clk_aat) > 0.0
        result = engine.run()
    finally:
        module.close()

    assert result["status"] == "ok"
    sizing = result["diff_sizing"]["sizing"]
    assert sizing["best_loss"] < sizing["initial_loss"]
    committed_rounds = [r for r in sizing["rounds"] if not r.get("no_action", False)]
    assert committed_rounds
    # Capture register (instance 3) must be among accepted VT swaps and match PlaceObj._timing_loss!
    accepted_all = {
        inst_id
        for r in committed_rounds
        for inst_id in r["incremental_commit"]["accepted_instance_ids"]
    }
    assert 3 in accepted_all
    for r in committed_rounds:
        inc = r["incremental_commit"]
        assert inc["status"] == "ok"
        assert inc["final_loss"] == pytest.approx(r["loss"], rel=1e-6, abs=1e-6)

