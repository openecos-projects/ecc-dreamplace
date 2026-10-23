"""Terminal GR finalize must not invoke a second GPUGR."""

from types import SimpleNamespace
from pathlib import Path

import torch
from dreamplace.flows import gr_sizing


def test_finalize_reuses_optimization_lane_route(tmp_path, monkeypatch):
    output_dir = tmp_path / "gr_parasitics"
    output_dir.mkdir()

    tree = SimpleNamespace(
        pin_to_vertex=torch.tensor([0, 1]),
        pin_grid_vertex=torch.tensor([0, 1]),
        parent=torch.zeros(2),
        edge_parent=torch.empty(0),
        valid_pin_ids=torch.arange(2),
        net_ids=torch.tensor([0]),
        filtered_net_count=0,
        wire_cap=torch.zeros(1),
        model_report={},
    )
    session = gr_sizing.GRSizingSession(
        op=SimpleNamespace(snapshot=tree, forward_calls=3, forward_seconds=1.5),
        report={"router_calls": 1, "timing_rc_mode": "gr"},
        pin_names={"U:Y": 0, "V:A": 1},
    )

    def boom(*args, **kwargs):
        raise AssertionError("finalize must not construct XplaceGPUGR / run_gpugr")

    monkeypatch.setattr(gr_sizing, "XplaceGPUGR", boom)

    class FakeParasiticsOp:
        def __init__(self, snapshot):
            self.snapshot = snapshot
            self.forward_calls = 0
            self.forward_seconds = 0.0

    monkeypatch.setattr(gr_sizing, "GRParasiticsOp", FakeParasiticsOp)

    class FakePlacer:
        def __init__(self, params, placedb, _):
            assert placedb.gr_sizing is not None
            assert placedb.gr_sizing.report["reused_optimization_lane_route"] is True
            assert placedb.gr_sizing.report["router_calls"] == 0
            self.pos = [torch.zeros(2)]
            self.last_global_place_model = SimpleNamespace(
                timing_obj=lambda pos, surrogate_mode=None: (
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    None,
                    None,
                ),
                op_collections=SimpleNamespace(
                    elmore_delay_op=None,
                    timing_propagation_op=SimpleNamespace(
                        last_total_slew_violation=0.0,
                        last_total_cap_violation=0.0,
                    ),
                ),
            )

        def run_sta_only(self, params, placedb):
            return float("inf"), float("inf"), {
                "timing_stage_summary": {
                    "sta": {"wns": 0.0, "tns": 0.0, "timing_objective": 0.0}
                }
            }

    import dreamplace.NonLinearPlace as nlp_mod

    monkeypatch.setattr(nlp_mod, "NonLinearPlace", FakePlacer)

    import dreamplace.flows.gr_sizing_config as gr_cfg
    import dreamplace.flows.flow_config as flow_cfg

    monkeypatch.setattr(gr_cfg, "configure_gr_sizing", lambda params: None)
    monkeypatch.setattr(flow_cfg, "apply_flow_defaults", lambda params: None)

    class FakeEngine:
        def __init__(self, params=None):
            self.params = params or SimpleNamespace(
                timing_rc_mode="gr",
                flow_kind="sizing",
                result_dir=str(tmp_path),
                design_inputs={},
            )
            self.placedb = SimpleNamespace(
                gr_sizing=session,
                pin_names=["U:Y", "V:A"],
                rawdb=SimpleNamespace(backend_caps=SimpleNamespace(to_dict=lambda: {})),
            )
            self.data_manager = object()

        def write_back(self, path):
            Path(path).write_text("def")

        def write_verilog(self, path):
            Path(path).write_text("v")

        def setup_rawdb(self, data_manager):
            return None

        def setup_placedb(self):
            self.placedb = SimpleNamespace(
                pin_names=["U:Y", "V:A"],
                gr_sizing=None,
                rawdb=SimpleNamespace(backend_caps=SimpleNamespace(to_dict=lambda: {})),
            )

    engine = FakeEngine()
    result = gr_sizing.finalize_gr_sizing(engine, {"status": "ok"})

    assert result["gr_sizing"]["router_calls"] == 1
    assert result["gr_sizing"]["terminal_snapshot"]["reused_optimization_lane_route"] is True
    assert (output_dir / "sizing.json").is_file()
