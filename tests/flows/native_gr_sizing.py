"""Run the strict GR lifecycle on a real clocked native fixture.

Invoke through the source-build qualification launcher. Observers record calls
and state without substituting native routing, timing, mutation or legalization.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from dreamplace.flows.flow_config import resolve_flow_config
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo
from dreamplace.Params import Params
from dreamplace.PlaceObj import PlaceObj
from dreamplace.Placer import PlacementEngine
from gr_sizing_clock_gradient import check_clock_gradient
from gr_sizing_fixture import make_fixture

from chipcompiler.data import PDK, EccData, EccStep, OriginDesign, StepEnum, Workspace
from chipcompiler.tools.ecc.module import ECCToolsModule
from chipcompiler.tools.ecc_dreamplace.module import DreamplaceModule, DreamplaceRunMode


def name_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def run(output, phase):
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[5]
    fixture = make_fixture(output / "inputs")
    params = Params()
    params.fromJson(
        json.loads(
            (root / "chipcompiler/tools/ecc_dreamplace/configs/dreamplace_ecc.json").read_text()
        )
    )
    params.fromJson(
        {
            "flow_kind": phase,
            "timing_surrogate_mode": "mixed",
            "timing_rc_mode": "gr",
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
            "result_dir": str(output / "result"),
            "def_input": str(fixture / "fixture.def"),
            "verilog_input": str(fixture / "fixture.v"),
            "lef_input": [str(fixture / "fixture.lef")],
            "design_inputs": {
                "lib": [str(fixture / "fixture.lib")],
                "sdc": str(fixture / "fixture.sdc"),
                "work_dir": str(output / "native_sta"),
            },
        }
    )
    config_path = output / "dreamplace.json"
    materialized = params.toJson()
    config_path.write_text(
        json.dumps(resolve_flow_config(materialized, explicit_keys=tuple(materialized)))
    )
    workspace = Workspace(
        directory=output,
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
        data=EccData(steps={StepEnum.PLACEMENT.value: output / "result"}),
    )
    wrapper = DreamplaceModule(
        workspace, step, None, fixture / "fixture.def", fixture / "fixture.v", None, None
    )
    params = wrapper._build_params(Params, mode=DreamplaceRunMode.PLACEMENT)
    module = ECCToolsModule()
    module.init_config(str(fixture / "db.json"), output / "native", output / "feature")
    module.init_techlef(str(fixture / "fixture.lef"))
    assert module.read_def(str(fixture / "fixture.def"))
    engine = PlacementEngine(params)
    engine.setup_rawdb(module)
    calls = {"forward": 0, "rebuild": 0}
    original_forward, original_rebuild = SteinerTopo.forward, SteinerTopo.rebuild_tree
    original_setup = PlacementEngine.setup_placedb
    native_states = []
    route_inputs = []
    original_route = XplaceGPUGR.run_gpugr
    route_shapes = []
    gradient_proof = []
    original_gradient = PlaceObj.obj_and_grad_fn

    def route(owner, *args, **kwargs):
        route_inputs.append(kwargs["input_def"])
        result = original_route(owner, *args, **kwargs)
        pack = result["timing_route_pack"]
        shapes = pack["pin_shape_dbu"].reshape(-1, 4)
        route_shapes.append(dict(zip(pack["pin_names"], shapes.tolist(), strict=True)))
        return result

    def gradient(model, pos):
        if not gradient_proof:
            gradient_proof.append(check_clock_gradient(model))
        return original_gradient(model, pos)

    def setup(current):
        result = original_setup(current)
        db = current.placedb
        cells = [name_text(value) for value in db.flat_libcell_names]
        masters = {}
        for index, name in enumerate(db.node_names):
            family = int(db.inst_main_id[index])
            if family >= 0:
                cell = int(db.main_id_2_cell_id_start[family]) + int(db.inst_libcell_offset[index])
                masters[name_text(name)] = [
                    cells[cell],
                    float(db.node_size_x[index]),
                    float(db.node_size_y[index]),
                ]
        pins = {
            name_text(name): [float(db.pin_offset_x[index]), float(db.pin_offset_y[index])]
            for index, name in enumerate(db.pin_names)
        }
        native_states.append({"masters": masters, "pin_offsets": pins})
        return result

    def forward(op, *args, **kwargs):
        calls["forward"] += 1
        return original_forward(op, *args, **kwargs)

    def rebuild(op, *args, **kwargs):
        calls["rebuild"] += 1
        return original_rebuild(op, *args, **kwargs)

    SteinerTopo.forward, SteinerTopo.rebuild_tree = forward, rebuild
    PlacementEngine.setup_placedb = setup
    XplaceGPUGR.run_gpugr = route
    if phase == "sizing":
        PlaceObj.obj_and_grad_fn = gradient
    try:
        result = engine.run()
        assert result["status"] == "ok", result
        assert calls == {"forward": 0, "rebuild": 0}, calls
        model = engine.placer.last_global_place_model
        assert model.op_collections.elmore_delay_op is engine.placedb.gr_sizing.op
        assert torch.isfinite(model.op_collections.elmore_delay_op.loads["rise"]).all()
        assert len(route_inputs) == (2 if phase == "sizing" else 1), route_inputs
        best = None
        if phase == "sizing":
            assert result["gr_sizing"]["router_calls"] == 2
            assert result["gr_sizing"]["backend_caps"]["supports_apply_sizing"]
            assert engine.placer.last_placement_debug_summary["optimizer_steps"] == 50
            assert len(native_states) == 2
            before, after = (state["masters"] for state in native_states)
            changed = {name: value for name, value in after.items() if value != before[name]}
            assert changed and any("DFF" in value[0] for value in changed.values())
            assert any("H7L" in value[0] for value in changed.values())
            def_text = Path(result["gr_sizing"]["def"]).read_text()
            verilog = Path(result["gr_sizing"]["verilog"]).read_text()
            for name, (master, width, height) in after.items():
                assert f"- {name} {master}" in def_text, (name, master)
                assert master in verilog
                assert width == int(master.split("X")[1][0]) and height == 1.0
            assert native_states[0]["pin_offsets"] != native_states[1]["pin_offsets"]
            for name, (_, width, _) in changed.items():
                pin = f"{name}:Q" if name == "launch" else f"{name}:Y"
                lx, _, hx, _ = route_shapes[1][pin]
                assert hx - lx == width * 1000
                assert route_shapes[0][pin] != route_shapes[1][pin]
            best = engine.placer.last_continuous_early_stop_state
            assert best["restored"] and best["loss_source"] == "combined_timing_loss"
            best = {
                key: best[key]
                for key in (
                    "best_iteration",
                    "best_loss",
                    "restore_iteration",
                    "restored",
                    "loss_source",
                )
            }
        report = {
            "phase": phase,
            "result": result,
            "timing_flute_calls": calls,
            "native_states": native_states,
            "debug": getattr(engine.placer, "last_placement_debug_summary", None),
            "actual_route_inputs": route_inputs,
            "route_pin_shapes_dbu": route_shapes,
            "real_clock_gradient": gradient_proof,
            "best_restore": best,
            "effective": params.toJson(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        SteinerTopo.forward, SteinerTopo.rebuild_tree = original_forward, original_rebuild
        PlacementEngine.setup_placedb = original_setup
        XplaceGPUGR.run_gpugr = original_route
        PlaceObj.obj_and_grad_fn = original_gradient
        module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("sta", "sizing"), default="sta")
    args = parser.parse_args()
    run(args.output, args.phase)
