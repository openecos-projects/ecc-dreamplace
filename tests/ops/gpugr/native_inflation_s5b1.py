"""Bounded native GP qualification with the natural inflation threshold.

Invoke explicitly with the clocked fixture manifest and placement RC Tcl.
Observers wrap production entry points without changing loss, counts or maps.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from dreamplace.flows.flow_config import resolve_flow_config
from dreamplace.flows.inflation_s5b1 import InflationS5B1
from dreamplace.ops.routability.cooptimization_area import capture_area
from dreamplace.PlaceObj import PlaceObj
from native_refresh_route import make_engine, snapshot


def run(args):
    logging.basicConfig(level=logging.INFO)
    args.backend = "ecc"
    overrides = {
        "flow_kind": "placement", "timing_opt_enabled": 1,
        "global_place_flag": 1, "legalize_flag": 0, "detailed_place_flag": 0,
        "buffering_fixed_buffer_master": "BUFX4H7L", "random_center_init_flag": 0,
        "num_bins_x": 32, "num_bins_y": 32, "stop_overflow": 0.,
        "placement_initial_learning_rate_max": .1,
        "gpugr_bottom_routing_layer": "MET2", "gpugr_top_routing_layer": "MET3",
        "gpugr_final_eval_flag": 0, "gpugr_area_adjust_congestion_mode": "max_hv_effective",
        "l_shape_update_interval": 15, "l_shape_auto_disable_flag": 0,
        "l_shape_use_ggr_topology": 0, "enhanced_inflation_flag": 1,
        "max_num_area_adjust": 2, "enhanced_inflation_max_rounds": 2,
        "legalize_before_each_inflation_flag": 0,
        "adjust_pin_area_flag": 0, "area_adjust_stop_ratio": 0.,
        "route_area_adjust_stop_ratio": 0., "RePlAce_skip_energy_flag": 0,
        "global_place_stages": [{
            "num_bins_x": 32, "num_bins_y": 32, "iteration": 300,
            "learning_rate": .01, "wirelength": "weighted_average", "optimizer": "adam",
            "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1,
            "learning_rate_decay": 1.,
        }],
    }
    engine, module = make_engine(
        args, resolve_flow_config(overrides, explicit_keys=tuple(overrides))
    )
    before = snapshot(engine.placedb)["counts"]
    windows, densities, steps = [], [], []
    observed_model = [None]
    original_window = InflationS5B1.before_inflation
    original_weight = PlaceObj.initialize_density_weight
    original_gradient = PlaceObj.obj_and_grad_fn
    original_step = torch.optim.Adam.step

    def window(owner, model, pos, **kwargs):
        overflow = float(model.overflow)
        result = original_window(owner, model, pos, **kwargs)
        if result is not None:
            assert overflow < .15
            assert snapshot(engine.placedb)["counts"] == before
            windows.append({"trigger_overflow": overflow, **result})
        return result

    def weight(model, params, db):
        result = original_weight(model, params, db)
        assert model._timing_geometry_cache is None
        with torch.no_grad():
            expected = model.placement_density(model.data_collections.pos[0])
        torch.testing.assert_close(model.init_density, expected)
        model._timing_geometry_cache = None
        densities.append({"initial_density": float(expected), "weight": float(result)})
        return result

    def gradient(model, pos):
        result = original_gradient(model, pos)
        observed_model[0] = model
        return result

    def step(optimizer, *step_args, **kwargs):
        model = observed_model[0]
        pos = engine.placer.pos[0]
        previous = pos.detach().clone()
        result = original_step(optimizer, *step_args, **kwargs)
        assert torch.isfinite(pos).all() and torch.isfinite(pos.grad).all()
        state = model._active_segment_count_state_for_virtual_density()
        steps.append({
            "displacement_norm": float((pos.detach() - previous).norm()),
            "overflow": float(model.overflow),
            "route_gradient_norm": model.l_shape_last_grad_norm,
            "virtual_count": 0 if state is None else int(state.z_param.detach().sum()),
        })
        return result

    InflationS5B1.before_inflation = window
    PlaceObj.initialize_density_weight = weight
    PlaceObj.obj_and_grad_fn = gradient
    torch.optim.Adam.step = step
    try:
        terminal_result = engine.run() if args.terminal else None
        if not args.terminal:
            engine.place()
        assert windows, "No natural inflation attempt triggered S5B1"
        assert len(windows) <= 5
        assert len(steps) > 1 and any(s["displacement_norm"] > 0 for s in steps)
        if args.terminal:
            assert terminal_result["status"] == "ok"
            assert terminal_result["inflation_s5b1_terminal"]["new_gp_steps"] == 0
            area = terminal_result["inflation_s5b1_terminal"]["gp_area"]
        else:
            owner = engine.placer.inflation_s5b1
            area = capture_area(
                engine.placer.data_collections, engine.placedb,
                owner.virtual_area(observed_model[0]), capacity=owner.capacity,
            ).__dict__
        report = {
            "qualification": "production GP; original SDC, RC, route maps and 0.15 trigger",
            "overrides": overrides, "windows": windows, "density_initializations": densities,
            "steps": steps, "final_area": area, "terminal_result": terminal_result,
            "native_counts_unchanged_during_gp": before,
            "debug": engine.placer.last_placement_debug_summary,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        InflationS5B1.before_inflation = original_window
        PlaceObj.initialize_density_weight = original_weight
        PlaceObj.obj_and_grad_fn = original_gradient
        torch.optim.Adam.step = original_step
        module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--terminal", action="store_true")
    run(parser.parse_args())
