"""Check weight recalibration after a real native GP inflation boundary.

Run with --manifest <ECC fixture> --rc-tcl <matched RC> --output <new directory>.
The probe stops after the first post-inflation gradient, without a QoR claim.
"""

import argparse
import json
import math
from pathlib import Path

import dreamplace.PlaceObj as place_obj
from dreamplace.flows.flow_config import resolve_flow_config
from dreamplace.ops.routability.routability_controller import RoutabilityController
from native_refresh_route import make_engine


class ProbeComplete(Exception):
    pass


def run(args):
    events = []
    originals = (
        RoutabilityController.record_inflation,
        place_obj.PlaceObj.reset_l_shape_weight_state,
        place_obj.apply_l_shape_gradient,
    )

    def inflation(controller):
        originals[0](controller)
        events.append({"event": "inflation", "gp_steps": controller._gp_steps})

    def reset(model):
        pending = bool(events) and events[-1]["event"] == "inflation"
        before = model._l_shape_weight_initialized
        originals[1](model)
        if pending:
            assert not model._l_shape_weight_initialized
            events.append({
                "event": "reset", "initialized_before": before,
                "density_weight": float(model.density_weight.item()),
            })

    def gradient(model, pos, objective):
        pending = bool(events)
        if pending:
            assert events[-1]["event"] == "reset", events
            assert not model._l_shape_weight_initialized
        result = originals[2](model, pos, objective)
        if pending and model._l_shape_weight_initialized:
            assert math.isclose(
                model.l_shape_last_weight_candidate, model.l_shape_last_target_weight,
                rel_tol=1e-6, abs_tol=1e-9,
            )
            events.append({
                "event": "calibration", "weight": model.l_shape_last_weight,
                "candidate": model.l_shape_last_weight_candidate,
                "target": model.l_shape_last_target_weight,
                "gradient_ratio": model.l_shape_last_grad_ratio,
            })
            raise ProbeComplete
        return result

    overrides = {
        "flow_kind": "placement", "timing_opt_enabled": 0,
        "placement_sizing_mode": "place_only", "sizing_parameterization": "logits",
        "with_sta": 0, "timing_eval_flag": 0, "diff_timing_driven_placement": 0,
        "differentiable_timing_obj": 0, "routability_opt_flag": 1,
        "l_shape_routability_flag": 1, "l_shape_gradient_mode": "translation",
        "l_shape_capacity_al_enable": 0, "l_shape_use_ggr_topology": 0,
        "l_shape_overflow_threshold": 1.1, "l_shape_auto_disable_flag": 0,
        "l_shape_keep_during_inflation": 1, "l_direction_use_gpugr": 1,
        "l_shape_grad_target_ratio": 0.1, "l_shape_grad_target_ratio_max": 0.1,
        "adjust_gpugr_area_flag": 1, "node_area_adjust_overflow": 1.1,
        "max_num_area_adjust": 1, "inflation_min_interval": 0,
        "enable_fillers": 1, "gpugr_bottom_routing_layer": "MET2",
        "gpugr_top_routing_layer": "MET3", "gpugr_final_eval_flag": 0,
        "global_place_flag": 1, "legalize_flag": 1, "detailed_place_flag": 0,
        "stop_overflow": 0.0, "placement_initial_learning_rate_max": 0.1,
        "global_place_stages": [{
            "num_bins_x": 32, "num_bins_y": 32, "iteration": 30,
            "learning_rate": 0.01, "wirelength": "weighted_average",
            "optimizer": "adam", "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1, "learning_rate_decay": 1.0,
        }],
    }
    engine, module = make_engine(
        args, resolve_flow_config(overrides, explicit_keys=tuple(overrides)),
    )
    RoutabilityController.record_inflation = inflation
    place_obj.PlaceObj.reset_l_shape_weight_state = reset
    place_obj.apply_l_shape_gradient = gradient
    try:
        try:
            engine.place()
        except ProbeComplete:
            pass
        assert [event["event"] for event in events] == ["inflation", "reset", "calibration"]
        (args.output / "report.json").write_text(json.dumps({
            "status": "qualified", "scope": "native GP inflation lifecycle",
            "termination": "intentional stop after post-inflation calibration",
            "events": events,
        }, indent=2) + "\n")
    finally:
        RoutabilityController.record_inflation = originals[0]
        place_obj.PlaceObj.reset_l_shape_weight_state = originals[1]
        place_obj.apply_l_shape_gradient = originals[2]
        module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(backend="ecc")
    run(parser.parse_args())
