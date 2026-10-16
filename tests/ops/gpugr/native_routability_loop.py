"""Observe a bounded real ECC L-shape/inflation optimizer chain on GCD."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from native_refresh_route import make_engine
from dreamplace.PlaceObj import PlaceObj
from dreamplace.ops.routability.routability_controller import RoutabilityController


def run(args):
    logging.basicConfig(level=logging.INFO)
    args.backend = "ecc"
    overrides = {
        "with_sta": 0,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "flow_kind": "placement",
        "gp_noise_ratio": 0,
        "l_shape_routability_flag": 1,
        "l_shape_overflow_threshold": 1.0,
        "l_shape_use_ggr_topology": 1,
        "l_direction_use_gpugr": 1,
        "l_shape_capacity_al_enable": 1,
        "soft_l_assignment": 0,
        "l_shape_keep_during_inflation": 1,
        "l_shape_update_interval": 1000,
        "l_shape_auto_disable_flag": 0,
        "gpugr_final_eval_flag": 0,
        "gpugr_final_eval_rrr_iters": 0,
        "gpugr_bottom_routing_layer": "MET2",
        "gpugr_top_routing_layer": "MET3",
        "adjust_gpugr_area_flag": 1,
        "adjust_route_area_flag": 1,
        "gpugr_area_adjust_congestion_mode": "max_hv_effective",
        "area_adjust_stop_ratio": 0.0,
        "route_area_adjust_stop_ratio": 0.0,
        "stop_overflow": 0.0,
        "adjust_pin_area_flag": 0,
        "adjust_nctugr_area_flag": 0,
        "node_area_adjust_overflow": 1.0,
        "max_num_area_adjust": 1,
        "enhanced_inflation_flag": 1,
        "legalize_before_each_inflation_flag": 0,
        "global_place_stages": [
            {
                "num_bins_x": 32,
                "num_bins_y": 32,
                "iteration": 10,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "adam",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
                "learning_rate_decay": 1,
            }
        ],
    }
    engine, module = make_engine(args, overrides)
    initial_sizes = engine.placedb.node_size_x.copy()
    steps, geometry = [], []
    observed_model = [None]
    original_gradient = PlaceObj.obj_and_grad_fn
    original_step = torch.optim.Adam.step
    original_geometry = RoutabilityController.after_geometry_change

    def gradient(model, pos):
        result = original_gradient(model, pos)
        observed_model[0] = model
        return result

    def step(optimizer, *step_args, **step_kwargs):
        model = observed_model[0]
        position = engine.placer.pos[0]
        before = position.detach().clone()
        result = original_step(optimizer, *step_args, **step_kwargs)
        route_grad = getattr(model, "l_shape_last_grad_norm", None)
        if route_grad is not None and float(route_grad) > 0:
            delta = (position.detach() - before).norm().item()
            assert np.isfinite(delta) and delta > 0
            assert torch.isfinite(position.grad).all()
            steps.append(
                {
                    "route_gradient_norm": float(route_grad),
                    "position_delta_norm": delta,
                    "weight": float(model.l_shape_routability_weight.item()),
                    "position_before": before.tolist(),
                    "position_after": position.detach().tolist(),
                    "gradient": position.grad.detach().tolist(),
                }
            )
        return result

    def changed(controller, *callback_args, **kwargs):
        result = original_geometry(controller, *callback_args, **kwargs)
        model = kwargs["model"]
        geometry.append(model.data_collections.node_size_x.detach().cpu().tolist())
        return result

    PlaceObj.obj_and_grad_fn = gradient
    torch.optim.Adam.step = step
    RoutabilityController.after_geometry_change = changed
    try:
        engine.place()
    finally:
        PlaceObj.obj_and_grad_fn = original_gradient
        torch.optim.Adam.step = original_step
        RoutabilityController.after_geometry_change = original_geometry
    assert steps, "No native L-shape gradient reached a real optimizer step"
    assert geometry, "No inflation geometry update occurred"
    restored = engine.placer.data_collections.node_size_x.detach().cpu().numpy()
    np.testing.assert_allclose(restored, initial_sizes, rtol=1e-6, atol=1e-6)
    report = {
        "overrides": overrides,
        "optimizer_route_steps": steps,
        "geometry_update_count": len(geometry),
        "physical_geometry_restored": True,
        "debug": engine.placer.last_placement_debug_summary,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    run(parser.parse_args())
