"""Probe the actual PlaceObj pin/Steiner/RC/timing chain at fixed legal masters."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from dreamplace.flows.hold_objective import HoldObjective


def verify_position_gradients(
    owner, model, pos, output, *, surrogate_mode="lut_only", hold_margin_ps=150.0
):
    output = Path(output)
    output.mkdir(exist_ok=False)
    op = owner.ops.timing_propagation_op
    assert owner.placedb.gr_sizing is None, "Position derivatives need live placement RC"
    saved = (
        model.params.placement_sizing_mode,
        model.params.timing_objective_lane,
        model.params.differentiable_timing_obj,
        model.hold_objective,
    )
    model.params.placement_sizing_mode = "place_only"
    model.params.timing_objective_lane = "timing_only"
    model.params.differentiable_timing_obj = 1
    topology = owner.ops.steiner_topo_op
    epoch = topology.topology_generation
    nodes = int(model.data_collections.node_size_x.numel())
    movable = int(owner.placedb.num_movable_nodes)
    base = pos.detach().clone()
    raw = {}
    report = {
        "scope": (
            "Actual fixed-master PlaceObj position -> pin position -> Steiner coordinates "
            "-> live placement RC -> timing -> setup/hold. Frozen topology, float64; "
            f"devices recorded. Two hold modes; hold margin {hold_margin_ps:g} ps. "
            "No optimizer iteration or QoR claim."
        ),
        "surrogate_mode": surrogate_mode or getattr(model.params, "timing_surrogate_mode", None),
        "surrogate_flags": list(
            op._resolve_surrogate_mode(
                surrogate_mode or getattr(model.params, "timing_surrogate_mode", None)
            )
        ),
        "timing_device": str(op.device),
        "geometry_device": str(pos.device),
        "topology_generation": epoch,
        "position_shape": list(pos.shape),
        "node_count": nodes,
        "movable_count": movable,
        "modes": {},
    }
    try:

        def evaluate(p):
            model._timing_geometry_cache = None
            values = model.timing_obj(p, surrogate_mode=surrogate_mode)
            ev = op.last_evaluation
            assert topology.topology_generation == epoch
            assert op._active_discrete_action_context is None
            losses = {
                "setup_tns": -ev.setup_tns,
                "setup_wns": -ev.setup_wns,
                "combined": model._timing_loss(*values),
            }
            if model.hold_objective.active:
                losses["hold"] = model.hold_objective.loss(ev)
            return losses, ev

        for mode in ("report_only", "active"):
            model.hold_objective = (
                HoldObjective(0.0, 0.0, hold_margin_ps)
                if mode == "report_only"
                else HoldObjective(1.0, 0.01, hold_margin_ps)
            )
            p = base.clone().requires_grad_(requires_grad=True)
            with torch.enable_grad():
                losses, ev = evaluate(p)
            assert ev.hold is not None
            hold_grad = bool(ev.hold.context_slack.requires_grad)
            assert hold_grad == (mode == "active"), (mode, hold_grad)
            grads = {}
            for key, value in losses.items():
                (g,) = torch.autograd.grad(value, p, retain_graph=True, allow_unused=False)
                assert bool(torch.isfinite(g).all()), key
                grads[key] = g.detach()
                raw[mode + "_" + key + "_gradient"] = g.detach().cpu().numpy()
            raw[mode + "_setup_slack"] = (
                op._sparse_runtime.last_result.endpoint_slack.detach().cpu().numpy()
            )
            raw[mode + "_hold_slack"] = ev.hold.endpoint_slack.detach().cpu().numpy()
            item = {
                "losses": {k: float(v.detach()) for k, v in losses.items()},
                "hold_graph_enabled": hold_grad,
                "objective_terms": copy.deepcopy(model.last_timing_objective_terms),
                "rows": [],
            }
            for key, g in grads.items():
                selected = []
                for axis, start in (("x", 0), ("y", nodes)):
                    scores = g[start : start + movable].abs()
                    ids = torch.argsort(scores, descending=True, stable=True)[:2]
                    selected.extend((axis, start + int(i)) for i in ids if float(scores[i]) > 1e-8)
                for axis, index in selected:
                    finite = []
                    for epsilon in (1e-4, 1e-5):
                        points = []
                        for sign in (1.0, -1.0):
                            shifted = base.clone()
                            shifted[index] += sign * epsilon
                            with torch.no_grad():
                                values, _ = evaluate(shifted)
                            points.append(float(values[key]))
                        finite.append((points[0] - points[1]) / (2 * epsilon))
                    actual = float(g[index])
                    error = max(abs(x - actual) for x in finite)
                    tolerance = max(2e-5, 2e-4 * abs(actual))
                    item["rows"].append(
                        {
                            "loss": key,
                            "axis": axis,
                            "coordinate": index,
                            "instance": index if axis == "x" else index - nodes,
                            "backward": actual,
                            "finite_difference": finite,
                            "max_error": error,
                            "tolerance": tolerance,
                            "passed": error <= tolerance,
                        }
                    )
            item["passed"] = bool(item["rows"]) and all(x["passed"] for x in item["rows"])
            item["nonzero_movables"] = {
                k: int(torch.count_nonzero(torch.cat((g[:movable], g[nodes : nodes + movable]))))
                for k, g in grads.items()
            }
            report["modes"][mode] = item
            print("POSITION_PROBE", mode, item["passed"], "samples", len(item["rows"]), flush=True)
        report["setup_gradient_mode_parity"] = {
            k: bool(
                np.allclose(
                    raw["report_only_" + k + "_gradient"],
                    raw["active_" + k + "_gradient"],
                    atol=1e-8,
                    rtol=1e-7,
                )
            )
            for k in ("setup_tns", "setup_wns")
        }
        report["passed"] = all(x["passed"] for x in report["modes"].values()) and all(
            report["setup_gradient_mode_parity"].values()
        )
    finally:
        (
            model.params.placement_sizing_mode,
            model.params.timing_objective_lane,
            model.params.differentiable_timing_obj,
            model.hold_objective,
        ) = saved
        model._timing_geometry_cache = None
    np.savez_compressed(output / "position-gradients.npz", **raw)
    (output / "position-gradients.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "modes"}, indent=2), flush=True)
    assert report["passed"], (
        "Position gradient finite-difference gate failed; see position-gradients.json"
    )
