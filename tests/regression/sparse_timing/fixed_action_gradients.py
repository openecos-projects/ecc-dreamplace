"""Replay all four legal action derivatives on a frozen native timing model.

Uses production candidate mapping, live pin caps, LUT timing and backward.
Wire RC is zero. Each scalar objective uses one backward for the whole N x 4
matrix. No action is committed, and position gradients or finite action deltas
are outside this qualification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import preflight
import torch
from dreamplace.ops.timing_propagation.discrete_actions import build_discrete_action_context
from native_hold_case import NativeHoldCase


def save_snapshot(case, output, device, reference_caps):
    context = case.context
    context.validate(device=device, dtype=torch.float64)
    caps = torch.as_tensor(reference_caps, dtype=torch.float64, device=device)

    def forward():
        case.timing(
            {"rise": case.zeros, "fall": case.zeros},
            {"rise": case.zeros, "fall": case.zeros},
            {"rise": caps[:, 0], "fall": caps[:, 1]},
            surrogate_mode="lut_only",
        )
        return case.timing._sparse_runtime.last_result

    case.timing._active_discrete_action_context = None
    with torch.no_grad():
        baseline = forward()
    case.timing._active_discrete_action_context = context
    forward()
    result = case.timing._sparse_runtime.last_result
    torch.testing.assert_close(
        (result.endpoint_slack, result.check_slack, *result.lanes),
        (baseline.endpoint_slack, baseline.check_slack, *baseline.lanes),
        atol=1e-10, rtol=1e-12,
    )
    arrays = {
        "valid_actions": context.valid_instances,
        "source_cell_ids": case.data.inst_cell_id,
        "target_cell_ids": context.target_cell_ids,
        "endpoint_pins": result.endpoint_pins,
        "endpoint_slack": result.endpoint_slack,
        "check_slack": result.check_slack,
        "required": result.required,
    }
    for name, value in (("tns", result.tns), ("wns", result.wns)):
        gradient = None
        if value.requires_grad:
            (gradient,) = torch.autograd.grad(-value, context.beta, retain_graph=True,
                                             allow_unused=True)
        arrays[name + "_gradient"] = context.action_scores(gradient)
        arrays[name] = value
    hold = case.forward(context.beta)
    arrays.update(hold_endpoint_pins=case.timing.end_points,
                  hold_endpoint_slack=hold.endpoint_slack,
                  hold_context_slack=hold.context_slack)
    # Raw hold TNS is tested without a margin; designs with no negative hold
    # endpoints correctly have a zero derivative and do not prove active hold.
    gradient = None
    if hold.hold_tns.requires_grad:
        (gradient,) = torch.autograd.grad(-hold.hold_tns, context.beta,
                                         allow_unused=True)
    arrays["hold_tns_gradient"] = context.action_scores(gradient)
    if context.beta.is_cuda:
        torch.cuda.synchronize(context.beta.device)
    np.savez_compressed(
        output / f"actions-{device}.npz",
        **{name: value.detach().cpu().numpy() for name, value in arrays.items()},
    )
    return {
        "device": str(context.beta.device),
        "dtype": str(context.beta.dtype),
        "setup_loads": "frozen native max loads with live max-pin-cap action deltas",
        "beta_zero_matches_fixed_master": True,
        "action_names": context.action_names,
        "shape": list(context.beta.shape),
        "valid_actions_per_direction": context.valid_instances.sum(0).tolist(),
        "tns_nonzero_per_direction": (arrays["tns_gradient"].abs() > 1e-8).sum(0).tolist(),
        "native_action_backward_used": context.native_action_backward_used,
        "tns_ps": result.tns.item(),
        "wns_ps": result.wns.item(),
        "hold_tns_ps": hold.hold_tns.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "ecc-root", "native-module"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timing-mode", choices=("complete", "model_only"), required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--aggregation-mode", choices=("hard", "smooth"), default="smooth")
    parser.add_argument("--tau-ps", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if "cuda" in args.devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    def export(pydb, output):
        view = preflight.save_export(pydb, output)
        base = dict(np.load(output / "base_arrays.npz"))
        min_view = pydb.timing_min_sparse
        np.savez_compressed(output / "min_sparse_arrays.npz", **min_view["arrays"])
        (output / "min_sparse_metadata.json").write_text(json.dumps(
            {key: value for key, value in min_view.items() if key != "arrays"}, indent=2
        ))
        np.savez_compressed(output / "min_qualification.npz", **{
            name: np.asarray(getattr(pydb, name)) for name in (
                "endpoints_min_valid", "endpoints_min_reason",
                "endpoints_timing_check_min_valid", "endpoints_timing_check_min_reason",
            )
        })
        reports = []
        for device in args.devices:
            case = NativeHoldCase(pydb, view, base, device=device)
            case.timing.timing_aggregation_mode = args.aggregation_mode
            case.timing.timing_aggregation_tau_ps = args.tau_ps
            for name in ("flat_lib_pin_rcap", "flat_lib_pin_fcap"):
                setattr(case.data, name, torch.tensor(getattr(pydb, name),
                                                     dtype=torch.float64, device=device))
            case.context = build_discrete_action_context(
                case.data, case.candidates, action_names=case.context.action_names,
                pin_names=pydb.flat_lib_pin_names, device=device, dtype=torch.float64,
                drive_load_feedback=True,
            )
            case.context.evaluate_targets = True
            case.model._active_discrete_action_context = case.context
            reports.append(save_snapshot(case, output, device,
                                         view.arrays["pin_reference_cap"]))
        (output / "action_summary.json").write_text(json.dumps(reports, indent=2))
        return view

    preflight.run(args, export=export)


if __name__ == "__main__":
    main()
