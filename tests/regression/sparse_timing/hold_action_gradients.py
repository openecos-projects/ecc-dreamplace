"""Compare one-backward native-library hold actions with local finite differences.

Example: pass the same --manifest/--ecc-root/--native-module as preflight.py.
The diagnostic margin exercises cell paths when raw negative hold paths are IO
feedthroughs. It never changes production defaults or claims hold closure.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import preflight
import torch
from dreamplace.flows.hold_objective import HoldObjective
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    _select_quad_actions,
)
from dreamplace.ops.timing_propagation import lut_entry_2d_op
from native_hold_case import NativeHoldCase


def verify(case, output, *, margin_ps, samples, epsilon):
    beta = case.context.beta
    if beta.is_cuda:
        torch.cuda.synchronize(beta.device)
    started = time.perf_counter()
    result = case.forward(beta)
    objective = HoldObjective(1, 0.01, margin_ps)
    loss = objective.loss(SimpleNamespace(hold=result))
    (gradient,) = torch.autograd.grad(loss, beta)
    if beta.is_cuda:
        torch.cuda.synchronize(beta.device)
    elapsed = time.perf_counter() - started
    valid = case.context.valid_instances.clone()
    check_owners = case.context.check_inst_ids
    if check_owners is None:
        check_owners = beta.new_empty(0, dtype=torch.long)
    check_owners = check_owners[check_owners >= 0]
    clock_end = int(case.timing.flat_inst_arcs_by_level_start[1])
    clock_owners = case.context.inst_ids[:clock_end]
    clock_owners = clock_owners[clock_owners >= 0]
    rows = []
    for action, name in enumerate(ACTION_NAMES):
        scores = torch.where(valid[:, action], gradient[:, action].abs(), 0)
        selected = torch.argsort(scores, descending=True, stable=True)[:samples]
        for owners in (check_owners, clock_owners):
            if owners.numel():
                selected = torch.cat((selected, owners[torch.argmax(scores[owners])].reshape(1)))
        selected = torch.unique(selected)
        selected = selected[scores[selected] > 1e-8]
        for node in selected.tolist():
            values = []
            for scale in (1, 0.1):
                eps = epsilon * scale
                plus, minus = beta.detach().clone(), beta.detach().clone()
                plus[node, action], minus[node, action] = eps, -eps
                with torch.no_grad():
                    p = objective.loss(SimpleNamespace(hold=case.forward(plus))).item()
                    m = objective.loss(SimpleNamespace(hold=case.forward(minus))).item()
                values.append((p - m) / (2 * eps))
            actual = gradient[node, action].item()
            error = max(abs(actual - value) for value in values)
            tolerance = max(1e-5, 1e-4 * abs(actual))
            source = int(case.data.inst_cell_id[node])
            target = int(case.context.target_cell_ids[node, action])
            rows.append(
                {
                    "instance_id": node,
                    "action": name,
                    "source": case.placedb.flat_libcell_names[source].decode(),
                    "target": case.placedb.flat_libcell_names[target].decode(),
                    "has_hold_check": bool((check_owners == node).any()),
                    "has_clock_to_q": bool((clock_owners == node).any()),
                    "backward": actual,
                    "finite_difference": values,
                    "max_absolute_error": error,
                    "tolerance": tolerance,
                    "passed": error <= tolerance,
                }
            )
    expected = set(ACTION_NAMES)
    covered = {row["action"] for row in rows}
    native = Path(lut_entry_2d_op._select_module(beta).__file__)
    selected, budget, _, selection_backend = _select_quad_actions(
        gradient,
        valid,
        case.data.inst_is_sizeable,
        shared_budget_percent=1.0,
        blocked_instances=None,
        size_up_percent=0.0,
        size_down_percent=0.0,
        vt_percent=0.0,
    )
    report = {
        "status": "passed"
        if rows and covered == expected and all(r["passed"] for r in rows)
        else "failed",
        "scope": (
            "Real native library and legal four-action table, float64, ideal clocks, "
            "live min pin/SDC loads, zero wire RC; local beta=0 derivative only"
        ),
        "diagnostic_margin_ps": margin_ps,
        "device": str(beta.device),
        "torch_version": torch.__version__,
        "objective": {"wns": objective.wns, "tns": objective.tns},
        "hold_wns_ps": result.hold_wns.item(),
        "hold_tns_ps": result.hold_tns.item(),
        "endpoints_below_diagnostic_margin": int((result.endpoint_slack < margin_ps).sum()),
        "positive_slack_min_ps": (
            float(result.endpoint_slack[result.endpoint_slack > 0].min())
            if bool((result.endpoint_slack > 0).any()) else None
        ),
        "instance_count": beta.shape[0],
        "valid_actions_per_direction": valid.sum(0).tolist(),
        "nonzero_gradients_per_direction": ((gradient.abs() > 1e-8) & valid).sum(0).tolist(),
        "forward_backward_seconds": elapsed,
        "lut_native_module": str(native),
        "lut_native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "epsilon": epsilon,
        "diagnostic_selection": {
            "shared_budget_percent": 1.0,
            "budget": budget,
            "backend": selection_backend,
            "selected": torch.nonzero(selected).tolist(),
        },
        "rows": rows,
        "limitations": [
            "Diagnostic margin is not raw hold closure",
            "No position-to-RC, native early kernel or full-flow qualification",
            "Finite beta=1 action and joint-step changes are not local derivatives",
        ],
    }
    np.savez_compressed(
        output / "hold_action_gradients.npz",
        gradient=gradient.cpu().numpy(),
        valid=valid.cpu().numpy(),
        selected=selected.cpu().numpy(),
        target_cells=case.context.target_cell_ids.cpu().numpy(),
        context_slack=result.context_slack.detach().cpu().numpy(),
        endpoint_slack=result.endpoint_slack.detach().cpu().numpy(),
    )
    (output / "hold_action_gradients.json").write_text(json.dumps(report, indent=2))
    print("HOLD_ACTION_GRADIENTS", json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("native-library four-direction hold gradient gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ecc-root", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timing-mode", choices=("complete", "model_only"), default="model_only")
    parser.add_argument("--margin-ps", type=float, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def export(pydb, output):
        view = preflight.save_export(pydb, output)
        case = NativeHoldCase(
            pydb, view, dict(np.load(output / "base_arrays.npz")), device=args.device
        )
        verify(case, output, margin_ps=args.margin_ps, samples=args.samples, epsilon=args.epsilon)
        return view

    preflight.run(args, export=export)


if __name__ == "__main__":
    main()
