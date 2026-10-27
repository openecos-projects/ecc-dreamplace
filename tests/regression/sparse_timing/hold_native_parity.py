"""Compare production early reference/native values and six input adjoints."""

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
from dreamplace.ops.timing_propagation import lut_entry_2d_op
from dreamplace.ops.timing_propagation.early_propagation import EarlyPropagation
from native_hold_case import NativeHoldCase


def verify(case, output, margin):
    case.model._active_discrete_action_context = None
    case.timing._active_discrete_action_context = None
    base = tuple(cap.detach() for cap in case.loads()) + (case.zeros,) * 4
    report = {}
    for mode in ("hard", "smooth"):
        case.timing.timing_aggregation_mode = mode
        results, gradients, samples, dispatch = [], [], [], []
        for native in (False, True):
            inputs = tuple(value.clone().requires_grad_() for value in base)
            op = EarlyPropagation(case.hold, use_native=native)
            if case.zeros.is_cuda:
                torch.cuda.synchronize(case.zeros.device)
            started = time.perf_counter()
            result = op.forward(case.timing, inputs[2:4], inputs[4:6], inputs[:2])
            loss = HoldObjective(1, 0.01, margin).loss(SimpleNamespace(hold=result))
            gradients.append(torch.stack(torch.autograd.grad(loss, inputs)))
            if case.zeros.is_cuda:
                torch.cuda.synchronize(case.zeros.device)
            results.append(
                {
                    "context_slack": result.context_slack,
                    "endpoint_slack": result.endpoint_slack,
                    "slew": torch.stack(result.slews),
                    "arrival": result.state_arrivals if result.state_arrivals is not None else torch.stack(
                        [torch.stack(value) for value in result.arrivals.values()]
                    ),
                }
            )
            samples.append(time.perf_counter() - started)
            dispatch.append(op.native_dispatch.copy())
        errors = {}
        for field in results[0]:
            a, b = (result[field] for result in results)
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-12)
            finite = torch.isfinite(a) & torch.isfinite(b)
            errors[field] = float((a[finite] - b[finite]).abs().max())
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-9, rtol=1e-9)
        errors["six_input_gradients"] = float((gradients[0] - gradients[1]).abs().max())
        assert all(dispatch[1][kind] > 0 for kind in ("cell", "net"))
        report[mode] = {
            "max_absolute_errors": errors,
            "seconds_reference_native": samples,
            "dispatch_reference_native": dispatch,
        }
    native = Path(lut_entry_2d_op._select_module(case.zeros).__file__)
    summary = {
        "status": "passed",
        "scope": (
            "Real native export, float64, ideal clocks, zero wire RC baseline; "
            "live min loads, independent load/delay/impulse input derivatives"
        ),
        "diagnostic_margin_ps": margin,
        "device": str(case.zeros.device),
        "torch_version": torch.__version__,
        "lut_native_module": str(native),
        "lut_native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "modes": report,
        "limitations": [
            "No cell beta action or position-to-RC gradient comparison",
            "No full-flow qualification",
            "Single-run timings are diagnostic",
        ],
    }
    (output / "hold_native_parity.json").write_text(json.dumps(summary, indent=2))
    print("HOLD_NATIVE_PARITY", json.dumps(summary), flush=True)


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
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def export(pydb, output):
        view = preflight.save_export(pydb, output)
        case = NativeHoldCase(
            pydb, view, dict(np.load(output / "base_arrays.npz")), device=args.device
        )
        verify(case, output, args.margin_ps)
        return view

    preflight.run(args, export=export)


if __name__ == "__main__":
    main()
