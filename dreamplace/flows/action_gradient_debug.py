"""Forward-only checks of the action interpolation; never used by selection."""

import torch


def probe_action_interpolation(evaluate, context, instance, action):
    """Compare small positive beta steps with the base-point adjoint.

    The normal deferred operator queries targets during backward at beta=0.
    A finite difference must explicitly evaluate target LUTs in forward.
    """

    previous = getattr(context, "evaluate_targets", False)
    saved_beta = context.beta.detach().clone()
    records = []
    try:
        context.evaluate_targets = True
        with torch.no_grad():
            context.beta.zero_()
            steps = (
                (0.0, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0)
                if context.beta.dtype == torch.float64
                else (0.0, 0.001, 0.01, 0.1, 1.0)
            )
            for step in steps:
                context.beta[instance, action] = step
                metrics = evaluate()
                records.append({"beta": step, **metrics})
    finally:
        with torch.no_grad():
            context.beta.copy_(saved_beta)
        context.evaluate_targets = previous
    base = records[0]
    for row in records[1:]:
        for metric in ("tns", "wns"):
            row["delta_" + metric] = row[metric] - base[metric]
            row["slope_" + metric] = row["delta_" + metric] / row["beta"]
    return records
