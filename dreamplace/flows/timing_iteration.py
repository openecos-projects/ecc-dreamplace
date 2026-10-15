"""Diff-TDP gate refresh at its existing metric/objective boundary."""
from dataclasses import dataclass
from typing import Callable
import time
import logging
import torch

@dataclass(frozen=True)
class TimingIterationContext:
    enabled: Callable
    uses_net_weight: Callable
    project_weights: Callable
    refresh_topology: Callable
    carrier: Callable
    reset_weights: Callable

def update_gate(context, params, placedb, model, pos, iteration,
                diff_tdp_gate_status, joint_step):
    if (
        context.enabled(params)
        and diff_tdp_gate_status["topology_refresh_due"]
    ):
        if context.uses_net_weight(params):
            context.project_weights(
                params,
                placedb,
                model,
                pos,
                iteration=iteration,
                gate_status=diff_tdp_gate_status,
                topology_prepared=joint_step.topology_prepared,
            )
            model.use_timing_obj = False
        else:
            t_steiner = time.time()
            if not joint_step.topology_prepared:
                with torch.no_grad():
                    context.refresh_topology(pos)
                joint_step.topology_prepared = True
            diff_tdp_update_ms = (time.time() - t_steiner) * 1000
            logging.info(
                "Diff TDP enabled timing objective at iteration=%d "
                "overflow=%.6f threshold=%.6f interval=%d "
                "carrier=%s steiner_update_ms=%.3f",
                int(iteration),
                float(diff_tdp_gate_status["overflow"]),
                float(diff_tdp_gate_status["threshold"]),
                int(diff_tdp_gate_status["interval"]),
                context.carrier(params),
                diff_tdp_update_ms,
            )
    elif (
        context.enabled(params)
        and not diff_tdp_gate_status["timing_objective_active"]
    ):
        context.reset_weights(
            params,
            placedb,
            iteration=iteration,
            reason=diff_tdp_gate_status["reason"],
        )
        logging.debug(
            "Diff TDP gate skipped at iteration=%d reason=%s "
            "overflow=%s threshold=%.6f interval=%d",
            int(iteration),
            diff_tdp_gate_status["reason"],
            diff_tdp_gate_status["overflow"],
            float(diff_tdp_gate_status["threshold"]),
            int(diff_tdp_gate_status["interval"]),
        )
