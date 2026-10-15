"""Best-state selection and restore for ECC continuous optimization.

The call-scoped context references the existing live tensors and restore
callbacks. Native writeback and stop scheduling remain with the caller.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import dreamplace.BasicPlace as BasicPlace
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState
from dreamplace.ops.buffer_insertion.optimization_state import capture_buffer_optimization_snapshot, restore_buffer_optimization_snapshot

@dataclass
class EarlyStopContext:
    data_collections: object
    placedb: object
    params: object
    oscillation_state: object
    sizing_parameterization: Callable
    continuous_size_parameter: Callable
    discrete_actions_owned: bool
    sync_runtime_cells: Callable
    sizing_mode: Callable
    finite_scalar: Callable
    capture_best: Callable
    loss_and_source: Callable

def continuous_early_stop_loss_and_source(context, params, metric):
    if metric is None:
        return None, None
    combined_timing_loss = context.finite_scalar(
        getattr(metric, "combined_timing_loss", None)
    )
    if combined_timing_loss is not None:
        return combined_timing_loss, "combined_timing_loss"
    if (
        context.sizing_mode(params) == "size_only"
        and bool(getattr(params, "with_sta", False))
        and bool(getattr(params, "differentiable_timing_obj", False))
    ):
        return None, None
    objective = context.finite_scalar(getattr(metric, "objective", None))
    if objective is not None:
        return objective, "objective"
    return None, None


def continuous_early_stop_loss(context, params, metric):
    loss, _source = context.loss_and_source(params, metric)
    return loss


def build_continuous_early_stop_state(params):
    patience = getattr(params, "early_stop_patience", None)
    min_delta = getattr(params, "early_stop_min_delta", None)
    restore_best = bool(getattr(params, "early_stop_restore_best", False))
    patience_value = None if patience is None else int(patience)
    stop_enabled = patience_value is not None and patience_value > 0
    enabled = stop_enabled or restore_best
    return {
        "enabled": enabled,
        "patience": patience_value,
        "min_delta": 0.0 if min_delta is None else max(0.0, float(min_delta)),
        "restore_best": restore_best,
        "best_loss": None,
        "best_iteration": None,
        "best_state": None,
        "loss_source": None,
        "last_loss": None,
        "last_iteration": None,
        "last_loss_source": None,
        "no_improve_rounds": 0,
        "triggered": False,
        "trigger_iteration": None,
        "trigger_reason": None,
        "restored": False,
        "restore_iteration": None,
        "restore_reason": None,
    }


def capture_continuous_early_stop_best_state(context, snapshot=None):
    data_collections = getattr(context, "data_collections", None)
    placedb = getattr(context, "placedb", None)

    def capture_tensor(name):
        if isinstance(snapshot, dict) and snapshot.get(name) is not None:
            return snapshot[name].detach().clone()
        tensor = None if data_collections is None else getattr(data_collections, name, None)
        if tensor is None:
            return None
        return tensor.detach().clone()

    def capture_array(name):
        value = None if placedb is None else getattr(placedb, name, None)
        if value is None:
            return None
        return np.array(value, copy=True)

    best_state = {
        "size_logits": capture_tensor("size_logits"),
        "real_size": capture_tensor("real_size"),
        "continuous_size": capture_tensor("continuous_size"),
        "continuous_size_parameter": capture_tensor(
            "continuous_size_parameter"
        ),
        "effective_initial_size": capture_tensor("effective_initial_size"),
        "sizing_parameterization": (
            (
                snapshot.get("sizing_parameterization")
                or context.sizing_parameterization()
            )
            if isinstance(snapshot, dict)
            else context.sizing_parameterization()
        ),
        "vt_logits": capture_tensor("vt_logits"),
        "inst_cell_id": capture_tensor("inst_cell_id"),
        "inst_libcell_offset": capture_tensor("inst_libcell_offset"),
        "placedb_inst_cell_id": capture_array("inst_cell_id"),
        "placedb_inst_libcell_offset": capture_array("inst_libcell_offset"),
    }
    best_state.update(
        capture_buffer_optimization_snapshot(
            getattr(data_collections, "buffer_optimization_state", None)
        )
    )
    oscillation_state = getattr(context, "oscillation_state", None)
    best_state["quad_oscillation_state"] = (
        None if oscillation_state is None else oscillation_state.snapshot()
    )
    return best_state


def restore_continuous_early_stop_best_state(context, state, reason="continuous_best_state_restore"):
    summary = {
        "enabled": bool(isinstance(state, dict) and state.get("enabled")),
        "restore_best": bool(isinstance(state, dict) and state.get("restore_best")),
        "restored": False,
        "restore_reason": None,
        "best_iteration": None,
        "best_loss": None,
    }
    if not isinstance(state, dict) or not state.get("enabled") or not state.get("restore_best"):
        return summary
    best_state = state.get("best_state")
    if not isinstance(best_state, dict):
        summary["restore_reason"] = "best_state_unavailable"
        return summary

    data_collections = getattr(context, "data_collections", None)
    placedb = getattr(context, "placedb", None)

    def restore_tensor(name):
        source = best_state.get(name)
        target = None if data_collections is None else getattr(data_collections, name, None)
        if source is None or target is None:
            return
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    restored_cell_updates = None
    sizing_runtime_sync = None
    if context.discrete_actions_owned:
        sizing_restore_summary = {
            "restored": False,
            "restore_reason": "milestone_owned_discrete_sizing_state",
        }
        buffer_restore_summary = {
            "restored": False,
            "restore_reason": "route_b_owned_discrete_state",
            "restored_keys": [],
        }
    else:
        current_cells = getattr(data_collections, "inst_cell_id", None)
        saved_cells = best_state.get("inst_cell_id")
        if current_cells is not None and saved_cells is not None:
            saved_cells = saved_cells.to(device=current_cells.device)
            changed_ids = torch.nonzero(
                current_cells != saved_cells, as_tuple=False
            ).flatten()
            if changed_ids.numel():
                restored_cell_updates = {
                    "applied_instance_ids": changed_ids.cpu().tolist(),
                    "applied_cell_ids": saved_cells[changed_ids].cpu().tolist(),
                }
        saved_parameterization = str(
            best_state.get(
                "sizing_parameterization",
                context.sizing_parameterization(),
            )
        ).strip().lower()
        current_parameterization = context.sizing_parameterization()
        saved_continuous_size = best_state.get("continuous_size")
        if saved_parameterization == current_parameterization:
            source = best_state.get("continuous_size_parameter")
            target = context.continuous_size_parameter()
            if source is not None and target is not None:
                with torch.no_grad():
                    target.copy_(
                        source.to(device=target.device, dtype=target.dtype)
                    )
            else:
                restore_tensor("size_logits")
                restore_tensor("real_size")
        elif saved_continuous_size is None:
            raise RuntimeError(
                "best sizing snapshot lacks continuous size for parameterization conversion"
            )
        elif current_parameterization == "logits":
            target = context.continuous_size_parameter()
            converted = BasicPlace.size_var_to_logits(
                saved_continuous_size,
                data_collections.inst_size_lower,
                data_collections.inst_size_upper,
            )
            with torch.no_grad():
                target.copy_(converted.to(device=target.device, dtype=target.dtype))
        elif current_parameterization == "real_size":
            target = context.continuous_size_parameter()
            with torch.no_grad():
                target.copy_(
                    saved_continuous_size.to(
                        device=target.device,
                        dtype=target.dtype,
                    )
                )
        else:
            raise RuntimeError(
                "unsupported sizing parameterization in best-state restore"
            )
        restore_tensor("effective_initial_size")
        restore_tensor("vt_logits")
        restore_tensor("inst_cell_id")
        restore_tensor("inst_libcell_offset")
        sizing_restore_summary = {
            "restored": True,
            "restore_reason": reason,
        }
        buffer_restore_summary = restore_buffer_optimization_snapshot(
            getattr(data_collections, "buffer_optimization_state", None),
            best_state,
        )

    if placedb is not None and not context.discrete_actions_owned:
        for state_key, attr_name in (
            ("placedb_inst_cell_id", "inst_cell_id"),
            ("placedb_inst_libcell_offset", "inst_libcell_offset"),
        ):
            source = best_state.get(state_key)
            target = getattr(placedb, attr_name, None)
            if source is not None and target is not None:
                target[...] = source

    if restored_cell_updates is not None:
        # Restore the selected masters and their physical/timing state as
        # one transaction. Final projection can report no changed cells,
        # so it cannot repair stale last-iteration geometry for us.
        sizing_runtime_sync = context.sync_runtime_cells(
            getattr(context, "params", None), restored_cell_updates
        )

    if "quad_oscillation_state" in best_state:
        saved_oscillation = best_state["quad_oscillation_state"]
        if saved_oscillation is None:
            context.oscillation_state = None
        else:
            oscillation_state = getattr(context, "oscillation_state", None)
            if oscillation_state is None:
                oscillation_state = CellOscillationState()
            oscillation_state.restore(saved_oscillation)
            context.oscillation_state = oscillation_state

    state["restored"] = True
    state["restore_iteration"] = state.get("best_iteration")
    state["restore_reason"] = reason
    summary.update(
        {
            "restored": True,
            "restore_reason": reason,
            "best_iteration": state.get("best_iteration"),
            "best_loss": state.get("best_loss"),
            "sizing_restored": bool(sizing_restore_summary.get("restored")),
            "sizing_restore_reason": sizing_restore_summary.get(
                "restore_reason"
            ),
            "sizing_runtime_sync": sizing_runtime_sync,
            "buffer_restored": bool(buffer_restore_summary.get("restored")),
            "buffer_restore_reason": buffer_restore_summary.get("restore_reason"),
        }
    )
    return summary


def update_continuous_early_stop_state(
    context,
    state,
    combined_timing_loss,
    iteration,
    loss_source="combined_timing_loss",
):
    if not state.get("enabled"):
        return False
    if combined_timing_loss is None:
        return False

    loss = float(combined_timing_loss)
    source = str(loss_source or "combined_timing_loss")
    state["last_loss"] = loss
    state["last_iteration"] = int(iteration)
    state["last_loss_source"] = source

    best_loss = state.get("best_loss")
    min_delta = float(state.get("min_delta", 0.0))
    if best_loss is None or loss < best_loss - min_delta:
        state["best_loss"] = loss
        state["best_iteration"] = int(iteration)
        state["loss_source"] = source
        if state.get("restore_best"):
            best_state = state.pop("candidate_state", None)
            if not isinstance(best_state, dict):
                best_state = context.capture_best()
            state["best_state"] = best_state
        state["no_improve_rounds"] = 0
        return False

    state["no_improve_rounds"] = int(state.get("no_improve_rounds", 0)) + 1
    patience = state.get("patience")
    if patience is not None and int(patience) > 0 and state["no_improve_rounds"] >= int(patience):
        state["triggered"] = True
        state["trigger_iteration"] = int(iteration)
        state["trigger_reason"] = "continuous_no_improvement"
        return True
    return False


def continuous_early_stop_summary(state):
    if not isinstance(state, dict):
        return None
    return {
        "enabled": bool(state.get("enabled")),
        "patience": state.get("patience"),
        "min_delta": state.get("min_delta"),
        "restore_best": bool(state.get("restore_best")),
        "best_loss": state.get("best_loss"),
        "best_iteration": state.get("best_iteration"),
        "loss_source": state.get("loss_source"),
        "best_state_available": isinstance(state.get("best_state"), dict),
        "last_loss": state.get("last_loss"),
        "last_iteration": state.get("last_iteration"),
        "last_loss_source": state.get("last_loss_source"),
        "no_improve_rounds": state.get("no_improve_rounds"),
        "triggered": bool(state.get("triggered")),
        "trigger_iteration": state.get("trigger_iteration"),
        "trigger_reason": state.get("trigger_reason"),
        "restored": bool(state.get("restored")),
        "restore_iteration": state.get("restore_iteration"),
        "restore_reason": state.get("restore_reason"),
    }


