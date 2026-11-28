"""Joint preparation and post-step actions; native commit stays in PlacementEngine."""


from dataclasses import dataclass
from typing import Callable
from dreamplace.flows.gp.metrics import update_metric_timing
import copy
import logging
from dreamplace.flows.gp.metrics import flatten_metrics, last_metric_from_metrics
from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
import numpy as np
import torch
from dreamplace.ops.buffer_insertion.buffering_lane import _action_digest, _tensor_digest


@dataclass
class JointStepState:
    event: object = None
    prepared: object = None
    sizing_frame: object = None
    topology_prepared: bool = False
    pin2pin_update: object = None
    consumed_generation: object = None
    objective_recorded: bool = False
    eval_count_before: object = None

@dataclass(frozen=True)
class JointIterationContext:
    timing_op: object
    prepare: Callable
    rebuild_pin2pin: Callable
    capture_frame: Callable
    rebootstrap: Callable
    overflow_transaction: Callable
    physical_context: Callable
    update_artifact: Callable
    requires_commit: Callable
    virtual_window: Callable
    publish_request: Callable

def prepare_step(context, coordinator, params, placedb, model, pos, optimizer,
                 optimizer_name, iteration, cur_metric, joint_step, profile_now, profile_add):
    if coordinator is not None and coordinator.is_segment_direct_joint:
        _profile_t = profile_now()
        prepared_segment_iteration = context.prepare(
            model=model,
            pos=pos,
            overflow=model.overflow,
            iteration=iteration,
        )
        profile_add("refresh_live_timing_topology_ms", _profile_t)
        if (
            isinstance(prepared_segment_iteration, dict)
            and prepared_segment_iteration.get("mode")
            == "pin2pin_rebootstrap"
        ):
            joint_step.prepared = prepared_segment_iteration
            observation = prepared_segment_iteration["observation"]
            joint_step.topology_prepared = bool(
                observation.get("activation_due")
            )
            if observation.get("activation_due"):
                joint_step.pin2pin_update = (
                    context.rebuild_pin2pin(
                        params=params,
                        placedb=placedb,
                        model=model,
                        pos=pos,
                        optimizer=optimizer,
                        iteration=iteration,
                        overflow=model.overflow[-1],
                        reason="activation_bootstrap",
                        force_clear_accumulation=True,
                    )
                )
                update_metric_timing(
                    cur_metric,
                    *joint_step.pin2pin_update["timing"][:3],
                    context.timing_op,
                )
        else:
            joint_step.event = prepared_segment_iteration
            joint_step.topology_prepared = True
        if (
            optimizer_name.lower() == "nesterov"
            and coordinator.uses_overflow_milestone_actions
            and not coordinator.uses_pin2pin_rebootstrap_schedule
            and joint_step.event is not None
        ):
            _profile_t = profile_now()
            joint_step.sizing_frame = (
                context.capture_frame(
                    params=params,
                    model=model,
                    pos=pos,
                    event=joint_step.event,
                    evaluate_objective=True,
                )
            )
            profile_add(
                "segment_milestone_sizing_frame_ms",
                _profile_t,
            )


def after_step(context, coordinator, params, placedb, model, pos, optimizer,
               optimizer_name, iteration, cur_metric, joint_step):
    if coordinator is not None:
        route_b_record = None
        if (
            coordinator.uses_pin2pin_rebootstrap_schedule
            and joint_step.consumed_generation is not None
        ):
            if (
                optimizer_name.lower() == "nesterov"
                and not joint_step.objective_recorded
            ):
                coordinator.record_pin2pin_nesterov_step(
                    iteration=iteration,
                    generation_id=(
                        joint_step.consumed_generation
                    ),
                    optimizer_eval_count_before=(
                        joint_step.eval_count_before
                    ),
                    optimizer_eval_count_after=sum(
                        int(group.get("obj_eval_count", 0) or 0)
                        for group in optimizer.param_groups
                    ),
                )
                joint_step.objective_recorded = True
            else:
                coordinator.record_pin2pin_placement_step(
                    iteration=iteration,
                    generation_id=(
                        joint_step.consumed_generation
                    ),
                )
            needs_clean_confirmation = bool(
                coordinator.pin2pin_pair_generations.is_timing_clean
                and not coordinator.segment_milestones.queue_is_empty
                and not coordinator.pin2pin_pair_generations.install_count_at_iteration(
                    iteration
                )
            )
            if needs_clean_confirmation:
                confirmation = context.rebuild_pin2pin(
                    params=params,
                    placedb=placedb,
                    model=model,
                    pos=pos,
                    optimizer=optimizer,
                    iteration=iteration,
                    overflow=model.overflow[-1],
                    reason="milestone_clean_confirmation",
                    force_clear_accumulation=True,
                )
                joint_step.pin2pin_update = confirmation
                update_metric_timing(
                    cur_metric,
                    *confirmation["timing"][:3],
                    context.timing_op,
                )
                if confirmation["generation"]["status"] == "timing_clean":
                    route_b_record = (
                        coordinator.skip_pin2pin_milestone_after_timing_clean_confirmation(
                            iteration=iteration,
                            overflow=float(model.overflow[-1]),
                            consumed_pair_generation=(
                                joint_step.consumed_generation
                            ),
                            confirmation_pair_generation=(
                                confirmation["generation"][
                                    "generation_id"
                                ]
                            ),
                            evidence={
                                "wns": confirmation["payload"]["wns"],
                                "tns": confirmation["payload"]["tns"],
                                "pair_count": confirmation["payload"][
                                    "pin2pin_pair_count"
                                ],
                            },
                        )
                    )
            else:
                joint_step.event = (
                    coordinator.begin_pin2pin_milestone_after_step(
                        iteration=iteration,
                        overflow=float(model.overflow[-1]),
                    )
                )
                if joint_step.event is not None:
                    route_b_record = context.rebootstrap(
                        params=params,
                        placedb=placedb,
                        model=model,
                        pos=pos,
                        optimizer=optimizer,
                        event=joint_step.event,
                    )
        elif joint_step.event is not None:
            if coordinator.uses_fixed_window_actions:
                route_b_record = coordinator.defer_segment_route_b(
                    event=joint_step.event,
                )
            elif coordinator.uses_overflow_milestone_actions:
                route_b_record = (
                    context.overflow_transaction(
                        params=params,
                        model=model,
                        pos=pos,
                        optimizer=optimizer,
                        event=joint_step.event,
                        sizing_frame=joint_step.sizing_frame,
                    )
                )
        if route_b_record is not None:
            transition = dict(route_b_record.get("transition", {}) or {})
            setattr(
                cur_metric,
                "joint_segment_milestone",
                route_b_record.get("milestone"),
            )
            setattr(
                cur_metric,
                "joint_segment_route_b_actions",
                transition.get("accepted_action_count", 0),
            )
            setattr(
                cur_metric,
                "joint_segment_buffer_count",
                transition.get("buffer_count_after", 0),
            )
        clamp_metrics = coordinator.post_step_clamp(
            iteration=iteration,
            optimizer=optimizer,
        )
        if clamp_metrics:
            for key, value in clamp_metrics.items():
                setattr(cur_metric, f"joint_{key}", value)
        overflow_value = float(model.overflow[-1])
        if coordinator.should_commit_buffers(iteration, overflow_value):
            commit_result = coordinator.prepare_buffer_commit_request(
                iteration=iteration,
                optimizer=optimizer,
                reason="joint_buffer_periodic_commit",
                final_physical_context=context.physical_context(
                    pos, iteration=iteration,
                ),
            )
            context.publish_request(coordinator.last_buffer_commit_request, commit_result)
            setattr(
                cur_metric,
                "joint_buffer_commit_status",
                commit_result.get("status"),
            )
            setattr(
                cur_metric,
                "joint_projected_buffer_count",
                commit_result.get("projected_buffer_count"),
            )
            context.update_artifact(params)
            if context.requires_commit(
                commit_result
            ):
                return dict(commit_result)
        virtual_window_record = (
            context.virtual_window(
                params=params,
                model=model,
                pos=pos,
                optimizer=optimizer,
                iteration=iteration,
            )
        )
        if virtual_window_record is not None:
            setattr(
                cur_metric,
                "joint_segment_virtual_window_index",
                virtual_window_record.get("window_index"),
            )
            setattr(
                cur_metric,
                "joint_segment_virtual_window_actions",
                virtual_window_record.get(
                    "accepted_action_count",
                    0,
                ),
            )

    return None


@dataclass
class JointCommitContext:
    joint_coordinator: object
    trace: object
    artifact: object
    last_metrics: object
    physical_context: Callable
    projection_context: Callable
    publish_request: Callable
    update_artifact: Callable
    write_artifact: Callable
    write_metrics: Callable
    write_debug: Callable
    post_commit_result: object = None

def finalize_joint_buffer_commit(
    context,
    params,
    *,
    iteration,
    optimizer=None,
    final_pos=None,
):
    coordinator = getattr(context, "joint_coordinator", None)
    if coordinator is None:
        return None
    trace = getattr(context, "trace", None)
    if trace:
        context.update_artifact(params)
        return None
    live_projection_context = None
    final_physical_context = None
    if final_pos is not None:
        final_physical_context = context.physical_context(
            final_pos, iteration=iteration,
        )
    if coordinator.is_segment_direct_joint:
        if final_pos is None:
            raise RuntimeError(
                "segment joint final commit requires the final placement snapshot"
            )
        live_projection_context = context.projection_context(
            final_pos,
            iteration=iteration,
        )
        if live_projection_context is None:
            result = {
                "status": "skipped",
                "reason": "segment_joint_window_not_activated",
                "iteration": int(iteration),
            }
            context.update_artifact(params)
            return result
    commit_result = coordinator.prepare_buffer_commit_request(
        iteration=iteration,
        optimizer=optimizer,
        reason="joint_buffer_final_commit",
        live_projection_context=live_projection_context,
        final_physical_context=final_physical_context,
    )
    context.publish_request(coordinator.last_buffer_commit_request, commit_result, reset_trace=True)
    context.update_artifact(params)
    return commit_result


def joint_post_commit_control_result(commit_result):
    if (
        isinstance(commit_result, dict)
        and commit_result.get("status") == "request_ready"
    ):
        request = dict(commit_result.get("commit_request", {}) or {})
        return {
            "status": "pending_buffer_commit_request",
            "topology_mutated": False,
            "refresh_required": False,
            "refresh_mode": request.get("refresh_mode", "topo"),
            "rebuild_mode": request.get("rebuild_mode", "topo"),
            "continuation_policy": "execute_in_placement_engine",
            "accepted_action_count": 0,
            "commit_request": request,
        }
    commit = {}
    if isinstance(commit_result, dict):
        commit = dict(commit_result.get("commit", {}) or {})
    accepted_action_count = commit.get("accepted_action_count")
    try:
        accepted_action_count = int(accepted_action_count or 0)
    except (TypeError, ValueError):
        accepted_action_count = 0
    topology_mutated = (
        commit.get("status") == "accepted" and accepted_action_count > 0
    )
    return {
        "status": (
            "safe_stopped_after_accepted_buffer_commit"
            if topology_mutated
            else "completed"
        ),
        "topology_mutated": bool(topology_mutated),
        "refresh_required": bool(topology_mutated),
        "refresh_mode": commit.get("refresh_mode", "topo"),
        "rebuild_mode": commit.get("rebuild_mode", "topo"),
        "continuation_policy": (
            "rebuild_placer_outer_loop"
            if topology_mutated
            else "continue_same_topology"
        ),
        "accepted_action_count": accepted_action_count,
        "commit": commit,
    }


def joint_buffer_request_requires_outer_commit(request_result, request):
    if not isinstance(request_result, dict):
        return False
    if request_result.get("status") != "request_ready":
        return False
    return bool(
        request is not None
        and bool(getattr(request, "commit_enabled", False))
        and not bool(getattr(request, "is_noop", False))
    )


def safe_stop_after_joint_buffer_request(
    context,
    params,
    *,
    iteration,
    place_stage_metrics,
    processed_metrics,
    commit_result,
):
    logging.info(
        "joint buffer commit request is ready; stop the inner window before "
        "physical mutation so PlacementEngine can own commit and rebuild"
    )
    if not place_stage_metrics:
        place_stage_metrics = copy.deepcopy(
            [
                metric
                for metric in flatten_metrics(getattr(context, "last_metrics", []))
                if metric is not None
            ]
        )
    write_crash_stage_marker(
        "nonlinearplace_joint_buffer_request_safe_stop",
        "start",
        iteration=iteration,
        projected_buffer_count=commit_result.get("projected_buffer_count"),
    )
    try:
        context.write_metrics(params, place_stage_metrics, stage_name="place")
    except Exception as exc:
        logging.warning(
            "failed to write place metrics before joint buffer commit safe-stop: %s",
            exc,
        )
    summary_payload = getattr(context, "artifact", None)
    if isinstance(summary_payload, dict):
        summary = summary_payload.get("summary")
        if isinstance(summary, dict):
            summary["post_commit_control_policy"] = {
                "status": "safe_stopped_before_buffer_commit",
                "reason": "physical_buffer_commit_owned_by_placement_engine",
                "projected_buffer_count": commit_result.get(
                    "projected_buffer_count"
                ),
            }
            summary["post_commit_control_result"] = (
                joint_post_commit_control_result(commit_result)
            )
            context.write_artifact(params, summary)
    write_crash_stage_marker(
        "nonlinearplace_joint_buffer_request_safe_stop",
        "done",
        iteration=iteration,
        reason="pending_buffer_commit_request",
    )
    processed_metrics = dict(processed_metrics or {})
    processed_metrics["joint_buffer_commit_safe_stop"] = {
        "status": "safe_stopped_before_buffer_commit",
        "projected_buffer_count": commit_result.get("projected_buffer_count"),
    }
    processed_metrics["post_commit_control_result"] = (
        joint_post_commit_control_result(commit_result)
    )
    context.post_commit_result = dict(
        processed_metrics["post_commit_control_result"]
    )
    context.write_debug(
        params,
        iteration=iteration,
        optimizer_steps=len(place_stage_metrics),
        processed_metrics=processed_metrics,
        last_metric=last_metric_from_metrics(place_stage_metrics),
        status="ok",
        stop_reason="pending_buffer_commit_request",
        entered_legalization=False,
        skipped_legalization_reason=None,
    )
    return float("nan"), float("nan"), processed_metrics


def final_physical_context(placer, pos, *, iteration):
    placedb = placer.placedb
    params = placer.params
    num_movable = int(placedb.num_movable_nodes)
    num_nodes = int(placedb.num_nodes)
    scale = float(params.scale_factor or 1.0)
    shift_x, shift_y = params.shift_factor
    final_pos = pos.detach().cpu().double()
    # Freeze integer DBU once, before a float32 PlaceDB round-trip can change
    # which side of a half-DBU boundary the optimized coordinate falls on.
    node_x = (final_pos[:num_movable] / scale + shift_x).round().contiguous()
    node_y = (final_pos[num_nodes : num_nodes + num_movable] / scale + shift_y).round().contiguous()
    final_ids = placer.data_collections.inst_cell_id.detach().cpu().long().clone()
    initial_ids = getattr(placer, "segment_joint_initial_inst_cell_id", None)
    if initial_ids is None:
        # The native export is the baseline for this optimizer window. The
        # live PlaceDB cell IDs may already reflect discrete optimizer steps.
        pydb = placedb.pydb
        main_ids = torch.as_tensor(np.asarray(pydb.inst_main_id), dtype=torch.long)
        starts = torch.as_tensor(np.asarray(pydb.main_id_2_cell_id_start), dtype=torch.long)
        offsets = torch.as_tensor(np.asarray(pydb.inst_libcell_offset), dtype=torch.long)
        initial_ids = torch.where(main_ids >= 0, starts[main_ids.clamp(min=0)] + offsets, -1)
    if initial_ids.numel() != final_ids.numel():
        raise RuntimeError("joint initial/final cell-ID lengths differ")

    def decoded(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    widths = placedb.flat_libcell_width
    heights = placedb.flat_libcell_height
    actions = []
    for inst_id in torch.nonzero(final_ids != initial_ids).flatten().tolist():
        before_id = int(initial_ids[inst_id])
        after_id = int(final_ids[inst_id])
        before_area = float(widths[before_id] * heights[before_id])
        after_area = float(widths[after_id] * heights[after_id])
        actions.append(
            {
                "instance_id": inst_id,
                "instance_name": decoded(placedb.node_names[inst_id]),
                "before_cell_id": before_id,
                "after_cell_id": after_id,
                "before_master": decoded(placedb.flat_libcell_names[before_id]),
                "after_master": decoded(placedb.flat_libcell_names[after_id]),
                "before_area_internal": before_area,
                "after_area_internal": after_area,
                "area_delta_internal": after_area - before_area,
            }
        )
    return {
        "iteration": int(iteration),
        "placement_node_x": node_x,
        "placement_node_y": node_y,
        "sizing_cell_ids": final_ids,
        "sizing_actions": actions,
        "sizing_action_count": len(actions),
        "placement_snapshot_digest": _tensor_digest((("node_x", node_x), ("node_y", node_y))),
    }


def commit_joint_physical_actions(engine, request, *, mutation_backend, output_dir):
    """Use the existing native sizing and buffer transactions in one joint boundary.

    Each native transaction owns its rollback. This boundary does not promise
    rollback of an accepted sizing transaction if the subsequent buffer batch fails.
    """
    if _action_digest(request.sizing_actions) != request.sizing_action_digest:
        raise ValueError("joint sizing actions do not match action digest")
    if _action_digest(request.actions) != request.action_digest:
        raise ValueError("buffer commit request actions do not match action digest")
    placedb = engine.placedb
    params = engine.params
    placedb.inst_cell_id = np.asarray(request.sizing_cell_ids, dtype=np.int32)
    scale = float(params.scale_factor)
    shift_x, shift_y = params.shift_factor
    node_x = (np.asarray(request.placement_node_x) - shift_x) * scale
    node_y = (np.asarray(request.placement_node_y) - shift_y) * scale
    placedb.apply(params, node_x, node_y)
    sizing = dict(placedb.last_sizing_writeback_summary or {})
    accepted = int(sizing.get("accepted_count", sizing.get("applied", 0)))
    if accepted != len(request.sizing_actions):
        raise RuntimeError(
            f"joint sizing count mismatch: requested={len(request.sizing_actions)}, "
            f"accepted={accepted}"
        )
    commit = dict(mutation_backend.commit_buffers(request, output_dir=output_dir) or {})
    commit.update(
        {
            "placement_snapshot_digest": request.placement_snapshot_digest,
            "placement_coordinate_count": len(node_x),
            "sizing_action_count": accepted,
            "sizing_action_digest": request.sizing_action_digest,
            "sizing_result": sizing,
            "atomic_across_sizing_and_buffer": False,
        }
    )
    return commit


_PROFILE_NAME = "proximal_alternating_v1"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def proximal_profile_enabled(params):
    profile = str(getattr(params, "joint_quality_profile", "") or "")
    explicit = str(getattr(params, "joint_proximal_enabled", "")).strip().lower()
    return profile == _PROFILE_NAME or explicit in _TRUE_VALUES


def _float_param(params, name, default=0.0):
    try:
        return float(getattr(params, name, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _tensor_summary(tensor):
    value = tensor.detach()
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.numel() == 0:
        result.update({"numel": 0, "mean": None, "min": None, "max": None})
        return result
    scalar = value.float()
    result.update(
        {
            "numel": int(value.numel()),
            "mean": float(scalar.mean().detach().cpu().item()),
            "min": float(scalar.min().detach().cpu().item()),
            "max": float(scalar.max().detach().cpu().item()),
        }
    )
    return result


def _scalar_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class JointProximalObjective:
    """Native-scale proximal/resource terms for the joint flow objective."""

    def __init__(self, params):
        self.params = params
        self.enabled = proximal_profile_enabled(params)
        self.anchors = {}
        self._anchorable_names_observed = set()
        self.anchor_metadata = {}
        self.last_summary = {"enabled": bool(self.enabled)}

    def lambdas(self):
        return {
            "lambda_x": _float_param(self.params, "joint_proximal_lambda_x", 0.0),
            "lambda_s": _float_param(self.params, "joint_proximal_lambda_s", 0.0),
            "lambda_z": _float_param(self.params, "joint_proximal_lambda_z", 0.0),
            "lambda_b": _float_param(self.params, "joint_proximal_lambda_b", 0.0),
            "mu": _float_param(self.params, "joint_proximal_buffer_mu", 0.0),
        }

    def _buffer_state(self, model):
        data_collections = getattr(model, "data_collections", None)
        state = getattr(model, "buffer_optimization_state", None)
        if state is None and data_collections is not None:
            state = getattr(data_collections, "buffer_optimization_state", None)
        if state is None and data_collections is not None:
            state = getattr(data_collections, "buffer_segment_count_state", None)
        return state

    def _state_tensors(self, model, pos):
        data_collections = getattr(model, "data_collections", None)
        buffer_state = self._buffer_state(model)
        tensors = {
            "x": pos,
            "size_logits": getattr(data_collections, "size_logits", None),
            "vt_logits": getattr(data_collections, "vt_logits", None),
            "z_param": getattr(buffer_state, "z_param", None),
            "bu_logits": getattr(buffer_state, "bu_logits", None),
            "bsu_index_param": getattr(buffer_state, "bsu_index_param", None),
        }
        return {name: value for name, value in tensors.items() if torch.is_tensor(value)}

    def _metadata(self, model):
        data_collections = getattr(model, "data_collections", None)
        payload = None
        if data_collections is not None:
            payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
        payload_metadata = {}
        if isinstance(payload, dict):
            payload_metadata = dict(payload.get("metadata", {}) or {})
        return {
            "topology_generation": getattr(data_collections, "topology_generation", None),
            "openroad_design_generation": getattr(
                data_collections,
                "openroad_design_generation",
                None,
            ),
            "buffer_payload_metadata": payload_metadata,
        }

    def capture_anchors(self, model, pos, *, force=False):
        if not self.enabled:
            self.last_summary = {"enabled": False, "reason": "profile_disabled"}
            return self.last_summary
        if self.anchors and not force:
            tensors = self._state_tensors(model, pos)
            missing_anchors = {
                name: tensor.detach().clone()
                for name, tensor in tensors.items()
                if name not in self.anchors
                and name not in self._anchorable_names_observed
                and (tensor.requires_grad or name == "x")
            }
            if missing_anchors:
                self.anchors.update(missing_anchors)
                self._anchorable_names_observed.update(missing_anchors)
                self.anchor_metadata = self._metadata(model)
                return self.anchor_summary(status="extended")
            return self.anchor_summary(status="reused")
        tensors = self._state_tensors(model, pos)
        self.anchors = {
            name: tensor.detach().clone()
            for name, tensor in tensors.items()
            if tensor.requires_grad or name == "x"
        }
        self._anchorable_names_observed = set(self.anchors)
        self.anchor_metadata = self._metadata(model)
        return self.anchor_summary(status="created")

    def anchor_summary(self, *, status):
        return {
            "status": status,
            "metadata": copy.deepcopy(self.anchor_metadata),
            "tensors": {
                name: _tensor_summary(tensor) for name, tensor in self.anchors.items()
            },
        }

    def _mean_square(self, tensor, anchor, name):
        if tensor.shape != anchor.shape:
            raise ValueError(
                f"joint proximal anchor shape mismatch for {name}: "
                f"tensor={tuple(tensor.shape)} anchor={tuple(anchor.shape)}"
            )
        if tensor.numel() == 0:
            return tensor.sum() * 0.0
        delta = tensor - anchor.to(device=tensor.device, dtype=tensor.dtype)
        return 0.5 * torch.mean(delta * delta)

    def adjust_discrete_count_gradient(self, z_tensor, raw_grad):
        """Replace the differential P_z term with its integer +1 increment."""
        if not torch.is_tensor(z_tensor) or not torch.is_tensor(raw_grad):
            raise TypeError("z_tensor and raw_grad must be tensors")
        if z_tensor.shape != raw_grad.shape:
            raise ValueError("z_tensor and raw_grad must have equal shape")
        action_grad = raw_grad.detach().clone()
        anchor = self.anchors.get("z_param")
        lambda_z = self.lambdas()["lambda_z"]
        if anchor is None or lambda_z == 0.0 or z_tensor.numel() == 0:
            return action_grad, {
                "enabled": False,
                "reason": "missing_anchor_or_zero_lambda",
                "finite_increment_applied": False,
            }

        current = z_tensor.detach()
        anchor = anchor.to(device=current.device, dtype=current.dtype)
        normalizer = float(current.numel())
        differential = lambda_z * (current - anchor) / normalizer
        finite_increment = (
            lambda_z
            * ((current + 1.0 - anchor).square() - (current - anchor).square())
            / (2.0 * normalizer)
        )
        correction = finite_increment - differential
        action_grad.add_(correction.to(device=action_grad.device, dtype=action_grad.dtype))
        return action_grad, {
            "enabled": True,
            "finite_increment_applied": True,
            "lambda_z": float(lambda_z),
            "normalizer": int(current.numel()),
            "correction_min": float(correction.min().cpu().item()),
            "correction_max": float(correction.max().cpu().item()),
            "correction_norm": float(correction.float().norm().cpu().item()),
        }

    def _term_or_missing(self, tensors, names, lambda_value, label, missing):
        active_terms = []
        for name in names:
            tensor = tensors.get(name)
            anchor = self.anchors.get(name)
            if tensor is None:
                continue
            if anchor is None:
                if lambda_value != 0.0:
                    missing.append(name)
                continue
            active_terms.append(self._mean_square(tensor, anchor, name))
        if active_terms:
            return sum(active_terms) / float(len(active_terms))
        if lambda_value != 0.0:
            missing.append(label)
        return None

    def add_to_objective(self, model, obj, pos):
        if not self.enabled:
            self.last_summary = {"enabled": False, "reason": "profile_disabled"}
            return obj, self.last_summary
        anchor_summary = self.capture_anchors(model, pos)
        tensors = self._state_tensors(model, pos)
        lambdas = self.lambdas()
        missing = []
        z_tensor = tensors.get("z_param")
        if z_tensor is None:
            z_tensor = tensors.get("bu_logits")
        # The segment lane is deliberately created after its overflow gate.
        # During initial learning-rate estimation there is no buffer state yet;
        # defer P_z until the lane is activated instead of treating it as a
        # malformed active state.
        lambda_z = lambdas["lambda_z"] if z_tensor is not None else 0.0
        terms = {
            "P_x": self._term_or_missing(
                tensors,
                ("x",),
                lambdas["lambda_x"],
                "placement",
                missing,
            ),
            "P_s": self._term_or_missing(
                tensors,
                ("size_logits", "vt_logits"),
                lambdas["lambda_s"],
                "sizing",
                missing,
            ),
            "P_z": self._term_or_missing(
                tensors,
                ("z_param", "bu_logits"),
                lambda_z,
                "buffer_count",
                missing,
            ),
            "P_b": self._term_or_missing(
                tensors,
                ("bsu_index_param",),
                lambdas["lambda_b"],
                "buffer_size",
                missing,
            ),
        }
        if missing:
            raise ValueError(
                "joint proximal profile has nonzero lambda but missing anchors/state: "
                + ", ".join(sorted(set(missing)))
            )
        if z_tensor is not None:
            resource_values = torch.relu(z_tensor)
            if z_tensor.numel():
                terms["P_buf"] = torch.mean(resource_values)
            else:
                terms["P_buf"] = z_tensor.sum() * 0.0
        elif lambdas["mu"] != 0.0:
            raise ValueError("joint proximal buffer resource term requires z_param or bu_logits")
        else:
            terms["P_buf"] = None

        weighted_terms = {
            "lambda_x_P_x": None if terms["P_x"] is None else lambdas["lambda_x"] * terms["P_x"],
            "lambda_s_P_s": None if terms["P_s"] is None else lambdas["lambda_s"] * terms["P_s"],
            "lambda_z_P_z": None if terms["P_z"] is None else lambdas["lambda_z"] * terms["P_z"],
            "lambda_b_P_b": None if terms["P_b"] is None else lambdas["lambda_b"] * terms["P_b"],
            "mu_P_buf": None if terms["P_buf"] is None else lambdas["mu"] * terms["P_buf"],
        }
        total = None
        for value in weighted_terms.values():
            if value is None:
                continue
            total = value if total is None else total + value
        if total is None:
            total = obj.new_zeros(())
        augmented_obj = obj + total
        obj_abs = abs(_scalar_or_none(obj) or 0.0)
        denom = max(obj_abs, 1.0e-12)
        weighted_scalars = {
            name: _scalar_or_none(value) for name, value in weighted_terms.items()
        }
        ratios = {
            name: (None if value is None else abs(float(value)) / denom)
            for name, value in weighted_scalars.items()
        }
        self.last_summary = {
            "enabled": True,
            "profile": _PROFILE_NAME,
            "obj_fn": _scalar_or_none(obj),
            "total_obj": _scalar_or_none(augmented_obj),
            "raw_terms": {name: _scalar_or_none(value) for name, value in terms.items()},
            "weighted_terms": weighted_scalars,
            "total_weighted": _scalar_or_none(total),
            "lambdas": lambdas,
            "lambda_scale_probe": {
                "denominator_abs_obj_fn": float(denom),
                "weighted_ratio_to_abs_obj_fn": ratios,
                "target_band": [0.01, 0.05],
                "band_is_calibration_warning": True,
            },
            "anchors": anchor_summary,
            "active_blocks": {
                "placement": terms["P_x"] is not None,
                "sizing": terms["P_s"] is not None,
                "buffer_count": terms["P_z"] is not None,
                "buffer_size": terms["P_b"] is not None,
                "buffer_resource": terms["P_buf"] is not None,
            },
            "gates": {
                "missing_anchor": False,
                "missing_state": [],
            },
        }
        return augmented_obj, self.last_summary
