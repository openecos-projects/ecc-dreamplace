"""Joint preparation and post-step actions; native commit stays in PlacementEngine."""
from dataclasses import dataclass
from typing import Callable
from dreamplace.flows.metric_state import update_metric_timing

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
