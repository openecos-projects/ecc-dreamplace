import copy
import hashlib
import json
import logging
import math
import os
import time

import torch

from dreamplace.ops.buffer_insertion.discrete_virtual_scheduler import (
    schedule_net_gradient_actions,
    take_transition_prefix,
    transition_backtracking_sizes,
)


SEGMENT_DIRECT_JOINT_PROFILE = "segment_count_direct_joint_v1"
SEGMENT_DIRECT_JOINT_MILESTONES = (0.30, 0.25, 0.20, 0.15, 0.10)
SEGMENT_ACTION_CALIBRATION_SAMPLE_COUNT = 8
SEGMENT_OVERFLOW_MILESTONE_SCHEDULE = "overflow_milestones"
SEGMENT_FIXED_WINDOW_SCHEDULE = "fixed_window_compat"


def _tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        average_rank = 0.5 * ((begin + 1) + end)
        for position in range(begin, end):
            ranks[order[position]] = average_rank
        begin = end
    return ranks


def _rank_correlation(lhs, rhs):
    if len(lhs) != len(rhs) or len(lhs) < 2:
        return None
    lhs_rank = _average_ranks(lhs)
    rhs_rank = _average_ranks(rhs)
    lhs_mean = sum(lhs_rank) / len(lhs_rank)
    rhs_mean = sum(rhs_rank) / len(rhs_rank)
    numerator = sum(
        (left - lhs_mean) * (right - rhs_mean)
        for left, right in zip(lhs_rank, rhs_rank)
    )
    lhs_norm = math.sqrt(sum((value - lhs_mean) ** 2 for value in lhs_rank))
    rhs_norm = math.sqrt(sum((value - rhs_mean) ** 2 for value in rhs_rank))
    denominator = lhs_norm * rhs_norm
    return None if denominator == 0.0 else numerator / denominator


def _same_nonzero_sign(lhs, rhs):
    return (lhs > 0.0 and rhs > 0.0) or (lhs < 0.0 and rhs < 0.0)


class Pin2PinPairGenerationState:
    """Lifecycle evidence for the packed Pin2Pin placement objective."""

    VALID_STATUSES = frozenset(("timing_violating", "timing_clean"))

    def __init__(self, *, update_interval):
        update_interval = int(update_interval)
        if update_interval <= 0:
            raise ValueError("Pin2Pin update interval must be positive")
        self.update_interval = update_interval
        self._next_generation_id = 1
        self._current_generation = None
        self._generation_history = []
        self._install_count_by_iteration = {}
        self._pending_objective_evaluations = set()
        self.trace = []

    @property
    def active(self):
        return self._current_generation is not None

    @property
    def current_generation(self):
        return copy.deepcopy(self._current_generation)

    @property
    def current_generation_id(self):
        if self._current_generation is None:
            return None
        return int(self._current_generation["generation_id"])

    @property
    def is_timing_clean(self):
        return bool(
            self._current_generation is not None
            and self._current_generation["status"] == "timing_clean"
        )

    @property
    def pending(self):
        return bool(
            self._current_generation is not None
            and not self._current_generation["valid"]
        )

    @property
    def periodic_update_due(self):
        return bool(
            self._current_generation is not None
            and self._current_generation["valid"]
            and int(self._current_generation["consumed_step_count"])
            >= self.update_interval
        )

    @property
    def can_execute_milestone(self):
        return bool(
            self._current_generation is not None
            and self._current_generation["valid"]
            and int(self._current_generation["objective_eval_count"]) > 0
            and int(self._current_generation["consumed_step_count"]) > 0
        )

    def _require_current(self, generation_id):
        if (
            self._current_generation is None
            or int(generation_id) != self.current_generation_id
        ):
            raise RuntimeError(
                f"pair generation {generation_id} is not the current generation"
            )
        if not self._current_generation["valid"]:
            raise RuntimeError(
                f"pair generation {generation_id} has been invalidated"
            )
        return self._current_generation

    def install(self, *, iteration, reason, status, pair_count, metadata=None):
        iteration = int(iteration)
        status = str(status)
        pair_count = int(pair_count)
        if status not in self.VALID_STATUSES:
            raise ValueError(f"invalid Pin2Pin generation status: {status}")
        if status == "timing_clean" and pair_count != 0:
            raise ValueError("timing-clean Pin2Pin generation must contain zero pairs")
        if status == "timing_violating" and pair_count <= 0:
            raise ValueError(
                "timing-violating Pin2Pin generation requires a positive pair count"
            )
        install_count = int(self._install_count_by_iteration.get(iteration, 0))
        if install_count:
            raise RuntimeError(
                f"a Pin2Pin generation was already installed at iteration {iteration}"
            )
        self._install_count_by_iteration[iteration] = install_count + 1
        self._pending_objective_evaluations.clear()
        generation = {
            "generation_id": int(self._next_generation_id),
            "installed_iteration": iteration,
            "install_reason": str(reason),
            "status": status,
            "pair_count": pair_count,
            "valid": True,
            "objective_eval_count": 0,
            "consumed_step_count": 0,
            "objective_eval_iterations": [],
            "consumed_step_iterations": [],
            "metadata": copy.deepcopy(metadata or {}),
        }
        self._next_generation_id += 1
        self._current_generation = generation
        self._generation_history.append(generation)
        self.trace.append(
            {
                "event": "installed",
                "iteration": iteration,
                "generation_id": generation["generation_id"],
                "reason": str(reason),
                "status": status,
                "pair_count": pair_count,
            }
        )
        return copy.deepcopy(generation)

    def record_objective_evaluation(self, *, iteration, generation_id):
        generation = self._require_current(generation_id)
        iteration = int(iteration)
        key = (int(generation_id), iteration)
        if key in self._pending_objective_evaluations:
            raise RuntimeError(
                "pair generation already has an unconsumed objective evaluation "
                f"at iteration {iteration}"
            )
        generation["objective_eval_count"] += 1
        generation["objective_eval_iterations"].append(iteration)
        self._pending_objective_evaluations.add(key)
        self.trace.append(
            {
                "event": "objective_evaluated",
                "iteration": iteration,
                "generation_id": int(generation_id),
            }
        )

    def record_placement_step(self, *, iteration, generation_id):
        generation = self._require_current(generation_id)
        iteration = int(iteration)
        key = (int(generation_id), iteration)
        if key not in self._pending_objective_evaluations:
            raise RuntimeError(
                "Pin2Pin placement step requires an objective evaluation from "
                "the same generation and iteration"
            )
        self._pending_objective_evaluations.remove(key)
        generation["consumed_step_count"] += 1
        generation["consumed_step_iterations"].append(iteration)
        self.trace.append(
            {
                "event": "placement_step_consumed",
                "iteration": iteration,
                "generation_id": int(generation_id),
            }
        )

    def invalidate(self, *, iteration, reason):
        if self._current_generation is None:
            raise RuntimeError("cannot invalidate an absent Pin2Pin generation")
        self._current_generation["valid"] = False
        self._current_generation["invalidated_iteration"] = int(iteration)
        self._current_generation["invalidation_reason"] = str(reason)
        self._pending_objective_evaluations.clear()
        self.trace.append(
            {
                "event": "invalidated",
                "iteration": int(iteration),
                "generation_id": self.current_generation_id,
                "reason": str(reason),
            }
        )

    def install_count_at_iteration(self, iteration):
        return int(self._install_count_by_iteration.get(int(iteration), 0))

    def summarize(self):
        current = self.current_generation
        consumed_steps = int(
            0 if current is None else current["consumed_step_count"]
        )
        return {
            "update_interval": self.update_interval,
            "current_generation": current,
            "generation_history": copy.deepcopy(self._generation_history),
            "install_count_by_iteration": {
                str(key): int(value)
                for key, value in sorted(self._install_count_by_iteration.items())
            },
            "next_periodic_due_after_consumed_steps": self.update_interval,
            "consumed_steps_until_periodic_update": max(
                0, self.update_interval - consumed_steps
            ),
            "trace": copy.deepcopy(self.trace),
        }


class SegmentJointMilestoneState:
    """One-shot overflow events for the frozen placement-buffering window."""

    REBOOTSTRAP_TRANSACTION_STAGES = (
        "tdp_objective_and_step_consumed",
        "sizing_micro_loop_completed",
        "buffering_gradient_ready",
        "route_b_applied",
        "tdp_state_invalidated",
        "tdp_rebootstrap_completed",
    )

    TRANSACTION_STAGES = (
        "timing_frame_ready",
        "sizing_gradient_frozen",
        "placement_updated",
        "sizing_applied",
        "runtime_state_refreshed",
        "buffering_gradient_ready",
        "route_b_applied",
    )

    def __init__(self, milestones=SEGMENT_DIRECT_JOINT_MILESTONES):
        values = tuple(float(value) for value in milestones)
        if not values or any(
            values[index] <= values[index + 1]
            for index in range(len(values) - 1)
        ):
            raise ValueError("segment joint milestones must be strictly descending")
        self.milestones = values
        self.status = {value: "pending" for value in values}
        self.previous_overflow = None
        self.activation_iteration = None
        self.frozen_topology_generation = None
        self._next_event_id = 0
        self._pending_event = None
        self._milestone_queue = []
        self._last_transaction_iteration = None
        self.terminal_failure = None
        self.queue_trace = []
        self.trace = []

    @property
    def active(self):
        return self.activation_iteration is not None

    @property
    def pending_event(self):
        return None if self._pending_event is None else copy.deepcopy(self._pending_event)

    @property
    def queue(self):
        return [float(record["milestone"]) for record in self._milestone_queue]

    @property
    def transaction_active(self):
        return self._pending_event is not None

    @property
    def queue_is_empty(self):
        return not self._milestone_queue

    @property
    def has_pending_work(self):
        return self.transaction_active or not self.queue_is_empty

    def activate(self, *, iteration, overflow):
        value = float(overflow)
        if not math.isfinite(value):
            raise ValueError("segment joint activation overflow must be finite")
        activated = self.activation_iteration is None
        if activated:
            self.activation_iteration = int(iteration)
        return {
            "activated": activated,
            "iteration": int(self.activation_iteration),
            "overflow": value,
        }

    def queue_crossed(self, *, overflow, iteration):
        value = float(overflow)
        if not math.isfinite(value):
            raise ValueError("segment joint overflow must be finite")
        if self.terminal_failure is not None:
            raise RuntimeError("segment joint milestone state has failed terminally")
        previous = self.previous_overflow
        self.previous_overflow = value
        queue_before = self.queue
        newly_queued = []
        if self.active:
            for threshold in self.milestones:
                if self.status[threshold] != "pending" or value > threshold:
                    continue
                self.status[threshold] = "queued"
                self._milestone_queue.append(
                    {
                        "milestone": threshold,
                        "queued_iteration": int(iteration),
                        "queued_overflow": value,
                        "previous_overflow": previous,
                    }
                )
                newly_queued.append(threshold)
        observation = {
            "iteration": int(iteration),
            "overflow": value,
            "previous_overflow": previous,
            "newly_queued": list(newly_queued),
            "queue_before": queue_before,
            "queue_after": self.queue,
        }
        self.queue_trace.append(copy.deepcopy(observation))
        return observation

    def begin_next_after_step(
        self,
        *,
        iteration,
        overflow,
        pair_generation,
        pair_generation_consumed,
        pair_rebuild_serviced,
    ):
        if self.terminal_failure is not None:
            raise RuntimeError("segment joint milestone state has failed terminally")
        if self._pending_event is not None:
            raise RuntimeError("segment joint milestone event was not consumed")
        if not self.active or not self._milestone_queue:
            return None
        iteration = int(iteration)
        if pair_rebuild_serviced or self._last_transaction_iteration == iteration:
            return None
        if pair_generation is None or not pair_generation_consumed:
            return None
        queued = self._milestone_queue.pop(0)
        milestone = float(queued["milestone"])
        self.status[milestone] = "ready"
        first_stage = self.REBOOTSTRAP_TRANSACTION_STAGES[0]
        event = {
            "event_id": int(self._next_event_id),
            "iteration": iteration,
            "overflow": float(overflow),
            "previous_overflow": queued.get("previous_overflow"),
            "milestone": milestone,
            "activation": False,
            "skipped": [],
            "queued_iteration": int(queued["queued_iteration"]),
            "queued_overflow": float(queued["queued_overflow"]),
            "queue_before": [milestone, *self.queue],
            "queue_after_begin": self.queue,
            "pair_generation_before_milestone": int(pair_generation),
            "transaction_stage": first_stage,
            "transaction_stages": list(self.REBOOTSTRAP_TRANSACTION_STAGES),
            "stage_trace": [
                {
                    "stage": first_stage,
                    "status": "ok",
                    "evidence": {
                        "pair_generation": int(pair_generation),
                        "placement_iteration": iteration,
                    },
                }
            ],
        }
        self._next_event_id += 1
        self._pending_event = event
        return copy.deepcopy(event)

    def observe(self, *, overflow, iteration):
        value = float(overflow)
        if not math.isfinite(value):
            raise ValueError("segment joint overflow must be finite")
        if self._pending_event is not None:
            raise RuntimeError("segment joint milestone event was not consumed")
        if self.terminal_failure is not None:
            raise RuntimeError("segment joint milestone state has failed terminally")

        previous = self.previous_overflow
        self.previous_overflow = value
        selected = None
        activation = False
        skipped = []
        if not self.active:
            if value > self.milestones[0]:
                return None
            crossed = [threshold for threshold in self.milestones if value <= threshold]
            selected = min(crossed)
            activation = True
            for threshold in crossed:
                if threshold > selected:
                    self.status[threshold] = "skipped_before_activation"
                    skipped.append(threshold)
            self.activation_iteration = int(iteration)
        elif previous is not None and value < previous:
            crossed = [
                threshold
                for threshold in self.milestones
                if self.status[threshold] == "pending"
                and previous > threshold
                and value <= threshold
            ]
            if crossed:
                selected = min(crossed)
                for threshold in crossed:
                    if threshold > selected:
                        self.status[threshold] = "skipped_multi_crossing"
                        skipped.append(threshold)

        if selected is None:
            return None
        self.status[selected] = "ready"
        event = {
            "event_id": int(self._next_event_id),
            "iteration": int(iteration),
            "overflow": value,
            "previous_overflow": previous,
            "milestone": selected,
            "activation": activation,
            "skipped": list(skipped),
            "transaction_stage": "timing_frame_ready",
            "stage_trace": [
                {
                    "stage": "timing_frame_ready",
                    "status": "ok",
                }
            ],
        }
        self._next_event_id += 1
        self._pending_event = event
        return dict(event)

    def advance(self, event, *, stage, evidence=None):
        if self._pending_event is None:
            raise RuntimeError("no segment joint milestone event is pending")
        if int(event.get("event_id", -1)) != int(self._pending_event["event_id"]):
            raise ValueError("segment joint milestone event identity mismatch")
        stage = str(stage)
        transaction_stages = tuple(
            self._pending_event.get("transaction_stages") or self.TRANSACTION_STAGES
        )
        if stage not in transaction_stages:
            raise ValueError(f"invalid segment joint transaction stage: {stage}")
        current = str(self._pending_event.get("transaction_stage"))
        current_index = transaction_stages.index(current)
        stage_index = transaction_stages.index(stage)
        if stage_index != current_index + 1:
            raise RuntimeError(
                "segment joint transaction stage must advance exactly once: "
                f"{current} -> {stage}"
            )
        self._pending_event["transaction_stage"] = stage
        self._pending_event.setdefault("stage_trace", []).append(
            {
                "stage": stage,
                "status": "ok",
                "evidence": copy.deepcopy(evidence or {}),
            }
        )
        return self.pending_event

    def fail(self, event, *, stage, error, evidence=None):
        if self._pending_event is None:
            raise RuntimeError("no segment joint milestone event is pending")
        if int(event.get("event_id", -1)) != int(self._pending_event["event_id"]):
            raise ValueError("segment joint milestone event identity mismatch")
        record = copy.deepcopy(self._pending_event)
        record.update(
            {
                "status": "failed_terminal",
                "failed_stage": str(stage),
                "error": str(error),
                "failure_evidence": copy.deepcopy(evidence or {}),
                "frozen_topology_generation": self.frozen_topology_generation,
            }
        )
        milestone = float(record["milestone"])
        self.status[milestone] = "failed_terminal"
        self.trace.append(record)
        self.terminal_failure = copy.deepcopy(record)
        self._last_transaction_iteration = int(record["iteration"])
        self._pending_event = None
        return copy.deepcopy(record)

    def set_frozen_topology_generation(self, generation):
        generation = int(generation)
        if generation <= 0:
            raise ValueError("frozen topology generation must be positive")
        if self.frozen_topology_generation not in (None, generation):
            raise RuntimeError("segment joint topology generation changed in-window")
        self.frozen_topology_generation = generation
        return generation

    def consume(self, event, *, transition, require_stage=None):
        if self._pending_event is None:
            raise RuntimeError("no segment joint milestone event is pending")
        if int(event.get("event_id", -1)) != int(self._pending_event["event_id"]):
            raise ValueError("segment joint milestone event identity mismatch")
        if require_stage is not None and str(
            self._pending_event.get("transaction_stage")
        ) != str(require_stage):
            raise RuntimeError(
                "segment joint milestone transaction is incomplete: "
                f"expected {require_stage}, got "
                f"{self._pending_event.get('transaction_stage')}"
            )
        milestone = float(self._pending_event["milestone"])
        record = dict(self._pending_event)
        record["status"] = "consumed"
        record["transition"] = dict(transition or {})
        record["frozen_topology_generation"] = self.frozen_topology_generation
        self.status[milestone] = "consumed"
        self.trace.append(record)
        self._last_transaction_iteration = int(record["iteration"])
        self._pending_event = None
        return dict(record)

    def skip(self, event, *, reason, evidence=None):
        if self._pending_event is None:
            raise RuntimeError("no segment joint milestone event is pending")
        if int(event.get("event_id", -1)) != int(self._pending_event["event_id"]):
            raise ValueError("segment joint milestone event identity mismatch")
        record = copy.deepcopy(self._pending_event)
        record.update(
            {
                "status": "skipped_timing_clean",
                "skip_reason": str(reason),
                "skip_evidence": copy.deepcopy(evidence or {}),
                "frozen_topology_generation": self.frozen_topology_generation,
            }
        )
        milestone = float(record["milestone"])
        self.status[milestone] = "skipped_timing_clean"
        self.trace.append(record)
        self._last_transaction_iteration = int(record["iteration"])
        self._pending_event = None
        return copy.deepcopy(record)

    def skip_next_after_timing_clean_confirmation(
        self,
        *,
        iteration,
        overflow,
        consumed_pair_generation,
        confirmation_pair_generation,
        evidence=None,
    ):
        if self.terminal_failure is not None:
            raise RuntimeError("segment joint milestone state has failed terminally")
        if self._pending_event is not None:
            raise RuntimeError("segment joint milestone event was not consumed")
        if not self._milestone_queue:
            return None
        iteration = int(iteration)
        if self._last_transaction_iteration == iteration:
            return None
        queued = self._milestone_queue.pop(0)
        milestone = float(queued["milestone"])
        record = {
            "event_id": int(self._next_event_id),
            "iteration": iteration,
            "overflow": float(overflow),
            "previous_overflow": queued.get("previous_overflow"),
            "milestone": milestone,
            "queued_iteration": int(queued["queued_iteration"]),
            "queued_overflow": float(queued["queued_overflow"]),
            "queue_before": [milestone, *self.queue],
            "queue_after": self.queue,
            "pair_generation_before_milestone": int(consumed_pair_generation),
            "pair_generation_after_milestone": int(
                confirmation_pair_generation
            ),
            "status": "skipped_timing_clean",
            "skip_reason": "timing_clean_confirmed",
            "skip_evidence": copy.deepcopy(evidence or {}),
            "transition": {
                "accepted_action_count": 0,
                "buffer_count_before": None,
                "buffer_count_after": None,
            },
            "frozen_topology_generation": self.frozen_topology_generation,
            "stage_trace": [
                {
                    "stage": "tdp_objective_and_step_consumed",
                    "status": "ok",
                    "evidence": {
                        "pair_generation": int(consumed_pair_generation),
                        "placement_iteration": iteration,
                    },
                },
                {
                    "stage": "timing_clean_confirmation",
                    "status": "ok",
                    "evidence": copy.deepcopy(evidence or {}),
                },
            ],
        }
        self._next_event_id += 1
        self.status[milestone] = "skipped_timing_clean"
        self.trace.append(copy.deepcopy(record))
        self._last_transaction_iteration = iteration
        return copy.deepcopy(record)

    def summarize(self):
        return {
            "milestones": list(self.milestones),
            "status": [
                {"milestone": threshold, "status": self.status[threshold]}
                for threshold in self.milestones
            ],
            "activation_iteration": self.activation_iteration,
            "previous_overflow": self.previous_overflow,
            "frozen_topology_generation": self.frozen_topology_generation,
            "queue": self.queue,
            "queue_trace": copy.deepcopy(self.queue_trace),
            "last_transaction_iteration": self._last_transaction_iteration,
            "pending_event": self.pending_event,
            "terminal_failure": copy.deepcopy(self.terminal_failure),
            "trace": copy.deepcopy(self.trace),
        }


class JointCoordinator:
    """Thin lifecycle coordinator for placement + sizing + buffering flow."""

    def __init__(self, params, *, buffering_lane=None):
        self.params = params
        self.buffering_lane = buffering_lane
        self.last_buffer_commit_request = None
        self.segment_action_schedule_mode = str(
            getattr(
                params,
                "joint_segment_action_schedule_mode",
                SEGMENT_OVERFLOW_MILESTONE_SCHEDULE,
            )
            or SEGMENT_OVERFLOW_MILESTONE_SCHEDULE
        )
        if self.is_segment_direct_joint and self.segment_action_schedule_mode not in {
            SEGMENT_OVERFLOW_MILESTONE_SCHEDULE,
            SEGMENT_FIXED_WINDOW_SCHEDULE,
        }:
            raise ValueError(
                "invalid segment joint action schedule mode: "
                f"{self.segment_action_schedule_mode}"
            )
        self.segment_milestones = (
            SegmentJointMilestoneState(
                milestones=getattr(
                    params,
                    "joint_segment_milestones",
                    SEGMENT_DIRECT_JOINT_MILESTONES,
                )
            )
            if self.is_segment_direct_joint
            else None
        )
        self.pin2pin_pair_generations = (
            Pin2PinPairGenerationState(
                update_interval=int(
                    getattr(params, "net_weighting_update_interval", 15) or 15
                )
            )
            if self.uses_pin2pin_rebootstrap_schedule
            else None
        )
        self._segment_lane_prepared = False
        self._segment_action_calibration_done = False
        self._segment_route_b_applied = False
        self.segment_route_b_enabled = bool(
            getattr(params, "joint_segment_route_b_enabled", True)
        )
        self.sizing_enabled = bool(
            getattr(params, "joint_segment_sizing_enabled", True)
        )
        self._proximal_objective = None
        self.summary = {
            "status": "initialized",
            "flow_kind": str(getattr(params, "flow_kind", "")),
            "placement_sizing_mode": str(
                getattr(params, "placement_sizing_mode", "")
            ),
            "sizing_enabled": self.sizing_enabled,
            "coordinator_role": "lifecycle_only",
            "uses_joint_op": False,
            "segment_route_b_enabled": self.segment_route_b_enabled,
            "segment_action_schedule_mode": self.segment_action_schedule_mode,
            "activation": {
                "overflow_gate": self.overflow_gate,
                "placement_active_from_start": True,
                "sizing_active_after_overflow_gate": self.sizing_enabled,
                "buffering_active_after_overflow_gate": True,
                "buffer_commit_active_after_overflow_gate": True,
            },
        }
        if self.segment_milestones is not None:
            self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        if self.pin2pin_pair_generations is not None:
            self.summary["pin2pin_pair_generations"] = (
                self.pin2pin_pair_generations.summarize()
            )

    @property
    def is_segment_direct_joint(self):
        return (
            str(getattr(self.params, "joint_quality_profile", "") or "")
            == SEGMENT_DIRECT_JOINT_PROFILE
        )

    @property
    def overflow_gate(self):
        gate = float(getattr(self.params, "joint_buffer_overflow_gate", 0.2))
        if getattr(self.params, "_timing_topology_enable_overflow_threshold_explicit", False):
            gate = max(
                gate,
                float(getattr(self.params, "timing_topology_enable_overflow_threshold", gate)),
            )
        return gate

    @property
    def uses_overflow_milestone_actions(self):
        return (
            self.is_segment_direct_joint
            and self.segment_action_schedule_mode
            == SEGMENT_OVERFLOW_MILESTONE_SCHEDULE
        )

    @property
    def uses_fixed_window_actions(self):
        return (
            self.is_segment_direct_joint
            and self.segment_action_schedule_mode == SEGMENT_FIXED_WINDOW_SCHEDULE
        )

    @property
    def uses_pin2pin_rebootstrap_schedule(self):
        return bool(
            self.uses_overflow_milestone_actions
            and getattr(
                self.params,
                "joint_segment_pin2pin_rebootstrap_enabled",
                True,
            )
            and getattr(self.params, "enable_net_weighting", False)
            and str(getattr(self.params, "net_weighting_scheme", "") or "").lower()
            == "pin2pin"
        )

    @property
    def commit_period(self):
        return int(getattr(self.params, "joint_buffer_commit_period", 100))

    def should_enable_joint_lanes(self, overflow):
        try:
            return float(overflow) <= self.overflow_gate
        except (TypeError, ValueError):
            return False

    def should_commit_buffers(self, iteration, overflow):
        if self.is_segment_direct_joint:
            return False
        period = self.commit_period
        return (
            period > 0
            and int(iteration) > 0
            and int(iteration) % period == 0
            and self.should_enable_joint_lanes(overflow)
        )

    def prepare_lanes(self, *, model=None, data_collections=None):
        self._proximal_objective = getattr(model, "joint_proximal_objective", None)
        lane_summary = None
        if self.is_segment_direct_joint:
            self.summary["prepared"] = False
            self.summary["lane_prepare_deferred_until_activation"] = True
            return self
        if self.buffering_lane is not None:
            self.buffering_lane.prepare(model=model, params=self.params)
            if data_collections is not None:
                self.buffering_lane.attach_state(data_collections)
            lane_summary = self.buffering_lane.summarize()
        self.summary["prepared"] = True
        if lane_summary is not None:
            self.summary["buffering_lane"] = lane_summary
        return self

    def activate_segment_lane(self, *, model, data_collections, topology_generation):
        if not self.is_segment_direct_joint:
            raise RuntimeError("segment joint lane activation requires the canonical profile")
        self._proximal_objective = getattr(model, "joint_proximal_objective", None)
        self.segment_milestones.set_frozen_topology_generation(topology_generation)
        if not self._segment_lane_prepared:
            if self.buffering_lane is None:
                raise RuntimeError("segment joint lane is unavailable")
            existing_state = getattr(data_collections, "buffer_segment_count_state", None)
            state = self.buffering_lane._current_state()
            reuse_existing_state = existing_state is not None and existing_state is state
            if not reuse_existing_state:
                self.buffering_lane.prepare(model=model, params=self.params)
                self.buffering_lane.attach_state(data_collections)
                state = self.buffering_lane._current_state()
            else:
                self.buffering_lane.attach_state(data_collections)
            if state is None or not hasattr(state, "z_param"):
                raise RuntimeError("segment joint lane did not build SegmentCountState")
            if not reuse_existing_state:
                with torch.no_grad():
                    state.z_param.zero_()
                if not bool((state.z_param.detach() == 0.0).all()):
                    raise RuntimeError("segment joint count state must activate from zero")
            self._segment_lane_prepared = True
            self.summary["prepared"] = True
            self.summary["lane_prepare_deferred_until_activation"] = False
            self.summary["buffering_lane"] = self.buffering_lane.summarize()
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        return self.buffering_lane

    def collect_param_groups(self, *, placement_params=None, sizing_params=None):
        groups = []
        if placement_params:
            groups.append(
                {
                    "params": list(placement_params),
                    "group_name": "placement",
                    "joint_lane": "placement",
                    "joint_activation": "always",
                }
            )
        if sizing_params and not self.is_segment_direct_joint:
            groups.append(
                {
                    "params": list(sizing_params),
                    "group_name": "sizing",
                    "joint_lane": "sizing",
                    "joint_activation": "overflow_gate",
                }
            )
        buffer_params = []
        if self.buffering_lane is not None and not self.is_segment_direct_joint:
            buffer_params = list(self.buffering_lane.optimizer_parameters())
        if buffer_params:
            groups.append(
                {
                    "params": buffer_params,
                    "group_name": "buffering",
                    "joint_lane": "buffering",
                    "joint_activation": "overflow_gate",
                }
            )
        self.summary["param_groups"] = [
            {
                "group_name": group.get("group_name"),
                "joint_lane": group.get("joint_lane"),
                "joint_activation": group.get("joint_activation"),
                "param_count": len(group.get("params", ())),
            }
            for group in groups
        ]
        return groups

    def observe_segment_milestone(self, *, overflow, iteration):
        if not self.is_segment_direct_joint:
            return None
        event = self.segment_milestones.observe(
            overflow=overflow,
            iteration=iteration,
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        return event

    def observe_pin2pin_segment_iteration(self, *, overflow, iteration):
        if not self.uses_pin2pin_rebootstrap_schedule:
            raise RuntimeError(
                "Pin2Pin segment observation requires the rebootstrap schedule"
            )
        overflow = float(overflow)
        activation_threshold = float(
            getattr(
                self.params,
                "timing_topology_enable_overflow_threshold",
                0.35,
            )
        )
        activation_due = bool(
            not self.segment_milestones.active
            and overflow <= activation_threshold
        )
        activation = None
        if activation_due:
            activation = self.segment_milestones.activate(
                iteration=iteration,
                overflow=overflow,
            )
        observation = self.segment_milestones.queue_crossed(
            overflow=overflow,
            iteration=iteration,
        )
        result = {
            **observation,
            "activation_threshold": activation_threshold,
            "activation_due": activation_due,
            "activation": activation,
        }
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_pin2pin_segment_observation"] = copy.deepcopy(result)
        return result

    def install_pin2pin_generation(
        self,
        *,
        iteration,
        reason,
        status,
        pair_count,
        metadata=None,
    ):
        if self.pin2pin_pair_generations is None:
            raise RuntimeError("Pin2Pin generation state is unavailable")
        generation = self.pin2pin_pair_generations.install(
            iteration=iteration,
            reason=reason,
            status=status,
            pair_count=pair_count,
            metadata=metadata,
        )
        self.summary["pin2pin_pair_generations"] = (
            self.pin2pin_pair_generations.summarize()
        )
        return generation

    def record_pin2pin_objective_evaluation(self, *, iteration, generation_id):
        if self.pin2pin_pair_generations is None:
            raise RuntimeError("Pin2Pin generation state is unavailable")
        self.pin2pin_pair_generations.record_objective_evaluation(
            iteration=iteration,
            generation_id=generation_id,
        )
        self.summary["pin2pin_pair_generations"] = (
            self.pin2pin_pair_generations.summarize()
        )

    def record_pin2pin_placement_step(self, *, iteration, generation_id):
        if self.pin2pin_pair_generations is None:
            raise RuntimeError("Pin2Pin generation state is unavailable")
        self.pin2pin_pair_generations.record_placement_step(
            iteration=iteration,
            generation_id=generation_id,
        )
        self.summary["pin2pin_pair_generations"] = (
            self.pin2pin_pair_generations.summarize()
        )

    def record_pin2pin_nesterov_step(
        self,
        *,
        iteration,
        generation_id,
        optimizer_eval_count_before,
        optimizer_eval_count_after,
    ):
        before = int(optimizer_eval_count_before)
        after = int(optimizer_eval_count_after)
        if after <= before:
            raise RuntimeError(
                "Nesterov Pin2Pin step did not evaluate the current objective"
            )
        self.record_pin2pin_objective_evaluation(
            iteration=iteration,
            generation_id=generation_id,
        )
        self.record_pin2pin_placement_step(
            iteration=iteration,
            generation_id=generation_id,
        )
        evidence = {
            "iteration": int(iteration),
            "pair_generation": int(generation_id),
            "optimizer_eval_count_before": before,
            "optimizer_eval_count_after": after,
            "optimizer_eval_count_delta": after - before,
        }
        self.summary["last_pin2pin_nesterov_consumption"] = evidence
        return copy.deepcopy(evidence)

    def begin_pin2pin_milestone_after_step(self, *, iteration, overflow):
        if self.pin2pin_pair_generations is None:
            raise RuntimeError("Pin2Pin generation state is unavailable")
        generation = self.pin2pin_pair_generations
        event = self.segment_milestones.begin_next_after_step(
            iteration=iteration,
            overflow=overflow,
            pair_generation=generation.current_generation_id,
            pair_generation_consumed=generation.can_execute_milestone,
            pair_rebuild_serviced=bool(
                generation.install_count_at_iteration(iteration)
            ),
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        return event

    def skip_pin2pin_milestone_after_timing_clean_confirmation(
        self,
        *,
        iteration,
        overflow,
        consumed_pair_generation,
        confirmation_pair_generation,
        evidence=None,
    ):
        record = self.segment_milestones.skip_next_after_timing_clean_confirmation(
            iteration=iteration,
            overflow=overflow,
            consumed_pair_generation=consumed_pair_generation,
            confirmation_pair_generation=confirmation_pair_generation,
            evidence=evidence,
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_milestone"] = copy.deepcopy(record)
        return record

    def advance_segment_milestone(self, *, event, stage, evidence=None):
        if not self.uses_overflow_milestone_actions:
            raise RuntimeError(
                "milestone transaction stages require overflow_milestones scheduling"
            )
        pending = self.segment_milestones.advance(
            event,
            stage=stage,
            evidence=evidence,
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        return pending

    def fail_segment_milestone(self, *, event, stage, error, evidence=None):
        record = self.segment_milestones.fail(
            event,
            stage=stage,
            error=error,
            evidence=evidence,
        )
        self.summary["status"] = "failed_terminal"
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_milestone_failure"] = record
        return record

    def mark_incomplete_pending_milestone(self, *, iteration, reason):
        pair_summary = (
            None
            if self.pin2pin_pair_generations is None
            else self.pin2pin_pair_generations.summarize()
        )
        result = {
            "status": "incomplete_pending_milestone",
            "iteration": int(iteration),
            "reason": str(reason),
            "queue": self.segment_milestones.queue,
            "active_transaction": self.segment_milestones.pending_event,
            "pair_generation_pending": bool(
                self.pin2pin_pair_generations is not None
                and self.pin2pin_pair_generations.pending
            ),
            "pair_generation": (
                None if pair_summary is None else pair_summary["current_generation"]
            ),
        }
        self.summary["status"] = result["status"]
        self.summary["incomplete_pending_milestone"] = copy.deepcopy(result)
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        return result

    @staticmethod
    def _optimizer_owns_parameter(optimizer, parameter):
        return any(
            any(candidate is parameter for candidate in group.get("params", ()))
            for group in getattr(optimizer, "param_groups", ())
        )

    @staticmethod
    def _evaluate_timing_loss(model, pos):
        with torch.no_grad():
            wns, tns, ws, ts = model.timing_obj(pos)
            loss = model._timing_loss(wns, tns, ws, ts)
        if not torch.is_tensor(loss) or int(loss.numel()) != 1:
            raise RuntimeError("segment Route-B timing loss must be scalar")
        value = float(loss.detach().cpu().item())
        if not math.isfinite(value):
            raise RuntimeError("segment Route-B timing loss is non-finite")
        return value

    @staticmethod
    def _buffering_gradient_baseline_loss(buffering_gradient):
        if not isinstance(buffering_gradient, dict):
            return None
        if str(buffering_gradient.get("status", "")) != "ready":
            return None
        if "timing_loss" not in buffering_gradient:
            return None
        value = buffering_gradient["timing_loss"]
        if torch.is_tensor(value):
            if int(value.numel()) != 1:
                raise RuntimeError(
                    "refreshed segment buffering-gradient timing loss must be scalar"
                )
            value = value.detach().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "refreshed segment buffering-gradient timing loss is invalid"
            ) from error
        if not math.isfinite(value):
            raise RuntimeError(
                "refreshed segment buffering-gradient timing loss is non-finite"
            )
        return value

    def _apply_segment_route_b_transition(
        self,
        *,
        optimizer=None,
        model=None,
        pos=None,
        baseline_loss=None,
    ):
        if not self._segment_lane_prepared:
            raise RuntimeError("segment joint lane is not active")
        state = self.buffering_lane._current_state()
        if self._optimizer_owns_parameter(optimizer, state.z_param):
            raise RuntimeError("segment joint z_param must not belong to the optimizer")
        if optimizer is not None and state.z_param in getattr(optimizer, "state", {}):
            raise RuntimeError("segment joint z_param must not have optimizer state")
        raw_grad = state.z_param.grad
        if raw_grad is None:
            raise RuntimeError("shared timing backward did not populate z_param.grad")
        if not bool(torch.isfinite(raw_grad).all()):
            raise RuntimeError("segment joint z_param.grad contains non-finite values")
        counts_before = state.z_param.detach().clone()
        if not bool((counts_before == torch.round(counts_before)).all()):
            raise RuntimeError("segment joint z must be integer before Route-B")
        action_grad = raw_grad.detach().clone()
        proximal_adjustment = {
            "enabled": False,
            "reason": "missing_proximal_objective",
            "finite_increment_applied": False,
        }
        if self._proximal_objective is not None:
            action_grad, proximal_adjustment = (
                self._proximal_objective.adjust_discrete_count_gradient(
                    state.z_param,
                    raw_grad,
                )
            )
        transition = schedule_net_gradient_actions(
            segment_net_index=state.segment_net_index,
            net_ids=state.net_ids,
            segment_ids=state.segment_ids,
            counts=counts_before,
            raw_grad=action_grad,
            max_count=state.max_repeater_count,
            selection_fraction=float(
                getattr(
                    self.params,
                    "joint_segment_buffering_selection_fraction",
                    0.001,
                )
            ),
        )
        gradient_selected_action_count = int(transition.accepted_count)
        batch_acceptance = {
            "policy": "gradient_only_without_objective_evaluator",
            "gradient_selected_action_count": gradient_selected_action_count,
            "trial_prefixes": [],
        }
        if (model is None) != (pos is None):
            raise ValueError("model and pos must be provided together for Route-B acceptance")
        has_objective_evaluator = (
            model is not None
            and callable(getattr(model, "timing_obj", None))
            and callable(getattr(model, "_timing_loss", None))
        )
        profile_enabled = bool(
            getattr(self.params, "timing_obj_profile", False)
            or getattr(self.params, "timing_propagation_profile", False)
        )

        def profile_clock():
            if profile_enabled and torch.is_tensor(pos) and pos.is_cuda:
                torch.cuda.synchronize(pos.device)
            return time.perf_counter()

        if has_objective_evaluator and gradient_selected_action_count > 0:
            timing_eval_profile = {
                "enabled": bool(profile_enabled),
                "synchronized": bool(profile_enabled and pos.is_cuda),
                "baseline_ms": None,
                "trial_ms": [],
            }
            if baseline_loss is None:
                baseline_started_at = profile_clock()
                baseline_loss = self._evaluate_timing_loss(model, pos)
                baseline_finished_at = profile_clock()
                baseline_source = "timing_obj_evaluation"
                baseline_evaluation_count = 1
                timing_eval_profile["baseline_ms"] = float(
                    (baseline_finished_at - baseline_started_at) * 1000.0
                )
            else:
                try:
                    baseline_loss = float(baseline_loss)
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Route-B cached baseline timing loss is invalid"
                    ) from error
                if not math.isfinite(baseline_loss):
                    raise RuntimeError(
                        "Route-B cached baseline timing loss is non-finite"
                    )
                baseline_source = "refreshed_buffer_gradient"
                baseline_evaluation_count = 0
            batch_acceptance = {
                "policy": "monotonic_loss_prefix_backtracking",
                "gradient_selected_action_count": gradient_selected_action_count,
                "baseline_loss": baseline_loss,
                "baseline_source": baseline_source,
                "baseline_evaluation_count": baseline_evaluation_count,
                "trial_evaluation_count": 0,
                "restore_evaluation_count": 0,
                "trial_prefixes": [],
            }
            for selected_count in transition_backtracking_sizes(
                gradient_selected_action_count
            ):
                candidate = take_transition_prefix(
                    transition,
                    counts=counts_before,
                    selected_count=selected_count,
                )
                with torch.no_grad():
                    state.z_param.copy_(candidate.next_counts)
                trial_started_at = profile_clock()
                trial_loss = self._evaluate_timing_loss(model, pos)
                trial_finished_at = profile_clock()
                batch_acceptance["trial_evaluation_count"] += 1
                actual_improvement = baseline_loss - trial_loss
                trial_record = {
                    "selected_action_count": int(selected_count),
                    "loss": trial_loss,
                    "actual_improvement": actual_improvement,
                    "accepted": bool(actual_improvement > 0.0),
                }
                if profile_enabled:
                    trial_ms = float(
                        (trial_finished_at - trial_started_at) * 1000.0
                    )
                    trial_record["timing_eval_ms"] = trial_ms
                    timing_eval_profile["trial_ms"].append(trial_ms)
                batch_acceptance["trial_prefixes"].append(trial_record)
                if actual_improvement > 0.0:
                    transition = candidate
                    break
            else:
                with torch.no_grad():
                    state.z_param.copy_(counts_before)
                if not torch.equal(state.z_param.detach(), counts_before):
                    raise RuntimeError("segment Route-B rejection did not restore z state")
                transition = take_transition_prefix(
                    transition,
                    counts=counts_before,
                    selected_count=0,
                )
                batch_acceptance["restored_loss"] = baseline_loss
                batch_acceptance["restored_loss_source"] = (
                    "baseline_reused_after_exact_z_restore"
                )
                batch_acceptance["restored_loss_evaluated"] = False
                batch_acceptance["restoration_state_exact"] = True
            if profile_enabled:
                batch_acceptance["timing_eval_profile"] = timing_eval_profile
        with torch.no_grad():
            state.z_param.copy_(transition.next_counts)
        counts_after = state.z_param.detach().clone()
        buffer_count_before = int(counts_before.sum().cpu().item())
        buffer_count_after = int(counts_after.sum().cpu().item())
        if buffer_count_after - buffer_count_before != int(transition.accepted_count):
            raise RuntimeError(
                "segment joint Route-B count delta does not match accepted actions"
            )
        selected_ids = transition.selected_segment_ids.detach().cpu().tolist()
        digest = hashlib.sha256(
            ",".join(str(int(value)) for value in selected_ids).encode("ascii")
        ).hexdigest()
        return {
            "accepted_action_count": int(transition.accepted_count),
            "gradient_selected_action_count": gradient_selected_action_count,
            "affected_net_count": int(transition.affected_net_count),
            "eligible_net_count": int(transition.eligible_net_count),
            "prefix_size": int(transition.prefix_size),
            "positive_prefix_count": int(transition.positive_prefix_count),
            "selected_segment_ids": [int(value) for value in selected_ids],
            "selected_segment_digest": digest,
            "raw_grad_norm": float(raw_grad.detach().float().norm().cpu().item()),
            "action_grad_norm": float(action_grad.detach().float().norm().cpu().item()),
            "raw_grad_nonzero_count": int((raw_grad != 0.0).sum().cpu().item()),
            "action_grad_nonzero_count": int((action_grad != 0.0).sum().cpu().item()),
            "proximal_adjustment": proximal_adjustment,
            "batch_acceptance": batch_acceptance,
            "input_state_digest": _tensor_sha256(counts_before),
            "output_state_digest": _tensor_sha256(counts_after),
            "buffer_count_before": buffer_count_before,
            "buffer_count_after": buffer_count_after,
            "optimizer_owns_z": False,
            "optimizer_z_state_count": 0,
        }

    def apply_segment_route_b(
        self,
        *,
        event,
        optimizer=None,
        model=None,
        pos=None,
    ):
        if not self.is_segment_direct_joint:
            raise RuntimeError("Route-B joint transition requires the canonical profile")
        if not self.uses_overflow_milestone_actions:
            raise RuntimeError("milestone Route-B requires overflow_milestones scheduling")
        if event is None:
            return None
        transition_summary = self._apply_segment_route_b_transition(
            optimizer=optimizer,
            model=model,
            pos=pos,
        )
        record = self.segment_milestones.consume(
            event,
            transition=transition_summary,
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_route_b"] = record
        return record

    def complete_segment_milestone(
        self,
        *,
        event,
        sizing,
        refresh,
        buffering_gradient,
        optimizer=None,
        model=None,
        pos=None,
    ):
        if not self.uses_overflow_milestone_actions:
            raise RuntimeError(
                "milestone completion requires overflow_milestones scheduling"
            )
        route_b_started_at = time.perf_counter()
        if self.segment_route_b_enabled:
            route_b = self._apply_segment_route_b_transition(
                optimizer=optimizer,
                model=model,
                pos=pos,
                baseline_loss=self._buffering_gradient_baseline_loss(
                    buffering_gradient
                ),
            )
            route_b["status"] = "applied"
        else:
            state = self.buffering_lane._current_state()
            state_digest = _tensor_sha256(state.z_param)
            buffer_count = int(state.z_param.detach().sum().cpu().item())
            route_b = {
                "status": "disabled",
                "reason": "joint_segment_route_b_disabled",
                "accepted_action_count": 0,
                "selected_segment_ids": [],
                "input_state_digest": state_digest,
                "output_state_digest": state_digest,
                "buffer_count_before": buffer_count,
                "buffer_count_after": buffer_count,
                "optimizer_owns_z": False,
                "optimizer_z_state_count": 0,
            }
        route_b["runtime_ms"] = float(
            (time.perf_counter() - route_b_started_at) * 1000.0
        )
        self.advance_segment_milestone(
            event=event,
            stage="route_b_applied",
            evidence={
                "status": route_b.get("status"),
                "accepted_action_count": int(
                    route_b.get("accepted_action_count", 0) or 0
                ),
                "buffer_count_after": int(route_b.get("buffer_count_after", 0) or 0),
                "input_state_digest": route_b.get("input_state_digest"),
                "output_state_digest": route_b.get("output_state_digest"),
                "runtime_ms": route_b.get("runtime_ms"),
            },
        )
        sizing = copy.deepcopy(sizing or {})
        refresh = copy.deepcopy(refresh or {})
        buffering_gradient = copy.deepcopy(buffering_gradient or {})
        runtime_ms = {
            "sizing_gradient": float(
                dict(sizing.get("frame", {}) or {}).get("runtime_ms", 0.0) or 0.0
            ),
            "sizing_select": float(sizing.get("runtime_ms", 0.0) or 0.0),
            "refresh": float(refresh.get("runtime_ms", 0.0) or 0.0),
            "buffer_timing_forward_backward": float(
                buffering_gradient.get("runtime_ms", 0.0) or 0.0
            ),
            "route_b": float(route_b.get("runtime_ms", 0.0) or 0.0),
        }
        runtime_ms["total"] = float(sum(runtime_ms.values()))
        transition = {
            "sizing": sizing,
            "refresh": refresh,
            "buffering_gradient": buffering_gradient,
            "route_b": copy.deepcopy(route_b),
            "runtime_ms": runtime_ms,
            "accepted_action_count": int(
                route_b.get("accepted_action_count", 0) or 0
            ),
            "buffer_count_before": int(route_b.get("buffer_count_before", 0) or 0),
            "buffer_count_after": int(route_b.get("buffer_count_after", 0) or 0),
        }
        record = self.segment_milestones.consume(
            event,
            transition=transition,
            require_stage="route_b_applied",
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_route_b"] = record
        self.summary["last_segment_milestone"] = record
        return record

    def apply_rebootstrap_milestone_route_b(
        self,
        *,
        event,
        buffering_gradient=None,
        optimizer=None,
        model=None,
        pos=None,
    ):
        if not self.uses_pin2pin_rebootstrap_schedule:
            raise RuntimeError("rebootstrap Route-B requires the Pin2Pin schedule")
        started_at = time.perf_counter()
        if self.segment_route_b_enabled:
            route_b = self._apply_segment_route_b_transition(
                optimizer=optimizer,
                model=model,
                pos=pos,
                baseline_loss=self._buffering_gradient_baseline_loss(
                    buffering_gradient
                ),
            )
            route_b["status"] = "applied"
        else:
            state = self.buffering_lane._current_state()
            state_digest = _tensor_sha256(state.z_param)
            buffer_count = int(state.z_param.detach().sum().cpu().item())
            route_b = {
                "status": "disabled",
                "reason": "joint_segment_route_b_disabled",
                "accepted_action_count": 0,
                "selected_segment_ids": [],
                "input_state_digest": state_digest,
                "output_state_digest": state_digest,
                "buffer_count_before": buffer_count,
                "buffer_count_after": buffer_count,
            }
        route_b["runtime_ms"] = float(
            (time.perf_counter() - started_at) * 1000.0
        )
        self.advance_segment_milestone(
            event=event,
            stage="route_b_applied",
            evidence={
                "status": route_b.get("status"),
                "accepted_action_count": int(
                    route_b.get("accepted_action_count", 0) or 0
                ),
                "buffer_count_after": int(
                    route_b.get("buffer_count_after", 0) or 0
                ),
                "input_state_digest": route_b.get("input_state_digest"),
                "output_state_digest": route_b.get("output_state_digest"),
                "runtime_ms": route_b["runtime_ms"],
            },
        )
        return route_b

    def complete_rebootstrap_segment_milestone(self, *, event, transition):
        record = self.segment_milestones.consume(
            event,
            transition=copy.deepcopy(transition or {}),
            require_stage="tdp_rebootstrap_completed",
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_route_b"] = record
        self.summary["last_segment_milestone"] = record
        return record

    def defer_segment_route_b(self, *, event):
        """Consume activation metadata; apply Route-B at the window boundary."""
        if not self.is_segment_direct_joint:
            raise RuntimeError("segment joint transition requires the canonical profile")
        if event is None:
            return None
        if not self.uses_fixed_window_actions:
            raise RuntimeError("Route-B deferral requires fixed_window_compat scheduling")
        state = self.buffering_lane._current_state()
        record = self.segment_milestones.consume(
            event,
            transition={
                "accepted_action_count": 0,
                "deferred_to_window_boundary": True,
                "buffer_count_before": int(
                    state.z_param.detach().sum().cpu().item()
                ),
            },
        )
        self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        self.summary["last_segment_route_b_activation"] = record
        return record

    def apply_segment_route_b_at_window_boundary(
        self,
        *,
        iteration,
        optimizer=None,
        model=None,
        pos=None,
    ):
        """Run one timing-only integer update after the placement block."""
        if self.is_segment_direct_joint and not self.uses_fixed_window_actions:
            return None
        if not self.is_segment_direct_joint or not self._segment_lane_prepared:
            return None
        if self._segment_route_b_applied:
            return self.summary.get("last_segment_route_b_window_boundary")
        if not self.segment_route_b_enabled:
            state = self.buffering_lane._current_state()
            buffer_count = int(state.z_param.detach().sum().cpu().item())
            record = {
                "status": "disabled",
                "reason": "joint_segment_route_b_disabled",
                "iteration": int(iteration),
                "transition": {
                    "accepted_action_count": 0,
                    "buffer_count_before": buffer_count,
                    "buffer_count_after": buffer_count,
                },
                "frozen_topology_generation": (
                    self.segment_milestones.frozen_topology_generation
                ),
            }
            self._segment_route_b_applied = True
            self.summary["last_segment_route_b_window_boundary"] = record
            self.summary["last_segment_route_b"] = record
            return record
        transition_summary = self._apply_segment_route_b_transition(
            optimizer=optimizer,
            model=model,
            pos=pos,
        )
        record = {
            "status": "window_boundary",
            "iteration": int(iteration),
            "transition": transition_summary,
            "frozen_topology_generation": self.segment_milestones.frozen_topology_generation,
        }
        self._segment_route_b_applied = True
        self.summary["last_segment_route_b_window_boundary"] = record
        self.summary["last_segment_route_b"] = record
        return record

    def apply_segment_virtual_window_boundary(
        self,
        *,
        model,
        pos,
        optimizer,
        iteration,
        window_index,
        placement_steps,
        following_placement_steps,
        final,
    ):
        """Apply one discrete block without rebuilding the placement optimizer."""
        if not self.is_segment_direct_joint:
            return None
        if not self.uses_fixed_window_actions:
            return None
        if self._segment_lane_prepared:
            route_b = self.apply_segment_route_b_at_window_boundary(
                iteration=iteration,
                optimizer=optimizer,
                model=model,
                pos=pos,
            )
        else:
            route_b = {
                "status": "inactive",
                "reason": "segment_joint_window_not_activated",
                "iteration": int(iteration),
                "transition": {
                    "accepted_action_count": 0,
                    "buffer_count_before": 0,
                    "buffer_count_after": 0,
                },
                "frozen_topology_generation": None,
            }
        record = {
            "window_index": int(window_index),
            "status": str(route_b.get("status") or "unknown"),
            "boundary": "final" if final else "virtual",
            "actual_placement_steps": int(placement_steps),
            "following_placement_steps": int(following_placement_steps),
            "route_b": copy.deepcopy(route_b),
            "accepted_action_count": int(
                dict(route_b.get("transition", {}) or {}).get(
                    "accepted_action_count",
                    0,
                )
                or 0
            ),
        }
        windows = self.summary.setdefault("segment_virtual_windows", [])
        windows.append(record)
        if not final:
            self._segment_route_b_applied = False
        if following_placement_steps > 0 and self._proximal_objective is not None:
            record["next_anchor"] = self._proximal_objective.capture_anchors(
                model,
                pos,
                force=True,
            )
        return record

    def calibrate_segment_actions(self, *, model, pos, event):
        """Compare the shared z gradient with explicit +1 timing-loss actions."""
        if not self.is_segment_direct_joint or event is None:
            return None
        if self._segment_action_calibration_done:
            return self.summary.get("segment_action_calibration")
        if not self._segment_lane_prepared:
            raise RuntimeError("segment action calibration requires an active lane")

        state = self.buffering_lane._current_state()
        raw_grad = state.z_param.grad
        if raw_grad is None:
            raise RuntimeError("segment action calibration requires shared z_param.grad")
        if not bool(torch.isfinite(raw_grad).all()):
            raise RuntimeError("segment action calibration received non-finite grad_z")
        counts = state.z_param.detach().clone()
        if not bool((counts == torch.round(counts)).all()):
            raise RuntimeError("segment action calibration requires integer z")

        transition = schedule_net_gradient_actions(
            segment_net_index=state.segment_net_index,
            net_ids=state.net_ids,
            segment_ids=state.segment_ids,
            counts=counts,
            raw_grad=raw_grad,
            max_count=state.max_repeater_count,
        )
        sample_indices = transition.selected_segment_indices[
            :SEGMENT_ACTION_CALIBRATION_SAMPLE_COUNT
        ]
        sample_indices_cpu = sample_indices.detach().cpu().tolist()
        raw_grad_snapshot = raw_grad.detach().clone()
        started_at = time.perf_counter()

        def evaluate_timing_loss():
            with torch.no_grad():
                wns, tns, ws, ts = model.timing_obj(pos)
                loss = model._timing_loss(wns, tns, ws, ts)
            if not torch.is_tensor(loss) or loss.numel() != 1:
                raise RuntimeError("segment action calibration timing loss must be scalar")
            value = float(loss.detach().cpu().item())
            if not math.isfinite(value):
                raise RuntimeError("segment action calibration timing loss is non-finite")
            return value

        baseline_loss = None
        restored_loss = None
        action_rows = []
        try:
            baseline_loss = evaluate_timing_loss()
            for segment_index in sample_indices_cpu:
                segment_index = int(segment_index)
                z_before = float(counts[segment_index].cpu().item())
                with torch.no_grad():
                    state.z_param[segment_index] = z_before + 1.0
                plus_one_loss = evaluate_timing_loss()
                with torch.no_grad():
                    state.z_param[segment_index] = z_before

                predicted_improvement = float(
                    -raw_grad_snapshot[segment_index].detach().cpu().item()
                )
                explicit_improvement = float(baseline_loss - plus_one_loss)
                net_index = int(
                    state.segment_net_index[segment_index].detach().cpu().item()
                )
                action_rows.append(
                    {
                        "segment_index": segment_index,
                        "segment_id": int(
                            state.segment_ids[segment_index].detach().cpu().item()
                        ),
                        "net_index": net_index,
                        "net_id": int(state.net_ids[net_index].detach().cpu().item()),
                        "z_before": z_before,
                        "z_after": z_before + 1.0,
                        "raw_grad_z": float(
                            raw_grad_snapshot[segment_index].detach().cpu().item()
                        ),
                        "predicted_improvement": predicted_improvement,
                        "plus_one_loss": plus_one_loss,
                        "explicit_improvement": explicit_improvement,
                        "sign_agreement": _same_nonzero_sign(
                            predicted_improvement,
                            explicit_improvement,
                        ),
                    }
                )
        finally:
            with torch.no_grad():
                state.z_param.copy_(counts)
            if baseline_loss is not None:
                restored_loss = evaluate_timing_loss()

        if not torch.equal(state.z_param.detach(), counts):
            raise RuntimeError("segment action calibration did not restore integer z")
        if state.z_param.grad is None or not torch.equal(
            state.z_param.grad.detach(), raw_grad_snapshot
        ):
            raise RuntimeError("segment action calibration changed shared z_param.grad")
        if not math.isclose(
            baseline_loss,
            restored_loss,
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise RuntimeError("segment action calibration did not restore timing state")

        predicted = [row["predicted_improvement"] for row in action_rows]
        explicit = [row["explicit_improvement"] for row in action_rows]
        agreement_count = sum(bool(row["sign_agreement"]) for row in action_rows)
        payload = {
            "artifact": "segment_joint_action_calibration",
            "artifact_version": 1,
            "profile": SEGMENT_DIRECT_JOINT_PROFILE,
            "objective": "PlaceObj._timing_loss(PlaceObj.timing_obj(pos))",
            "timing_frame": "same_x_t_z_t_before_optimizer_and_route_b",
            "event_id": int(event["event_id"]),
            "iteration": int(event["iteration"]),
            "milestone": float(event["milestone"]),
            "topology_generation": self.segment_milestones.frozen_topology_generation,
            "sample_policy": "first_route_b_selected_unique_net_actions",
            "sample_count_requested": SEGMENT_ACTION_CALIBRATION_SAMPLE_COUNT,
            "sample_count": len(action_rows),
            "baseline_loss": baseline_loss,
            "restored_loss": restored_loss,
            "no_second_backward": True,
            "integer_state_restored": True,
            "shared_gradient_preserved": True,
            "actions": action_rows,
            "sign_agreement_count": agreement_count,
            "sign_agreement_ratio": (
                None if not action_rows else agreement_count / len(action_rows)
            ),
            "rank_correlation": _rank_correlation(predicted, explicit),
            "runtime_ms": float((time.perf_counter() - started_at) * 1000.0),
        }
        result_dir = str(getattr(self.params, "result_dir", "") or "")
        artifact_path = ""
        if result_dir:
            design_name = getattr(self.params, "design_name", "design")
            design_name = design_name() if callable(design_name) else str(design_name)
            artifact_path = os.path.join(
                result_dir,
                f"{design_name}_segment_joint_action_calibration.json",
            )
            os.makedirs(result_dir, exist_ok=True)
            with open(artifact_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
        payload["artifact_path"] = artifact_path
        self._segment_action_calibration_done = True
        self.summary["segment_action_calibration"] = payload
        logging.info(
            "segment joint action calibration samples=%d sign_agreement=%s "
            "rank_correlation=%s wall_ms=%.3f",
            len(action_rows),
            payload["sign_agreement_ratio"],
            payload["rank_correlation"],
            payload["runtime_ms"],
        )
        return copy.deepcopy(payload)

    def record_segment_joint_backward_evidence(
        self,
        *,
        model,
        pos,
        iteration,
        overflow,
    ):
        if not self.is_segment_direct_joint or not self.segment_milestones.active:
            return None
        state = self.buffering_lane._current_state()
        provider = getattr(model, "_relaxed_buffer_dynamic_net_provider", None)
        provider_metadata = dict(getattr(provider, "metadata", {}) or {})
        gradient_stats = copy.deepcopy(
            provider_metadata.get("live_geometry_gradient_stats", {}) or {}
        )
        z_value = state.z_param.detach()
        z_grad = state.z_param.grad
        pos_grad = getattr(pos, "grad", None)

        def tensor_stats(value):
            if value is None:
                return None
            detached = value.detach()
            finite = torch.isfinite(detached)
            finite_values = detached[finite]
            return {
                "count": int(detached.numel()),
                "finite_count": int(finite.sum().cpu().item()),
                "nonzero_count": int((finite_values != 0.0).sum().cpu().item()),
                "norm": float(finite_values.float().norm().cpu().item()),
                "max_abs": (
                    0.0
                    if not int(finite_values.numel())
                    else float(finite_values.abs().max().cpu().item())
                ),
            }

        explicit_source_mapping_count = int(
            provider_metadata.get(
                "live_rc_explicit_source_edge_mapping_count",
                0,
            )
            or 0
        )
        identity_source_mapping_count = int(
            provider_metadata.get(
                "live_rc_identity_source_edge_mapping_count",
                0,
            )
            or 0
        )
        objective_profile = dict(
            getattr(model, "debug_obj_and_grad_profile", {}) or {}
        )
        timing_op = getattr(
            getattr(model, "op_collections", None),
            "timing_propagation_op",
            None,
        )
        timing_profile = dict(getattr(timing_op, "last_profile_payload", {}) or {})
        virtual_density_op = getattr(
            getattr(model, "op_collections", None),
            "virtual_cell_density_op",
            None,
        )
        entry = {
            "iteration": int(iteration),
            "overflow": float(overflow),
            "topology_generation": self.segment_milestones.frozen_topology_generation,
            "live_geometry_forward_id": provider_metadata.get(
                "live_geometry_forward_id"
            ),
            "live_geometry_topology_epoch": provider_metadata.get(
                "live_geometry_topology_epoch"
            ),
            "live_geometry_census": copy.deepcopy(
                provider_metadata.get("live_geometry_census")
            ),
            "rectilinear_path_policy": provider_metadata.get(
                "live_geometry_rectilinear_path_policy"
            ),
            "source_edge_mapping_status": (
                "explicit_source_prepared_edge_index"
                if explicit_source_mapping_count > 0
                else "identity_full_view"
                if identity_source_mapping_count > 0
                else "missing"
            ),
            "source_edge_mapping_view_count": (
                explicit_source_mapping_count + identity_source_mapping_count
            ),
            "source_edge_mapping_index_count": int(
                provider_metadata.get("live_rc_source_edge_index_count", 0) or 0
            ),
            "backend_requested": provider_metadata.get(
                "segment_count_timing_backend_requested"
            ),
            "backend_used": provider_metadata.get(
                "segment_count_timing_backend_used"
            ),
            "backend_fallback_reason": provider_metadata.get(
                "segment_count_timing_backend_fallback_reason"
            ),
            "segment_transfer_backend_used": provider_metadata.get(
                "segment_transfer_backend_used"
            ),
            "live_rc_full_build_count": provider_metadata.get(
                "live_rc_full_build_count", 0
            ),
            "live_rc_full_build_ms": provider_metadata.get(
                "live_rc_full_build_ms", 0.0
            ),
            "live_rc_gather_count": provider_metadata.get(
                "live_rc_gather_count", 0
            ),
            "live_rc_gather_ms": provider_metadata.get("live_rc_gather_ms", 0.0),
            "segment_transfer_forward_ms": provider_metadata.get(
                "segment_transfer_forward_ms", 0.0
            ),
            "timing_propagation_runtime_ms": (
                None
                if timing_op is None
                else float(getattr(timing_op, "last_runtime_seconds", 0.0) or 0.0)
                * 1000.0
            ),
            "timing_propagation_stage_runtime_ms": copy.deepcopy(
                timing_profile.get("stage_runtime_ms", {}) or {}
            ),
            "objective_forward_ms": objective_profile.get(
                "obj_and_grad_obj_fn_ms"
            ),
            "objective_backward_ms": objective_profile.get(
                "objective_backward_ms"
            ),
            "objective_total_ms": objective_profile.get("obj_and_grad_total_ms"),
            "z_distribution": {
                "count": int(z_value.numel()),
                "sum": float(z_value.sum().cpu().item()),
                "min": float(z_value.min().cpu().item()),
                "max": float(z_value.max().cpu().item()),
                "nonzero_count": int((z_value != 0.0).sum().cpu().item()),
                "integer": bool((z_value == torch.round(z_value)).all().cpu().item()),
            },
            "grad_pos": tensor_stats(pos_grad),
            "grad_z": tensor_stats(z_grad),
            "live_geometry_gradients": gradient_stats,
            "virtual_cell_density": copy.deepcopy(
                getattr(virtual_density_op, "last_metadata", {}) or {}
            ),
        }
        trace = self.summary.setdefault("segment_joint_backward_evidence_trace", [])
        trace.append(entry)
        self.summary["last_segment_joint_backward_evidence"] = entry
        return copy.deepcopy(entry)

    def apply_activation_to_optimizer(self, optimizer, *, overflow, sizing_lr, buffering_lr):
        enabled = self.should_enable_joint_lanes(overflow)
        for group in getattr(optimizer, "param_groups", ()):
            lane = group.get("joint_lane") or group.get("group_name")
            if lane == "sizing":
                group["lr"] = float(sizing_lr) if enabled else 0.0
            elif lane == "buffering":
                group["lr"] = float(buffering_lr) if enabled else 0.0
        self.summary["last_activation"] = {
            "enabled": bool(enabled),
            "overflow": None if overflow is None else float(overflow),
            "overflow_gate": self.overflow_gate,
        }
        return enabled

    def post_step_clamp(self, *, iteration, optimizer=None):
        metrics = {}
        if self.is_segment_direct_joint:
            if not self._segment_lane_prepared:
                return metrics
            state = self.buffering_lane._current_state()
            counts = state.z_param.detach()
            if not bool((counts == torch.round(counts)).all()):
                raise RuntimeError("segment joint z became fractional")
            if not bool(
                ((counts >= 0.0) & (counts <= float(state.max_repeater_count))).all()
            ):
                raise RuntimeError("segment joint z left its legal range")
            metrics["z_param_min"] = float(counts.min().cpu().item()) if counts.numel() else None
            metrics["z_param_max"] = float(counts.max().cpu().item()) if counts.numel() else None
            metrics["z_integer_invariant"] = True
            return metrics
        if self.buffering_lane is not None:
            self.buffering_lane.after_step(iteration, None, metrics)
        if metrics:
            self.summary["last_post_step_clamp"] = dict(metrics)
        return metrics

    def gradient_metrics(self, optimizer):
        result = {}
        for group in getattr(optimizer, "param_groups", ()):
            lane = group.get("joint_lane") or group.get("group_name") or "unknown"
            sq_sum = 0.0
            nonnull = 0
            for param in group.get("params", ()):
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                nonnull += 1
                grad_detached = grad.detach().float()
                if grad_detached.numel() == 0:
                    continue
                sq_sum += float(torch.sum(grad_detached * grad_detached).cpu().item())
            result[f"{lane}_grad_param_count"] = int(nonnull)
            result[f"{lane}_grad_norm"] = math.sqrt(max(sq_sum, 0.0))
        self.summary["last_gradient_metrics"] = dict(result)
        return result

    def prepare_buffer_commit_request(
        self,
        *,
        iteration,
        optimizer=None,
        reason="joint_buffer_commit",
        live_projection_context=None,
        final_physical_context=None,
    ):
        if self.buffering_lane is None:
            result = {"status": "skipped", "reason": "missing_buffering_lane"}
            self.summary["last_commit"] = result
            return result
        total_start = time.perf_counter()
        logging.info("joint buffer projection start iteration=%s", iteration)
        projection_start = time.perf_counter()
        projection_result = self.buffering_lane.project(
            iteration=iteration,
            final=reason == "joint_buffer_final_commit",
            live_projection_context=live_projection_context,
        )
        projection_wall_ms = float((time.perf_counter() - projection_start) * 1000.0)
        logging.info(
            "joint buffer final projection done projected=%s wall_ms=%.3f",
            int(getattr(projection_result, "projected_buffer_count", 0) or 0),
            projection_wall_ms,
        )
        refresh_start = time.perf_counter()
        refresh_summary = self.buffering_lane.refresh_runtime_state(projection_result)
        refresh_wall_ms = float((time.perf_counter() - refresh_start) * 1000.0)
        logging.info("joint buffer runtime refresh done wall_ms=%.3f", refresh_wall_ms)
        request_start = time.perf_counter()
        final_physical_context = dict(final_physical_context or {})
        commit_request = self.buffering_lane.build_commit_request(
            projection_result,
            iteration=iteration,
            reason=reason,
            runtime_refresh=refresh_summary,
            sizing_actions=final_physical_context.get("sizing_actions"),
            sizing_cell_ids=final_physical_context.get("sizing_cell_ids"),
            placement_node_x=final_physical_context.get("placement_node_x"),
            placement_node_y=final_physical_context.get("placement_node_y"),
        )
        request_wall_ms = float((time.perf_counter() - request_start) * 1000.0)
        self.last_buffer_commit_request = commit_request
        result = {
            "status": "request_ready",
            "iteration": int(iteration),
            "projected_buffer_count": int(
                getattr(projection_result, "projected_buffer_count", 0) or 0
            ),
            "sizing_action_count": len(
                getattr(commit_request, "sizing_actions", ()) or ()
            ),
            "projection_wall_ms": projection_wall_ms,
            "runtime_refresh_wall_ms": refresh_wall_ms,
            "request_wall_ms": request_wall_ms,
            "total_wall_ms": float((time.perf_counter() - total_start) * 1000.0),
            "runtime_refresh": dict(refresh_summary or {}),
            "commit_request": commit_request.to_summary(),
        }
        self.summary["last_commit_request"] = result
        return result

    def summarize(self):
        if self.buffering_lane is not None:
            self.summary["buffering_lane"] = self.buffering_lane.summarize()
        if self.segment_milestones is not None:
            self.summary["segment_direct_joint"] = self.segment_milestones.summarize()
        if self.pin2pin_pair_generations is not None:
            self.summary["pin2pin_pair_generations"] = (
                self.pin2pin_pair_generations.summarize()
            )
        return copy.deepcopy(self.summary)
