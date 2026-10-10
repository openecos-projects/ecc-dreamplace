#!/usr/bin/env python

import math
from dataclasses import dataclass, field
from typing import Iterable


def _as_float_list(values) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    return [float(value) for value in values]


def _as_int_list(values) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    return [int(value) for value in values]


@dataclass
class DynamicCriticalEndpointSelector:
    top_k: int = 256
    slack_window_ps: float = 100.0
    refresh_interval: int = 10
    hysteresis_interval: int = 3
    full_refresh_interval: int = 50
    active_endpoint_ids: set[int] = field(default_factory=set, init=False)
    last_selected_iteration: dict[int, int] = field(default_factory=dict, init=False)
    last_stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if float(self.slack_window_ps) < 0.0:
            raise ValueError("slack_window_ps must be non-negative")
        if int(self.refresh_interval) <= 0:
            raise ValueError("refresh_interval must be positive")
        if int(self.hysteresis_interval) < 0:
            raise ValueError("hysteresis_interval must be non-negative")
        if int(self.full_refresh_interval) < 0:
            raise ValueError("full_refresh_interval must be non-negative")
        self.top_k = int(self.top_k)
        self.slack_window_ps = float(self.slack_window_ps)
        self.refresh_interval = int(self.refresh_interval)
        self.hysteresis_interval = int(self.hysteresis_interval)
        self.full_refresh_interval = int(self.full_refresh_interval)

    def should_refresh(self, iteration: int) -> bool:
        return int(iteration) % self.refresh_interval == 0

    def refresh(
        self,
        iteration: int,
        endpoint_ids: Iterable[int],
        endpoint_slack_ps: Iterable[float],
    ) -> tuple[set[int], dict]:
        endpoint_ids = _as_int_list(endpoint_ids)
        endpoint_slack_ps = _as_float_list(endpoint_slack_ps)
        if len(endpoint_ids) != len(endpoint_slack_ps):
            raise ValueError("endpoint_ids and endpoint_slack_ps must have the same length")

        previous_active = set(self.active_endpoint_ids)
        finite_pairs = [
            (endpoint_id, slack)
            for endpoint_id, slack in zip(endpoint_ids, endpoint_slack_ps)
            if math.isfinite(slack)
        ]
        if not finite_pairs:
            self.active_endpoint_ids = set()
            self.last_stats = self._build_stats(
                iteration=iteration,
                previous_active=previous_active,
                active=set(),
                top_k_selected=set(),
                slack_window_selected=set(),
                hysteresis_selected=set(),
                full_refresh_selected=set(),
                finite_slacks=[],
            )
            return set(), dict(self.last_stats)

        finite_pairs.sort(key=lambda item: (item[1], item[0]))
        top_k_selected = {endpoint_id for endpoint_id, _ in finite_pairs[: self.top_k]}
        wns = finite_pairs[0][1]
        slack_threshold = wns + self.slack_window_ps
        slack_window_selected = {
            endpoint_id for endpoint_id, slack in finite_pairs if slack <= slack_threshold
        }
        base_selected = top_k_selected | slack_window_selected

        full_refresh_selected: set[int] = set()
        if (
            self.full_refresh_interval > 0
            and int(iteration) > 0
            and int(iteration) % self.full_refresh_interval == 0
        ):
            full_refresh_selected = {endpoint_id for endpoint_id, _ in finite_pairs}

        hysteresis_selected = {
            endpoint_id
            for endpoint_id, selected_iteration in self.last_selected_iteration.items()
            if int(iteration) - int(selected_iteration) <= self.hysteresis_interval
        }
        valid_endpoint_set = {endpoint_id for endpoint_id, _ in finite_pairs}
        hysteresis_selected &= valid_endpoint_set

        active = base_selected | hysteresis_selected | full_refresh_selected

        for endpoint_id in base_selected | full_refresh_selected:
            self.last_selected_iteration[endpoint_id] = int(iteration)
        stale_cutoff = int(iteration) - max(self.hysteresis_interval, self.refresh_interval)
        self.last_selected_iteration = {
            endpoint_id: selected_iteration
            for endpoint_id, selected_iteration in self.last_selected_iteration.items()
            if selected_iteration >= stale_cutoff and endpoint_id in valid_endpoint_set
        }

        self.active_endpoint_ids = set(active)
        self.last_stats = self._build_stats(
            iteration=iteration,
            previous_active=previous_active,
            active=active,
            top_k_selected=top_k_selected,
            slack_window_selected=slack_window_selected,
            hysteresis_selected=hysteresis_selected - base_selected - full_refresh_selected,
            full_refresh_selected=full_refresh_selected,
            finite_slacks=[slack for _, slack in finite_pairs],
        )
        return set(active), dict(self.last_stats)

    def _build_stats(
        self,
        *,
        iteration: int,
        previous_active: set[int],
        active: set[int],
        top_k_selected: set[int],
        slack_window_selected: set[int],
        hysteresis_selected: set[int],
        full_refresh_selected: set[int],
        finite_slacks: list[float],
    ) -> dict:
        overlap = active & previous_active
        previous_count = len(previous_active)
        if finite_slacks:
            wns = min(finite_slacks)
            tns = sum(min(slack, 0.0) for slack in finite_slacks)
        else:
            wns = None
            tns = None
        return {
            "iteration": int(iteration),
            "active_endpoint_count": len(active),
            "previous_active_endpoint_count": previous_count,
            "active_overlap_count": len(overlap),
            "active_overlap_fraction": (
                float(len(overlap)) / float(previous_count) if previous_count else None
            ),
            "selected_by_top_k_count": len(top_k_selected),
            "selected_by_slack_window_count": len(slack_window_selected),
            "selected_by_hysteresis_count": len(hysteresis_selected),
            "selected_by_full_refresh_count": len(full_refresh_selected),
            "full_endpoint_count": len(finite_slacks),
            "full_endpoint_wns_ps": wns,
            "full_endpoint_tns_ps": tns,
        }
