"""Drift-calibrated temperature of the AT fan-in merges in the timing VJP (timing_at_merge_vjp_tau="drift").

At an AT merge the VJP splits the output adjoint over the fan-in candidates by softmax(AT / tau).  With the gap
between the two best candidates drifting between weight publications by a noise of scale sigma, the probability
that a candidate is the maximum at the next publication is Phi(gap / sigma) ~ logistic(1.70 * gap / sigma), i.e. a
softmax with tau = sigma / 1.70.  sigma is measured, not tuned: in the reverse of publication k, at each AT merge,
the gap of the two best sources of publication k-1 is re-evaluated at the same candidate positions; sigma is
1.4826 x the adjoint-weighted median absolute drift.  It sets tau for publication k+1 (lower bound: the base tau).
Only the VJP softmax of the AT merges uses it; the forward and every other merge keep the base tau.
(docs/2026-10-30-todo-reorganization-after-method-freeze.md 11)
"""

from __future__ import annotations

import math

import torch

LOGISTIC_SCALE = 1.70  # logistic(1.70 x) approximates Phi(x)
MAD_TO_SIGMA = 1.4826


def top_two(groups, values, count):
    """Per group: positions of the best and second-best value (second = len(values) if absent) and their gap."""
    big = values.numel()
    position = torch.arange(big, device=values.device)
    best = torch.full((count,), -math.inf, device=values.device, dtype=values.dtype).scatter_reduce(
        0, groups, values, "amax", include_self=False)
    first = torch.full((count,), big, device=values.device, dtype=torch.long).scatter_reduce(
        0, groups, torch.where(values == best[groups], position, torch.full_like(position, big)), "amin")
    rest = torch.where(position == first[groups], torch.full_like(values, -math.inf), values)
    second = torch.full((count,), -math.inf, device=values.device, dtype=values.dtype).scatter_reduce(
        0, groups, rest, "amax", include_self=False)
    second_pos = torch.full((count,), big, device=values.device, dtype=torch.long).scatter_reduce(
        0, groups, torch.where((rest == second[groups]) & torch.isfinite(rest), position, torch.full_like(position, big)),
        "amin")
    return first, second_pos, best - second


def weighted_median(values, weights):
    order = torch.argsort(values)
    cumulative = torch.cumsum(weights[order], 0)
    index = int(torch.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(values[order][min(index, values.numel() - 1)])


class AtMergeTemperature:
    """Per-publication sigma of the AT merge gap drift and the tau it implies for the next publication.

    ``observe`` only queues device tensors; ``finish`` runs top_two for all merges of the reverse at once (the groups
    of each merge offset past the previous ones) and gathers the drift from the previous publication's stored
    positions.  With the same merge layout as that publication and no repeated key, one gather replaces the per-merge
    ones; otherwise each merge is matched by key as before.  The values, their order and the host syncs of
    ``finish`` are the only ones: a reverse used to sync three times per merge on boolean indexing.
    """

    def __init__(
        self,
        base_tau,
        *,
        step_normalize: bool = False,
        min_horizon_steps: float = 0.0,
        leg_floor_tau_ps: float = 0.0,
    ):
        self.base_tau = float(base_tau)
        self.step_normalize = bool(step_normalize)
        self.min_horizon_steps = max(float(min_horizon_steps or 0.0), 0.0)
        self.leg_floor_tau_ps = max(float(leg_floor_tau_ps or 0.0), 0.0)
        self.tau = max(self.base_tau, self.leg_floor_tau_ps)
        self.previous = None  # (layout, first, second, gap): global positions of the last observed reverse
        self.current = None
        self.forward_id = None
        self.history = []
        self._pending_iteration = None
        self._pending_next_steps = None
        self._last_iteration = None
        self._ema_horizon = None

    def set_schedule_context(self, *, iteration=None, predicted_next_steps=None):
        """Supply outer GP iteration and predicted upcoming refresh interval before ``begin``."""
        self._pending_iteration = None if iteration is None else int(iteration)
        self._pending_next_steps = (
            None
            if predicted_next_steps is None
            else max(float(predicted_next_steps), 1.0)
        )

    def begin(self, forward_id):
        """Start observing the reverse of a new forward; a repeated reverse of the same forward is not observed."""
        if forward_id == self.forward_id:
            self.current = None
            return
        self.forward_id = forward_id
        self.current = dict(
            layout=[],
            groups=[],
            values=[],
            weight=[],
            seen=[],
            group_count=0,
            iteration=self._pending_iteration,
            predicted_next_steps=self._pending_next_steps,
        )
        self._pending_iteration = None
        self._pending_next_steps = None

    def observe(self, key, groups, values, count, weight):
        """Queue one AT merge (device tensors only; the host syncs happen once, in ``finish``)."""
        if self.current is None or values.numel() == 0 or count == 0:
            return
        weight = weight.detach().double().reshape(-1)
        self.current["layout"].append((key, values.numel(), count))
        self.current["groups"].append(groups.long().reshape(-1) + self.current["group_count"])
        self.current["group_count"] += count
        self.current["values"].append(values.detach().to(torch.float64, copy=True).reshape(-1))
        self.current["weight"].append(weight)
        self.current["seen"].append(weight.sum())

    def _drift(self, layout, values, weight):
        """|gap now - gap then| at the previous publication's candidate positions and their weights, merge by merge."""
        if self.previous is None:
            return [], []
        p_layout, p_first, p_second, p_gap = self.previous
        if layout == p_layout and len({entry[0] for entry in layout}) == len(layout):
            ok = (p_second < values.numel()) & torch.isfinite(p_gap)
            now = values[p_first[ok]] - values[p_second[ok]]
            return [(now - p_gap[ok]).abs()], [weight[ok]]
        last, value_base, group_base = {}, 0, 0  # a key observed twice keeps its last observation
        for key, n, count in p_layout:
            last[key] = (value_base, group_base, n, count)
            value_base, group_base = value_base + n, group_base + count
        drift, weights, value_base, group_base = [], [], 0, 0
        for key, n, count in layout:
            old = last.get(key)
            if old is not None and old[2:] == (n, count):
                p_value, p_group = old[0], old[1]
                first = p_first[p_group : p_group + count] - p_value
                second = p_second[p_group : p_group + count] - p_value
                gap = p_gap[p_group : p_group + count]
                ok = (second < n) & torch.isfinite(gap)
                local = values[value_base : value_base + n]
                drift.append((local[first[ok]] - local[second[ok]] - gap[ok]).abs())
                weights.append(weight[group_base : group_base + count][ok])
            value_base, group_base = value_base + n, group_base + count
        return drift, weights

    def finish(self):
        """Close the observed reverse; returns its record (or None if this reverse was not observed)."""
        if self.current is None:
            return None
        current, used, sigma, matched = self.current, self.tau, None, 0.0
        cur_iter = current.get("iteration")
        next_steps = current.get("predicted_next_steps")
        elapsed_steps = (
            int(cur_iter) - int(self._last_iteration)
            if (
                cur_iter is not None
                and self._last_iteration is not None
                and int(cur_iter) > int(self._last_iteration)
            )
            else None
        )
        if cur_iter is not None:
            self._last_iteration = int(cur_iter)

        seen = float(torch.stack(current["seen"]).sum()) if current["seen"] else 0.0
        stored, drift, weight = None, [], []
        if current["layout"]:
            values = torch.cat(current["values"])
            first, second, gap = top_two(torch.cat(current["groups"]), values, current["group_count"])
            drift, weight = self._drift(current["layout"], values, torch.cat(current["weight"]))
            stored = (current["layout"], first, second, gap)

        if not self.step_normalize and self.leg_floor_tau_ps <= 0.0:
            if drift:
                drift, weight = torch.cat(drift).cpu(), torch.cat(weight).cpu()
                matched = float(weight.sum())
                if matched > 0:
                    sigma = MAD_TO_SIGMA * weighted_median(drift, weight)
                    self.tau = max(self.base_tau, sigma / LOGISTIC_SCALE)
            record = dict(
                forward_id=self.forward_id,
                tau_used_ps=used,
                sigma_delta_ps=sigma,
                tau_next_ps=self.tau,
                matched_weight_share=(matched / seen if seen > 0 else None),
            )
        else:
            floor_tau = max(self.base_tau, self.leg_floor_tau_ps)
            if next_steps is not None and float(next_steps) > 0.0:
                ns_f = float(next_steps)
                self._ema_horizon = (
                    ns_f
                    if self._ema_horizon is None
                    else (0.65 * ns_f + 0.35 * self._ema_horizon)
                )
            raw_horizon = (
                self._ema_horizon
                if self._ema_horizon is not None
                else (
                    float(elapsed_steps)
                    if elapsed_steps is not None
                    else max(self.min_horizon_steps, 1.0)
                )
            )
            eff_horizon = max(float(raw_horizon), self.min_horizon_steps, 1.0)
            sigma_step = None
            if drift:
                drift, weight = torch.cat(drift).cpu(), torch.cat(weight).cpu()
                matched = float(weight.sum())
                if matched > 0:
                    sigma = MAD_TO_SIGMA * weighted_median(drift, weight)
                    if self.step_normalize:
                        denom_steps = (
                            float(elapsed_steps)
                            if (elapsed_steps is not None and elapsed_steps > 0)
                            else eff_horizon
                        )
                        sigma_step = sigma / max(denom_steps, 1.0)
                        drift_tau = (sigma_step * eff_horizon) / LOGISTIC_SCALE
                    else:
                        drift_tau = sigma / LOGISTIC_SCALE
                    self.tau = max(floor_tau, drift_tau)
                else:
                    self.tau = max(floor_tau, self.tau)
            else:
                self.tau = max(floor_tau, self.tau)
            record = dict(
                forward_id=self.forward_id,
                tau_used_ps=used,
                sigma_delta_ps=sigma,
                sigma_step_ps=sigma_step,
                elapsed_steps=elapsed_steps,
                eff_horizon_steps=eff_horizon,
                leg_floor_tau_ps=self.leg_floor_tau_ps,
                tau_next_ps=self.tau,
                matched_weight_share=(matched / seen if seen > 0 else None),
            )
        self.history.append(record)
        self.previous, self.current = stored, None
        return record
