"""State policy for enabling, disabling, and recovering L-shape guidance."""

import logging
import math

from dreamplace.ops.routability.profile_timing import l_shape_log_verbose


class LShapePolicy:
    def __init__(self, params):
        self.params = params

    def reset_auto_disable_state(self, model):
        model._l_shape_auto_disabled = False
        model._l_shape_auto_disable_state = {
            "update_count": 0,
            "best_lcost": None,
            "best_iteration": None,
            "prev_ov_ema": None,
            "last_ov_ema_improvement": None,
            "overshoot_streak": 0,
            "rebound_streak": 0,
            "plateau_streak": 0,
        }

    def reset_reenable_state(self, model):
        base_threshold = float(getattr(self.params, "l_shape_overflow_threshold", 0.2))
        model._l_shape_reenable_threshold_base = base_threshold
        model._l_shape_reenable_threshold = base_threshold
        model._l_shape_reenable_count = 0
        model._l_shape_reenable_last_overflow = None
        model._l_shape_reenable_descend_streak = 0
        model._l_shape_ratio_last = None
        model._l_shape_ratio_rise_streak = 0
        model._l_shape_ratio_guard_disabled = False
        model._l_shape_inflation_guard_disabled = False

    @staticmethod
    def reset_overflow_state(model):
        model._l_shape_overflow_ema = None
        model._l_shape_overflow_last = None

    @staticmethod
    def reset_reenable_progress(model):
        model._l_shape_reenable_last_overflow = None
        model._l_shape_reenable_descend_streak = 0

    def update_overflow_target(self, model, iteration, overflow, overflow_ratio):
        params = self.params
        beta = float(getattr(params, "l_shape_overflow_ema_beta", 0.8))
        beta = max(0.0, min(0.999, beta))
        if not hasattr(model, "_l_shape_overflow_ema") or model._l_shape_overflow_ema is None:
            model._l_shape_overflow_ema = overflow_ratio
            model._l_shape_overflow_last = overflow_ratio
            if l_shape_log_verbose(params) >= 1:
                logging.info(
                    "L-shape overflow outer-loop initialized: "
                    "ov_raw=%.6e, ov_ratio=%.6e, target_ratio=%.4f",
                    overflow, overflow_ratio, model.l_shape_grad_target_ratio,
                )
        else:
            prev_ema = float(model._l_shape_overflow_ema)
            ema = beta * prev_ema + (1.0 - beta) * overflow_ratio
            delta = ema - prev_ema
            model._l_shape_overflow_last = overflow_ratio
            model._l_shape_overflow_ema = ema

            deadband = float(getattr(params, "l_shape_overflow_deadband", 1e-4))
            if abs(delta) > deadband:
                k = float(getattr(params, "l_shape_overflow_update_k", 2.0))
                old_ratio = float(model.l_shape_grad_target_ratio)
                ratio_min = float(getattr(params, "l_shape_grad_target_ratio_min", 0.05))
                ratio_max = float(getattr(params, "l_shape_grad_target_ratio_max", 0.2))
                if ratio_min > ratio_max:
                    ratio_min, ratio_max = ratio_max, ratio_min
                new_ratio = old_ratio * math.exp(k * delta)
                new_ratio = max(ratio_min, min(ratio_max, new_ratio))
                model.l_shape_grad_target_ratio = new_ratio
                if l_shape_log_verbose(params) >= 1:
                    logging.info(
                        "L-shape overflow outer-loop iter=%d: "
                        "ov_raw=%.6e, ov_ratio=%.6e, ov_ema %.6e->%.6e, delta=%.3e, "
                        "target_ratio %.4f->%.4f",
                        iteration, overflow, overflow_ratio, prev_ema, ema, delta,
                        old_ratio, new_ratio,
                    )
            else:
                logging.debug(
                    "L-shape overflow outer-loop iter=%d: "
                    "ov_raw=%.6e, ov_ratio=%.6e, ov_ema %.6e->%.6e, delta=%.3e "
                    "(deadband=%.3e), target_ratio=%.4f",
                    iteration, overflow, overflow_ratio, prev_ema, ema, delta,
                    deadband, model.l_shape_grad_target_ratio,
                )
        self.maybe_auto_disable(model, iteration, outer_update=True)

    def disable_for_recovery(
        self,
        model,
        iteration,
        reason,
        update_threshold=False,
        inflation_round=None,
        ratio_info=None,
    ):
        if not getattr(model, "use_l_shape_routability", False):
            return

        model.use_l_shape_routability = False
        model.enable_l_shape_routability = False
        if hasattr(model, "reset_l_shape_weight_state"):
            model.reset_l_shape_weight_state()
        self.reset_auto_disable_state(model)

        base_threshold = float(
            getattr(
                model,
                "_l_shape_reenable_threshold_base",
                getattr(self.params, "l_shape_overflow_threshold", 0.2),
            )
        )
        current_threshold = float(
            getattr(model, "_l_shape_reenable_threshold", base_threshold)
        )
        reenable_count = int(getattr(model, "_l_shape_reenable_count", 0))
        next_threshold = current_threshold
        if update_threshold and not bool(getattr(self.params, "timing_opt_enabled", False)):
            reenable_count += 1
            next_threshold = current_threshold * 0.7
            model._l_shape_reenable_count = reenable_count
            model._l_shape_reenable_threshold = next_threshold

        model._l_shape_reenable_last_overflow = None
        model._l_shape_reenable_descend_streak = 0
        model._l_shape_ratio_last = None
        model._l_shape_ratio_rise_streak = 0

        if reason == "inflation" and bool(getattr(self.params, "timing_opt_enabled", False)):
            model._l_shape_reenable_threshold = base_threshold
            model._l_shape_reenable_count = 0
            logging.info("L-shape paused for inflation round %d; resumes below %.4f",
                         inflation_round, base_threshold)
        elif reason == "inflation":
            model._l_shape_inflation_guard_disabled = True
            logging.info(
                "L-shape disabled due to inflation (round %d) and permanently disabled "
                "for the remaining placement (next_threshold=%.4f, base_threshold=%.4f, reenable_count=%d)",
                inflation_round,
                next_threshold,
                base_threshold,
                reenable_count,
            )
        elif reason == "demand_supply_ratio_limit":
            model._l_shape_ratio_guard_disabled = True
            logging.info(
                "L-shape disabled due to demand/supply ratio limit at iteration %d: "
                "ratio=%.4f > 0.6500; permanently disabled for the remaining placement "
                "(reenable_threshold=%.4f, reenable_count=%d)",
                iteration,
                float((ratio_info or {}).get("current_ratio", float("nan"))),
                current_threshold,
                reenable_count,
            )
        elif reason == "demand_supply_ratio_rise":
            model._l_shape_ratio_guard_disabled = True
            logging.info(
                "L-shape disabled due to demand/supply ratio surge at iteration %d: "
                "ratio %.4f -> %.4f (rise_streak=%d, rel_increase=%.2f%%, "
                "permanently disabled for the remaining placement; "
                "reenable_threshold=%.4f, reenable_count=%d)",
                iteration,
                float((ratio_info or {}).get("prev_ratio", float("nan"))),
                float((ratio_info or {}).get("current_ratio", float("nan"))),
                int((ratio_info or {}).get("rise_streak", 0)),
                float((ratio_info or {}).get("rel_increase_pct", 0.0)),
                current_threshold,
                reenable_count,
            )

    def maybe_auto_disable(self, model, iteration, outer_update=False):
        params = self.params
        if not getattr(params, "l_shape_auto_disable_flag", False):
            return
        if not getattr(model, "use_l_shape_routability", False):
            return
        if getattr(model, "_l_shape_auto_disabled", False):
            return

        l_shape_op = getattr(model, "l_shape_routability_op", None)
        if l_shape_op is None or not getattr(l_shape_op, "soft_l_assignment", False):
            return
        if bool(getattr(model, "l_shape_fast_mode", False)) and not bool(
            getattr(model, "l_shape_energy_valid", True)
        ):
            if outer_update and l_shape_log_verbose(params) >= 1:
                logging.info(
                    "Skip L-shape auto-disable cost-rebound check because "
                    "l_shape_fast_mode invalidates scalar energy"
                )
            return

        current_cost = getattr(model, "l_shape_last_cost", None)
        current_grad_ratio = getattr(model, "l_shape_last_grad_ratio", None)
        current_ov_ema = getattr(model, "_l_shape_overflow_ema", None)
        if current_cost is None or current_grad_ratio is None or current_ov_ema is None:
            return
        if not math.isfinite(current_cost) or not math.isfinite(current_grad_ratio):
            return

        state = getattr(model, "_l_shape_auto_disable_state", None)
        if not isinstance(state, dict):
            self.reset_auto_disable_state(model)
            state = model._l_shape_auto_disable_state

        if outer_update:
            state["update_count"] += 1

        best_cost = state.get("best_lcost")
        if best_cost is None or current_cost < best_cost:
            state["best_lcost"] = current_cost
            state["best_iteration"] = iteration
            state["rebound_streak"] = 0
        else:
            rebound_ratio = float(getattr(params, "l_shape_auto_disable_lcost_rebound_ratio", 0.05))
            rebound_threshold = best_cost * (1.0 + rebound_ratio)
            state["rebound_streak"] = state["rebound_streak"] + 1 if current_cost > rebound_threshold else 0

        target_ratio = float(model.l_shape_grad_target_ratio)
        overshoot_margin = float(
            getattr(params, "l_shape_auto_disable_grad_overshoot_margin", 0.01)
        )
        state["overshoot_streak"] = (
            state["overshoot_streak"] + 1
            if current_grad_ratio > target_ratio + overshoot_margin
            else 0
        )

        prev_ov_ema = state.get("prev_ov_ema")
        if outer_update:
            if prev_ov_ema is not None and math.isfinite(prev_ov_ema):
                plateau_eps = float(
                    getattr(params, "l_shape_auto_disable_ov_ema_plateau_eps", 1e-3)
                )
                ema_improvement = prev_ov_ema - current_ov_ema
                state["plateau_streak"] = (
                    state["plateau_streak"] + 1 if ema_improvement <= plateau_eps else 0
                )
                state["last_ov_ema_improvement"] = ema_improvement
            state["prev_ov_ema"] = float(current_ov_ema)

        warmup_updates = max(0, int(getattr(params, "l_shape_auto_disable_warmup_updates", 3)))
        patience = max(1, int(getattr(params, "l_shape_auto_disable_patience", 2)))
        if (
            state["update_count"] <= warmup_updates
            or state["overshoot_streak"] < patience
            or state["rebound_streak"] < patience
            or state["plateau_streak"] < patience
        ):
            return

        model.use_l_shape_routability = False
        model.enable_l_shape_routability = False
        model._l_shape_auto_disabled = True
        state["disabled_at"] = iteration
        rebound_pct = 0.0
        if state.get("best_lcost"):
            rebound_pct = (
                (current_cost - state["best_lcost"])
                / max(state["best_lcost"], 1e-12)
                * 100.0
            )
        logging.info(
            "L-shape auto-disabled at iteration %d: LShapeCostRaw %.6e rebounded from best %.6e@iter=%s by %.2f%%, "
            "LGradRatio %.4f > LTargetRatio %.4f + %.4f, ov_ema improvement %.3e.",
            iteration,
            current_cost,
            state.get("best_lcost", float("nan")),
            state.get("best_iteration"),
            rebound_pct,
            current_grad_ratio,
            target_ratio,
            overshoot_margin,
            state.get("last_ov_ema_improvement", float("nan")),
        )

    def maybe_disable_by_ratio(self, model, iteration):
        if not getattr(model, "use_l_shape_routability", False):
            return
        if int(getattr(model, "_l_shape_reenable_count", 0)) <= 0:
            return

        soft_summary = getattr(model, "soft_l_last_summary", None) or {}
        current_ratio = soft_summary.get("current_demand_supply_ratio")
        if current_ratio is None:
            current_ratio = soft_summary.get("target_demand_supply_ratio")
        if current_ratio is None or not math.isfinite(float(current_ratio)):
            return
        current_ratio = float(current_ratio)

        absolute_limit = 0.65
        relative_rise_limit = 0.03
        prev_ratio = getattr(model, "_l_shape_ratio_last", None)
        rise_streak = int(getattr(model, "_l_shape_ratio_rise_streak", 0))
        rel_increase = None
        if prev_ratio is not None and math.isfinite(prev_ratio) and prev_ratio > 1e-9:
            rel_increase = (current_ratio - prev_ratio) / prev_ratio
            rise_streak = rise_streak + 1 if current_ratio > prev_ratio and rel_increase > relative_rise_limit else 0
        else:
            rise_streak = 0

        model._l_shape_ratio_last = current_ratio
        model._l_shape_ratio_rise_streak = rise_streak
        if current_ratio > absolute_limit:
            self.disable_for_recovery(
                model,
                iteration,
                reason="demand_supply_ratio_limit",
                ratio_info={"current_ratio": current_ratio},
            )
        elif rise_streak >= 2 and rel_increase is not None:
            self.disable_for_recovery(
                model,
                iteration,
                reason="demand_supply_ratio_rise",
                ratio_info={
                    "prev_ratio": prev_ratio,
                    "current_ratio": current_ratio,
                    "rise_streak": rise_streak,
                    "rel_increase_pct": rel_increase * 100.0,
                },
            )

    def maybe_reenable(self, model, current_overflow):
        if model.enable_l_shape_routability:
            return
        if getattr(model, "_l_shape_auto_disabled", False):
            return
        if getattr(model, "_l_shape_ratio_guard_disabled", False):
            return
        if getattr(model, "_l_shape_inflation_guard_disabled", False):
            return

        threshold = float(
            getattr(
                model,
                "_l_shape_reenable_threshold",
                getattr(self.params, "l_shape_overflow_threshold", 0.2),
            )
        )
        reenable_count = int(getattr(model, "_l_shape_reenable_count", 0))
        if reenable_count <= 0:
            if current_overflow < threshold:
                model.enable_l_shape_routability = True
            return

        last_overflow = getattr(model, "_l_shape_reenable_last_overflow", None)
        descend_streak = int(getattr(model, "_l_shape_reenable_descend_streak", 0))
        if (
            last_overflow is not None
            and math.isfinite(last_overflow)
            and current_overflow < last_overflow - 1e-6
        ):
            descend_streak += 1
        else:
            descend_streak = 0
        model._l_shape_reenable_last_overflow = current_overflow
        model._l_shape_reenable_descend_streak = descend_streak
        if descend_streak >= 5 and current_overflow < threshold:
            model.enable_l_shape_routability = True


__all__ = ["LShapePolicy"]
