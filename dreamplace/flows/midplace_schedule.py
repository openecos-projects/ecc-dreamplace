"""Preserve consumed window counters, while rebuilding all topology-owned caches."""

import copy
from dataclasses import dataclass, field

from dreamplace.flows.timing_opt_schedule import TimingOptWindowSchedule


@dataclass
class MidplaceSchedule:
    timing: TimingOptWindowSchedule | None = None
    timing_round: int = 0
    timing_summaries: list = field(default_factory=list)
    gp_steps: int = 0
    last_inflation_step: int | None = None
    inflation_rounds: dict[int, int] = field(default_factory=dict)
    use_restart_density_weight: bool = False
    gamma_before_window: float | None = None
    trigger_lgamma_step: int | None = None
    trigger_overflow: object = None
    trigger_goverflow: object = None
    llambda_flat_iteration: int = 0
    last_llambda_metric: object = None
    diff_tdp_gate_summary: dict | None = None
    diff_tdp_step_status: dict | None = None
    legacy_net_weight_gate_summary: dict | None = None
    legacy_net_weight_gate_open: bool = False
    legacy_net_weight_update_count: int = 0
    use_timing_obj: bool = False
    timing_coeff_state: tuple | None = None
    timing_grad_balance_state: dict | None = None
    enable_l_shape_routability: bool = False
    use_l_shape_routability: bool = False
    l_shape_state: dict = field(default_factory=dict)

    def capture(
        self,
        placer,
        model,
        stage,
        *,
        consume_inflation_window=False,
        iteration=None,
        window_summary=None,
        cursor=None,
        metric=None,
    ):
        owner = getattr(placer, "inflation_s5b1", None)
        if owner is not None:
            self.timing = copy.deepcopy(owner.schedule)
            self.timing_round = owner.round_index
            self.timing_summaries = copy.deepcopy(owner.summaries)
        controller = placer._routability_controller
        self.gp_steps = controller._gp_steps
        self.last_inflation_step = controller._last_inflation_step
        state = getattr(model, "inflation_state", None)
        if state is not None:
            self.inflation_rounds[stage] = int(state.num_area_adjust)
        self.use_restart_density_weight = True
        gamma = getattr(model, "gamma", None)
        if gamma is not None:
            self.gamma_before_window = (
                float(gamma.detach().cpu().item())
                if hasattr(gamma, "detach")
                else float(gamma)
            )
        if cursor is not None:
            self.trigger_lgamma_step = int(cursor.gamma)
        eval_metric = (
            metric if metric is not None else getattr(placer, "_last_eval_metric", None)
        )
        if eval_metric is not None:
            ov = getattr(eval_metric, "overflow", None)
            if ov is not None:
                self.trigger_overflow = (
                    ov.detach().clone() if hasattr(ov, "detach") else copy.deepcopy(ov)
                )
            gov = getattr(eval_metric, "goverflow", None)
            if gov is not None:
                self.trigger_goverflow = (
                    gov.detach().clone() if hasattr(gov, "detach") else copy.deepcopy(gov)
                )
            self.last_llambda_metric = copy.deepcopy(eval_metric)
        elif getattr(model, "overflow", None) is not None:
            ov = model.overflow
            self.trigger_overflow = (
                ov.detach().clone() if hasattr(ov, "detach") else copy.deepcopy(ov)
            )
        self.llambda_flat_iteration = int(
            getattr(placer, "_llambda_flat_iteration", 0) or 0
        )
        self.diff_tdp_gate_summary = copy.deepcopy(
            getattr(placer, "_diff_tdp_gate_summary", None)
        )
        self.diff_tdp_step_status = copy.deepcopy(
            getattr(placer, "_diff_tdp_step_status", None)
        )
        self.legacy_net_weight_gate_summary = copy.deepcopy(
            getattr(placer, "_legacy_net_weight_gate_summary", None)
        )
        self.legacy_net_weight_gate_open = bool(
            getattr(placer, "_legacy_net_weight_gate_open", False)
        )
        self.legacy_net_weight_update_count = int(
            getattr(placer, "_legacy_net_weight_update_count", 0)
        )
        self.use_timing_obj = bool(getattr(model, "use_timing_obj", False))
        base = getattr(model, "_timing_coeff_growth_base", None)
        if base is not None:
            self.timing_coeff_state = (
                tuple(base),
                float(getattr(model, "_timing_coeff_growth_multiplier", 1.0)),
                float(getattr(model, "timing_wns_coeff", base[0])),
                float(getattr(model, "timing_tns_coeff", base[1])),
            )
        balance = getattr(placer, "_timing_grad_balance_state", None)
        if balance is None:
            balance = getattr(model, "timing_grad_balance_summary", None)
        self.timing_grad_balance_state = copy.deepcopy(balance)
        self.enable_l_shape_routability = bool(
            getattr(model, "enable_l_shape_routability", False)
        )
        self.use_l_shape_routability = bool(
            getattr(model, "use_l_shape_routability", False)
        )
        self.l_shape_state = {
            key: copy.deepcopy(getattr(model, key))
            for key in (
                "l_shape_grad_target_ratio",
                "_l_shape_overflow_ema",
                "_l_shape_overflow_last",
                "_l_shape_auto_disabled",
                "_l_shape_auto_disable_state",
                "_l_shape_reenable_threshold_base",
                "_l_shape_reenable_threshold",
                "_l_shape_reenable_count",
                "_l_shape_ratio_guard_disabled",
                "_l_shape_inflation_guard_disabled",
            )
            if hasattr(model, key)
        }
        if consume_inflation_window:
            self.record_inflation_window(
                stage=stage,
                iteration=iteration,
                window_summary=window_summary,
            )

    def record_inflation_window(self, *, stage, iteration=None, window_summary=None):
        if self.timing is not None:
            event = self.timing.inflation_event(
                iteration=int(0 if iteration is None else iteration),
                stage=int(stage),
            )
            if window_summary is not None:
                entry = copy.deepcopy(window_summary)
                if event is not None:
                    entry["trigger"] = event
                entry["window"] = self.timing.window_count
                self.timing_summaries.append(entry)
        self.last_inflation_step = self.gp_steps
        self.inflation_rounds[stage] = int(self.inflation_rounds.get(stage, 0)) + 1
        self.use_restart_density_weight = True

    def restore_l_shape_state(self, model, params):
        for key, value in self.l_shape_state.items():
            setattr(model, key, copy.deepcopy(value))
        if self.enable_l_shape_routability and bool(
            getattr(params, "l_shape_keep_during_inflation", False)
        ):
            model.enable_l_shape_routability = True

    def restore_model_state(self, placer, model, params):
        if (
            self.gamma_before_window is not None
            and hasattr(model, "gamma")
            and model.gamma is not None
        ):
            if hasattr(model.gamma, "data"):
                model.gamma.data.fill_(float(self.gamma_before_window))
            else:
                model.gamma = float(self.gamma_before_window)
        if self.diff_tdp_gate_summary is not None:
            placer._diff_tdp_gate_summary = copy.deepcopy(self.diff_tdp_gate_summary)
        if self.diff_tdp_step_status is not None:
            placer._diff_tdp_step_status = copy.deepcopy(self.diff_tdp_step_status)
        if self.legacy_net_weight_gate_summary is not None:
            placer._legacy_net_weight_gate_summary = copy.deepcopy(
                self.legacy_net_weight_gate_summary
            )
            placer._legacy_net_weight_gate_open = bool(self.legacy_net_weight_gate_open)
            placer._legacy_net_weight_update_count = int(
                self.legacy_net_weight_update_count
            )
        if self.use_timing_obj or bool(
            (self.diff_tdp_step_status or {}).get("timing_objective_active", False)
        ):
            model.use_timing_obj = True
        if self.timing_coeff_state is not None:
            base, multiplier, wns_coeff, tns_coeff = self.timing_coeff_state
            model._timing_coeff_growth_base = base
            model._timing_coeff_growth_multiplier = multiplier
            model.timing_wns_coeff = wns_coeff
            model.timing_tns_coeff = tns_coeff
        if (
            self.timing_grad_balance_state is not None
            and not bool(getattr(params, "timing_grad_balance_reset_on_restart", True))
        ):
            placer._timing_grad_balance_state = copy.deepcopy(
                self.timing_grad_balance_state
            )
            if hasattr(model, "apply_timing_grad_balance_state"):
                model.apply_timing_grad_balance_state(self.timing_grad_balance_state)
        self.restore_l_shape_state(model, params)

    def restore(self, owner, controller):
        if self.timing is not None:
            if owner is None:
                raise RuntimeError("midplace lost its S5B1 owner during runtime reconstruction")
            owner.schedule = copy.deepcopy(self.timing)
            owner.milestones = owner.schedule.milestones
            owner.pending_milestones = owner.schedule.pending_milestones
            owner.window_count = owner.schedule.window_count
            owner.round_index = self.timing_round
            owner.summaries = copy.deepcopy(self.timing_summaries)
            if callable(getattr(owner, "_write_summaries", None)) and owner.summaries:
                owner._write_summaries()
            if owner.lane is not None:
                raise RuntimeError("fresh S5B1 owner unexpectedly retained virtual topology")
        controller._gp_steps = self.gp_steps
        controller._last_inflation_step = self.last_inflation_step
        return {
            "virtual_topology": "discarded_and_rebuilt",
            "routability_cache": "fresh",
            "timing_windows_completed": 0 if self.timing is None else self.timing.window_count,
            "pending_timing_milestones": []
            if self.timing is None
            else list(self.timing.pending_milestones),
            "inflation_rounds_completed": dict(self.inflation_rounds),
            "gp_steps_completed": self.gp_steps,
        }
