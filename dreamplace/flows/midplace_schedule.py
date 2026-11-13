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
    inflated_movable_sizes: dict[str, tuple[float, float]] = field(default_factory=dict)
    oscillation_snapshot: dict | None = None
    committed_buffer_count: int = 0
    committed_buffer_records: list = field(default_factory=list)
    alternating_completed_rounds: int = 0
    alternating_placement_step_count: int = 0
    alternating_trace: list = field(default_factory=list)

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
        alternating = getattr(placer, "alternating_discrete", None)
        owner = getattr(placer, "inflation_s5b1", None)
        if owner is None and alternating is not None:
            owner = getattr(alternating, "owner", None)
        if owner is not None:
            self.timing = copy.deepcopy(owner.schedule)
            self.timing_round = owner.round_index
            self.timing_summaries = copy.deepcopy(owner.summaries)
            osc = getattr(owner, "oscillation", None)
            if osc is not None and callable(getattr(osc, "snapshot", None)):
                self.oscillation_snapshot = copy.deepcopy(osc.snapshot())
            self.committed_buffer_count = int(
                getattr(owner, "committed_buffer_count", 0) or 0
            )
            self.committed_buffer_records = copy.deepcopy(
                list(getattr(owner, "committed_buffer_records", ()) or ())
            )
        if alternating is not None:
            self.alternating_completed_rounds = int(alternating.completed_rounds)
            self.alternating_placement_step_count = int(
                alternating.placement_step_count
            )
            self.alternating_trace = copy.deepcopy(list(alternating.trace))
        dc = getattr(placer, "data_collections", None)
        placedb = getattr(placer, "placedb", None)
        if (
            dc is not None
            and placedb is not None
            and getattr(dc, "node_size_x", None) is not None
            and getattr(dc, "original_node_size_x", None) is not None
        ):
            count = int(placedb.num_movable_nodes)
            node_names = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in placedb.node_names[:count]
            ]
            wx = dc.node_size_x[:count].detach().cpu().tolist()
            wy = dc.node_size_y[:count].detach().cpu().tolist()
            owx = dc.original_node_size_x[:count].detach().cpu().tolist()
            owy = dc.original_node_size_y[:count].detach().cpu().tolist()
            self.inflated_movable_sizes = {
                name: (float(w), float(h))
                for name, w, h, ow, oh in zip(node_names, wx, wy, owx, owy, strict=True)
                if w > ow + 1e-12 or h > oh + 1e-12
            }
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
                "_l_shape_paused_for_inflation",
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
        if self.timing is not None and owner is not None:
            target = getattr(owner, "owner", owner)
            target.schedule = copy.deepcopy(self.timing)
            target.milestones = target.schedule.milestones
            target.pending_milestones = target.schedule.pending_milestones
            target.window_count = target.schedule.window_count
            target.round_index = self.timing_round
            target.summaries = copy.deepcopy(self.timing_summaries)
            if self.oscillation_snapshot is not None and hasattr(target, "oscillation"):
                target.oscillation.restore(copy.deepcopy(self.oscillation_snapshot))
            target.committed_buffer_count = int(self.committed_buffer_count)
            target.committed_buffer_records = copy.deepcopy(
                self.committed_buffer_records
            )
            if callable(getattr(target, "_write_summaries", None)) and target.summaries:
                target._write_summaries()
            if target.lane is not None:
                raise RuntimeError("fresh S5B1 owner unexpectedly retained virtual topology")
        elif self.timing is not None and owner is None:
            # AlternatingDiscreteController is constructed later in NonLinearPlace.__call__;
            # bind_alternating_controller restores it via restore_alternating().
            if not self.alternating_trace and self.alternating_completed_rounds == 0:
                raise RuntimeError("midplace lost its S5B1 owner during runtime reconstruction")
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
            "inflated_movable_nodes_captured": len(self.inflated_movable_sizes),
        }

    def restore_alternating(self, alternating):
        if alternating is None:
            return
        owner = alternating.owner
        if self.timing is not None:
            owner.schedule = copy.deepcopy(self.timing)
            owner.milestones = owner.schedule.milestones
            owner.pending_milestones = owner.schedule.pending_milestones
            owner.window_count = owner.schedule.window_count
            owner.round_index = self.timing_round
            owner.summaries = copy.deepcopy(self.timing_summaries)
        if self.oscillation_snapshot is not None and hasattr(owner, "oscillation"):
            owner.oscillation.restore(copy.deepcopy(self.oscillation_snapshot))
        owner.committed_buffer_count = int(self.committed_buffer_count)
        owner.committed_buffer_records = copy.deepcopy(self.committed_buffer_records)
        alternating.completed_rounds = int(self.alternating_completed_rounds)
        alternating.placement_step_count = int(self.alternating_placement_step_count)
        alternating.trace = copy.deepcopy(self.alternating_trace)
        if alternating.trace:
            alternating.last_result = alternating.trace[-1]

    def restore_inflated_geometry(self, placer, placedb, cooptimization=None):
        if not self.inflated_movable_sizes:
            return {"restored_inflated_nodes": 0}
        import torch

        from dreamplace.ops.routability.cooptimization_area import (
            capture_area,
            coordinate_area,
        )

        dc = getattr(placer, "data_collections", None)
        if dc is None or placedb is None:
            return {"restored_inflated_nodes": 0}
        if not hasattr(dc, "original_node_size_x") or dc.original_node_size_x is None:
            dc.original_node_size_x = dc.node_size_x.detach().clone()
        if not hasattr(dc, "original_node_size_y") or dc.original_node_size_y is None:
            dc.original_node_size_y = dc.node_size_y.detach().clone()
        if not hasattr(dc, "original_pin_offset_x") or dc.original_pin_offset_x is None:
            dc.original_pin_offset_x = dc.pin_offset_x.detach().clone()
        if not hasattr(dc, "original_pin_offset_y") or dc.original_pin_offset_y is None:
            dc.original_pin_offset_y = dc.pin_offset_y.detach().clone()

        count = int(placedb.num_movable_nodes)
        inflated_sizes = self.inflated_movable_sizes
        matched_ids = []
        target_wx = []
        target_wy = []
        for node_id, raw_name in enumerate(placedb.node_names[:count]):
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            inflated = inflated_sizes.get(name)
            if inflated is not None:
                matched_ids.append(node_id)
                target_wx.append(float(inflated[0]))
                target_wy.append(float(inflated[1]))
        if not matched_ids:
            return {"restored_inflated_nodes": 0}

        pos = placer.pos[0]
        num_nodes = int(placedb.num_nodes)
        device = dc.node_size_x.device
        capacity = getattr(cooptimization, "capacity", None)

        with torch.no_grad():
            idx = torch.tensor(matched_ids, device=device, dtype=torch.long)
            req_w = torch.tensor(target_wx, device=device, dtype=torch.float64)
            req_h = torch.tensor(target_wy, device=device, dtype=torch.float64)
            ow = dc.original_node_size_x[idx].double()
            oh = dc.original_node_size_y[idx].double()
            nw = torch.maximum(ow, req_w)
            nh = torch.maximum(oh, req_h)
            active = (nw > ow + 1e-12) | (nh > oh + 1e-12)
            restored = int(active.sum().item())
            if restored == 0:
                return {"restored_inflated_nodes": 0}

            before = capture_area(dc, placedb, 0.0, capacity=capacity)
            active_idx = idx[active]
            nw_active = nw[active]
            nh_active = nh[active]
            dx = (nw_active - ow[active]) * 0.5
            dy = (nh_active - oh[active]) * 0.5

            dc.node_size_x[active_idx] = nw_active.to(dtype=dc.node_size_x.dtype)
            dc.node_size_y[active_idx] = nh_active.to(dtype=dc.node_size_y.dtype)
            pos.data[active_idx] -= dx.to(dtype=pos.dtype)
            pos.data[num_nodes + active_idx] -= dy.to(dtype=pos.dtype)

            raw_starts = getattr(dc, "flat_node2pin_start_map", None)
            if raw_starts is None:
                raw_starts = placedb.flat_node2pin_start_map
            starts_t = torch.as_tensor(raw_starts, device=device, dtype=torch.long)
            pin_starts = starts_t[active_idx]
            pin_counts = (starts_t[active_idx + 1] - pin_starts).clamp(min=0)
            total_pins = int(pin_counts.sum().item())
            if total_pins > 0:
                node_offsets = torch.repeat_interleave(
                    torch.arange(active_idx.numel(), device=device, dtype=torch.long),
                    pin_counts,
                )
                group_starts = torch.cumsum(pin_counts, 0) - pin_counts
                within = (
                    torch.arange(total_pins, device=device, dtype=torch.long)
                    - group_starts[node_offsets]
                )
                csr_idx = pin_starts[node_offsets] + within
                pins = dc.flat_node2pin_map[csr_idx].long()
                dc.pin_offset_x[pins] = dc.original_pin_offset_x[pins] + dx[
                    node_offsets
                ].to(dtype=dc.pin_offset_x.dtype)
                dc.pin_offset_y[pins] = dc.original_pin_offset_y[pins] + dy[
                    node_offsets
                ].to(dtype=dc.pin_offset_y.dtype)

            coordinate_area(before, dc, placedb, pos, 0.0)
            placedb.cooptimization_geometry = dc
        return {"restored_inflated_nodes": restored}
