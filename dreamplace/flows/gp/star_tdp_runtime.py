"""AutoDMP R0_reference (A_bilinear_bplus) Star-Pin-Pair TDP runtime for ecc-dreamplace.

Encapsulates data-net and leaf-clock (LCB/ICG -> FF) StarPinPair construction,
FLUTE teacher RC calibration, homothety slope calibration, input-gradient EMA
accumulation, B+ load clamping, kappa raw-L1 dose calibration, 2N MM row-sum
Hessian curvature, and Nesterov momentum transport without touching routability,
sizing (InflationS5B1), or buffering (segment virtual + MidplaceRepair).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from dreamplace.ops.buffer_insertion.net_eligibility import clock_net_ids
from dreamplace.ops.net_subgraph_timing.segment_count_live_geometry import (
    build_live_edge_geometry,
)
from dreamplace.ops.star_pin_pair import (
    ClockRCCalibrationState,
    InputAccumulator,
    SlopeCalibration,
    StarPinPair,
    StarRCCalibrationState,
    apply_satisfied_slack_release,
    calibration_cap_digest,
    calibration_policy_digest,
    clamp_load_length_weight,
    clock_rowsum_hessian,
    compute_delay_sigmoid_retention_beta,
    compute_slack_retention_beta,
    estimate_slack_drift_tau,
    launch_capture_clock_pair_adjoint,
    next_lower_overflow_milestone,
    pair_delay_adjoint_from_pin_grads,
    rowsum_hessian,
    summarize_drift_tau_history,
    timing_transport_factor,
    transport_momentum,
)
from dreamplace.ops.star_pin_pair.rc_calibration import (
    pair_wirelength_by_net,
    tree_wirelength_by_net,
)
from dreamplace.ops.star_pin_pair.slope_calibration import teacher_rc_like
from dreamplace.ops.steiner_topo import steiner_topo
from dreamplace.ops.timing_propagation.at_merge_temperature import AtMergeTemperature
from dreamplace.ops.timing_propagation.lcb_ff_timing import (
    apply_lcb_ff_to_compact_clk_pins,
    build_clock_star_steiner_net2pin,
    build_lcb_ff_clock_star_domain,
    compute_lcb_ff_clock_state,
    pin_is_input,
    pin_is_output,
)


def _scalar_overflow(overflow) -> Optional[float]:
    if overflow is None:
        return None
    if torch.is_tensor(overflow):
        if overflow.numel() == 0:
            return None
        val = float(overflow.detach().reshape(-1)[-1].item())
    else:
        val = float(overflow)
    return val if math.isfinite(val) else None


def _movable_l1(grad: torch.Tensor, movable_count: int) -> float:
    g2 = grad.detach().double().reshape(2, -1)[:, : int(movable_count)]
    return float(g2.abs().sum().item())


def _filter_timing_arcs_by_net_degree(
    net2driver,
    net_arcs,
    net_arcs_start,
    data_collections,
    ignore_net_degree,
):
    """Filter high-fanout timing nets (>= ignore_net_degree) to match HPWL/SteinerTopo."""
    if (
        ignore_net_degree is None
        or int(ignore_net_degree) <= 0
        or net2driver is None
        or net_arcs is None
        or net_arcs_start is None
    ):
        return net_arcs, net_arcs_start
    flat_start = getattr(data_collections, "flat_net2pin_start_map", None)
    pin2net = getattr(data_collections, "pin2net_map", None)
    if flat_start is None or pin2net is None:
        return net_arcs, net_arcs_start

    starts = torch.as_tensor(net_arcs_start).reshape(-1)
    arcs = torch.as_tensor(net_arcs)
    if starts.numel() < 2 or arcs.ndim != 2 or arcs.shape[0] == 0:
        return net_arcs, net_arcs_start

    f_start = torch.as_tensor(flat_start).reshape(-1).detach().cpu().long()
    if f_start.numel() < 2:
        return net_arcs, net_arcs_start
    native_deg = f_start[1:] - f_start[:-1]
    num_native_nets = int(native_deg.numel())
    if num_native_nets <= 0:
        return net_arcs, net_arcs_start

    p2n = torch.as_tensor(pin2net).reshape(-1).detach().cpu().long()
    drv = torch.as_tensor(net2driver).reshape(-1).detach().cpu().long()
    starts_cpu = starts.detach().cpu().long()
    t_deg = starts_cpu[1:] - starts_cpu[:-1]

    valid_drv = (t_deg > 0) & (drv >= 0) & (drv < int(p2n.numel()))
    mapped_net = torch.zeros_like(drv)
    if bool(valid_drv.any().item()):
        mapped_net[valid_drv] = p2n[drv[valid_drv]]
    valid_net = valid_drv & (mapped_net >= 0) & (mapped_net < num_native_nets)
    ignored_tnet = torch.zeros_like(valid_drv)
    if bool(valid_net.any().item()):
        ignored_tnet[valid_net] = native_deg[mapped_net[valid_net]] >= int(
            ignore_net_degree
        )
    if not bool(ignored_tnet.any().item()):
        return net_arcs, net_arcs_start

    net_ids_per_arc = torch.repeat_interleave(
        torch.arange(int(t_deg.numel()), dtype=torch.long), t_deg
    )
    keep_arc = ~ignored_tnet[net_ids_per_arc]
    new_t_deg = torch.where(ignored_tnet, torch.zeros_like(t_deg), t_deg)
    new_starts = torch.zeros_like(starts_cpu)
    new_starts[1:] = torch.cumsum(new_t_deg, dim=0)
    filtered_arcs = arcs[keep_arc.to(device=arcs.device)].contiguous()
    filtered_starts = new_starts.to(
        device=starts.device, dtype=starts.dtype
    ).contiguous()
    logging.info(
        "StarTdp filtered %d timing nets (%d -> %d pairs) with degree >= %d",
        int(ignored_tnet.sum().item()),
        int(arcs.shape[0]),
        int(filtered_arcs.shape[0]),
        int(ignore_net_degree),
    )
    return filtered_arcs, filtered_starts


class StarTdpRuntime:
    """Manages Star-Pin-Pair TDP snapshots, objective, curvature, and Nesterov transport."""

    DEFAULT_KAPPA = 0.04752809509417255

    def __init__(self, *, model, params, placedb, data_collections, op_collections):
        self.model = model
        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections

        timing_op = getattr(op_collections, "timing_propagation_op", None)
        elmore_op = getattr(op_collections, "elmore_delay_op", None)
        if timing_op is None or elmore_op is None:
            raise RuntimeError(
                "timing_placement_carrier=star_pin_pair requires timing_propagation_op and elmore_delay_op (with_sta=1)"
            )

        self.scale_factor = float(
            getattr(elmore_op, "scale_factor", getattr(params, "scale_factor", 1.0))
            or 1.0
        )
        self.dbu = float(
            getattr(elmore_op, "dbu", getattr(placedb, "dbu", 1.0)) or 1.0
        )
        self.coordinate_to_um = self.scale_factor * self.dbu
        self.r_unit = float(elmore_op.r_unit)
        self.c_unit = float(elmore_op.c_unit)
        self.num_pins = int(placedb.num_pins)

        self.distance_model = (
            str(getattr(params, "star_distance_model", "lse") or "lse").strip().lower()
        )
        if self.distance_model not in ("l1", "wa", "lse"):
            raise ValueError(
                f"star_distance_model must be l1, wa, or lse, got {self.distance_model!r}"
            )
        self.loss_mode = (
            str(getattr(params, "star_timing_loss_mode", "tns_only") or "tns_only")
            .strip()
            .lower()
        )
        if self.loss_mode not in ("tns_only", "combined"):
            raise ValueError(
                f"star_timing_loss_mode must be tns_only or combined, got {self.loss_mode!r}"
            )
        self.smooth_extremum = bool(int(getattr(params, "star_smooth_extremum", 1)))
        self.extremum_tau_ps = float(
            getattr(params, "star_extremum_tau_ps", 1.0) or 1.0
        )
        if not math.isfinite(self.extremum_tau_ps) or self.extremum_tau_ps <= 0.0:
            raise ValueError("star_extremum_tau_ps must be finite and positive")
        self.at_merge_vjp_tau = (
            str(getattr(params, "star_at_merge_vjp_tau", "drift") or "drift")
            .strip()
            .lower()
        )
        if self.at_merge_vjp_tau not in ("drift", "fixed"):
            raise ValueError(
                f"star_at_merge_vjp_tau must be drift or fixed, got {self.at_merge_vjp_tau!r}"
            )
        self.load_sign = (
            str(getattr(params, "star_load_sign", "clamp") or "clamp").strip().lower()
        )
        if self.load_sign not in ("clamp", "signed", "zero"):
            raise ValueError(
                f"star_load_sign must be clamp, signed, or zero, got {self.load_sign!r}"
            )
        self.load_geometry = (
            str(getattr(params, "star_load_geometry", "star") or "star")
            .strip()
            .lower()
        )
        if self.load_geometry not in ("star", "wa_net"):
            raise ValueError(
                f"star_load_geometry must be star or wa_net, got {self.load_geometry!r}"
            )

        self.requested_kappa = float(
            getattr(params, "star_kappa", self.DEFAULT_KAPPA) or self.DEFAULT_KAPPA
        )
        raw_max_kappa = getattr(params, "star_max_kappa", 0.0)
        self.max_kappa = (
            0.0 if raw_max_kappa is None else float(raw_max_kappa)
        )
        raw_post_restart_max = getattr(params, "star_post_restart_max_kappa", 0.0)
        self.post_restart_max_kappa = (
            0.0 if raw_post_restart_max is None else max(0.0, float(raw_post_restart_max))
        )
        fixed_lambda_raw = getattr(params, "star_fixed_lambda", None)
        self.fixed_lambda = (
            float(fixed_lambda_raw)
            if fixed_lambda_raw is not None and float(fixed_lambda_raw) > 0.0
            else None
        )
        if self.fixed_lambda is None and (
            not math.isfinite(self.requested_kappa) or self.requested_kappa <= 0.0
        ):
            raise ValueError("star_kappa must be finite and positive")
        if not math.isfinite(self.max_kappa):
            raise ValueError("star_max_kappa must be finite")
        if self.max_kappa > 0.0 and self.max_kappa < self.requested_kappa:
            self.max_kappa = float(self.requested_kappa)
        if (
            self.post_restart_max_kappa > 0.0
            and self.post_restart_max_kappa < self.requested_kappa
        ):
            self.post_restart_max_kappa = float(self.requested_kappa)

        self.restart_history_mode = (
            str(getattr(params, "star_restart_history_mode", "reset") or "reset")
            .strip()
            .lower()
        )
        if self.restart_history_mode not in (
            "reset",
            "slack_beta",
            "delay_sigmoid",
            "selective_sigmoid",
        ):
            raise ValueError(
                "star_restart_history_mode must be reset, slack_beta, delay_sigmoid, "
                f"or selective_sigmoid, got {self.restart_history_mode!r}"
            )
        self.recal_m_eff_scale_fix = bool(
            int(getattr(params, "star_recal_m_eff_scale_fix", 1))
        )
        self.restart_beta_tau_mode = (
            str(getattr(params, "star_restart_beta_tau_mode", "drift") or "drift")
            .strip()
            .lower()
        )
        if self.restart_beta_tau_mode not in ("drift", "fixed"):
            raise ValueError(
                f"star_restart_beta_tau_mode must be drift or fixed, got {self.restart_beta_tau_mode!r}"
            )
        raw_beta_scale = getattr(params, "star_restart_beta_tau_scale", 1.0)
        self.restart_beta_tau_scale = (
            1.0 if raw_beta_scale is None else max(0.0, float(raw_beta_scale))
        )
        raw_beta_min = getattr(params, "star_restart_beta_min_tau_ps", 1.0)
        self.restart_beta_min_tau_ps = (
            1.0 if raw_beta_min is None else max(1e-6, float(raw_beta_min))
        )
        raw_beta_fixed = getattr(params, "star_restart_beta_fixed_tau_ps", 40.0)
        self.restart_beta_fixed_tau_ps = (
            40.0 if raw_beta_fixed is None else max(1e-6, float(raw_beta_fixed))
        )
        self.endpoint_slack_tau_mode = (
            str(getattr(params, "star_endpoint_slack_tau_mode", "fixed") or "fixed")
            .strip()
            .lower()
        )
        if self.endpoint_slack_tau_mode not in ("fixed", "drift", "post_restart_drift"):
            raise ValueError(
                "star_endpoint_slack_tau_mode must be fixed, drift, or post_restart_drift, "
                f"got {self.endpoint_slack_tau_mode!r}"
            )
        raw_ep_scale = getattr(params, "star_endpoint_slack_tau_scale", 1.0)
        self.endpoint_slack_tau_scale = (
            1.0 if raw_ep_scale is None else max(0.0, float(raw_ep_scale))
        )

        self.rc_calibration_enabled = bool(
            getattr(params, "star_rc_calibration_enabled", 1)
        )
        self.rc_resample_every_refresh = bool(
            getattr(params, "star_rc_resample_every_refresh", 0)
        )
        self.rc_overflow_step = float(
            getattr(params, "star_rc_calibration_overflow_step", 0.05) or 0.05
        )
        self.slope_calibration_enabled = bool(
            getattr(params, "star_slope_calibration_enabled", 1)
        )
        self.lcb_ff_clock_enabled = bool(
            getattr(params, "star_lcb_ff_clock_enabled", 1)
        )
        self.clock_weight_scale = float(
            getattr(params, "star_clock_weight_scale", 1.0)
        )
        self.curvature_precondition_enabled = bool(
            getattr(params, "star_curvature_precondition_enabled", 1)
        )
        self.elmore_rowsum_axes = int(
            getattr(params, "star_elmore_rowsum_axes", 2) or 2
        )
        self.full_linear_slope = bool(
            getattr(params, "star_rowsum_full_linear_slope", 1)
        )
        self.momentum_transport_envelope = float(
            getattr(params, "star_momentum_transport_envelope", 0.0) or 0.0
        )
        raw_transport_timing = getattr(params, "star_momentum_transport_timing", None)
        if self.momentum_transport_envelope >= 1.0:
            self.momentum_transport_timing = False
        elif raw_transport_timing is None:
            self.momentum_transport_timing = True
        else:
            self.momentum_transport_timing = bool(int(raw_transport_timing))
        self.refresh_nesterov_driver = bool(
            getattr(params, "star_refresh_nesterov_state", 1)
        )

        # Build data StarPinPair operator.
        net2driver = getattr(
            data_collections,
            "net2driver_pin_map",
            getattr(timing_op, "net2driver_pin_map", None),
        )
        net_arcs = getattr(
            data_collections,
            "net_flat_arcs",
            getattr(timing_op, "net_flat_arcs", None),
        )
        net_arcs_start = getattr(
            data_collections,
            "net_flat_arcs_start",
            getattr(timing_op, "net_flat_arcs_start", None),
        )
        net_arcs, net_arcs_start = _filter_timing_arcs_by_net_degree(
            net2driver,
            net_arcs,
            net_arcs_start,
            data_collections,
            getattr(params, "ignore_net_degree", 100),
        )
        self.star = StarPinPair(
            net2driver_pin_map=net2driver,
            timing_net_flat_arcs=net_arcs,
            timing_net_flat_arcs_start=net_arcs_start,
            num_pins=self.num_pins,
            r_unit=self.r_unit,
            c_unit=self.c_unit,
            scale_factor=self.scale_factor,
            dbu=self.dbu,
            identity_namespace="data",
            load_geometry=self.load_geometry,
        )
        op_collections.star_pin_pair_op = self.star

        # Build optional movable leaf-clock (LCB/ICG -> FF) Star domain.
        self.clock_domain = None
        self.clock_star: Optional[StarPinPair] = None
        self.clock_steiner_topo = None
        self.clock_teacher_rc = None
        clock_pins = getattr(timing_op, "clock_pins", None)
        if (
            self.lcb_ff_clock_enabled
            and clock_pins is not None
            and int(torch.as_tensor(clock_pins).numel()) > 0
        ):
            self.clock_domain = build_lcb_ff_clock_star_domain(
                clock_pins_native=clock_pins,
                pin2net_map=data_collections.pin2net_map,
                pin2node_map=data_collections.pin2node_map,
                pin_direct=self._resolve_pin_direct(timing_op, net2driver),
                flat_net2pin_map=data_collections.flat_net2pin_map,
                flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
                placedb=placedb,
                params=params,
            )
            op_collections.clock_star_domain = self.clock_domain
            if self.clock_domain.plan.leaf_pair_count() > 0:
                star_dev = self.star.sink_pin_ids.device
                self.clock_star = StarPinPair(
                    net2driver_pin_map=self.clock_domain.net2driver_pin_map.to(
                        device=star_dev
                    ),
                    timing_net_flat_arcs=self.clock_domain.timing_net_flat_arcs.to(
                        device=star_dev
                    ),
                    timing_net_flat_arcs_start=self.clock_domain.timing_net_flat_arcs_start.to(
                        device=star_dev
                    ),
                    num_pins=int(self.clock_domain.compact_to_native_pin_ids.numel()),
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    scale_factor=self.scale_factor,
                    dbu=self.dbu,
                    identity_namespace="clock",
                )
                self.clock_star.compact_to_native_pin_ids = (
                    self.clock_domain.compact_to_native_pin_ids.to(device=star_dev)
                )
                self.clock_star.clock_row_to_compact_pin_ids = (
                    self.clock_domain.clock_row_to_compact_pin_ids.to(device=star_dev)
                )
                op_collections.clock_star_pin_pair_op = self.clock_star
                clk_net2pin, clk_start = build_clock_star_steiner_net2pin(
                    self.clock_star
                )
                self.clock_steiner_topo = steiner_topo.SteinerTopo(
                    clk_net2pin.cpu(),
                    clk_start.cpu(),
                    ignore_net_degree=int(
                        getattr(params, "ignore_net_degree", 100) or 100
                    ),
                    deterministic_flag=bool(
                        getattr(params, "deterministic_flag", False)
                    ),
                )
                self.clock_teacher_rc = teacher_rc_like(elmore_op)

        self.slope_calibrator: Optional[SlopeCalibration] = None
        self.data_steiner_topo = None
        self.data_teacher_rc = None

        input_history_mode = (
            str(
                getattr(params, "star_input_history_mode", "accumulate")
                or "accumulate"
            )
            .strip()
            .lower()
        )
        self.input_history_mode = input_history_mode
        self.input_history_gain = float(
            getattr(params, "star_input_history_gain", 0.2) or 0.2
        )
        raw_half_life = getattr(params, "star_input_history_half_life", 0.0)
        self.input_history_half_life = (
            150.0
            if input_history_mode == "ema" and (raw_half_life is None or float(raw_half_life) <= 0.0)
            else float(raw_half_life or 0.0)
        )
        raw_carry_ratio = getattr(
            params, "star_input_history_max_carry_ratio", 0.0
        )
        self.input_history_max_carry_ratio = (
            0.0 if raw_carry_ratio is None else max(0.0, float(raw_carry_ratio))
        )
        self.input_history_reference_interval = max(
            0.0, float(getattr(params, "star_input_history_reference_interval", 0.0) or 0.0)
        )
        self.satisfied_release_mode = (
            str(getattr(params, "star_satisfied_release_mode", "off") or "off")
            .strip()
            .lower()
        )
        self.satisfied_release_threshold_ps = float(
            getattr(params, "star_satisfied_release_threshold_ps", 60.0) or 60.0
        )
        self.satisfied_release_band_ps = float(
            getattr(params, "star_satisfied_release_band_ps", 140.0) or 140.0
        )
        self.satisfied_release_tau_mult = float(
            getattr(params, "star_satisfied_release_tau_mult", 2.0) or 2.0
        )
        self.satisfied_release_half_life = float(
            getattr(params, "star_satisfied_release_half_life", 45.0) or 45.0
        )
        self.at_merge_step_norm = bool(
            getattr(params, "timing_at_merge_vjp_tau_step_norm", 0)
        )
        self.at_merge_min_horizon_steps = max(
            0.0, float(getattr(params, "timing_at_merge_vjp_tau_min_horizon_steps", 0.0) or 0.0)
        )
        self.at_merge_leg_floor_rows = max(
            0.0, float(getattr(params, "timing_at_merge_vjp_tau_leg_floor_rows", 0.0) or 0.0)
        )
        self._history_virtual_iter = 0
        self._last_raw_iter: Optional[int] = None
        self.input_accumulator: Optional[InputAccumulator] = None
        self._reset_input_accumulator()
        self.at_merge_temperature: Optional[AtMergeTemperature] = None
        self._reset_at_merge_temperature()

        self.data_rc_cal_state: Optional[StarRCCalibrationState] = None
        self.clock_rc_cal_state: Optional[ClockRCCalibrationState] = None
        self.rc_next_overflow: Optional[float] = None
        self.last_cap_digest: Optional[str] = None
        self.rc_teacher_call_count = 0

        self._cap_epoch = 0
        self._cached_caps_key = None
        self._cached_real_caps: Optional[Dict[str, torch.Tensor]] = None
        self._cached_full_caps: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

        self.gate_active = False
        self.geometry_dirty = False
        self.initial_lambda_star = (
            float(self.fixed_lambda) if self.fixed_lambda else 0.0
        )
        self.base_lambda_star = (
            float(self.fixed_lambda) if self.fixed_lambda else 0.0
        )
        self._fallback_base_lambda_star = (
            float(self.fixed_lambda) if self.fixed_lambda else 0.0
        )
        self.lambda_star = float(self.fixed_lambda) if self.fixed_lambda else 0.0
        self.last_realized_kappa = 0.0
        self.kappa_matched = self.fixed_lambda is not None
        self.kappa_deferred_count = 0
        self.kappa_calibration_record: Optional[Dict[str, object]] = None
        self._refreshes_since_restart = 0
        self.refresh_count = 0
        self.last_refresh_summary: Optional[Dict[str, object]] = None
        self.last_objective_value: Optional[torch.Tensor] = None
        self._committed_pin_slack: Optional[Dict[str, torch.Tensor]] = None
        self._committed_master_digest: Optional[str] = None
        self._committed_pin_cap_sig: Optional[torch.Tensor] = None
        self._restored_mutated_cap_mask: Optional[torch.Tensor] = None
        self.slack_drift_tau_history: List[float] = []
        self._extra_at_merge_tau_history: List[float] = []
        self._pending_slack_beta_rescale = False
        self._has_restarted = False
        self._pre_restart_realized_kappa = 0.0
        self._pending_recal_carry_ratio: Optional[float] = None
        self._pending_recal_m_eff = 1.0

        self.step_context: Optional[Dict[str, object]] = None
        self._step_alpha_start: Optional[float] = None
        self.transport_state: Dict[str, Any] = {
            "last": None,
            "carried": None,
            "base": None,
            "timing": None,
            "terms": None,
            "carried_terms": None,
        }
        self._current_outer_iteration = 0
        self._current_outer_overflow: Optional[float] = None
        self._first_active_iteration: Optional[int] = None
        self.last_timing_curvature_2n: Optional[torch.Tensor] = None

        self.macro_exclusion_field = None
        if bool(getattr(params, "macro_exclusion_field_flag", 0)):
            from dreamplace.ops.macro_exclusion_field import MacroExclusionField

            self.macro_exclusion_field = MacroExclusionField(
                placedb,
                data_collections,
                data_collections.pos[0],
                ov_start=float(getattr(params, "macro_exclusion_ov_start", 0.20)),
                ov_full=float(getattr(params, "macro_exclusion_ov_full", 0.085)),
                beta_max=float(getattr(params, "macro_exclusion_beta_max", 1.10)),
                bg_wl_scale=float(getattr(params, "macro_exclusion_bg_wl_scale", 0.0)),
                wl_gate_scale=float(
                    getattr(params, "macro_exclusion_wl_gate_scale", 0.35)
                ),
                warmup_iters=int(getattr(params, "macro_exclusion_warmup_iters", 110)),
                ramp_iters=int(getattr(params, "macro_exclusion_ramp_iters", 40)),
                collar_rows=float(getattr(params, "macro_exclusion_collar_rows", 1.0)),
                distance_blend=float(
                    getattr(params, "macro_exclusion_distance_blend", 0.15)
                ),
                profile_mode=str(
                    getattr(params, "macro_exclusion_profile_mode", "tent_rim")
                ),
                ridge_smooth_rows=float(
                    getattr(params, "macro_exclusion_ridge_smooth_rows", 3.0)
                ),
                slack_gate_mode=str(
                    getattr(params, "macro_exclusion_slack_gate_mode", "kappa_evac")
                ),
                slack_margin_ps=float(
                    getattr(params, "macro_exclusion_slack_margin_ps", 300.0)
                ),
                slack_tau_ps=float(
                    getattr(params, "macro_exclusion_slack_tau_ps", 100.0)
                ),
                depth_ps_per_um=float(
                    getattr(params, "macro_exclusion_depth_ps_per_um", 2.5)
                ),
                crit_wl_scale=float(
                    getattr(params, "macro_exclusion_crit_wl_scale", 0.65)
                ),
                ratchet_decay=float(
                    getattr(params, "macro_exclusion_ratchet_decay", 1.0)
                ),
                kappa_slack=float(
                    getattr(params, "macro_exclusion_kappa_slack", 1.0)
                ),
                row_margin=float(
                    getattr(params, "macro_exclusion_row_margin", 5.0)
                ),
                force_mode=str(
                    getattr(params, "macro_exclusion_force_mode", "taylor_kkt")
                ),
                escape_slope_mode=str(
                    getattr(params, "macro_exclusion_escape_slope_mode", "global")
                ),
                gate_release_mode=str(
                    getattr(params, "macro_exclusion_gate_release_mode", "ratchet")
                ),
                hyst_kappa_band=float(
                    getattr(params, "macro_exclusion_hyst_kappa_band", 0.75)
                ),
                interior_evac_boost=float(
                    getattr(params, "macro_exclusion_interior_evac_boost", 0.50)
                ),
                bell_sigma_rows=float(
                    getattr(params, "macro_exclusion_bell_sigma_rows", 16.0)
                ),
                bell_bias=float(
                    getattr(params, "macro_exclusion_bell_bias", 0.80)
                ),
            )
            op_collections.macro_exclusion_field_op = self.macro_exclusion_field

        logging.info(
            "StarTdpRuntime initialized: data_pairs=%d leaf_clock_nets=%d leaf_clock_pairs=%d "
            "distance=%s loss=%s smooth=%d(tau=%.4g,vjp=%s,ep=%s) load_sign=%s kappa=%.6g "
            "max_kappa=%.6g(post=%.6g) history=%s(hl=%.4g,carry=%.4g,restart=%s/%s) "
            "slope_cal=%d rc_cal=%d(every=%d) curv=%d macro_field=%d(obs=%d)",
            int(self.star.pair_count),
            0
            if self.clock_domain is None
            else len(self.clock_domain.plan.leaf_net_ids),
            0 if self.clock_star is None else int(self.clock_star.pair_count),
            self.distance_model,
            self.loss_mode,
            int(self.smooth_extremum),
            self.extremum_tau_ps,
            self.at_merge_vjp_tau,
            self.endpoint_slack_tau_mode,
            self.load_sign,
            self.requested_kappa,
            self.max_kappa,
            self.post_restart_max_kappa,
            self.input_history_mode,
            self.input_history_half_life,
            self.input_history_max_carry_ratio,
            self.restart_history_mode,
            self.restart_beta_tau_mode,
            int(self.slope_calibration_enabled),
            int(self.rc_calibration_enabled),
            int(self.rc_resample_every_refresh),
            int(self.curvature_precondition_enabled),
            int(self.macro_exclusion_field is not None),
            int(
                getattr(self.macro_exclusion_field, "metadata", {}).get(
                    "total_obstacle_boxes", 0
                )
            )
            if self.macro_exclusion_field is not None
            else 0,
        )

    def _compute_leg_floor_tau_ps(self) -> float:
        if self.at_merge_leg_floor_rows <= 0.0:
            return 0.0
        pos_per_um = max(float(getattr(self.star, "coordinate_to_um", 1.0) or 1.0), 1e-6)
        row_h_um = max(float(getattr(self.placedb, "row_height", 0.0) or 0.0) / pos_per_um, 0.0)
        if row_h_um <= 0.0:
            return 0.0
        disp_um = self.at_merge_leg_floor_rows * row_h_um
        r_u = max(float(getattr(self.star, "r_unit", 0.0) or 0.0), 0.0)
        c_u = max(float(getattr(self.star, "c_unit", 0.0) or 0.0), 0.0)
        slope_est = max(r_u * (0.005 + c_u * max(disp_um, 20.0)), 0.15)
        return (math.sqrt(2.0) * slope_est * disp_um) / 1.70

    def _reset_at_merge_temperature(self) -> None:
        if self.smooth_extremum and self.at_merge_vjp_tau == "drift":
            self.at_merge_temperature = AtMergeTemperature(
                self.extremum_tau_ps,
                step_normalize=self.at_merge_step_norm,
                min_horizon_steps=self.at_merge_min_horizon_steps,
                leg_floor_tau_ps=self._compute_leg_floor_tau_ps(),
            )
        else:
            self.at_merge_temperature = None

    def _reset_input_accumulator(self) -> None:
        self._history_virtual_iter = 0
        self._last_raw_iter = None
        if self.input_history_mode != "off":
            self.input_accumulator = InputAccumulator(
                gain=self.input_history_gain,
                half_life=self.input_history_half_life,
                mode=self.input_history_mode,
                max_carry_ratio=self.input_history_max_carry_ratio,
                reference_interval=self.input_history_reference_interval,
            )
        else:
            self.input_accumulator = None

    def _current_master_digest(self) -> Optional[str]:
        offsets = getattr(self.data_collections, "inst_libcell_offset", None)
        if offsets is None or not torch.is_tensor(offsets) or offsets.numel() <= 0:
            return None
        raw = offsets.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _current_pin_cap_signature(
        self, pin_tokens: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Per-pin ``(node_master_offset, net_degree, net_pin_token_sum)`` CPU int64 signature."""
        offsets = getattr(self.data_collections, "inst_libcell_offset", None)
        p2node = getattr(self.data_collections, "pin2node_map", None)
        p2net = getattr(self.data_collections, "pin2net_map", None)
        if offsets is None or p2node is None or p2net is None or self.num_pins <= 0:
            return None
        off_cpu = torch.as_tensor(offsets).detach().cpu().reshape(-1).long()
        node_cpu = torch.as_tensor(p2node).detach().cpu().reshape(-1)[: self.num_pins].long()
        net_cpu = torch.as_tensor(p2net).detach().cpu().reshape(-1)[: self.num_pins].long()
        if int(node_cpu.numel()) != self.num_pins or int(net_cpu.numel()) != self.num_pins:
            return None
        valid_node = (node_cpu >= 0) & (node_cpu < int(off_cpu.numel()))
        master_col = torch.full((self.num_pins,), -1, dtype=torch.long)
        if bool(valid_node.any().item()):
            master_col[valid_node] = off_cpu[node_cpu[valid_node]]
        tokens = (
            torch.arange(self.num_pins, dtype=torch.long)
            if pin_tokens is None
            else pin_tokens.detach().cpu().reshape(-1)[: self.num_pins].long()
        )
        valid_net = net_cpu >= 0
        deg_col = torch.zeros((self.num_pins,), dtype=torch.long)
        sum_col = torch.zeros((self.num_pins,), dtype=torch.long)
        if bool(valid_net.any().item()):
            num_nets = int(net_cpu[valid_net].max().item()) + 1
            net_deg = torch.zeros((num_nets,), dtype=torch.long)
            net_sum = torch.zeros((num_nets,), dtype=torch.long)
            ones = torch.ones_like(net_cpu[valid_net])
            net_deg.index_add_(0, net_cpu[valid_net], ones)
            net_sum.index_add_(0, net_cpu[valid_net], tokens[valid_net])
            deg_col[valid_net] = net_deg[net_cpu[valid_net]]
            sum_col[valid_net] = net_sum[net_cpu[valid_net]]
        return torch.stack((master_col, deg_col, sum_col), dim=1)

    def _at_merge_tau_samples(self) -> List[float]:
        samples = [
            float(v)
            for v in self._extra_at_merge_tau_history
            if v is not None and math.isfinite(float(v)) and float(v) > 0.0
        ]
        if self.at_merge_temperature is not None:
            for rec in self.at_merge_temperature.history:
                if (
                    rec is not None
                    and rec.get("sigma_delta_ps") is not None
                    and rec.get("tau_next_ps") is not None
                ):
                    val = float(rec["tau_next_ps"])
                    if math.isfinite(val) and val > 0.0:
                        samples.append(val)
        return samples

    def _resolve_beta_tau_ps(self) -> float:
        if self.restart_beta_tau_mode == "fixed":
            return max(1e-6, float(self.restart_beta_fixed_tau_ps))
        return summarize_drift_tau_history(
            self.slack_drift_tau_history,
            at_merge_tau_history=self._at_merge_tau_samples(),
            base_tau_ps=self.extremum_tau_ps,
            scale=self.restart_beta_tau_scale,
            min_tau_ps=self.restart_beta_min_tau_ps,
        )

    def _resolve_endpoint_zero_tau_ps(self) -> Optional[float]:
        if not self.smooth_extremum or self.endpoint_slack_tau_mode == "fixed":
            return None
        if self.endpoint_slack_tau_mode == "post_restart_drift" and not self._has_restarted:
            return None
        return summarize_drift_tau_history(
            self.slack_drift_tau_history,
            at_merge_tau_history=self._at_merge_tau_samples(),
            base_tau_ps=self.extremum_tau_ps,
            scale=self.endpoint_slack_tau_scale,
            min_tau_ps=self.extremum_tau_ps,
        )

    def _reset_transport_state(self) -> None:
        self.transport_state["last"] = None
        self.transport_state["carried"] = None
        self.transport_state["base"] = None
        self.transport_state["timing"] = None
        self.transport_state["terms"] = None
        self.transport_state["carried_terms"] = None

    def _prepare_restart_transition(self, *, recalibrate_kappa: bool) -> None:
        self._reset_transport_state()
        self.data_rc_cal_state = None
        self.clock_rc_cal_state = None
        self.rc_next_overflow = None
        self.last_cap_digest = None
        self._refreshes_since_restart = 0
        self._has_restarted = True
        if self.macro_exclusion_field is not None:
            hist_decay = (
                0.5
                if self.restart_history_mode
                in ("slack_beta", "delay_sigmoid", "selective_sigmoid")
                else 0.0
            )
            self.macro_exclusion_field.reset_history(decay=hist_decay)
        at_samples = self._at_merge_tau_samples()
        if at_samples:
            self._extra_at_merge_tau_history = list(at_samples)
        if self.restart_history_mode in (
            "slack_beta",
            "delay_sigmoid",
            "selective_sigmoid",
        ):
            self._last_raw_iter = None
            if (
                self.input_accumulator is not None
                and self.input_accumulator.values is not None
            ):
                self._pending_slack_beta_rescale = True
            if self.at_merge_temperature is not None:
                self.at_merge_temperature.previous = None
                self.at_merge_temperature.current = None
                if at_samples:
                    self.at_merge_temperature.tau = summarize_drift_tau_history(
                        (),
                        at_merge_tau_history=at_samples,
                        base_tau_ps=self.extremum_tau_ps,
                        scale=1.0,
                        min_tau_ps=self.extremum_tau_ps,
                    )
        else:
            self._reset_input_accumulator()
            self._reset_at_merge_temperature()
            if self.at_merge_temperature is not None and at_samples:
                self.at_merge_temperature.tau = summarize_drift_tau_history(
                    (),
                    at_merge_tau_history=at_samples,
                    base_tau_ps=self.extremum_tau_ps,
                    scale=1.0,
                    min_tau_ps=self.extremum_tau_ps,
                )
            self._pending_slack_beta_rescale = False
            self._pending_recal_carry_ratio = None

        if recalibrate_kappa and self.fixed_lambda is None:
            if self.base_lambda_star > 0.0 and math.isfinite(self.base_lambda_star):
                self._fallback_base_lambda_star = float(self.base_lambda_star)
            if (
                self.kappa_matched
                and self._pending_recal_carry_ratio is None
                and self.last_realized_kappa > 0.0
                and math.isfinite(self.last_realized_kappa)
            ):
                self._pre_restart_realized_kappa = float(self.last_realized_kappa)
            self.initial_lambda_star = 0.0
            self.base_lambda_star = 0.0
            self.lambda_star = 0.0
            self.last_realized_kappa = 0.0
            self.kappa_matched = False
            self.kappa_deferred_count = 0

    def _ensure_slope_calibrator(self) -> Optional[SlopeCalibration]:
        if not self.slope_calibration_enabled or int(self.star.pair_count) <= 0:
            return None
        if not (
            hasattr(self.data_collections, "flat_net2pin_map")
            and hasattr(self.data_collections, "flat_net2pin_start_map")
            and hasattr(self.data_collections, "pin2net_map")
        ):
            return None
        if (
            self.slope_calibrator is None
            or self.slope_calibrator.pair_topology_generation
            != int(self.star.pair_topology_generation)
        ):
            self.slope_calibrator = SlopeCalibration(
                placedb=self.placedb,
                params=self.params,
                data_collections=self.data_collections,
                star=self.star,
                elmore_delay_op=self.op_collections.elmore_delay_op,
            )
        return self.slope_calibrator

    def _ensure_data_teacher(self):
        calibrator = self._ensure_slope_calibrator()
        if calibrator is not None:
            return calibrator.topo, calibrator.rc
        if self.data_steiner_topo is None or self.data_teacher_rc is None:
            net2pin = self.data_collections.flat_net2pin_map.detach().cpu()
            start = self.data_collections.flat_net2pin_start_map.detach().cpu()
            ignored_clock = sorted(
                clock_net_ids(self.placedb, self.data_collections)
            )
            self.data_steiner_topo = steiner_topo.SteinerTopo(
                net2pin,
                start,
                ignore_net_degree=getattr(
                    self.params, "ignore_net_degree", None
                ),
                ignored_net_ids=ignored_clock,
                deterministic_flag=bool(
                    getattr(self.params, "deterministic_flag", False)
                ),
            )
            self.data_teacher_rc = teacher_rc_like(
                self.op_collections.elmore_delay_op
            )
        return self.data_steiner_topo, self.data_teacher_rc

    def _resolve_pin_direct(self, timing_op, net2driver) -> list[str]:
        """Resolve IO/direction strings; synthesize when placedb.pin_direct stores orientations (ecc_py)."""
        raw = getattr(
            self.placedb,
            "pin_direct",
            getattr(self.data_collections, "pin_direct", None),
        )
        if raw is not None and len(raw) == self.num_pins:
            if any(pin_is_output(item) or pin_is_input(item) for item in raw[:64]):
                return list(raw)
        directions = ["INPUT"] * self.num_pins
        output_mask = getattr(self.model, "output_pin_mask", None)
        if output_mask is not None and torch.is_tensor(output_mask):
            out_idx = torch.nonzero(output_mask[: self.num_pins]).reshape(-1).tolist()
            for pin_id in out_idx:
                directions[int(pin_id)] = "OUTPUT"
        if net2driver is not None:
            drv_list = (
                torch.as_tensor(net2driver).detach().reshape(-1).long().cpu().tolist()
            )
            for pin_id in drv_list:
                if 0 <= int(pin_id) < self.num_pins:
                    directions[int(pin_id)] = "OUTPUT"
        raw_pydb = getattr(getattr(self.placedb, "rawdb", None), "pydb", None)
        raw_net2driver = getattr(raw_pydb, "net2driver_pin_map", None)
        if raw_net2driver is not None:
            for pin_id in raw_net2driver:
                if 0 <= int(pin_id) < self.num_pins:
                    directions[int(pin_id)] = "OUTPUT"
        level_arcs = getattr(timing_op, "flat_inst_arcs_by_level", None)
        if level_arcs is not None and torch.is_tensor(level_arcs) and level_arcs.numel():
            arcs_2d = (
                level_arcs
                if level_arcs.ndim == 2
                else level_arcs.reshape(-1, 8)
            )
            dst_pins = (
                arcs_2d[:, 1].detach().reshape(-1).long().cpu().unique().tolist()
            )
            for pin_id in dst_pins:
                if 0 <= int(pin_id) < self.num_pins:
                    directions[int(pin_id)] = "OUTPUT"
        else:
            inst_arcs = getattr(timing_op, "inst_flat_arcs", None)
            if (
                inst_arcs is not None
                and torch.is_tensor(inst_arcs)
                and inst_arcs.ndim == 2
                and inst_arcs.numel()
            ):
                dst_pins = (
                    inst_arcs[:, 1].detach().reshape(-1).long().cpu().unique().tolist()
                )
                for pin_id in dst_pins:
                    if 0 <= int(pin_id) < self.num_pins:
                        directions[int(pin_id)] = "OUTPUT"
        start_pts = getattr(
            self.data_collections,
            "start_points",
            getattr(timing_op, "start_points", None),
        )
        if start_pts is not None and torch.is_tensor(start_pts) and start_pts.numel():
            for pin_id in start_pts.detach().reshape(-1).long().cpu().tolist():
                if 0 <= int(pin_id) < self.num_pins:
                    directions[int(pin_id)] = "OUTPUT"
        return directions

    # ------------------------------------------------------------------
    # Adapter methods consumed by rowsum_hessian / clock_rowsum_hessian
    # ------------------------------------------------------------------
    @property
    def segment_timing_coefficient(self) -> float:
        return float(self.lambda_star) if self.is_objective_active else 0.0

    def _star_pin_pair_op(self) -> StarPinPair:
        return self.star

    def _segment_timing_star_load_curvature(self) -> bool:
        return self.load_sign != "zero"

    def _segment_timing_star_pair_objective(self) -> str:
        return "elmore"

    def _segment_timing_cut_distance_model(self) -> str:
        return self.distance_model

    def _segment_timing_cut_effective_smoothing_gamma(self) -> float:
        gamma = getattr(self.model, "gamma", None)
        if torch.is_tensor(gamma):
            return float(gamma.detach().reshape(-1)[0].item())
        return float(gamma or 1.0)

    def _driver_sink_star_base_cap_map(self, device=None) -> Dict[str, torch.Tensor]:
        real_caps, _ = self._resolve_pin_caps(
            device=device if device is not None else self.star.sink_pin_ids.device
        )
        return real_caps

    def _segment_timing_clock_weight_config(self) -> Tuple[str, float]:
        if not self.lcb_ff_clock_enabled or self.clock_star is None:
            return "off", 0.0
        return "launch_capture", float(self.clock_weight_scale)

    def _lcb_ff_clock_compact_pin_pos(self, pin_pos: torch.Tensor) -> torch.Tensor:
        if self.clock_star is None:
            return pin_pos.new_empty(0)
        full_count = int(pin_pos.numel() // 2)
        native_ids = self.clock_star.compact_to_native_pin_ids.to(
            device=pin_pos.device, dtype=torch.long
        )
        return torch.cat(
            (
                pin_pos[:full_count].index_select(0, native_ids),
                pin_pos[full_count:].index_select(0, native_ids),
            ),
            dim=0,
        )

    def _lcb_ff_clock_base_cap_map(self, device=None) -> Dict[str, torch.Tensor]:
        if self.clock_star is None:
            empty = torch.zeros(0, dtype=torch.float32)
            return {"generic": empty, "rise": empty, "fall": empty}
        target_device = (
            device if device is not None else self.clock_star.sink_pin_ids.device
        )
        real_caps, _ = self._resolve_pin_caps(device=target_device)
        native_ids = self.clock_star.compact_to_native_pin_ids.to(
            device=target_device, dtype=torch.long
        )
        return {
            key: value.index_select(0, native_ids) for key, value in real_caps.items()
        }

    def record_outer_step(self, *, iteration: int, overflow=None) -> None:
        self._current_outer_iteration = int(iteration)
        ov = _scalar_overflow(overflow)
        if ov is not None:
            self._current_outer_overflow = ov

    def _segment_timing_cut_schedule_iteration(self) -> int:
        return int(self._current_outer_iteration)

    def _segment_timing_snapshot_start_iter(self) -> int:
        if self._first_active_iteration is not None:
            return int(self._first_active_iteration)
        return int(getattr(self.params, "timing_opt_iter", 200))

    def current_overflow(self) -> Optional[float]:
        if self._current_outer_overflow is not None:
            return float(self._current_outer_overflow)
        ov = getattr(self.model, "overflow", None)
        return _scalar_overflow(ov)

    # ------------------------------------------------------------------
    # State & invalidation hooks
    # ------------------------------------------------------------------
    @property
    def snapshot_generation(self) -> int:
        return int(self.star.snapshot_generation)

    @property
    def is_objective_active(self) -> bool:
        return (
            bool(self.gate_active)
            and self.snapshot_generation > 0
            and math.isfinite(self.lambda_star)
            and self.lambda_star > 0.0
        )

    @property
    def is_macro_field_active(self) -> bool:
        return (
            self.macro_exclusion_field is not None
            and bool(self.macro_exclusion_field.has_obstacles)
            and bool(self.gate_active)
            and self.snapshot_generation > 0
        )

    @property
    def driver_signature(self):
        if (
            self.macro_exclusion_field is not None
            and bool(self.macro_exclusion_field.has_obstacles)
        ):
            return (
                self.snapshot_generation,
                bool(self.is_objective_active),
                round(float(self.lambda_star), 12),
                bool(self.is_macro_field_active),
            )
        return (
            self.snapshot_generation,
            bool(self.is_objective_active),
            round(float(self.lambda_star), 12),
        )

    def set_gate_active(self, active: bool) -> None:
        was_active = bool(self.gate_active)
        self.gate_active = bool(active)
        if not self.gate_active:
            self._reset_transport_state()
            if was_active:
                self._prepare_restart_transition(
                    recalibrate_kappa=bool(
                        getattr(self.params, "star_kappa_recalibrate_on_restart", 0)
                    )
                )

    def invalidate_after_geometry_or_sizing_change(self) -> None:
        """Mark pin-cap caches dirty after S5B1 discrete sizing or area inflation."""
        self._cap_epoch += 1
        self._cached_caps_key = None
        self._cached_real_caps = None
        self._cached_full_caps = None
        self.geometry_dirty = True
        self.star._timing_precondition_cache_key = None
        self.star._timing_precondition_cache_x = None
        self.star._timing_precondition_cache_y = None
        if self.clock_star is not None:
            self.clock_star._timing_precondition_cache_key = None
            self.clock_star._timing_precondition_cache_x = None
            self.clock_star._timing_precondition_cache_y = None
        if self.macro_exclusion_field is not None:
            self.macro_exclusion_field.sync_movable_geometry(self.data_collections)

    def reset_for_restart(self, *, recalibrate_kappa: bool = False) -> Dict[str, object]:
        """Reset momentum transport, RC anchor, and history across GP/routability restarts."""
        self.invalidate_after_geometry_or_sizing_change()
        self._prepare_restart_transition(recalibrate_kappa=bool(recalibrate_kappa))
        return {
            "recalibrate_kappa": bool(recalibrate_kappa),
            "lambda_star": float(self.lambda_star),
            "snapshot_generation": int(self.snapshot_generation),
        }

    def export_midplace_state(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "initial_lambda_star": float(self.initial_lambda_star),
            "base_lambda_star": float(self.base_lambda_star),
            "fallback_base_lambda_star": float(self._fallback_base_lambda_star),
            "lambda_star": float(self.lambda_star),
            "kappa_matched": bool(self.kappa_matched),
            "kappa_deferred_count": int(self.kappa_deferred_count),
            "kappa_calibration_record": self.kappa_calibration_record,
            "slack_drift_tau_history": list(self.slack_drift_tau_history),
            "at_merge_tau_history": self._at_merge_tau_samples(),
            "has_restarted": bool(self._has_restarted),
            "pre_restart_realized_kappa": float(
                self.last_realized_kappa
                if (
                    self.kappa_matched
                    and self._pending_recal_carry_ratio is None
                    and self.last_realized_kappa > 0.0
                    and math.isfinite(self.last_realized_kappa)
                )
                else self._pre_restart_realized_kappa
            ),
        }
        if (
            self.restart_history_mode
            in ("slack_beta", "delay_sigmoid", "selective_sigmoid")
            and self.input_accumulator is not None
            and self.input_accumulator.values is not None
            and (
                self._committed_pin_slack is not None
                or self.restart_history_mode in ("delay_sigmoid", "selective_sigmoid")
            )
        ):
            raw_pin_names = getattr(self.placedb, "pin_names", None)
            pin_names = (
                tuple(raw_pin_names[: self.num_pins])
                if raw_pin_names is not None and len(raw_pin_names) >= self.num_pins
                else None
            )
            payload["num_pins"] = int(self.num_pins)
            payload["pin_names"] = pin_names
            payload["accumulator_values"] = {
                group: {
                    edge: vec.detach().cpu().clone()
                    for edge, vec in edges.items()
                }
                for group, edges in self.input_accumulator.values.items()
            }
            if self._committed_pin_slack is not None:
                payload["committed_pin_slack"] = {
                    edge: vec.detach().cpu().clone()
                    for edge, vec in self._committed_pin_slack.items()
                }
            if self._committed_pin_cap_sig is not None:
                payload["committed_pin_cap_sig"] = (
                    self._committed_pin_cap_sig.detach().cpu().clone()
                )
        return payload

    def restore_midplace_state(
        self, state: Optional[Mapping[str, object]], *, recalibrate_kappa: bool
    ) -> None:
        if not state:
            return
        self._has_restarted = True
        self.slack_drift_tau_history = [
            float(v)
            for v in state.get("slack_drift_tau_history", ())
            if v is not None and math.isfinite(float(v)) and float(v) > 0.0
        ]
        self._extra_at_merge_tau_history = [
            float(v)
            for v in state.get("at_merge_tau_history", ())
            if v is not None and math.isfinite(float(v)) and float(v) > 0.0
        ]
        if self.at_merge_temperature is not None:
            at_samples = self._at_merge_tau_samples()
            if at_samples:
                self.at_merge_temperature.tau = summarize_drift_tau_history(
                    (),
                    at_merge_tau_history=at_samples,
                    base_tau_ps=self.extremum_tau_ps,
                    scale=1.0,
                    min_tau_ps=self.extremum_tau_ps,
                )

        if state:
            prev_rk = float(state.get("pre_restart_realized_kappa", 0.0) or 0.0)
            if math.isfinite(prev_rk) and prev_rk > 0.0:
                self._pre_restart_realized_kappa = prev_rk
        if self.fixed_lambda is None:
            lam = float(state.get("lambda_star", 0.0) or 0.0)
            base_lam = float(state.get("base_lambda_star", lam) or lam)
            fb_lam = float(state.get("fallback_base_lambda_star", base_lam) or base_lam)
            init_lam = float(state.get("initial_lambda_star", base_lam) or base_lam)
            if math.isfinite(fb_lam) and fb_lam > 0.0:
                self._fallback_base_lambda_star = float(fb_lam)
            if not recalibrate_kappa and state.get("kappa_matched") and math.isfinite(lam) and lam > 0.0:
                self.base_lambda_star = (
                    base_lam if math.isfinite(base_lam) and base_lam > 0.0 else lam
                )
                self._fallback_base_lambda_star = float(self.base_lambda_star)
                self.initial_lambda_star = (
                    init_lam
                    if math.isfinite(init_lam) and init_lam > 0.0
                    else self.base_lambda_star
                )
                self.lambda_star = lam
                self.kappa_matched = True
                self.kappa_calibration_record = state.get("kappa_calibration_record")

        acc_vals = state.get("accumulator_values")
        prev_slack = state.get("committed_pin_slack")
        prev_cap_sig = state.get("committed_pin_cap_sig")
        if (
            self.restart_history_mode
            in ("slack_beta", "delay_sigmoid", "selective_sigmoid")
            and self.input_accumulator is not None
            and isinstance(acc_vals, Mapping)
            and (
                isinstance(prev_slack, Mapping)
                or self.restart_history_mode in ("delay_sigmoid", "selective_sigmoid")
            )
        ):
            old_n = int(state.get("num_pins", 0) or 0)
            if old_n > 0 and self.num_pins > 0:
                rc_device = self.star.sink_pin_ids.device
                old_names = state.get("pin_names")
                raw_new_names = getattr(self.placedb, "pin_names", None)
                new_names = (
                    tuple(raw_new_names[: self.num_pins])
                    if raw_new_names is not None and len(raw_new_names) >= self.num_pins
                    else None
                )
                src_idx = None
                dst_idx = None
                if (
                    old_names is not None
                    and new_names is not None
                    and len(old_names) == old_n
                ):
                    prefix = min(old_n, self.num_pins)
                    if new_names[:prefix] == old_names[:prefix]:
                        src_idx = torch.arange(prefix, dtype=torch.long)
                        dst_idx = src_idx
                    else:
                        old_map = {name: i for i, name in enumerate(old_names)}
                        pairs = [
                            (old_map[name], j)
                            for j, name in enumerate(new_names)
                            if name in old_map
                        ]
                        if pairs:
                            src_idx = torch.tensor(
                                [p[0] for p in pairs], dtype=torch.long
                            )
                            dst_idx = torch.tensor(
                                [p[1] for p in pairs], dtype=torch.long
                            )
                else:
                    prefix = min(old_n, self.num_pins)
                    src_idx = torch.arange(prefix, dtype=torch.long)
                    dst_idx = src_idx

                if src_idx is not None and dst_idx is not None and int(src_idx.numel()) > 0:
                    mutated_cap_mask_cpu = torch.ones((self.num_pins,), dtype=torch.bool)
                    if (
                        self.restart_history_mode == "selective_sigmoid"
                        and torch.is_tensor(prev_cap_sig)
                        and int(prev_cap_sig.shape[0]) == old_n
                    ):
                        pin_tokens = torch.full(
                            (self.num_pins,), -1_000_000_000, dtype=torch.long
                        )
                        pin_tokens[dst_idx] = src_idx
                        cur_sig = self._current_pin_cap_signature(pin_tokens=pin_tokens)
                        if cur_sig is not None:
                            same_sig = (
                                cur_sig[dst_idx] == prev_cap_sig.detach().cpu().long()[src_idx]
                            ).all(dim=1)
                            mutated_cap_mask_cpu[dst_idx] = ~same_sig

                    remapped_acc: Dict[str, Dict[str, torch.Tensor]] = {}
                    for group, edges in acc_vals.items():
                        remapped_acc[group] = {}
                        for edge, vec in edges.items():
                            dst_t = torch.zeros((self.num_pins,), dtype=torch.float32)
                            if not (
                                self.restart_history_mode == "delay_sigmoid"
                                and group == "cap"
                            ):
                                src_t = torch.as_tensor(vec, dtype=torch.float32).cpu()
                                dst_t[dst_idx] = src_t[src_idx]
                                if (
                                    self.restart_history_mode == "selective_sigmoid"
                                    and group == "cap"
                                ):
                                    dst_t[mutated_cap_mask_cpu] = 0.0
                            remapped_acc[group][edge] = dst_t.to(device=rc_device)
                    remapped_slack: Optional[Dict[str, torch.Tensor]] = None
                    if isinstance(prev_slack, Mapping):
                        remapped_slack = {}
                        for edge, vec in prev_slack.items():
                            src_s = torch.as_tensor(vec, dtype=torch.float32).cpu()
                            dst_s = torch.full(
                                (self.num_pins,), float("inf"), dtype=torch.float32
                            )
                            dst_s[dst_idx] = src_s[src_idx]
                            remapped_slack[edge] = dst_s.to(device=rc_device)

                    self.input_accumulator.values = remapped_acc
                    self.input_accumulator.identity = (
                        id(self.placedb),
                        self.num_pins,
                        int(self.star.pair_topology_generation),
                    )
                    self.input_accumulator.iteration = 0
                    self.input_accumulator.generation = 1
                    self._history_virtual_iter = 0
                    self._last_raw_iter = None
                    self._committed_pin_slack = remapped_slack
                    self._restored_mutated_cap_mask = (
                        mutated_cap_mask_cpu.to(device=rc_device)
                        if self.restart_history_mode == "selective_sigmoid"
                        else None
                    )
                    self._pending_slack_beta_rescale = True

    # ------------------------------------------------------------------
    # Capacitance & FLUTE teacher RC calibration
    # ------------------------------------------------------------------
    def _effective_surrogate_mode(self):
        surrogate_mode = getattr(self.params, "timing_surrogate_mode", None)
        if surrogate_mode is not None:
            return surrogate_mode
        model_params = getattr(self.model, "params", None)
        if model_params is not None:
            mode = getattr(model_params, "timing_surrogate_mode", None)
            if mode is not None:
                return mode
        return "lut_only"

    def _resolve_pin_caps(
        self, *, device: torch.device
    ) -> Tuple[
        Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        offsets = self.data_collections.inst_libcell_offset
        eff_mode = self._effective_surrogate_mode()
        key = (
            str(device),
            int(offsets.data_ptr()),
            int(self._cap_epoch),
            str(eff_mode),
        )
        if (
            self._cached_caps_key == key
            and self._cached_real_caps is not None
            and self._cached_full_caps is not None
        ):
            return self._cached_real_caps, self._cached_full_caps

        had_mode = hasattr(self.model, "_active_timing_surrogate_mode")
        prev_mode = getattr(self.model, "_active_timing_surrogate_mode", None)
        had_ctx = hasattr(self.model, "_active_discrete_action_context")
        prev_ctx = getattr(self.model, "_active_discrete_action_context", None)
        self.model._active_timing_surrogate_mode = eff_mode
        self.model._active_discrete_action_context = None
        try:
            with torch.no_grad():
                c_base, c_rise, c_fall = self.model.pin_caps_op(
                    offsets, self.data_collections
                )
                full_caps = (
                    c_base.detach().to(device=device),
                    c_rise.detach().to(device=device),
                    c_fall.detach().to(device=device),
                )
                real_caps = {
                    "generic": full_caps[0][: self.num_pins].contiguous(),
                    "rise": full_caps[1][: self.num_pins].contiguous(),
                    "fall": full_caps[2][: self.num_pins].contiguous(),
                }
        finally:
            if had_mode:
                self.model._active_timing_surrogate_mode = prev_mode
            elif hasattr(self.model, "_active_timing_surrogate_mode"):
                delattr(self.model, "_active_timing_surrogate_mode")
            if had_ctx:
                self.model._active_discrete_action_context = prev_ctx
            elif hasattr(self.model, "_active_discrete_action_context"):
                delattr(self.model, "_active_discrete_action_context")
        self._cached_caps_key = key
        self._cached_real_caps = real_caps
        self._cached_full_caps = full_caps
        return real_caps, full_caps

    def _policy_identity(self) -> Tuple[str, str]:
        mode_id = "generic,rise,fall"
        clk_digest = calibration_policy_digest(
            {
                "lcb_ff": bool(self.lcb_ff_clock_enabled),
                "leaf_nets": 0
                if self.clock_domain is None
                else len(self.clock_domain.plan.leaf_net_ids),
            }
        )
        return mode_id, clk_digest

    def _should_resample_rc_calibration(
        self, *, overflow_value: Optional[float], cap_digest: str
    ) -> bool:
        if not self.rc_calibration_enabled:
            return False
        if self.rc_resample_every_refresh:
            return True
        if self.data_rc_cal_state is None or self.last_cap_digest != cap_digest:
            return True
        if overflow_value is None or self.rc_next_overflow is None:
            return False
        return overflow_value <= self.rc_next_overflow

    def needs_live_topology_refresh(
        self, pos: torch.Tensor, *, overflow=None
    ) -> bool:
        return bool(
            callable(getattr(self.model, "_relaxed_buffer_timing_enabled", None))
            and self.model._relaxed_buffer_timing_enabled()
        )

    def _rebuild_live_steiner_topology_fallback(self, pos: torch.Tensor) -> None:
        self.model._timing_geometry_cache = None
        if getattr(self.placedb, "gr_sizing", None) is not None:
            return
        topo_op = getattr(self.op_collections, "steiner_topo_op", None)
        if topo_op is None:
            return
        pin_pos = self.op_collections.pin_pos_op(pos.detach())
        pin_pos_cpu = pin_pos.detach().cpu().contiguous()
        if bool(getattr(topo_op, "topology_frozen", False)) or not hasattr(
            topo_op, "rebuild_tree"
        ):
            new_x, new_y = topo_op(pin_pos_cpu)
            update_kind = "frozen_geometry_forward"
        else:
            topo_op.rebuild_tree(pin_pos_cpu)
            new_x = topo_op.newx
            new_y = topo_op.newy
            update_kind = "topology_rebuild"
        if hasattr(topo_op, "net_flat_topo_sort"):
            from dreamplace.flows.gp.topology_refresh import (
                TimingTopologyContext,
                publish_live_timing_topology,
            )

            publish_live_timing_topology(
                TimingTopologyContext(
                    data_collections=self.data_collections,
                    op_collections=self.op_collections,
                    placedb=self.placedb,
                    params=self.params,
                    segment_direct_joint_enabled=False,
                    record_stage=lambda *_args, **_kwargs: None,
                    record_bytes=lambda *_args, **_kwargs: None,
                    refresh_callback=None,
                ),
                new_x,
                new_y,
                update_kind=update_kind,
            )

    def _sample_teacher_rc_calibration(
        self,
        *,
        pos: torch.Tensor,
        pin_pos: torch.Tensor,
        real_caps: Dict[str, torch.Tensor],
        full_caps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cap_digest: str,
        overflow_value: Optional[float],
        iteration: int,
        topology_already_rebuilt: bool,
    ) -> Optional[tuple]:
        shared_tree = None
        with torch.no_grad():
            length_um = self.star.length_um(pin_pos.detach())
            star_wl = pair_wirelength_by_net(
                length_um, self.star.pair_net_ids, self.star.num_nets
            )
            if getattr(self.placedb, "gr_sizing", None) is not None:
                _, new_x, new_y = self.model._timing_geometry_for_pos(pos.detach())
                elmore_dev = new_x.device
                c_g = full_caps[0].to(elmore_dev)
                c_r = full_caps[1].to(elmore_dev)
                c_f = full_caps[2].to(elmore_dev)
                _, t_loads, t_delays, _, _, t_impulses = (
                    self.op_collections.elmore_delay_op(c_g, c_r, c_f)
                )
                teacher_wl = star_wl.detach().clone()
            elif hasattr(self.data_collections, "flat_net2pin_map") and hasattr(
                self.data_collections, "flat_net2pin_start_map"
            ):
                d_topo, d_rc = self._ensure_data_teacher()
                cpu_ppos = (
                    pin_pos.detach().to("cpu", dtype=torch.float32).contiguous()
                )
                tree = d_topo.rebuild_tree(cpu_ppos, resolve_l_directions=False)
                if (
                    self.slope_calibrator is not None
                    and d_topo is self.slope_calibrator.topo
                ):
                    shared_tree = tree
                sort, sort_start, fa, to, to_start, frm = tree
                dnx, dny = d_topo(cpu_ppos)
                dc_g = real_caps["generic"].detach().to("cpu", dtype=dnx.dtype)
                dc_r = real_caps["rise"].detach().to("cpu", dtype=dnx.dtype)
                dc_f = real_caps["fall"].detach().to("cpu", dtype=dnx.dtype)
                _, t_loads, t_delays, _, _, t_impulses = d_rc(
                    dnx, dny, sort, sort_start, fa, to_start, to, frm, dc_g, dc_r, dc_f
                )
                edge_geom = build_live_edge_geometry(
                    dnx,
                    dny,
                    frm,
                    to,
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    scale_factor=self.scale_factor,
                    dbu=self.dbu,
                )
                native_wl = tree_wirelength_by_net(
                    edge_length_um=edge_geom["length"],
                    net_flat_topo_sort=sort,
                    net_flat_topo_sort_start=sort_start,
                    flat_pin_to=to,
                ).to(device=star_wl.device, dtype=star_wl.dtype)
                if int(native_wl.numel()) == int(self.star.num_nets):
                    teacher_wl = native_wl
                else:
                    p2n = torch.as_tensor(
                        self.data_collections.pin2net_map,
                        device=star_wl.device,
                        dtype=torch.long,
                    ).reshape(-1)
                    drv = self.star.net_driver_pin_ids.to(
                        device=star_wl.device, dtype=torch.long
                    )
                    valid_drv = drv >= 0
                    mapped_nets = p2n[drv.clamp(min=0)].clamp(
                        min=0, max=max(0, int(native_wl.numel()) - 1)
                    )
                    teacher_wl = torch.where(
                        valid_drv,
                        native_wl.index_select(0, mapped_nets),
                        torch.zeros_like(star_wl),
                    )
            else:
                if not topology_already_rebuilt:
                    self._rebuild_live_steiner_topology_fallback(pos.detach())
                _, new_x, new_y = self.model._timing_geometry_for_pos(pos.detach())
                elmore_dev = new_x.device
                c_g = full_caps[0].to(elmore_dev)
                c_r = full_caps[1].to(elmore_dev)
                c_f = full_caps[2].to(elmore_dev)
                _, t_loads, t_delays, _, _, t_impulses = (
                    self.op_collections.elmore_delay_op(
                        new_x,
                        new_y,
                        self.data_collections.net_flat_topo_sort,
                        self.data_collections.net_flat_topo_sort_start,
                        self.data_collections.pin_fa,
                        self.data_collections.flat_pin_to_start,
                        self.data_collections.flat_pin_to,
                        self.data_collections.flat_pin_from,
                        c_g,
                        c_r,
                        c_f,
                    )
                )
                edge_geom = build_live_edge_geometry(
                    new_x,
                    new_y,
                    self.data_collections.flat_pin_from,
                    self.data_collections.flat_pin_to,
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    scale_factor=self.scale_factor,
                    dbu=self.dbu,
                )
                native_wl = tree_wirelength_by_net(
                    edge_length_um=edge_geom["length"],
                    net_flat_topo_sort=self.data_collections.net_flat_topo_sort,
                    net_flat_topo_sort_start=self.data_collections.net_flat_topo_sort_start,
                    flat_pin_to=self.data_collections.flat_pin_to,
                ).to(device=star_wl.device, dtype=star_wl.dtype)
                if int(native_wl.numel()) == int(self.star.num_nets):
                    teacher_wl = native_wl
                else:
                    p2n = torch.as_tensor(
                        self.data_collections.pin2net_map,
                        device=star_wl.device,
                        dtype=torch.long,
                    ).reshape(-1)
                    drv = self.star.net_driver_pin_ids.to(
                        device=star_wl.device, dtype=torch.long
                    )
                    valid_drv = drv >= 0
                    mapped_nets = p2n[drv.clamp(min=0)].clamp(
                        min=0, max=max(0, int(native_wl.numel()) - 1)
                    )
                    teacher_wl = torch.where(
                        valid_drv,
                        native_wl.index_select(0, mapped_nets),
                        torch.zeros_like(star_wl),
                    )
            teacher_loads = {
                k: v[: self.num_pins].detach().to(pin_pos.device)
                for k, v in t_loads.items()
            }
            teacher_delays = {
                k: v[: self.num_pins].detach().to(pin_pos.device)
                for k, v in t_delays.items()
            }
            teacher_impulses = {
                k: v[: self.num_pins].detach().to(pin_pos.device)
                for k, v in t_impulses.items()
            }

            _, s_loads, s_delays, _, _, _s_impulses = self.star.rc_forward(
                length_um, real_caps
            )
            mode_id, clk_digest = self._policy_identity()
            next_gen = int(self.rc_teacher_call_count) + 1
            self.data_rc_cal_state = StarRCCalibrationState.from_reference(
                pair_identity_hash=self.star.pair_identity_hash,
                pair_topology_generation=int(self.star.pair_topology_generation),
                source_snapshot_generation=int(self.star.snapshot_generation),
                teacher_topology_generation=next_gen,
                position=pin_pos.detach(),
                cap_digest=cap_digest,
                r_unit=self.r_unit,
                c_unit=self.c_unit,
                coordinate_to_um=self.coordinate_to_um,
                timing_mode_identity=mode_id,
                clock_policy_digest=clk_digest,
                pair_net_ids=self.star.pair_net_ids,
                driver_pin_ids=self.star.driver_pin_ids,
                sink_pin_ids=self.star.sink_pin_ids,
                base_caps=real_caps,
                star_wirelength_by_net=star_wl,
                teacher_wirelength_by_net=teacher_wl,
                star_loads=s_loads,
                teacher_loads=teacher_loads,
                star_delays=s_delays,
                teacher_delays=teacher_delays,
                teacher_impulses=teacher_impulses,
                anchor_iteration=int(iteration),
                anchor_overflow=overflow_value,
            )

            if (
                self.clock_star is not None
                and self.clock_steiner_topo is not None
                and self.clock_teacher_rc is not None
            ):
                compact_pos = self._lcb_ff_clock_compact_pin_pos(pin_pos.detach())
                compact_caps = self._lcb_ff_clock_base_cap_map(
                    device=compact_pos.device
                )
                cpu_cpos = compact_pos.detach().to("cpu", dtype=torch.float32).contiguous()
                tree = self.clock_steiner_topo.rebuild_tree(
                    cpu_cpos, resolve_l_directions=False
                )
                sort, sort_start, fa, to, to_start, frm = tree
                cnx, cny = self.clock_steiner_topo(cpu_cpos)
                cc_g = compact_caps["generic"].detach().to("cpu", dtype=cnx.dtype)
                cc_r = compact_caps["rise"].detach().to("cpu", dtype=cnx.dtype)
                cc_f = compact_caps["fall"].detach().to("cpu", dtype=cnx.dtype)
                _, ct_loads, ct_delays, _, _, ct_impulses = self.clock_teacher_rc(
                    cnx, cny, sort, sort_start, fa, to_start, to, frm, cc_g, cc_r, cc_f
                )
                n_clk_pins = int(self.clock_star.num_pins)
                clk_t_loads = {
                    k: v[:n_clk_pins].detach().to(compact_pos.device)
                    for k, v in ct_loads.items()
                }
                clk_t_delays = {
                    k: v[:n_clk_pins].detach().to(compact_pos.device)
                    for k, v in ct_delays.items()
                }
                clk_t_impulses = {
                    k: v[:n_clk_pins].detach().to(compact_pos.device)
                    for k, v in ct_impulses.items()
                }
                clk_len_um = self.clock_star.length_um(compact_pos)
                clk_star_wl = pair_wirelength_by_net(
                    clk_len_um, self.clock_star.pair_net_ids, self.clock_star.num_nets
                )
                clk_edge_geom = build_live_edge_geometry(
                    cnx,
                    cny,
                    frm,
                    to,
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    scale_factor=self.scale_factor,
                    dbu=self.dbu,
                )
                clk_teacher_wl = tree_wirelength_by_net(
                    edge_length_um=clk_edge_geom["length"],
                    net_flat_topo_sort=sort,
                    net_flat_topo_sort_start=sort_start,
                    flat_pin_to=to,
                ).to(device=clk_star_wl.device, dtype=clk_star_wl.dtype)
                _, cs_loads, cs_delays, _, _, _cs_impulses = self.clock_star.rc_forward(
                    clk_len_um, compact_caps
                )
                clk_cap_dig = calibration_cap_digest(compact_caps)
                self.clock_rc_cal_state = ClockRCCalibrationState.from_reference(
                    pair_identity_hash=self.clock_star.pair_identity_hash,
                    pair_topology_generation=int(
                        self.clock_star.pair_topology_generation
                    ),
                    source_snapshot_generation=int(self.clock_star.snapshot_generation),
                    teacher_topology_generation=next_gen,
                    position=compact_pos,
                    cap_digest=clk_cap_dig,
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    coordinate_to_um=self.coordinate_to_um,
                    timing_mode_identity=mode_id,
                    clock_policy_digest=clk_digest,
                    pair_net_ids=self.clock_star.pair_net_ids,
                    driver_pin_ids=self.clock_star.driver_pin_ids,
                    sink_pin_ids=self.clock_star.sink_pin_ids,
                    base_caps=compact_caps,
                    star_wirelength_by_net=clk_star_wl,
                    teacher_wirelength_by_net=clk_teacher_wl,
                    star_loads=cs_loads,
                    teacher_loads=clk_t_loads,
                    star_delays=cs_delays,
                    teacher_delays=clk_t_delays,
                    teacher_impulses=clk_t_impulses,
                    anchor_iteration=int(iteration),
                    anchor_overflow=overflow_value,
                )

            self.last_cap_digest = cap_digest
            self.rc_teacher_call_count += 1
            if overflow_value is not None:
                raw_act = float(
                    getattr(
                        self.params,
                        "timing_start_overflow",
                        getattr(self.params, "overflow_threshold", 0.5),
                    )
                    or 0.5
                )
                act_ovf = min(max(raw_act, 1e-3), 1.0)
                step_ovf = min(max(self.rc_overflow_step, 1e-4), act_ovf)
                self.rc_next_overflow = next_lower_overflow_milestone(
                    overflow_value,
                    activation_overflow=act_ovf,
                    overflow_step=step_ovf,
                )
        return shared_tree

    # ------------------------------------------------------------------
    # Snapshot publication & kappa dose calibration
    # ------------------------------------------------------------------
    def refresh_snapshot(
        self,
        pos: torch.Tensor,
        *,
        iteration: int,
        overflow=None,
        topology_already_rebuilt: bool = False,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        self.gate_active = True
        overflow_val = _scalar_overflow(overflow)
        self._current_outer_iteration = int(iteration)
        if overflow_val is not None:
            self._current_outer_overflow = overflow_val
        if self._first_active_iteration is None:
            self._first_active_iteration = int(iteration)
        timing_op = self.op_collections.timing_propagation_op

        cur_master_digest = self._current_master_digest()
        cur_cap_sig = (
            self._current_pin_cap_signature()
            if self.restart_history_mode == "selective_sigmoid"
            else None
        )
        master_changed = (
            self._committed_master_digest is not None
            and cur_master_digest is not None
            and cur_master_digest != self._committed_master_digest
        )
        if master_changed:
            self._has_restarted = True
            at_samples = self._at_merge_tau_samples()
            if at_samples:
                self._extra_at_merge_tau_history = list(at_samples)
            if self.at_merge_temperature is not None:
                self.at_merge_temperature.previous = None
                self.at_merge_temperature.current = None
                if at_samples:
                    self.at_merge_temperature.tau = summarize_drift_tau_history(
                        (),
                        at_merge_tau_history=at_samples,
                        base_tau_ps=self.extremum_tau_ps,
                        scale=1.0,
                        min_tau_ps=self.extremum_tau_ps,
                    )
            if self.restart_history_mode in (
            "slack_beta",
            "delay_sigmoid",
            "selective_sigmoid",
        ):
                if (
                    self.input_accumulator is not None
                    and self.input_accumulator.values is not None
                ):
                    self._pending_slack_beta_rescale = True
            else:
                self._reset_input_accumulator()
                self._pending_slack_beta_rescale = False
                self._pending_recal_carry_ratio = None
            if self.fixed_lambda is None and bool(
                getattr(self.params, "star_kappa_recalibrate_on_restart", 0)
            ):
                if self.base_lambda_star > 0.0 and math.isfinite(self.base_lambda_star):
                    self._fallback_base_lambda_star = float(self.base_lambda_star)
                if (
                    self.kappa_matched
                    and self._pending_recal_carry_ratio is None
                    and self.last_realized_kappa > 0.0
                    and math.isfinite(self.last_realized_kappa)
                ):
                    self._pre_restart_realized_kappa = float(self.last_realized_kappa)
                self.initial_lambda_star = 0.0
                self.base_lambda_star = 0.0
                self.lambda_star = 0.0
                self.last_realized_kappa = 0.0
                self.kappa_matched = False
                self.kappa_deferred_count = 0

        with torch.no_grad():
            pin_pos = self.op_collections.pin_pos_op(pos.detach())
        rc_device = pin_pos.device
        real_caps, full_caps = self._resolve_pin_caps(device=rc_device)
        cap_digest = calibration_cap_digest(real_caps)

        need_teacher = self._should_resample_rc_calibration(
            overflow_value=overflow_val, cap_digest=cap_digest
        )
        relaxed_buffering_active = bool(
            callable(getattr(self.model, "_relaxed_buffer_timing_enabled", None))
            and self.model._relaxed_buffer_timing_enabled()
        )
        shared_teacher_tree = None
        if need_teacher:
            shared_teacher_tree = self._sample_teacher_rc_calibration(
                pos=pos,
                pin_pos=pin_pos,
                real_caps=real_caps,
                full_caps=full_caps,
                cap_digest=cap_digest,
                overflow_value=overflow_val,
                iteration=iteration,
                topology_already_rebuilt=topology_already_rebuilt,
            )
        if relaxed_buffering_active and not topology_already_rebuilt:
            with torch.no_grad():
                self._rebuild_live_steiner_topology_fallback(pos.detach())
            topology_already_rebuilt = True

        # 1. Calibrated data Star RC forward (detached geometry, leaf RC tensors for VJP).
        mode_id, clk_digest = self._policy_identity()
        with torch.no_grad():
            length_um = self.star.length_um(pin_pos.detach())
            _, raw_loads, raw_delays, _, _, raw_impulses = self.star.rc_forward(
                length_um, real_caps
            )
            if self.data_rc_cal_state is not None:
                raw_loads, raw_delays, raw_impulses = self.data_rc_cal_state.apply(
                    star=self.star,
                    position=pin_pos.detach(),
                    cap_digest=cap_digest,
                    loads=raw_loads,
                    delays=raw_delays,
                    impulses=raw_impulses,
                    timing_mode_identity=mode_id,
                    clock_policy_digest=clk_digest,
                )

            # 2. Optional leaf-clock LCB/ICG -> FF Star state.
            clk_aat_compact = None
            clk_tran_compact = None
            clk_leaf_mask = None
            if self.clock_domain is not None and self.clock_star is not None:
                compact_pos = self._lcb_ff_clock_compact_pin_pos(pin_pos.detach())
                compact_caps = self._lcb_ff_clock_base_cap_map(
                    device=compact_pos.device
                )
                clk_cap_dig = calibration_cap_digest(compact_caps)
                clk_len_um = self.clock_star.length_um(compact_pos)
                _, clk_loads, clk_delays, _, _, clk_impulses = (
                    self.clock_star.rc_forward(
                        clk_len_um, compact_caps, validate=False
                    )
                )
                if self.clock_rc_cal_state is not None:
                    clk_loads, clk_delays, clk_impulses = (
                        self.clock_rc_cal_state.apply(
                            star=self.clock_star,
                            position=compact_pos,
                            cap_digest=clk_cap_dig,
                            loads=clk_loads,
                            delays=clk_delays,
                            impulses=clk_impulses,
                            timing_mode_identity=mode_id,
                            clock_policy_digest=clk_digest,
                        )
                    )
                clk_rc_inputs = {
                    "model": (
                        "clock_star_calibrated"
                        if self.clock_rc_cal_state is not None
                        else "clock_star_raw"
                    ),
                    "loads": clk_loads,
                    "delays": clk_delays,
                    "impulses": clk_impulses,
                }
                clock_state = compute_lcb_ff_clock_state(
                    plan=self.clock_domain.plan,
                    pin_pos=compact_pos,
                    base_caps=compact_caps,
                    r_unit=self.r_unit,
                    c_unit=self.c_unit,
                    coordinate_to_um=self.coordinate_to_um,
                    timing_op=timing_op,
                    data_collections=self.data_collections,
                    rc_inputs=clk_rc_inputs,
                )
                apply_lcb_ff_to_compact_clk_pins(timing_op, clock_state)
                clk_aat_compact = torch.stack(
                    (
                        clock_state["late_rise_arrival_ps"],
                        clock_state["late_fall_arrival_ps"],
                    ),
                    dim=0,
                )
                clk_tran_compact = torch.stack(
                    (
                        clock_state["late_rise_slew_ps"],
                        clock_state["late_fall_slew_ps"],
                    ),
                    dim=0,
                )
                clk_leaf_mask = ~self.clock_domain.plan.sink_is_root.to(
                    device=compact_pos.device
                )

        # 3. Differentiable TimingPropagation forward + VJP wrt delay, cap, impulse, and clk_pin_aat.
        timing_dev = getattr(timing_op, "device", rc_device)
        timing_dtype = getattr(timing_op, "dtype", None)
        delays_leaf = {
            k: v.detach()
            .to(device=timing_dev, dtype=timing_dtype or v.dtype)
            .requires_grad_(k in ("rise", "fall"))
            for k, v in raw_delays.items()
        }
        impulses_leaf = {
            k: v.detach()
            .to(device=timing_dev, dtype=timing_dtype or v.dtype)
            .requires_grad_(k in ("rise", "fall"))
            for k, v in raw_impulses.items()
        }
        loads_leaf = {
            k: v.detach()
            .to(device=timing_dev, dtype=timing_dtype or v.dtype)
            .requires_grad_(k in ("rise", "fall"))
            for k, v in raw_loads.items()
        }
        clk_aat_leaf = None
        clk_tran_arg = None
        clk_mask_arg = None
        if (
            clk_aat_compact is not None
            and self.clock_star is not None
            and self.clock_weight_scale != 0.0
        ):
            clk_aat_leaf = (
                clk_aat_compact.detach()
                .to(device=timing_dev, dtype=timing_dtype or clk_aat_compact.dtype)
                .requires_grad_(True)
            )
            clk_tran_arg = clk_tran_compact.detach().to(
                device=timing_dev, dtype=timing_dtype or clk_tran_compact.dtype
            )
            clk_mask_arg = (
                None
                if clk_leaf_mask is None
                else clk_leaf_mask.detach().to(device=timing_dev)
            )

        dynamic_net_provider = None
        dynamic_net_arc_inputs = None
        live_geometry_forward_id = None
        if relaxed_buffering_active:
            integration_mode = self.model._relaxed_buffer_timing_integration_mode()
            if integration_mode == "dynamic_net_provider":
                _, new_x, new_y = self.model._timing_geometry_for_pos(pos.detach())
                elmore_dev = new_x.device
                dynamic_net_provider = (
                    self.model._build_relaxed_buffer_dynamic_net_provider()
                )
                live_geometry_forward_id = self.model._bind_segment_live_geometry(
                    dynamic_net_provider,
                    new_x=new_x,
                    new_y=new_y,
                    pin_capacitance_by_sense={
                        "base": full_caps[0].to(elmore_dev),
                        "rise": full_caps[1].to(elmore_dev),
                        "fall": full_caps[2].to(elmore_dev),
                    },
                )
            elif integration_mode == "precomputed_overlay":
                dynamic_net_arc_inputs = (
                    self.model._build_relaxed_buffer_dynamic_net_arc_inputs(delays_leaf)
                )

        surrogate_mode = self._effective_surrogate_mode()
        self.model._active_timing_surrogate_mode = surrogate_mode
        self.model._active_discrete_action_context = None
        timing_op._active_discrete_action_context = None
        sparse_runtime = getattr(timing_op, "_sparse_runtime", None)
        if sparse_runtime is not None:
            sparse_runtime.timing._active_discrete_action_context = None
        prev_log_top = getattr(timing_op, "log_top_violations", True)
        prev_log_stage = getattr(timing_op, "log_stage_progress", True)
        prev_log_mismatch = getattr(timing_op, "log_violation_shape_mismatch", True)
        prev_req_slack = getattr(timing_op, "_request_pin_slack_snapshot", False)
        want_pin_slack = bool(
            self.restart_history_mode
            in ("slack_beta", "delay_sigmoid", "selective_sigmoid")
            or self.endpoint_slack_tau_mode in ("drift", "post_restart_drift")
            or self.satisfied_release_mode != "off"
            or (
                self.macro_exclusion_field is not None
                and self.macro_exclusion_field.has_obstacles
                and self.macro_exclusion_field.slack_gate_mode != "uniform"
            )
        )
        timing_op.log_top_violations = False
        timing_op.log_stage_progress = False
        timing_op.log_violation_shape_mismatch = False
        timing_op._request_pin_slack_snapshot = want_pin_slack

        timing_kwargs = {
            "surrogate_mode": surrogate_mode,
            "dynamic_net_arc_inputs": dynamic_net_arc_inputs,
            "clk_pin_aat": clk_aat_leaf,
            "clk_pin_tran": clk_tran_arg,
            "clk_pin_tran_mask": clk_mask_arg,
        }
        if getattr(timing_op, "hold_rc", None) is not None:
            _, new_x, new_y = self.model._timing_geometry_for_pos(pos.detach())
            timing_kwargs["early_rc_inputs"] = timing_op.hold_rc.evaluate(
                self.model, new_x, new_y, full_caps[0].to(new_x.device)
            )
        if dynamic_net_provider is not None:
            timing_kwargs["dynamic_net_provider"] = dynamic_net_provider
        ep_zero_tau_used = self._resolve_endpoint_zero_tau_ps()
        if self.smooth_extremum:
            timing_kwargs["smooth_extremum"] = True
            timing_kwargs["extremum_tau_ps"] = float(self.extremum_tau_ps)
            timing_kwargs["at_merge_temperature"] = self.at_merge_temperature
            if ep_zero_tau_used is not None:
                timing_kwargs["endpoint_zero_tau_ps"] = float(ep_zero_tau_used)

        at_merge_record = None
        if self.at_merge_temperature is not None:
            pred_steps = float(getattr(self.params, "timing_opt_step", 15) or 15)
            self.at_merge_temperature.set_schedule_context(
                iteration=int(iteration),
                predicted_next_steps=pred_steps,
            )
            self.at_merge_temperature.begin(int(self.refresh_count) + 1)
        try:
            with torch.enable_grad():
                wns, tns, ws, ts = timing_op(
                    delays_leaf, impulses_leaf, loads_leaf, **timing_kwargs
                )
                if self.loss_mode == "tns_only":
                    soft_loss = (
                        getattr(timing_op, "last_soft_tns_loss", None)
                        if self.smooth_extremum
                        else None
                    )
                    timing_loss = soft_loss if soft_loss is not None else -tns
                else:
                    raw_wns_c = getattr(self.params, "timing_wns_coeff", None)
                    raw_tns_c = getattr(self.params, "timing_tns_coeff", None)
                    wns_coeff = 0.01 if raw_wns_c is None else float(raw_wns_c)
                    tns_coeff = 0.0001 if raw_tns_c is None else float(raw_tns_c)
                    timing_loss = -(wns_coeff * wns + tns_coeff * tns)

                grad_inputs = [
                    delays_leaf["rise"],
                    delays_leaf["fall"],
                    loads_leaf["rise"],
                    loads_leaf["fall"],
                    impulses_leaf["rise"],
                    impulses_leaf["fall"],
                ]
                if clk_aat_leaf is not None:
                    grad_inputs.append(clk_aat_leaf)

                if torch.is_tensor(timing_loss) and timing_loss.requires_grad:
                    raw_grads = torch.autograd.grad(
                        timing_loss,
                        grad_inputs,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                else:
                    raw_grads = tuple(None for _ in grad_inputs)
            if self.at_merge_temperature is not None:
                at_merge_record = self.at_merge_temperature.finish()
        finally:
            if (
                self.at_merge_temperature is not None
                and self.at_merge_temperature.current is not None
            ):
                self.at_merge_temperature.current = None
            timing_op.log_top_violations = prev_log_top
            timing_op.log_stage_progress = prev_log_stage
            timing_op.log_violation_shape_mismatch = prev_log_mismatch
            timing_op._request_pin_slack_snapshot = prev_req_slack
            if live_geometry_forward_id is not None and dynamic_net_provider is not None:
                dynamic_net_provider.release_live_geometry(
                    forward_id=live_geometry_forward_id
                )

        # Record latest STA telemetry on model for NonLinearPlace / S5B1 visibility.
        self.model.wns = wns.detach()
        self.model.tns = tns.detach()
        self.model.ws = ws.detach() if torch.is_tensor(ws) else ws
        self.model.ts = ts.detach() if torch.is_tensor(ts) else ts

        cur_pin_slack: Optional[Dict[str, torch.Tensor]] = None
        if want_pin_slack:
            raw_rs = getattr(timing_op, "pin_rslack_live_snapshot", None)
            raw_fs = getattr(timing_op, "pin_fslack_live_snapshot", None)
            if (
                raw_rs is not None
                and raw_fs is not None
                and torch.is_tensor(raw_rs)
                and torch.is_tensor(raw_fs)
                and int(raw_rs.numel()) >= self.num_pins
                and int(raw_fs.numel()) >= self.num_pins
            ):
                cur_pin_slack = {
                    "rise": raw_rs.detach()[: self.num_pins].to(
                        device=rc_device, dtype=torch.float32
                    ),
                    "fall": raw_fs.detach()[: self.num_pins].to(
                        device=rc_device, dtype=torch.float32
                    ),
                }

        def _grad_or_zero(g, ref):
            if g is None:
                return torch.zeros(
                    (self.num_pins,), dtype=torch.float32, device=rc_device
                )
            return g.detach()[: self.num_pins].to(
                device=rc_device, dtype=torch.float32
            )

        fresh_gradients: Dict[str, Dict[str, torch.Tensor]] = {
            "delay": {
                "rise": _grad_or_zero(raw_grads[0], delays_leaf["rise"]),
                "fall": _grad_or_zero(raw_grads[1], delays_leaf["fall"]),
            },
            "cap": {
                "rise": _grad_or_zero(raw_grads[2], loads_leaf["rise"]),
                "fall": _grad_or_zero(raw_grads[3], loads_leaf["fall"]),
            },
            "impulse": {
                "rise": _grad_or_zero(raw_grads[4], impulses_leaf["rise"]),
                "fall": _grad_or_zero(raw_grads[5], impulses_leaf["fall"]),
            },
        }
        clock_gradients = None
        if clk_aat_leaf is not None:
            clk_g = raw_grads[6]
            if clk_g is None:
                clk_g = torch.zeros_like(clk_aat_leaf)
            clk_g = clk_g.detach().to(device=rc_device, dtype=torch.float32)
            clock_gradients = {
                "clk_pin_r_aat": clk_g[0],
                "clk_pin_f_aat": clk_g[1],
            }

        # 4. InputAccumulator history + optional post-restart drift-tau slack_beta retention.
        carry_beta = None
        beta_tau_used: Optional[float] = None
        slack_drift_tau_obs: Optional[float] = None
        need_beta_rescale = bool(self._pending_slack_beta_rescale or master_changed)
        if (
            need_beta_rescale
            and self.restart_history_mode == "slack_beta"
            and self.input_accumulator is not None
            and self.input_accumulator.values is not None
            and self._committed_pin_slack is not None
            and cur_pin_slack is not None
        ):
            beta_tau_used = self._resolve_beta_tau_ps()
            carry_beta = compute_slack_retention_beta(
                self._committed_pin_slack,
                cur_pin_slack,
                tau_ps=beta_tau_used,
            )
        elif (
            need_beta_rescale
            and self.restart_history_mode in ("delay_sigmoid", "selective_sigmoid")
            and self.input_accumulator is not None
            and self.input_accumulator.values is not None
            and cur_pin_slack is not None
        ):
            beta_tau_used = self._resolve_beta_tau_ps()
            mut_mask = None
            if self.restart_history_mode == "selective_sigmoid":
                if self._restored_mutated_cap_mask is not None:
                    mut_mask = self._restored_mutated_cap_mask
                elif (
                    cur_cap_sig is not None
                    and self._committed_pin_cap_sig is not None
                    and cur_cap_sig.shape == self._committed_pin_cap_sig.shape
                ):
                    mut_mask = (cur_cap_sig != self._committed_pin_cap_sig).any(dim=1).to(
                        device=rc_device
                    )
            carry_beta = compute_delay_sigmoid_retention_beta(
                cur_pin_slack,
                tau_ps=beta_tau_used,
                mutated_cap_pin_mask=mut_mask,
            )
            self._restored_mutated_cap_mask = None
        elif (
            not need_beta_rescale
            and self._committed_pin_slack is not None
            and cur_pin_slack is not None
        ):
            pin_weights = {}
            for edge in ("rise", "fall"):
                w_pin = fresh_gradients["delay"][edge].abs()
                if (
                    self.input_accumulator is not None
                    and self.input_accumulator.values is not None
                ):
                    w_pin = (
                        w_pin
                        + self.input_accumulator.values["delay"][edge].detach().abs()
                    )
                pin_weights[edge] = w_pin
            slack_drift_tau_obs, _ = estimate_slack_drift_tau(
                self._committed_pin_slack,
                cur_pin_slack,
                pin_weights=pin_weights,
                base_tau_ps=self.extremum_tau_ps,
            )

        pending_history = None
        beta_carry_ratio: Optional[float] = None
        effective_gradients = dict(fresh_gradients)
        if self.input_accumulator is not None:
            identity = (
                id(self.placedb),
                self.num_pins,
                int(self.star.pair_topology_generation),
            )
            if (
                self.input_accumulator.identity is not None
                and identity != self.input_accumulator.identity
                and carry_beta is None
            ):
                self._reset_input_accumulator()
            raw_iter = int(iteration)
            if self._last_raw_iter is None:
                self._history_virtual_iter = (
                    max(0, raw_iter)
                    if self.input_accumulator.iteration is None
                    else max(int(self.input_accumulator.iteration) + 1, raw_iter)
                )
            elif raw_iter > self._last_raw_iter:
                self._history_virtual_iter += raw_iter - self._last_raw_iter
            else:
                self._history_virtual_iter += 1
            history_candidate = self.input_accumulator.prepare(
                fresh_gradients,
                identity=identity,
                pin_count=self.num_pins,
                iteration=int(self._history_virtual_iter),
                carry_beta=carry_beta,
                allow_identity_change=bool(carry_beta is not None),
            )
            if self.satisfied_release_mode != "off" and cur_pin_slack is not None:
                min_pin_s = torch.minimum(
                    cur_pin_slack["rise"], cur_pin_slack["fall"]
                )
                merge_tau_cur = (
                    float(self.at_merge_temperature.tau)
                    if self.at_merge_temperature is not None
                    else float(self.extremum_tau_ps)
                )
                rel_rec = apply_satisfied_slack_release(
                    fresh_gradients,
                    history_candidate["values"],
                    min_pin_s,
                    mode=self.satisfied_release_mode,
                    threshold_ps=self.satisfied_release_threshold_ps,
                    band_ps=self.satisfied_release_band_ps,
                    tau_mult=self.satisfied_release_tau_mult,
                    merge_tau_ps=merge_tau_cur,
                    half_life_iters=self.satisfied_release_half_life,
                    elapsed_iters=float(history_candidate.get("elapsed") or 15.0),
                )
                if rel_rec is not None:
                    history_candidate["satisfied_release"] = rel_rec
            beta_carry_ratio = history_candidate.get("beta_carry_ratio")
            pending_history = (history_candidate, raw_iter)
            effective_gradients.update(history_candidate["values"])
            if carry_beta is not None and beta_carry_ratio is not None:
                if self._pending_recal_carry_ratio is None:
                    self._pending_recal_carry_ratio = float(beta_carry_ratio)
                else:
                    self._pending_recal_carry_ratio = float(
                        self._pending_recal_carry_ratio * float(beta_carry_ratio)
                    )
            elif self.restart_history_mode not in (
                "slack_beta",
                "delay_sigmoid",
                "selective_sigmoid",
            ):
                self._pending_recal_carry_ratio = None

        # 5. Project pair delay weight K and pair load-length weight B -> (B+, B_raw).
        raw_weight = pair_delay_adjoint_from_pin_grads(
            self.star.sink_pin_ids,
            effective_gradients["delay"]["rise"],
            effective_gradients["delay"]["fall"],
        )["pair"].reshape(-1)

        if self.load_sign == "zero":
            raw_b = torch.zeros_like(raw_weight)
        else:
            raw_b = self.star.load_length_adjoint(
                effective_gradients["cap"], self.data_rc_cal_state
            ).reshape(-1)
        pub_b, curv_b = clamp_load_length_weight(raw_b, mode=self.load_sign)

        # 6. Per-net FLUTE teacher homothety slope calibration s_n on K.
        slope_meta = None
        pub_weight = raw_weight
        calibrator = self._ensure_slope_calibrator()
        if calibrator is not None:
            pub_weight, slope_meta = calibrator.rescale_weights(
                pin_pos=pin_pos.detach(),
                base_caps_generic=real_caps["generic"],
                raw_weight=raw_weight,
                iteration=int(iteration),
                prebuilt_tree=shared_teacher_tree,
            )

        # Compute post-restart effective accumulation multiple m_eff in consistent
        # pair-force units (ps/um) relative to a single fresh refresh step.
        u_scale = (
            1.0
            if (
                self.input_accumulator is None
                or self.input_accumulator.mode == "replace"
            )
            else float(self.input_accumulator.gain)
        )
        if not self.recal_m_eff_scale_fix:
            fresh_delay_l1 = float(
                fresh_gradients["delay"]["rise"].detach().abs().sum().item()
                + fresh_gradients["delay"]["fall"].detach().abs().sum().item()
            )
            eff_delay_l1 = float(
                effective_gradients["delay"]["rise"].detach().abs().sum().item()
                + effective_gradients["delay"]["fall"].detach().abs().sum().item()
            )
            self._pending_recal_m_eff = (
                max(1.0, eff_delay_l1 / fresh_delay_l1)
                if fresh_delay_l1 > 1e-12
                else 1.0
            )
        else:
            with torch.no_grad():
                fresh_raw_k = (
                    pair_delay_adjoint_from_pin_grads(
                        self.star.sink_pin_ids,
                        fresh_gradients["delay"]["rise"],
                        fresh_gradients["delay"]["fall"],
                    )["pair"].reshape(-1)
                    * u_scale
                )
                if calibrator is not None and raw_weight.numel() > 0:
                    s_pair = torch.where(
                        raw_weight.abs() > 1e-30,
                        pub_weight / raw_weight,
                        torch.ones_like(raw_weight),
                    )
                    fresh_pub_k = fresh_raw_k * s_pair
                else:
                    fresh_pub_k = fresh_raw_k
                if self.load_sign == "zero":
                    fresh_pub_b = torch.zeros_like(fresh_pub_k)
                else:
                    fresh_cap = {
                        edge: fresh_gradients["cap"][edge] * u_scale
                        for edge in ("rise", "fall")
                    }
                    fresh_raw_b = self.star.load_length_adjoint(
                        fresh_cap, self.data_rc_cal_state
                    ).reshape(-1)
                    fresh_pub_b, _ = clamp_load_length_weight(
                        fresh_raw_b, mode=self.load_sign
                    )
                pair_len = length_um.reshape(-1)
                sink_cap_vec = real_caps["generic"][self.star.sink_pin_ids]
                rc_slope_unit = float(self.r_unit) * (
                    sink_cap_vec + float(self.c_unit) * pair_len
                )
                eff_force_l1 = float(
                    (pub_weight.detach().abs() * rc_slope_unit + pub_b.detach().abs())
                    .sum()
                    .item()
                )
                fresh_force_l1 = float(
                    (fresh_pub_k.detach().abs() * rc_slope_unit + fresh_pub_b.detach().abs())
                    .sum()
                    .item()
                )
                if fresh_force_l1 > 1e-12:
                    self._pending_recal_m_eff = max(1.0, eff_force_l1 / fresh_force_l1)
                else:
                    fresh_delay_scaled = float(
                        fresh_gradients["delay"]["rise"].detach().abs().sum().item()
                        + fresh_gradients["delay"]["fall"].detach().abs().sum().item()
                    ) * u_scale
                    eff_delay_l1 = float(
                        effective_gradients["delay"]["rise"].detach().abs().sum().item()
                        + effective_gradients["delay"]["fall"].detach().abs().sum().item()
                    )
                    self._pending_recal_m_eff = (
                        max(1.0, eff_delay_l1 / fresh_delay_scaled)
                        if fresh_delay_scaled > 1e-12
                        else 1.0
                    )

        # 7. Publish data Star snapshot & leaf-clock Star snapshot.
        candidate = self.star.prepare_snapshot(
            raw_weight=pub_weight,
            length_um=length_um.reshape(-1),
            load_length_weight=pub_b,
            curvature_load_length_weight=curv_b,
            metadata={
                "iteration": int(iteration),
                "overflow": overflow_val,
                "loss_mode": self.loss_mode,
                "load_sign": self.load_sign,
                "slope_calibration": slope_meta,
            },
        )
        star_meta = self.star.publish_snapshot(candidate)
        if pending_history is not None and self.input_accumulator is not None:
            history_candidate, raw_iter = pending_history
            self.input_accumulator.commit(history_candidate)
            self._last_raw_iter = raw_iter
        self._pending_slack_beta_rescale = False
        if slack_drift_tau_obs is not None:
            self.slack_drift_tau_history.append(float(slack_drift_tau_obs))
        if cur_pin_slack is not None:
            self._committed_pin_slack = cur_pin_slack
        if cur_master_digest is not None:
            self._committed_master_digest = cur_master_digest
        if cur_cap_sig is not None:
            self._committed_pin_cap_sig = cur_cap_sig

        clock_weight_l1 = 0.0
        if self.clock_star is not None:
            if clock_gradients is not None and self.clock_weight_scale != 0.0:
                clk_w, _clk_receipt = launch_capture_clock_pair_adjoint(
                    clock_gradients=clock_gradients,
                    pair_sink_pin_ids=self.clock_star.sink_pin_ids,
                    clock_row_to_compact_pin_ids=self.clock_star.clock_row_to_compact_pin_ids,
                )
                clk_w = (
                    clk_w.detach().to(
                        device=self.clock_star.sink_pin_ids.device,
                        dtype=torch.float32,
                    )
                    * float(self.clock_weight_scale)
                )
            else:
                clk_w = torch.zeros(
                    (int(self.clock_star.pair_count),),
                    dtype=torch.float32,
                    device=self.clock_star.sink_pin_ids.device,
                )
            self.clock_star.frozen_weight = clk_w
            self.clock_star.frozen_weight_seen = clk_w > 0.0
            self.clock_star.snapshot_generation = int(self.star.snapshot_generation)
            self.clock_star._timing_precondition_cache_key = None
            self.clock_star._timing_precondition_cache_x = None
            self.clock_star._timing_precondition_cache_y = None
            clock_weight_l1 = float(clk_w.abs().sum().item())

        # 8. Calibrate & regulate lambda_star against live movable timing/WL L1 norms.
        live_tns_val = float(self.model.tns.item())
        if self.fixed_lambda is None:
            self._regulate_lambda_star(
                pos.detach(),
                iteration=int(iteration),
                live_tns=live_tns_val,
                overflow=overflow_val,
            )

        self.geometry_dirty = False
        self._refreshes_since_restart += 1
        self.refresh_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        vjp_tau_used = (
            float(at_merge_record["tau_used_ps"])
            if at_merge_record is not None and at_merge_record.get("tau_used_ps") is not None
            else float(self.extremum_tau_ps)
        )
        vjp_tau_next = (
            float(at_merge_record["tau_next_ps"])
            if at_merge_record is not None and at_merge_record.get("tau_next_ps") is not None
            else float(self.extremum_tau_ps)
        )
        summary = {
            "iteration": int(iteration),
            "overflow": overflow_val,
            "snapshot_generation": int(star_meta["snapshot_generation"]),
            "wns": float(self.model.wns.item()),
            "tns": live_tns_val,
            "lambda_star": float(self.lambda_star),
            "realized_kappa": float(self.last_realized_kappa),
            "kappa_matched": bool(self.kappa_matched),
            "k_l1": float(pub_weight.detach().abs().sum().item()),
            "b_pub_l1": float(pub_b.detach().abs().sum().item()),
            "clock_k_l1": clock_weight_l1,
            "teacher_resampled": bool(need_teacher),
            "vjp_tau_used_ps": vjp_tau_used,
            "vjp_tau_next_ps": vjp_tau_next,
            "endpoint_zero_tau_ps": (
                float(self.extremum_tau_ps)
                if ep_zero_tau_used is None
                else float(ep_zero_tau_used)
            ),
            "slack_drift_tau_ps": slack_drift_tau_obs,
            "beta_tau_ps": beta_tau_used,
            "beta_carry_ratio": (
                float(beta_carry_ratio)
                if beta_carry_ratio is not None and carry_beta is not None
                else None
            ),
            "elapsed_ms": elapsed_ms,
        }
        self.last_refresh_summary = summary
        logging.info(
            "StarTdp refresh iter=%d gen=%d ovf=%s wns=%.4f tns=%.4f "
            "lambda=%.6g real_k=%.4f K_l1=%.4g B+_l1=%.4g clkK_l1=%.4g "
            "vjp_tau=%.4g->%.4g ep_tau=%.4g beta=%s teacher=%d %.2fms",
            int(iteration),
            int(star_meta["snapshot_generation"]),
            "None" if overflow_val is None else f"{overflow_val:.4f}",
            summary["wns"],
            summary["tns"],
            summary["lambda_star"],
            summary["realized_kappa"],
            summary["k_l1"],
            summary["b_pub_l1"],
            clock_weight_l1,
            vjp_tau_used,
            vjp_tau_next,
            summary["endpoint_zero_tau_ps"],
            (
                "off"
                if summary["beta_carry_ratio"] is None
                else f"{summary['beta_carry_ratio']:.3f}@tau={beta_tau_used:.4g}"
            ),
            int(need_teacher),
            elapsed_ms,
        )
        return summary

    def _unit_surrogate_term(self, pos: torch.Tensor) -> torch.Tensor:
        pin_pos = self.op_collections.pin_pos_op(pos)
        real_caps = self._driver_sink_star_base_cap_map(device=pin_pos.device)
        gamma_val = self._segment_timing_cut_effective_smoothing_gamma()
        data_gamma = (
            gamma_val
            if (
                self.distance_model in ("wa", "lse")
                or self.star.load_geometry == "wa_net"
            )
            else None
        )
        clk_gamma = (
            gamma_val if self.distance_model in ("wa", "lse") else None
        )
        total = self.star.structured_delay_objective(
            pin_pos,
            base_caps=real_caps,
            distance_model=self.distance_model,
            smoothing_gamma=data_gamma,
        )
        if (
            self.clock_star is not None
            and self.lcb_ff_clock_enabled
            and self.clock_weight_scale != 0.0
            and int(self.clock_star.pair_count) > 0
            and int(self.clock_star.frozen_weight.numel())
            == int(self.clock_star.pair_count)
        ):
            compact_pos = self._lcb_ff_clock_compact_pin_pos(pin_pos)
            compact_caps = self._lcb_ff_clock_base_cap_map(device=compact_pos.device)
            clk_term = self.clock_star.structured_delay_objective(
                compact_pos,
                base_caps=compact_caps,
                distance_model=self.distance_model,
                smoothing_gamma=clk_gamma,
            )
            total = total + clk_term
        return total

    def _regulate_lambda_star(
        self,
        pos_detached: torch.Tensor,
        *,
        iteration: int,
        live_tns: float,
        overflow: Optional[float] = None,
    ) -> None:
        movable = int(self.placedb.num_movable_nodes)
        with torch.enable_grad():
            probe_pos = pos_detached.detach().clone().requires_grad_(True)
            unit_term = self._unit_surrogate_term(probe_pos)
            if not torch.is_tensor(unit_term) or not unit_term.requires_grad:
                timing_l1 = 0.0
            else:
                grad_t = torch.autograd.grad(
                    unit_term, probe_pos, retain_graph=False, allow_unused=True
                )[0]
                timing_l1 = (
                    0.0 if grad_t is None else _movable_l1(grad_t, movable)
                )
            if timing_l1 <= 0.0:
                self.lambda_star = 0.0
                self.last_realized_kappa = 0.0
                if not self.kappa_matched:
                    self.kappa_deferred_count += 1
                    logging.info(
                        "StarTdp kappa calibration deferred at iter=%d (zero movable timing support)",
                        int(iteration),
                    )
                return

            wl_term = self.op_collections.wirelength_op(probe_pos)
            grad_wl = torch.autograd.grad(
                wl_term, probe_pos, retain_graph=False, allow_unused=True
            )[0]
            wl_l1 = 0.0 if grad_wl is None else _movable_l1(grad_wl, movable)

        if wl_l1 <= 0.0 or not math.isfinite(wl_l1 / timing_l1):
            self.lambda_star = 0.0
            self.last_realized_kappa = 0.0
            if not self.kappa_matched:
                self.kappa_deferred_count += 1
            return

        c_ratio = wl_l1 / timing_l1
        eff_max_kappa = (
            float(self.max_kappa)
            if self.max_kappa > 0.0
            else (float(self.post_restart_max_kappa) if self._has_restarted else 0.0)
        )
        if not self.kappa_matched:
            if live_tns >= 0.0 and (
                self._fallback_base_lambda_star > 0.0 or self.smooth_extremum
            ):
                self.kappa_deferred_count += 1
                if self._fallback_base_lambda_star > 0.0:
                    base_lam = float(self._fallback_base_lambda_star)
                    base_realized = base_lam * timing_l1 / wl_l1
                    lam = base_lam
                    if eff_max_kappa > 0.0 and base_realized > eff_max_kappa:
                        lam = float(eff_max_kappa * c_ratio)
                    self.lambda_star = float(lam)
                    self.last_realized_kappa = float(
                        self.lambda_star * timing_l1 / wl_l1
                    )
                    logging.info(
                        "StarTdp post-restart kappa recalibration deferred at iter=%d (live_tns=%.4f >= 0); "
                        "holding fallback base_lambda_star=%.6g (lambda=%.6g, real_k=%.4f)",
                        int(iteration),
                        float(live_tns),
                        base_lam,
                        self.lambda_star,
                        self.last_realized_kappa,
                    )
                else:
                    self.lambda_star = 0.0
                    self.last_realized_kappa = 0.0
                    logging.info(
                        "StarTdp kappa calibration deferred at iter=%d (live_tns=%.4f >= 0; smooth tail only)",
                        int(iteration),
                        float(live_tns),
                    )
                return

            target_kappa = float(self.requested_kappa)
            if (
                self._has_restarted
                and self._pending_recal_carry_ratio is not None
                and self._pending_recal_carry_ratio > 0.0
                and self._pre_restart_realized_kappa > 0.0
            ):
                carried_kappa = float(
                    self._pending_recal_carry_ratio * self._pre_restart_realized_kappa
                )
                target_kappa = min(
                    float(self.requested_kappa) + carried_kappa,
                    float(self.requested_kappa) * max(1.0, float(self._pending_recal_m_eff)),
                )
                if eff_max_kappa > 0.0:
                    target_kappa = min(target_kappa, eff_max_kappa)
            lam = target_kappa * c_ratio
            if not math.isfinite(lam) or lam <= 0.0:
                raise RuntimeError(f"invalid calibrated Star lambda: {lam}")
            self.initial_lambda_star = float(lam)
            self.base_lambda_star = float(lam)
            self._fallback_base_lambda_star = float(lam)
            self.lambda_star = float(lam)
            self.kappa_matched = True
            self._pending_recal_carry_ratio = None
            self.last_realized_kappa = float(
                self.lambda_star * timing_l1 / wl_l1
            )
            self.kappa_calibration_record = {
                "iteration": int(iteration),
                "snapshot_generation": int(self.snapshot_generation),
                "requested_kappa": float(self.requested_kappa),
                "target_kappa": float(target_kappa),
                "max_kappa": float(eff_max_kappa),
                "raw_wl_l1": float(wl_l1),
                "raw_unit_timing_l1": float(timing_l1),
                "c_wl_over_timing": float(c_ratio),
                "lambda_star": float(lam),
                "deferred_publications": int(self.kappa_deferred_count),
            }
            logging.info(
                "StarTdp calibrated lambda_star=%.6g at iter=%d (kappa=%.6g, target_kappa=%.6g, max_kappa=%.6g, ||dWL||_1=%.4g, ||dT_1||_1=%.4g, deferred=%d)",
                self.lambda_star,
                int(iteration),
                self.requested_kappa,
                target_kappa,
                eff_max_kappa,
                wl_l1,
                timing_l1,
                self.kappa_deferred_count,
            )
            return

        base_lam = float(self.base_lambda_star)
        if base_lam <= 0.0:
            base_lam = float(self.requested_kappa * c_ratio)
            self.base_lambda_star = base_lam
        self._fallback_base_lambda_star = float(base_lam)
        if self.initial_lambda_star <= 0.0:
            self.initial_lambda_star = base_lam
        base_realized = base_lam * timing_l1 / wl_l1

        lam = base_lam
        if eff_max_kappa > 0.0 and base_realized > eff_max_kappa:
            lam = float(eff_max_kappa * c_ratio)
        self.lambda_star = float(lam)
        self.last_realized_kappa = float(self.lambda_star * timing_l1 / wl_l1)

    # ------------------------------------------------------------------
    # GP objective & 2N MM curvature + Nesterov hooks
    # ------------------------------------------------------------------
    def objective(self, pos: torch.Tensor) -> Optional[torch.Tensor]:
        total: Optional[torch.Tensor] = None
        if self.is_objective_active:
            unit_val = self._unit_surrogate_term(pos)
            total = unit_val * float(self.lambda_star)
        if self.is_macro_field_active and self.macro_exclusion_field is not None:
            macro_val = self.macro_exclusion_field.objective(
                self, pos, self.current_overflow()
            )
            if float(getattr(self.macro_exclusion_field, "last_beta", 0.0)) > 0.0:
                total = macro_val if total is None else (total + macro_val)
        self.last_objective_value = None if total is None else total.detach()
        return total

    def effective_curvature_2n(
        self, *, fallback_pos: torch.Tensor, fallback_alpha: float = 1.0
    ) -> Optional[torch.Tensor]:
        """Return ``alpha_k * (H_timing + H_macro)`` of shape ``[2, num_nodes]`` when active."""
        if not self.curvature_precondition_enabled or not (
            self.is_objective_active or self.is_macro_field_active
        ):
            self.last_timing_curvature_2n = None
            return None
        ctx = self.step_context
        if ctx is not None:
            if ctx.get("hessian") is None:
                pos_eval = ctx["position"]
                with torch.no_grad():
                    if self.is_objective_active:
                        h_timing, gamma = rowsum_hessian(
                            self,
                            pos_eval,
                            elmore_axes=self.elmore_rowsum_axes,
                            full_slope=self.full_linear_slope,
                        )
                        if self.lcb_ff_clock_enabled and self.clock_star is not None:
                            h_timing = h_timing + clock_rowsum_hessian(
                                self,
                                pos_eval,
                                gamma=gamma,
                                elmore_axes=self.elmore_rowsum_axes,
                                full_slope=self.full_linear_slope,
                            )
                        h_timing = h_timing.detach()
                    else:
                        h_timing = pos_eval.new_zeros((2, int(self.placedb.num_nodes)))
                    h_total = h_timing
                    if self.is_macro_field_active and self.macro_exclusion_field is not None:
                        h_macro = self.macro_exclusion_field.diagonal_hessian(
                            self, pos_eval, self.current_overflow()
                        )
                        h_total = (h_timing + h_macro).detach()
                    ctx["timing_hessian"] = h_timing
                    ctx["hessian"] = h_total
            alpha = float(ctx.get("alpha", fallback_alpha))
            self.last_timing_curvature_2n = ctx["timing_hessian"] * alpha
            return ctx["hessian"] * alpha

        pos_eval = fallback_pos.detach()
        with torch.no_grad():
            if self.is_objective_active:
                h_timing, gamma = rowsum_hessian(
                    self,
                    pos_eval,
                    elmore_axes=self.elmore_rowsum_axes,
                    full_slope=self.full_linear_slope,
                )
                if self.lcb_ff_clock_enabled and self.clock_star is not None:
                    h_timing = h_timing + clock_rowsum_hessian(
                        self,
                        pos_eval,
                        gamma=gamma,
                        elmore_axes=self.elmore_rowsum_axes,
                        full_slope=self.full_linear_slope,
                    )
                h_timing = h_timing.detach()
            else:
                h_timing = pos_eval.new_zeros((2, int(self.placedb.num_nodes)))
            h_total = h_timing
            if self.is_macro_field_active and self.macro_exclusion_field is not None:
                h_macro = self.macro_exclusion_field.diagonal_hessian(
                    self, pos_eval, self.current_overflow()
                )
                h_total = (h_timing + h_macro).detach()
            alpha = float(fallback_alpha)
            self.last_timing_curvature_2n = h_timing * alpha
            return h_total * alpha

    def record_precondition_base_only(self, base_1n: torch.Tensor) -> None:
        if not (self.momentum_transport_timing or self.momentum_transport_envelope >= 1.0):
            return
        base_flat = torch.cat((base_1n, base_1n), dim=0).detach()
        if self.momentum_transport_envelope >= 1.0:
            self.transport_state["last"] = base_flat.clamp(min=1.0).clone()
        if self.momentum_transport_timing:
            self.transport_state["base"] = base_flat.clone()
            self.transport_state["timing"] = torch.zeros_like(base_flat)

    def record_precondition_diagonal(
        self,
        precond_2n: torch.Tensor,
        *,
        base_1n: Optional[torch.Tensor] = None,
        curv_2n: Optional[torch.Tensor] = None,
    ) -> None:
        if not (self.is_objective_active or self.is_macro_field_active):
            return
        if self.momentum_transport_envelope >= 1.0 or self.momentum_transport_timing:
            self.transport_state["last"] = precond_2n.reshape(-1).detach().clone()
        if self.momentum_transport_timing and base_1n is not None and curv_2n is not None:
            base_flat = torch.cat((base_1n, base_1n), dim=0).detach().clone()
            curv_flat = (
                curv_2n.to(device=base_flat.device, dtype=base_flat.dtype)
                .reshape(-1)
                .detach()
                .clone()
            )
            self.transport_state["base"] = base_flat
            self.transport_state["timing"] = curv_flat

    def begin_nesterov_step(self, position: torch.Tensor, alpha_start: float) -> None:
        clean_alpha = (
            float(alpha_start)
            if math.isfinite(alpha_start) and alpha_start > 0.0
            else 1.0
        )
        self._step_alpha_start = clean_alpha
        if not (self.is_objective_active or self.is_macro_field_active):
            self.step_context = None
            return
        self.step_context = {
            "position": position.detach().clone(),
            "alpha": clean_alpha,
            "timing_hessian": None,
            "hessian": None,
        }

    def end_nesterov_step(
        self, u_next: Optional[torch.Tensor], v_next: Optional[torch.Tensor]
    ) -> None:
        alpha_start = self._step_alpha_start
        self.step_context = None
        self._step_alpha_start = None
        if not (self.momentum_transport_timing or self.momentum_transport_envelope >= 1.0):
            self._reset_transport_state()
            return
        if (
            self.momentum_transport_timing
            and alpha_start is not None
            and self.transport_state.get("base") is not None
            and self.transport_state.get("timing") is not None
        ):
            self.transport_state["terms"] = (
                self.transport_state["base"],
                self.transport_state["timing"],
                float(alpha_start),
            )
        if self.is_objective_active or self.is_macro_field_active:
            if self.momentum_transport_envelope >= 1.0:
                last = self.transport_state.get("last")
                carried = self.transport_state.get("carried")
                if (
                    last is not None
                    and carried is not None
                    and u_next is not None
                    and v_next is not None
                    and carried.shape == last.shape
                    and last.numel() == u_next.numel()
                ):
                    with torch.no_grad():
                        transport_momentum(
                            u_next,
                            v_next,
                            carried,
                            last,
                            self.momentum_transport_envelope,
                        )
            if (
                self.momentum_transport_timing
                and self.transport_state.get("carried_terms") is not None
                and self.transport_state.get("terms") is not None
                and u_next is not None
                and v_next is not None
            ):
                (base_used, timing_used, alpha_used), (_, timing_next, alpha_next) = (
                    self.transport_state["carried_terms"],
                    self.transport_state["terms"],
                )
                if base_used.numel() == u_next.numel() and timing_next.numel() == u_next.numel():
                    with torch.no_grad():
                        factor = timing_transport_factor(
                            base_used,
                            timing_used,
                            alpha_used,
                            timing_next,
                            alpha_next,
                        )
                        if bool((factor < 1).any()):
                            u_next.copy_(
                                v_next - factor.to(u_next.dtype) * (v_next - u_next)
                            )
        self.transport_state["carried"] = self.transport_state.get("last")
        self.transport_state["carried_terms"] = self.transport_state.get("terms")
