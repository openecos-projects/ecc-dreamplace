"""Configuration contract for the standalone ``diff_opt`` flow.

The ECC step owns the public mode/flow fields, while this module keeps the
DreamPlace side independent from chipcompiler.  A flow tuple always means one
discrete action phase; it never means a GP iteration or an optimizer step.
"""


from dataclasses import dataclass
import math
import copy
from dreamplace.ops.gpugr.xplace_backend import normalize_gpugr_backend
from collections.abc import Mapping, Sequence


__all__ = [
    "DIFF_OPT_ACTIONS",
    "DIFF_OPT_COEFFICIENT_NAMES",
    "DIFF_OPT_DEFAULT_COEFFICIENTS",
    "DIFF_OPT_MODES",
    "DIFF_OPT_SCHEMA_VERSION",
    "HOLD_COEFFICIENT_NAMES",
    "DiffFlowSpec",
    "TimingCoefficients",
    "canonical_diff_opt_flows",
    "parse_diff_opt_flows",
    "DIFF_SIZING_ACTION_MASKS",
    "DIFF_SIZING_ACTIONS",
    "DiffSizingFlowSpec",
    "canonical_diff_sizing_flows",
    "parse_diff_sizing_flows",
]


DIFF_OPT_SCHEMA_VERSION = 1
DIFF_OPT_MODES = frozenset(("sizing", "buffering", "joint"))
DIFF_OPT_ACTIONS = frozenset(("sz", "bf"))
DIFF_OPT_COEFFICIENT_NAMES = ("wns", "tns", "slew", "cap")
HOLD_COEFFICIENT_NAMES = ("hold_wns", "hold_tns")
DIFF_OPT_DEFAULT_COEFFICIENTS = {
    "wns": 500.0,
    "tns": 5.0,
    "slew": 1.0,
    "cap": 1.0,
}


@dataclass(frozen=True)
class TimingCoefficients:
    wns: float
    tns: float
    slew: float
    cap: float
    hold_wns: float = 0.0
    hold_tns: float = 0.0

    @classmethod
    def from_value(cls, value, *, field="coefficients"):
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError(f"{field} must contain [wns, tns, slew, cap]")
            value = dict(zip(DIFF_OPT_COEFFICIENT_NAMES, value, strict=True))
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a mapping or [wns, tns, slew, cap] list")
        unknown = set(value) - {*DIFF_OPT_COEFFICIENT_NAMES, *HOLD_COEFFICIENT_NAMES}
        if unknown:
            raise ValueError(f"{field} contains unknown keys: {sorted(unknown)}")
        result = {}
        for name in (*DIFF_OPT_COEFFICIENT_NAMES, *HOLD_COEFFICIENT_NAMES):
            raw = value.get(name, DIFF_OPT_DEFAULT_COEFFICIENTS.get(name, 0.0))
            if isinstance(raw, bool):
                raise ValueError(f"{field}.{name} must be a finite non-negative number")
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field}.{name} must be a finite non-negative number") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{field}.{name} must be a finite non-negative number")
            result[name] = number
        if not any(result.values()):
            raise ValueError(f"{field} must contain at least one positive coefficient")
        return cls(**result)

    def to_dict(self):
        values = {
            name: float(getattr(self, name)) for name in DIFF_OPT_COEFFICIENT_NAMES
        }
        values.update(
            {
                name: float(getattr(self, name))
                for name in HOLD_COEFFICIENT_NAMES
                if getattr(self, name)
            }
        )
        return values


@dataclass(frozen=True)
class DiffFlowSpec:
    action: str
    steps: int
    coefficients: TimingCoefficients

    def __post_init__(self):
        action = str(self.action).strip().lower()
        if action not in DIFF_OPT_ACTIONS:
            raise ValueError(f"unsupported diff_opt action: {self.action!r}")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ValueError("diff_opt steps must be a positive integer")
        if self.steps <= 0:
            raise ValueError("diff_opt steps must be a positive integer")
        object.__setattr__(self, "action", action)

    def to_list(self):
        return [self.action, int(self.steps), self.coefficients.to_dict()]


def _default_flows(mode):
    coefficients = TimingCoefficients.from_value(DIFF_OPT_DEFAULT_COEFFICIENTS)
    defaults = {
        "sizing": (("sz", 50),),
        "buffering": (("bf", 10),),
        "joint": (("sz", 5), ("bf", 1)),
    }
    return tuple(DiffFlowSpec(action, steps, coefficients) for action, steps in defaults[mode])


def parse_diff_opt_flows(mode="buffering", flows=None):
    """Parse and validate the canonical diff_opt mode and ordered flow list."""

    mode = str(mode or "buffering").strip().lower()
    if mode not in DIFF_OPT_MODES:
        raise ValueError(
            f"unsupported diff_optimization_mode: {mode!r}; "
            f"expected one of {sorted(DIFF_OPT_MODES)}"
        )
    if flows is None:
        parsed = _default_flows(mode)
    else:
        if not isinstance(flows, (list, tuple)) or not flows:
            raise ValueError("diff_optimization_flows must be a non-empty list")
        parsed = []
        for index, item in enumerate(flows):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    f"diff_optimization_flows[{index}] must be "
                    "[action, steps, [wns, tns, slew, cap]]"
                )
            action, steps, coefficients = item
            parsed.append(
                DiffFlowSpec(
                    str(action).strip().lower(),
                    steps,
                    TimingCoefficients.from_value(
                        coefficients,
                        field=f"diff_optimization_flows[{index}][2]",
                    ),
                )
            )
        parsed = tuple(parsed)

    allowed = {"sz"} if mode == "sizing" else {"bf"} if mode == "buffering" else DIFF_OPT_ACTIONS
    invalid = [spec.action for spec in parsed if spec.action not in allowed]
    if invalid:
        raise ValueError(
            f"diff_optimization_mode={mode!r} does not allow actions {sorted(set(invalid))}"
        )
    return mode, tuple(parsed)


def canonical_diff_opt_flows(mode="buffering", flows=None):
    mode, parsed = parse_diff_opt_flows(mode, flows)
    return mode, [spec.to_list() for spec in parsed]


DIFF_SIZING_ACTIONS = frozenset(("quad", "up", "down", "vt", "vt_fast", "vt_slow"))
DIFF_SIZING_ACTION_MASKS = {
    "quad": (True, True, True, True),
    "up": (True, False, False, True),
    "down": (False, True, True, False),
    "vt": (False, False, True, True),
    "vt_fast": (False, False, False, True),
    "vt_slow": (False, False, True, False),
}


@dataclass(frozen=True)
class DiffSizingFlowSpec:
    action: str
    rounds: int
    coefficients: TimingCoefficients
    selection_percent: float

    def __post_init__(self):
        action = str(self.action).strip().lower()
        if action not in DIFF_SIZING_ACTIONS:
            raise ValueError(
                f"unsupported diff_sizing action: {self.action!r}; "
                f"expected one of {sorted(DIFF_SIZING_ACTIONS)}"
            )
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise ValueError("diff_sizing phase rounds must be a positive integer")
        if self.rounds <= 0:
            raise ValueError("diff_sizing phase rounds must be a positive integer")
        if isinstance(self.selection_percent, bool):
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            )
        try:
            percent = float(self.selection_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            ) from exc
        if not math.isfinite(percent) or not 0.0 < percent <= 100.0:
            raise ValueError(
                "diff_sizing phase selection_percent must be in (0, 100]"
            )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "selection_percent", percent)

    @property
    def action_mask(self):
        return DIFF_SIZING_ACTION_MASKS[self.action]

    def to_list(self):
        return [
            self.action,
            int(self.rounds),
            self.coefficients.to_dict(),
            float(self.selection_percent),
        ]


def parse_diff_sizing_flows(value):
    """Parse ``[[action, rounds, coeffs, percent], ...]`` or return ``None``."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("diff_sizing_optimization_flows must be a non-empty list")
    parsed = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(
                f"diff_sizing_optimization_flows[{index}] must be "
                "[action, rounds, [wns, tns, slew, cap], selection_percent]"
            )
        action, rounds, coefficients, selection_percent = item
        parsed.append(
            DiffSizingFlowSpec(
                action=str(action),
                rounds=rounds,
                coefficients=TimingCoefficients.from_value(
                    coefficients,
                    field=f"diff_sizing_optimization_flows[{index}][2]",
                ),
                selection_percent=selection_percent,
            )
        )
    return tuple(parsed)


def canonical_diff_sizing_flows(value):
    parsed = parse_diff_sizing_flows(value)
    return None if parsed is None else [spec.to_list() for spec in parsed]


TIMING_OPT_COEFFICIENT_FIELDS = {
    "wns": "timing_wns_coeff",
    "tns": "timing_tns_coeff",
    "slew": "timing_slew_weight",
    "cap": "timing_cap_weight",
}

DEFAULT_TIMING_OPT_COEFFICIENTS = {
    "mode": "fixed", "wns": 500.0, "tns": 5.0, "cap": 1.0, "slew": 1.0,
}

TIMING_OPT_ALIASES = (
    ("timing_opt_enabled", "inflation_s5b1_enabled"),
    ("timing_opt_buffering_enabled", "inflation_s5b1_buffering_enabled"),
    ("timing_opt_max_windows", "inflation_s5b1_max_windows"),
    ("timing_opt_overflow_milestones", "inflation_s5b1_overflow_milestones"),
    ("timing_opt_sizing_rounds", "inflation_sizing_rounds"),
)

_DEFAULTS = {
    "timing_opt_enabled": 0,
    "timing_opt_buffering_enabled": 0,
    "timing_opt_max_windows": 5,
    "timing_opt_overflow_milestones": [],
    "timing_opt_sizing_rounds": 10,
}
_LEGACY_NAMES = {legacy for _, legacy in TIMING_OPT_ALIASES}


def _values_equal(left, right):
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return list(left) == list(right)
    return left == right


def normalize_timing_opt_coefficients(policy):
    """Validate a sizing-window policy; fixed mode uses outer timing weight 1."""
    if not isinstance(policy, dict):
        raise ValueError("timing_opt_coefficients must be an object")
    mode = policy.get("mode")
    if mode not in ("inherit", "fixed"):
        raise ValueError("timing_opt_coefficients.mode must be inherit or fixed")
    required = {"mode", *TIMING_OPT_COEFFICIENT_FIELDS}
    fields = required | set(HOLD_COEFFICIENT_NAMES)
    if set(policy) - fields or (mode == "fixed" and not required <= set(policy)):
        raise ValueError(
            "timing_opt_coefficients allows mode, wns, tns, slew, cap, hold_wns and hold_tns; "
            "fixed mode requires mode, wns, tns, slew and cap"
        )
    normalized = {"mode": mode}
    for name in (*TIMING_OPT_COEFFICIENT_FIELDS, *HOLD_COEFFICIENT_NAMES):
        if name not in policy:
            continue
        value = policy[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(
                f"timing_opt_coefficients.{name} must be finite and nonnegative"
            )
        normalized[name] = float(value)
    return normalized


def normalize_timing_opt_params(params):
    """Normalize old placement window fields to the ``timing_opt_*`` names.

    Existing profiles may still contain ``inflation_s5b1_*`` fields.  A
    canonical schema default may coexist with one of those legacy values; the
    explicit legacy value then replaces that default.  If both spellings are
    explicitly supplied with different values, fail before flow defaults can
    make a silent choice.
    """
    for canonical, legacy in TIMING_OPT_ALIASES:
        has_canonical = hasattr(params, canonical)
        has_legacy = hasattr(params, legacy)
        canonical_explicit = bool(
            getattr(params, f"_{canonical}_explicit", False)
        )
        legacy_explicit = bool(getattr(params, f"_{legacy}_explicit", False))
        default = _DEFAULTS[canonical]

        if has_canonical and has_legacy:
            canonical_value = getattr(params, canonical)
            legacy_value = getattr(params, legacy)
            if not _values_equal(canonical_value, legacy_value):
                if legacy_explicit and not canonical_explicit:
                    canonical_value = legacy_value
                elif canonical_explicit and not legacy_explicit:
                    pass
                elif not canonical_explicit and not legacy_explicit and _values_equal(
                    canonical_value, default
                ):
                    canonical_value = legacy_value
                else:
                    raise ValueError(
                        "conflicting timing optimization parameters: "
                        f"{canonical}={canonical_value!r} and "
                        f"{legacy}={legacy_value!r}"
                    )
        elif has_legacy:
            canonical_value = getattr(params, legacy)
        elif has_canonical:
            canonical_value = getattr(params, canonical)
        else:
            canonical_value = copy.deepcopy(default)

        setattr(params, canonical, canonical_value)
        if legacy_explicit:
            setattr(params, f"_{canonical}_explicit", True)
        # Consume input aliases once. A stale legacy value must not undo an
        # internal canonical update, such as disabling windows for legalization.
        if has_legacy:
            delattr(params, legacy)
        if hasattr(params, f"_{legacy}_explicit"):
            delattr(params, f"_{legacy}_explicit")
    params.timing_opt_coefficients = normalize_timing_opt_coefficients(
        getattr(params, "timing_opt_coefficients", DEFAULT_TIMING_OPT_COEFFICIENTS)
    )
    return params


def is_legacy_timing_opt_name(name):
    return name in _LEGACY_NAMES


def configure_timing_placement_carrier(params):
    carrier = getattr(params, "timing_placement_carrier", "star_pin_pair")
    selected = getattr(params, "_timing_placement_carrier_explicit", False)
    legacy_pin2pin = bool(getattr(params, "pin2pin_net_weighting", 0)) or (
        str(getattr(params, "net_weighting_scheme", "") or "").lower() == "pin2pin"
    )
    timing_active = (
        not legacy_pin2pin
        and bool(getattr(params, "with_sta", 0))
        and (
            bool(getattr(params, "differentiable_timing_obj", 0))
            or bool(getattr(params, "diff_timing_driven_placement", 0))
            or bool(getattr(params, "timing_opt_enabled", 0))
        )
    )
    if not (selected or carrier == "pin2pin" or (carrier == "star_pin_pair" and timing_active)):
        return
    if not getattr(params, "global_place_flag", True):
        return
    if carrier not in {"direct_loss", "star_pin_pair", "pin2pin", "gradient_net_weight"}:
        raise ValueError(
            "timing_placement_carrier must be direct_loss, star_pin_pair, pin2pin, or gradient_net_weight"
        )
    if carrier == "gradient_net_weight":
        return
    pin2pin = carrier == "pin2pin"
    controls = {
        "with_sta": 1,
        "timing_eval_flag": 1,
        "diff_timing_driven_placement": int(not pin2pin),
        "differentiable_timing_obj": int(not pin2pin),
        "enable_net_weighting": int(pin2pin),
        "pin2pin_net_weighting": int(pin2pin),
        "net_weighting_scheme": "pin2pin" if pin2pin else "lilith",
    }
    for name, value in controls.items():
        setattr(params, name, value)
        setattr(params, f"_{name}_explicit", True)
    if carrier == "star_pin_pair":
        star_defaults = {
            "star_distance_model": "lse",
            "star_load_geometry": "wa_net",
            "star_recal_m_eff_scale_fix": 1,
            "star_restart_history_mode": "selective_sigmoid",
            "star_restart_beta_tau_mode": "drift",
            "star_restart_beta_tau_scale": 1.0,
            "star_restart_beta_min_tau_ps": 1.0,
            "star_restart_beta_fixed_tau_ps": 40.0,
            "star_kappa_recalibrate_on_restart": 1,
            "star_post_restart_max_kappa": 0.20,
            "star_endpoint_slack_tau_mode": "post_restart_drift",
            "star_endpoint_slack_tau_scale": 1.0,
            "star_momentum_transport_envelope": 0.0,
            "star_momentum_transport_timing": (
                0
                if float(getattr(params, "star_momentum_transport_envelope", 0.0) or 0.0)
                >= 1.0
                else 1
            ),
            "macro_exclusion_field_flag": 1,
            "macro_exclusion_ov_start": 0.2,
            "macro_exclusion_ov_full": 0.085,
            "macro_exclusion_beta_max": 1.1,
            "macro_exclusion_bg_wl_scale": 0.0,
            "macro_exclusion_wl_gate_scale": 0.35,
            "macro_exclusion_warmup_iters": 110,
            "macro_exclusion_ramp_iters": 40,
            "macro_exclusion_collar_rows": 1.0,
            "macro_exclusion_distance_blend": 0.15,
            "macro_exclusion_profile_mode": "tent_rim",
            "macro_exclusion_ridge_smooth_rows": 3.0,
            "macro_exclusion_slack_gate_mode": "kappa_evac",
            "macro_exclusion_slack_margin_ps": 300.0,
            "macro_exclusion_slack_tau_ps": 100.0,
            "macro_exclusion_depth_ps_per_um": 2.5,
            "macro_exclusion_crit_wl_scale": 0.65,
            "macro_exclusion_ratchet_decay": 1.0,
            "macro_exclusion_kappa_slack": 1.0,
            "macro_exclusion_row_margin": 5.0,
            "macro_exclusion_force_mode": "taylor_kkt",
            "macro_exclusion_escape_slope_mode": "global",
            "macro_exclusion_gate_release_mode": "ratchet",
            "macro_exclusion_hyst_kappa_band": 0.75,
            "macro_exclusion_interior_evac_boost": 0.50,
            "macro_exclusion_bell_sigma_rows": 16.0,
            "macro_exclusion_bell_bias": 0.80,
            "star_input_history_reference_interval": 0.0,
            "star_satisfied_release_mode": "off",
            "star_satisfied_release_threshold_ps": 60.0,
            "star_satisfied_release_band_ps": 140.0,
            "star_satisfied_release_tau_mult": 2.0,
            "star_satisfied_release_half_life": 45.0,
            "timing_at_merge_vjp_tau_step_norm": 0,
            "timing_at_merge_vjp_tau_min_horizon_steps": 0.0,
            "timing_at_merge_vjp_tau_leg_floor_rows": 0.0,
            "d_metric_legalize_flag": 1,
            "d_metric_legalize_anchor_boost": 0.5,
            "d_metric_legalize_max_weight": 64.0,
        }
        for name, value in star_defaults.items():
            if not hasattr(params, name):
                setattr(params, name, value)


def configure_gr_sizing(params):
    mode = str(getattr(params, "timing_rc_mode", "placement"))
    if mode not in {"placement", "gr", "spef"}:
        raise ValueError("timing_rc_mode must be placement, gr, or spef")
    params.timing_rc_mode = mode
    terminal_gr_sta_enabled = getattr(params, "diff_sizing_terminal_gr_sta_enabled", 0)
    if terminal_gr_sta_enabled not in (0, 1, False, True):
        raise ValueError("diff_sizing_terminal_gr_sta_enabled must be 0 or 1")
    params.diff_sizing_terminal_gr_sta_enabled = int(bool(terminal_gr_sta_enabled))
    raw_scale = getattr(params, "spef_coupling_cap_scale", 1.0)
    if isinstance(raw_scale, bool):
        raise ValueError("spef_coupling_cap_scale must be a finite nonnegative float")
    try:
        coupling_scale = float(1.0 if raw_scale is None else raw_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("spef_coupling_cap_scale must be a finite nonnegative float") from exc
    if not math.isfinite(coupling_scale) or coupling_scale < 0.0:
        raise ValueError("spef_coupling_cap_scale must be a finite nonnegative float")
    params.spef_coupling_cap_scale = coupling_scale
    reconcile_lumped_cap = getattr(params, "spef_reconcile_lumped_cap", 0)
    if reconcile_lumped_cap not in (0, 1, False, True):
        raise ValueError("spef_reconcile_lumped_cap must be 0 or 1")
    params.spef_reconcile_lumped_cap = int(bool(reconcile_lumped_cap))
    params.spef_input = str(getattr(params, "spef_input", "") or "").strip()
    if mode == "placement":
        return
    if getattr(params, "place_io_engine", "ieda") != "ecc" or params.gpu:
        raise ValueError(f"{mode.upper()} timing requires the native ECC backend and gpu=0")
    if mode == "gr":
        backend = normalize_gpugr_backend(getattr(params, "gpugr_backend", "auto"))
        if backend == "cuda":
            raise ValueError(
                "GR timing requires a CPU routing backend: cpu_pr, cpu_pr_mt, or cpu_pr_maze"
            )
        # This production lane is CPU-qualified even when the host has CUDA.
        params.gpugr_backend = "cpu_pr_mt" if backend == "auto" else backend
        rrr_iters = int(getattr(params, "gr_sizing_rrr_iters", 0))
        if rrr_iters < 0:
            raise ValueError("gr_sizing_rrr_iters must be non-negative")
        if rrr_iters > 0 and params.gpugr_backend != "cpu_pr_maze":
            raise ValueError("positive gr_sizing_rrr_iters requires gpugr_backend=cpu_pr_maze")
    elif params.diff_sizing_terminal_gr_sta_enabled:
        raise ValueError("diff_sizing_terminal_gr_sta_enabled requires timing_rc_mode=gr")
    if params.flow_kind not in {"sizing", "sta"}:
        raise ValueError("GR timing supports standalone sizing or read-only STA only")
    if getattr(params, "placement_sizing_mode", "place_only") == "joint":
        raise ValueError("GR timing does not support joint placement/sizing")
    unsupported = (
        "timing_opt_enabled",
        "routability_opt_flag",
        "l_shape_routability_flag",
        "adjust_gpugr_area_flag",
        "gpugr_final_eval_flag",
        "enable_relaxed_buffer_timing",
        "buffering_continuous_relaxed_optimization",
        "buffering_segment_count_tns_gradient",
        "joint_segment_virtual_density_enabled",
        "enable_net_weighting",
        "pin2pin_net_weighting",
    )
    active = [name for name in unsupported if getattr(params, name, False)]
    if active:
        raise ValueError("GR timing does not support: " + ", ".join(active))
    params.with_sta = 1
    params.diff_timing_driven_placement = 0
    params.cell_model_schema = "main_id_arc_offset_piecewise_linear"
    params.piecewise_gradient_mode = "native_piecewise"
    if not getattr(params, "_timing_propagation_device_explicit", False):
        params.timing_propagation_device = "cpu"
    params.enable_fillers = 0
    params.gp_noise_ratio = 0.0
    params.random_center_init_flag = 0
    params.detailed_place_flag = 0
    params.detailed_place_engine = ""
    if params.flow_kind == "sta":
        params.differentiable_timing_obj = 0
        params.placement_sizing_mode = "place_only"
        params.sizing_parameterization = "logits"
        params.real_size_execution_mode = "continuous_only"
        params.real_size_warmup_steps = 0
        params.continuous_size_dynamics_mode = "none"
        params.global_place_flag = 0
        params.legalize_flag = 0
        params.timing_surrogate_mode = "lut_only"
        return
    params.placement_sizing_mode = "size_only"
    continuous_steps = getattr(params, "diff_sizing_continuous_steps", 0)
    if (
        isinstance(continuous_steps, bool)
        or not isinstance(continuous_steps, int)
        or not 0 <= continuous_steps <= 1000
    ):
        raise ValueError("diff_sizing_continuous_steps must be an integer in [0, 1000]")
    params.diff_sizing_continuous_steps = continuous_steps
    action_mode = str(
        getattr(params, "timing_action_gradient_mode", "lut_action")
        or "lut_action"
    )
    if action_mode not in {"off", "lut_action"}:
        raise ValueError(
            "timing_action_gradient_mode must be one of: off, lut_action"
        )
    # lut_action evaluates candidates directly with Liberty LUT + incremental STA;
    # only the legacy continuous gradient mode needs surrogate clock-to-Q arcs.
    params.timing_surrogate_mode = (
        "lut_only" if action_mode == "lut_action" else "surrogate_only"
    )
    if action_mode == "lut_action":
        if continuous_steps != 0:
            raise ValueError(
                "lut_action standalone diff_sizing requires "
                "diff_sizing_continuous_steps=0"
            )
        # The action path commits one legal master per selected instance.  It
        # owns real-size coordinates directly; logits would make the owner
        # write physical sizes into a logits tensor.
        params.sizing_parameterization = "real_size"
        params.real_size_execution_mode = "continuous_only"
        params.real_size_warmup_steps = 0
        params.continuous_size_dynamics_mode = "none"
    elif continuous_steps == 0:
        params.sizing_parameterization = "logits"
        params.real_size_execution_mode = "continuous_only"
        params.real_size_warmup_steps = 0
        params.continuous_size_dynamics_mode = "discrete_gradient_topk"
    else:
        params.sizing_parameterization = "real_size"
        params.real_size_execution_mode = "warmup_to_discrete"
        params.real_size_warmup_steps = continuous_steps
        params.continuous_size_dynamics_mode = "none"
    params.discrete_gradient_topk_shared_budget = 1
    params.discrete_gradient_topk_shared_budget_percent = 1.0
    params.discrete_gradient_topk_up_percent = 1.0
    params.discrete_gradient_topk_down_percent = 1.0
    params.discrete_gradient_topk_vt_percent = 1.0
    params.discrete_gradient_topk_preserve_vt = False
    params.timing_objective_lane = "timing_slew_cap"
    params.differentiable_timing_obj = 1
    params.early_stop_patience = 0
    params.early_stop_restore_best = True
    params.global_place_flag = 1
    params.legalize_flag = 1
    stage = dict(params.global_place_stages[0])
    stage.update(
        iteration=50, optimizer="adam", Llambda_density_weight_iteration=1, Lsub_iteration=1
    )
    params.global_place_stages = [stage]


def _nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _hold_coefficients(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object with wns and tns")
    unknown = set(value) - {"wns", "tns"}
    if unknown:
        raise ValueError(f"{name} only allows wns and tns")
    return {
        field: _nonnegative_number(value.get(field, 0.0), f"{name}.{field}")
        for field in ("wns", "tns")
    }


def _phase_hold_coefficients(value, name):
    if value is None:
        return {"hold_wns": 0.0, "hold_tns": 0.0}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {
        field: _nonnegative_number(value.get(field, 0.0), f"{name}.{field}")
        for field in ("hold_wns", "hold_tns")
    }


def _is_placement_flow(flow_kind):
    value = getattr(flow_kind, "value", flow_kind)
    return value == "placement"


def validate_hold_configuration(params, flow_kind):
    """Reject unsupported hold combinations before database construction.

    This intentionally validates only the public configuration contract. Schema
    and Liberty identity checks remain at the producer/consumer boundary where
    the actual database is available.
    """
    enabled = getattr(params, "timing_hold_enabled", False)
    if enabled not in (False, True, 0, 1):
        raise ValueError("timing_hold_enabled must be a boolean")
    enabled = bool(enabled)
    margin = _nonnegative_number(
        getattr(params, "timing_hold_margin_ns", 0.0), "timing_hold_margin_ns"
    )
    coefficients = _hold_coefficients(
        getattr(params, "timing_hold_coefficients", {"wns": 0.0, "tns": 0.0}),
        "timing_hold_coefficients",
    )
    phase_values = []
    for name in ("timing_opt_coefficients", "diff_sizing_coefficients"):
        policy = getattr(params, name, None)
        if policy is not None:
            phase_values.append(_phase_hold_coefficients(policy, name))
    active = any(coefficients.values()) or any(
        value for phase in phase_values for value in phase.values()
    )
    if active and not enabled:
        raise ValueError("nonzero hold coefficients require timing_hold_enabled=true")
    if enabled and not bool(getattr(params, "with_sta", False)):
        raise ValueError("timing_hold_enabled requires with_sta=1")
    paths = getattr(params, "timing_hold_lib_paths", [])
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise ValueError("timing_hold_lib_paths must be a list of Liberty paths")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("timing_hold_lib_paths must contain non-empty strings")
    if paths and not enabled:
        raise ValueError("timing_hold_lib_paths requires timing_hold_enabled=true")
    if paths:
        raise ValueError(
            "timing_hold_lib_paths requires the independent early Liberty corner "
            "adapter, which is not available in this migration checkpoint"
        )
    if (
        active
        and _is_placement_flow(flow_kind)
        and getattr(params, "global_place_flag", True)
        and getattr(params, "placement_sizing_mode", "place_only") != "size_only"
        and getattr(params, "timing_placement_carrier", "direct_loss") != "direct_loss"
    ):
        raise ValueError(
            "active placement hold optimization currently requires "
            "timing_placement_carrier=direct_loss"
        )
    params.timing_hold_margin_ns = margin
    params.timing_hold_coefficients = coefficients
    return params
