import copy
import hashlib
import json
import logging
import math
import os
from enum import Enum
from types import SimpleNamespace

from dreamplace.flows.buffering_profiles import (
    BUFFERING_DEFAULTS,
    CANDIDATE_BUFFERING_DEFAULTS,
    SEGMENT_BUFFERING_DEFAULTS,
)
from dreamplace.flows.flow_profiles import (
    FlowKind,
    INFLATION_S5B1_DEFAULTS,
    JOINT_DEFAULTS as _JOINT_DEFAULTS,
    PLACEMENT_DEFAULTS as _PLACEMENT_DEFAULTS,
    PIN2PIN_TIMING_PLACEMENT_DEFAULTS as _PIN2PIN_TIMING_PLACEMENT_DEFAULTS,
    PROXIMAL_ALTERNATING_JOINT_DEFAULTS as _PROXIMAL_ALTERNATING_JOINT_DEFAULTS,
    SEGMENT_COUNT_DIRECT_JOINT_DEFAULTS as _SEGMENT_COUNT_DIRECT_JOINT_DEFAULTS,
    SEGMENT_JOINT_FIXED_WINDOW_SCHEDULE,
    SEGMENT_JOINT_OVERFLOW_MILESTONE_SCHEDULE,
    SIZING_DEFAULTS as _SIZING_DEFAULTS,
    STAGED_JOINT_SIZING_ITERATIONS,
    STAGED_JOINT_SIZING_STAGE_DEFAULTS,
    STAGED_JOINT_TDP_ITERATIONS,
    STAGED_JOINT_TDP_STAGE_DEFAULTS,
    STAGED_SMOKE_JOINT_DEFAULTS as _STAGED_SMOKE_JOINT_DEFAULTS,
    STA_DEFAULTS as _STA_DEFAULTS,
    TIMING_DRIVEN_PLACEMENT_DEFAULTS as _DIFF_TIMING_PLACEMENT_DEFAULTS,
)
from dreamplace.flows.timing_placement_config import configure_timing_placement_carrier
from dreamplace.flows.timing_opt_config import (
    is_legacy_timing_opt_name,
    normalize_timing_opt_params,
)


_BUFFERING_FLAGS = (
    "buffering_continuous_relaxed_optimization",
    "buffering_segment_count_tns_gradient",
)


def _param_enabled(params, name):
    return bool(getattr(params, name, False))


def _has_any_enabled(params, names):
    return any(_param_enabled(params, name) for name in names)


def infer_flow_kind(params):
    explicit = getattr(params, "flow_kind", None) or getattr(params, "flow", None)
    if explicit:
        return FlowKind(str(explicit))

    # A schema default describes the representation without selecting the flow.
    # Legacy programmatic callers supply the mode directly, without CLI markers.
    mode_selected = _param_enabled(params, "buffering_mode") and getattr(
        params, "_buffering_mode_explicit", True
    )
    if mode_selected or _has_any_enabled(params, _BUFFERING_FLAGS):
        return FlowKind.BUFFERING

    mode = getattr(params, "placement_sizing_mode", "place_only")
    if mode == "size_only":
        return FlowKind.SIZING
    if mode == "joint":
        return FlowKind.JOINT
    return FlowKind.PLACEMENT


def _set_default_if_missing(params, name, value):
    if not hasattr(params, name):
        setattr(params, name, value)


def _normalize_openroad_vt_suffixes(params):
    """Normalize the configured OpenROAD VT suffix list for bridge input."""
    raw = getattr(params, "openroad_vt_suffixes", ("H7H", "H7R", "H7L"))
    if isinstance(raw, str):
        raw = raw.replace(";", ",").split(",")
    try:
        values = [str(value).strip() for value in raw]
    except TypeError as exc:
        raise ValueError("openroad_vt_suffixes must be a list or comma-separated string") from exc
    values = list(dict.fromkeys(value for value in values if value))
    if not values:
        raise ValueError("openroad_vt_suffixes must contain at least one non-empty suffix")
    params.openroad_vt_suffixes = tuple(values)


def _resolve_size_parameterization_defaults(params):
    parameterization = str(
        getattr(params, "sizing_parameterization", "logits") or "logits"
    ).strip().lower()
    if parameterization not in {"logits", "real_size"}:
        raise ValueError(
            "unsupported sizing_parameterization: "
            f"{parameterization!r}; expected 'logits' or 'real_size'"
        )
    execution_mode = str(
        getattr(params, "real_size_execution_mode", "continuous_only")
        or "continuous_only"
    ).strip().lower()
    mode = str(
        getattr(params, "continuous_size_dynamics_mode", "none") or "none"
    ).strip().lower()
    if execution_mode not in {"continuous_only", "warmup_to_discrete"}:
        raise ValueError(
            "unsupported real_size_execution_mode: "
            f"{execution_mode!r}; expected 'continuous_only' or "
            "'warmup_to_discrete'"
        )
    try:
        learning_rate = float(getattr(params, "real_size_learning_rate", 0.1))
    except (TypeError, ValueError):
        raise ValueError("real_size_learning_rate must be a positive finite float")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("real_size_learning_rate must be a positive finite float")
    raw_warmup_steps = getattr(params, "real_size_warmup_steps", 0) or 0
    try:
        warmup_steps_float = float(raw_warmup_steps)
    except (TypeError, ValueError):
        raise ValueError("real_size_warmup_steps must be a nonnegative integer")
    if (
        not math.isfinite(warmup_steps_float)
        or not warmup_steps_float.is_integer()
        or warmup_steps_float < 0.0
    ):
        raise ValueError("real_size_warmup_steps must be a nonnegative integer")
    warmup_steps = int(warmup_steps_float)
    if parameterization == "real_size" and execution_mode == "warmup_to_discrete":
        if warmup_steps <= 0:
            raise ValueError(
                "real_size_warmup_steps must be positive for "
                "real_size_execution_mode='warmup_to_discrete'"
            )
    if parameterization == "real_size":
        # The first phase owns the continuous update. The legacy discrete
        # dynamics are enabled only by the explicit transition.
        if (
            (
                getattr(params, "_continuous_size_dynamics_mode_explicit", False)
                or getattr(
                    params,
                    "_continuous_size_dynamics_mode_json_declared",
                    False,
                )
            )
            and mode not in ("", "none")
        ):
            raise ValueError(
                "real_size parameterization requires "
                "continuous_size_dynamics_mode='none' before the explicit "
                "warm-up transition"
            )
        params.continuous_size_dynamics_mode = "none"
    params.sizing_parameterization = parameterization
    params.real_size_execution_mode = execution_mode
    params.real_size_learning_rate = learning_rate
    params.real_size_warmup_steps = warmup_steps


def _set_flow_default(params, name, value, *, force):
    if getattr(params, f"_{name}_explicit", False):
        return
    if name in {
        "buffering_mode",
        "buffering_segment_strategy",
        "buffering_candidate_strategy",
    } and hasattr(params, name):
        return
    if force or not hasattr(params, name):
        setattr(params, name, value)


def _pin2pin_timing_placement_enabled(params):
    return bool(
        getattr(params, "enable_net_weighting", False)
        and str(getattr(params, "net_weighting_scheme", "") or "").lower()
        == "pin2pin"
    )


def _apply_pin2pin_timing_placement_defaults(params, *, force):
    if not _pin2pin_timing_placement_enabled(params):
        return False

    if (
        getattr(params, "_differentiable_timing_obj_explicit", False)
        and bool(getattr(params, "differentiable_timing_obj", False))
    ):
        raise ValueError(
            "pin2pin timing placement is incompatible with "
            "differentiable_timing_obj=1; disable the differentiable timing objective"
        )
    if (
        getattr(params, "_diff_timing_driven_placement_explicit", False)
        and bool(getattr(params, "diff_timing_driven_placement", False))
    ):
        raise ValueError(
            "pin2pin timing placement uses legacy net weighting; "
            "set diff_timing_driven_placement=0"
        )

    for name, value in _PIN2PIN_TIMING_PLACEMENT_DEFAULTS:
        _set_flow_default(params, name, value, force=force)
    return True


def _resolve_timing_propagation_device_default(params):
    """Let timing-active flows follow the selected placement device by default."""
    if not _param_enabled(params, "with_sta"):
        return
    if getattr(params, "_timing_propagation_device_explicit", False):
        return
    params.timing_propagation_device = "inherit"


def _set_sizing_stage_iteration_default(params, *, force):
    stages = getattr(params, "global_place_stages", None)
    if not stages:
        params.global_place_stages = [{"iteration": 100, "optimizer": "adam"}]
        return
    first_stage = stages[0]
    if not isinstance(first_stage, dict):
        return
    if (
        not getattr(params, "_global_place_stages_explicit", False)
        and (force or "iteration" not in first_stage)
    ):
        first_stage["iteration"] = 100
    if force or "optimizer" not in first_stage:
        first_stage["optimizer"] = "adam"


def _set_joint_stage_optimizer_default(params):
    """Keep joint flows on an optimizer that can carry non-placement groups.

    Placement params files default to nesterov. A joint run may add a
    buffering or sizing parameter group, and nesterov rejects any group other
    than "placement". Switch the inherited optimizer before the run starts.
    """
    stages = getattr(params, "global_place_stages", None) or []
    sizing_enabled = bool(getattr(params, "joint_segment_sizing_enabled", True))
    no_sizing_segment_joint = (
        not sizing_enabled
        and _joint_quality_profile(params) == "segment_count_direct_joint_v1"
        and getattr(params, "placement_sizing_mode", "") == "place_only"
    )
    if no_sizing_segment_joint:
        return

    replaced = False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("optimizer", "")).lower() != "nesterov":
            continue
        stage["optimizer"] = "adam"
        replaced = True
    if replaced:
        logging.warning(
            "joint flow uses %s parameters; switched the global placement stage "
            "optimizer from nesterov to adam",
            "buffering/sizing" if sizing_enabled else "buffering without sizing",
        )


def _set_sta_stage_iteration_default(params, *, force):
    if getattr(params, "_global_place_stages_explicit", False):
        return
    stages = getattr(params, "global_place_stages", None)
    if not stages:
        params.global_place_stages = [{"iteration": 0}]
        return
    first_stage = stages[0]
    if not isinstance(first_stage, dict):
        return
    if force or "iteration" not in first_stage:
        first_stage["iteration"] = 0


def _buffering_mode(params):
    return str(getattr(params, "buffering_mode", "segment"))


def _joint_quality_profile(params):
    return str(getattr(params, "joint_quality_profile", "") or "")


def _mode_buffering_defaults(params):
    if _buffering_mode(params) == "candidate":
        return CANDIDATE_BUFFERING_DEFAULTS
    return SEGMENT_BUFFERING_DEFAULTS


def _buffering_segment_strategy(params):
    return str(
        getattr(params, "buffering_segment_strategy", "continuous") or "continuous"
    ).strip()


def _buffering_candidate_strategy(params):
    return str(
        getattr(params, "buffering_candidate_strategy", "continuous") or "continuous"
    ).strip()


def _set_discrete_route_default(params, name, value):
    if getattr(params, f"_{name}_explicit", False):
        current = getattr(params, name, None)
        if current != value:
            raise ValueError(
                f"discrete_net_gradient requires {name}={value!r}; "
                f"received explicit {current!r}"
            )
    setattr(params, name, value)


def _set_canonical_profile_value(params, name, value):
    if getattr(params, f"_{name}_explicit", False):
        current = getattr(params, name, None)
        if current != value:
            raise ValueError(
                f"canonical joint profile requires {name}={value!r}; "
                f"received explicit {current!r}"
            )
    setattr(params, name, value)


def _resolve_segment_direct_joint_schedule(params):
    if _joint_quality_profile(params) != "segment_count_direct_joint_v1":
        return

    window_steps = int(
        getattr(params, "joint_segment_virtual_window_steps", 0) or 0
    )
    warmup_steps = int(
        getattr(params, "joint_segment_virtual_warmup_steps", 0) or 0
    )
    outer_iterations = int(getattr(params, "joint_buffer_outer_iterations", 1) or 1)
    if window_steps > 0:
        schedule_mode = SEGMENT_JOINT_FIXED_WINDOW_SCHEDULE
    else:
        if warmup_steps != 0:
            raise ValueError(
                "segment_count_direct_joint_v1 requires zero virtual warmup "
                "when fixed-window scheduling is disabled"
            )
        if outer_iterations != 1:
            raise ValueError(
                "overflow_milestones requires joint_buffer_outer_iterations=1"
            )
        schedule_mode = SEGMENT_JOINT_OVERFLOW_MILESTONE_SCHEDULE
        milestones = tuple(
            float(value)
            for value in getattr(params, "joint_segment_milestones", ())
        )
        if not milestones or any(not math.isfinite(value) for value in milestones):
            raise ValueError("joint segment milestones must be finite and nonempty")
        if any(
            milestones[index] <= milestones[index + 1]
            for index in range(len(milestones) - 1)
        ):
            raise ValueError("joint segment milestones must be strictly descending")
        activation = float(
            getattr(params, "timing_topology_enable_overflow_threshold", 0.35)
        )
        if milestones[0] > activation:
            raise ValueError(
                "the first joint segment milestone must not exceed TDP activation"
            )
        sizing_enabled = bool(getattr(params, "joint_segment_sizing_enabled", True))
        if sizing_enabled:
            sizing_rounds = int(getattr(params, "joint_segment_sizing_rounds", 0))
            if sizing_rounds <= 0:
                raise ValueError("joint segment sizing rounds must be positive")
            sizing_up_percent = float(
                getattr(params, "joint_segment_sizing_up_percent", float("nan"))
            )
            if (
                not math.isfinite(sizing_up_percent)
                or not 0.0 < sizing_up_percent <= 100.0
            ):
                raise ValueError("joint segment sizing percentage must be in (0, 100]")
        buffering_rounds = int(
            getattr(params, "joint_segment_buffering_rounds", 0)
        )
        if buffering_rounds != 1:
            raise ValueError("overflow milestones require one Route-B round")
        buffering_fraction = float(
            getattr(
                params,
                "joint_segment_buffering_selection_fraction",
                float("nan"),
            )
        )
        if not math.isfinite(buffering_fraction) or not 0.0 < buffering_fraction <= 1.0:
            raise ValueError(
                "joint segment buffering selection fraction must be in (0, 1]"
            )
        params.joint_segment_milestones = milestones
    params.joint_segment_action_schedule_mode = schedule_mode


def _apply_discrete_net_gradient_defaults(params, flow_kind):
    mode = _buffering_mode(params)
    segment_strategy = _buffering_segment_strategy(params)
    candidate_strategy = _buffering_candidate_strategy(params)
    if segment_strategy not in {"continuous", "discrete_net_gradient"}:
        raise ValueError(f"unsupported buffering segment strategy: {segment_strategy}")
    if candidate_strategy not in {"continuous", "discrete_net_gradient"}:
        raise ValueError(f"unsupported buffering candidate strategy: {candidate_strategy}")
    if mode == "segment":
        if candidate_strategy != "continuous":
            raise ValueError(
                "buffering_candidate_strategy=discrete_net_gradient requires "
                "buffering_mode=candidate"
            )
        strategy = segment_strategy
    elif mode == "candidate":
        if segment_strategy != "continuous":
            raise ValueError("discrete_net_gradient requires buffering_mode=segment")
        strategy = candidate_strategy
    else:
        raise ValueError(f"unsupported buffering mode: {mode}")
    if strategy == "continuous":
        return
    canonical_joint = (
        flow_kind == FlowKind.JOINT
        and _joint_quality_profile(params) == "segment_count_direct_joint_v1"
    )
    inflation_placement = (
        flow_kind == FlowKind.PLACEMENT and _param_enabled(params, "timing_opt_enabled")
    )
    if flow_kind != FlowKind.BUFFERING and not canonical_joint and not inflation_placement:
        raise ValueError("discrete_net_gradient requires standalone flow_kind=buffering")
    if (
        getattr(params, "buffering_fixed_bsu_index", None) is None
        and not str(
            getattr(params, "buffering_fixed_buffer_master", "") or ""
        ).strip()
    ):
        raise ValueError(
            "discrete_net_gradient requires buffering_fixed_buffer_master "
            "or buffering_fixed_bsu_index"
        )
    if bool(getattr(params, "buffering_segment_capacity_enabled", False)):
        raise ValueError("discrete_net_gradient does not allow segment capacity control")
    if getattr(params, "buffering_max_selected_actions", None) is not None:
        raise ValueError("discrete_net_gradient does not allow max selected actions")
    if bool(
        getattr(params, "buffering_segment_projection_require_setup_criticality", False)
    ):
        raise ValueError("discrete_net_gradient does not allow criticality filtering")
    if mode == "candidate":
        if bool(getattr(params, "_buffering_continuous_lr_explicit", False)):
            raise ValueError(
                "candidate discrete_net_gradient does not use buffering_continuous_lr"
            )
        return
    _set_discrete_route_default(params, "buffering_segment_count_z_init", 0.0)
    _set_discrete_route_default(
        params,
        "buffering_segment_integer_projection_interval",
        0,
    )
    _set_discrete_route_default(
        params,
        "buffering_segment_integer_projection_start_step",
        0,
    )
    _set_discrete_route_default(
        params,
        "buffering_segment_integer_projection_project_bsu",
        0,
    )
    _set_discrete_route_default(
        params,
        "buffering_segment_projection_min_z_to_insert",
        0.5,
    )


def _normalize_effective_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _normalize_effective_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_effective_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolved_params_manifest(params):
    if not getattr(params, "_flow_defaults_resolved", False):
        raise ValueError("effective parameter manifest requires resolved Params")
    params_json = params.toJson() if hasattr(params, "toJson") else vars(params)
    resolved_params = {
        str(key): _normalize_effective_value(value)
        for key, value in sorted(params_json.items(), key=lambda item: str(item[0]))
        if not str(key).startswith("_") and not is_legacy_timing_opt_name(str(key))
    }
    environment_overrides = {
        key: value
        for key, value in sorted(os.environ.items())
        if (
            key.startswith(("AIMP_", "AUTODMP_", "DREAMPLACE_", "PLACE_IO_"))
            or key == "CUDA_VISIBLE_DEVICES"
        )
    }
    payload = {
        "schema_version": 1,
        "resolved_params": resolved_params,
        "environment_overrides": environment_overrides,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "content_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def apply_flow_defaults(params):
    normalize_timing_opt_params(params)
    explicit_flow = getattr(params, "flow_kind", None) or getattr(params, "flow", None)
    flow_kind = infer_flow_kind(params)
    if flow_kind == FlowKind.PHYSICAL_ECO:
        raise ValueError(
            "flow_kind 'physical_eco' is not supported by the canonical flow; "
            "use an explicit supported flow kind"
        )
    if flow_kind != FlowKind.PLACEMENT:
        if _param_enabled(params, "timing_opt_enabled") and not (
            getattr(params, "_timing_opt_enabled_explicit", False)
            or getattr(params, "_inflation_s5b1_enabled_explicit", False)
        ):
            # The ECC template is placement-first and enables timing windows
            # by default. A caller selecting sizing/STA/buffering must not
            # inherit that placement-only lane unless explicitly requested.
            params.timing_opt_enabled = 0
        for name in (
            "routability_opt_flag",
            "l_shape_routability_flag",
            "adjust_gpugr_area_flag",
            "gpugr_final_eval_flag",
            "enable_net_weighting",
            "pin2pin_net_weighting",
            "joint_segment_virtual_density_enabled",
        ):
            if not getattr(params, f"_{name}_explicit", False):
                setattr(params, name, 0)
    if flow_kind == FlowKind.PLACEMENT:
        configure_timing_placement_carrier(params)
        params.flow_kind = FlowKind.PLACEMENT.value
        force_defaults = explicit_flow is not None
        timing_driven_placement = _param_enabled(
            params, "diff_timing_driven_placement"
        )
        for name, value in _PLACEMENT_DEFAULTS:
            if name == "diff_timing_driven_placement" and timing_driven_placement:
                continue
            _set_flow_default(params, name, value, force=force_defaults)
        if timing_driven_placement:
            for name, value in _DIFF_TIMING_PLACEMENT_DEFAULTS:
                _set_flow_default(params, name, value, force=force_defaults)
        _apply_pin2pin_timing_placement_defaults(params, force=force_defaults)
    if flow_kind == FlowKind.STA:
        params.flow_kind = FlowKind.STA.value
        force_defaults = explicit_flow is not None
        for name, value in _STA_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        _set_sta_stage_iteration_default(params, force=force_defaults)
    if flow_kind == FlowKind.SIZING:
        params.flow_kind = FlowKind.SIZING.value
        force_defaults = explicit_flow is not None
        for name, value in _SIZING_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        _set_sizing_stage_iteration_default(params, force=force_defaults)
    if flow_kind == FlowKind.JOINT:
        params.flow_kind = FlowKind.JOINT.value
        force_defaults = explicit_flow is not None
        for name, value in BUFFERING_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        for name, value in SEGMENT_BUFFERING_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        for name, value in _JOINT_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        if _joint_quality_profile(params) == "proximal_alternating_v1":
            for name, value in _PROXIMAL_ALTERNATING_JOINT_DEFAULTS:
                _set_flow_default(params, name, value, force=True)
        if _joint_quality_profile(params) == "staged_smoke_v1":
            for name, value in _STAGED_SMOKE_JOINT_DEFAULTS:
                _set_flow_default(params, name, value, force=True)
        if _joint_quality_profile(params) == "segment_count_direct_joint_v1":
            for name, value in _SEGMENT_COUNT_DIRECT_JOINT_DEFAULTS:
                if name in {
                    "timing_topology_enable_overflow_threshold",
                    "net_weighting_update_interval",
                    "joint_segment_milestones",
                    "joint_segment_sizing_rounds",
                    "joint_segment_sizing_up_percent",
                    "joint_segment_buffering_rounds",
                    "joint_segment_buffering_selection_fraction",
                    "joint_segment_pin2pin_rebootstrap_enabled",
                } and getattr(params, f"_{name}_explicit", False):
                    continue
                if (
                    name == "buffering_commit_enabled"
                    and getattr(params, "_buffering_commit_enabled_explicit", False)
                ):
                    continue
                if (
                    name == "joint_segment_sizing_enabled"
                    and getattr(
                        params,
                        "_joint_segment_sizing_enabled_explicit",
                        False,
                    )
                ):
                    continue
                if (
                    name == "joint_segment_virtual_density_enabled"
                    and getattr(
                        params,
                        "_joint_segment_virtual_density_enabled_explicit",
                        False,
                    )
                ):
                    continue
                if (
                    name == "joint_segment_route_b_enabled"
                    and getattr(
                        params,
                        "_joint_segment_route_b_enabled_explicit",
                        False,
                    )
                ):
                    continue
                if (
                    name == "joint_segment_virtual_window_steps"
                    and getattr(
                        params,
                        "_joint_segment_virtual_window_steps_explicit",
                        False,
                    )
                ):
                    continue
                if (
                    name == "joint_segment_virtual_warmup_steps"
                    and getattr(
                        params,
                        "_joint_segment_virtual_warmup_steps_explicit",
                        False,
                    )
                ):
                    continue
                if (
                    name == "buffering_fixed_bsu_index"
                    and getattr(params, "_buffering_fixed_bsu_index_explicit", False)
                ):
                    continue
                if (
                    name == "buffering_fixed_buffer_master"
                    and getattr(
                        params,
                        "_buffering_fixed_buffer_master_explicit",
                        False,
                    )
                ):
                    continue
                _set_canonical_profile_value(params, name, value)
            _resolve_segment_direct_joint_schedule(params)
            if not bool(getattr(params, "joint_segment_sizing_enabled", True)):
                if getattr(params, "_placement_sizing_mode_explicit", False) and getattr(
                    params, "placement_sizing_mode", None
                ) != "place_only":
                    raise ValueError(
                        "joint no-sizing requires placement_sizing_mode='place_only'"
                    )
                # No-sizing must not create size/vt tensors or a sizing
                # optimizer group. The segment lane still owns buffering.
                params.placement_sizing_mode = "place_only"
        _set_joint_stage_optimizer_default(params)
        if (
            getattr(params, "_diff_timing_driven_placement_explicit", False)
            and not _param_enabled(params, "diff_timing_driven_placement")
        ):
            params.differentiable_timing_obj = 0
    if flow_kind == FlowKind.BUFFERING:
        params.flow_kind = FlowKind.BUFFERING.value
        force_defaults = explicit_flow is not None
        for name, value in BUFFERING_DEFAULTS:
            _set_flow_default(params, name, value, force=force_defaults)
        for name, value in _mode_buffering_defaults(params):
            _set_flow_default(params, name, value, force=force_defaults)
    if _param_enabled(params, "timing_opt_enabled"):
        if flow_kind != FlowKind.PLACEMENT:
            raise ValueError("timing optimization requires the placement flow")
        if bool(getattr(params, "l_shape_use_ggr_topology", False)):
            raise ValueError(
                "timing optimization retains buffered nets through the FLUTE topology provider"
            )
        for name, upper_bound in (
            ("timing_opt_max_windows", 5),
            ("timing_opt_sizing_rounds", 10),
        ):
            value = getattr(params, name, 5)
            if isinstance(value, bool) or int(value) != value or not 1 <= value <= upper_bound:
                raise ValueError(f"{name} must be an integer in [1, {upper_bound}]")
            setattr(params, name, int(value))
        window_defaults = dict(INFLATION_S5B1_DEFAULTS)
        if _pin2pin_timing_placement_enabled(params):
            window_defaults.update({
                name: value for name, value in _PIN2PIN_TIMING_PLACEMENT_DEFAULTS
                if name in {
                    "diff_timing_driven_placement", "differentiable_timing_obj",
                    "enable_net_weighting", "pin2pin_net_weighting",
                    "timing_placement_carrier", "net_weighting_scheme",
                }
            })
        for name, value in window_defaults.items():
            if name in {
                "placement_sizing_mode", "sizing_parameterization", "continuous_size_dynamics_mode",
                "buffering_mode", "buffering_segment_strategy", "buffering_segment_count_z_init",
                "buffering_segment_count_timing_backend", "buffering_segment_transfer_backend",
                "buffering_segment_live_geometry", "buffering_segment_integer_projection_interval",
                "buffering_segment_integer_projection_start_step", "buffering_segment_integer_projection_project_bsu",
                "buffering_discrete_count_proximal_lambda", "relaxed_buffer_timing_integration_mode",
                "joint_segment_virtual_density_enabled", "with_sta", "diff_timing_driven_placement",
                "l_shape_use_ggr_topology", "l_shape_capacity_al_enable",
                "differentiable_timing_obj", "timing_placement_carrier", "critical_endpoint_pruning_mode",
                "enable_net_weighting", "pin2pin_net_weighting", "enhanced_inflation_replay_best_round_flag",
            }:
                _set_discrete_route_default(params, name, value)
            else:
                _set_flow_default(params, name, value, force=True)
    _resolve_timing_propagation_device_default(params)
    if _param_enabled(params, "timing_opt_enabled"):
        params.timing_propagation_device = "cpu"
    _normalize_openroad_vt_suffixes(params)
    _resolve_size_parameterization_defaults(params)
    _apply_discrete_net_gradient_defaults(params, flow_kind)
    from dreamplace.flows.gr_sizing_config import configure_gr_sizing

    configure_gr_sizing(params)
    params._flow_defaults_resolved = True
    return flow_kind


def resolve_flow_config(config, *, explicit_keys=()):
    """Return a copied JSON-compatible config with canonical flow defaults."""
    if not isinstance(config, dict):
        raise TypeError("DreamPlace flow config must be a dictionary")
    params = SimpleNamespace(**copy.deepcopy(config))
    for name in explicit_keys:
        setattr(params, f"_{name}_explicit", True)
    apply_flow_defaults(params)
    return {
        key: copy.deepcopy(value)
        for key, value in vars(params).items()
        if not key.startswith("_") and not is_legacy_timing_opt_name(key)
    }


def resolved_flow_kind(params):
    if not getattr(params, "_flow_defaults_resolved", False):
        raise ValueError(
            "run_optimization_flow requires Params resolved by "
            "build_effective_params_from_args or apply_flow_defaults"
        )
    return FlowKind(str(getattr(params, "flow_kind", "")))
