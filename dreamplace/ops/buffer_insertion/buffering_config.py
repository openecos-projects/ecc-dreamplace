from dataclasses import dataclass
from typing import Any


BUFFERING_MODES = ("segment", "candidate")
SEGMENT_STRATEGIES = ("continuous", "discrete_net_gradient")
CANDIDATE_STRATEGIES = ("continuous", "discrete_net_gradient")

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _enabled(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _as_nonempty_string(value):
    if value is None:
        return ""
    value = str(value).strip()
    return value if value else ""


def _flow_kind_value(flow_kind):
    if flow_kind is None:
        return None
    return getattr(flow_kind, "value", str(flow_kind))


def _get_param(params: Any, name: str, default=None):
    return getattr(params, name, default)


@dataclass(frozen=True)
class BufferingConfig:
    mode: str
    segment_strategy: str
    candidate_strategy: str
    output_dir: str
    continuous_steps: int
    continuous_lr: float
    max_selected_actions: int | None
    max_repeaters_per_segment: int
    fixed_buffer_master: str = ""
    fixed_bsu_index: int | None = None
    discrete_count_proximal_lambda: float = 0.0
    route_b_selection_fraction: float = 0.001
    segment_integer_projection_interval: int = 0
    segment_integer_projection_start_step: int = 0
    segment_integer_projection_project_bsu: bool = False
    segment_integer_projection_reset_optimizer_state: bool = True
    segment_projection_min_z_to_insert: float = 0.5
    segment_projection_require_setup_criticality: bool = False
    # TODO(segment-capacity): Keep this experimental path disabled by default
    # until projected/committed capacity fidelity and multi-design QoR close.
    segment_capacity_enabled: bool = False
    segment_capacity_grid: str = "auto"
    commit_enabled: bool = False
    committed_def_path: str = ""
    post_commit_pysta_json: str = ""

    @property
    def active_strategy(self):
        return (
            self.segment_strategy
            if self.mode == "segment"
            else self.candidate_strategy
        )


def _default_output_dir(params):
    explicit = _as_nonempty_string(_get_param(params, "buffering_output_dir", ""))
    if explicit:
        return explicit
    result_dir = _as_nonempty_string(getattr(params, "result_dir", "results")) or "results"
    design_name = "unknown_design"
    if hasattr(params, "design_name"):
        design_attr = getattr(params, "design_name")
        design_name = design_attr() if callable(design_attr) else str(design_attr)
    elif hasattr(params, "base_design_name"):
        design_name = str(getattr(params, "base_design_name"))
    return f"{result_dir}/{design_name}/buffering_inner_loop"


def _resolve_mode(params):
    mode = _get_param(params, "buffering_mode", None)
    if mode is None:
        mode = "segment"
    mode = str(mode)
    if mode not in BUFFERING_MODES:
        raise ValueError(f"unsupported buffering mode: {mode}")
    return mode


def _resolve_segment_strategy(params):
    strategy = str(
        _get_param(params, "buffering_segment_strategy", "continuous") or "continuous"
    ).strip()
    if strategy not in SEGMENT_STRATEGIES:
        raise ValueError(f"unsupported buffering segment strategy: {strategy}")
    return strategy


def _resolve_candidate_strategy(params):
    strategy = str(
        _get_param(params, "buffering_candidate_strategy", "continuous")
        or "continuous"
    ).strip()
    if strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"unsupported buffering candidate strategy: {strategy}")
    return strategy


def _resolve_int(params, public_name, default):
    value = _get_param(params, public_name, None)
    if value is None:
        value = default
    return int(value)


def _resolve_float(params, public_name, default):
    value = _get_param(params, public_name, None)
    if value is None:
        value = default
    return float(value)


def _resolve_optional_int(params, name):
    value = _get_param(params, name, None)
    return None if value is None else int(value)


def build_buffering_config_from_params(params, flow_kind=None):
    inferred_flow = _flow_kind_value(flow_kind or getattr(params, "flow_kind", None))
    commit_enabled = bool(
        inferred_flow in ("buffering", "joint")
        and _enabled(_get_param(params, "buffering_commit_enabled", 0))
    )
    mode = _resolve_mode(params)
    segment_strategy = _resolve_segment_strategy(params)
    candidate_strategy = _resolve_candidate_strategy(params)
    if mode == "segment" and candidate_strategy != "continuous":
        raise ValueError(
            "buffering_candidate_strategy=discrete_net_gradient requires "
            "buffering_mode=candidate"
        )
    if mode == "candidate" and segment_strategy != "continuous":
        raise ValueError("discrete_net_gradient requires buffering_mode=segment")
    fixed_bsu_index = _resolve_optional_int(params, "buffering_fixed_bsu_index")
    if fixed_bsu_index is not None and fixed_bsu_index < 0:
        raise ValueError("buffering_fixed_bsu_index must be nonnegative")
    fixed_buffer_master = _as_nonempty_string(
        _get_param(params, "buffering_fixed_buffer_master", "")
    )
    discrete_count_proximal_lambda = _resolve_float(
        params,
        "buffering_discrete_count_proximal_lambda",
        0.0,
    )
    if discrete_count_proximal_lambda < 0.0:
        raise ValueError("buffering_discrete_count_proximal_lambda must be nonnegative")
    route_b_selection_fraction = _resolve_float(
        params,
        "buffering_route_b_selection_fraction",
        0.001,
    )
    if not 0.0 < route_b_selection_fraction <= 1.0:
        raise ValueError("buffering_route_b_selection_fraction must lie within (0, 1]")
    max_selected_actions = _resolve_optional_int(params, "buffering_max_selected_actions")
    max_repeaters_per_segment = _resolve_int(
        params,
        "buffering_max_repeaters_per_segment",
        _get_param(params, "joint_buffer_max_per_segment", 3),
    )
    if max_repeaters_per_segment <= 0:
        raise ValueError("buffering_max_repeaters_per_segment must be positive")
    continuous_steps = _resolve_int(
        params,
        "buffering_continuous_steps",
        120,
    )
    segment_capacity_enabled = _enabled(
        _get_param(params, "buffering_segment_capacity_enabled", 0)
    )
    segment_capacity_grid = str(
        _get_param(params, "buffering_segment_capacity_grid", "auto") or "auto"
    ).strip().lower()
    if segment_capacity_grid not in {"auto", "4", "8"}:
        raise ValueError(
            "buffering_segment_capacity_grid must be one of auto, 4, or 8"
        )
    if segment_capacity_enabled:
        if mode != "segment":
            raise ValueError("segment capacity control requires buffering_mode=segment")
        if segment_capacity_grid == "auto":
            raise ValueError(
                "segment capacity control requires an explicit capacity grid: 4 or 8"
            )
        if fixed_bsu_index is None and not fixed_buffer_master:
            raise ValueError(
                "segment capacity control requires buffering_fixed_buffer_master "
                "or buffering_fixed_bsu_index"
            )
        if max_selected_actions is not None:
            raise ValueError(
                "segment capacity control does not allow max selected actions"
            )
        if _enabled(
            _get_param(params, "buffering_segment_projection_require_setup_criticality", 0)
        ):
            raise ValueError(
                "segment capacity control does not allow criticality action filtering"
            )
    segment_integer_projection_interval = (
        0
        if segment_capacity_enabled
        else _resolve_int(
            params,
            "buffering_segment_integer_projection_interval",
            100,
        )
    )
    segment_integer_projection_start_step = _resolve_int(
        params,
        "buffering_segment_integer_projection_start_step",
        100,
    )
    segment_integer_projection_project_bsu = _enabled(
        _get_param(params, "buffering_segment_integer_projection_project_bsu", 0)
    )
    segment_integer_projection_reset_optimizer_state = _enabled(
        _get_param(
            params,
            "buffering_segment_integer_projection_reset_optimizer_state",
            1,
        )
    )
    segment_projection_min_z_to_insert = _resolve_float(
        params,
        "buffering_segment_projection_min_z_to_insert",
        0.5,
    )
    segment_projection_require_setup_criticality = _enabled(
        _get_param(
            params,
            "buffering_segment_projection_require_setup_criticality",
            0,
        )
    )
    active_strategy = (
        segment_strategy if mode == "segment" else candidate_strategy
    )
    if active_strategy == "discrete_net_gradient":
        if continuous_steps <= 0:
            raise ValueError("buffering_continuous_steps must be positive")
        canonical_joint = (
            inferred_flow == "joint"
            and str(getattr(params, "joint_quality_profile", "") or "")
            == "segment_count_direct_joint_v1"
        )
        inflation_placement = inferred_flow == "placement" and _enabled(
            _get_param(params, "timing_opt_enabled", False)
        )
        if inferred_flow != "buffering" and not canonical_joint and not inflation_placement:
            raise ValueError(
                "discrete_net_gradient requires standalone flow_kind=buffering"
            )
        if fixed_bsu_index is None and not fixed_buffer_master:
            raise ValueError(
                "discrete_net_gradient requires buffering_fixed_buffer_master "
                "or buffering_fixed_bsu_index"
            )
        if segment_capacity_enabled:
            raise ValueError(
                "discrete_net_gradient does not allow segment capacity control"
            )
        if max_selected_actions is not None:
            raise ValueError(
                "discrete_net_gradient does not allow max selected actions"
            )
        if mode == "candidate":
            pass
        elif segment_integer_projection_interval != 0:
            raise ValueError(
                "discrete_net_gradient requires periodic integer projection disabled"
            )
        if mode == "segment" and segment_integer_projection_start_step != 0:
            raise ValueError(
                "discrete_net_gradient requires integer projection start step 0"
            )
        if mode == "segment" and segment_integer_projection_project_bsu:
            raise ValueError(
                "discrete_net_gradient does not allow bsu integer projection"
            )
        if segment_projection_require_setup_criticality:
            raise ValueError(
                "discrete_net_gradient does not allow criticality filtering"
            )
        if mode == "candidate" and bool(
            _get_param(params, "_buffering_continuous_lr_explicit", False)
        ):
            raise ValueError(
                "candidate discrete_net_gradient does not use buffering_continuous_lr"
            )
        if (
            mode == "segment"
            and abs(segment_projection_min_z_to_insert - 0.5) > 1.0e-12
        ):
            raise ValueError(
                "discrete_net_gradient requires min_z_to_insert=0.5"
            )
        if mode == "segment":
            initial_z = _resolve_float(params, "buffering_segment_count_z_init", 0.0)
            if abs(initial_z) > 1.0e-12:
                raise ValueError("discrete_net_gradient requires z_init=0")
    return BufferingConfig(
        mode=mode,
        segment_strategy=segment_strategy,
        candidate_strategy=candidate_strategy,
        output_dir=_default_output_dir(params),
        continuous_steps=continuous_steps,
        continuous_lr=_resolve_float(
            params,
            "buffering_continuous_lr",
            0.01,
        ),
        max_selected_actions=max_selected_actions,
        max_repeaters_per_segment=max_repeaters_per_segment,
        fixed_buffer_master=fixed_buffer_master,
        fixed_bsu_index=fixed_bsu_index,
        discrete_count_proximal_lambda=discrete_count_proximal_lambda,
        route_b_selection_fraction=route_b_selection_fraction,
        segment_integer_projection_interval=segment_integer_projection_interval,
        segment_integer_projection_start_step=segment_integer_projection_start_step,
        segment_integer_projection_project_bsu=segment_integer_projection_project_bsu,
        segment_integer_projection_reset_optimizer_state=(
            segment_integer_projection_reset_optimizer_state
        ),
        segment_projection_min_z_to_insert=segment_projection_min_z_to_insert,
        segment_projection_require_setup_criticality=(
            segment_projection_require_setup_criticality
        ),
        segment_capacity_enabled=segment_capacity_enabled,
        segment_capacity_grid=segment_capacity_grid,
        commit_enabled=commit_enabled,
        committed_def_path=_as_nonempty_string(
            _get_param(params, "buffering_committed_def_path", "")
        ),
        post_commit_pysta_json=_as_nonempty_string(
            _get_param(params, "buffering_post_commit_pysta_json", "")
        ),
    )
