"""Canonical AutoDMP CLI parameter construction.

`dreamplace/Placer.py` is still the executable entry. This module owns the
shared CLI surface and effective-params construction for placement,
diff-sizing, buffering, and joint flows. The legacy `physical_eco` flow kind is
recognized only to report a stable unsupported-flow error. Legacy scripts under the
top-level `baseline/` directory are compatibility wrappers, not the place for
new canonical defaults.
"""

import argparse
import copy
import json
import math
import os

from dreamplace.Params import Params
from dreamplace.ops.openroad_handoff import launcher as handoff_launcher
from dreamplace.flows.flow_config import (
    FlowKind,
    apply_flow_defaults,
    resolved_params_manifest,
)

def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return value


def _nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative float")
    return value


def _percentage(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise argparse.ArgumentTypeError("expected a percentage in [0, 100]")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return value


def _density_float(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("expected a float in (0, 1]")
    return value


def _overflow_float(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("expected a float in [0, 1]")
    return value


def _add_buffering_public_args(parser):
    parser.add_argument("--buffering-output-dir")
    parser.add_argument("--buffering-mode", choices=("segment", "candidate"))
    parser.add_argument("--buffering-max-selected-actions", type=_nonnegative_int)
    parser.add_argument("--buffering-continuous-steps", type=_positive_int)
    parser.add_argument("--buffering-continuous-lr", type=float)
    parser.add_argument("--buffering-max-repeaters-per-segment", type=_positive_int)
    parser.add_argument(
        "--buffering-segment-strategy",
        choices=("continuous", "discrete_net_gradient"),
    )
    parser.add_argument(
        "--buffering-candidate-strategy",
        choices=("continuous", "discrete_net_gradient"),
    )
    parser.add_argument(
        "--buffering-segment-driver-cap-mode",
        choices=("residual", "direct"),
        help="Segment driver load model; direct is an experimental alternative to residual calibration.",
    )
    parser.add_argument("--buffering-segment-count-z-init", type=float)
    parser.add_argument(
        "--buffering-fixed-buffer-master",
        help=(
            "Exact legal buffer master or auto:x4; resolved to the family-local "
            "BSU index after loading Liberty metadata."
        ),
    )
    parser.add_argument("--buffering-fixed-bsu-index", type=_nonnegative_int)
    parser.add_argument("--buffering-discrete-count-proximal-lambda", type=float)
    parser.add_argument(
        "--buffering-route-b-selection-fraction",
        type=_density_float,
    )
    parser.add_argument(
        "--buffering-segment-capacity-enabled",
        type=_nonnegative_int,
        help=(
            "Enable experimental fixed-placement per-bin segment-buffer "
            "capacity control."
        ),
    )
    parser.add_argument(
        "--buffering-segment-capacity-grid",
        choices=("4", "8"),
        help="Capacity-grid coarsening: 4x4 or 8x8 base bins.",
    )
    parser.add_argument("--buffering-segment-count-timing-backend")
    parser.add_argument(
        "--buffering-segment-transfer-backend",
        choices=("static_size_table", "segment_transfer_python", "segment_transfer_native"),
    )
    parser.add_argument("--buffering-segment-integer-projection-interval", type=_nonnegative_int)
    parser.add_argument("--buffering-segment-integer-projection-start-step", type=_nonnegative_int)
    parser.add_argument(
        "--buffering-segment-integer-projection-project-bsu",
        type=_nonnegative_int,
    )
    parser.add_argument(
        "--buffering-segment-integer-projection-reset-optimizer-state",
        type=_nonnegative_int,
    )
    parser.add_argument("--buffering-commit-enabled", type=_nonnegative_int)
    parser.add_argument("--buffering-committed-def-path")
    parser.add_argument("--buffering-post-commit-pysta-json")
    parser.add_argument("--joint-buffer-commit-period", type=_nonnegative_int)
    parser.add_argument("--joint-buffer-outer-iterations", type=_positive_int)
    parser.add_argument("--joint-buffer-max-per-segment", type=_positive_int)
    parser.add_argument("--joint-segment-virtual-window-steps", type=_positive_int)
    parser.add_argument("--joint-segment-virtual-warmup-steps", type=_nonnegative_int)
    parser.add_argument(
        "--joint-segment-milestone",
        dest="joint_segment_milestones",
        action="append",
        type=_overflow_float,
    )
    parser.add_argument("--joint-segment-sizing-rounds", type=_positive_int)
    parser.add_argument("--joint-segment-sizing-up-percent", type=_percentage)
    parser.add_argument("--joint-segment-buffering-rounds", type=_positive_int)
    parser.add_argument(
        "--joint-segment-buffering-selection-fraction",
        type=_density_float,
    )
    parser.add_argument(
        "--joint-segment-pin2pin-rebootstrap-enabled",
        type=_nonnegative_int,
    )
    parser.add_argument(
        "--joint-segment-sizing-enabled",
        type=_nonnegative_int,
        help="Enable milestone-owned one-step discrete gate sizing.",
    )
    parser.add_argument(
        "--joint-segment-route-b-enabled",
        type=_nonnegative_int,
        help="Allow the canonical segment joint profile to mutate integer z at window boundaries.",
    )
    parser.add_argument(
        "--joint-segment-virtual-density-enabled",
        type=_nonnegative_int,
        help="Enable virtual buffer cells in the placement density objective.",
    )
    parser.add_argument(
        "--joint-post-commit-openroad-eco",
        choices=("none", "resynth_once"),
        help="Optional OpenROAD ECO pass after accepted buffer commit and before pydb refresh.",
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run DREAMPlace placement with optional OpenROAD handoff overrides."
    )
    parser.add_argument("params_json", nargs="?", help="DREAMPlace parameter JSON")
    parser.add_argument("--workspace", help="Workspace/run directory")
    parser.add_argument(
        "--flow-kind",
        choices=tuple(flow.value for flow in FlowKind),
        help="Recommended high-level flow entry, e.g. sizing for diff gate sizing.",
    )
    parser.add_argument(
        "--joint-quality-profile",
        choices=(
            "proximal_alternating_v1",
            "staged_smoke_v1",
            "segment_count_direct_joint_v1",
        ),
        help="Compact joint optimization profile.",
    )
    parser.add_argument("--joint-staged-smoke-outer-iterations", type=_positive_int)
    parser.add_argument("--result-dir")
    parser.add_argument("--base-design-name")
    parser.add_argument("--output-def", help="Final DEF path written after placement")
    parser.add_argument(
        "--output-verilog",
        help="Final Verilog path written after placement or buffering",
    )
    parser.add_argument("--place-io-engine", choices=handoff_launcher.PLACE_IO_ENGINES)
    parser.add_argument("--def-input")
    parser.add_argument("--verilog-input")
    parser.add_argument("--tech-lef", action="append", default=[])
    parser.add_argument("--lef", action="append", default=[])
    parser.add_argument("--lib", action="append", default=[])
    parser.add_argument("--sdc")
    parser.add_argument("--rc-tcl")
    parser.add_argument("--target-density", type=_density_float)
    parser.add_argument("--stop-overflow", type=_overflow_float)
    parser.add_argument("--gpu", type=_nonnegative_int)
    parser.add_argument("--gpu-id", type=_nonnegative_int)
    parser.add_argument(
        "--timing-propagation-device",
        choices=("inherit", "cpu", "cuda"),
        help="Execution device for timing propagation; inherit follows --gpu.",
    )
    parser.add_argument(
        "--surrogate-cache-root",
        help=(
            "Optional disk cache directory for the fitted cell surrogate "
            "coefficients. Unset by default: the fit costs seconds, so runs "
            "refit in memory and write nothing."
        ),
    )
    parser.add_argument("--enable-fillers", type=_nonnegative_int)
    parser.add_argument("--gp-noise-ratio", type=_nonnegative_float)
    parser.add_argument("--legalize", type=_nonnegative_int)
    parser.add_argument("--random-center-init", type=_nonnegative_int)
    parser.add_argument(
        "--placement-optimizer",
        help="Override global_place_stages[*].optimizer for placement.",
    )
    parser.add_argument(
        "--placement-initial-learning-rate-max",
        type=_positive_float,
        help=(
            "Optional upper bound for the estimated initial placement learning "
            "rate when restarting from an existing placement."
        ),
    )
    parser.add_argument(
        "--placement-overflow-tolerance",
        type=_nonnegative_float,
        help=(
            "Numerical tolerance added to stop_overflow when deciding whether "
            "an explicit external legalization handoff is allowed."
        ),
    )
    parser.add_argument("--init-loc-perc-x", type=float)
    parser.add_argument("--init-loc-perc-y", type=float)
    parser.add_argument("--plot-interval", type=_positive_int)
    parser.add_argument("--plot", type=_nonnegative_int)
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        help="Override global_place_stages[0].iteration.",
    )
    parser.add_argument(
        "--num-bins-x",
        type=_positive_int,
        help="Override the horizontal density-bin count for placement.",
    )
    parser.add_argument(
        "--num-bins-y",
        type=_positive_int,
        help="Override the vertical density-bin count for placement.",
    )
    parser.add_argument("--with-sta", action="store_true")
    parser.add_argument("--handoff-restart-policy")
    parser.add_argument("--handoff-mode", choices=handoff_launcher.HANDOFF_MODES)
    parser.add_argument("--handoff-trigger-period", type=_positive_int)
    parser.add_argument(
        "--max-handoff-overflow",
        type=handoff_launcher.optional_gate_float,
    )
    parser.add_argument("--setup-margin", type=float)
    parser.add_argument("--diff-timing-driven-placement", type=_nonnegative_int)
    parser.add_argument("--enable-net-weighting", type=_nonnegative_int)
    parser.add_argument(
        "--net-weighting-scheme",
        choices=("adams", "lilith", "pin2pin"),
    )
    parser.add_argument("--pin2pin-weight", type=_nonnegative_float)
    parser.add_argument("--pin2pin-min-weight", type=_nonnegative_float)
    parser.add_argument("--pin2pin-max-weight", type=_nonnegative_float)
    parser.add_argument("--pin2pin-accumulate-weight", type=_nonnegative_float)
    parser.add_argument(
        "--net-weighting-npaths",
        type=_percentage,
        help="Legacy net-selection percentage; 3.0 matches Efficient-TDP and 0 keeps all.",
    )
    parser.add_argument(
        "--net-weighting-nendpoints",
        type=_nonnegative_int,
        help="Maximum violating endpoint-transition states selected per Pin2Pin update; 0 keeps all.",
    )
    parser.add_argument("--net-weighting-update-interval", type=_positive_int)
    parser.add_argument("--timing-topology-refresh-interval", type=_nonnegative_int)
    parser.add_argument("--timing-topology-enable-overflow-threshold", type=float)
    parser.add_argument("--timing-wns-coeff", type=float)
    parser.add_argument("--timing-tns-coeff", type=float)
    parser.add_argument("--timing-grad-balance-target-ratio", type=_nonnegative_float)
    parser.add_argument(
        "--timing-placement-carrier",
        choices=("direct_loss", "pin2pin", "gradient_net_weight"),
    )
    parser.add_argument("--timing-gradient-net-weight-scale", type=float)
    parser.add_argument("--timing-gradient-net-weight-max", type=float)
    parser.add_argument(
        "--openroad-vt-suffixes",
        help="Comma-separated OpenROAD VT suffixes, for example H7H,H7R,H7L.",
    )
    parser.add_argument("--discrete-gradient-topk-up-percent", type=float)
    parser.add_argument("--discrete-gradient-topk-down-percent", type=float)
    parser.add_argument(
        "--discrete-gradient-topk-vt-percent",
        type=_percentage,
        help="Percentage of improving one-step VT moves committed per iteration.",
    )
    parser.add_argument(
        "--discrete-gradient-topk-shared-budget",
        type=int,
        choices=(0, 1),
        help="Select one best size/VT action per cell under a shared cell budget.",
    )
    parser.add_argument(
        "--discrete-gradient-topk-shared-budget-percent",
        type=_percentage,
        help="Percentage of sizeable cells updated each round in shared-budget mode (sizing default: 1).",
    )
    parser.add_argument(
        "--discrete-gradient-topk-oscillation-veto",
        type=int,
        choices=(0, 1),
        help="Freeze cells for two rounds after repeated committed master reversals (shared budget only).",
    )
    parser.add_argument(
        "--continuous-size-dynamics-mode",
        choices=(
            "none",
            "phased_trust",
            "late_adaptive_guard",
            "late_zero_vio_guard",
            "late_accept_reject",
            "late_commit_feedback",
            "discrete_gradient_topk",
        ),
        help="Explicitly select the post-Adam sizing dynamics policy.",
    )
    parser.add_argument(
        "--sizing-parameterization",
        choices=("logits", "real_size"),
        help="Continuous sizing coordinate used by the sizing optimizer.",
    )
    parser.add_argument("--real-size-learning-rate", type=_positive_float)
    parser.add_argument(
        "--real-size-execution-mode",
        choices=("continuous_only", "warmup_to_discrete"),
    )
    parser.add_argument("--real-size-warmup-steps", type=_nonnegative_int)
    parser.add_argument(
        "--real-size-transition-backend",
        choices=("reference", "vectorized", "native"),
    )
    parser.add_argument(
        "--real-size-transition-artifact-policy",
        choices=("summary_only", "full"),
    )
    parser.add_argument("--real-size-transition-profile", type=_nonnegative_int)
    parser.add_argument(
        "--size-interpolated-pin-native-op",
        choices=("auto", "on", "off"),
        help="Control native size-interpolated pin property op for sizing/STA flows.",
    )
    _add_buffering_public_args(parser)
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Print effective params JSON and exit before constructing PlacementEngine.",
    )
    return parser


def _load_params_json(path):
    params = Params()
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as stream:
            schema = json.load(stream)
        params.fromJson({key: value.get("default") for key, value in schema.items()})
    params._buffering_mode_explicit = False
    if path:
        with open(path, "r") as stream:
            config = json.load(stream)
        params.fromJson(config)
        for name in (
            "buffering_mode",
            "buffering_fixed_buffer_master",
            "buffering_fixed_bsu_index",
            "continuous_size_dynamics_mode",
            "sizing_parameterization",
            "real_size_learning_rate",
            "real_size_execution_mode",
            "real_size_warmup_steps",
            "net_weighting_nendpoints",
            "timing_opt_enabled",
            "timing_opt_buffering_enabled",
            "timing_opt_max_windows",
            "timing_opt_overflow_milestones",
            "timing_opt_sizing_rounds",
            "inflation_s5b1_enabled",
            "inflation_s5b1_buffering_enabled",
            "inflation_s5b1_max_windows",
            "inflation_s5b1_overflow_milestones",
            "inflation_sizing_rounds",
        ):
            if name in config:
                if name == "continuous_size_dynamics_mode":
                    # Keep placement-flow legacy override semantics intact.
                    # real-size validation inspects this declaration separately.
                    setattr(params, "_continuous_size_dynamics_mode_json_declared", True)
                else:
                    setattr(params, f"_{name}_explicit", True)
    _apply_legacy_param_defaults(params)
    return params


def _apply_legacy_param_defaults(params):
    # Existing placement code reads these fields, but they are absent from params.json.
    defaults = {
        "auto_adjust_bins": False,
        "differentiable_timing_obj": False,
        "max_net_weight": "inf",
        "momentum_decay_factor": 0.5,
        # Older timing-placement configs omit this Nesterov option. Match the
        # historical default used by params_timing.json.
        "use_bb": 0,
    }
    for key, value in defaults.items():
        if key not in params.__dict__:
            setattr(params, key, value)


def _existing_design_inputs(params):
    existing = copy.deepcopy(getattr(params, "design_inputs", {}) or {})
    existing.setdefault("tech_lef", [])
    existing.setdefault("lef", [])
    existing.setdefault("lib", [])
    return existing


def _apply_design_input_overrides(params, args):
    design_inputs = _existing_design_inputs(params)
    changed = False
    if args.tech_lef:
        design_inputs["tech_lef"] = list(args.tech_lef)
        changed = True
    if args.lef:
        design_inputs["lef"] = list(args.lef)
        changed = True
    if args.lib:
        design_inputs["lib"] = list(args.lib)
        changed = True
    if args.def_input:
        design_inputs["def"] = args.def_input
        params.def_input = args.def_input
        changed = True
    if args.sdc:
        design_inputs["sdc"] = args.sdc
        changed = True
    if args.rc_tcl:
        design_inputs["rc_tcl"] = args.rc_tcl
        changed = True
    if args.verilog_input:
        params.verilog_input = args.verilog_input

    if changed:
        params.design_inputs = design_inputs
    if args.tech_lef or args.lef:
        params.lef_input = list(design_inputs.get("tech_lef") or []) + list(
            design_inputs.get("lef") or []
        )


def _load_workspace_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as stream:
        return json.load(stream)


def _apply_workspace_design_inputs(params, workspace_dir):
    """Derive params.design_inputs from a workspace config tree.

    The OpenROAD backend is design-inputs-first: it never parses the workspace
    itself, while iEDA/BOOKSHELF resolve paths from config/ on their own.  Read
    config/tech_path.json and config/design_path.json here so that
    `Placer.py <params.json> --workspace <ws>` works for every backend instead
    of failing with an empty design_inputs.

    Explicit params.design_inputs and CLI overrides win; the workspace only
    fills the gaps.
    """
    if not workspace_dir:
        return
    config_root = os.path.join(workspace_dir, "config")
    tech = _load_workspace_config(os.path.join(config_root, "tech_path.json"))
    design = _load_workspace_config(os.path.join(config_root, "design_path.json"))
    if not tech and not design:
        return

    design_inputs = _existing_design_inputs(params)

    def fill(key, value):
        if value and not design_inputs.get(key):
            design_inputs[key] = value

    tech_lef = tech.get("tech_lef_path")
    fill("tech_lef", [tech_lef] if tech_lef else [])
    fill("lef", list(tech.get("lef_paths") or []))
    fill("lib", list(tech.get("lib_paths") or []))
    fill("rc_tcl", tech.get("rc_tcl"))
    fill("def", design.get("def_input_path"))
    fill("sdc", design.get("sdc_path"))
    params.design_inputs = design_inputs


def _resolve_rc_tcl(params, args):
    if args.rc_tcl:
        return args.rc_tcl
    design_inputs = getattr(params, "design_inputs", {}) or {}
    return design_inputs.get("rc_tcl")


def _existing_pre_repair_tcl(params):
    handoff = getattr(params, "openroad_handoff", {}) or {}
    buffer_insertion = handoff.get("buffer_insertion", {}) or {}
    options = buffer_insertion.get("options", {}) or {}
    return options.get("pre_repair_tcl")


def _existing_handoff_write_def(params):
    handoff = getattr(params, "openroad_handoff", {}) or {}
    buffer_insertion = handoff.get("buffer_insertion", {}) or {}
    return buffer_insertion.get("write_def_after")


def _apply_output_def_override(params, output_def):
    if not output_def:
        return
    handoff = getattr(params, "openroad_handoff", None)
    if not isinstance(handoff, dict):
        return
    buffer_insertion = handoff.get("buffer_insertion")
    if not isinstance(buffer_insertion, dict):
        return
    buffer_insertion["write_def_after"] = output_def


def _apply_stage_iteration_override(params, iterations):
    if iterations is None:
        return
    stages = getattr(params, "global_place_stages", None)
    if not stages:
        stages = [{}]
        params.global_place_stages = stages
    if not isinstance(stages[0], dict):
        stages[0] = {}
    stages[0]["iteration"] = int(iterations)
    params._global_place_stages_explicit = True


def _apply_stage_optimizer_override(params, optimizer):
    if optimizer is None:
        return
    stages = getattr(params, "global_place_stages", None)
    if not stages:
        stages = [{}]
        params.global_place_stages = stages
    for stage in stages:
        if isinstance(stage, dict):
            stage["optimizer"] = str(optimizer)
    params._global_place_stages_explicit = True
    params._placement_optimizer_explicit = True


def _apply_bin_count_overrides(params, num_bins_x, num_bins_y):
    if num_bins_x is None and num_bins_y is None:
        return
    if num_bins_x is not None:
        params.num_bins_x = int(num_bins_x)
    if num_bins_y is not None:
        params.num_bins_y = int(num_bins_y)
    stages = getattr(params, "global_place_stages", None)
    if not stages:
        stages = [{}]
        params.global_place_stages = stages
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if num_bins_x is not None:
            stage["num_bins_x"] = int(num_bins_x)
        if num_bins_y is not None:
            stage["num_bins_y"] = int(num_bins_y)
    params._global_place_stages_explicit = True
    params._placement_bin_count_explicit = True


def _apply_param_overrides(
    params,
    args,
    specs,
    *,
    skip_falsy=False,
    unmarked=(),
    renames=None,
):
    """Apply homogeneous CLI overrides and their explicit-value markers."""
    renames = renames or {}
    for arg_name, convert in specs:
        value = getattr(args, arg_name, None)
        if value is None or (skip_falsy and not value):
            continue
        param_name = renames.get(arg_name, arg_name)
        setattr(params, param_name, convert(value))
        if arg_name not in unmarked:
            setattr(params, f"_{param_name}_explicit", True)


_BUFFERING_VALUE_OVERRIDES = (
    ("buffering_max_selected_actions", int),
    ("buffering_continuous_steps", int),
    ("buffering_continuous_lr", float),
    ("buffering_max_repeaters_per_segment", int),
    ("buffering_segment_count_z_init", float),
    ("buffering_fixed_bsu_index", int),
    ("buffering_discrete_count_proximal_lambda", float),
    ("buffering_route_b_selection_fraction", float),
    ("buffering_segment_capacity_enabled", int),
    ("buffering_segment_integer_projection_interval", int),
    ("buffering_segment_integer_projection_start_step", int),
    ("buffering_segment_integer_projection_project_bsu", int),
    ("buffering_segment_integer_projection_reset_optimizer_state", int),
    ("buffering_commit_enabled", int),
    ("joint_buffer_outer_iterations", int),
    ("joint_segment_virtual_window_steps", int),
    ("joint_segment_virtual_warmup_steps", int),
    ("joint_segment_sizing_rounds", int),
    ("joint_segment_sizing_up_percent", float),
    ("joint_segment_buffering_rounds", int),
    ("joint_segment_buffering_selection_fraction", float),
    ("joint_segment_pin2pin_rebootstrap_enabled", int),
    ("joint_segment_sizing_enabled", int),
    ("joint_segment_route_b_enabled", int),
    ("joint_segment_virtual_density_enabled", int),
)


_BUFFERING_STRING_OVERRIDES = (
    ("buffering_mode", str),
    ("buffering_output_dir", str),
    ("buffering_fixed_buffer_master", str),
    ("buffering_segment_strategy", str),
    ("buffering_candidate_strategy", str),
    ("buffering_segment_driver_cap_mode", str),
    ("buffering_segment_capacity_grid", str),
    ("buffering_segment_count_timing_backend", str),
    ("buffering_segment_transfer_backend", str),
    ("buffering_committed_def_path", str),
    ("buffering_post_commit_pysta_json", str),
    ("joint_post_commit_openroad_eco", str),
)


def _apply_buffering_alias_overrides(params, args):
    _apply_param_overrides(params, args, _BUFFERING_VALUE_OVERRIDES)
    _apply_param_overrides(
        params,
        args,
        _BUFFERING_STRING_OVERRIDES,
        skip_falsy=True,
        unmarked=("buffering_segment_driver_cap_mode",),
    )
    if args.joint_buffer_commit_period is not None:
        value = int(args.joint_buffer_commit_period)
        params.joint_buffer_commit_period = value
        params.buffering_segment_integer_projection_interval = value
        params._joint_buffer_commit_period_explicit = True
        params._buffering_segment_integer_projection_interval_explicit = True
    if args.joint_buffer_outer_iterations is not None:
        params.joint_buffer_outer_iterations = int(args.joint_buffer_outer_iterations)
        params._joint_buffer_outer_iterations_explicit = True
    if args.joint_buffer_max_per_segment is not None:
        value = int(args.joint_buffer_max_per_segment)
        params.joint_buffer_max_per_segment = value
        params.buffering_max_repeaters_per_segment = value
        params._joint_buffer_max_per_segment_explicit = True
        params._buffering_max_repeaters_per_segment_explicit = True
    if args.joint_segment_milestones:
        params.joint_segment_milestones = tuple(
            float(value) for value in args.joint_segment_milestones
        )
        params._joint_segment_milestones_explicit = True


def uses_buffer_only_handoff(params):
    handoff = getattr(params, "openroad_handoff", {}) or {}
    if not handoff.get("enabled"):
        return False
    buffer_insertion = handoff.get("buffer_insertion", {}) or {}
    return buffer_insertion.get("strategy") in ("buffer-only", "buffer_only")


_MAIN_VALUE_OVERRIDES = (
    ("joint_staged_smoke_outer_iterations", int),
    ("target_density", float),
    ("stop_overflow", float),
    ("gpu", int),
    ("gpu_id", int),
    ("enable_fillers", int),
    ("gp_noise_ratio", float),
    ("legalize", int),
    ("random_center_init", int),
    ("init_loc_perc_x", float),
    ("init_loc_perc_y", float),
    ("plot_interval", int),
    ("plot", int),
    ("placement_initial_learning_rate_max", float),
    ("placement_overflow_tolerance", float),
    ("diff_timing_driven_placement", int),
    ("enable_net_weighting", int),
    ("pin2pin_weight", float),
    ("pin2pin_min_weight", float),
    ("pin2pin_max_weight", float),
    ("pin2pin_accumulate_weight", float),
    ("net_weighting_npaths", float),
    ("net_weighting_nendpoints", int),
    ("net_weighting_update_interval", int),
    ("timing_topology_refresh_interval", int),
    ("timing_topology_enable_overflow_threshold", float),
    ("timing_wns_coeff", float),
    ("timing_tns_coeff", float),
    ("timing_grad_balance_target_ratio", float),
    ("timing_gradient_net_weight_scale", float),
    ("timing_gradient_net_weight_max", float),
    ("discrete_gradient_topk_up_percent", float),
    ("discrete_gradient_topk_down_percent", float),
    ("discrete_gradient_topk_vt_percent", float),
    ("discrete_gradient_topk_shared_budget", int),
    ("discrete_gradient_topk_shared_budget_percent", float),
    ("discrete_gradient_topk_oscillation_veto", int),
    ("real_size_learning_rate", float),
    ("real_size_warmup_steps", int),
    ("real_size_transition_profile", int),
)


_MAIN_STRING_OVERRIDES = (
    ("flow_kind", str),
    ("joint_quality_profile", str),
    ("place_io_engine", str),
    ("result_dir", str),
    ("base_design_name", str),
    ("timing_propagation_device", str),
    ("surrogate_cache_root", str),
    ("handoff_restart_policy", str),
    ("net_weighting_scheme", str),
    ("timing_placement_carrier", str),
    ("continuous_size_dynamics_mode", str),
    ("sizing_parameterization", str),
    ("real_size_execution_mode", str),
    ("real_size_transition_backend", str),
    ("real_size_transition_artifact_policy", str),
    ("size_interpolated_pin_native_op", str),
    ("openroad_vt_suffixes", str),
)


_MAIN_UNMARKED_OVERRIDES = {
    "target_density",
    "stop_overflow",
    "plot_interval",
    "flow_kind",
    "joint_quality_profile",
    "place_io_engine",
    "result_dir",
    "base_design_name",
    "handoff_restart_policy",
}


_MAIN_PARAM_RENAMES = {
    "legalize": "legalize_flag",
    "random_center_init": "random_center_init_flag",
    "plot": "plot_flag",
}


def build_effective_params_from_args(args):
    params = _load_params_json(args.params_json)

    for specs, skip_falsy in (
        (_MAIN_VALUE_OVERRIDES, False),
        (_MAIN_STRING_OVERRIDES, True),
    ):
        _apply_param_overrides(
            params,
            args,
            specs,
            skip_falsy=skip_falsy,
            unmarked=_MAIN_UNMARKED_OVERRIDES,
            renames=_MAIN_PARAM_RENAMES,
        )
    _apply_stage_iteration_override(params, args.iterations)
    _apply_stage_optimizer_override(params, args.placement_optimizer)
    _apply_bin_count_overrides(params, args.num_bins_x, args.num_bins_y)
    if args.with_sta:
        params.with_sta = 1
        params._with_sta_explicit = True

    pin2pin_min_weight = getattr(params, "pin2pin_min_weight", None)
    pin2pin_max_weight = getattr(params, "pin2pin_max_weight", None)
    if (
        pin2pin_min_weight is not None
        and pin2pin_max_weight is not None
        and float(pin2pin_min_weight) > float(pin2pin_max_weight)
    ):
        raise ValueError("pin2pin_min_weight must not exceed pin2pin_max_weight")

    _apply_design_input_overrides(params, args)

    workspace = args.workspace or getattr(params, "result_dir", None) or os.getcwd()
    _apply_workspace_design_inputs(params, str(workspace))

    output_def = args.output_def or _existing_handoff_write_def(params)
    if args.handoff_mode:
        rc_tcl = _resolve_rc_tcl(params, args)
        pre_repair_tcl = _existing_pre_repair_tcl(params)
        if (
            args.handoff_mode == "buffer-only"
            and not rc_tcl
            and not pre_repair_tcl
        ):
            raise ValueError("buffer-only handoff requires --rc-tcl or custom pre_repair_tcl")
        params.openroad_handoff = handoff_launcher.build_openroad_handoff_config(
            output_def=output_def,
            rc_tcl=rc_tcl,
            handoff_mode=args.handoff_mode,
            trigger_period=args.handoff_trigger_period,
            max_handoff_overflow=args.max_handoff_overflow,
            pre_repair_tcl=pre_repair_tcl,
        )
    else:
        _apply_output_def_override(params, args.output_def)

    _apply_buffering_alias_overrides(params, args)

    apply_flow_defaults(params)

    launch = {
        "workspace": str(workspace),
        "result_dir": getattr(params, "result_dir", None),
        "output_def": output_def,
        "output_verilog": (
            None if args.output_verilog is None else str(args.output_verilog)
        ),
        "setup_margin": (
            None if args.setup_margin is None else float(args.setup_margin)
        ),
        "effective_params_manifest": resolved_params_manifest(params),
    }
    return params, launch
