import argparse
import copy
import glob
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import types
from collections import Counter


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
AIEDA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
INSTALL_ROOT = os.path.join(PROJECT_ROOT, "install")
PARAMS_SCHEMA = os.path.join(PROJECT_ROOT, "dreamplace", "params.json")
BASELINE_ROOT = (
    "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/ablation_runs/"
    "2026-03-20_cx55_soft_L_full_ablation_v1/workspaces/baseline"
)
TECH_LEF = "/nfs/share/home/qiming/icsprout55-pdk/prtech/techLEF/N551P6M.lef"

TARGET_DESIGNS = ("APU", "PPU", "BM64")
VARIANTS = (
    "no_handoff",
    "handoff_no_buffer",
    "handoff_repair_design",
    "buffer-only",
)
DEFAULT_VARIANTS = VARIANTS[:3]

NORMAL_PARAM_FIELDS = (
    "global_place_stages",
    "num_bins_x",
    "num_bins_y",
    "target_density",
    "density_weight",
    "random_seed",
    "stop_overflow",
    "routability_opt_flag",
    "with_sta",
    "legalize_flag",
    "detailed_place_flag",
    "num_threads",
    "auto_adjust_bins",
    "get_congestion_map",
    "route_info_input",
)

INTERESTING_LOG_PATTERNS = (
    "[ERROR]",
    "Traceback",
    "Exception",
    "RuntimeError",
    "AssertionError",
    "failed",
    "placement failed",
    "OpenROAD handoff triggered",
    "OpenROAD buffer insertion executed",
    "OpenROAD handoff skipped",
    "requires_runtimedb_rebuild",
    "runtimedb_rebuild_performed",
    "ODB-",
    "DIVERGENCE",
    "segmentation",
    "abort",
    "mismatch",
    "invalid",
)

ERROR_LOG_PATTERNS = (
    "[ERROR]",
    "Traceback",
    "Exception",
    "RuntimeError",
    "AssertionError",
    "placement failed",
    "ODB-0289",
    "DIVERGENCE",
    "segmentation",
    "abort",
)

ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+),.*?"
    r"Obj\s+(?P<objective>[0-9.+\-Ee]+),\s+"
    r"DensityWeight\s+(?P<density_weight>[0-9.+\-Ee]+),\s+"
    r"HPWL\s+(?P<hpwl>[0-9.+\-Ee]+),\s+"
    r"Overflow\s+(?P<overflow>[0-9.+\-Ee]+),\s+"
    r"MaxDensity\s+(?P<max_density>[0-9.+\-Ee]+)"
)


def load_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def write_json(path, payload):
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)


def require_paths(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise RuntimeError("missing required paths: %s" % ", ".join(missing))


def load_source_module(module_name, relative_path):
    module_path = os.path.join(PROJECT_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_default_params_dict():
    schema = load_json(PARAMS_SCHEMA)
    return {key: value.get("default") for key, value in schema.items()}


def install_optional_ieda_dummy_modules():
    sys.modules.setdefault("third_party.iEDA", types.ModuleType("third_party.iEDA"))
    sys.modules.setdefault("third_party.iEDA.bin", types.ModuleType("third_party.iEDA.bin"))
    sys.modules.setdefault(
        "third_party.iEDA.bin.ieda_py",
        types.ModuleType("third_party.iEDA.bin.ieda_py"),
    )


def prepare_runtime_modules(mpl_config_dir):
    extension_pattern = os.path.join(
        INSTALL_ROOT,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "placeio_openroad_cpp*.so",
    )
    if not glob.glob(extension_pattern):
        raise RuntimeError("OpenROAD bridge extension is not available under install/")

    require_paths((PARAMS_SCHEMA,))
    os.environ["MPLCONFIGDIR"] = mpl_config_dir
    os.makedirs(mpl_config_dir, exist_ok=True)

    if INSTALL_ROOT not in sys.path:
        sys.path.insert(0, INSTALL_ROOT)
    for extra_path in (PROJECT_ROOT, AIEDA_ROOT):
        if extra_path not in sys.path:
            sys.path.insert(1, extra_path)

    install_optional_ieda_dummy_modules()

    import dreamplace.configure as configure

    configure.compile_configurations["CUDA_FOUND"] = ""

    params_module = load_source_module("dreamplace.Params", "dreamplace/Params.py")
    if not getattr(params_module.Params, "_cx55_default_fallback", False):
        params_module.Params.__getattr__ = lambda self, name: False
        params_module.Params._cx55_default_fallback = True

    load_source_module(
        "dreamplace.ops.openroad_handoff.session",
        "dreamplace/ops/openroad_handoff/session.py",
    )
    load_source_module(
        "dreamplace.ops.openroad_handoff.controller",
        "dreamplace/ops/openroad_handoff/controller.py",
    )
    load_source_module(
        "dreamplace.ops.openroad_handoff",
        "dreamplace/ops/openroad_handoff/__init__.py",
    )
    load_source_module("dreamplace.BasicPlace", "dreamplace/BasicPlace.py")
    load_source_module("dreamplace.macroPlaceDB", "dreamplace/macroPlaceDB.py")
    load_source_module("dreamplace.NonLinearPlace", "dreamplace/NonLinearPlace.py")
    load_source_module("dreamplace.Placer", "dreamplace/Placer.py")

    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    if not hasattr(placeio_openroad.PlaceIOFunction, "build_buffer_insertion_command"):
        placeio_openroad.PlaceIOFunction.build_buffer_insertion_command = staticmethod(
            placeio_openroad.PlaceIOFunction._build_tcl_command
        )

    from dreamplace.Params import Params
    import dreamplace.Placer as Placer

    return Params, Placer


def design_workspace(design):
    return os.path.join(BASELINE_ROOT, design, "workspace")


def load_design_config(design):
    workspace = design_workspace(design)
    config_root = os.path.join(workspace, "config")
    param_path = os.path.join(config_root, "dreamplace_config", "param.json")
    tech_path = os.path.join(config_root, "tech_path.json")
    design_path = os.path.join(config_root, "design_path.json")
    workspace_path = os.path.join(config_root, "workspace.json")

    param_data = load_json(param_path)
    tech_paths = load_json(tech_path)
    design_paths = load_json(design_path)
    workspace_meta = load_json(workspace_path)
    workspace_info = workspace_meta.get("workspace", {}) or {}

    def_input = param_data.get("def_input") or design_paths["def_input_path"]
    verilog_input = param_data.get("verilog_input") or design_paths["verilog_input_path"]
    sdc_path = design_paths["sdc_path"]
    lef_paths = list(tech_paths.get("lef_paths", []))
    lib_paths = list(tech_paths.get("lib_paths", []))

    require_paths(
        [
            param_path,
            tech_path,
            design_path,
            workspace_path,
            TECH_LEF,
            def_input,
            verilog_input,
            sdc_path,
        ]
        + lef_paths
        + lib_paths
    )

    return {
        "design": workspace_info.get("design") or design,
        "process_node": workspace_info.get("process_node") or "cx55",
        "workspace": workspace,
        "config_root": config_root,
        "param_path": param_path,
        "param_data": param_data,
        "tech_lef": TECH_LEF,
        "lef_paths": lef_paths,
        "lib_paths": lib_paths,
        "def_input": def_input,
        "verilog_input": verilog_input,
        "sdc_path": sdc_path,
    }


def normal_param_snapshot(param_data):
    return {key: copy.deepcopy(param_data.get(key)) for key in NORMAL_PARAM_FIELDS}


def artifact_root_for(design, variant, artifact_base=None):
    base = artifact_base or os.path.join(THIS_DIR, "artifacts")
    return os.path.join(base, "%s_cx55_n551_ablation_iter100_%s" % (design, variant))


def make_run_dir(artifact_root, design, variant):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = "%s_cx55_%s_%s" % (design, variant, timestamp)
    run_dir = os.path.join(artifact_root, name)
    if os.path.exists(run_dir):
        run_dir = "%s_%d" % (run_dir, os.getpid())
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def variant_handoff_config(variant, final_def):
    if variant == "no_handoff":
        return {
            "enabled": False,
            "trigger": {"mode": "disabled"},
        }

    buffer_insertion = {
        "enabled": variant != "handoff_no_buffer",
        "strategy": "repair_design",
        "options": {"max_wire_length": 10},
    }
    if variant == "buffer-only":
        buffer_insertion = {
            "enabled": True,
            "strategy": "buffer-only",
            "options": {
                "pre_repair_tcl": [
                    "set_wire_rc -signal -layer MET3",
                    "estimate_parasitics -placement",
                ],
            },
        }
    if buffer_insertion["enabled"]:
        buffer_insertion["write_def_after"] = final_def

    return {
        "enabled": True,
        "trigger": {"mode": "disabled"},
        "buffer_insertion": buffer_insertion,
    }


def build_params(params_cls, design_config, run_dir, final_def, variant, plot_interval=None):
    params = params_cls()
    params.fromJson(load_default_params_dict())
    params.fromJson(copy.deepcopy(design_config["param_data"]))

    params.place_io_engine = "openroad"
    params.gpu = 0
    params.with_sta = 0
    params.routability_opt_flag = 0
    params.result_dir = run_dir
    if plot_interval is not None:
        params.plot_interval = int(plot_interval)
    params.base_design_name = "%s_cx55_ablation_iter100_%s" % (
        design_config["design"],
        variant,
    )
    params.design_inputs = {
        "tech_lef": [design_config["tech_lef"]],
        "lef": list(design_config["lef_paths"]),
        "def": design_config["def_input"],
        "sdc": design_config["sdc_path"],
        "lib": list(design_config["lib_paths"]),
    }
    params.lef_input = list(params.design_inputs["tech_lef"]) + list(
        params.design_inputs["lef"]
    )
    params.def_input = params.design_inputs["def"]
    params.verilog_input = design_config["verilog_input"]
    params.openroad_handoff = variant_handoff_config(variant, final_def)
    return params


def install_exact_once_trigger(placedb, target_iter=100):
    original_create = placedb.create_openroad_handoff_controller
    state = {"attempted": False}

    def exact_create(controller_params):
        controller = original_create(controller_params)
        if controller is None:
            return None

        def exact_should(event):
            if state["attempted"]:
                return False, None
            iteration = event.get("absolute_iteration", event.get("iteration"))
            if iteration is None:
                return False, None
            if int(iteration) == target_iter:
                state["attempted"] = True
                return True, {
                    "trigger_mode": "exact_iteration",
                    "trigger_reason": "exact_iteration@iter=%d" % target_iter,
                }
            return False, None

        controller.should_handoff = exact_should
        return controller

    placedb.create_openroad_handoff_controller = exact_create


def install_periodic_trigger(placedb, trigger_period=50):
    original_create = placedb.create_openroad_handoff_controller
    attempted_iterations = set()

    def periodic_create(controller_params):
        controller = original_create(controller_params)
        if controller is None:
            return None

        def periodic_should(event):
            iteration = event.get("absolute_iteration", event.get("iteration"))
            if iteration is None:
                return False, None
            iteration = int(iteration)
            if iteration <= 0 or iteration % trigger_period != 0:
                return False, None
            if iteration in attempted_iterations:
                return False, None
            attempted_iterations.add(iteration)
            return True, {
                "trigger_mode": "periodic_iteration",
                "trigger_reason": "periodic_iteration@period=%d@iter=%d"
                % (trigger_period, iteration),
            }

        controller.should_handoff = periodic_should
        return controller

    placedb.create_openroad_handoff_controller = periodic_create


def configure_regression_logging():
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        force=True,
    )


class StdoutStderrTee:
    def __init__(self, log_path, mirror_to_stdout=True):
        self.log_path = log_path
        self.mirror_to_stdout = mirror_to_stdout
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._read_fd = None
        self._writer_thread = None
        self._log_stream = None

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        self._read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._log_stream = open(self.log_path, "ab", buffering=0)
        self._writer_thread = threading.Thread(target=self._pump, daemon=True)
        self._writer_thread.start()
        return self

    def _pump(self):
        while True:
            chunk = os.read(self._read_fd, 4096)
            if not chunk:
                break
            if self.mirror_to_stdout:
                os.write(self._saved_stdout_fd, chunk)
            self._log_stream.write(chunk)

    def __exit__(self, exc_type, exc_value, traceback_value):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        self._writer_thread.join()
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._read_fd)
        self._log_stream.close()


def read_text(path):
    try:
        with open(path, "r", errors="replace") as stream:
            return stream.read()
    except OSError:
        return ""


def empty_policy_status_counts():
    return {"clean": 0, "violated": 0, "unknown": 0}


def empty_policy_violation_totals():
    return {
        "non_buffer_master_change_count": 0,
        "non_buffer_size_change_count": 0,
        "non_buffer_orient_change_count": 0,
        "added_non_buffer_count": 0,
        "removed_non_buffer_count": 0,
        "connectivity_violation_count": 0,
        "total_violation_count": 0,
    }


def merge_policy_summary(policy, status_counts, violation_totals, unknown_reason_counts):
    if not isinstance(policy, dict):
        status_counts["unknown"] += 1
        return
    status = policy.get("status")
    if isinstance(status, str) and status in status_counts:
        status_counts[status] += 1
    else:
        status_counts["unknown"] += 1
    violation_counts = policy.get("violation_counts", {}) or {}
    if not isinstance(violation_counts, dict):
        violation_counts = {}
    for field_name in violation_totals:
        value = violation_counts.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        violation_totals[field_name] += int(value)
    unknown_reasons = policy.get("unknown_reasons", []) or []
    if not isinstance(unknown_reasons, (list, tuple)):
        unknown_reasons = []
    for unknown in unknown_reasons:
        if not isinstance(unknown, dict):
            continue
        reason = unknown.get("reason")
        if isinstance(reason, str) and reason:
            unknown_reason_counts[reason] += 1


def finalized_session_status(session_summary):
    event_count = session_summary.get("event_count", 0)
    if not is_int_value(event_count) or event_count <= 0:
        return "not_started"
    statuses = session_summary.get("status_counts", {}) or {}
    if not isinstance(statuses, dict):
        statuses = {}
    success_count = statuses.get("success", 0)
    if not is_int_value(success_count):
        success_count = 0
    return "completed" if success_count == event_count else "partial"


def parse_iteration_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def event_iteration(event):
    for field_name in ("absolute_iteration", "iteration"):
        if field_name not in event:
            continue
        iteration = parse_iteration_value(event.get(field_name))
        if iteration is not None:
            return iteration
    return None


def iter_dict_events(event_history):
    if event_history is None or isinstance(event_history, (str, bytes)):
        return
    try:
        iterator = iter(event_history)
    except TypeError:
        return
    for event in iterator:
        if isinstance(event, dict):
            yield event


def build_session_summary(event_history):
    valid_events = list(iter_dict_events(event_history))
    mutation_kind_counts = {
        "no_mutation": 0,
        "placement_only": 0,
        "geometry_changed": 0,
        "topology_changed": 0,
    }
    buffer_churn_totals = {
        "added_buffer_count": 0,
        "removed_buffer_count": 0,
        "surviving_buffer_count": 0,
    }
    status_counts = Counter()
    event_iterations = []
    buffer_only_policy_status_counts = empty_policy_status_counts()
    buffer_only_policy_violation_totals = empty_policy_violation_totals()
    buffer_only_policy_unknown_reason_counts = Counter()
    latest_metric_snapshot = None

    for event in valid_events:
        status = event.get("status")
        if isinstance(status, str) and status:
            status_counts[status] += 1
        iteration = event_iteration(event)
        if iteration is not None:
            event_iterations.append(iteration)
        mutation_kind = event.get("mutation_kind")
        if mutation_kind in mutation_kind_counts:
            mutation_kind_counts[mutation_kind] += 1
        for field_name in buffer_churn_totals:
            value = event.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            buffer_churn_totals[field_name] += int(value)
        merge_policy_summary(
            event.get("buffer_only_policy"),
            buffer_only_policy_status_counts,
            buffer_only_policy_violation_totals,
            buffer_only_policy_unknown_reason_counts,
        )
        metric_snapshot = event.get("metric_snapshot")
        if isinstance(metric_snapshot, dict):
            latest_metric_snapshot = copy.deepcopy(metric_snapshot)

    summary = {
        "status": "not_started",
        "event_count": len(valid_events),
        "event_iterations": event_iterations,
        "status_counts": dict(status_counts),
        "mutation_kind_counts": mutation_kind_counts,
        "buffer_churn_totals": buffer_churn_totals,
        "buffer_only_policy_status_counts": buffer_only_policy_status_counts,
        "buffer_only_policy_violation_totals": buffer_only_policy_violation_totals,
        "buffer_only_policy_unknown_reason_counts": dict(
            buffer_only_policy_unknown_reason_counts
        ),
        "latest_metric_snapshot": latest_metric_snapshot,
        "events_tail": copy.deepcopy(valid_events[-3:]),
    }
    summary["status"] = finalized_session_status(summary)
    return summary


def build_handoff_strategy_summary(
    params,
    command_builder,
    target_iter=100,
    trigger_period=None,
):
    handoff_config = getattr(params, "openroad_handoff", {}) or {}
    handoff_enabled = bool(handoff_config.get("enabled", False))
    buffer_cfg = handoff_config.get("buffer_insertion", {}) or {}
    strategy = buffer_cfg.get("strategy")
    options = copy.deepcopy(buffer_cfg.get("options", {}) or {})
    command = None
    if handoff_enabled and strategy:
        command = command_builder(strategy, options)
    return {
        "handoff_enabled": handoff_enabled,
        "trigger_policy": (
            (
                "periodic_absolute_iteration_%d" % trigger_period
                if trigger_period
                else "exact_once_absolute_iteration_%d" % target_iter
            )
            if handoff_enabled
            else "disabled"
        ),
        "buffer_insertion_enabled": bool(buffer_cfg.get("enabled", False)),
        "buffer_insertion_strategy": strategy,
        "buffer_insertion_options": options,
        "buffer_insertion_command": command,
    }


def parse_last_iteration(log_text):
    last = None
    for line in log_text.splitlines():
        match = ITERATION_RE.search(line)
        if not match:
            continue
        last = {
            "line": line,
            "iteration": int(match.group("iteration")),
            "objective": float(match.group("objective")),
            "density_weight": float(match.group("density_weight")),
            "hpwl": float(match.group("hpwl")),
            "overflow": float(match.group("overflow")),
            "max_density": float(match.group("max_density")),
        }
    return last


def matching_log_lines(log_text, patterns, max_lines=80):
    matches = []
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for line in log_text.splitlines():
        lowered_line = line.lower()
        if any(pattern.lower() in lowered_line for pattern in lowered_patterns):
            matches.append(line)
    return matches[-max_lines:]


REPAIR_MOVE_SEQUENCE_RE = re.compile(r"Repair move sequence:\s*(?P<moves>.*?)\s*$")
INSERTED_BUFFER_RE = re.compile(r"Inserted\s+(?P<count>\d+)\s+.*buffers?")
REMOVED_BUFFER_RE = re.compile(r"Removed\s+(?P<count>\d+)\s+buffers?")
RESIZED_INSTANCE_RE = re.compile(r"Resized\s+(?P<count>\d+)\s+instances?")
TIMING_EVIDENCE_RE = re.compile(r"\b(WNS|TNS|violating endpoints?|slack)\b", re.IGNORECASE)


def is_int_value(value):
    return not isinstance(value, bool) and isinstance(value, int)


def int_count(mapping, field_name):
    value = mapping.get(field_name)
    if is_int_value(value):
        return value
    return None


def all_zero_int_counts(mapping, field_names):
    for field_name in field_names:
        if int_count(mapping, field_name) != 0:
            return False
    return True


def int_counts_are_present_zero(mapping, field_names):
    for field_name in field_names:
        if field_name not in mapping:
            return False
        value = mapping.get(field_name)
        if not is_int_value(value) or value != 0:
            return False
    return True


def extract_openroad_diagnostic_evidence(log_text, event_history):
    repair_move_sequences = []
    inserted_buffer_log_count = 0
    removed_buffer_log_count = 0
    resized_instance_log_count = 0
    for line in log_text.splitlines():
        move_match = REPAIR_MOVE_SEQUENCE_RE.search(line)
        if move_match:
            repair_move_sequences.append(move_match.group("moves").strip())
        insert_match = INSERTED_BUFFER_RE.search(line)
        if insert_match:
            inserted_buffer_log_count += int(insert_match.group("count"))
        remove_match = REMOVED_BUFFER_RE.search(line)
        if remove_match:
            removed_buffer_log_count += int(remove_match.group("count"))
        resize_match = RESIZED_INSTANCE_RE.search(line)
        if resize_match:
            resized_instance_log_count += int(resize_match.group("count"))

    labels = set()
    move_text = " ".join(repair_move_sequences)
    has_buffer_moves = "BufferMove" in move_text or "SplitLoadMove" in move_text
    has_timing_evidence = bool(TIMING_EVIDENCE_RE.search(log_text))
    has_clean_zero_allowed_policy = False
    all_events_have_zero_churn = True
    for event in iter_dict_events(event_history):
        if not int_counts_are_present_zero(
            event,
            ("added_buffer_count", "removed_buffer_count"),
        ):
            all_events_have_zero_churn = False
        policy = event.get("buffer_only_policy") or {}
        if not isinstance(policy, dict):
            policy = {}
        allowed = policy.get("allowed_change_counts", {}) or {}
        if not isinstance(allowed, dict):
            allowed = {}
        violations = policy.get("violation_counts", {}) or {}
        if not isinstance(violations, dict):
            violations = {}
        repair_resize_violation_count = (
            int_count(violations, "non_buffer_master_change_count") or 0
        ) + (int_count(violations, "non_buffer_size_change_count") or 0)
        if (
            event.get("buffer_insertion_strategy") == "repair_design"
            and policy.get("status") == "violated"
            and repair_resize_violation_count > 0
            and resized_instance_log_count > 0
        ):
            labels.add("consistent_with_repair_design_driver_resize")
        if (
            policy.get("status") == "clean"
            and all_zero_int_counts(
                allowed,
                ("added_buffer_count", "removed_buffer_count", "resized_buffer_count"),
            )
            and int_counts_are_present_zero(
                event,
                ("added_buffer_count", "removed_buffer_count"),
            )
        ):
            has_clean_zero_allowed_policy = True
    if has_clean_zero_allowed_policy and all_events_have_zero_churn:
        labels.add("clean_no_buffer_churn")
        if has_buffer_moves:
            labels.add("buffer_moves_available_but_not_accepted")
    if not has_timing_evidence:
        labels.add("insufficient_timing_evidence")

    return {
        "repair_move_sequences": repair_move_sequences,
        "inserted_buffer_log_count": inserted_buffer_log_count,
        "removed_buffer_log_count": removed_buffer_log_count,
        "resized_instance_log_count": resized_instance_log_count,
        "diagnostic_labels": sorted(labels),
    }


def run_one(
    design,
    variant,
    artifact_base=None,
    target_iter=100,
    trigger_period=None,
    plot_interval=None,
):
    design_config = load_design_config(design)
    artifact_root = artifact_root_for(design_config["design"], variant, artifact_base)
    os.makedirs(artifact_root, exist_ok=True)
    run_dir = make_run_dir(artifact_root, design_config["design"], variant)
    log_path = os.path.join(run_dir, "run.log")
    final_def = os.path.join(
        run_dir,
        "%s_cx55_ablation_iter100_%s.def" % (design_config["design"], variant),
    )

    Params, Placer = prepare_runtime_modules(os.path.join(run_dir, "matplotlib"))
    params = build_params(
        Params,
        design_config,
        run_dir,
        final_def,
        variant,
        plot_interval=plot_interval,
    )

    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    handoff_strategy = build_handoff_strategy_summary(
        params,
        placeio_openroad.PlaceIOFunction.build_buffer_insertion_command,
        target_iter=target_iter,
        trigger_period=trigger_period,
    )

    start = time.time()
    placement_result = None
    event_history = []
    failure = None

    with StdoutStderrTee(log_path):
        configure_regression_logging()
        logging.info("design=%s", design_config["design"])
        logging.info("variant=%s", variant)
        logging.info("param_path=%s", design_config["param_path"])
        logging.info("tech_lef=%s", design_config["tech_lef"])
        logging.info("def_input=%s", design_config["def_input"])
        logging.info("verilog_input=%s", design_config["verilog_input"])
        logging.info("sdc_path=%s", design_config["sdc_path"])
        logging.info("artifact_root=%s", artifact_root)
        logging.info("run_dir=%s", run_dir)
        engine = None
        try:
            engine = Placer.PlacementEngine(params)
            engine.setup_rawdb(data_manager=types.SimpleNamespace(dir_workspace=run_dir))
            if trigger_period:
                install_periodic_trigger(engine.placedb, trigger_period=trigger_period)
            else:
                install_exact_once_trigger(engine.placedb, target_iter=target_iter)
            placement_result = engine.run()
            session = getattr(engine.placedb, "handoff_session", None)
            if session is not None:
                event_history = copy.deepcopy(session.event_history)
        except Exception as exc:
            logging.exception("cx55 ablation iter100 run failed")
            if engine is not None and getattr(engine, "placedb", None) is not None:
                session = getattr(engine.placedb, "handoff_session", None)
                if session is not None:
                    event_history = copy.deepcopy(session.event_history)
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    elapsed = time.time() - start
    run_log_text = read_text(log_path)
    session_summary = build_session_summary(event_history)
    summary = {
        "design": design_config["design"],
        "variant": variant,
        "elapsed_sec": elapsed,
        "config": {
            "workspace": os.path.dirname(design_config["workspace"]),
            "param_path": design_config["param_path"],
            "tech_lef": design_config["tech_lef"],
            "def_input": design_config["def_input"],
            "verilog_input": design_config["verilog_input"],
            "plot_interval": getattr(params, "plot_interval", None),
            "normal_param_snapshot": normal_param_snapshot(design_config["param_data"]),
        },
        "handoff_strategy": handoff_strategy,
        "placement_result": placement_result,
        "last_iteration": parse_last_iteration(run_log_text),
        "session": session_summary,
        "openroad_diagnostic_evidence": extract_openroad_diagnostic_evidence(
            run_log_text, event_history
        ),
        "failure": failure,
        "final_def": final_def,
        "final_def_exists": os.path.exists(final_def),
        "run_dir": run_dir,
        "log_path": log_path,
        "manifest_path": os.path.join(run_dir, "run_manifest.json"),
        "summary_path": os.path.join(run_dir, "summary.json"),
        "interesting_log_lines_tail": matching_log_lines(
            run_log_text,
            INTERESTING_LOG_PATTERNS,
        ),
        "error_log_matches": matching_log_lines(run_log_text, ERROR_LOG_PATTERNS),
    }
    write_json(summary["summary_path"], summary)
    write_json(summary["manifest_path"], summary)
    write_json(
        os.path.join(artifact_root, "latest_run.json"),
        {
            "run_dir": run_dir,
            "summary_path": summary["summary_path"],
        },
    )
    return summary


def repair_design_was_placement_only(summary):
    counts = summary.get("session", {}).get("mutation_kind_counts", {})
    return (
        counts.get("placement_only", 0) > 0
        and counts.get("topology_changed", 0) == 0
        and counts.get("geometry_changed", 0) == 0
    )


def parse_csv(values):
    result = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", dest="designs", action="append", choices=TARGET_DESIGNS)
    parser.add_argument("--variant", dest="variants", action="append", choices=VARIANTS)
    parser.add_argument("--artifact-base")
    parser.add_argument("--target-iter", type=int, default=100)
    parser.add_argument("--trigger-period", type=int)
    parser.add_argument("--plot-interval", type=int)
    parser.add_argument("--combined-output")
    parser.add_argument("--include-buffer-only-if-needed", action="store_true")
    args = parser.parse_args(argv)
    if args.trigger_period is not None and args.trigger_period <= 0:
        parser.error("--trigger-period must be positive")
    if args.plot_interval is not None and args.plot_interval <= 0:
        parser.error("--plot-interval must be positive")

    designs = parse_csv(args.designs or list(TARGET_DESIGNS))
    variants = parse_csv(args.variants or list(DEFAULT_VARIANTS))
    summaries = []
    failed = False

    for design in designs:
        design_repair_summary = None
        for variant in variants:
            summary = run_one(
                design,
                variant,
                artifact_base=args.artifact_base,
                target_iter=args.target_iter,
                trigger_period=args.trigger_period,
                plot_interval=args.plot_interval,
            )
            summaries.append(summary)
            if summary.get("failure") is not None:
                failed = True
            if variant == "handoff_repair_design":
                design_repair_summary = summary
            print(
                "%s %s failure=%s hpwl=%s overflow=%s mutations=%s churn=%s"
                % (
                    summary["design"],
                    summary["variant"],
                    summary["failure"],
                    (summary.get("placement_result") or {}).get("hpwl"),
                    (summary.get("placement_result") or {}).get("overflow"),
                    summary["session"]["mutation_kind_counts"],
                    summary["session"]["buffer_churn_totals"],
                )
            )

        if (
            args.include_buffer_only_if_needed
            and "buffer-only" not in variants
            and design_repair_summary is not None
            and repair_design_was_placement_only(design_repair_summary)
        ):
            summary = run_one(
                design,
                "buffer-only",
                artifact_base=args.artifact_base,
                target_iter=args.target_iter,
                trigger_period=args.trigger_period,
                plot_interval=args.plot_interval,
            )
            summaries.append(summary)
            if summary.get("failure") is not None:
                failed = True
            print(
                "%s %s failure=%s hpwl=%s overflow=%s mutations=%s churn=%s"
                % (
                    summary["design"],
                    summary["variant"],
                    summary["failure"],
                    (summary.get("placement_result") or {}).get("hpwl"),
                    (summary.get("placement_result") or {}).get("overflow"),
                    summary["session"]["mutation_kind_counts"],
                    summary["session"]["buffer_churn_totals"],
                )
            )

    if args.combined_output:
        write_json(args.combined_output, summaries)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
