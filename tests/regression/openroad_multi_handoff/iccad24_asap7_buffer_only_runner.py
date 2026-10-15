import argparse
import copy
import json
import logging
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import cx55_ablation_iter100_matrix as matrix
import def_validation

AUTODMP_ROOT = Path(__file__).resolve().parents[3]
if str(AUTODMP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTODMP_ROOT))
from dreamplace.ops.openroad_handoff import launcher as handoff_launcher


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_ROOT = Path("/nfs/share/home/zhaoxueyan/iccad24-benchmark")
DEFAULT_ARTIFACT_BASE = THIS_DIR / "artifacts" / "iccad24_asap7_buffer_only"
HANDOFF_MODES = handoff_launcher.HANDOFF_MODES
HANDOFF_RESTART_POLICIES = ("cold", "warm_schedule")
PLACE_IO_ENGINES = handoff_launcher.PLACE_IO_ENGINES

CASE_ORDER = (
    "NV_NVDLA_partition_m",
    "hidden1",
    "NV_NVDLA_partition_p",
    "ariane136",
    "hidden2",
    "mempool_tile_wrap",
    "hidden3",
    "hidden4",
    "aes_256",
    "hidden5",
)

ASAP7_RC_FILE = Path("setRC.tcl")
ASAP7_LEF_FILES = (
    Path("lef/asap7_tech_1x_201209.lef"),
    Path("lef/asap7sc7p5t_27_R_1x_201211.lef"),
    Path("lef/sram_asap7_16x256_1rw.lef"),
    Path("lef/sram_asap7_32x256_1rw.lef"),
    Path("lef/sram_asap7_64x256_1rw.lef"),
    Path("lef/sram_asap7_64x64_1rw.lef"),
)
ASAP7_LIB_FILES = (
    Path("lib/asap7sc7p5t_AO_RVT_FF_nldm_201020.lib"),
    Path("lib/asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib"),
    Path("lib/asap7sc7p5t_OA_RVT_FF_nldm_201020.lib"),
    Path("lib/asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib"),
    Path("lib/asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib"),
    Path("lib/sram_asap7_16x256_1rw.lib"),
    Path("lib/sram_asap7_32x256_1rw.lib"),
    Path("lib/sram_asap7_64x256_1rw.lib"),
    Path("lib/sram_asap7_64x64_1rw.lib"),
)

SRAM_RE = re.compile(r"\b(sram_asap7_[A-Za-z0-9_]+)\b")
DEF_COUNT_RE = re.compile(r"^\s*(COMPONENTS|PINS|NETS)\s+(\d+)\s*;")
DIEAREA_RE = re.compile(r"^\s*DIEAREA\s+(.+?)\s*;")


def _load_placeio_ieda():
    import dreamplace.ops.placeio_ieda.place_io as placeio_ieda

    return placeio_ieda


def load_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)


def require_existing(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise RuntimeError("missing required paths: %s" % ", ".join(missing))


def log_run_stage(stage, state, **fields):
    parts = ["stage=%s" % stage, "state=%s" % state]
    parts.extend("%s=%s" % (key, value) for key, value in sorted(fields.items()))
    logging.info(" ".join(parts))


@contextmanager
def logged_run_stage(stage, **fields):
    log_run_stage(stage, "begin", **fields)
    try:
        yield
    except Exception:
        log_run_stage(stage, "error", **fields)
        raise
    else:
        log_run_stage(stage, "end", **fields)


def _path_mtime_or_none(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def run_subprocess_with_stale_log_watchdog(
    command,
    log_path,
    stale_log_timeout_sec,
    poll_interval_sec=30,
    popen_cls=subprocess.Popen,
    now=time.monotonic,
    sleep=time.sleep,
    killpg=os.killpg,
    log_mtime_func=_path_mtime_or_none,
):
    start_time = now()
    last_progress_time = start_time
    last_log_mtime = log_mtime_func(log_path)
    process = popen_cls(command, start_new_session=True)

    while True:
        returncode = process.poll()
        if returncode is not None:
            return {
                "status": "completed",
                "returncode": returncode,
                "elapsed_sec": now() - start_time,
                "idle_sec": now() - last_progress_time,
            }

        current_log_mtime = log_mtime_func(log_path)
        if current_log_mtime is not None and current_log_mtime != last_log_mtime:
            last_log_mtime = current_log_mtime
            last_progress_time = now()

        idle_sec = now() - last_progress_time
        if idle_sec >= float(stale_log_timeout_sec):
            killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=30)
            return {
                "status": "stale_log_timeout",
                "returncode": returncode,
                "elapsed_sec": now() - start_time,
                "idle_sec": idle_sec,
                "stale_log_timeout_sec": float(stale_log_timeout_sec),
                "log_path": str(log_path),
            }

        sleep(poll_interval_sec)


def parse_csv(values):
    result = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def remap_workspace_case_path(path_value, benchmark_root=DEFAULT_BENCHMARK_ROOT):
    if not path_value:
        return path_value
    path_text = str(path_value)
    marker = "iccad24-benchmark/"
    marker_index = path_text.find(marker)
    if marker_index < 0:
        return path_text
    suffix = path_text[marker_index + len(marker) :]
    return str(Path(benchmark_root) / suffix)


def asap7_tech_bundle(asap7_root=None):
    asap7_root = Path(asap7_root or DEFAULT_BENCHMARK_ROOT / "ASAP7")
    tech_lef = asap7_root / ASAP7_LEF_FILES[0]
    lef_paths = [asap7_root / path for path in ASAP7_LEF_FILES[1:]]
    lib_paths = [asap7_root / path for path in ASAP7_LIB_FILES]
    rc_tcl = asap7_root / ASAP7_RC_FILE
    require_existing([tech_lef, rc_tcl] + lef_paths + lib_paths)
    return {
        "tech_lef": str(tech_lef),
        "lef_paths": [str(path) for path in lef_paths],
        "lib_paths": [str(path) for path in lib_paths],
        "rc_tcl": str(rc_tcl),
    }


def count_size_rows(size_path):
    if not Path(size_path).exists():
        return None
    count = 0
    with open(size_path, "r") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def parse_def_counts(def_path):
    counts = {}
    if not Path(def_path).exists():
        return counts
    with open(def_path, "r", errors="replace") as stream:
        for line in stream:
            count_match = DEF_COUNT_RE.match(line)
            if count_match:
                counts[count_match.group(1).lower()] = int(count_match.group(2))
                continue
            diearea_match = DIEAREA_RE.match(line)
            if diearea_match:
                counts["diearea"] = diearea_match.group(1).strip()
            if {"components", "pins", "nets"}.issubset(counts):
                break
    return counts


def count_sram_refs(verilog_path):
    counts = {}
    if not Path(verilog_path).exists():
        return counts
    with open(verilog_path, "r", errors="replace") as stream:
        for line in stream:
            for match in SRAM_RE.finditer(line):
                name = match.group(1)
                counts[name] = counts.get(name, 0) + 1
    return counts


def selected_sdc_path(case_dir, case_name, benchmark_root):
    design_path_json = case_dir / "workspace" / "config" / "design_path.json"
    if design_path_json.exists():
        configured = load_json(design_path_json).get("sdc_path")
        remapped = Path(remap_workspace_case_path(configured, benchmark_root))
        if remapped.exists():
            return str(remapped), "config"

    compat_sdc = (
        case_dir
        / "workspace"
        / "output"
        / "dreamplace"
        / "compat"
        / ("%s_optimizer_compat.sdc" % case_name)
    )
    if compat_sdc.exists():
        return str(compat_sdc), "compat"

    return str(case_dir / ("%s.sdc" % case_name)), "original"


def discover_case(case_name, benchmark_root=DEFAULT_BENCHMARK_ROOT):
    benchmark_root = Path(benchmark_root)
    case_dir = benchmark_root / "design" / case_name
    param_path = case_dir / "workspace" / "dreamplace_config" / "param.json"
    if not param_path.exists():
        fallback = case_dir / "workspace" / "config" / "dreamplace_config" / "param.json"
        param_path = fallback if fallback.exists() else param_path
    def_input = case_dir / ("%s.def" % case_name)
    verilog_input = case_dir / ("%s.v" % case_name)
    size_path = case_dir / ("%s.size" % case_name)
    sdc_path, sdc_kind = selected_sdc_path(case_dir, case_name, benchmark_root)

    require_existing([case_dir, param_path, def_input, verilog_input, sdc_path, size_path])
    def_counts = parse_def_counts(def_input)
    return {
        "design": case_name,
        "case_dir": str(case_dir),
        "workspace": str(case_dir / "workspace"),
        "param_path": str(param_path),
        "def_input": str(def_input),
        "verilog_input": str(verilog_input),
        "sdc_path": str(sdc_path),
        "sdc_kind": sdc_kind,
        "size_path": str(size_path),
        "component_count": count_size_rows(size_path),
        "def_counts": def_counts,
        "sram_refs": count_sram_refs(verilog_input),
    }


def discover_cases(benchmark_root=DEFAULT_BENCHMARK_ROOT, requested_cases=None):
    benchmark_root = Path(benchmark_root)
    design_root = benchmark_root / "design"
    if requested_cases:
        names = list(requested_cases)
    else:
        available = {path.name for path in design_root.iterdir() if path.is_dir()}
        names = [name for name in CASE_ORDER if name in available]
        extras = sorted(available.difference(names))
        names.extend(extras)
    cases = [discover_case(name, benchmark_root=benchmark_root) for name in names]
    return sorted(
        cases,
        key=lambda case: (
            CASE_ORDER.index(case["design"])
            if case["design"] in CASE_ORDER
            else len(CASE_ORDER),
            case.get("component_count") or 0,
            case["design"],
        ),
    )


def buffer_only_repair_command(margin):
    return handoff_launcher.buffer_only_repair_command(margin)


def install_buffer_only_margin_command_patch(margin):
    return handoff_launcher.install_buffer_only_margin_command_patch(margin)


def metric_float(metric, name):
    if metric is None or not hasattr(metric, name):
        return None
    value = getattr(metric, name)
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_gate_float(value):
    return handoff_launcher.optional_gate_float(value)


def install_iccad24_handoff_trigger(
    placedb,
    trigger_period=50,
    max_handoff_overflow=None,
):
    original_create = placedb.create_openroad_handoff_controller
    attempted_iterations = set()

    def gated_create(controller_params):
        controller = original_create(controller_params)
        if controller is None:
            return None

        def gated_should(event):
            iteration = event.get("absolute_iteration", event.get("iteration"))
            if iteration is None:
                return False, None
            iteration = int(iteration)
            if iteration <= 0 or iteration % int(trigger_period) != 0:
                return False, None
            if iteration in attempted_iterations:
                return False, None
            if max_handoff_overflow is not None:
                overflow = metric_float(event.get("metric"), "overflow")
                threshold = float(max_handoff_overflow)
                if overflow is None or not math.isfinite(overflow) or overflow > threshold:
                    return False, None
            attempted_iterations.add(iteration)
            reason = "periodic_iteration@period=%d@iter=%d" % (
                int(trigger_period),
                iteration,
            )
            if max_handoff_overflow is not None:
                reason += "@overflow<=%.12g" % float(max_handoff_overflow)
            return True, {
                "trigger_mode": "periodic_iteration",
                "trigger_reason": reason,
            }

        controller.should_handoff = gated_should
        return controller

    placedb.create_openroad_handoff_controller = gated_create


def variant_handoff_config(
    final_def,
    rc_tcl,
    handoff_mode="buffer-only",
    trigger_period=None,
    max_handoff_overflow=None,
):
    return handoff_launcher.build_openroad_handoff_config(
        output_def=final_def,
        rc_tcl=rc_tcl,
        handoff_mode=handoff_mode,
        trigger_period=trigger_period,
        max_handoff_overflow=max_handoff_overflow,
    )


def effective_target_iter(param_data, target_iter=None):
    if target_iter is not None:
        return int(target_iter), "override"
    stages = param_data.get("global_place_stages") or []
    if not stages or "iteration" not in stages[0]:
        raise RuntimeError("missing native global_place_stages[0].iteration")
    return int(stages[0]["iteration"]), "native"


def install_real_ieda_binding():
    for module_name in (
        "third_party.iEDA.bin.ieda_py",
        "third_party.iEDA.bin",
        "third_party.iEDA",
    ):
        sys.modules.pop(module_name, None)
    from third_party.iEDA.bin import ieda_py as real_ieda

    placeio_ieda = _load_placeio_ieda()
    return placeio_ieda.bind_ieda_module(real_ieda)


def prepare_ieda_workspace(case_config, tech_bundle, run_dir):
    workspace = Path(run_dir) / "workspace"
    source_config = Path(case_config["workspace"]) / "config"
    target_config = workspace / "config"
    if target_config.exists():
        shutil.rmtree(target_config)
    shutil.copytree(source_config, target_config)

    design_path = target_config / "design_path.json"
    design_data = load_json(design_path)
    design_data["def_input_path"] = case_config["def_input"]
    design_data["verilog_input_path"] = case_config["verilog_input"]
    design_data["sdc_path"] = case_config["sdc_path"]
    design_data.setdefault("spef_path", "")
    design_data.setdefault("mp_tcl", "")
    write_json(design_path, design_data)

    tech_path = target_config / "tech_path.json"
    tech_data = load_json(tech_path)
    tech_data["tech_lef_path"] = tech_bundle["tech_lef"]
    tech_data["lef_paths"] = list(tech_bundle["lef_paths"])
    tech_data["lib_paths"] = list(tech_bundle["lib_paths"])
    write_json(tech_path, tech_data)

    db_config_path = target_config / "iEDA_config" / "db_default_config.json"
    db_config = load_json(db_config_path)
    input_config = db_config.setdefault("INPUT", {})
    input_config["tech_lef_path"] = tech_bundle["tech_lef"]
    input_config["lef_paths"] = list(tech_bundle["lef_paths"])
    input_config["def_path"] = case_config["def_input"]
    input_config["verilog_path"] = case_config["verilog_input"]
    input_config["lib_path"] = list(tech_bundle["lib_paths"])
    input_config["sdc_path"] = case_config["sdc_path"]
    db_config.setdefault("OUTPUT", {})["output_dir_path"] = str(
        workspace / "output" / "iEDA" / "result"
    )
    write_json(db_config_path, db_config)

    return workspace


def build_params(
    params_cls,
    case_config,
    tech_bundle,
    run_dir,
    final_def,
    target_iter=None,
    plot_interval=None,
    with_sta=False,
    handoff_mode="buffer-only",
    handoff_restart_policy="cold",
    target_density_override=None,
    stop_overflow_override=None,
    preserve_routability_opt=False,
    place_io_engine="openroad",
    trigger_period=None,
    max_handoff_overflow=None,
):
    params = params_cls()
    params.fromJson(matrix.load_default_params_dict())
    param_data = load_json(case_config["param_path"])
    params.fromJson(copy.deepcopy(param_data))

    stages = copy.deepcopy(getattr(params, "global_place_stages", []) or [])
    if not stages:
        if target_iter is None:
            raise RuntimeError(
                "global_place_stages is empty and --target-iter was not provided"
            )
        stages = [{"iteration": int(target_iter)}]
    elif target_iter is not None:
        stages[0]["iteration"] = int(target_iter)

    params.global_place_stages = stages
    params.place_io_engine = place_io_engine
    params.gpu = 0
    params.with_sta = 1 if with_sta else 0
    params.handoff_restart_policy = handoff_restart_policy
    if not preserve_routability_opt:
        params.routability_opt_flag = 0
    if target_density_override is not None:
        params.target_density = float(target_density_override)
    if stop_overflow_override is not None:
        params.stop_overflow = float(stop_overflow_override)
    params.result_dir = str(run_dir)
    params.base_design_name = "iccad24_asap7_%s_%s" % (
        case_config["design"],
        handoff_mode.replace("-", "_"),
    )
    if plot_interval is not None:
        params.plot_interval = int(plot_interval)

    params.design_inputs = {
        "tech_lef": [tech_bundle["tech_lef"]],
        "lef": list(tech_bundle["lef_paths"]),
        "def": case_config["def_input"],
        "sdc": case_config["sdc_path"],
        "lib": list(tech_bundle["lib_paths"]),
        "rc_tcl": tech_bundle["rc_tcl"],
    }
    params.lef_input = list(params.design_inputs["tech_lef"]) + list(
        params.design_inputs["lef"]
    )
    params.def_input = params.design_inputs["def"]
    params.verilog_input = case_config["verilog_input"]
    params.openroad_handoff = variant_handoff_config(
        final_def,
        tech_bundle["rc_tcl"],
        handoff_mode=handoff_mode,
        trigger_period=trigger_period,
        max_handoff_overflow=max_handoff_overflow,
    )
    return params, param_data


def format_float_label(value):
    text = "%.12g" % float(value)
    return text.replace("-", "m").replace(".", "p")


def artifact_root_for(
    design,
    artifact_base,
    effective_iter,
    iteration_source,
    trigger_period,
    margin,
    with_sta=False,
    handoff_mode="buffer-only",
    handoff_restart_policy="cold",
    target_density_override=None,
    stop_overflow_override=None,
    preserve_routability_opt=False,
):
    iter_label = (
        "native%d" % int(effective_iter)
        if iteration_source == "native"
        else "iter%d" % int(effective_iter)
    )
    sta_label = "_sta1" if with_sta else ""
    override_label = ""
    if target_density_override is not None:
        override_label += "_td%s" % format_float_label(target_density_override)
    if stop_overflow_override is not None:
        override_label += "_so%s" % format_float_label(stop_overflow_override)
    if handoff_restart_policy != "cold":
        override_label += "_rst%s" % handoff_restart_policy
    if preserve_routability_opt:
        override_label += "_roptnative"
    return (
        Path(artifact_base)
        / (
            "%s_iccad24_asap7_%s_%s%s_period%d_margin%.12g%s"
            % (
                design,
                handoff_mode.replace("-", "_"),
                iter_label,
                sta_label,
                int(trigger_period),
                float(margin),
                override_label,
            )
        )
    )


def make_run_dir(artifact_root, design):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(artifact_root) / ("%s_%s" % (design, timestamp))
    if run_dir.exists():
        run_dir = Path("%s_%d" % (run_dir, os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def no_buffer_reason(summary):
    session = summary.get("session", {}) or {}
    churn = session.get("buffer_churn_totals", {}) or {}
    diagnostics = summary.get("openroad_diagnostic_evidence", {}) or {}
    labels = diagnostics.get("diagnostic_labels", []) or []
    if summary.get("failure"):
        failure = summary["failure"]
        return "run_failure:%s:%s" % (failure.get("type"), failure.get("message"))
    if int(churn.get("added_buffer_count", 0) or 0) > 0:
        return "inserted_buffer"
    if int(session.get("event_count", 0) or 0) <= 0:
        return "no_handoff_event"
    if session.get("status") != "completed":
        return "handoff_not_completed:%s" % session.get("status")
    if "insufficient_timing_evidence" in labels:
        return "insufficient_timing_evidence"
    if "clean_no_buffer_churn" in labels:
        return "clean_no_buffer_churn"
    return "repair_timing_completed_without_buffer_churn"


def smoke_passed(summary):
    session = summary.get("session", {}) or {}
    return (
        summary.get("failure") is None
        and int(session.get("event_count", 0) or 0) > 0
        and session.get("status") == "completed"
    )


def run_one(
    design,
    benchmark_root=DEFAULT_BENCHMARK_ROOT,
    artifact_base=DEFAULT_ARTIFACT_BASE,
    target_iter=None,
    trigger_period=50,
    margin=0,
    plot_interval=None,
    with_sta=False,
    handoff_mode="buffer-only",
    handoff_restart_policy="cold",
    target_density_override=None,
    stop_overflow_override=None,
    max_handoff_overflow=0.2,
    preserve_routability_opt=False,
    place_io_engine="openroad",
):
    case_config = discover_case(design, benchmark_root=benchmark_root)
    param_data_for_iter = load_json(case_config["param_path"])
    effective_iter, iteration_source = effective_target_iter(
        param_data_for_iter,
        target_iter=target_iter,
    )
    tech_bundle = asap7_tech_bundle(Path(benchmark_root) / "ASAP7")
    artifact_root = artifact_root_for(
        case_config["design"],
        artifact_base,
        effective_iter,
        iteration_source,
        trigger_period,
        margin,
        with_sta=with_sta,
        handoff_mode=handoff_mode,
        handoff_restart_policy=handoff_restart_policy,
        target_density_override=target_density_override,
        stop_overflow_override=stop_overflow_override,
        preserve_routability_opt=preserve_routability_opt,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(artifact_root, case_config["design"])
    log_path = run_dir / "run.log"
    final_def = run_dir / (
        "%s_iccad24_asap7_%s.def"
        % (case_config["design"], handoff_mode.replace("-", "_"))
    )
    data_manager_workspace = run_dir
    if place_io_engine == "ieda":
        data_manager_workspace = prepare_ieda_workspace(
            case_config,
            tech_bundle,
            run_dir,
        )

    Params, Placer = matrix.prepare_runtime_modules(str(run_dir / "matplotlib"))
    if place_io_engine == "ieda":
        install_real_ieda_binding()
    margin_command = install_buffer_only_margin_command_patch(margin)
    params, original_param_data = build_params(
        Params,
        case_config,
        tech_bundle,
        run_dir,
        str(final_def),
        target_iter=target_iter,
        plot_interval=plot_interval,
        with_sta=with_sta,
        handoff_mode=handoff_mode,
        handoff_restart_policy=handoff_restart_policy,
        target_density_override=target_density_override,
        stop_overflow_override=stop_overflow_override,
        preserve_routability_opt=preserve_routability_opt,
        place_io_engine=place_io_engine,
        trigger_period=trigger_period,
        max_handoff_overflow=max_handoff_overflow,
    )

    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    handoff_strategy = matrix.build_handoff_strategy_summary(
        params,
        placeio_openroad.PlaceIOFunction.build_buffer_insertion_command,
        target_iter=effective_iter,
        trigger_period=trigger_period,
    )

    start = time.time()
    placement_result = None
    event_history = []
    failure = None
    final_def_validation = {"status": "not_run", "summary": None, "failure": ""}

    mirror_run_output = place_io_engine != "ieda"
    with matrix.StdoutStderrTee(
        str(log_path),
        mirror_to_stdout=mirror_run_output,
    ):
        matrix.configure_regression_logging()
        logging.info("design=%s", case_config["design"])
        logging.info("benchmark_root=%s", benchmark_root)
        logging.info("param_path=%s", case_config["param_path"])
        logging.info("tech_lef=%s", tech_bundle["tech_lef"])
        logging.info("rc_tcl=%s", tech_bundle["rc_tcl"])
        logging.info("def_input=%s", case_config["def_input"])
        logging.info("verilog_input=%s", case_config["verilog_input"])
        logging.info("sdc_path=%s", case_config["sdc_path"])
        logging.info("sdc_kind=%s", case_config["sdc_kind"])
        logging.info("margin_command=%s", margin_command)
        logging.info("place_io_engine=%s", place_io_engine)
        logging.info("trigger_period=%s target_iter=%s", trigger_period, target_iter)
        logging.info("max_handoff_overflow=%s", max_handoff_overflow)
        logging.info("run_dir=%s", run_dir)
        logging.info("data_manager_workspace=%s", data_manager_workspace)
        engine = None
        try:
            if place_io_engine == "ieda":
                with logged_run_stage(
                    "ieda_read_def",
                    place_io_engine=place_io_engine,
                    run_dir=run_dir,
                ):
                    placeio_ieda = _load_placeio_ieda()
                    ieda_io = placeio_ieda.PlaceIOFunction.make_io(
                        str(data_manager_workspace),
                        input_def=case_config["def_input"],
                        input_verilog=case_config["verilog_input"],
                    )
                    ieda_io.read_def(case_config["def_input"], read_verilog=False)
            with logged_run_stage(
                "placement_engine_init",
                place_io_engine=place_io_engine,
                run_dir=run_dir,
            ):
                engine = Placer.PlacementEngine(params)
            with logged_run_stage(
                "setup_rawdb",
                place_io_engine=place_io_engine,
                run_dir=run_dir,
            ):
                engine.setup_rawdb(
                    data_manager=types.SimpleNamespace(
                        dir_workspace=str(data_manager_workspace)
                    )
                )
            with logged_run_stage(
                "engine_run",
                place_io_engine=place_io_engine,
                run_dir=run_dir,
            ):
                placement_result = engine.run()
            if placement_result is not None:
                with logged_run_stage(
                    "write_back",
                    place_io_engine=place_io_engine,
                    run_dir=run_dir,
                ):
                    engine.write_back(str(final_def))
            session = getattr(engine.placedb, "handoff_session", None)
            if session is not None:
                event_history = copy.deepcopy(session.event_history)
        except Exception as exc:
            logging.exception("iccad24 asap7 buffer-only run failed")
            if engine is not None and getattr(engine, "placedb", None) is not None:
                session = getattr(engine.placedb, "handoff_session", None)
                if session is not None:
                    event_history = copy.deepcopy(session.event_history)
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    if final_def.exists():
        try:
            final_def_validation["summary"] = (
                def_validation.validate_final_def_coordinates(final_def)
            )
            final_def_validation["status"] = "passed"
        except def_validation.DefCoordinateValidationError as exc:
            final_def_validation["summary"] = (
                def_validation.parse_def_placement_summary(final_def)
            )
            final_def_validation["status"] = "failed"
            final_def_validation["failure"] = str(exc)

    elapsed = time.time() - start
    run_log_text = matrix.read_text(str(log_path))
    session_summary = matrix.build_session_summary(event_history)
    summary = {
        "design": case_config["design"],
        "variant": handoff_mode,
        "handoff_mode": handoff_mode,
        "with_sta": bool(with_sta),
        "place_io_engine": place_io_engine,
        "elapsed_sec": elapsed,
        "native_target_iter": int(
            (original_param_data.get("global_place_stages") or [{}])[0].get(
                "iteration", effective_iter
            )
        ),
        "effective_target_iter": int(effective_iter),
        "iteration_source": iteration_source,
        "target_iter": int(effective_iter),
        "trigger_period": int(trigger_period),
        "max_handoff_overflow": (
            float(max_handoff_overflow)
            if max_handoff_overflow is not None
            else None
        ),
        "setup_margin": float(margin),
        "case": case_config,
        "tech_bundle": tech_bundle,
        "config": {
            "param_path": case_config["param_path"],
            "with_sta": bool(with_sta),
            "place_io_engine": place_io_engine,
            "handoff_restart_policy": handoff_restart_policy,
            "plot_interval": getattr(params, "plot_interval", None),
            "handoff_trigger": {
                "trigger_period": int(trigger_period),
                "max_handoff_overflow": (
                    float(max_handoff_overflow)
                    if max_handoff_overflow is not None
                    else None
                ),
            },
            "diagnostic_param_overrides": {
                key: value
                for key, value in {
                    "target_density": (
                        float(target_density_override)
                        if target_density_override is not None
                        else None
                    ),
                    "stop_overflow": (
                        float(stop_overflow_override)
                        if stop_overflow_override is not None
                        else None
                    ),
                }.items()
                if value is not None
            },
            "preserve_routability_opt": bool(preserve_routability_opt),
            "normal_param_snapshot": matrix.normal_param_snapshot(original_param_data),
        },
        "handoff_strategy": handoff_strategy,
        "placement_result": placement_result,
        "last_iteration": matrix.parse_last_iteration(run_log_text),
        "session": session_summary,
        "openroad_diagnostic_evidence": matrix.extract_openroad_diagnostic_evidence(
            run_log_text, event_history
        ),
        "failure": failure,
        "final_def": str(final_def),
        "final_def_exists": final_def.exists(),
        "final_def_validation": final_def_validation,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "manifest_path": str(run_dir / "run_manifest.json"),
        "summary_path": str(run_dir / "summary.json"),
        "interesting_log_lines_tail": matrix.matching_log_lines(
            run_log_text,
            matrix.INTERESTING_LOG_PATTERNS,
        ),
        "error_log_matches": matrix.matching_log_lines(
            run_log_text,
            matrix.ERROR_LOG_PATTERNS,
        ),
    }
    summary["no_buffer_reason"] = no_buffer_reason(summary)

    write_json(summary["summary_path"], summary)
    write_json(summary["manifest_path"], summary)
    write_json(
        artifact_root / "latest_run.json",
        {
            "run_dir": summary["run_dir"],
            "summary_path": summary["summary_path"],
        },
    )
    return summary


def build_manifest(benchmark_root=DEFAULT_BENCHMARK_ROOT, requested_cases=None):
    benchmark_root = Path(benchmark_root)
    cases = discover_cases(benchmark_root=benchmark_root, requested_cases=requested_cases)
    return {
        "benchmark_root": str(benchmark_root),
        "asap7": asap7_tech_bundle(benchmark_root / "ASAP7"),
        "case_count": len(cases),
        "cases": cases,
    }


def print_run_line(summary):
    session = summary.get("session", {}) or {}
    churn = session.get("buffer_churn_totals", {}) or {}
    mutations = session.get("mutation_kind_counts", {}) or {}
    print(
        (
            "design=%s status=%s failure=%s events=%s last_iter=%s "
            "topology=%s added=%s removed=%s reason=%s run_dir=%s"
        )
        % (
            summary.get("design"),
            session.get("status"),
            summary.get("failure"),
            session.get("event_count"),
            summary.get("last_iteration"),
            mutations.get("topology_changed"),
            churn.get("added_buffer_count"),
            churn.get("removed_buffer_count"),
            summary.get("no_buffer_reason"),
            summary.get("run_dir"),
        ),
        flush=True,
    )


def write_combined(path, summaries, manifest):
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest,
        "summaries": summaries,
    }
    write_json(path, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--artifact-base", type=Path, default=DEFAULT_ARTIFACT_BASE)
    parser.add_argument("--design", dest="designs", action="append")
    parser.add_argument("--smoke-design", default="NV_NVDLA_partition_m")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--target-iter",
        type=int,
        default=None,
        help="Override native DreamPlace iteration count. Omit for full native run.",
    )
    parser.add_argument("--trigger-period", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--plot-interval", type=int)
    parser.add_argument("--with-sta", action="store_true")
    parser.add_argument("--place-io-engine", choices=PLACE_IO_ENGINES, default="openroad")
    parser.add_argument("--handoff-mode", choices=HANDOFF_MODES, default="buffer-only")
    parser.add_argument(
        "--handoff-restart-policy",
        choices=HANDOFF_RESTART_POLICIES,
        default="cold",
    )
    parser.add_argument("--target-density-override", type=float)
    parser.add_argument("--stop-overflow-override", type=float)
    parser.add_argument("--max-handoff-overflow", type=optional_gate_float, default=0.2)
    parser.add_argument("--preserve-routability-opt", action="store_true")
    parser.add_argument("--combined-output")
    args = parser.parse_args(argv)

    if args.target_iter is not None and args.target_iter <= 0:
        parser.error("--target-iter must be positive")
    if args.trigger_period <= 0:
        parser.error("--trigger-period must be positive")
    if args.plot_interval is not None and args.plot_interval <= 0:
        parser.error("--plot-interval must be positive")
    if args.target_density_override is not None and not (
        0 < args.target_density_override <= 1
    ):
        parser.error("--target-density-override must be in (0, 1]")
    if args.stop_overflow_override is not None and not (
        0 <= args.stop_overflow_override <= 1
    ):
        parser.error("--stop-overflow-override must be in [0, 1]")
    if args.max_handoff_overflow is not None and not (
        0 <= args.max_handoff_overflow <= 1
    ):
        parser.error("--max-handoff-overflow must be in [0, 1]")

    requested_cases = parse_csv(args.designs or [])
    manifest = build_manifest(
        benchmark_root=args.benchmark_root,
        requested_cases=requested_cases or None,
    )
    if args.manifest_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    args.artifact_base.mkdir(parents=True, exist_ok=True)
    combined_output = Path(args.combined_output) if args.combined_output else (
        args.artifact_base / "iccad24_asap7_buffer_only_combined_summary.json"
    )
    designs = [case["design"] for case in manifest["cases"]]
    summaries = []

    smoke_summary = None
    if not args.skip_smoke:
        smoke_summary = run_one(
            args.smoke_design,
            benchmark_root=args.benchmark_root,
            artifact_base=args.artifact_base,
            target_iter=args.target_iter,
            trigger_period=args.trigger_period,
            margin=args.margin,
            plot_interval=args.plot_interval,
            with_sta=args.with_sta,
            handoff_mode=args.handoff_mode,
            handoff_restart_policy=args.handoff_restart_policy,
            target_density_override=args.target_density_override,
            stop_overflow_override=args.stop_overflow_override,
            max_handoff_overflow=args.max_handoff_overflow,
            preserve_routability_opt=args.preserve_routability_opt,
            place_io_engine=args.place_io_engine,
        )
        summaries.append(smoke_summary)
        print_run_line(smoke_summary)
        write_combined(combined_output, summaries, manifest)
        if not smoke_passed(smoke_summary):
            print(
                "smoke_failed: stopping before full run; combined_summary=%s"
                % combined_output,
                file=sys.stderr,
            )
            return 2

    for design in designs:
        if smoke_summary is not None and design == args.smoke_design:
            continue
        summary = run_one(
            design,
            benchmark_root=args.benchmark_root,
            artifact_base=args.artifact_base,
            target_iter=args.target_iter,
            trigger_period=args.trigger_period,
            margin=args.margin,
            plot_interval=args.plot_interval,
            with_sta=args.with_sta,
            handoff_mode=args.handoff_mode,
            handoff_restart_policy=args.handoff_restart_policy,
            target_density_override=args.target_density_override,
            stop_overflow_override=args.stop_overflow_override,
            max_handoff_overflow=args.max_handoff_overflow,
            preserve_routability_opt=args.preserve_routability_opt,
            place_io_engine=args.place_io_engine,
        )
        summaries.append(summary)
        print_run_line(summary)
        write_combined(combined_output, summaries, manifest)

    print("combined_summary=%s" % combined_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
