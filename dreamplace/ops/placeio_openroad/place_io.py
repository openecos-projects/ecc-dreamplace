from torch.autograd import Function
import glob
import importlib.util
import os
import time
from pathlib import Path


def _load_placeio_openroad_cpp():
    module_name = "dreamplace.ops.placeio_openroad.placeio_openroad_cpp"
    repo_root = Path(__file__).resolve().parents[3]
    search_patterns = []
    explicit_path = os.environ.get("DREAMPLACE_PLACEIO_OPENROAD_CPP", "").strip()
    if explicit_path:
        search_patterns.append(explicit_path)
    build_dir = os.environ.get("DREAMPLACE_PLACEIO_OPENROAD_BUILD_DIR", "").strip()
    if build_dir:
        search_patterns.append(os.path.join(build_dir, "**", "placeio_openroad_cpp*.so"))
    search_patterns.extend(
        [
            str(repo_root / "build_codex" / "**" / "placeio_openroad_cpp*.so"),
            str(repo_root / "build" / "**" / "placeio_openroad_cpp*.so"),
            str(repo_root / "build_openroad_schema_check" / "**" / "placeio_openroad_cpp*.so"),
        ]
    )

    candidates = []
    for pattern in search_patterns:
        if os.path.isfile(pattern):
            candidates.append(pattern)
        else:
            candidates.extend(glob.glob(pattern, recursive=True))
    candidates = sorted(set(candidates), key=lambda path: os.path.getmtime(path), reverse=True)
    for candidate in candidates:
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        import dreamplace.ops.placeio_openroad.placeio_openroad_cpp as module
        return module
    except (ImportError, OSError):
        pass

    raise ImportError(
        "Cannot import placeio_openroad_cpp. Build placeio_openroad_cpp or set "
        "DREAMPLACE_PLACEIO_OPENROAD_CPP/DREAMPLACE_PLACEIO_OPENROAD_BUILD_DIR."
    )


placeio_openroad_cpp = _load_placeio_openroad_cpp()

DEFAULT_PLACE_IO_ENGINE = "openroad"
DEFAULT_OPENROAD_VT_SUFFIXES = ("H7H", "H7R", "H7L")
PLACE_IO_ENGINE = os.environ.get("PLACE_IO_ENGINE", "").strip().lower()
SUPPORTED_OPENROAD_REFRESH_MODES = ("topo", "all")
DEFAULT_BUFFER_INSERTION_STRATEGY = "repair_design"
EXPERIMENTAL_BUFFER_ONLY_STRATEGY = "buffer-only"
EXPERIMENTAL_BUFFER_ONLY_COMMAND = (
    'repair_timing -setup -sequence "unbuffer,buffer,split" '
    "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap"
)
PRE_REPAIR_TCL_OPTION = "pre_repair_tcl"
_BUFFER_INSERTION_STRATEGY_ALIASES = {
    "repair_design": {
        "profile": "repair_design",
        "command": "repair_design",
        "kind": "baseline",
        "experimental": False,
        "validation": "default_baseline",
    },
    "buffer-only": {
        "profile": EXPERIMENTAL_BUFFER_ONLY_STRATEGY,
        "command": EXPERIMENTAL_BUFFER_ONLY_COMMAND,
        "kind": "experimental",
        "experimental": True,
        "validation": "log_validated_only",
    },
    "buffer_only": {
        "profile": EXPERIMENTAL_BUFFER_ONLY_STRATEGY,
        "command": EXPERIMENTAL_BUFFER_ONLY_COMMAND,
        "kind": "experimental",
        "experimental": True,
        "validation": "log_validated_only",
    },
}


def _normalize_buffer_insertion_strategy_alias(strategy):
    return strategy.strip().lower().replace(" ", "_")


def _validate_openroad_refresh_mode(refresh_mode):
    mode = str(refresh_mode or "topo").strip().lower()
    if mode not in SUPPORTED_OPENROAD_REFRESH_MODES:
        raise RuntimeError(
            "unsupported OpenROAD refresh_mode=%r; expected one of %s"
            % (refresh_mode, ", ".join(SUPPORTED_OPENROAD_REFRESH_MODES))
        )
    return mode


def _normalize_openroad_vt_suffixes(value):
    if value is None:
        value = DEFAULT_OPENROAD_VT_SUFFIXES
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    suffixes = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not suffixes:
        raise RuntimeError("openroad_vt_suffixes must contain at least one suffix")
    return suffixes


class SolutionFileFormat(object):
    BOOKSHELF = 0
    DEF = 1


class PlaceIOFunction(Function):
    last_read_profile = {}

    @staticmethod
    def describe_buffer_insertion_strategy(strategy=None):
        strategy = (strategy or DEFAULT_BUFFER_INSERTION_STRATEGY).strip()
        if not strategy:
            raise RuntimeError("buffer insertion strategy must not be empty")
        profile = _BUFFER_INSERTION_STRATEGY_ALIASES.get(
            _normalize_buffer_insertion_strategy_alias(strategy)
        )
        if profile is not None:
            return dict(profile)
        return {
            "profile": strategy,
            "command": strategy,
            "kind": "custom",
            "experimental": False,
            "validation": "custom_unvalidated",
        }

    @staticmethod
    def _format_tcl_value(value):
        return "{%s}" % value

    @staticmethod
    def _build_tcl_command(strategy, options=None):
        strategy = (strategy or "").strip()
        if not strategy:
            raise RuntimeError("buffer insertion strategy must not be empty")

        options = options or {}
        cmd = [strategy]
        for key, value in options.items():
            if value is None or value is False:
                continue
            # Preserve the caller-provided Tcl flag spelling because OpenROAD
            # commands in this tree use mixed conventions such as -slew_margin.
            flag = "-%s" % str(key)
            if value is True:
                cmd.append(flag)
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    cmd.extend([flag, PlaceIOFunction._format_tcl_value(str(item))])
                continue
            cmd.extend([flag, PlaceIOFunction._format_tcl_value(str(value))])
        return " ".join(cmd)

    @staticmethod
    def _split_pre_repair_tcl_options(options=None):
        command_options = dict(options or {})
        pre_repair_tcl = command_options.pop(PRE_REPAIR_TCL_OPTION, None)
        if pre_repair_tcl is None:
            return [], command_options
        if isinstance(pre_repair_tcl, str):
            pre_commands = [pre_repair_tcl]
        else:
            pre_commands = list(pre_repair_tcl)
        return [
            str(command).strip()
            for command in pre_commands
            if command is not None and str(command).strip()
        ], command_options

    @staticmethod
    def build_buffer_insertion_command(strategy, options=None):
        pre_commands, command_options = (
            PlaceIOFunction._split_pre_repair_tcl_options(options)
        )
        profile = PlaceIOFunction.describe_buffer_insertion_strategy(strategy)
        if profile["profile"] == EXPERIMENTAL_BUFFER_ONLY_STRATEGY:
            main_command = profile["command"]
        else:
            main_command = PlaceIOFunction._build_tcl_command(
                profile["command"],
                command_options,
            )
        return "\n".join(pre_commands + [main_command])

    @staticmethod
    def build_one_net_buffer_command(net_name, options=None):
        net_name = str(net_name or "").strip()
        if not net_name:
            raise RuntimeError("one-net buffer insertion requires a net name")
        options = options or {}
        max_wire_length = options.get("max_wire_length", 0)
        slew_margin = options.get("slew_margin", 0)
        cap_margin = options.get("cap_margin", 0)
        return "rsz::repair_net_cmd [get_net %s] %s %s %s" % (
            PlaceIOFunction._format_tcl_value(net_name),
            max_wire_length,
            slew_margin,
            cap_margin,
        )


    @staticmethod
    def read(params):
        profile = {}
        args = "DREAMPlace"
        # OpenROAD is design-inputs-first: workspace resolution happens before
        # this adapter, then params.design_inputs carries tech LEF, LEF, DEF,
        # Liberty, SDC, and optional RC Tcl paths into OpenROAD.
        design_inputs = getattr(params, "design_inputs", None) or {}
        place_io_engine = PLACE_IO_ENGINE or getattr(
            params, "place_io_engine", DEFAULT_PLACE_IO_ENGINE
        )
        num_threads = int(getattr(params, "num_threads", 8) or 8)
        args += " --num_threads %d" % max(1, num_threads)
        for suffix in _normalize_openroad_vt_suffixes(
            getattr(params, "openroad_vt_suffixes", None)
        ):
            args += " --vt_suffix %s" % suffix
        profile["openroad_vt_suffixes"] = list(
            _normalize_openroad_vt_suffixes(
                getattr(params, "openroad_vt_suffixes", None)
            )
        )

        if place_io_engine == "openroad" and not design_inputs:
            raise RuntimeError(
                "OpenROAD place_io requires params.design_inputs from workspace paths; "
                "got empty design_inputs"
            )

        lef_inputs = []
        tech_lef = design_inputs.get("tech_lef")
        if tech_lef:
            if isinstance(tech_lef, list):
                lef_inputs.extend(tech_lef)
            else:
                lef_inputs.append(tech_lef)

        data_lefs = design_inputs.get("lef")
        if data_lefs:
            if isinstance(data_lefs, list):
                lef_inputs.extend(data_lefs)
            else:
                lef_inputs.append(data_lefs)

        if not lef_inputs and "lef_input" in params.__dict__:
            lef_inputs = params.lef_input
        if lef_inputs:
            if isinstance(lef_inputs, list):
                for lef in lef_inputs:
                    args += " --lef_input %s" % (lef)
            else:
                args += " --lef_input %s" % (lef_inputs)

        def_input = design_inputs.get("def")
        if not def_input and "def_input" in params.__dict__:
            def_input = params.def_input
        if def_input:
            args += " --def_input %s" % (def_input)

        liberty_inputs = design_inputs.get("lib")
        if liberty_inputs:
            if isinstance(liberty_inputs, list):
                for liberty in liberty_inputs:
                    args += " --lib_input %s" % (liberty)
            else:
                args += " --lib_input %s" % (liberty_inputs)

        sdc_input = design_inputs.get("sdc")
        if sdc_input:
            args += " --sdc_input %s" % (sdc_input)

        if place_io_engine == "openroad":
            if not lef_inputs:
                raise RuntimeError(
                    "OpenROAD place_io requires design_inputs['tech_lef']/['lef'] (workspace LEF paths)"
                )
            if not def_input:
                raise RuntimeError(
                    "OpenROAD place_io requires design_inputs['def'] (workspace DEF path)"
                )
        stage_start = time.perf_counter()
        raw_db = placeio_openroad_cpp.forward(args.split(" "))
        profile["setup_rawdb_cpp_forward_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        if place_io_engine == "openroad":
            stage_start = time.perf_counter()
            raw_db.eval_tcl_string("set_ideal_network [all_clocks]")
            profile["setup_rawdb_set_ideal_network_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000.0
        rc_tcl = design_inputs.get("rc_tcl")
        if place_io_engine == "openroad" and rc_tcl:
            stage_start = time.perf_counter()
            raw_db.eval_tcl_string(
                "source %s" % PlaceIOFunction._format_tcl_value(str(rc_tcl))
            )
            profile["setup_rawdb_source_rc_tcl_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000.0
            stage_start = time.perf_counter()
            raw_db.eval_tcl_string("estimate_parasitics -placement")
            profile["setup_rawdb_estimate_parasitics_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000.0
            design_inputs["parasitics_initialization"] = "placement"
            design_inputs["rc_tcl_configured"] = True
        elif place_io_engine == "openroad":
            design_inputs["parasitics_initialization"] = "none"
            design_inputs["rc_tcl_configured"] = False
        PlaceIOFunction.last_read_profile = profile
        return raw_db

    @staticmethod
    def pydb(raw_db):
        pydb = placeio_openroad_cpp.pydb(raw_db)
        export_profile = getattr(pydb, "export_profile", None)
        if isinstance(export_profile, dict):
            PlaceIOFunction.last_read_profile.update(export_profile)
        return pydb

    @staticmethod
    def sync_from_openroad(raw_db, refresh_mode="topo", return_summary=False):
        mode = _validate_openroad_refresh_mode(refresh_mode)
        stage_start = time.perf_counter()
        pydb = raw_db.sync_from_openroad()
        summary = {
            "status": "ok",
            "requested_refresh_mode": mode,
            "effective_refresh_mode": mode,
            "source": "raw_db.sync_from_openroad",
            "elapsed_ms": (time.perf_counter() - stage_start) * 1000.0,
        }
        if return_summary:
            return pydb, summary
        return pydb

    @staticmethod
    def export_topology_snapshot(raw_db):
        return raw_db.export_pydb_view()

    @staticmethod
    def refresh_topology_snapshot(raw_db):
        return raw_db.sync_from_openroad()

    @staticmethod
    def write(raw_db, filename, sol_file_format, node_x, node_y):
        return placeio_openroad_cpp.write(raw_db, filename, sol_file_format, node_x, node_y)

    @staticmethod
    def apply(raw_db, node_x, node_y):
        return placeio_openroad_cpp.apply(raw_db, node_x, node_y)

    @staticmethod
    def apply_sizing(raw_db, inst_cell_ids, cell_master_names):
        if hasattr(raw_db, "apply_sizing"):
            return raw_db.apply_sizing(inst_cell_ids, cell_master_names)
        return placeio_openroad_cpp.apply_sizing(
            raw_db, inst_cell_ids, cell_master_names
        )

    @staticmethod
    def query_diff_guided_batch_timing_metrics(raw_db):
        if hasattr(raw_db, "query_diff_guided_batch_timing_metrics"):
            return raw_db.query_diff_guided_batch_timing_metrics()
        return placeio_openroad_cpp.query_diff_guided_batch_timing_metrics(raw_db)

    @staticmethod
    def query_diff_guided_batch_timing_samples(raw_db, sample_request=None):
        sample_request = {} if sample_request is None else sample_request
        if hasattr(raw_db, "query_diff_guided_batch_timing_samples"):
            return raw_db.query_diff_guided_batch_timing_samples(sample_request)
        return placeio_openroad_cpp.query_diff_guided_batch_timing_samples(
            raw_db, sample_request
        )

    @staticmethod
    def query_diff_guided_batch_dynamic_conflict_signature(raw_db, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "query_diff_guided_batch_dynamic_conflict_signature"):
            return raw_db.query_diff_guided_batch_dynamic_conflict_signature(config)
        return placeio_openroad_cpp.query_diff_guided_batch_dynamic_conflict_signature(
            raw_db, config
        )

    @staticmethod
    def evaluate_diff_guided_batch_actions(raw_db, actions, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "evaluate_diff_guided_batch_actions"):
            return raw_db.evaluate_diff_guided_batch_actions(actions, config)
        return placeio_openroad_cpp.evaluate_diff_guided_batch_actions(
            raw_db, actions, config
        )

    @staticmethod
    def apply_diff_guided_batch_transaction(raw_db, action_results, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "apply_diff_guided_batch_transaction"):
            return raw_db.apply_diff_guided_batch_transaction(action_results, config)
        return placeio_openroad_cpp.apply_diff_guided_batch_transaction(
            raw_db, action_results, config
        )

    @staticmethod
    def run_diff_guided_batch_loop(raw_db, seed_queue, conflict_precompute, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "run_diff_guided_batch_loop"):
            return raw_db.run_diff_guided_batch_loop(seed_queue, conflict_precompute, config)
        return placeio_openroad_cpp.run_diff_guided_batch_loop(
            raw_db, seed_queue, conflict_precompute, config
        )

    @staticmethod
    def run_diff_guided_batch_loop_compact(raw_db, action_buffers, conflict_precompute, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "run_diff_guided_batch_loop_compact"):
            return raw_db.run_diff_guided_batch_loop_compact(action_buffers, conflict_precompute, config)
        return placeio_openroad_cpp.run_diff_guided_batch_loop_compact(
            raw_db, action_buffers, conflict_precompute, config
        )

    @staticmethod
    def run_one_net_buffer(raw_db, net_name, config=None):
        config = {} if config is None else config
        return raw_db.run_one_net_buffer(net_name, config)

    @staticmethod
    def run_coordinate_buffer_insert(raw_db, action, config=None):
        config = {} if config is None else config
        if hasattr(raw_db, "run_coordinate_buffer_insert"):
            return raw_db.run_coordinate_buffer_insert(action, config)
        return placeio_openroad_cpp.run_coordinate_buffer_insert(raw_db, action, config)

    @staticmethod
    def eval_tcl_string(raw_db, cmd):
        return raw_db.eval_tcl_string(cmd)

    @staticmethod
    def sync_to_openroad(raw_db, node_x, node_y):
        return raw_db.sync_to_openroad(node_x, node_y)

    @staticmethod
    def buffer_insertion(raw_db, strategy="repair_design", options=None):
        cmd = PlaceIOFunction.build_buffer_insertion_command(strategy, options)
        return raw_db.run_buffer_insertion(cmd)
