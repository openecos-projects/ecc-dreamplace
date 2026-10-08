import copy
import json
import signal
import sys
import tempfile
import unittest
import types
from pathlib import Path
from unittest.mock import patch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import iccad24_asap7_buffer_only_runner as runner


def write_file(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class Iccad24Asap7BufferOnlyRunnerTest(unittest.TestCase):
    class FakeParams:
        def fromJson(self, payload):
            for key, value in payload.items():
                setattr(self, key, copy.deepcopy(value))

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name) / "iccad24-benchmark"
        self.case_name = "tiny_case"
        self.case_dir = self.root / "design" / self.case_name

        write_file(self.case_dir / f"{self.case_name}.def", "VERSION 5.8 ;\n")
        write_file(self.case_dir / f"{self.case_name}.v", "module tiny_case(); endmodule\n")
        write_file(self.case_dir / f"{self.case_name}.sdc", "create_clock -period 1 clk\n")
        write_file(self.case_dir / f"{self.case_name}.size", "u0 BUF\nu1 BUF\n")

        self.compat_sdc = (
            self.case_dir
            / "workspace"
            / "output"
            / "dreamplace"
            / "compat"
            / f"{self.case_name}_optimizer_compat.sdc"
        )
        write_file(self.compat_sdc, "create_clock -period 1 clk\n")

        stale_sdc = (
            "/home/zhaoxueyan/papers/PhD_zhongqi/third_party/"
            "benchmark-admm-gatesizing/workspace_case/iccad24-benchmark/"
            f"design/{self.case_name}/workspace/output/dreamplace/compat/"
            f"{self.case_name}_optimizer_compat.sdc"
        )
        write_file(
            self.case_dir / "workspace" / "config" / "design_path.json",
            json.dumps({"sdc_path": stale_sdc}),
        )
        write_file(
            self.case_dir / "workspace" / "config" / "tech_path.json",
            json.dumps(
                {
                    "tech_lef_path": str(self.root / "ASAP7" / runner.ASAP7_LEF_FILES[0]),
                    "lef_paths": [
                        str(self.root / "ASAP7" / path)
                        for path in runner.ASAP7_LEF_FILES[1:]
                    ],
                    "lib_paths": [
                        str(self.root / "ASAP7" / path)
                        for path in runner.ASAP7_LIB_FILES
                    ],
                }
            ),
        )
        write_file(
            self.case_dir / "workspace" / "config" / "workspace.json",
            json.dumps(
                {
                    "workspace": {
                        "process_node": "asap7",
                        "version": "V1",
                        "project": self.case_name,
                        "design": self.case_name,
                        "task": "run_data",
                    }
                }
            ),
        )
        write_file(
            self.case_dir / "workspace" / "config" / "iEDA_config" / "db_default_config.json",
            json.dumps(
                {
                    "INPUT": {
                        "tech_lef_path": str(self.root / "ASAP7" / runner.ASAP7_LEF_FILES[0]),
                        "lef_paths": [
                            str(self.root / "ASAP7" / path)
                            for path in runner.ASAP7_LEF_FILES[1:]
                        ],
                        "def_path": stale_sdc.replace(".sdc", ".def"),
                        "verilog_path": stale_sdc.replace(".sdc", ".v"),
                        "lib_path": [
                            str(self.root / "ASAP7" / path)
                            for path in runner.ASAP7_LIB_FILES
                        ],
                        "sdc_path": stale_sdc,
                    },
                    "OUTPUT": {"output_dir_path": "/stale/output"},
                }
            ),
        )
        write_file(
            self.case_dir / "workspace" / "config" / "iEDA_config" / "flow_config.json",
            json.dumps({}),
        )
        write_file(
            self.case_dir / "workspace" / "dreamplace_config" / "param.json",
            json.dumps(
                {
                    "global_place_stages": [{"iteration": 3000}],
                    "num_bins_x": 16,
                    "num_bins_y": 16,
                    "routability_opt_flag": 1,
                }
            ),
        )

        asap7 = self.root / "ASAP7"
        for path in runner.ASAP7_LEF_FILES + runner.ASAP7_LIB_FILES:
            write_file(asap7 / path, "placeholder\n")
        write_file(asap7 / runner.ASAP7_RC_FILE, "set_wire_rc -signal -layer M3\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_discover_case_remaps_config_sdc_and_counts_size(self):
        case = runner.discover_case(self.case_name, benchmark_root=self.root)

        self.assertEqual(case["design"], self.case_name)
        self.assertEqual(case["sdc_path"], str(self.compat_sdc))
        self.assertEqual(case["sdc_kind"], "config")
        self.assertEqual(case["component_count"], 2)
        self.assertEqual(
            case["param_path"],
            str(self.case_dir / "workspace" / "dreamplace_config" / "param.json"),
        )

    def test_asap7_bundle_uses_full_lef_lib_and_rc_set(self):
        bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        self.assertTrue(bundle["tech_lef"].endswith("asap7_tech_1x_201209.lef"))
        self.assertEqual(len(bundle["lef_paths"]), len(runner.ASAP7_LEF_FILES) - 1)
        self.assertEqual(len(bundle["lib_paths"]), len(runner.ASAP7_LIB_FILES))
        self.assertTrue(bundle["rc_tcl"].endswith("setRC.tcl"))

    def test_buffer_only_command_sets_zero_margin_explicitly(self):
        command = runner.buffer_only_repair_command(0)

        self.assertIn("repair_timing -setup", command)
        self.assertIn("-setup_margin 0", command)
        self.assertIn('-sequence "unbuffer,buffer,split"', command)
        self.assertIn("-skip_pin_swap", command)
        self.assertIn("-skip_gate_cloning", command)
        self.assertIn("-skip_size_down", command)
        self.assertNotIn("-skip_vt_swap", command)
        self.assertNotIn("-skip_crit_vt_swap", command)

    def test_build_params_preserves_native_iteration_when_target_iter_is_none(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
        )

        self.assertEqual(params.global_place_stages[0]["iteration"], 3000)

    def test_build_params_keeps_explicit_target_iter_override(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=100,
        )

        self.assertEqual(params.global_place_stages[0]["iteration"], 100)

    def test_build_params_enables_with_sta_when_requested(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            with_sta=True,
        )

        self.assertEqual(params.with_sta, 1)
        self.assertEqual(params.design_inputs["rc_tcl"], tech_bundle["rc_tcl"])

    def test_build_params_defaults_to_openroad_place_io_engine(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
        )

        self.assertEqual(params.place_io_engine, "openroad")

    def test_build_params_can_select_ieda_place_io_engine(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            place_io_engine="ieda",
        )

        self.assertEqual(params.place_io_engine, "ieda")

    def test_log_run_stage_emits_machine_readable_progress_marker(self):
        with self.assertLogs(level="INFO") as captured:
            runner.log_run_stage(
                "setup_rawdb",
                "begin",
                place_io_engine="ieda",
                run_dir="/tmp/run",
            )

        self.assertIn(
            "stage=setup_rawdb state=begin place_io_engine=ieda run_dir=/tmp/run",
            captured.output[0],
        )

    def test_stale_log_watchdog_terminates_process_group_after_no_new_log(self):
        class FakeClock:
            def __init__(self):
                self.value = 0.0

            def now(self):
                return self.value

            def sleep(self, seconds):
                self.value += seconds

        class FakeProcess:
            instances = []

            def __init__(self, command, **kwargs):
                self.command = command
                self.kwargs = kwargs
                self.pid = 12345
                self.returncode = None
                FakeProcess.instances.append(self)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        killed = []
        clock = FakeClock()

        def fake_killpg(pid, sig):
            killed.append((pid, sig))
            FakeProcess.instances[0].returncode = -int(sig)

        result = runner.run_subprocess_with_stale_log_watchdog(
            ["python3", "long_run.py"],
            Path(self.tmpdir.name) / "run.log",
            stale_log_timeout_sec=3,
            poll_interval_sec=1,
            popen_cls=FakeProcess,
            now=clock.now,
            sleep=clock.sleep,
            killpg=fake_killpg,
            log_mtime_func=lambda path: 10.0,
        )

        self.assertEqual(result["status"], "stale_log_timeout")
        self.assertEqual(killed, [(12345, signal.SIGTERM)])
        self.assertTrue(FakeProcess.instances[0].kwargs["start_new_session"])
        self.assertGreaterEqual(result["idle_sec"], 3)

    def test_prepare_ieda_workspace_copies_config_and_rewrites_case_paths(self):
        run_dir = Path(self.tmpdir.name) / "run"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        workspace = runner.prepare_ieda_workspace(
            case,
            tech_bundle,
            run_dir,
        )

        self.assertEqual(workspace, run_dir / "workspace")
        design_path = json.loads((workspace / "config" / "design_path.json").read_text())
        tech_path = json.loads((workspace / "config" / "tech_path.json").read_text())
        db_config = json.loads(
            (workspace / "config" / "iEDA_config" / "db_default_config.json").read_text()
        )
        self.assertEqual(design_path["def_input_path"], case["def_input"])
        self.assertEqual(design_path["verilog_input_path"], case["verilog_input"])
        self.assertEqual(design_path["sdc_path"], case["sdc_path"])
        self.assertEqual(tech_path["tech_lef_path"], tech_bundle["tech_lef"])
        self.assertEqual(tech_path["lef_paths"], tech_bundle["lef_paths"])
        self.assertEqual(tech_path["lib_paths"], tech_bundle["lib_paths"])
        self.assertEqual(db_config["INPUT"]["def_path"], case["def_input"])
        self.assertEqual(db_config["INPUT"]["sdc_path"], case["sdc_path"])
        self.assertEqual(
            db_config["OUTPUT"]["output_dir_path"],
            str(workspace / "output" / "iEDA" / "result"),
        )

    def test_build_params_sets_default_cold_handoff_restart_policy(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
        )

        self.assertEqual(params.handoff_restart_policy, "cold")

    def test_build_params_sets_warm_schedule_handoff_restart_policy(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            handoff_restart_policy="warm_schedule",
        )

        self.assertEqual(params.handoff_restart_policy, "warm_schedule")

    def test_build_params_can_disable_handoff_for_no_handoff_diagnostic(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            handoff_mode="no-handoff",
        )

        self.assertEqual(params.openroad_handoff["enabled"], False)
        self.assertEqual(params.openroad_handoff["trigger"], {"mode": "disabled"})
        self.assertEqual(params.base_design_name, "iccad24_asap7_tiny_case_no_handoff")

    def test_build_params_uses_production_interval_guard_config(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            trigger_period=75,
            max_handoff_overflow=0.15,
        )

        self.assertEqual(params.openroad_handoff["trigger"]["mode"], "interval")
        self.assertEqual(params.openroad_handoff["trigger"]["interval"], 75)
        self.assertEqual(
            params.openroad_handoff["trigger"]["guards"],
            [{"name": "overflow", "op": "<=", "value": 0.15}],
        )
        self.assertEqual(
            params.openroad_handoff["trigger"]["dedupe_by"],
            "absolute_iteration",
        )

    def test_build_params_applies_density_overrides_for_diagnostics(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            target_density_override=0.45,
            stop_overflow_override=0.55,
        )

        self.assertEqual(params.target_density, 0.45)
        self.assertEqual(params.stop_overflow, 0.55)

    def test_build_params_can_preserve_native_routability_opt_flag(self):
        run_dir = Path(self.tmpdir.name) / "run"
        final_def = run_dir / "out.def"
        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")

        params, _ = runner.build_params(
            params_cls=self.FakeParams,
            case_config=case,
            tech_bundle=tech_bundle,
            run_dir=run_dir,
            final_def=str(final_def),
            target_iter=None,
            preserve_routability_opt=True,
        )

        self.assertEqual(params.routability_opt_flag, 1)

    def test_effective_target_iter_reports_native_and_override_sources(self):
        self.assertEqual(
            runner.effective_target_iter(
                {"global_place_stages": [{"iteration": 3000}]},
                target_iter=None,
            ),
            (3000, "native"),
        )
        self.assertEqual(
            runner.effective_target_iter(
                {"global_place_stages": [{"iteration": 3000}]},
                target_iter=100,
            ),
            (100, "override"),
        )

    def test_artifact_root_labels_native_and_override_iterations(self):
        base = Path(self.tmpdir.name) / "artifacts"

        native_root = runner.artifact_root_for(
            "hidden5", base, 3000, "native", 50, 0
        )
        override_root = runner.artifact_root_for(
            "hidden5", base, 100, "override", 50, 0
        )

        self.assertEqual(
            native_root.name,
            "hidden5_iccad24_asap7_buffer_only_native3000_period50_margin0",
        )
        self.assertEqual(
            override_root.name,
            "hidden5_iccad24_asap7_buffer_only_iter100_period50_margin0",
        )

    def test_artifact_root_labels_with_sta_artifacts_distinctly(self):
        base = Path(self.tmpdir.name) / "artifacts"

        native_root = runner.artifact_root_for(
            "hidden5", base, 3000, "native", 50, 0, with_sta=False
        )
        sta_root = runner.artifact_root_for(
            "hidden5", base, 3000, "native", 50, 0, with_sta=True
        )

        self.assertEqual(
            native_root.name,
            "hidden5_iccad24_asap7_buffer_only_native3000_period50_margin0",
        )
        self.assertEqual(
            sta_root.name,
            "hidden5_iccad24_asap7_buffer_only_native3000_sta1_period50_margin0",
        )

    def test_artifact_root_labels_no_handoff_and_density_overrides(self):
        base = Path(self.tmpdir.name) / "artifacts"

        root = runner.artifact_root_for(
            "hidden3",
            base,
            3000,
            "native",
            50,
            0,
            handoff_mode="no-handoff",
            target_density_override=0.45,
            stop_overflow_override=0.55,
        )

        self.assertEqual(
            root.name,
            "hidden3_iccad24_asap7_no_handoff_native3000_period50_margin0_td0p45_so0p55",
        )

    def test_artifact_root_labels_preserved_routability_control(self):
        base = Path(self.tmpdir.name) / "artifacts"

        root = runner.artifact_root_for(
            "hidden3",
            base,
            3000,
            "native",
            50,
            0,
            handoff_mode="no-handoff",
            preserve_routability_opt=True,
        )

        self.assertEqual(
            root.name,
            "hidden3_iccad24_asap7_no_handoff_native3000_period50_margin0_roptnative",
        )

    def test_iccad24_trigger_skips_periodic_handoff_above_overflow_gate(self):
        class FakeController:
            pass

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        runner.install_iccad24_handoff_trigger(
            placedb,
            trigger_period=50,
            max_handoff_overflow=0.2,
        )
        controller = placedb.create_openroad_handoff_controller({})

        requested, decision = controller.should_handoff(
            {
                "absolute_iteration": 50,
                "metric": types.SimpleNamespace(overflow=0.5),
            }
        )

        self.assertFalse(requested)
        self.assertIsNone(decision)

    def test_iccad24_trigger_runs_periodic_handoff_below_overflow_gate(self):
        class FakeController:
            pass

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        runner.install_iccad24_handoff_trigger(
            placedb,
            trigger_period=50,
            max_handoff_overflow=0.2,
        )
        controller = placedb.create_openroad_handoff_controller({})

        requested, decision = controller.should_handoff(
            {
                "absolute_iteration": 100,
                "metric": types.SimpleNamespace(overflow=0.15),
            }
        )

        self.assertTrue(requested)
        self.assertEqual(decision["trigger_mode"], "periodic_iteration")
        self.assertIn("period=50@iter=100", decision["trigger_reason"])
        self.assertIn("overflow<=0.2", decision["trigger_reason"])

    def test_iccad24_trigger_skips_handoff_when_overflow_is_nan(self):
        class FakeController:
            pass

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        runner.install_iccad24_handoff_trigger(
            placedb,
            trigger_period=50,
            max_handoff_overflow=0.2,
        )
        controller = placedb.create_openroad_handoff_controller({})

        requested, decision = controller.should_handoff(
            {
                "absolute_iteration": 50,
                "metric": types.SimpleNamespace(overflow=float("nan")),
            }
        )

        self.assertFalse(requested)
        self.assertIsNone(decision)

    def test_iccad24_trigger_preserves_periodic_behavior_without_overflow_gate(self):
        class FakeController:
            pass

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        runner.install_iccad24_handoff_trigger(
            placedb,
            trigger_period=50,
            max_handoff_overflow=None,
        )
        controller = placedb.create_openroad_handoff_controller({})

        requested, decision = controller.should_handoff(
            {
                "absolute_iteration": 50,
                "metric": types.SimpleNamespace(overflow=0.9),
            }
        )

        self.assertTrue(requested)
        self.assertEqual(decision["trigger_mode"], "periodic_iteration")
        self.assertEqual(
            decision["trigger_reason"],
            "periodic_iteration@period=50@iter=50",
        )

    def test_run_one_records_with_sta_and_handoff_trigger_in_summary(self):
        class FakePlacementResult:
            pass

        class FakeSession:
            def __init__(self):
                self.event_history = [{"kind": "handoff"}]

        class FakePlacedb:
            def __init__(self):
                self.handoff_session = FakeSession()

            def create_openroad_handoff_controller(self, params):
                return types.SimpleNamespace()

        class FakeEngine:
            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()

            def setup_rawdb(self, data_manager):
                self.data_manager = data_manager

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                write_file(Path(def_file), "VERSION 5.8 ;\nEND DESIGN\n")

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "build_params", return_value=(self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]})
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "completed", "event_count": 1, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.object(
            runner.matrix, "install_periodic_trigger"
        ), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                with_sta=True,
                max_handoff_overflow=0.2,
            )

        self.assertTrue(summary["with_sta"])
        self.assertTrue(summary["config"]["with_sta"])
        self.assertEqual(summary["max_handoff_overflow"], 0.2)
        self.assertEqual(
            summary["config"]["handoff_trigger"],
            {
                "trigger_period": 50,
                "max_handoff_overflow": 0.2,
            },
        )

    def test_run_one_records_diagnostic_mode_and_density_overrides(self):
        class FakePlacementResult:
            pass

        build_params_calls = []

        class FakePlacedb:
            handoff_session = None

        class FakeEngine:
            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()

            def setup_rawdb(self, data_manager):
                self.data_manager = data_manager

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                write_file(Path(def_file), "VERSION 5.8 ;\nEND DESIGN\n")

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        def fake_build_params(*args, **kwargs):
            build_params_calls.append(kwargs)
            return self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]}

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "build_params", side_effect=fake_build_params
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "completed", "event_count": 0, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.object(
            runner.matrix, "install_periodic_trigger"
        ), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                handoff_mode="no-handoff",
                target_density_override=0.45,
                stop_overflow_override=0.55,
                preserve_routability_opt=True,
            )

        self.assertEqual(summary["handoff_mode"], "no-handoff")
        self.assertTrue(build_params_calls[0]["preserve_routability_opt"])
        self.assertEqual(
            summary["config"]["diagnostic_param_overrides"],
            {
                "target_density": 0.45,
                "stop_overflow": 0.55,
            },
        )

    def test_run_one_initializes_ieda_workspace_before_ieda_setup_rawdb(self):
        init_calls = []
        setup_workspaces = []
        tee_calls = []

        class FakePlacementResult:
            pass

        class FakePlacedb:
            handoff_session = None

        class FakeEngine:
            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()

            def setup_rawdb(self, data_manager):
                setup_workspaces.append(data_manager.dir_workspace)

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                write_file(Path(def_file), "VERSION 5.8 ;\nEND DESIGN\n")

        class FakeIEDAIO:
            def __init__(self, dir_workspace, input_def=None, input_verilog=None):
                self.dir_workspace = dir_workspace
                self.input_def = input_def
                self.input_verilog = input_verilog

            def read_def(self, input_def="", read_verilog=False):
                init_calls.append(
                    {
                        "dir_workspace": self.dir_workspace,
                        "input_def": input_def,
                        "read_verilog": read_verilog,
                    }
                )

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path
                tee_calls.append({"path": path, **kwargs})

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_ops.__path__ = []
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_placeio_ieda = types.ModuleType("dreamplace.ops.placeio_ieda")
        fake_placeio_ieda_place_io = types.ModuleType("dreamplace.ops.placeio_ieda.place_io")
        fake_placeio_ieda_place_io.PlaceIOFunction = types.SimpleNamespace(
            make_io=lambda *args, **kwargs: FakeIEDAIO(*args, **kwargs),
        )
        fake_placeio_ieda.place_io = fake_placeio_ieda_place_io
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad
        fake_ops.placeio_ieda = fake_placeio_ieda

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "install_real_ieda_binding"
        ), patch.object(
            runner, "build_params", return_value=(self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]})
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "not_started", "event_count": 0, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
                "dreamplace.ops.placeio_ieda": fake_placeio_ieda,
                "dreamplace.ops.placeio_ieda.place_io": fake_placeio_ieda_place_io,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                handoff_mode="no-handoff",
                place_io_engine="ieda",
            )

        expected_workspace = str(run_dir / "workspace")
        self.assertEqual(setup_workspaces, [expected_workspace])
        self.assertEqual(init_calls[0]["dir_workspace"], expected_workspace)
        self.assertEqual(init_calls[0]["input_def"], case["def_input"])
        self.assertFalse(init_calls[0]["read_verilog"])
        self.assertEqual(summary["place_io_engine"], "ieda")
        self.assertEqual(summary["config"]["place_io_engine"], "ieda")
        self.assertEqual(tee_calls[0]["mirror_to_stdout"], False)

    def test_run_one_writes_final_def_after_successful_no_handoff_run(self):
        class FakePlacementResult:
            pass

        class FakePlacedb:
            handoff_session = None

        class FakeEngine:
            instances = []

            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()
                self.write_back_calls = []
                FakeEngine.instances.append(self)

            def setup_rawdb(self, data_manager):
                self.data_manager = data_manager

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                self.write_back_calls.append(def_file)
                write_file(Path(def_file), "VERSION 5.8 ;\nEND DESIGN\n")

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "build_params", return_value=(self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]})
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "not_started", "event_count": 0, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                handoff_mode="no-handoff",
            )

        self.assertTrue(summary["final_def_exists"])
        self.assertEqual(FakeEngine.instances[0].write_back_calls, [summary["final_def"]])

    def test_run_one_records_final_def_coordinate_validation_failure(self):
        class FakePlacementResult:
            pass

        class FakePlacedb:
            handoff_session = None

        class FakeEngine:
            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()

            def setup_rawdb(self, data_manager):
                self.data_manager = data_manager

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                write_file(
                    Path(def_file),
                    "\n".join(
                        [
                            "VERSION 5.8 ;",
                            "DIEAREA ( 0 0 ) ( 1000 900 ) ;",
                            "COMPONENTS 1 ;",
                            "- u0 BUF_X1 + PLACED ( 54000 54000 ) N ;",
                            "END COMPONENTS",
                            "END DESIGN",
                            "",
                        ]
                    ),
                )

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "build_params", return_value=(self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]})
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "not_started", "event_count": 0, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                handoff_mode="no-handoff",
            )

        self.assertTrue(summary["final_def_exists"])
        self.assertIsNone(summary["failure"])
        self.assertEqual(summary["final_def_validation"]["status"], "failed")
        self.assertIn("outside DIEAREA", summary["final_def_validation"]["failure"])

    def test_run_one_validates_written_final_def_even_when_run_records_failure(self):
        class FakePlacementResult:
            pass

        class FakePlacedb:
            handoff_session = None

        class FakeEngine:
            def __init__(self, params):
                self.params = params
                self.placedb = FakePlacedb()

            def setup_rawdb(self, data_manager):
                self.data_manager = data_manager

            def run(self):
                return FakePlacementResult()

            def write_back(self, def_file):
                write_file(
                    Path(def_file),
                    "\n".join(
                        [
                            "VERSION 5.8 ;",
                            "DIEAREA ( 0 0 ) ( 1000 900 ) ;",
                            "COMPONENTS 1 ;",
                            "- u0 BUF_X1 + PLACED ( 500 400 ) N ;",
                            "END COMPONENTS",
                            "END DESIGN",
                            "",
                        ]
                    ),
                )
                raise RuntimeError("post-write failure")

        class DummyTee:
            def __init__(self, path, **kwargs):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_placeio = types.SimpleNamespace(
            PlaceIOFunction=types.SimpleNamespace(build_buffer_insertion_command=lambda: "cmd"),
            _BUFFER_INSERTION_STRATEGY_ALIASES={
                "buffer-only": {"command": "cmd"},
                "buffer_only": {"command": "cmd"},
            },
        )
        fake_dreamplace = types.ModuleType("dreamplace")
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_placeio_openroad = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_placeio_openroad.place_io = fake_placeio
        fake_dreamplace.ops = fake_ops
        fake_ops.placeio_openroad = fake_placeio_openroad

        case = runner.discover_case(self.case_name, benchmark_root=self.root)
        tech_bundle = runner.asap7_tech_bundle(self.root / "ASAP7")
        run_dir = Path(self.tmpdir.name) / "run"
        run_dir.mkdir()

        with patch.object(runner, "discover_case", return_value=case), patch.object(
            runner, "asap7_tech_bundle", return_value=tech_bundle
        ), patch.object(runner, "effective_target_iter", return_value=(3000, "native")), patch.object(
            runner, "artifact_root_for", return_value=Path(self.tmpdir.name) / "artifacts"
        ), patch.object(runner, "make_run_dir", return_value=run_dir), patch.object(
            runner, "install_buffer_only_margin_command_patch", return_value="cmd"
        ), patch.object(
            runner, "build_params", return_value=(self.FakeParams(), {"global_place_stages": [{"iteration": 3000}]})
        ), patch.object(runner.matrix, "prepare_runtime_modules", return_value=(object(), types.SimpleNamespace(PlacementEngine=FakeEngine))), patch.object(
            runner.matrix, "build_handoff_strategy_summary", return_value={"strategy": "summary"}
        ), patch.object(runner.matrix, "StdoutStderrTee", DummyTee), patch.object(
            runner.matrix, "configure_regression_logging"
        ), patch.object(runner.matrix, "read_text", return_value=""), patch.object(
            runner.matrix, "build_session_summary", return_value={"status": "not_started", "event_count": 0, "buffer_churn_totals": {}}
        ), patch.object(runner.matrix, "parse_last_iteration", return_value=7), patch.object(
            runner.matrix, "extract_openroad_diagnostic_evidence", return_value={"diagnostic_labels": []}
        ), patch.object(runner.matrix, "matching_log_lines", return_value=[]), patch.dict(
            sys.modules,
            {
                "dreamplace": fake_dreamplace,
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_placeio,
            },
        ):
            summary = runner.run_one(
                self.case_name,
                benchmark_root=self.root,
                artifact_base=Path(self.tmpdir.name) / "artifacts",
                handoff_mode="no-handoff",
            )

        self.assertTrue(summary["final_def_exists"])
        self.assertEqual(summary["failure"]["type"], "RuntimeError")
        self.assertEqual(summary["final_def_validation"]["status"], "passed")
        self.assertEqual(
            summary["final_def_validation"]["summary"]["movable_outside_die_count"],
            0,
        )

    def test_main_threads_place_io_engine_through_cli(self):
        recorded = []

        def fake_run_one(*args, **kwargs):
            recorded.append(kwargs)
            return {
                "design": "tiny_case",
                "session": {"status": "completed", "event_count": 1, "buffer_churn_totals": {}},
                "failure": None,
                "last_iteration": 7,
                "run_dir": str(Path(self.tmpdir.name) / "run"),
                "no_buffer_reason": "no_handoff_event",
                "with_sta": kwargs["with_sta"],
            }

        with patch.object(
            runner,
            "build_manifest",
            return_value={"cases": [{"design": self.case_name}], "benchmark_root": str(self.root), "asap7": {}},
        ), patch.object(runner, "run_one", side_effect=fake_run_one), patch.object(
            runner, "print_run_line"
        ), patch.object(runner, "write_combined"), patch.object(
            runner, "smoke_passed", return_value=True
        ):
            exit_code = runner.main(
                [
                    "--benchmark-root",
                    str(self.root),
                    "--artifact-base",
                    str(Path(self.tmpdir.name) / "artifacts"),
                    "--design",
                    self.case_name,
                    "--skip-smoke",
                    "--with-sta",
                    "--place-io-engine",
                    "ieda",
                    "--max-handoff-overflow",
                    "0.2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded[0]["with_sta"], True)
        self.assertEqual(recorded[0]["place_io_engine"], "ieda")
        self.assertEqual(recorded[0]["max_handoff_overflow"], 0.2)

    def test_main_allows_cli_to_disable_max_handoff_overflow_gate(self):
        recorded = []

        def fake_run_one(*args, **kwargs):
            recorded.append(kwargs)
            return {
                "design": "tiny_case",
                "session": {"status": "completed", "event_count": 1, "buffer_churn_totals": {}},
                "failure": None,
                "last_iteration": 7,
                "run_dir": str(Path(self.tmpdir.name) / "run"),
                "no_buffer_reason": "no_handoff_event",
                "with_sta": kwargs["with_sta"],
            }

        with patch.object(
            runner,
            "build_manifest",
            return_value={"cases": [{"design": self.case_name}], "benchmark_root": str(self.root), "asap7": {}},
        ), patch.object(runner, "run_one", side_effect=fake_run_one), patch.object(
            runner, "print_run_line"
        ), patch.object(runner, "write_combined"), patch.object(
            runner, "smoke_passed", return_value=True
        ):
            exit_code = runner.main(
                [
                    "--benchmark-root",
                    str(self.root),
                    "--artifact-base",
                    str(Path(self.tmpdir.name) / "artifacts"),
                    "--design",
                    self.case_name,
                    "--skip-smoke",
                    "--max-handoff-overflow",
                    "none",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIsNone(recorded[0]["max_handoff_overflow"])

    def test_main_threads_diagnostic_cli_options(self):
        recorded = []

        def fake_run_one(*args, **kwargs):
            recorded.append(kwargs)
            return {
                "design": "tiny_case",
                "session": {"status": "completed", "event_count": 0, "buffer_churn_totals": {}},
                "failure": None,
                "last_iteration": 7,
                "run_dir": str(Path(self.tmpdir.name) / "run"),
                "no_buffer_reason": "no_handoff_event",
                "with_sta": kwargs["with_sta"],
            }

        with patch.object(
            runner,
            "build_manifest",
            return_value={"cases": [{"design": self.case_name}], "benchmark_root": str(self.root), "asap7": {}},
        ), patch.object(runner, "run_one", side_effect=fake_run_one), patch.object(
            runner, "print_run_line"
        ), patch.object(runner, "write_combined"), patch.object(
            runner, "smoke_passed", return_value=True
        ):
            exit_code = runner.main(
                [
                    "--benchmark-root",
                    str(self.root),
                    "--artifact-base",
                    str(Path(self.tmpdir.name) / "artifacts"),
                    "--design",
                    self.case_name,
                    "--skip-smoke",
                    "--handoff-mode",
                    "no-handoff",
                    "--target-density-override",
                    "0.45",
                    "--stop-overflow-override",
                    "0.55",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded[0]["handoff_mode"], "no-handoff")
        self.assertEqual(recorded[0]["target_density_override"], 0.45)
        self.assertEqual(recorded[0]["stop_overflow_override"], 0.55)

    def test_main_threads_handoff_restart_policy_cli_option(self):
        recorded = []

        def fake_run_one(*args, **kwargs):
            recorded.append(kwargs)
            return {
                "design": "tiny_case",
                "session": {"status": "completed", "event_count": 0, "buffer_churn_totals": {}},
                "failure": None,
                "last_iteration": 7,
                "run_dir": str(Path(self.tmpdir.name) / "run"),
                "no_buffer_reason": "no_handoff_event",
                "with_sta": kwargs["with_sta"],
            }

        with patch.object(
            runner,
            "build_manifest",
            return_value={"cases": [{"design": self.case_name}], "benchmark_root": str(self.root), "asap7": {}},
        ), patch.object(runner, "run_one", side_effect=fake_run_one), patch.object(
            runner, "print_run_line"
        ), patch.object(runner, "write_combined"), patch.object(
            runner, "smoke_passed", return_value=True
        ):
            exit_code = runner.main(
                [
                    "--benchmark-root",
                    str(self.root),
                    "--artifact-base",
                    str(Path(self.tmpdir.name) / "artifacts"),
                    "--design",
                    self.case_name,
                    "--skip-smoke",
                    "--handoff-restart-policy",
                    "warm_schedule",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded[0]["handoff_restart_policy"], "warm_schedule")

    def test_main_threads_preserve_routability_opt_cli_option(self):
        recorded = []

        def fake_run_one(*args, **kwargs):
            recorded.append(kwargs)
            return {
                "design": "tiny_case",
                "session": {"status": "completed", "event_count": 0, "buffer_churn_totals": {}},
                "failure": None,
                "last_iteration": 7,
                "run_dir": str(Path(self.tmpdir.name) / "run"),
                "no_buffer_reason": "no_handoff_event",
                "with_sta": kwargs["with_sta"],
            }

        with patch.object(
            runner,
            "build_manifest",
            return_value={"cases": [{"design": self.case_name}], "benchmark_root": str(self.root), "asap7": {}},
        ), patch.object(runner, "run_one", side_effect=fake_run_one), patch.object(
            runner, "print_run_line"
        ), patch.object(runner, "write_combined"), patch.object(
            runner, "smoke_passed", return_value=True
        ):
            exit_code = runner.main(
                [
                    "--benchmark-root",
                    str(self.root),
                    "--artifact-base",
                    str(Path(self.tmpdir.name) / "artifacts"),
                    "--design",
                    self.case_name,
                    "--skip-smoke",
                    "--handoff-mode",
                    "no-handoff",
                    "--preserve-routability-opt",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(recorded[0]["preserve_routability_opt"])


if __name__ == "__main__":
    unittest.main()
