import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("run_staged_joint", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StagedJointRegressionTest(unittest.TestCase):
    def test_openroad_tcl_uses_requested_thread_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            openroad = root / "openroad"
            openroad.touch()
            with mock.patch.object(runner, "run_command", return_value=0) as run:
                result = runner.run_openroad_tcl(
                    openroad_bin=openroad,
                    tcl_path=root / "run.tcl",
                    log_path=root / "run.log",
                    dry_run=False,
                    openroad_num_threads=16,
                )

        self.assertEqual(result["status"], "ok")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-threads") + 1], "16")

    def test_stage1_command_enables_diff_tdp_without_buffering_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_autodmp_stage_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "stage1_result",
                flow_kind="placement",
                def_input=root / "design" / case / f"{case}.def",
                output_def=root / "stage1.def",
                iterations=400,
            )

        self.assertIn("--flow-kind", command)
        self.assertIn("placement", command)
        self.assertIn("--diff-timing-driven-placement", command)
        self.assertIn("--with-sta", command)
        self.assertIn("--output-def", command)
        self.assertIn(str(root / "stage1.def"), command)
        forbidden = {
            "--buffering-mode",
            "--buffering-max-selected-actions",
            "--buffering-segment-count-z-init",
        }
        self.assertTrue(forbidden.isdisjoint(set(command)))

    def test_stage1_command_supports_strict_tdp_gate_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_autodmp_stage_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "stage1_result",
                flow_kind="placement",
                def_input=root / "design" / case / f"{case}.def",
                output_def=root / "stage1.def",
                iterations=2000,
                diff_tdp_enabled=False,
                tdp_enable_threshold=0.15,
                enable_net_weighting=0,
            )

        diff_index = command.index("--diff-timing-driven-placement")
        self.assertEqual(command[diff_index + 1], "0")
        threshold_index = command.index("--timing-topology-enable-overflow-threshold")
        self.assertEqual(command[threshold_index + 1], "0.15")
        net_weight_index = command.index("--enable-net-weighting")
        self.assertEqual(command[net_weight_index + 1], "0")

    def test_stage2_command_uses_stage1_def_as_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_autodmp_stage_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "stage2_result",
                flow_kind="sizing",
                def_input=root / "stage1.def",
                output_def=root / "stage2.def",
                iterations=100,
            )

        flow_index = command.index("--flow-kind")
        self.assertEqual(command[flow_index + 1], "sizing")
        def_index = command.index("--def-input")
        self.assertEqual(command[def_index + 1], str(root / "stage1.def"))
        output_index = command.index("--output-def")
        self.assertEqual(command[output_index + 1], str(root / "stage2.def"))
        self.assertNotIn("--diff-timing-driven-placement", command)

    def test_stage_commands_can_pin_autodmp_place_io_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_autodmp_stage_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "stage2_result",
                flow_kind="sizing",
                def_input=root / "stage1.def",
                output_def=root / "stage2.def",
                iterations=100,
                place_io_engine="openroad",
            )

        engine_index = command.index("--place-io-engine")
        self.assertEqual(command[engine_index + 1], "openroad")

    def test_tools_checkpoint_legalizes_once_without_repair_or_verilog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            tcl = runner.generate_tools_checkpoint_tcl(
                benchmark_root=root,
                case=case,
                def_input=root / "stage.def",
                output_def=root / "stage_legal.def",
                input_inventory=root / "input_inventory.tsv",
                evaluated_inventory=root / "evaluated_inventory.tsv",
                dpl_report=root / "dpl_report.json",
                detailed_placement_search_window="full_core",
            )

        self.assertEqual(tcl.count("\ndetailed_placement -max_displacement "), 1)
        self.assertIn("detailed_placement -max_displacement", tcl)
        self.assertIn('emit_metric "detailed_placement_search_window" "full_core"', tcl)
        self.assertIn("check_placement", tcl)
        self.assertIn("-report_file_name", tcl)
        self.assertIn("create_macro_halo_blockages", tcl)
        self.assertIn("destroy_macro_halo_blockages", tcl)
        self.assertEqual(tcl.count("dump_inventory"), 3)
        self.assertIn("estimate_parasitics -placement", tcl)
        self.assertNotIn("repair_design", tcl)
        self.assertNotIn("repair_timing", tcl)
        self.assertNotIn("read_verilog", tcl)

    def test_tools_release_plan_freezes_psb_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            output_root = root / "output"
            returncode = runner.main(
                [
                    "--benchmark-root",
                    str(root),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "unit",
                    "--case",
                    case,
                    "--python-bin",
                    "/python",
                    "--openroad-bin",
                    "/openroad",
                    "--openroad-num-threads",
                    "16",
                    "--tools-release-method",
                    "autodmp_staged_psb",
                    "--dry-run-config",
                ]
            )
            method_root = output_root / "unit" / case / "autodmp_staged_psb"
            result = json.loads((method_root / "result.json").read_text())
            plan = json.loads((method_root / "command_plan.json").read_text())

        self.assertEqual(returncode, 0)
        self.assertEqual(result["status"], "plan_only")
        self.assertEqual(
            plan["order"],
            [
                "pin2pin_placement",
                "placement_checkpoint_legalization",
                "fixed_position_sizing",
                "sizing_checkpoint_legalization",
                "equal_spaced_route_b",
            ],
        )
        placement = plan["commands"]["pin2pin_placement"]
        sizing = plan["commands"]["fixed_position_sizing"]
        buffering = plan["commands"]["equal_spaced_route_b"]
        self.assertEqual(placement[placement.index("--iterations") + 1], "3000")
        self.assertEqual(
            placement[placement.index("--openroad-num-threads") + 1], "16"
        )
        self.assertEqual(placement[placement.index("--pin2pin-weight") + 1], "0.0005")
        self.assertIn("--enable-fillers", placement)
        self.assertEqual(sizing[sizing.index("--iterations") + 1], "50")
        self.assertEqual(
            sizing[sizing.index("--discrete-gradient-topk-up-percent") + 1],
            "30.0",
        )
        self.assertEqual(
            buffering[buffering.index("--segment-strategy") + 1],
            "discrete_net_gradient",
        )
        self.assertEqual(
            buffering[buffering.index("--def-input") + 1],
            result["source_artifacts"]["sizing_legalized_def"],
        )

    def test_repair_tcl_emits_report_sequence_around_repair_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            tcl = runner.generate_openroad_repair_tcl(
                benchmark_root=root,
                case=case,
                def_input=root / "stage2.def",
                output_dir=root / "repair",
                repair_mode="repair_design",
                output_def=root / "repair" / "out.def",
            )

        self.assertIn('safe_metric "pre_repair_wns" {worst_slack -max}', tcl)
        self.assertIn('safe_metric "pre_repair_tns" {total_negative_slack -max}', tcl)
        self.assertIn("\nrepair_design\n", tcl)
        self.assertIn('refresh_timing "post_repair"', tcl)
        self.assertIn('safe_metric "post_repair_wns" {worst_slack -max}', tcl)
        self.assertIn('safe_metric "post_repair_tns" {total_negative_slack -max}', tcl)
        self.assertIn("write_def", tcl)

    def test_repair_tcl_supports_repair_timing_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            tcl = runner.generate_openroad_repair_tcl(
                benchmark_root=root,
                case=case,
                def_input=root / "stage2.def",
                output_dir=root / "repair",
                repair_mode="repair_timing_setup",
            )

        self.assertIn("\nrepair_timing -setup\n", tcl)

    def test_baseline_tcl_requests_timing_driven_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            tcl = runner.generate_openroad_repair_tcl(
                benchmark_root=root,
                case=case,
                def_input=root / "design" / case / f"{case}.def",
                output_dir=root / "baseline",
                repair_mode="repair_design",
                run_timing_driven_placement=True,
            )

        self.assertIn('emit_metric "baseline_timing_driven_placement_requested" 1', tcl)
        self.assertIn("global_placement -timing_driven", tcl)
        self.assertIn("detailed_placement", tcl)
        self.assertIn("\nrepair_design\n", tcl)

    def test_parse_openroad_metrics_converts_numeric_values(self):
        metrics = runner.parse_openroad_metrics(
            """
            METRIC|pre_repair_tns|-143.5
            METRIC|post_repair_tns|-10.25
            METRIC|parasitics_source|placement
            """
        )

        self.assertEqual(metrics["pre_repair_tns"], -143.5)
        self.assertEqual(metrics["post_repair_tns"], -10.25)
        self.assertEqual(metrics["parasitics_source"], "placement")

    def test_build_row_marks_failed_stage_without_qor_claim(self):
        row = runner.build_row(
            date="2026-06-26",
            run_id="unit",
            case="NV_NVDLA_partition_m",
            flow_name="autodmp_staged",
            repair_mode="repair_design",
            stage1={"status": "failed", "failure": "boom", "runtime_sec": 1.0},
            stage2=None,
            repair={"status": "skipped", "metrics": {}},
            runtime_total_sec=1.0,
            command_plan_path=Path("/tmp/command_plan.json"),
        )

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failed_stage"], "stage1")
        self.assertEqual(row["failure"], "boom")
        self.assertIsNone(row["post_repair_tns"])

    def test_build_row_accepts_stage1_only_without_stage2_failure(self):
        row = runner.build_row(
            date="2026-06-26",
            run_id="unit",
            case="NV_NVDLA_partition_m",
            flow_name="autodmp_staged",
            repair_mode="repair_design",
            stage1={"status": "dry_run", "runtime_sec": 0.0},
            stage2=None,
            repair={"status": "dry_run", "runtime_sec": 0.0, "metrics": {}},
            runtime_total_sec=0.0,
            command_plan_path=Path("/tmp/command_plan.json"),
            stage1_output_def=Path("/tmp/stage1.def"),
            stage2_output_def=None,
        )

        self.assertEqual(row["status"], "dry_run")
        self.assertEqual(row["failed_stage"], "")
        self.assertEqual(row["stage2_status"], "")
        self.assertEqual(row["stage2_output_def"], "")

    def test_resolve_cases_rejects_unknown_case(self):
        with self.assertRaises(SystemExit):
            runner.resolve_cases(["not_a_case"])

    @staticmethod
    def _write_case(root, case):
        case_dir = root / "design" / case
        workspace = case_dir / "workspace" / "config" / "dreamplace_config"
        workspace.mkdir(parents=True)
        (workspace / "param.json").write_text("{}", encoding="utf-8")
        for suffix in (".def", ".v", ".sdc"):
            (case_dir / f"{case}{suffix}").write_text("", encoding="utf-8")
        (root / "stage1.def").write_text("", encoding="utf-8")
        (root / "stage2.def").write_text("", encoding="utf-8")
        asap7 = root / "ASAP7"
        for subdir in ("lef", "lib"):
            (asap7 / subdir).mkdir(parents=True, exist_ok=True)
        (asap7 / "setRC.tcl").write_text("", encoding="utf-8")
        (asap7 / "lef" / "asap7_tech_1x_201209.lef").write_text("", encoding="utf-8")
        for lef in (
            "asap7sc7p5t_27_R_1x_201211.lef",
            "sram_asap7_16x256_1rw.lef",
            "sram_asap7_32x256_1rw.lef",
            "sram_asap7_64x256_1rw.lef",
            "sram_asap7_64x64_1rw.lef",
        ):
            (asap7 / "lef" / lef).write_text("", encoding="utf-8")
        for lib in (
            "asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
            "sram_asap7_16x256_1rw.lib",
            "sram_asap7_32x256_1rw.lib",
            "sram_asap7_64x256_1rw.lib",
            "sram_asap7_64x64_1rw.lib",
        ):
            (asap7 / "lib" / lib).write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
