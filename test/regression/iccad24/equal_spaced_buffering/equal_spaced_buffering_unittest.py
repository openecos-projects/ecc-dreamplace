import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("equal_spaced_buffering_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
RUNTIME_PROFILE_PATH = Path(__file__).resolve().parent / "route_b_runtime_profile.py"
runtime_spec = importlib.util.spec_from_file_location(
    "equal_spaced_buffering_runtime_profile",
    RUNTIME_PROFILE_PATH,
)
runtime_profile = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime_profile)


class EqualSpacedBufferingRunnerTest(unittest.TestCase):
    def _benchmark(self, root: Path, case: str = "toy") -> Path:
        asap7 = root / "ASAP7"
        for path in (
            asap7 / "lef" / "asap7_tech_1x_201209.lef",
            asap7 / "lef" / "asap7sc7p5t_27_R_1x_201211.lef",
            asap7 / "lef" / "sram_asap7_16x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_32x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_64x256_1rw.lef",
            asap7 / "lef" / "sram_asap7_64x64_1rw.lef",
            asap7 / "setRC.tcl",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        for name in (
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
            path = asap7 / "lib" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        design = root / "design" / case
        (design / "workspace" / "config" / "dreamplace_config").mkdir(parents=True)
        (design / f"{case}.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
        (design / f"{case}.v").write_text("module toy; endmodule\n", encoding="utf-8")
        (design / f"{case}.sdc").write_text("", encoding="utf-8")
        (design / "workspace" / "config" / "dreamplace_config" / "param.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return root

    def test_standalone_paths_are_derived_from_autodmp_root(self):
        self.assertEqual(runner.AUTODMP_ROOT, RUNNER_PATH.parents[4])
        with mock.patch.dict(os.environ, {}, clear=True):
            output_root = runner._path_from_env(
                "AUTODMP_BUFFERING_OUTPUT_ROOT",
                runner.AUTODMP_ROOT
                / "logs"
                / "regression"
                / "iccad24"
                / "equal_spaced_buffering",
            )
            benchmark_root = runner._path_from_env(
                "AUTODMP_ICCAD24_BENCHMARK_ROOT",
                runner.AUTODMP_ROOT / "benchmarks" / "iccad24_benchmark",
            )

        self.assertEqual(
            output_root,
            runner.AUTODMP_ROOT
            / "logs"
            / "regression"
            / "iccad24"
            / "equal_spaced_buffering",
        )
        self.assertEqual(
            benchmark_root,
            runner.AUTODMP_ROOT / "benchmarks" / "iccad24_benchmark",
        )

    def test_standalone_paths_accept_environment_overrides(self):
        with mock.patch.dict(os.environ, {"AUTODMP_TEST_PATH": "~/autodmp-input"}):
            resolved = runner._path_from_env("AUTODMP_TEST_PATH", "/fallback")

        self.assertEqual(resolved, Path("~/autodmp-input").expanduser())

    def test_standalone_path_falls_back_without_environment_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            resolved = runner._path_from_env("AUTODMP_TEST_PATH", "/fallback")

        self.assertEqual(resolved, Path("/fallback"))

    def test_profile_defines_only_three_methods_and_one_final_commit(self):
        profile = runner._load_profile()

        self.assertEqual(tuple(profile["methods"]), ("b0", "b1", "ours"))
        self.assertEqual(profile["segment_mode"]["max_repeaters_per_segment"], 3)
        self.assertEqual(profile["segment_mode"]["gpu_id"], 1)
        self.assertEqual(profile["segment_mode"]["integer_projection_interval"], 0)
        self.assertTrue(profile["segment_mode"]["one_final_hard_commit"])

    def test_runtime_profile_accepts_only_valid_early_route_b_termination(self):
        self.assertTrue(
            runtime_profile._valid_round_execution(
                {"iterations": 5, "terminal_reason": "max_rounds"},
                5,
            )
        )
        self.assertTrue(
            runtime_profile._valid_round_execution(
                {"iterations": 2, "terminal_reason": "no_positive_action"},
                5,
            )
        )
        self.assertTrue(
            runtime_profile._valid_round_execution(
                {"iterations": 2, "terminal_reason": "no_improving_prefix"},
                5,
            )
        )
        self.assertFalse(
            runtime_profile._valid_round_execution(
                {"iterations": 2, "terminal_reason": "max_rounds"},
                5,
            )
        )

    def test_runtime_profile_detailed_timing_is_explicitly_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text("{}\n", encoding="ascii")
            normal_path = root / "normal.json"
            detailed_path = root / "detailed.json"

            normal = runtime_profile._write_runtime_params(source, normal_path)
            detailed = runtime_profile._write_runtime_params(
                source,
                detailed_path,
                detailed_timing_profile=True,
            )

        for name in (
            "timing_obj_profile",
            "timing_propagation_profile",
            "buffering_dynamic_provider_profile",
        ):
            self.assertFalse(normal[name])
            self.assertTrue(detailed[name])

    def test_local_greedy_method_is_not_a_supported_primary_method(self):
        with self.assertRaisesRegex(ValueError, "unsupported selection: b2"):
            runner._resolve_selected(["b2"], runner.METHODS)

    def test_b1_tcl_is_def_only_and_uses_strict_buffer_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            tcl = runner.generate_openroad_tcl(
                benchmark_root=root,
                case="toy",
                def_input=root / "design" / "toy" / "toy.def",
                output_def=root / "out.def",
                power_report=root / "power.rpt",
                method="b1",
                buffer_master="HB3xp67_ASAP7_75t_R",
                b1_max_buffer_percent=3.0,
            )

        self.assertNotIn("read_verilog", tcl)
        self.assertIn('repair_timing -setup -sequence "buffer"', tcl)
        self.assertIn("set_dont_use [get_lib_cells *]", tcl)
        self.assertIn("unset_dont_use $allowed_buffer_cells", tcl)
        self.assertNotIn("unbuffer,buffer,split", tcl)
        self.assertNotIn("repair_design", tcl)

    def test_b1_rd_rt_tcl_disables_sizing_cells_and_keeps_buffer_only_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            tcl = runner.generate_openroad_tcl(
                benchmark_root=root,
                case="toy",
                def_input=root / "design" / "toy" / "toy.def",
                output_def=root / "out.def",
                power_report=root / "power.rpt",
                method="b1_rd_rt",
                buffer_master="HB3xp67_ASAP7_75t_R",
                b1_max_buffer_percent=3.0,
            )

        self.assertIn("\nrepair_design\n", tcl)
        self.assertIn('repair_timing -setup -sequence "buffer"', tcl)
        self.assertLess(tcl.index("repair_design\n"), tcl.index("repair_timing -setup"))
        self.assertIn("set_dont_use [get_lib_cells *]", tcl)
        self.assertIn("unset_dont_use $allowed_buffer_cells", tcl)
        self.assertNotIn("set_dont_touch", tcl)

    def test_source_only_b1_rd_rt_keeps_internal_dpl_but_skips_final_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            tcl = runner.generate_openroad_tcl(
                benchmark_root=root,
                case="toy",
                def_input=root / "design" / "toy" / "toy.def",
                output_def=root / "out.def",
                power_report=root / "power.rpt",
                method="b1_rd_rt",
                buffer_master="HB3xp67_ASAP7_75t_R",
                b1_max_buffer_percent=3.0,
                source_only=True,
            )

        self.assertEqual(tcl.count("detailed_placement"), 1)
        self.assertNotIn('safe_metric "wns_ns"', tcl)
        self.assertIn("write_def", tcl)

    def test_ours_command_freezes_bsu_and_disables_periodic_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            profile = runner._load_profile()
            command = runner.build_ours_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case="toy",
                result_dir=root / "result",
                committed_def=root / "committed.def",
                profile=profile,
            )

        self.assertIn("--buffering-fixed-bsu-index", command)
        self.assertEqual(Path(command[1]), runner.AUTODMP_ROOT / "dreamplace" / "Placer.py")
        self.assertIn("--place-io-engine", command)
        self.assertEqual(command[command.index("--place-io-engine") + 1], "openroad")
        self.assertIn("--gpu-id", command)
        self.assertEqual(command[command.index("--gpu-id") + 1], "1")
        self.assertIn("--buffering-segment-integer-projection-interval", command)
        interval_index = command.index("--buffering-segment-integer-projection-interval")
        self.assertEqual(command[interval_index + 1], "0")
        self.assertIn("--buffering-commit-enabled", command)
        self.assertEqual(command[command.index("--buffering-commit-enabled") + 1], "1")

    def test_ours_command_allows_campaign_logical_gpu_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            command = runner.build_ours_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case="toy",
                result_dir=root / "result",
                committed_def=root / "committed.def",
                profile=runner._load_profile(),
                logical_gpu_id=0,
            )

        self.assertEqual(command[command.index("--gpu-id") + 1], "0")

    def test_ours_command_accepts_staged_fixed_position_input_def(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            staged_def = root / "stage2_legalized.def"
            staged_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            command = runner.build_ours_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case="toy",
                result_dir=root / "result",
                committed_def=root / "committed.def",
                profile=runner._load_profile(),
                logical_gpu_id=0,
                def_input=staged_def,
            )

        self.assertEqual(
            command[command.index("--def-input") + 1],
            str(staged_def),
        )

    def test_ours_command_can_enable_explicit_capacity_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            command = runner.build_ours_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case="toy",
                result_dir=root / "result",
                committed_def=root / "committed.def",
                profile=runner._load_profile(),
                capacity_grid="8",
            )

        self.assertEqual(
            command[command.index("--buffering-segment-capacity-enabled") + 1],
            "1",
        )
        self.assertEqual(
            command[command.index("--buffering-segment-capacity-grid") + 1],
            "8",
        )

    def test_no_action_commit_is_a_valid_discrete_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text("{}\n", encoding="ascii")
            summary = {
                "status": "completed",
                "state_kind": "segment_count",
                "full_design_scope": True,
                "max_repeater_count": 3,
                "fixed_bsu_index": 7,
                "buffer_size_optimization": False,
                "strategy": "discrete_net_gradient",
                "inner_loop": {
                    "projection_count": 1,
                    "periodic_integer_projection_count": 0,
                    "terminal_metrics": {"tns": 0.0},
                    "trace_path": str(trace_path),
                    "terminal_reason": "no_positive_action",
                },
                "commit": {
                    "status": "skipped",
                    "reason": "no_projected_buffer_actions",
                    "action_count": 0,
                },
            }

            failures = runner._ours_contract_failures(
                summary,
                runner._load_profile(),
                segment_strategy="discrete_net_gradient",
            )

        self.assertEqual(failures, [])

    def test_no_improving_prefix_is_a_valid_no_action_commit(self):
        summary = {
            "status": "completed",
            "inner_loop": {"terminal_reason": "no_improving_prefix"},
            "commit": {
                "status": "skipped",
                "reason": "no_projected_buffer_actions",
                "action_count": 0,
            },
        }

        self.assertTrue(runner._is_no_action_commit(summary))

    def test_no_action_commit_does_not_hide_failed_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text("{}\n", encoding="ascii")
            summary = {
                "status": "completed",
                "state_kind": "segment_count",
                "full_design_scope": True,
                "max_repeater_count": 3,
                "fixed_bsu_index": 7,
                "buffer_size_optimization": False,
                "strategy": "discrete_net_gradient",
                "inner_loop": {
                    "projection_count": 1,
                    "periodic_integer_projection_count": 0,
                    "terminal_metrics": {"tns": 0.0},
                    "trace_path": str(trace_path),
                    "terminal_reason": "no_positive_action",
                },
                "commit": {
                    "status": "skipped",
                    "reason": "no_projected_buffer_actions",
                    "action_count": 0,
                    "failed_action_count": 1,
                },
            }

            failures = runner._ours_contract_failures(
                summary,
                runner._load_profile(),
                segment_strategy="discrete_net_gradient",
            )

        self.assertIn("failed_buffer_actions", failures)

    def test_run_ours_no_action_returns_input_def_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            input_def = root / "design" / "toy" / "toy.def"
            result_dir = root / "run" / "ours" / "autodmp_result"
            result_dir.mkdir(parents=True)
            trace_path = result_dir / "trace.json"
            trace_path.write_text("{}\n", encoding="ascii")
            summary = {
                "status": "completed",
                "state_kind": "segment_count",
                "full_design_scope": True,
                "max_repeater_count": 3,
                "fixed_bsu_index": 7,
                "buffer_size_optimization": False,
                "strategy": "discrete_net_gradient",
                "inner_loop": {
                    "projection_count": 1,
                    "periodic_integer_projection_count": 0,
                    "terminal_metrics": {"tns": 0.0},
                    "trace_path": str(trace_path),
                    "terminal_reason": "no_positive_action",
                },
                "commit": {
                    "status": "skipped",
                    "reason": "no_projected_buffer_actions",
                    "action_count": 0,
                },
            }
            (result_dir / "toy_buffering_inner_loop_summary.json").write_text(
                runner.json.dumps({"summary": summary}) + "\n",
                encoding="ascii",
            )

            with mock.patch.object(runner, "build_ours_command", return_value=["fake"]), mock.patch.object(
                runner,
                "_run_command",
                return_value={"status": "ok", "returncode": 0, "runtime_sec": 1.0},
            ):
                result = runner._run_ours(
                    python_bin=Path("/python"),
                    openroad_bin=Path("/openroad"),
                    benchmark_root=root,
                    case="toy",
                    case_root=root / "run",
                    profile=runner._load_profile(),
                    plan_only=False,
                    segment_strategy="discrete_net_gradient",
                    source_only=True,
                )

            audit = runner._read_json(root / "run" / "ours" / "action_audit.json")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(Path(result["final_def"]), input_def)
        self.assertEqual(result["buffer_count"], 0)
        self.assertEqual(result["action_audit_status"], "skipped")
        self.assertEqual(audit["reason"], "no_projected_buffer_actions")
        self.assertFalse(result["topology_mutated"])

    def test_discrete_route_command_overrides_continuous_start_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            command = runner.build_ours_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case="toy",
                result_dir=root / "result",
                committed_def=root / "committed.def",
                profile=runner._load_profile(),
                segment_strategy="discrete_net_gradient",
            )

        self.assertEqual(
            command[command.index("--buffering-segment-strategy") + 1],
            "discrete_net_gradient",
        )
        self.assertEqual(
            command[command.index("--buffering-segment-count-z-init") + 1],
            "0.0",
        )
        self.assertEqual(
            command[command.index("--buffering-segment-integer-projection-start-step") + 1],
            "0",
        )

    def test_capacity_contract_requires_selected_grid_and_strict_feasibility(self):
        summary = {
            "segment_capacity_enabled": True,
            "segment_capacity_selected_grid_factor": 8,
            "inner_loop": {
                "metrics_trace": [
                    {"segment_capacity_max_violation": 9.0e-5},
                ]
            },
        }

        self.assertEqual(runner._capacity_contract_failures(summary, "8"), [])
        summary["inner_loop"]["metrics_trace"][0]["segment_capacity_max_violation"] = 1.1e-4
        self.assertEqual(
            runner._capacity_contract_failures(summary, "8"),
            ["final_segment_capacity_infeasible"],
        )
        self.assertIn(
            "segment_capacity_grid_mismatch",
            runner._capacity_contract_failures(summary, "4"),
        )

    def test_plan_only_capacity_grid_is_forwarded_only_to_ours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            output_root = root / "runs"
            profile = runner._load_profile()
            profile["suite"] = ["toy"]
            profile_path = root / "capacity_profile.json"
            runner._write_json(profile_path, profile)
            status = runner.main(
                [
                    "--profile",
                    str(profile_path),
                    "--benchmark-root",
                    str(root),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "capacity_grid",
                    "--case",
                    "toy",
                    "--method",
                    "ours",
                    "--capacity-grid",
                    "4",
                    "--plan-only",
                ]
            )
            command = (
                output_root
                / "capacity_grid"
                / "toy"
                / "ours"
                / "command.txt"
            ).read_text(encoding="utf-8")
            manifest = runner._read_json(output_root / "capacity_grid" / "manifest.json")

        self.assertEqual(status, 0)
        self.assertIn("--buffering-segment-capacity-enabled 1", command)
        self.assertIn("--buffering-segment-capacity-grid 4", command)
        self.assertEqual(manifest["capacity_grid"], "4")

    def test_fidelity_counts_and_profile_are_diagnostic_and_one_final_commit(self):
        profile = runner._load_profile()
        fidelity = runner._fidelity_profile(profile)

        self.assertEqual(runner._parse_fidelity_counts("3,1,0,3", nmax=3), (0, 1, 3))
        with self.assertRaises(ValueError):
            runner._parse_fidelity_counts("4", nmax=3)
        self.assertEqual(profile["segment_mode"]["continuous_steps"], 120)
        self.assertEqual(fidelity["segment_mode"]["continuous_steps"], 1)
        self.assertEqual(fidelity["segment_mode"]["continuous_lr"], 0.0)
        self.assertEqual(fidelity["segment_mode"]["z_init"], 0.0)
        self.assertEqual(fidelity["segment_mode"]["integer_projection_interval"], 0)
        self.assertTrue(fidelity["segment_mode"]["one_final_hard_commit"])

    def test_fidelity_evaluation_uses_stage_order_and_chain_slew(self):
        observation = {
            "repeater_count": 2,
            "pysta_device_probe_rows": [
                {
                    "sense": "rise",
                    "analytic_transfer_repeater_count": 2,
                    "analytic_transfer_buffer_delays": [10.0, 8.0],
                    "analytic_transfer_buffer_input_slews": [5.0, 4.0],
                    "analytic_transfer_buffer_output_loads": [0.01, 0.02],
                    "analytic_transfer_buffer_output_slews": [4.0, 3.0],
                }
            ],
            "committed_arc_samples": [
                {
                    "liberty_gate_delay_ps": 10.5,
                    "input_slew_ps": 5.2,
                    "output_load_cap_pf": 0.012,
                    "liberty_output_slew_ps": 3.2,
                },
                {
                    "liberty_gate_delay_ps": 8.2,
                    "input_slew_ps": 4.1,
                    "output_load_cap_pf": 0.021,
                    "liberty_output_slew_ps": 3.1,
                },
            ],
        }

        decision, payload = runner._evaluate_fidelity_observations([observation])

        self.assertEqual(decision, "suitable_for_relaxed_stage_fidelity")
        stages = payload["comparisons"][0]["stages"]
        self.assertEqual(stages[0]["output_slew_reference"], "next_stage_input_slew_ps")
        self.assertEqual(stages[1]["output_slew_reference"], "final_stage_liberty_output_slew_ps")

    def test_committed_segment_arc_samples_preserve_stage_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "commit.json"
            artifact.write_text(
                """{
  \"results\": [
    {
      \"segment_id\": 17,
      \"action_id\": 2,
      \"segment_split_index\": 1,
      \"segment_split_count_on_edge\": 2,
      \"segment_split_ratio\": 0.6666666667,
      \"inserted_buffer_name\": \"b1\",
      \"hard_commit_diagnostic\": {
        \"cell_arc_delay_samples\": {
          \"status\": \"ok\",
          \"buffer_master_name\": \"BUF\",
          \"selected_max_delay_sample\": {
            \"input_slew\": 20.0,
            \"output_load_cap_pf\": 0.02,
            \"liberty_gate_delay\": 30.0,
            \"liberty_output_slew\": 40.0
          },
          \"pin_arrival_delta\": {\"r_arrival_delta\": 31.0, \"f_arrival_delta\": 32.0}
        }
      }
    },
    {
      \"segment_id\": 17,
      \"action_id\": 1,
      \"segment_split_index\": 0,
      \"segment_split_count_on_edge\": 2,
      \"segment_split_ratio\": 0.3333333333,
      \"inserted_buffer_name\": \"b0\",
      \"hard_commit_diagnostic\": {\"cell_arc_delay_samples\": {\"status\": \"ok\"}}
    }
  ]
}\n""",
                encoding="utf-8",
            )
            samples = runner._committed_segment_arc_samples(
                {"commit": {"artifact_path": str(artifact)}},
                segment_id=17,
            )

        self.assertEqual([sample["inserted_buffer_name"] for sample in samples], ["b0", "b1"])
        self.assertEqual(samples[1]["liberty_gate_delay_ps"], 30.0)
        self.assertEqual(samples[1]["pin_arrival_delta_ps"]["fall"], 32.0)

    def test_fidelity_plan_only_does_not_read_an_empty_summary_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._benchmark(Path(tmp))
            report = runner._run_segment_count_fidelity(
                python_bin=Path("/python"),
                openroad_bin=Path("/openroad"),
                benchmark_root=root,
                case="toy",
                run_root=root / "runs",
                profile=runner._load_profile(),
                fidelity_segment_id=7,
                counts=(0, 1),
                plan_only=True,
            )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            [row["repeater_count"] for row in report["observations"]],
            [0, 1],
        )

    def test_table_reports_b0_delta_and_action_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "comparison.md"
            runner._write_table(
                table,
                [
                    {
                        "case": "toy",
                        "method": "b0",
                        "status": "pass",
                        "wns_ns": -0.2,
                        "tns_ns": -10.0,
                        "action_audit_status": "pass",
                    },
                    {
                        "case": "toy",
                        "method": "ours",
                        "status": "pass",
                        "wns_ns": -0.1,
                        "tns_ns": -4.0,
                        "action_audit_status": "pass",
                    },
                ],
                "unit",
            )
            contents = table.read_text(encoding="utf-8")

        self.assertIn("Delta TNS vs B0 (ns)", contents)
        self.assertIn("| toy | ours | pass | -0.100000 | -4.000000 | 6.000000 |", contents)
        self.assertIn("| toy | ours | pass | -0.100000 | -4.000000 | 6.000000 |  |", contents)
        self.assertIn("| pass |", contents)

    def test_def_audit_rejects_existing_master_change_and_wrong_new_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.def"
            after = root / "after.def"
            before.write_text(
                "COMPONENTS 1 ;\n- u0 NAND2 ;\nEND COMPONENTS\n",
                encoding="utf-8",
            )
            after.write_text(
                "COMPONENTS 2 ;\n- u0 INV ;\n- rebuffer0 HB1 ;\nEND COMPONENTS\n",
                encoding="utf-8",
            )
            audit = runner.audit_def_mutation(
                input_def=before,
                final_def=after,
                expected_buffer_master="HB3",
                require_new_buffers=True,
            )

        self.assertEqual(audit["status"], "failed")
        self.assertIn("existing_instance_master_changed", audit["failures"])
        self.assertIn("new_instance_not_expected_buffer_master", audit["failures"])


if __name__ == "__main__":
    unittest.main()
