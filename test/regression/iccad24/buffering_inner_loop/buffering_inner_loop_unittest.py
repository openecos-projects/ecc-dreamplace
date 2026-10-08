import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("run_buffering_inner_loop", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BufferingInnerLoopRegressionTest(unittest.TestCase):
    def test_candidate_route_b_profile_freezes_production_recipe(self):
        profile = runner._load_candidate_profile(runner.PROFILE_PATH)

        self.assertEqual(profile["suite"], ["NV_NVDLA_partition_m"])
        self.assertEqual(profile["candidate_strategy"], "discrete_net_gradient")
        self.assertEqual(profile["rounds"], 5)
        self.assertEqual(profile["fixed_bsu_index"], 7)
        self.assertEqual(profile["selection_fraction"], 0.001)
        self.assertEqual(
            profile["candidate_net_subgraph_backend"],
            "cuda_explicit_autograd",
        )
        self.assertEqual(profile["candidate_generation"]["mode"], "equal_count")
        self.assertEqual(
            profile["candidate_generation"]["max_candidates_per_segment"],
            3,
        )
        self.assertTrue(profile["one_final_coordinate_commit"])

    def test_candidate_route_b_profile_writes_equal_count_k_to_effective_params(self):
        profile = runner._load_candidate_profile(runner.PROFILE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "params.json"
            output = Path(tmp) / "effective.json"
            canonical.write_text("{}\n", encoding="utf-8")

            runner._write_profile_params(
                canonical_params_path=canonical,
                output_path=output,
                profile=profile,
            )
            effective = runner._read_json(output)

        self.assertEqual(effective["buffering_candidate_policy"], "segment_only")
        self.assertEqual(effective["buffering_max_repeaters_per_segment"], 3)
        self.assertEqual(effective["buffering_include_tree_node_candidates"], 0)
        self.assertEqual(effective["timing_propagation_device"], "cuda")

    def test_summarize_inner_loop_result_passes_for_canonical_completed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "NV_NVDLA_partition_m_buffering_inner_loop_summary.json"
            summary_path.write_text(
                """
                {
                  "artifact_version": 1,
                  "metadata": {
                    "artifact_scope": "buffering_inner_loop",
                    "canonical_entry": "NonLinearPlace._maybe_run_buffering_inner_loop"
                  },
                  "summary": {
                    "status": "completed",
                    "mode": "candidate",
                    "state_build_status": "ok",
                    "candidate_count": 14609,
                    "net_count": 27732,
                    "last_projected_buffer_count": 14609,
                    "runtime_refresh": {
                      "status": "ok",
                      "projected_buffer_count": 14609,
                      "affected_net_count": 2211
                    },
                    "inner_loop": {
                      "iterations": 1,
                      "projection_count": 2,
                      "metrics_trace": [
                        {"loss": 143.0, "tns": -143.0, "wns": -0.53}
                      ]
                    },
                    "commit": {
                      "status": "disabled",
                      "reason": "commit_not_requested"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_inner_loop_result(
                case="NV_NVDLA_partition_m",
                run_id="unit",
                run_root=Path(tmp),
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_returncode=0,
                buffering_mode="candidate",
                continuous_steps=1,
                continuous_lr=0.01,
                max_selected_actions=None,
                commit_enabled=False,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["summary_status"], "completed")
        self.assertEqual(row["state_build_status"], "ok")
        self.assertEqual(row["candidate_count"], 14609)
        self.assertEqual(row["runtime_refresh_status"], "ok")
        self.assertEqual(row["commit_status"], "disabled")
        self.assertEqual(row["loss_initial"], 143.0)
        self.assertEqual(row["loss_final"], 143.0)
        self.assertEqual(row["tns_initial"], -143.0)
        self.assertEqual(row["tns_final"], -143.0)

    def test_summarize_inner_loop_result_accepts_structured_enabled_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "status": "completed",
                    "state_build_status": "ok",
                    "candidate_count": 1,
                    "runtime_refresh": {"status": "ok"},
                    "commit": {
                      "status": "accepted",
                      "action_count": 1,
                      "attempted_action_count": 1,
                      "accepted_action_count": 1,
                      "before_wns": -1.0,
                      "before_tns": -10.0,
                      "after_wns": -0.9,
                      "after_tns": -8.0,
                      "delta_wns": 0.1,
                      "delta_tns": 2.0
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_inner_loop_result(
                case="NV_NVDLA_partition_m",
                run_id="unit",
                run_root=Path(tmp),
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_returncode=0,
                buffering_mode="candidate",
                continuous_steps=1,
                continuous_lr=0.01,
                max_selected_actions=10,
                commit_enabled=True,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["commit_status"], "accepted")
        self.assertEqual(row["commit_accepted_action_count"], 1)
        self.assertEqual(row["commit_delta_tns"], 2.0)

    def test_summarize_inner_loop_result_passes_for_segment_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "status": "completed",
                    "mode": "segment",
                    "state_build_status": "ok",
                    "state_kind": "segment_count",
                    "segment_count": 123,
                    "affected_net_count": 17,
                    "net_count": 27732,
                    "last_projected_buffer_count": 9,
                    "segment_count_timing_backend_used": "prepared_python",
                    "runtime_refresh": {
                      "status": "ok",
                      "projected_buffer_count": 9,
                      "affected_net_count": 5
                    },
                    "inner_loop": {
                      "iterations": 1,
                      "projection_count": 2,
                      "metrics_trace": [
                        {
                          "loss": 10.0,
                          "tns": -10.0,
                          "wns": -0.1,
                          "grad_bu_norm": 2.0,
                          "grad_z_norm": 3.0,
                          "grad_bsu_norm": 4.0
                        }
                      ]
                    },
                    "commit": {
                      "status": "disabled",
                      "reason": "commit_not_requested"
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_inner_loop_result(
                case="NV_NVDLA_partition_m",
                run_id="unit",
                run_root=Path(tmp),
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_returncode=0,
                buffering_mode="segment",
                continuous_steps=1,
                continuous_lr=0.01,
                max_selected_actions=None,
                commit_enabled=False,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["state_kind"], "segment_count")
        self.assertEqual(row["segment_count"], 123)
        self.assertEqual(row["affected_net_count"], 17)
        self.assertEqual(row["projected_buffer_count"], 9)
        self.assertEqual(row["grad_z_initial_norm"], 3.0)
        self.assertEqual(row["grad_z_final_norm"], 3.0)
        self.assertEqual(row["grad_bu_initial_norm"], 2.0)
        self.assertEqual(row["grad_bu_final_norm"], 2.0)
        self.assertEqual(row["grad_bsu_initial_norm"], 4.0)
        self.assertEqual(row["grad_bsu_final_norm"], 4.0)
        self.assertEqual(row["segment_count_timing_backend_used"], "prepared_python")

    def test_build_placer_command_uses_canonical_buffering_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iccad24_benchmark"
            case = "NV_NVDLA_partition_m"
            case_dir = root / "design" / case
            workspace_config = case_dir / "workspace" / "config" / "dreamplace_config"
            workspace_config.mkdir(parents=True)
            (case_dir / f"{case}.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
            (case_dir / f"{case}.v").write_text(f"module {case}; endmodule\n", encoding="utf-8")
            (case_dir / f"{case}.sdc").write_text("", encoding="utf-8")
            (workspace_config / "param.json").write_text("{}", encoding="utf-8")
            for path in runner.asap7_inputs(root).values():
                if isinstance(path, list):
                    for item in path:
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_text("", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

            command = runner.build_placer_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case=case,
                result_dir=Path(tmp) / "result",
                buffering_mode="segment",
                continuous_steps=1,
                continuous_lr=0.01,
                max_selected_actions=100,
                commit_enabled=True,
            )
            candidate_command = runner.build_placer_command(
                python_bin=Path("/python"),
                benchmark_root=root,
                case=case,
                result_dir=Path(tmp) / "candidate_result",
                buffering_mode="candidate",
                candidate_strategy="discrete_net_gradient",
                fixed_bsu_index=7,
                gpu_id=1,
                continuous_steps=5,
                continuous_lr=None,
                committed_def_path=Path(tmp) / "candidate_committed.def",
                commit_enabled=True,
            )

        self.assertIn("--flow-kind", command)
        self.assertIn("buffering", command)
        self.assertIn("--buffering-mode", command)
        self.assertIn("segment", command)
        self.assertIn("--with-sta", command)
        self.assertIn("--buffering-continuous-steps", command)
        self.assertIn("--buffering-continuous-lr", command)
        self.assertIn("--buffering-max-selected-actions", command)
        self.assertIn("100", command)
        self.assertIn("--buffering-commit-enabled", command)
        self.assertIn("1", command)
        self.assertFalse(any(flag.startswith("--buffering-") and "du" in flag for flag in command))
        self.assertIn("--buffering-candidate-strategy", candidate_command)
        self.assertIn("discrete_net_gradient", candidate_command)
        self.assertIn("--buffering-fixed-bsu-index", candidate_command)
        self.assertIn("7", candidate_command)
        self.assertIn("--gpu-id", candidate_command)
        self.assertIn("--buffering-committed-def-path", candidate_command)
        self.assertNotIn("--buffering-continuous-lr", candidate_command)
        self.assertNotIn("--buffering-max-selected-actions", candidate_command)

    def test_campaign_gpu_binding_and_source_only_are_public_cli_options(self):
        args = runner.build_arg_parser().parse_args(
            [
                "--cuda-visible-devices",
                "2",
                "--logical-gpu-id",
                "0",
                "--source-only",
            ]
        )

        self.assertEqual(args.cuda_visible_devices, "2")
        self.assertEqual(args.logical_gpu_id, 0)
        self.assertTrue(args.source_only)


if __name__ == "__main__":
    unittest.main()
