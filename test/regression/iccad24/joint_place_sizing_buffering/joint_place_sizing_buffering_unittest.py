import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("run_joint_place_sizing_buffering", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class JointPlaceSizingBufferingRegressionTest(unittest.TestCase):
    def test_build_placer_command_uses_canonical_joint_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=25,
                outer_iterations=1,
                max_buffers_per_segment=2,
                gate="gate_a",
            )

        self.assertIn("--flow-kind", command)
        self.assertIn("joint", command)
        self.assertIn("--joint-buffer-commit-period", command)
        self.assertIn("25", command)
        self.assertIn("--joint-buffer-max-per-segment", command)
        self.assertIn("2", command)

    def test_build_placer_command_can_select_proximal_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=100,
                outer_iterations=1,
                max_buffers_per_segment=3,
                gate="gate_a",
                joint_quality_profile="proximal_alternating_v1",
            )

        self.assertIn("--joint-quality-profile", command)
        self.assertIn("proximal_alternating_v1", command)

    def test_build_placer_command_can_select_segment_direct_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=0,
                outer_iterations=2,
                max_buffers_per_segment=3,
                gate="gate_c",
                joint_quality_profile="segment_count_direct_joint_v1",
            )

        self.assertIn("--joint-quality-profile", command)
        self.assertIn("segment_count_direct_joint_v1", command)
        period_index = command.index("--joint-buffer-commit-period")
        self.assertEqual(command[period_index + 1], "0")

    def test_gate_b_command_forces_short_commit_without_real_commit_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=1,
                outer_iterations=1,
                max_buffers_per_segment=3,
                gate="gate_b",
            )

        self.assertIn("--timing-topology-enable-overflow-threshold", command)
        self.assertIn("1.1", command)
        self.assertIn("--buffering-commit-enabled", command)
        self.assertIn("0", command)

    def test_segment_direct_gate_b_keeps_fixed_milestones_and_disables_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=1,
                commit_period=100,
                outer_iterations=1,
                max_buffers_per_segment=3,
                gate="gate_b",
                joint_quality_profile="segment_count_direct_joint_v1",
            )

        period_index = command.index("--joint-buffer-commit-period")
        self.assertEqual(command[period_index + 1], "0")
        self.assertNotIn("--timing-topology-enable-overflow-threshold", command)
        commit_index = command.index("--buffering-commit-enabled")
        self.assertEqual(command[commit_index + 1], "0")

    def test_gate_c_command_enables_real_commit_without_forcing_overflow_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=1,
                outer_iterations=1,
                max_buffers_per_segment=3,
                gate="gate_c",
            )

        self.assertIn("--buffering-commit-enabled", command)
        self.assertIn("1", command)
        self.assertIn("--buffering-committed-def-path", command)
        self.assertNotIn("--timing-topology-enable-overflow-threshold", command)

    def test_extra_placer_args_are_appended_for_focused_smokes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                iterations=2,
                commit_period=100,
                outer_iterations=1,
                max_buffers_per_segment=3,
                gate="gate_c",
                extra_placer_args=[
                    "--buffering-segment-count-z-init",
                    "0.6",
                    "--buffering-max-selected-actions",
                    "1",
                ],
            )

        self.assertEqual(
            command[-4:],
            [
                "--buffering-segment-count-z-init",
                "0.6",
                "--buffering-max-selected-actions",
                "1",
            ],
        )

    def test_summarize_joint_result_requires_commit_trace_for_gate_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text('{"summary": {"buffer_commit_trace": []}}', encoding="utf-8")

            row = runner.summarize_joint_result(
                case="NV_NVDLA_partition_m",
                gate="gate_b",
                run_id="unit",
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_path=Path(tmp) / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["failure"], "missing_gate_b_commit_trace")

    def test_summarize_joint_result_accepts_gate_b_commit_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "proximal": {
                      "enabled": true,
                      "profile": "proximal_alternating_v1",
                      "total_weighted": 0.125
                    },
                    "buffering_lane": {
                      "projection": {
                        "min_z_to_insert": 0.15,
                        "projected_candidate_count_before_topk": 8,
                        "projected_candidate_count_after_criticality_filter": 5,
                        "criticality_filtered_candidate_count": 3,
                        "projected_candidate_count": 3,
                        "criticality_overlay_status": "ok",
                        "criticality_overlay_updated_sink_count": 32,
                        "criticality_overlay_negative_sink_count": 7,
                        "selected_candidate_diagnostics": [
                          {
                            "segment_id": 17,
                            "net_name": "n17",
                            "z_value": 0.42,
                            "bsu_value": 0.75,
                            "projection_selection_score": 12.5,
                            "downstream_sink_criticality_sum": 25.0
                          }
                        ],
                        "z_value_distribution": {
                          "max": 0.42,
                          "ge_min_insert_count": 8,
                          "ge_0p5_count": 0
                        }
                      }
                    },
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "projected_buffer_count": 3,
                        "commit": {
                          "status": "disabled",
                          "transaction_status": "disabled",
                          "qor_status": "not_applicable_no_accepted_commit",
                          "qor_pass": false,
                          "accepted_action_count": 0
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="NV_NVDLA_partition_m",
                gate="gate_b",
                run_id="unit",
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_path=Path(tmp) / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["commit_count"], 1)
        self.assertEqual(row["inserted_buffers"], 0)
        self.assertEqual(row["joint_quality_profile"], "proximal_alternating_v1")
        self.assertEqual(row["proximal_enabled"], True)
        self.assertEqual(row["proximal_total_weighted"], 0.125)
        self.assertEqual(row["segment_projection_min_z_to_insert"], 0.15)
        self.assertEqual(row["segment_projection_candidates_before_topk"], 8)
        self.assertEqual(row["segment_projection_candidates_after_criticality_filter"], 5)
        self.assertEqual(row["segment_projection_criticality_filtered_count"], 3)
        self.assertEqual(row["segment_projection_candidates"], 3)
        self.assertEqual(row["segment_criticality_overlay_status"], "ok")
        self.assertEqual(row["segment_criticality_overlay_updated_sink_count"], 32)
        self.assertEqual(row["segment_criticality_overlay_negative_sink_count"], 7)
        self.assertEqual(row["segment_z_max"], 0.42)
        self.assertEqual(row["segment_z_ge_min_insert_count"], 8)
        self.assertEqual(row["segment_z_ge_0p5_count"], 0)
        self.assertEqual(row["commit_transaction_status"], "disabled")
        self.assertEqual(row["commit_qor_status"], "not_applicable_no_accepted_commit")
        self.assertEqual(row["commit_qor_pass"], False)
        self.assertEqual(row["selected_segment_id"], 17)
        self.assertEqual(row["selected_net_name"], "n17")
        self.assertEqual(row["selected_z_value"], 0.42)
        self.assertEqual(row["selected_bsu_value"], 0.75)
        self.assertEqual(row["selected_projection_score"], 12.5)
        self.assertEqual(row["selected_downstream_criticality_sum"], 25.0)

    def test_summarize_joint_result_reports_transaction_pass_and_qor_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "accepted",
                          "transaction_status": "accepted",
                          "qor_status": "fail_tns_nonpositive",
                          "qor_pass": false,
                          "commit_enabled": true,
                          "accepted_action_count": 1,
                          "delta_tns": -0.639
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_joint_post_rebuild_summary.json").write_text(
                """
                {
                  "post_rebuild_result": {
                    "status": "ok",
                    "post_rebuild_timing_status": "ok",
                    "fresh_nonlinear_place_constructed": true
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_buffer_commit_pydb_refresh_summary.json").write_text(
                """
                {
                  "refresh_status": "ok",
                  "refresh_generation": 1
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=30,
                commit_period=100,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["inserted_buffers"], 1)
        self.assertEqual(row["commit_transaction_status"], "accepted")
        self.assertEqual(row["commit_qor_status"], "fail_tns_nonpositive")
        self.assertEqual(row["commit_qor_pass"], False)
        self.assertEqual(row["committed_delta_tns"], -0.639)

    def test_summarize_joint_result_requires_gate_c_real_commit_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "disabled",
                          "commit_enabled": false
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="NV_NVDLA_partition_m",
                gate="gate_c",
                run_id="unit",
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_path=Path(tmp) / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["failure"], "gate_c_commit_disabled")

    def test_summarize_joint_result_accepts_gate_c_structured_skipped_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "skipped",
                          "reason": "no_projected_buffer_actions",
                          "commit_enabled": true,
                          "accepted_action_count": 0
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="NV_NVDLA_partition_m",
                gate="gate_c",
                run_id="unit",
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_path=Path(tmp) / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["commit_count"], 1)
        self.assertEqual(row["inserted_buffers"], 0)

    def test_summarize_joint_result_requires_post_rebuild_after_accepted_gate_c_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "accepted",
                          "commit_enabled": true,
                          "accepted_action_count": 1
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=Path(tmp),
                log_path=Path(tmp) / "run.log",
                command_path=Path(tmp) / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["failure"], "missing_gate_c_post_rebuild_summary")

    def test_summarize_joint_result_requires_refresh_summary_after_accepted_gate_c_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "accepted",
                          "commit_enabled": true,
                          "accepted_action_count": 1
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_joint_post_rebuild_summary.json").write_text(
                """
                {
                  "post_rebuild_result": {
                    "status": "ok",
                    "post_rebuild_timing_status": "ok",
                    "fresh_nonlinear_place_constructed": true,
                    "openroad_eco_mode": "resynth_once",
                    "openroad_eco_status": "ok",
                    "openroad_eco_summary_path": "/tmp/unit_eco.json"
                  }
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["failure"], "missing_gate_c_refresh_summary")

    def test_summarize_joint_result_accepts_refresh_and_post_rebuild_after_accepted_gate_c_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "accepted",
                          "commit_enabled": true,
                          "accepted_action_count": 1
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_joint_post_rebuild_summary.json").write_text(
                """
                {
                  "post_rebuild_result": {
                    "status": "ok",
                    "post_rebuild_timing_status": "ok",
                    "fresh_nonlinear_place_constructed": true
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_buffer_commit_pydb_refresh_summary.json").write_text(
                """
                {
                  "artifact_scope": "buffer_commit_pydb_refresh",
                  "refresh_status": "ok",
                  "refresh_generation": 1
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["refresh_status"], "ok")
        self.assertEqual(row["refresh_generation"], 1)
        self.assertEqual(row["post_rebuild_status"], "ok")
        self.assertEqual(row["post_rebuild_timing_status"], "ok")
        self.assertTrue(row["fresh_nonlinear_place_constructed"])
        self.assertIsNone(row["openroad_eco_mode"])
        self.assertIsNone(row["openroad_eco_status"])
        self.assertEqual(row["openroad_eco_summary_path"], "")

    def test_summarize_joint_result_accepts_segment_joint_terminal_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "completed",
                        "commit": {
                          "status": "accepted",
                          "transaction_status": "accepted",
                          "accepted_action_count": 2,
                          "after_wns": -8.0,
                          "after_tns": -70.0,
                          "delta_tns": 30.0
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_segment_joint_committed_verification.json").write_text(
                """
                {
                  "status": "ok",
                  "qor_status": "pass",
                  "delta_tns": 30.0,
                  "added_area_um2": 2.5,
                  "final_def_sha256": "fixture-sha256",
                  "opensta_after": {"wns": -8.0, "tns": -70.0},
                  "final_placement_metrics": {"overflow": 0.12},
                  "refresh_summary": {
                    "status": "ok",
                    "rebuild": {"topology_generation": 1}
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_segment_joint_def_only_opensta_replay.json").write_text(
                """
                {
                  "status": "ok",
                  "load_contract": {
                    "design_carrier": "final_def",
                    "fresh_openroad_bridge": true,
                    "verilog_read": false,
                    "final_def_sha256": "fixture-sha256"
                  },
                  "opensta_after": {"wns": -7.75, "tns": -69.5},
                  "def_only_vs_in_process_gap": {"wns": 0.25, "tns": 0.5}
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=30,
                commit_period=0,
                max_buffers_per_segment=3,
                summary_path=summary_path,
                joint_quality_profile="segment_count_direct_joint_v1",
                outer_iterations=2,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["joint_quality_profile"], "segment_count_direct_joint_v1")
        self.assertEqual(row["commit_period"], 0)
        self.assertEqual(row["outer_iterations"], 2)
        self.assertEqual(row["terminal_status"], "ok")
        self.assertEqual(row["terminal_qor_status"], "pass")
        self.assertEqual(row["def_only_opensta_status"], "ok")
        self.assertEqual(row["def_only_opensta_wns"], -7.75)
        self.assertEqual(row["def_only_opensta_tns"], -69.5)
        self.assertEqual(row["def_only_vs_in_process_wns"], 0.25)
        self.assertEqual(row["def_only_vs_in_process_tns"], 0.5)
        self.assertTrue(row["def_only_final_def_hash_match"])
        self.assertEqual(row["terminal_added_area_um2"], 2.5)
        self.assertEqual(row["terminal_final_overflow"], 0.12)
        self.assertEqual(row["committed_delta_tns"], 30.0)
        self.assertEqual(row["refresh_status"], "ok")
        self.assertEqual(row["refresh_generation"], 1)

    def test_segment_joint_gate_c_rejects_missing_def_only_opensta_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [{
                      "status": "completed",
                      "commit": {
                        "status": "accepted",
                        "accepted_action_count": 1
                      }
                    }]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_segment_joint_committed_verification.json").write_text(
                """
                {
                  "status": "ok",
                  "qor_status": "pass",
                  "final_def_sha256": "fixture-sha256",
                  "opensta_after": {"wns": -8.0, "tns": -70.0},
                  "refresh_summary": {"status": "ok"}
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=0,
                max_buffers_per_segment=3,
                summary_path=summary_path,
                joint_quality_profile="segment_count_direct_joint_v1",
            )

        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["failure"], "missing_gate_c_def_only_opensta_replay")

    def test_summarize_joint_result_reports_optional_openroad_eco_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            summary_path = result_dir / "unit_joint_flow_summary.json"
            summary_path.write_text(
                """
                {
                  "summary": {
                    "buffer_commit_trace": [
                      {
                        "status": "ok",
                        "commit": {
                          "status": "accepted",
                          "commit_enabled": true,
                          "accepted_action_count": 1
                        }
                      }
                    ]
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_joint_post_rebuild_summary.json").write_text(
                """
                {
                  "post_rebuild_result": {
                    "status": "ok",
                    "post_rebuild_timing_status": "ok",
                    "fresh_nonlinear_place_constructed": true,
                    "openroad_eco_mode": "resynth_once",
                    "openroad_eco_status": "ok",
                    "openroad_eco_summary_path": "/tmp/unit_eco.json"
                  }
                }
                """,
                encoding="utf-8",
            )
            (result_dir / "unit_buffer_commit_pydb_refresh_summary.json").write_text(
                """
                {
                  "artifact_scope": "buffer_commit_pydb_refresh",
                  "refresh_status": "ok",
                  "refresh_generation": 1
                }
                """,
                encoding="utf-8",
            )

            row = runner.summarize_joint_result(
                case="unit",
                gate="gate_c",
                run_id="unit",
                result_dir=result_dir,
                log_path=result_dir / "run.log",
                command_path=result_dir / "command.txt",
                returncode=0,
                iterations=2,
                commit_period=1,
                max_buffers_per_segment=3,
                summary_path=summary_path,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["openroad_eco_mode"], "resynth_once")
        self.assertEqual(row["openroad_eco_status"], "ok")
        self.assertEqual(row["openroad_eco_summary_path"], "/tmp/unit_eco.json")

    def test_average_step_runtime_uses_full_step_log_not_metric_eval_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                """
                [INFO] DREAMPlace - iteration 0, time 0.800ms
                [INFO] DREAMPlace - full step 10000.000 ms
                [INFO] DREAMPlace - full step 12000.000 ms
                """,
                encoding="utf-8",
            )
            records = [{"runtime_seconds": 0.0008}, {"runtime_seconds": 0.0009}]

            value = runner._average_step_runtime(records, log_path)

        self.assertAlmostEqual(value, 11.0)

    def test_average_step_runtime_ignores_runtime_seconds_without_full_step_evidence(self):
        records = [{"runtime_seconds": 0.0008}, {"runtime_seconds": 0.0009}]

        value = runner._average_step_runtime(records)

        self.assertIsNone(value)

    @staticmethod
    def _write_case(root, case):
        case_dir = root / "design" / case
        workspace = case_dir / "workspace" / "config" / "dreamplace_config"
        workspace.mkdir(parents=True)
        (workspace / "param.json").write_text("{}", encoding="utf-8")
        for suffix in (".def", ".v", ".sdc"):
            (case_dir / f"{case}{suffix}").write_text("", encoding="utf-8")
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
