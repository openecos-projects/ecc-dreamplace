"""Timing stage, endpoint classification and changed-cell artifact contracts."""

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn

from dreamplace.NonLinearPlace import NonLinearPlace


class TimingArtifactsTest(unittest.TestCase):
    def _make_metric_record(
        self,
        iteration,
        wns,
        tns,
        timing_objective,
        slew_violation,
        cap_violation,
        leakage,
    ):
        return SimpleNamespace(
            iteration=iteration,
            detailed_step=None,
            objective=None,
            wirelength=None,
            density=None,
            density_weight=None,
            hpwl=None,
            overflow=None,
            max_density=None,
            gamma=None,
            wns=wns,
            tns=tns,
            timing_objective=timing_objective,
            slew_violation=slew_violation,
            cap_violation=cap_violation,
            leakage=leakage,
            eval_time=0.1,
            size_min=None,
            size_mean=None,
            size_max=None,
            vt_class_proportions=None,
        )


    def test_write_timing_stage_artifact_exports_strict_stage_compare_latest(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )

            with open(os.path.join(tmpdir, "FFT_place_stage_endpoint_timing_compare_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "python_wns": -120.0,
                        "python_tns": -1000.0,
                        "ieda_wns": -140.0,
                        "ieda_tns": -1200.0,
                        "backend_metadata": {
                            "backend": "openroad",
                            "sta_reference_role": "golden_opensta",
                            "sta_is_golden": True,
                            "parasitics_initialization": "placement",
                            "sta_state_status": "validated_golden",
                        },
                    },
                    f,
                )
            with open(os.path.join(tmpdir, "FFT_endpoint_timing_compare_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "python_wns": -100.0,
                        "python_tns": -900.0,
                        "ieda_wns": -110.0,
                        "ieda_tns": -950.0,
                        "backend_metadata": {
                            "backend": "openroad",
                            "sta_reference_role": "golden_opensta",
                            "sta_is_golden": True,
                            "parasitics_initialization": "placement",
                            "sta_state_status": "validated_golden",
                        },
                    },
                    f,
                )
            with open(os.path.join(tmpdir, "FFT_place_stage_endpoint_timing_compare.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"])
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "u0/Y",
                        "cpp_slack": -140.0,
                        "py_slack": -120.0,
                        "slack_delta": 20.0,
                    }
                )
            with open(os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"])
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "u1/Y",
                        "cpp_slack": -110.0,
                        "py_slack": -100.0,
                        "slack_delta": 10.0,
                    }
                )
            with open(os.path.join(tmpdir, "FFT_projection.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "inst_id",
                        "changed_cell",
                        "current_size",
                        "continuous_size",
                        "projected_size",
                        "term_timing_penalty",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "inst_id": 3,
                        "changed_cell": True,
                        "current_size": 1.0,
                        "continuous_size": 1.4,
                        "projected_size": 1.1,
                        "term_timing_penalty": 0.6,
                    }
                )
            with open(os.path.join(tmpdir, "FFT_projection_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"num_changed_cells": 1}, f)

            placer._write_timing_stage_artifact(
                params,
                [self._make_metric_record(1, -120.0, -1000.0, -10.0, 1.2, 0.4, 0.2)],
                [self._make_metric_record(2, -100.0, -900.0, -8.5, 0.8, 0.2, 0.15)],
            )

            latest_path = os.path.join(tmpdir, "FFT_stage_compare_latest.json")
            self.assertTrue(os.path.exists(latest_path))
            with open(latest_path, encoding="utf-8") as f:
                latest = json.load(f)

            self.assertEqual(
                latest["metadata"]["strict_stage_order"],
                ["pre_projection", "post_projection", "post_legalization"],
            )
            self.assertIn("pre_projection", latest)
            self.assertIn("post_projection", latest)
            self.assertIn("post_legalization", latest)
            for stage_name in ("pre_projection", "post_projection", "post_legalization"):
                self.assertIn("WNS", latest[stage_name])
                self.assertIn("TNS", latest[stage_name])
                self.assertIn("SlewVio", latest[stage_name])
                self.assertIn("CapVio", latest[stage_name])
                self.assertIn("Leakage", latest[stage_name])
                self.assertIn("primary", latest[stage_name])
                self.assertIn("secondary", latest[stage_name])
            self.assertEqual(latest["changed_cell_summary"]["num_changed_cells"], 1)
            self.assertEqual(latest["changed_cell_summary"]["rows"][0]["inst_id"], 3)
            self.assertEqual(
                latest["changed_cell_summary"]["rows"][0]["critical_endpoint_cone_membership"],
                True,
            )
            self.assertEqual(latest["top_critical_endpoints"]["pre_projection"][0]["pin_name"], "u0/Y")


    def test_write_timing_stage_artifact_prefers_dedicated_post_projection_snapshot(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )

            placer._compute_post_projection_timing_snapshot = lambda *args, **kwargs: {
                "record": {
                    "iteration": 1,
                    "wns": -80.0,
                    "tns": -700.0,
                    "timing_objective": -7.0,
                    "combined_timing_loss": 0.078,
                    "slew_violation": 0.55,
                    "cap_violation": 0.11,
                    "leakage": 0.14,
                },
                "endpoint_payload": {
                    "source": "AutoDMP_projected",
                    "available": True,
                    "wns": -0.08,
                    "tns": -0.70,
                },
                "top_endpoints": [
                    {
                        "pin_name": "u_proj/Y",
                        "autodmp_slack_ns": -0.08,
                    }
                ],
            }

            placer._write_timing_stage_artifact(
                params,
                [self._make_metric_record(1, -120.0, -1000.0, -10.0, 1.2, 0.4, 0.2)],
                [self._make_metric_record(2, -100.0, -900.0, -8.5, 0.8, 0.2, 0.15)],
            )

            with open(os.path.join(tmpdir, "FFT_timing_stage_summary.json"), encoding="utf-8") as f:
                summary = json.load(f)
            with open(os.path.join(tmpdir, "FFT_stage_compare_latest.json"), encoding="utf-8") as f:
                latest = json.load(f)

            self.assertEqual(summary["post_projection"]["wns"], -80.0)
            self.assertEqual(summary["post_projection"]["slew_violation"], 0.55)
            self.assertEqual(
                latest["top_critical_endpoints"]["post_projection"][0]["pin_name"],
                "u_proj/Y",
            )


    def test_write_timing_stage_artifact_includes_continuous_early_stop_summary(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.last_continuous_early_stop_state = {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.001,
            "best_loss": 9.8,
            "best_iteration": 12,
            "last_loss": 9.82,
            "last_iteration": 17,
            "no_improve_rounds": 5,
            "triggered": True,
            "trigger_iteration": 17,
            "trigger_reason": "continuous_no_improvement",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )

            placer._write_timing_stage_artifact(
                params,
                [self._make_metric_record(1, -120.0, -1000.0, -10.0, 1.2, 0.4, 0.2)],
                [self._make_metric_record(2, -100.0, -900.0, -8.5, 0.8, 0.2, 0.15)],
            )

            with open(os.path.join(tmpdir, "FFT_timing_stage_summary.json"), encoding="utf-8") as f:
                summary = json.load(f)

            self.assertEqual(summary["continuous_early_stop"]["best_iteration"], 12)
            self.assertTrue(summary["continuous_early_stop"]["triggered"])
            self.assertEqual(summary["continuous_early_stop"]["trigger_iteration"], 17)


    def test_stage_compare_latest_marks_changed_cells_by_endpoint_cone_membership(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )
            placer.placedb = SimpleNamespace(
                pin_names=[b"u3/A", b"u4/A", b"u4/Y"],
            )
            placer.data_collections = SimpleNamespace(
                pin2node_map=torch.tensor([3, 4, 4], dtype=torch.int64),
                flat_pin_to_graph_reverse=torch.tensor([0], dtype=torch.int64),
                flat_pin_to_graph_start_reverse=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            )
            placer._build_endpoint_stage_payload = lambda params_arg, stage_name: (
                {"source": "test", "available": True},
                [{"pin_name": "u4/Y"}] if stage_name == "post_projection" else [],
            )

            with open(os.path.join(tmpdir, "FFT_projection.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "inst_id",
                        "changed_cell",
                        "current_size",
                        "continuous_size",
                        "projected_size",
                        "term_timing_penalty",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "inst_id": 3,
                        "changed_cell": True,
                        "current_size": 1.0,
                        "continuous_size": 1.2,
                        "projected_size": 1.1,
                        "term_timing_penalty": 0.0,
                    }
                )
            with open(os.path.join(tmpdir, "FFT_projection_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"num_changed_cells": 1}, f)

            placer._write_stage_compare_latest_artifact(
                params,
                {
                    "pre_projection": {
                        "wns": -120.0,
                        "tns": -1000.0,
                        "slew_violation": 1.2,
                        "cap_violation": 0.4,
                        "leakage": 0.2,
                    },
                    "post_projection": {
                        "wns": -100.0,
                        "tns": -900.0,
                        "slew_violation": 0.8,
                        "cap_violation": 0.2,
                        "leakage": 0.18,
                    },
                    "post_legalization": {
                        "wns": -90.0,
                        "tns": -850.0,
                        "slew_violation": 0.7,
                        "cap_violation": 0.1,
                        "leakage": 0.16,
                    },
                },
            )

            with open(os.path.join(tmpdir, "FFT_stage_compare_latest.json"), encoding="utf-8") as f:
                latest = json.load(f)

            self.assertTrue(latest["changed_cell_summary"]["rows"][0]["critical_endpoint_cone_membership"])


    def test_refresh_stage_compare_latest_artifact_reloads_projection_summary_and_counterfactual_metadata(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )
            placer._build_endpoint_stage_payload = lambda params_arg, stage_name: (
                {"source": "test", "available": True, "wns": -0.1, "tns": -1.0},
                [{"pin_name": "u0/Y"}] if stage_name == "pre_projection" else [],
            )
            timing_stage_summary = {
                "pre_projection": {
                    "wns": -100.0,
                    "tns": -1000.0,
                    "slew_violation": 1.0,
                    "cap_violation": 0.5,
                    "leakage": 0.2,
                },
                "post_projection": {
                    "wns": -95.0,
                    "tns": -900.0,
                    "slew_violation": 0.8,
                    "cap_violation": 0.4,
                    "leakage": 0.18,
                },
                "post_legalization": {
                    "wns": -90.0,
                    "tns": -850.0,
                    "slew_violation": 0.7,
                    "cap_violation": 0.3,
                    "leakage": 0.16,
                },
            }
            projection_csv_path = os.path.join(tmpdir, "FFT_projection.csv")
            projection_summary_path = os.path.join(tmpdir, "FFT_projection_summary.json")

            with open(projection_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "inst_id",
                        "changed_cell",
                        "current_size",
                        "continuous_size",
                        "projected_size",
                        "term_timing_penalty",
                        "metadata_current_candidate_present",
                        "metadata_current_candidate_total_score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "inst_id": 11,
                        "changed_cell": True,
                        "current_size": 1.0,
                        "continuous_size": 1.4,
                        "projected_size": 1.1,
                        "term_timing_penalty": 0.6,
                        "metadata_current_candidate_present": True,
                        "metadata_current_candidate_total_score": 1.25,
                    }
                )
            with open(projection_summary_path, "w", encoding="utf-8") as f:
                json.dump({"num_changed_cells": 1}, f)

            placer._write_stage_compare_latest_artifact(params, timing_stage_summary)

            with open(projection_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "inst_id",
                        "changed_cell",
                        "current_size",
                        "continuous_size",
                        "projected_size",
                        "term_timing_penalty",
                        "metadata_current_candidate_present",
                        "metadata_current_candidate_total_score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "inst_id": 22,
                        "changed_cell": True,
                        "current_size": 1.0,
                        "continuous_size": 1.5,
                        "projected_size": 1.2,
                        "term_timing_penalty": 0.9,
                        "metadata_current_candidate_present": True,
                        "metadata_current_candidate_total_score": 2.5,
                    }
                )
                writer.writerow(
                    {
                        "inst_id": 23,
                        "changed_cell": True,
                        "current_size": 1.0,
                        "continuous_size": 1.1,
                        "projected_size": 1.0,
                        "term_timing_penalty": 0.0,
                        "metadata_current_candidate_present": False,
                        "metadata_current_candidate_total_score": 3.5,
                    }
                )
            with open(projection_summary_path, "w", encoding="utf-8") as f:
                json.dump({"num_changed_cells": 2}, f)

            self.assertTrue(
                hasattr(placer, "_refresh_stage_compare_latest_artifact"),
                "latest stage compare artifact should expose an explicit refresh helper",
            )
            refreshed = placer._refresh_stage_compare_latest_artifact(params, timing_stage_summary)

            self.assertEqual(refreshed["changed_cell_summary"]["num_changed_cells"], 2)
            self.assertEqual(refreshed["changed_cell_summary"]["rows"][0]["inst_id"], 22)
            self.assertEqual(
                refreshed["changed_cell_summary"]["rows"][0]["metadata_current_candidate_total_score"],
                2.5,
            )


    def test_stage_compare_ignores_endpoint_payloads_older_than_current_run(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )
            placer._stage_compare_run_started_at = 2000.0
            summary_path = os.path.join(
                tmpdir,
                "FFT_endpoint_timing_compare_summary.json",
            )
            csv_path = os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ieda_wns": -100.0,
                        "ieda_tns": -300.0,
                        "python_wns": -90.0,
                        "python_tns": -250.0,
                        "num_ieda_negative_slack": 3,
                    },
                    f,
                )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "stale/Y",
                        "cpp_slack": -100.0,
                        "py_slack": -90.0,
                        "slack_delta": 10.0,
                    }
                )
            os.utime(summary_path, (1000.0, 1000.0))
            os.utime(csv_path, (1000.0, 1000.0))

            primary, top_rows = placer._build_endpoint_stage_payload(
                params,
                "post_legalization",
            )

            self.assertFalse(primary["available"])
            self.assertEqual(primary["stale_reason"], "older_than_current_run")
            self.assertIsNone(primary["wns"])
            self.assertEqual(top_rows, [])


    def test_stage_compare_keeps_diagnostic_backend_payload_out_of_primary_metrics(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )
            with open(os.path.join(tmpdir, "FFT_endpoint_timing_compare_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backend_wns": -568.0,
                        "backend_tns": -30892.0,
                        "num_backend_negative_slack": 141,
                        "python_wns": -525.0,
                        "python_tns": -107548.0,
                        "backend_metadata": {
                            "backend": "ieda",
                            "sta_reference_role": "diagnostic_exporter",
                            "sta_is_golden": False,
                            "parasitics_initialization": "unverified",
                            "sta_state_status": "diagnostic_unverified",
                        },
                    },
                    f,
                )
            with open(os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "u0/D",
                        "cpp_slack": -568.0,
                        "py_slack": -525.0,
                        "slack_delta": -43.0,
                    }
                )

            latest = placer._write_stage_compare_latest_artifact(
                params,
                {
                    "pre_projection": {},
                    "post_projection": {},
                    "post_legalization": {
                        "wns": -525.0,
                        "tns": -107548.0,
                        "slew_violation": 0.0,
                        "cap_violation": 0.0,
                        "leakage": 0.0,
                    },
                },
            )

            stage = latest["post_legalization"]
            self.assertAlmostEqual(stage["WNS"], -0.525)
            self.assertAlmostEqual(stage["TNS"], -107.548)
            self.assertEqual(stage["primary"]["source"], "AutoDMP")
            self.assertEqual(stage["diagnostic"]["sta_reference_role"], "diagnostic_exporter")
            self.assertFalse(stage["diagnostic"]["sta_is_golden"])
            self.assertAlmostEqual(stage["diagnostic"]["wns"], -0.568)
            self.assertEqual(latest["top_critical_endpoints"]["post_legalization"], [])
            self.assertEqual(
                stage["diagnostic_top_endpoints"][0]["pin_name"],
                "u0/D",
            )
