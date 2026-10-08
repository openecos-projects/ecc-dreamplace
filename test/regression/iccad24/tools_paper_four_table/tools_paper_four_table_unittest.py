#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapters = _load("artifact_adapters")
aggregate = _load("aggregate")
contract = _load("campaign_contract")


class ToolsPaperFourTableTest(unittest.TestCase):
    def _track_a(self, path: Path):
        path.write_text(
            json.dumps(
                {
                    "artifact": adapters.TRACK_A_ARTIFACT,
                    "artifact_version": 1,
                    "status": "pass",
                    "metrics": {
                        "raw_hpwl_dbu": 100,
                        "legalized_hpwl_dbu": 101,
                        "wns_ns": -0.1,
                        "tns_ns": -10.0,
                        "violating_endpoint_count": 2,
                        "slew_violation_count": 0,
                        "slew_violation_total": 0.0,
                        "cap_violation_count": 0,
                        "cap_violation_total": 0.0,
                        "total_cell_area_um2": 12.0,
                        "power_total_w": 0.1,
                        "placement_valid": 1,
                    },
                    "mutation": {
                        "resize_count": 3,
                        "inserted_buffer_count": 4,
                        "added_cell_area_um2": 1.0,
                    },
                    "artifacts": {
                        "evaluated_def": "/tmp/final.def",
                        "evaluated_def_sha256": "a" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )

    def _pin2pin_case_result(
        self,
        *,
        source_method: str,
        log_path: Path,
    ) -> dict:
        return {
            "artifact": adapters.PLACEMENT_RESULT_ARTIFACT,
            "artifact_version": adapters.PLACEMENT_RESULT_VERSION,
            "case": "NV_NVDLA_partition_m",
            "status": "pass",
            "failures": [],
            "r0": {
                "manifest": {
                    "coordinate_identity": {
                        "movable_nodes_xy_sha256": "b" * 64,
                    }
                }
            },
            "methods": {
                source_method: {
                    "status": "pass",
                    "execution": {
                        "status": "ok",
                        "runtime_sec": 10.0,
                        "log_path": str(log_path),
                    },
                    "milestones": {
                        "legacy_net_weight": {
                            "status": "updated",
                            "update_count": 2,
                            "pin2pin_backend": "cuda",
                            "pin2pin_path_pair_backend": (
                                "cpp_openmp_transition_aware"
                            ),
                            "pin2pin_native_summary": {
                                "state_domain": "setup_plus_recovery",
                            },
                        }
                    },
                    "raw_def": "/tmp/raw.def",
                    "raw_def_sha256": "c" * 64,
                }
            },
        }

    @staticmethod
    def _coordinated_physical_commit() -> dict:
        return {
            "status": "ok",
            "full_action_order": [
                "coordinate_sync",
                "apply_sizing",
                "insert_buffers",
                "legalize",
                "check_placement",
                "estimate_parasitics",
                "timing_refresh",
            ],
            "sizing_master_verification": {
                "status": "ok",
                "expected_count": 4,
                "matched_count": 4,
                "missing_count": 0,
                "mismatch_count": 0,
            },
            "accepted_buffer_count": 3,
            "buffer_count": 3,
            "legalization_status": "ok",
            "parasitic_refresh_status": "ok",
            "def_only_opensta_verification_status": "ok",
        }

    def test_track_a_adapter_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            self._track_a(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["artifact_version"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported artifact_version"):
                adapters.read_json_artifact(
                    path,
                    artifact=adapters.TRACK_A_ARTIFACT,
                    versions={1},
                )

    def test_track_a_adapter_normalizes_public_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            self._track_a(path)
            row = adapters.normalize_track_a_row(
                track_a_path=path,
                campaign_id="campaign",
                table_id="buffering",
                case="NV_NVDLA_partition_m",
                method_id="autodmp_equal_spaced_route_b",
                attempt_id="attempt",
                input_domain="D_post",
                required_cuda=True,
                input_metrics={"wns_ns": -0.2, "tns_ns": -20.0},
                source_artifact={"path": "/tmp/source.json"},
                runtime={"end_to_end_runtime_sec": 2.0},
            )
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["mutation"]["inserted_buffer_count"], 4)
        self.assertEqual(row["final_metrics"]["tns_ns"], -10.0)

    def test_tns_tolerance_and_normalizations(self):
        self.assertEqual(aggregate.classify_tns_gap(0.0005), "tie")
        self.assertEqual(aggregate.classify_tns_gap(0.002), "win")
        self.assertEqual(aggregate.classify_tns_gap(-0.002), "loss")
        self.assertEqual(aggregate.relative_pareto_change(100.05, 100.0), 0.0)
        rows = [
            {
                "table_id": "sizing",
                "case": "case0",
                "method_id": "or_pure_sizeup",
                "status": "pass",
                "attempt_id": "or",
                "input_metrics": {"wns_ns": -1.0, "tns_ns": -100.0},
                "final_metrics": {"wns_ns": -0.5, "tns_ns": -50.0},
            },
            {
                "table_id": "sizing",
                "case": "case0",
                "method_id": "autodmp_diff_sizing",
                "status": "pass",
                "attempt_id": "ours",
                "input_metrics": {"wns_ns": -1.0, "tns_ns": -100.0},
                "final_metrics": {"wns_ns": -0.4, "tns_ns": -40.0},
            },
        ]
        enriched = {
            row["method_id"]: row for row in aggregate.enrich_rows(rows)
        }
        ours = enriched["autodmp_diff_sizing"]
        self.assertEqual(ours["state_delta_tns_ns"], 60.0)
        self.assertEqual(ours["normalized_state_gain"], 0.6)
        self.assertEqual(ours["matched_baseline_gap_tns_ns"], 10.0)
        self.assertEqual(ours["normalized_baseline_advantage"], 0.1)
        self.assertEqual(ours["tns_result_vs_baseline"], "win")

    def test_table_generation_is_deterministic(self):
        rows = []
        for method, final_tns in (
            ("or_pure_sizeup", -50.0),
            ("autodmp_diff_sizing", -40.0),
        ):
            rows.append(
                {
                    "table_id": "sizing",
                    "case": "case0",
                    "method_id": method,
                    "status": "pass",
                    "attempt_id": method,
                    "terminal_reason": None,
                    "input_metrics": {"wns_ns": -1.0, "tns_ns": -100.0},
                    "final_metrics": {"wns_ns": -0.5, "tns_ns": final_tns},
                    "mutation": {},
                    "runtime": {},
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            aggregate.write_table_files(
                first,
                rows,
                table_order=["sizing"],
                case_order=["case0"],
                method_order={"sizing": contract.PRIMARY_METHODS["sizing"]},
            )
            aggregate.write_table_files(
                second,
                reversed(rows),
                table_order=["sizing"],
                case_order=["case0"],
                method_order={"sizing": contract.PRIMARY_METHODS["sizing"]},
            )
            for suffix in ("json", "csv", "md"):
                first_data = (first / f"table3_sizing.{suffix}").read_bytes()
                second_data = (second / f"table3_sizing.{suffix}").read_bytes()
                self.assertEqual(first_data, second_data)
            self.assertEqual(
                (first / "aggregate_summary.json").read_bytes(),
                (second / "aggregate_summary.json").read_bytes(),
            )

    def test_aggregate_uses_matched_runtime_group_and_never_sums_tns(self):
        rows = []
        for method, tns, runtime, group in (
            ("or_pure_sizeup", -50.0, 20.0, "sizing:source_plus_track_a_v1"),
            (
                "autodmp_diff_sizing",
                -40.0,
                10.0,
                "sizing:source_plus_track_a_v1",
            ),
        ):
            rows.append(
                {
                    "table_id": "sizing",
                    "case": "case0",
                    "method_id": method,
                    "status": "pass",
                    "attempt_id": method,
                    "input_metrics": {"tns_ns": -100.0},
                    "final_metrics": {
                        "tns_ns": tns,
                        "legalized_hpwl_dbu": 100,
                        "total_cell_area_um2": 10.0,
                        "power_total_w": 0.1,
                    },
                    "mutation": {},
                    "runtime": {
                        "end_to_end_runtime_sec": runtime,
                        "runtime_comparable_group": group,
                    },
                }
            )
        summary = aggregate.summarize_tables(rows)
        ours = summary["tables"]["sizing"]["methods"]["autodmp_diff_sizing"]
        self.assertEqual(ours["tns_wins"], 1)
        self.assertEqual(
            ours["median_matched_end_to_end_speedup_baseline_over_method"],
            2.0,
        )
        self.assertNotIn("tns_sum", json.dumps(summary, sort_keys=True))
        rows[1]["runtime"]["runtime_comparable_group"] = "different-boundary"
        unmatched = aggregate.summarize_tables(rows)
        self.assertIsNone(
            unmatched["tables"]["sizing"]["methods"][
                "autodmp_diff_sizing"
            ]["median_matched_end_to_end_speedup_baseline_over_method"]
        )

    def test_failure_row_cannot_use_nonterminal_status(self):
        with self.assertRaisesRegex(ValueError, "invalid failure status"):
            adapters.terminal_failure_row(
                campaign_id="campaign",
                table_id="placement",
                case="case0",
                method_id="method0",
                attempt_id="attempt0",
                input_domain="R0",
                required_cuda=True,
                status="not_run",
                terminal_reason="missing",
            )

    def test_sizing_adapter_normalizes_public_result_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a_root = root / "track_a"
            track_a_root.mkdir()
            result_path = root / "result.json"
            source_log = root / "source.log"
            source_log.write_text(
                "[INFO] optimizer adam takes 39.218 seconds\n"
                "[INFO] TOOLS_METRIC|gpu_peak_allocated_bytes|1024\n"
                "[INFO] TOOLS_METRIC|gpu_peak_reserved_bytes|2048\n",
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "artifact": "diff_sizing_tools_method_result",
                        "artifact_version": 1,
                        "case": "NV_NVDLA_partition_m",
                        "method_id": "autodmp_diff_sizing",
                        "status": "pass",
                        "failures": [],
                        "source_execution": {
                            "status": "ok",
                            "runtime_sec": 74.0,
                            "log_path": str(source_log),
                        },
                        "track_a_execution": {
                            "status": "ok",
                            "runtime_sec": 12.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            track_a = {
                "artifact": "canonical_def_only_track_a",
                "artifact_version": 1,
                "status": "pass",
                "metrics": {
                    "raw_hpwl_dbu": 10,
                    "legalized_hpwl_dbu": 11,
                    "wns_ns": -0.2,
                    "tns_ns": -10.0,
                    "violating_endpoint_count": 1,
                    "slew_violation_count": 0,
                    "slew_violation_total": 0.0,
                    "cap_violation_count": 0,
                    "cap_violation_total": 0.0,
                    "total_cell_area_um2": 100.0,
                    "power_total_w": 0.01,
                },
                "mutation": {"resize_count": 5},
                "artifacts": {},
            }
            (track_a_root / "track_a_summary.json").write_text(
                json.dumps(track_a),
                encoding="utf-8",
            )

            row = adapters.normalize_sizing_result(
                result_path=result_path,
                campaign_id="campaign",
                attempt_id="attempt",
                input_metrics={"wns_ns": -0.5, "tns_ns": -100.0},
                required_cuda=True,
            )

        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["final_metrics"]["tns_ns"], -10.0)
        self.assertEqual(row["mutation"]["resize_count"], 5)
        self.assertEqual(row["runtime"]["core_optimizer_runtime_sec"], 39.218)
        self.assertEqual(row["runtime"]["physical_transaction_runtime_sec"], 34.782)
        self.assertEqual(row["runtime"]["end_to_end_runtime_sec"], 86.0)

    def test_openroad_native_full_flow_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a_path = root / "track_a.json"
            self._track_a(track_a_path)
            result_path = root / "source.json"
            payload = self._pin2pin_case_result(
                source_method="OR-native-full-flow",
                log_path=root / "unused.log",
            )
            method = payload["methods"]["OR-native-full-flow"]
            method["metrics"] = {"core_runtime_sec": 8.0}
            method["native_timing_driven_actions"] = {"status": "ok"}
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            row = adapters.normalize_full_flow_result(
                result_path=result_path,
                track_a_path=track_a_path,
                track_a_launcher_runtime_sec=2.0,
                campaign_id="campaign",
                attempt_id="attempt",
                method_id="or_native_full_flow",
                required_cuda=False,
            )
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["runtime"]["core_optimizer_runtime_sec"], 8.0)
        self.assertEqual(
            row["source_artifact"]["r0_movable_xy_sha256"],
            "b" * 64,
        )

    def test_staged_full_flow_adapter_and_stage_order_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a_path = root / "track_a.json"
            self._track_a(track_a_path)
            placement_log = root / "placement.log"
            sizing_log = root / "sizing.log"
            placement_log.write_text(
                "optimizer nesterov takes 1.0 seconds\n"
                "TOOLS_METRIC|gpu_peak_allocated_bytes|1024\n"
                "TOOLS_METRIC|gpu_peak_reserved_bytes|2048\n",
                encoding="utf-8",
            )
            sizing_log.write_text(
                "optimizer adam takes 1.0 seconds\n", encoding="utf-8"
            )
            placement_path = root / "placement.json"
            placement_path.write_text(
                json.dumps(
                    self._pin2pin_case_result(
                        source_method="TDP-pin2pin",
                        log_path=placement_log,
                    )
                ),
                encoding="utf-8",
            )
            buffering_path = root / "buffering.json"
            buffering_log = root / "buffering.log"
            buffering_log.write_text("buffering complete\n", encoding="utf-8")
            buffering_path.write_text(
                json.dumps(
                    {
                        "artifact": adapters.EQUAL_BUFFERING_RESULT_ARTIFACT,
                        "artifact_version": adapters.EQUAL_BUFFERING_RESULT_VERSION,
                        "result": {
                            "status": "pass",
                            "run_log": str(buffering_log),
                        },
                        "summary": {
                            "inner_loop": {
                                "metrics_trace": [{"step_wall_ms": 100.0}],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            stage_order = [
                "pin2pin_placement",
                "placement_checkpoint_legalization",
                "fixed_position_sizing",
                "sizing_checkpoint_legalization",
                "equal_spaced_route_b",
            ]
            stages = {
                name: {"status": "ok", "runtime_sec": 2.0}
                for name in stage_order
            }
            stages["fixed_position_sizing"]["log_path"] = str(sizing_log)
            result_path = root / "staged.json"
            result = {
                "artifact": adapters.FULL_FLOW_RESULT_ARTIFACT,
                "artifact_version": adapters.FULL_FLOW_RESULT_VERSION,
                "case": "NV_NVDLA_partition_m",
                "method_id": "autodmp_staged_psb",
                "status": "pass",
                "failures": [],
                "stage_order": stage_order,
                "stages": stages,
                "raw_def": "/tmp/raw.def",
                "raw_def_sha256": "c" * 64,
                "source_artifacts": {
                    "placement_case_summary": str(placement_path),
                    "buffering_result": str(buffering_path),
                },
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            row = adapters.normalize_full_flow_result(
                result_path=result_path,
                track_a_path=track_a_path,
                track_a_launcher_runtime_sec=2.0,
                campaign_id="campaign",
                attempt_id="attempt",
                method_id="autodmp_staged_psb",
                required_cuda=True,
            )
            self.assertEqual(row["status"], "pass")
            result["stage_order"] = list(reversed(stage_order))
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage order mismatch"):
                adapters.normalize_full_flow_result(
                    result_path=result_path,
                    track_a_path=track_a_path,
                    track_a_launcher_runtime_sec=2.0,
                    campaign_id="campaign",
                    attempt_id="attempt",
                    method_id="autodmp_staged_psb",
                    required_cuda=True,
                )

    def test_coordinated_full_flow_rejects_mechanism_and_readback_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a_path = root / "track_a.json"
            self._track_a(track_a_path)
            source_log = root / "source.log"
            source_log.write_text(
                "optimizer nesterov takes 8.0 seconds\n"
                "TOOLS_METRIC|gpu_peak_allocated_bytes|1024\n"
                "TOOLS_METRIC|gpu_peak_reserved_bytes|2048\n",
                encoding="utf-8",
            )
            result = self._pin2pin_case_result(
                source_method="TDP-pin2pin-SB",
                log_path=source_log,
            )
            method = result["methods"]["TDP-pin2pin-SB"]
            method["mechanism_audit"] = {"status": "pass", "failures": []}
            method["physical_commit"] = self._coordinated_physical_commit()
            result_path = root / "coordinated.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            row = adapters.normalize_full_flow_result(
                result_path=result_path,
                track_a_path=track_a_path,
                track_a_launcher_runtime_sec=2.0,
                campaign_id="campaign",
                attempt_id="attempt",
                method_id="autodmp_coordinated_psb",
                required_cuda=True,
            )
            self.assertEqual(row["status"], "pass")

            method["mechanism_audit"] = {
                "status": "failed",
                "failures": ["fixture"],
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mechanism audit failed"):
                adapters.normalize_full_flow_result(
                    result_path=result_path,
                    track_a_path=track_a_path,
                    track_a_launcher_runtime_sec=2.0,
                    campaign_id="campaign",
                    attempt_id="attempt",
                    method_id="autodmp_coordinated_psb",
                    required_cuda=True,
                )

            method["mechanism_audit"] = {"status": "pass", "failures": []}
            method["physical_commit"]["sizing_master_verification"][
                "mismatch_count"
            ] = 1
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sizing readback failed"):
                adapters.normalize_full_flow_result(
                    result_path=result_path,
                    track_a_path=track_a_path,
                    track_a_launcher_runtime_sec=2.0,
                    campaign_id="campaign",
                    attempt_id="attempt",
                    method_id="autodmp_coordinated_psb",
                    required_cuda=True,
                )

            method["physical_commit"] = self._coordinated_physical_commit()
            method["physical_commit"]["accepted_buffer_count"] = 2
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "buffer readback failed"):
                adapters.normalize_full_flow_result(
                    result_path=result_path,
                    track_a_path=track_a_path,
                    track_a_launcher_runtime_sec=2.0,
                    campaign_id="campaign",
                    attempt_id="attempt",
                    method_id="autodmp_coordinated_psb",
                    required_cuda=True,
                )


if __name__ == "__main__":
    unittest.main()
