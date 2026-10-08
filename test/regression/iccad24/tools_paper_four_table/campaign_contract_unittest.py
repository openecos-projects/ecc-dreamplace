#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "campaign_contract.py"
PROFILE_PATH = SCRIPT_DIR / "profiles" / "tools_release_v1.json"
SPEC = importlib.util.spec_from_file_location("tools_campaign_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


class CampaignContractTest(unittest.TestCase):
    def setUp(self):
        self.profile = contract.load_release_profile(PROFILE_PATH)

    def _execution(self, cases=None):
        return {
            "artifact": contract.EXECUTION_ARTIFACT,
            "artifact_version": 1,
            "release_profile_digest": contract.digest_payload(self.profile),
            "selected_cases": list(cases or contract.ICCAD24_CASES),
            "selected_tables": list(contract.PRIMARY_TABLE_ORDER),
            "autodmp_revision": "abc123",
            "autodmp_tracked_diff_sha256": "d" * 64,
            "source_file_sha256": {"run.py": "b" * 64},
            "native_library_sha256": {"timing.so": "c" * 64},
            "openroad_path": "/tools/openroad",
            "openroad_sha256": "a" * 64,
            "python_path": "/env/python",
            "torch_version": "2.0.0",
            "cuda_runtime_version": "12.0",
            "cuda_visible_devices": "2",
            "logical_gpu_index": 0,
            "gpu_model": "GPU",
            "gpu_uuid": "GPU-uuid",
            "gpu_preflight_compute_processes": [],
            "cpu_model": "CPU",
            "cpu_count": 64,
            "omp_num_threads": 16,
            "torch_num_threads": 16,
            "torch_num_interop_threads": 1,
            "openroad_num_threads": 16,
            "case_concurrency": 1,
            "measurement_policy": "cold_process_per_attempt",
        }

    def test_profile_and_matrix_are_exactly_120_rows(self):
        rows = contract.primary_matrix()
        self.assertEqual(contract.EXPECTED_PRIMARY_ROWS, 120)
        self.assertEqual(len(rows), 120)
        self.assertNotIn(
            "efficient_tdp_qualified",
            {row["method_id"] for row in rows},
        )

    def test_profile_has_no_gpu_id(self):
        self.assertNotIn("gpu_id", json.dumps(self.profile, sort_keys=True))

    def test_profile_rejects_parameter_and_case_overrides(self):
        broken = copy.deepcopy(self.profile)
        broken["placement"]["max_steps"] = 500
        with self.assertRaisesRegex(ValueError, "placement optimizer/max_steps"):
            contract.validate_release_profile(broken)
        broken = copy.deepcopy(self.profile)
        broken["gpu_id"] = 2
        with self.assertRaisesRegex(ValueError, "physical gpu_id"):
            contract.validate_release_profile(broken)
        broken = copy.deepcopy(self.profile)
        broken["per_case"] = {"hidden5": {"placement": {"max_steps": 500}}}
        with self.assertRaisesRegex(ValueError, "per-case"):
            contract.validate_release_profile(broken)
        broken = copy.deepcopy(self.profile)
        broken["track_a"]["detailed_placement_search_window"] = "default"
        with self.assertRaisesRegex(ValueError, "canonical Track A"):
            contract.validate_release_profile(broken)

    def test_pilot_and_full_share_release_but_not_campaign_identity(self):
        pilot_execution = self._execution(
            ["NV_NVDLA_partition_m", "ariane136", "hidden5"]
        )
        full_execution = self._execution()
        pilot = contract.campaign_identity(self.profile, pilot_execution)
        full = contract.campaign_identity(self.profile, full_execution)
        self.assertEqual(
            pilot["release_profile_digest"],
            full["release_profile_digest"],
        )
        self.assertNotEqual(
            pilot["execution_manifest_digest"],
            full["execution_manifest_digest"],
        )
        self.assertNotEqual(pilot["campaign_id"], full["campaign_id"])

    def test_execution_manifest_requires_serial_fixed_thread_policy(self):
        broken = self._execution()
        broken["case_concurrency"] = 2
        with self.assertRaisesRegex(ValueError, "serial case execution"):
            contract.validate_execution_manifest(broken, profile=self.profile)
        broken = self._execution()
        broken.pop("omp_num_threads")
        with self.assertRaisesRegex(ValueError, "missing execution manifest"):
            contract.validate_execution_manifest(broken, profile=self.profile)

    def test_selected_cases_are_canonical(self):
        selected = contract.selected_cases(["hidden5", "NV_NVDLA_partition_m"])
        self.assertEqual(selected, ("NV_NVDLA_partition_m", "hidden5"))
        with self.assertRaisesRegex(ValueError, "unknown case"):
            contract.selected_cases(["unknown"])

    def test_completed_matrix_rejects_not_run_and_missing_attempt(self):
        rows = contract.primary_matrix(["NV_NVDLA_partition_m"])
        with self.assertRaisesRegex(ValueError, "not terminal"):
            contract.validate_completed_matrix(
                rows,
                cases=["NV_NVDLA_partition_m"],
            )
        completed = []
        for index, row in enumerate(rows):
            completed.append(
                {
                    **row,
                    "status": "pass",
                    "attempt_id": f"attempt-{index}",
                }
            )
        contract.validate_completed_matrix(
            completed,
            cases=["NV_NVDLA_partition_m"],
        )
        completed[0].pop("attempt_id")
        with self.assertRaisesRegex(ValueError, "missing attempt_id"):
            contract.validate_completed_matrix(
                completed,
                cases=["NV_NVDLA_partition_m"],
            )

    def test_completed_matrix_honors_selected_tables(self):
        rows = [
            {
                **row,
                "status": "execution_failed",
                "attempt_id": f"attempt-{index}",
                "terminal_reason": "fixture failure",
            }
            for index, row in enumerate(
                contract.primary_matrix(["NV_NVDLA_partition_m"])
            )
            if row["table_id"] == "full_flow"
        ]
        contract.validate_completed_matrix(
            rows,
            cases=["NV_NVDLA_partition_m"],
            tables=["full_flow"],
        )
        with self.assertRaisesRegex(ValueError, "required denominator"):
            contract.validate_completed_matrix(
                rows[:-1],
                cases=["NV_NVDLA_partition_m"],
                tables=["full_flow"],
            )

    def test_normalized_row_validator_rejects_missing_track_a_metric(self):
        row = {
            "artifact": contract.NORMALIZED_ROW_ARTIFACT,
            "artifact_version": contract.NORMALIZED_ROW_VERSION,
            "campaign_id": "campaign",
            "table_id": "sizing",
            "case": "NV_NVDLA_partition_m",
            "method_id": "autodmp_diff_sizing",
            "attempt_id": "attempt-1",
            "input_domain": "D_post",
            "required_cuda": True,
            "status": "pass",
            "terminal_reason": None,
            "input_metrics": {"wns_ns": -1.0, "tns_ns": -100.0},
            "final_metrics": {
                name: 1 for name in contract.REQUIRED_TRACK_A_METRICS
            },
            "mutation": {},
            "runtime": {
                "initialization_runtime_sec": 0.0,
                "core_optimizer_runtime_sec": 1.0,
                "physical_transaction_runtime_sec": 2.0,
                "legalization_runtime_sec": 0.0,
                "track_a_runtime_sec": 3.0,
                "end_to_end_runtime_sec": 6.0,
                "peak_rss_kb": 1024,
                "gpu_peak_allocated_bytes": 1024,
                "gpu_peak_reserved_bytes": 2048,
                "runtime_comparable_group": "sizing:source_plus_track_a_v1",
            },
            "source_artifact": {"path": "/tmp/source.json"},
            "track_a_artifact": {
                "path": "/tmp/track_a.json",
                "evaluated_def_sha256": "a" * 64,
            },
        }
        contract.validate_normalized_row(row, campaign_id="campaign")
        row["final_metrics"].pop("tns_ns")
        with self.assertRaisesRegex(ValueError, "missing Track A metrics"):
            contract.validate_normalized_row(row, campaign_id="campaign")

    def test_digest_is_deterministic_and_rejects_nan(self):
        first = contract.digest_payload(self.profile)
        second = contract.digest_payload(copy.deepcopy(self.profile))
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            contract.digest_payload({"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
