#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import campaign_contract as contract
import qualification


SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = SCRIPT_DIR / "profiles" / "tools_release_v1.json"
RUN_SPEC = importlib.util.spec_from_file_location(
    "tools_paper_four_table_orchestrator",
    SCRIPT_DIR / "orchestrator.py",
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
campaign_run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(campaign_run)


class QualificationTest(unittest.TestCase):
    def setUp(self):
        self.profile = contract.load_release_profile(PROFILE_PATH)

    def _execution(self) -> dict:
        return {
            "autodmp_revision": "revision",
            "autodmp_tracked_diff_sha256": "a" * 64,
            "source_file_sha256": {"run.py": "b" * 64},
            "native_library_sha256": {"timing.so": "c" * 64},
            "openroad_sha256": "d" * 64,
            "openroad_version": "OpenROAD v1",
            "python_path": "/env/python",
            "python_version": "Python 3.11",
            "torch_version": "2.0.0",
            "cuda_runtime_version": "12.0",
            "gpu_model": "GPU",
            "gpu_uuid": "GPU-uuid",
            "gpu_preflight_compute_processes": [],
            "cpu_model": "CPU",
            "omp_num_threads": 16,
            "torch_num_threads": 16,
            "torch_num_interop_threads": 1,
            "openroad_num_threads": 16,
            "measurement_policy": "cold_process_per_attempt",
            "benchmark_root": "/benchmarks/iccad24",
        }

    def test_source_commands_pin_manifest_openroad_thread_count(self):
        args = SimpleNamespace(
            python_bin=Path("/python"),
            openroad_bin=Path("/openroad"),
            benchmark_root=Path("/benchmark"),
            profile=PROFILE_PATH,
            gpu_id=5,
            cpu_threads=16,
        )
        placement, *_ = campaign_run._placement_source_command(
            args=args,
            profile=self.profile,
            case="NV_NVDLA_partition_m",
            method_id="or_gpl_plain",
            attempt_root=Path("/attempt/placement"),
        )
        sizing, *_ = campaign_run._sizing_source_command(
            args=args,
            case="NV_NVDLA_partition_m",
            method_id="or_pure_sizeup",
            attempt_root=Path("/attempt/sizing"),
        )
        buffering_commands = [
            campaign_run._buffering_source_command(
                args=args,
                case="NV_NVDLA_partition_m",
                method_id=method_id,
                attempt_root=Path(f"/attempt/{method_id}"),
                campaign_manifest_path=Path("/campaign/manifest.json"),
            )[0]
            for method_id in (
                "or_repair_design_buffer_only",
                "autodmp_candidate_route_b",
                "autodmp_equal_spaced_route_b",
            )
        ]
        full_flow, *_ = campaign_run._full_flow_source_command(
            args=args,
            profile=self.profile,
            case="NV_NVDLA_partition_m",
            method_id="or_native_full_flow",
            attempt_root=Path("/attempt/full_flow"),
        )
        staged, *_ = campaign_run._full_flow_source_command(
            args=args,
            profile=self.profile,
            case="NV_NVDLA_partition_m",
            method_id="autodmp_staged_psb",
            attempt_root=Path("/attempt/staged"),
        )
        track_a = campaign_run._track_a_command(
            args=args,
            profile=self.profile,
            case="NV_NVDLA_partition_m",
            method_id="method",
            reference_def=Path("/reference.def"),
            def_input=Path("/input.def"),
            output_dir=Path("/track_a"),
        )
        for command in (
            placement,
            sizing,
            *buffering_commands,
            full_flow,
            staged,
            track_a,
        ):
            self.assertEqual(
                command[command.index("--openroad-num-threads") + 1], "16"
            )

    def _reuse_fixture(self, root: Path) -> tuple[Path, dict]:
        source_path = root / "source.json"
        source_path.write_text("{}\n", encoding="utf-8")
        raw_def = root / "raw.def"
        raw_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
        r0_def = root / "r0.def"
        r0_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
        evaluated_def = root / "evaluated.def"
        evaluated_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
        track_a_path = root / "track_a.json"
        track_a_path.write_text("{}\n", encoding="utf-8")
        prior_campaign = "prior-campaign"
        row = {
            "artifact": contract.NORMALIZED_ROW_ARTIFACT,
            "artifact_version": contract.NORMALIZED_ROW_VERSION,
            "campaign_id": prior_campaign,
            "table_id": "full_flow",
            "case": "NV_NVDLA_partition_m",
            "method_id": "autodmp_coordinated_psb",
            "attempt_id": "prior-attempt",
            "input_domain": "R0",
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
                "runtime_comparable_group": "full_flow:source_plus_track_a_v1",
            },
            "source_artifact": {
                "path": str(source_path),
                "raw_def": str(raw_def),
                "raw_def_sha256": qualification.sha256(raw_def),
                "r0_movable_xy_sha256": "e" * 64,
                "r0_def": str(r0_def),
                "r0_def_sha256": qualification.sha256(r0_def),
            },
            "track_a_artifact": {
                "path": str(track_a_path),
                "evaluated_def": str(evaluated_def),
                "evaluated_def_sha256": qualification.sha256(evaluated_def),
            },
        }
        row = qualification.seal_passing_row(row)
        normalized_path = root / "normalized_row.json"
        normalized_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        campaign_manifest = root / "campaign_manifest.json"
        campaign_manifest.write_text(
            json.dumps(
                {
                    "artifact": qualification.CAMPAIGN_MANIFEST_ARTIFACT,
                    "artifact_version": 1,
                    "campaign_id": prior_campaign,
                    "release_profile_digest": contract.digest_payload(self.profile),
                    "execution_manifest": self._execution(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        attempt_manifest = root / "attempt_manifest.json"
        attempt_manifest.write_text(
            json.dumps(
                {
                    "artifact": qualification.ATTEMPT_MANIFEST_ARTIFACT,
                    "artifact_version": 1,
                    "campaign_id": prior_campaign,
                    "attempt_id": "prior-attempt",
                    "status": "pass",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        qualification_path = root / "qualification.json"
        qualification_path.write_text(
            json.dumps(
                {
                    "artifact": qualification.ATTEMPT_QUALIFICATION_ARTIFACT,
                    "artifact_version": 1,
                    "attempt_id": "prior-attempt",
                    "status": "pass",
                    "terminal_status": "pass",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        paths = {
            "normalized_row": normalized_path,
            "campaign_manifest": campaign_manifest,
            "attempt_manifest": attempt_manifest,
            "qualification": qualification_path,
        }
        entry = {
            "table_id": "full_flow",
            "case": "NV_NVDLA_partition_m",
            "method_id": "autodmp_coordinated_psb",
        }
        for name, path in paths.items():
            entry[name] = str(path)
            entry[f"{name}_sha256"] = qualification.sha256(path)
        index_path = root / "reuse_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "artifact": qualification.REUSE_INDEX_ARTIFACT,
                    "artifact_version": qualification.REUSE_INDEX_VERSION,
                    "entries": [entry],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return index_path, entry

    def test_explicit_reuse_index_qualifies_exact_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path, entry = self._reuse_fixture(Path(tmp))
            loaded = qualification.load_reuse_index(index_path)
            row, report = qualification.qualify_reuse_entry(
                loaded[("full_flow", "NV_NVDLA_partition_m", "autodmp_coordinated_psb")],
                current_profile=self.profile,
                current_execution=self._execution(),
                expected_key=(
                    "full_flow",
                    "NV_NVDLA_partition_m",
                    "autodmp_coordinated_psb",
                ),
            )
            self.assertEqual(row["status"], "pass")
            self.assertEqual(report["status"], "pass")
            Path(entry["normalized_row"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
                qualification.qualify_reuse_entry(
                    entry,
                    current_profile=self.profile,
                    current_execution=self._execution(),
                    expected_key=(
                        "full_flow",
                        "NV_NVDLA_partition_m",
                        "autodmp_coordinated_psb",
                    ),
                )

    def test_reuse_rejects_source_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, entry = self._reuse_fixture(Path(tmp))
            changed = copy.deepcopy(self._execution())
            changed["openroad_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                ValueError, "execution_identity_mismatch:openroad_sha256"
            ):
                qualification.qualify_reuse_entry(
                    entry,
                    current_profile=self.profile,
                    current_execution=changed,
                    expected_key=(
                        "full_flow",
                        "NV_NVDLA_partition_m",
                        "autodmp_coordinated_psb",
                    ),
                )

    def test_r0_identity_mismatch_becomes_terminal_provenance_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            r0_def = run_root / "r0.def"
            r0_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            base = {
                "status": "pass",
                "source_artifact": {
                    "r0_movable_xy_sha256": "a" * 64,
                    "r0_def": str(r0_def),
                    "r0_def_sha256": qualification.sha256(r0_def),
                },
            }
            first = campaign_run._record_r0_identity(
                base,
                run_root=run_root,
                campaign_id="campaign",
                table_id="placement",
                case="NV_NVDLA_partition_m",
                method_id="autodmp_p_ctrl",
                attempt_id="attempt-1",
                required_cuda=True,
            )
            self.assertEqual(first["status"], "pass")
            changed = copy.deepcopy(base)
            changed["source_artifact"]["r0_movable_xy_sha256"] = "b" * 64
            second = campaign_run._record_r0_identity(
                changed,
                run_root=run_root,
                campaign_id="campaign",
                table_id="full_flow",
                case="NV_NVDLA_partition_m",
                method_id="autodmp_coordinated_psb",
                attempt_id="attempt-2",
                required_cuda=True,
            )
            self.assertEqual(second["status"], "provenance_mismatch")

    def test_attempt_command_collects_peak_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = campaign_run._run_attempt_command(
                ["/bin/sh", "-c", "printf test"],
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                environment={},
            )
            self.assertEqual(result["returncode"], 0)
            self.assertGreaterEqual(result["peak_rss_kb"], 0)
            self.assertTrue(Path(result["peak_rss_path"]).is_file())

    def test_attempt_command_preserves_nonzero_exit_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = campaign_run._run_attempt_command(
                ["/bin/sh", "-c", "exit 7"],
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                environment={},
            )
            self.assertEqual(result["returncode"], 7)
            self.assertGreaterEqual(result["peak_rss_kb"], 0)

    def test_finalization_emits_audits_and_rejects_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            campaign_id = "campaign"
            rows = []
            for index, method_id in enumerate(contract.PRIMARY_METHODS["full_flow"]):
                row = campaign_run.adapters.terminal_failure_row(
                    campaign_id=campaign_id,
                    table_id="full_flow",
                    case="NV_NVDLA_partition_m",
                    method_id=method_id,
                    attempt_id=f"attempt-{index}",
                    input_domain="R0",
                    required_cuda=method_id != "or_native_full_flow",
                    status="execution_failed",
                    terminal_reason="fixture failure",
                )
                rows.append(row)
                attempt_root = (
                    run_root
                    / "attempts"
                    / "full_flow"
                    / "NV_NVDLA_partition_m"
                    / method_id
                    / row["attempt_id"]
                )
                campaign_run._write_json(attempt_root / "normalized_row.json", row)
            campaign_run._write_json(
                run_root / "campaign_manifest.json",
                {"artifact": "fixture", "campaign_id": campaign_id},
            )
            campaign_run._write_json(run_root / "normalized_rows.json", rows)
            artifacts = campaign_run._write_final_tables(
                run_root=run_root,
                rows=rows,
                profile=self.profile,
                campaign_id=campaign_id,
                selected_cases=("NV_NVDLA_partition_m",),
                selected_tables=("full_flow",),
            )
            self.assertIn("campaign_reports", artifacts)
            self.assertTrue((run_root / "artifact_index.json").is_file())
            completion = json.loads(
                (run_root / "completion_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completion["status"], "complete_with_terminal_failures")
            with self.assertRaisesRegex(ValueError, "required denominator"):
                campaign_run._write_final_tables(
                    run_root=run_root,
                    rows=rows[:-1],
                    profile=self.profile,
                    campaign_id=campaign_id,
                    selected_cases=("NV_NVDLA_partition_m",),
                    selected_tables=("full_flow",),
                )


if __name__ == "__main__":
    unittest.main()
