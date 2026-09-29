import json
import tempfile
import unittest
from pathlib import Path

from dreamplace.ops.gpugr.evidence import terminal_status, write_evidence_record


class EvidenceWriterTest(unittest.TestCase):
    def test_pass_record_is_atomic_and_contains_verified_artifact(self):
        with tempfile.TemporaryDirectory(prefix="gpugr_evidence_") as directory:
            root = Path(directory)
            artifact = root / "result.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            output = write_evidence_record(
                root / "evidence",
                {
                    "claim_level": "L1",
                    "gate": "toy",
                    "run_id": "run-1",
                    "status": "pass",
                    "command": ["python", "smoke.py"],
                },
                artifact_paths={"result": artifact},
            )

            self.assertEqual(output, root / "evidence" / "toy" / "run-1.json")
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertTrue(payload["artifacts"]["result"]["hash_matches"] is None)
            self.assertTrue(payload["artifacts"]["result"]["exists"])
            self.assertEqual(list((output.parent).glob("*.tmp")), [])

    def test_pass_record_rejects_missing_or_mismatched_artifact(self):
        with tempfile.TemporaryDirectory(prefix="gpugr_evidence_") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid artifacts"):
                write_evidence_record(
                    root,
                    {"claim_level": "L2", "gate": "clean", "status": "pass"},
                    artifact_paths={"missing": root / "missing.json"},
                )

            artifact = root / "result.json"
            artifact.write_text("current\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                write_evidence_record(
                    root,
                    {
                        "claim_level": "L2",
                        "gate": "clean",
                        "status": "pass",
                    },
                    artifact_paths={"result": {"path": artifact, "sha256": "0" * 64}},
                )

    def test_non_pass_record_keeps_artifact_failure_and_process_status(self):
        with tempfile.TemporaryDirectory(prefix="gpugr_evidence_") as directory:
            output = write_evidence_record(
                directory,
                {
                    "claim_level": "L3",
                    "gate": "rocket",
                    "run_id": "run-2",
                    "status": "incomplete",
                    "exit_code": -11,
                },
                artifact_paths={"def": Path(directory) / "missing.def"},
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "incomplete")
            self.assertIn("artifact_validation_errors", payload)

    def test_terminal_status_never_promotes_signal_or_missing_marker(self):
        self.assertEqual(
            terminal_status(returncode=0, terminal_marker=True),
            "pass",
        )
        self.assertEqual(
            terminal_status(returncode=1, terminal_marker=True),
            "fail",
        )
        self.assertEqual(
            terminal_status(returncode=-11, terminal_marker=True),
            "incomplete",
        )
        self.assertEqual(
            terminal_status(returncode=0, terminal_marker=False),
            "incomplete",
        )
        self.assertEqual(
            terminal_status(returncode=None, timed_out=True),
            "incomplete",
        )


if __name__ == "__main__":
    unittest.main()
