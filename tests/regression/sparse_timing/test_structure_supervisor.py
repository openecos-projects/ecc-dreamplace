"""Exercise real subprocess completion and resource-budget termination."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from structure_supervisor import run


class StructureSupervisorTest(unittest.TestCase):
    def execute(self, source, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            limits = {
                "max_wall_s": 10,
                "max_rss_bytes": 1024**3,
                "max_states": 100,
                "max_edges": 100,
                **overrides,
            }
            path = root / "limits.json"
            path.write_text(json.dumps(limits))
            report = run(
                [sys.executable, "-c", source],
                limits_path=path,
                log_path=root / "child.log",
                report_path=root / "supervisor.json",
            )
            self.assertEqual(report, json.loads((root / "supervisor.json").read_text()))
            self.assertEqual(limits, json.loads(path.read_text()))
            return report, (root / "child.log").read_text()

    def test_completed_command_preserves_output(self):
        report, output = self.execute("print('native structure ready')")
        self.assertEqual(
            (report["status"], report["returncode"], report["limit_exceeded"]),
            ("completed", 0, None),
        )
        self.assertEqual(output, "native structure ready\n")

    def test_failed_child_is_not_completed(self):
        report, _ = self.execute("raise SystemExit(7)")
        self.assertEqual((report["status"], report["returncode"]), ("failed", 7))

    def test_wall_budget_terminates_child(self):
        report, _ = self.execute("import time; time.sleep(30)", max_wall_s=0.1)
        self.assertEqual(
            (report["status"], report["limit_exceeded"]), ("budget_exceeded", "max_wall_s")
        )
        self.assertLess(report["returncode"], 0)

    def test_rss_budget_terminates_child(self):
        report, _ = self.execute(
            "import time; data=bytearray(32*1024**2); time.sleep(30)", max_rss_bytes=16 * 1024**2
        )
        self.assertEqual(
            (report["status"], report["limit_exceeded"]), ("budget_exceeded", "max_rss_bytes")
        )
        self.assertLess(report["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
