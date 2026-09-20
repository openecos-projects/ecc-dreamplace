import json
import tempfile
import unittest
from pathlib import Path

from dreamplace.ops.gpugr.cugr_evidence import write_cugr_evidence


class CugrEvidenceTest(unittest.TestCase):
    def test_success_record_includes_input_hashes_and_backend_metrics(self):
        with tempfile.TemporaryDirectory(prefix="cugr_evidence_") as directory:
            root = Path(directory)
            def_path = root / "input.def"
            lef_path = root / "tech.lef"
            result_path = root / "metrics.json"
            def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            lef_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            result_path.write_text("{}\n", encoding="utf-8")

            output = write_cugr_evidence(
                result_dir=root,
                run_id="run-1",
                input_def=str(def_path),
                lefs=[str(lef_path)],
                artifact_paths={"metrics": result_path},
                metrics={
                    "cugr_total_passes": 1,
                    "cugr_source_dirty": 1,
                },
                source_commit="test-sha",
                source_dirty=True,
                returncode=0,
                terminal_marker=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["metrics"]["cugr_total_passes"], 1)
            self.assertTrue(payload["artifacts"]["input_def"]["sha256"])
            self.assertTrue(payload["artifacts"]["input_lef_0"]["sha256"])


if __name__ == "__main__":
    unittest.main()
