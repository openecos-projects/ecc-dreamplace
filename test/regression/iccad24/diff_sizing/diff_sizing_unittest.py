#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("diff_sizing_tools_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DiffSizingToolsTest(unittest.TestCase):
    def test_autodmp_command_is_fixed_position_production_sizing(self):
        command = runner.build_autodmp_command(
            python_bin=Path("/python"),
            benchmark_root=Path("/bench"),
            case="case0",
            result_dir=Path("/out/result"),
            output_def=Path("/out/raw.def"),
            iterations=50,
            logical_gpu_id=0,
            plan_only=False,
        )
        joined = " ".join(map(str, command))
        self.assertIn("--flow-kind sizing", joined)
        self.assertIn("--iterations 50", joined)
        self.assertIn("--legalize 0", joined)
        self.assertIn("--enable-fillers 0", joined)
        self.assertIn("--discrete-gradient-topk-up-percent 30.0", joined)
        self.assertIn("--discrete-gradient-topk-down-percent 0.0", joined)

    def test_openroad_tcl_is_pure_sizeup_without_dpl(self):
        tcl = runner.build_openroad_sizeup_tcl(
            benchmark_root=Path("/bench"),
            case="case0",
            def_input=Path("/bench/design/case0/case0.def"),
            output_def=Path("/out/raw.def"),
        )
        self.assertIn('repair_timing -setup -sequence "sizeup"', tcl)
        self.assertIn("refresh_timing", tcl)
        self.assertNotIn("\nupdate_timing\n", tcl)
        self.assertIn("-skip_pin_swap", tcl)
        self.assertIn("-skip_gate_cloning", tcl)
        self.assertIn("-skip_size_down", tcl)
        self.assertNotIn("detailed_placement", tcl)
        self.assertNotIn("repair_design", tcl)

    def test_track_a_command_is_def_only_public_cli(self):
        command = runner.build_track_a_command(
            python_bin=Path("/python"),
            openroad_bin=Path("/openroad"),
            benchmark_root=Path("/bench"),
            case="case0",
            method_id="autodmp_diff_sizing",
            reference_def=Path("/reference.def"),
            committed_def=Path("/committed.def"),
            output_dir=Path("/out"),
            buffer_master="BUFx4",
            plan_only=True,
        )
        joined = " ".join(map(str, command))
        self.assertIn("def_only_track_a.py", joined)
        self.assertIn("--reference-def /reference.def", joined)
        self.assertIn("--def-input /committed.def", joined)
        self.assertNotIn("--verilog", joined)
        self.assertIn("--plan-only", command)

    def test_method_row_rejects_coordinate_or_topology_mutation(self):
        row = runner._method_row(
            method_id="autodmp_diff_sizing",
            case="case0",
            method_root=Path("/out"),
            source_command=[],
            track_a_command=[],
            source_execution={"status": "ok", "committed_def_exists": True},
            track_a_execution={"status": "ok"},
            track_a_summary={
                "status": "pass",
                "mutation": {
                    "coordinate_changed_count": 1,
                    "inserted_buffer_count": 0,
                    "inserted_other_count": 0,
                    "deleted_instance_count": 0,
                },
            },
            plan_only=False,
        )
        self.assertEqual(row["status"], "failed")
        self.assertIn("method_changed_coordinates_before_track_a", row["failures"])

    def test_known_positive_autodmp_sizing_rejects_zero_resize_writeback(self):
        row = runner._method_row(
            method_id="autodmp_diff_sizing",
            case="NV_NVDLA_partition_m",
            method_root=Path("/out"),
            source_command=[],
            track_a_command=[],
            source_execution={"status": "ok", "committed_def_exists": True},
            track_a_execution={"status": "ok"},
            track_a_summary={
                "status": "pass",
                "mutation": {
                    "coordinate_changed_count": 0,
                    "inserted_buffer_count": 0,
                    "inserted_other_count": 0,
                    "deleted_instance_count": 0,
                    "resize_count": 0,
                },
            },
            plan_only=False,
        )

        self.assertEqual(row["status"], "failed")
        self.assertIn(
            "known_positive_sizing_state_has_zero_resizes",
            row["failures"],
        )

    def test_plan_only_emits_both_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = runner.main(
                [
                    "--case",
                    "NV_NVDLA_partition_m",
                    "--output-root",
                    str(root),
                    "--run-id",
                    "plan",
                    "--plan-only",
                ]
            )
            summary = json.loads(
                (root / "plan" / "campaign_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual({row["status"] for row in summary["rows"]}, {"plan_only"})


if __name__ == "__main__":
    unittest.main()
