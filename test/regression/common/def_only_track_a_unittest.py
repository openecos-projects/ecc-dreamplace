#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("def_only_track_a.py")
SPEC = importlib.util.spec_from_file_location("def_only_track_a", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
track_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(track_a)


class DefOnlyTrackATest(unittest.TestCase):
    LIBERTY_BASENAMES = track_a.EXPECTED_ICCAD24_LIBERTY_BASENAMES

    def test_blocked_layer_waiver_keeps_other_failures_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            self.assertFalse(track_a.blocked_layer_only_waiver(path)["applied"])
            for categories, expected in [
                ({"Blocked_layers_failures": {"violations": [{"sources": []}]}}, True),
                ({"Blocked_layers_failures": {"violations": []}}, False),
                ({"Blocked_layers_failures": {"violations": [{}]},
                  "Overlap_failures": {"violations": [{}]}}, False),
                ({"Row_failures": {"violations": [{}]}}, False),
                ({"Unexpected_failure": {}}, False),
            ]:
                path.write_text(json.dumps({"DPL": {"category": categories}}))
                self.assertEqual(track_a.blocked_layer_only_waiver(path)["applied"], expected)
            path.write_text("invalid json")
            self.assertFalse(track_a.blocked_layer_only_waiver(path)["applied"])

    def test_blocked_layer_waiver_requires_explicit_cli_flag(self):
        parser = track_a.build_arg_parser()
        self.assertFalse(parser.get_default("ignore_blocked_layer_violations"))

    def _libs(self, root: Path) -> list[Path]:
        return [root / name for name in self.LIBERTY_BASENAMES]

    def _evaluation_tcl(self, root: Path) -> str:
        return track_a.build_evaluation_tcl(
            def_input=root / "committed.def",
            tech_lef=root / "tech.lef",
            lefs=[root / "cells.lef"],
            libs=self._libs(root),
            sdc=root / "design.sdc",
            rc_tcl=root / "setRC.tcl",
            output_def=root / "evaluated.def",
            inventory_path=root / "inventory.tsv",
            evaluated_inventory_path=root / "evaluated_inventory.tsv",
            dpl_report_path=root / "dpl_report.json",
            endpoint_csv=root / "endpoints.csv",
            power_report=root / "power.rpt",
        )

    def test_evaluation_tcl_has_one_dpl_and_no_repair_or_verilog(self):
        with tempfile.TemporaryDirectory() as tmp:
            tcl = self._evaluation_tcl(Path(tmp))
        self.assertEqual(tcl.count("\ndetailed_placement -max_displacement "), 1)
        self.assertIn('emit_metric "detailed_placement_search_window" "full_core"', tcl)
        self.assertNotIn("read_verilog", tcl)
        self.assertNotIn("repair_design", tcl)
        self.assertNotIn("repair_timing", tcl)
        self.assertNotIn("buffer_ports", tcl)
        self.assertIn("estimate_parasitics -placement", tcl)
        self.assertIn("set_power_activity -input -activity 0.1 -duty 0.5", tcl)
        self.assertIn('emit_metric "raw_hpwl_dbu"', tcl)
        self.assertIn('emit_metric "legalized_hpwl_dbu"', tcl)
        self.assertIn("-report_file_name", tcl)
        self.assertIn("create_macro_halo_blockages", tcl)
        self.assertIn("destroy_macro_halo_blockages", tcl)
        self.assertEqual(tcl.count("dump_inventory"), 3)

    def test_input_state_tcl_has_no_dpl_or_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tcl = track_a.build_evaluation_tcl(
                def_input=root / "r0.def",
                tech_lef=root / "tech.lef",
                lefs=[root / "cells.lef"],
                libs=self._libs(root),
                sdc=root / "design.sdc",
                rc_tcl=root / "setRC.tcl",
                output_def=root / "evaluated.def",
                inventory_path=root / "inventory.tsv",
                evaluated_inventory_path=root / "evaluated_inventory.tsv",
                dpl_report_path=root / "dpl_report.json",
                endpoint_csv=root / "endpoints.csv",
                power_report=root / "power.rpt",
                evaluation_mode="input_state",
            )
        self.assertNotIn("detailed_placement", tcl)
        self.assertNotIn("check_placement", tcl)
        self.assertNotIn("repair_design", tcl)
        self.assertNotIn("repair_timing", tcl)
        self.assertIn('emit_metric "placement_valid" -1', tcl)
        self.assertIn("estimate_parasitics -placement", tcl)

    def test_physical_finalize_tcl_has_one_dpl_and_distinct_artifact_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tcl = track_a.build_evaluation_tcl(
                def_input=root / "committed.def",
                tech_lef=root / "tech.lef",
                lefs=[root / "cells.lef"],
                libs=self._libs(root),
                sdc=root / "design.sdc",
                rc_tcl=root / "setRC.tcl",
                output_def=root / "evaluated.def",
                inventory_path=root / "inventory.tsv",
                evaluated_inventory_path=root / "evaluated_inventory.tsv",
                dpl_report_path=root / "dpl_report.json",
                endpoint_csv=root / "endpoints.csv",
                power_report=root / "power.rpt",
                evaluation_mode="physical_finalize",
            )
        self.assertEqual(tcl.count("\ndetailed_placement -max_displacement "), 1)
        self.assertIn("estimate_parasitics -placement", tcl)
        self.assertNotIn("read_verilog", tcl)

    def test_fixed_state_checks_legality_without_moving_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tcl = track_a.build_evaluation_tcl(
                def_input=root / "committed.def",
                tech_lef=root / "tech.lef",
                lefs=[root / "cells.lef"],
                libs=self._libs(root),
                sdc=root / "design.sdc",
                rc_tcl=root / "setRC.tcl",
                output_def=root / "evaluated.def",
                inventory_path=root / "inventory.tsv",
                evaluated_inventory_path=root / "evaluated_inventory.tsv",
                dpl_report_path=root / "dpl_report.json",
                endpoint_csv=root / "endpoints.csv",
                power_report=root / "power.rpt",
                evaluation_mode="fixed_state",
            )
        self.assertIn("check_placement -verbose", tcl)
        self.assertNotIn("detailed_placement", tcl)
        self.assertNotIn('emit_metric "placement_valid" -1', tcl)
        self.assertIn("estimate_parasitics -placement", tcl)

    def test_inventory_tcl_does_not_load_timing_or_modify_placement(self):
        root = Path("/tmp/fixture")
        tcl = track_a.build_inventory_tcl(
            def_input=root / "reference.def",
            tech_lef=root / "tech.lef",
            lefs=[root / "cells.lef"],
            inventory_path=root / "inventory.tsv",
        )
        self.assertNotIn("read_sdc", tcl)
        self.assertNotIn("read_liberty", tcl)
        self.assertNotIn("detailed_placement", tcl)
        self.assertIn("dump_inventory", tcl)

    def test_parse_metrics_preserves_zero_and_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.log"
            path.write_text(
                "METRIC|wns_ns|-0.125\n"
                "METRIC|tns_ns|0\n"
                "METRIC|placement_valid|1\n",
                encoding="utf-8",
            )
            metrics = track_a.parse_metrics(path)
        self.assertEqual(metrics["wns_ns"], -0.125)
        self.assertEqual(metrics["tns_ns"], 0)
        self.assertEqual(metrics["placement_valid"], 1)

    def test_read_inventory_and_mutation_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_path = root / "reference.tsv"
            committed_path = root / "committed.tsv"
            reference_path.write_text(
                "instance\tmaster\tis_block\tarea_um2\tx_dbu\ty_dbu\torient\tplacement_status\tbbox_x_min_dbu\tbbox_y_min_dbu\tbbox_x_max_dbu\tbbox_y_max_dbu\tkeepout_x_min_dbu\tkeepout_y_min_dbu\tkeepout_x_max_dbu\tkeepout_y_max_dbu\n"
                "u0\tINVx1\t0\t1.0\t0\t0\tR0\tPLACED\t0\t0\t1\t1\t0\t0\t1\t1\n"
                "u1\tINVx1\t0\t1.0\t10\t0\tR0\tPLACED\t10\t0\t11\t1\t10\t0\t11\t1\n"
                "mem\tSRAM\t1\t100.0\t20\t0\tR0\tFIRM\t20\t0\t30\t10\t20\t0\t30\t10\n",
                encoding="utf-8",
            )
            committed_path.write_text(
                "instance\tmaster\tis_block\tarea_um2\tx_dbu\ty_dbu\torient\tplacement_status\tbbox_x_min_dbu\tbbox_y_min_dbu\tbbox_x_max_dbu\tbbox_y_max_dbu\tkeepout_x_min_dbu\tkeepout_y_min_dbu\tkeepout_x_max_dbu\tkeepout_y_max_dbu\n"
                "u0\tINVx2\t0\t2.0\t0\t0\tR0\tPLACED\t0\t0\t2\t1\t0\t0\t2\t1\n"
                "u1\tINVx1\t0\t1.0\t11\t0\tR0\tPLACED\t11\t0\t12\t1\t11\t0\t12\t1\n"
                "buf0\tBUFx4\t0\t0.5\t5\t0\tR0\tPLACED\t5\t0\t6\t1\t5\t0\t6\t1\n"
                "mem\tSRAM\t1\t100.0\t20\t0\tR0\tFIRM\t20\t0\t30\t10\t20\t0\t30\t10\n",
                encoding="utf-8",
            )
            summary = track_a.summarize_mutations(
                track_a.read_inventory(reference_path),
                track_a.read_inventory(committed_path),
                buffer_masters=["BUFx4"],
            )
        self.assertEqual(summary["resize_count"], 1)
        self.assertEqual(summary["resized_instances"], ["u0"])
        self.assertEqual(summary["inserted_buffer_count"], 1)
        self.assertEqual(summary["inserted_other_count"], 0)
        self.assertEqual(summary["deleted_instance_count"], 0)
        self.assertEqual(summary["coordinate_changed_count"], 1)
        self.assertEqual(summary["coordinate_changed_instances"], ["u1"])
        self.assertAlmostEqual(summary["added_cell_area_um2"], 1.5)

    def test_unknown_or_duplicate_inventory_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.tsv"
            path.write_text(
                "instance\tmaster\tis_block\tarea_um2\tx_dbu\ty_dbu\torient\tplacement_status\tbbox_x_min_dbu\tbbox_y_min_dbu\tbbox_x_max_dbu\tbbox_y_max_dbu\tkeepout_x_min_dbu\tkeepout_y_min_dbu\tkeepout_x_max_dbu\tkeepout_y_max_dbu\n"
                "u0\tINVx1\t0\t1.0\t0\t0\tR0\tPLACED\t0\t0\t1\t1\t0\t0\t1\t1\n"
                "u0\tINVx2\t0\t2.0\t0\t0\tR0\tPLACED\t0\t0\t2\t1\t0\t0\t2\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate inventory"):
                track_a.read_inventory(path)

    def test_complete_iccad24_liberty_manifest_is_required(self):
        paths = [Path(name) for name in self.LIBERTY_BASENAMES]
        track_a.validate_complete_iccad24_liberty_manifest(paths)

        with self.assertRaisesRegex(ValueError, "complete 9-Liberty"):
            track_a.validate_complete_iccad24_liberty_manifest(paths[:-1])

        with self.assertRaisesRegex(ValueError, "SIMPLE_RVT"):
            track_a.validate_complete_iccad24_liberty_manifest(
                [path for path in paths if "SIMPLE_RVT" not in path.name]
            )

        with self.assertRaisesRegex(ValueError, "duplicates"):
            track_a.validate_complete_iccad24_liberty_manifest(paths + [paths[0]])

    def test_evaluation_tcl_rejects_incomplete_liberty_manifest(self):
        with self.assertRaisesRegex(ValueError, "complete 9-Liberty"):
            track_a.build_evaluation_tcl(
                def_input=Path("/tmp/committed.def"),
                tech_lef=Path("/tmp/tech.lef"),
                lefs=[Path("/tmp/cells.lef")],
                libs=[Path(name) for name in self.LIBERTY_BASENAMES[:-1]],
                sdc=Path("/tmp/design.sdc"),
                rc_tcl=Path("/tmp/setRC.tcl"),
                output_def=Path("/tmp/evaluated.def"),
                inventory_path=Path("/tmp/inventory.tsv"),
                evaluated_inventory_path=Path("/tmp/evaluated_inventory.tsv"),
                dpl_report_path=Path("/tmp/dpl_report.json"),
                endpoint_csv=Path("/tmp/endpoints.csv"),
                power_report=Path("/tmp/power.rpt"),
            )

    def test_plan_only_writes_tcl_and_nonterminal_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            rc = track_a.main(
                [
                    "--case",
                    "case0",
                    "--method-id",
                    "method0",
                    "--openroad-bin",
                    str(root / "openroad"),
                    "--openroad-num-threads",
                    "16",
                    "--reference-def",
                    str(root / "reference.def"),
                    "--def-input",
                    str(root / "committed.def"),
                    "--tech-lef",
                    str(root / "tech.lef"),
                    "--lef",
                    str(root / "cells.lef"),
                    *sum(
                        (["--lib", str(root / name)] for name in self.LIBERTY_BASENAMES),
                        [],
                    ),
                    "--sdc",
                    str(root / "design.sdc"),
                    "--rc-tcl",
                    str(root / "setRC.tcl"),
                    "--output-dir",
                    str(output_dir),
                    "--plan-only",
                ]
            )
            summary = json.loads(
                (output_dir / "track_a_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(rc, 0)
            self.assertEqual(summary["artifact"], track_a.ARTIFACT_KIND)
            self.assertEqual(summary["artifact_version"], 1)
            self.assertEqual(summary["status"], "plan_only")
            self.assertTrue((output_dir / "evaluate.tcl").is_file())
            self.assertTrue((output_dir / "reference_inventory.tcl").is_file())
            commands = json.loads(
                (output_dir / "commands.json").read_text(encoding="utf-8")
            )
            for command in commands.values():
                self.assertEqual(command[command.index("-threads") + 1], "16")


if __name__ == "__main__":
    unittest.main()
