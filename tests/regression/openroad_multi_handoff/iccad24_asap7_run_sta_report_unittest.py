import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import iccad24_asap7_run_sta_report as report


class Iccad24Asap7RunStaReportTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_expand_variants_uses_cartesian_product_and_stable_labels(self):
        variants = report.expand_variants(
            handoff_modes=["buffer-only", "no-handoff"],
            target_density_overrides=[0.2],
            stop_overflow_overrides=[0.075, None],
            with_sta_values=[False],
            max_handoff_overflows=[0.2],
        )

        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "buffer_only_td0p2_so0p075_sta0_mho0p2",
                "buffer_only_td0p2_soNative_sta0_mho0p2",
                "no_handoff_td0p2_so0p075_sta0_mho0p2",
                "no_handoff_td0p2_soNative_sta0_mho0p2",
            ],
        )

    def test_expand_variants_labels_warm_restart_policy(self):
        variants = report.expand_variants(
            handoff_modes=["buffer-only"],
            target_density_overrides=[0.2],
            stop_overflow_overrides=[0.075],
            with_sta_values=[False],
            max_handoff_overflows=[0.2],
            handoff_restart_policies=["cold", "warm_schedule"],
        )

        self.assertEqual(
            [variant["label"] for variant in variants],
            [
                "buffer_only_td0p2_so0p075_sta0_mho0p2",
                "buffer_only_td0p2_so0p075_sta0_mho0p2_rstwarm_schedule",
            ],
        )

    def test_build_report_row_preserves_failure_without_sta_metrics(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "buffer-only",
            "with_sta": False,
            "failure": {"type": "RuntimeError", "message": "boom"},
            "final_def_exists": False,
            "run_dir": "run/hidden3",
            "config": {"diagnostic_param_overrides": {"target_density": 0.2}},
        }

        row = report.build_report_row(
            placement,
            variant_label="buffer_only",
            sta_result=None,
        )

        self.assertEqual(row["design"], "hidden3")
        self.assertEqual(row["mode_config"], "buffer_only")
        self.assertEqual(row["status"], "placement_failed")
        self.assertIn("RuntimeError: boom", row["failure"])
        self.assertIsNone(row["WNS(ns)"])

    def test_build_report_row_skips_sta_when_final_def_validation_failed(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "no-handoff",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {
                "status": "failed",
                "failure": "1/1 movable components outside DIEAREA",
            },
            "run_dir": "run/hidden3",
            "config": {"diagnostic_param_overrides": {}},
        }

        row = report.build_report_row(
            placement,
            variant_label="no_handoff",
            sta_result=None,
        )

        self.assertEqual(row["status"], "skipped_invalid_final_def")
        self.assertIn("outside DIEAREA", row["failure"])
        self.assertIsNone(row["WNS(ns)"])

    def test_build_report_row_skips_sta_when_final_def_validation_not_run(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "no-handoff",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {"status": "not_run", "failure": ""},
            "run_dir": "run/hidden3",
            "config": {"diagnostic_param_overrides": {}},
        }

        row = report.build_report_row(
            placement,
            variant_label="no_handoff",
            sta_result=None,
        )

        self.assertEqual(row["status"], "skipped_unvalidated_final_def")
        self.assertIn("final DEF validation status is not_run", row["failure"])
        self.assertIsNone(row["WNS(ns)"])

    def test_build_report_row_labels_preserved_routability_control(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "no-handoff",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "run_dir": "run/hidden3",
            "config": {
                "diagnostic_param_overrides": {},
                "preserve_routability_opt": True,
            },
        }

        row = report.build_report_row(
            placement,
            variant_label="no_handoff_tdNative_soNative_sta0_mho0p2",
            sta_result={"status": "sta_completed", "failure": "", "metrics": {}},
        )

        self.assertEqual(
            row["mode_config"],
            "no_handoff_tdNative_soNative_sta0_mho0p2_roptnative",
        )

    def test_build_report_row_surfaces_handoff_and_buffer_metadata(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "buffer-only",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {"status": "passed"},
            "run_dir": "run/hidden3",
            "handoff_summary": {
                "handoff_count": 2,
                "added_buffer_count": 5,
                "removed_buffer_count": 1,
                "surviving_buffer_count": 4,
            },
            "config": {
                "diagnostic_param_overrides": {
                    "target_density": 0.5,
                    "stop_overflow": 0.075,
                },
                "handoff_restart_policy": "cold",
            },
            "final_def": "run/hidden3/final.def",
        }
        sta_result = {
            "status": "sta_completed",
            "failure": "",
            "artifact_dir": "sta/hidden3",
            "log_path": "sta/hidden3/openroad_sta.log",
            "metrics": {"wns_ns": -0.1, "tns_ns": -1.0},
        }

        row = report.build_report_row(
            placement,
            variant_label="buffer_only_td0p5_so0p075_sta0_mho0p2",
            sta_result=sta_result,
        )

        self.assertEqual(row["handoff_count"], 2)
        self.assertEqual(row["added_buffer_count"], 5)
        self.assertEqual(row["removed_buffer_count"], 1)
        self.assertEqual(row["surviving_buffer_count"], 4)
        self.assertEqual(row["final_def_validation_status"], "passed")
        self.assertEqual(row["final_def"], "run/hidden3/final.def")
        self.assertEqual(row["sta_log_path"], "sta/hidden3/openroad_sta.log")

    def test_build_report_row_reads_runner_session_handoff_metadata(self):
        placement = {
            "design": "hidden3",
            "handoff_mode": "buffer-only",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {"status": "passed"},
            "run_dir": "run/hidden3",
            "session": {
                "event_count": 3,
                "buffer_churn_totals": {
                    "added_buffer_count": 7,
                    "removed_buffer_count": 2,
                    "surviving_buffer_count": 5,
                },
            },
            "config": {"diagnostic_param_overrides": {}},
        }

        row = report.build_report_row(
            placement,
            variant_label="buffer_only_td0p5_so0p075_sta0_mho0p2",
            sta_result={"status": "sta_completed", "failure": "", "metrics": {}},
        )

        self.assertEqual(row["handoff_count"], 3)
        self.assertEqual(row["added_buffer_count"], 7)
        self.assertEqual(row["removed_buffer_count"], 2)
        self.assertEqual(row["surviving_buffer_count"], 5)

    def test_generate_sta_tcl_reads_asap7_inputs_and_emits_metrics(self):
        sta_dir = self.root / "sta"
        tcl = report.generate_sta_tcl(
            design="hidden3",
            tech_bundle={
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": ["/tech/cells.lib"],
                "rc_tcl": "/tech/setRC.tcl",
            },
            final_def="/run/hidden3.def",
            sdc_path="/design/hidden3.sdc",
            sta_dir=sta_dir,
        )

        self.assertIn("read_lef {/tech/asap7.tech.lef}", tcl)
        self.assertIn("read_lef {/tech/cells.lef}", tcl)
        self.assertIn("read_liberty {/tech/cells.lib}", tcl)
        self.assertIn("read_def {/run/hidden3.def}", tcl)
        self.assertIn("read_sdc {/design/hidden3.sdc}", tcl)
        self.assertIn("set_cmd_units -time ns -capacitance fF", tcl)
        self.assertIn("source {/tech/setRC.tcl}", tcl)
        self.assertIn("estimate_parasitics -placement", tcl)
        self.assertIn("report_cell_usage -file", tcl)
        self.assertNotIn("design_power", tcl)
        self.assertNotIn("total_leakage_uw", tcl)
        self.assertIn(
            "report_cell_usage -file {%s}" % (sta_dir / "cell_usage.rpt").resolve(),
            tcl,
        )
        self.assertIn(
            "report_check_types -max_slew -violators -digits 6 > {%s}"
            % (sta_dir / "max_slew_violators.rpt").resolve(),
            tcl,
        )
        self.assertIn(
            "report_check_types -max_capacitance -violators -digits 6 > {%s}"
            % (sta_dir / "max_capacitance_violators.rpt").resolve(),
            tcl,
        )
        self.assertIn("report_power > {%s}" % (sta_dir / "power.rpt").resolve(), tcl)
        self.assertIn("sta::network_instance_count", tcl)
        self.assertIn("sta::worst_slack -max", tcl)
        self.assertIn("sta::total_negative_slack -max", tcl)
        self.assertIn("report_check_types -max_slew -violators", tcl)
        self.assertIn("report_check_types -max_capacitance -violators", tcl)

    def test_generate_sta_tcl_resolves_relative_artifact_report_paths(self):
        relative_sta_dir = Path("relative") / "sta"
        tcl = report.generate_sta_tcl(
            design="hidden3",
            tech_bundle={
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": [],
                "lib_paths": [],
                "rc_tcl": "/tech/setRC.tcl",
            },
            final_def="/run/hidden3.def",
            sdc_path="/design/hidden3.sdc",
            sta_dir=relative_sta_dir,
        )

        self.assertIn(
            "report_cell_usage -file {%s}"
            % (relative_sta_dir / "cell_usage.rpt").resolve(),
            tcl,
        )
        self.assertNotIn("report_cell_usage -file {relative/sta", tcl)
        self.assertIn(
            "report_power > {%s}" % (relative_sta_dir / "power.rpt").resolve(),
            tcl,
        )
        self.assertNotIn("report_power > {relative/sta", tcl)

    def test_generate_sta_tcl_emits_ns_labels_for_timing_metrics(self):
        lib = self.root / "cells.lib"
        lib.write_text(
            "\n".join(
                [
                    "library(test) {",
                    '  time_unit : "1ps";',
                    "}",
                ]
            )
        )
        tcl = report.generate_sta_tcl(
            design="hidden3",
            tech_bundle={
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": [],
                "lib_paths": [str(lib)],
                "rc_tcl": "/tech/setRC.tcl",
            },
            final_def="/run/hidden3.def",
            sdc_path="/design/hidden3.sdc",
            sta_dir=self.root / "sta",
        )

        self.assertIn("set_cmd_units -time ns -capacitance fF", tcl)
        self.assertIn('AUTODMP_STA_METRIC wns_ns %.12g ns', tcl)
        self.assertIn('AUTODMP_STA_METRIC tns_ns %.12g ns', tcl)
        self.assertNotIn('AUTODMP_STA_METRIC wns_ns %%.12g', tcl)
        self.assertNotIn('AUTODMP_STA_METRIC tns_ns %%.12g', tcl)
        self.assertNotIn('AUTODMP_STA_METRIC wns_ns %.12g 1ps', tcl)
        self.assertNotIn('AUTODMP_STA_METRIC tns_ns %.12g 1ps', tcl)

    def test_parse_sta_metric_lines_converts_numbers_and_units(self):
        metrics = report.parse_sta_metric_lines(
            "\n".join(
                [
                    "AUTODMP_STA_METRIC gate_count 123 count",
                    "AUTODMP_STA_METRIC wns_ns -0.125 ns",
                    "AUTODMP_STA_METRIC tns_ns -10.5 ns",
                    "AUTODMP_STA_METRIC total_leakage_uw 42.25 uW",
                ]
            )
        )

        self.assertEqual(metrics["gate_count"], 123)
        self.assertEqual(metrics["wns_ns"], -0.125)
        self.assertEqual(metrics["tns_ns"], -10.5)
        self.assertEqual(metrics["total_leakage_uw"], 42.25)

    def test_parse_sta_metric_lines_preserves_ns_named_metrics_from_legacy_ps_labels(self):
        metrics = report.parse_sta_metric_lines(
            "\n".join(
                [
                    "AUTODMP_STA_METRIC gate_count 123 count",
                    "AUTODMP_STA_METRIC wns_ns -0.526997294053 1ps",
                    "AUTODMP_STA_METRIC tns_ns -142.143140392 1ps",
                    "AUTODMP_STA_METRIC total_leakage_uw 42.25 uW",
                ]
            )
        )

        self.assertEqual(metrics["gate_count"], 123)
        self.assertAlmostEqual(metrics["wns_ns"], -0.526997294053)
        self.assertAlmostEqual(metrics["tns_ns"], -142.143140392)
        self.assertEqual(metrics["total_leakage_uw"], 42.25)

    def test_parse_power_report_returns_total_leakage_in_microwatts(self):
        power_report = "\n".join(
            [
                "Group                  Internal  Switching    Leakage      Total",
                "                          Power      Power      Power      Power (Watts)",
                "----------------------------------------------------------------",
                "Total                  5.12e-04   3.73e-04   1.00e-09   8.85e-04 100.0%",
                "                          57.8%      42.2%       0.0%",
            ]
        )

        self.assertAlmostEqual(
            report.parse_total_leakage_uw(power_report),
            0.001,
        )

    def test_sum_violation_report_slacks_returns_positive_total(self):
        text = "\n".join(
            [
                "max slew",
                "",
                "Pin                                    Limit    Slew   Slack",
                "------------------------------------------------------------",
                "rdrv/Y                                  1.48    3.33   -1.85 (VIOLATED)",
                "r0/D                                    1.50    3.33   -1.83 (VIOLATED)",
                "met/D                                   1.50    1.20    0.30 (MET)",
            ]
        )

        self.assertAlmostEqual(report.sum_violation_report_slacks(text), 3.68)

    def test_sum_violation_report_slacks_scales_ps_timing_to_ns(self):
        text = "\n".join(
            [
                "max slew",
                "",
                "Pin                                    Limit    Slew   Slack",
                "------------------------------------------------------------",
                "rdrv/Y                                  320.0  9341.0 -9021.0 (VIOLATED)",
                "r0/D                                    320.0   900.0  -580.0 (VIOLATED)",
            ]
        )

        self.assertAlmostEqual(
            report.sum_violation_report_slacks(text, time_unit="ps"),
            9.601,
        )
        self.assertAlmostEqual(
            report.sum_violation_report_slacks(text, time_unit="ns"),
            9601.0,
        )

    def test_run_sta_for_summary_uses_ns_timing_metrics_after_set_cmd_units(self):
        sta_base = self.root / "sta"
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        sdc = self.root / "hidden3.sdc"
        sdc.write_text("create_clock -period 2100 clk\n")
        lib = self.root / "cells.lib"
        lib.write_text(
            "\n".join(
                [
                    "library(test) {",
                    '  time_unit : "1ps";',
                    "  capacitive_load_unit (1,ff);",
                    "}",
                ]
            )
        )
        placement = {
            "design": "hidden3",
            "final_def": str(final_def),
            "case": {"sdc_path": str(sdc)},
            "tech_bundle": {
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": [str(lib)],
                "rc_tcl": "/tech/setRC.tcl",
            },
        }

        def fake_run(cmd, cwd, stdout, stderr, text):
            Path(cwd, "max_slew_violators.rpt").write_text(
                "Pin Limit Slew Slack\nu/Y 320.0 9341.0 -9021.0 (VIOLATED)\n"
            )
            Path(cwd, "max_capacitance_violators.rpt").write_text(
                "Pin Limit Cap Slack\nu/Y 1.0 3.0 -2.5 (VIOLATED)\n"
            )
            Path(cwd, "power.rpt").write_text(
                "Total 1.00e-06 2.00e-06 9.50e-06 1.25e-05 100.0%\n"
            )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="\n".join(
                    [
                        "AUTODMP_STA_METRIC gate_count 123 count",
                        "AUTODMP_STA_METRIC wns_ns -32.6124 ns",
                        "AUTODMP_STA_METRIC tns_ns -237211.204323 ns",
                    ]
                ),
                stderr="",
            )

        with patch.object(report.subprocess, "run", side_effect=fake_run):
            result = report.run_sta_for_summary(
                placement,
                sta_base,
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_completed")
        self.assertAlmostEqual(result["metrics"]["wns_ns"], -32.6124)
        self.assertAlmostEqual(result["metrics"]["tns_ns"], -237211.204323)
        self.assertAlmostEqual(
            result["metrics"]["total_slew_violation_difference_ns"],
            9021.0,
        )
        self.assertEqual(
            result["metrics"]["total_load_capacitance_violation_difference_ff"],
            2.5,
        )

    def test_run_sta_for_summary_writes_artifacts_and_parses_metrics(self):
        sta_base = self.root / "sta"
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        sdc = self.root / "hidden3.sdc"
        sdc.write_text("create_clock -period 1 clk\n")
        placement = {
            "design": "hidden3",
            "final_def": str(final_def),
            "case": {"sdc_path": str(sdc)},
            "tech_bundle": {
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": ["/tech/cells.lib"],
                "rc_tcl": "/tech/setRC.tcl",
            },
        }

        def fake_run(cmd, cwd, stdout, stderr, text):
            self.assertTrue(Path(cmd[1]).is_absolute())
            self.assertIn(str(final_def.resolve()), Path(cmd[1]).read_text())
            self.assertIn(str(sdc.resolve()), Path(cmd[1]).read_text())
            Path(cwd, "max_slew_violators.rpt").write_text(
                "Pin Limit Slew Slack\nu/Y 1.0 2.0 -0.5 (VIOLATED)\n"
            )
            Path(cwd, "max_capacitance_violators.rpt").write_text(
                "Pin Limit Cap Slack\nu/Y 1.0 3.0 -2.5 (VIOLATED)\n"
            )
            Path(cwd, "power.rpt").write_text(
                "Total 1.00e-06 2.00e-06 9.50e-06 1.25e-05 100.0%\n"
            )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="\n".join(
                    [
                        "AUTODMP_STA_METRIC gate_count 123 count",
                        "AUTODMP_STA_METRIC wns_ns -0.1 ns",
                        "AUTODMP_STA_METRIC tns_ns -2.0 ns",
                    ]
                ),
                stderr="",
            )

        with patch.object(report.subprocess, "run", side_effect=fake_run):
            result = report.run_sta_for_summary(
                placement,
                sta_base,
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_completed")
        self.assertEqual(result["metrics"]["gate_count"], 123)
        self.assertEqual(result["metrics"]["total_slew_violation_difference_ns"], 0.5)
        self.assertEqual(result["metrics"]["total_leakage_uw"], 9.5)
        self.assertEqual(
            result["metrics"]["total_load_capacitance_violation_difference_ff"],
            2.5,
        )
        self.assertTrue(Path(result["tcl_path"]).exists())
        self.assertTrue(Path(result["metrics_path"]).exists())

    def test_run_sta_for_summary_resolves_relative_design_paths_in_tcl(self):
        sta_base = self.root / "sta"
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        sdc = self.root / "hidden3.sdc"
        sdc.write_text("create_clock -period 1 clk\n")
        relative_def = os.path.relpath(final_def, Path.cwd())
        relative_sdc = os.path.relpath(sdc, Path.cwd())
        placement = {
            "design": "hidden3",
            "final_def": str(relative_def),
            "case": {"sdc_path": str(relative_sdc)},
            "tech_bundle": {
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": ["/tech/cells.lib"],
                "rc_tcl": "/tech/setRC.tcl",
            },
        }

        def fake_run(cmd, cwd, stdout, stderr, text):
            tcl_text = Path(cmd[1]).read_text()
            self.assertIn("read_def {%s}" % final_def.resolve(), tcl_text)
            self.assertIn("read_sdc {%s}" % sdc.resolve(), tcl_text)
            Path(cwd, "power.rpt").write_text(
                "Total 1.00e-06 2.00e-06 9.50e-06 1.25e-05 100.0%\n"
            )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="\n".join(
                    [
                        "AUTODMP_STA_METRIC gate_count 123 count",
                        "AUTODMP_STA_METRIC wns_ns -0.1 ns",
                        "AUTODMP_STA_METRIC tns_ns -2.0 ns",
                    ]
                ),
                stderr="",
            )

        with patch.object(report.subprocess, "run", side_effect=fake_run):
            result = report.run_sta_for_summary(
                placement,
                sta_base,
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_completed")

    def test_run_sta_for_summary_marks_missing_scalar_metrics_as_parse_failure(self):
        sta_base = self.root / "sta"
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        sdc = self.root / "hidden3.sdc"
        sdc.write_text("create_clock -period 1 clk\n")
        placement = {
            "design": "hidden3",
            "final_def": str(final_def),
            "case": {"sdc_path": str(sdc)},
            "tech_bundle": {
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": ["/tech/cells.lib"],
                "rc_tcl": "/tech/setRC.tcl",
            },
        }

        with patch.object(
            report.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["openroad"],
                0,
                stdout="[ERROR STA-0340] cannot open Tcl\n",
                stderr="",
            ),
        ):
            result = report.run_sta_for_summary(
                placement,
                sta_base,
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_parse_failed")
        self.assertIn("missing STA metrics", result["failure"])
        self.assertIn("gate_count", result["parse_warnings"][0])

    def test_run_sta_for_summary_records_openroad_failure(self):
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        placement = {
            "design": "hidden3",
            "final_def": str(final_def),
            "case": {"sdc_path": str(self.root / "missing.sdc")},
            "tech_bundle": {
                "tech_lef": "tech.lef",
                "lef_paths": [],
                "lib_paths": [],
                "rc_tcl": "setRC.tcl",
            },
        }

        with patch.object(
            report.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["openroad"],
                1,
                stdout="",
                stderr="bad",
            ),
        ):
            result = report.run_sta_for_summary(
                placement,
                self.root / "sta",
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_failed")
        self.assertIn("exit 1", result["failure"])

    def test_run_sta_for_summary_records_missing_openroad_binary(self):
        final_def = self.root / "hidden3.def"
        final_def.write_text("VERSION 5.8 ;\nEND DESIGN\n")
        placement = {
            "design": "hidden3",
            "final_def": str(final_def),
            "case": {"sdc_path": str(self.root / "missing.sdc")},
            "tech_bundle": {
                "tech_lef": "tech.lef",
                "lef_paths": [],
                "lib_paths": [],
                "rc_tcl": "setRC.tcl",
            },
        }

        with patch.object(
            report.subprocess,
            "run",
            side_effect=FileNotFoundError("openroad"),
        ):
            result = report.run_sta_for_summary(
                placement,
                self.root / "sta",
                openroad_bin="openroad",
                variant_label="buffer_only",
            )

        self.assertEqual(result["status"], "sta_failed")
        self.assertIn("openroad binary not found", result["failure"])
        self.assertTrue(Path(result["metrics_path"]).exists())

    def test_run_variant_skips_sta_when_final_def_validation_failed(self):
        variant = {
            "label": "no_handoff_tdNative_soNative_sta0_mho0p2",
            "handoff_mode": "no-handoff",
            "target_density_override": None,
            "stop_overflow_override": None,
            "with_sta": False,
            "max_handoff_overflow": 0.2,
        }
        args = type(
            "Args",
            (),
            {
                "dry_run": False,
                "target_iter": None,
                "trigger_period": 50,
                "margin": 0.0,
                "plot_interval": None,
                "preserve_routability_opt": False,
                "openroad_bin": "openroad",
            },
        )()
        placement_summary = {
            "design": "hidden3",
            "handoff_mode": "no-handoff",
            "with_sta": False,
            "max_handoff_overflow": 0.2,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {
                "status": "failed",
                "failure": "1/1 movable components outside DIEAREA",
            },
            "run_dir": str(self.root / "placement" / "hidden3"),
            "config": {"diagnostic_param_overrides": {}},
        }

        with patch.object(
            report.placement_runner,
            "run_one",
            return_value=placement_summary,
        ), patch.object(report, "run_sta_for_summary") as run_sta:
            result = report.run_variant(
                variant,
                designs=["hidden3"],
                benchmark_root=self.root / "benchmark",
                run_dir=self.root / "run",
                args=args,
            )

        run_sta.assert_not_called()
        self.assertEqual(result["sta_results"], [])
        self.assertEqual(result["rows"][0]["status"], "skipped_invalid_final_def")

    def test_run_variant_skips_sta_when_final_def_validation_not_passed(self):
        variant = {
            "label": "no_handoff_tdNative_soNative_sta0_mho0p2",
            "handoff_mode": "no-handoff",
            "target_density_override": None,
            "stop_overflow_override": None,
            "with_sta": False,
            "max_handoff_overflow": 0.2,
        }
        args = type(
            "Args",
            (),
            {
                "dry_run": False,
                "target_iter": None,
                "trigger_period": 50,
                "margin": 0.0,
                "plot_interval": None,
                "preserve_routability_opt": False,
                "openroad_bin": "openroad",
            },
        )()
        placement_summary = {
            "design": "hidden3",
            "handoff_mode": "no-handoff",
            "with_sta": False,
            "max_handoff_overflow": 0.2,
            "failure": None,
            "final_def_exists": True,
            "final_def_validation": {"status": "not_run", "failure": ""},
            "run_dir": str(self.root / "placement" / "hidden3"),
            "config": {"diagnostic_param_overrides": {}},
        }

        with patch.object(
            report.placement_runner,
            "run_one",
            return_value=placement_summary,
        ), patch.object(report, "run_sta_for_summary") as run_sta:
            result = report.run_variant(
                variant,
                designs=["hidden3"],
                benchmark_root=self.root / "benchmark",
                run_dir=self.root / "run",
                args=args,
            )

        run_sta.assert_not_called()
        self.assertEqual(result["sta_results"], [])
        self.assertEqual(result["rows"][0]["status"], "skipped_unvalidated_final_def")

    def test_main_accepts_with_sta_flag(self):
        artifact_base = self.root / "artifacts"

        exit_code = report.main(
            [
                "--artifact-base",
                str(artifact_base),
                "--design",
                "hidden3",
                "--dry-run",
                "--no-timestamp",
                "--with-sta",
            ]
        )

        self.assertEqual(exit_code, 0)
        summary_json = artifact_base / "run_sta_summary.json"
        data = report.json.loads(summary_json.read_text())
        self.assertTrue(data["rows"][0]["with_sta"])

    def test_main_dry_run_reports_warm_restart_policy(self):
        artifact_base = self.root / "artifacts"

        exit_code = report.main(
            [
                "--artifact-base",
                str(artifact_base),
                "--design",
                "hidden3",
                "--dry-run",
                "--no-timestamp",
                "--handoff-restart-policy",
                "warm_schedule",
            ]
        )

        self.assertEqual(exit_code, 0)
        summary_json = artifact_base / "run_sta_summary.json"
        data = report.json.loads(summary_json.read_text())
        row = data["rows"][0]
        self.assertEqual(row["handoff_restart_policy"], "warm_schedule")
        self.assertIn("_rstwarm_schedule", row["mode_config"])

    def test_baseline_family_from_row_classifies_sources(self):
        cases = [
            ({"source": "original", "mode_config": "original_def"}, "original_def_opensta"),
            (
                {"source": "original_repaired", "mode_config": "original_def_repair_timing"},
                "original_def_openroad_repair_opensta",
            ),
            (
                {"source": "cold", "mode_config": "no_handoff_td0p5_so0p075_sta0_mho0p2"},
                "autodmp_no_handoff_opensta",
            ),
            (
                {"source": "cold", "mode_config": "no_handoff_repair_td0p5_so0p075_sta0_mho0p2"},
                "autodmp_no_handoff_openroad_repair_opensta",
            ),
            (
                {"source": "cold", "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2"},
                "autodmp_handoff_cold_opensta",
            ),
            (
                {"source": "warm", "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2_rstwarm_schedule"},
                "autodmp_handoff_warm_opensta",
            ),
        ]

        self.assertEqual(
            [report.baseline_family_from_row(row) for row, _ in cases],
            [expected for _, expected in cases],
        )

    def test_add_metric_deltas_uses_original_and_same_density_no_handoff(self):
        rows = [
            {
                "design": "hidden3",
                "source": "original",
                "mode_config": "original_def",
                "status": "sta_completed",
                "WNS(ns)": -1.0,
                "TNS(ns)": -10.0,
                "total_slew_violation_difference(ns)": 5.0,
                "target_density": None,
            },
            {
                "design": "hidden3",
                "source": "original_repaired",
                "mode_config": "original_def_repair_timing",
                "status": "sta_completed",
                "WNS(ns)": -0.5,
                "TNS(ns)": -5.0,
                "total_slew_violation_difference(ns)": 3.0,
                "target_density": None,
            },
            {
                "design": "hidden3",
                "source": "cold",
                "mode_config": "no_handoff_td0p5_so0p075_sta0_mho0p2",
                "status": "sta_completed",
                "WNS(ns)": -0.8,
                "TNS(ns)": -8.0,
                "total_slew_violation_difference(ns)": 4.0,
                "target_density": 0.5,
            },
            {
                "design": "hidden3",
                "source": "cold",
                "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2",
                "status": "sta_completed",
                "WNS(ns)": -0.2,
                "TNS(ns)": -2.0,
                "total_slew_violation_difference(ns)": 6.0,
                "target_density": 0.5,
            },
        ]

        enriched = report.add_metric_deltas(rows)
        handoff = enriched[3]

        self.assertEqual(handoff["baseline_family"], "autodmp_handoff_cold_opensta")
        self.assertAlmostEqual(handoff["delta_vs_original_WNS(ns)"], 0.8)
        self.assertAlmostEqual(handoff["delta_vs_original_TNS(ns)"], 8.0)
        self.assertAlmostEqual(
            handoff["delta_vs_same_density_no_handoff_WNS(ns)"],
            0.6,
        )
        self.assertAlmostEqual(
            handoff["delta_vs_same_density_no_handoff_total_slew_violation_difference(ns)"],
            2.0,
        )
        self.assertAlmostEqual(
            handoff["delta_vs_original_repaired_WNS(ns)"],
            0.3,
        )

    def test_grouped_comparison_markdown_uses_research_factor_columns(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden3",
                    "source": "cold",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2",
                    "handoff_mode": "buffer-only",
                    "target_density": 0.5,
                    "stop_overflow": 0.075,
                    "max_handoff_overflow": 0.2,
                    "with_sta": False,
                    "handoff_restart_policy": "cold",
                    "final_def_validation_status": "passed",
                    "handoff_count": 2,
                    "added_buffer_count": 5,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 4,
                    "status": "sta_completed",
                    "WNS(ns)": -0.2,
                    "TNS(ns)": -2.0,
                    "total_slew_violation_difference(ns)": 6.0,
                    "Total Load Capacitance Violation Difference(fF)": 8.0,
                    "Total Leakage(uw)": 4.0,
                    "gate_count": 11,
                },
            ]
        )

        markdown = report.render_grouped_comparison_markdown(rows)
        header = next(
            line for line in markdown.splitlines() if line.startswith("| casename |")
        )

        self.assertIn("| handoff mode |", header)
        self.assertIn("| target density |", header)
        self.assertIn("| stop overflow |", header)
        self.assertIn("| max handoff overflow |", header)
        self.assertIn("| with STA |", header)
        self.assertIn("| restart policy |", header)
        self.assertIn("| final DEF validation |", header)
        self.assertIn("| handoff count |", header)
        self.assertIn("| added buffers |", header)
        self.assertIn("| removed buffers |", header)
        self.assertIn("| surviving buffers |", header)
        self.assertNotIn("mode/config", header)
        self.assertNotIn("delta vs", header)
        self.assertIn(
            "| hidden3 | AutoDMP + Handoff + Cold | cold | buffer-only | 0.5 | 0.075 | 0.2 | False | cold | passed | 2 | 5 | 1 | 4 | sta_completed |",
            markdown,
        )

    def test_grouped_comparison_markdown_fills_mho_from_mode_config(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden3",
                    "source": "cold",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2",
                    "handoff_mode": "buffer-only",
                    "target_density": 0.5,
                    "stop_overflow": 0.075,
                    "max_handoff_overflow": None,
                    "with_sta": False,
                    "handoff_restart_policy": "cold",
                    "final_def_validation_status": "passed",
                    "handoff_count": 2,
                    "added_buffer_count": 5,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 4,
                    "status": "sta_completed",
                    "WNS(ns)": -0.2,
                    "TNS(ns)": -2.0,
                    "total_slew_violation_difference(ns)": 6.0,
                    "Total Load Capacitance Violation Difference(fF)": 8.0,
                    "Total Leakage(uw)": 4.0,
                    "gate_count": 11,
                },
            ]
        )

        markdown = report.render_grouped_comparison_markdown(rows)

        self.assertIn(
            "| hidden3 | AutoDMP + Handoff + Cold | cold | buffer-only | 0.5 | 0.075 | 0.2 | False | cold | passed |",
            markdown,
        )

    def test_render_grouped_comparison_markdown_groups_by_case_and_bolds_best(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden3",
                    "source": "original",
                    "mode_config": "original_def",
                    "status": "sta_completed",
                    "WNS(ns)": -1.0,
                    "TNS(ns)": -10.0,
                    "total_slew_violation_difference(ns)": 5.0,
                    "Total Load Capacitance Violation Difference(fF)": 7.0,
                    "Total Leakage(uw)": 3.0,
                    "gate_count": 10,
                },
                {
                    "design": "hidden3",
                    "source": "cold",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2",
                    "status": "sta_completed",
                    "WNS(ns)": -0.2,
                    "TNS(ns)": -2.0,
                    "total_slew_violation_difference(ns)": 6.0,
                    "Total Load Capacitance Violation Difference(fF)": 8.0,
                    "Total Leakage(uw)": 4.0,
                    "gate_count": 11,
                },
            ]
        )

        markdown = report.render_grouped_comparison_markdown(rows)

        self.assertIn("## hidden3", markdown)
        self.assertIn(
            "| baseline family | source | handoff mode | target density |",
            markdown,
        )
        self.assertNotIn("mode/config", markdown)
        self.assertNotIn("delta vs original WNS(ns)", markdown)
        self.assertNotIn("delta vs same-density no-handoff WNS(ns)", markdown)
        self.assertIn("**-0.2**", markdown)
        self.assertIn("**-2**", markdown)
        self.assertIn("**5**", markdown)
        self.assertIn("**7**", markdown)

    def test_render_rows_csv_includes_baseline_family_and_deltas(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden3",
                    "source": "original",
                    "mode_config": "original_def",
                    "status": "sta_completed",
                    "WNS(ns)": -1.0,
                    "TNS(ns)": -10.0,
                }
            ]
        )

        csv_text = report.render_rows_csv(rows)

        self.assertIn("baseline_family", csv_text.splitlines()[0])
        self.assertIn("original_def_opensta", csv_text)
        self.assertIn("delta_vs_original_WNS(ns)", csv_text.splitlines()[0])

    def test_render_rows_csv_puts_casename_first_and_uses_display_family(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden3",
                    "source": "warm",
                    "handoff_restart_policy": "warm_schedule",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2_rstwarm_schedule",
                    "status": "sta_completed",
                    "WNS(ns)": -1.0,
                    "TNS(ns)": -10.0,
                }
            ]
        )

        csv_lines = report.render_rows_csv(rows).splitlines()

        self.assertEqual(csv_lines[0].split(",")[0], "casename")
        self.assertIn("baseline_family_display", csv_lines[0])
        self.assertIn("AutoDMP + Handoff + Warm", csv_lines[1])
        self.assertNotIn("OpenSTA", csv_lines[1])

    def test_baseline_family_display_uses_plus_delimiters_and_symmetric_restart_names(self):
        rows = report.add_metric_deltas(
            [
                {
                    "design": "hidden2",
                    "source": "no_handoff",
                    "handoff_mode": "no-handoff",
                    "handoff_restart_policy": "cold",
                    "mode_config": "no_handoff_td0p5_so0p075_sta0_mho0p2",
                    "status": "sta_completed",
                    "WNS(ns)": -1.2,
                    "TNS(ns)": -12.0,
                },
                {
                    "design": "hidden3",
                    "source": "cold",
                    "handoff_restart_policy": "cold",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2",
                    "status": "sta_completed",
                    "WNS(ns)": -1.0,
                    "TNS(ns)": -10.0,
                },
                {
                    "design": "hidden3",
                    "source": "warm",
                    "handoff_restart_policy": "warm_schedule",
                    "mode_config": "buffer_only_td0p5_so0p075_sta0_mho0p2_rstwarm_schedule",
                    "status": "sta_completed",
                    "WNS(ns)": -0.8,
                    "TNS(ns)": -8.0,
                },
            ]
        )

        self.assertEqual(
            rows[0]["baseline_family_display"],
            "AutoDMP + No_Handoff",
        )
        self.assertEqual(
            rows[1]["baseline_family_display"],
            "AutoDMP + Handoff + Cold",
        )
        self.assertEqual(
            rows[2]["baseline_family_display"],
            "AutoDMP + Handoff + Warm",
        )

    def test_baseline_family_display_keeps_original_def_as_single_token(self):
        self.assertEqual(
            report.baseline_family_display_from_key("original_def_opensta"),
            "Original_DEF",
        )
        self.assertEqual(
            report.baseline_family_display_from_key(
                "original_def_openroad_repair_opensta"
            ),
            "Original_DEF + OpenROAD Repair",
        )

    def test_main_writes_enriched_json_markdown_and_csv_in_dry_run(self):
        run_dir = self.root / "report"

        exit_code = report.main(
            [
                "--artifact-base",
                str(run_dir),
                "--no-timestamp",
                "--dry-run",
                "--design",
                "hidden3",
                "--handoff-mode",
                "buffer-only",
                "--target-density-override",
                "0.5",
                "--stop-overflow-override",
                "0.075",
                "--max-handoff-overflow",
                "0.2",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue((run_dir / "run_sta_summary.json").exists())
        self.assertTrue((run_dir / "run_sta_summary.md").exists())
        self.assertTrue((run_dir / "run_sta_summary.csv").exists())
        self.assertTrue((run_dir / "comparison_summary.md").exists())
        payload = report.json.loads((run_dir / "run_sta_summary.json").read_text())
        self.assertIn("enriched_rows", payload)
        self.assertIn("baseline_family", payload["enriched_rows"][0])

    def test_build_original_def_summary_creates_valid_external_baseline_shape(self):
        original_def = self.root / "hidden3.def"
        original_def.write_text(
            "\n".join(
                [
                    "VERSION 5.8 ;",
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 1 ;",
                    "- u0 BUF + PLACED ( 10 20 ) N ;",
                    "END COMPONENTS",
                    "END DESIGN",
                ]
            )
            + "\n"
        )
        case = {
            "design": "hidden3",
            "def_input": str(original_def),
            "sdc_path": "/bench/design/hidden3/hidden3.sdc",
        }
        tech_bundle = {"tech_lef": "tech.lef", "lef_paths": [], "lib_paths": [], "rc_tcl": "setRC.tcl"}

        summary = report.build_original_def_summary(
            case,
            tech_bundle,
            artifact_dir="/tmp/original/hidden3",
        )

        self.assertEqual(summary["design"], "hidden3")
        self.assertEqual(summary["handoff_mode"], "original")
        self.assertEqual(summary["source"], "original")
        self.assertEqual(summary["final_def"], str(original_def))
        self.assertTrue(summary["final_def_exists"])
        self.assertEqual(summary["final_def_validation"]["status"], "passed")
        self.assertEqual(summary["config"]["handoff_restart_policy"], "cold")

    def test_generate_original_def_repair_tcl_uses_buffer_only_repair_command(self):
        output_def = self.root / "repair" / "hidden3_repaired.def"

        tcl = report.generate_original_def_repair_tcl(
            design="hidden3",
            tech_bundle={
                "tech_lef": "/tech/asap7.tech.lef",
                "lef_paths": ["/tech/cells.lef"],
                "lib_paths": ["/tech/cells.lib"],
                "rc_tcl": "/tech/setRC.tcl",
            },
            input_def="/design/hidden3.def",
            sdc_path="/design/hidden3.sdc",
            output_def=output_def,
            repair_dir=self.root / "repair",
            margin=0.0,
        )

        self.assertIn("read_def {/design/hidden3.def}", tcl)
        self.assertIn("read_sdc {/design/hidden3.sdc}", tcl)
        self.assertIn("set_cmd_units -time ns -capacitance fF", tcl)
        self.assertIn("repair_timing -setup", tcl)
        self.assertIn('-sequence "unbuffer,buffer,split"', tcl)
        self.assertIn("-skip_pin_swap", tcl)
        self.assertIn("-skip_gate_cloning", tcl)
        self.assertIn("-skip_size_down", tcl)
        self.assertNotIn("-skip_vt_swap", tcl)
        self.assertIn("write_def {%s}" % output_def.resolve(), tcl)

    def test_run_original_def_repair_summary_writes_repaired_baseline_shape(self):
        original_def = self.root / "hidden3.def"
        original_def.write_text(
            "\n".join(
                [
                    "VERSION 5.8 ;",
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 1 ;",
                    "- u0 BUF + PLACED ( 10 20 ) N ;",
                    "END COMPONENTS",
                    "END DESIGN",
                ]
            )
            + "\n"
        )
        case = {
            "design": "hidden3",
            "def_input": str(original_def),
            "sdc_path": str(self.root / "hidden3.sdc"),
        }
        tech_bundle = {"tech_lef": "tech.lef", "lef_paths": [], "lib_paths": [], "rc_tcl": "setRC.tcl"}
        artifact_dir = self.root / "repair_artifacts"

        def fake_run(cmd, cwd, stdout, stderr, text):
            repaired_def = artifact_dir / "hidden3.original_repaired.def"
            repaired_def.write_text(original_def.read_text())
            return subprocess.CompletedProcess(cmd, 0, stdout="repair ok\n", stderr="")

        with patch.object(report.subprocess, "run", side_effect=fake_run):
            summary = report.run_original_def_repair_summary(
                case,
                tech_bundle,
                artifact_dir=artifact_dir,
                openroad_bin="openroad",
                margin=0.0,
            )

        self.assertEqual(summary["design"], "hidden3")
        self.assertEqual(summary["handoff_mode"], "original_repair")
        self.assertEqual(summary["source"], "original_repaired")
        self.assertTrue(summary["final_def_exists"])
        self.assertEqual(summary["final_def_validation"]["status"], "passed")
        self.assertTrue(Path(summary["repair_tcl_path"]).exists())
        self.assertTrue(Path(summary["repair_log_path"]).exists())

    def test_run_original_def_repair_summary_treats_openroad_error_log_as_failure(self):
        original_def = self.root / "hidden3.def"
        original_def.write_text(
            "\n".join(
                [
                    "VERSION 5.8 ;",
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 1 ;",
                    "- u0 BUF + PLACED ( 10 20 ) N ;",
                    "END COMPONENTS",
                    "END DESIGN",
                ]
            )
            + "\n"
        )
        case = {
            "design": "hidden3",
            "def_input": str(original_def),
            "sdc_path": str(self.root / "hidden3.sdc"),
        }
        tech_bundle = {
            "tech_lef": "tech.lef",
            "lef_paths": [],
            "lib_paths": [],
            "rc_tcl": "setRC.tcl",
        }

        def fake_run(cmd, cwd, stdout, stderr, text):
            del cwd, stdout, stderr, text
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="[ERROR STA-0562] repair_timing -skip_vt_swap is not a known keyword\n",
                stderr="",
            )

        with patch.object(report.subprocess, "run", side_effect=fake_run):
            summary = report.run_original_def_repair_summary(
                case,
                tech_bundle,
                artifact_dir=self.root / "repair_error_artifacts",
                openroad_bin="openroad",
                margin=0.0,
            )

        self.assertEqual(summary["failure"]["type"], "OpenROADRepairFailed")
        self.assertIn("STA-0562", summary["failure"]["message"])
        self.assertFalse(summary["final_def_exists"])

    def test_main_dry_run_can_include_original_def_baseline_row(self):
        run_dir = self.root / "report_original"

        exit_code = report.main(
            [
                "--artifact-base",
                str(run_dir),
                "--no-timestamp",
                "--dry-run",
                "--include-original-def-baseline",
                "--design",
                "hidden3",
            ]
        )

        self.assertEqual(exit_code, 0)
        payload = report.json.loads((run_dir / "run_sta_summary.json").read_text())
        families = [row["baseline_family"] for row in payload["enriched_rows"]]
        self.assertIn("original_def_opensta", families)

    def test_main_dry_run_can_include_original_def_repair_baseline_row(self):
        run_dir = self.root / "report_original_repair"

        exit_code = report.main(
            [
                "--artifact-base",
                str(run_dir),
                "--no-timestamp",
                "--dry-run",
                "--include-original-def-repair-baseline",
                "--design",
                "hidden3",
            ]
        )

        self.assertEqual(exit_code, 0)
        payload = report.json.loads((run_dir / "run_sta_summary.json").read_text())
        families = [row["baseline_family"] for row in payload["enriched_rows"]]
        self.assertIn("original_def_openroad_repair_opensta", families)

    def test_main_original_def_baseline_skips_sta_when_validation_fails(self):
        run_dir = self.root / "report_invalid_original"
        invalid_def = self.root / "invalid.def"
        invalid_def.write_text(
            "\n".join(
                [
                    "VERSION 5.8 ;",
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 1 ;",
                    "- u0 BUF + PLACED ( 10000 10000 ) N ;",
                    "END COMPONENTS",
                    "END DESIGN",
                ]
            )
            + "\n"
        )
        manifest = {
            "cases": [
                {
                    "design": "hidden3",
                    "def_input": str(invalid_def),
                    "sdc_path": str(self.root / "hidden3.sdc"),
                }
            ]
        }

        with patch.object(report.placement_runner, "build_manifest", return_value=manifest), \
            patch.object(
                report.placement_runner,
                "asap7_tech_bundle",
                return_value={
                    "tech_lef": "tech.lef",
                    "lef_paths": [],
                    "lib_paths": [],
                    "rc_tcl": "setRC.tcl",
                },
            ), patch.object(
                report,
                "run_variant",
                return_value={
                    "variant": {},
                    "placement_summaries": [],
                    "sta_results": [],
                    "rows": [],
                },
            ), patch.object(report, "run_sta_for_summary") as run_sta:
            exit_code = report.main(
                [
                    "--artifact-base",
                    str(run_dir),
                    "--no-timestamp",
                    "--include-original-def-baseline",
                    "--design",
                    "hidden3",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_sta.assert_not_called()
        payload = report.json.loads((run_dir / "run_sta_summary.json").read_text())
        self.assertEqual(
            payload["enriched_rows"][0]["status"],
            "skipped_invalid_final_def",
        )

    def test_main_original_def_repair_skips_sta_when_validation_fails(self):
        run_dir = self.root / "report_invalid_original_repair"
        manifest = {
            "cases": [
                {
                    "design": "hidden3",
                    "def_input": str(self.root / "hidden3.def"),
                    "sdc_path": str(self.root / "hidden3.sdc"),
                }
            ]
        }
        repair_summary = {
            "design": "hidden3",
            "handoff_mode": "original_repair",
            "source": "original_repaired",
            "with_sta": False,
            "failure": None,
            "final_def_exists": True,
            "final_def": str(self.root / "repaired.def"),
            "final_def_validation": {"status": "failed", "failure": "outside DIEAREA"},
            "run_dir": str(self.root / "repair"),
            "config": {"diagnostic_param_overrides": {}, "handoff_restart_policy": "cold"},
        }

        with patch.object(report.placement_runner, "build_manifest", return_value=manifest), \
            patch.object(
                report.placement_runner,
                "asap7_tech_bundle",
                return_value={
                    "tech_lef": "tech.lef",
                    "lef_paths": [],
                    "lib_paths": [],
                    "rc_tcl": "setRC.tcl",
                },
            ), patch.object(
                report,
                "run_original_def_repair_summary",
                return_value=repair_summary,
            ), patch.object(
                report,
                "run_variant",
                return_value={
                    "variant": {},
                    "placement_summaries": [],
                    "sta_results": [],
                    "rows": [],
                },
            ), patch.object(report, "run_sta_for_summary") as run_sta:
            exit_code = report.main(
                [
                    "--artifact-base",
                    str(run_dir),
                    "--no-timestamp",
                    "--include-original-def-repair-baseline",
                    "--design",
                    "hidden3",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_sta.assert_not_called()
        payload = report.json.loads((run_dir / "run_sta_summary.json").read_text())
        self.assertEqual(
            payload["enriched_rows"][0]["baseline_family"],
            "original_def_openroad_repair_opensta",
        )
        self.assertEqual(
            payload["enriched_rows"][0]["status"],
            "skipped_invalid_final_def",
        )

    def test_render_markdown_table_contains_required_columns(self):
        rows = [
            {
                "design": "hidden3",
                "mode_config": "buffer_only_td0p2",
                "gate_count": 123,
                "WNS(ns)": -0.1,
                "TNS(ns)": -2.0,
                "total_slew_violation_difference(ns)": 0.5,
                "Total Load Capacitance Violation Difference(fF)": 2.5,
                "Total Leakage(uw)": 9.5,
                "artifact_dir": "sta/hidden3",
                "status": "sta_completed",
                "failure": "",
            }
        ]

        markdown = report.render_markdown_table(rows)

        self.assertIn(
            "| casename | design/case | baseline family | mode/config | gate count | WNS(ns) |",
            markdown,
        )
        self.assertIn(
            "| hidden3 | hidden3 | AutoDMP + Handoff + Cold | buffer_only_td0p2 | 123 | -0.1 | -2 | 0.5 | 2.5 | 9.5 | sta/hidden3 | sta_completed |  |",
            markdown,
        )

    def test_render_markdown_table_fills_missing_baseline_family_display_from_row_shape(self):
        rows = [
            {
                "design": "hidden3",
                "source": "original",
                "baseline_family": None,
                "baseline_family_display": None,
                "mode_config": "original_def",
                "gate_count": 123,
                "WNS(ns)": -0.1,
                "TNS(ns)": -2.0,
                "total_slew_violation_difference(ns)": 0.5,
                "Total Load Capacitance Violation Difference(fF)": 2.5,
                "Total Leakage(uw)": 9.5,
                "artifact_dir": "sta/hidden3",
                "status": "sta_completed",
                "failure": "",
            }
        ]

        markdown = report.render_markdown_table(rows)

        self.assertIn("Original_DEF", markdown)

    def test_main_dry_run_writes_markdown_without_running_placement(self):
        artifact_base = self.root / "artifacts"

        exit_code = report.main(
            [
                "--artifact-base",
                str(artifact_base),
                "--design",
                "hidden3",
                "--dry-run",
                "--no-timestamp",
            ]
        )

        self.assertEqual(exit_code, 0)
        summary_md = artifact_base / "run_sta_summary.md"
        summary_json = artifact_base / "run_sta_summary.json"
        self.assertTrue(summary_md.exists())
        self.assertTrue(summary_json.exists())
        self.assertIn("hidden3", summary_md.read_text())


if __name__ == "__main__":
    unittest.main()
