import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
OPENROAD_PYDB_EXPORT_IMPL = (
    RUNNER_PATH.parents[4]
    / "dreamplace"
    / "ops"
    / "placeio_openroad"
    / "src"
    / "openroad_pyplacedb_export_impl.cpp"
)
spec = importlib.util.spec_from_file_location("run_sta_compare", RUNNER_PATH)
sta_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sta_compare)


class StaCompareRegressionTest(unittest.TestCase):
    def test_unconstrained_outputs_keep_the_invalid_required_time_sentinel(self):
        source = OPENROAD_PYDB_EXPORT_IMPL.read_text(encoding="utf-8")

        self.assertIn("!sta->sdc()->hasOutputDelay(pin)", source)
        self.assertIn("return invalid_required", source)
        self.assertIn(
            "return period - outputDelayForPython(sta, pin, rf, time_unit, 0.0)",
            source,
        )

    def test_openroad_import_covers_fixed_timing_macros_and_bus_pin_metadata(self):
        source = OPENROAD_PYDB_EXPORT_IMPL.read_text(encoding="utf-8")

        self.assertIn("struct TimingInstRecord", source)
        self.assertIn("buildTimingInstView", source)
        self.assertIn("const auto timing_insts = buildTimingInstView", source)
        self.assertIn("liberty_cell->timingArcSets().empty()", source)
        self.assertIn("liberty_cell->hasInferedRegTimingArcs()", source)
        self.assertIn(
            "const int fixed_node_offset = static_cast<int>(movable_insts.size())",
            source,
        )
        self.assertIn("libertyPortLookupName", source)
        self.assertIn("LEF macro pins are commonly bit-blasted", source)
        self.assertIn("The physical ITerm and its net remain distinct", source)
        self.assertIn(
            "buildBasicTiming(raw_db, pins, nets, iterm_pin_ids, bterm_pin_ids, timing_insts, pydb)",
            source,
        )
        self.assertGreaterEqual(
            source.count("for (const auto& timing_inst : timing_insts)"),
            4,
        )

    def test_resolve_cases_defaults_to_all_iccad24_cases(self):
        self.assertEqual(sta_compare.resolve_cases(["all"]), list(sta_compare.ICCAD24_CASES))
        self.assertEqual(sta_compare.resolve_cases(None), list(sta_compare.ICCAD24_CASES))

    def test_resolve_cases_accepts_comma_separated_subset(self):
        self.assertEqual(
            sta_compare.resolve_cases(["aes_256,NV_NVDLA_partition_m"]),
            ["aes_256", "NV_NVDLA_partition_m"],
        )

    def test_build_summary_row_converts_autodmp_ps_and_computes_delta(self):
        openroad = {
            "status": "ok",
            "metrics": {
                "setup_wns": -0.4,
                "setup_tns": -10.0,
                "slew_violation": 0.2,
                "cap_violation": 0.3,
            },
        }
        autodmp = {
            "status": "ok",
            "endpoint_summary": {
                "python_wns": -500.0,
                "python_tns": -12000.0,
                "ieda_wns": -450.0,
                "ieda_tns": -11000.0,
            },
            "timing_summary": {
                "post_legalization": {
                    "wns": -480.0,
                    "tns": -11500.0,
                    "slew_violation": 1,
                    "cap_violation": 2,
                }
            },
        }

        row = sta_compare.build_summary_row("aes_256", openroad, autodmp)

        self.assertEqual(row["status"], "ok")
        self.assertAlmostEqual(row["autodmp_python_wns_ns"], -0.5)
        self.assertAlmostEqual(row["autodmp_python_tns_ns"], -12.0)
        self.assertAlmostEqual(row["delta_post_wns_ns_vs_openroad"], -0.08)
        self.assertAlmostEqual(row["delta_post_tns_ns_vs_openroad"], -1.5)
        self.assertAlmostEqual(row["delta_python_wns_ns_vs_openroad"], -0.1)
        self.assertAlmostEqual(row["delta_backend_tns_ns_vs_openroad"], -1.0)
        self.assertEqual(row["autodmp_slew_violation"], 1)

    def test_build_summary_row_allows_openroad_only_smoke(self):
        row = sta_compare.build_summary_row(
            "NV_NVDLA_partition_m",
            {"status": "ok", "metrics": {"setup_wns": -0.5}},
            {"status": "skipped", "failure": ""},
        )

        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["openroad_status"], "ok")
        self.assertEqual(row["autodmp_status"], "skipped")

    def test_generate_openroad_sta_tcl_reads_design_and_emits_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iccad24_benchmark"
            case_dir = root / "design" / "aes_256"
            workspace_config = case_dir / "workspace" / "config" / "dreamplace_config"
            workspace_config.mkdir(parents=True)
            (case_dir / "aes_256.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
            (case_dir / "aes_256.v").write_text("module aes_256; endmodule\n", encoding="utf-8")
            (case_dir / "aes_256.sdc").write_text("", encoding="utf-8")
            (workspace_config / "param.json").write_text("{}", encoding="utf-8")
            for path in sta_compare.asap7_inputs(root).values():
                if isinstance(path, list):
                    for item in path:
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_text("", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

            tcl = sta_compare.generate_openroad_sta_tcl(
                benchmark_root=root,
                case="aes_256",
                parasitics="placement",
                output_dir=Path(tmp) / "out",
            )

        self.assertIn("read_lef", tcl)
        self.assertIn("read_def -continue_on_errors", tcl)
        self.assertIn("read_verilog", tcl)
        self.assertIn("read_sdc", tcl)
        self.assertIn("estimate_parasitics -placement", tcl)
        self.assertIn('safe_metric "setup_wns" {worst_slack -max}', tcl)
        self.assertIn('emit_metric "slew_violation"', tcl)
        self.assertNotIn("global_route", tcl)

    def test_generate_openroad_sta_tcl_accepts_def_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iccad24_benchmark"
            case_dir = root / "design" / "aes_256"
            workspace_config = case_dir / "workspace" / "config" / "dreamplace_config"
            workspace_config.mkdir(parents=True)
            (case_dir / "aes_256.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
            (case_dir / "aes_256.v").write_text("module aes_256; endmodule\n", encoding="utf-8")
            (case_dir / "aes_256.sdc").write_text("", encoding="utf-8")
            (workspace_config / "param.json").write_text("{}", encoding="utf-8")
            override_def = Path(tmp) / "placed.def"
            override_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            for path in sta_compare.asap7_inputs(root).values():
                if isinstance(path, list):
                    for item in path:
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_text("", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

            tcl = sta_compare.generate_openroad_sta_tcl(
                benchmark_root=root,
                case="aes_256",
                parasitics="placement",
                output_dir=Path(tmp) / "out",
                def_input=override_def,
            )

        self.assertIn(f"read_def -continue_on_errors {{{override_def}}}", tcl)
        self.assertNotIn(f"read_def -continue_on_errors {{{root / 'design' / 'aes_256' / 'aes_256.def'}}}", tcl)

    def test_run_autodmp_case_accepts_def_override_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iccad24_benchmark"
            case_dir = root / "design" / "aes_256"
            workspace_config = case_dir / "workspace" / "config" / "dreamplace_config"
            workspace_config.mkdir(parents=True)
            (case_dir / "aes_256.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
            (case_dir / "aes_256.v").write_text("module aes_256; endmodule\n", encoding="utf-8")
            (case_dir / "aes_256.sdc").write_text("", encoding="utf-8")
            (workspace_config / "param.json").write_text("{}", encoding="utf-8")
            override_def = Path(tmp) / "placed.def"
            override_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
            for path in sta_compare.asap7_inputs(root).values():
                if isinstance(path, list):
                    for item in path:
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_text("", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

            result = sta_compare.run_autodmp_case(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case="aes_256",
                output_dir=Path(tmp) / "out",
                iterations=1,
                flow_kind="sta",
                def_input=override_def,
                dry_run=True,
            )

        command = result["command"]
        self.assertEqual(command[command.index("--def-input") + 1], str(override_def))

    def test_generate_openroad_sta_tcl_can_measure_pure_sta_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iccad24_benchmark"
            case_dir = root / "design" / "aes_256"
            workspace_config = case_dir / "workspace" / "config" / "dreamplace_config"
            workspace_config.mkdir(parents=True)
            (case_dir / "aes_256.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
            (case_dir / "aes_256.v").write_text("module aes_256; endmodule\n", encoding="utf-8")
            (case_dir / "aes_256.sdc").write_text("", encoding="utf-8")
            (workspace_config / "param.json").write_text("{}", encoding="utf-8")
            for path in sta_compare.asap7_inputs(root).values():
                if isinstance(path, list):
                    for item in path:
                        item.parent.mkdir(parents=True, exist_ok=True)
                        item.write_text("", encoding="utf-8")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

            tcl = sta_compare.generate_openroad_sta_tcl(
                benchmark_root=root,
                case="aes_256",
                parasitics="placement",
                output_dir=Path(tmp) / "out",
                sta_query_repeats=7,
            )

        self.assertIn("proc measure_setup_sta_query", tcl)
        self.assertIn("measure_setup_sta_query 7", tcl)
        self.assertIn("update", tcl)
        self.assertIn("worst_slack -max", tcl)
        self.assertIn("total_negative_slack -max", tcl)
        self.assertIn('emit_metric "sta_query_first_sec"', tcl)
        self.assertIn('emit_metric "sta_query_avg_sec"', tcl)


if __name__ == "__main__":
    unittest.main()
