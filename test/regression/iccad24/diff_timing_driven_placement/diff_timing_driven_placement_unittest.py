import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("run_diff_tdp", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DiffTimingDrivenPlacementRegressionTest(unittest.TestCase):
    def test_build_placer_command_keeps_placement_only_timing_flags_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                output_def=root / "final.def",
                mode="placement_only",
                iterations=30,
                gated_overflow=0.2,
                refresh_interval=10,
            )

        self.assertIn("--flow-kind", command)
        self.assertIn("placement", command)
        self.assertIn("--output-def", command)
        self.assertIn(str(root / "final.def"), command)
        self.assertNotIn("--with-sta", command)
        self.assertNotIn("--diff-timing-driven-placement", command)

    def test_build_placer_command_adds_gated_diff_tdp_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                output_def=root / "final.def",
                mode="diff_tdp_gated",
                iterations=30,
                gated_overflow=0.2,
                refresh_interval=10,
            )

        self.assertIn("--with-sta", command)
        self.assertIn("--diff-timing-driven-placement", command)
        threshold_index = command.index("--timing-topology-enable-overflow-threshold")
        self.assertEqual(command[threshold_index + 1], "0.2")

    def test_build_placer_command_uses_ungated_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                output_def=root / "final.def",
                mode="diff_tdp_ungated",
                iterations=30,
                gated_overflow=0.2,
                refresh_interval=10,
            )

        threshold_index = command.index("--timing-topology-enable-overflow-threshold")
        self.assertEqual(command[threshold_index + 1], "1.1")

    def test_build_placer_command_appends_raw_placer_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = "NV_NVDLA_partition_m"
            self._write_case(root, case)
            command = runner.build_placer_command(
                python_bin=Path("/usr/bin/python"),
                benchmark_root=root,
                case=case,
                result_dir=root / "result",
                output_def=root / "final.def",
                mode="diff_tdp_gated",
                iterations=30,
                gated_overflow=0.2,
                refresh_interval=10,
                placer_args=[
                    "--timing-wns-coeff",
                    "0.002",
                    "--timing-tns-coeff",
                    "0.00002",
                ],
            )

        self.assertEqual(command[-4:], [
            "--timing-wns-coeff",
            "0.002",
            "--timing-tns-coeff",
            "0.00002",
        ])

    def test_parse_placement_log_extracts_final_metrics_and_timing_events(self):
        parsed = runner.parse_placement_log(
            """
            [INFO] iteration    0, Obj 1.000000E+00, WL 2.000000E+00, Overflow 4.000000E-01
            [INFO] Diff TDP enabled timing objective at iteration=20 overflow=0.190000 threshold=0.200000 interval=10 steiner_update_ms=12.0
            [INFO] iteration   20, Obj 3.000000E+00, WL 4.000000E+00, Overflow 1.900000E-01, WNS -5.0E-01, TNS -1.430000E+02, TimingObj 2.0E+00
            """
        )

        self.assertEqual(parsed["final_metrics"]["iteration"], 20)
        self.assertAlmostEqual(parsed["final_metrics"]["overflow"], 0.19)
        self.assertAlmostEqual(parsed["final_metrics"]["tns"], -143.0)
        self.assertEqual(len(parsed["timing_events"]), 1)
        self.assertEqual(parsed["timing_events"][0]["iteration"], 20)
        self.assertAlmostEqual(parsed["timing_events"][0]["overflow"], 0.19)

    def test_parse_placement_log_extracts_gradient_net_weight_events(self):
        parsed = runner.parse_placement_log(
            """
            [INFO] Diff TDP gradient net-weight update at iteration=10 overflow=0.196471 non_unit_nets=8765 max_weight=8.000 timing_loss=118.959137
            [INFO] iteration   10, Obj 3.000000E+00, WL 4.000000E+00, Overflow 1.900000E-01, WNS -5.0E-01, TNS -1.430000E+02, TimingObj 2.0E+00
            """
        )

        self.assertEqual(len(parsed["timing_events"]), 1)
        self.assertEqual(parsed["timing_events"][0]["iteration"], 10)
        self.assertAlmostEqual(parsed["timing_events"][0]["overflow"], 0.196471)
        self.assertEqual(parsed["timing_events"][0]["carrier"], "gradient_net_weight")

    @staticmethod
    def _write_case(root, case):
        case_dir = root / "design" / case
        workspace = case_dir / "workspace" / "config" / "dreamplace_config"
        workspace.mkdir(parents=True)
        (workspace / "param.json").write_text("{}", encoding="utf-8")
        for suffix in (".def", ".v", ".sdc"):
            (case_dir / f"{case}{suffix}").write_text("", encoding="utf-8")
        asap7 = root / "ASAP7"
        for subdir in ("lef", "lib"):
            (asap7 / subdir).mkdir(parents=True, exist_ok=True)
        (asap7 / "setRC.tcl").write_text("", encoding="utf-8")
        (asap7 / "lef" / "asap7_tech_1x_201209.lef").write_text("", encoding="utf-8")
        for lef in (
            "asap7sc7p5t_27_R_1x_201211.lef",
            "sram_asap7_16x256_1rw.lef",
            "sram_asap7_32x256_1rw.lef",
            "sram_asap7_64x256_1rw.lef",
            "sram_asap7_64x64_1rw.lef",
        ):
            (asap7 / "lef" / lef).write_text("", encoding="utf-8")
        for lib in (
            "asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
            "asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
            "sram_asap7_16x256_1rw.lib",
            "sram_asap7_32x256_1rw.lib",
            "sram_asap7_64x256_1rw.lib",
            "sram_asap7_64x64_1rw.lib",
        ):
            (asap7 / "lib" / lib).write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
