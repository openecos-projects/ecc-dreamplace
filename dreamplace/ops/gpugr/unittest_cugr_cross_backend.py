import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dreamplace.ops.gpugr.cugr_backend import CugrGPUGR
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR


_XPLACE_ROOT = Path(
    os.environ.get(
        "ECC_XPLACE_ROOT",
        Path(__file__).resolve().parents[3] / "thirdparty" / "xplace",
    )
).expanduser()
_SAMPLE_ROOT = _XPLACE_ROOT / "thirdparty" / "cu-gr" / "toys" / "iccad2019c" / "ispd18_sample"


def _native_available():
    cpybin = _XPLACE_ROOT / "cpp_to_py" / "cpybin"
    return (
        (_SAMPLE_ROOT / "ispd18_sample.input.def").is_file()
        and (_SAMPLE_ROOT / "ispd18_sample.input.lef").is_file()
        and any(cpybin.glob("cugr*.so"))
        and any(cpybin.glob("gpugr*.so"))
        and any(cpybin.glob("io_parser*.so"))
    )


@unittest.skipUnless(_native_available(), "CUGR and CPU_PR runtimes are unavailable")
class CugrCrossBackendTest(unittest.TestCase):
    def test_cugr_and_cpu_pr_share_grid_and_legality_contract(self):
        lef = str(_SAMPLE_ROOT / "ispd18_sample.input.lef")
        design_def = str(_SAMPLE_ROOT / "ispd18_sample.input.def")
        params = SimpleNamespace(
            lef_input=[lef],
            result_dir="/tmp/cugr_cross_backend",
            design_name=lambda: "ispd18_sample",
            cugr_session_cache_enable=True,
        )

        cpu_pr = XplaceGPUGR(params, None).run_gpugr(
            input_def=design_def,
            out_dir=params.result_dir,
            design_name="ispd18_sample",
            route_xsize=32,
            route_ysize=32,
            threads=1,
            rrr_iters=0,
            skip_m1_route=True,
            backend="cpu_pr",
            include_route_entries=True,
        )
        cugr = CugrGPUGR(params, None).run_gpugr(
            input_def=design_def,
            out_dir=params.result_dir,
            design_name="ispd18_sample",
            route_xsize=32,
            route_ysize=32,
            threads=1,
            rrr_iters=0,
            skip_m1_route=True,
            backend="cugr",
            include_route_entries=True,
        )

        for result in (cpu_pr, cugr):
            maps = result["maps"]
            self.assertEqual(tuple(maps["capacity_map"].shape), (9, 32, 32))
            self.assertEqual(tuple(maps["dmd_map"].shape), (9, 32, 32))
            self.assertTrue(torch.isfinite(maps["capacity_map"]).all())
            self.assertTrue(torch.isfinite(maps["dmd_map"]).all())
            self.assertTrue(torch.all(maps["capacity_map"] >= 0))
            self.assertGreaterEqual(len(result["route_entries"]), 1)

            for net in result["route_entries"]:
                for entry in net["entries"]:
                    self.assertGreaterEqual(entry["layer_idx"], 0)
                    self.assertLess(entry["layer_idx"], 9)
                    for coordinate in (
                        entry["grid_x1"],
                        entry["grid_y1"],
                        entry["grid_x2"],
                        entry["grid_y2"],
                    ):
                        self.assertGreaterEqual(coordinate, 0)
                        self.assertLess(coordinate, 32)

        self.assertEqual(cpu_pr["metrics"]["gpugr_backend"], "cpu_pr")
        self.assertEqual(cugr["metrics"]["gpugr_backend"], "cugr")
        self.assertEqual(cugr["metrics"]["cugr_total_passes"], 1)
        self.assertEqual(cugr["metrics"]["gpugr_rrr_iters"], 0)


if __name__ == "__main__":
    unittest.main()
