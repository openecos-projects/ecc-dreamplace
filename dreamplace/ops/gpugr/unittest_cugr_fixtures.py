import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dreamplace.ops.gpugr.cugr_backend import CugrGPUGR


_XPLACE_ROOT = Path(
    os.environ.get(
        "ECC_XPLACE_ROOT",
        Path(__file__).resolve().parents[3] / "thirdparty" / "xplace",
    )
).expanduser()
_SAMPLE_ROOT = _XPLACE_ROOT / "thirdparty" / "cu-gr" / "toys" / "iccad2019c" / "ispd18_sample"
_NET_NAMES = (
    "net1237",
    "net1240",
    "net1233",
    "net1236",
    "net1234",
    "net1232",
    "net1231",
    "net1239",
    "net1235",
    "net1238",
    "net1230",
)


def _native_available():
    # Do not import a native extension while unittest discovers modules.  The
    # Xplace loader must preload libpython and establish the intended cpybin
    # package first; importing here can cache a failed module in cpp_to_py and
    # poison later io_parser/gpugr imports.  The test body performs the real
    # capability check through the normal backend path.
    return (
        (_SAMPLE_ROOT / "ispd18_sample.input.def").is_file()
        and (_SAMPLE_ROOT / "ispd18_sample.input.lef").is_file()
        and any((_XPLACE_ROOT / "cpp_to_py" / "cpybin").glob("cugr*.so"))
    )


@unittest.skipUnless(_native_available(), "source-built CUGR extension or fixture is unavailable")
class CugrFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lef = _SAMPLE_ROOT / "ispd18_sample.input.lef"
        cls.design_def = _SAMPLE_ROOT / "ispd18_sample.input.def"
        cls.placedb = SimpleNamespace(
            node_names=tuple(f"inst{index}" for index in range(22)),
            net_names=_NET_NAMES,
            flat_net2pin_map=tuple(range(22)),
            flat_net2pin_start_map=tuple(range(0, 23, 2)),
            num_physical_nodes=22,
            num_pins=22,
            num_nets=11,
        )
        cls.params = SimpleNamespace(
            lef_input=[str(cls.lef)],
            result_dir="/tmp/cugr_fixture_results",
            design_name=lambda: "ispd18_sample",
            cugr_session_cache_enable=True,
        )

    def setUp(self):
        self.operator = CugrGPUGR(self.params, self.placedb)
        self.cugr = self.operator._import_cugr()
        self.cugr.reset()

    def tearDown(self):
        self.cugr.reset()

    def _run(self, **overrides):
        request = {
            "input_def": str(self.design_def),
            "out_dir": "/tmp/cugr_fixture_results",
            "design_name": "ispd18_sample",
            "route_xsize": 32,
            "route_ysize": 32,
            "threads": 8,
            "rrr_iters": 0,
            "skip_m1_route": True,
            "backend": "cugr",
            "include_route_entries": True,
            "include_l_shape_topology_pack": True,
            "topology_net_name_to_id": {name: index for index, name in enumerate(_NET_NAMES)},
            "topology_flat_net2pin_map": tuple(range(22)),
            "topology_flat_net2pin_start_map": tuple(range(0, 23, 2)),
            "topology_num_pins": 22,
            "topology_num_nets": 11,
            "save_artifacts": True,
        }
        request.update(overrides)
        return self.operator.run_gpugr(**request)

    def test_medium_fixture_returns_maps_routes_and_topology(self):
        result = self._run()
        metrics = result["metrics"]
        self.assertEqual(metrics["cugr_total_passes"], 1)
        self.assertEqual(metrics["gpugr_rrr_iters"], 0)
        self.assertEqual(metrics["cugr_threads"], 8)
        self.assertEqual(metrics["cugr_route_grid_x"], 32)
        self.assertEqual(metrics["cugr_route_grid_y"], 32)
        self.assertGreater(metrics["cugr_pass0_worker_count"], 0)
        self.assertGreater(metrics["elapsed_sec"], 0.0)
        self.assertGreater(metrics["cugr_parse_sec"], 0.0)
        self.assertGreater(metrics["cugr_route_sec"], 0.0)
        self.assertGreater(metrics["cugr_pack_sec"], 0.0)
        self.assertGreaterEqual(metrics["gr_est_shorts"], 0.0)
        self.assertEqual(len(metrics["cugr_fixed_usage_sha256"]), 64)
        self.assertEqual(len(metrics["cugr_movable_usage_sha256"]), 64)
        self.assertEqual(tuple(result["maps"]["dmd_map"].shape), (9, 32, 32))
        self.assertTrue(torch.isfinite(result["maps"]["dmd_map"]).all())
        self.assertEqual(len(result["route_entries"]), 11)
        self.assertEqual(result["l_shape_topology_pack"]["metadata"]["num_nets"], 11)
        for path in result["artifact_paths"].values():
            self.assertTrue(Path(path).is_file(), path)

    def test_medium_fixture_reuses_session_and_reproduces_maps(self):
        first = self._run(save_artifacts=False)
        second = self._run(save_artifacts=False)
        self.assertEqual(first["metrics"]["cugr_session_mode"], "full_reparse_reset")
        self.assertEqual(second["metrics"]["cugr_session_mode"], "coordinate_refresh")
        self.assertEqual(second["metrics"]["cugr_session_cache_hit"], 1)
        torch.testing.assert_close(first["maps"]["dmd_map"], second["maps"]["dmd_map"])
        self.assertEqual(first["l_shape_topology_pack"]["metadata"], second["l_shape_topology_pack"]["metadata"])

    def test_request_boundary_rejects_unsupported_or_invalid_options(self):
        with self.assertRaises(ValueError):
            self.operator.run_gpugr(
                input_def=str(self.design_def),
                route_xsize=0,
                route_ysize=32,
                threads=1,
            )
        with self.assertRaises(RuntimeError):
            self.operator.run_gpugr(
                input_def=str(self.design_def),
                route_xsize=32,
                route_ysize=32,
                guide_path="/tmp/unsupported.guide",
            )
        with self.assertRaises(TypeError):
            self.operator.run_gpugr(
                input_def=str(self.design_def),
                route_xsize=32,
                route_ysize=32,
                unexpected_option=True,
            )


if __name__ == "__main__":
    unittest.main()
