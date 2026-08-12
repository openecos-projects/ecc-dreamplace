import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from dreamplace.ops.steiner_topo import steiner_topo


class SteinerTopoTest(unittest.TestCase):
    def setUp(self):
        self.flat_net2pin_map = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        self.flat_net2pin_start_map = torch.tensor([0, 4], dtype=torch.int32)
        self.pos = torch.tensor([0.0, 10.0, 20.0, 30.0, 0.0, 20.0, 10.0, 30.0])

    def test_rebuild_tree_passes_packaged_flute_lut_files(self):
        outputs = tuple(torch.empty(0) for _ in range(12))
        operator = steiner_topo.SteinerTopo(
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
        )

        with mock.patch.object(
            steiner_topo.steiner_topo_cpp,
            "build_tree",
            return_value=outputs,
        ) as build_tree:
            operator.rebuild_tree(self.pos)

        args = build_tree.call_args.args
        self.assertEqual(
            args[4:],
            (
                str(steiner_topo._FLUTE_POWV_FILE),
                str(steiner_topo._FLUTE_POST_FILE),
            ),
        )
        self.assertTrue(Path(args[4]).is_file())
        self.assertTrue(Path(args[5]).is_file())

    def test_build_tree_rejects_missing_flute_lut_files(self):
        with TemporaryDirectory() as temp_dir:
            missing_lut = str(Path(temp_dir) / "missing.dat")
            cases = (
                (missing_lut, str(steiner_topo._FLUTE_POST_FILE), "POWV"),
                (str(steiner_topo._FLUTE_POWV_FILE), missing_lut, "POST"),
            )

            for powv_file, post_file, lut_name in cases:
                with self.subTest(lut_name=lut_name):
                    with self.assertRaisesRegex(
                        RuntimeError, f"Flute {lut_name} LUT file does not exist"
                    ):
                        steiner_topo.steiner_topo_cpp.build_tree(
                            self.pos,
                            self.flat_net2pin_map,
                            self.flat_net2pin_start_map,
                            self.flat_net2pin_map.numel(),
                            powv_file,
                            post_file,
                        )


if __name__ == "__main__":
    unittest.main()
