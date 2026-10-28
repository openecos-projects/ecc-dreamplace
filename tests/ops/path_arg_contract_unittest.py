"""Contract tests for filesystem-path arguments on the pybind boundary.

Every converted binding must accept ``str``, ``pathlib.Path`` and a custom
``os.PathLike`` object, and must reject ``None`` and non-path values with
``TypeError``. The behaviour of a path argument must be identical for all
accepted spellings.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dreamplace.ops.draw_place import draw_place_cpp
from dreamplace.ops.rmst_wl import rmst_wl_cpp
from dreamplace.ops.steiner_topo import steiner_topo_cpp
from dreamplace.ops.steiner_topo.steiner_topo import (
    _FLUTE_POST_FILE,
    _FLUTE_POWV_FILE,
)


class _CustomPathLike:
    """Minimal os.PathLike implementation that is not a pathlib.Path."""

    def __init__(self, path):
        self._path = os.fspath(path)

    def __fspath__(self):
        return self._path


PATH_LIKE_FACTORIES = (str, Path, _CustomPathLike)


def _draw_place_arguments(num_movable_nodes=2):
    dtype = np.float32
    xx = np.array([1.0, 2.0]).astype(dtype)
    yy = np.array([3.0, 1.5]).astype(dtype)
    node_size_x = np.array([0.5, 1.0]).astype(dtype)
    node_size_y = np.array([1.0, 1.0]).astype(dtype)
    pin2node_map = np.array([0, 0, 1, 1]).astype(np.int32)
    pin_offset_x = np.array([0.1, 0.3, 0.2, 0.6]).astype(dtype)
    pin_offset_y = np.array([0.1, 0.8, 0.2, 0.6]).astype(dtype)
    return dict(
        pos=torch.from_numpy(np.concatenate([xx, yy])),
        node_size_x=torch.from_numpy(node_size_x),
        node_size_y=torch.from_numpy(node_size_y),
        pin_offset_x=torch.from_numpy(pin_offset_x),
        pin_offset_y=torch.from_numpy(pin_offset_y),
        pin2node_map=torch.from_numpy(pin2node_map),
        xl=1.0,
        yl=1.0,
        xh=5.0,
        yh=5.0,
        site_width=1.0,
        row_height=2.0,
        bin_size_x=2.0,
        bin_size_y=2.0,
        num_movable_nodes=num_movable_nodes,
        num_filler_nodes=0,
        show_fillers=True,
    )


class DrawPlacePathArgumentTest(unittest.TestCase):
    def test_accepts_path_like_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            for factory in PATH_LIKE_FACTORIES:
                with self.subTest(factory=factory.__name__):
                    filename = factory(Path(directory) / "placement.png")
                    self.assertTrue(
                        draw_place_cpp.forward(
                            filename=filename, **_draw_place_arguments()
                        )
                    )
                    self.assertTrue((Path(directory) / "placement.png").is_file())

    def test_rejects_non_path_filename(self):
        arguments = _draw_place_arguments()
        for bad_value in (None, 5):
            with self.subTest(value=bad_value):
                with self.assertRaises(TypeError):
                    draw_place_cpp.forward(filename=bad_value, **arguments)


def _two_pin_net_arguments():
    # One net with two pins; enough for the topology/wirelength launchers.
    return dict(
        pos=torch.tensor([0.0, 0.0, 1000.0, 1000.0]),
        flat_netpin=torch.tensor([0, 1], dtype=torch.int32),
        netpin_start=torch.tensor([0, 2], dtype=torch.int32),
        # rmst_wl masks nets with degree >= ignore_net_degree, so leave room.
        ignore_net_degree=3,
    )


class SteinerTopoPathArgumentTest(unittest.TestCase):
    def test_accepts_path_like_lut_paths(self):
        arguments = _two_pin_net_arguments()
        for factory in PATH_LIKE_FACTORIES:
            with self.subTest(factory=factory.__name__):
                outputs = steiner_topo_cpp.build_tree(
                    arguments["pos"],
                    arguments["flat_netpin"],
                    arguments["netpin_start"],
                    arguments["ignore_net_degree"],
                    factory(_FLUTE_POWV_FILE),
                    factory(_FLUTE_POST_FILE),
                )
                self.assertTrue(len(outputs) > 0)

    def test_rejects_non_path_lut_paths(self):
        arguments = _two_pin_net_arguments()
        for bad_value in (None, 5):
            with self.subTest(value=bad_value):
                with self.assertRaises(TypeError):
                    steiner_topo_cpp.build_tree(
                        arguments["pos"],
                        arguments["flat_netpin"],
                        arguments["netpin_start"],
                        arguments["ignore_net_degree"],
                        bad_value,
                        _FLUTE_POST_FILE,
                    )


def _three_pin_net_arguments():
    # flute returns zero wirelength for two-pin nets, so use a three-pin net.
    return dict(
        pos=torch.tensor([0.0, 0.0, 1000.0, 1000.0, 0.0, 1000.0]),
        flat_netpin=torch.tensor([0, 1, 2], dtype=torch.int32),
        netpin_start=torch.tensor([0, 3], dtype=torch.int32),
        ignore_net_degree=10,
    )


class RmstWlPathArgumentTest(unittest.TestCase):
    def test_accepts_path_like_lut_paths(self):
        arguments = _three_pin_net_arguments()
        for factory in PATH_LIKE_FACTORIES:
            with self.subTest(factory=factory.__name__):
                rmst_wl = torch.zeros(1)
                ret = rmst_wl_cpp.forward(
                    arguments["pos"],
                    arguments["flat_netpin"],
                    arguments["netpin_start"],
                    arguments["ignore_net_degree"],
                    True,
                    factory(_FLUTE_POWV_FILE),
                    factory(_FLUTE_POST_FILE),
                    rmst_wl,
                )
                self.assertEqual(ret, 0)
                self.assertGreater(float(rmst_wl[0]), 0.0)

    def test_rejects_non_path_lut_paths(self):
        arguments = _three_pin_net_arguments()
        for bad_value in (None, 5):
            with self.subTest(value=bad_value):
                with self.assertRaises(TypeError):
                    rmst_wl_cpp.forward(
                        arguments["pos"],
                        arguments["flat_netpin"],
                        arguments["netpin_start"],
                        arguments["ignore_net_degree"],
                        True,
                        bad_value,
                        _FLUTE_POST_FILE,
                        torch.zeros(1),
                    )


if __name__ == "__main__":
    unittest.main()
