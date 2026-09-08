import os
import sys
import types
import unittest

import torch

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR  # noqa: E402


class _RouteForce:
    def __init__(self):
        self._wire = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        self._via = torch.tensor([[[0.0]], [[1.0]], [[1.0]], [[2.0]]])
        self._capacity = torch.tensor([[[4.0]], [[8.0]], [[8.0]], [[12.0]]])
        self._fixed = torch.tensor([[[1.0]], [[1.0]], [[2.0]], [[2.0]]])
        self._movable = torch.tensor([[[0.0]], [[1.0]], [[1.0]], [[2.0]]])

    def dmd_map(self):
        demand = self._wire + self._via
        return demand, self._wire, self._via

    def cap_map(self):
        return self._capacity

    def raw_wire_dmd_map(self):
        return self._wire

    def fix_usage_map(self):
        return self._fixed

    def mov_usage_map(self):
        return self._movable


class GPUGRCongestionMapTest(unittest.TestCase):
    def test_effective_directional_maps_include_vias_and_obstacle_usage(self):
        backend = XplaceGPUGR(params=None, placedb=None)
        gpdb = types.SimpleNamespace(m1direction=lambda: 0)

        maps = backend._compute_maps(_RouteForce(), gpdb, skip_m1_route=True)

        torch.testing.assert_close(maps["cg_map_h_effective_raw"], torch.tensor([[0.8]]))
        torch.testing.assert_close(maps["cg_map_v_effective_raw"], torch.tensor([[9.0 / 14.0]]))
        torch.testing.assert_close(maps["cg_map_h_effective_overflow"], torch.tensor([[0.0]]))
        torch.testing.assert_close(maps["cg_map_v_effective_overflow"], torch.tensor([[0.0]]))


if __name__ == "__main__":
    unittest.main()
