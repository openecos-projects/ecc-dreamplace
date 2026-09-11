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
    def __init__(self, routing_layer_range=(0, 3)):
        self._wire = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        self._via = torch.tensor([[[0.0]], [[1.0]], [[1.0]], [[2.0]]])
        self._capacity = torch.tensor([[[4.0]], [[8.0]], [[8.0]], [[12.0]]])
        self._fixed = torch.tensor([[[1.0]], [[1.0]], [[2.0]], [[2.0]]])
        self._movable = torch.tensor([[[0.0]], [[1.0]], [[1.0]], [[2.0]]])
        self._routing_layer_range = routing_layer_range

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

    def routing_layer_range(self):
        return self._routing_layer_range

    def routing_layer_names(self):
        return ["MET1", "MET2", "MET3", "MET4"]


class GPUGRCongestionMapTest(unittest.TestCase):
    def test_effective_directional_maps_include_vias_and_obstacle_usage(self):
        backend = XplaceGPUGR(params=None, placedb=None)
        gpdb = types.SimpleNamespace(m1direction=lambda: 0)

        maps = backend._compute_maps(_RouteForce(), gpdb, skip_m1_route=True)

        torch.testing.assert_close(maps["cg_map_h_effective_raw"], torch.tensor([[0.8]]))
        torch.testing.assert_close(maps["cg_map_v_effective_raw"], torch.tensor([[9.0 / 14.0]]))
        torch.testing.assert_close(maps["cg_map_h_effective_overflow"], torch.tensor([[0.0]]))
        torch.testing.assert_close(maps["cg_map_v_effective_overflow"], torch.tensor([[0.0]]))

    def test_restricted_routing_layer_window_excludes_disabled_layers(self):
        backend = XplaceGPUGR(params=None, placedb=None)
        gpdb = types.SimpleNamespace(m1direction=lambda: 0)

        maps = backend._compute_maps(
            _RouteForce(routing_layer_range=(1, 2)), gpdb, skip_m1_route=False
        )

        torch.testing.assert_close(
            maps["dmd_map"], torch.tensor([[[0.0]], [[3.0]], [[4.0]], [[0.0]]])
        )
        torch.testing.assert_close(
            maps["capacity_map"], torch.tensor([[[0.0]], [[8.0]], [[8.0]], [[0.0]]])
        )
        torch.testing.assert_close(maps["cg_map_union_raw"], torch.tensor([[7.0 / 16.0]]))
        torch.testing.assert_close(maps["cg_map_h_raw"], torch.tensor([[4.0 / 8.0]]))
        torch.testing.assert_close(maps["cg_map_v_raw"], torch.tensor([[3.0 / 8.0]]))

    def test_empty_directional_window_returns_zero_instead_of_nan(self):
        backend = XplaceGPUGR(params=None, placedb=None)
        gpdb = types.SimpleNamespace(m1direction=lambda: 0)

        maps = backend._compute_maps(
            _RouteForce(routing_layer_range=(2, 2)), gpdb, skip_m1_route=False
        )

        torch.testing.assert_close(maps["cg_map_v_raw"], torch.zeros((1, 1)))
        self.assertTrue(torch.isfinite(maps["cg_map_v_raw"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_maps_normalize_cpu_obstacle_maps_to_route_device(self):
        routeforce = _RouteForce()
        routeforce._wire = routeforce._wire.cuda()
        routeforce._via = routeforce._via.cuda()
        routeforce._capacity = routeforce._capacity.cuda()

        backend = XplaceGPUGR(params=None, placedb=None)
        gpdb = types.SimpleNamespace(m1direction=lambda: 0)
        maps = backend._compute_maps(routeforce, gpdb, skip_m1_route=True)

        self.assertTrue(all(value.device.type == "cuda" for value in maps.values()))
        torch.testing.assert_close(
            maps["cg_map_h_effective_raw"],
            torch.tensor([[0.8]], device="cuda"),
        )
        torch.testing.assert_close(
            maps["cg_map_v_effective_raw"],
            torch.tensor([[9.0 / 14.0]], device="cuda"),
        )


if __name__ == "__main__":
    unittest.main()
