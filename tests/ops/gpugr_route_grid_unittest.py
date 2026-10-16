import unittest
from types import SimpleNamespace

from dreamplace.ops.routability.gpugr_context import compute_route_grid_like_xplace


class GpugrRouteGridTest(unittest.TestCase):
    def test_explicit_route_grid_is_preserved_when_auto_adjust_is_disabled(self):
        params = SimpleNamespace(
            route_num_bins_x=512,
            route_num_bins_y=512,
            auto_adjust_bins=0,
            num_bins_x=32,
            num_bins_y=32,
        )
        placedb = SimpleNamespace(
            num_bins_y=32,
            xl=0.0,
            yl=0.0,
            xh=100.0,
            yh=100.0,
        )

        self.assertEqual(
            compute_route_grid_like_xplace(params, placedb),
            (512, 512),
        )

    def test_auto_adjust_keeps_xplace_rule(self):
        params = SimpleNamespace(
            route_num_bins_x=512,
            route_num_bins_y=512,
            auto_adjust_bins=1,
            num_bins_x=32,
            num_bins_y=32,
        )
        placedb = SimpleNamespace(
            num_bins_y=32,
            xl=0.0,
            yl=0.0,
            xh=200.0,
            yh=100.0,
        )

        self.assertEqual(
            compute_route_grid_like_xplace(params, placedb),
            (64, 32),
        )


if __name__ == "__main__":
    unittest.main()
