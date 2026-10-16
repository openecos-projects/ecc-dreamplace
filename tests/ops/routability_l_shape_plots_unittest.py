"""Area-mode diagnostics remain usable with planar and split H/V maps."""

from types import SimpleNamespace
import unittest

import torch

from dreamplace.ops.routability.l_shape_density_plots import _build_l_shape_electric_plot_maps


class LShapeDensityPlotTest(unittest.TestCase):
    def test_area_density_payload(self):
        density = torch.tensor([[1., 3.], [2., 2.]])
        supply = torch.full((2, 2), 2.)
        demand = torch.tensor([[3., 1.], [1., 3.]])
        for split in (False, True):
            with self.subTest(split=split):
                count = 2 if split else 1
                op = SimpleNamespace(
                    cached_density_map=count * density,
                    density_op=SimpleNamespace(
                        target_density=count * supply,
                        target_demand=count * demand,
                        area_per_track=torch.tensor(1.),
                        bin_size_x=1., bin_size_y=1.,
                    ),
                )
                if split:
                    op.cached_density_map_h = density
                    op.cached_density_map_v = density
                    for suffix in ("h", "v"):
                        setattr(op.density_op, f"target_density_{suffix}", supply)
                        setattr(op.density_op, f"target_demand_{suffix}", demand)
                payload = _build_l_shape_electric_plot_maps(op)
                actual = {
                    key: payload[key].tolist() for key in (
                        "surrogate_rho_map", "surrogate_overflow_map", "ggr_overflow_map"
                    )
                }
                self.assertEqual(actual, {
                    "surrogate_rho_map": [[-count, count], [0., 0.]],
                    "surrogate_overflow_map": [[0., count], [0., 0.]],
                    "ggr_overflow_map": [[count, 0.], [0., count]],
                })


if __name__ == "__main__":
    unittest.main()
