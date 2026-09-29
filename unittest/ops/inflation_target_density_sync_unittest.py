import os
import sys
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import torch

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)


class InflationTargetDensitySyncTest(unittest.TestCase):
    def test_electric_potential_refreshes_fixed_map_after_target_density_change(self):
        import types

        from dreamplace import PlaceObj as place_obj_module

        dtype = torch.float64

        class DataCollections:
            def __init__(self):
                self.node_size_x = torch.tensor([2.0, 2.0, 1.0], dtype=dtype)
                self.node_size_y = torch.tensor([2.0, 2.0, 1.0], dtype=dtype)
                self.target_density = torch.tensor([0.5], dtype=dtype)
                self.sorted_node_map = torch.tensor([0], dtype=torch.int32)
                self.movable_macro_mask = torch.tensor([True])
                self.node2fence_region_map = None

            def bin_center_x_padded(self, placedb, padding, num_bins_x):
                return torch.arange(0.5, num_bins_x, dtype=dtype)

            def bin_center_y_padded(self, placedb, padding, num_bins_y):
                return torch.arange(0.5, num_bins_y, dtype=dtype)

        data_collections = DataCollections()
        placedb = types.SimpleNamespace(
            xl=0.0,
            yl=0.0,
            xh=4.0,
            yh=4.0,
            row_height=1.0,
            node_size_x=data_collections.node_size_x.numpy(),
            node_size_y=data_collections.node_size_y.numpy(),
            num_movable_nodes=1,
            num_terminals=1,
            num_terminal_NIs=0,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[],
            target_density_fence_region=[],
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=0.0,
            ieda_m2_pg_rail_blockage_flag=0,
            m2_pg_rail_density_weight=0.0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)
        potential = place_obj.build_electric_potential(
            params,
            placedb,
            data_collections,
            num_bins_x=4,
            num_bins_y=4,
            name="target-density-sync-test",
        )

        self.assertIs(potential.target_density, data_collections.target_density)
        self.assertAlmostEqual(float(potential.ratio[0]), 0.5)

        pos = torch.tensor(
            [0.0, 2.0, 3.0, 0.0, 2.0, 3.0],
            dtype=dtype,
        )
        potential(pos)
        fixed_map_before = potential.initial_density_map.clone()

        data_collections.target_density.fill_(0.75)
        potential.reset()
        potential(pos)

        self.assertAlmostEqual(float(potential.ratio[0]), 0.75)
        self.assertTrue(torch.allclose(
            potential.initial_density_map,
            fixed_map_before * 1.5,
        ))
        self.assertTrue(
            torch.allclose(
                potential.initial_density_map,
                potential.m2_pg_rail_fixed_without_rail_raw_map * 0.75,
            )
        )

        potential.reset()
        pos.requires_grad_(True)
        potential(pos).sum().backward()
        self.assertIsNotNone(pos.grad)


if __name__ == "__main__":
    unittest.main()
