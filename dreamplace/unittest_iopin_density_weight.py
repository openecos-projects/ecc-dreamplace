import os
import sys
import types
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.electric_potential.electric_overflow import ElectricOverflow  # noqa: E402


def _install_ieda_stubs():
    tools = types.ModuleType("tools")
    ieda = types.ModuleType("tools.iEDA")
    module = types.ModuleType("tools.iEDA.module")
    sta = types.ModuleType("tools.iEDA.module.sta")
    gpugr = types.ModuleType("tools.iEDA.module.gpugr")

    class IEDASta:
        pass

    class IEDAGPUGR:
        pass

    sta.IEDASta = IEDASta
    gpugr.IEDAGPUGR = IEDAGPUGR
    sys.modules.setdefault("tools", tools)
    sys.modules.setdefault("tools.iEDA", ieda)
    sys.modules.setdefault("tools.iEDA.module", module)
    sys.modules.setdefault("tools.iEDA.module.sta", sta)
    sys.modules.setdefault("tools.iEDA.module.gpugr", gpugr)


def _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=1):
    dtype = torch.float64
    node_size_x = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=dtype)
    node_size_y = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=dtype)
    bin_center_x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    bin_center_y = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    return ElectricOverflow(
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        bin_center_x=bin_center_x,
        bin_center_y=bin_center_y,
        target_density=0.25,
        xl=0.0,
        yl=0.0,
        xh=4.0,
        yh=4.0,
        bin_size_x=1.0,
        bin_size_y=1.0,
        num_movable_nodes=1,
        num_terminals=1,
        num_filler_nodes=1,
        padding=0,
        deterministic_flag=0,
        sorted_node_map=torch.tensor([0], dtype=torch.int32),
        num_terminal_NIs=num_terminal_NIs,
        iopin_density_weight=iopin_density_weight,
    )


def _make_pos():
    dtype = torch.float64
    x = torch.tensor([0.0, 0.0, 2.0, 3.0], dtype=dtype)
    y = torch.tensor([0.0, 0.0, 2.0, 3.0], dtype=dtype)
    return torch.cat([x, y]).contiguous()


class IOPinDensityWeightMapTest(unittest.TestCase):
    def test_zero_weight_matches_fixed_terminal_density(self):
        pos = _make_pos()
        baseline = _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=0)
        weighted_off = _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=1)

        baseline.compute_initial_density_map(pos)
        weighted_off.compute_initial_density_map(pos)

        self.assertTrue(
            torch.allclose(baseline.initial_density_map, weighted_off.initial_density_map)
        )

    def test_weighted_delta_uses_raw_terminal_ni_density(self):
        pos = _make_pos()
        weighted_off = _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=1)
        weighted_on = _make_density_op(iopin_density_weight=3.0, num_terminal_NIs=1)
        raw_iopin = _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=0)

        weighted_off.compute_initial_density_map(pos)
        weighted_on.compute_initial_density_map(pos)

        iopin_start = weighted_on.num_movable_nodes + weighted_on.num_terminals
        packed_pos = torch.cat(
            [
                pos[iopin_start:iopin_start + weighted_on.num_terminal_NIs],
                pos[pos.numel() // 2 + iopin_start:pos.numel() // 2 + iopin_start + weighted_on.num_terminal_NIs],
            ]
        ).contiguous()
        raw_iopin.node_size_x = weighted_on.node_size_x[iopin_start:iopin_start + weighted_on.num_terminal_NIs]
        raw_iopin.node_size_y = weighted_on.node_size_y[iopin_start:iopin_start + weighted_on.num_terminal_NIs]
        raw_iopin.num_movable_nodes = 0
        raw_iopin.num_terminals = weighted_on.num_terminal_NIs
        raw_iopin.compute_initial_density_map(packed_pos)
        raw_iopin.initial_density_map.div_(raw_iopin.target_density)

        delta = weighted_on.initial_density_map - weighted_off.initial_density_map

        self.assertTrue(torch.count_nonzero(delta).item() > 0)
        self.assertTrue(torch.allclose(delta, 3.0 * raw_iopin.initial_density_map))
        self.assertFalse(
            torch.allclose(
                delta,
                3.0 * weighted_on.target_density * raw_iopin.initial_density_map,
            )
        )

    def test_weighted_delta_uses_terminal_ni_slice_not_movable_slice(self):
        pos = _make_pos()
        weighted_off = _make_density_op(iopin_density_weight=0.0, num_terminal_NIs=1)
        weighted_on = _make_density_op(iopin_density_weight=3.0, num_terminal_NIs=1)

        weighted_off.compute_initial_density_map(pos)
        weighted_on.compute_initial_density_map(pos)

        delta = weighted_on.initial_density_map - weighted_off.initial_density_map
        self.assertEqual(delta[2, 2].item(), 3.0)
        self.assertEqual(delta[0, 0].item(), 0.0)

    def test_logging_reports_enabled_iopin_density(self):
        pos = _make_pos()
        op = _make_density_op(iopin_density_weight=3.0, num_terminal_NIs=1)

        with self.assertLogs(level="INFO") as logs:
            op.compute_initial_density_map(pos)

        self.assertIn(
            "I/O pin density increment enabled: weight=3.0, num_terminal_NIs=1",
            "\n".join(logs.output),
        )

    def test_region_design_builders_force_effective_weight_to_zero(self):
        _install_ieda_stubs()
        from dreamplace import PlaceObj as place_obj_module

        dtype = torch.float64

        class DataCollections:
            node_size_x = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            node_size_y = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            target_density = torch.tensor(0.5, dtype=dtype)
            sorted_node_map = torch.tensor([0], dtype=torch.int32)
            movable_macro_mask = None
            node2fence_region_map = torch.tensor([0], dtype=torch.int32)

            def bin_center_x_padded(self, placedb, padding, num_bins_x):
                return torch.tensor([0.5, 1.5], dtype=dtype)

            def bin_center_y_padded(self, placedb, padding, num_bins_y):
                return torch.tensor([0.5, 1.5], dtype=dtype)

        placedb = types.SimpleNamespace(
            xl=0.0,
            yl=0.0,
            xh=2.0,
            yh=2.0,
            row_height=1.0,
            node_size_x=torch.tensor([1.0, 1.0, 1.0], dtype=dtype).numpy(),
            node_size_y=torch.tensor([1.0, 1.0, 1.0], dtype=dtype).numpy(),
            num_movable_nodes=1,
            num_terminals=1,
            num_terminal_NIs=1,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[object()],
            target_density_fence_region=[0.5, 0.5],
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=3.0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)

        overflow = place_obj.build_electric_overflow(
            params, placedb, DataCollections(), 2, 2
        )
        potential = place_obj.build_electric_potential(
            params, placedb, DataCollections(), 2, 2, name="test"
        )

        self.assertEqual(overflow.iopin_density_weight, 0.0)
        self.assertEqual(potential.iopin_density_weight, 0.0)

    def test_non_region_builders_pass_effective_weight_and_terminal_ni_count(self):
        _install_ieda_stubs()
        from dreamplace import PlaceObj as place_obj_module

        dtype = torch.float64

        class DataCollections:
            node_size_x = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            node_size_y = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            target_density = torch.tensor(0.5, dtype=dtype)
            sorted_node_map = torch.tensor([0], dtype=torch.int32)
            movable_macro_mask = None
            node2fence_region_map = None

            def bin_center_x_padded(self, placedb, padding, num_bins_x):
                return torch.tensor([0.5, 1.5], dtype=dtype)

            def bin_center_y_padded(self, placedb, padding, num_bins_y):
                return torch.tensor([0.5, 1.5], dtype=dtype)

        placedb = types.SimpleNamespace(
            xl=0.0,
            yl=0.0,
            xh=2.0,
            yh=2.0,
            row_height=1.0,
            node_size_x=torch.tensor([1.0, 1.0, 1.0], dtype=dtype).numpy(),
            node_size_y=torch.tensor([1.0, 1.0, 1.0], dtype=dtype).numpy(),
            num_movable_nodes=1,
            num_terminals=1,
            num_terminal_NIs=1,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[],
            target_density_fence_region=[],
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=3.0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)

        overflow = place_obj.build_electric_overflow(
            params, placedb, DataCollections(), 2, 2
        )
        potential = place_obj.build_electric_potential(
            params, placedb, DataCollections(), 2, 2, name="test"
        )

        self.assertEqual(overflow.iopin_density_weight, 3.0)
        self.assertEqual(potential.iopin_density_weight, 3.0)
        self.assertEqual(overflow.num_terminal_NIs, 1)
        self.assertEqual(potential.num_terminal_NIs, 1)


if __name__ == "__main__":
    unittest.main()
