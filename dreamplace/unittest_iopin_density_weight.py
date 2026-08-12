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


def _make_density_op(
    iopin_density_weight=0.0,
    num_terminal_NIs=1,
    m2_pg_rail_density_boxes=None,
    m2_pg_rail_density_weight=1.0,
):
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
        m2_pg_rail_density_boxes=m2_pg_rail_density_boxes,
        m2_pg_rail_density_weight=m2_pg_rail_density_weight,
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


class M2PgRailSoftDensityMapTest(unittest.TestCase):
    def test_zero_rail_density_weight_matches_fixed_without_rail_map(self):
        pos = _make_pos()
        baseline = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=None,
        )
        rail_weight_zero = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=torch.tensor(
                [[1.0, 1.0, 3.0, 3.0]], dtype=torch.float64
            ),
            m2_pg_rail_density_weight=0.0,
        )

        baseline.compute_initial_density_map(pos)
        rail_weight_zero.compute_initial_density_map(pos)

        self.assertTrue(
            torch.allclose(
                baseline.initial_density_map,
                rail_weight_zero.initial_density_map,
            )
        )
        self.assertTrue(
            torch.allclose(
                rail_weight_zero.m2_pg_rail_raw_delta_map,
                torch.zeros_like(rail_weight_zero.m2_pg_rail_raw_delta_map),
            )
        )

    def test_empty_rail_boxes_match_fixed_without_rail_map(self):
        pos = _make_pos()
        baseline = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=None,
        )
        empty_rail = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=torch.empty((0, 4), dtype=torch.float64),
        )

        baseline.compute_initial_density_map(pos)
        empty_rail.compute_initial_density_map(pos)

        self.assertTrue(
            torch.allclose(baseline.initial_density_map, empty_rail.initial_density_map)
        )
        self.assertTrue(
            torch.allclose(
                empty_rail.m2_pg_rail_combined_raw_map,
                empty_rail.m2_pg_rail_fixed_without_rail_raw_map,
            )
        )
        self.assertTrue(
            torch.allclose(
                empty_rail.m2_pg_rail_raw_delta_map,
                torch.zeros_like(empty_rail.m2_pg_rail_raw_delta_map),
            )
        )

    def test_non_empty_rail_boxes_use_combined_backend_before_scaling(self):
        pos = _make_pos()
        rail_boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float64)
        rail_op = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=rail_boxes,
        )

        rail_op.compute_initial_density_map(pos)

        num_nodes = pos.numel() // 2
        fixed_x = pos[rail_op.num_movable_nodes:rail_op.num_movable_nodes + rail_op.num_terminals]
        fixed_y = pos[
            num_nodes + rail_op.num_movable_nodes:
            num_nodes + rail_op.num_movable_nodes + rail_op.num_terminals
        ]
        packed_pos = torch.cat(
            [
                fixed_x,
                rail_boxes[:, 0],
                fixed_y,
                rail_boxes[:, 1],
            ]
        ).contiguous()
        packed_node_size_x = torch.cat(
            [
                rail_op.node_size_x[
                    rail_op.num_movable_nodes:
                    rail_op.num_movable_nodes + rail_op.num_terminals
                ],
                rail_boxes[:, 2] - rail_boxes[:, 0],
            ]
        ).contiguous()
        packed_node_size_y = torch.cat(
            [
                rail_op.node_size_y[
                    rail_op.num_movable_nodes:
                    rail_op.num_movable_nodes + rail_op.num_terminals
                ],
                rail_boxes[:, 3] - rail_boxes[:, 1],
            ]
        ).contiguous()
        expected_combined_raw = rail_op._fixed_density_map(
            packed_pos,
            packed_node_size_x,
            packed_node_size_y,
            num_movable_nodes=0,
            num_terminals=rail_op.num_terminals + rail_boxes.size(0),
        )
        expected_fixed_raw = rail_op._fixed_density_map(
            pos,
            rail_op.node_size_x,
            rail_op.node_size_y,
            rail_op.num_movable_nodes,
            rail_op.num_terminals,
        )

        self.assertTrue(
            torch.allclose(rail_op.m2_pg_rail_combined_raw_map, expected_combined_raw)
        )
        self.assertTrue(
            torch.allclose(
                rail_op.initial_density_map,
                expected_combined_raw * rail_op.target_density,
            )
        )
        self.assertTrue(
            torch.allclose(
                rail_op.m2_pg_rail_raw_delta_map,
                expected_combined_raw - expected_fixed_raw,
            )
        )

    def test_rail_density_weight_scales_backend_raw_delta(self):
        pos = _make_pos()
        rail_boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float64)
        rail_op = _make_density_op(
            iopin_density_weight=0.0,
            num_terminal_NIs=0,
            m2_pg_rail_density_boxes=rail_boxes,
            m2_pg_rail_density_weight=2.0,
        )

        rail_op.compute_initial_density_map(pos)

        expected_weighted_raw = (
            rail_op.m2_pg_rail_fixed_without_rail_raw_map
            + 2.0 * rail_op.m2_pg_rail_raw_delta_map
        )
        self.assertTrue(
            torch.allclose(
                rail_op.initial_density_map,
                expected_weighted_raw * rail_op.target_density,
            )
        )

    def test_non_region_builders_default_m2_pg_rail_density_weight_to_one(self):
        _install_ieda_stubs()
        from dreamplace import PlaceObj as place_obj_module

        dtype = torch.float64
        rail_boxes = torch.tensor([[0.0, 0.0, 2.0, 1.0]], dtype=dtype)

        class DataCollections:
            node_size_x = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            node_size_y = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            target_density = torch.tensor(0.5, dtype=dtype)
            sorted_node_map = torch.tensor([0], dtype=torch.int32)
            movable_macro_mask = None
            node2fence_region_map = None
            m2_pg_rail_density_boxes = rail_boxes

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
            num_terminal_NIs=0,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[],
            target_density_fence_region=[],
            m2_pg_rail_density_boxes=rail_boxes,
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=0.0,
            ieda_m2_pg_rail_blockage_flag=0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)

        overflow = place_obj.build_electric_overflow(
            params, placedb, DataCollections(), 2, 2
        )

        self.assertEqual(overflow.m2_pg_rail_density_weight, 1.0)
        self.assertIsNotNone(overflow.m2_pg_rail_density_boxes)

    def test_hard_ieda_m2_flag_disables_soft_density_even_with_weight(self):
        _install_ieda_stubs()
        from dreamplace import PlaceObj as place_obj_module

        dtype = torch.float64
        rail_boxes = torch.tensor([[0.0, 0.0, 2.0, 1.0]], dtype=dtype)

        class DataCollections:
            node_size_x = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            node_size_y = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
            target_density = torch.tensor(0.5, dtype=dtype)
            sorted_node_map = torch.tensor([0], dtype=torch.int32)
            movable_macro_mask = None
            node2fence_region_map = None
            m2_pg_rail_density_boxes = rail_boxes

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
            num_terminal_NIs=0,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[],
            target_density_fence_region=[],
            m2_pg_rail_density_boxes=rail_boxes,
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=0.0,
            ieda_m2_pg_rail_blockage_flag=1,
            m2_pg_rail_density_weight=1.0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)

        overflow = place_obj.build_electric_overflow(
            params, placedb, DataCollections(), 2, 2
        )

        self.assertEqual(overflow.m2_pg_rail_density_weight, 0.0)
        self.assertIsNone(overflow.m2_pg_rail_density_boxes)

    def test_region_specific_electric_potential_skips_rail_soft_density(self):
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
            num_terminal_NIs=0,
            num_filler_nodes=1,
            num_nodes=3,
            regions=[object()],
            target_density_fence_region=[0.5],
            filler_start_map=[0, 1],
            m2_pg_rail_density_boxes=torch.tensor(
                [[0.0, 0.0, 2.0, 1.0]],
                dtype=dtype,
            ),
        )
        params = types.SimpleNamespace(
            deterministic_flag=0,
            RePlAce_skip_energy_flag=1,
            iopin_density_weight=0.0,
        )
        place_obj = place_obj_module.PlaceObj.__new__(place_obj_module.PlaceObj)

        with self.assertLogs(level="INFO") as logs:
            potential = place_obj.build_electric_potential(
                params,
                placedb,
                DataCollections(),
                2,
                2,
                name="test",
                region_id=0,
                fence_regions=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=dtype),
            )

        self.assertIsNone(potential.m2_pg_rail_density_boxes)
        self.assertIn(
            "M2 PG rail soft density skipped for fence-region electric field",
            "\n".join(logs.output),
        )


if __name__ == "__main__":
    unittest.main()
