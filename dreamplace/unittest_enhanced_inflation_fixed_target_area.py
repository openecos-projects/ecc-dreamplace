import os
import sys
import tempfile
import unittest

import numpy as np
import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)
AIEDA_ROOT = os.path.abspath(os.path.join(AUTODMP_ROOT, "..", ".."))
if AIEDA_ROOT not in sys.path:
    sys.path.insert(0, AIEDA_ROOT)

from dreamplace.ops.adjust_node_area import adjust_node_area  # noqa: E402
from dreamplace.ops.routability import enhanced_inflation_controller  # noqa: E402
from dreamplace import PlaceObj  # noqa: E402


class _FixedRouteArea(torch.nn.Module):
    def __init__(self, route_area):
        super().__init__()
        self.route_area = route_area

    def forward(self, pos, node_size_x, node_size_y, utilization_map):
        return self.route_area.to(device=node_size_x.device, dtype=node_size_x.dtype)


class _Params:
    routability_opt_flag = True
    enhanced_inflation_flag = True
    enhanced_inflation_use_target_area = 0
    modularity_inflation_flag = False
    max_route_opt_adjust_rate = 10.0
    route_opt_adjust_exponent = 1.0
    max_pin_opt_adjust_rate = 10.0
    area_adjust_stop_ratio = 0.001
    route_area_adjust_stop_ratio = 0.001
    pin_area_adjust_stop_ratio = 0.001


class _PlotParams:
    plot_flag = True
    modularity_plot_flag = True

    def __init__(self, result_dir):
        self.result_dir = result_dir

    @staticmethod
    def design_name():
        return "plot_test"


class _PlaceDB:
    num_movable_nodes = 2
    num_filler_nodes = 2
    num_nodes = 4
    routing_grid_xl = 0.0
    routing_grid_yl = 0.0
    routing_grid_xh = 100.0
    routing_grid_yh = 100.0
    num_routing_grids_x = 1
    num_routing_grids_y = 1


class _DataCollections:
    def __init__(self):
        self.node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        self.node_size_y = torch.tensor([6.0, 6.0, 4.0, 4.0])
        self.node_areas = self.node_size_x * self.node_size_y
        self.target_density = torch.tensor(0.5)
        self.flat_node2pin_map = torch.empty(0, dtype=torch.int64)
        self.flat_node2pin_start_map = torch.zeros(3, dtype=torch.int64)
        self.pin_weights = None
        self.pin_offset_x = torch.empty(0)
        self.pin_offset_y = torch.empty(0)
        self.unit_pin_capacity = 1.0


class EnhancedInflationFixedTargetAreaTest(unittest.TestCase):
    def _make_adjust_op(
        self,
        total_place_area=200.0,
        total_whitespace_area=140.0,
        inflation_area_budget_ratio=0.1,
        params=None,
    ):
        op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=torch.empty(0, dtype=torch.int64),
            flat_node2pin_start_map=torch.zeros(3, dtype=torch.int64),
            pin_weights=None,
            xl=0.0,
            yl=0.0,
            xh=100.0,
            yh=100.0,
            num_movable_nodes=2,
            num_filler_nodes=2,
            route_num_bins_x=1,
            route_num_bins_y=1,
            pin_num_bins_x=1,
            pin_num_bins_y=1,
            total_place_area=total_place_area,
            total_whitespace_area=total_whitespace_area,
            max_route_opt_adjust_rate=10.0,
            route_opt_adjust_exponent=1.0,
            max_pin_opt_adjust_rate=10.0,
            inflation_area_budget_ratio=inflation_area_budget_ratio,
            area_adjust_stop_ratio=0.001,
            route_area_adjust_stop_ratio=0.001,
            pin_area_adjust_stop_ratio=0.001,
            unit_pin_capacity=1.0,
            params=params,
        )
        return op

    def _run_adjust(self, op, node_size_x, node_size_y, target_density, route_area, fixed_target_area):
        op.compute_node_area_route = _FixedRouteArea(torch.tensor(route_area))
        pos = torch.tensor(
            [
                0.0,
                10.0,
                20.0,
                30.0,
                0.0,
                10.0,
                20.0,
                30.0,
            ]
        )
        pin_offset_x = torch.empty(0)
        pin_offset_y = torch.empty(0)
        route_utilization_map = torch.ones(1, 1)

        original_update = adjust_node_area.update_pin_offset_cpp.forward
        adjust_node_area.update_pin_offset_cpp.forward = lambda *args: None
        try:
            return op(
                pos,
                node_size_x,
                node_size_y,
                pin_offset_x,
                pin_offset_y,
                target_density,
                route_utilization_map,
                None,
                fixed_target_area=fixed_target_area,
            )
        finally:
            adjust_node_area.update_pin_offset_cpp.forward = original_update

    def test_enhanced_inflation_consumes_whitespace_proportionally(self):
        node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        node_size_y = torch.tensor([6.0, 6.0, 4.0, 4.0])
        target_density = torch.tensor(0.5)

        result = self._run_adjust(
            self._make_adjust_op(),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[40.0, 30.0],
            fixed_target_area=100.0,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(float((node_size_x[:2] * node_size_y[:2]).sum()), 70.0, places=5)
        self.assertAlmostEqual(
            float((node_size_x[-2:] * node_size_y[-2:]).sum()),
            37.142857,
            places=5,
        )
        self.assertAlmostEqual(float(target_density), 0.535714, places=6)

    def test_enhanced_inflation_retains_proportional_filler_reserve(self):
        node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        node_size_y = torch.tensor([9.0, 9.0, 1.0, 1.0])
        target_density = torch.tensor(0.5)

        result = self._run_adjust(
            self._make_adjust_op(total_whitespace_area=200.0),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[75.0, 45.0],
            fixed_target_area=100.0,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(float((node_size_x[:2] * node_size_y[:2]).sum()), 110.0, places=5)
        self.assertAlmostEqual(
            float((node_size_x[-2:] * node_size_y[-2:]).sum()),
            8.181818,
            places=5,
        )
        self.assertAlmostEqual(float(target_density), 0.590909, places=6)

    def test_configured_ratio_caps_inflation_at_whitespace_budget(self):
        node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        node_size_y = torch.tensor([6.0, 6.0, 4.0, 4.0])
        target_density = torch.tensor(0.5)

        result = self._run_adjust(
            self._make_adjust_op(inflation_area_budget_ratio=0.05),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[40.0, 30.0],
            fixed_target_area=100.0,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(
            float((node_size_x[:2] * node_size_y[:2]).sum()), 67.0, places=5
        )
        self.assertAlmostEqual(
            float((node_size_x[-2:] * node_size_y[-2:]).sum()), 38.0, places=5
        )
        self.assertAlmostEqual(float(target_density), 0.525, places=6)

    def test_configured_ratio_caps_inflation_without_fixed_target_area(self):
        node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        node_size_y = torch.tensor([6.0, 6.0, 4.0, 4.0])
        target_density = torch.tensor(0.5)

        result = self._run_adjust(
            self._make_adjust_op(inflation_area_budget_ratio=0.05),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[40.0, 30.0],
            fixed_target_area=None,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(
            float((node_size_x[:2] * node_size_y[:2]).sum()), 67.0, places=5
        )
        self.assertAlmostEqual(
            float((node_size_x[-2:] * node_size_y[-2:]).sum()), 40.0, places=5
        )
        self.assertAlmostEqual(float(target_density), 0.535, places=6)

    def test_enhanced_inflation_without_filler_updates_density(self):
        node_size_x = torch.tensor([5.0, 5.0, 0.0, 0.0])
        node_size_y = torch.tensor([6.0, 6.0, 0.0, 0.0])
        target_density = torch.tensor(0.3)

        result = self._run_adjust(
            self._make_adjust_op(),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[40.0, 30.0],
            fixed_target_area=60.0,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(
            float((node_size_x[:2] * node_size_y[:2]).sum()), 70.0, places=5
        )
        self.assertAlmostEqual(float(target_density), 0.35, places=6)

    def test_enhanced_inflation_at_physical_limit_consumes_all_filler(self):
        node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        node_size_y = torch.tensor([9.0, 9.0, 1.0, 1.0])
        target_density = torch.tensor(1.0)

        result = self._run_adjust(
            self._make_adjust_op(
                total_place_area=100.0,
                total_whitespace_area=10.0,
                inflation_area_budget_ratio=1.0,
            ),
            node_size_x,
            node_size_y,
            target_density,
            route_area=[75.0, 45.0],
            fixed_target_area=100.0,
        )

        self.assertEqual(result, (True, True, False))
        self.assertAlmostEqual(
            float((node_size_x[:2] * node_size_y[:2]).sum()), 100.0, places=5
        )
        self.assertAlmostEqual(
            float((node_size_x[-2:] * node_size_y[-2:]).sum()), 0.0, places=5
        )
        self.assertAlmostEqual(float(target_density), 1.0, places=6)

    def test_enhanced_state_always_captures_baseline_target_area(self):
        state = enhanced_inflation_controller.create_inflation_state(
            _Params(),
            _PlaceDB(),
            _DataCollections(),
        )

        self.assertEqual(state.controller_mode, "enhanced_inflation")
        self.assertAlmostEqual(state.target_area, 100.0, places=6)

    def test_build_adjust_node_area_syncs_node_areas_after_inflation(self):
        data_collections = _DataCollections()
        placedb = _PlaceDB()
        params = _Params()
        place_obj = PlaceObj.PlaceObj.__new__(PlaceObj.PlaceObj)
        adjust_op = PlaceObj.PlaceObj.build_adjust_node_area(
            place_obj,
            params,
            placedb,
            data_collections,
        )
        adjust_op._enhanced_adjust_node_area_impl.compute_node_area_route = _FixedRouteArea(
            torch.tensor([40.0, 30.0])
        )
        pos = torch.tensor([0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0])
        original_update = adjust_node_area.update_pin_offset_cpp.forward
        adjust_node_area.update_pin_offset_cpp.forward = lambda *args: None
        try:
            result = adjust_op(
                pos,
                torch.ones(1, 1),
                None,
                fixed_target_area=100.0,
            )
        finally:
            adjust_node_area.update_pin_offset_cpp.forward = original_update

        self.assertEqual(result, (True, True, False))
        torch.testing.assert_close(
            data_collections.node_areas,
            data_collections.node_size_x * data_collections.node_size_y,
        )

    def test_restore_current_round_geometry_syncs_node_areas(self):
        data_collections = _DataCollections()
        placedb = _PlaceDB()
        state = enhanced_inflation_controller.create_inflation_state(
            _Params(),
            placedb,
            data_collections,
        )
        pos = torch.tensor([0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0])
        enhanced_inflation_controller.begin_inflation_round(
            state,
            data_collections,
            placedb,
            pos=pos,
            round_idx=0,
            stage_idx=0,
            iteration=10,
            overflow=torch.tensor(0.1),
            route_map_source="gpugr",
            adjust_area_flag=True,
            adjust_route_area_flag=True,
            adjust_pin_area_flag=False,
        )
        data_collections.node_size_x.mul_(2.0)
        data_collections.node_size_y.mul_(3.0)
        data_collections.node_areas.copy_(
            data_collections.node_size_x * data_collections.node_size_y
        )

        self.assertTrue(
            enhanced_inflation_controller.restore_current_round_geometry(
                state,
                data_collections,
                placedb,
                pos,
            )
        )
        torch.testing.assert_close(
            data_collections.node_areas,
            data_collections.node_size_x * data_collections.node_size_y,
        )

    def test_route_inflation_plots_preserve_each_numeric_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = self._make_adjust_op(params=_PlotParams(tmpdir))
            route_power_map = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
            route_effective_map = route_power_map.clamp(max=2.0)
            old_movable_area = torch.tensor([10.0, 10.0])
            actual_area_increment = torch.tensor([0.0, 2.5])

            color_limits = op._maybe_plot_route_inflation_maps(
                route_power_map=route_power_map,
                route_effective_map=route_effective_map,
                scale_factor=0.25,
                old_movable_area=old_movable_area,
                actual_area_increment=actual_area_increment,
                inflation_round=3,
            )

            self.assertEqual(color_limits, (1.0, 1.25))
            plot_dir = os.path.join(tmpdir, "plot_test", "plot")
            for filename in (
                "route3_raw.png",
                "route3_effective.png",
                "route3.png",
                "route3_maps.npz",
            ):
                self.assertTrue(os.path.isfile(os.path.join(plot_dir, filename)))

            with np.load(os.path.join(plot_dir, "route3_maps.npz")) as maps:
                np.testing.assert_allclose(maps["raw_power_map"], route_power_map.numpy())
                np.testing.assert_allclose(
                    maps["effective_map"], route_effective_map.numpy()
                )
                np.testing.assert_allclose(
                    maps["applied_map"],
                    np.asarray([[1.0, 1.25], [1.25, 1.25]]),
                )
                np.testing.assert_allclose(maps["shared_color_limits"], [1.0, 1.25])


if __name__ == "__main__":
    unittest.main()
