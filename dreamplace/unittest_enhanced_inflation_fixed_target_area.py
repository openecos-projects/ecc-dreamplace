import os
import sys
import unittest

import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.adjust_node_area import adjust_node_area  # noqa: E402
from dreamplace.ops.routability import enhanced_inflation_controller  # noqa: E402


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


class _PlaceDB:
    num_movable_nodes = 2
    num_filler_nodes = 2
    num_nodes = 4


class _DataCollections:
    def __init__(self):
        self.node_size_x = torch.tensor([5.0, 5.0, 5.0, 5.0])
        self.node_size_y = torch.tensor([6.0, 6.0, 4.0, 4.0])
        self.target_density = torch.tensor(0.5)


class EnhancedInflationFixedTargetAreaTest(unittest.TestCase):
    def _make_adjust_op(self, total_place_area=200.0, total_whitespace_area=140.0):
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
            area_adjust_stop_ratio=0.001,
            route_area_adjust_stop_ratio=0.001,
            pin_area_adjust_stop_ratio=0.001,
            unit_pin_capacity=1.0,
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

    def test_fixed_target_area_consumes_filler_and_preserves_target_density(self):
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
        self.assertAlmostEqual(float((node_size_x[-2:] * node_size_y[-2:]).sum()), 30.0, places=5)
        self.assertAlmostEqual(float(target_density), 0.5, places=6)

    def test_fixed_target_area_caps_movable_growth_at_remaining_budget(self):
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
        self.assertAlmostEqual(float((node_size_x[:2] * node_size_y[:2]).sum()), 100.0, places=5)
        self.assertAlmostEqual(float((node_size_x[-2:] * node_size_y[-2:]).sum()), 0.0, places=5)
        self.assertAlmostEqual(float(target_density), 0.5, places=6)

    def test_enhanced_state_always_captures_baseline_target_area(self):
        state = enhanced_inflation_controller.create_inflation_state(
            _Params(),
            _PlaceDB(),
            _DataCollections(),
        )

        self.assertEqual(state.controller_mode, "enhanced_inflation")
        self.assertAlmostEqual(state.target_area, 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
