import json
import os
import sys
import types
import unittest

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTODMP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.routability.l_shape_electric_potential import (
    LShapeElectricPotential,
)
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
    validate_ggr_l_shape_topology_params,
)


def _make_op(enable=True):
    shape = (2, 2)
    capacity = torch.ones(shape, dtype=torch.float32)
    demand = torch.zeros(shape, dtype=torch.float32)
    fixed = torch.zeros(shape, dtype=torch.float32)
    return LShapeElectricPotential(
        xl=0.0,
        yl=0.0,
        xh=2.0,
        yh=2.0,
        bin_size_x=1.0,
        bin_size_y=1.0,
        num_bins_x=2,
        num_bins_y=2,
        target_density=capacity,
        target_demand=demand,
        supply_original=capacity,
        target_density_h=capacity,
        target_density_v=capacity,
        target_demand_h=demand,
        target_demand_v=demand,
        supply_original_h=capacity,
        supply_original_v=capacity,
        fix_usage_map=fixed,
        fix_usage_map_h=fixed,
        fix_usage_map_v=fixed,
        capacity_al_enable=enable,
        log_verbose=0,
    )


class LShapeCapacityALTest(unittest.TestCase):
    def test_schema_exposes_only_enable_flag(self):
        params_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "params.json"
        )
        params_path = os.path.abspath(params_path)
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("l_shape_capacity_al_enable", params)
        forbidden = (
            "l_shape_capacity_al_rho",
            "l_shape_capacity_al_rho_init",
            "l_shape_capacity_al_lambda_max",
            "l_shape_capacity_al_update_interval",
            "l_shape_capacity_al_grad_ratio_cap",
        )
        for name in forbidden:
            self.assertNotIn(name, params)

    def test_capacity_al_requires_hard_ggr_mode(self):
        base = dict(
            l_shape_capacity_al_enable=1,
            l_shape_use_ggr_topology=1,
            l_direction_use_gpugr=1,
            soft_l_assignment=0,
            l_shape_use_xplace_weight_schedule=0,
        )
        validate_ggr_l_shape_topology_params(types.SimpleNamespace(**base))

        for key, value in (
            ("l_shape_use_ggr_topology", 0),
            ("l_direction_use_gpugr", 0),
            ("soft_l_assignment", 1),
            ("l_shape_use_xplace_weight_schedule", 1),
        ):
            bad = dict(base)
            bad[key] = value
            with self.assertRaises(RuntimeError, msg=key):
                validate_ggr_l_shape_topology_params(types.SimpleNamespace(**bad))

    def test_lambda_maps_are_detached_and_iteration_gated(self):
        op = _make_op(enable=True)
        g_h = torch.tensor(
            [[0.5, -0.5], [2.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        g_v = torch.tensor(
            [[-1.0, 0.25], [0.1, -0.2]], dtype=torch.float32, requires_grad=True
        )

        q_h, q_v = op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=True,
            placement_iteration_id=7,
            reset_key=("snapshot-a",),
        )

        self.assertEqual(op.lambda_h.shape, g_h.shape)
        self.assertEqual(op.lambda_v.shape, g_v.shape)
        self.assertEqual(op.lambda_h.device, g_h.device)
        self.assertEqual(op.lambda_v.dtype, g_v.dtype)
        self.assertFalse(op.lambda_h.requires_grad)
        self.assertFalse(op.lambda_v.requires_grad)
        self.assertTrue(torch.all(op.lambda_h >= 0))
        self.assertTrue(torch.all(op.lambda_h <= op._capacity_al_lambda_max))
        self.assertTrue(torch.allclose(q_h, torch.relu(g_h.detach())))
        self.assertTrue(torch.allclose(q_v, torch.relu(g_v.detach())))

        lambda_h_once = op.lambda_h.clone()
        lambda_v_once = op.lambda_v.clone()
        op._capacity_al_source_maps(
            g_h * 10.0,
            g_v * 10.0,
            update_lambda=True,
            placement_iteration_id=7,
            reset_key=("snapshot-a",),
        )
        self.assertTrue(torch.equal(op.lambda_h, lambda_h_once))
        self.assertTrue(torch.equal(op.lambda_v, lambda_v_once))

        q_h_next, _ = op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=False,
            placement_iteration_id=8,
            reset_key=("snapshot-a",),
        )
        self.assertTrue(torch.allclose(q_h_next, torch.relu(lambda_h_once + g_h.detach())))

        op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=False,
            placement_iteration_id=8,
            reset_key=("snapshot-b",),
        )
        self.assertTrue(torch.equal(op.lambda_h, lambda_h_once))
        self.assertTrue(torch.equal(op.lambda_v, lambda_v_once))

        g_negative = torch.full_like(g_h, -1.0)
        op._capacity_al_source_maps(
            g_negative,
            g_negative,
            update_lambda=True,
            placement_iteration_id=8,
            reset_key=("snapshot-b",),
        )
        self.assertTrue(torch.count_nonzero(op.lambda_h).item() == 0)
        self.assertTrue(torch.count_nonzero(op.lambda_v).item() == 0)

    def test_disabled_source_does_not_allocate_lambda(self):
        op = _make_op(enable=False)
        g_h = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        g_v = torch.tensor([[-0.25, 0.75]], dtype=torch.float32)

        q_h, q_v = op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=True,
            placement_iteration_id=1,
            reset_key=("snapshot",),
        )

        self.assertTrue(torch.equal(q_h, torch.relu(g_h)))
        self.assertTrue(torch.equal(q_v, torch.relu(g_v)))
        self.assertIsNone(op.lambda_h)
        self.assertIsNone(op.lambda_v)

    def test_enabled_read_only_source_does_not_allocate_or_reset_lambda(self):
        op = _make_op(enable=True)
        g_h = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        g_v = torch.tensor([[-0.25, 0.75]], dtype=torch.float32)

        q_h, q_v = op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=False,
            placement_iteration_id=None,
            reset_key=("snapshot-a",),
        )

        self.assertTrue(torch.equal(q_h, torch.relu(g_h)))
        self.assertTrue(torch.equal(q_v, torch.relu(g_v)))
        self.assertIsNone(op.lambda_h)
        self.assertIsNone(op.lambda_v)
        self.assertEqual(op._capacity_al_reset_count, 0)

        op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=True,
            placement_iteration_id=1,
            reset_key=("snapshot-a",),
        )
        lambda_h_once = op.lambda_h.clone()
        lambda_v_once = op.lambda_v.clone()
        reset_count_once = op._capacity_al_reset_count

        op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=False,
            placement_iteration_id=None,
            reset_key=("snapshot-b",),
        )

        self.assertTrue(torch.equal(op.lambda_h, lambda_h_once))
        self.assertTrue(torch.equal(op.lambda_v, lambda_v_once))
        self.assertEqual(op._capacity_al_reset_count, reset_count_once)

    def test_forward_uses_al_source_without_breaking_segment_gradients(self):
        shape = (4, 4)
        capacity = torch.ones(shape, dtype=torch.float32) * 0.01
        demand = torch.zeros(shape, dtype=torch.float32)
        fixed = torch.zeros(shape, dtype=torch.float32)
        op = LShapeElectricPotential(
            xl=0.0,
            yl=0.0,
            xh=4.0,
            yh=4.0,
            bin_size_x=1.0,
            bin_size_y=1.0,
            num_bins_x=4,
            num_bins_y=4,
            target_density=capacity,
            target_demand=demand,
            supply_original=capacity,
            target_density_h=capacity,
            target_density_v=capacity,
            target_demand_h=demand,
            target_demand_v=demand,
            supply_original_h=capacity,
            supply_original_v=capacity,
            fix_usage_map=fixed,
            fix_usage_map_h=fixed,
            fix_usage_map_v=fixed,
            capacity_al_enable=True,
            log_verbose=2,
        )
        segment_pos = torch.tensor(
            [0.2, 0.2, 2.2, 0.2], dtype=torch.float32, requires_grad=True
        )
        segment_size_x = torch.tensor([1.0, 0.2], dtype=torch.float32)
        segment_size_y = torch.tensor([0.2, 1.0], dtype=torch.float32)
        segment_is_horizontal = torch.tensor([True, False])

        with self.assertLogs(
            "dreamplace.ops.routability.l_shape_electric_potential",
            level="INFO",
        ) as captured:
            energy = op(
                segment_pos,
                segment_size_x,
                segment_size_y,
                segment_is_horizontal,
                update_capacity_al_lambda=True,
                placement_iteration_id=11,
            )
        energy.backward()

        log_text = "\n".join(captured.output)
        self.assertIn("g_h_sum=", log_text)
        self.assertIn("g_v_sum=", log_text)
        self.assertIn("g_h_pos_bins=", log_text)
        self.assertIn("g_v_pos_bins=", log_text)

        self.assertGreater(float(energy.detach().item()), 0.0)
        self.assertIsNotNone(segment_pos.grad)
        self.assertGreater(float(segment_pos.grad.norm().item()), 0.0)
        self.assertTrue(op.last_al_stats["updated"])
        self.assertIn("E_cap_smooth_total", op.last_al_stats)
        self.assertIn("Pq_h_min", op.last_al_stats)
        self.assertGreater(float(op.lambda_h.sum().item()), 0.0)

        lambda_h_once = op.lambda_h.clone()
        op(
            segment_pos.detach().clone().requires_grad_(True),
            segment_size_x,
            segment_size_y,
            segment_is_horizontal,
            update_capacity_al_lambda=True,
            placement_iteration_id=11,
        )
        self.assertTrue(torch.equal(op.lambda_h, lambda_h_once))


if __name__ == "__main__":
    unittest.main()
