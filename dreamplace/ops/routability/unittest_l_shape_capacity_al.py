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
    SegmentElectricPotentialFunction,
    build_boundary_source_map,
    build_fixed_macro_source_maps,
    compute_boundary_fixed_usage,
    compute_macro_fixed_usage,
    compute_track_rho_components,
    compute_fixed_macro_overlap_stats,
    compute_movable_displacement_stats,
)
from dreamplace.ops.routability.l_shape_routability import (
    LShapeRoutabilityOp,
    _build_l_shape_electric_plot_maps,
    _build_l_shape_true_source_plot_maps,
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


def _make_placedb(
    *,
    fixed_macro_mask=(True,),
    fixed_macro_idx=None,
    fixed_slice_start=1,
    node_x=(0.0, 0.0),
    node_y=(0.0, 0.0),
    node_size_x=(1.0, 1.0),
    node_size_y=(1.0, 1.0),
):
    fixed_slice = slice(fixed_slice_start, fixed_slice_start + len(fixed_macro_mask))
    if fixed_macro_idx is None:
        fixed_macro_idx = [
            fixed_slice.start + idx
            for idx, selected in enumerate(fixed_macro_mask)
            if selected
        ]
    return types.SimpleNamespace(
        xl=0.0,
        yl=0.0,
        xh=2.0,
        yh=2.0,
        fixed_slice=fixed_slice,
        fixed_macro_mask=torch.tensor(fixed_macro_mask, dtype=torch.bool),
        fixed_macro_idx=torch.tensor(fixed_macro_idx, dtype=torch.int64),
        node_x=torch.tensor(node_x, dtype=torch.float32),
        node_y=torch.tensor(node_y, dtype=torch.float32),
        node_size_x=torch.tensor(node_size_x, dtype=torch.float32),
        node_size_y=torch.tensor(node_size_y, dtype=torch.float32),
    )


def _forward_inputs(dtype=torch.float32):
    segment_pos = torch.tensor(
        [0.2, 0.2, 2.2, 0.2], dtype=dtype, requires_grad=True
    )
    segment_size_x = torch.tensor([1.0, 0.2], dtype=dtype)
    segment_size_y = torch.tensor([0.2, 1.0], dtype=dtype)
    segment_is_horizontal = torch.tensor([True, False])
    return segment_pos, segment_size_x, segment_size_y, segment_is_horizontal


def _make_forward_op(enable=True, macro_source_value=0.0, capacity_value=0.01):
    shape = (4, 4)
    capacity = torch.ones(shape, dtype=torch.float32) * float(capacity_value)
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
        capacity_al_enable=enable,
        log_verbose=0,
    )
    op.macro_source_map.fill_(float(macro_source_value))
    op.macro_body_source_map.copy_(op.macro_source_map)
    op.macro_exclusion_stats.update(
        {
            "macro_exclusion_enabled": True,
            "macro_count": 1 if macro_source_value > 0 else 0,
            "macro_body_bins": int((op.macro_body_source_map > 0).sum().item()),
            "macro_source_active_bins": int((op.macro_source_map > 0).sum().item()),
            "macro_source_max": float(op.macro_source_map.max().item()),
            "macro_source_sum": float(op.macro_source_map.sum().item()),
            "macro_body_source_max": float(op.macro_body_source_map.max().item()),
            "macro_body_source_sum": float(op.macro_body_source_map.sum().item()),
        }
    )
    return op


class LShapeCapacityALTest(unittest.TestCase):
    def test_fixed_macro_source_rasterizes_body_overlap(self):
        placedb = _make_placedb(
            fixed_macro_mask=(True, True),
            node_x=(0.0, 0.0, 0.5),
            node_y=(0.0, 0.0, 0.5),
            node_size_x=(0.0, 1.0, 1.0),
            node_size_y=(0.0, 1.0, 1.0),
        )

        body, halo, source, stats = build_fixed_macro_source_maps(
            placedb,
            xl=0.0,
            yl=0.0,
            bin_size_x=1.0,
            bin_size_y=1.0,
            num_bins_x=2,
            num_bins_y=2,
        )

        self.assertEqual(tuple(body.shape), (2, 2))
        self.assertAlmostEqual(float(body[0, 0].item()), 1.0)
        self.assertAlmostEqual(float(body[1, 1].item()), 0.25)
        self.assertAlmostEqual(float(body[0, 1].item()), 0.25)
        self.assertAlmostEqual(float(body[1, 0].item()), 0.25)
        self.assertTrue(torch.equal(halo, torch.zeros_like(halo)))
        self.assertTrue(torch.equal(source, body))
        self.assertEqual(stats["macro_count"], 2)
        self.assertEqual(stats["macro_set_source"], "fixed_macro_mask_heuristic")
        self.assertEqual(stats["macro_source_coordinate_system"], "autodmp_scaled")

    def test_fixed_macro_source_overlapping_partials_saturate_with_max(self):
        placedb = _make_placedb(
            fixed_macro_mask=(True, True),
            node_x=(0.0, 0.0, 0.0),
            node_y=(0.0, 0.0, 0.0),
            node_size_x=(0.0, 0.5, 0.5),
            node_size_y=(0.0, 0.5, 0.5),
        )

        body, _, _, _ = build_fixed_macro_source_maps(
            placedb,
            xl=0.0,
            yl=0.0,
            bin_size_x=1.0,
            bin_size_y=1.0,
            num_bins_x=2,
            num_bins_y=2,
        )

        self.assertAlmostEqual(float(body[0, 0].item()), 0.25)

    def test_fixed_macro_source_requires_heuristic_idx_mask_consistency(self):
        placedb = _make_placedb(
            fixed_macro_mask=(True,),
            fixed_macro_idx=(0,),
        )

        with self.assertRaises(RuntimeError):
            build_fixed_macro_source_maps(
                placedb,
                xl=0.0,
                yl=0.0,
                bin_size_x=1.0,
                bin_size_y=1.0,
                num_bins_x=2,
                num_bins_y=2,
            )

    def test_empty_fixed_macro_set_is_zero_source(self):
        placedb = _make_placedb(
            fixed_macro_mask=(False,),
            fixed_macro_idx=(),
        )

        body, halo, source, stats = build_fixed_macro_source_maps(
            placedb,
            xl=0.0,
            yl=0.0,
            bin_size_x=1.0,
            bin_size_y=1.0,
            num_bins_x=2,
            num_bins_y=2,
        )

        self.assertEqual(stats["macro_count"], 0)
        self.assertEqual(stats["macro_source_active_bins"], 0)
        self.assertTrue(torch.equal(body, torch.zeros_like(body)))
        self.assertTrue(torch.equal(halo, torch.zeros_like(halo)))
        self.assertTrue(torch.equal(source, torch.zeros_like(source)))

    def test_fixed_macro_overlap_stats_use_movable_cells_and_heuristic_macros(self):
        placedb = _make_placedb(
            fixed_macro_mask=(True,),
            fixed_slice_start=2,
            node_x=(0.0, 2.0, 0.0),
            node_y=(0.0, 0.0, 0.0),
            node_size_x=(1.0, 1.0, 1.0),
            node_size_y=(1.0, 1.0, 1.0),
        )
        placedb.num_movable_nodes = 2
        pos = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        node_size_x = torch.tensor([1.0, 1.0, 1.0])
        node_size_y = torch.tensor([1.0, 1.0, 1.0])

        stats = compute_fixed_macro_overlap_stats(
            pos, node_size_x, node_size_y, placedb
        )

        self.assertEqual(stats["fixed_macro_overlap_macro_count"], 1)
        self.assertEqual(stats["fixed_macro_overlap_cell_count"], 1)
        self.assertEqual(stats["fixed_macro_overlap_pair_count"], 1)
        self.assertAlmostEqual(stats["fixed_macro_overlap_area"], 1.0)
        self.assertAlmostEqual(stats["fixed_macro_overlap_area_ratio"], 0.5)

    def test_movable_displacement_stats_report_legalization_cleanup_distance(self):
        placedb = _make_placedb(fixed_macro_mask=(False,), fixed_macro_idx=())
        placedb.num_movable_nodes = 2
        before = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        after = torch.tensor([3.0, 2.0, 0.0, 4.0, 0.0, 0.0])

        stats = compute_movable_displacement_stats(before, after, placedb)

        self.assertEqual(stats["movable_displacement_moved_count"], 1)
        self.assertAlmostEqual(stats["movable_displacement_max"], 5.0)
        self.assertAlmostEqual(stats["movable_displacement_mean"], 2.5)
        self.assertAlmostEqual(stats["movable_displacement_sum"], 5.0)

    def test_macro_source_registered_buffers_and_target_refresh_preserves_it(self):
        shape = (2, 2)
        capacity = torch.ones(shape, dtype=torch.float32)
        demand = torch.zeros(shape, dtype=torch.float32)
        fixed = torch.zeros(shape, dtype=torch.float32)
        placedb = _make_placedb(
            fixed_macro_mask=(True,),
            node_x=(0.0, 0.0),
            node_y=(0.0, 0.0),
            node_size_x=(0.0, 1.0),
            node_size_y=(0.0, 1.0),
        )
        op = LShapeElectricPotential(
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
            placedb=placedb,
        )

        self.assertIn("macro_body_source_map", dict(op.named_buffers()))
        self.assertIn("macro_halo_source_map", dict(op.named_buffers()))
        self.assertIn("macro_source_map", dict(op.named_buffers()))
        macro_before = op.macro_source_map.clone()
        op.set_directional_targets(target_density_h=capacity * 2.0)
        self.assertTrue(torch.equal(op.macro_source_map, macro_before))

    def test_routability_update_targets_preserves_macro_source_buffer(self):
        shape = (2, 2)
        capacity = torch.ones(shape, dtype=torch.float32)
        demand = torch.zeros(shape, dtype=torch.float32)
        fixed = torch.zeros(shape, dtype=torch.float32)
        placedb = _make_placedb(
            fixed_macro_mask=(True,),
            node_x=(0.0, 0.0),
            node_y=(0.0, 0.0),
            node_size_x=(0.0, 1.0),
            node_size_y=(0.0, 1.0),
        )
        params = types.SimpleNamespace(
            route_wire_width=0.0,
            soft_l_assignment=False,
            soft_l_use_resolver_prior=True,
            soft_l_temperature=1.0,
            soft_l_prior_bias=1.0,
            soft_l_min_weight=0.05,
            soft_l_tie_break_delta=0.10,
            soft_l_background_weight=0.05,
            soft_l_self_overflow_weight=0.5,
            soft_l_hotspot_weight=2.0,
            soft_l_hotspot_ramp_ratio=0.03,
            soft_l_adaptive_tau=False,
            soft_l_adaptive_scale=1.0,
            soft_l_same_net_diag_split_sigma=1.0,
            soft_l_same_net_diag_split_min_support=1e-6,
            soft_l_same_net_diag_split_max_distance=0.0,
            l_shape_profile_flag=False,
            l_shape_capacity_al_enable=False,
            deterministic_flag=False,
            l_shape_debug_hash_flag=False,
            l_shape_debug_hash_start_iter=-1,
            l_shape_debug_hash_end_iter=-1,
            l_shape_debug_hash_sample=4096,
            l_shape_edge_net_ids_cache_flag=1,
            soft_l_debug_update_interval=10,
            l_shape_log_verbose=0,
        )
        routability = LShapeRoutabilityOp(
            placedb,
            params,
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
        )
        macro_before = routability.density_op.macro_source_map.clone()

        routability.update_targets(
            target_density=capacity * 2.0,
            target_demand=demand + 0.1,
            raw_wire_demand_map=demand + 0.2,
            supply_original=capacity * 3.0,
            target_density_h=capacity * 4.0,
            target_density_v=capacity * 5.0,
            target_demand_h=demand + 0.3,
            target_demand_v=demand + 0.4,
            raw_wire_demand_map_h=demand + 0.5,
            raw_wire_demand_map_v=demand + 0.6,
            supply_original_h=capacity * 6.0,
            supply_original_v=capacity * 7.0,
            fix_usage_map=fixed + 0.1,
            fix_usage_map_h=fixed + 0.2,
            fix_usage_map_v=fixed + 0.3,
        )

        self.assertTrue(torch.equal(routability.density_op.macro_source_map, macro_before))

    def test_macro_fixed_usage_uses_capacity_capped_p95_and_preserves_base_fixed_usage(self):
        op = _make_op(enable=True)
        op.macro_source_map = torch.tensor(
            [[1.0, 0.0], [0.5, 1.0]], dtype=torch.float32
        )
        capacity = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64
        )
        fixed = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        )
        segment_tracks = torch.zeros_like(capacity)
        segment_tracks[0, 0] = 3.0

        components = compute_track_rho_components(
            density_seg_area=segment_tracks,
            supply_original=capacity,
            fix_usage=fixed,
            bin_area=1.0,
            macro_source_map=op.macro_source_map,
        )

        self.assertEqual(components["macro_usage"].dtype, torch.float32)
        self.assertEqual(components["base_fixed_usage"].dtype, torch.float32)
        self.assertEqual(components["initial_density_tracks"].dtype, torch.float32)
        self.assertEqual(components["rho_map"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                components["macro_usage"],
                torch.tensor([[10.0, 0.0], [15.0, 38.5]], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.equal(components["base_fixed_usage"], fixed.to(dtype=torch.float32)))
        self.assertTrue(
            torch.equal(
                components["initial_density_tracks"],
                fixed.to(dtype=torch.float32) + components["macro_usage"],
            )
        )
        self.assertAlmostEqual(components["macro_usage_reference_p95"], 38.5)
        self.assertAlmostEqual(float(components["residual_tracks"][0, 0]), 0.4)

    def test_electric_potential_routing_buffers_are_normalized_to_float32(self):
        shape = (2, 2)
        capacity = torch.ones(shape, dtype=torch.float64)
        demand = torch.zeros(shape, dtype=torch.float64)
        fixed = torch.zeros(shape, dtype=torch.float64)
        op = LShapeElectricPotential(
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
            raw_wire_demand_map=demand,
            supply_original=capacity,
            target_density_h=capacity,
            target_density_v=capacity,
            target_demand_h=demand,
            target_demand_v=demand,
            raw_wire_demand_map_h=demand,
            raw_wire_demand_map_v=demand,
            supply_original_h=capacity,
            supply_original_v=capacity,
            fix_usage_map=fixed,
            fix_usage_map_h=fixed,
            fix_usage_map_v=fixed,
            capacity_al_enable=True,
            log_verbose=0,
        )
        for name in (
            "target_density",
            "target_demand",
            "raw_wire_demand_map",
            "supply_original",
            "target_density_h",
            "target_density_v",
            "target_demand_h",
            "target_demand_v",
            "raw_wire_demand_map_h",
            "raw_wire_demand_map_v",
            "supply_original_h",
            "supply_original_v",
            "fix_usage_map",
            "fix_usage_map_h",
            "fix_usage_map_v",
            "macro_body_source_map",
            "macro_halo_source_map",
            "macro_source_map",
            "boundary_source_map",
            "area_per_track",
        ):
            self.assertEqual(getattr(op, name).dtype, torch.float32, name)

        op.set_target_density(capacity * 2.0)
        op.set_target_demand(demand + 1.0)
        op.set_raw_wire_demand_map(demand + 2.0)
        op.set_supply_original(capacity * 3.0)
        op.set_fix_usage_map(fixed + 0.25)
        op.set_directional_targets(
            target_density_h=capacity * 4.0,
            target_density_v=capacity * 5.0,
            target_demand_h=demand + 3.0,
            target_demand_v=demand + 4.0,
            raw_wire_demand_map_h=demand + 5.0,
            raw_wire_demand_map_v=demand + 6.0,
            supply_original_h=capacity * 6.0,
            supply_original_v=capacity * 7.0,
            fix_usage_map_h=fixed + 0.5,
            fix_usage_map_v=fixed + 0.75,
        )
        for name in (
            "target_density",
            "target_demand",
            "raw_wire_demand_map",
            "supply_original",
            "target_density_h",
            "target_density_v",
            "target_demand_h",
            "target_demand_v",
            "raw_wire_demand_map_h",
            "raw_wire_demand_map_v",
            "supply_original_h",
            "supply_original_v",
            "fix_usage_map",
            "fix_usage_map_h",
            "fix_usage_map_v",
        ):
            self.assertEqual(getattr(op, name).dtype, torch.float32, name)

    def test_macro_fixed_usage_can_be_computed_directly_for_direction(self):
        macro_source = torch.tensor(
            [[1.0, 0.0], [0.25, 1.0]], dtype=torch.float32
        )
        capacity = torch.tensor(
            [[2.0, 3.0], [4.0, 10.0]], dtype=torch.float32
        )

        macro_usage, p95 = compute_macro_fixed_usage(macro_source, capacity)

        expected_p95 = float(torch.quantile(capacity.reshape(-1), 0.95).item())
        self.assertAlmostEqual(p95, expected_p95, places=5)
        self.assertTrue(
            torch.equal(
                macro_usage,
                torch.tensor([[2.0, 0.0], [1.0, expected_p95]], dtype=torch.float32),
            )
        )

    def test_macro_fixed_usage_zero_source_skips_capacity_reference(self):
        macro_source = torch.zeros((2, 2), dtype=torch.float32)
        capacity = torch.tensor(
            [[2.0, 3.0], [4.0, 10.0]], dtype=torch.float32
        )

        macro_usage, p95 = compute_macro_fixed_usage(macro_source, capacity)

        self.assertEqual(p95, 0.0)
        self.assertTrue(torch.equal(macro_usage, torch.zeros_like(capacity)))

    def test_boundary_source_uses_edge_ring_and_capacity_capped_p95(self):
        source, stats = build_boundary_source_map(
            num_bins_x=4,
            num_bins_y=5,
            width_bins=1,
            strength=0.5,
        )

        self.assertEqual(tuple(source.shape), (4, 5))
        self.assertAlmostEqual(float(source[0, 0].item()), 0.5)
        self.assertAlmostEqual(float(source[3, 4].item()), 0.5)
        self.assertAlmostEqual(float(source[1, 2].item()), 0.0)
        self.assertEqual(stats["boundary_source_active_bins"], 14)
        self.assertEqual(stats["boundary_source_width_bins"], 1)

        capacity = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 100.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
            ],
            dtype=torch.float32,
        )
        usage, p95 = compute_boundary_fixed_usage(source, capacity)

        expected_p95 = float(torch.quantile(capacity.reshape(-1), 0.95).item())
        self.assertAlmostEqual(p95, expected_p95, places=5)
        self.assertAlmostEqual(float(usage[0, 0].item()), 0.5)
        self.assertAlmostEqual(float(usage[0, 4].item()), expected_p95 * 0.5, places=4)
        self.assertAlmostEqual(float(usage[1, 2].item()), 0.0)

    def test_track_rho_components_include_boundary_usage_as_fixed_occupancy(self):
        capacity = torch.ones((3, 3), dtype=torch.float32) * 10.0
        fixed = torch.ones((3, 3), dtype=torch.float32)
        segment_tracks = torch.zeros_like(capacity)
        boundary_source, _ = build_boundary_source_map(3, 3, width_bins=1, strength=0.25)

        components = compute_track_rho_components(
            density_seg_area=segment_tracks,
            supply_original=capacity,
            fix_usage=fixed,
            bin_area=1.0,
            boundary_source_map=boundary_source,
        )

        self.assertTrue(
            torch.equal(
                components["boundary_usage"],
                boundary_source * 10.0,
            )
        )
        self.assertTrue(
            torch.equal(
                components["initial_density_tracks"],
                fixed + components["boundary_usage"],
            )
        )
        self.assertAlmostEqual(float(components["residual_tracks"][0, 0].item()), -0.65)
        self.assertAlmostEqual(float(components["residual_tracks"][1, 1].item()), -0.9)

    def test_capacity_al_lambda_ignores_macro_source(self):
        op = _make_op(enable=True)
        g_h = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
        g_v = torch.tensor([[-0.5, 0.25]], dtype=torch.float32)
        op.macro_source_map = torch.ones_like(g_h)

        op._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=True,
            placement_iteration_id=1,
            reset_key=("same-residual",),
        )
        lambda_h_with_macro = op.lambda_h.clone()
        lambda_v_with_macro = op.lambda_v.clone()

        op2 = _make_op(enable=True)
        op2.macro_source_map = torch.zeros_like(g_h)
        op2._capacity_al_source_maps(
            g_h,
            g_v,
            update_lambda=True,
            placement_iteration_id=1,
            reset_key=("same-residual",),
        )

        self.assertTrue(torch.equal(lambda_h_with_macro, op2.lambda_h))
        self.assertTrue(torch.equal(lambda_v_with_macro, op2.lambda_v))

    def test_forward_macro_occupancy_changes_overflow_without_mutating_base_fixed_usage(self):
        segment_pos, segment_size_x, segment_size_y, segment_is_horizontal = _forward_inputs()
        op = _make_forward_op(enable=True, macro_source_value=1.0, capacity_value=1.0)
        fix_usage_h_before = op.fix_usage_map_h.clone()
        fix_usage_v_before = op.fix_usage_map_v.clone()

        op(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_is_horizontal,
            update_capacity_al_lambda=True,
            placement_iteration_id=1,
        )
        overflow_with_macro = SegmentElectricPotentialFunction.last_overflow_map.clone()

        op_no_macro = _make_forward_op(enable=True, macro_source_value=0.0, capacity_value=1.0)
        segment_pos2, segment_size_x2, segment_size_y2, segment_is_horizontal2 = _forward_inputs()
        op_no_macro(
            segment_pos2,
            segment_size_x2,
            segment_size_y2,
            segment_is_horizontal2,
            update_capacity_al_lambda=True,
            placement_iteration_id=1,
        )
        overflow_without_macro = SegmentElectricPotentialFunction.last_overflow_map.clone()

        self.assertTrue(torch.equal(op.fix_usage_map_h, fix_usage_h_before))
        self.assertTrue(torch.equal(op.fix_usage_map_v, fix_usage_v_before))
        self.assertGreater(
            float(overflow_with_macro.sum().item()),
            float(overflow_without_macro.sum().item()),
        )
        self.assertGreater(float(op.last_macro_exclusion_stats["macro_usage_sum"]), 0.0)

    def test_forward_capacity_al_lambda_includes_macro_occupancy(self):
        segment_pos, segment_size_x, segment_size_y, segment_is_horizontal = _forward_inputs()
        op = _make_forward_op(enable=True, macro_source_value=1.0, capacity_value=1.0)
        op(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_is_horizontal,
            update_capacity_al_lambda=True,
            placement_iteration_id=3,
        )

        segment_pos2, segment_size_x2, segment_size_y2, segment_is_horizontal2 = _forward_inputs()
        op_no_macro = _make_forward_op(enable=True, macro_source_value=0.0, capacity_value=1.0)
        op_no_macro(
            segment_pos2,
            segment_size_x2,
            segment_size_y2,
            segment_is_horizontal2,
            update_capacity_al_lambda=True,
            placement_iteration_id=3,
        )

        self.assertGreater(float(op.lambda_h.sum().item()), float(op_no_macro.lambda_h.sum().item()))
        self.assertGreater(float(op.lambda_v.sum().item()), float(op_no_macro.lambda_v.sum().item()))
        self.assertGreater(float(op.last_al_stats["g_h_sum"]), float(op_no_macro.last_al_stats["g_h_sum"]))
        self.assertGreater(float(op.last_al_stats["g_v_sum"]), float(op_no_macro.last_al_stats["g_v_sum"]))

    def test_true_source_plot_payload_uses_forward_rho_maps(self):
        l_shape_op = types.SimpleNamespace(density_op=_make_op(enable=True))
        SegmentElectricPotentialFunction.last_rho_map_h = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        )
        SegmentElectricPotentialFunction.last_rho_map_v = torch.tensor(
            [[0.5, 0.25], [0.125, 0.0]], dtype=torch.float64
        )
        SegmentElectricPotentialFunction.last_rho_map = (
            SegmentElectricPotentialFunction.last_rho_map_h
            + SegmentElectricPotentialFunction.last_rho_map_v
        )

        payload = _build_l_shape_true_source_plot_maps(l_shape_op)

        self.assertTrue(payload["split_available"])
        self.assertEqual(payload["source_name"], "forward_rho")
        self.assertTrue(
            torch.equal(
                payload["source_map_h"],
                SegmentElectricPotentialFunction.last_rho_map_h.to(dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["source_map_v"],
                SegmentElectricPotentialFunction.last_rho_map_v.to(dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["source_map"],
                SegmentElectricPotentialFunction.last_rho_map.to(dtype=torch.float32),
            )
        )

    def test_true_source_plot_payload_exposes_macro_source_map(self):
        density_op = _make_op(enable=True)
        density_op.macro_source_map = torch.tensor(
            [[0.0, 1.0], [0.5, 0.0]], dtype=torch.float64
        )
        density_op.macro_body_source_map = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
        )
        density_op.macro_halo_source_map = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64
        )
        l_shape_op = types.SimpleNamespace(density_op=density_op)
        SegmentElectricPotentialFunction.last_rho_map = torch.ones(
            (2, 2), dtype=torch.float32
        )
        SegmentElectricPotentialFunction.last_rho_map_h = None
        SegmentElectricPotentialFunction.last_rho_map_v = None

        payload = _build_l_shape_true_source_plot_maps(l_shape_op)

        self.assertTrue(
            torch.equal(
                payload["macro_source_map"],
                density_op.macro_source_map.to(dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["macro_body_source_map"],
                density_op.macro_body_source_map.to(dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["macro_halo_source_map"],
                density_op.macro_halo_source_map.to(dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["macro_mask"],
                torch.tensor([[False, True], [True, False]]),
            )
        )

    def test_electric_plot_payload_exposes_macro_usage_initial_density(self):
        capacity_h = torch.ones((2, 2), dtype=torch.float32) * 10.0
        capacity_v = torch.ones((2, 2), dtype=torch.float32) * 20.0
        fixed_h = torch.ones((2, 2), dtype=torch.float32)
        fixed_v = torch.ones((2, 2), dtype=torch.float32) * 2.0
        demand = torch.zeros((2, 2), dtype=torch.float32)
        density_op = _make_op(enable=True)
        density_op.blockage_initial_density = True
        density_op.bin_size_x = 1.0
        density_op.bin_size_y = 1.0
        density_op.target_density = capacity_h + capacity_v
        density_op.target_demand = demand
        density_op.target_density_h = capacity_h
        density_op.target_density_v = capacity_v
        density_op.target_demand_h = demand
        density_op.target_demand_v = demand
        density_op.supply_original_h = capacity_h
        density_op.supply_original_v = capacity_v
        density_op.macro_source_map = torch.tensor(
            [[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32
        )
        density_op.boundary_source_map = torch.tensor(
            [[0.25, 0.0], [0.0, 0.25]], dtype=torch.float32
        )
        l_shape_op = types.SimpleNamespace(
            density_op=density_op,
            cached_density_map=torch.zeros((2, 2), dtype=torch.float32),
            cached_density_map_h=torch.zeros((2, 2), dtype=torch.float32),
            cached_density_map_v=torch.zeros((2, 2), dtype=torch.float32),
            fix_usage_map=fixed_h + fixed_v,
            fix_usage_map_h=fixed_h,
            fix_usage_map_v=fixed_v,
        )

        payload = _build_l_shape_electric_plot_maps(l_shape_op)

        self.assertTrue(
            torch.equal(
                payload["macro_usage_map_h"],
                torch.tensor([[10.0, 0.0], [5.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["macro_usage_map_v"],
                torch.tensor([[20.0, 0.0], [10.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["boundary_usage_map_h"],
                torch.tensor([[2.5, 0.0], [0.0, 2.5]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["boundary_usage_map_v"],
                torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.equal(payload["base_fixed_usage_map"], fixed_h + fixed_v))
        self.assertTrue(
            torch.equal(
                payload["initial_density_map"],
                (
                    payload["base_fixed_usage_map"]
                    + payload["macro_usage_map"]
                    + payload["boundary_usage_map"]
                ),
            )
        )

    def test_schema_exposes_only_enable_flag(self):
        params_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "params.json"
        )
        params_path = os.path.abspath(params_path)
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("l_shape_capacity_al_enable", params)
        self.assertIn("l_shape_overflow_threshold", params)
        self.assertEqual(params["l_shape_overflow_threshold"]["default"], 0.2)
        forbidden = (
            "l_shape_capacity_al_rho",
            "l_shape_capacity_al_rho_init",
            "l_shape_capacity_al_lambda_max",
            "l_shape_capacity_al_update_interval",
            "l_shape_capacity_al_grad_ratio_cap",
            "l_shape_diff_source_mode",
            "l_shape_diff_source_sink_scale",
            "l_shape_diff_source_clip_value",
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
