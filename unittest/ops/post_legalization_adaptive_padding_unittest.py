import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dreamplace.Params import Params
from dreamplace.ops.routability import route_evaluation
from dreamplace.ops.routability.post_legalization_adaptive_padding import (
    allocate_padding_sites,
    build_smoothed_overflow_map,
    compute_cell_box_overlap_stats,
    score_cells_from_overflow,
)


class PostLegalizationAdaptivePaddingTest(unittest.TestCase):
    def test_gpugr_operator_receives_params_and_placedb(self):
        params = SimpleNamespace(
            post_legalization_adaptive_padding_flag=1,
            legalize_flag=1,
            post_legalization_padding_rrr_iters=0,
            result_dir="/tmp",
            gpu_id=0,
            num_threads=1,
            gpugr_backend="cpu_pr",
            design_name=lambda: "cpu_padding_test",
        )
        placedb = SimpleNamespace(regions=[])
        operator = mock.Mock()
        operator.run_gpugr.side_effect = RuntimeError("stop after operator dispatch")

        with (
            mock.patch.object(route_evaluation, "write_back_movable_lpos"),
            mock.patch.object(
                route_evaluation,
                "compute_route_grid_like_xplace",
                return_value=(8, 8),
            ),
            mock.patch.object(
                route_evaluation,
                "get_cached_gpugr_operator",
                return_value=operator,
            ) as get_operator,
            self.assertRaisesRegex(RuntimeError, "stop after operator dispatch"),
        ):
            route_evaluation.run_post_legalization_adaptive_padding(
                params,
                placedb,
                pos=object(),
                model=object(),
            )

        get_operator.assert_called_once_with(params, placedb)

    def test_scores_wide_cell_from_all_overlapped_bins(self):
        horizontal = torch.zeros((4, 2), dtype=torch.float32)
        vertical = torch.zeros_like(horizontal)
        vertical[2, 0] = 2.0
        overflow = build_smoothed_overflow_map(horizontal, vertical, smooth_kernel=1)
        pos = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        size_x = torch.tensor([0.5, 2.0], dtype=torch.float32)
        size_y = torch.tensor([0.5, 0.5], dtype=torch.float32)

        scores = score_cells_from_overflow(
            pos=pos,
            node_size_x=size_x,
            node_size_y=size_y,
            num_nodes=2,
            num_movable_nodes=2,
            overflow_xy=overflow,
            grid_xl=0.0,
            grid_yl=0.0,
            grid_xh=4.0,
            grid_yh=2.0,
        )

        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 2.0)

    def test_allocates_padding_with_independent_row_segment_budgets(self):
        # A fixed blockage [9, 11] splits the row into two 9-site segments.
        pos = torch.tensor([0.0, 12.0, 9.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        size_x = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        size_y = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        plan = allocate_padding_sites(
            scores=np.asarray([2.0, 1.0], dtype=np.float32),
            pos=pos,
            node_size_x=size_x,
            node_size_y=size_y,
            num_nodes=3,
            num_movable_nodes=2,
            num_physical_nodes=3,
            xl=0.0,
            yl=0.0,
            xh=20.0,
            yh=1.0,
            site_width=1.0,
            row_height=1.0,
            hot_cell_ratio=1.0,
            row_free_ratio=0.5,
            max_padding_sites=1,
        )

        np.testing.assert_array_equal(plan["padding_sites"], [1, 1])
        self.assertEqual(plan["allocated_count"], 2)
        self.assertEqual(plan["total_added_sites"], 4)

    def test_capacity_clipping_keeps_highest_score(self):
        pos = torch.tensor([0.0, 2.0, 4.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        size_x = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        size_y = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        plan = allocate_padding_sites(
            scores=np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
            pos=pos,
            node_size_x=size_x,
            node_size_y=size_y,
            num_nodes=3,
            num_movable_nodes=3,
            num_physical_nodes=3,
            xl=0.0,
            yl=0.0,
            xh=8.0,
            yh=1.0,
            site_width=1.0,
            row_height=1.0,
            hot_cell_ratio=1.0,
            row_free_ratio=1.0,
            max_padding_sites=1,
        )

        np.testing.assert_array_equal(plan["padding_sites"], [0, 1, 0])
        self.assertEqual(plan["total_free_sites"], 2)

    def test_cell_box_overlap_stats(self):
        pos = torch.tensor([0.0, 3.0, 0.0, 0.0], dtype=torch.float32)
        size_x = torch.tensor([2.0, 2.0], dtype=torch.float32)
        size_y = torch.tensor([1.0, 1.0], dtype=torch.float32)
        boxes = torch.tensor([[1.0, 0.0, 4.0, 1.0]], dtype=torch.float32)

        stats = compute_cell_box_overlap_stats(
            pos,
            size_x,
            size_y,
            boxes,
            num_nodes=2,
            num_movable_nodes=2,
        )

        self.assertEqual(stats["overlap_count"], 2)
        self.assertEqual(stats["overlap_area"], 2.0)

    def test_params_normalize_padding_values(self):
        params = Params()
        params.fromJson(
            {
                "post_legalization_adaptive_padding_flag": "1",
                "post_legalization_padding_hot_cell_ratio": "0.2",
                "post_legalization_padding_row_free_ratio": "0.5",
                "post_legalization_padding_max_sites": "1",
                "post_legalization_padding_smooth_kernel": "3",
                "post_legalization_padding_max_retries": "4",
                "post_legalization_padding_rrr_iters": "0",
            }
        )

        self.assertEqual(params.post_legalization_adaptive_padding_flag, 1)
        self.assertEqual(params.post_legalization_padding_max_sites, 1)
        self.assertEqual(params.post_legalization_padding_smooth_kernel, 3)

    def test_params_reject_even_smoothing_kernel(self):
        params = Params()
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            params.fromJson({"post_legalization_padding_smooth_kernel": 2})


if __name__ == "__main__":
    unittest.main()
