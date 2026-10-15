import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from dreamplace.ops.routability import adaptive_padding_cpp
from dreamplace.ops.routability import iterative_legalization_padding as iterative
from dreamplace.ops.routability.post_legalization_adaptive_padding import (
    allocate_padding_sites,
    score_cells_from_overflow,
    select_top_overflow_bins,
)


class IterativeLegalizationPaddingTest(unittest.TestCase):
    def test_native_planner_matches_python_on_ties_obstacles_and_cumulative_sizes(self):
        horizontal = np.array([[2.0], [2.0], [0.0], [4.0], [1.0], [0.0]])
        vertical = np.array([[0.0], [0.0], [3.0], [0.0], [0.0], [0.0]])
        physical_pos = np.array([1.0, 3.0, 13.0, 16.0, 6.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0])
        physical_size = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
        height = np.ones(5)
        rows = np.array([[0.0, 0.0, 10.0, 1.0], [12.0, 0.0, 18.0, 1.0]])
        eligible = np.array([True, True, True, False])
        for ratio in (0.2, 0.6, 1.0):
            for cumulative in (np.array([0, 0, 0, 0]), np.array([1, 1, 0, 0])):
                logical_pos = physical_pos.copy()
                logical_pos[:4] -= cumulative
                logical_size = physical_size.copy()
                logical_size[:4] += 2 * cumulative
                native = adaptive_padding_cpp.plan_padding(
                    horizontal, vertical, np.linspace(0, 18, 7), np.array([0., 1.]),
                    physical_pos, logical_pos,
                    physical_size, logical_size, height, rows, eligible,
                    5, 4, 5, 0.0, 0.0, 18.0, 1.0, 1.0, 1.0, ratio, 1.0,
                )
                self.assertTrue(native["scores"].flags.c_contiguous)
                self.assertTrue(native["padding_sites"].flags.c_contiguous)
                hot = select_top_overflow_bins(horizontal, vertical, ratio)
                scores = score_cells_from_overflow(
                    physical_pos, physical_size, height, 5, 4, hot,
                    0.0, 0.0, 18.0, 1.0,
                )
                expected = allocate_padding_sites(
                    scores, logical_pos, logical_size, height, 5, 4, 5,
                    0.0, 0.0, 18.0, 1.0, 1.0, 1.0,
                    hot_cell_ratio=1.0, row_free_ratio=1.0,
                    max_padding_sites=1, eligible_mask=eligible, rows=rows,
                )
                np.testing.assert_array_equal(native["scores"], expected["scores"])
                np.testing.assert_array_equal(native["padding_sites"], expected["padding_sites"])
                self.assertEqual(native["hot_bins_count"], np.count_nonzero(hot))
                for name in ("positive_score_count", "allocated_count", "total_free_sites"):
                    self.assertEqual(native[name], expected[name])

    def test_native_planner_uses_all_free_sites_and_excludes_row_gaps(self):
        plan = adaptive_padding_cpp.plan_padding(
            np.array([[1.0]]), np.zeros((1, 1)),
            np.array([0.0, 40.0]), np.array([0.0, 1.0]),
            np.array([1.0, 19.0, 23.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 19.0, 23.0, 0.0, 0.0, 0.0]),
            np.ones(3), np.ones(3), np.ones(3),
            np.array([[0.0, 0.0, 18.0, 1.0], [22.0, 0.0, 40.0, 1.0]]),
            np.ones(3, dtype=bool), 3, 3, 3,
            0.0, 0.0, 40.0, 1.0, 1.0, 1.0, 0.2, 1.0,
        )
        np.testing.assert_array_equal(plan["padding_sites"], [1, 0, 1])
        self.assertEqual(plan["total_free_sites"], 34)

    def test_route_grid_origin_and_last_bin_differ_from_row_bounds(self):
        horizontal = np.array([[0.0], [5.0], [0.0], [0.0]])
        pos = np.array([0.5, 0.0])
        args = (pos, pos, np.ones(1), np.ones(1), np.ones(1),
                np.array([[0.0, 0.0, 8.0, 1.0]]),
                np.ones(1, dtype=bool), 1, 1, 1,
                0.0, 0.0, 8.0, 1.0, 1.0, 1.0, 0.2, 1.0)
        actual = adaptive_padding_cpp.plan_padding(
            horizontal, np.zeros_like(horizontal),
            np.array([-2.0, 0.0, 2.0, 4.0, 8.0]), np.array([0.0, 1.0]), *args,
        )
        core_grid = adaptive_padding_cpp.plan_padding(
            horizontal, np.zeros_like(horizontal),
            np.linspace(0.0, 8.0, 5), np.array([0.0, 1.0]), *args,
        )
        np.testing.assert_array_equal(actual["scores"], [5.0])
        np.testing.assert_array_equal(actual["padding_sites"], [1])
        np.testing.assert_array_equal(core_grid["scores"], [0.0])

    def test_native_planner_limits_ranked_cells_without_reducing_row_capacity(self):
        pos = np.r_[np.arange(5, dtype=float), np.zeros(5)]
        result = adaptive_padding_cpp.plan_padding(
            np.array([[4.0]]), np.zeros((1, 1)),
            np.array([0.0, 30.0]), np.array([0.0, 1.0]),
            pos, pos, np.ones(5), np.ones(5), np.ones(5),
            np.array([[0.0, 0.0, 30.0, 1.0]]),
            np.ones(5, dtype=bool), 5, 5, 5,
            0.0, 0.0, 30.0, 1.0, 1.0, 1.0, 0.2, 0.2,
        )
        np.testing.assert_array_equal(result["padding_sites"], [1, 0, 0, 0, 0])
        self.assertEqual(result["total_free_sites"], 25)
        self.assertEqual(result["positive_score_count"], 5)

    def test_selects_top_positive_bins_and_maps_overlapping_cells(self):
        horizontal = torch.tensor([[0.0], [1.0], [3.0], [2.0]])
        vertical = torch.tensor([[0.0], [4.0], [0.0], [0.0]])
        hot = select_top_overflow_bins(horizontal, vertical, ratio=0.2)
        np.testing.assert_array_equal(hot, [[0.0], [4.0], [0.0], [0.0]])
        scores = score_cells_from_overflow(
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.5, 1.5]), torch.tensor([1.0, 1.0]),
            2, 2, hot, 0.0, 0.0, 4.0, 1.0,
        )
        np.testing.assert_array_equal(scores, [0.0, 4.0])

    def test_row_intervals_exclude_unplaceable_gaps(self):
        plan = allocate_padding_sites(
            scores=np.array([4.0, 10.0, 3.0]),
            pos=torch.tensor([1.0, 19.0, 23.0, 0.0, 0.0, 0.0]),
            node_size_x=torch.tensor([1.0, 1.0, 1.0]),
            node_size_y=torch.tensor([1.0, 1.0, 1.0]),
            num_nodes=3, num_movable_nodes=3, num_physical_nodes=3,
            xl=0.0, yl=0.0, xh=40.0, yh=1.0,
            site_width=1.0, row_height=1.0, hot_cell_ratio=1.0,
            row_free_ratio=1.0, max_padding_sites=1,
            rows=np.array([[0, 0, 18, 1], [22, 0, 40, 1]]),
        )
        self.assertEqual(plan["total_free_sites"], 34)
        self.assertEqual(plan["total_budget_sites"], 34)
        np.testing.assert_array_equal(plan["padding_sites"], [1, 0, 1])

    def test_three_rounds_keep_cumulative_sizes(self):
        params = SimpleNamespace(
            post_legalization_padding_rounds=3,
            post_legalization_padding_overflow_bin_ratio=0.2,
            post_legalization_padding_hot_cell_ratio=1.0,
            post_legalization_padding_row_free_ratio=1.0,
            post_legalization_padding_rrr_iters=0,
            shift_factor=[0, 0], scale_factor=1.0,
            result_dir="/tmp", num_threads=2, gpugr_backend="cpu_pr_mt",
            design_name=lambda: "tiny",
        )
        placedb = SimpleNamespace(
            regions=[], num_movable_nodes=1, num_nodes=1, num_physical_nodes=1,
            xl=0.0, yl=0.0, xh=30.0, yh=1.0, site_width=1.0,
            row_height=1.0, rows=np.array([[0.0, 0.0, 30.0, 1.0]]),
        )
        data = SimpleNamespace(
            node_size_x=torch.tensor([1.0]), node_size_y=torch.tensor([1.0]),
            movable_macro_mask=torch.tensor([False]),
        )
        router = mock.Mock()
        route_result = {
            "maps": {
                "cg_map_h_overflow": torch.tensor([[4.0], [0.0], [0.0], [0.0]]),
                "cg_map_v_overflow": torch.zeros((4, 1)),
            },
            "route_grid_edges_dbu": (np.linspace(0.0, 30.0, 5), np.array([0.0, 1.0])),
        }
        for fail_round, expected in ((None, [1, 2, 3]), (2, [1, 2])):
            requested = []

            def legalize(*, requested=requested, fail_round=fail_round, **kwargs):
                requested.append(kwargs["padding_sites"].copy())
                if len(requested) == fail_round:
                    return kwargs["pos"], {"rollback": True}
                return kwargs["pos"] + torch.tensor([1.0, 0.0]), {"rollback": False}

            model = SimpleNamespace(
                data_collections=data, run_adaptive_padding_legalization=legalize,
            )
            router.reset_mock()
            router.run_gpugr.side_effect = [
                {**route_result, "metrics": {"num_overflow_nets": 1, "gr_est_shorts": short}}
                for short in (4.0, 3.0, 2.0, 1.0)
            ]
            with (
                mock.patch.object(iterative, "write_back_movable_lpos"),
                mock.patch.object(iterative, "compute_route_grid_like_xplace", return_value=(4, 1)),
                mock.patch.object(iterative, "get_cached_gpugr_operator", return_value=router),
                self.assertLogs(level="INFO") as captured,
            ):
                result, stats = iterative.run_iterative_legalization_padding(
                    params, placedb, torch.tensor([0.0, 0.0]), model,
                )
            np.testing.assert_array_equal(requested, np.array(expected)[:, None])
            count = len(expected) if fail_round is None else fail_round - 1
            self.assertEqual(stats, {
                "completed_rounds": count, "cumulative_padding_sites": count,
                "effective_cells_by_round": [1] * count + ([0] if fail_round else []),
            })
            torch.testing.assert_close(result, torch.tensor([float(count), 0.0]))
            self.assertEqual(router.run_gpugr.call_count, count + 1)
            for round_id, effective in enumerate(stats["effective_cells_by_round"], 1):
                self.assertIn(
                    f"round {round_id}/3:", captured.output[round_id - 1],
                )
                self.assertIn(
                    f"effective_cells={effective}", captured.output[round_id - 1],
                )

    def test_worse_routing_restores_previous_legal_position(self):
        params = SimpleNamespace(
            post_legalization_padding_rounds=3,
            post_legalization_padding_overflow_bin_ratio=0.2,
            post_legalization_padding_hot_cell_ratio=1.0,
            shift_factor=[0, 0], scale_factor=1.0,
            result_dir="/tmp", num_threads=1, design_name=lambda: "tiny",
        )
        placedb = SimpleNamespace(
            regions=[], num_nodes=1, num_movable_nodes=1, num_physical_nodes=1,
            rows=np.array([[0.0, 0.0, 12.0, 1.0]]),
            xl=0.0, yl=0.0, xh=12.0, yh=1.0, site_width=1.0, row_height=1.0,
        )
        router = mock.Mock()
        route_result = {
            "maps": {
                "cg_map_h_overflow": np.array([[4.0]]),
                "cg_map_v_overflow": np.zeros((1, 1)),
            },
            "route_grid_edges_dbu": (np.array([0.0, 12.0]), np.array([0.0, 1.0])),
        }
        router.run_gpugr.side_effect = [
            {**route_result, "metrics": {"num_overflow_nets": 1, "gr_est_shorts": short}}
            for short in (2.0, 3.0)
        ]
        data = SimpleNamespace(node_size_x=torch.ones(1), node_size_y=torch.ones(1))
        model = SimpleNamespace(
            data_collections=data,
            run_adaptive_padding_legalization=lambda **kwargs: (
                kwargs["pos"] + torch.tensor([1.0, 0.0]), {"rollback": False}
            ),
        )
        with (
            mock.patch.object(iterative, "write_back_movable_lpos") as write_back,
            mock.patch.object(iterative, "compute_route_grid_like_xplace", return_value=(1, 1)),
            mock.patch.object(iterative, "get_cached_gpugr_operator", return_value=router),
            self.assertLogs(level="INFO") as captured,
        ):
            result, stats = iterative.run_iterative_legalization_padding(
                params, placedb, torch.tensor([0.0, 0.0]), model,
            )
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))
        self.assertEqual(stats["completed_rounds"], 0)
        self.assertEqual(stats["effective_cells_by_round"], [0])
        self.assertEqual(router.run_gpugr.call_count, 2)
        self.assertEqual(write_back.call_count, 3)
        self.assertIn("accepted=0", captured.output[0])

    def test_no_overflow_reports_zero_effective_cells(self):
        params = SimpleNamespace(
            post_legalization_padding_rounds=3,
            post_legalization_padding_overflow_bin_ratio=0.2,
            post_legalization_padding_hot_cell_ratio=1.0,
            shift_factor=[0, 0], scale_factor=1.0,
            result_dir="/tmp", num_threads=1, design_name=lambda: "tiny",
        )
        placedb = SimpleNamespace(
            regions=[], rows=[[0, 0, 10, 1]], num_movable_nodes=1,
            num_nodes=1, num_physical_nodes=1, xl=0.0, yl=0.0,
            xh=10.0, yh=1.0, site_width=1.0, row_height=1.0,
        )
        router = mock.Mock()
        router.run_gpugr.return_value = {
            "maps": {
                "cg_map_h_overflow": torch.zeros((2, 2)),
                "cg_map_v_overflow": torch.zeros((2, 2)),
            },
            "route_grid_edges_dbu": (np.linspace(0.0, 10.0, 3), np.linspace(0.0, 1.0, 3)),
        }
        model = SimpleNamespace(
            data_collections=SimpleNamespace(
                node_size_x=torch.ones(1), node_size_y=torch.ones(1),
            ),
            run_adaptive_padding_legalization=mock.Mock(),
        )
        with (
            mock.patch.object(iterative, "write_back_movable_lpos"),
            mock.patch.object(iterative, "compute_route_grid_like_xplace", return_value=(2, 2)),
            mock.patch.object(iterative, "get_cached_gpugr_operator", return_value=router),
            self.assertLogs(level="INFO") as captured,
        ):
            _, stats = iterative.run_iterative_legalization_padding(
                params, placedb, torch.tensor([0.0, 0.0]), model,
            )
        self.assertEqual(stats["effective_cells_by_round"], [0])
        self.assertIn("effective_cells=0", captured.output[0])
        model.run_adaptive_padding_legalization.assert_not_called()

    def test_full_row_budget_accounts_for_cumulative_padding(self):
        base_pos = torch.tensor([1.0, 0.0])
        size_x = torch.tensor([1.0])
        size_y = torch.tensor([1.0])
        padding = np.array([1])
        logical_pos = base_pos.clone()
        logical_pos[0] -= 1.0
        logical_size_x = size_x + 2 * torch.tensor(padding, dtype=size_x.dtype)
        plan = allocate_padding_sites(
            scores=np.array([3.0]), pos=logical_pos,
            node_size_x=logical_size_x, node_size_y=size_y,
            num_nodes=1, num_movable_nodes=1, num_physical_nodes=1,
            xl=0.0, yl=0.0, xh=12.0, yh=1.0,
            site_width=1.0, row_height=1.0, hot_cell_ratio=1.0,
            row_free_ratio=1.0, max_padding_sites=1,
            rows=np.array([[0.0, 0.0, 12.0, 1.0]]),
        )
        self.assertEqual(plan["total_free_sites"], 9)
        self.assertEqual(plan["total_budget_sites"], 9)
        np.testing.assert_array_equal(plan["padding_sites"], [1])


if __name__ == "__main__":
    unittest.main()
