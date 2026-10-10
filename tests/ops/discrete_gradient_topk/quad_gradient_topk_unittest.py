import math
import unittest
from types import SimpleNamespace

import torch
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.discrete_gradient_topk import sizes_to_logits
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    SIZE_DOWN,
    SIZE_UP,
    VT_SLOWDOWN,
    VT_SPEEDUP,
    _select_shared_budget,
    apply_discrete_quad_gradient_topk_update,
    build_quad_candidate_table,
)


class QuadGradientTopkTest(unittest.TestCase):
    def test_real_size_gradient_selects_one_legal_move_in_shared_cell_budget(self):
        real_size = torch.ones(2, dtype=torch.float64, requires_grad=True)
        vt_logits = torch.tensor([[math.log(1e-3), 0.0, math.log(1e-3)]] * 2,
                                 dtype=torch.float64, requires_grad=True)
        probability = torch.softmax(vt_logits, dim=1)
        loss = -4 * real_size[0] - real_size[1] - 3 * probability[1, 2]
        size_grad, vt_grad = torch.autograd.grad(loss, (real_size, vt_logits))
        next_size, next_vt, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=real_size.detach(), size_grad=size_grad,
            size_parameterization="real_size", vt_logits=vt_logits.detach(), vt_grad=vt_grad,
            inst_size_lower=torch.zeros(2, dtype=torch.float64),
            inst_size_upper=torch.full((2,), 2.0, dtype=torch.float64),
            inst_vt_mask=torch.ones((2, 3), dtype=torch.bool),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool), inst_cell_id=torch.tensor([1, 1]),
            flat_libcell_info=self._flat_info(),
            flat_libcell_leakage=torch.tensor([1., 2., 4., 2., 4., 8.], dtype=torch.float64),
            size_up_percent=0.0, size_down_percent=0.0, vt_percent=0.0,
            shared_budget_percent=1.0,
        )
        torch.testing.assert_close(
            (next_size, next_vt.argmax(dim=1)),
            (torch.tensor([2., 1.], dtype=torch.float64), torch.tensor([1, 1])),
        )
        self.assertEqual({key: summary[key] for key in (
            "applied_actions", "applied_instance_ids", "applied_cell_ids", "shared_budget_cells",
        )}, {
            "applied_actions": ["size_up"], "applied_instance_ids": [0],
            "applied_cell_ids": [4], "shared_budget_cells": 1,
        })

    def _flat_info(self):
        rows = []
        for size in (1.0, 2.0):
            for vt in (0.0, 1.0, 2.0):
                rows.append([float(len(rows)), 10.0, size, vt])
        return torch.tensor(rows, dtype=torch.float64)

    def test_candidate_table_keeps_size_and_vt_actions_on_separate_axes(self):
        table = build_quad_candidate_table(self._flat_info())
        current_cell = 1  # size=1, vt=1

        self.assertEqual(int(table.target_cell_ids[current_cell, SIZE_UP]), 4)
        self.assertEqual(int(table.target_cell_ids[current_cell, SIZE_DOWN]), -1)
        self.assertEqual(int(table.target_cell_ids[current_cell, VT_SLOWDOWN]), 0)
        self.assertEqual(int(table.target_cell_ids[current_cell, VT_SPEEDUP]), 2)

    def test_shared_budget_uses_sizeable_cells_and_one_best_action_per_cell(self):
        scores = torch.tensor(
            [[-5.0, 1.0, 1.0, -6.0], [-6.0, 1.0, 1.0, -5.0]]
            + [[-4.0, 1.0, 1.0, 1.0]] * 9
            + [[-100.0, 1.0, 1.0, 1.0]],
        )
        sizeable = torch.tensor([True] * 11 + [False])
        valid = sizeable[:, None].expand_as(scores)

        selected, budget = _select_shared_budget(scores, valid, sizeable, 10.0)

        self.assertEqual(budget, 2)  # ceil(11 * 10%), not 10% of candidates
        self.assertEqual(torch.nonzero(selected).tolist(), [[0, VT_SPEEDUP], [1, SIZE_UP]])
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(int(selected.sum(dim=1).max()), 1)
        self.assertEqual(int(_select_shared_budget(scores, valid, sizeable, 0.0)[0].sum()), 0)

        only_one = valid.clone()
        only_one[1:] = False
        selected, budget = _select_shared_budget(scores, only_one, sizeable, 10.0)
        self.assertEqual(budget, 2)
        self.assertEqual(int(selected.sum()), 1)

    def test_shared_budget_matches_independent_single_move_objective_deltas(self):
        lower = torch.zeros(2, dtype=torch.float64)
        upper = torch.full((2,), 2.0, dtype=torch.float64)
        size_logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        vt_logits = torch.tensor(
            [[math.log(1.0e-3), 0.0, math.log(1.0e-3)]] * 2,
            dtype=torch.float64,
            requires_grad=True,
        )
        size = lower + torch.sigmoid(size_logits) * (upper - lower)
        probability = torch.softmax(vt_logits, dim=1)
        objective = -4.0 * size[0] - size[1] - 2.0 * probability[0, 2] - 3.0 * probability[1, 2]
        size_grad, vt_grad = torch.autograd.grad(objective, (size_logits, vt_logits))

        _, _, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=size_logits.detach(),
            size_grad=size_grad,
            vt_logits=vt_logits.detach(),
            vt_grad=vt_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((2, 3), dtype=torch.bool),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([1, 1]),
            flat_libcell_info=self._flat_info(),
            flat_libcell_leakage=torch.tensor([1., 2., 4., 2., 4., 8.], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=0.0,
            shared_budget_percent=50.0,
        )

        self.assertEqual(summary["selection_policy"], "shared_cell_budget")
        self.assertEqual(summary["shared_budget_cells"], 1)
        self.assertEqual(summary["applied_actions"], ["size_up"])
        self.assertEqual(summary["applied_instance_ids"], [0])
        exact_size_delta = -4.0 * (2.0 - float(size[0].detach()))
        exact_vt_delta = -3.0 * (1.0 - float(probability[1, 2].detach()))
        self.assertAlmostEqual(
            summary["selected_scores"][0]["predicted_delta_obj"], exact_size_delta
        )
        self.assertLess(exact_size_delta, exact_vt_delta)

        _, _, all_actions = apply_discrete_quad_gradient_topk_update(
            size_logits=size_logits.detach(),
            size_grad=size_grad,
            vt_logits=vt_logits.detach(),
            vt_grad=vt_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((2, 3), dtype=torch.bool),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([1, 1]),
            flat_libcell_info=self._flat_info(),
            flat_libcell_leakage=torch.tensor([1., 2., 4., 2., 4., 8.], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=0.0,
            shared_budget_percent=100.0,
        )
        self.assertEqual(all_actions["applied_actions"], ["size_up", "vt_speedup"])
        self.assertAlmostEqual(
            all_actions["selected_scores"][1]["predicted_delta_obj"],
            exact_vt_delta,
            delta=0.01,
        )

    def test_shared_budget_single_vt_selects_only_size_neighbors(self):
        flat_info = torch.tensor(
            [[0., 10., 1., 0.], [1., 10., 2., 0.]], dtype=torch.float64
        )
        size_logits = torch.zeros(1, dtype=torch.float64)
        next_sizes, next_vts, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.tensor([-1.0], dtype=torch.float64),
            vt_logits=torch.zeros((1, 1), dtype=torch.float64),
            vt_grad=torch.zeros((1, 1), dtype=torch.float64),
            inst_size_lower=torch.zeros(1, dtype=torch.float64),
            inst_size_upper=torch.full((1,), 2.0, dtype=torch.float64),
            inst_vt_mask=torch.ones((1, 1), dtype=torch.bool),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([0]),
            flat_libcell_info=flat_info,
            flat_libcell_leakage=torch.tensor([1., 2.], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=0.0,
            shared_budget_percent=10.0,
        )
        self.assertEqual(summary["applied_actions"], ["size_up"])
        self.assertTrue(torch.equal(next_vts, torch.zeros((1, 1), dtype=torch.float64)))
        self.assertFalse(torch.equal(next_sizes, size_logits))

    def test_frozen_cell_is_excluded_before_topk_and_budget_refills(self):
        _, _, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=torch.zeros(2, dtype=torch.float64),
            size_grad=torch.tensor([-2.0, -1.0], dtype=torch.float64),
            vt_logits=torch.zeros((2, 1), dtype=torch.float64),
            vt_grad=torch.zeros((2, 1), dtype=torch.float64),
            inst_size_lower=torch.zeros(2, dtype=torch.float64),
            inst_size_upper=torch.full((2,), 2.0, dtype=torch.float64),
            inst_vt_mask=torch.ones((2, 1), dtype=torch.bool),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([0, 0]),
            flat_libcell_info=torch.tensor(
                [[0., 10., 1., 0.], [1., 10., 2., 0.]], dtype=torch.float64
            ),
            flat_libcell_leakage=torch.tensor([1., 2.], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=0.0,
            shared_budget_percent=50.0,
            blocked_instances=torch.tensor([True, False]),
        )
        self.assertEqual(summary["applied_instance_ids"], [1])
        self.assertEqual(summary["shared_budget_cells"], 1)
        self.assertEqual(summary["num_blocked_instances"], 1)
        self.assertEqual(summary["num_prevented_moves"], 1)
        self.assertEqual(summary["num_replacement_actions"], 1)

    def test_vt_neighbors_follow_leakage_not_backend_integer_order(self):
        flat_info = torch.tensor(
            [
                [0.0, 10.0, 1.0, 0.0],
                [1.0, 10.0, 1.0, 1.0],
                [2.0, 10.0, 1.0, 2.0],
            ],
            dtype=torch.float64,
        )
        table = build_quad_candidate_table(
            flat_info,
            torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64),
        )

        self.assertEqual(int(table.target_cell_ids[2, VT_SLOWDOWN]), 0)
        self.assertEqual(int(table.target_cell_ids[2, VT_SPEEDUP]), 1)

    def test_vt_gradient_commits_exact_neighbor_without_changing_size(self):
        flat_info = self._flat_info()
        lower = torch.tensor([1.0], dtype=torch.float64)
        upper = torch.tensor([2.0], dtype=torch.float64)
        size_logits = sizes_to_logits(torch.tensor([1.0], dtype=torch.float64), lower, upper)
        vt_logits = torch.tensor(
            [[math.log(1.0e-6), 0.0, math.log(1.0e-6)]],
            dtype=torch.float64,
            requires_grad=True,
        )
        vt_probs = torch.softmax(vt_logits, dim=1)
        (-vt_probs[:, 2].sum()).backward()

        next_sizes, next_vts, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.zeros_like(size_logits),
            vt_logits=vt_logits.detach(),
            vt_grad=vt_logits.grad.detach(),
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((1, 3), dtype=torch.bool),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([1], dtype=torch.long),
            flat_libcell_info=flat_info,
            flat_libcell_leakage=torch.tensor([1.0, 2.0, 4.0, 2.0, 4.0, 8.0], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=100.0,
        )

        self.assertTrue(torch.allclose(next_sizes, size_logits))
        self.assertEqual(int(torch.argmax(next_vts, dim=1)[0]), 2)
        self.assertEqual(summary["applied_cell_ids"], [2])
        self.assertEqual(summary["applied_actions"], ["vt_speedup"])
        self.assertEqual(summary["num_vt_applied"], 1)

    def test_missing_same_size_vt_neighbor_does_not_create_coupled_action(self):
        flat_info = torch.tensor(
            [
                [0.0, 10.0, 1.0, 0.0],
                [1.0, 10.0, 2.0, 0.0],
                [2.0, 10.0, 2.0, 1.0],
            ],
            dtype=torch.float64,
        )
        table = build_quad_candidate_table(flat_info)

        self.assertEqual(int(table.target_cell_ids[0, SIZE_UP]), 1)
        self.assertEqual(int(table.target_cell_ids[0, VT_SPEEDUP]), -1)
        self.assertEqual(int(table.target_cell_ids[1, VT_SPEEDUP]), 2)

    def test_non_sizeable_instance_may_have_no_library_cell(self):
        flat_info = self._flat_info()
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([2.0, 2.0], dtype=torch.float64)
        size_logits = sizes_to_logits(lower, lower, upper)
        vt_logits = torch.tensor(
            [
                [math.log(1.0e-6), 0.0, math.log(1.0e-6)],
                [0.0, math.log(1.0e-6), math.log(1.0e-6)],
            ],
            dtype=torch.float64,
        )

        next_sizes, next_vts, summary = apply_discrete_quad_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.zeros_like(size_logits),
            vt_logits=vt_logits,
            vt_grad=torch.zeros_like(vt_logits),
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((2, 3), dtype=torch.bool),
            inst_is_sizeable=torch.tensor([True, False]),
            inst_cell_id=torch.tensor([1, -1], dtype=torch.long),
            flat_libcell_info=flat_info,
            flat_libcell_leakage=torch.tensor([1.0, 2.0, 4.0, 2.0, 4.0, 8.0], dtype=torch.float64),
            size_up_percent=0.0,
            size_down_percent=0.0,
            vt_percent=100.0,
        )

        self.assertTrue(torch.equal(next_sizes, size_logits))
        self.assertTrue(torch.equal(next_vts, vt_logits))
        self.assertEqual(summary["num_sizeable_instances"], 1)

    def test_post_adam_dynamics_discards_continuous_vt_step_and_commits_neighbor(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        lower = torch.tensor([1.0], dtype=torch.float64)
        upper = torch.tensor([2.0], dtype=torch.float64)
        size_logits = torch.nn.Parameter(
            sizes_to_logits(torch.tensor([1.0], dtype=torch.float64), lower, upper)
        )
        vt_logits = torch.nn.Parameter(
            torch.tensor(
                [[math.log(1.0e-6), 0.0, math.log(1.0e-6)]],
                dtype=torch.float64,
            )
        )
        before = {
            "size_logits": size_logits.detach().clone(),
            "vt_logits": vt_logits.detach().clone(),
        }
        size_logits.grad = torch.zeros_like(size_logits)
        vt_probs = torch.softmax(vt_logits, dim=1)
        vt_logits.grad = torch.autograd.grad(-vt_probs[:, 2].sum(), vt_logits)[0]
        with torch.no_grad():
            size_logits.add_(0.25)
            vt_logits.add_(0.5)

        placer.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((1, 3), dtype=torch.bool),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([1], dtype=torch.long),
            flat_libcell_info=self._flat_info(),
            flat_libcell_leakage=torch.tensor([1.0, 2.0, 4.0, 2.0, 4.0, 8.0], dtype=torch.float64),
        )
        placer.discrete_gradient_topk_candidate_cache = None
        placer.discrete_gradient_topk_candidate_cache_key = None
        placer._sync_discrete_gradient_topk_runtime_cells = lambda _params, _summary: {
            "runtime_cell_state_synced": True
        }
        placer._should_skip_discrete_gradient_topk_artifact_write = lambda _params: True
        params = SimpleNamespace(
            continuous_size_dynamics_mode="discrete_gradient_topk",
            discrete_gradient_topk_up_percent=0.0,
            discrete_gradient_topk_down_percent=0.0,
            discrete_gradient_topk_vt_percent=100.0,
        )

        summary = placer._apply_continuous_size_logits_dynamics(
            params=params,
            before_snapshot=before,
            iteration=0,
            total_iterations=10,
        )

        self.assertTrue(torch.allclose(size_logits, before["size_logits"]))
        self.assertEqual(int(torch.argmax(vt_logits, dim=1)[0]), 2)
        self.assertEqual(summary["discrete_gradient_topk"]["applied_cell_ids"], [2])
        self.assertEqual(summary["discrete_gradient_topk_num_vt_applied"], 1)

    def test_shared_budget_uses_default_ten_percent_in_single_vt_runtime(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        lower = torch.zeros(1, dtype=torch.float64)
        upper = torch.full((1,), 2.0, dtype=torch.float64)
        size_logits = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        vt_logits = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        size_logits.grad = torch.tensor([-1.0], dtype=torch.float64)
        placer.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_vt_mask=torch.ones((1, 1), dtype=torch.bool),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([0]),
            flat_libcell_info=torch.tensor(
                [[0., 10., 1., 0.], [1., 10., 2., 0.]], dtype=torch.float64
            ),
            flat_libcell_leakage=torch.tensor([1., 2.], dtype=torch.float64),
        )
        placer.discrete_gradient_topk_candidate_cache = None
        placer.discrete_gradient_topk_candidate_cache_key = None
        placer._sync_discrete_gradient_topk_runtime_cells = lambda _params, _summary: {
            "runtime_cell_state_synced": True
        }
        placer._should_skip_discrete_gradient_topk_artifact_write = lambda _params: True
        params = SimpleNamespace(
            placement_sizing_mode="size_only",
            continuous_size_dynamics_mode="discrete_gradient_topk",
            discrete_gradient_topk_shared_budget=True,
        )
        config = placer._continuous_size_dynamics_config(params)
        self.assertEqual(config["discrete_gradient_topk_shared_budget_percent"], 1.0)

        summary = placer._apply_continuous_size_logits_dynamics(
            params=params,
            before_snapshot={"size_logits": size_logits.detach().clone(),
                             "vt_logits": vt_logits.detach().clone()},
            iteration=0,
            total_iterations=10,
        )["discrete_gradient_topk"]

        self.assertEqual(summary["applied_actions"], ["size_up"])
        self.assertEqual(summary["shared_budget_cells"], 1)
        self.assertEqual(summary["config"]["shared_budget_percent"], 1.0)

    def test_single_vt_library_keeps_legacy_size_only_selector(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        lower = torch.tensor([1.0], dtype=torch.float64)
        upper = torch.tensor([2.0], dtype=torch.float64)
        size_logits = torch.nn.Parameter(
            sizes_to_logits(torch.tensor([1.0], dtype=torch.float64), lower, upper)
        )
        vt_logits = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        before = {
            "size_logits": size_logits.detach().clone(),
            "vt_logits": vt_logits.detach().clone(),
        }
        size_logits.grad = torch.tensor([-1.0], dtype=torch.float64)
        vt_logits.grad = torch.zeros_like(vt_logits)
        placer.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
        )
        placer._select_discrete_gradient_topk = lambda **_kwargs: (
            before["size_logits"],
            torch.tensor([1.0], dtype=torch.float64),
            {
                "mode": "discrete_gradient_topk",
                "num_up_applied": 0,
                "num_down_applied": 0,
                "num_changed_instances": 0,
            },
            None,
        )
        placer._sync_discrete_gradient_topk_runtime_cells = lambda _params, _summary: {
            "runtime_cell_state_synced": False
        }
        placer._should_skip_discrete_gradient_topk_artifact_write = lambda _params: True
        params = SimpleNamespace(
            continuous_size_dynamics_mode="discrete_gradient_topk",
            discrete_gradient_topk_up_percent=30.0,
            discrete_gradient_topk_down_percent=0.0,
            discrete_gradient_topk_vt_percent=100.0,
        )

        summary = placer._apply_continuous_size_logits_dynamics(
            params=params,
            before_snapshot=before,
            iteration=0,
            total_iterations=10,
        )

        self.assertEqual(
            summary["discrete_gradient_topk"]["config"]["ranking_mode"],
            "real_size_taylor",
        )
        self.assertTrue(
            summary["discrete_gradient_topk"]["config"]["preserve_vt"]
        )
        self.assertTrue(torch.equal(vt_logits, before["vt_logits"]))


if __name__ == "__main__":
    unittest.main()
