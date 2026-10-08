import unittest
from unittest import mock

import torch

import dreamplace.ops.discrete_gradient_topk.discrete_gradient_topk as topk_impl
from dreamplace.ops.discrete_gradient_topk.discrete_gradient_topk import (
    apply_discrete_gradient_topk_update,
    build_discrete_gradient_topk_candidate_cache,
    build_family_candidate_table,
    build_instance_candidate_table,
    sizes_to_logits,
)


class DiscreteGradientTopkHelperTest(unittest.TestCase):
    def _flat_libcell_info(self):
        # Columns follow the sizing bridge convention used by the optimizer plan:
        # [unused, main_id, size, vt].
        return torch.tensor(
            [
                [0.0, 10.0, 1.0, 0.0],
                [0.0, 10.0, 2.0, 0.0],
                [0.0, 10.0, 4.0, 0.0],
                [0.0, 10.0, 8.0, 1.0],
                [0.0, 20.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )

    def test_candidate_table_precomputes_instance_specific_legal_logits(self):
        flat_info = self._flat_libcell_info()
        inst_cell_id = torch.tensor([0, 1], dtype=torch.long)
        lower = torch.tensor([1.0, 0.5], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0], dtype=torch.float64)

        family_table = build_family_candidate_table(flat_info, preserve_vt=True)
        instance_table = build_instance_candidate_table(
            family_table=family_table,
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            inst_size_lower=lower,
            inst_size_upper=upper,
            eps=1e-6,
        )

        size_two_logits = instance_table["legal_logits"][:, 1]
        expected = sizes_to_logits(
            torch.tensor([2.0, 2.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )

        self.assertTrue(torch.allclose(size_two_logits, expected))
        self.assertNotAlmostEqual(float(size_two_logits[0]), float(size_two_logits[1]))

    def test_delta_logit_is_indexed_from_precomputed_legal_logits(self):
        flat_info = self._flat_libcell_info()
        inst_cell_id = torch.tensor([1], dtype=torch.long)
        lower = torch.tensor([1.0], dtype=torch.float64)
        upper = torch.tensor([4.0], dtype=torch.float64)
        family_table = build_family_candidate_table(flat_info, preserve_vt=True)
        instance_table = build_instance_candidate_table(
            family_table=family_table,
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            inst_size_lower=lower,
            inst_size_upper=upper,
            eps=1e-6,
        )

        current_index = int(instance_table["current_legal_index"][0])
        current_logit = instance_table["legal_logits"][0, current_index]
        target_logit = instance_table["legal_logits"][0, current_index + 1]
        indexed_delta = target_logit - current_logit

        self.assertAlmostEqual(
            float(indexed_delta),
            float(instance_table["legal_logits"][0, 2] - instance_table["legal_logits"][0, 1]),
        )

    def test_cached_candidate_table_matches_uncached_reference_after_cell_update(self):
        flat_info = self._flat_libcell_info()
        lower = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
        size_logits = sizes_to_logits(
            torch.tensor([2.0, 2.0, 4.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )
        size_grad = torch.tensor([-1.0, -0.5, 1.5], dtype=torch.float64)
        cached_from_initial_state = build_discrete_gradient_topk_candidate_cache(
            inst_cell_id=torch.tensor([0, 0, 2], dtype=torch.long),
            flat_libcell_info=flat_info,
            inst_size_lower=lower,
            inst_size_upper=upper,
            eps=1e-6,
            preserve_vt=True,
        )
        updated_inst_cell_id = torch.tensor([1, 1, 2], dtype=torch.long)

        uncached_logits, uncached_sizes, uncached_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(3, dtype=torch.bool),
            inst_cell_id=updated_inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
        )
        cached_logits, cached_sizes, cached_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(3, dtype=torch.bool),
            inst_cell_id=updated_inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=cached_from_initial_state,
        )

        self.assertTrue(torch.allclose(cached_logits, uncached_logits))
        self.assertTrue(torch.allclose(cached_sizes, uncached_sizes))
        self.assertEqual(cached_summary["applied_instance_ids"], uncached_summary["applied_instance_ids"])
        self.assertEqual(cached_summary["applied_cell_ids"], uncached_summary["applied_cell_ids"])
        self.assertEqual(cached_summary["ranking"], uncached_summary["ranking"])

    def test_tensor_selection_backend_matches_python_reference_for_logit_taylor(self):
        flat_info = self._flat_libcell_info()
        lower = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
        size_logits = sizes_to_logits(
            torch.tensor([1.0, 2.0, 4.0, 2.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )
        size_grad = torch.tensor([-1.0, -0.25, 0.75, 0.5], dtype=torch.float64)
        inst_cell_id = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        instance_table = build_discrete_gradient_topk_candidate_cache(
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            inst_size_lower=lower,
            inst_size_upper=upper,
            eps=1e-6,
            preserve_vt=True,
        )

        ranking_mode = "taylor_delta_logit"
        reference_logits, reference_sizes, reference_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(4, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode=ranking_mode,
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="python",
        )
        tensor_logits, tensor_sizes, tensor_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(4, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode=ranking_mode,
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertTrue(torch.allclose(tensor_logits, reference_logits))
        self.assertTrue(torch.allclose(tensor_sizes, reference_sizes))
        self.assertEqual(tensor_summary["applied_instance_ids"], reference_summary["applied_instance_ids"])
        self.assertEqual(tensor_summary["applied_cell_ids"], reference_summary["applied_cell_ids"])
        self.assertEqual(
            tensor_summary["applied_legal_index_deltas"],
            reference_summary["applied_legal_index_deltas"],
        )
        self.assertEqual(tensor_summary["ranking"]["mode"], ranking_mode)
        self.assertEqual(
            tensor_summary["ranking"]["num_candidate_moves"],
            reference_summary["ranking"]["num_candidate_moves"],
        )
        self.assertEqual(
            tensor_summary["ranking"]["num_improving_candidates"],
            reference_summary["ranking"]["num_improving_candidates"],
        )
        self.assertEqual(
            tensor_summary["ranking"]["num_non_improving_selected"],
            reference_summary["ranking"]["num_non_improving_selected"],
        )
        for key in (
            "predicted_delta_obj_min",
            "predicted_delta_obj_mean",
            "predicted_delta_obj_max",
            "predicted_improvement_threshold",
            "delta_logit_mean_abs",
            "delta_logit_max_abs",
        ):
            self.assertAlmostEqual(
                tensor_summary["ranking"][key],
                reference_summary["ranking"][key],
            )

    def test_tensor_selection_backend_matches_python_reference_for_raw_gradient(self):
        flat_info = self._flat_libcell_info()
        lower = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
        size_logits = sizes_to_logits(
            torch.tensor([1.0, 1.0, 4.0, 2.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )
        size_grad = torch.tensor([-2.0, -0.5, 1.5, 0.75], dtype=torch.float64)
        inst_cell_id = torch.tensor([0, 0, 2, 1], dtype=torch.long)
        instance_table = build_discrete_gradient_topk_candidate_cache(
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            inst_size_lower=lower,
            inst_size_upper=upper,
            eps=1e-6,
            preserve_vt=True,
        )

        reference_logits, reference_sizes, reference_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(4, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=50.0,
            down_percent=50.0,
            preserve_vt=True,
            ranking_mode="raw_gradient",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="python",
        )
        tensor_logits, tensor_sizes, tensor_summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(4, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=50.0,
            down_percent=50.0,
            preserve_vt=True,
            ranking_mode="raw_gradient",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertTrue(torch.allclose(tensor_logits, reference_logits))
        self.assertTrue(torch.allclose(tensor_sizes, reference_sizes))
        self.assertEqual(tensor_summary["ranking"]["selection_backend"], "tensor")
        self.assertEqual(tensor_summary["applied_instance_ids"], reference_summary["applied_instance_ids"])
        self.assertEqual(tensor_summary["applied_cell_ids"], reference_summary["applied_cell_ids"])
        self.assertEqual(
            tensor_summary["applied_legal_index_deltas"],
            reference_summary["applied_legal_index_deltas"],
        )
        self.assertGreater(tensor_summary["ranking"]["num_candidate_moves"], 0)
        self.assertGreater(tensor_summary["ranking"]["num_improving_candidates"], 0)
        self.assertEqual(tensor_summary["ranking"]["num_non_improving_selected"], 0)

    def test_raw_gradient_ranking_keeps_legacy_up_down_order_for_ablation(self):
        flat_info = self._flat_libcell_info()
        size_logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        size_grad = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float64)
        inst_cell_id = torch.tensor([0, 0, 2], dtype=torch.long)

        target_logits, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=torch.ones(3, dtype=torch.float64),
            inst_size_upper=torch.full((3,), 4.0, dtype=torch.float64),
            inst_is_sizeable=torch.ones(3, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=50.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode="raw_gradient",
            step_mode="direct_one_step",
        )

        self.assertGreater(float(target_sizes[0]), 1.0)
        self.assertEqual(float(target_sizes[1]), 1.0)
        self.assertLess(float(target_sizes[2]), 4.0)
        self.assertEqual(summary["num_up_selected"], 1)
        self.assertEqual(summary["num_down_selected"], 1)
        self.assertEqual(summary["ranking"]["mode"], "raw_gradient")
        self.assertEqual(target_logits.dtype, size_logits.dtype)

    def test_taylor_delta_logit_ranks_by_grad_times_delta_logit_not_raw_grad(self):
        flat_info = self._flat_libcell_info()
        inst_cell_id = torch.tensor([0, 1], dtype=torch.long)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0], dtype=torch.float64)
        # Instance 0 has the larger raw gradient magnitude, but it is at size 1
        # where the next logit step is smaller than instance 1's size 2 -> 4 step.
        size_logits = sizes_to_logits(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )
        size_grad = torch.tensor([-2.0, -1.9], dtype=torch.float64)

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=50.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="taylor_delta_logit",
            step_mode="direct_one_step",
        )

        self.assertEqual(float(target_sizes[0]), 1.0)
        self.assertEqual(float(target_sizes[1]), 4.0)
        self.assertLess(summary["ranking"]["predicted_delta_obj_min"], 0.0)
        self.assertEqual(summary["ranking"]["num_non_improving_selected"], 0)

    def test_taylor_delta_candidate_is_retired(self):
        flat_info = self._flat_libcell_info()
        with self.assertRaisesRegex(ValueError, "retired; use real_size_taylor"):
            apply_discrete_gradient_topk_update(
                size_logits=torch.tensor([0.0], dtype=torch.float64),
                size_grad=torch.tensor([-1.0], dtype=torch.float64),
                inst_size_lower=torch.tensor([1.0], dtype=torch.float64),
                inst_size_upper=torch.tensor([4.0], dtype=torch.float64),
                inst_is_sizeable=torch.ones(1, dtype=torch.bool),
                inst_cell_id=torch.tensor([0], dtype=torch.long),
                flat_libcell_info=flat_info,
                up_percent=100.0,
                down_percent=0.0,
                preserve_vt=True,
                ranking_mode="taylor_delta_candidate",
                step_mode="direct_one_step",
            )

    def test_real_size_taylor_direct_one_step_ranks_by_applied_one_step_move(self):
        instance_table = {
            "legal_sizes": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=torch.float64,
            ),
            "legal_logits": torch.tensor(
                [
                    [0.0, 1.0, 100.0],
                    [0.0, 2.0, 3.0],
                ],
                dtype=torch.float64,
            ),
            "legal_cell_ids": torch.tensor(
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                dtype=torch.long,
            ),
            "candidate_mask": torch.ones((2, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([0, 0], dtype=torch.long),
        }

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=torch.tensor([0.0, 0.0], dtype=torch.float64),
            size_grad=torch.tensor([-1.0, -1.0], dtype=torch.float64),
            inst_size_lower=torch.tensor([1.0, 1.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([3.0, 3.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([10, 20], dtype=torch.long),
            flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
            up_percent=50.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertEqual(summary["applied_instance_ids"], [0])
        self.assertEqual(summary["num_up_improving_instances"], 2)
        self.assertEqual(summary["num_down_improving_instances"], 0)
        self.assertEqual(summary["ranking"]["num_up_improving_instances"], 2)
        self.assertEqual(summary["num_up_applied"], 1)
        self.assertEqual(summary["num_down_applied"], 0)
        self.assertEqual(float(target_sizes[0]), 2.0)
        self.assertEqual(float(target_sizes[1]), 1.0)

    def test_tensor_selection_skips_zero_percent_direction(self):
        instance_table = {
            "legal_sizes": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=torch.float64,
            ),
            "legal_logits": torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ],
                dtype=torch.float64,
            ),
            "legal_cell_ids": torch.tensor(
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                dtype=torch.long,
            ),
            "candidate_mask": torch.ones((2, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([1, 1], dtype=torch.long),
        }

        original = topk_impl._taylor_tensor_select_direction
        calls = []

        def record_direction(*, direction_mask, percent, **kwargs):
            calls.append(float(percent))
            return original(direction_mask=direction_mask, percent=percent, **kwargs)

        with mock.patch.object(
            topk_impl,
            "_taylor_tensor_select_direction",
            side_effect=record_direction,
        ):
            _, target_sizes, summary = apply_discrete_gradient_topk_update(
                size_logits=torch.tensor([0.0, 0.0], dtype=torch.float64),
                size_grad=torch.tensor([-1.0, 1.0], dtype=torch.float64),
                inst_size_lower=torch.tensor([1.0, 1.0], dtype=torch.float64),
                inst_size_upper=torch.tensor([3.0, 3.0], dtype=torch.float64),
                inst_is_sizeable=torch.ones(2, dtype=torch.bool),
                inst_cell_id=torch.tensor([11, 21], dtype=torch.long),
                flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
                up_percent=100.0,
                down_percent=0.0,
                preserve_vt=True,
                ranking_mode="real_size_taylor",
                step_mode="direct_one_step",
                instance_candidate_table=instance_table,
                selection_backend="tensor",
            )

        self.assertEqual(calls, [100.0])
        self.assertEqual(summary["num_down_selected"], 0)
        self.assertEqual(float(target_sizes[0]), 3.0)
        self.assertEqual(float(target_sizes[1]), 2.0)

    def test_real_size_taylor_ranks_by_real_size_delta_not_delta_logit(self):
        instance_table = {
            "legal_sizes": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 8.0, 9.0],
                ],
                dtype=torch.float64,
            ),
            "legal_logits": torch.tensor(
                [
                    [0.0, 100.0, 101.0],
                    [0.0, 2.0, 3.0],
                ],
                dtype=torch.float64,
            ),
            "legal_cell_ids": torch.tensor(
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                dtype=torch.long,
            ),
            "candidate_mask": torch.ones((2, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([0, 0], dtype=torch.long),
        }

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=torch.tensor([0.0, 0.0], dtype=torch.float64),
            size_grad=torch.tensor([-1.0, -0.3], dtype=torch.float64),
            inst_size_lower=torch.tensor([1.0, 1.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([9.0, 9.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([10, 20], dtype=torch.long),
            flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
            up_percent=50.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertEqual(summary["applied_instance_ids"], [1])
        self.assertEqual(summary["ranking"]["mode"], "real_size_taylor")
        self.assertEqual(summary["ranking"]["score_coordinate"], "real_size")
        self.assertEqual(summary["ranking"]["delta_value_name"], "delta_real_size")
        self.assertEqual(float(target_sizes[0]), 1.0)
        self.assertEqual(float(target_sizes[1]), 8.0)

    def test_timing_coordinate_taylor_alias_uses_real_size_taylor_path(self):
        instance_table = {
            "legal_sizes": torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 8.0, 9.0],
                ],
                dtype=torch.float64,
            ),
            "legal_logits": torch.tensor(
                [
                    [0.0, 100.0, 101.0],
                    [0.0, 2.0, 3.0],
                ],
                dtype=torch.float64,
            ),
            "legal_cell_ids": torch.tensor(
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                dtype=torch.long,
            ),
            "candidate_mask": torch.ones((2, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([0, 0], dtype=torch.long),
        }

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=torch.tensor([0.0, 0.0], dtype=torch.float64),
            size_grad=torch.tensor([-1.0, -0.3], dtype=torch.float64),
            inst_size_lower=torch.tensor([1.0, 1.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([9.0, 9.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([10, 20], dtype=torch.long),
            flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
            up_percent=50.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="timing_coordinate_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertEqual(summary["ranking_mode"], "timing_coordinate_taylor")
        self.assertEqual(summary["ranking"]["mode"], "timing_coordinate_taylor")
        self.assertEqual(summary["ranking"]["score_coordinate"], "real_size")
        self.assertEqual(float(target_sizes[0]), 1.0)
        self.assertEqual(float(target_sizes[1]), 8.0)

    def test_leakage_taylor_ranks_by_log_leakage_delta_not_real_size(self):
        instance_table = {
            "legal_sizes": torch.tensor(
                [
                    [0.67, 0.67, 2.0],
                    [1.0, 8.0, 9.0],
                ],
                dtype=torch.float64,
            ),
            "legal_logits": torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 100.0, 101.0],
                ],
                dtype=torch.float64,
            ),
            "legal_cell_ids": torch.tensor(
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                dtype=torch.long,
            ),
            "candidate_mask": torch.ones((2, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([0, 0], dtype=torch.long),
        }
        # The first row mimics HB-like names where parsed real_size can tie.
        # leakage_taylor must still see cell 11 as a valid stronger candidate.
        flat_libcell_leakage = torch.zeros(23, dtype=torch.float64)
        flat_libcell_leakage[10] = 3.0
        flat_libcell_leakage[11] = 4.0
        flat_libcell_leakage[12] = 5.0
        flat_libcell_leakage[20] = 1.0
        flat_libcell_leakage[21] = 10.0
        flat_libcell_leakage[22] = 11.0

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=torch.tensor([0.0, 0.0], dtype=torch.float64),
            size_grad=torch.tensor([-1.0, -0.1], dtype=torch.float64),
            inst_size_lower=torch.tensor([0.0, 0.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([10.0, 10.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([10, 20], dtype=torch.long),
            flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
            flat_libcell_leakage=flat_libcell_leakage,
            up_percent=50.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="leakage_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertEqual(summary["applied_instance_ids"], [0])
        self.assertEqual(summary["applied_cell_ids"], [11])
        self.assertEqual(summary["ranking"]["mode"], "leakage_taylor")
        self.assertEqual(summary["ranking"]["score_coordinate"], "log_leakage")
        self.assertEqual(summary["ranking"]["delta_value_name"], "delta_log_leakage")
        self.assertEqual(summary["ranking"]["leakage_coord_mode"], "log_leakage")
        self.assertIsNotNone(summary["ranking"]["delta_log_leakage_mean_abs"])
        self.assertGreater(summary["ranking"]["delta_log_leakage_max_abs"], 0.0)
        self.assertEqual(float(target_sizes[0]), 0.67)
        self.assertEqual(float(target_sizes[1]), 1.0)

    def test_leakage_taylor_direct_step_uses_leakage_direction_not_legal_index(self):
        instance_table = {
            "legal_sizes": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
            "legal_logits": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64),
            "legal_cell_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "candidate_mask": torch.ones((1, 3), dtype=torch.bool),
            "current_legal_index": torch.tensor([1], dtype=torch.long),
        }
        flat_libcell_leakage = torch.zeros(13, dtype=torch.float64)
        flat_libcell_leakage[10] = 5.0
        flat_libcell_leakage[11] = 3.0
        flat_libcell_leakage[12] = 2.0

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=torch.tensor([0.0], dtype=torch.float64),
            size_grad=torch.tensor([-1.0], dtype=torch.float64),
            inst_size_lower=torch.tensor([0.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([10.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([11], dtype=torch.long),
            flat_libcell_info=torch.empty((0, 4), dtype=torch.float64),
            flat_libcell_leakage=flat_libcell_leakage,
            up_percent=100.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="leakage_taylor",
            step_mode="direct_one_step",
            instance_candidate_table=instance_table,
            selection_backend="tensor",
        )

        self.assertEqual(summary["applied_cell_ids"], [10])
        self.assertEqual(summary["num_up_selected"], 1)
        self.assertEqual(summary["num_down_selected"], 0)
        self.assertEqual(summary["ranking"]["direction_coordinate"], "log_leakage")
        self.assertGreater(summary["ranking"]["selected_up_mean_delta_log_leakage"], 0.0)
        self.assertEqual(float(target_sizes[0]), 1.0)

    def test_leakage_taylor_requires_flat_libcell_leakage(self):
        with self.assertRaisesRegex(ValueError, "leakage_taylor requires flat_libcell_leakage"):
            apply_discrete_gradient_topk_update(
                size_logits=torch.tensor([0.0], dtype=torch.float64),
                size_grad=torch.tensor([-1.0], dtype=torch.float64),
                inst_size_lower=torch.tensor([0.0], dtype=torch.float64),
                inst_size_upper=torch.tensor([10.0], dtype=torch.float64),
                inst_is_sizeable=torch.ones(1, dtype=torch.bool),
                inst_cell_id=torch.tensor([0], dtype=torch.long),
                flat_libcell_info=self._flat_libcell_info(),
                up_percent=100.0,
                down_percent=0.0,
                preserve_vt=True,
                ranking_mode="leakage_taylor",
                step_mode="direct_one_step",
            )

    def test_summary_records_applied_cell_ids_after_direct_one_step_policy(self):
        flat_info = self._flat_libcell_info()
        inst_cell_id = torch.tensor([0], dtype=torch.long)
        lower = torch.tensor([1.0], dtype=torch.float64)
        upper = torch.tensor([4.0], dtype=torch.float64)
        size_logits = sizes_to_logits(
            torch.tensor([1.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.tensor([-1.0], dtype=torch.float64),
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
        )

        self.assertEqual(float(target_sizes[0]), 2.0)
        self.assertEqual(summary["applied_cell_ids"][0], 1)
        self.assertEqual(summary["applied_instance_ids"], [0])
        self.assertEqual(summary["applied_legal_index_deltas"], [1])
        self.assertEqual(summary["step"]["mean_abs_legal_index_delta"], 1.0)
        self.assertEqual(summary["step"]["max_abs_legal_index_delta"], 1)

    def test_taylor_delta_logit_records_delta_logit_and_predicted_delta_stats(self):
        flat_info = self._flat_libcell_info()
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        upper = torch.tensor([4.0, 4.0], dtype=torch.float64)
        size_logits = sizes_to_logits(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            lower,
            upper,
            eps=1e-6,
        )

        _, _, summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.tensor([-1.0, 0.5], dtype=torch.float64),
            inst_size_lower=lower,
            inst_size_upper=upper,
            inst_is_sizeable=torch.ones(2, dtype=torch.bool),
            inst_cell_id=torch.tensor([0, 1], dtype=torch.long),
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=100.0,
            preserve_vt=True,
            ranking_mode="taylor_delta_logit",
            step_mode="direct_one_step",
        )

        ranking = summary["ranking"]
        self.assertIsNotNone(ranking["predicted_delta_obj_min"])
        self.assertIsNotNone(ranking["predicted_delta_obj_mean"])
        self.assertIsNotNone(ranking["predicted_delta_obj_max"])
        self.assertIsNotNone(ranking["predicted_improvement_threshold"])
        self.assertIsNotNone(ranking["delta_logit_mean_abs"])
        self.assertIsNotNone(ranking["delta_logit_max_abs"])
        self.assertGreater(ranking["delta_logit_max_abs"], 0.0)

    def test_real_size_taylor_rejects_non_improving_candidate_by_default(self):
        flat_info = self._flat_libcell_info()
        size_logits = sizes_to_logits(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([4.0], dtype=torch.float64),
            eps=1e-6,
        )

        _, target_sizes, summary = apply_discrete_gradient_topk_update(
            size_logits=size_logits,
            size_grad=torch.tensor([0.5], dtype=torch.float64),
            inst_size_lower=torch.tensor([1.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([4.0], dtype=torch.float64),
            inst_is_sizeable=torch.ones(1, dtype=torch.bool),
            inst_cell_id=torch.tensor([1], dtype=torch.long),
            flat_libcell_info=flat_info,
            up_percent=100.0,
            down_percent=0.0,
            preserve_vt=True,
            ranking_mode="real_size_taylor",
            step_mode="direct_one_step",
        )

        self.assertEqual(float(target_sizes[0]), 2.0)
        self.assertEqual(summary["num_changed_instances"], 0)
        self.assertEqual(summary["ranking"]["num_non_improving_selected"], 0)

    def test_preserve_vt_filters_candidates_by_current_vt(self):
        flat_info = self._flat_libcell_info()
        family_table = build_family_candidate_table(flat_info, preserve_vt=True)
        instance_table = build_instance_candidate_table(
            family_table=family_table,
            inst_cell_id=torch.tensor([0], dtype=torch.long),
            flat_libcell_info=flat_info,
            inst_size_lower=torch.tensor([1.0], dtype=torch.float64),
            inst_size_upper=torch.tensor([8.0], dtype=torch.float64),
            eps=1e-6,
        )

        legal_cell_ids = instance_table["legal_cell_ids"][0][instance_table["candidate_mask"][0]]
        self.assertNotIn(3, [int(item) for item in legal_cell_ids.tolist()])

    def test_lagrangian_relaxation_step_mode_is_explicit_future_work(self):
        flat_info = self._flat_libcell_info()
        with self.assertRaisesRegex(NotImplementedError, "future work"):
            apply_discrete_gradient_topk_update(
                size_logits=torch.tensor([0.0], dtype=torch.float64),
                size_grad=torch.tensor([-1.0], dtype=torch.float64),
                inst_size_lower=torch.tensor([1.0], dtype=torch.float64),
                inst_size_upper=torch.tensor([4.0], dtype=torch.float64),
                inst_is_sizeable=torch.ones(1, dtype=torch.bool),
                inst_cell_id=torch.tensor([0], dtype=torch.long),
                flat_libcell_info=flat_info,
                up_percent=100.0,
                down_percent=0.0,
                preserve_vt=True,
                ranking_mode="real_size_taylor",
                step_mode="lagrangian_relaxation",
            )

    def test_invalid_step_mode_is_rejected(self):
        flat_info = self._flat_libcell_info()
        with self.assertRaisesRegex(ValueError, "invalid discrete gradient top-k step mode"):
            apply_discrete_gradient_topk_update(
                size_logits=torch.tensor([0.0], dtype=torch.float64),
                size_grad=torch.tensor([-1.0], dtype=torch.float64),
                inst_size_lower=torch.tensor([1.0], dtype=torch.float64),
                inst_size_upper=torch.tensor([4.0], dtype=torch.float64),
                inst_is_sizeable=torch.ones(1, dtype=torch.bool),
                inst_cell_id=torch.tensor([0], dtype=torch.long),
                flat_libcell_info=flat_info,
                up_percent=100.0,
                down_percent=0.0,
                preserve_vt=True,
                ranking_mode="real_size_taylor",
                step_mode="bad_step",
            )


if __name__ == "__main__":
    unittest.main()
