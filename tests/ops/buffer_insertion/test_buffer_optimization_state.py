import unittest

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    interpolate_buffer_values_by_bsu_index,
)


class BufferOptimizationStateTest(unittest.TestCase):
    def test_build_state_creates_trainable_candidate_variables(self):
        candidates = [
            {"candidate_id": 10, "tree_node_id": 3, "net_id": 7, "buffer_main_type_index": 1},
            {"candidate_id": 11, "tree_node_id": 4, "net_id": 7, "buffer_main_type_index": 1},
        ]

        state = build_buffer_optimization_state(
            candidates,
            buffer_main_type_index=1,
            legal_buffer_count=4,
            initial_bu_logit=-2.0,
            initial_bsu_index=1.25,
            dtype=torch.float64,
        )

        self.assertTrue(torch.equal(state.candidate_ids, torch.tensor([10, 11], dtype=torch.long)))
        self.assertTrue(torch.equal(state.candidate_node_id, torch.tensor([3, 4], dtype=torch.long)))
        self.assertTrue(torch.equal(state.candidate_net_id, torch.tensor([7, 7], dtype=torch.long)))
        self.assertEqual(tuple(state.bu_logits.shape), (2,))
        self.assertEqual(tuple(state.bsu_index_param.shape), (2,))
        self.assertTrue(state.bu_logits.requires_grad)
        self.assertTrue(state.bsu_index_param.requires_grad)
        self.assertAlmostEqual(float(state.relaxed_bu()[0]), float(torch.sigmoid(torch.tensor(-2.0))))
        self.assertTrue(torch.allclose(state.bsu_index(), torch.tensor([1.25, 1.25], dtype=torch.float64)))

    def test_state_rejects_wrong_buffer_family(self):
        candidates = [
            {"candidate_id": 1, "tree_node_id": 2, "net_id": 3, "buffer_main_type_index": 2}
        ]

        with self.assertRaisesRegex(ValueError, "buffer_main_type_index"):
            build_buffer_optimization_state(
                candidates,
                buffer_main_type_index=1,
                legal_buffer_count=4,
            )

    def test_bsu_index_interpolation_has_gradient_inside_legal_range(self):
        bsu_index = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.float64))
        per_size_input_cap = torch.tensor([[0.10, 0.20, 0.50, 0.90]], dtype=torch.float64)
        per_size_delay = torch.tensor([[10.0, 8.0, 4.0, 2.0]], dtype=torch.float64)
        per_size_output_slew = torch.tensor([[1.0, 0.8, 0.5, 0.4]], dtype=torch.float64)

        result = interpolate_buffer_values_by_bsu_index(
            bsu_index,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        loss = result["buffer_delay"].sum()
        loss.backward()

        self.assertAlmostEqual(float(result["buffer_delay"][0]), 7.0)
        self.assertIsNotNone(bsu_index.grad)
        self.assertAlmostEqual(float(bsu_index.grad[0]), -4.0)
        self.assertEqual(result["bsu_index_status"]["clamp"], "none")
        self.assertEqual(result["bsu_index_status"]["coordinate_source"], "current_relaxed")

    def test_bsu_index_interpolation_reports_guard_clamp(self):
        bsu_index = torch.nn.Parameter(torch.tensor([9.0], dtype=torch.float32))
        per_size = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        result = interpolate_buffer_values_by_bsu_index(
            bsu_index,
            per_size_input_cap=per_size,
            per_size_delay=per_size,
            per_size_output_slew=per_size,
        )

        self.assertTrue(result["bsu_index_status"]["clamped"])
        self.assertEqual(result["bsu_index_status"]["clamp"], "high")
        self.assertAlmostEqual(float(result["buffer_delay"][0]), 2.0)


if __name__ == "__main__":
    unittest.main()
