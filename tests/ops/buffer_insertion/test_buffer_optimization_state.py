import unittest

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
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



if __name__ == "__main__":
    unittest.main()
