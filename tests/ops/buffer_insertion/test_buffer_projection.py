import unittest

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    project_buffer_optimization_state,
)
from dreamplace.ops.buffer_insertion.coordinate_backend import (
    build_coordinate_backend_contract,
)
from dreamplace.ops.physical_action import build_commit_action_candidates


class BufferProjectionTest(unittest.TestCase):
    def _candidates(self):
        return [
            {
                "candidate_id": 0,
                "tree_node_id": 10,
                "net_id": 1,
                "net_name": "n1",
                "driver_pin_id": 100,
                "driver_pin_name": "drv/Y",
                "load_pin_id": 200,
                "load_pin_name": "sink/A",
                "downstream_pin_ids": [200],
                "downstream_pin_names": ["sink/A"],
                "x_dbu": 100,
                "y_dbu": 200,
                "buffer_main_type_index": 1,
            },
            {
                "candidate_id": 1,
                "tree_node_id": 11,
                "net_id": 1,
                "net_name": "n1",
                "driver_pin_id": 100,
                "driver_pin_name": "drv/Y",
                "load_pin_id": 201,
                "load_pin_name": "sink2/A",
                "downstream_pin_ids": [201],
                "downstream_pin_names": ["sink2/A"],
                "x_dbu": 300,
                "y_dbu": 400,
                "buffer_main_type_index": 1,
            },
        ]

    def test_projection_selects_topk_and_rounds_legal_bsu(self):
        state = build_buffer_optimization_state(
            self._candidates(),
            buffer_main_type_index=1,
            legal_buffer_count=4,
        )
        with torch.no_grad():
            state.bu_logits.copy_(torch.tensor([-3.0, 3.0]))
            state.bsu_index_param.copy_(torch.tensor([0.25, 2.6]))

        virtual_state = project_buffer_optimization_state(state, top_k=1)

        self.assertEqual(virtual_state.virtual_buffer_count, 1)
        self.assertEqual(list(virtual_state.entries), [1])
        self.assertEqual(virtual_state.entries[1]["bsu"], 3)
        self.assertEqual(virtual_state.entries[1]["x_dbu"], 300)
        self.assertEqual(virtual_state.trace[0]["action"], "insert")

    def test_projection_threshold_filters_and_keeps_score_order(self):
        state = build_buffer_optimization_state(
            self._candidates(),
            buffer_main_type_index=1,
            legal_buffer_count=4,
        )
        with torch.no_grad():
            state.bu_logits.copy_(torch.tensor([-3.0, 3.0]))
            state.bsu_index_param.copy_(torch.tensor([1.0, 2.0]))

        virtual_state = project_buffer_optimization_state(state, threshold=0.5)

        self.assertEqual(virtual_state.virtual_buffer_count, 1)
        self.assertEqual(list(virtual_state.entries), [1])
        self.assertGreater(virtual_state.entries[1]["buffer_projection_score"], 0.5)

    def test_projected_virtual_state_builds_compact_action_payload(self):
        state = build_buffer_optimization_state(
            self._candidates(),
            buffer_main_type_index=1,
            legal_buffer_count=4,
        )
        with torch.no_grad():
            state.bu_logits.copy_(torch.tensor([-3.0, 3.0]))
            state.bsu_index_param.copy_(torch.tensor([0.25, 2.6]))

        virtual_state = project_buffer_optimization_state(state, top_k=1)
        artifact = build_commit_action_candidates(
            virtual_state,
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
            executable_by_compact_batch_v1=False,
        )

        self.assertTrue(artifact["schema_compatible_with_compact_action"])
        self.assertFalse(artifact["executable_by_compact_batch_v1"])
        self.assertEqual(len(artifact["actions"]), 1)
        action = artifact["actions"][0]
        self.assertEqual(action["action_kind"], "buffer_insert")
        self.assertEqual(action["action_id"], 1)
        self.assertEqual(action["target_size_idx"], 3)
        self.assertEqual(action["buffer_master_name"], "BUF_X8")
        self.assertEqual(action["candidate_location_x_dbu"], 300.0)
        self.assertEqual(action["candidate_location_y_dbu"], 400.0)
        self.assertEqual(action["net_name"], "n1")
        self.assertEqual(action["downstream_pin_names"], ["sink2/A"])

    def test_projected_compact_payload_is_coordinate_contract_ready(self):
        state = build_buffer_optimization_state(
            self._candidates(),
            buffer_main_type_index=1,
            legal_buffer_count=4,
        )
        with torch.no_grad():
            state.bu_logits.copy_(torch.tensor([-3.0, 3.0]))
            state.bsu_index_param.copy_(torch.tensor([0.25, 2.6]))

        virtual_state = project_buffer_optimization_state(state, top_k=1)
        artifact = build_commit_action_candidates(
            virtual_state,
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
        )
        contract = build_coordinate_backend_contract(artifact["actions"])

        self.assertEqual(contract["status"], "contract_ready_backend_not_implemented")
        self.assertFalse(contract["qor_evidence"])
        self.assertEqual(contract["payload_ready_count"], 1)
        action = artifact["actions"][0]
        self.assertNotIn("bu_logits", action)
        self.assertNotIn("bsu_index_param", action)
        self.assertNotIn("bsu_index_value", action)
        self.assertEqual(action["bsu"], 3)
        self.assertEqual(action["buffer_main_type_index"], 1)

    def test_projection_rejects_missing_coordinate_payload(self):
        state = build_buffer_optimization_state(
            [
                {
                    "candidate_id": 0,
                    "tree_node_id": 10,
                    "net_id": 1,
                    "buffer_main_type_index": 1,
                }
            ],
            buffer_main_type_index=1,
            legal_buffer_count=4,
        )

        with self.assertRaisesRegex(ValueError, "coordinate"):
            project_buffer_optimization_state(state, top_k=1)


if __name__ == "__main__":
    unittest.main()
