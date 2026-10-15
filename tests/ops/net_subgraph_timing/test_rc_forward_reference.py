import unittest

from dreamplace.ops.net_subgraph_timing import compute_buffer_aware_rc_forward


class RcForwardReferenceTest(unittest.TestCase):
    def _tree(self):
        return {
            "root_node_id": 0,
            "children_by_node": {0: [1], 1: [2], 2: []},
            "parent_by_node": {0: None, 1: 0, 2: 1},
            "edge_rc": {
                (0, 1): {"r": 2.0},
                (1, 2): {"r": 3.0},
            },
            "node_cap": {0: 0.0, 1: 1.0, 2: 4.0},
            "sink_nodes": [2],
        }

    def test_all_bu_zero_matches_original_mode(self):
        tree = self._tree()
        original = compute_buffer_aware_rc_forward(tree, mode="original")
        buffer_forward = compute_buffer_aware_rc_forward(
            tree,
            mode="buffer_forward",
            virtual_buffers={1: {"bu": 0, "bsu": 0}},
            buffer_library={0: {"input_cap": 0.5, "delay": 7.0}},
        )

        self.assertEqual(buffer_forward["lin_by_node"], original["lin_by_node"])
        self.assertEqual(buffer_forward["arrival_by_node"], original["arrival_by_node"])

    def test_inserted_buffer_partitions_upstream_load_and_adds_delay(self):
        result = compute_buffer_aware_rc_forward(
            self._tree(),
            mode="buffer_forward",
            virtual_buffers={1: {"bu": 1, "bsu": 0}},
            buffer_library={0: {"input_cap": 0.5, "delay": 7.0}},
        )

        self.assertEqual(result["lin_by_node"][1], 0.5)
        self.assertEqual(result["lout_by_node"][1], 5.0)
        self.assertEqual(result["arrival_by_node"][1], 1.0 + 7.0)
        self.assertEqual(result["arrival_by_node"][2], 8.0 + 12.0)

    def test_wire_capacitance_is_split_to_edge_endpoints(self):
        result = compute_buffer_aware_rc_forward(
            {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 2.0, "c": 2.0}},
                "node_cap": {0: 0.0, 1: 4.0},
                "sink_nodes": [1],
            },
            mode="original",
        )

        self.assertEqual(result["effective_node_cap"][0], 1.0)
        self.assertEqual(result["effective_node_cap"][1], 5.0)
        self.assertEqual(result["lin_by_node"][1], 5.0)
        self.assertEqual(result["arrival_by_node"][1], 10.0)

    def test_bsu_on_bu_zero_does_not_affect_timing(self):
        tree = self._tree()
        bsu0 = compute_buffer_aware_rc_forward(
            tree,
            mode="buffer_forward",
            virtual_buffers={1: {"bu": 0, "bsu": 0}},
            buffer_library={
                0: {"input_cap": 0.5, "delay": 7.0},
                1: {"input_cap": 9.0, "delay": 99.0},
            },
        )
        bsu1 = compute_buffer_aware_rc_forward(
            tree,
            mode="buffer_forward",
            virtual_buffers={1: {"bu": 0, "bsu": 1}},
            buffer_library={
                0: {"input_cap": 0.5, "delay": 7.0},
                1: {"input_cap": 9.0, "delay": 99.0},
            },
        )

        self.assertEqual(bsu0["arrival_by_node"], bsu1["arrival_by_node"])

    def test_missing_buffer_library_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_buffer_aware_rc_forward(
                self._tree(),
                mode="buffer_forward",
                virtual_buffers={1: {"bu": 1, "bsu": 9}},
                buffer_library={},
            )


if __name__ == "__main__":
    unittest.main()
