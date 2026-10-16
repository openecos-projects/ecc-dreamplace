import unittest

import torch

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)


class SegmentCountTensorBuilderTest(unittest.TestCase):
    def _branch_net(self):
        return {
            "net_id": 42,
            "net_name": "branch",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0), 2: (0, 100), 3: (150, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [3], 2: [], 3: []},
                "edge_rc": {
                    (0, 1): {"r": 10.0, "c": 0.5},
                    (0, 2): {"r": 6.0, "c": 0.25},
                    (1, 3): {"r": 4.0, "c": 0.125},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 5.0, 3: 3.0},
                "sink_nodes": [2, 3],
            },
        }

    def _partial_segment_net(self):
        net = self._branch_net()
        net["net_id"] = 43
        net["rc_tree"] = dict(net["rc_tree"])
        net["rc_tree"]["edge_rc"] = {
            (0, 1): {"r": 10.0, "c": 0.5},
            (0, 2): {"r": 6.0, "c": 0.25},
        }
        return net

    def _state(self, nets):
        return build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.0,
            initial_bsu_index=1.0,
            dtype=torch.float64,
        )

    def test_builds_flat_topology_and_segment_edge_mapping(self):
        net = self._branch_net()
        state = self._state([net])

        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)

        self.assertEqual(inputs["metadata"]["net_ids"], [42])
        self.assertEqual(inputs["net_ids"].tolist(), [42])
        self.assertEqual(inputs["net_topo_start"].tolist(), [0, 4])
        self.assertEqual(inputs["flat_topo_node_id"].tolist(), [0, 1, 3, 2])
        self.assertEqual(inputs["edge_start"].tolist(), [0, 2, 3, 3, 3])
        self.assertEqual(inputs["edge_parent_node_id"].tolist(), [0, 0, 1])
        self.assertEqual(inputs["edge_child_node_id"].tolist(), [1, 2, 3])
        self.assertEqual(inputs["edge_parent_compact_id"].tolist(), [0, 0, 1])
        self.assertEqual(inputs["edge_child_compact_id"].tolist(), [1, 3, 2])
        self.assertEqual(inputs["edge_net_index"].tolist(), [0, 0, 0])
        self.assertEqual(inputs["edge_to_segment_id"].tolist(), [0, 1, 2])
        torch.testing.assert_close(
            inputs["edge_resistance"],
            torch.tensor([10.0, 6.0, 4.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            inputs["edge_capacitance"],
            torch.tensor([0.5, 0.25, 0.125], dtype=torch.float64),
        )
        torch.testing.assert_close(
            inputs["node_capacitance"],
            torch.tensor([0.0, 1.0, 3.0, 5.0], dtype=torch.float64),
        )
        self.assertEqual(inputs["sink_node_id"].tolist(), [2, 3])
        self.assertEqual(inputs["sink_net_id"].tolist(), [42, 42])
        self.assertEqual(inputs["sink_node_compact_id"].tolist(), [3, 2])
        self.assertEqual(inputs["driver_pin_id"].tolist(), [0])
        self.assertEqual(inputs["parent_cap_fraction"].shape, (3, 4))
        self.assertEqual(inputs["child_cap_fraction"].shape, (3, 4))
        self.assertEqual(inputs["segment_sub_resistance_fraction"].shape, (3, 4, 4))
        torch.testing.assert_close(
            inputs["parent_cap_fraction"][0],
            torch.tensor([1.0, 0.5, 0.33, 0.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            inputs["child_cap_fraction"][0],
            torch.tensor([1.0, 0.5, 0.33, 0.25], dtype=torch.float64),
        )

    def test_marks_edges_without_segment_state_as_minus_one(self):
        net = self._partial_segment_net()
        state = self._state([net])

        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)

        self.assertEqual(len(state.segment_rows), 2)
        self.assertEqual(inputs["edge_parent_node_id"].tolist(), [0, 0, 1])
        self.assertEqual(inputs["edge_child_node_id"].tolist(), [1, 2, 3])
        self.assertEqual(inputs["edge_to_segment_id"].tolist(), [0, 1, -1])

    def test_quantized_zero_length_segment_uses_finite_equal_fractions(self):
        net = self._branch_net()
        state = self._state([net])
        rows = [dict(row) for row in state.segment_rows]
        rows[0]["child_x_dbu"] = rows[0]["parent_x_dbu"]
        rows[0]["child_y_dbu"] = rows[0]["parent_y_dbu"]
        state.segment_rows = tuple(rows)

        inputs = build_segment_count_timing_inputs(
            [net], state, dtype=torch.float64
        )

        expected = torch.tensor([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=torch.float64)
        self.assertTrue(torch.isfinite(inputs["parent_cap_fraction"]).all())
        self.assertTrue(torch.isfinite(inputs["child_cap_fraction"]).all())
        self.assertTrue(
            torch.isfinite(inputs["segment_sub_resistance_fraction"]).all()
        )
        torch.testing.assert_close(inputs["parent_cap_fraction"][0], expected)
        torch.testing.assert_close(inputs["child_cap_fraction"][0], expected)
        for count in range(1, state.max_repeater_count + 1):
            torch.testing.assert_close(
                inputs["segment_sub_resistance_fraction"][0, count, : count + 1],
                torch.full(
                    (count + 1,),
                    1.0 / float(count + 1),
                    dtype=torch.float64,
                ),
            )

    def test_state_static_topology_matches_rc_tree_fallback(self):
        net = self._branch_net()
        state = self._state([net])

        static_inputs = build_segment_count_timing_inputs(
            [net], state, dtype=torch.float64
        )
        self.assertEqual(
            static_inputs["metadata"]["topology_source"], "state_static"
        )
        net.pop("_segment_count_static_topology")
        fallback_inputs = build_segment_count_timing_inputs(
            [net], state, dtype=torch.float64
        )
        self.assertEqual(
            fallback_inputs["metadata"]["topology_source"], "rc_tree"
        )
        for key, static_value in static_inputs.items():
            if key == "metadata":
                continue
            torch.testing.assert_close(static_value, fallback_inputs[key])


if __name__ == "__main__":
    unittest.main()
