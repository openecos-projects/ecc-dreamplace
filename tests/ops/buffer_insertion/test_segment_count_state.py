import unittest

import torch

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)


class SegmentCountStateTest(unittest.TestCase):
    def _line_net(self):
        return {
            "net_id": 41,
            "net_name": "line",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 10.0, "c": 4.0}},
                "node_cap": {0: 0.0, 1: 6.0},
                "sink_nodes": [1],
            },
        }

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
                    (0, 1): {"r": 10.0, "c": 4.0},
                    (0, 2): {"r": 6.0, "c": 2.0},
                    (1, 3): {"r": 4.0, "c": 1.0},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 5.0, 3: 3.0},
                "sink_nodes": [2, 3],
            },
        }

    def test_line_net_builds_one_segment_state(self):
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
            initial_z=0.05,
            initial_bsu_index=2.5,
            fixed_bsu_index=2,
            dtype=torch.float64,
        )

        self.assertEqual(state.summary["full_design_scope"], True)
        self.assertIsNone(state.summary["max_nets"])
        self.assertIsNone(state.summary["max_candidates"])
        self.assertEqual(state.summary["eligible_segment_count"], 1)
        self.assertEqual(tuple(state.z_param.shape), (1,))
        self.assertEqual(tuple(state.bsu_index_param.shape), (1,))
        self.assertTrue(torch.allclose(state.z_value(), torch.tensor([0.05], dtype=torch.float64)))
        self.assertTrue(torch.allclose(state.bsu_index(), torch.tensor([2.0], dtype=torch.float64)))
        self.assertEqual(state.fixed_bsu_index, 2)
        self.assertEqual(state.summary["fixed_bsu_index"], 2)
        self.assertEqual(state.segment_rows[0]["net_id"], 41)
        self.assertEqual(state.segment_rows[0]["parent_node_id"], 0)
        self.assertEqual(state.segment_rows[0]["child_node_id"], 1)
        self.assertEqual(state.segment_rows[0]["buffer_main_type_index"], 7)

    def test_branch_net_preserves_one_segment_per_tree_edge(self):
        state = build_segment_count_state(
            [self._branch_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=2,
        )

        edges = [
            (row["net_id"], row["parent_node_id"], row["child_node_id"])
            for row in state.segment_rows
        ]
        self.assertEqual(edges, [(42, 0, 1), (42, 0, 2), (42, 1, 3)])
        self.assertEqual(state.segment_ids.tolist(), [0, 1, 2])
        self.assertEqual(state.segment_net_id.tolist(), [42, 42, 42])
        self.assertEqual(state.parent_node_id.tolist(), [0, 0, 1])
        self.assertEqual(state.child_node_id.tolist(), [1, 2, 3])
        self.assertEqual(state.summary["eligible_segment_count"], 3)

    def test_previously_buffered_segment_survives_coincident_endpoints(self):
        net = self._line_net()
        net['coordinates'][1] = net['coordinates'][0]
        net['rc_tree']['edge_rc'][(0, 1)] = {'r': 0., 'c': 0.}
        state = build_segment_count_state(
            [net], buffer_main_type_index=7, legal_buffer_count=8,
            max_repeater_count=3, initial_z=0., retained_edges={(41, 0, 1)},
        )
        self.assertEqual(
            list(zip(state.segment_net_id.tolist(), state.parent_node_id.tolist(),
                     state.child_node_id.tolist())), [(41, 0, 1)],
        )

    def test_skips_missing_coordinates_missing_rc_and_zero_length(self):
        missing_coordinate = self._line_net()
        missing_coordinate["net_id"] = 1
        missing_coordinate["coordinates"] = {0: (0, 0)}

        missing_rc = self._line_net()
        missing_rc["net_id"] = 2
        missing_rc["rc_tree"] = dict(missing_rc["rc_tree"])
        missing_rc["rc_tree"]["edge_rc"] = {}

        zero_length = self._line_net()
        zero_length["net_id"] = 3
        zero_length["coordinates"] = {0: (5, 5), 1: (5, 5)}

        valid = self._line_net()
        valid["net_id"] = 4

        state = build_segment_count_state(
            [missing_coordinate, missing_rc, zero_length, valid],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=2,
        )

        self.assertEqual(state.summary["eligible_segment_count"], 1)
        self.assertEqual(state.summary["skipped_missing_coordinate_count"], 1)
        self.assertEqual(state.summary["skipped_missing_rc_count"], 1)
        self.assertEqual(state.summary["skipped_zero_length_count"], 1)
        self.assertEqual(state.segment_rows[0]["net_id"], 4)

    def test_rejects_bounded_default_scope(self):
        with self.assertRaisesRegex(ValueError, "full-design segment scope"):
            build_segment_count_state(
                [self._line_net()],
                buffer_main_type_index=7,
                legal_buffer_count=8,
                max_repeater_count=2,
                max_nets=1,
            )

    def test_rejects_fixed_bsu_outside_legal_buffer_table(self):
        with self.assertRaisesRegex(ValueError, "reference a legal buffer"):
            build_segment_count_state(
                [self._line_net()],
                buffer_main_type_index=7,
                legal_buffer_count=2,
                max_repeater_count=2,
                fixed_bsu_index=2,
            )


if __name__ == "__main__":
    unittest.main()
