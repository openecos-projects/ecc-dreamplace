import unittest

import torch

from dreamplace.ops.net_subgraph_timing import (
    build_net_subgraph_timing_inputs,
    net_subgraph_forward_native,
)


class NetSubgraphTensorBuilderTest(unittest.TestCase):
    def test_builds_compact_native_inputs_from_du_net_records(self):
        nets = [
            {
                "net_id": 10,
                "driver_pin_id": 100,
                "rc_tree": {
                    "root_node_id": 100,
                    "children_by_node": {100: [300, 200], 300: [], 200: []},
                    "edge_rc": {
                        (100, 300): {"r": 3.0, "c": 0.6},
                        (100, 200): {"r": 2.0, "c": 0.4},
                    },
                    "node_cap": {100: 0.0, 300: 3.0, 200: 2.0},
                    "sink_nodes": [300, 200],
                },
            },
            {
                "net_id": 20,
                "driver_pin_id": 900,
                "rc_tree": {
                    "root_node_id": 900,
                    "children_by_node": {900: [950], 950: []},
                    "edge_rc": {
                        (900, 950): {"r": 5.0, "c": 1.0},
                    },
                    "node_cap": {900: 0.0, 950: 4.0},
                    "sink_nodes": [950],
                },
            },
        ]
        candidates = [
            {
                "node_id": 300,
                "bu": 1.0,
                "buffer_input_cap": 0.5,
                "buffer_delay": 7.0,
                "buffer_output_slew": 0.2,
            },
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=candidates,
            driver_arrival_by_net={10: 1.0, 20: 2.0},
            driver_slew_by_net={10: 0.1, 20: 0.2},
            dtype=torch.float32,
        )

        self.assertEqual(inputs["metadata"]["compact_node_to_original"], [100, 300, 200, 900, 950])
        self.assertEqual(inputs["metadata"]["original_node_to_compact"], {100: 0, 300: 1, 200: 2, 900: 3, 950: 4})
        self.assertTrue(torch.equal(inputs["net_flat_topo_sort"], torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["net_flat_topo_sort_start"], torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["pin_fa"], torch.tensor([-1, 0, 0, -1, 3], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["flat_pin_to_start"], torch.tensor([0, 2, 2, 2, 3, 3], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["flat_pin_to"], torch.tensor([1, 2, 4], dtype=torch.int32)))
        self.assertTrue(torch.allclose(inputs["edge_resistance"], torch.tensor([0.0, 3.0, 2.0, 0.0, 5.0])))
        self.assertTrue(torch.allclose(inputs["edge_capacitance"], torch.tensor([0.0, 0.6, 0.4, 0.0, 1.0])))
        self.assertTrue(torch.allclose(inputs["node_capacitance"], torch.tensor([0.0, 3.0, 2.0, 0.0, 4.0])))
        self.assertTrue(torch.equal(inputs["candidate_node_id"], torch.tensor([1], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["sink_node_id"], torch.tensor([1, 2, 4], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["sink_pin_id"], torch.tensor([300, 200, 950], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["sink_net_index"], torch.tensor([10, 10, 20], dtype=torch.int32)))

        native_inputs = {key: inputs[key] for key in inputs["native_input_keys"]}
        result = net_subgraph_forward_native(**native_inputs)
        self.assertTrue(torch.equal(result["sink_net_index"], inputs["sink_net_index"]))
        self.assertTrue(torch.allclose(result["sink_load"], torch.tensor([0.5, 2.2, 4.5])))

    def test_rejects_candidate_outside_selected_subgraphs(self):
        nets = [
            {
                "net_id": 10,
                "driver_pin_id": 100,
                "rc_tree": {
                    "root_node_id": 100,
                    "children_by_node": {100: [200], 200: []},
                    "edge_rc": {(100, 200): {"r": 2.0, "c": 0.4}},
                    "node_cap": {100: 0.0, 200: 2.0},
                    "sink_nodes": [200],
                },
            },
        ]

        with self.assertRaisesRegex(ValueError, "candidate node_id 999"):
            build_net_subgraph_timing_inputs(
                nets,
                candidates=[
                    {
                        "node_id": 999,
                        "bu": 1.0,
                        "buffer_input_cap": 0.5,
                        "buffer_delay": 7.0,
                        "buffer_output_slew": 0.2,
                    },
                ],
            )

    def test_keeps_duplicate_local_tree_node_ids_separate_by_net(self):
        nets = [
            {
                "net_id": 10,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: [2], 2: []},
                    "edge_rc": {
                        (0, 1): {"r": 1.0, "c": 0.0},
                        (1, 2): {"r": 1.0, "c": 0.0},
                    },
                    "node_cap": {0: 0.0, 1: 1.0, 2: 2.0},
                    "sink_nodes": [2],
                },
            },
            {
                "net_id": 20,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: [2], 2: []},
                    "edge_rc": {
                        (0, 1): {"r": 2.0, "c": 0.0},
                        (1, 2): {"r": 3.0, "c": 0.0},
                    },
                    "node_cap": {0: 0.0, 1: 4.0, 2: 5.0},
                    "sink_nodes": [2],
                },
            },
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "net_id": 20,
                    "node_id": 1,
                    "bu": 1.0,
                    "buffer_input_cap": 0.5,
                    "buffer_delay": 7.0,
                    "buffer_output_slew": 0.2,
                },
            ],
        )

        self.assertEqual(inputs["metadata"]["compact_node_to_original"], [0, 1, 2, 0, 1, 2])
        self.assertEqual(
            inputs["metadata"]["original_node_to_compact_by_net"],
            {10: {0: 0, 1: 1, 2: 2}, 20: {0: 3, 1: 4, 2: 5}},
        )
        self.assertTrue(torch.equal(inputs["net_flat_topo_sort"], torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["net_flat_topo_sort_start"], torch.tensor([0, 3, 6], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["candidate_node_id"], torch.tensor([4], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["sink_node_id"], torch.tensor([2, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["sink_net_index"], torch.tensor([10, 20], dtype=torch.int32)))

    def test_rejects_ambiguous_candidate_without_net_id_when_node_id_is_duplicated(self):
        nets = [
            {
                "net_id": 10,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: []},
                    "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}},
                    "node_cap": {0: 0.0, 1: 1.0},
                    "sink_nodes": [1],
                },
            },
            {
                "net_id": 20,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: []},
                    "edge_rc": {(0, 1): {"r": 2.0, "c": 0.0}},
                    "node_cap": {0: 0.0, 1: 2.0},
                    "sink_nodes": [1],
                },
            },
        ]

        with self.assertRaisesRegex(ValueError, "ambiguous candidate node_id 1"):
            build_net_subgraph_timing_inputs(
                nets,
                candidates=[
                    {
                        "node_id": 1,
                        "bu": 1.0,
                        "buffer_input_cap": 0.5,
                        "buffer_delay": 7.0,
                        "buffer_output_slew": 0.2,
                    },
                ],
            )

    def test_splits_segment_candidate_edge_and_preserves_rc(self):
        nets = [
            {
                "net_id": 30,
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
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "candidate_id": 7,
                    "net_id": 30,
                    "tree_node_id": None,
                    "parent_node_id": 0,
                    "child_node_ids": [1],
                    "x_dbu": 25,
                    "y_dbu": 0,
                    "bu": 1.0,
                    "buffer_input_cap": 0.5,
                    "buffer_delay": 1.0,
                    "buffer_output_slew": 0.2,
                },
            ],
        )

        self.assertEqual(inputs["metadata"]["compact_node_to_original"], [0, -1, 1])
        self.assertEqual(
            inputs["metadata"]["synthetic_segment_splits"],
            [
                {
                    "net_id": 30,
                    "synthetic_node_id": -1,
                    "parent_node_id": 0,
                    "child_node_id": 1,
                    "split_ratio": 0.25,
                    "candidate_index": 0,
                }
            ],
        )
        self.assertTrue(torch.equal(inputs["net_flat_topo_sort"], torch.tensor([0, 1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["pin_fa"], torch.tensor([-1, 0, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["flat_pin_to_start"], torch.tensor([0, 1, 2, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["flat_pin_to"], torch.tensor([1, 2], dtype=torch.int32)))
        self.assertTrue(torch.allclose(inputs["edge_resistance"], torch.tensor([0.0, 2.5, 7.5])))
        self.assertTrue(torch.allclose(inputs["edge_capacitance"], torch.tensor([0.0, 1.0, 3.0])))
        self.assertTrue(torch.allclose(inputs["node_capacitance"], torch.tensor([0.0, 0.0, 6.0])))
        self.assertTrue(torch.equal(inputs["candidate_node_id"], torch.tensor([1], dtype=torch.int32)))

    def test_segment_split_uses_explicit_ratio_across_coordinate_domains(self):
        nets = [
            {
                "net_id": 30,
                "driver_pin_id": 0,
                "coordinates": {0: (10, 0), 1: (110, 0)},
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: []},
                    "edge_rc": {(0, 1): {"r": 10.0, "c": 4.0}},
                    "node_cap": {0: 0.0, 1: 6.0},
                    "sink_nodes": [1],
                },
            }
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "candidate_id": 7,
                    "net_id": 30,
                    "tree_node_id": None,
                    "parent_node_id": 0,
                    "child_node_ids": [1],
                    "x_dbu": 25_000,
                    "y_dbu": 0,
                    "segment_split_ratio": 0.25,
                    "bu": 1.0,
                },
            ],
        )

        self.assertEqual(
            inputs["metadata"]["synthetic_segment_splits"][0]["split_ratio"],
            0.25,
        )
        self.assertTrue(
            torch.allclose(
                inputs["edge_resistance"],
                torch.tensor([0.0, 2.5, 7.5]),
            )
        )

    def test_splits_multiple_segment_candidates_on_same_edge(self):
        nets = [
            {
                "net_id": 31,
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
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "candidate_id": 7,
                    "net_id": 31,
                    "tree_node_id": None,
                    "parent_node_id": 0,
                    "child_node_ids": [1],
                    "x_dbu": 25,
                    "y_dbu": 0,
                    "bu": 1.0,
                },
                {
                    "candidate_id": 8,
                    "net_id": 31,
                    "tree_node_id": None,
                    "parent_node_id": 0,
                    "child_node_ids": [1],
                    "x_dbu": 75,
                    "y_dbu": 0,
                    "bu": 1.0,
                },
            ],
        )

        self.assertEqual(inputs["metadata"]["compact_node_to_original"], [0, -1, -2, 1])
        self.assertTrue(torch.equal(inputs["candidate_node_id"], torch.tensor([1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["pin_fa"], torch.tensor([-1, 0, 1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["flat_pin_to"], torch.tensor([1, 2, 3], dtype=torch.int32)))
        self.assertTrue(torch.allclose(inputs["edge_resistance"], torch.tensor([0.0, 2.5, 5.0, 2.5])))
        self.assertTrue(torch.allclose(inputs["edge_capacitance"], torch.tensor([0.0, 1.0, 2.0, 1.0])))

    def test_same_net_multiple_candidates_share_one_tensor_build(self):
        nets = [
            {
                "net_id": 40,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1, 2], 1: [], 2: []},
                    "edge_rc": {
                        (0, 1): {"r": 1.0, "c": 0.1},
                        (0, 2): {"r": 2.0, "c": 0.2},
                    },
                    "node_cap": {0: 0.0, 1: 1.0, 2: 2.0},
                    "sink_nodes": [1, 2],
                },
            }
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "candidate_id": 100,
                    "net_id": 40,
                    "node_id": 1,
                    "bu": 0.25,
                    "buffer_input_cap": 0.5,
                    "buffer_delay": 1.0,
                    "buffer_output_slew": 0.2,
                },
                {
                    "candidate_id": 101,
                    "net_id": 40,
                    "node_id": 2,
                    "bu": 0.75,
                    "buffer_input_cap": 0.6,
                    "buffer_delay": 1.2,
                    "buffer_output_slew": 0.3,
                },
            ],
        )

        self.assertTrue(torch.equal(inputs["candidate_node_id"], torch.tensor([1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(inputs["candidate_net_id"], torch.tensor([40, 40], dtype=torch.int32)))
        self.assertEqual(inputs["metadata"]["candidate_count"], 2)
        self.assertEqual(inputs["metadata"]["candidate_ids"], [100, 101])
        self.assertEqual(inputs["metadata"]["candidate_net_id"], [40, 40])
        self.assertEqual(inputs["metadata"]["net_ids"], [40])

    def test_compact_runtime_metadata_keeps_only_provider_mappings(self):
        nets = [
            {
                "net_id": 40,
                "driver_pin_id": 0,
                "rc_tree": {
                    "root_node_id": 0,
                    "children_by_node": {0: [1], 1: []},
                    "edge_rc": {(0, 1): {"r": 1.0, "c": 0.1}},
                    "node_cap": {0: 0.0, 1: 1.0},
                    "sink_nodes": [1],
                },
            }
        ]

        inputs = build_net_subgraph_timing_inputs(
            nets,
            candidates=[
                {
                    "candidate_id": 100,
                    "net_id": 40,
                    "node_id": 1,
                    "bu": 0.0,
                }
            ],
            compact_runtime_metadata=True,
        )

        self.assertEqual(inputs["metadata"], {
            "compact_node_to_original": [0, 1],
            "candidate_count": 1,
            "net_ids": [40],
        })


if __name__ == "__main__":
    unittest.main()
