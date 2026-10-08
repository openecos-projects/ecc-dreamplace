import unittest

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
)
from dreamplace.ops.net_subgraph_timing import (
    build_relaxed_buffer_dynamic_net_arc_inputs,
    build_relaxed_buffer_timing_payload,
    build_dynamic_net_arc_inputs_from_net_subgraphs,
    build_timing_pin_net_inputs,
)


class NetSubgraphTimingAdapterTest(unittest.TestCase):
    def test_builds_pin_domain_timing_inputs_from_sink_outputs(self):
        result = {
            "sink_arrival": torch.tensor([1.5, 2.5], dtype=torch.float32),
            "sink_net_delay": torch.tensor([1.25, 2.25], dtype=torch.float32),
            "sink_slew": torch.tensor([0.15, 0.25], dtype=torch.float32),
            "sink_net_impulse": torch.tensor([0.015, 0.025], dtype=torch.float32),
            "sink_load": torch.tensor([3.0, 4.0], dtype=torch.float32),
            "sink_net_index": torch.tensor([10, 20], dtype=torch.int32),
        }

        adapted = build_timing_pin_net_inputs(
            result,
            sink_pin_id=torch.tensor([3, 1], dtype=torch.int32),
            num_pins=5,
            fill_value=-1.0,
        )

        expected_delay = torch.tensor([-1.0, 2.25, -1.0, 1.25, -1.0])
        expected_impulse = torch.tensor([-1.0, 0.025, -1.0, 0.015, -1.0])
        expected_cap = torch.tensor([-1.0, 4.0, -1.0, 3.0, -1.0])
        self.assertTrue(torch.allclose(adapted["pin_net_delays"]["rise"], expected_delay))
        self.assertTrue(torch.allclose(adapted["pin_net_delays"]["fall"], expected_delay))
        self.assertTrue(torch.allclose(adapted["pin_net_impulses"]["rise"], expected_impulse))
        self.assertTrue(torch.allclose(adapted["pin_net_impulses"]["fall"], expected_impulse))
        self.assertTrue(torch.allclose(adapted["pin_net_caps"]["rise"], expected_cap))
        self.assertTrue(torch.allclose(adapted["pin_net_caps"]["fall"], expected_cap))
        self.assertTrue(torch.equal(adapted["sink_pin_id"], torch.tensor([3, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(adapted["sink_net_index"], torch.tensor([10, 20], dtype=torch.int32)))

    def test_pin_domain_adapter_exposes_dynamic_net_arc_semantics(self):
        result = {
            "sink_arrival": torch.tensor([6.0], dtype=torch.float32),
            "sink_net_delay": torch.tensor([5.0], dtype=torch.float32),
            "sink_slew": torch.tensor([0.2], dtype=torch.float32),
            "sink_net_impulse": torch.tensor([0.03], dtype=torch.float32),
            "sink_load": torch.tensor([0.5], dtype=torch.float32),
        }

        adapted = build_timing_pin_net_inputs(
            result,
            sink_pin_id=torch.tensor([1], dtype=torch.int32),
            num_pins=3,
        )

        semantics = adapted["net_arc_semantics"]
        self.assertEqual(semantics["schema_name"], "buffering_dynamic_net_arc_semantics")
        self.assertEqual(semantics["pin_net_delay_semantics"], "relative_net_arc_delay")
        self.assertEqual(
            semantics["pin_net_impulse_semantics"],
            "relative_net_arc_slew_square_delta",
        )
        self.assertFalse(semantics["includes_driver_arrival"])
        self.assertTrue(semantics["includes_buffer_intrinsic_delay"])
        self.assertTrue(semantics["includes_upstream_wire_delay"])
        self.assertTrue(semantics["includes_downstream_wire_delay"])
        self.assertTrue(semantics["timing_op_relative_net_arc_delay_calibrated"])

    def test_rejects_duplicate_sink_pins_by_default(self):
        result = {
            "sink_arrival": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "sink_net_delay": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "sink_slew": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "sink_net_impulse": torch.tensor([0.01, 0.04], dtype=torch.float32),
            "sink_load": torch.tensor([3.0, 4.0], dtype=torch.float32),
        }

        with self.assertRaisesRegex(ValueError, "duplicate sink_pin_id"):
            build_timing_pin_net_inputs(
                result,
                sink_pin_id=torch.tensor([1, 1], dtype=torch.int32),
                num_pins=3,
            )

    def test_builds_dynamic_net_arc_inputs_from_net_subgraphs(self):
        adapted = build_dynamic_net_arc_inputs_from_net_subgraphs(
            [
                {
                    "net_id": 7,
                    "driver_pin_id": 0,
                    "rc_tree": {
                        "root_node_id": 0,
                        "children_by_node": {0: [1], 1: []},
                        "edge_rc": {(0, 1): {"r": 2.0, "c": 0.0}},
                        "node_cap": {0: 0.0, 1: 3.0},
                        "sink_nodes": [1],
                    },
                }
            ],
            candidates=[
                {
                    "net_id": 7,
                    "node_id": 1,
                    "bu": 1.0,
                    "buffer_input_cap": 0.5,
                    "buffer_delay": 4.0,
                    "buffer_output_slew": 0.2,
                }
            ],
            num_pins=3,
            driver_arrival_by_net={7: 1.0},
            driver_slew_by_net={7: 0.1},
            fill_value=0.0,
            dtype=torch.float32,
        )

        self.assertEqual(adapted["source"], "net_subgraph_timing_cpp")
        self.assertEqual(adapted["net_subgraph_timing_mode"], "native_fixed_state_forward")
        self.assertTrue(torch.equal(adapted["sink_pin_id"], torch.tensor([1], dtype=torch.int32)))
        self.assertTrue(torch.equal(adapted["sink_net_index"], torch.tensor([7], dtype=torch.int32)))
        self.assertTrue(torch.allclose(adapted["pin_net_delays"]["rise"], torch.tensor([0.0, 5.0, 0.0])))
        self.assertTrue(torch.allclose(adapted["pin_net_impulses"]["rise"], torch.tensor([0.0, 0.03, 0.0])))
        self.assertTrue(torch.allclose(adapted["pin_net_caps"]["rise"], torch.tensor([0.5, 0.5, 0.0])))
        self.assertEqual(adapted["metadata"]["net_ids"], [7])
        self.assertEqual(adapted["metadata"]["driver_pin_ids"], [0])
        self.assertEqual(
            adapted["metadata"]["net_arc_semantics"]["pin_net_delay_semantics"],
            "relative_net_arc_delay",
        )
        self.assertTrue(
            adapted["metadata"]["net_arc_semantics"][
                "timing_op_relative_net_arc_delay_calibrated"
            ]
        )

    def test_dynamic_inputs_update_driver_cap_for_sibling_load_replacement(self):
        adapted = build_dynamic_net_arc_inputs_from_net_subgraphs(
            [
                {
                    "net_id": 9,
                    "driver_pin_id": 0,
                    "rc_tree": {
                        "root_node_id": 0,
                        "children_by_node": {0: [1, 2], 1: [], 2: []},
                        "edge_rc": {
                            (0, 1): {"r": 1.0, "c": 0.0},
                            (0, 2): {"r": 1.0, "c": 0.0},
                        },
                        "node_cap": {0: 0.0, 1: 0.3, 2: 1.0},
                        "sink_nodes": [1, 2],
                    },
                }
            ],
            candidates=[
                {
                    "net_id": 9,
                    "node_id": 1,
                    "bu": 1.0,
                    "buffer_input_cap": 0.8,
                    "buffer_delay": 4.0,
                    "buffer_output_slew": 0.2,
                }
            ],
            num_pins=3,
            driver_arrival_by_net={9: 0.0},
            driver_slew_by_net={9: 0.1},
            fill_value=0.0,
            dtype=torch.float32,
        )

        # The driver cell must see the post-buffer effective net load:
        # moved branch load 0.3 is replaced by buffer input cap 0.8, while
        # sibling load 1.0 remains, so root lin = 1.8.
        self.assertTrue(
            torch.allclose(
                adapted["pin_net_caps"]["rise"],
                torch.tensor([1.8, 0.8, 1.0]),
            )
        )

    def test_relaxed_payload_dynamic_inputs_update_driver_cap(self):
        candidate = {
            "candidate_id": 0,
            "net_id": 9,
            "node_id": 1,
            "tree_node_id": 1,
            "buffer_main_type_index": 1,
        }
        state = build_buffer_optimization_state(
            [candidate],
            buffer_main_type_index=1,
            legal_buffer_count=2,
            initial_bu_logit=30.0,
            initial_bsu_index=1.0,
        )
        net = {
            "net_id": 9,
            "driver_pin_id": 0,
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [], 2: []},
                "edge_rc": {
                    (0, 1): {"r": 1.0, "c": 0.0},
                    (0, 2): {"r": 1.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 0.3, 2: 1.0},
                "sink_nodes": [1, 2],
            },
        }

        payload = build_relaxed_buffer_timing_payload(
            [net],
            [candidate],
            buffer_state=state,
            per_size_input_cap=torch.tensor([[0.4, 0.8]]),
            per_size_delay=torch.tensor([[2.0, 4.0]]),
            per_size_output_slew=torch.tensor([[0.3, 0.2]]),
            num_pins=3,
            driver_arrival_by_net={9: 0.0},
            driver_slew_by_net={9: 0.1},
            dtype=torch.float32,
        )

        adapted = build_relaxed_buffer_dynamic_net_arc_inputs(
            buffer_state=state,
            **payload,
        )

        self.assertTrue(
            torch.allclose(
                adapted["pin_net_caps"]["rise"],
                torch.tensor([1.8, 0.8, 1.0]),
                atol=1e-5,
            )
        )
        self.assertEqual(adapted["metadata"]["driver_pin_ids"], [0])
        self.assertEqual(
            adapted["metadata"]["driver_net_cap_source"],
            "net_subgraph_root_lin",
        )

    def test_relaxed_payload_can_filter_to_candidate_downstream_sink_pins(self):
        candidate = {
            "candidate_id": 0,
            "net_id": 9,
            "node_id": 1,
            "tree_node_id": 1,
            "buffer_main_type_index": 1,
            "downstream_pin_ids": [1],
            "downstream_pin_count": 1,
        }
        state = build_buffer_optimization_state(
            [candidate],
            buffer_main_type_index=1,
            legal_buffer_count=2,
            initial_bu_logit=30.0,
            initial_bsu_index=1.0,
        )
        net = {
            "net_id": 9,
            "driver_pin_id": 0,
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [], 2: []},
                "edge_rc": {
                    (0, 1): {"r": 1.0, "c": 0.0},
                    (0, 2): {"r": 1.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 0.3, 2: 1.0},
                "sink_nodes": [1, 2],
            },
        }

        payload = build_relaxed_buffer_timing_payload(
            [net],
            [candidate],
            buffer_state=state,
            per_size_input_cap=torch.tensor([[0.4, 0.8]]),
            per_size_delay=torch.tensor([[2.0, 4.0]]),
            per_size_output_slew=torch.tensor([[0.3, 0.2]]),
            num_pins=3,
            driver_arrival_by_net={9: 0.0},
            driver_slew_by_net={9: 0.1},
            sink_filter_mode="candidate_downstream",
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(payload["sink_pin_id"], torch.tensor([1], dtype=torch.int32)))
        self.assertTrue(torch.equal(payload["sink_node_id"], torch.tensor([1], dtype=torch.int32)))
        self.assertTrue(torch.equal(payload["sink_net_index"], torch.tensor([9], dtype=torch.int32)))
        sink_filter = payload["metadata"]["sink_filter"]
        self.assertEqual(sink_filter["mode"], "candidate_downstream")
        self.assertEqual(sink_filter["status"], "filtered")
        self.assertEqual(sink_filter["all_net_sink_count"], 2)
        self.assertEqual(sink_filter["filtered_sink_count"], 1)
        self.assertEqual(sink_filter["candidate_downstream_sink_count"], 1)
        self.assertEqual(sink_filter["candidate_downstream_pin_ids_by_net"], {9: [1]})

    def test_relaxed_payload_accepts_segment_candidate_without_node_id(self):
        candidate = {
            "candidate_id": 0,
            "net_id": 9,
            "parent_node_id": 0,
            "child_node_ids": [2],
            "x_dbu": 50,
            "y_dbu": 0,
            "buffer_main_type_index": 1,
            "downstream_pin_ids": [2],
            "downstream_pin_count": 1,
        }
        state_candidate = dict(candidate, synthetic_node_id=-1)
        state = build_buffer_optimization_state(
            [state_candidate],
            buffer_main_type_index=1,
            legal_buffer_count=2,
            initial_bu_logit=30.0,
            initial_bsu_index=1.0,
        )
        net = {
            "net_id": 9,
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (0, 100), 2: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [], 2: []},
                "edge_rc": {
                    (0, 1): {"r": 1.0, "c": 0.0},
                    (0, 2): {"r": 2.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 0.3, 2: 1.0},
                "sink_nodes": [1, 2],
            },
        }

        payload = build_relaxed_buffer_timing_payload(
            [net],
            [candidate],
            buffer_state=state,
            per_size_input_cap=torch.tensor([[0.4, 0.8]]),
            per_size_delay=torch.tensor([[2.0, 4.0]]),
            per_size_output_slew=torch.tensor([[0.3, 0.2]]),
            num_pins=3,
            driver_arrival_by_net={9: 0.0},
            driver_slew_by_net={9: 0.1},
            sink_filter_mode="candidate_downstream",
            dtype=torch.float32,
        )

        self.assertEqual(payload["metadata"]["compact_node_to_original"], [0, 1, -1, 2])
        self.assertEqual(int(payload["candidate_node_id"][0]), 2)
        self.assertTrue(torch.equal(payload["sink_pin_id"], torch.tensor([2], dtype=torch.int32)))

    def test_relaxed_buffer_dynamic_inputs_preserve_current_approx_coordinate_source(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=2,
            initial_bu_logit=0.0,
            initial_bsu_index=1.0,
        )

        def coordinate_callback(**kwargs):
            delay = kwargs["candidate_input_slew"].view(-1, 1) + torch.tensor(
                [[1.0, 2.0]],
                dtype=kwargs["candidate_input_slew"].dtype,
                device=kwargs["candidate_input_slew"].device,
            )
            return {
                "per_size_input_cap": kwargs["per_size_input_cap"],
                "per_size_delay": delay,
                "per_size_output_slew": kwargs["per_size_output_slew"],
                "coordinate_source": "current_approx_probe_slew_lout",
            }

        adapted = build_relaxed_buffer_dynamic_net_arc_inputs(
            buffer_state=state,
            per_size_input_cap=torch.tensor([[0.2, 0.4]]),
            per_size_delay=torch.tensor([[1.0, 1.0]]),
            per_size_output_slew=torch.tensor([[0.5, 0.25]]),
            sink_pin_id=torch.tensor([2], dtype=torch.int32),
            num_pins=4,
            coordinate_source="current_approx",
            buffer_coordinate_callback=coordinate_callback,
            net_flat_topo_sort=torch.tensor([0, 1, 2], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 3], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0, 1], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            flat_pin_to=torch.tensor([1, 2], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 2.0, 3.0]),
            node_capacitance=torch.tensor([0.0, 1.0, 4.0]),
            edge_capacitance=torch.tensor([0.0, 0.0, 0.0]),
            driver_arrival=torch.tensor([0.0]),
            driver_slew=torch.tensor([0.1]),
            sink_node_id=torch.tensor([2], dtype=torch.int32),
        )

        self.assertEqual(adapted["buffer_coordinate_source"], "current_approx_probe_slew_lout")
        self.assertEqual(
            adapted["metadata"]["bsu_index_status"]["coordinate_source"],
            "current_approx_probe_slew_lout",
        )
        self.assertGreater(float(adapted["pin_net_delays"]["rise"][2]), 1.0)


if __name__ == "__main__":
    unittest.main()
