import unittest

import torch

from dreamplace.ops.net_subgraph_timing import (
    build_equal_spaced_segment_candidates,
    evaluate_expanded_segment_reference,
    net_subgraph_forward,
    segment_repeater_transfer,
)


def _native_inputs_only(inputs):
    return {key: inputs[key] for key in inputs["native_input_keys"]}


class RecordingSurrogate:
    def __init__(self, *, input_cap=0.5, delay=1.25, output_slew=0.2):
        self.input_cap = float(input_cap)
        self.delay = float(delay)
        self.output_slew = float(output_slew)
        self.calls = []

    def __call__(self, candidate, *, bsu, input_slew, output_cap):
        self.calls.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "bsu": int(bsu),
                "input_slew": float(input_slew),
                "output_cap": float(output_cap),
            }
        )
        return {
            "buffer_input_cap": self.input_cap,
            "buffer_delay": self.delay,
            "buffer_output_slew": self.output_slew,
            "delay_source": "test_surrogate",
            "transition_source": "test_surrogate",
        }


class SegmentRepeaterTransferTest(unittest.TestCase):
    def _line_net(self):
        return {
            "net_id": 41,
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

    def test_generates_equal_spaced_candidates_and_rejects_bad_segments(self):
        net = self._line_net()

        self.assertEqual(
            build_equal_spaced_segment_candidates(
                net,
                parent_node_id=0,
                child_node_id=1,
                repeater_count=0,
            ),
            [],
        )

        dbu_candidates = build_equal_spaced_segment_candidates(
            {
                **net,
                "coordinates_dbu": {0: (1000, 2000), 1: (3000, 2000)},
            },
            parent_node_id=0,
            child_node_id=1,
            repeater_count=1,
            coordinate_key="coordinates_dbu",
        )
        self.assertEqual(
            (dbu_candidates[0]["x_dbu"], dbu_candidates[0]["y_dbu"]),
            (2000, 2000),
        )

        candidates = build_equal_spaced_segment_candidates(
            net,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=3,
            bsu=2,
            buffer_main_type_index=9,
            candidate_id_start=7,
        )

        self.assertEqual([candidate["candidate_id"] for candidate in candidates], [7, 8, 9])
        self.assertEqual([candidate["segment_split_index"] for candidate in candidates], [0, 1, 2])
        self.assertEqual(
            [candidate["segment_split_ratio"] for candidate in candidates],
            [0.25, 0.5, 0.75],
        )
        self.assertEqual([candidate["x_dbu"] for candidate in candidates], [25, 50, 75])
        for candidate in candidates:
            self.assertEqual(candidate["net_id"], 41)
            self.assertIsNone(candidate["tree_node_id"])
            self.assertEqual(candidate["parent_node_id"], 0)
            self.assertEqual(candidate["child_node_ids"], [1])
            self.assertEqual(candidate["y_dbu"], 0)
            self.assertEqual(candidate["segment_split_count_on_edge"], 3)
            self.assertEqual(candidate["source_flags"], ["equal_spaced_segment_count"])
            self.assertEqual(candidate["buffer_main_type_index"], 9)
            self.assertEqual(candidate["bsu"], 2)
            self.assertEqual(candidate["bu"], 0.0)

        with self.assertRaisesRegex(ValueError, "repeater_count"):
            build_equal_spaced_segment_candidates(net, parent_node_id=0, child_node_id=1, repeater_count=-1)

        missing_coordinate = dict(net)
        missing_coordinate["coordinates"] = {0: (0, 0)}
        with self.assertRaisesRegex(ValueError, "coordinate"):
            build_equal_spaced_segment_candidates(
                missing_coordinate,
                parent_node_id=0,
                child_node_id=1,
                repeater_count=1,
            )

        with self.assertRaisesRegex(ValueError, "topology"):
            build_equal_spaced_segment_candidates(net, parent_node_id=1, child_node_id=0, repeater_count=1)

    def test_transfer_definitions_match_expanded_oracle_and_surrogate_load_cap(self):
        surrogate = RecordingSurrogate(input_cap=0.4, delay=1.5, output_slew=0.25)

        result = evaluate_expanded_segment_reference(
            [self._line_net()],
            net_id=41,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=1,
            bsu=3,
            buffer_surrogate=surrogate,
            driver_slew_by_net={41: 0.1},
        )

        diagnostics = result["diagnostics"]
        self.assertEqual(len(surrogate.calls), 1)
        self.assertAlmostEqual(
            surrogate.calls[0]["output_cap"],
            diagnostics["coordinate_candidate_lout"][0],
        )
        self.assertAlmostEqual(
            surrogate.calls[0]["input_slew"],
            diagnostics["coordinate_candidate_input_slew"][0],
        )

        coordinate_inputs = diagnostics["coordinate_inputs"]
        first_synthetic = diagnostics["coordinate_node_ids"]["candidate_compact_nodes"][0]
        self.assertAlmostEqual(float(coordinate_inputs["node_capacitance"][first_synthetic]), 0.0)

        expanded = diagnostics["expanded_result"]
        node_ids = diagnostics["expanded_node_ids"]
        expected_delay = (
            expanded["arrival_in"][node_ids["child_compact"]]
            - expanded["arrival_out"][node_ids["parent_compact"]]
        )
        self.assertTrue(torch.allclose(result["delay"], expected_delay))
        self.assertTrue(torch.allclose(result["output_slew"], expanded["slew_in"][node_ids["child_compact"]]))
        self.assertTrue(
            torch.allclose(
                result["upstream_visible_input_cap"],
                expanded["lin"][node_ids["first_synthetic_compact"]],
            )
        )
        self.assertTrue(torch.isfinite(result["delay"]))
        self.assertTrue(torch.isfinite(result["output_slew"]))
        self.assertTrue(torch.isfinite(result["upstream_visible_input_cap"]))

    def test_zero_repeater_degenerates_to_ordinary_edge_and_child_input_load(self):
        result = segment_repeater_transfer(
            [self._line_net()],
            net_id=41,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=0,
            driver_slew_by_net={41: 0.1},
        )

        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["coordinate_candidates"], [])
        self.assertEqual(diagnostics["expanded_candidates"], [])

        expanded = diagnostics["expanded_result"]
        node_ids = diagnostics["expanded_node_ids"]
        self.assertTrue(
            torch.allclose(
                result["delay"],
                expanded["arrival_in"][node_ids["child_compact"]]
                - expanded["arrival_out"][node_ids["parent_compact"]],
            )
        )
        self.assertTrue(
            torch.allclose(
                result["upstream_visible_input_cap"],
                expanded["lin"][node_ids["child_compact"]],
            )
        )

    def test_expanded_native_matches_python_reference_and_preserves_sibling_branch(self):
        surrogate = RecordingSurrogate(input_cap=0.6, delay=2.0, output_slew=0.3)
        result = evaluate_expanded_segment_reference(
            [self._branch_net()],
            net_id=42,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=2,
            bsu=1,
            buffer_surrogate=surrogate,
            driver_slew_by_net={42: 0.2},
        )
        diagnostics = result["diagnostics"]
        expanded_inputs = diagnostics["expanded_inputs"]
        native = diagnostics["expanded_result"]
        reference = net_subgraph_forward(**_native_inputs_only(expanded_inputs))

        for key in (
            "lout",
            "lin",
            "arrival_in",
            "arrival_out",
            "slew_in",
            "slew_out",
            "sink_arrival",
            "sink_slew",
            "sink_load",
        ):
            self.assertTrue(torch.allclose(native[key], reference[key], atol=1e-6), key)

        by_net = expanded_inputs["metadata"]["original_node_to_compact_by_net"][42]
        root_compact = by_net[0]
        sibling_compact = by_net[2]
        self.assertEqual(int(expanded_inputs["pin_fa"][sibling_compact]), root_compact)

        node_ids = diagnostics["expanded_node_ids"]
        self.assertTrue(
            torch.allclose(
                result["upstream_visible_input_cap"],
                native["lin"][node_ids["first_synthetic_compact"]],
            )
        )

    def test_rejects_missing_bsu_for_inserted_repeater(self):
        with self.assertRaisesRegex(ValueError, "bsu"):
            evaluate_expanded_segment_reference(
                [self._line_net()],
                net_id=41,
                parent_node_id=0,
                child_node_id=1,
                repeater_count=1,
                buffer_surrogate=RecordingSurrogate(),
            )


if __name__ == "__main__":
    unittest.main()
