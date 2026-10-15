import math
import unittest

import torch

from dreamplace.ops.net_subgraph_timing import (
    compute_buffer_aware_rc_forward,
    net_subgraph_forward,
)


class NetSubgraphTimingTest(unittest.TestCase):
    def _line_topology(self):
        return {
            "net_flat_topo_sort": torch.tensor([0, 1, 2], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 3], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0, 1], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1, 2], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 2.0, 3.0]),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0]),
            "driver_arrival": torch.tensor([0.0]),
            "driver_slew": torch.tensor([0.1]),
        }

    def test_no_buffer_matches_tree_load_delay_slew_forward(self):
        topo = self._line_topology()

        result = net_subgraph_forward(
            **topo,
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([0.0]),
            buffer_input_cap=torch.tensor([0.5]),
            buffer_delay=torch.tensor([7.0]),
            buffer_output_slew=torch.tensor([0.2]),
        )

        self.assertAlmostEqual(float(result["lin"][1]), 5.0)
        self.assertAlmostEqual(float(result["arrival_out"][1]), 10.0)
        self.assertAlmostEqual(float(result["arrival_out"][2]), 22.0)
        expected_slew_1 = math.sqrt(0.1 * 0.1 + (math.log(10.0) * 10.0) ** 2)
        self.assertAlmostEqual(float(result["slew_out"][1]), expected_slew_1, places=5)

    def test_inserted_buffer_partitions_load_and_resets_output_slew(self):
        topo = self._line_topology()

        result = net_subgraph_forward(
            **topo,
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([1.0]),
            buffer_input_cap=torch.tensor([0.5]),
            buffer_delay=torch.tensor([7.0]),
            buffer_output_slew=torch.tensor([0.2]),
        )

        self.assertAlmostEqual(float(result["lout"][1]), 5.0)
        self.assertAlmostEqual(float(result["lin"][1]), 0.5)
        self.assertAlmostEqual(float(result["arrival_in"][1]), 1.0)
        self.assertAlmostEqual(float(result["arrival_out"][1]), 8.0)
        self.assertAlmostEqual(float(result["arrival_out"][2]), 20.0)
        self.assertAlmostEqual(float(result["slew_out"][1]), 0.2)

    def test_wire_edge_capacitance_matches_reference_split(self):
        result = net_subgraph_forward(
            net_flat_topo_sort=torch.tensor([0, 1], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 2], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 1], dtype=torch.int32),
            flat_pin_to=torch.tensor([1], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 2.0]),
            edge_capacitance=torch.tensor([0.0, 2.0]),
            node_capacitance=torch.tensor([0.0, 4.0]),
            driver_arrival=torch.tensor([0.0]),
            driver_slew=torch.tensor([0.1]),
            candidate_node_id=torch.tensor([], dtype=torch.int32),
            candidate_bu=torch.tensor([]),
            buffer_input_cap=torch.tensor([]),
            buffer_delay=torch.tensor([]),
            buffer_output_slew=torch.tensor([]),
        )
        reference = compute_buffer_aware_rc_forward(
            {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 2.0, "c": 2.0}},
                "node_cap": {0: 0.0, 1: 4.0},
                "sink_nodes": [1],
            },
            mode="original",
        )

        self.assertAlmostEqual(float(result["effective_node_cap"][0]), 1.0)
        self.assertAlmostEqual(float(result["effective_node_cap"][1]), 5.0)
        self.assertAlmostEqual(float(result["lin"][1]), reference["lin_by_node"][1])
        self.assertAlmostEqual(float(result["arrival_out"][1]), reference["arrival_by_node"][1])

    def test_multi_net_batch_returns_sink_outputs_for_timing_propagation(self):
        result = net_subgraph_forward(
            net_flat_topo_sort=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 2, 4], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0, -1, 2], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32),
            flat_pin_to=torch.tensor([1, 3], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 4.0, 0.0, 5.0]),
            node_capacitance=torch.tensor([0.0, 2.0, 0.0, 3.0]),
            driver_arrival=torch.tensor([10.0, 20.0]),
            driver_slew=torch.tensor([0.1, 0.2]),
            candidate_node_id=torch.tensor([3], dtype=torch.int32),
            candidate_bu=torch.tensor([1.0]),
            buffer_input_cap=torch.tensor([0.5]),
            buffer_delay=torch.tensor([6.0]),
            buffer_output_slew=torch.tensor([0.25]),
            sink_node_id=torch.tensor([1, 3], dtype=torch.int32),
        )

        self.assertAlmostEqual(float(result["arrival_out"][1]), 18.0)
        self.assertAlmostEqual(float(result["arrival_out"][3]), 28.5)
        self.assertTrue(torch.allclose(result["sink_arrival"], torch.tensor([18.0, 28.5])))
        self.assertTrue(torch.allclose(result["sink_net_delay"], torch.tensor([8.0, 8.5])))
        self.assertTrue(
            torch.allclose(
                result["sink_net_impulse"],
                result["sink_slew"] * result["sink_slew"]
                - torch.tensor([0.1 * 0.1, 0.2 * 0.2]),
            )
        )
        self.assertTrue(torch.allclose(result["sink_load"], torch.tensor([2.0, 0.5])))
        self.assertTrue(torch.allclose(result["sink_cap"], torch.tensor([2.0, 3.0])))
        self.assertAlmostEqual(float(result["sink_slew"][1]), 0.25)

    def test_branch_topology_sink_outputs_include_net_index(self):
        result = net_subgraph_forward(
            net_flat_topo_sort=torch.tensor([0, 1, 3, 2], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 4], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0, 0, 1], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32),
            flat_pin_to=torch.tensor([1, 2, 3], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 2.0, 4.0, 3.0]),
            node_capacitance=torch.tensor([0.0, 1.0, 4.0, 2.0]),
            driver_arrival=torch.tensor([0.0]),
            driver_slew=torch.tensor([0.1]),
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([1.0]),
            buffer_input_cap=torch.tensor([0.5]),
            buffer_delay=torch.tensor([7.0]),
            buffer_output_slew=torch.tensor([0.2]),
            sink_node_id=torch.tensor([2, 3], dtype=torch.int32),
            sink_net_index=torch.tensor([0, 0], dtype=torch.int32),
        )

        self.assertAlmostEqual(float(result["lin"][1]), 0.5)
        self.assertAlmostEqual(float(result["arrival_out"][1]), 8.0)
        self.assertAlmostEqual(float(result["arrival_out"][2]), 16.0)
        self.assertAlmostEqual(float(result["arrival_out"][3]), 14.0)
        self.assertTrue(torch.equal(result["sink_net_index"], torch.tensor([0, 0], dtype=torch.int32)))
        self.assertTrue(torch.allclose(result["sink_arrival"], torch.tensor([16.0, 14.0])))


if __name__ == "__main__":
    unittest.main()
