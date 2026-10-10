import unittest

import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    SegmentTransferResult,
    analytic_segment_transfer,
    compare_transfer_results,
    expanded_segment_transfer_oracle,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


class RecordingSurrogate:
    def __init__(self, *, input_cap, delay, output_slew):
        self.input_cap = float(input_cap)
        self.delay = float(delay)
        self.output_slew = float(output_slew)

    def __call__(self, candidate, *, bsu, input_slew, output_cap):
        return {
            "buffer_input_cap": self.input_cap,
            "buffer_delay": self.delay,
            "buffer_output_slew": self.output_slew,
            "input_slew": float(input_slew),
            "output_cap": float(output_cap),
        }


def _line_net(edge_cap=0.0):
    return {
        "net_id": 7,
        "driver_pin_id": 0,
        "coordinates": {0: (0, 0), 1: (100, 0)},
        "rc_tree": {
            "root_node_id": 0,
            "children_by_node": {0: [1], 1: []},
            "edge_rc": {(0, 1): {"r": 2.0, "c": float(edge_cap)}},
            "node_cap": {0: 0.0, 1: 3.0},
            "sink_nodes": [1],
        },
    }


class SegmentTransferOracleAlignmentTest(unittest.TestCase):
    def test_n0_matches_expanded_oracle_on_single_edge(self):
        oracle = expanded_segment_transfer_oracle(
            [_line_net(edge_cap=0.0)],
            net_id=7,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=0,
            driver_slew_by_net={7: 0.5},
            dtype=torch.float64,
        )
        analytic = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(0.5, dtype=torch.float64),
                downstream_load=torch.tensor(3.0, dtype=torch.float64),
                edge_resistance=torch.tensor(2.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=0,
            )
        )
        reference = SegmentTransferResult(
            upstream_visible_load=oracle["upstream_visible_input_cap"],
            segment_delay=oracle["delay"],
            output_slew=oracle["output_slew"],
            output_arrival=oracle["delay"],
            diagnostics=oracle["diagnostics"],
        )
        comparison = compare_transfer_results(reference, analytic)
        torch.testing.assert_close(comparison["max_abs_diff"], torch.tensor(0.0, dtype=torch.float64))

    def test_n1_oracle_path_is_available_but_not_the_analytic_hot_path(self):
        oracle = expanded_segment_transfer_oracle(
            [_line_net(edge_cap=0.0)],
            net_id=7,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=1,
            bsu=0,
            buffer_surrogate=RecordingSurrogate(
                input_cap=0.4,
                delay=1.25,
                output_slew=0.75,
            ),
            driver_slew_by_net={7: 0.5},
            dtype=torch.float64,
        )
        analytic = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(0.5, dtype=torch.float64),
                downstream_load=torch.tensor(3.0, dtype=torch.float64),
                edge_resistance=torch.tensor(2.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=torch.tensor(0.0, dtype=torch.float64),
                buffer_device=default_fake_buffer_device(),
            )
        )

        self.assertIn("expanded_inputs", oracle["diagnostics"])
        self.assertNotIn("expanded_inputs", analytic.diagnostics)
        self.assertTrue(torch.isfinite(oracle["delay"]))
        self.assertTrue(torch.isfinite(analytic.segment_delay))


if __name__ == "__main__":
    unittest.main()
