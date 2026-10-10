import unittest

import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    analytic_segment_transfer,
    build_segment_transfer_feature_row,
    expanded_segment_transfer_oracle,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.models import (
    build_segment_transfer_model,
)


class SegmentTransferBoundaryTest(unittest.TestCase):
    def test_oracle_wrapper_is_separate_from_analytic_transfer(self):
        self.assertIsNotNone(expanded_segment_transfer_oracle)
        self.assertNotEqual(analytic_segment_transfer, expanded_segment_transfer_oracle)

    def test_feature_builder_exposes_local_contract(self):
        transfer_input = SegmentTransferInput(
            input_arrival=0.0,
            input_slew=torch.tensor(1.0, dtype=torch.float64),
            downstream_load=torch.tensor(2.0, dtype=torch.float64),
            edge_resistance=torch.tensor(3.0, dtype=torch.float64),
            edge_capacitance=torch.tensor(4.0, dtype=torch.float64),
            upstream_retained_capacitance=torch.tensor(0.25, dtype=torch.float64),
            repeater_count=1,
            bsu_index=torch.tensor(0.5, dtype=torch.float64),
            buffer_device=default_fake_buffer_device(),
        )
        result = analytic_segment_transfer(transfer_input)
        row = build_segment_transfer_feature_row(transfer_input, result)

        for key in (
            "input_slew",
            "downstream_load",
            "edge_resistance",
            "edge_capacitance",
            "upstream_retained_capacitance",
            "repeater_count",
            "bsu_index",
            "segment_delay",
            "output_slew",
            "upstream_visible_load",
        ):
            self.assertIn(key, row)

    def test_residual_model_requires_explicit_registry_selection(self):
        model = build_segment_transfer_model("residual_mlp", input_dim=6, hidden_dim=4)
        output = model(torch.ones(2, 6))
        self.assertEqual(tuple(output.shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_segment_transfer_model("default", input_dim=6)


if __name__ == "__main__":
    unittest.main()
