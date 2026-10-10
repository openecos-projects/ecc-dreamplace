import unittest

import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    BufferDeviceLut,
    SegmentTransferInput,
    analytic_segment_transfer,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.schema import (
    SegmentTransferDatasetSchema,
)


class SegmentTransferSchemaTest(unittest.TestCase):
    def test_package_import_and_n0_structured_output(self):
        transfer_input = SegmentTransferInput(
            input_arrival=torch.tensor(2.0, dtype=torch.float64),
            input_slew=torch.tensor(0.5, dtype=torch.float64),
            downstream_load=torch.tensor(3.0, dtype=torch.float64),
            edge_resistance=torch.tensor(4.0, dtype=torch.float64),
            edge_capacitance=torch.tensor(2.0, dtype=torch.float64),
            repeater_count=0,
        )

        result = analytic_segment_transfer(transfer_input)

        self.assertTrue(torch.is_tensor(result.upstream_visible_load))
        self.assertTrue(torch.is_tensor(result.segment_delay))
        self.assertTrue(torch.is_tensor(result.output_slew))
        self.assertTrue(torch.is_tensor(result.output_arrival))
        self.assertEqual(result.diagnostics["repeater_count"], 0)
        self.assertNotIn("buffer_delay", result.diagnostics)

    def test_invalid_local_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "input_slew"):
            analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=0.0,
                    input_slew=-1.0,
                    downstream_load=1.0,
                    edge_resistance=1.0,
                    edge_capacitance=1.0,
                    repeater_count=0,
                )
            )

        with self.assertRaisesRegex(ValueError, "edge_resistance"):
            analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=0.0,
                    input_slew=1.0,
                    downstream_load=1.0,
                    edge_resistance=-1.0,
                    edge_capacitance=1.0,
                    repeater_count=0,
                )
            )

        with self.assertRaisesRegex(ValueError, "buffer_device"):
            analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=0.0,
                    input_slew=1.0,
                    downstream_load=1.0,
                    edge_resistance=1.0,
                    edge_capacitance=1.0,
                    repeater_count=1,
                )
            )

        with self.assertRaisesRegex(ValueError, "repeater_count"):
            analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=0.0,
                    input_slew=1.0,
                    downstream_load=1.0,
                    edge_resistance=1.0,
                    edge_capacitance=1.0,
                    repeater_count=-1,
                    buffer_device=default_fake_buffer_device(),
                )
            )

    def test_buffer_device_lut_schema_rejects_static_tables_for_n1(self):
        bad_device = BufferDeviceLut(
            input_cap_by_size=torch.tensor([0.4, 0.8], dtype=torch.float64),
            input_slew_axis=torch.tensor([0.0, 10.0], dtype=torch.float64),
            output_load_axis=torch.tensor([0.0, 10.0], dtype=torch.float64),
            delay_lut=torch.tensor([1.0, 2.0], dtype=torch.float64),
            output_slew_lut=torch.tensor([0.5, 0.8], dtype=torch.float64),
        )

        with self.assertRaisesRegex(ValueError, "delay_lut"):
            analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=0.0,
                    input_slew=1.0,
                    downstream_load=1.0,
                    edge_resistance=1.0,
                    edge_capacitance=1.0,
                    repeater_count=1,
                    buffer_device=bad_device,
                )
            )

    def test_dataset_schema_includes_local_transfer_features_and_labels(self):
        schema = SegmentTransferDatasetSchema()

        self.assertIn("input_arrival", schema.feature_names)
        self.assertIn("input_slew", schema.feature_names)
        self.assertIn("downstream_load", schema.feature_names)
        self.assertIn("edge_resistance", schema.feature_names)
        self.assertIn("edge_capacitance", schema.feature_names)
        self.assertIn("upstream_retained_capacitance", schema.feature_names)
        self.assertIn("bsu_index", schema.feature_names)
        self.assertIn("upstream_visible_load", schema.label_names)
        self.assertIn("segment_delay", schema.label_names)
        self.assertIn("output_arrival", schema.label_names)
        self.assertIn("output_slew", schema.label_names)


if __name__ == "__main__":
    unittest.main()
