import unittest
from types import SimpleNamespace

import torch

from dreamplace.ops.buffer_insertion.buffer_library import (
    lookup_buffer_delay_for_bsu,
    lookup_buffer_transition_for_bsu,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    analytic_segment_transfer,
    build_buffer_device_lut_from_metadata,
)


def _metadata():
    return SimpleNamespace(
        cell_id_2_libpin_id_start=[0, 2, 4],
        flat_lib_pin_cap=[0.02, 0.0, 0.04, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2],
        f_delay_flat_luts_values=[
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        f_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
        f_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
        f_delay_flat_luts_dim=[[2, 2], [2, 2]],
        r_delay_flat_luts_values=[
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 5.0, 7.0, 9.0],
        ],
        r_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
        r_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
        r_delay_flat_luts_dim=[[2, 2], [2, 2]],
        f_trans_flat_luts_values=[
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
        ],
        f_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
        f_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
        f_trans_flat_luts_dim=[[2, 2], [2, 2]],
        r_trans_flat_luts_values=[
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9],
        ],
        r_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
        r_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
        r_trans_flat_luts_dim=[[2, 2], [2, 2]],
    )


def _contract():
    return {
        "status": "ok",
        "legal_cell_ids": [0, 1],
        "legal_master_names": ["BUF_X1", "BUF_X2"],
        "legal_size_values": [1.0, 2.0],
    }


class SegmentTransferMetadataAdapterTest(unittest.TestCase):
    def test_phase_device_propagates_distinct_liberty_delay_slew_and_input_cap(self):
        metadata = _metadata()
        metadata.flat_lib_pin_rcap = [0.03, 0.0, 0.06, 0.0]
        metadata.flat_lib_pin_fcap = [0.01, 0.0, 0.02, 0.0]
        device = build_buffer_device_lut_from_metadata(metadata, _contract())
        for sense, delay, slew, cap in (("rise", 6.0, 0.6, 0.06), ("fall", 5.0, 0.5, 0.02)):
            result = analytic_segment_transfer(SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(5.0, dtype=torch.float64),
                downstream_load=torch.tensor(10.0, dtype=torch.float64),
                edge_resistance=torch.tensor(0.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1, bsu_index=1.0, buffer_device=device.phase_luts[sense],
            ))
            torch.testing.assert_close(
                torch.stack([result.output_arrival, result.output_slew,
                             result.diagnostics["buffer_input_cap"]]),
                torch.tensor([delay, slew, cap], dtype=torch.float64),
            )

    def test_metadata_adapter_matches_existing_bsu_lut_lookup(self):
        metadata = _metadata()
        contract = _contract()
        device = build_buffer_device_lut_from_metadata(metadata, contract)

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(5.0, dtype=torch.float64),
                downstream_load=torch.tensor(10.0, dtype=torch.float64),
                edge_resistance=torch.tensor(0.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=torch.tensor(1.0, dtype=torch.float64),
                buffer_device=device,
            )
        )

        expected_delay, delay_source = lookup_buffer_delay_for_bsu(
            metadata,
            contract,
            bsu=1,
            input_slew=5.0,
            output_cap=10.0,
        )
        expected_slew, transition_source = lookup_buffer_transition_for_bsu(
            metadata,
            contract,
            bsu=1,
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertEqual(device.source, "metadata_contract_lut")
        self.assertEqual(len(device.arc_luts_by_size), 2)
        self.assertEqual(
            sorted(
                (row["arc_id"], row["prefix"])
                for row in device.arc_luts_by_size[1]
            ),
            [(1, "f"), (1, "r")],
        )
        self.assertEqual(delay_source, "lut_bilinear_interpolation")
        self.assertEqual(transition_source, "lut_bilinear_interpolation")
        torch.testing.assert_close(
            result.diagnostics["buffer_delay"],
            torch.tensor(expected_delay, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_output_slew"],
            torch.tensor(expected_slew, dtype=torch.float64),
        )
        torch.testing.assert_close(result.diagnostics["buffer_input_cap"], torch.tensor(0.04, dtype=torch.float64))

    def test_fractional_bsu_interpolates_legal_buffer_lut_tensors(self):
        device = build_buffer_device_lut_from_metadata(_metadata(), _contract())
        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(5.0, dtype=torch.float64),
                downstream_load=torch.tensor(10.0, dtype=torch.float64),
                edge_resistance=torch.tensor(0.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
                buffer_device=device,
            )
        )

        self.assertAlmostEqual(float(result.diagnostics["buffer_delay"]), 5.0)

    def test_resamples_mismatched_lut_axes_across_legal_sizes(self):
        metadata = _metadata()
        metadata.f_delay_flat_luts_trans_table[1] = [0.0, 5.0]
        metadata.r_delay_flat_luts_trans_table[1] = [0.0, 5.0]
        metadata.f_trans_flat_luts_trans_table[1] = [0.0, 5.0]
        metadata.r_trans_flat_luts_trans_table[1] = [0.0, 5.0]

        device = build_buffer_device_lut_from_metadata(metadata, _contract())
        tensors = device.tensors(dtype=torch.float64, device=torch.device("cpu"))
        torch.testing.assert_close(
            tensors["input_slew_axis"],
            torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64),
        )

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(5.0, dtype=torch.float64),
                downstream_load=torch.tensor(10.0, dtype=torch.float64),
                edge_resistance=torch.tensor(0.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=torch.tensor(1.0, dtype=torch.float64),
                buffer_device=device,
            )
        )
        expected_delay, _source = lookup_buffer_delay_for_bsu(
            metadata,
            _contract(),
            bsu=1,
            input_slew=5.0,
            output_cap=10.0,
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_delay"],
            torch.tensor(expected_delay, dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
