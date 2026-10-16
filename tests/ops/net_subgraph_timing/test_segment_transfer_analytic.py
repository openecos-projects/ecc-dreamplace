import unittest

import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    analytic_segment_transfer,
    compare_transfer_results,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data import (
    build_deterministic_segment_transfer_dataset,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


class SegmentTransferAnalyticTest(unittest.TestCase):
    def test_n0_matches_documented_elmore_baseline_and_has_gradients(self):
        input_slew = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        downstream_load = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        edge_resistance = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
        edge_capacitance = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(2.0, dtype=torch.float64),
                input_slew=input_slew,
                downstream_load=downstream_load,
                edge_resistance=edge_resistance,
                edge_capacitance=edge_capacitance,
                repeater_count=0,
            )
        )

        expected_delay = edge_resistance * (downstream_load + 0.5 * edge_capacitance)
        torch.testing.assert_close(result.segment_delay, expected_delay)
        torch.testing.assert_close(result.upstream_visible_load, downstream_load + edge_capacitance)
        self.assertNotIn("buffer_input_cap", result.diagnostics)

        loss = result.segment_delay + result.output_slew + result.upstream_visible_load
        loss.backward()
        for tensor in (input_slew, downstream_load, edge_resistance, edge_capacitance):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad))

    def test_n1_delay_and_slew_depend_on_input_slew_and_output_load(self):
        device = default_fake_buffer_device()

        def run(input_slew, downstream_load):
            return analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=torch.tensor(0.0, dtype=torch.float64),
                    input_slew=torch.tensor(input_slew, dtype=torch.float64),
                    downstream_load=torch.tensor(downstream_load, dtype=torch.float64),
                    edge_resistance=torch.tensor(2.0, dtype=torch.float64),
                    edge_capacitance=torch.tensor(1.0, dtype=torch.float64),
                    repeater_count=1,
                    bsu_index=torch.tensor(0.5, dtype=torch.float64),
                    buffer_device=device,
                )
            )

        low_slew = run(1.0, 3.0)
        high_slew = run(8.0, 3.0)
        high_load = run(1.0, 8.0)

        self.assertNotEqual(
            float(low_slew.diagnostics["buffer_delay"]),
            float(high_slew.diagnostics["buffer_delay"]),
        )
        self.assertNotEqual(
            float(low_slew.diagnostics["buffer_output_slew"]),
            float(high_slew.diagnostics["buffer_output_slew"]),
        )
        self.assertNotEqual(
            float(low_slew.diagnostics["buffer_delay"]),
            float(high_load.diagnostics["buffer_delay"]),
        )
        self.assertNotEqual(
            float(low_slew.diagnostics["buffer_output_slew"]),
            float(high_load.diagnostics["buffer_output_slew"]),
        )
        for key in (
            "buffer_input_slew",
            "buffer_output_load",
            "buffer_delay",
            "buffer_output_slew",
        ):
            self.assertIn(key, low_slew.diagnostics)

    def test_n1_device_lookup_extrapolates_like_opensta_for_out_of_range_load(self):
        device = default_fake_buffer_device()
        output_load = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=torch.tensor(0.0, dtype=torch.float64),
                downstream_load=output_load,
                edge_resistance=torch.tensor(0.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(0.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=torch.tensor(0.0, dtype=torch.float64),
                buffer_device=device,
            )
        )

        # The fake delay table has slope 0.2 per load-table index and load
        # axis spacing 5.0, so output_load=15 extrapolates to index alpha=3.
        torch.testing.assert_close(
            result.diagnostics["buffer_delay"],
            torch.tensor(1.6, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_output_slew"],
            torch.tensor(0.8, dtype=torch.float64),
        )
        result.diagnostics["buffer_delay"].backward()
        torch.testing.assert_close(
            output_load.grad,
            torch.tensor(0.04, dtype=torch.float64),
        )

    def test_n1_gradients_flow_to_bsu_and_local_inputs(self):
        device = default_fake_buffer_device()
        input_slew = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
        downstream_load = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        bsu_index = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(0.0, dtype=torch.float64),
                input_slew=input_slew,
                downstream_load=downstream_load,
                edge_resistance=torch.tensor(2.0, dtype=torch.float64),
                edge_capacitance=torch.tensor(1.0, dtype=torch.float64),
                repeater_count=1,
                bsu_index=bsu_index,
                buffer_device=device,
            )
        )
        loss = result.segment_delay + result.output_slew + result.upstream_visible_load
        loss.backward()

        for tensor in (input_slew, downstream_load, bsu_index):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad))
            self.assertNotEqual(float(tensor.grad), 0.0)

    def test_n1_retained_upstream_cap_affects_parent_visible_load_only(self):
        device = default_fake_buffer_device()
        retained_cap = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

        base_input = dict(
            input_arrival=torch.tensor(0.0, dtype=torch.float64),
            input_slew=torch.tensor(4.0, dtype=torch.float64),
            downstream_load=torch.tensor(3.0, dtype=torch.float64),
            edge_resistance=torch.tensor(2.0, dtype=torch.float64),
            edge_capacitance=torch.tensor(1.0, dtype=torch.float64),
            repeater_count=1,
            bsu_index=torch.tensor(0.5, dtype=torch.float64),
            buffer_device=device,
        )
        without_retained = analytic_segment_transfer(SegmentTransferInput(**base_input))
        with_retained = analytic_segment_transfer(
            SegmentTransferInput(
                **base_input,
                upstream_retained_capacitance=retained_cap,
            )
        )

        torch.testing.assert_close(
            with_retained.upstream_visible_load,
            without_retained.upstream_visible_load + retained_cap,
        )
        torch.testing.assert_close(
            with_retained.segment_delay,
            without_retained.segment_delay,
        )
        torch.testing.assert_close(
            with_retained.output_slew,
            without_retained.output_slew,
        )
        self.assertIn(
            "upstream_retained_capacitance",
            with_retained.diagnostics,
        )

        with_retained.upstream_visible_load.backward()
        self.assertIsNotNone(retained_cap.grad)
        torch.testing.assert_close(retained_cap.grad, torch.tensor(1.0, dtype=torch.float64))

    def test_dataset_sampler_and_validation_helper_are_local(self):
        rows = build_deterministic_segment_transfer_dataset()
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertIn("input", row)
            self.assertIn("label", row)
            self.assertIn("segment_delay", row["label"])
            self.assertIn("output_arrival", row["label"])

        result = analytic_segment_transfer(rows[0]["input"])
        comparison = compare_transfer_results(result, result)
        torch.testing.assert_close(comparison["max_abs_diff"], torch.tensor(0.0, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
