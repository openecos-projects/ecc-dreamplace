import unittest

import torch

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_prepared_timing import (
    segment_count_prepared_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    segment_count_relaxed_timing,
    _split_fractions,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    batched_segment_transfer,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


class SegmentCountPreparedTimingTest(unittest.TestCase):
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
                    (0, 1): {"r": 10.0, "c": 0.5},
                    (0, 2): {"r": 6.0, "c": 0.25},
                    (1, 3): {"r": 4.0, "c": 0.125},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 5.0, 3: 3.0},
                "sink_nodes": [2, 3],
            },
        }

    def _state(self, nets, *, z=1.25, bsu=1.5):
        return build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=z,
            initial_bsu_index=bsu,
            dtype=torch.float64,
        )

    def _tables(self, segment_count):
        input_cap = torch.tensor([0.3, 0.4, 0.6, 0.9], dtype=torch.float64).repeat(segment_count, 1)
        delay = torch.tensor([1.0, 2.0, 3.5, 5.0], dtype=torch.float64).repeat(segment_count, 1)
        output_slew = torch.tensor([0.10, 0.20, 0.35, 0.50], dtype=torch.float64).repeat(segment_count, 1)
        return input_cap, delay, output_slew

    def _line_net(self):
        return {
            "net_id": 7,
            "net_name": "line",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 2.0, "c": 0.5}},
                "node_cap": {0: 0.0, 1: 3.0},
                "sink_nodes": [1],
            },
        }

    def test_prepared_evaluator_matches_reference_values(self):
        net = self._branch_net()
        state = self._state([net])
        tables = self._tables(len(state.segment_rows))

        reference = segment_count_relaxed_timing(
            [net],
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival_by_net={42: torch.tensor(4.0, dtype=torch.float64)},
            driver_slew_by_net={42: torch.tensor(0.2, dtype=torch.float64)},
        )
        prepared = segment_count_prepared_relaxed_timing(
            build_segment_count_timing_inputs([net], state, dtype=torch.float64),
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=torch.tensor([4.0], dtype=torch.float64),
            driver_slew=torch.tensor([0.2], dtype=torch.float64),
            canonical_equal_spacing=True,
        )

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
            "sink_node_id",
            "sink_net_index",
        ):
            torch.testing.assert_close(prepared[key], reference[key])

    def test_prepared_evaluator_matches_reference_gradients(self):
        net = self._branch_net()
        reference_state = self._state([net])
        prepared_state = self._state([net])
        tables = self._tables(len(reference_state.segment_rows))

        reference = segment_count_relaxed_timing(
            [net],
            segment_state=reference_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_slew_by_net={42: torch.tensor(0.2, dtype=torch.float64)},
        )
        prepared = segment_count_prepared_relaxed_timing(
            build_segment_count_timing_inputs([net], prepared_state, dtype=torch.float64),
            segment_state=prepared_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_slew=torch.tensor([0.2], dtype=torch.float64),
            canonical_equal_spacing=True,
        )

        (reference["sink_arrival"].sum() + reference["sink_slew"].sum()).backward()
        (prepared["sink_arrival"].sum() + prepared["sink_slew"].sum()).backward()

        torch.testing.assert_close(
            prepared_state.z_param.grad,
            reference_state.z_param.grad,
        )
        torch.testing.assert_close(
            prepared_state.bsu_index_param.grad,
            reference_state.bsu_index_param.grad,
        )

    def test_live_rc_override_preserves_values_and_returns_rc_gradients(self):
        net = self._line_net()
        state = self._state([net], z=1.0, bsu=1.0)
        tables = self._tables(len(state.segment_rows))
        prepared_inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
        )
        static_result = segment_count_prepared_relaxed_timing(
            prepared_inputs,
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_slew=torch.tensor([0.2], dtype=torch.float64),
        )
        edge_r = prepared_inputs["edge_resistance"].clone().requires_grad_(True)
        edge_c = prepared_inputs["edge_capacitance"].clone().requires_grad_(True)
        live_result = segment_count_prepared_relaxed_timing(
            prepared_inputs,
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_slew=torch.tensor([0.2], dtype=torch.float64),
            edge_resistance_override=edge_r,
            edge_capacitance_override=edge_c,
        )

        for key in ("sink_arrival", "sink_slew", "sink_load"):
            torch.testing.assert_close(live_result[key], static_result[key])
        objective = (
            live_result["sink_arrival"].sum()
            + live_result["sink_slew"].sum()
            + live_result["sink_load"].sum()
        )
        grad_r, grad_c = torch.autograd.grad(objective, (edge_r, edge_c))
        self.assertTrue(bool(torch.isfinite(grad_r).all()))
        self.assertTrue(bool(torch.isfinite(grad_c).all()))
        self.assertTrue(bool((grad_r != 0.0).any()))
        self.assertTrue(bool((grad_c != 0.0).any()))
        self.assertTrue(live_result["metadata"]["live_edge_override"])

    def test_live_rc_override_matches_finite_difference(self):
        net = self._line_net()
        state = self._state([net], z=1.0, bsu=1.0)
        tables = self._tables(len(state.segment_rows))
        prepared_inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
        )

        def objective(edge_r, edge_c):
            result = segment_count_prepared_relaxed_timing(
                prepared_inputs,
                segment_state=state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_slew=torch.tensor([0.2], dtype=torch.float64),
                edge_resistance_override=edge_r,
                edge_capacitance_override=edge_c,
                canonical_equal_spacing=True,
            )
            return (
                result["sink_arrival"].sum()
                + result["sink_slew"].sum()
                + result["sink_load"].sum()
            )

        edge_r = prepared_inputs["edge_resistance"].clone().requires_grad_(True)
        edge_c = prepared_inputs["edge_capacitance"].clone().requires_grad_(True)
        grad_r, grad_c = torch.autograd.grad(
            objective(edge_r, edge_c),
            (edge_r, edge_c),
        )
        epsilon = 1.0e-6

        def central_difference(base_r, base_c, *, vary_r):
            plus_r = base_r.detach().clone()
            minus_r = base_r.detach().clone()
            plus_c = base_c.detach().clone()
            minus_c = base_c.detach().clone()
            if vary_r:
                plus_r[0] += epsilon
                minus_r[0] -= epsilon
            else:
                plus_c[0] += epsilon
                minus_c[0] -= epsilon
            return (
                objective(plus_r, plus_c) - objective(minus_r, minus_c)
            ) / (2.0 * epsilon)

        torch.testing.assert_close(
            grad_r[0],
            central_difference(edge_r, edge_c, vary_r=True),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        torch.testing.assert_close(
            grad_c[0],
            central_difference(edge_r, edge_c, vary_r=False),
            rtol=1.0e-6,
            atol=1.0e-8,
        )

    def test_live_rc_override_requires_matching_pair_and_edge_count(self):
        net = self._line_net()
        state = self._state([net])
        tables = self._tables(len(state.segment_rows))
        prepared_inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
        )
        kwargs = {
            "segment_state": state,
            "per_size_input_cap": tables[0],
            "per_size_delay": tables[1],
            "per_size_output_slew": tables[2],
        }
        with self.assertRaisesRegex(ValueError, "required together"):
            segment_count_prepared_relaxed_timing(
                prepared_inputs,
                edge_resistance_override=torch.tensor([2.0]),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "prepared edge count"):
            segment_count_prepared_relaxed_timing(
                prepared_inputs,
                edge_resistance_override=torch.tensor([2.0, 3.0]),
                edge_capacitance_override=torch.tensor([0.5, 0.6]),
                **kwargs,
            )

    def _assert_prepared_segment_transfer_backend_matches_batched_single_edge(
        self,
        transfer_backend,
    ):
        net = self._line_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=2,
            max_repeater_count=3,
            initial_z=2.0,
            initial_bsu_index=0.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        device = default_fake_buffer_device(dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64)
        driver_slew = torch.tensor([0.2], dtype=torch.float64)

        prepared = segment_count_prepared_relaxed_timing(
            build_segment_count_timing_inputs([net], state, dtype=torch.float64),
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            transfer_backend=transfer_backend,
            buffer_device_lut=device,
        )
        device_tensors = device.tensors(dtype=torch.float64, device=torch.device("cpu"))
        fractions = _split_fractions(state.segment_rows[0], 2)
        expected = batched_segment_transfer(
            input_arrival=prepared["segment_parent_arrival"].detach(),
            input_slew=prepared["segment_parent_slew"].detach(),
            downstream_load=prepared["segment_downstream_load"].detach(),
            edge_resistance=torch.tensor([2.0], dtype=torch.float64),
            edge_capacitance=torch.tensor([0.5], dtype=torch.float64),
            repeater_count=torch.tensor([2], dtype=torch.long),
            split_fractions=torch.tensor([fractions + [0.0]], dtype=torch.float64),
            bsu_index=torch.tensor([0.5], dtype=torch.float64),
            upstream_retained_cap=torch.tensor([0.0], dtype=torch.float64),
            buffer_input_cap_by_size=device_tensors["input_cap_by_size"],
            buffer_slew_axis=device_tensors["input_slew_axis"],
            buffer_load_axis=device_tensors["output_load_axis"],
            buffer_delay_lut=device_tensors["delay_lut"],
            buffer_output_slew_lut=device_tensors["output_slew_lut"],
        )

        torch.testing.assert_close(prepared["segment_delay"], expected["segment_delay"])
        torch.testing.assert_close(prepared["segment_output_slew"], expected["output_slew"])
        torch.testing.assert_close(prepared["sink_arrival"], expected["output_arrival"])
        torch.testing.assert_close(prepared["sink_slew"], expected["output_slew"])
        self.assertEqual(
            prepared["metadata"]["segment_transfer_backend"],
            transfer_backend,
        )

    def test_prepared_segment_transfer_python_backend_matches_batched_single_edge(self):
        self._assert_prepared_segment_transfer_backend_matches_batched_single_edge(
            "segment_transfer_python"
        )

    def test_prepared_segment_transfer_native_backend_matches_batched_single_edge(self):
        self._assert_prepared_segment_transfer_backend_matches_batched_single_edge(
            "segment_transfer_native"
        )

    def test_segment_transfer_zero_count_matches_unbuffered_elmore_load(self):
        net = self._line_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=2,
            max_repeater_count=3,
            initial_z=0.0,
            initial_bsu_index=0.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        kwargs = {
            "segment_state": state,
            "per_size_input_cap": tables[0],
            "per_size_delay": tables[1],
            "per_size_output_slew": tables[2],
            "driver_arrival": torch.tensor([4.0], dtype=torch.float64),
            "driver_slew": torch.tensor([0.2], dtype=torch.float64),
        }
        unbuffered = segment_count_prepared_relaxed_timing(inputs, **kwargs)

        for backend in ("segment_transfer_python", "segment_transfer_native"):
            transferred = segment_count_prepared_relaxed_timing(
                inputs,
                transfer_backend=backend,
                buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
                **kwargs,
            )
            torch.testing.assert_close(
                transferred["segment_delay"],
                unbuffered["segment_delay"],
            )
            torch.testing.assert_close(
                transferred["sink_arrival"],
                unbuffered["sink_arrival"],
            )
            torch.testing.assert_close(
                transferred["sink_slew"],
                unbuffered["sink_slew"],
            )


if __name__ == "__main__":
    unittest.main()
