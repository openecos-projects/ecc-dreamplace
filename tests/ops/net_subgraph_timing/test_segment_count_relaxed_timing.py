import inspect
import unittest

import torch

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    _interpolate_count_states,
    segment_count_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_repeater_transfer import (
    segment_repeater_transfer,
)


class TableSurrogate:
    def __init__(self, input_cap, delay, output_slew):
        self.input_cap = list(input_cap)
        self.delay = list(delay)
        self.output_slew = list(output_slew)

    def __call__(self, candidate, *, bsu, input_slew, output_cap):
        index = int(bsu)
        return {
            "buffer_input_cap": float(self.input_cap[index]),
            "buffer_delay": float(self.delay[index]),
            "buffer_output_slew": float(self.output_slew[index]),
        }


class SegmentCountRelaxedTimingTest(unittest.TestCase):
    def _line_net(self):
        return {
            "net_id": 41,
            "net_name": "line",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 10.0, "c": 0.0}},
                "node_cap": {0: 0.0, 1: 6.0},
                "sink_nodes": [1],
            },
        }

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
                    (0, 1): {"r": 10.0, "c": 0.0},
                    (0, 2): {"r": 6.0, "c": 0.0},
                    (1, 3): {"r": 4.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 5.0, 3: 3.0},
                "sink_nodes": [2, 3],
            },
        }

    def _tables(self, segment_count, dtype=torch.float64):
        input_cap = torch.tensor([0.3, 0.4, 0.6, 0.9], dtype=dtype).repeat(segment_count, 1)
        delay = torch.tensor([1.0, 2.0, 3.5, 5.0], dtype=dtype).repeat(segment_count, 1)
        output_slew = torch.tensor([0.10, 0.20, 0.35, 0.50], dtype=dtype).repeat(segment_count, 1)
        return input_cap, delay, output_slew

    def test_zero_count_uses_right_adjacent_state_gradient(self):
        z_value = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        states = [
            torch.tensor(3.0, dtype=torch.float64),
            torch.tensor(7.0, dtype=torch.float64),
            torch.tensor(15.0, dtype=torch.float64),
        ]

        result = _interpolate_count_states(states, z_value, max_repeater_count=2)
        gradient, = torch.autograd.grad(result, z_value)

        self.assertEqual(float(result), 3.0)
        self.assertEqual(float(gradient), 4.0)

    def _state(self, nets, *, z=0.0, bsu=1.0, max_repeater_count=3):
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=max_repeater_count,
            initial_z=z,
            initial_bsu_index=bsu,
            dtype=torch.float64,
        )
        return state

    def test_integer_count_endpoints_match_expanded_reference_on_line_net(self):
        net = self._line_net()
        for repeater_count in range(4):
            state = self._state([net], z=float(repeater_count), bsu=1.0)
            per_size_input_cap, per_size_delay, per_size_output_slew = self._tables(
                len(state.segment_rows)
            )

            result = segment_count_relaxed_timing(
                [net],
                segment_state=state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                driver_slew_by_net={41: 0.1},
            )
            reference = segment_repeater_transfer(
                [net],
                net_id=41,
                parent_node_id=0,
                child_node_id=1,
                repeater_count=repeater_count,
                bsu=1 if repeater_count else None,
                buffer_surrogate=TableSurrogate(
                    per_size_input_cap[0].tolist(),
                    per_size_delay[0].tolist(),
                    per_size_output_slew[0].tolist(),
                )
                if repeater_count
                else None,
                driver_slew_by_net={41: 0.1},
                dtype=torch.float64,
            )

            torch.testing.assert_close(result["segment_delay"][0], reference["delay"], atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(result["segment_output_slew"][0], reference["output_slew"], atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(
                result["segment_upstream_visible_input_cap"][0],
                reference["upstream_visible_input_cap"],
                atol=1e-7,
                rtol=1e-7,
            )

    def test_branch_net_endpoint_does_not_decompose_independent_paths(self):
        net = self._branch_net()
        state = self._state([net], z=0.0, bsu=1.0, max_repeater_count=3)
        with torch.no_grad():
            state.z_param.zero_()
            state.z_param[0] = 2.0
        per_size_input_cap, per_size_delay, per_size_output_slew = self._tables(
            len(state.segment_rows)
        )

        result = segment_count_relaxed_timing(
            [net],
            segment_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            driver_slew_by_net={42: 0.2},
        )
        reference = segment_repeater_transfer(
            [net],
            net_id=42,
            parent_node_id=0,
            child_node_id=1,
            repeater_count=2,
            bsu=1,
            buffer_surrogate=TableSurrogate(
                per_size_input_cap[0].tolist(),
                per_size_delay[0].tolist(),
                per_size_output_slew[0].tolist(),
            ),
            driver_slew_by_net={42: 0.2},
            dtype=torch.float64,
        )

        self.assertEqual(result["metadata"]["net_count"], 1)
        self.assertEqual(result["metadata"]["segment_count_state_count"], 3)
        torch.testing.assert_close(result["segment_delay"][0], reference["delay"], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(result["segment_output_slew"][0], reference["output_slew"], atol=1e-7, rtol=1e-7)

    def test_fractional_count_interpolates_adjacent_integer_states(self):
        net = self._line_net()
        per_size = None
        integer_outputs = []
        for z_value in (1.0, 2.0):
            state = self._state([net], z=z_value, bsu=1.0)
            per_size = self._tables(len(state.segment_rows))
            integer_outputs.append(
                segment_count_relaxed_timing(
                    [net],
                    segment_state=state,
                    per_size_input_cap=per_size[0],
                    per_size_delay=per_size[1],
                    per_size_output_slew=per_size[2],
                    driver_slew_by_net={41: 0.1},
                )
            )

        fractional_state = self._state([net], z=1.25, bsu=1.0)
        result = segment_count_relaxed_timing(
            [net],
            segment_state=fractional_state,
            per_size_input_cap=per_size[0],
            per_size_delay=per_size[1],
            per_size_output_slew=per_size[2],
            driver_slew_by_net={41: 0.1},
        )

        expected_delay = 0.75 * integer_outputs[0]["segment_delay"][0] + 0.25 * integer_outputs[1]["segment_delay"][0]
        expected_slew = 0.75 * integer_outputs[0]["segment_output_slew"][0] + 0.25 * integer_outputs[1]["segment_output_slew"][0]
        torch.testing.assert_close(result["segment_delay"][0], expected_delay)
        torch.testing.assert_close(result["segment_output_slew"][0], expected_slew)

    def test_integer_count_gradient_has_the_next_action_direction(self):
        net = self._line_net()
        tables = self._tables(1)

        def evaluate(z_value, *, requires_grad):
            state = self._state([net], z=z_value, bsu=1.0)
            if not requires_grad:
                state.z_param.requires_grad_(False)
            result = segment_count_relaxed_timing(
                [net],
                segment_state=state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_slew_by_net={41: 0.1},
            )
            loss = result["sink_arrival"].sum() + result["sink_slew"].sum()
            return state, loss

        state, loss_at_zero = evaluate(0.0, requires_grad=True)
        grad_z, = torch.autograd.grad(loss_at_zero, state.z_param)
        _next_state, loss_at_one = evaluate(1.0, requires_grad=False)
        discrete_delta = loss_at_one - loss_at_zero.detach()

        self.assertTrue(torch.isfinite(grad_z).all())
        self.assertNotEqual(float(grad_z.abs().max()), 0.0)
        self.assertGreater(float((grad_z * discrete_delta).item()), 0.0)

    def test_fractional_bsu_and_count_backpropagate_gradients(self):
        net = self._line_net()
        state = self._state([net], z=1.25, bsu=1.5)
        per_size_input_cap, per_size_delay, per_size_output_slew = self._tables(
            len(state.segment_rows)
        )

        result = segment_count_relaxed_timing(
            [net],
            segment_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            driver_slew_by_net={41: 0.1},
        )
        loss = result["sink_arrival"].sum() + result["sink_slew"].sum()
        loss.backward()

        self.assertIsNotNone(state.z_param.grad)
        self.assertIsNotNone(state.bsu_index_param.grad)
        self.assertTrue(torch.isfinite(state.z_param.grad).all())
        self.assertTrue(torch.isfinite(state.bsu_index_param.grad).all())
        self.assertNotEqual(float(state.z_param.grad.abs().max()), 0.0)
        self.assertNotEqual(float(state.bsu_index_param.grad.abs().max()), 0.0)

    def test_integer_bsu_has_forward_size_gradient_until_upper_bound(self):
        net = self._line_net()
        state = self._state([net], z=1.25, bsu=1.0)
        per_size_input_cap, per_size_delay, per_size_output_slew = self._tables(
            len(state.segment_rows)
        )

        result = segment_count_relaxed_timing(
            [net],
            segment_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            driver_slew_by_net={41: 0.1},
        )
        loss = result["sink_arrival"].sum() + result["sink_slew"].sum()
        loss.backward()

        self.assertIsNotNone(state.bsu_index_param.grad)
        self.assertTrue(torch.isfinite(state.bsu_index_param.grad).all())
        self.assertNotEqual(float(state.bsu_index_param.grad.abs().max()), 0.0)

    def test_online_implementation_does_not_call_reference_or_rebuild_candidates(self):
        source = inspect.getsource(segment_count_relaxed_timing)

        self.assertNotIn("segment_repeater_transfer", source)
        self.assertNotIn("build_net_subgraph_timing_inputs", source)
        self.assertNotIn("int(bsu", source)
        self.assertNotIn("float(bsu", source)


if __name__ == "__main__":
    unittest.main()
