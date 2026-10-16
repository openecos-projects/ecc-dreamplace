import unittest

import torch

from dreamplace.ops.net_subgraph_timing import (
    net_subgraph_forward,
    net_subgraph_forward_cuda_explicit_autograd,
    net_subgraph_forward_native_explicit_autograd,
    net_subgraph_forward_native,
    net_subgraph_forward_native_recompute_autograd,
)


class NetSubgraphTimingCppTest(unittest.TestCase):
    def _assert_outputs_match(self, inputs):
        reference = net_subgraph_forward(**inputs)
        native = net_subgraph_forward_native(**inputs)

        for key in (
            "lout",
            "lin",
            "effective_node_cap",
            "arrival_in",
            "arrival_out",
            "slew_in",
            "slew_out",
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "sink_cap",
        ):
            self.assertTrue(torch.allclose(native[key], reference[key], atol=1e-6), key)

        self.assertTrue(torch.equal(native["sink_net_index"], reference["sink_net_index"]))

    def test_branch_topology_matches_python_reference(self):
        self._assert_outputs_match(
            {
                "net_flat_topo_sort": torch.tensor([0, 1, 3, 2], dtype=torch.int32),
                "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.int32),
                "pin_fa": torch.tensor([-1, 0, 0, 1], dtype=torch.int32),
                "flat_pin_to_start": torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32),
                "flat_pin_to": torch.tensor([1, 2, 3], dtype=torch.int32),
                "edge_resistance": torch.tensor([0.0, 2.0, 4.0, 3.0]),
                "edge_capacitance": torch.tensor([0.0, 0.5, 1.0, 0.25]),
                "node_capacitance": torch.tensor([0.0, 1.0, 4.0, 2.0]),
                "driver_arrival": torch.tensor([0.0]),
                "driver_slew": torch.tensor([0.1]),
                "candidate_node_id": torch.tensor([1], dtype=torch.int32),
                "candidate_bu": torch.tensor([1.0]),
                "buffer_input_cap": torch.tensor([0.5]),
                "buffer_delay": torch.tensor([7.0]),
                "buffer_output_slew": torch.tensor([0.2]),
                "sink_node_id": torch.tensor([2, 3], dtype=torch.int32),
                "sink_net_index": torch.tensor([0, 0], dtype=torch.int32),
            }
        )

    def test_multi_net_batch_matches_python_reference(self):
        self._assert_outputs_match(
            {
                "net_flat_topo_sort": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                "net_flat_topo_sort_start": torch.tensor([0, 2, 4], dtype=torch.int32),
                "pin_fa": torch.tensor([-1, 0, -1, 2], dtype=torch.int32),
                "flat_pin_to_start": torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32),
                "flat_pin_to": torch.tensor([1, 3], dtype=torch.int32),
                "edge_resistance": torch.tensor([0.0, 4.0, 0.0, 5.0]),
                "node_capacitance": torch.tensor([0.0, 2.0, 0.0, 3.0]),
                "driver_arrival": torch.tensor([10.0, 20.0]),
                "driver_slew": torch.tensor([0.1, 0.2]),
                "candidate_node_id": torch.tensor([3], dtype=torch.int32),
                "candidate_bu": torch.tensor([1.0]),
                "buffer_input_cap": torch.tensor([0.5]),
                "buffer_delay": torch.tensor([6.0]),
                "buffer_output_slew": torch.tensor([0.25]),
                "sink_node_id": torch.tensor([1, 3], dtype=torch.int32),
                "sink_net_index": torch.tensor([0, 1], dtype=torch.int32),
            }
        )

    def test_native_recompute_autograd_matches_python_reference_gradients(self):
        static = {
            "net_flat_topo_sort": torch.tensor([0, 1, 2], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 3], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0, 1], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1, 2], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 2.0, 3.0]),
            "edge_capacitance": torch.tensor([0.0, 0.0, 0.0]),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0]),
            "candidate_node_id": torch.tensor([1], dtype=torch.int32),
            "sink_node_id": torch.tensor([2], dtype=torch.int32),
        }

        def run(forward_fn):
            driver_arrival = torch.tensor([0.0], requires_grad=True)
            driver_slew = torch.tensor([0.1], requires_grad=True)
            candidate_bu = torch.tensor([0.5], requires_grad=True)
            buffer_input_cap = torch.tensor([0.4], requires_grad=True)
            buffer_delay = torch.tensor([5.0], requires_grad=True)
            buffer_output_slew = torch.tensor([0.5], requires_grad=True)
            result = forward_fn(
                **static,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                candidate_bu=candidate_bu,
                buffer_input_cap=buffer_input_cap,
                buffer_delay=buffer_delay,
                buffer_output_slew=buffer_output_slew,
            )
            loss = (
                result["sink_arrival"].sum()
                + 0.1 * result["sink_slew"].sum()
                + 0.01 * result["sink_load"].sum()
            )
            loss.backward()
            return result, (
                driver_arrival.grad,
                driver_slew.grad,
                candidate_bu.grad,
                buffer_input_cap.grad,
                buffer_delay.grad,
                buffer_output_slew.grad,
            )

        reference, reference_grads = run(net_subgraph_forward)
        native, native_grads = run(net_subgraph_forward_native_recompute_autograd)

        for key in ("sink_arrival", "sink_slew", "sink_load", "sink_net_delay"):
            self.assertTrue(torch.allclose(native[key], reference[key], atol=1e-6), key)
        for index, (native_grad, reference_grad) in enumerate(zip(native_grads, reference_grads)):
            self.assertIsNotNone(native_grad, index)
            self.assertTrue(torch.allclose(native_grad, reference_grad, atol=1e-6), index)

    def test_native_explicit_autograd_matches_python_reference_gradients(self):
        static = {
            "net_flat_topo_sort": torch.tensor([0, 1, 3, 2], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0, 0, 1], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1, 2, 3], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 2.0, 4.0, 3.0]),
            "edge_capacitance": torch.tensor([0.0, 0.5, 1.0, 0.25]),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0, 2.0]),
            "candidate_node_id": torch.tensor([1, 2], dtype=torch.int32),
            "sink_node_id": torch.tensor([2, 3], dtype=torch.int32),
            "sink_net_index": torch.tensor([0, 0], dtype=torch.int32),
        }

        def run(forward_fn):
            driver_arrival = torch.tensor([0.0], requires_grad=True)
            driver_slew = torch.tensor([0.1], requires_grad=True)
            candidate_bu = torch.tensor([0.5, 0.25], requires_grad=True)
            buffer_input_cap = torch.tensor([0.4, 0.6], requires_grad=True)
            buffer_delay = torch.tensor([5.0, 7.0], requires_grad=True)
            buffer_output_slew = torch.tensor([0.5, 0.7], requires_grad=True)
            result = forward_fn(
                **static,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                candidate_bu=candidate_bu,
                buffer_input_cap=buffer_input_cap,
                buffer_delay=buffer_delay,
                buffer_output_slew=buffer_output_slew,
            )
            loss = (
                result["sink_arrival"].sum()
                + 0.1 * result["sink_slew"].sum()
                + 0.01 * result["sink_load"].sum()
                + 0.001 * result["sink_net_delay"].sum()
                + 0.0001 * result["sink_net_impulse"].sum()
            )
            loss.backward()
            return result, (
                driver_arrival.grad,
                driver_slew.grad,
                candidate_bu.grad,
                buffer_input_cap.grad,
                buffer_delay.grad,
                buffer_output_slew.grad,
            )

        reference, reference_grads = run(net_subgraph_forward)
        explicit, explicit_grads = run(net_subgraph_forward_native_explicit_autograd)

        for key in (
            "lout",
            "lin",
            "arrival_out",
            "slew_out",
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "sink_net_delay",
            "sink_net_impulse",
        ):
            self.assertTrue(torch.allclose(explicit[key], reference[key], atol=1e-6), key)
        for index, (explicit_grad, reference_grad) in enumerate(zip(explicit_grads, reference_grads)):
            self.assertIsNotNone(explicit_grad, index)
            self.assertTrue(torch.allclose(explicit_grad, reference_grad, atol=1e-5), index)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_explicit_autograd_matches_python_reference_outputs_and_gradients(self):
        static = {
            "net_flat_topo_sort": torch.tensor([0, 1, 3, 2], dtype=torch.long),
            "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.long),
            "pin_fa": torch.tensor([-1, 0, 0, 1], dtype=torch.long),
            "flat_pin_to_start": torch.tensor([0, 2, 3, 3, 3], dtype=torch.long),
            "flat_pin_to": torch.tensor([1, 2, 3], dtype=torch.long),
            "edge_resistance": torch.tensor([0.0, 2.0, 4.0, 3.0], dtype=torch.float64),
            "edge_capacitance": torch.tensor([0.0, 0.5, 1.0, 0.25], dtype=torch.float64),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0, 2.0], dtype=torch.float64),
            "candidate_node_id": torch.tensor([1, 2], dtype=torch.long),
            "sink_node_id": torch.tensor([2, 3], dtype=torch.long),
            "sink_net_index": torch.tensor([0, 0], dtype=torch.long),
        }

        def run(forward_fn, device):
            inputs = {key: value.to(device) for key, value in static.items()}
            variables = [
                torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True),
                torch.tensor([0.1], dtype=torch.float64, device=device, requires_grad=True),
                torch.tensor([0.5, 0.25], dtype=torch.float64, device=device, requires_grad=True),
                torch.tensor([0.4, 0.6], dtype=torch.float64, device=device, requires_grad=True),
                torch.tensor([5.0, 7.0], dtype=torch.float64, device=device, requires_grad=True),
                torch.tensor([0.5, 0.7], dtype=torch.float64, device=device, requires_grad=True),
            ]
            result = forward_fn(
                **inputs,
                driver_arrival=variables[0],
                driver_slew=variables[1],
                candidate_bu=variables[2],
                buffer_input_cap=variables[3],
                buffer_delay=variables[4],
                buffer_output_slew=variables[5],
            )
            loss = (
                0.001 * result["lout"].sum()
                + 0.002 * result["lin"].sum()
                + 0.003 * result["arrival_in"].sum()
                + 0.004 * result["arrival_out"].sum()
                + 0.005 * result["slew_in"].sum()
                + 0.006 * result["slew_out"].sum()
                + result["sink_arrival"].sum()
                + 0.1 * result["sink_slew"].sum()
                + 0.01 * result["sink_load"].sum()
                + 0.001 * result["sink_net_delay"].sum()
                + 0.0001 * result["sink_net_impulse"].sum()
            )
            loss.backward()
            return (
                {key: value.detach().cpu() for key, value in result.items()},
                [value.grad.detach().cpu() for value in variables],
            )

        reference, reference_grads = run(net_subgraph_forward, torch.device("cpu"))
        cuda, cuda_grads = run(
            net_subgraph_forward_cuda_explicit_autograd,
            torch.device("cuda"),
        )
        for key in (
            "lout",
            "lin",
            "effective_node_cap",
            "arrival_in",
            "arrival_out",
            "slew_in",
            "slew_out",
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "sink_cap",
            "sink_net_delay",
            "sink_net_impulse",
        ):
            self.assertTrue(torch.allclose(cuda[key], reference[key], atol=1e-10), key)
        for index, (cuda_grad, reference_grad) in enumerate(zip(cuda_grads, reference_grads)):
            self.assertTrue(
                torch.allclose(cuda_grad, reference_grad, atol=1e-9, rtol=1e-9),
                index,
            )


if __name__ == "__main__":
    unittest.main()
