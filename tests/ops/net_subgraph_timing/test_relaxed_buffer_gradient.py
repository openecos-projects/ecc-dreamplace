import unittest

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
    build_discrete_candidate_state,
)
from dreamplace.ops.net_subgraph_timing import (
    net_subgraph_forward,
    net_subgraph_forward_relaxed,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    lookup_buffer_device_per_size,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


class RelaxedBufferGradientTest(unittest.TestCase):
    def _line_topology(self):
        return {
            "net_flat_topo_sort": torch.tensor([0, 1, 2], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 3], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0, 1], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1, 2], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 2.0, 3.0]),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0]),
            "edge_capacitance": torch.tensor([0.0, 0.0, 0.0]),
            "driver_arrival": torch.tensor([0.0]),
            "driver_slew": torch.tensor([0.1]),
            "sink_node_id": torch.tensor([2], dtype=torch.int32),
        }

    def _branch_topology(self):
        return {
            "net_flat_topo_sort": torch.tensor([0, 1, 3, 2], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0, 0, 1], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1, 2, 3], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 2.0, 4.0, 3.0]),
            "node_capacitance": torch.tensor([0.0, 1.0, 4.0, 2.0]),
            "edge_capacitance": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "driver_arrival": torch.tensor([0.0]),
            "driver_slew": torch.tensor([0.1]),
            "sink_node_id": torch.tensor([2, 3], dtype=torch.int32),
        }

    def test_toy_tns_loss_backpropagates_to_bu_and_bsu_index(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.0,
            initial_bsu_index=0.5,
        )
        per_size_input_cap = torch.tensor([[0.20, 0.40, 0.80]])
        per_size_delay = torch.tensor([[9.0, 5.0, 2.0]])
        per_size_output_slew = torch.tensor([[0.9, 0.5, 0.2]])
        result = net_subgraph_forward_relaxed(
            net_flat_topo_sort=torch.tensor([0, 1, 2], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 3], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0, 1], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            flat_pin_to=torch.tensor([1, 2], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 2.0, 3.0]),
            node_capacitance=torch.tensor([0.0, 1.0, 4.0]),
            edge_capacitance=torch.tensor([0.0, 0.0, 0.0]),
            driver_arrival=torch.tensor([0.0]),
            driver_slew=torch.tensor([0.1]),
            buffer_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
            sink_node_id=torch.tensor([2], dtype=torch.int32),
        )
        loss = torch.relu(result["sink_arrival"][0] - torch.tensor(12.0))
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(state.bu_logits.grad)
        self.assertIsNotNone(state.bsu_index_param.grad)
        self.assertTrue(torch.isfinite(state.bu_logits.grad).all())
        self.assertTrue(torch.isfinite(state.bsu_index_param.grad).all())
        self.assertNotEqual(float(state.bu_logits.grad.abs().sum()), 0.0)
        self.assertNotEqual(float(state.bsu_index_param.grad.abs().sum()), 0.0)

    def test_relaxed_near_zero_bu_matches_no_buffer_reference(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=-30.0,
            initial_bsu_index=1.5,
        )
        per_size_input_cap = torch.tensor([[0.20, 0.40, 0.80]])
        per_size_delay = torch.tensor([[9.0, 5.0, 2.0]])
        per_size_output_slew = torch.tensor([[0.9, 0.5, 0.2]])

        relaxed = net_subgraph_forward_relaxed(
            **self._line_topology(),
            buffer_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        no_buffer = net_subgraph_forward(
            **self._line_topology(),
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([0.0]),
            buffer_input_cap=torch.tensor([0.0]),
            buffer_delay=torch.tensor([0.0]),
            buffer_output_slew=torch.tensor([0.0]),
        )

        self.assertTrue(torch.allclose(relaxed["arrival_out"], no_buffer["arrival_out"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["slew_out"], no_buffer["slew_out"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["lin"], no_buffer["lin"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["sink_load"], no_buffer["sink_load"], atol=1e-5))

    def test_relaxed_integer_bsu_matches_fixed_inserted_buffer_reference(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=30.0,
            initial_bsu_index=2.0,
        )
        per_size_input_cap = torch.tensor([[0.20, 0.40, 0.80]])
        per_size_delay = torch.tensor([[9.0, 5.0, 2.0]])
        per_size_output_slew = torch.tensor([[0.9, 0.5, 0.2]])

        relaxed = net_subgraph_forward_relaxed(
            **self._line_topology(),
            buffer_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        fixed = net_subgraph_forward(
            **self._line_topology(),
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([1.0]),
            buffer_input_cap=torch.tensor([0.80]),
            buffer_delay=torch.tensor([2.0]),
            buffer_output_slew=torch.tensor([0.2]),
        )

        self.assertTrue(torch.allclose(relaxed["arrival_out"], fixed["arrival_out"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["slew_out"], fixed["slew_out"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["lin"], fixed["lin"], atol=1e-5))
        self.assertTrue(torch.allclose(relaxed["sink_arrival"], fixed["sink_arrival"], atol=1e-5))

    def test_critical_candidate_has_stronger_gradient_than_irrelevant_branch_candidate(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
                {"candidate_id": 1, "tree_node_id": 2, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.0,
            initial_bsu_index=0.5,
        )
        per_size_input_cap = torch.tensor(
            [
                [0.20, 0.40, 0.80],
                [0.20, 0.40, 0.80],
            ]
        )
        per_size_delay = torch.tensor(
            [
                [9.0, 5.0, 2.0],
                [9.0, 5.0, 2.0],
            ]
        )
        per_size_output_slew = torch.tensor(
            [
                [0.9, 0.5, 0.2],
                [0.9, 0.5, 0.2],
            ]
        )

        result = net_subgraph_forward_relaxed(
            **self._branch_topology(),
            buffer_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        critical_sink_arrival = result["sink_arrival"][1]
        loss = torch.relu(critical_sink_arrival - torch.tensor(8.0))
        loss.backward()

        critical_bu_grad = float(state.bu_logits.grad[0].abs())
        irrelevant_bu_grad = float(state.bu_logits.grad[1].abs())
        critical_bsu_grad = float(state.bsu_index_param.grad[0].abs())
        irrelevant_bsu_grad = float(state.bsu_index_param.grad[1].abs())

        self.assertGreater(float(loss), 0.0)
        self.assertGreater(critical_bu_grad, 0.0)
        self.assertGreater(critical_bsu_grad, 0.0)
        self.assertGreater(critical_bu_grad, irrelevant_bu_grad)
        self.assertGreater(critical_bsu_grad, irrelevant_bsu_grad)

    def test_same_net_multiple_candidate_gradients_from_one_backward(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
                {"candidate_id": 1, "tree_node_id": 2, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.0,
            initial_bsu_index=0.5,
        )
        per_size_input_cap = torch.tensor(
            [
                [0.20, 0.40, 0.80],
                [0.20, 0.40, 0.80],
            ]
        )
        per_size_delay = torch.tensor(
            [
                [9.0, 5.0, 2.0],
                [9.0, 5.0, 2.0],
            ]
        )
        per_size_output_slew = torch.tensor(
            [
                [0.9, 0.5, 0.2],
                [0.9, 0.5, 0.2],
            ]
        )

        result = net_subgraph_forward_relaxed(
            **self._line_topology(),
            buffer_state=state,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        loss = torch.relu(result["sink_arrival"][0] - torch.tensor(12.0))
        loss.backward()

        self.assertEqual(tuple(state.bu_logits.grad.shape), (2,))
        self.assertEqual(tuple(state.bsu_index_param.grad.shape), (2,))
        self.assertTrue(torch.isfinite(state.bu_logits.grad).all())
        self.assertTrue(torch.isfinite(state.bsu_index_param.grad).all())
        self.assertGreater(float(state.bu_logits.grad.abs().min()), 0.0)
        self.assertGreater(float(state.bsu_index_param.grad.abs().min()), 0.0)

    def test_current_approx_coordinate_callback_updates_buffer_tables_from_candidate_slew_and_load(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 0, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.0,
            initial_bsu_index=1.0,
        )
        static_input_cap = torch.tensor([[0.20, 0.40, 0.80]])
        static_delay = torch.tensor([[1.0, 1.0, 1.0]])
        static_output_slew = torch.tensor([[0.9, 0.5, 0.2]])
        topology = self._line_topology()
        topology["edge_capacitance"] = torch.tensor([0.0, 2.0, 4.0])

        def coordinate_callback(**kwargs):
            candidate_input_slew = kwargs["candidate_input_slew"]
            candidate_output_cap = kwargs["candidate_output_cap"]
            size_offsets = torch.tensor(
                [[0.0, 1.0, 2.0]],
                dtype=candidate_input_slew.dtype,
                device=candidate_input_slew.device,
            )
            coordinate_delay = (
                candidate_input_slew.view(-1, 1)
                + candidate_output_cap.view(-1, 1)
                + size_offsets
            )
            return {
                "per_size_input_cap": kwargs["per_size_input_cap"],
                "per_size_delay": coordinate_delay,
                "per_size_output_slew": kwargs["per_size_output_slew"],
                "coordinate_source": "current_approx_probe_slew_lout",
            }

        result = net_subgraph_forward_relaxed(
            **topology,
            buffer_state=state,
            per_size_input_cap=static_input_cap,
            per_size_delay=static_delay,
            per_size_output_slew=static_output_slew,
            buffer_coordinate_callback=coordinate_callback,
            coordinate_source="current_approx",
        )
        loss = result["sink_arrival"].sum()
        loss.backward()

        model = result["relaxed_buffer_model"]
        self.assertEqual(
            model["bsu_index_status"]["coordinate_source"],
            "current_approx_probe_slew_lout",
        )
        self.assertTrue(torch.allclose(model["candidate_input_slew"], result["coordinate_probe"]["slew_in"][state.candidate_node_id]))
        self.assertTrue(
            torch.allclose(
                model["candidate_output_cap"],
                result["coordinate_probe"]["lout"][state.candidate_node_id]
                - torch.tensor([1.0]),
            )
        )
        self.assertGreater(float(model["buffer_delay"][0]), 1.0)
        self.assertIsNotNone(state.bu_logits.grad)
        self.assertNotEqual(float(state.bu_logits.grad.abs().sum()), 0.0)

    def test_liberty_coordinate_callback_bu_gradient_matches_finite_difference(self):
        buffer_device = default_fake_buffer_device(dtype=torch.float32)
        per_size_input_cap = torch.tensor([[0.20, 0.40]])
        per_size_delay = torch.tensor([[1.0, 1.0]])
        per_size_output_slew = torch.tensor([[0.9, 0.5]])
        topology = self._line_topology()
        topology["edge_capacitance"] = torch.tensor([0.0, 2.0, 4.0])

        def evaluate(logit, *, backward):
            state = build_buffer_optimization_state(
                [
                    {
                        "candidate_id": 0,
                        "tree_node_id": 1,
                        "net_id": 0,
                        "buffer_main_type_index": 1,
                    },
                ],
                buffer_main_type_index=1,
                legal_buffer_count=2,
                initial_bu_logit=logit,
                initial_bsu_index=0.5,
            )

            def coordinate_callback(**kwargs):
                return lookup_buffer_device_per_size(
                    buffer_device,
                    input_slew=kwargs["candidate_input_slew"],
                    output_load=kwargs["candidate_output_cap"],
                )

            result = net_subgraph_forward_relaxed(
                **topology,
                buffer_state=state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                buffer_coordinate_callback=coordinate_callback,
                forward_backend="native_explicit_autograd",
            )
            objective = (
                result["sink_arrival"].sum()
                + 0.1 * result["sink_slew"].sum()
                + 0.01 * result["lin"][0]
            )
            if backward:
                objective.backward()
                return float(objective.detach()), float(state.bu_logits.grad[0])
            return float(objective.detach())

        logit = 0.2
        _value, analytic_grad = evaluate(logit, backward=True)
        epsilon = 1e-3
        finite_difference = (
            evaluate(logit + epsilon, backward=False)
            - evaluate(logit - epsilon, backward=False)
        ) / (2.0 * epsilon)

        self.assertAlmostEqual(analytic_grad, finite_difference, delta=2e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fixed_bsu_fused_cuda_matches_two_pass_liberty_path(self):
        device = torch.device("cuda")
        buffer_device = default_fake_buffer_device(dtype=torch.float32)
        device_lut = buffer_device.tensors(dtype=torch.float32, device=device)
        candidates = [
            {
                "candidate_id": 0,
                "candidate_node_id": 1,
                "net_id": 0,
                "buffer_main_type_index": 1,
                "x_dbu": 100,
                "y_dbu": 0,
            }
        ]
        topology = {
            key: value.to(device=device)
            for key, value in self._line_topology().items()
        }
        topology["sink_net_index"] = torch.tensor(
            [0], dtype=torch.long, device=device
        )
        per_size_input_cap = device_lut["input_cap_by_size"].view(1, -1)
        per_size_delay = torch.tensor([[1.0, 1.0]], device=device)
        per_size_output_slew = torch.tensor([[0.9, 0.5]], device=device)

        def evaluate(*, fused):
            state = build_discrete_candidate_state(
                candidates,
                buffer_main_type_index=1,
                legal_buffer_count=2,
                fixed_bsu_index=1,
                device=device,
            )

            def coordinate_callback(**kwargs):
                return lookup_buffer_device_per_size(
                    buffer_device,
                    input_slew=kwargs["candidate_input_slew"],
                    output_load=kwargs["candidate_output_cap"],
                )

            result = net_subgraph_forward_relaxed(
                **topology,
                buffer_state=state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                buffer_coordinate_callback=coordinate_callback,
                fixed_bsu_device_lut=device_lut if fused else None,
                fixed_bsu_index=1 if fused else None,
                forward_backend="cuda_explicit_autograd",
            )
            loss = (
                result["sink_arrival"].sum()
                + 0.1 * result["sink_slew"].sum()
                + 0.01 * result["lin"].sum()
            )
            grad, = torch.autograd.grad(loss, state.activation_param)
            return result, grad

        reference, reference_grad = evaluate(fused=False)
        fused, fused_grad = evaluate(fused=True)
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
            torch.testing.assert_close(fused[key], reference[key])
        for key in (
            "arrival_in",
            "arrival_out",
            "slew_in",
            "slew_out",
            "sink_arrival",
            "sink_slew",
            "sink_net_delay",
            "sink_net_impulse",
        ):
            torch.testing.assert_close(
                fused["coordinate_probe"][key],
                reference["coordinate_probe"][key],
            )
        for key in (
            "buffer_input_cap",
            "buffer_delay",
            "buffer_output_slew",
            "candidate_input_slew",
            "candidate_output_cap",
        ):
            torch.testing.assert_close(
                fused["relaxed_buffer_model"][key],
                reference["relaxed_buffer_model"][key],
            )
        torch.testing.assert_close(fused_grad, reference_grad)

    def test_native_autograd_backends_match_python_relaxed_gradients(self):
        per_size_input_cap = torch.tensor([[0.20, 0.40, 0.80]])
        per_size_delay = torch.tensor([[9.0, 5.0, 2.0]])
        per_size_output_slew = torch.tensor([[0.9, 0.5, 0.2]])
        topology = self._line_topology()
        topology["edge_capacitance"] = torch.tensor([0.0, 2.0, 4.0])

        def run(backend):
            state = build_buffer_optimization_state(
                [
                    {
                        "candidate_id": 0,
                        "tree_node_id": 1,
                        "net_id": 0,
                        "buffer_main_type_index": 1,
                    },
                ],
                buffer_main_type_index=1,
                legal_buffer_count=3,
                initial_bu_logit=0.0,
                initial_bsu_index=0.5,
            )
            result = net_subgraph_forward_relaxed(
                **topology,
                buffer_state=state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                forward_backend=backend,
            )
            loss = (
                torch.relu(result["sink_arrival"][0] - torch.tensor(12.0))
                + 0.1 * result["sink_slew"][0]
                + 0.01 * result["sink_load"][0]
            )
            loss.backward()
            return result, state.bu_logits.grad.detach(), state.bsu_index_param.grad.detach()

        python_result, python_bu_grad, python_bsu_grad = run("python")
        for backend in ("native_recompute_autograd", "native_explicit_autograd"):
            native_result, native_bu_grad, native_bsu_grad = run(backend)

            self.assertTrue(
                torch.allclose(
                    native_result["sink_arrival"],
                    python_result["sink_arrival"],
                    atol=1e-6,
                ),
                backend,
            )
            self.assertTrue(torch.allclose(native_bu_grad, python_bu_grad, atol=1e-6), backend)
            self.assertTrue(torch.allclose(native_bsu_grad, python_bsu_grad, atol=1e-6), backend)


if __name__ == "__main__":
    unittest.main()
