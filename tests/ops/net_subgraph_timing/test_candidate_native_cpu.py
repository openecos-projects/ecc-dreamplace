"""The CPU reverse pass preserves its driver boundary and optional wire C."""

import pytest
import torch
from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import (
    net_subgraph_forward,
    net_subgraph_forward_native_explicit_autograd,
)
from test_candidate_geometry_gradients import _inputs


def test_inferred_sink_nets_preserve_repeated_sink_values_and_gradients():
    results = []
    for inferred in (False, True):
        inputs = _inputs(0.4, [0, 0.5, 0.7, 0.2])
        inputs["sink_node_id"] = torch.tensor([2, 2, 3], dtype=torch.int32)
        inputs["sink_net_index"] = None if inferred else torch.zeros(3, dtype=torch.int32)
        names = (
            "driver_arrival",
            "driver_slew",
            "candidate_bu",
            "buffer_input_cap",
            "buffer_delay",
            "buffer_output_slew",
            "edge_resistance",
            "node_capacitance",
        )
        variables = tuple(inputs[name].requires_grad_() for name in names)
        outputs = net_subgraph_forward_native_explicit_autograd(**inputs)
        values = tuple(
            outputs[name]
            for name in ("sink_arrival", "sink_slew", "sink_net_delay", "sink_net_impulse")
        )
        gradients = torch.autograd.grad(sum(value.sum() for value in values), variables)
        results.append((values, gradients))
    torch.testing.assert_close(results[1], results[0], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("activation", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("output", ["sink_net_delay", "sink_net_impulse"])
def test_root_candidate_preserves_input_driver_relative_output_derivatives(activation, output):
    results = []
    for function in (net_subgraph_forward, net_subgraph_forward_native_explicit_autograd):
        inputs = _inputs(activation, [0, 0.5, 0.7, 0.2])
        inputs["candidate_node_id"] = torch.tensor([0], dtype=torch.int32)
        names = (
            "driver_arrival",
            "driver_slew",
            "candidate_bu",
            "buffer_input_cap",
            "buffer_delay",
            "buffer_output_slew",
            "edge_resistance",
            "node_capacitance",
        )
        variables = tuple(inputs[name].requires_grad_() for name in names)
        value = function(**inputs)[output]
        gradients = torch.autograd.grad(value.sum(), variables, allow_unused=True)
        results.append(
            (
                value,
                tuple(
                    torch.zeros_like(v) if g is None else g
                    for v, g in zip(variables, gradients, strict=True)
                ),
            )
        )
    torch.testing.assert_close(results[1], results[0], rtol=1e-9, atol=1e-9)
