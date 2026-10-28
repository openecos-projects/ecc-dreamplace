import pytest
import torch
from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import (
    net_subgraph_forward,
    net_subgraph_forward_native_explicit_autograd,
    net_subgraph_forward_native_recompute_autograd,
)


def _inputs(bu, wire_cap):
    return {
        "net_flat_topo_sort": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.int32),
        "pin_fa": torch.tensor([-1, 0, 1, 1], dtype=torch.int32),
        "flat_pin_to_start": torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32),
        "flat_pin_to": torch.tensor([1, 2, 3], dtype=torch.int32),
        "edge_resistance": torch.tensor([0.0, 2.0, 3.0, 1.0], dtype=torch.float64),
        "node_capacitance": torch.tensor([0.0, 0.4, 1.0, 0.6], dtype=torch.float64),
        "edge_capacitance": torch.tensor(wire_cap, dtype=torch.float64),
        "driver_arrival": torch.tensor([0.2], dtype=torch.float64),
        "driver_slew": torch.tensor([0.1], dtype=torch.float64),
        "candidate_node_id": torch.tensor([1], dtype=torch.int32),
        "candidate_bu": torch.tensor([bu], dtype=torch.float64),
        "buffer_input_cap": torch.tensor([0.2], dtype=torch.float64),
        "buffer_delay": torch.tensor([0.1], dtype=torch.float64),
        "buffer_output_slew": torch.tensor([0.1], dtype=torch.float64),
        "sink_node_id": torch.tensor([2, 3], dtype=torch.int32),
        "sink_net_index": torch.tensor([0, 0], dtype=torch.int32),
    }


def _reference(**inputs):
    if not inputs["edge_capacitance"].numel():
        inputs["edge_capacitance"] = None
    return net_subgraph_forward(**inputs)


@pytest.mark.parametrize("bu", [0.0, 0.3, 1.0])
@pytest.mark.parametrize(
    "output",
    [
        "sink_arrival",
        "sink_slew",
        "sink_load",
        "sink_net_delay",
        "sink_net_impulse",
        "effective_node_cap",
        "sink_cap",
    ],
)
@pytest.mark.parametrize("wire_cap", [[], [0.0, 0.5, 0.7, 0.2]])
@pytest.mark.parametrize(
    "native",
    [net_subgraph_forward_native_explicit_autograd, net_subgraph_forward_native_recompute_autograd],
)
def test_native_candidate_rc_gradients_match_reference_and_finite_difference(
    bu, output, wire_cap, native
):
    gradients = []
    names = ("edge_resistance", "node_capacitance", "edge_capacitance")
    for forward in (_reference, native):
        inputs = _inputs(bu, wire_cap)
        variables = tuple(inputs[name].requires_grad_() for name in names)
        loss = forward(**inputs)[output].sum()
        grads = torch.autograd.grad(loss, variables, allow_unused=True)
        gradients.append(
            tuple(
                torch.zeros_like(var) if grad is None else grad
                for var, grad in zip(variables, grads, strict=True)
            )
        )
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    inputs = _inputs(bu, wire_cap)
    for name, actual in zip(names, gradients[1], strict=True):
        finite = torch.zeros_like(actual)
        for index in range(actual.numel()):
            if name == "edge_resistance" and index == 0:
                continue
            original = inputs[name][index].item()
            inputs[name][index] = original + 1e-6
            plus = _reference(**inputs)[output].sum()
            inputs[name][index] = original - 1e-6
            minus = _reference(**inputs)[output].sum()
            inputs[name][index] = original
            finite[index] = (plus - minus) / 2e-6
        torch.testing.assert_close(actual, finite, rtol=1e-6, atol=1e-7)
