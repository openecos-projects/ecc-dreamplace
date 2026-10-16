"""RC derivatives from the candidate timing reverse pass."""

import math

import torch


def candidate_node_coverage(topology, node_count):
    """Packed forests cover every node; padded inputs may omit trailing nodes."""
    if topology.numel() == node_count:
        return None
    covered = torch.zeros(node_count, dtype=torch.bool, device=topology.device)
    return covered.index_fill(0, topology.long(), True)


def mask_unvisited_candidate_adjoints(adjoints, covered):
    if covered is None:
        return
    # Native forward leaves omitted node outputs at constant zero. Seeds for
    # those outputs must not become RC or typed candidate input derivatives.
    for name in (
        "grad_lout",
        "grad_lin",
        "grad_arrival_in",
        "grad_arrival_out",
        "grad_slew_in",
        "grad_slew_out",
    ):
        adjoints[name] = torch.where(covered, adjoints[name], 0.0)


def candidate_rc_gradients_from_adjoint(
    *,
    parents,
    edge_resistance,
    lin,
    slew_in,
    adjoints,
    output_gradients,
    sink_node_id,
    required,
    node_coverage=None,
):
    """Complete RC derivatives without replaying the tree in Python autograd."""
    if not any(required):
        return (None, None, None)
    nonroot = parents >= 0
    resistance_gradient = None
    if required[0]:
        denominator = torch.where(slew_in != 0, slew_in, 1.0)
        slew_gradient = (
            adjoints["grad_slew_in"]
            * math.log(10.0) ** 2
            * edge_resistance
            * lin.square()
            / denominator
        )
        resistance_gradient = torch.where(
            nonroot,
            adjoints["grad_arrival_in"] * lin + slew_gradient,
            0.0,
        )
    node_gradient = None
    edge_gradient = None
    if required[1] or required[2]:
        node_gradient = adjoints["grad_lout"]
        direct = output_gradients.get("effective_node_cap")
        if direct is not None:
            node_gradient = node_gradient + direct
        direct_sink = output_gradients.get("sink_cap")
        if direct_sink is not None:
            node_gradient = node_gradient.index_add(0, sink_node_id.long(), direct_sink)
        if node_coverage is not None:
            node_gradient = torch.where(node_coverage, node_gradient, 0.0)
        if required[2]:
            edge_gradient = torch.where(
                nonroot,
                0.5 * (node_gradient + node_gradient[parents.clamp(min=0).long()]),
                0.0,
            )
    return (
        resistance_gradient,
        node_gradient if required[1] else None,
        edge_gradient,
    )
