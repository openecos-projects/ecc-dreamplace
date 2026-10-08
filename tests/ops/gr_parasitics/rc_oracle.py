"""Scalar physical equations independent of the production RC kernels."""

import torch


def scalar_oracle(tree, pin_cap):
    cap = tree.wire_cap.tolist()
    for pin, vertex in enumerate(tree.pin_to_vertex.tolist()):
        if vertex >= 0:
            cap[vertex] += float(pin_cap[pin].detach())
    parent, order = tree.parent.tolist(), tree.topo_order.tolist()
    resistance = tree.incoming_resistance.tolist()
    load, delay, ldelay, beta = cap.copy(), [0.0] * len(cap), [0.0] * len(cap), [0.0] * len(cap)
    for vertex in reversed(order):
        if parent[vertex] >= 0:
            load[parent[vertex]] += load[vertex]
    for vertex in order:
        p = parent[vertex]
        if p >= 0:
            delay[vertex] = delay[p] + resistance[vertex] * load[vertex]
    ldelay = [c * d for c, d in zip(cap, delay, strict=True)]
    for vertex in reversed(order):
        if parent[vertex] >= 0:
            ldelay[parent[vertex]] += ldelay[vertex]
    for vertex in order:
        p = parent[vertex]
        if p >= 0:
            beta[vertex] = beta[p] + resistance[vertex] * ldelay[vertex]
    impulse = [2 * b - d * d for b, d in zip(beta, delay, strict=True)]
    return [
        torch.tensor([values[v] for v in tree.pin_to_vertex], dtype=torch.float64)
        for values in (cap, load, delay, ldelay, beta, impulse)
    ]
