"""Gate B: independent scalar moments and analytical/FD pin-cap gradients."""

from dataclasses import replace

import pytest
import torch
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.route_snapshot import prepare_snapshot
from rc_oracle import scalar_oracle


def path_resistance(tree, first, second):
    def ancestors(vertex):
        result = set()
        while vertex >= 0:
            result.add(vertex)
            vertex = int(tree.parent[vertex])
        return result

    common = ancestors(first) & ancestors(second)
    return sum(float(tree.incoming_resistance[v]) for v in common)


def test_rc_moments_and_cap_fd_match_scalar_oracle(snapshot_inputs):
    op = GRParasiticsOp.prepare(*snapshot_inputs, dtype=torch.float64)
    cap = torch.tensor([0.01, 0.02, 0.0], dtype=torch.float64, requires_grad=True)
    modes = ("generic", "rise", "fall")
    values = op(cap, cap * 1.2, cap * 0.8)
    for mode, scale in zip(modes, (1.0, 1.2, 0.8), strict=True):
        expected = scalar_oracle(op.snapshot, cap * scale)
        for group, golden in zip(values, expected, strict=True):
            torch.testing.assert_close(group[mode], golden, rtol=1e-9, atol=1e-12)
    for group_index in range(1, 6):
        for mode in modes:
            result = values[group_index][mode][0]
            grad = torch.autograd.grad(result, cap, retain_graph=True)[0]
            for pin in range(3):
                step = 1e-4 * max(abs(float(cap[pin].detach())), 1e-3)
                plus, minus = cap.detach().clone(), cap.detach().clone()
                plus[pin] += step
                minus[pin] -= step
                if cap[pin] == 0:
                    minus = cap.detach()
                    denominator = step
                else:
                    denominator = 2 * step
                high = op(plus, plus * 1.2, plus * 0.8)[group_index][mode][0]
                low = op(minus, minus * 1.2, minus * 0.8)[group_index][mode][0]
                fd = (high - low) / denominator
                assert torch.isfinite(grad[pin])
                assert float(grad[pin]) == pytest.approx(float(fd), rel=1e-3, abs=1e-6)
    # Root cap affects root load but cannot affect downstream delay (root R=0).
    delay_grad = torch.autograd.grad(values[2]["generic"][0], cap)[0]
    vertex = op.snapshot.pin_to_vertex
    expected = [path_resistance(op.snapshot, int(vertex[0]), int(v)) for v in vertex]
    torch.testing.assert_close(delay_grad, torch.tensor(expected, dtype=torch.float64))
    smoke = GRParasiticsOp.prepare(*snapshot_inputs)
    result = smoke(cap.detach().float(), cap.detach().float(), cap.detach().float())
    assert all(torch.isfinite(group["generic"]).all() for group in result)


def test_shared_vertex_and_cross_sink_gradient(snapshot_inputs):
    tree = prepare_snapshot(*snapshot_inputs, dtype=torch.float64)
    # Numerical attachment contract: two physical loads share one electrical
    # vertex. Geometry/routing qualification is owned by Gate A above.
    mapping = tree.pin_to_vertex.clone()
    mapping[1] = mapping[0]
    shared = replace(tree, pin_to_vertex=mapping, valid_vertex_ids=mapping[tree.valid_pin_ids])
    op = GRParasiticsOp(shared)
    cap = torch.tensor([0.01, 0.02, 0.0], dtype=torch.float64, requires_grad=True)
    values = op(cap, cap, cap)
    golden = scalar_oracle(shared, cap)
    for group, expected in zip(values, golden, strict=True):
        torch.testing.assert_close(group["generic"], expected, rtol=1e-9, atol=1e-12)
    grad = torch.autograd.grad(values[2]["generic"][0], cap)[0]
    path_r = path_resistance(shared, int(mapping[0]), int(mapping[0]))
    torch.testing.assert_close(grad, torch.tensor([path_r, path_r, 0.0], dtype=torch.float64))
    assert path_r > 0
