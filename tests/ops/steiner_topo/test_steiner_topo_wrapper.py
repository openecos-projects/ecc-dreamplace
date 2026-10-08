import torch
import pytest

from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo
from dreamplace.ops.steiner_topo import steiner_topo as steiner_topo_module


def test_rebuild_tree_normalizes_pin_positions_to_cpu(monkeypatch):
    captured = {}

    def fake_build_tree(pos, flat_net2pin_map, flat_net2pin_start_map, ignore_net_degree,
                        powv_file, post_file, deterministic_flag):
        captured["is_cuda"] = bool(pos.is_cuda)
        captured["is_contiguous"] = bool(pos.is_contiguous())
        device = pos.device
        return (
            torch.zeros(2, device=device),
            torch.zeros(2, device=device),
            torch.zeros(2, device=device),
            torch.zeros(2, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            torch.tensor([-1, 0], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
        )

    monkeypatch.setattr(steiner_topo_module.steiner_topo_cpp, "build_tree", fake_build_tree)
    topo = SteinerTopo(
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
    )
    if torch.cuda.is_available():
        pin_pos = torch.zeros(4, device="cuda")
    else:
        pin_pos = torch.zeros(4)

    topo.rebuild_tree(pin_pos)

    assert captured["is_cuda"] is False
    assert captured["is_contiguous"] is True
    assert topo.topology_generation == 1
    assert topo.rebuild_count == 1


def test_frozen_topology_allows_coordinate_forward_but_rejects_rebuild(monkeypatch):
    def fake_build_tree(pos, *args, **kwargs):
        device = pos.device
        return (
            torch.tensor([0.0, 10.0], device=device),
            torch.zeros(2, device=device),
            torch.tensor([0.0, 1.0], device=device),
            torch.zeros(2, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            torch.tensor([-1, 0], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
        )

    def fake_forward(pos, pin_relate_x, pin_relate_y, num_vertices, deterministic_flag):
        return pos[:num_vertices], torch.zeros_like(pos[:num_vertices])

    def fake_backward(grad_x, grad_y, pos, pin_relate_x, pin_relate_y):
        return grad_x + grad_y

    monkeypatch.setattr(steiner_topo_module.steiner_topo_cpp, "build_tree", fake_build_tree)
    monkeypatch.setattr(steiner_topo_module.steiner_topo_cpp, "forward", fake_forward)
    monkeypatch.setattr(steiner_topo_module.steiner_topo_cpp, "backward", fake_backward)
    topo = SteinerTopo(
        flat_net2pin_map=torch.tensor([0, 1], dtype=torch.int32),
        flat_net2pin_start_map=torch.tensor([0, 2], dtype=torch.int32),
    )
    pos = torch.tensor([0.0, 10.0], requires_grad=True)

    topo.rebuild_tree(pos)
    frozen_generation = topo.freeze_topology()
    new_x, new_y = topo(pos)
    (new_x.sum() + new_y.sum()).backward()

    assert frozen_generation == 1
    assert topo.frozen_topology_generation == 1
    assert topo.forward_count == 1
    assert pos.grad is not None
    topo.topology_generation = 2
    with pytest.raises(RuntimeError, match="frozen topology generation changed"):
        topo(pos)
    topo.topology_generation = 1
    with pytest.raises(RuntimeError, match="topology is frozen"):
        topo.rebuild_tree(pos)

    assert topo.unfreeze_topology() == 1
    topo.rebuild_tree(pos)
    assert topo.topology_generation == 2
