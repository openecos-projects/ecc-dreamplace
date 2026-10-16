"""Topology lifetime and view boundaries for deterministic CUDA reductions."""

import pytest
import torch

from dreamplace.ops.timing_propagation.scatter_groups import scatter_groups


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recreated_views_share_groups_and_inplace_edits_invalidate_them(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    arcs = torch.tensor([[2, 1], [0, 1], [2, 0]], device=device)
    first = scatter_groups(arcs[:, 0], 4)
    assert scatter_groups(arcs[:, 0], 4) is first
    outputs = scatter_groups(arcs[:, 1], 4)
    assert outputs is not first
    torch.testing.assert_close(
        first, (arcs.new_tensor([1, 0, 2]), arcs.new_tensor([0, 1, 1, 3, 3]))
    )
    torch.testing.assert_close(
        outputs, (arcs.new_tensor([2, 0, 1]), arcs.new_tensor([0, 1, 3, 3, 3]))
    )

    arcs[0, 0] = 1
    updated = scatter_groups(arcs[:, 0], 4)
    assert updated is not first
    torch.testing.assert_close(
        updated, (arcs.new_tensor([1, 0, 2]), arcs.new_tensor([0, 1, 2, 3, 3]))
    )
    assert scatter_groups(arcs[:, 1], 4) is not outputs


def test_subranges_strides_and_destination_counts_do_not_reuse_incompatible_groups():
    index = torch.tensor([1, 0, 1, 0])
    whole = scatter_groups(index, 2)
    for view, count, order, offsets in (
        (index[:2], 2, [1, 0], [0, 1, 2]),
        (index[::2], 2, [0, 1], [0, 0, 2]),
        (index, 3, [1, 3, 0, 2], [0, 2, 4, 4]),
    ):
        actual = scatter_groups(view, count)
        assert actual is not whole
        torch.testing.assert_close(actual, (index.new_tensor(order), index.new_tensor(offsets)))


def test_inference_topology_rebuilds_without_a_version_counter():
    with torch.inference_mode():
        index = torch.tensor([1, 0, 1])
        actual = scatter_groups(index, 2)
        torch.testing.assert_close(
            actual, (index.new_tensor([1, 0, 2]), index.new_tensor([0, 1, 3]))
        )
        index[0] = 0
        updated = scatter_groups(index, 2)
        torch.testing.assert_close(
            updated, (index.new_tensor([0, 1, 2]), index.new_tensor([0, 2, 3]))
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_groups_are_reused_only_on_the_stream_that_constructed_them():
    index = torch.tensor([1, 0, 1], device="cuda")
    first = scatter_groups(index, 2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        second = scatter_groups(index, 2)
        assert second is not first
        assert scatter_groups(index, 2) is second
    stream.synchronize()
    torch.testing.assert_close(first, second)
    assert scatter_groups(index, 2) is first
