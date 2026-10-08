"""Excluded GR nets stay outside the immutable electrical snapshot."""

import numpy as np
import torch
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.route_snapshot import prepare_snapshot


def test_native_excluded_net_has_no_rc_tree(snapshot_inputs):
    pack, rc, mapping, identity = snapshot_inputs
    excluded = dict(pack)
    excluded["net_status"] = np.full_like(pack["net_status"], 4)
    snapshot = prepare_snapshot(excluded, rc, mapping, identity)
    assert snapshot.filtered_net_count == 1
    assert snapshot.net_ids.numel() == snapshot.parent.numel() == 0
    assert not snapshot.valid_pin_mask.any()
    assert mapping.eligible_nets.all()  # The caller's mask stays intact.
    cap = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    groups = GRParasiticsOp(snapshot)(cap, cap, cap)
    for group in groups[:2]:
        for value in group.values():
            torch.testing.assert_close(value, cap)
    for group in groups[2:]:
        for value in group.values():
            torch.testing.assert_close(value, torch.zeros_like(cap))
    groups[1]["generic"].sum().backward()
    torch.testing.assert_close(cap.grad, torch.ones_like(cap))
