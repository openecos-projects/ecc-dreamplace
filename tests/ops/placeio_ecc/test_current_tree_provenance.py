"""Current-tree source IDs must belong to the carrier used for projection."""

from types import SimpleNamespace

from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from test_timing_carrier import _base


def test_current_tree_keeps_carrier_identity_when_python_steiner_ids_change(monkeypatch):
    base = _base()
    carrier_tree = {
        "net_flat_topo_sort_start": [0, 2, 6],
        "net_flat_topo_sort": [0, 1, 2, 4, 5, 3],
        "pin_fa": [-1, 0, -1, 5, 2, 4],
        "flat_pin_to_start": [0, 1, 1, 2, 2, 3, 4],
        "flat_pin_to": [1, 4, 5, 3],
        "flat_pin_from": [0, 2, 4, 5],
        "node_x": [0.0, 10000.0, 10000.0, 100000.0, 30000.0, 60000.0],
        "node_y": [0.0] * 6,
    }
    # The geometry/connectivity are identical, but Python's Steiner rebuild
    # gives the two non-pin vertices the opposite numeric identities.
    python_tree = {
        **carrier_tree,
        "net_flat_topo_sort": [0, 1, 2, 5, 4, 3],
        "pin_fa": [-1, 0, -1, 4, 5, 2],
        "flat_pin_to_start": [0, 1, 1, 2, 2, 3, 4],
        "flat_pin_to": [1, 5, 3, 4],
        "flat_pin_from": [0, 2, 4, 5],
        "node_x": [0.0, 10000.0, 10000.0, 100000.0, 60000.0, 30000.0],
    }
    reference = RebufferTimingModel(base, use_pin_offset=False, steiner_view=carrier_tree)
    active = RebufferTimingModel(base, use_pin_offset=False, steiner_view=python_tree)
    expected = RebufferPlanContext(base, reference).native_current_tree_payload(1)
    runtime = SimpleNamespace(
        model=active,
        payload={
            "steiner": steiner_payload(base, reference),
            "original_net_by_net": [0, 1],
        },
    )
    context = RebufferPlanContext(base, active)
    monkeypatch.setattr(context, "_runtime", lambda: runtime)

    actual = context.native_current_tree_payload(1)

    assert active.baseline.as_dict() == reference.baseline.as_dict()
    assert rebuffer_cpp.export_current_driver_tree(actual["current_tree"]()) == (
        rebuffer_cpp.export_current_driver_tree(expected["current_tree"]())
    )
