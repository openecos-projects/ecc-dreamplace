"""Repeated source edges preserve locations and loads in placeio_ecc."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp as native
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from test_timing_carrier import _base


def test_complete_wire_chain_projects_in_driver_order_with_original_steiner_ids():
    base = _base()
    model = RebufferTimingModel(base, use_pin_offset=False)
    state = native.incremental_topology_from_steiner(
        steiner_payload(base, model), 1, 2, model.r_unit, model.c_unit,
    )
    view = native.incremental_topology_view(state)
    edge = next(item for item in view["edges"] if item["child"] == 3)
    source, target = view["coordinates"][edge["parent"]], view["coordinates"][3]
    payload = native.export_overlay_carrier(base, steiner_payload(base, model), [], [])
    actions = [
        {
            "net_id": 1, "driver_pin": 2, "downstream_pins": [3], "buffer_cell_index": 1,
            "parent_node_id": 0, "child_node_id": 1,
            "source_parent_node_id": edge["parent"], "source_child_node_id": 3,
            "source_edge_fraction": fraction, "segment_tree_depth": 0,
            "x_dbu": source[0] + fraction * (target[0] - source[0]),
            "y_dbu": source[1] + fraction * (target[1] - source[1]),
        }
        for fraction in (0.75, 0.5, 0.25)
    ]
    projected = list(native.project_overlay_actions(base, payload, [state], actions))
    assert [action["source_edge_fraction"] for action in projected] == [0.25, 0.5, 0.75]
    assert [action["x_dbu"] for action in projected] == pytest.approx([a["x_dbu"] for a in actions[::-1]])
    assert [action["downstream_pins"] for action in projected] == [[3]] * 3
    assert len({(a["parent_node_id"], a["child_node_id"]) for a in projected}) == 3
    # Projection simulates the bundle on copies, preserving the active source.
    assert native.incremental_topology_view(state) == view
