from types import SimpleNamespace

import torch
from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
from test_incremental_timing import STATE_FIELDS
from test_overlay_carrier import _model
from test_timing_carrier import _base


def test_persistent_alternative_does_not_mutate_saved_checkpoint():
    base = _base()
    timing = _model(base)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=base), timing_model=timing)
    session.begin()
    empty = session.checkpoint()["checkpoint_id"]
    action = {
        "action_kind": "buffer_insert",
        "net_name": "n1",
        "driver_pin_name": "u/Y",
        "downstream_pin_names": ["OUT"],
        "buffer_master_name": "BUF1",
        "candidate_location_x_dbu": 50000,
        "candidate_location_y_dbu": 0,
    }
    session.apply([action])
    first = session.evaluate()
    first_graph = timing.current_graph()
    saved = {field: getattr(timing.model, field).clone() for field in STATE_FIELDS}
    checkpoint = session.checkpoint()["checkpoint_id"]
    session.restore(empty)
    session.apply([{**action, "buffer_master_name": "BUF2", "candidate_location_x_dbu": 75000}])
    result = session.evaluate()
    assert result["topology_source"] == "native_session_patch_handle"

    for field, values in saved.items():
        torch.testing.assert_close(
            getattr(first_graph.model.model, field), values, rtol=0, atol=1e-8
        )
    session.restore(checkpoint)
    assert session.evaluate() == first
    for field, values in saved.items():
        torch.testing.assert_close(getattr(timing.model, field), values, rtol=0, atol=1e-8)
