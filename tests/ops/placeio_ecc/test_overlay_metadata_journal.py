"""Native metadata checkpoints share immutable records and restore exact roots."""

import pytest
from dreamplace.ops.placeio_ecc.buffer_overlay import _json_safe
from tests.ops.placeio_ecc.test_native_timing_campaign import _action
from tests.ops.placeio_ecc.test_native_timing_evaluation import _session
from tests.ops.rebuffer.test_persistent_timing_resize import _same_state


def test_metadata_journal_restores_nested_native_state_without_json_copy(monkeypatch):
    _, session, bridge = _session()
    overlay = session.overlay
    old_root = session.checkpoint()["checkpoint_id"]
    overlay.enable_metadata_journal()
    session.apply([_action(overlay)])
    session.evaluate()
    checkpoint = session.checkpoint()["checkpoint_id"]
    before = _json_safe(overlay.actions)
    metadata = overlay._checkpoints[checkpoint]
    assert metadata["actions"][0] is overlay.actions[0]
    with pytest.raises(TypeError, match="immutable"):
        overlay.actions[0]["downstream_pin_names"].append("unexpected")
    state = bridge.graph.state_view()
    session.apply([_action(overlay)])
    session.evaluate()

    def forbidden(*_args, **_kwargs):
        pytest.fail("journal checkpoint/restore deep-copied JSON metadata")

    with monkeypatch.context() as patch:
        patch.setattr("dreamplace.ops.placeio_ecc.buffer_overlay._json_safe", forbidden)
        session.checkpoint()
        session.restore(checkpoint)
    assert _json_safe(overlay.actions) == before
    _same_state(bridge.graph.state_view(), state, bitwise=True)
    session.restore(old_root)
    assert overlay.actions == []


def test_record_publication_isolates_inputs_exports_and_checkpoint_forks():
    _, session, _ = _session()
    overlay = session.overlay
    overlay.enable_metadata_journal()
    action = _action(overlay)
    session.apply([action])
    checkpoint = session.checkpoint()["checkpoint_id"]
    before = _json_safe(overlay.actions)
    action["downstream_pin_names"].append("foreign")
    exported = _json_safe(overlay.actions)
    exported[0]["downstream_pin_names"].append("changed export")
    assert _json_safe(overlay.actions) == before
    target = overlay.actions[0]["session_buffer_id"]
    session.apply([{"action_kind": "buffer_remove", "session_buffer_id": target}])
    session.restore(checkpoint)
    assert _json_safe(overlay.actions) == before
    session.apply([_action(overlay)])
    session.restore(checkpoint)
    assert _json_safe(overlay.actions) == before
    assert overlay.net_patches is overlay._checkpoints[checkpoint]["net_patches"]
