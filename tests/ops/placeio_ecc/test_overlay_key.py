from types import SimpleNamespace

from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay, action_digest
from dreamplace.ops.placeio_ecc.current_timing import overlay_key


def _overlay():
    return BufferOverlay(SimpleNamespace(net_names=["n0", "n1"]))


def test_overlay_key_tracks_every_list_mutation_and_restore():
    overlay = _overlay()
    a, b = {"net_name": "n0", "session_buffer_id": 0}, {"net_name": "n1", "session_buffer_id": 1}
    keys = []
    for mutate in (
        lambda: overlay.actions.append(a),
        lambda: overlay.actions.append(b),
        lambda: overlay.actions.pop(0),
        lambda: setattr(overlay, "actions", [a, b]),
        lambda: overlay.actions.__setitem__(1, a),
    ):
        mutate()
        keys.append(overlay_key(overlay))
        assert keys[-1] == action_digest(list(overlay.actions))
    assert keys[0] != keys[1] and keys[2] != keys[1] and keys[3] == keys[1]


def test_overlay_key_is_cached_between_queries(monkeypatch):
    import dreamplace.ops.placeio_ecc.buffer_overlay as module

    overlay = _overlay()
    overlay.actions.append({"net_name": "n0", "session_buffer_id": 0})
    calls = []
    real = module.action_digest
    monkeypatch.setattr(module, "action_digest", lambda actions: calls.append(1) or real(actions))
    first = overlay_key(overlay)
    assert [overlay_key(overlay) for _ in range(5)] == [first] * 5
    assert len(calls) == 1
