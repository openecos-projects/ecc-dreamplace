"""Dependency grouping must preserve the accepted timing trajectory."""

from dreamplace.ops.placeio_ecc.commit_verify import commit_buffer_actions_verified
from test_commit_verify import FakeBackend, _action, _install


class EpochBackend(FakeBackend):
    def _timing(self):
        # Each accepted prefix improves timing. Moving the third mutation
        # ahead of the second creates an unqualified intermediate state.
        gain = {
            frozenset(): 0.0,
            frozenset({0}): 10.0,
            frozenset({1}): 5.0,
            frozenset({0, 1}): 15.0,
            frozenset({0, 2}): -10.0,
            frozenset({0, 1, 2}): 20.0,
        }[frozenset(self.committed)]
        return {"wns_ps": -100.0 + gain, "tns_ps": -1000.0 + 10.0 * gain}


def test_interleaved_source_net_dependencies_preserve_accepted_epoch_order(monkeypatch):
    backend = EpochBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {})
    _install(monkeypatch, backend)
    actions = [
        {**_action(0), "net_name": "n0", "tree_bundle_id": "epoch0"},
        {**_action(1), "net_name": "n1", "tree_bundle_id": "epoch1"},
        {**_action(2), "net_name": "n0", "tree_bundle_id": "epoch2"},
    ]
    result = commit_buffer_actions_verified(
        backend, actions, chunk_size=1, reader=lambda pydb: pydb.timing
    )

    assert (result["status"], result["kept_actions"], result["reverted_count"]) == (
        "ok",
        actions,
        0,
    )
    assert backend.calls == [actions]
    assert result["timing_after"] == {"wns_ps": -80.0, "tns_ps": -800.0}
