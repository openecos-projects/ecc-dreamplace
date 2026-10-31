"""Physical verification must preserve complete native tree dependencies."""

import pytest
from dreamplace.ops.placeio_ecc.commit_verify import commit_buffer_actions_verified
from test_commit_verify import FakeBackend, _action, _install


class JointTreeBackend(FakeBackend):
    def _timing(self):
        gain = 5.0 if len(self.committed) == 2 else -5.0 if self.committed else 0.0
        return {"wns_ps": -100.0 + gain, "tns_ps": -1000.0 + 10 * gain}


@pytest.mark.parametrize("chunk_size", [1, 8])
def test_cross_net_native_bundle_keeps_joint_gain_even_when_larger_than_chunk(
    monkeypatch, chunk_size
):
    backend = JointTreeBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {})
    _install(monkeypatch, backend)
    actions = [{**_action(i), "tree_bundle_id": "tree:epoch0"} for i in range(2)]
    result = commit_buffer_actions_verified(
        backend,
        actions,
        chunk_size=chunk_size,
        reader=lambda pydb: pydb.timing,
    )
    assert result["core"] == "cpp"
    assert (result["status"], result["kept_actions"], backend.committed) == ("ok", actions, {0, 1})
    assert [record["bundle_count"] for record in result["chunks"]] == [1]
    assert result["timing_after"] == {"wns_ps": -95.0, "tns_ps": -950.0}


def test_failed_complete_tree_is_never_retried_as_individual_source_nets(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {0: (5.0, 50.0), 1: (-10.0, -100.0)},
    )
    _install(monkeypatch, backend)
    actions = [{**_action(i), "tree_bundle_id": "tree:epoch0"} for i in range(2)]
    result = commit_buffer_actions_verified(backend, actions, reader=lambda pydb: pydb.timing)
    assert (result["kept_count"], result["reverted_count"], backend.committed) == (0, 2, set())
    assert [record["status"] for record in result["chunks"]] == ["reverted"]
    assert [len(call) for call in backend.calls] == [2, 2]


def test_shared_source_net_connects_tree_bundles_across_accepted_epochs(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {i: (1.0, 10.0) for i in range(5)},
    )
    _install(monkeypatch, backend)
    actions = [
        {**_action(0), "net_name": "n0", "tree_bundle_id": "tree0"},
        {**_action(1), "net_name": "n1", "tree_bundle_id": "tree0"},
        {**_action(2), "net_name": "n1", "tree_bundle_id": "tree1"},
        {**_action(3), "net_name": "n2", "tree_bundle_id": "tree1"},
        {**_action(4), "net_name": "n3", "tree_bundle_id": "tree2"},
    ]
    result = commit_buffer_actions_verified(
        backend,
        actions,
        chunk_size=1,
        reader=lambda pydb: pydb.timing,
    )
    assert result["kept_actions"] == actions
    assert [len(call) for call in backend.calls] == [4, 1]
    assert [record["bundle_count"] for record in result["chunks"]] == [1, 1]
