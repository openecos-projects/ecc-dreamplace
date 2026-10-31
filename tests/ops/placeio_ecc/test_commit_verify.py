from copy import deepcopy
from types import SimpleNamespace

import pytest
from dreamplace.ops.placeio_ecc import commit_verify
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction


class FakePydb:
    timing_schema_version = 2
    node_names = ()
    pin_names = ()
    net_names = ()

    def __init__(self, timing):
        self.timing = dict(timing)


class FakeBackend:
    """Apply/refresh double whose timing is a function of the committed set."""

    def __init__(self, base, deltas):
        self.base = dict(base)
        self.deltas = dict(deltas)
        self.committed = set()
        self.pydb = FakePydb(self._timing())
        self.timing_enabled = True
        self.calls = []
        self._owner = {}

    def _timing(self):
        wns = self.base["wns_ps"] + sum(self.deltas[i][0] for i in self.committed)
        tns = self.base["tns_ps"] + sum(self.deltas[i][1] for i in self.committed)
        return {"wns_ps": wns, "tns_ps": tns}

    def apply_buffer_actions(self, actions, digest):
        self.calls.append(deepcopy(list(actions)))
        inserted = []
        for action in actions:
            if action.get("action_kind") == "buffer_remove":
                self.committed.discard(self._owner[action["instance_name"]])
                continue
            owner = action["probe_id"]
            name = f"buffer_coord_{owner}"
            self._owner[name] = owner
            self.committed.add(owner)
            inserted.append(name)
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "removed_count": len(actions) - len(inserted),
            "committed_buffer_instance_names": inserted,
        }

    def refresh(self):
        self.pydb.timing = self._timing()
        return self.pydb, {"status": "ok"}


def _action(index):
    return {
        "action_kind": "buffer_insert",
        "probe_id": index,
        "net_name": f"net{index}",
        "buffer_master_name": "BUFX1H7R",
        "candidate_location_x_dbu": 100 * index,
        "candidate_location_y_dbu": 100 * index,
        "downstream_pin_names": [f"u{index}/a"],
    }


def _install(monkeypatch, backend):
    monkeypatch.setattr(
        PlaceIOFunction,
        "apply_buffer_actions",
        staticmethod(lambda raw_db, actions, digest: backend.apply_buffer_actions(actions, digest)),
    )
    monkeypatch.setattr(
        commit_verify,
        "refresh_python_timing_carrier",
        lambda raw_db: backend.refresh(),
    )


def test_commit_verify_splits_a_regressing_chunk_to_isolate_the_bad_action(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {
            0: (5.0, 50.0),
            1: (5.0, 50.0),
            2: (-80.0, -20000.0),
            3: (5.0, 50.0),
        },
    )
    _install(monkeypatch, backend)
    actions = [_action(index) for index in range(4)]

    result = commit_verify.commit_buffer_actions_verified(
        backend, actions, chunk_size=4, reader=lambda pydb: dict(pydb.timing)
    )

    assert result["status"] == "ok"
    assert result["kept_count"] == 3
    assert result["reverted_count"] == 1
    assert [chunk["status"] for chunk in result["chunks"]] == [
        "split",
        "kept",
        "split",
        "reverted",
        "kept",
    ]
    assert backend.committed == {0, 1, 3}
    assert result["timing_after"] == {"wns_ps": -85.0, "tns_ps": -850.0}


def test_commit_verify_keeps_every_improving_chunk(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {0: (5.0, 50.0), 1: (5.0, 50.0)},
    )
    _install(monkeypatch, backend)

    result = commit_verify.commit_buffer_actions_verified(
        backend, [_action(0), _action(1)], chunk_size=1, reader=lambda pydb: dict(pydb.timing)
    )

    assert [chunk["status"] for chunk in result["chunks"]] == ["kept", "kept"]
    assert result["reverted_count"] == 0
    assert result["timing_after"] == {"wns_ps": -90.0, "tns_ps": -900.0}


def test_commit_verify_without_chunking_verifies_one_transaction(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {0: (5.0, 50.0), 1: (5.0, 50.0)},
    )
    _install(monkeypatch, backend)

    result = commit_verify.commit_buffer_actions_verified(
        backend, [_action(0), _action(1)], reader=lambda pydb: dict(pydb.timing)
    )

    assert result["chunk_count"] == 1
    assert result["kept_count"] == 2
    assert result["timing_after"] == {"wns_ps": -90.0, "tns_ps": -900.0}


@pytest.mark.parametrize("chunk_size", [1, 100])
def test_same_net_chain_is_atomic_across_chunking_and_retries(monkeypatch, chunk_size):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {0: (5.0, 50.0), 1: (-80.0, -20000.0), 2: (5.0, 50.0)},
    )
    _install(monkeypatch, backend)
    actions = [_action(i) for i in range(3)]
    actions[1]["net_name"] = actions[0]["net_name"]

    result = commit_verify.commit_buffer_actions_verified(
        backend,
        actions,
        chunk_size=chunk_size,
        reader=lambda pydb: pydb.timing,
    )

    assert result["status"] == "ok"
    assert (result["kept_count"], result["reverted_count"]) == (1, 2)
    assert result["timing_after"] == {"wns_ps": -95.0, "tns_ps": -950.0}
    assert [len(call) for call in backend.calls] == (
        [2, 2, 1] if chunk_size == 1 else [3, 3, 2, 2, 1]
    )
    assert [r["verified_status"] for r in result["action_results"]] == [
        "reverted",
        "reverted",
        "kept",
    ]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None])
def test_invalid_post_commit_timing_rolls_back_and_fails(monkeypatch, invalid):
    backend = FakeBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {0: (5.0, 50.0)})
    _install(monkeypatch, backend)

    def reader(pydb):
        if backend.committed:
            return {"wns_ps": invalid, "tns_ps": -950.0}
        return pydb.timing

    result = commit_verify.commit_buffer_actions_verified(backend, [_action(0)], reader=reader)

    assert result["status"] == "timing_failed"
    assert backend.committed == set()
    assert result["timing_after"] == backend.base
    assert result["kept_count"] == 0
    assert result["timing_failed_count"] == 1


def test_refresh_exception_is_undone_and_reported(monkeypatch):
    backend = FakeBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {0: (5.0, 50.0)})
    _install(monkeypatch, backend)

    def refresh(raw_db):
        if backend.committed:
            raise RuntimeError("unmapped timing endpoint")
        return backend.refresh()

    monkeypatch.setattr(commit_verify, "refresh_python_timing_carrier", refresh)
    result = commit_verify.commit_buffer_actions_verified(
        backend, [_action(0)], reader=lambda pydb: pydb.timing
    )

    assert result["status"] == "refresh_failed"
    assert result["timing_after"] == backend.base
    assert backend.committed == set()
    assert result["action_results"][0]["verified_status"] == "refresh_failed"


def test_commit_verify_skips_without_actions():
    result = commit_verify.commit_buffer_actions_verified(
        SimpleNamespace(pydb=FakePydb({})), [], chunk_size=4
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "no_actions"


def test_missing_baseline_metric_prevents_native_mutation(monkeypatch):
    backend = FakeBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {0: (5.0, 50.0)})
    _install(monkeypatch, backend)
    result = commit_verify.commit_buffer_actions_verified(
        backend,
        [_action(0)],
        reader=lambda pydb: {"wns_ps": -100.0},
    )

    assert result["status"] == "timing_failed"
    assert backend.calls == []
    assert "timing_after" not in result


def test_partial_native_acceptance_is_reverted_and_cannot_be_kept(monkeypatch):
    backend = FakeBackend(
        {"wns_ps": -100.0, "tns_ps": -1000.0},
        {0: (5.0, 50.0), 1: (5.0, 50.0)},
    )
    _install(monkeypatch, backend)

    def partial(raw_db, actions, digest):
        if actions[0]["action_kind"] == "buffer_remove":
            return backend.apply_buffer_actions(actions, digest)
        return backend.apply_buffer_actions(actions[:1], digest)

    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(partial))
    result = commit_verify.commit_buffer_actions_verified(
        backend,
        [_action(0), _action(1)],
        reader=lambda pydb: pydb.timing,
    )

    assert result["status"] == "commit_failed"
    assert result["kept_count"] == 0
    assert result["timing_after"] == backend.base
    assert backend.committed == set()


def test_sparse_schema_is_rejected_before_native_mutation(monkeypatch):
    backend = FakeBackend({"wns_ps": -100.0, "tns_ps": -1000.0}, {0: (5.0, 50.0)})
    backend.pydb.timing_schema_version = 3
    _install(monkeypatch, backend)

    result = commit_verify.commit_buffer_actions_verified(
        backend,
        [_action(0)],
        reader=lambda pydb: pydb.timing,
    )

    assert result["status"] == "unsupported_timing_schema"
    assert backend.calls == []
