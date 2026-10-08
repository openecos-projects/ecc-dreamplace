from types import SimpleNamespace

import pytest

from dreamplace.ops.buffer_insertion.buffering_lane import (
    BufferCommitRequest,
    _action_digest,
)
from dreamplace.ops.placeio_common.backend_contract import (
    ieda_backend_caps,
    openroad_backend_caps,
)
from dreamplace.ops.placeio_common.physical_mutation import (
    physical_mutation_backend_for,
)


def _request(*, actions=(), commit_enabled=True, committed_def_path=""):
    actions = tuple(actions)
    return BufferCommitRequest(
        version=1,
        mode="candidate",
        actions=actions,
        action_digest=_action_digest(actions),
        iteration=3,
        reason="unit",
        mutation_kind="topology-changing",
        commit_enabled=commit_enabled,
        committed_def_path=committed_def_path,
        refresh_mode="topo",
        rebuild_mode="topo",
        projection_metadata={},
        runtime_refresh_summary={},
    )


def _placedb(caps, bridge=None):
    return SimpleNamespace(
        backend_caps=caps.to_dict(),
        openroad_bridge=bridge,
    )


def test_adapter_exposes_an_isolated_capability_snapshot():
    openroad_caps = openroad_backend_caps().to_dict()
    ieda_caps = ieda_backend_caps().to_dict()
    openroad = physical_mutation_backend_for(_placedb(openroad_backend_caps()))
    ieda = physical_mutation_backend_for(_placedb(ieda_backend_caps()))

    assert openroad.capabilities() == openroad_caps
    assert ieda.capabilities() == ieda_caps
    snapshot = openroad.capabilities()
    snapshot["backend"] = "tampered"
    assert openroad.capabilities()["backend"] == "openroad"


def test_openroad_adapter_commits_existing_coordinate_action_backend(monkeypatch, tmp_path):
    calls = []

    class Bridge:
        def __init__(self):
            self.def_paths = []

        def write_def(self, path):
            self.def_paths.append(path)

    bridge = Bridge()
    placedb = _placedb(openroad_backend_caps(), bridge)
    committed_def = tmp_path / "committed.def"
    request = _request(
        actions=({"action_id": 4},),
        committed_def_path=str(committed_def),
    )

    def fake_commit(actions, *, placedb, output_dir):
        calls.append((tuple(actions), placedb, output_dir))
        return {
            "status": "accepted",
            "accepted_action_count": 1,
            "action_count": 1,
        }

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.coordinate_backend.commit_coordinate_buffer_actions",
        fake_commit,
    )

    result = physical_mutation_backend_for(placedb).commit_buffers(
        request,
        output_dir=str(tmp_path),
    )

    assert calls == [(request.actions, placedb, str(tmp_path))]
    assert result["status"] == "accepted"
    assert result["backend"] == "openroad"
    assert result["refresh_required"]
    assert result["refresh_mode"] == "topo"
    assert result["rebuild_mode"] == "topo"
    assert result["committed_def_path"] == str(committed_def)
    assert bridge.def_paths == [str(committed_def)]


def test_ieda_adapter_rejects_buffer_commit_before_backend_mutation(monkeypatch):
    placedb = _placedb(ieda_backend_caps())
    request = _request(actions=({"action_id": 4},))

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.coordinate_backend.commit_coordinate_buffer_actions",
        lambda *args, **kwargs: pytest.fail("unsupported iEDA must not mutate"),
    )

    result = physical_mutation_backend_for(placedb).commit_buffers(request)

    assert result["status"] == "unsupported"
    assert result["reason"] == "unsupported_backend_capability"
    assert result["backend"] == "ieda"
    assert result["required_capability"] == "supports_buffer_commit"
    assert result["attempted_action_count"] == 0
    assert result["accepted_action_count"] == 0
    assert not result["topology_mutated"]


def test_adapter_rejects_tampered_request_before_coordinate_backend(monkeypatch):
    placedb = _placedb(openroad_backend_caps())
    request = _request(actions=({"action_id": 4},))
    request.actions[0]["action_id"] = 5

    monkeypatch.setattr(
        "dreamplace.ops.buffer_insertion.coordinate_backend.commit_coordinate_buffer_actions",
        lambda *args, **kwargs: pytest.fail("tampered request must not mutate"),
    )

    with pytest.raises(ValueError, match="do not match action digest"):
        physical_mutation_backend_for(placedb).commit_buffers(request)


def test_openroad_adapter_refreshes_through_placedb_bridge_contract():
    calls = []
    placedb = _placedb(openroad_backend_caps())

    def refresh_from_openroad_bridge(*, refresh_mode, rebuild_mode):
        calls.append((refresh_mode, rebuild_mode))
        return {"status": "ok", "rebuild": {"topology_generation": 3}}

    placedb.refresh_from_openroad_bridge = refresh_from_openroad_bridge

    result = physical_mutation_backend_for(placedb).refresh(
        refresh_mode="topo",
        rebuild_mode="topo",
    )

    assert calls == [("topo", "topo")]
    assert result["status"] == "ok"


def test_unknown_backend_is_explicitly_unsupported_without_refresh():
    placedb = SimpleNamespace(backend_caps={"backend": "unknown"})
    backend = physical_mutation_backend_for(placedb)

    result = backend.commit_buffers(_request(actions=({"action_id": 4},)))
    refresh = backend.refresh(refresh_mode="topo", rebuild_mode="topo")

    assert result["status"] == "unsupported"
    assert refresh["status"] == "unsupported"
    assert refresh["required_capability"] == "supports_committed_refresh"
