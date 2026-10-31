from copy import deepcopy
from types import SimpleNamespace

import pytest
from dreamplace.ops.placeio_ecc.buffer_session import (
    BufferOverlay,
    PlaceIOBufferSession,
    action_digest,
)
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
from dreamplace.ops.rebuffer import rebuffer_cpp


class FakePydb:
    timing_schema_version = 2

    def __init__(self):
        self.net_names = ("net0", "net1")
        self.pin_names = ("driver", "p2", "p3")
        self.apply_calls = []

    def apply_buffer_actions(self, actions, digest):
        self.apply_calls.append((deepcopy(actions), digest))
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "rejected_count": 0,
            "failed_count": 0,
        }


def _raw_db():
    pydb = FakePydb()
    return SimpleNamespace(
        pydb=pydb,
        timing_enabled=True,
        refresh_generation=0,
    )


def _action(net_name, x, y, *, load="u/a"):
    return {
        "action_kind": "buffer_insert",
        "net_name": net_name,
        "buffer_master_name": "BUFX1H7R",
        "candidate_location_x_dbu": x,
        "candidate_location_y_dbu": y,
        "downstream_pin_names": [load],
    }


class FakeNativeTimingModel:
    def _net_id(self, net_name):
        assert net_name == "net0"
        return 0

    def _pin_id(self, pin_name):
        return {"p2": 2, "p3": 3}[pin_name]

    def _net_record(self, _net_id):
        return {
            "net_id": 0,
            "driver_pin_id": 0,
            "rc_tree": {
                "children_by_node": {0: [1], 1: [2, 3], 2: [], 3: []},
                "edge_rc": {
                    (0, 1): {"r": 10.0, "c": 2.0},
                    (1, 2): {"r": 5.0, "c": 1.0},
                    (1, 3): {"r": 6.0, "c": 1.0},
                },
                "sink_nodes": [2, 3],
            },
        }


class FakeNativeGeometryTimingModel(FakeNativeTimingModel):
    def create_net_overlay(self, net_name):
        assert net_name == "net0"
        return rebuffer_cpp.create_incremental_topology(
            0,
            0,
            4,
            {
                "edges": [
                    {"parent": 0, "child": 1, "r": 1.0, "c": 1.0},
                    {"parent": 1, "child": 2, "r": 1.0, "c": 1.0},
                    {"parent": 1, "child": 3, "r": 1.0, "c": 1.0},
                ],
                "node_loads": {2: [2], 3: [3]},
            },
        )


def test_overlay_groups_multi_buffer_and_cross_net_actions():
    pydb = FakePydb()
    overlay = BufferOverlay(pydb)

    result = overlay.apply(
        [
            _action("net0", 100, 200, load="u0/a"),
            _action("net0", 300, 400, load="u1/a"),
            _action("net1", 500, 600, load="u2/a"),
        ]
    )

    assert result["applied_count"] == 3
    assert overlay.summary()["buffer_count"] == 3
    assert overlay.summary()["net_count"] == 2
    assert overlay.summary()["net_patches"]["net0"]["inserted_buffer_ids"] == [0, 1]
    assert pydb.apply_calls == []


def test_native_campaign_checkpoint_restores_multiple_net_topologies():
    def topology(net_id, driver, load):
        return rebuffer_cpp.create_incremental_topology(
            net_id,
            driver,
            load + 1,
            {
                "edges": [{"parent": driver, "child": load, "r": 1.0, "c": 1.0}],
                "node_loads": {load: [load]},
            },
        )

    first = topology(0, 0, 1)
    second = topology(1, 2, 3)
    campaign = rebuffer_cpp.create_repair_timing_campaign_session()
    campaign.attach_topology(0, first)
    campaign.attach_topology(1, second)
    checkpoint = campaign.checkpoint()

    def action(action_id, driver, load):
        return {
            "action_id": action_id,
            "buffer_master_name": "BUF",
            "downstream_pin_ids": [load],
            "parent_node_id": driver,
            "child_node_id": load,
            "split_ratio": 0.5,
            "x_dbu": 0,
            "y_dbu": 0,
        }

    campaign.apply_candidate(0, action(0, 0, 1))
    campaign.apply_candidate(1, action(1, 2, 3))
    assert campaign.summary()[0]["buffers"]
    assert campaign.summary()[1]["buffers"]
    assert campaign.summary()["action_ledger"] == {0: "active", 1: "active"}

    campaign.restore(checkpoint)
    assert campaign.summary()[0]["buffers"] == []
    assert campaign.summary()[1]["buffers"] == []
    assert campaign.summary()["action_ledger"] == {}


def test_native_campaign_records_candidate_lifecycle_and_restores_action_state():
    campaign = rebuffer_cpp.create_repair_timing_campaign_session()

    observation = {
        "variant_index": 2,
        "status": "ok",
        "wns_ps": -10.0,
        "tns_ps": -20.0,
        "endpoint_slack_ps": -5.0,
        "action_ids": [7, 8],
    }
    campaign.record_candidate(3, observation, "candidate_observation")
    campaign.record_candidate(3, observation, "selected_deferred")
    campaign.record_action_state(3, 7, "accepted")
    checkpoint = campaign.checkpoint()
    campaign.record_action_state(3, 7, "pending")

    restored = campaign.restore(checkpoint)
    assert restored["action_ledger"] == {7: "accepted"}
    history = restored["ledger_history"]
    assert [entry["state"] for entry in history[:2]] == [
        "candidate_observation",
        "selected_deferred",
    ]
    assert history[-1]["state"] == "pending"


def test_native_campaign_replaces_current_tree_atomically():
    topology = rebuffer_cpp.create_incremental_topology(
        0,
        0,
        2,
        {
            "edges": [{"parent": 0, "child": 1, "r": 1.0, "c": 1.0}],
            "node_loads": {1: [1]},
        },
    )
    campaign = rebuffer_cpp.create_repair_timing_campaign_session()
    campaign.attach_topology(0, topology)

    def action(action_id):
        return {
            "action_id": action_id,
            "buffer_master_name": "BUF",
            "downstream_pin_ids": [1],
            "parent_node_id": 0,
            "child_node_id": 1,
            "split_ratio": 0.5,
        }

    campaign.apply_candidate(0, action(0))
    replaced = campaign.replace_candidate_tree(0, [action(10)])

    assert replaced["removed_action_ids"] == [0]
    assert [item["action_id"] for item in replaced["applied"]] == [10]
    assert [item["action_id"] for item in campaign.summary()[0]["buffers"]] == [10]
    assert campaign.summary()["action_ledger"] == {0: "replaced", 10: "active"}


def test_overlay_uses_native_campaign_for_current_tree_replacement():
    overlay = BufferOverlay(
        FakePydb(), timing_model=FakeNativeGeometryTimingModel()
    )
    first = {
        **_action("net0", 100, 200, load="p2"),
        "parent_node_id": 1,
        "child_node_id": 2,
        "split_ratio": 0.4,
    }
    second = {
        **_action("net0", 300, 400, load="p2"),
        "parent_node_id": 1,
        "child_node_id": 2,
        "split_ratio": 0.6,
    }
    overlay.apply([first])
    replaced = overlay.replace_session_tree("net0", [second])

    assert replaced["replacement_mode"] == "native_campaign"
    assert replaced["removed_session_buffer_ids"] == [0]
    assert replaced["session_buffer_ids"] == [1]
    buffers = overlay.summary()["native_topologies"]["net0"]["buffers"]
    assert [buffer["action_id"] for buffer in buffers] == [1]


def test_overlay_updates_native_incremental_topology_and_checkpoint():
    pydb = FakePydb()
    overlay = BufferOverlay(pydb, timing_model=FakeNativeTimingModel())
    action = _action("net0", 100, 200, load="p2")
    action.update(
        {
            "parent_node_id": 1,
            "child_node_id": 2,
            "split_ratio": 0.4,
        }
    )

    applied = overlay.apply([action])
    assert applied["applied_count"] == 1
    assert overlay.summary()["native_topology_events"][0]["status"] == "ok"
    checkpoint = overlay.checkpoint("native")
    first_view = overlay.summary()["native_topologies"]["net0"]
    second = _action("net0", 300, 400, load="p3")
    second.update(
        {
            "parent_node_id": 1,
            "child_node_id": 3,
            "split_ratio": 0.5,
        }
    )
    overlay.apply([second])
    overlay.restore(checkpoint["checkpoint_id"])
    view = overlay.summary()["native_topologies"]["net0"]
    assert [buffer["action_id"] for buffer in view["buffers"]] == [0]
    assert view == first_view


def test_checkpoint_restore_and_rollback_preserve_base_identity():
    raw_db = _raw_db()
    base = raw_db.pydb
    session = PlaceIOBufferSession(raw_db)

    assert session.begin()["status"] == "ok"
    session.apply([_action("net0", 100, 200)])
    checkpoint = session.checkpoint("accepted")
    session.apply([_action("net1", 300, 400)])
    assert session.overlay.summary()["buffer_count"] == 2

    restored = session.restore(checkpoint["checkpoint_id"])
    assert restored["action_count"] == 1
    assert session.overlay.summary()["buffer_count"] == 1
    assert raw_db.pydb is base

    rolled_back = session.rollback()
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["pydb_identity_preserved"] is True
    assert raw_db.pydb is base
    assert session.overlay.summary()["action_count"] == 0


def test_restore_drops_new_native_net_and_allows_same_action_again():
    overlay = BufferOverlay(FakePydb(), timing_model=FakeNativeTimingModel())
    action = {
        **_action("net0", 100, 200, load="p2"),
        "parent_node_id": 1,
        "child_node_id": 2,
        "split_ratio": 0.4,
    }
    checkpoint = overlay.checkpoint("empty")
    before = overlay.summary()
    overlay.apply([action])
    overlay.restore(checkpoint["checkpoint_id"])
    assert overlay.summary() == before
    overlay.apply([action])
    assert len(overlay.summary()["native_topologies"]["net0"]["buffers"]) == 1


def test_restore_reinstates_removed_action_and_topology():
    overlay = BufferOverlay(FakePydb(), timing_model=FakeNativeTimingModel())
    action = {
        **_action("net0", 100, 200, load="p2"),
        "parent_node_id": 1,
        "child_node_id": 2,
        "split_ratio": 0.4,
    }
    overlay.apply([action])
    checkpoint = overlay.checkpoint("inserted")
    before = overlay.summary()
    overlay.remove_session_buffers()
    overlay.restore(checkpoint["checkpoint_id"])
    assert overlay.summary() == before


def test_failed_batch_restores_native_topology_and_action_ledger():
    overlay = BufferOverlay(FakePydb(), timing_model=FakeNativeTimingModel())
    action = {
        **_action("net0", 100, 200, load="p2"),
        "parent_node_id": 1,
        "child_node_id": 2,
        "split_ratio": 0.4,
    }
    before = overlay.summary()
    with pytest.raises(ValueError, match="active_edge"):
        overlay.apply([action, action])
    assert overlay.summary() == before


def test_seed_rebuilds_verified_topology_without_touching_native_db():
    raw_db = _raw_db()
    action = _action("net0", 100, 200, load="p2")
    action.update(
        {
            "parent_node_id": 1,
            "child_node_id": 2,
            "split_ratio": 0.4,
        }
    )
    session = PlaceIOBufferSession(
        raw_db,
        timing_model=FakeNativeTimingModel(),
    )

    session.begin()
    seeded = session.seed([action])

    assert seeded["seeded_count"] == 1
    assert session.overlay.summary()["native_topology_events"][0]["status"] == "ok"
    assert raw_db.pydb.apply_calls == []
    assert session.overlay.summary()["native_topologies"]["net0"]["epoch"] == 1


def test_evaluate_is_explicit_and_never_writes_native_db(monkeypatch):
    raw_db = _raw_db()
    base = raw_db.pydb
    seen = []

    def timing_model(base_pydb, overlay):
        seen.append((base_pydb, overlay.epoch, overlay.summary()["buffer_count"]))
        return {"status": "ok", "wns": -0.5}

    session = PlaceIOBufferSession(raw_db, timing_model=timing_model)
    session.begin()
    session.apply([_action("net0", 100, 200)])

    result = session.evaluate()

    assert result["wns"] == -0.5
    assert seen == [(base, 1, 1)]
    assert raw_db.pydb is base
    assert base.apply_calls == []


def test_commit_uses_one_batch_and_one_refresh(monkeypatch):
    raw_db = _raw_db()
    base = raw_db.pydb
    new_pydb = FakePydb()
    commit_calls = []
    refresh_calls = []

    def fake_apply(_raw_db, actions, digest):
        commit_calls.append((list(actions), digest))
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "rejected_count": 0,
            "failed_count": 0,
        }

    def fake_refresh(_raw_db):
        refresh_calls.append(True)
        _raw_db.pydb = new_pydb
        _raw_db.refresh_generation += 1
        return new_pydb, {"status": "ok", "generation": 1}

    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(fake_apply))
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier", fake_refresh
    )

    session = PlaceIOBufferSession(raw_db)
    session.begin()
    session.apply([_action("net0", 100, 200), _action("net1", 300, 400)])

    committed = session.commit()

    assert committed["status"] == "committed"
    assert len(commit_calls) == 1
    assert commit_calls[0][1] == action_digest(session.overlay.actions)
    assert refresh_calls == [True]
    assert raw_db.pydb is new_pydb
    assert raw_db.pydb is not base

    second = session.commit()
    assert second["status"] == "already_committed"
    assert len(commit_calls) == 1
    assert len(refresh_calls) == 1


def test_preflight_failure_does_not_refresh_and_can_rollback(monkeypatch):
    raw_db = _raw_db()
    base = raw_db.pydb
    refresh_calls = []

    monkeypatch.setattr(
        PlaceIOFunction,
        "apply_buffer_actions",
        staticmethod(
            lambda _raw_db, _actions, _digest: {
                "status": "rejected",
                "accepted_count": 0,
                "rejected_count": 1,
                "failed_count": 0,
            }
        ),
    )
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier",
        staticmethod(lambda _raw_db: refresh_calls.append(True)),
    )

    session = PlaceIOBufferSession(raw_db)
    session.begin()
    session.apply([_action("net0", 100, 200)])

    failed = session.commit()
    assert failed["status"] == "rejected"
    assert failed["native_db_mutated"] is False
    assert refresh_calls == []
    assert raw_db.pydb is base
    assert session.rollback()["pydb_identity_preserved"] is True


def test_unknown_net_is_rejected_before_overlay_mutation():
    overlay = BufferOverlay(FakePydb())
    with pytest.raises(ValueError, match="unknown net_name"):
        overlay.apply([_action("missing", 100, 200)])
    assert overlay.summary()["action_count"] == 0


def test_remove_session_buffer_before_commit(monkeypatch):
    raw_db = _raw_db()
    commit_calls = []

    def fake_apply(_raw_db, actions, digest):
        commit_calls.append((list(actions), digest))
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "rejected_count": 0,
            "failed_count": 0,
        }

    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(fake_apply))
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier",
        staticmethod(lambda _raw_db: (_raw_db, {"status": "ok"})),
    )

    session = PlaceIOBufferSession(raw_db)
    session.begin()
    session.apply([_action("net0", 100, 200), _action("net1", 300, 400)])
    removed = session.strip_net("net0")

    assert removed["removed_count"] == 1
    assert session.overlay.summary()["buffer_count"] == 1
    assert session.overlay.summary()["net_count"] == 1

    committed = session.commit()

    assert committed["status"] == "committed"
    assert len(commit_calls) == 1
    assert len(commit_calls[0][0]) == 1
    assert commit_calls[0][1] == action_digest(session.overlay.actions)


def test_remove_unknown_session_buffer_is_rejected():
    overlay = BufferOverlay(FakePydb())
    with pytest.raises(ValueError, match="unknown session_buffer_id"):
        overlay.apply([{"action_kind": "buffer_remove", "session_buffer_id": 99}])
    assert overlay.summary()["action_count"] == 0


def test_remove_base_buffers_commits_native_removal_batch(monkeypatch):
    raw_db = _raw_db()
    base = raw_db.pydb
    new_pydb = FakePydb()
    commit_calls = []
    refresh_calls = []

    def fake_apply(_raw_db, actions, digest):
        commit_calls.append((list(actions), digest))
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "removed_count": len(actions),
            "rejected_count": 0,
            "failed_count": 0,
        }

    def fake_refresh(_raw_db):
        refresh_calls.append(True)
        _raw_db.pydb = new_pydb
        _raw_db.refresh_generation += 1
        return new_pydb, {"status": "ok", "generation": 1}

    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(fake_apply))
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier", fake_refresh
    )

    session = PlaceIOBufferSession(raw_db)
    session.begin()
    planned = session.remove_base_buffers(["buffer_coord0", "buffer_coord1"], net_name="net0")

    assert planned["removed_count"] == 2
    assert session.overlay.summary()["base_remove_count"] == 2
    assert session.overlay.summary()["buffer_count"] == 0

    committed = session.commit()

    assert committed["status"] == "committed"
    assert committed["removed_count"] == 2
    assert committed["inserted_count"] == 0
    assert len(commit_calls) == 1
    assert commit_calls[0][0] == [
        {"action_kind": "buffer_remove", "instance_name": "buffer_coord0", "net_name": "net0"},
        {"action_kind": "buffer_remove", "instance_name": "buffer_coord1", "net_name": "net0"},
    ]
    assert commit_calls[0][1] == action_digest(commit_calls[0][0])
    assert refresh_calls == [True]
    assert raw_db.pydb is new_pydb
    assert raw_db.pydb is not base


def test_mixed_base_removal_and_insert_commit_is_rejected():
    raw_db = _raw_db()
    session = PlaceIOBufferSession(raw_db)
    session.begin()
    session.remove_base_buffers(["buffer_coord0"])
    session.apply([_action("net0", 100, 200)])

    with pytest.raises(ValueError, match="mixed base buffer removal and insert"):
        session.commit()


def test_refresh_failure_does_not_repeat_the_completed_native_commit(monkeypatch):
    raw_db = _raw_db()
    calls = []

    def apply(raw, actions, digest):
        calls.append(list(actions))
        return {"status": "accepted", "accepted_count": len(actions)}

    def failed_refresh(raw):
        raise ValueError("unmapped timing identity")

    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(apply))
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier",
        failed_refresh,
    )
    session = PlaceIOBufferSession(raw_db)
    session.begin()
    session.apply([_action("net0", 100, 200)])

    first = session.commit()
    second = session.commit()

    assert first["status"] == "refresh_failed"
    assert first["native_db_mutated"] is True
    assert second["status"] == "already_committed"
    assert second["commit_summary"] == first
    assert len(calls) == 1
