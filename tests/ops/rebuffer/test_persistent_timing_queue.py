"""Graph state must feed the existing native queue policy without Python snapshots."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _split
from tests.ops.rebuffer.test_persistent_timing_paths import _predecessors
from tests.ops.rebuffer.test_persistent_timing_propagation import _checks_base
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _reference_queue(owner, original_net_by_net=None):
    context = RebufferPlanContext(owner.pydb, owner)
    assert context.status == "ok", context.reasons
    views = context.native_queue_views()
    details = owner.native_setup_path_details() or {}
    records = []
    for endpoint in owner.model.end_points.tolist():
        slack = float(views["timing"]["pin_slack"][endpoint])
        if slack >= 0:
            continue
        path = details.get(endpoint, {}).get("path")
        if not path:
            paths = owner.model.get_critical_paths([endpoint], K=1)
            path = paths[0] if paths else []
        if path:
            records.append(
                {
                    "endpoint_pin": endpoint,
                    "endpoint_pin_name": str(owner.pydb.pin_names[endpoint]),
                    "slack_ps": slack,
                    "path_pins": path,
                    "path_timing": context.native_path_context(path, path_details=details),
                }
            )
    mapping = views["original_net_by_net"] if original_net_by_net is None else original_net_by_net
    snapshot = rebuffer_cpp.build_native_timing_queue_state(
        views["design"], views["timing"], views["liberty"], records, mapping
    )
    return snapshot, views


def _initialize(owner, graph):
    _, views = _reference_queue(owner)
    graph.initialize_queue(views["liberty"], views["design"]["node_is_buffer"].tolist())
    return views


def _normalize(snapshot, pin_names, net_names, original_names, config=None):
    records = rebuffer_cpp.select_timing_queue_items(snapshot, config or {})
    normalized = []
    for item in records:
        record = dict(item)
        record["endpoint_pin"] = pin_names[item["endpoint_pin"]]
        record["path_pins"] = [pin_names[pin] for pin in item["path_pins"]]
        record["driver_pin_id"] = pin_names[item["driver_pin_id"]]
        record["root_net_id"] = net_names[item["root_net_id"]]
        record["net_ids"] = [original_names[net] for net in item["net_ids"]]
        record["tree_key"] = (record["root_net_id"], record["driver_pin_id"])
        normalized.append(record)
    return normalized


def _assert_queue(graph, reference, base, original_net_by_net=None, config=None):
    expected, _ = _reference_queue(reference, original_net_by_net)
    actual = graph.timing_queue_state(list(range(len(base.net_names))))
    metadata = graph.mutation_view()
    observed = _normalize(
        actual, metadata["pin_names"], metadata["net_names"], base.net_names, config
    )
    desired = _normalize(
        expected, reference.pydb.pin_names, reference.pydb.net_names, base.net_names, config
    )
    assert [{key: val for key, val in row.items() if key != "slack_ps"} for row in observed] == [
        {key: val for key, val in row.items() if key != "slack_ps"} for row in desired
    ]
    np.testing.assert_allclose(
        [row["slack_ps"] for row in observed],
        [row["slack_ps"] for row in desired],
        rtol=0,
        atol=1e-6,
    )
    assert actual["snapshot_statistics"] == expected["snapshot_statistics"]
    return actual


@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("with_clock", [False, True])
def test_queue_matches_existing_python_records_and_native_policy(sense, with_clock):
    base = _checks_base() if with_clock else _base()
    base.flat_inst_arcs_by_level[-1][4] = sense
    owner, graph = _graph(_predecessors(base))
    graph.propagate()
    _initialize(owner, graph)

    _assert_queue(graph, owner, base)
    _assert_queue(
        graph,
        owner,
        base,
        config={"native_tree_round_robin": True, "native_tree_round_robin_endpoint_window": 1},
    )


def test_recovery_only_endpoint_uses_shared_legacy_path_fallback():
    base = _checks_base()
    base.end_points = base.endpoint_pin_ids = [1, 3]
    base.outcaps = [0.0, 0.0]
    base.endpoints_rRAT = base.endpoints_fRAT = [-1000.0, 0.0]
    base.endpoints_max_valid = np.ones((2, 2), dtype=bool)
    base.endpoints_max_reason = np.zeros((2, 2), dtype=np.int32)
    base.endpoints_timing_check_arcs[:, 1] = 1
    owner, graph = _graph(_predecessors(base))
    graph.propagate()
    _initialize(owner, graph)

    result = _assert_queue(graph, owner, base)

    assert result["graph_queue_statistics"]["legacy_fallback_count"] == 1


def _current_reference(owner, graph, actions):
    exported = graph.export_carrier(
        steiner_payload(owner.pydb, owner), actions, owner.r_unit, owner.c_unit
    )
    pydb = CommittedTimingPyDB(owner.pydb, SimpleNamespace(**exported["physical"]))
    reference = RebufferTimingModel(pydb, use_pin_offset=False, steiner_view=exported["steiner"])
    return reference, exported["original_net_by_net"]


def test_queue_survives_mutation_restore_compact_and_preserves_old_snapshot():
    owner, graph = _graph(_base())
    graph.propagate()
    _initialize(owner, graph)
    checkpoint = graph.checkpoint()
    original = _assert_queue(graph, owner, owner.pydb)
    original_records = rebuffer_cpp.select_timing_queue_items(original, {})
    actions = []
    for action_id in (7, 3):
        action, _ = _split(graph, 1, 3, action_id)
        actions.append(action)
        graph.propagate_incremental()
        reference, net_map = _current_reference(owner, graph, actions)
        _assert_queue(graph, reference, owner.pydb, net_map)
    assert rebuffer_cpp.select_timing_queue_items(original, {}) == original_records
    graph.restore(checkpoint)
    _assert_queue(graph, owner, owner.pydb)
    for action_id in (9, 2):
        _split(graph, 1, 3, action_id)
    graph.remove_buffer(1, 9)
    graph.propagate_incremental()
    before = graph.timing_queue_state([0, 1])
    before_records = rebuffer_cpp.select_timing_queue_items(before, {})
    graph.commit(graph.checkpoint())
    mapping = graph.compact()
    after = graph.timing_queue_state([0, 1])
    after_records = rebuffer_cpp.select_timing_queue_items(after, {})
    assert [r["slack_ps"] for r in after_records] == [r["slack_ps"] for r in before_records]
    for old, new in zip(before_records, after_records, strict=True):
        assert new["path_pins"] == [int(mapping["pin_old_to_new"][pin]) for pin in old["path_pins"]]


def test_queue_reads_native_storage_without_python_timing_or_array_exports(monkeypatch):
    owner, graph = _graph(_base())
    graph.propagate()
    _initialize(owner, graph)
    retained = graph.state_view(copy=False)
    before = graph.state_view_statistics(), graph.summary()

    def forbidden(*args, **kwargs):
        pytest.fail("graph queue called Python timing or path extraction")

    monkeypatch.setattr(owner, "native_setup_path_details", forbidden)
    monkeypatch.setattr(owner.model, "get_critical_paths", forbidden)
    result = graph.timing_queue_state([0, 1])

    assert rebuffer_cpp.select_timing_queue_items(result, {})
    assert (graph.state_view_statistics(), graph.summary()) == before
    np.testing.assert_array_equal(retained["pin_rAAT"], owner.model.pin_rAAT_live_snapshot)
    _split(graph, 1, 3, 7)
    with pytest.raises(ValueError, match="current native propagation"):
        graph.timing_queue_state([0, 1])


def test_native_queue_scheduler_accepts_graph_owned_state():
    owner, graph = _graph(_base())
    graph.propagate()
    _initialize(owner, graph)
    reference, _ = _reference_queue(owner)
    native = graph.timing_queue_state([0, 1])
    requests = []
    for snapshot in (reference, native):
        current = []

        def provider(request, records=current):
            records.append(dict(request))
            return {"status": "no_actions"}

        result = rebuffer_cpp.run_repair_timing_queue_operator(
            lambda value=snapshot: value,
            provider,
            lambda label: None,
            lambda actions: None,
            lambda: {"status": "ok", **owner.baseline.as_dict()},
            lambda token: None,
            {"max_passes": 3, "native_tree_queue": True},
        )
        assert result["stop_reason"] == "queue_exhausted"
        requests.append(current)
    assert requests[0] == requests[1]


def test_queue_initialization_owns_inputs_and_rejects_mismatched_metadata():
    owner, graph = _graph(_base())
    graph.propagate()
    _, views = _reference_queue(owner)
    flags = views["design"]["node_is_buffer"].tolist()
    invalid = deepcopy(views["liberty"])
    invalid["names"][0] = "wrong"
    before = graph.summary()
    with pytest.raises(ValueError, match="identity"):
        graph.initialize_queue(invalid, flags)
    with pytest.raises(ValueError, match="flags"):
        graph.initialize_queue(views["liberty"], [])
    wrong_units = {**views["liberty"], "dbu": views["liberty"]["dbu"] + 1}
    with pytest.raises(ValueError, match="units"):
        graph.initialize_queue(wrong_units, flags)
    assert graph.summary() == before
    graph.initialize_queue(views["liberty"], flags)
    original = rebuffer_cpp.select_timing_queue_items(graph.timing_queue_state([0, 1]), {})
    views["liberty"]["names"][0] = "changed_after_import"
    flags[:] = [False] * len(flags)
    assert rebuffer_cpp.select_timing_queue_items(graph.timing_queue_state([0, 1]), {}) == original
    with pytest.raises(ValueError, match="immutable"):
        graph.initialize_queue(views["liberty"], flags)
    with pytest.raises(ValueError, match="net identities"):
        graph.timing_queue_state([0])
