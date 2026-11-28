"""Graph-owned paths retain the Python TP extractor's selection and transitions."""

import numpy as np
import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.propagation_payload import persistent_propagation_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_buffers import _reference, _split
from tests.ops.rebuffer.test_persistent_timing_propagation import _checks_base
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _predecessors(base):
    """Keep modified synthetic fixtures' CSR consistent with their cell/net rows."""
    incoming = [[] for _ in base.pin_names]
    clock_end = base.flat_inst_arcs_by_level_start[1]
    for arc, row in enumerate(base.flat_inst_arcs_by_level):
        if arc >= clock_end:
            incoming[row[1]].append((row[0], arc))
    for source, sink in base.net_flat_arcs:
        incoming[sink].append((source, -1))
    base.pin_pred_start = [0]
    base.pin_pred_pin, base.pin_pred_arc_id = [], []
    for row in incoming:
        base.pin_pred_pin.extend(pin for pin, _ in row)
        base.pin_pred_arc_id.extend(arc for _, arc in row)
        base.pin_pred_start.append(len(base.pin_pred_pin))
    return base


def _make(base):
    owner = RebufferTimingModel(_predecessors(base), use_pin_offset=False)
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    graph.initialize_propagation(persistent_propagation_payload(owner))
    graph.propagate()
    return owner, graph


def _expected(owner, **kwargs):
    assert owner.model.capture_critical_path_snapshot_from_state()
    batch = owner.model.extract_setup_critical_paths(**kwargs)
    paths, cursor = {}, 0
    for index, endpoint in enumerate(batch.endpoint_pins.tolist()):
        begin, end = batch.path_offsets[index : index + 2].tolist()
        pins = batch.path_pins[begin:end].tolist()
        transitions = batch.path_transitions[begin:end].tolist()
        arcs = batch.path_arc_ids[cursor : cursor + end - begin - 1].tolist()
        cursor += end - begin - 1
        if endpoint in paths:
            continue
        paths[endpoint] = {
            "path": pins,
            "transitions": transitions,
            "arc_ids": arcs,
            "endpoint_transition": int(batch.endpoint_transitions[index]),
            "endpoint_slack": float(batch.endpoint_slacks[index]),
            "valid": bool(batch.path_valid[index]),
            "endpoint_test_id": int(batch.endpoint_test_ids[index]),
            "invalid_reason": int(batch.invalid_reason[index]),
            "max_residual_ps": float(batch.max_residual_ps[index]),
            "path_arrivals": [
                float(
                    (
                        owner.model.pin_rAAT_live_snapshot
                        if transition == 0
                        else owner.model.pin_fAAT_live_snapshot
                    )[pin]
                )
                for pin, transition in zip(pins, transitions, strict=True)
            ],
        }
    counts = {
        name: int(getattr(batch, name))
        for name in (
            "failing_state_count",
            "selected_state_count",
            "valid_path_count",
            "invalid_path_count",
        )
    }
    return paths, counts


def _assert_paths(graph, owner, **kwargs):
    expected, counts = _expected(owner, **kwargs)
    result = graph.setup_path_details(**kwargs)
    actual = {pin: dict(record) for pin, record in result["paths"].items()}
    for record in actual.values():
        assert record.pop("topology_epoch") == graph.summary()["epoch"]
    numeric = ("endpoint_slack", "max_residual_ps", "path_arrivals")
    assert {
        pin: {key: value for key, value in record.items() if key not in numeric}
        for pin, record in actual.items()
    } == {
        pin: {key: value for key, value in record.items() if key not in numeric}
        for pin, record in expected.items()
    }
    for pin, record in actual.items():
        for field in numeric:
            np.testing.assert_allclose(record[field], expected[pin][field], rtol=0, atol=1e-6)
    assert {name: result["statistics"][name] for name in counts} == counts
    assert result["statistics"]["timing_state_values_copied"] == 0
    return result


@pytest.mark.parametrize("sense", [-1, 0, 1])
@pytest.mark.parametrize("hard_pairwise", [False, True])
def test_graph_paths_use_shared_transition_aware_backtrace(monkeypatch, sense, hard_pairwise):
    monkeypatch.setenv("AIMP_FORCE_HARD_PAIRWISE_MAX", str(int(hard_pairwise)))
    base = _base()
    base.flat_inst_arcs_by_level[0][4] = sense
    owner, graph = _make(base)

    result = _assert_paths(graph, owner)

    assert result["paths"][3]["path"] == [0, 1, 2, 3]
    assert result["paths"][3]["valid"]


@pytest.mark.parametrize("global_k", [0, 1, 2, 4])
@pytest.mark.parametrize("qualified", [True, False])
def test_per_constraint_deadlines_test_ids_and_qualification(global_k, qualified):
    base = _checks_base()
    base.endpoints_constraint_max_valid = np.array([[True, False], [False, qualified]])
    base.endpoints_constraint_max_reason = (~base.endpoints_constraint_max_valid).astype(np.int32)
    base.endpoints_max_valid = np.array([[True, qualified]])
    base.endpoints_max_reason = (~base.endpoints_max_valid).astype(np.int32)
    base.endpoints_timing_check_max_valid[:, 1] &= qualified
    base.endpoints_timing_check_max_reason = (~base.endpoints_timing_check_max_valid).astype(
        np.int32
    )
    owner, graph = _make(base)

    result = _assert_paths(graph, owner, global_k=global_k)

    assert result["statistics"]["endpoint_record_count"] == 1 + int(qualified)
    assert result["paths"][3]["endpoint_test_id"] in (0, 1)
    if not qualified:
        assert result["paths"][3]["endpoint_transition"] == 0


@pytest.mark.parametrize("with_setup", [False, True])
def test_recovery_only_pin_uses_current_tp_fallback_rule(with_setup):
    base = _checks_base()
    base.end_points = base.endpoint_pin_ids = [1, 3]
    base.endpoints_rRAT = base.endpoints_fRAT = [-1000.0, 0.0]
    base.endpoints_max_valid = np.ones((2, 2), dtype=bool)
    base.endpoints_max_reason = np.zeros((2, 2), dtype=np.int32)
    base.outcaps = [0.0, 0.0]
    base.endpoints_timing_check_arcs[:, 1] = 1
    base.endpoints_timing_check_max_valid[:] = False
    base.endpoints_timing_check_max_reason[:] = 1
    if with_setup:
        base.endpoints_timing_check_arcs = np.concatenate(
            (base.endpoints_timing_check_arcs, [[1, 3, 0, 0, 1, 0, 1, 0]])
        )
        base.endpoints_timing_check_max_valid = np.concatenate(
            (base.endpoints_timing_check_max_valid, [[True, True]])
        )
        base.endpoints_timing_check_max_reason = (~base.endpoints_timing_check_max_valid).astype(
            np.int32
        )
    else:
        base.endpoints_constraint_arcs = np.empty((0, 8), dtype=np.int64)
        base.endpoints_constraint_max_valid = np.empty((0, 2), dtype=bool)
        base.endpoints_constraint_max_reason = np.empty((0, 2), dtype=np.int32)
    owner, graph = _make(base)

    result = _assert_paths(graph, owner)

    assert (1 in result["paths"]) != with_setup
    assert (result["paths"][3]["endpoint_test_id"] >= 0) == with_setup


@pytest.mark.parametrize("async_launch", [False, True])
def test_clock_local_arcs_are_excluded_and_async_launch_can_continue(async_launch):
    base = _checks_base()
    base.start_points = [0, 2]
    base.inrdelays = [100.0 if async_launch else -1000.0, 0.0]
    base.infdelays = list(base.inrdelays)
    base.inrtrans, base.inftrans = [15.0, 15.0], [20.0, 20.0]
    owner, graph = _make(base)

    result = _assert_paths(graph, owner)

    record = result["paths"][3]
    assert record["valid"]
    assert 0 not in record["arc_ids"]  # local clock id 0 maps to physical pin 1
    assert record["path"] == ([0, 1, 2, 3] if async_launch else [2, 3])


def test_duplicate_pin_pair_arcs_keep_deterministic_tie_order():
    base = _base()
    base.flat_inst_arcs_by_level.append([1, 2, 0, 0, 0, 0, 1, 0])
    base.flat_inst_arcs_by_level_start[-1] = 2
    owner, graph = _make(base)

    first = _assert_paths(graph, owner)
    second = _assert_paths(graph, owner)

    assert first["paths"] == second["paths"]


def test_relevelization_changes_flat_arc_tie_order_and_restore_recovers_it():
    base = _base()
    # Two equal predecessors with different source depths after a split.
    base.flat_inst_arcs_by_level.append([0, 2, 0, 0, 1, 0, 1, 0])
    base.flat_inst_arcs_by_level_start[-1] = 2
    for prefix in ("r_delay", "f_delay"):
        setattr(base, f"{prefix}_flat_luts_values", [[0.0] * 4] * 2)
    owner = RebufferTimingModel(_predecessors(base), r_unit=1e-100, c_unit=0.0)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    graph.initialize_propagation(persistent_propagation_payload(owner))
    graph.initialize_mutations()
    topology = rebuffer_cpp.incremental_topology_from_steiner(
        steiner_payload(base, owner), 0, 0, owner.r_unit, owner.c_unit
    )
    graph.attach_net_topology_state(0, topology)
    graph.propagate()
    checkpoint = graph.checkpoint()
    original = graph.setup_path_details()["paths"]
    assert original[3]["path"] == [0, 1, 2, 3]

    _split(graph, 0, 1, 9)
    graph.propagate_incremental()

    assert graph.mutation_index_oracle()["status"] == "ok"
    reordered = graph.setup_path_details()["paths"][3]
    assert reordered["path"] == [0, 2, 3]
    assert reordered["arc_ids"] == [1, -1]
    graph.restore(checkpoint)
    assert graph.mutation_index_oracle()["status"] == "ok"
    assert graph.setup_path_details()["paths"] == original


@pytest.mark.parametrize("max_depth", [1, 2, 3, 10])
def test_depth_invalidity_matches_existing_extractor(max_depth):
    owner, graph = _make(_base())

    _assert_paths(graph, owner, max_depth=max_depth)


def test_filter_empty_duplicates_and_nonfailing_endpoint():
    owner, graph = _make(_base())
    for requested in ([], [3], [3, 3], [0, 3], [0]):
        _assert_paths(graph, owner, endpoint_pins=requested)


def test_stale_state_is_rejected_and_rollback_forks_do_not_reuse_paths():
    owner, graph = _make(_base())
    checkpoint = graph.checkpoint()
    first = graph.setup_path_details()
    rc = graph.state_view()["pin_net_delay_rise"]
    rc[3] += 100.0
    graph.set_state({"pin_net_delay_rise": rc})
    with pytest.raises(ValueError, match="current native propagation"):
        graph.setup_path_details()
    graph.propagate_incremental()
    changed = graph.setup_path_details()
    assert changed["paths"] != first["paths"]
    graph.restore(checkpoint)
    assert graph.setup_path_details()["paths"] == first["paths"]
    rc[3] += 200.0
    graph.set_state({"pin_net_delay_rise": rc})
    graph.propagate_incremental()
    fork = graph.setup_path_details()
    assert fork["statistics"]["topology_epoch"] == changed["statistics"]["topology_epoch"]
    assert fork["paths"] != changed["paths"]


def test_readonly_path_extraction_does_not_export_or_detach_state_views(monkeypatch):
    owner, graph = _make(_base())
    snapshot = graph.state_view(copy=False)
    before = graph.state_view_statistics(), graph.summary()

    def forbidden(*args, **kwargs):
        pytest.fail("native graph path extraction called the Python timing owner")

    monkeypatch.setattr(owner, "native_setup_path_details", forbidden)
    monkeypatch.setattr(owner.model, "capture_critical_path_snapshot_from_state", forbidden)
    result = graph.setup_path_details()

    assert result["paths"]
    assert (graph.state_view_statistics(), graph.summary()) == before
    np.testing.assert_array_equal(snapshot["pin_rAAT"], owner.model.pin_rAAT_live_snapshot)


def test_insert_remove_resize_restore_and_compact_paths_match_fresh_python_tp():
    owner, graph = _graph(_base())
    graph.propagate()
    original = graph.setup_path_details()["paths"]
    actions = []
    for action_id in (7, 3):
        action, _ = _split(graph, 1, 3, action_id)
        actions.append(action)
        graph.propagate_incremental()
    reference = _reference(owner, graph, actions)
    native = graph.setup_path_details()["paths"][3]
    endpoint = reference.pydb.pin_names.index(owner.pydb.pin_names[3])
    expected = reference.native_setup_path_details()[endpoint]
    names = graph.mutation_view()["pin_names"]
    assert [names[pin] for pin in native["path"]] == [
        reference.pydb.pin_names[pin] for pin in expected["path"]
    ]
    assert native["transitions"] == expected["transitions"]
    np.testing.assert_allclose(
        native["path_arrivals"], expected["path_arrivals"], rtol=0, atol=1e-6
    )
    graph.resize_buffer(1, {"action_id": 7, "buffer_master_name": "BUF2"})
    graph.remove_buffer(1, 3)
    graph.propagate_incremental()
    graph.commit(graph.checkpoint())
    before = graph.setup_path_details()["paths"]
    compact = graph.compact()
    after = graph.setup_path_details()["paths"]
    mapping = compact["pin_old_to_new"]
    assert after[3]["path"] == [int(mapping[pin]) for pin in before[3]["path"]]
    assert after[3]["path_arrivals"] == before[3]["path_arrivals"]
    # A separate graph retains a root checkpoint so a different mutation fork
    # can be compared at the same native epoch without an epoch-only cache.
    _, rollback = _graph(_base())
    rollback.propagate()
    checkpoint = rollback.checkpoint()
    _split(rollback, 1, 3, 8)
    rollback.propagate_incremental()
    rollback.restore(checkpoint)
    assert rollback.setup_path_details()["paths"] == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_k": -1},
        {"residual_tolerance_ps": 0.0},
        {"residual_tolerance_ps": float("nan")},
        {"endpoint_pins": [-1]},
        {"endpoint_pins": [4]},
    ],
)
def test_invalid_request_preserves_graph_state(kwargs):
    _, graph = _make(_base())
    before = graph.summary()
    with pytest.raises(ValueError):
        graph.setup_path_details(**kwargs)
    assert graph.summary() == before


def test_imported_state_needs_native_baseline_before_paths():
    owner = RebufferTimingModel(_base(), use_pin_offset=False)
    graph = rebuffer_cpp.create_persistent_timing_graph(owner.pydb)
    graph.initialize_timing_state(persistent_timing_state_payload(owner))
    graph.initialize_propagation(persistent_propagation_payload(owner))
    with pytest.raises(ValueError, match="current native propagation"):
        graph.setup_path_details()
    graph.propagate()
    _assert_paths(graph, owner)
