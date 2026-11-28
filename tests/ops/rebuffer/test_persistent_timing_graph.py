from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from dreamplace.ops.placeio_ecc.persistent_timing import _compact_current_pydb
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base


TIMING_STATE_FIELDS = (
    "pin_rAAT",
    "pin_fAAT",
    "pin_rRAT",
    "pin_fRAT",
    "pin_rtran",
    "pin_ftran",
    "cell_arc_rr_delays",
    "cell_arc_fr_delays",
    "cell_arc_rf_delays",
    "cell_arc_ff_delays",
)


def _assert_fresh_timing_parity(current, reference):
    assert current.baseline.as_dict() == pytest.approx(
        reference.baseline.as_dict(), abs=1e-8, rel=0
    )
    for field in TIMING_STATE_FIELDS:
        torch.testing.assert_close(
            getattr(current.model, field),
            getattr(reference.model, field),
            rtol=0,
            atol=1e-8,
        )
    torch.testing.assert_close(
        current.model.get_pin_slack(), reference.model.get_pin_slack(), rtol=0, atol=1e-8
    )


def test_persistent_timing_graph_owns_index_and_journaled_state():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())

    initial = graph.summary()
    assert initial["status"] == "ok"
    assert initial["pin_count"] == 4
    assert initial["cell_arc_count"] == 1
    assert initial["net_arc_count"] == 2
    assert initial["journal_size"] == 0

    graph.set_state({"pin_rAAT": np.array([0.0, 1.0, 2.0, 3.0])})
    checkpoint = graph.checkpoint()
    graph.set_state(
        {
            "pin_rAAT": np.array([10.0, 11.0, 12.0, 13.0]),
            "pin_rRAT": np.array([20.0, 21.0, 22.0, 23.0]),
        }
    )

    restored = graph.restore(checkpoint)
    assert restored["journal_size"] == 1
    view = graph.state_view()
    np.testing.assert_array_equal(view["pin_rAAT"], [0.0, 1.0, 2.0, 3.0])
    assert "pin_rRAT" not in view

    committed = graph.commit(checkpoint)
    assert committed["journal_size"] == 0
    assert committed["state_fields"] == ["pin_rAAT"]


def test_persistent_timing_graph_journals_single_net_split_and_rc_state():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())
    graph.attach_net_topology(
        7,
        0,
        4,
        {
            "edges": [
                {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
                {"parent": 1, "child": 2, "r": 5.0, "c": 1.0},
                {"parent": 1, "child": 3, "r": 6.0, "c": 1.0},
            ],
            "node_loads": {2: [20], 3: [30]},
        },
    )
    checkpoint = graph.checkpoint()

    split = graph.apply_buffer_split(
        7,
        {
            "action_id": 101,
            "parent_node_id": 1,
            "child_node_id": 2,
            "split_ratio": 0.4,
            "downstream_pin_ids": [20],
        },
    )
    assert split["input_node"] == 4
    assert split["output_node"] == 5

    rc = graph.build_net_rc(7)
    assert rc["affected_nodes"] == [0, 1, 2, 3, 4, 5]
    assert rc["parent"] == [0, 0, 5, 1, 1, 5]
    rc_result = graph.evaluate_net_rc(
        7,
        rc,
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        {"slots": [{"input": [0.1, 0.1, 0.1], "output": [0.2, 0.2, 0.2]}]},
        4,
    )
    state = graph.state_view()
    np.testing.assert_array_equal(
        state["net_7_loads_generic"], rc_result["loads_generic"]
    )
    assert graph.summary()["net_count"] == 1
    assert graph.topology_view(7)["buffers"][0]["action_id"] == 101

    restored = graph.restore(checkpoint)
    assert restored["journal_size"] == 0
    assert graph.topology_view(7)["buffers"] == []

    graph.apply_buffer_split(
        7,
        {
            "action_id": 101,
            "parent_node_id": 1,
            "child_node_id": 2,
            "split_ratio": 0.4,
            "downstream_pin_ids": [20],
        },
    )
    assert graph.remove_buffer(7, 101)["action_id"] == 101
    assert graph.topology_view(7)["buffers"] == []


def test_persistent_timing_graph_can_clear_shared_topology_bindings():
    graph = rebuffer_cpp.create_persistent_timing_graph(_base())
    graph.attach_net_topology(
        7,
        0,
        4,
        {
            "edges": [
                {"parent": 0, "child": 1, "r": 10.0, "c": 2.0},
                {"parent": 1, "child": 2, "r": 5.0, "c": 1.0},
            ],
            "node_loads": {2: [20]},
        },
    )
    assert graph.summary()["net_count"] == 1

    cleared = graph.clear_net_topologies()

    assert cleared == {"status": "ok", "removed_count": 1, "net_count": 0}
    assert graph.summary()["net_count"] == 0
    assert list(graph.state_view()) == ["status", "epoch"]


def test_persistent_timing_graph_patch_matches_fresh_python_timing():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)

    initial_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial_state)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )

    action = {
        "session_buffer_id": 0,
        "buffer_master_name": "BUF2",
        "candidate_location_x_dbu": 50000.0,
        "candidate_location_y_dbu": 0.0,
        "net_name": "n1",
        "downstream_pin_names": ["OUT"],
    }
    split = {
        "action_id": 0,
        "parent_node_id": 2,
        "child_node_id": 3,
        "split_ratio": 4.0 / 9.0,
        "downstream_pin_ids": [3],
        "x_dbu": 50000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF2",
    }
    graph.apply_buffer_split(1, split)

    reference_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    rebuffer_cpp.apply_incremental_buffer_split(reference_state, split)
    action_payload = {**action, "downstream_pin_ids": [3]}
    graph_patch = graph.build_timing_patch(
        1, action_payload, int(baseline.new_x.numel()), 0
    )
    reference_patch = rebuffer_cpp.build_incremental_timing_patch(
        base, reference_state, action_payload, int(baseline.new_x.numel()), 0
    )
    for field in (
        "parent",
        "child_start",
        "child",
        "topo_start",
        "topo",
        "edge_resistance",
        "edge_capacitance",
        "affected_nodes",
    ):
        np.testing.assert_array_equal(
            np.asarray(graph_patch["rc"][field]),
            np.asarray(reference_patch["rc"][field]),
        )
    assert graph_patch["graph"]["num_pins"] == reference_patch["graph"]["num_pins"]

    physical_payload = rebuffer_cpp.export_overlay_carrier(
        base, base_steiner, [reference_state], [action]
    )
    reference = RebufferTimingModel(
        CommittedTimingPyDB(base, SimpleNamespace(**physical_payload["physical"])),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    compact_pydb = _compact_current_pydb(
        base, graph_patch["graph"], [action], 0
    )
    current = RebufferTimingModel(
        compact_pydb,
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )

    assert current.baseline.as_dict() == pytest.approx(
        reference.baseline.as_dict(), abs=1e-8, rel=0
    )
    for field in (
        "pin_rAAT",
        "pin_fAAT",
        "pin_rRAT",
        "pin_fRAT",
        "pin_rtran",
        "pin_ftran",
        "cell_arc_rr_delays",
        "cell_arc_fr_delays",
        "cell_arc_rf_delays",
        "cell_arc_ff_delays",
    ):
        torch.testing.assert_close(
            getattr(current.model, field),
            getattr(reference.model, field),
            rtol=0,
            atol=1e-8,
        )
    torch.testing.assert_close(
        current.model.get_pin_slack(), reference.model.get_pin_slack(), rtol=0, atol=1e-8
    )


def test_persistent_timing_graph_repeated_insert_restore_matches_fresh_python_timing():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)
    initial_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial_state)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )
    rng = random.Random(8137)
    state_fields = (
        "pin_rAAT",
        "pin_fAAT",
        "pin_rRAT",
        "pin_fRAT",
        "pin_rtran",
        "pin_ftran",
        "cell_arc_rr_delays",
        "cell_arc_fr_delays",
        "cell_arc_rf_delays",
        "cell_arc_ff_delays",
    )

    for attempt in range(100):
        ratio = rng.uniform(0.1, 0.9)
        master = "BUF1" if attempt % 2 else "BUF2"
        x_dbu = 10000.0 + ratio * 90000.0
        action = {
            "session_buffer_id": attempt,
            "buffer_master_name": master,
            "candidate_location_x_dbu": x_dbu,
            "candidate_location_y_dbu": 0.0,
            "net_name": "n1",
            "downstream_pin_names": ["OUT"],
        }
        split = {
            "action_id": attempt,
            "parent_node_id": 2,
            "child_node_id": 3,
            "split_ratio": ratio,
            "downstream_pin_ids": [3],
            "x_dbu": x_dbu,
            "y_dbu": 0.0,
            "buffer_master_name": master,
        }
        checkpoint = graph.checkpoint()
        graph.apply_buffer_split(1, split)

        reference_state = rebuffer_cpp.incremental_topology_from_steiner(
            base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
        )
        rebuffer_cpp.apply_incremental_buffer_split(reference_state, split)
        action_payload = {**action, "downstream_pin_ids": [3]}
        graph_patch = graph.build_timing_patch(
            1, action_payload, int(baseline.new_x.numel()), 0
        )
        reference_patch = rebuffer_cpp.build_incremental_timing_patch(
            base, reference_state, action_payload, int(baseline.new_x.numel()), 0
        )
        np.testing.assert_array_equal(
            np.asarray(graph_patch["rc"]["affected_nodes"]),
            np.asarray(reference_patch["rc"]["affected_nodes"]),
        )
        np.testing.assert_allclose(
            np.asarray(graph_patch["rc"]["edge_resistance"]),
            np.asarray(reference_patch["rc"]["edge_resistance"]),
            rtol=0,
            atol=1e-12,
        )

        physical_payload = rebuffer_cpp.export_overlay_carrier(
            base, base_steiner, [reference_state], [action]
        )
        reference = RebufferTimingModel(
            CommittedTimingPyDB(
                base, SimpleNamespace(**physical_payload["physical"])
            ),
            use_pin_offset=False,
            steiner_view=physical_payload["steiner"],
        )
        compact_pydb = _compact_current_pydb(
            base, graph_patch["graph"], [action], 0
        )
        current = RebufferTimingModel(
            compact_pydb,
            use_pin_offset=False,
            steiner_view=physical_payload["steiner"],
        )
        assert current.baseline.as_dict() == pytest.approx(
            reference.baseline.as_dict(), abs=1e-8, rel=0
        )
        for field in state_fields:
            torch.testing.assert_close(
                getattr(current.model, field),
                getattr(reference.model, field),
                rtol=0,
                atol=1e-8,
            )
        torch.testing.assert_close(
            current.model.get_pin_slack(),
            reference.model.get_pin_slack(),
            rtol=0,
            atol=1e-8,
        )

        graph.restore(checkpoint)
        assert graph.topology_view(1)["buffers"] == []
        assert list(graph.state_view()) == ["status", "epoch"]


def test_persistent_timing_graph_two_buffer_chain_and_lifo_remove_match_fresh_timing():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)
    initial_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial_state)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )
    split_first = {
        "action_id": 0,
        "parent_node_id": 2,
        "child_node_id": 3,
        "split_ratio": 4.0 / 9.0,
        "downstream_pin_ids": [3],
        "x_dbu": 50000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF2",
    }
    split_second = {
        "action_id": 1,
        "parent_node_id": 5,
        "child_node_id": 3,
        "split_ratio": 0.5,
        "downstream_pin_ids": [3],
        "x_dbu": 75000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF1",
    }
    action_first = {
        "session_buffer_id": 0,
        "buffer_master_name": "BUF2",
        "candidate_location_x_dbu": 50000.0,
        "candidate_location_y_dbu": 0.0,
        "net_name": "n1",
        "downstream_pin_names": ["OUT"],
    }
    action_second = {
        "session_buffer_id": 1,
        "buffer_master_name": "BUF1",
        "candidate_location_x_dbu": 75000.0,
        "candidate_location_y_dbu": 0.0,
        "net_name": "n1",
        "downstream_pin_names": ["OUT"],
    }

    graph.apply_buffer_split(1, split_first)
    graph.apply_buffer_split(1, split_second)
    reference_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    rebuffer_cpp.apply_incremental_buffer_split(reference_state, split_first)
    rebuffer_cpp.apply_incremental_buffer_split(reference_state, split_second)
    second_payload = {**action_second, "downstream_pin_ids": [3]}
    graph_patch = graph.build_timing_patch(
        1, second_payload, int(baseline.new_x.numel()), 0
    )
    reference_patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        reference_state,
        second_payload,
        int(baseline.new_x.numel()),
        0,
    )
    assert graph_patch["buffer_count"] == 2
    assert graph_patch["graph"]["num_pins"] == len(base.pin_names) + 4
    for field in (
        "parent",
        "child_start",
        "child",
        "topo_start",
        "topo",
        "edge_resistance",
        "edge_capacitance",
        "affected_nodes",
    ):
        np.testing.assert_array_equal(
            np.asarray(graph_patch["rc"][field]),
            np.asarray(reference_patch["rc"][field]),
        )

    physical_payload = rebuffer_cpp.export_overlay_carrier(
        base,
        base_steiner,
        [reference_state],
        [action_first, action_second],
    )
    reference = RebufferTimingModel(
        CommittedTimingPyDB(
            base, SimpleNamespace(**physical_payload["physical"])
        ),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    current = RebufferTimingModel(
        _compact_current_pydb(
            base, graph_patch["graph"], [action_first, action_second], 0
        ),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    _assert_fresh_timing_parity(current, reference)

    assert graph.remove_buffer(1, 1)["action_id"] == 1
    assert rebuffer_cpp.remove_incremental_buffer(reference_state, 1)["action_id"] == 1
    first_payload = {**action_first, "downstream_pin_ids": [3]}
    graph_patch = graph.build_timing_patch(
        1, first_payload, int(baseline.new_x.numel()), 0
    )
    reference_patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        reference_state,
        first_payload,
        int(baseline.new_x.numel()),
        0,
    )
    np.testing.assert_array_equal(
        np.asarray(graph_patch["rc"]["affected_nodes"]),
        np.asarray(reference_patch["rc"]["affected_nodes"]),
    )
    physical_payload = rebuffer_cpp.export_overlay_carrier(
        base, base_steiner, [reference_state], [action_first]
    )
    reference = RebufferTimingModel(
        CommittedTimingPyDB(
            base, SimpleNamespace(**physical_payload["physical"])
        ),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    current = RebufferTimingModel(
        _compact_current_pydb(base, graph_patch["graph"], [action_first], 0),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    _assert_fresh_timing_parity(current, reference)
    assert graph.remove_buffer(1, 0)["action_id"] == 0
    assert rebuffer_cpp.remove_incremental_buffer(reference_state, 0)["action_id"] == 0
    assert graph.topology_view(1)["buffers"] == []


def test_persistent_timing_graph_resize_updates_patch_and_restores_master():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)
    initial_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial_state)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )
    split = {
        "action_id": 0,
        "parent_node_id": 2,
        "child_node_id": 3,
        "split_ratio": 4.0 / 9.0,
        "downstream_pin_ids": [3],
        "x_dbu": 50000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF2",
    }
    graph.apply_buffer_split(1, split)
    reference_state = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    rebuffer_cpp.apply_incremental_buffer_split(reference_state, split)
    checkpoint = graph.checkpoint()

    resize = {"action_id": 0, "buffer_master_name": "BUF1"}
    assert graph.resize_buffer(1, resize)["status"] == "ok"
    assert rebuffer_cpp.resize_incremental_buffer(reference_state, resize)["status"] == "ok"
    assert graph.topology_view(1)["buffers"][0]["buffer_master_name"] == "BUF1"

    action = {
        "session_buffer_id": 0,
        "buffer_master_name": "BUF1",
        "candidate_location_x_dbu": 50000.0,
        "candidate_location_y_dbu": 0.0,
        "net_name": "n1",
        "downstream_pin_names": ["OUT"],
    }
    action_payload = {**action, "downstream_pin_ids": [3]}
    graph_patch = graph.build_timing_patch(
        1, action_payload, int(baseline.new_x.numel()), 0
    )
    reference_patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        reference_state,
        action_payload,
        int(baseline.new_x.numel()),
        0,
    )
    np.testing.assert_array_equal(
        np.asarray(graph_patch["rc"]["affected_nodes"]),
        np.asarray(reference_patch["rc"]["affected_nodes"]),
    )
    physical_payload = rebuffer_cpp.export_overlay_carrier(
        base, base_steiner, [reference_state], [action]
    )
    reference = RebufferTimingModel(
        CommittedTimingPyDB(
            base, SimpleNamespace(**physical_payload["physical"])
        ),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    current = RebufferTimingModel(
        _compact_current_pydb(base, graph_patch["graph"], [action], 0),
        use_pin_offset=False,
        steiner_view=physical_payload["steiner"],
    )
    _assert_fresh_timing_parity(current, reference)

    graph.restore(checkpoint)
    assert graph.topology_view(1)["buffers"][0]["buffer_master_name"] == "BUF2"


def test_persistent_timing_graph_cross_net_export_and_partial_restore_match_fresh_timing():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    specifications = (
        (0, 0, 1, "BUF2", 5000.0, 10),
        (1, 2, 3, "BUF1", 55000.0, 11),
    )
    reference_states = []
    actions = []
    splits = []
    for net_id, driver, child, master, x_dbu, action_id in specifications:
        initial = rebuffer_cpp.incremental_topology_from_steiner(
            base_steiner, net_id, driver, baseline.r_unit, baseline.c_unit
        )
        initial_view = rebuffer_cpp.incremental_topology_view(initial)
        graph.attach_net_topology(
            net_id,
            driver,
            4,
            {
                "edges": initial_view["edges"],
                "node_loads": initial_view["node_loads"],
                "coordinates": initial_view["coordinates"],
            },
        )
        split = {
            "action_id": action_id,
            "parent_node_id": driver,
            "child_node_id": child,
            "split_ratio": 0.5,
            "downstream_pin_ids": [child],
            "x_dbu": x_dbu,
            "y_dbu": 0.0,
            "buffer_master_name": master,
        }
        reference = rebuffer_cpp.incremental_topology_from_steiner(
            base_steiner, net_id, driver, baseline.r_unit, baseline.c_unit
        )
        reference_states.append(reference)
        rebuffer_cpp.apply_incremental_buffer_split(reference, split)
        splits.append(split)
        actions.append(
            {
                "session_buffer_id": action_id,
                "buffer_master_name": master,
                "candidate_location_x_dbu": x_dbu,
                "candidate_location_y_dbu": 0.0,
                "net_name": base.net_names[net_id],
                "downstream_pin_names": [base.pin_names[child]],
            }
        )
    checkpoint_before_actions = graph.checkpoint()
    graph.apply_buffer_split(0, splits[0])
    checkpoint_after_net0 = graph.checkpoint()
    graph.apply_buffer_split(1, splits[1])
    exported = graph.export_carrier(
        base_steiner, actions, baseline.r_unit, baseline.c_unit
    )
    reference_export = rebuffer_cpp.export_overlay_carrier(
        base,
        base_steiner,
        reference_states,
        actions,
        baseline.r_unit,
        baseline.c_unit,
    )
    assert exported["physical"]["pin_names"] == reference_export["physical"]["pin_names"]
    np.testing.assert_allclose(
        np.asarray(exported["physical"]["node_x"]),
        np.asarray(reference_export["physical"]["node_x"]),
        rtol=0,
        atol=1e-12,
    )
    current = RebufferTimingModel(
        CommittedTimingPyDB(
            base, SimpleNamespace(**exported["physical"])
        ),
        use_pin_offset=False,
        steiner_view=exported["steiner"],
    )
    reference = RebufferTimingModel(
        CommittedTimingPyDB(
            base, SimpleNamespace(**reference_export["physical"])
        ),
        use_pin_offset=False,
        steiner_view=reference_export["steiner"],
    )
    _assert_fresh_timing_parity(current, reference)

    graph.restore(checkpoint_after_net0)
    assert [row["action_id"] for row in graph.topology_view(0)["buffers"]] == [10]
    assert graph.topology_view(1)["buffers"] == []
    graph.restore(checkpoint_before_actions)
    assert graph.topology_view(0)["buffers"] == []
    assert graph.topology_view(1)["buffers"] == []


def test_persistent_timing_graph_relevelizes_deep_chain_before_restore():
    base = _base()
    baseline = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, baseline)
    initial = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )
    checkpoint = graph.checkpoint()
    reference = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, baseline.r_unit, baseline.c_unit
    )
    last_action = None
    for action_id in range(6):
        view = graph.topology_view(1)
        edge = next(edge for edge in view["edges"] if edge["child"] == 3)
        source = view["coordinates"][edge["parent"]]
        target = view["coordinates"][3]
        x_dbu = source[0] + 0.5 * (target[0] - source[0])
        last_action = {
            "action_id": action_id,
            "parent_node_id": edge["parent"],
            "child_node_id": 3,
            "split_ratio": 0.5,
            "downstream_pin_ids": [3],
            "x_dbu": x_dbu,
            "y_dbu": 0.0,
            "buffer_master_name": "BUF1",
        }
        graph.apply_buffer_split(1, last_action)
        rebuffer_cpp.apply_incremental_buffer_split(reference, last_action)

    levels = graph.relevelize_net(1)
    assert levels["status"] == "ok"
    assert levels["max_level"] == 6
    assert levels["levels"][3] == 6
    assert len(levels["topological_order"]) == 14
    graph_patch = graph.build_timing_patch(
        1,
        {**last_action, "downstream_pin_ids": [3]},
        int(baseline.new_x.numel()),
        0,
    )
    reference_patch = rebuffer_cpp.build_incremental_timing_patch(
        base,
        reference,
        {**last_action, "downstream_pin_ids": [3]},
        int(baseline.new_x.numel()),
        0,
    )
    np.testing.assert_array_equal(
        np.asarray(graph_patch["rc"]["affected_nodes"]),
        np.asarray(reference_patch["rc"]["affected_nodes"]),
    )

    graph.restore(checkpoint)
    restored_levels = graph.relevelize_net(1)
    assert restored_levels["max_level"] == 0
    assert graph.topology_view(1)["buffers"] == []


def test_persistent_timing_graph_compacts_sparse_node_namespace():
    base = _base()
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        21,
        {
            "edges": [
                {"parent": 2, "child": 10, "r": 1.0, "c": 0.1},
                {"parent": 10, "child": 20, "r": 2.0, "c": 0.2},
                {"parent": 20, "child": 3, "r": 3.0, "c": 0.3},
            ],
            "node_loads": {3: [3]},
            "coordinates": {
                2: [10000.0, 0.0],
                10: [30000.0, 0.0],
                20: [70000.0, 0.0],
                3: [100000.0, 0.0],
            },
        },
    )
    compacted = graph.compact_net_topology(1)
    assert compacted["status"] == "ok"
    assert compacted["changed"] is True
    assert compacted["mapping"] == {10: 4, 20: 5}
    assert compacted["next_node_id"] == 6
    view = graph.topology_view(1)
    assert view["edges"] == [
        {"parent": 2, "child": 4, "r": 1.0, "c": 0.1},
        {"parent": 4, "child": 5, "r": 2.0, "c": 0.2},
        {"parent": 5, "child": 3, "r": 3.0, "c": 0.3},
    ]
    rc = graph.build_net_rc(1)
    assert rc["affected_nodes"] == [2, 3, 4, 5]
    assert rc["parent"] == [-1, -1, 2, 5, 2, 4]
    checkpoint = graph.checkpoint()
    unchanged = graph.compact_net_topology(1)
    assert unchanged["changed"] is False
    graph.restore(checkpoint)
    assert graph.topology_view(1)["edges"] == view["edges"]


def test_persistent_timing_graph_filters_python_tp_critical_path_subset():
    base = _base()
    timing = RebufferTimingModel(base, use_pin_offset=False)
    details = timing.native_setup_path_details([3])
    assert details and 3 in details
    graph = rebuffer_cpp.create_persistent_timing_graph(base)

    subset = graph.critical_path_subset(details, [3])
    assert subset["status"] == "ok"
    assert subset["endpoint_count"] == 1
    assert subset["invalid"] == []
    assert subset["paths"][3]["path"] == details[3]["path"]
    assert subset["paths"][3]["transitions"] == details[3]["transitions"]

    empty = graph.critical_path_subset(details, [0])
    assert empty["endpoint_count"] == 0
    assert empty["paths"] == {}

    invalid = graph.critical_path_subset(
        {3: {"path": [0, 3], "transitions": [1, 1]}}, [3]
    )
    assert invalid["endpoint_count"] == 0
    assert invalid["invalid"][0]["endpoint_pin"] == 3


def test_persistent_timing_graph_finalizes_action_mapping_and_provenance():
    base = _base()
    timing = RebufferTimingModel(base, use_pin_offset=False)
    base_steiner = steiner_payload(base, timing)
    initial = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, timing.r_unit, timing.c_unit
    )
    initial_view = rebuffer_cpp.incremental_topology_view(initial)
    graph = rebuffer_cpp.create_persistent_timing_graph(base)
    graph.attach_net_topology(
        1,
        2,
        4,
        {
            "edges": initial_view["edges"],
            "node_loads": initial_view["node_loads"],
            "coordinates": initial_view["coordinates"],
        },
    )
    split_first = {
        "action_id": 0,
        "parent_node_id": 2,
        "child_node_id": 3,
        "split_ratio": 4.0 / 9.0,
        "downstream_pin_ids": [3],
        "x_dbu": 50000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF2",
    }
    split_second = {
        "action_id": 1,
        "parent_node_id": 5,
        "child_node_id": 3,
        "split_ratio": 0.5,
        "downstream_pin_ids": [3],
        "x_dbu": 75000.0,
        "y_dbu": 0.0,
        "buffer_master_name": "BUF1",
    }
    graph.apply_buffer_split(1, split_first)
    graph.apply_buffer_split(1, split_second)
    reference = rebuffer_cpp.incremental_topology_from_steiner(
        base_steiner, 1, 2, timing.r_unit, timing.c_unit
    )
    rebuffer_cpp.apply_incremental_buffer_split(reference, split_first)
    rebuffer_cpp.apply_incremental_buffer_split(reference, split_second)
    actions = [
        {
            "session_buffer_id": 0,
            "buffer_master_name": "BUF2",
            "candidate_location_x_dbu": 50000.0,
            "candidate_location_y_dbu": 0.0,
            "net_name": "n1",
            "downstream_pin_names": ["OUT"],
        },
        {
            "session_buffer_id": 1,
            "buffer_master_name": "BUF1",
            "candidate_location_x_dbu": 75000.0,
            "candidate_location_y_dbu": 0.0,
            "net_name": "n1",
            "downstream_pin_names": ["OUT"],
        },
    ]
    native = graph.finalize_actions(actions)
    reference_final = rebuffer_cpp.finalize_overlay_actions(
        base, [reference], actions
    )
    assert list(native) == list(reference_final)
    assert native[0]["parent_action_id"] == -1
    assert native[1]["parent_action_id"] == 0
    assert native[1]["segment_tree_depth"] > native[0]["segment_tree_depth"]
    from dreamplace.ops.placeio_ecc.buffer_overlay import action_digest

    assert action_digest(list(native)) == action_digest(list(reference_final))
