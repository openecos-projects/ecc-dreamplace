from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dreamplace.ops.rebuffer.rebuffer_cpp")

from dreamplace.ops.placeio_ecc.current_driver_tree import (  # noqa: E402
    begin_current_driver_tree_session,
    current_driver_tree_reachable_net_ids,
    current_driver_tree_root_net_id,
    import_current_driver_tree,
)


def _pydb():
    return SimpleNamespace(
        dbu=1000,
        flat_net2pin_map=[0, 1, 2, 3, 4],
        flat_net2pin_start_map=[0, 3, 5],
        net2driver_pin_map=[0, 3],
        pin2node_map=[0, 1, 2, 1, 3],
        pin2net_map=[0, 0, 0, 1, 1],
        pin_offset_x=[0, 0, 0, 0, 0],
        pin_offset_y=[0, 0, 0, 0, 0],
        node_x=[0, 1000, 2000, 3000],
        node_y=[0, 0, 1000, 0],
        node_size_x=[100, 100, 100, 100],
        node_size_y=[100, 100, 100, 100],
        node_is_buffer=[False, True, False, False],
        node_is_fixed=[False, False, False, False],
        inst_main_id=[10, 20, 30, 40],
        inst_libcell_offset=[0, 0, 0, 0],
        main_id_2_cell_id_start=[0, 1, 2, 3, 4],
        cell_id_2_libpin_id_start=[0, 1, 2, 3, 4],
        pin_2_libpin_offset=[0, 0, 0, 0, 0],
        pin_direct=[-1, 1, 1, -1, 1],
        flat_lib_pin_cap=[0.0, 0.42, 0.0, 0.42],
        flat_lib_pin_rcap=[0.0, 0.42, 0.0, 0.42],
        flat_lib_pin_cap_limit=[0.0] * 4,
        flat_lib_pin_slew_limit=[0.0] * 4,
        flat_lib_pin_drive_resistance=[0.0] * 4,
        flat_libcell_info=[
            ["DRV", 0, 1.0, 0.0],
            ["BUF", 0, 1.0, 0.0],
            ["LOAD", 0, 1.0, 0.0],
            ["LOAD2", 0, 1.0, 0.0],
        ],
        flat_libcell_names=["DRV", "BUF", "LOAD", "LOAD2"],
        net_names=["n0", "n1"],
        pin_names=["drv/Y", "buf/A", "load/A", "buf/Y", "load2/A"],
        node_names=["drv", "buf", "load", "load2"],
        clock_pins=[],
        clock_net_names=[],
        rows=[(0, 0, 10000, 1000)],
        site_width=500,
        row_height=1000,
        num_movable_nodes=4,
        buffer_main_type_index=1,
        buffer_main_type_candidate_indices=[1],
    )


def _topology():
    return {
        "status": "ok",
        "topology_source": "fixture",
        "net_ids": [0, 1],
        "net_flat_topo_sort_start": [0, 3, 5],
        "net_flat_topo_sort": [0, 1, 2, 3, 4],
        "pin_fa": [-1, 0, 0, -1, 3],
        "steiner_node_base": 5,
        "node_x": {0: 0, 1: 1000, 2: 2000, 3: 1000, 4: 3000},
        "node_y": {0: 0, 1: 0, 2: 1000, 3: 0, 4: 0},
    }


def test_current_driver_tree_imports_buffer_bridge_and_round_trips():
    tree = import_current_driver_tree(_pydb(), _topology(), root_net_id=0)

    assert tree["status"] == "ok"
    assert tree["net_ids"] == [0, 1]
    assert tree["root_net_id"] == 0
    assert tree["driver_pin_id"] == 0
    assert tree["buffer_count"] == 1
    assert tree["round_trip"]["valid"] is True
    assert tree["round_trip"]["tree_digest"] == tree["tree_digest"]
    assert tree["node_count"] == 6
    assert tree["edge_count"] == 5

    kinds = {node["kind_name"] for node in tree["nodes"]}
    assert {"driver", "buffer_input", "buffer", "buffer_output", "logic_load"} <= kinds
    buffer = next(node for node in tree["nodes"] if node["kind_name"] == "buffer")
    assert buffer["source_instance_id"] == 1
    assert buffer["existing_buffer"] is True


def test_current_driver_tree_accepts_dense_coordinate_arrays():
    dense = {
        **_topology(),
        "node_x": np.array([0.0, 1000.0, 2000.0, 1000.0, 3000.0, 99.0]),
        "node_y": np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 99.0]),
    }

    assert import_current_driver_tree(_pydb(), dense, root_net_id=0) == import_current_driver_tree(
        _pydb(), _topology(), root_net_id=0
    )


def test_current_driver_tree_can_start_at_existing_buffer_output():
    tree = import_current_driver_tree(_pydb(), _topology(), root_net_id=1)

    assert tree["status"] == "ok"
    assert tree["root_net_id"] == 1
    assert tree["driver_pin_id"] == 3
    assert tree["net_ids"] == [1]
    assert tree["buffer_count"] == 0
    root = next(node for node in tree["nodes"] if node["id"] == tree["root_node_id"])
    assert root["kind_name"] == "buffer_output"


def test_current_driver_tree_reachable_nets_follow_buffer_output_bridge():
    assert current_driver_tree_reachable_net_ids(_pydb(), 0) == [0, 1]
    assert current_driver_tree_reachable_net_ids(_pydb(), 1) == [1]
    assert current_driver_tree_root_net_id(_pydb(), 0) == 0
    assert current_driver_tree_root_net_id(_pydb(), 1) == 0


def test_current_driver_tree_round_trip_rejects_tampered_digest():
    tree = import_current_driver_tree(_pydb(), _topology(), root_net_id=0)
    tree["tree_digest"] = "tampered"

    from dreamplace.ops.rebuffer import rebuffer_cpp

    result = rebuffer_cpp.validate_current_driver_tree_export(tree)
    assert result["status"] == "unsupported"
    assert result["reason"] == "current_tree_digest_mismatch"


def test_current_driver_tree_fails_closed_without_buffer_metadata():
    pydb = _pydb()
    pydb.node_is_buffer = None
    del pydb.buffer_main_type_candidate_indices

    result = import_current_driver_tree(pydb, _topology(), root_net_id=0)

    assert result["status"] == "unsupported"
    assert result["unsupported_reasons"] == ["missing_node_is_buffer"]


def test_placeio_ecc_tree_session_wraps_native_checkpoint_apply_restore():
    pydb = _pydb()
    session = begin_current_driver_tree_session(
        pydb, _topology(), root_net_id=0
    )
    checkpoint = session.checkpoint()
    source = session.summary()
    source_tree = import_current_driver_tree(pydb, _topology(), root_net_id=0)
    buffer = next(node for node in source_tree["nodes"] if node["kind_name"] == "buffer")
    mutation = {
        "kind": 1,
        "node_id": buffer["id"],
        "source_instance_id": buffer["source_instance_id"],
        "parent_node_id": buffer["parent_node_id"],
        "child_node_id": buffer["child_node_ids"][0],
        "target_cell_index": -1,
        "target_main_id": -1,
    }

    applied = session.apply_mutation(mutation)
    assert applied["transaction_state"] == "APPLIED"
    assert session.summary()["active_node_count"] == source["source_node_count"] - 1

    restored = session.restore(checkpoint)
    assert restored["transaction_state"] == "RESTORED"
    assert session.summary()["active_tree_digest"] == source["source_tree_digest"]


def test_placeio_ecc_tree_session_routes_python_tp_result_to_native_journal():
    session = begin_current_driver_tree_session(
        _pydb(), _topology(), root_net_id=0
    )
    checkpoint = session.checkpoint()
    source_tree = import_current_driver_tree(_pydb(), _topology(), root_net_id=0)
    buffer = next(node for node in source_tree["nodes"] if node["kind_name"] == "buffer")
    mutation = {
        "kind": 1,
        "node_id": buffer["id"],
        "source_instance_id": buffer["source_instance_id"],
        "parent_node_id": buffer["parent_node_id"],
        "child_node_id": buffer["child_node_ids"][0],
        "target_cell_index": -1,
        "target_main_id": -1,
    }
    applied = session.apply_mutation(mutation)

    observed = {}

    def evaluate(candidate):
        observed.update(candidate)
        return {
            "status": "evaluated",
            "timing_epoch": candidate["timing_epoch"],
            "endpoint_scope_digest": "scope-fixture",
            "wns_ps": -10.0,
            "tns_ps": -20.0,
            "worst_slack_ps": -10.0,
            "violating_endpoint_count": 1,
            "max_slew_violation": 0.0,
            "max_cap_violation": 0.0,
            "area": 1.0,
            "buffer_count": applied["buffer_count"],
            "score_digest": "score-fixture",
            "failure_reason": "",
        }

    evaluated = session.evaluate_candidate(
        evaluate,
        design_identity={"native_build_id": "fixture"},
        affected_instance_ids=[buffer["source_instance_id"]],
        affected_net_ids=[0, 1],
        endpoint_scope=[2],
        timing_epoch=7,
        requested_metrics=["wns_ps", "tns_ps"],
    )
    assert observed["design_identity"] == {"native_build_id": "fixture"}
    assert observed["candidate_tree_digest"] == applied["candidate_tree_digest"]
    assert observed["transaction_identity"]["timing_epoch"] == 7
    assert evaluated["transaction_state"] == "EVALUATED"

    accepted = session.accept_candidate()
    assert accepted["transaction_state"] == "ACCEPTED"
    assert session.summary()["transaction_identity"]["transaction_state"] == "ACCEPTED"

    # An accepted native candidate cannot be evaluated a second time through a
    # stale Python callback or silently restored as if it were still pending.
    with pytest.raises(RuntimeError):
        session.evaluate_candidate(evaluate, timing_epoch=7)
    assert checkpoint["checkpoint_id"] == 0


