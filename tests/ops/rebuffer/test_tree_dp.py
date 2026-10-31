from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("dreamplace.ops.rebuffer.rebuffer_cpp")

from dreamplace.ops.placeio_ecc.current_driver_tree import (  # noqa: E402
    import_current_driver_tree,
)
from dreamplace.ops.rebuffer import rebuffer_cpp  # noqa: E402
from test_repair_design import _characterization, _liberty, _timing  # noqa: E402


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
        node_is_buffer=[False, False, False, False],
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


def _single_edge_topology():
    return {
        "status": "ok",
        "topology_source": "exhaustive_fixture",
        "net_ids": [1],
        "net_flat_topo_sort_start": [0, 2],
        "net_flat_topo_sort": [3, 4],
        "pin_fa": [-1, -1, -1, -1, 3],
        "steiner_node_base": 5,
        "node_x": {3: 0, 4: 100_000},
        "node_y": {3: 0, 4: 0},
    }


def _serial_buffer_pydb():
    pydb = deepcopy(_pydb())
    pydb.flat_net2pin_map = [0, 1, 6, 2, 3, 4, 5]
    pydb.flat_net2pin_start_map = [0, 3, 5, 7]
    pydb.net2driver_pin_map = [0, 2, 4]
    pydb.pin2node_map = [0, 1, 1, 2, 2, 3, 4]
    pydb.pin2net_map = [0, 0, 1, 1, 2, 2, 0]
    pydb.pin_offset_x = [0] * 7
    pydb.pin_offset_y = [0] * 7
    pydb.pin_2_libpin_offset = [0] * 7
    pydb.pin_direct = [-1, 1, -1, 1, -1, 1, 1]
    pydb.node_x = [0, 1000, 2000, 3000, 1000]
    pydb.node_y = [0, 0, 0, 0, 1000]
    pydb.node_size_x = [100] * 5
    pydb.node_size_y = [100] * 5
    pydb.node_is_buffer = [False, True, True, False, False]
    pydb.node_is_fixed = [False] * 5
    pydb.inst_main_id = [0, 0, 0, 1, 1]
    pydb.inst_libcell_offset = [0, 1, 1, 0, 0]
    pydb.pin_names = [
        "drv/Y",
        "buf1/A",
        "buf1/Y",
        "buf2/A",
        "buf2/Y",
        "load/A",
        "branch/A",
    ]
    pydb.node_names = ["drv", "buf1", "buf2", "load", "branch"]
    pydb.net_names = ["n0", "n1", "n2"]
    pydb.num_movable_nodes = 5
    return pydb


def _serial_buffer_topology():
    return {
        "status": "ok",
        "topology_source": "serial_buffer_fixture",
        "net_ids": [0, 1, 2],
        "net_flat_topo_sort_start": [0, 3, 5, 7],
        "net_flat_topo_sort": [0, 1, 6, 2, 3, 4, 5],
        "pin_fa": [-1, 0, -1, 2, -1, 4, 0],
        "steiner_node_base": 7,
        "node_x": {0: 0, 1: 1000, 2: 2000, 3: 3000, 4: 2000, 5: 3000, 6: 1000},
        "node_y": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1000},
    }


from dreamplace.ops.rebuffer.pydb_adapter import (  # noqa: E402
    build_rebuffer_design_from_pydb,
)


def _unbuffered_tree():
    pydb = _pydb()
    pydb.node_is_buffer = [False] * len(pydb.node_names)
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = import_current_driver_tree(pydb, _topology(), root_net_id=0)
    native_tree = rebuffer_cpp.import_current_driver_tree(design, _topology(), 0)
    return design, tree, native_tree


def test_joint_tree_vgdp_emits_complete_tree_candidates_and_trace():
    design, tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing["pin_slack"] = [0.0, -0.2, -0.1, 0.0, 0.0]
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
        native_tree,
    )

    assert result["status"] == "ok"
    assert result["tree_node_count"] == tree["node_count"]
    assert result["tree_edge_count"] == tree["edge_count"]
    assert result["generated_option_count"] > 0
    assert result["surviving_option_count"] >= 1
    assert result["trace"]
    assert all(
        candidate["actions"] for candidate in result["candidates"] if candidate["buffer_count"] > 0
    )
    assert all(
        action["segment_tree_depth"] >= 0
        for candidate in result["candidates"]
        for action in candidate["actions"]
    )


def test_tree_vgdp_actions_materialize_as_a_valid_native_candidate_tree():
    design, _tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing["pin_slack"] = [0.0, -0.2, -0.1, 0.0, 0.0]
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 0.01},
        native_tree,
    )
    candidate = next(
        candidate for candidate in result["candidates"] if candidate["buffer_count"] > 0
    )

    invalid_action = deepcopy(candidate["actions"])
    invalid_action[0]["source_parent_node_id"] += 1
    with pytest.raises(ValueError, match="physical_source_identity_mismatch"):
        rebuffer_cpp.apply_current_driver_tree_buffer_actions(
            native_tree, _liberty(), invalid_action
        )

    candidate_tree = rebuffer_cpp.apply_current_driver_tree_buffer_actions(
        native_tree, _liberty(), candidate["actions"]
    )
    candidate_view = rebuffer_cpp.export_current_driver_tree(candidate_tree)
    source_view = rebuffer_cpp.export_current_driver_tree(native_tree)
    assert candidate_view["status"] == "ok"
    assert candidate_view["buffer_count"] == candidate["buffer_count"]
    assert candidate_view["node_count"] == source_view["node_count"] + candidate["buffer_count"]
    assert candidate_view["edge_count"] == candidate_view["node_count"] - 1
    assert candidate_view["topology_epoch"] == source_view["topology_epoch"] + 1
    assert all(
        not node["existing_buffer"]
        for node in candidate_view["nodes"]
        if node["kind_name"] == "buffer"
        and node["source_instance_id"] < 0
    )

    transaction = rebuffer_cpp.create_current_driver_tree_transaction(native_tree)
    applied = transaction.apply_candidate_tree(candidate_tree, source_view["tree_digest"])
    assert applied["transaction_state"] == "APPLIED"
    assert applied["candidate_tree_digest"] == candidate_view["tree_digest"]
    with pytest.raises(ValueError, match="stale_source_tree_identity"):
        transaction.apply_candidate_tree(candidate_tree, "stale")


def test_joint_tree_vgdp_is_deterministic_and_keeps_pass_through_candidate():
    design, tree, native_tree = _unbuffered_tree()
    timing = _timing()
    timing["pin_slack"] = [0.0, -0.2, -0.1, 0.0, 0.0]
    config = {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4}
    first = rebuffer_cpp.plan_current_driver_tree(
        design, timing, _liberty(), _characterization(), config, native_tree
    )
    second = rebuffer_cpp.plan_current_driver_tree(
        design, timing, _liberty(), _characterization(), config, native_tree
    )

    assert first == second
    assert any(candidate["buffer_count"] == 0 for candidate in first["candidates"])


def test_joint_tree_vgdp_does_not_insert_on_existing_buffer_cell_edges():
    pydb = _serial_buffer_pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(
        design, _serial_buffer_topology(), root_net_id=0
    )
    source = rebuffer_cpp.export_current_driver_tree(tree)
    cell_edges = {
        (edge["parent_node_id"], edge["child_node_id"])
        for edge in source["edges"]
        if edge["kind"] == 1
    }
    timing = _timing()
    timing["pin_slack"] = [0.0, -10.0, -10.0, 0.0, -10.0, -10.0, -10.0]
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
        tree,
    )

    assert result["status"] == "ok"
    assert all(
        (action["parent_node_id"], action["child_node_id"]) not in cell_edges
        for candidate in result["candidates"]
        for action in candidate["actions"]
    )


def test_joint_tree_vgdp_merges_two_branch_options_exhaustively():
    pydb = _pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    topology = _topology()
    topology["node_x"] = {0: 0, 1: 100_000, 2: 100_000, 3: 0, 4: 0}
    topology["node_y"] = {0: 0, 1: 0, 2: 100_000, 3: 0, 4: 0}
    tree = rebuffer_cpp.import_current_driver_tree(design, topology, 0)
    timing = _timing()
    timing["pin_slew_rise"] = [0.2] * 5
    timing["pin_slew_fall"] = [0.2] * 5
    timing["pin_load"] = [0.05, 0.001, 0.001, 0.05, 0.001]
    timing["pin_cap"] = [0.001] * 5
    timing["pin_max_slew"] = [0.1] * 5
    timing["pin_max_cap"] = [0.01] * 5
    timing["pin_slack"] = [0.0, -100.0, -90.0, 0.0, 0.0]
    result = rebuffer_cpp.plan_current_driver_tree(
        design,
        timing,
        _liberty(),
        _characterization(),
        {"tree_enforce_electrical": False, "r_unit": 0.914, "c_unit": 1.89e-4},
        tree,
    )

    assert result["status"] == "ok"
    assert result["tree_branch_count"] == 1
    assert {candidate["buffer_count"] for candidate in result["candidates"]} == {0, 1, 2}
    assert [len(candidate["actions"]) for candidate in result["candidates"]] == [
        candidate["buffer_count"] for candidate in result["candidates"]
    ]
    assert all(
        {action["child_node_id"] for action in candidate["actions"]}
        <= {1, 2}
        for candidate in result["candidates"]
    )
    assert all(
        action["segment_tree_depth"] == 0
        for candidate in result["candidates"]
        for action in candidate["actions"]
    )
    assert any(
        item["reason"] == "dominated" and item["node_id"] == 0
        for item in result["trace"]
    )


def test_existing_tree_mutations_keep_remove_and_rewrite_validate():
    pydb = _pydb()
    pydb.node_is_buffer = [False, True, False, False]
    pydb.inst_main_id = [0, 0, 1, 0]
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(design, _topology(), 0)
    mutations = list(
        rebuffer_cpp.enumerate_current_driver_tree_mutations(
            design, _liberty(), tree
        )
    )

    assert {mutation["kind"] for mutation in mutations} >= {0, 1, 2}
    keep = next(mutation for mutation in mutations if mutation["kind"] == 0)
    remove = next(mutation for mutation in mutations if mutation["kind"] == 1)
    rewrite = next(
        mutation
        for mutation in mutations
        if mutation["kind"] == 2 and mutation["target_cell_index"] != 0
    )
    source = rebuffer_cpp.export_current_driver_tree(tree)

    kept = rebuffer_cpp.apply_current_driver_tree_mutation(tree, keep)
    assert kept["tree_digest"] == source["tree_digest"]

    removed = rebuffer_cpp.apply_current_driver_tree_mutation(tree, remove)
    assert removed["status"] == "ok"
    assert removed["node_count"] == source["node_count"] - 1
    assert removed["buffer_count"] == 0
    assert removed["edge_count"] == removed["node_count"] - 1

    rewritten = rebuffer_cpp.apply_current_driver_tree_mutation(tree, rewrite)
    assert rewritten["status"] == "ok"
    assert rewritten["node_count"] == source["node_count"]
    assert rewritten["buffer_count"] == 1
    buffer_node = next(
        node for node in rewritten["nodes"] if node["kind_name"] == "buffer"
    )
    assert buffer_node["master_id"] == source["nodes"][buffer_node["id"]]["master_id"]
    assert buffer_node["cell_id"] == rewrite["target_cell_index"]
    assert rewrite["target_main_id"] == buffer_node["master_id"]


def test_tree_area_recovery_enumerates_native_pass_through_removal():
    pydb = _pydb()
    pydb.node_is_buffer = [False, True, False, False]
    pydb.inst_main_id = [0, 0, 1, 0]
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(design, _topology(), 0)

    candidates = list(
        rebuffer_cpp.enumerate_tree_area_recovery(design, _liberty(), tree)
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["kind"] == 1
    assert candidate["recovery_kind"] == "remove_buffer_pass_through"
    assert candidate["buffer_count_delta"] == -1
    assert candidate["area_saved"] > 0.0
    removed = rebuffer_cpp.apply_current_driver_tree_mutation(tree, candidate)
    assert removed["status"] == "ok"
    assert removed["buffer_count"] == 0

    transaction = rebuffer_cpp.create_current_driver_tree_transaction(tree)
    transaction_candidates = transaction.area_recovery_candidates(
        design, _liberty()
    )
    assert transaction_candidates[0]["node_id"] == candidate["node_id"]
    assert transaction_candidates[0]["area_saved"] == candidate["area_saved"]


def test_tree_area_recovery_enumerates_smaller_same_family_rewrite():
    pydb = _pydb()
    pydb.node_is_buffer = [False, True, False, False]
    pydb.inst_main_id = [0, 0, 1, 0]
    pydb.inst_libcell_offset = [0, 1, 0, 0]
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(design, _topology(), 0)

    candidates = list(
        rebuffer_cpp.enumerate_tree_area_recovery(design, _liberty(), tree)
    )
    rewrite = next(
        candidate
        for candidate in candidates
        if candidate["recovery_kind"] == "rewrite_buffer_smaller_master"
    )

    assert rewrite["kind"] == 2
    assert rewrite["target_cell_index"] == 0
    assert rewrite["area_saved"] == 1.0
    rewritten = rebuffer_cpp.apply_current_driver_tree_mutation(tree, rewrite)
    assert rewritten["status"] == "ok"
    assert rewritten["buffer_count"] == 1
    node = next(item for item in rewritten["nodes"] if item["kind_name"] == "buffer")
    assert node["cell_id"] == 0


def test_native_tree_area_recovery_owns_accept_restore_and_ledger():
    pydb = _serial_buffer_pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(
        design, _serial_buffer_topology(), 0
    )
    transaction = rebuffer_cpp.create_current_driver_tree_transaction(tree)
    source_node_count = transaction.summary()["source_node_count"]
    physical_checkpoints = []
    physical_applies = []
    physical_restores = []

    def physical_checkpoint(label, mutation):
        token = len(physical_checkpoints)
        physical_checkpoints.append(
            (label, token, mutation["source_instance_id"])
        )
        return {"checkpoint_id": token}

    def physical_apply(mutation, tree_result):
        physical_applies.append(
            (mutation["source_instance_id"], tree_result["candidate_tree_digest"])
        )
        return {"status": "applied"}

    def evaluate(mutation, tree_result):
        if mutation is None:
            return {
                "status": "evaluated",
                "wns_ps": -10.0,
                "tns_ps": -20.0,
            }
        if mutation["source_instance_id"] == 1:
            return {
                "status": "evaluated",
                "wns_ps": -10.5,
                "tns_ps": -20.0,
            }
        return {
            "status": "evaluated",
            "wns_ps": -10.0,
            "tns_ps": -20.0,
        }

    def physical_restore(token):
        physical_restores.append(int(token))
        return {"status": "restored"}

    result = rebuffer_cpp.run_tree_area_recovery(
        transaction,
        design,
        {**_liberty(), "cell_area": [2.0, 2.0, 1.0]},
        physical_checkpoint,
        physical_apply,
        evaluate,
        physical_restore,
        {},
    )

    assert result["core"] == "cpp"
    assert result["accepted_count"] == 1
    assert result["area_saved"] == 2.0
    assert [row["status"] for row in result["ledger"]] == [
        "rejected",
        "accepted",
    ]
    assert len(physical_checkpoints) == 2
    assert len(physical_applies) == 2
    assert [item[2] for item in physical_checkpoints] == [1, 2]
    assert physical_restores == [0]
    assert transaction.summary()["active_node_count"] == source_node_count - 1
    assert transaction.summary()["transaction_state"] == "ACCEPTED"


def test_native_tree_area_recovery_reenumerates_after_accept():
    pydb = _serial_buffer_pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(
        design, _serial_buffer_topology(), 0
    )
    transaction = rebuffer_cpp.create_current_driver_tree_transaction(tree)
    source_node_count = transaction.summary()["source_node_count"]

    def physical_checkpoint(_label, mutation):
        return {"checkpoint_id": mutation["source_instance_id"]}

    def physical_apply(_mutation, _tree_result):
        return {"status": "applied"}

    def evaluate(_mutation, _tree_result):
        return {
            "status": "evaluated",
            "wns_ps": -10.0,
            "tns_ps": -20.0,
            "max_slew_violation": 0.0,
            "max_cap_violation": 0.0,
        }

    def physical_restore(_token):
        return {"status": "restored"}

    result = rebuffer_cpp.run_tree_area_recovery(
        transaction,
        design,
        {**_liberty(), "cell_area": [2.0, 2.0, 1.0]},
        physical_checkpoint,
        physical_apply,
        evaluate,
        physical_restore,
        {},
    )

    assert result["accepted_count"] == 2
    assert [row["status"] for row in result["ledger"]] == [
        "accepted",
        "accepted",
    ]
    assert transaction.summary()["active_node_count"] == source_node_count - 2


def test_existing_serial_buffers_and_branch_preserve_tree_identity():
    pydb = _serial_buffer_pydb()
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    topology = _serial_buffer_topology()
    tree = rebuffer_cpp.import_current_driver_tree(design, topology, 0)
    tree_view = rebuffer_cpp.export_current_driver_tree(tree)

    assert tree_view["buffer_count"] == 2
    assert tree_view["load_count"] == 2
    assert tree_view["edge_count"] == tree_view["node_count"] - 1
    assert tree_view["net_ids"] == [0, 1, 2]
    assert any(6 in node["load_pin_ids"] for node in tree_view["nodes"])

    mutations = list(
        rebuffer_cpp.enumerate_current_driver_tree_mutations(
            design, _liberty(), rebuffer_cpp.import_current_driver_tree(design, topology, 0)
        )
    )
    buffer_nodes = [
        node for node in tree_view["nodes"] if node["kind_name"] == "buffer"
    ]
    assert len(buffer_nodes) == 2
    assert {mutation["node_id"] for mutation in mutations if mutation["kind"] == 0} == {
        node["id"] for node in buffer_nodes
    }

    for node in buffer_nodes:
        remove = next(
            mutation
            for mutation in mutations
            if mutation["kind"] == 1 and mutation["node_id"] == node["id"]
        )
        removed = rebuffer_cpp.apply_current_driver_tree_mutation(tree, remove)
        assert removed["buffer_count"] == 1
        assert removed["edge_count"] == removed["node_count"] - 1
        rewrite = next(
            mutation
            for mutation in mutations
            if mutation["kind"] == 2
            and mutation["node_id"] == node["id"]
            and mutation["target_cell_index"] != node["cell_id"]
        )
        rewritten = rebuffer_cpp.apply_current_driver_tree_mutation(tree, rewrite)
        rewritten_node = next(
            item for item in rewritten["nodes"] if item["id"] == node["id"]
        )
        assert rewritten_node["master_id"] == node["master_id"]
        assert rewritten_node["cell_id"] == rewrite["target_cell_index"]

    transaction = rebuffer_cpp.create_current_driver_tree_transaction(tree)
    checkpoint = transaction.checkpoint()
    remove = next(mutation for mutation in mutations if mutation["kind"] == 1)
    transaction.apply_mutation(remove)
    assert transaction.summary()["active_node_count"] == tree_view["node_count"] - 1
    restored = transaction.restore(checkpoint)
    assert restored["candidate_tree_digest"] == tree_view["tree_digest"]


def test_current_tree_transaction_apply_and_restore_is_atomic():
    pydb = _pydb()
    pydb.node_is_buffer = [False, True, False, False]
    pydb.inst_main_id = [0, 0, 1, 0]
    design = build_rebuffer_design_from_pydb(pydb, require_buffer_metadata=True)
    tree = rebuffer_cpp.import_current_driver_tree(design, _topology(), 0)
    mutations = list(
        rebuffer_cpp.enumerate_current_driver_tree_mutations(
            design, _liberty(), tree
        )
    )
    remove = next(mutation for mutation in mutations if mutation["kind"] == 1)
    transaction = rebuffer_cpp.create_current_driver_tree_transaction(tree)
    checkpoint = transaction.checkpoint()
    source = transaction.summary()

    applied = transaction.apply_mutation(remove)
    assert applied["transaction_state"] == "APPLIED"
    assert applied["source_tree_digest"] == source["source_tree_digest"]
    assert applied["candidate_tree_digest"] != source["active_tree_digest"]
    assert transaction.summary()["active_node_count"] == source["source_node_count"] - 1

    restored = transaction.restore(checkpoint)
    assert restored["transaction_state"] == "RESTORED"
    assert restored["candidate_tree_digest"] == source["source_tree_digest"]
    assert transaction.summary()["active_tree_digest"] == source["source_tree_digest"]

