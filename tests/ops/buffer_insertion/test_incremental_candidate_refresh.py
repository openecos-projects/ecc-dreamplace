"""Regression tests for IncrementalCandidateSession and compact payload invariants."""

from types import SimpleNamespace

import torch
from dreamplace.ops.buffer_insertion.drive_type_table import DriveTypeEntry
from dreamplace.ops.buffer_insertion.incremental_candidate_refresh import (
    IncrementalCandidateSession,
)
from dreamplace.ops.buffer_insertion.incremental_compact_payload import (
    validate_compact_payload_invariants,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_typed_discrete_candidate_state,
)
from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
    RelaxedBufferDynamicNetProvider,
)


def _initial_three_net_fixture():
    """Build compact payload & typed candidate state with non-contiguous orig_net_ids [2, 7, 15].

    Simulates skipped clock/tie/1-pin nets so orig_net_id != net_idx and 15 >= len(orig_net_ids).
    Net 7 has an extra branch with a physical pin in its interior.
    """
    orig_net_ids = [2, 7, 15]
    # Pins: net 2: drv=1, sink=2; net 7: drv=3, sinks=4/7/8; net 15: drv=5, sink=6.
    records = [
        {
            "candidate_id": 101,
            "net_id": 2,
            "net_name": "n2",
            "buffer_main_type_index": 0,
            "driver_pin_id": 1,
            "driver_pin_name": "u2:Z",
            "load_pin_id": 2,
            "load_pin_name": "u2_sink:A",
            "downstream_pin_ids": (2,),
            "downstream_pin_names": ("u2_sink:A",),
            "parent_node_id": 1,
            "child_node_ids": (2,),
            "segment_split_ratio": 0.5,
            "synthetic_node_id": -1,
            "candidate_node_id": -1,
            "x_dbu": 500.0,
            "y_dbu": 0.0,
        },
        {
            "candidate_id": 102,
            "net_id": 7,
            "net_name": "n7",
            "buffer_main_type_index": 0,
            "driver_pin_id": 3,
            "driver_pin_name": "u7:Z",
            "load_pin_id": 4,
            "load_pin_name": "u7_sink:A",
            "downstream_pin_ids": (4,),
            "downstream_pin_names": ("u7_sink:A",),
            "parent_node_id": 3,
            "child_node_ids": (4,),
            "segment_split_ratio": 0.5,
            "synthetic_node_id": -2,
            "candidate_node_id": -2,
            "x_dbu": 1000.0,
            "y_dbu": 0.0,
        },
        {
            "candidate_id": 103,
            "net_id": 15,
            "net_name": "n15",
            "buffer_main_type_index": 0,
            "driver_pin_id": 5,
            "driver_pin_name": "u15:Z",
            "load_pin_id": 6,
            "load_pin_name": "u15_sink:A",
            "downstream_pin_ids": (6,),
            "downstream_pin_names": ("u15_sink:A",),
            "parent_node_id": 5,
            "child_node_ids": (6,),
            "segment_split_ratio": 0.5,
            "synthetic_node_id": -3,
            "candidate_node_id": -3,
            "x_dbu": 1500.0,
            "y_dbu": 0.0,
        },
    ]
    table = SimpleNamespace(
        entries=(
            DriveTypeEntry(
                drive_type_id=0,
                master_name="BUFX2",
                drive_strength=2.0,
                cell_id=10,
                bsu_index=0,
            ),
            DriveTypeEntry(
                drive_type_id=1,
                master_name="BUFX4",
                drive_strength=4.0,
                cell_id=11,
                bsu_index=1,
            ),
        ),
        to_metadata=lambda: {
            "masters": [
                {
                    "drive_type_id": 0,
                    "bsu_index": 0,
                    "cell_id": 10,
                    "master_name": "BUFX2",
                },
                {
                    "drive_type_id": 1,
                    "bsu_index": 1,
                    "cell_id": 11,
                    "master_name": "BUFX4",
                },
            ]
        },
    )
    state = build_typed_discrete_candidate_state(
        records,
        buffer_main_type_index=0,
        legal_buffer_count=2,
        drive_type_table=table,
    )
    state.candidate_node_id = torch.tensor([1, 4, 9], dtype=torch.long)

    payload = {
        "net_flat_topo_sort": torch.arange(11, dtype=torch.long),
        "net_flat_topo_sort_start": torch.tensor([0, 3, 8, 11], dtype=torch.long),
        "pin_fa": torch.tensor([-1, 0, 1, -1, 3, 4, 3, 6, -1, 8, 9], dtype=torch.long),
        "flat_pin_to_start": torch.tensor(
            [0, 1, 2, 2, 4, 5, 5, 6, 6, 7, 8, 8], dtype=torch.long
        ),
        "flat_pin_to": torch.tensor([1, 2, 4, 6, 5, 7, 9, 10], dtype=torch.long),
        "edge_resistance": torch.tensor(
            [0.0, 5.0, 5.0, 0.0, 10.0, 10.0, 6.0, 2.0, 0.0, 8.0, 8.0], dtype=torch.float32
        ),
        "edge_capacitance": torch.tensor(
            [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.2, 0.5, 0.0, 1.5, 1.5], dtype=torch.float32
        ),
        "node_capacitance": torch.tensor(
            [0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.5, 0.75, 0.0, 0.0, 5.0], dtype=torch.float32
        ),
        "candidate_node_id": torch.tensor([1, 4, 9], dtype=torch.long),
        "candidate_net_id": torch.tensor([2, 7, 15], dtype=torch.long),
        "candidate_bu": torch.zeros(3, dtype=torch.float32),
        "buffer_input_cap": torch.zeros(3, dtype=torch.float32),
        "buffer_delay": torch.zeros(3, dtype=torch.float32),
        "buffer_output_slew": torch.zeros(3, dtype=torch.float32),
        "per_size_input_cap": torch.tensor(
            [[1.2, 2.4]] * 3, dtype=torch.float32
        ),
        "per_size_delay": torch.tensor(
            [[15.0, 12.0]] * 3, dtype=torch.float32
        ),
        "per_size_output_slew": torch.tensor(
            [[8.0, 6.0]] * 3, dtype=torch.float32
        ),
        "driver_arrival": torch.zeros(3, dtype=torch.float32),
        "driver_slew": torch.ones(3, dtype=torch.float32),
        # sink_net_index holds design-level original net IDs.
        "sink_node_id": torch.tensor([2, 5, 6, 7, 10], dtype=torch.long),
        "sink_net_index": torch.tensor([2, 7, 7, 7, 15], dtype=torch.long),
        "sink_pin_id": torch.tensor([2, 4, 7, 8, 6], dtype=torch.long),
        "static_rc_edge_mask": torch.zeros(11, dtype=torch.bool),
        "metadata": {
            "net_ids": orig_net_ids,
            "compact_node_to_original": [1, -1, 2, 3, -1, 4, 7, 8, 5, -1, 6],
        },
    }
    validate_compact_payload_invariants(
        orig_net_ids=orig_net_ids, payload=payload, state=state
    )
    return orig_net_ids, payload, state


class _StubLane:
    def __init__(self, state, payload):
        self.buffer_optimization_state = state
        self.buffer_relaxed_timing_payload = payload
        self._full_candidate_topology_signature = "sig"
        self._full_candidate_structure_signature = "struct"

    def _current_state(self):
        return self.buffer_optimization_state

    def attach_state(self, _dc):
        return None

    def register_model_parameters(self, _model):
        return None


class _StubGate:
    def __init__(self, trees_by_call):
        self._trees_by_call = list(trees_by_call)
        self.remapped_calls = []

    def refresh_dirty_subnets(self, dirty_nets):
        return self._trees_by_call.pop(0)

    def remap_rejection_masks(self, **kwargs):
        self.remapped_calls.append(kwargs)


def test_refresh_round_preserves_noncontiguous_orig_net_sink_and_compact_candidate_contracts():
    """Keep physical input-side branches, sink alignment, and current subnet ownership."""
    orig_net_ids, payload, state = _initial_three_net_fixture()
    lane = _StubLane(state, payload)
    placedb = SimpleNamespace(
        num_pins=10,
        pin_names=[
            "",
            "u2:Z",
            "u2_sink:A",
            "u7:Z",
            "u7_sink:A",
            "u15:Z",
            "u15_sink:A",
            "u7_tap:A",
            "u7_tail:A",
        ],
    )
    model = SimpleNamespace(data_collections=None, placedb=placedb, op_collections=None)
    params = SimpleNamespace(
        dbu=1000,
        buffering_max_candidates_per_segment=2,
        buffering_include_tree_node_candidates=0,
        buffering_candidate_policy="rebuffer_sites",
        buffering_candidate_drive_type_mode="single_fixed",
    )

    # FLUTE visits the surviving tap and tail via the buffer input vertex on net 7.
    round0_trees = [
        {
            "net_id": 7,
            "driver_pin_id": 3,
            "root_vertex": 0,
            "topo": [0, 1, 2, 3],
            "parent": [-1, 0, 1, 2],
            "edge_r": [0.0, 4.0, 2.0, 1.0],
            "edge_c": [0.0, 0.8, 0.4, 0.2],
            "vertex_x_dbu": [0.0, 1000.0, 1500.0, 1800.0],
            "vertex_y_dbu": [0.0] * 4,
            "vertex_pin_id": [3, 20, 7, 8],
        },
        {
            "net_id": 100,
            "driver_pin_id": 21,
            "root_vertex": 0,
            "topo": [0, 1],
            "parent": [-1, 0],
            "edge_r": [0.0, 6.0],
            "edge_c": [0.0, 1.2],
            "vertex_x_dbu": [1000.0, 2000.0],
            "vertex_y_dbu": [0.0, 0.0],
            "vertex_pin_id": [21, 4],
        },
    ]
    # Round 1 inserts before the first buffer, moving its input and the tap to net 101.
    round1_trees = [
        {
            "net_id": 7,
            "driver_pin_id": 3,
            "root_vertex": 0,
            "topo": [0, 1],
            "parent": [-1, 0],
            "edge_r": [0.0, 2.0],
            "edge_c": [0.0, 0.4],
            "vertex_x_dbu": [0.0, 500.0],
            "vertex_y_dbu": [0.0, 0.0],
            "vertex_pin_id": [3, 22],
        },
        {
            "net_id": 101,
            "driver_pin_id": 23,
            "root_vertex": 0,
            "topo": [0, 1, 2, 3],
            "parent": [-1, 0, 1, 2],
            "edge_r": [0.0, 2.0, 2.0, 1.0],
            "edge_c": [0.0, 0.4, 0.4, 0.2],
            "vertex_x_dbu": [500.0, 1000.0, 1500.0, 1800.0],
            "vertex_y_dbu": [0.0] * 4,
            "vertex_pin_id": [23, 20, 7, 8],
        },
    ]
    gate = _StubGate([round0_trees, round1_trees])
    session = IncrementalCandidateSession(
        lane=lane, model=model, params=params, incremental_gate=gate
    )
    assert session.orig_sinks_by_net == {2: {2}, 7: {4, 7, 8}, 15: {6}}

    rep0 = session.refresh_round(
        accepted_action_infos=[
            {
                "row_idx": 1,
                "candidate_id": 102,
                "orig_net_id": 7,
                "drive_type_id": 1,
                "active_net": 7,
                "new_net": 100,
                "p_in": 20,
                "p_out": 21,
                "buf_node": 50,
                "x_dbu": 1000.0,
                "y_dbu": 0.0,
            }
        ],
        dirty_nets=[7, 100],
    )
    assert rep0["status"] == "ok"
    refreshed_payload = lane.buffer_relaxed_timing_payload
    refreshed_state = lane.buffer_optimization_state

    validate_compact_payload_invariants(
        orig_net_ids=orig_net_ids, payload=refreshed_payload, state=refreshed_state
    )
    assert refreshed_payload["sink_net_index"].tolist() == [2, 7, 7, 7, 15]
    assert refreshed_payload["sink_pin_id"].tolist() == [2, 4, 7, 8, 6]
    sink_nodes = dict(zip(
        refreshed_payload["sink_pin_id"].tolist(),
        refreshed_payload["sink_node_id"].tolist(),
        strict=True,
    ))
    parents = refreshed_payload["pin_fa"].tolist()
    locked_row = refreshed_state.candidate_ids.tolist().index(102)
    locked_node = int(refreshed_state.candidate_node_id[locked_row])
    junction = parents[locked_node]
    assert junction >= 0

    def ancestors(node):
        path = []
        while parents[node] >= 0:
            node = parents[node]
            path.append(node)
        return path

    assert locked_node in ancestors(sink_nodes[4])
    assert junction in ancestors(sink_nodes[7])
    assert sink_nodes[7] in ancestors(sink_nodes[8])
    assert locked_node not in ancestors(sink_nodes[8])
    assert refreshed_payload["node_capacitance"][sink_nodes[7]] == 0.5
    assert refreshed_payload["edge_resistance"][locked_node] == 0
    assert refreshed_payload["edge_capacitance"][locked_node] == 0

    # RelaxedBufferDynamicNetProvider must keep non-empty sink_indices for shifted net 15!
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=refreshed_payload,
        buffer_state=refreshed_state,
    )
    view_15 = provider._active_view((15,))
    assert int(view_15["sink_indices_cpu"].numel()) == 1
    view_7 = provider._active_view((7,))
    assert int(view_7["sink_indices_cpu"].numel()) == 3

    # Round 1: pick an upstream candidate on net 7 (cpp_net_id == 7) and insert upstream of buf1.
    upstream_row, upstream_rec = next(
        (idx, rec)
        for idx, rec in enumerate(refreshed_state.candidate_records)
        if rec.get("cpp_net_id") == 7
        and rec.get("cpp_child_vertices") == (1,)
        and int(rec["candidate_id"]) != 102
    )
    rep1 = session.refresh_round(
        accepted_action_infos=[
            {
                "row_idx": upstream_row,
                "candidate_id": int(upstream_rec["candidate_id"]),
                "orig_net_id": 7,
                "drive_type_id": 0,
                "active_net": 7,
                "new_net": 101,
                "p_in": 22,
                "p_out": 23,
                "buf_node": 51,
                "x_dbu": 500.0,
                "y_dbu": 0.0,
            }
        ],
        dirty_nets=[7, 101],
    )
    assert rep1["status"] == "ok"
    final_state = lane.buffer_optimization_state
    by_cid = {int(rec["candidate_id"]): rec for rec in final_state.candidate_records}
    upstream_locked = by_cid[int(upstream_rec["candidate_id"])]
    downstream_locked = by_cid[102]
    assert upstream_locked["segment_tree_depth"] == 0
    assert downstream_locked["segment_tree_depth"] == 1
    assert upstream_locked["downstream_pin_names"] == ("u7_sink:A", "u7_tap:A", "u7_tail:A")
    assert downstream_locked["downstream_pin_names"] == ("u7_sink:A",)
    assert downstream_locked["cpp_net_id"] == 101
    downstream_row = final_state.candidate_ids.tolist().index(102)
    assert int(final_state.net_ids[final_state.candidate_net_index[downstream_row]]) == 101
