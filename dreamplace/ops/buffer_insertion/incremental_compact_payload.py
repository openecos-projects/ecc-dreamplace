"""Compact timing subgraph splicing and invariant checks for 2n sub-net refresh."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from dreamplace.ops.buffer_insertion.optimization_state import (
    DiscreteCandidateState,
    TypedDiscreteCandidateState,
)


@dataclass
class LockedBufferInfo:
    """Metadata for an already-accepted virtual buffer across B1 rounds."""

    candidate_id: int
    drive_type_id: int
    p_in: int
    p_out: int
    upstream_subnet: int
    downstream_subnet: int
    buf_node: int
    record: dict
    buffer_depth: int = 0
    downstream_pin_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DirtyNetCandidateEntry:
    """Candidate or locked-buffer compact node binding inside a refreshed net."""

    compact_node_id: int
    record: dict
    active_drive_type: int | None
    allowed_mask: list[bool]


@dataclass(frozen=True)
class DirtyNetCompactPack:
    """Stitched compact tree slice for one dirty original net."""

    topo: list[int]
    pin_fa: list[int]
    children_by_local: list[list[int]]
    edge_r: list[float]
    edge_c: list[float]
    node_cap: list[float]
    orig_node: list[int]
    sinks: list[tuple[int, int]]
    candidates: list[DirtyNetCandidateEntry]


def validate_compact_payload_invariants(*, orig_net_ids, payload, state):
    """Assert compact-graph and sink/candidate index contracts match tensor_builder."""
    starts = payload["net_flat_topo_sort_start"].detach().cpu().tolist()
    num_compact_nodes = int(payload["net_flat_topo_sort"].numel())
    if len(starts) != len(orig_net_ids) + 1:
        raise ValueError(
            f"net_flat_topo_sort_start length {len(starts)} != {len(orig_net_ids) + 1}"
        )
    if int(starts[-1]) != num_compact_nodes:
        raise ValueError(
            f"Terminal start {starts[-1]} != compact node count {num_compact_nodes}"
        )

    net_row_by_orig = {int(nid): idx for idx, nid in enumerate(orig_net_ids)}
    sink_nodes = payload["sink_node_id"].detach().cpu().tolist()
    sink_orig_nets = payload["sink_net_index"].detach().cpu().tolist()
    for snode, snid in zip(sink_nodes, sink_orig_nets, strict=True):
        snid_i = int(snid)
        row = net_row_by_orig.get(snid_i)
        if row is None:
            raise ValueError(
                f"sink_net_index entry {snid_i} is not a valid orig_net_id"
            )
        lo = int(starts[row])
        hi = int(starts[row + 1])
        if not (lo <= int(snode) < hi):
            raise ValueError(
                f"sink_node_id {snode} outside net {snid_i} compact slice [{lo}, {hi})"
            )

    payload_cnodes = payload["candidate_node_id"].detach().cpu().tolist()
    state_cnodes = state.candidate_node_id.detach().cpu().tolist()
    if payload_cnodes != state_cnodes:
        raise ValueError(
            "state.candidate_node_id diverged from payload['candidate_node_id']"
        )
    for cnode in state_cnodes:
        if not (0 <= int(cnode) < num_compact_nodes):
            raise ValueError(
                f"candidate_node_id {cnode} outside [0, {num_compact_nodes})"
            )


def splice_compact_payload_and_state(
    *,
    orig_net_ids,
    dirty_packs_by_orig_net,
    payload,
    state,
    old_row_by_cid,
):
    """Splice refreshed dirty-net packs with shifted untouched compact-net slices."""
    is_typed = bool(getattr(state, "is_typed_candidate", False))
    dtype = payload["edge_resistance"].dtype
    payload_device = payload["edge_resistance"].device
    state_device = state.activation_param.device

    old_starts = payload["net_flat_topo_sort_start"].detach().cpu().tolist()
    old_topo = payload["net_flat_topo_sort"].detach().cpu().tolist()
    old_pin_fa = payload["pin_fa"].detach().cpu().tolist()
    old_child_start = payload["flat_pin_to_start"].detach().cpu().tolist()
    old_child_to = payload["flat_pin_to"].detach().cpu().tolist()
    old_edge_r = payload["edge_resistance"].detach().cpu().tolist()
    old_edge_c = payload["edge_capacitance"].detach().cpu().tolist()
    old_node_cap = payload["node_capacitance"].detach().cpu().tolist()
    old_orig_nodes = [
        int(v)
        for v in (payload.get("metadata") or {}).get(
            "compact_node_to_original", [-1] * len(old_topo)
        )
    ]
    old_static_mask = (
        payload["static_rc_edge_mask"].detach().cpu().tolist()
        if "static_rc_edge_mask" in payload
        else [False] * len(old_topo)
    )

    # NOTE: static_payload["sink_net_index"] stores design-level orig_net_id, NOT 0..N-1.
    old_sink_nodes = payload["sink_node_id"].detach().cpu().tolist()
    old_sink_pins = payload["sink_pin_id"].detach().cpu().tolist()
    old_sink_orig_nets = payload["sink_net_index"].detach().cpu().tolist()
    old_sinks_by_orig_net = {}
    for sn, sp, snid in zip(
        old_sink_nodes, old_sink_pins, old_sink_orig_nets, strict=True
    ):
        old_sinks_by_orig_net.setdefault(int(snid), []).append((int(sn), int(sp)))

    old_cand_ids = state.candidate_ids.detach().cpu().tolist()
    old_cand_nodes = payload["candidate_node_id"].detach().cpu().tolist()
    old_cand_net_ids = state.candidate_net_id.detach().cpu().tolist()
    old_acts = state.activation_param.detach().cpu().tolist()
    old_allowed = (
        state.drive_type_allowed.detach().cpu().tolist()
        if getattr(state, "drive_type_allowed", None) is not None
        else None
    )
    old_cands_by_net = {}
    for row, nid in enumerate(old_cand_net_ids):
        old_cands_by_net.setdefault(int(nid), []).append(row)

    new_topo = []
    new_starts = [0]
    new_pin_fa = []
    new_child_start = [0]
    new_child_to = []
    new_edge_r = []
    new_edge_c = []
    new_node_cap = []
    new_orig_nodes = []
    new_static_mask = []
    new_sink_nodes = []
    new_sink_pins = []
    new_sink_orig_nets = []

    new_records = []
    new_cand_ids = []
    new_cand_nodes = []
    new_cand_net_ids = []
    new_cand_group_ids = []
    new_acts = []
    new_allowed = []
    old_to_new_row = {}

    for net_idx, orig_net_id in enumerate(orig_net_ids):
        orig_nid = int(orig_net_id)
        base_offset = len(new_topo)
        pack = dirty_packs_by_orig_net.get(orig_nid)
        if pack is None:
            s = int(old_starts[net_idx])
            e = int(old_starts[net_idx + 1])
            delta = base_offset - s
            for g_old in range(s, e):
                new_topo.append(int(old_topo[g_old]) + delta)
                fa = int(old_pin_fa[g_old])
                new_pin_fa.append(-1 if fa < 0 else fa + delta)
                cs = int(old_child_start[g_old])
                ce = int(old_child_start[g_old + 1])
                for c_pos in range(cs, ce):
                    new_child_to.append(int(old_child_to[c_pos]) + delta)
                new_child_start.append(len(new_child_to))
                new_edge_r.append(float(old_edge_r[g_old]))
                new_edge_c.append(float(old_edge_c[g_old]))
                new_node_cap.append(float(old_node_cap[g_old]))
                new_orig_nodes.append(int(old_orig_nodes[g_old]))
                new_static_mask.append(bool(old_static_mask[g_old]))
            for sn, sp in old_sinks_by_orig_net.get(orig_nid, ()):
                new_sink_nodes.append(int(sn) + delta)
                new_sink_pins.append(int(sp))
                new_sink_orig_nets.append(orig_nid)
            for old_row in old_cands_by_net.get(orig_nid, ()):
                new_row = len(new_records)
                old_to_new_row[old_row] = new_row
                rec = dict(state.candidate_records[old_row])
                new_records.append(rec)
                new_cand_ids.append(int(old_cand_ids[old_row]))
                new_cand_nodes.append(int(old_cand_nodes[old_row]) + delta)
                new_cand_net_ids.append(orig_nid)
                new_cand_group_ids.append(int(rec.get("cpp_net_id", orig_nid)))
                new_acts.append(old_acts[old_row])
                if old_allowed is not None:
                    new_allowed.append(old_allowed[old_row])
            new_starts.append(len(new_topo))
            continue

        new_topo.extend(base_offset + local_idx for local_idx in pack.topo)
        new_pin_fa.extend(
            -1 if fa < 0 else base_offset + int(fa) for fa in pack.pin_fa
        )
        for children in pack.children_by_local:
            new_child_to.extend(base_offset + int(ch) for ch in children)
            new_child_start.append(len(new_child_to))
        new_edge_r.extend(pack.edge_r)
        new_edge_c.extend(pack.edge_c)
        new_node_cap.extend(pack.node_cap)
        new_orig_nodes.extend(pack.orig_node)
        new_static_mask.extend([True] * len(pack.topo))
        for local_sn, sp in pack.sinks:
            new_sink_nodes.append(base_offset + int(local_sn))
            new_sink_pins.append(int(sp))
            new_sink_orig_nets.append(orig_nid)
        for entry in pack.candidates:
            new_row = len(new_records)
            cid_val = int(entry.record["candidate_id"])
            if cid_val in old_row_by_cid:
                old_to_new_row[old_row_by_cid[cid_val]] = new_row
            new_records.append(entry.record)
            new_cand_ids.append(cid_val)
            new_cand_nodes.append(base_offset + int(entry.compact_node_id))
            new_cand_net_ids.append(orig_nid)
            new_cand_group_ids.append(int(entry.record.get("cpp_net_id", orig_nid)))
            if is_typed:
                act_vec = [0.0] * int(state.drive_type_count)
                if entry.active_drive_type is not None:
                    act_vec[int(entry.active_drive_type)] = 1.0
                new_acts.append(act_vec)
            else:
                new_acts.append(1.0 if entry.active_drive_type is not None else 0.0)
            if old_allowed is not None:
                new_allowed.append([bool(v) for v in entry.allowed_mask])
        new_starts.append(len(new_topo))

    group_net_ids = sorted(set(new_cand_group_ids))
    group_index_by_id = {gid: i for i, gid in enumerate(group_net_ids)}
    new_cand_net_index = [group_index_by_id[gid] for gid in new_cand_group_ids]

    act_tensor = torch.tensor(
        new_acts, dtype=state.activation_param.dtype, device=state_device
    )
    cand_node_tensor_state = torch.tensor(
        new_cand_nodes, dtype=torch.long, device=state_device
    )
    if is_typed:
        new_state = TypedDiscreteCandidateState(
            candidate_ids=torch.tensor(
                new_cand_ids, dtype=torch.long, device=state_device
            ),
            candidate_node_id=cand_node_tensor_state,
            candidate_net_id=torch.tensor(
                new_cand_net_ids, dtype=torch.long, device=state_device
            ),
            candidate_net_index=torch.tensor(
                new_cand_net_index, dtype=torch.long, device=state_device
            ),
            net_ids=torch.tensor(group_net_ids, dtype=torch.long, device=state_device),
            activation_param=nn.Parameter(act_tensor),
            drive_type_bsu_index=state.drive_type_bsu_index,
            drive_type_ids=state.drive_type_ids,
            drive_type_master_names=state.drive_type_master_names,
            drive_type_cell_ids=state.drive_type_cell_ids,
            legal_buffer_count=state.legal_buffer_count,
            buffer_main_type_index=state.buffer_main_type_index,
            candidate_records=new_records,
            drive_type_table_metadata=state.drive_type_table_metadata,
            drive_type_allowed=(
                torch.tensor(new_allowed, dtype=torch.bool, device=state_device)
                if new_allowed
                else None
            ),
        ).validate_binary_()
    else:
        new_state = DiscreteCandidateState(
            candidate_ids=torch.tensor(
                new_cand_ids, dtype=torch.long, device=state_device
            ),
            candidate_node_id=cand_node_tensor_state,
            candidate_net_id=torch.tensor(
                new_cand_net_ids, dtype=torch.long, device=state_device
            ),
            candidate_net_index=torch.tensor(
                new_cand_net_index, dtype=torch.long, device=state_device
            ),
            net_ids=torch.tensor(group_net_ids, dtype=torch.long, device=state_device),
            activation_param=nn.Parameter(act_tensor),
            fixed_bsu_index=state.fixed_bsu_index,
            legal_buffer_count=state.legal_buffer_count,
            buffer_main_type_index=state.buffer_main_type_index,
            candidate_records=new_records,
            fixed_buffer_master=getattr(state, "fixed_buffer_master", ""),
        ).validate_binary_()

    num_new_cands = len(new_records)
    new_payload = dict(payload)
    new_payload.pop("candidate_segment_columns", None)
    new_payload.update(
        {
            "net_flat_topo_sort": torch.tensor(
                new_topo, dtype=torch.long, device=payload_device
            ),
            "net_flat_topo_sort_start": torch.tensor(
                new_starts, dtype=torch.long, device=payload_device
            ),
            "pin_fa": torch.tensor(new_pin_fa, dtype=torch.long, device=payload_device),
            "flat_pin_to_start": torch.tensor(
                new_child_start, dtype=torch.long, device=payload_device
            ),
            "flat_pin_to": torch.tensor(
                new_child_to, dtype=torch.long, device=payload_device
            ),
            "edge_resistance": torch.tensor(
                new_edge_r, dtype=dtype, device=payload_device
            ),
            "node_capacitance": torch.tensor(
                new_node_cap, dtype=dtype, device=payload_device
            ),
            "edge_capacitance": torch.tensor(
                new_edge_c, dtype=dtype, device=payload_device
            ),
            "candidate_node_id": torch.tensor(
                new_cand_nodes, dtype=torch.long, device=payload_device
            ),
            "candidate_net_id": torch.tensor(
                new_cand_net_ids, dtype=torch.long, device=payload_device
            ),
            "candidate_bu": torch.zeros(
                num_new_cands, dtype=dtype, device=payload_device
            ),
            "buffer_input_cap": torch.zeros(
                num_new_cands, dtype=dtype, device=payload_device
            ),
            "buffer_delay": torch.zeros(
                num_new_cands, dtype=dtype, device=payload_device
            ),
            "buffer_output_slew": torch.zeros(
                num_new_cands, dtype=dtype, device=payload_device
            ),
            "sink_node_id": torch.tensor(
                new_sink_nodes, dtype=torch.long, device=payload_device
            ),
            "sink_net_index": torch.tensor(
                new_sink_orig_nets, dtype=torch.long, device=payload_device
            ),
            "sink_pin_id": torch.tensor(
                new_sink_pins, dtype=torch.long, device=payload_device
            ),
            "static_rc_edge_mask": torch.tensor(
                new_static_mask, dtype=torch.bool, device=payload_device
            ),
        }
    )
    for key in ("per_size_input_cap", "per_size_delay", "per_size_output_slew"):
        if key in payload and payload[key].ndim == 2 and payload[key].shape[0] > 0:
            new_payload[key] = payload[key][:1].expand(num_new_cands, -1).contiguous()

    new_meta = dict(payload.get("metadata") or {})
    new_meta["compact_node_to_original"] = new_orig_nodes
    new_meta["net_ids"] = list(orig_net_ids)
    new_meta["topology_source"] = "incremental_2n_subnet_refresh"
    new_meta["candidate_count"] = int(num_new_cands)
    new_payload["metadata"] = new_meta

    validate_compact_payload_invariants(
        orig_net_ids=orig_net_ids,
        payload=new_payload,
        state=new_state,
    )
    return new_payload, new_state, old_to_new_row, len(group_net_ids)
