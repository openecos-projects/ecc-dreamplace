"""Rebuild schema-v3 sparse timing views and hold pin mappings across carrier commits."""

import copy

import numpy as np

from .sparse_timing_schema import (
    INTEGER_COLUMNS,
    OTHER_COLUMNS,
    STATE_KEYS,
    model_digest,
    sparse_digest,
)

_HOLD_CONTEXT_PIN_FIELDS = (
    "start_pin",
    "native_start_pin",
    "launch_pin",
    "endpoint_pin",
    "capture_pin",
)
_HOLD_LAUNCH_PIN_FIELDS = ("start_pin", "launch_pin")


def _old_to_new_pin_map(rebuild_base, carrier) -> np.ndarray:
    pin_id_map = getattr(carrier, "pin_name2id_map", None)
    if not isinstance(pin_id_map, dict):
        pin_id_map = {str(name): idx for idx, name in enumerate(carrier.pin_names)}
    return np.array(
        [pin_id_map.get(str(name), -1) for name in rebuild_base.pin_names],
        dtype=np.int64,
    )


def remap_hold_carrier_metadata(rebuild_base, carrier, old_to_new_pin: np.ndarray | None = None):
    """Remap pin identities in hold_contexts and hold_launches onto the rebuilt carrier."""
    has_contexts = hasattr(rebuild_base, "hold_contexts") and rebuild_base.hold_contexts is not None
    has_launches = hasattr(rebuild_base, "hold_launches") and rebuild_base.hold_launches is not None
    if not has_contexts and not has_launches:
        return
    if old_to_new_pin is None:
        old_to_new_pin = _old_to_new_pin_map(rebuild_base, carrier)

    def _map_pin(pin_val: int, field: str) -> int:
        idx = int(pin_val)
        if idx < 0:
            return idx
        if idx >= len(old_to_new_pin) or old_to_new_pin[idx] < 0:
            raise ValueError(f"unmapped hold timing pin in {field}: {idx}")
        return int(old_to_new_pin[idx])

    if has_contexts:
        remapped_contexts = []
        for row in rebuild_base.hold_contexts:
            item = dict(row)
            for field in _HOLD_CONTEXT_PIN_FIELDS:
                if field in item:
                    item[field] = _map_pin(item[field], field)
            remapped_contexts.append(item)
        carrier.hold_contexts = remapped_contexts

    if has_launches:
        remapped_launches = []
        for row in rebuild_base.hold_launches:
            item = dict(row)
            for field in _HOLD_LAUNCH_PIN_FIELDS:
                if field in item:
                    item[field] = _map_pin(item[field], field)
            remapped_launches.append(item)
        carrier.hold_launches = remapped_launches


def _map_base_cell_arcs(
    rebuild_base,
    carrier,
    old_to_new_pin: np.ndarray,
    equivalent_arc_lut,
) -> tuple[np.ndarray, dict[str, int]]:
    new_node_map = {str(name): idx for idx, name in enumerate(carrier.node_names)}
    old_to_new_node = [new_node_map.get(str(name), -1) for name in rebuild_base.node_names]

    base_cells = np.asarray(rebuild_base.flat_inst_arcs_by_level, dtype=np.int64).reshape(-1, 8)
    new_cells = np.asarray(carrier.flat_inst_arcs_by_level, dtype=np.int64).reshape(-1, 8)
    base_levels = rebuild_base.flat_inst_arcs_by_level_start
    new_levels = carrier.flat_inst_arcs_by_level_start
    base_clk_end = int(base_levels[1]) if len(base_levels) > 1 else 0
    new_clk_end = int(new_levels[1]) if len(new_levels) > 1 else 0

    new_arc_lookup: dict[tuple[int, int, int, int, int], int] = {}
    for new_idx in range(len(new_cells)):
        row = new_cells[new_idx]
        key = (
            1 if new_idx < new_clk_end else 0,
            int(row[7]),
            int(row[0]),
            int(row[1]),
            int(row[3]),
        )
        new_arc_lookup[key] = new_idx

    main_start = rebuild_base.main_id_2_cell_id_start
    inst_main = carrier.inst_main_id
    inst_offset = carrier.inst_libcell_offset
    lib_arcs = rebuild_base.flat_libarc_info
    arc_start = rebuild_base.cell_id_2_arc_id_start
    lut_cache: dict[tuple[int, int, int], int] = {}

    old_arc_to_new = np.full(len(base_cells), -1, dtype=np.int64)
    for old_idx in range(len(base_cells)):
        row = base_cells[old_idx]
        old_owner, old_lut = int(row[7]), int(row[3])
        if old_owner < 0 or old_lut < 0 or old_owner >= len(old_to_new_node):
            continue
        new_owner = old_to_new_node[old_owner]
        if new_owner < 0:
            continue
        is_clk = 1 if old_idx < base_clk_end else 0
        new_src = int(row[0]) if is_clk else int(old_to_new_pin[int(row[0])])
        new_dst = int(old_to_new_pin[int(row[1])])
        if new_src < 0 or new_dst < 0:
            continue
        main_id = int(inst_main[new_owner])
        if main_id < 0:
            continue
        new_cell = int(main_start[main_id]) + int(inst_offset[new_owner])
        new_lut = equivalent_arc_lut(lib_arcs, arc_start, new_cell, old_lut, int(row[6]), lut_cache)
        old_arc_to_new[old_idx] = new_arc_lookup.get(
            (is_clk, new_owner, new_src, new_dst, new_lut), -1
        )
    return old_arc_to_new, new_node_map


def _legal_arc_mask(sense: int, trigger: int) -> int:
    sense_mask = 9 if sense == 1 else (6 if sense == -1 else 15)
    trig_mask = 3 if trigger == 1 else (12 if trigger == -1 else 15)
    return sense_mask & trig_mask


def _dag_longest_path_levels(
    num_states: int,
    cell_edges: np.ndarray,
    net_edges: np.ndarray,
    slew_only_edges: np.ndarray,
) -> np.ndarray:
    levels = np.zeros(num_states, dtype=np.int64)
    if num_states == 0:
        return levels
    parts_src = [e[:, 1] for e in (cell_edges, net_edges, slew_only_edges) if len(e)]
    if not parts_src:
        return levels
    all_src = np.concatenate(parts_src)
    all_dst = np.concatenate(
        [e[:, 2] for e in (cell_edges, net_edges, slew_only_edges) if len(e)]
    )
    changed = np.zeros(num_states, dtype=bool)
    changed[all_src] = True
    for _ in range(num_states + 1):
        active = changed[all_src]
        if not np.any(active):
            return levels
        prev = levels.copy()
        np.maximum.at(levels, all_dst[active], levels[all_src[active]] + 1)
        changed = levels > prev
    raise RuntimeError("ECC sparse timing schema: cycle in rebuilt sparse DAG")


def rebuild_sparse_view_dict(
    rebuild_base,
    carrier,
    raw_view: dict,
    *,
    equivalent_arc_lut,
    old_to_new_pin: np.ndarray | None = None,
) -> dict:
    """Rebuild a timing_sparse or timing_min_sparse payload for the committed carrier."""
    metadata = copy.deepcopy(raw_view["metadata"])
    base_arrays = raw_view["arrays"]

    if old_to_new_pin is None:
        old_to_new_pin = _old_to_new_pin_map(rebuild_base, carrier)
    old_pin_count = len(rebuild_base.pin_names)
    new_pin_count = len(carrier.pin_names)
    new_to_old_pin = np.full(new_pin_count, -1, dtype=np.int64)
    valid_old_pins = np.flatnonzero(old_to_new_pin >= 0)
    new_to_old_pin[old_to_new_pin[valid_old_pins]] = valid_old_pins

    old_arc_to_new, _ = _map_base_cell_arcs(
        rebuild_base, carrier, old_to_new_pin, equivalent_arc_lut
    )

    base_state_pin = base_arrays["state_pin"]
    mapped_state_pin = old_to_new_pin[base_state_pin]
    keep_state = mapped_state_pin >= 0
    kept_base_ids = np.flatnonzero(keep_state)
    base_to_temp = np.full(len(base_state_pin), -1, dtype=np.int64)
    base_to_temp[kept_base_ids] = np.arange(len(kept_base_ids), dtype=np.int64)

    base_domain = base_arrays["state_domain"]
    base_tag = base_arrays["state_tag"]
    base_launch_tag = base_arrays["state_launch_tag"]
    base_source_type = base_arrays["state_source_type"]
    base_role = base_arrays["state_role"]
    base_slew_group = base_arrays["state_slew_group"]
    base_valid = base_arrays["state_valid"]
    base_seed = base_arrays["state_seed"]
    base_reference = base_arrays["state_reference"]
    base_pin_start = base_arrays["pin_state_start"]

    next_temp_id = len(kept_base_ids)
    next_slew_group = int(base_slew_group.max(initial=-1)) + 1

    extra_pin: list[int] = []
    extra_domain: list[int] = []
    extra_tag: list[int] = []
    extra_launch_tag: list[int] = []
    extra_source_type: list[int] = []
    extra_role: list[int] = []
    extra_slew_group: list[int] = []
    extra_valid: list[tuple[int, int]] = []

    extra_cell_edges: list[tuple[int, int, int]] = []
    extra_cell_masks: list[int] = []
    extra_slew_edges: list[tuple[int, int, int]] = []
    extra_slew_masks: list[int] = []
    extra_net_edges: list[tuple[int, int, int]] = []
    extra_net_masks: list[int] = []
    extra_net_adj: list[np.ndarray] = []

    new_net_arcs = np.asarray(carrier.net_flat_arcs, dtype=np.int64).reshape(-1, 2)
    base_net_edges = base_arrays["net_edges"]
    base_net_masks = base_arrays["net_transition_mask"]
    base_net_adj = base_arrays["net_threshold_adjustments"]

    # Bulk-keep unchanged physical (driver, sink) net edges in vectorized NumPy.
    if len(new_net_arcs) and len(base_net_edges):
        arc_old_drv = new_to_old_pin[new_net_arcs[:, 0]]
        arc_old_snk = new_to_old_pin[new_net_arcs[:, 1]]
        existing_pairs = (arc_old_drv >= 0) & (arc_old_snk >= 0)
        active_pair_keys = (
            arc_old_drv[existing_pairs] * old_pin_count + arc_old_snk[existing_pairs]
        )
        edge_old_drv = base_state_pin[base_net_edges[:, 1]]
        edge_old_snk = base_net_edges[:, 0]
        edge_pair_keys = edge_old_drv * old_pin_count + edge_old_snk
        unchanged_net_mask = (
            np.isin(edge_pair_keys, active_pair_keys)
            & keep_state[base_net_edges[:, 1]]
            & keep_state[base_net_edges[:, 2]]
        )
    else:
        unchanged_net_mask = np.zeros(len(base_net_edges), dtype=bool)

    kept_net_idx = np.flatnonzero(unchanged_net_mask)
    split_net_idx = np.flatnonzero(~unchanged_net_mask)

    split_edges_by_pair: dict[tuple[int, int], list[int]] = {}
    default_drv_adj: dict[int, np.ndarray] = {}
    default_adj_row = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    base_pin2node = getattr(rebuild_base, "pin2node_map", None)
    base_inst_main = getattr(rebuild_base, "inst_main_id", None)
    for e_idx in split_net_idx:
        src_st = int(base_net_edges[e_idx, 1])
        dst_st = int(base_net_edges[e_idx, 2])
        if not (keep_state[src_st] and keep_state[dst_st]):
            continue
        old_drv = int(base_state_pin[src_st])
        old_snk = int(base_net_edges[e_idx, 0])
        split_edges_by_pair.setdefault((old_drv, old_snk), []).append(int(e_idx))
        if old_drv not in default_drv_adj:
            if (
                base_pin2node is not None
                and base_inst_main is not None
                and int(base_inst_main[int(base_pin2node[old_snk])]) >= 0
            ):
                default_drv_adj[old_drv] = base_net_adj[e_idx]
            else:
                default_drv_adj[old_drv] = base_net_adj[e_idx]

    old_node_names = set(str(name) for name in rebuild_base.node_names)
    new_buffer_nodes = [
        int(node_id)
        for node_id in getattr(carrier, "inst_topo_ids", ())
        if str(carrier.node_names[int(node_id)]) not in old_node_names
    ]

    if new_buffer_nodes:
        incoming_driver = np.full(new_pin_count, -1, dtype=np.int64)
        if len(new_net_arcs):
            incoming_driver[new_net_arcs[:, 1]] = new_net_arcs[:, 0]

        new_cells = np.asarray(carrier.flat_inst_arcs_by_level, dtype=np.int64).reshape(-1, 8)
        inst_arcs = carrier.inst_flat_arcs
        inst_arcs_start = carrier.inst_flat_arcs_start

        buf_root_old_drv: dict[int, int] = {}
        buf_pin_states: dict[int, dict[int, tuple[int, tuple[int, int]]]] = {}

        def _original_driver_states(old_pin: int) -> dict[int, tuple[int, tuple[int, int]]]:
            s_begin = int(base_pin_start[old_pin])
            s_end = int(base_pin_start[old_pin + 1])
            if s_begin >= s_end:
                return {}
            roles = base_role[s_begin:s_end]
            drv_role = 1 if np.any(roles == 1) else 0
            result: dict[int, tuple[int, tuple[int, int]]] = {}
            for st in range(s_begin, s_end):
                if not keep_state[st] or int(base_role[st]) != drv_role:
                    continue
                vr, vf = int(base_valid[st, 0]), int(base_valid[st, 1])
                if int(base_source_type[st]) != 0 and not (vr or vf):
                    continue
                result[st] = (int(base_to_temp[st]), (vr, vf))
            return result

        def _append_state(
            pin_id: int,
            st_u: int,
            slew_group: int,
            valid_pair: tuple[int, int],
        ) -> int:
            nonlocal next_temp_id
            tid = next_temp_id
            next_temp_id += 1
            extra_pin.append(pin_id)
            extra_domain.append(int(base_domain[st_u]))
            extra_tag.append(int(base_tag[st_u]))
            extra_launch_tag.append(int(base_launch_tag[st_u]))
            extra_source_type.append(int(base_source_type[st_u]))
            extra_role.append(0)
            extra_slew_group.append(slew_group)
            extra_valid.append(valid_pair)
            return tid

        for b_node in new_buffer_nodes:
            a_begin = int(inst_arcs_start[b_node])
            a_end = int(inst_arcs_start[b_node + 1])
            if a_begin >= a_end:
                continue
            b_sched_indices = [int(inst_arcs[pos]) for pos in range(a_begin, a_end)]
            b_arc_rows = [new_cells[idx] for idx in b_sched_indices]
            p_in = int(b_arc_rows[0][0])
            p_out = int(b_arc_rows[0][1])
            p_drv = int(incoming_driver[p_in])
            if p_drv < 0:
                continue
            old_drv = int(new_to_old_pin[p_drv])
            if old_drv >= 0:
                root_old_drv = old_drv
                upstream_states = _original_driver_states(old_drv)
                in_adj = default_drv_adj.get(old_drv, default_adj_row)
            else:
                root_old_drv = buf_root_old_drv.get(p_drv, -1)
                upstream_states = buf_pin_states.get(p_drv, {})
                in_adj = default_adj_row
            if root_old_drv < 0 or not upstream_states:
                continue

            sg_in = next_slew_group
            sg_out = next_slew_group + 1
            next_slew_group += 2
            out_state_map: dict[int, tuple[int, tuple[int, int]]] = {}
            arc_legals = [_legal_arc_mask(int(r[4]), int(r[5])) for r in b_arc_rows]

            for st_u, (up_tid, up_valid) in upstream_states.items():
                src_type = int(base_source_type[st_u])
                t_in = _append_state(p_in, st_u, sg_in, up_valid)
                in_mask = (
                    0
                    if src_type == 0
                    else ((1 if up_valid[0] else 0) | (8 if up_valid[1] else 0))
                )
                extra_net_edges.append((p_in, up_tid, t_in))
                extra_net_masks.append(in_mask)
                extra_net_adj.append(in_adj)

                if src_type == 0:
                    out_valid = (0, 0)
                    t_out = _append_state(p_out, st_u, sg_out, out_valid)
                    out_state_map[st_u] = (t_out, out_valid)
                    for sched_idx, legal in zip(b_sched_indices, arc_legals, strict=True):
                        if legal != 0:
                            extra_slew_edges.append((sched_idx, t_in, t_out))
                            extra_slew_masks.append(legal)
                else:
                    combined_legal = 0
                    for legal in arc_legals:
                        combined_legal |= legal
                    out_r = int(
                        (up_valid[0] and (combined_legal & 1 != 0))
                        or (up_valid[1] and (combined_legal & 4 != 0))
                    )
                    out_f = int(
                        (up_valid[0] and (combined_legal & 2 != 0))
                        or (up_valid[1] and (combined_legal & 8 != 0))
                    )
                    if not (out_r or out_f):
                        continue
                    out_valid = (out_r, out_f)
                    t_out = _append_state(p_out, st_u, sg_out, out_valid)
                    out_state_map[st_u] = (t_out, out_valid)
                    for sched_idx, legal in zip(b_sched_indices, arc_legals, strict=True):
                        cmask = 0
                        for ie in (0, 1):
                            if not up_valid[ie]:
                                continue
                            for oe in (0, 1):
                                bit = 1 << (2 * ie + oe)
                                if (legal & bit) and out_valid[oe]:
                                    cmask |= bit
                        if cmask != 0:
                            extra_cell_edges.append((sched_idx, t_in, t_out))
                            extra_cell_masks.append(cmask)

            buf_root_old_drv[p_out] = root_old_drv
            buf_pin_states[p_out] = out_state_map

        # Reconnect buffer output pins to downstream original sink pins.
        if len(new_net_arcs):
            buf_driven_rows = np.flatnonzero(
                (new_to_old_pin[new_net_arcs[:, 0]] < 0)
                & (new_to_old_pin[new_net_arcs[:, 1]] >= 0)
            )
            for arc_idx in buf_driven_rows:
                new_drv = int(new_net_arcs[arc_idx, 0])
                new_snk = int(new_net_arcs[arc_idx, 1])
                old_snk = int(new_to_old_pin[new_snk])
                root_old_drv = buf_root_old_drv.get(new_drv, -1)
                p_out_states = buf_pin_states.get(new_drv)
                if root_old_drv < 0 or not p_out_states:
                    continue
                for e_idx in split_edges_by_pair.get((root_old_drv, old_snk), ()):
                    st_u = int(base_net_edges[e_idx, 1])
                    st_snk = int(base_net_edges[e_idx, 2])
                    out_entry = p_out_states.get(st_u)
                    if out_entry is None:
                        continue
                    t_out, out_valid = out_entry
                    orig_mask = int(base_net_masks[e_idx])
                    if orig_mask == 0:
                        extra_net_edges.append((new_snk, t_out, int(base_to_temp[st_snk])))
                        extra_net_masks.append(0)
                        extra_net_adj.append(base_net_adj[e_idx])
                    else:
                        avail = (1 if out_valid[0] else 0) | (8 if out_valid[1] else 0)
                        mask = orig_mask & avail
                        if mask != 0:
                            extra_net_edges.append((new_snk, t_out, int(base_to_temp[st_snk])))
                            extra_net_masks.append(mask)
                            extra_net_adj.append(base_net_adj[e_idx])

    # Assemble transient state columns and sort into canonical schema order.
    if extra_pin:
        num_extra = len(extra_pin)
        state_pin = np.concatenate(
            (mapped_state_pin[kept_base_ids], np.asarray(extra_pin, dtype=np.int64))
        )
        state_domain = np.concatenate(
            (base_domain[kept_base_ids], np.asarray(extra_domain, dtype=np.int64))
        )
        state_tag = np.concatenate(
            (base_tag[kept_base_ids], np.asarray(extra_tag, dtype=np.int64))
        )
        state_launch_tag = np.concatenate(
            (base_launch_tag[kept_base_ids], np.asarray(extra_launch_tag, dtype=np.int64))
        )
        state_source_type = np.concatenate(
            (base_source_type[kept_base_ids], np.asarray(extra_source_type, dtype=np.int64))
        )
        state_role = np.concatenate(
            (base_role[kept_base_ids], np.asarray(extra_role, dtype=np.int64))
        )
        raw_slew_group = np.concatenate(
            (base_slew_group[kept_base_ids], np.asarray(extra_slew_group, dtype=np.int64))
        )
        state_valid = np.vstack(
            (base_valid[kept_base_ids], np.asarray(extra_valid, dtype=np.uint8))
        )
        state_seed = np.vstack(
            (base_seed[kept_base_ids], np.zeros((num_extra, 2), dtype=np.uint8))
        )
        extra_ref = np.tile(
            np.array([-9.0e7, -9.0e7, 0.0, 0.0], dtype=np.float64), (num_extra, 1)
        )
        state_reference = np.vstack((base_reference[kept_base_ids], extra_ref))
    else:
        state_pin = mapped_state_pin[kept_base_ids].copy()
        state_domain = base_domain[kept_base_ids].copy()
        state_tag = base_tag[kept_base_ids].copy()
        state_launch_tag = base_launch_tag[kept_base_ids].copy()
        state_source_type = base_source_type[kept_base_ids].copy()
        state_role = base_role[kept_base_ids].copy()
        raw_slew_group = base_slew_group[kept_base_ids].copy()
        state_valid = base_valid[kept_base_ids].copy()
        state_seed = base_seed[kept_base_ids].copy()
        state_reference = base_reference[kept_base_ids].copy()

    num_states = len(state_pin)
    order = np.lexsort((state_source_type, state_tag, state_domain, state_pin))
    temp_to_final = np.empty(num_states, dtype=np.int64)
    temp_to_final[order] = np.arange(num_states, dtype=np.int64)

    state_pin = state_pin[order]
    state_domain = state_domain[order]
    state_tag = state_tag[order]
    state_launch_tag = state_launch_tag[order]
    state_source_type = state_source_type[order]
    state_role = state_role[order]
    state_valid = state_valid[order]
    state_seed = state_seed[order]
    state_reference = state_reference[order]
    if num_states:
        _, state_slew_group = np.unique(raw_slew_group[order], return_inverse=True)
        state_slew_group = state_slew_group.astype(np.int64, copy=False)
    else:
        state_slew_group = np.empty(0, dtype=np.int64)

    pin_state_start = np.r_[
        0, np.cumsum(np.bincount(state_pin, minlength=new_pin_count), dtype=np.int64)
    ].astype(np.int64, copy=False)

    def _assemble_cell_edges(
        base_edge_arr: np.ndarray,
        base_mask_arr: np.ndarray,
        extra_edges_list: list[tuple[int, int, int]],
        extra_masks_list: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(base_edge_arr):
            new_base = old_arc_to_new[base_edge_arr[:, 0]]
            valid_e = (
                (new_base >= 0)
                & keep_state[base_edge_arr[:, 1]]
                & keep_state[base_edge_arr[:, 2]]
            )
            kept_e = base_edge_arr[valid_e]
            kept_edges = np.column_stack(
                (
                    new_base[valid_e],
                    temp_to_final[base_to_temp[kept_e[:, 1]]],
                    temp_to_final[base_to_temp[kept_e[:, 2]]],
                )
            )
            kept_masks = base_mask_arr[valid_e]
        else:
            kept_edges = np.empty((0, 3), dtype=np.int64)
            kept_masks = np.empty(0, dtype=np.int64)
        if extra_edges_list:
            ext_e = np.asarray(extra_edges_list, dtype=np.int64)
            ext_e[:, 1] = temp_to_final[ext_e[:, 1]]
            ext_e[:, 2] = temp_to_final[ext_e[:, 2]]
            ext_m = np.asarray(extra_masks_list, dtype=np.int64)
            return np.vstack((kept_edges, ext_e)), np.concatenate((kept_masks, ext_m))
        return kept_edges, kept_masks

    cell_edges, cell_masks = _assemble_cell_edges(
        base_arrays["cell_edges"],
        base_arrays["cell_transition_mask"],
        extra_cell_edges,
        extra_cell_masks,
    )
    slew_edges, slew_masks = _assemble_cell_edges(
        base_arrays["slew_only_cell_edges"],
        base_arrays["slew_only_cell_transition_mask"],
        extra_slew_edges,
        extra_slew_masks,
    )

    if len(kept_net_idx):
        kn = base_net_edges[kept_net_idx]
        kept_net_edges = np.column_stack(
            (
                old_to_new_pin[kn[:, 0]],
                temp_to_final[base_to_temp[kn[:, 1]]],
                temp_to_final[base_to_temp[kn[:, 2]]],
            )
        )
        kept_net_masks = base_net_masks[kept_net_idx]
        kept_net_adj = base_net_adj[kept_net_idx]
    else:
        kept_net_edges = np.empty((0, 3), dtype=np.int64)
        kept_net_masks = np.empty(0, dtype=np.int64)
        kept_net_adj = np.empty((0, 4), dtype=np.float64)

    if extra_net_edges:
        ext_ne = np.asarray(extra_net_edges, dtype=np.int64)
        ext_ne[:, 1] = temp_to_final[ext_ne[:, 1]]
        ext_ne[:, 2] = temp_to_final[ext_ne[:, 2]]
        ext_nm = np.asarray(extra_net_masks, dtype=np.int64)
        ext_na = np.vstack(extra_net_adj).astype(np.float64, copy=False)
        net_edges = np.vstack((kept_net_edges, ext_ne))
        net_masks = np.concatenate((kept_net_masks, ext_nm))
        net_adj = np.vstack((kept_net_adj, ext_na))
    else:
        net_edges = kept_net_edges
        net_masks = kept_net_masks
        net_adj = kept_net_adj

    if not extra_pin and np.all(keep_state):
        candidate_level = base_arrays["state_level"][order].astype(np.int64, copy=True)
        dag_ok = np.all(candidate_level >= 0) and all(
            len(e) == 0 or np.all(candidate_level[e[:, 1]] < candidate_level[e[:, 2]])
            for e in (cell_edges, net_edges, slew_edges)
        )
        state_level = (
            candidate_level
            if dag_ok
            else _dag_longest_path_levels(num_states, cell_edges, net_edges, slew_edges)
        )
    else:
        state_level = _dag_longest_path_levels(
            num_states, cell_edges, net_edges, slew_edges
        )

    pin_ref_cap = np.zeros((new_pin_count, 2), dtype=np.float64)
    pin_ref_cap[old_to_new_pin[valid_old_pins]] = base_arrays["pin_reference_cap"][
        valid_old_pins
    ]

    base_checks = base_arrays["check_index"]
    if len(base_checks):
        mapped_check_temp = base_to_temp[base_checks[:, 0]]
        if np.any(mapped_check_temp < 0):
            raise RuntimeError("ECC sparse timing schema: endpoint check state lost")
        check_index = base_checks.copy()
        check_index[:, 0] = temp_to_final[mapped_check_temp]
    else:
        check_index = np.empty((0, 7), dtype=np.int64)
    check_values = base_arrays["check_values"].copy()

    base_ctx = base_arrays["context_state"]
    if len(base_ctx):
        mapped_ctx_temp = base_to_temp[base_ctx]
        if np.any(mapped_ctx_temp < 0):
            raise RuntimeError("ECC sparse timing schema: hold context state lost")
        context_state = temp_to_final[mapped_ctx_temp]
    else:
        context_state = np.empty(0, dtype=np.int64)

    raw_arrays = {
        "pin_reference_cap": pin_ref_cap,
        "pin_state_start": pin_state_start,
        "state_pin": state_pin,
        "state_domain": state_domain,
        "state_tag": state_tag,
        "state_launch_tag": state_launch_tag,
        "state_source_type": state_source_type,
        "state_role": state_role,
        "state_level": state_level,
        "state_slew_group": state_slew_group,
        "state_valid": state_valid,
        "state_seed": state_seed,
        "state_reference": state_reference,
        "cell_edges": cell_edges,
        "slew_only_cell_edges": slew_edges,
        "net_edges": net_edges,
        "cell_transition_mask": cell_masks,
        "slew_only_cell_transition_mask": slew_masks,
        "net_transition_mask": net_masks,
        "net_threshold_adjustments": net_adj,
        "check_index": check_index,
        "context_state": context_state,
        "check_values": check_values,
    }

    arrays: dict[str, np.ndarray] = {}
    for name, cols in INTEGER_COLUMNS.items():
        arr = np.ascontiguousarray(raw_arrays[name], dtype=np.int64)
        arrays[name] = arr.reshape((-1, cols) if cols else (-1,))
    for name, (dtype, cols) in OTHER_COLUMNS.items():
        arr = np.ascontiguousarray(raw_arrays[name], dtype=dtype).reshape(-1, cols)
        arrays[name] = arr

    return {
        "metadata": metadata,
        "arrays": arrays,
        "semantic_digest": sparse_digest(arrays, metadata),
        "model_digest": model_digest(arrays, metadata),
        "state_key_digest": sparse_digest(
            {key: arrays[key] for key in STATE_KEYS},
            {
                "domains": metadata["domains"],
                "tags": metadata["tags"],
                "launch_tags": metadata["launch_tags"],
            },
        ),
    }


__all__ = [
    "rebuild_sparse_view_dict",
    "remap_hold_carrier_metadata",
]
