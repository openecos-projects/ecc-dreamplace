"""Map runtime timing arcs onto the four discrete master actions.

The timing graph owns one fixed arc topology.  An action changes the Liberty
master, but the graph keeps the current pin/state indices.  This module only
produces target Liberty rows and an all-arcs validity mask; it never rewrites
the topology.
"""

from dataclasses import dataclass

import torch
import time


_PAIR_CACHE_STATS = {"hits": 0, "misses": 0}


@dataclass(frozen=True)
class ActionArcTable:
    target_arc_ids: torch.Tensor
    target_senses: torch.Tensor
    valid_arcs: torch.Tensor
    inst_ids: torch.Tensor
    target_cell_ids: torch.Tensor
    valid_instances: torch.Tensor
    summary: dict


def _signature(row):
    return (
        int(row[0]),
        int(row[1]),
        int(row[4]),
        int(row[5]),
    )


def _decode_pin_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    decode = getattr(value, "decode", None)
    if callable(decode):
        return decode("utf-8", errors="replace")
    return str(value)


def _cell_pin_names(pin_names, pin_start, cell, *, pin_start_list=None, cell_pin_cache=None):
    if pin_names is None:
        return None
    if cell_pin_cache is not None:
        cached = cell_pin_cache[cell]
        if cached is not None:
            return cached
    if pin_start_list is not None:
        begin = int(pin_start_list[cell])
        end = int(pin_start_list[cell + 1])
    else:
        begin = int(pin_start[cell].item())
        end = int(pin_start[cell + 1].item())
    names = tuple(_decode_pin_name(value) for value in pin_names[begin:end])
    if cell_pin_cache is not None:
        cell_pin_cache[cell] = names
    return names


def _pin_names_identity_key(pin_names):
    if pin_names is None:
        return None
    count = len(pin_names)
    if count == 0:
        return ()
    if count <= 64:
        return tuple(_decode_pin_name(value) for value in pin_names)
    return (
        id(pin_names),
        count,
        _decode_pin_name(pin_names[0]),
        _decode_pin_name(pin_names[count // 2]),
        _decode_pin_name(pin_names[-1]),
    )


def _pair_mapping(
    source_cell,
    target_cell,
    arc_start,
    flat_arc_info,
    pin_start,
    pin_names=None,
    signatures=None,
    *,
    arc_start_list=None,
    pin_start_list=None,
    cell_pin_cache=None,
):
    """Return one target arc per source arc, or a structural failure."""

    cell_count = (
        len(arc_start_list) - 1
        if arc_start_list is not None
        else int(arc_start.numel()) - 1
    )
    if not (0 <= source_cell < cell_count and 0 <= target_cell < cell_count):
        return None, "target_cell", {}

    if arc_start_list is not None:
        source_begin = int(arc_start_list[source_cell])
        source_end = int(arc_start_list[source_cell + 1])
        target_begin = int(arc_start_list[target_cell])
        target_end = int(arc_start_list[target_cell + 1])
    else:
        source_begin = int(arc_start[source_cell].item())
        source_end = int(arc_start[source_cell + 1].item())
        target_begin = int(arc_start[target_cell].item())
        target_end = int(arc_start[target_cell + 1].item())
    if source_begin == source_end:
        return torch.empty(0, dtype=torch.long), None, {}

    if pin_start_list is not None:
        source_pin_count = int(pin_start_list[source_cell + 1] - pin_start_list[source_cell])
        target_pin_count = int(pin_start_list[target_cell + 1] - pin_start_list[target_cell])
    else:
        source_pin_count = int(pin_start[source_cell + 1] - pin_start[source_cell])
        target_pin_count = int(pin_start[target_cell + 1] - pin_start[target_cell])
    if source_pin_count != target_pin_count:
        return None, "pin_count", {
            "source_pin_count": source_pin_count,
            "target_pin_count": target_pin_count,
        }
    if pin_names is not None:
        source_names = _cell_pin_names(
            pin_names,
            pin_start,
            source_cell,
            pin_start_list=pin_start_list,
            cell_pin_cache=cell_pin_cache,
        )
        target_names = _cell_pin_names(
            pin_names,
            pin_start,
            target_cell,
            pin_start_list=pin_start_list,
            cell_pin_cache=cell_pin_cache,
        )
        if source_names != target_names:
            return None, "pin_identity", {
                "source_pin_count": source_pin_count,
                "target_pin_count": target_pin_count,
            }

    if signatures is None:
        source_signatures = [
            _signature(row)
            for row in flat_arc_info[source_begin:source_end].tolist()
        ]
        target_signatures = [
            _signature(row)
            for row in flat_arc_info[target_begin:target_end].tolist()
        ]
    else:
        source_signatures = list(signatures[source_begin:source_end])
        target_signatures = list(signatures[target_begin:target_end])
    if source_signatures == target_signatures:
        return (
            torch.arange(target_begin, target_end, dtype=torch.long),
            None,
            {
                "source_arc_count": source_end - source_begin,
                "target_arc_count": target_end - target_begin,
            },
        )
    source_groups = {}
    for offset, signature in enumerate(source_signatures):
        source_groups.setdefault(signature, []).append(source_begin + offset)
    target_groups = {}
    for offset, signature in enumerate(target_signatures):
        target_groups.setdefault(signature, []).append(target_begin + offset)

    if set(source_groups) != set(target_groups):
        return None, "arc_schema", {
            "missing_in_target": len(set(source_groups) - set(target_groups)),
            "missing_in_source": len(set(target_groups) - set(source_groups)),
        }
    for signature, source_ids in source_groups.items():
        if len(source_ids) != len(target_groups[signature]):
            return None, "arc_schema", {
                "duplicate_signature": signature,
                "source_arc_count": len(source_ids),
                "target_arc_count": len(target_groups[signature]),
            }

    mapping = torch.empty(source_end - source_begin, dtype=torch.long)
    for signature, source_ids in source_groups.items():
        target_ids = target_groups[signature]
        for source_id, target_id in zip(source_ids, target_ids, strict=True):
            mapping[source_id - source_begin] = int(target_id)
    return mapping, None, {
        "source_arc_count": source_end - source_begin,
        "target_arc_count": target_end - target_begin,
    }


def _cached_pair_mapping(
    source_cell,
    target_cell,
    arc_start,
    flat_arc_info,
    pin_start,
    pin_names=None,
    cache_owner=None,
    library_key=None,
    signatures=None,
    *,
    arc_start_list=None,
    pin_start_list=None,
    cell_pin_cache=None,
):
    owner = flat_arc_info if cache_owner is None else cache_owner
    cache = getattr(owner, "_lut_action_pair_cache", None)
    if cache is None or cache.get("__library_key__") != library_key:
        cache = {"__library_key__": library_key}
        try:
            owner._lut_action_pair_cache = cache
        except AttributeError:
            cache = None
    key = (int(source_cell), int(target_cell))
    if cache is not None and key in cache:
        _PAIR_CACHE_STATS["hits"] += 1
        return cache[key]
    _PAIR_CACHE_STATS["misses"] += 1
    result = _pair_mapping(
        int(source_cell),
        int(target_cell),
        arc_start,
        flat_arc_info,
        pin_start,
        pin_names,
        signatures,
        arc_start_list=arc_start_list,
        pin_start_list=pin_start_list,
        cell_pin_cache=cell_pin_cache,
    )
    if cache is not None:
        cache[key] = result
    return result


def build_action_arc_table(
    data,
    candidate_table,
    *,
    action_names,
    pin_names=None,
    runtime_arcs=None,
    output_cache=None,
    device=None,
):
    """Build target rows for every active runtime arc and action."""

    build_started_at = time.perf_counter()
    _PAIR_CACHE_STATS["hits"] = 0
    _PAIR_CACHE_STATS["misses"] = 0
    required = (
        "flat_inst_arcs_by_level",
        "inst_cell_id",
        "pin2node_map",
        "cell_id_2_arc_id_start",
        "flat_libarc_info",
        "cell_id_2_libpin_id_start",
        "inst_is_sizeable",
        "inst_vt_mask",
        "flat_libcell_info",
    )
    missing = [name for name in required if getattr(data, name, None) is None]
    if missing:
        raise RuntimeError(
            "lut_action requires missing runtime tensors: " + ", ".join(missing)
        )

    device = torch.device(
        data.inst_cell_id.device if device is None else device
    )
    num_actions = len(action_names)
    num_instances = int(data.inst_cell_id.numel())
    arcs = (
        data.flat_inst_arcs_by_level if runtime_arcs is None else runtime_arcs
    ).to(device=device, dtype=torch.long)
    if arcs.ndim != 2 or int(arcs.shape[1]) < 5:
        raise RuntimeError("flat_inst_arcs_by_level must have at least five columns")
    num_arcs = int(arcs.shape[0])

    current_cells = data.inst_cell_id.to(device=device, dtype=torch.long)
    targets_started_at = time.perf_counter()
    candidate_count = int(candidate_table.target_cell_ids.shape[0])
    safe_current = current_cells.clamp(
        min=0,
        max=max(candidate_count - 1, 0),
    )
    valid_current = (current_cells >= 0) & (
        current_cells < candidate_count
    )
    targets = candidate_table.target_cell_ids.to(
        device=device, dtype=torch.long
    )[safe_current]
    target_cells = torch.where(
        valid_current.unsqueeze(1),
        targets,
        torch.full_like(targets, -1),
    )
    sizeable = data.inst_is_sizeable.to(device=device).bool()
    target_valid = target_cells >= 0
    target_valid &= sizeable.unsqueeze(1)
    current_info_valid = valid_current & (
        safe_current < int(data.flat_libcell_info.shape[0])
    )
    target_valid &= current_info_valid.unsqueeze(1)
    vt_mask = data.inst_vt_mask.to(device=device).bool()
    for action_index in range(2, num_actions):
        valid = target_valid[:, action_index]
        if not bool(valid.any()):
            continue
        instance_rows = torch.nonzero(valid, as_tuple=False).flatten()
        target_vts = data.flat_libcell_info[
            target_cells[instance_rows, action_index].clamp(min=0).to(
                data.flat_libcell_info.device
            ),
            3,
        ].to(device=device, dtype=torch.long)
        allowed = (target_vts >= 0) & (target_vts < int(vt_mask.shape[1]))
        target_valid[instance_rows, action_index] &= allowed
        if bool(allowed.any()):
            checked_rows = instance_rows[allowed]
            target_valid[checked_rows, action_index] &= vt_mask[
                checked_rows,
                target_vts[allowed],
            ]

    runtime_key = (
        int(arcs.data_ptr()),
        tuple(int(value) for value in arcs.shape),
        str(device),
    )
    output_entry = None
    if output_cache is not None:
        output_entry = output_cache.get(runtime_key)
    output_buffer_reused = False
    if output_entry is None and output_cache is not None:
        output_entry = {}
        output_cache[runtime_key] = output_entry

    pin2node = data.pin2node_map
    output_pins = arcs[:, 1]
    topo_key = (
        int(pin2node.data_ptr()),
        int(pin2node._version),
        num_instances,
        num_arcs,
    )
    if (
        output_entry is not None
        and output_entry.get("topo_key") == topo_key
        and "output_pins" in output_entry
        and torch.equal(output_pins, output_entry["output_pins"])
    ):
        inst_ids = output_entry["inst_ids"]
        inst_ids_cpu = output_entry["inst_ids_cpu"]
        valid_inst_rows = output_entry["valid_inst_rows"]
        safe_inst_ids = output_entry["safe_inst_ids"]
        valid_inst_rows_dev = output_entry["valid_inst_rows_dev"]
        rows_dev = output_entry["rows_dev"]
        has_arcs = output_entry["has_arcs"]
        mapped_arc_count = output_entry["mapped_arc_count"]
    else:
        pin2node_dev = pin2node.to(device=device, dtype=torch.long)
        output_pins_long = output_pins.long()
        safe_output_pins = output_pins_long.clamp(
            min=0,
            max=max(int(pin2node_dev.numel()) - 1, 0),
        )
        inst_ids = torch.full(
            (num_arcs,),
            -1,
            dtype=torch.long,
            device=device,
        )
        valid_output = (output_pins_long >= 0) & (output_pins_long < int(pin2node_dev.numel()))
        if bool(valid_output.any()):
            mapped_nodes = pin2node_dev[safe_output_pins]
            valid_node = (mapped_nodes >= 0) & (mapped_nodes < num_instances)
            inst_ids[valid_output & valid_node] = mapped_nodes[valid_output & valid_node]
        inst_ids_cpu = inst_ids.detach().to(device="cpu", dtype=torch.long)
        valid_inst_rows = (inst_ids_cpu >= 0) & (inst_ids_cpu < num_instances)
        safe_inst_ids = inst_ids_cpu.clamp(min=0, max=max(num_instances - 1, 0))
        valid_inst_rows_dev = inst_ids >= 0
        rows_dev = inst_ids[valid_inst_rows_dev] if bool(valid_inst_rows_dev.any()) else inst_ids[:0]
        counts_1d = torch.zeros((num_instances,), dtype=torch.long, device=device)
        if rows_dev.numel():
            counts_1d.scatter_add_(0, rows_dev, torch.ones_like(rows_dev))
        has_arcs = (counts_1d > 0).unsqueeze(1)
        mapped_arc_count = int(rows_dev.numel())
        if output_entry is not None:
            output_entry.update(
                {
                    "topo_key": topo_key,
                    "output_pins": output_pins.clone(),
                    "inst_ids": inst_ids,
                    "inst_ids_cpu": inst_ids_cpu,
                    "valid_inst_rows": valid_inst_rows,
                    "safe_inst_ids": safe_inst_ids,
                    "valid_inst_rows_dev": valid_inst_rows_dev,
                    "rows_dev": rows_dev,
                    "has_arcs": has_arcs,
                    "mapped_arc_count": mapped_arc_count,
                }
            )
    targets_ms = (time.perf_counter() - targets_started_at) * 1000.0

    arc_start = data.cell_id_2_arc_id_start.detach().to(device="cpu", dtype=torch.long)
    flat_arc_info = data.flat_libarc_info.detach().to(device="cpu", dtype=torch.long)
    pin_start = data.cell_id_2_libpin_id_start.detach().to(
        device="cpu", dtype=torch.long
    )
    check_types = getattr(data, "flat_libarc_check_types", None)
    if check_types is not None and check_types.shape != (flat_arc_info.shape[0],):
        raise RuntimeError("lut_action library check classes do not align with library arcs")
    library_key = (
        int(data.cell_id_2_arc_id_start.data_ptr()),
        int(data.cell_id_2_arc_id_start._version),
        int(data.flat_libarc_info.data_ptr()),
        int(data.flat_libarc_info._version),
        int(data.cell_id_2_libpin_id_start.data_ptr()),
        int(data.cell_id_2_libpin_id_start._version),
        _pin_names_identity_key(pin_names),
        None if check_types is None else (int(check_types.data_ptr()), int(check_types._version)),
    )
    pin_identity_unavailable_reason = None
    if pin_names is not None:
        expected_pin_count = int(pin_start[-1].item()) if pin_start.numel() else 0
        if len(pin_names) == 0:
            pin_names = None
            pin_identity_unavailable_reason = "missing_flat_lib_pin_names"
        elif len(pin_names) != expected_pin_count:
            pin_identity_unavailable_reason = (
                "flat_lib_pin_names_length_mismatch:"
                f"names={len(pin_names)}:expected={expected_pin_count}"
            )
            pin_names = None
    arc_cells = arcs[:, 2].detach().to(device="cpu", dtype=torch.long)
    arc_ids = arcs[:, 3].detach().to(device="cpu", dtype=torch.long)

    def output_tensor(name, shape, *, dtype, fill_value=None, reset=True):
        nonlocal output_entry, output_buffer_reused
        if output_entry is not None:
            value = output_entry.get(name)
            if value is not None and tuple(value.shape) == tuple(shape):
                output_buffer_reused = True
                if reset:
                    if fill_value is not None:
                        value.fill_(fill_value)
                    else:
                        value.zero_()
                return value
        if fill_value is None:
            value = torch.zeros(shape, dtype=dtype, device=device)
        else:
            value = torch.full(shape, fill_value, dtype=dtype, device=device)
        if output_entry is not None:
            output_entry[name] = value
        return value

    instance_master = current_cells.detach().to(device="cpu", dtype=torch.long)
    candidate_targets_cpu = candidate_table.target_cell_ids.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    target_valid_cpu = target_valid.detach().to(device="cpu", dtype=torch.bool)
    pair_loop_started_at = time.perf_counter()

    cell_count = int(arc_start.numel()) - 1
    flat_arc_count = int(flat_arc_info.shape[0])
    used_cells = torch.unique(instance_master)
    used_cells = used_cells[(used_cells >= 0) & (used_cells < cell_count)]
    arc_cache = getattr(data, "_lut_action_arc_table_cache", None)
    if (
        arc_cache is None
        or arc_cache.get("library_key") != library_key
        or arc_cache.get("action_count") != num_actions
        or arc_cache.get("flat_arc_count") != flat_arc_count
        or arc_cache.get("cell_count") != cell_count
    ):
        arc_cache = {
            "library_key": library_key,
            "action_count": num_actions,
            "flat_arc_count": flat_arc_count,
            "cell_count": cell_count,
            "arc_start_list": arc_start.tolist(),
            "pin_start_list": pin_start.tolist(),
            "cell_pin_cache": [None] * cell_count,
            "arc_signatures": None,
            "mapped_arc_ids": [
                torch.full((flat_arc_count,), -1, dtype=torch.long)
                for _ in range(num_actions)
            ],
            # -2 means the cell has never been resolved.
            "mapping_target": [
                torch.full((cell_count,), -2, dtype=torch.long)
                for _ in range(num_actions)
            ],
            "bad_reason": [
                {
                    reason: torch.zeros(cell_count, dtype=torch.bool)
                    for reason in ("pin_count", "arc_schema", "pin_identity")
                }
                for _ in range(num_actions)
            ],
        }
        data._lut_action_arc_table_cache = arc_cache
    arc_start_list = arc_cache["arc_start_list"]
    pin_start_list = arc_cache["pin_start_list"]
    cell_pin_cache = arc_cache["cell_pin_cache"]
    arc_signatures = arc_cache["arc_signatures"]
    cached_changed_cells = 0
    for action_index in range(num_actions):
        mapped_arc_ids = arc_cache["mapped_arc_ids"][action_index]
        mapping_target = arc_cache["mapping_target"][action_index]
        bad_reason_cells = arc_cache["bad_reason"][action_index]
        target_for_cells = candidate_targets_cpu[:cell_count, action_index].clone()
        if int(target_for_cells.numel()) < cell_count:
            target_for_cells = torch.cat(
                (
                    target_for_cells,
                    torch.full(
                        (cell_count - int(target_for_cells.numel()),),
                        -1,
                        dtype=torch.long,
                    ),
                )
            )
        target_for_cells[
            (target_for_cells < 0) | (target_for_cells >= cell_count)
        ] = -1
        changed_cells = used_cells[
            target_for_cells[used_cells] != mapping_target[used_cells]
        ]
        cached_changed_cells += int(changed_cells.numel())
        if int(changed_cells.numel()) and arc_signatures is None:
            arc_signatures = [_signature(row) for row in flat_arc_info.tolist()]
            if check_types is not None:
                arc_signatures = [
                    (*signature, int(check_type))
                    for signature, check_type in zip(arc_signatures, check_types.detach().cpu().tolist(), strict=True)
                ]
            arc_cache["arc_signatures"] = arc_signatures
        for source_cell in changed_cells.tolist():
            source_begin = int(arc_start_list[source_cell])
            source_end = int(arc_start_list[source_cell + 1])
            if source_end > source_begin:
                mapped_arc_ids[source_begin:source_end] = -1
            for reason in bad_reason_cells:
                bad_reason_cells[reason][source_cell] = False
            target_cell = int(target_for_cells[source_cell].item())
            mapping_target[source_cell] = target_cell
            if target_cell < 0:
                continue
            mapping, reason, _details = _cached_pair_mapping(
                source_cell,
                target_cell,
                arc_start,
                flat_arc_info,
                pin_start,
                pin_names,
                cache_owner=data,
                library_key=library_key,
                signatures=arc_signatures,
                arc_start_list=arc_start_list,
                pin_start_list=pin_start_list,
                cell_pin_cache=cell_pin_cache,
            )
            if mapping is None:
                if reason in bad_reason_cells:
                    bad_reason_cells[reason][source_cell] = True
                continue
            if source_end > source_begin:
                mapped_arc_ids[source_begin:source_end] = mapping

    def evaluate_rows(
        row_indices,
        inst_master_vec,
        arc_cells_vec,
        arc_ids_vec,
        target_valid_mat,
        action_index,
        mapped_arc_ids,
        bad_reason_cells,
    ):
        r_valid_inst = valid_inst_rows[row_indices]
        r_safe_inst = safe_inst_ids[row_indices]
        r_arc_cells = arc_cells_vec[row_indices]
        r_arc_ids = arc_ids_vec[row_indices]
        r_valid_src = (r_arc_cells >= 0) & (r_arc_cells < cell_count)
        r_valid_rt_arc = (r_arc_ids >= 0) & (r_arc_ids < flat_arc_count)
        r_stale = r_valid_inst & r_valid_src & (inst_master_vec[r_safe_inst] != r_arc_cells)
        r_invalid_rt = ~(r_valid_inst & r_valid_src & r_valid_rt_arc)
        r_valid_rt = r_valid_inst & r_valid_src & r_valid_rt_arc & ~r_stale
        r_tvc = target_valid_mat[r_safe_inst, action_index] & r_valid_inst
        r_safe_arc_ids = r_arc_ids.clamp(min=0, max=max(flat_arc_count - 1, 0))
        r_safe_src_cells = r_arc_cells.clamp(min=0, max=max(cell_count - 1, 0))
        target_for_rows = mapped_arc_ids[r_safe_arc_ids]
        valid = r_valid_rt & r_tvc & (target_for_rows >= 0)
        failures = {
            "target_cell_missing": int((r_valid_inst & ~r_tvc).sum().item()),
            "pin_count": 0,
            "arc_schema": 0,
            "stale_instance_master": int((r_stale & r_tvc).sum().item()),
            "invalid_runtime_arc": int((r_invalid_rt & r_tvc).sum().item()),
            "pin_identity": 0,
        }
        for reason, bad_cells in bad_reason_cells.items():
            if bool(bad_cells.any()):
                failures[reason] = int(
                    (
                        r_valid_rt
                        & r_tvc
                        & bad_cells[r_safe_src_cells]
                        & (target_for_rows < 0)
                    )
                    .sum()
                    .item()
                )
        return target_for_rows, valid, failures

    inc_key = (
        library_key,
        topo_key,
        int(candidate_table.target_cell_ids.data_ptr()),
        int(candidate_table.target_cell_ids._version),
        int(data.inst_is_sizeable._version),
        int(data.inst_vt_mask._version),
    )
    can_incremental = (
        output_entry is not None
        and output_entry.get("inc_key") == inc_key
        and "prev_instance_master" in output_entry
    )
    dirty_arcs = None
    if can_incremental:
        changed_inst_mask = instance_master != output_entry["prev_instance_master"]
        changed_arc_mask = (
            (arc_cells != output_entry["prev_arc_cells"])
            | (arc_ids != output_entry["prev_arc_ids"])
            | (valid_inst_rows & changed_inst_mask[safe_inst_ids])
        )
        if int(changed_arc_mask.sum().item()) * 4 < num_arcs:
            dirty_arcs = torch.nonzero(changed_arc_mask, as_tuple=False).flatten()
        else:
            can_incremental = False

    if can_incremental and dirty_arcs is not None:
        target_arc_ids = output_tensor(
            "target_arc_ids", (num_arcs, num_actions), dtype=torch.long, fill_value=-1, reset=False
        )
        target_senses = output_tensor(
            "target_senses", (num_arcs, num_actions), dtype=torch.long, reset=False
        )
        valid_arcs = output_tensor(
            "valid_arcs", (num_arcs, num_actions), dtype=torch.bool, reset=False
        )
        bad_counts = output_tensor(
            "bad_counts", (num_instances, num_actions), dtype=torch.long, reset=False
        )
        failure_counts = {
            str(action): dict(output_entry["failure_counts"][str(action)])
            for action in action_names
        }
        if dirty_arcs.numel() > 0:
            d_arcs_dev = dirty_arcs.to(device=device)
            d_valid_inst_dev = valid_inst_rows[dirty_arcs].to(device=device)
            d_rows_dev = inst_ids[d_arcs_dev][d_valid_inst_dev]
            for action_index, action in enumerate(action_names):
                mapped_arc_ids = arc_cache["mapped_arc_ids"][action_index]
                bad_reason_cells = arc_cache["bad_reason"][action_index]
                target_for_rows, valid, new_fail = evaluate_rows(
                    dirty_arcs,
                    instance_master,
                    arc_cells,
                    arc_ids,
                    target_valid_cpu,
                    action_index,
                    mapped_arc_ids,
                    bad_reason_cells,
                )
                _, _, old_fail = evaluate_rows(
                    dirty_arcs,
                    output_entry["prev_instance_master"],
                    output_entry["prev_arc_cells"],
                    output_entry["prev_arc_ids"],
                    output_entry["prev_target_valid_cpu"],
                    action_index,
                    mapped_arc_ids,
                    bad_reason_cells,
                )
                old_valid_dev = valid_arcs[d_arcs_dev, action_index].clone()
                new_valid_dev = valid.to(device=device)
                target_arc_ids[d_arcs_dev, action_index] = torch.where(
                    valid, target_for_rows, torch.full_like(target_for_rows, -1)
                ).to(device=device)
                target_senses[d_arcs_dev, action_index] = torch.where(
                    valid,
                    flat_arc_info[target_for_rows.clamp(min=0), 4],
                    torch.zeros_like(target_for_rows),
                ).to(device=device)
                valid_arcs[d_arcs_dev, action_index] = new_valid_dev
                if d_rows_dev.numel():
                    delta_bad = (
                        (~new_valid_dev[d_valid_inst_dev]).long()
                        - (~old_valid_dev[d_valid_inst_dev]).long()
                    )
                    bad_counts[:, action_index].scatter_add_(0, d_rows_dev, delta_bad)
                for k in failure_counts[action]:
                    failure_counts[action][k] += new_fail[k] - old_fail[k]
            output_entry["prev_instance_master"].copy_(instance_master)
            output_entry["prev_arc_cells"].copy_(arc_cells)
            output_entry["prev_arc_ids"].copy_(arc_ids)
            output_entry["prev_target_valid_cpu"].copy_(target_valid_cpu)
            output_entry["failure_counts"] = {
                str(a): dict(failure_counts[str(a)]) for a in action_names
            }
        pair_loop_ms = (time.perf_counter() - pair_loop_started_at) * 1000.0
        assembly_started_at = time.perf_counter()
    else:
        target_arc_ids = output_tensor(
            "target_arc_ids", (num_arcs, num_actions), dtype=torch.long, fill_value=-1
        )
        target_senses = output_tensor(
            "target_senses", (num_arcs, num_actions), dtype=torch.long
        )
        valid_arcs = output_tensor(
            "valid_arcs", (num_arcs, num_actions), dtype=torch.bool
        )
        failure_counts = {}
        all_rows = slice(None)
        for action_index, action in enumerate(action_names):
            mapped_arc_ids = arc_cache["mapped_arc_ids"][action_index]
            bad_reason_cells = arc_cache["bad_reason"][action_index]
            target_for_rows, valid, failure_counts[str(action)] = evaluate_rows(
                all_rows,
                instance_master,
                arc_cells,
                arc_ids,
                target_valid_cpu,
                action_index,
                mapped_arc_ids,
                bad_reason_cells,
            )
            target_arc_ids[:, action_index] = torch.where(
                valid, target_for_rows, torch.full_like(target_for_rows, -1)
            ).to(device=device)
            target_senses[:, action_index] = torch.where(
                valid,
                flat_arc_info[target_for_rows.clamp(min=0), 4],
                torch.zeros_like(target_for_rows),
            ).to(device=device)
            valid_arcs[:, action_index] = valid.to(device=device)

        pair_loop_ms = (time.perf_counter() - pair_loop_started_at) * 1000.0
        assembly_started_at = time.perf_counter()
        bad_counts = output_tensor(
            "bad_counts", (num_instances, num_actions), dtype=torch.long
        )
        if rows_dev.numel():
            for action_index in range(num_actions):
                bad_counts[:, action_index].scatter_add_(
                    0,
                    rows_dev,
                    (~valid_arcs[valid_inst_rows_dev, action_index]).long(),
                )
        if output_entry is not None:
            output_entry["inc_key"] = inc_key
            output_entry["prev_instance_master"] = instance_master.clone()
            output_entry["prev_arc_cells"] = arc_cells.clone()
            output_entry["prev_arc_ids"] = arc_ids.clone()
            output_entry["prev_target_valid_cpu"] = target_valid_cpu.clone()
            output_entry["failure_counts"] = {
                str(a): dict(failure_counts[str(a)]) for a in action_names
            }
    valid_instances = target_valid & has_arcs & (bad_counts == 0)

    action_summary = {}
    sizeable_count = int(sizeable.sum().item())
    valid_arc_counts = [int(x) for x in valid_arcs.sum(dim=0).tolist()]
    valid_inst_counts = [int(x) for x in valid_instances.sum(dim=0).tolist()]
    for action_index, action in enumerate(action_names):
        valid_arc_cnt = valid_arc_counts[action_index]
        valid_inst_cnt = valid_inst_counts[action_index]
        action_summary[action] = {
            "valid_arc_count": valid_arc_cnt,
            "active_arc_count": num_arcs,
            "valid_instance_count": valid_inst_cnt,
            "mapped_arc_count": mapped_arc_count,
            "sizeable_instance_count": sizeable_count,
            "arc_valid_rate": (
                float(valid_arc_cnt) / num_arcs
                if num_arcs
                else 0.0
            ),
            "instance_valid_rate": (
                float(valid_inst_cnt)
                / sizeable_count
                if sizeable_count
                else 0.0
            ),
            "failures": failure_counts[action],
        }

    return ActionArcTable(
        target_arc_ids=target_arc_ids,
        target_senses=target_senses,
        valid_arcs=valid_arcs,
        inst_ids=inst_ids,
        target_cell_ids=target_cells,
        valid_instances=valid_instances,
        summary={
            "artifact": "lut_action_arc_mapping",
            "num_instances": num_instances,
            "num_arcs": num_arcs,
            "pin_identity_checked": pin_names is not None,
            "pin_identity_unavailable_reason": pin_identity_unavailable_reason,
            "pin_identity_fallback": (
                None if pin_names is not None else "arc_signature"
            ),
            "timing_ms": {
                "targets": float(targets_ms),
                "pair_loop": float(pair_loop_ms),
                "assembly": float(
                    (time.perf_counter() - assembly_started_at) * 1000.0
                ),
                "total": float((time.perf_counter() - build_started_at) * 1000.0),
            },
            "pair_cache": dict(_PAIR_CACHE_STATS),
            "cached_changed_cell_count": int(cached_changed_cells),
            "output_buffer_reused": bool(output_buffer_reused),
            "pair_cache_size": (
                len(getattr(flat_arc_info, "_lut_action_pair_cache", {}) or {})
            ),
            "flat_arc_table": {
                "data_ptr": int(flat_arc_info.data_ptr()),
                "version": int(flat_arc_info._version),
                "shape": [int(value) for value in flat_arc_info.shape],
            },
            "actions": action_summary,
        },
    )


__all__ = ["ActionArcTable", "build_action_arc_table"]
