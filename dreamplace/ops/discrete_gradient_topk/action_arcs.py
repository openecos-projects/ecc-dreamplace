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


def _cell_pin_names(pin_names, pin_start, cell):
    if pin_names is None:
        return None
    begin = int(pin_start[cell].item())
    end = int(pin_start[cell + 1].item())
    return tuple(_decode_pin_name(value) for value in pin_names[begin:end])


def _pair_mapping(
    source_cell,
    target_cell,
    arc_start,
    flat_arc_info,
    pin_start,
    pin_names=None,
    signatures=None,
):
    """Return one target arc per source arc, or a structural failure."""

    cell_count = int(arc_start.numel()) - 1
    if not (0 <= source_cell < cell_count and 0 <= target_cell < cell_count):
        return None, "target_cell", {}

    source_begin = int(arc_start[source_cell].item())
    source_end = int(arc_start[source_cell + 1].item())
    target_begin = int(arc_start[target_cell].item())
    target_end = int(arc_start[target_cell + 1].item())
    if source_begin == source_end:
        return torch.empty(0, dtype=torch.long), None, {}

    source_pin_count = int(pin_start[source_cell + 1] - pin_start[source_cell])
    target_pin_count = int(pin_start[target_cell + 1] - pin_start[target_cell])
    if source_pin_count != target_pin_count:
        return None, "pin_count", {
            "source_pin_count": source_pin_count,
            "target_pin_count": target_pin_count,
        }
    if pin_names is not None:
        source_names = _cell_pin_names(pin_names, pin_start, source_cell)
        target_names = _cell_pin_names(pin_names, pin_start, target_cell)
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

    pin2node = data.pin2node_map.to(device=device, dtype=torch.long)
    output_pins = arcs[:, 1].long()
    safe_output_pins = output_pins.clamp(
        min=0,
        max=max(int(pin2node.numel()) - 1, 0),
    )
    inst_ids = torch.full(
        (num_arcs,),
        -1,
        dtype=torch.long,
        device=device,
    )
    valid_output = (output_pins >= 0) & (output_pins < int(pin2node.numel()))
    if bool(valid_output.any()):
        mapped_nodes = pin2node[safe_output_pins]
        valid_node = (mapped_nodes >= 0) & (mapped_nodes < num_instances)
        inst_ids[valid_output & valid_node] = mapped_nodes[valid_output & valid_node]
    targets_ms = (time.perf_counter() - targets_started_at) * 1000.0

    arc_start = data.cell_id_2_arc_id_start.detach().to(device="cpu", dtype=torch.long)
    flat_arc_info = data.flat_libarc_info.detach().to(device="cpu", dtype=torch.long)
    pin_start = data.cell_id_2_libpin_id_start.detach().to(
        device="cpu", dtype=torch.long
    )
    pin_names = tuple(pin_names) if pin_names is not None else None
    library_key = (
        int(data.cell_id_2_arc_id_start.data_ptr()),
        int(data.cell_id_2_arc_id_start._version),
        int(data.flat_libarc_info.data_ptr()),
        int(data.flat_libarc_info._version),
        int(data.cell_id_2_libpin_id_start.data_ptr()),
        int(data.cell_id_2_libpin_id_start._version),
        None if pin_names is None else hash(pin_names),
    )
    pin_identity_unavailable_reason = None
    if pin_names is not None:
        expected_pin_count = int(pin_start[-1].item()) if pin_start.numel() else 0
        if not pin_names:
            pin_names = None
            pin_identity_unavailable_reason = "missing_flat_lib_pin_names"
        elif len(pin_names) != expected_pin_count:
            pin_identity_unavailable_reason = (
                "flat_lib_pin_names_length_mismatch:"
                f"names={len(pin_names)}:expected={expected_pin_count}"
            )
            pin_names = None
    arc_signatures = None
    arc_cells = arcs[:, 2].detach().to(device="cpu", dtype=torch.long)
    arc_ids = arcs[:, 3].detach().to(device="cpu", dtype=torch.long)

    runtime_key = (
        int(arcs.data_ptr()),
        int(arcs._version),
        tuple(int(value) for value in arcs.shape),
        str(device),
    )
    output_entry = None
    if output_cache is not None:
        output_entry = output_cache.get(runtime_key)
    output_buffer_reused = False

    def output_tensor(name, shape, *, dtype, fill_value=None):
        nonlocal output_entry, output_buffer_reused
        if output_entry is not None:
            value = output_entry.get(name)
            if value is not None and tuple(value.shape) == tuple(shape):
                output_buffer_reused = True
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

    if output_entry is None and output_cache is not None:
        output_entry = {}
        output_cache[runtime_key] = output_entry
    target_arc_ids = output_tensor(
        "target_arc_ids", (num_arcs, num_actions), dtype=torch.long, fill_value=-1
    )
    target_senses = output_tensor(
        "target_senses", (num_arcs, num_actions), dtype=torch.long
    )
    valid_arcs = output_tensor(
        "valid_arcs", (num_arcs, num_actions), dtype=torch.bool
    )
    failure_counts = {
        str(action): {
            "target_cell_missing": 0,
            "pin_count": 0,
            "arc_schema": 0,
            "stale_instance_master": 0,
            "invalid_runtime_arc": 0,
            "pin_identity": 0,
        }
        for action in action_names
    }
    instance_master = current_cells.detach().to(device="cpu", dtype=torch.long)
    target_cells_cpu = target_cells.detach().to(device="cpu", dtype=torch.long)
    candidate_targets_cpu = candidate_table.target_cell_ids.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    target_valid_cpu = target_valid.detach().to(device="cpu", dtype=torch.bool)
    inst_ids_cpu = inst_ids.detach().to(device="cpu", dtype=torch.long)
    pair_loop_started_at = time.perf_counter()

    cell_count = int(arc_start.numel()) - 1
    flat_arc_count = int(flat_arc_info.shape[0])
    valid_inst_rows = (inst_ids_cpu >= 0) & (inst_ids_cpu < num_instances)
    safe_inst_ids = inst_ids_cpu.clamp(min=0, max=max(num_instances - 1, 0))
    valid_source_cells = (arc_cells >= 0) & (arc_cells < cell_count)
    valid_runtime_arcs = (arc_ids >= 0) & (arc_ids < flat_arc_count)
    stale_rows = (
        valid_inst_rows
        & valid_source_cells
        & (instance_master[safe_inst_ids] != arc_cells)
    )
    invalid_runtime_rows = ~(valid_inst_rows & valid_source_cells & valid_runtime_arcs)
    valid_runtime_rows = (
        valid_inst_rows
        & valid_source_cells
        & valid_runtime_arcs
        & ~stale_rows
    )
    target_valid_rows = target_valid_cpu[safe_inst_ids]
    target_valid_rows &= valid_inst_rows.unsqueeze(1)
    safe_arc_ids = arc_ids.clamp(min=0, max=max(flat_arc_count - 1, 0))
    safe_source_cells = arc_cells.clamp(min=0, max=max(cell_count - 1, 0))
    used_cells = torch.unique(
        current_cells.detach().to(device="cpu", dtype=torch.long)
    )
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
    cached_changed_cells = 0
    for action_index, action in enumerate(action_names):
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
            arc_signatures = [
                (int(row[0]), int(row[1]), int(row[4]), int(row[5]))
                for row in flat_arc_info.tolist()
            ]
        for source_cell in changed_cells.tolist():
            source_begin = int(arc_start[source_cell].item())
            source_end = int(arc_start[source_cell + 1].item())
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
            )
            if mapping is None:
                if reason in bad_reason_cells:
                    bad_reason_cells[reason][source_cell] = True
                continue
            if source_end > source_begin:
                mapped_arc_ids[source_begin:source_end] = mapping

        target_for_rows = mapped_arc_ids[safe_arc_ids]
        valid = valid_runtime_rows & target_valid_rows[:, action_index]
        valid &= target_for_rows >= 0
        target_arc_ids[:, action_index] = torch.where(
            valid,
            target_for_rows,
            torch.full_like(target_for_rows, -1),
        ).to(device=device)
        target_senses[:, action_index] = torch.where(
            valid,
            flat_arc_info[target_for_rows.clamp(min=0), 4],
            torch.zeros_like(target_for_rows),
        ).to(device=device)
        valid_arcs[:, action_index] = valid.to(device=device)

        target_valid_column = target_valid_rows[:, action_index]
        failure_counts[action]["target_cell_missing"] = int(
            (valid_inst_rows & ~target_valid_column).sum().item()
        )
        failure_counts[action]["stale_instance_master"] = int(
            (stale_rows & target_valid_column).sum().item()
        )
        failure_counts[action]["invalid_runtime_arc"] = int(
            (invalid_runtime_rows & target_valid_column).sum().item()
        )
        for reason, bad_cells in bad_reason_cells.items():
            if not bool(bad_cells.any()):
                continue
            bad_source_rows = bad_cells[safe_source_cells]
            failure_counts[action][reason] = int(
                (
                    valid_runtime_rows
                    & target_valid_column
                    & bad_source_rows
                    & (target_for_rows < 0)
                )
                .sum()
                .item()
            )

    pair_loop_ms = (time.perf_counter() - pair_loop_started_at) * 1000.0
    assembly_started_at = time.perf_counter()
    counts = output_tensor(
        "counts", (num_instances, num_actions), dtype=torch.long
    )
    bad_counts = output_tensor(
        "bad_counts", (num_instances, num_actions), dtype=torch.long
    )
    valid_inst_rows = inst_ids >= 0
    if bool(valid_inst_rows.any()):
        rows = inst_ids[valid_inst_rows]
        for action_index in range(num_actions):
            counts[:, action_index].scatter_add_(
                0,
                rows,
                torch.ones_like(rows),
            )
            bad_counts[:, action_index].scatter_add_(
                0,
                rows,
                (~valid_arcs[valid_inst_rows, action_index]).long(),
            )
    valid_instances = target_valid & (counts > 0) & (bad_counts == 0)

    action_summary = {}
    sizeable_count = int(sizeable.sum().item())
    for action_index, action in enumerate(action_names):
        action_summary[action] = {
            "valid_arc_count": int(valid_arcs[:, action_index].sum().item()),
            "active_arc_count": num_arcs,
            "valid_instance_count": int(
                valid_instances[:, action_index].sum().item()
            ),
            "mapped_arc_count": int(
                (inst_ids >= 0).sum().item()
            ),
            "sizeable_instance_count": sizeable_count,
            "arc_valid_rate": (
                float(valid_arcs[:, action_index].sum().item()) / num_arcs
                if num_arcs
                else 0.0
            ),
            "instance_valid_rate": (
                float(valid_instances[:, action_index].sum().item())
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
