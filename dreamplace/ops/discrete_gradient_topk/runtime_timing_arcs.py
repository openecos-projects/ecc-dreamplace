"""Retarget fixed-topology Liberty arc rows after a master change."""

import numpy as np
import torch


def map_liberty_arc_transition(
    source_cell,
    target_cell,
    arc_start,
    flat_arc_info,
):
    """Map a fixed source-cell arc row onto a target Liberty cell."""
    source_cell = int(source_cell)
    target_cell = int(target_cell)
    starts = arc_start.detach().to(device="cpu", dtype=torch.long)
    info = flat_arc_info.detach().to(device="cpu", dtype=torch.long)
    cell_count = int(starts.numel()) - 1
    if not (0 <= source_cell < cell_count and 0 <= target_cell < cell_count):
        raise RuntimeError(
            "runtime Liberty arc transition references an invalid cell: "
            f"source={source_cell}, target={target_cell}"
        )

    def grouped_arcs(cell_id):
        begin = int(starts[cell_id].item())
        end = int(starts[cell_id + 1].item())
        groups = {}
        for arc_id in range(begin, end):
            row = info[arc_id]
            signature = (
                int(row[0].item()),
                int(row[1].item()),
                int(row[4].item()),
                int(row[5].item()),
            )
            groups.setdefault(signature, []).append(arc_id)
        return begin, end, groups

    source_begin, source_end, source_groups = grouped_arcs(source_cell)
    target_begin, target_end, target_groups = grouped_arcs(target_cell)
    missing_in_target = sorted(set(source_groups) - set(target_groups))
    missing_in_source = sorted(set(target_groups) - set(source_groups))
    target_expansions = {
        signature: (len(source_groups[signature]), len(target_groups[signature]))
        for signature in sorted(set(source_groups) & set(target_groups))
        if len(target_groups[signature]) > len(source_groups[signature])
    }
    if missing_in_target or missing_in_source or target_expansions:
        raise RuntimeError(
            "fixed timing topology cannot represent the target Liberty arc schema: "
            f"source_cell={source_cell}, target_cell={target_cell}, "
            f"missing_in_target={missing_in_target}, "
            f"missing_in_source={missing_in_source}, "
            f"target_expansions={target_expansions}"
        )

    mapping = torch.empty(source_end - source_begin, dtype=torch.long)
    collapsed_source_arcs = 0
    for signature, source_arc_ids in source_groups.items():
        target_arc_ids = target_groups[signature]
        for rank, source_arc_id in enumerate(source_arc_ids):
            target_rank = min(rank, len(target_arc_ids) - 1)
            mapping[source_arc_id - source_begin] = target_arc_ids[target_rank]
            collapsed_source_arcs += int(rank >= len(target_arc_ids))
    return mapping, {
        "source_cell": source_cell,
        "target_cell": target_cell,
        "source_arc_count": source_end - source_begin,
        "target_arc_count": target_end - target_begin,
        "collapsed_source_arc_count": collapsed_source_arcs,
    }


def sync_timing_arc_lut_rows(data, placedb, timing_op, changed_inst_ids):
    """Retarget fixed-topology timing arcs to the current Liberty masters."""
    required = (
        "pin2node_map",
        "inst_cell_id",
        "cell_id_2_arc_id_start",
        "flat_libarc_info",
    )
    missing = [name for name in required if getattr(data, name, None) is None]
    if missing:
        return {
            "status": "unavailable",
            "reason": "missing_runtime_arc_metadata",
            "missing": missing,
            "updated_row_count": 0,
        }

    pin2node = data.pin2node_map
    inst_cell_id = data.inst_cell_id
    arc_start = data.cell_id_2_arc_id_start
    flat_arc_info = data.flat_libarc_info
    changed = torch.as_tensor(
        changed_inst_ids,
        dtype=torch.long,
        device=inst_cell_id.device,
    ).reshape(-1)
    selected_inst = torch.zeros(
        int(inst_cell_id.numel()),
        dtype=torch.bool,
        device=inst_cell_id.device,
    )
    selected_inst[changed] = True

    table_summaries = {}
    transition_maps = {}
    transition_summaries = {}
    total_updated = 0
    with torch.no_grad():
        for name in ("flat_inst_arcs_by_level", "endpoints_constraint_arcs"):
            arcs = getattr(data, name, None)
            table_summary = {
                "row_count": 0,
                "selected_row_count": 0,
                "updated_row_count": 0,
                "semantic_remap_row_count": 0,
            }
            if not torch.is_tensor(arcs) or arcs.numel() == 0:
                table_summaries[name] = table_summary
                continue
            if arcs.ndim != 2 or int(arcs.shape[1]) < 4:
                raise RuntimeError(f"{name} must contain at least four columns")
            table_summary["row_count"] = int(arcs.shape[0])

            output_pin = arcs[:, 1].long().to(pin2node.device)
            valid_pin = (output_pin >= 0) & (output_pin < int(pin2node.numel()))
            safe_pin = output_pin.clamp(
                min=0,
                max=max(int(pin2node.numel()) - 1, 0),
            )
            node_id = pin2node[safe_pin].long().to(inst_cell_id.device)
            valid_node = valid_pin.to(inst_cell_id.device) & (node_id >= 0) & (
                node_id < int(inst_cell_id.numel())
            )
            selected = valid_node & selected_inst[
                node_id.clamp(
                    min=0,
                    max=max(int(inst_cell_id.numel()) - 1, 0),
                )
            ]
            selected_count = int(selected.sum().detach().cpu().item())
            table_summary["selected_row_count"] = selected_count
            if selected_count == 0:
                table_summaries[name] = table_summary
                continue

            arc_rows = torch.nonzero(selected, as_tuple=False).flatten()
            arc_rows_on_table = arc_rows.to(arcs.device)
            old_arc_id = arcs[arc_rows_on_table, 3].long().to(flat_arc_info.device)
            if not bool(
                ((old_arc_id >= 0) & (old_arc_id < int(flat_arc_info.shape[0])))
                .all()
                .detach()
                .cpu()
                .item()
            ):
                raise RuntimeError(f"{name} contains an invalid Liberty arc id")
            old_arc_info = flat_arc_info[old_arc_id]
            source_cell = old_arc_info[:, 2].long().to(arc_start.device)
            current_cell = inst_cell_id[node_id[arc_rows]].long().to(arc_start.device)
            if not bool(
                (
                    (source_cell >= 0)
                    & (source_cell + 1 < int(arc_start.numel()))
                    & (current_cell >= 0)
                    & (current_cell + 1 < int(arc_start.numel()))
                )
                .all()
                .detach()
                .cpu()
                .item()
            ):
                raise RuntimeError(f"{name} maps to an invalid runtime Liberty cell")
            cell_count = int(arc_start.numel()) - 1
            pair_code = source_cell * cell_count + current_cell
            current_arc_id = torch.empty_like(current_cell)
            old_arc_on_cell_device = old_arc_id.to(current_cell.device)
            for encoded_pair in torch.unique(pair_code).detach().cpu().tolist():
                pair_source = int(encoded_pair) // cell_count
                pair_target = int(encoded_pair) % cell_count
                pair = (pair_source, pair_target)
                if pair not in transition_maps:
                    mapping, mapping_summary = (
                        map_liberty_arc_transition(
                            pair_source,
                            pair_target,
                            arc_start,
                            flat_arc_info,
                        )
                    )
                    transition_maps[pair] = mapping
                    transition_summaries[pair] = mapping_summary
                pair_mask = pair_code == int(encoded_pair)
                source_begin = int(
                    arc_start[pair_source].detach().cpu().item()
                )
                source_offset = old_arc_on_cell_device[pair_mask] - source_begin
                mapping = transition_maps[pair].to(current_cell.device)
                current_arc_id[pair_mask] = mapping[source_offset]

            current_start = arc_start[current_cell].long()
            arc_offset = old_arc_info[:, 3].long().to(current_start.device)
            table_summary["semantic_remap_row_count"] = int(
                (current_arc_id != current_start + arc_offset)
                .sum()
                .detach()
                .cpu()
                .item()
            )
            current_end = arc_start[current_cell + 1].long()
            valid_current_arc = (
                (current_arc_id >= current_start)
                & (current_arc_id < current_end)
                & (current_arc_id >= 0)
                & (current_arc_id < int(flat_arc_info.shape[0]))
            )
            mapped_info = flat_arc_info[current_arc_id.to(flat_arc_info.device)]
            mapped_signature = mapped_info[:, [0, 1, 4, 5]].long()
            old_signature = old_arc_info[:, [0, 1, 4, 5]].long()
            valid_current_arc = valid_current_arc & (
                mapped_info[:, 2].long().to(valid_current_arc.device)
                == current_cell.to(valid_current_arc.device)
            ) & (
                mapped_signature.to(valid_current_arc.device)
                == old_signature.to(valid_current_arc.device)
            ).all(dim=1)
            if not bool(valid_current_arc.all().detach().cpu().item()):
                raise RuntimeError(
                    f"{name} cannot map a resized instance arc by semantic identity"
                )

            mapped_info = flat_arc_info[current_arc_id.to(flat_arc_info.device)]
            arcs[arc_rows_on_table, 2] = current_cell.to(
                device=arcs.device,
                dtype=arcs.dtype,
            )
            arcs[arc_rows_on_table, 3] = current_arc_id.to(
                device=arcs.device,
                dtype=arcs.dtype,
            )
            if int(arcs.shape[1]) >= 5 and int(mapped_info.shape[1]) >= 5:
                arcs[arc_rows_on_table, 4] = mapped_info[:, 4].to(
                    device=arcs.device,
                    dtype=arcs.dtype,
                )
            if int(arcs.shape[1]) >= 6 and int(mapped_info.shape[1]) >= 6:
                arcs[arc_rows_on_table, 5] = mapped_info[:, 5].to(
                    device=arcs.device,
                    dtype=arcs.dtype,
                )

            placedb_arcs = getattr(placedb, name, None)
            if isinstance(placedb_arcs, np.ndarray) and placedb_arcs.shape == tuple(
                arcs.shape
            ):
                rows_cpu = arc_rows.detach().cpu().numpy()
                placedb_arcs[rows_cpu] = arcs[arc_rows_on_table].detach().cpu().numpy()
            table_summary["updated_row_count"] = selected_count
            total_updated += selected_count
            table_summaries[name] = table_summary

    if timing_op is not None:
        timing_device = getattr(
            timing_op,
            "resolved_timing_propagation_device",
            getattr(timing_op, "device", data.flat_inst_arcs_by_level.device),
        )
        timing_op.flat_inst_arcs_by_level = data.flat_inst_arcs_by_level.to(
            timing_device
        )
        timing_op.endpoints_constraint_arcs = data.endpoints_constraint_arcs.to(
            timing_device
        )
        timing_op._cell_aat_static_cache = {}
        if hasattr(timing_op, "_build_device_contract"):
            timing_op.last_device_contract = timing_op._build_device_contract(
                {
                    "flat_inst_arcs_by_level": timing_op.flat_inst_arcs_by_level,
                    "endpoints_constraint_arcs": timing_op.endpoints_constraint_arcs,
                },
                scope="runtime_arc_sync",
            )
            timing_op._assert_device_contract(timing_op.last_device_contract)
    return {
        "status": "ok",
        "updated_row_count": int(total_updated),
        "tables": table_summaries,
        "transition_count": len(transition_summaries),
        "transitions": [
            transition_summaries[pair]
            for pair in sorted(transition_summaries)
        ],
        "timing_static_cache_invalidated": timing_op is not None,
    }
