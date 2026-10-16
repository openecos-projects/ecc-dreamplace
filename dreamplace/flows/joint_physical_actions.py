"""Capture and write the live placement/sizing state at a joint commit boundary."""

import numpy as np
import torch

from dreamplace.ops.buffer_insertion.buffering_lane import _action_digest, _tensor_digest


def final_physical_context(placer, pos, *, iteration):
    placedb = placer.placedb
    params = placer.params
    num_movable = int(placedb.num_movable_nodes)
    num_nodes = int(placedb.num_nodes)
    scale = float(params.scale_factor or 1.0)
    shift_x, shift_y = params.shift_factor
    final_pos = pos.detach().cpu().double()
    # Freeze integer DBU once, before a float32 PlaceDB round-trip can change
    # which side of a half-DBU boundary the optimized coordinate falls on.
    node_x = (final_pos[:num_movable] / scale + shift_x).round().contiguous()
    node_y = (final_pos[num_nodes : num_nodes + num_movable] / scale + shift_y).round().contiguous()
    final_ids = placer.data_collections.inst_cell_id.detach().cpu().long().clone()
    initial_ids = getattr(placer, "segment_joint_initial_inst_cell_id", None)
    if initial_ids is None:
        # The native export is the baseline for this optimizer window. The
        # live PlaceDB cell IDs may already reflect discrete optimizer steps.
        pydb = placedb.pydb
        main_ids = torch.as_tensor(np.asarray(pydb.inst_main_id), dtype=torch.long)
        starts = torch.as_tensor(np.asarray(pydb.main_id_2_cell_id_start), dtype=torch.long)
        offsets = torch.as_tensor(np.asarray(pydb.inst_libcell_offset), dtype=torch.long)
        initial_ids = torch.where(main_ids >= 0, starts[main_ids.clamp(min=0)] + offsets, -1)
    if initial_ids.numel() != final_ids.numel():
        raise RuntimeError("joint initial/final cell-ID lengths differ")

    def decoded(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    widths = placedb.flat_libcell_width
    heights = placedb.flat_libcell_height
    actions = []
    for inst_id in torch.nonzero(final_ids != initial_ids).flatten().tolist():
        before_id = int(initial_ids[inst_id])
        after_id = int(final_ids[inst_id])
        before_area = float(widths[before_id] * heights[before_id])
        after_area = float(widths[after_id] * heights[after_id])
        actions.append(
            {
                "instance_id": inst_id,
                "instance_name": decoded(placedb.node_names[inst_id]),
                "before_cell_id": before_id,
                "after_cell_id": after_id,
                "before_master": decoded(placedb.flat_libcell_names[before_id]),
                "after_master": decoded(placedb.flat_libcell_names[after_id]),
                "before_area_internal": before_area,
                "after_area_internal": after_area,
                "area_delta_internal": after_area - before_area,
            }
        )
    return {
        "iteration": int(iteration),
        "placement_node_x": node_x,
        "placement_node_y": node_y,
        "sizing_cell_ids": final_ids,
        "sizing_actions": actions,
        "sizing_action_count": len(actions),
        "placement_snapshot_digest": _tensor_digest((("node_x", node_x), ("node_y", node_y))),
    }


def commit_joint_physical_actions(engine, request, *, mutation_backend, output_dir):
    """Use the existing native sizing and buffer transactions in one joint boundary.

    Each native transaction owns its rollback. This boundary does not promise
    rollback of an accepted sizing transaction if the subsequent buffer batch fails.
    """
    if _action_digest(request.sizing_actions) != request.sizing_action_digest:
        raise ValueError("joint sizing actions do not match action digest")
    if _action_digest(request.actions) != request.action_digest:
        raise ValueError("buffer commit request actions do not match action digest")
    placedb = engine.placedb
    params = engine.params
    placedb.inst_cell_id = np.asarray(request.sizing_cell_ids, dtype=np.int32)
    scale = float(params.scale_factor)
    shift_x, shift_y = params.shift_factor
    node_x = (np.asarray(request.placement_node_x) - shift_x) * scale
    node_y = (np.asarray(request.placement_node_y) - shift_y) * scale
    placedb.apply(params, node_x, node_y)
    sizing = dict(placedb.last_sizing_writeback_summary or {})
    accepted = int(sizing.get("accepted_count", sizing.get("applied", 0)))
    if accepted != len(request.sizing_actions):
        raise RuntimeError(
            f"joint sizing count mismatch: requested={len(request.sizing_actions)}, "
            f"accepted={accepted}"
        )
    commit = dict(mutation_backend.commit_buffers(request, output_dir=output_dir) or {})
    commit.update(
        {
            "placement_snapshot_digest": request.placement_snapshot_digest,
            "placement_coordinate_count": len(node_x),
            "sizing_action_count": accepted,
            "sizing_action_digest": request.sizing_action_digest,
            "sizing_result": sizing,
            "atomic_across_sizing_and_buffer": False,
        }
    )
    return commit
