"""Inspect committed topology and fresh placement views at the repair boundary."""

import hashlib
import json

import numpy as np
import torch

from dreamplace.EvalMetrics import EvalMetrics
from dreamplace.flows.handoff_state import build_continuation_seed
from dreamplace.ops.rebuffer.pydb_adapter import _buffer_metadata


def plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"unsupported handoff artifact value: {type(value).__name__}")


def names(values):
    return [name.decode() if isinstance(name, bytes) else str(name) for name in values]


def topology_counts(placedb):
    pydb = placedb.rawdb.pydb
    buffer_mask, source = _buffer_metadata(pydb, int(pydb.num_nodes))
    if buffer_mask is None:
        raise RuntimeError("midplace requires qualified per-node buffer metadata")
    return {
        "nodes": int(pydb.num_nodes),
        "pins": len(pydb.pin_names),
        "nets": len(pydb.net_names),
        "cell_arcs": len(pydb.flat_inst_arcs_by_level),
        "movable": int(placedb.num_movable_nodes),
        "movable_density_area": float(
            np.dot(
                placedb.node_size_x[: placedb.num_movable_nodes],
                placedb.node_size_y[: placedb.num_movable_nodes],
            )
        ),
        "buffers": int(np.count_nonzero(buffer_mask)),
        "buffer_metadata_source": source,
        "topology_generation": int(getattr(placedb, "topology_generation", 0)),
        "refresh_generation": int(placedb.rawdb.refresh_generation),
    }


def action_identity(stage):
    payload = {
        name: {
            key: result["plan"].get(key, []) for key in ("actions", "resizes", "accepted_actions")
        }
        for name, result in stage["stages"].items()
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=plain).encode()).hexdigest()


def committed_seed(placedb, old_seed):
    """Use newly scaled committed coordinates; retain only old filler positions."""
    count = int(placedb.num_movable_nodes)
    seed = {
        "movable_node_names": names(placedb.node_names[:count]),
        "movable_x": np.asarray(placedb.node_x[:count]).tolist(),
        "movable_y": np.asarray(placedb.node_y[:count]).tolist(),
        "filler_x": list(old_seed["filler_x"]),
        "filler_y": list(old_seed["filler_y"]),
        "fixed_positions": fixed_positions(placedb),
    }
    if "fixed_positions" in old_seed and seed["fixed_positions"] != old_seed["fixed_positions"]:
        old, new = old_seed["fixed_positions"], seed["fixed_positions"]
        missing = sorted(set(old) - set(new))[:4]
        added = sorted(set(new) - set(old))[:4]
        moved = sorted(name for name in set(old) & set(new) if old[name] != new[name])[:4]
        raise RuntimeError(
            "native repair changed a fixed position or namespace:"
            f" count {len(old)}->{len(new)}"
            f" missing={missing} added={added} moved={moved}"
        )
    if "overflow_reference_area" in old_seed:
        seed["overflow_reference_area"] = old_seed["overflow_reference_area"]
    return seed


def validate_position_seed(placer, placedb, seed):
    count = int(placedb.num_movable_nodes)
    if names(placedb.node_names[:count]) != seed["movable_node_names"]:
        raise RuntimeError("committed position seed has a different movable namespace")
    actual = placer.pos[0].detach().cpu().numpy()
    expected = np.concatenate((seed["movable_x"], seed["movable_y"]))
    observed = np.concatenate(
        (actual[:count], actual[placedb.num_nodes : placedb.num_nodes + count])
    )
    if not np.array_equal(expected.astype(observed.dtype), observed):
        raise RuntimeError("fresh placer did not preserve the committed position seed")
    if fixed_positions(placedb, actual) != seed["fixed_positions"]:
        raise RuntimeError("fresh placer did not preserve a fixed position")
    return {
        "movable_positions_checked": count,
        "fixed_positions_checked": len(seed["fixed_positions"]),
        "max_seed_displacement": 0.0,
    }


def fixed_positions(placedb, pos=None):
    x = placedb.node_x if pos is None else pos[: placedb.num_nodes]
    y = placedb.node_y if pos is None else pos[placedb.num_nodes :]
    # Inserted buffers grow the movable range, so a positional slice would move
    # the fixed window and report a namespace change.  Identify fixed nodes by
    # name membership instead.
    movable = set(names(placedb.node_names[: placedb.num_movable_nodes]))
    return {
        name: [float(x[i]), float(y[i])]
        for i, name in enumerate(names(placedb.node_names))
        if i < placedb.num_physical_nodes
        and name not in movable
        # Synthetic blockage obstacles are renumbered when buffers are
        # inserted; they are not physical cells and cannot move.
        and not name.startswith("blockage")
    }


def capture_fillers(placedb, pos):
    seed = build_continuation_seed(
        placedb,
        {},
        pos,
        None,
        pos_to_numpy=lambda value: value.detach().cpu().numpy(),
        decode_node_name=lambda name: names([name])[0],
    )
    seed["fixed_positions"] = fixed_positions(placedb, pos.detach().cpu().numpy())
    return seed


def runtime_topology(placer, placedb, previous_names):
    """Validate CSR identity and insertion participation, without running legalize."""
    data, pydb = placer.data_collections, placedb.rawdb.pydb
    if placer.pos[0].numel() != 2 * placedb.num_nodes:
        raise RuntimeError("fresh placer position size does not match the committed topology")
    if not bool(torch.isfinite(placer.pos[0]).all()):
        raise RuntimeError("fresh placer has non-finite committed positions")
    for key in (
        "pin2node_map",
        "pin2net_map",
        "flat_node2pin_map",
        "flat_node2pin_start_map",
        "flat_net2pin_map",
        "flat_net2pin_start_map",
        "node2fence_region_map",
    ):
        actual = getattr(data, key).detach().cpu().numpy()
        if not np.array_equal(actual, getattr(placedb, key)):
            raise RuntimeError(f"fresh placer has stale {key}")
    all_names = names(placedb.node_names)
    native_ids = {name: i for i, name in enumerate(names(pydb.node_names))}
    buffer_mask, _ = _buffer_metadata(pydb, int(pydb.num_nodes))
    if buffer_mask is None:
        raise RuntimeError("committed topology lost per-node buffer metadata")
    records = []
    for node_id, name in enumerate(all_names):
        # Synthetic blockages are renumbered when buffers are inserted; they
        # are obstacles, never inserted timing cells.
        if name in previous_names or name.startswith("blockage"):
            continue
        raw_id = native_ids[name]
        if not bool(buffer_mask[raw_id]) or node_id >= placedb.num_movable_nodes:
            raise RuntimeError(f"inserted node is not a movable buffer: {name}")
        original = int(placedb.node2orig_node_map[node_id])
        region = int(placedb.node2fence_region_map[node_id])
        if original < 0 or original != int(pydb.node2orig_node_map[raw_id]):
            raise RuntimeError(f"inserted buffer has stale node2orig_node_map: {name}")
        if region < 0 or region != int(pydb.node2fence_region_map[raw_id]):
            raise RuntimeError(f"inserted buffer has stale node2fence_region_map: {name}")
        pin_start = int(placedb.flat_node2pin_start_map[node_id])
        pin_end = int(placedb.flat_node2pin_start_map[node_id + 1])
        if pin_end - pin_start < 2 or float(data.node_areas[node_id]) <= 0:
            raise RuntimeError(f"inserted buffer is missing pins or density area: {name}")
        records.append(
            {
                "name": name,
                "node_id": node_id,
                "native_node_id": raw_id,
                "node2orig_node_map": original,
                "fence_region": region if region < len(placedb.regions) else None,
                "movable": True,
                "node_is_buffer": True,
                "pin_count": pin_end - pin_start,
                "size": [float(data.node_size_x[node_id]), float(data.node_size_y[node_id])],
                "area": float(data.node_areas[node_id]),
                "seed_position": [
                    float(placer.pos[0][node_id]),
                    float(placer.pos[0][placedb.num_nodes + node_id]),
                ],
            }
        )
    return {
        "csr_matches_committed_placedb": True,
        "new_buffers": records,
        "physical_nodes": int(placedb.num_physical_nodes),
        "movable_nodes": int(placedb.num_movable_nodes),
        "timing_consumer_validation": "pending_first_optimizer_step",
        "legalization_input_nodes": int(data.node_size_x.numel()),
    }


def timing_validation(stage):
    """Report measured evaluation modes; fresh commit TP is a separate check."""
    timing = stage["stages"].get("repair_timing")
    if timing is None:
        return {"status": "not_requested", "incremental_full_fallback": None}
    plan = timing["plan"]
    native = plan.get("native_timing_graph_statistics", {})
    full_fallbacks = native.get("full_fallback_count")
    persistent = plan.get("persistent_timing_statistics", {})
    python_fallbacks = int(persistent.get("full_overlay_fallbacks", 0))
    fallback = bool(full_fallbacks or python_fallbacks) if full_fallbacks is not None else None
    return {
        "status": timing["status"],
        "native_candidate_planner": bool(plan.get("native_candidate_planner", False)),
        "evaluation_counts": native.get("evaluation_counts", {}),
        "python_materialization_count": native.get("python_materialization_count"),
        "python_full_overlay_fallbacks": python_fallbacks,
        "native_full_fallbacks": full_fallbacks,
        "incremental_full_fallback": fallback,
        "fresh_commit_verification": timing.get("commit", {}).get("timing_evaluator"),
    }


def post_step_overflow(placer, placedb, eval_ops, iteration):
    metric = EvalMetrics(iteration)
    with torch.no_grad():
        metric.evaluate(
            placedb, {"overflow": eval_ops["overflow"]}, placer.pos[0], placer.data_collections
        )
    return float(metric.overflow.detach().max().cpu())


def step_record(placer, placedb, model, metric, overflow, step, elapsed_ms, buffers):
    result = {
        "step": step,
        "raw_iteration": int(metric.iteration),
        "post_step_overflow": overflow,
        "wall_ms": elapsed_ms,
    }
    for key in (
        "objective",
        "hpwl",
        "wns",
        "tns",
        "timing_objective",
        "slew_violation",
        "cap_violation",
        "l_shape_cost",
        "l_shape_weighted_cost",
    ):
        value = getattr(metric, key, None)
        if value is not None:
            array = np.asarray(
                plain(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
            )
            if not np.isfinite(array).all():
                raise RuntimeError(f"post-handoff step has non-finite {key}")
            result[key] = array.tolist()
    result["timing_objective_enabled"] = bool(model.use_timing_obj)
    result["routability_objective_enabled"] = bool(model.use_l_shape_routability)
    if result.get("objective") is None:
        raise RuntimeError("post-handoff step has no executed objective")
    if model.use_timing_obj:
        terms = model.last_timing_objective_terms
        loss = terms.get("total_loss")
        if loss is None or not np.isfinite(loss):
            raise RuntimeError("enabled timing objective has no finite executed loss")
        result["timing_objective_state"] = terms
        if step == 1:
            timing = model.op_collections.timing_propagation_op
            pin_count = len(placedb.pin2net_map)
            if (
                timing is None
                or timing.num_pins < pin_count
                or not np.array_equal(
                    timing.pin_net[:pin_count].detach().cpu().numpy(), placedb.pin2net_map
                )
            ):
                raise RuntimeError("post-handoff timing consumer has stale physical pin mapping")
            result["timing_physical_pins_checked"] = pin_count
    if model.use_l_shape_routability:
        if result.get("l_shape_cost") is None:
            raise RuntimeError("enabled routability objective has no executed cost")
        if step == 1:
            route_db = model.l_shape_routability_op.placedb
            if not np.array_equal(route_db.pin2node_map, placedb.pin2node_map):
                raise RuntimeError("post-handoff routability consumer has stale pin mapping")
            result["routability_physical_pins_checked"] = len(placedb.pin2node_map)
    owner = getattr(placer, "inflation_s5b1", None)
    if owner is not None:
        result["timing_window_state"] = {
            "completed": owner.schedule.window_count,
            "pending_milestones": list(owner.schedule.pending_milestones),
            "virtual_lane_active": owner.lane is not None,
        }
    inflation = getattr(model, "inflation_state", None)
    if inflation is not None:
        result["inflation_rounds_completed"] = int(inflation.num_area_adjust)
    result["buffer_positions"] = [
        {
            "name": entry["name"],
            "position": [
                float(placer.pos[0][entry["node_id"]]),
                float(placer.pos[0][placedb.num_nodes + entry["node_id"]]),
            ],
            "moved": any(
                float(placer.pos[0][offset + entry["node_id"]]) != seed
                for offset, seed in zip((0, placedb.num_nodes), entry["seed_position"], strict=True)
            ),
            "inside_placement_boundary": (
                placedb.xl
                <= float(placer.pos[0][entry["node_id"]])
                <= placedb.xh - float(placer.data_collections.node_size_x[entry["node_id"]])
                and placedb.yl
                <= float(placer.pos[0][placedb.num_nodes + entry["node_id"]])
                <= placedb.yh - float(placer.data_collections.node_size_y[entry["node_id"]])
            ),
        }
        for entry in buffers
    ]
    if not bool(torch.isfinite(placer.pos[0]).all()):
        raise RuntimeError("post-handoff optimizer produced non-finite positions")
    expected_fixed = fixed_positions(placedb)
    if fixed_positions(placedb, placer.pos[0].detach().cpu().numpy()) != expected_fixed:
        raise RuntimeError("post-handoff optimizer moved a fixed position")
    result["fixed_positions_checked"] = len(expected_fixed)
    return result
