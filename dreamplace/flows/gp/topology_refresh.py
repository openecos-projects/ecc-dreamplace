"""ECC live timing topology refresh and freeze helpers."""


import time
from dataclasses import dataclass
import torch
from typing import Callable
import logging
from dataclasses import replace
from dreamplace.flows.gp import l_shape_iteration, topology_refresh as timing_iteration


@dataclass
class TimingTopologyContext:
    data_collections: object
    op_collections: object
    placedb: object
    params: object
    segment_direct_joint_enabled: bool
    record_stage: object
    record_bytes: object
    refresh_callback: object


def publish_live_timing_topology(context, new_x, new_y, *, update_kind):
    topo_op = context.op_collections.steiner_topo_op
    data = context.data_collections
    data.net_flat_topo_sort = topo_op.net_flat_topo_sort
    data.net_flat_topo_sort_start = topo_op.net_flat_topo_sort_start
    data.pin_fa = topo_op.pin_fa
    data.flat_pin_to = topo_op.flat_pin_to
    data.flat_pin_to_start = topo_op.flat_pin_to_start
    data.flat_pin_from = topo_op.flat_pin_from
    data.buffering_timing_topology = {
        "topology_source": "live_timing_topology",
        "topology_update_kind": str(update_kind),
        "topology_generation": int(getattr(topo_op, "topology_generation", 0)),
        "topology_frozen": bool(getattr(topo_op, "topology_frozen", False)),
        "frozen_topology_generation": getattr(
            topo_op, "frozen_topology_generation", None
        ),
        "net_flat_topo_sort": data.net_flat_topo_sort,
        "net_flat_topo_sort_start": data.net_flat_topo_sort_start,
        "pin_fa": data.pin_fa,
        "flat_pin_to": data.flat_pin_to,
        "flat_pin_from": data.flat_pin_from,
        "node_x": new_x,
        "node_y": new_y,
    }
    topology = data.buffering_timing_topology
    params = context.params
    placedb = context.placedb
    placedb_params = getattr(placedb, "params", None)
    scale_factor = getattr(
        params, "scale_factor", getattr(placedb_params, "scale_factor", 1.0)
    )
    shift_factor = getattr(
        params, "shift_factor", getattr(placedb_params, "shift_factor", (0.0, 0.0))
    )
    shift_x = float(shift_factor[0]) if len(shift_factor) > 0 else 0.0
    shift_y = float(shift_factor[1]) if len(shift_factor) > 1 else 0.0
    if scale_factor is None or float(scale_factor) == 0.0:
        scale_factor = 1.0
    topology["node_x_dbu"] = (
        new_x.to(torch.float64) / float(scale_factor) + shift_x
    ).round()
    topology["node_y_dbu"] = (
        new_y.to(torch.float64) / float(scale_factor) + shift_y
    ).round()
    if placedb is not None:
        topology["dbu"] = placedb.dbu
        data.buffering_timing_topology_dbu = placedb.dbu
        data.buffering_timing_topology_r_unit = getattr(placedb, "r_unit", None)
        data.buffering_timing_topology_c_unit = getattr(placedb, "c_unit", None)
    data.buffering_timing_topology_scale_factor = scale_factor
    return data.buffering_timing_topology


def refresh_live_timing_topology(context, pos):
    if getattr(context.placedb, "gr_sizing", None) is not None:
        pin_pos = context.op_collections.pin_pos_op(pos)
        x, y = pin_pos.chunk(2)
        return {
            "topology_source": "gr_snapshot",
            "topology_update_kind": "gr_snapshot_unchanged",
            "topology_generation": context.placedb.gr_sizing.op.snapshot.identity.generation,
            "node_x": x,
            "node_y": y,
        }
    started = time.perf_counter()
    pin_pos_started = time.perf_counter()
    pin_pos = context.op_collections.pin_pos_op(pos)
    context.record_stage("pin_pos_build_ms", pin_pos_started)
    if pin_pos.device.type != "cpu":
        context.record_bytes("device_to_host_bytes", pin_pos)
    topo_op = context.op_collections.steiner_topo_op
    if bool(getattr(topo_op, "topology_frozen", False)):
        sync_started = time.perf_counter()
        pin_pos_for_topology = pin_pos.cpu().contiguous()
        context.record_stage("pin_pos_device_sync_ms", sync_started)
        op_started = time.perf_counter()
        new_x, new_y = topo_op(pin_pos_for_topology)
        context.record_stage("topology_op_ms", op_started)
        update_kind = "frozen_geometry_forward"
    else:
        sync_started = time.perf_counter()
        pin_pos_for_topology = pin_pos.detach().cpu().contiguous()
        context.record_stage("pin_pos_device_sync_ms", sync_started)
        op_started = time.perf_counter()
        topo_op.rebuild_tree(pin_pos_for_topology)
        context.record_stage("topology_op_ms", op_started)
        new_x = topo_op.newx
        new_y = topo_op.newy
        update_kind = "topology_rebuild"

    topology = publish_live_timing_topology(
        context,
        new_x,
        new_y,
        update_kind=update_kind,
    )
    context.record_stage("topology_refresh_ms", started)
    return topology


def freeze_live_timing_topology(context, pos):
    topo_op = context.op_collections.steiner_topo_op
    if bool(getattr(topo_op, "topology_frozen", False)):
        return int(topo_op.frozen_topology_generation)
    context.refresh_callback(pos)
    generation = int(topo_op.freeze_topology())
    topology = context.data_collections.buffering_timing_topology
    topology["topology_frozen"] = True
    topology["frozen_topology_generation"] = generation
    topology["topology_update_kind"] = "activation_rebuild_and_freeze"
    return generation


def prepare_legacy_net_weight_timing_topology(context, pos):
    topo_op = context.op_collections.steiner_topo_op
    if context.segment_direct_joint_enabled and bool(
        getattr(topo_op, "topology_frozen", False)
    ):
        context.refresh_callback(pos)
        return
    (
        context.data_collections.net_flat_topo_sort,
        context.data_collections.net_flat_topo_sort_start,
        context.data_collections.pin_fa,
        context.data_collections.flat_pin_to,
        context.data_collections.flat_pin_to_start,
        context.data_collections.flat_pin_from,
    ) = topo_op.rebuild_tree(context.op_collections.pin_pos_op(pos))


@dataclass(frozen=True)
class TimingIterationContext:
    enabled: Callable
    uses_net_weight: Callable
    project_weights: Callable
    refresh_topology: Callable
    carrier: Callable
    reset_weights: Callable

def update_gate(context, params, placedb, model, pos, iteration,
                diff_tdp_gate_status, joint_step):
    if (
        context.enabled(params)
        and diff_tdp_gate_status["topology_refresh_due"]
    ):
        if context.uses_net_weight(params):
            context.project_weights(
                params,
                placedb,
                model,
                pos,
                iteration=iteration,
                gate_status=diff_tdp_gate_status,
                topology_prepared=joint_step.topology_prepared,
            )
            model.use_timing_obj = False
        else:
            t_steiner = time.time()
            if not joint_step.topology_prepared:
                with torch.no_grad():
                    context.refresh_topology(pos)
                joint_step.topology_prepared = True
            diff_tdp_update_ms = (time.time() - t_steiner) * 1000
            logging.info(
                "Diff TDP enabled timing objective at iteration=%d "
                "overflow=%.6f threshold=%.6f interval=%d "
                "carrier=%s steiner_update_ms=%.3f",
                int(iteration),
                float(diff_tdp_gate_status["overflow"]),
                float(diff_tdp_gate_status["threshold"]),
                int(diff_tdp_gate_status["interval"]),
                context.carrier(params),
                diff_tdp_update_ms,
            )
    elif (
        context.enabled(params)
        and not diff_tdp_gate_status["timing_objective_active"]
    ):
        context.reset_weights(
            params,
            placedb,
            iteration=iteration,
            reason=diff_tdp_gate_status["reason"],
        )
        logging.debug(
            "Diff TDP gate skipped at iteration=%d reason=%s "
            "overflow=%s threshold=%.6f interval=%d",
            int(iteration),
            diff_tdp_gate_status["reason"],
            diff_tdp_gate_status["overflow"],
            float(diff_tdp_gate_status["threshold"]),
            int(diff_tdp_gate_status["interval"]),
        )


def update_before_objective(
    timing_context,
    route_context,
    params,
    placedb,
    model,
    pos,
    iteration,
    cur_metric,
    l_shape_policy,
    gate_status,
    joint_step,
    publish_topology,
):
    if params.routability_opt_flag and getattr(params, "l_shape_routability_flag", 0):
        route_updated = l_shape_iteration.update_iteration(
            replace(route_context, topology_prepared=joint_step.topology_prepared),
            params,
            placedb,
            model,
            pos,
            iteration,
            cur_metric,
            l_shape_policy,
        )
        if route_updated:
            topo = route_context.op_collections.steiner_topo_op
            publish_topology(topo.newx, topo.newy, update_kind="routing_feedback")
            joint_step.topology_prepared = True
    timing_iteration.update_gate(
        timing_context,
        params,
        placedb,
        model,
        pos,
        iteration,
        gate_status,
        joint_step,
    )
