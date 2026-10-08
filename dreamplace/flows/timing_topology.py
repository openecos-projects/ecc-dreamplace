"""ECC live timing topology refresh and freeze helpers."""

import time
from dataclasses import dataclass

import torch


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
    params = context.params
    placedb = context.placedb
    placedb_params = getattr(placedb, "params", None)
    scale_factor = getattr(
        params,
        "scale_factor",
        getattr(placedb_params, "scale_factor", 1.0),
    )
    shift_factor = getattr(
        params,
        "shift_factor",
        getattr(placedb_params, "shift_factor", (0.0, 0.0)),
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
    data = context.data_collections
    if placedb is not None:
        data.buffering_timing_topology_dbu = getattr(placedb, "dbu", None)
        data.buffering_timing_topology_r_unit = getattr(placedb, "r_unit", None)
        data.buffering_timing_topology_c_unit = getattr(placedb, "c_unit", None)
    data.buffering_timing_topology_scale_factor = scale_factor
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
