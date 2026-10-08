"""Publish shared route/timing topology once, before evaluating the GP objective."""

from dataclasses import replace

from dreamplace.flows import l_shape_iteration, timing_iteration


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
