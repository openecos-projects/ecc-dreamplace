"""Bridge for the native ``repair_timing`` operator.

C++ owns endpoint/driver queue ordering, current-tree VGDP, candidate frontier
selection, acceptance, restore and stopping.  This module only transports the
current DreamPlace ``TimingPropagation`` state and placeio_ecc overlay into
the callbacks of ``rebuffer_cpp.run_repair_timing_native_snapshot_operator``.
"""

ARTIFACT = "placeio_ecc_timing_repair"
ARTIFACT_VERSION = 1


def _slack_vector(timing_model):
    from ..rebuffer.timing_model_payload import timing_pin_slack

    return timing_pin_slack(timing_model.model).cpu().numpy()


def _nets_on_path(pydb, path):
    nets = []
    for pin in path:
        if pin < 0 or pin >= len(pydb.pin2net_map):
            continue
        net = int(pydb.pin2net_map[pin])
        if net >= 0 and net not in nets:
            nets.append(net)
    return nets


def _path_load_slack_override(pydb, path, endpoint_slack):
    """Assign the endpoint slack to path loads, mirroring annotateLoadSlacks.

    The DP otherwise uses each load's local pin slack, which can be much
    better than the targeted endpoint when the path crosses a multi-input
    cell.  Overriding only the path loads keeps the rest of the net unchanged.
    """

    path_pins = {int(pin) for pin in path}
    override = {}
    for net in _nets_on_path(pydb, path):
        driver = int(pydb.net2driver_pin_map[net])
        begin = int(pydb.flat_net2pin_start_map[net])
        end = int(pydb.flat_net2pin_start_map[net + 1])
        for pin in pydb.flat_net2pin_map[begin:end]:
            pin = int(pin)
            if pin != driver and pin in path_pins:
                override[pin] = float(endpoint_slack)
    return override


def plan_timing_repair(session, timing_model, *, max_passes, config, context=None):
    """Run native ``repair_timing`` on an open ``PlaceIOBufferSession``."""

    if session.state != "open":
        raise RuntimeError("session.begin() must be called before planning")
    if context is None:
        from ..rebuffer.rebuffer_timing import RebufferPlanContext

        context = RebufferPlanContext(session.base_pydb, timing_model, config=config)
    if context.status != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(context.reasons),
            "passes": [],
            "accepted_actions": [],
            "accepted_count": 0,
        }

    from ..rebuffer import rebuffer_cpp
    from .electrical_observation import electrical_observation

    resolved = {**config, "max_passes": int(max_passes), "max_repairs_per_pass": 1}

    def evaluate():
        observation = dict(session.evaluate() or {})
        observation["electrical"] = electrical_observation(
            timing_model,
            max_fanout=resolved.get("max_fanout", 32),
            cap_margin=resolved.get("cap_margin", 0.0),
            slew_margin=resolved.get("slew_margin", 0.0),
        )
        slack = _slack_vector(timing_model)
        names = timing_model.active_pydb.pin_names
        observation["endpoint_slacks_by_name"] = {
            str(names[int(pin)]): float(slack[int(pin)])
            for pin in timing_model.model.end_points.detach().cpu().tolist()
        }
        return observation

    def timing_snapshot():
        # Materialization may replace the TP carrier and pin IDs. Capture
        # endpoints, paths and design views only after that transition.
        views = context.native_queue_views()
        if views.get("status") != "ok":
            return views
        model = timing_model.model
        slack = _slack_vector(timing_model)
        details = timing_model.native_setup_path_details() or {}
        records = []
        for endpoint in model.end_points.detach().cpu().tolist():
            endpoint = int(endpoint)
            endpoint_slack = float(slack[endpoint])
            if endpoint_slack >= 0.0:
                continue
            path = [int(pin) for pin in (details.get(endpoint) or {}).get("path", ())]
            if not path:
                paths = model.get_critical_paths([endpoint], K=1)
                path = [int(pin) for pin in (paths[0] if paths else ())]
            if not path:
                continue
            records.append(
                {
                    "endpoint_pin": endpoint,
                    "endpoint_pin_name": str(timing_model.active_pydb.pin_names[endpoint]),
                    "slack_ps": endpoint_slack,
                    "path_pins": path,
                    "path_timing": context.native_path_context(path),
                }
            )
        snapshot = dict(
            rebuffer_cpp.build_native_timing_queue_snapshot(
                views["design"],
                views["timing"],
                views["liberty"],
                records,
                views["original_net_by_net"],
            )
        )
        # Use the queue's native endpoint window so payload coverage and
        # scheduling include exactly the same path-driver records.
        payload_items = rebuffer_cpp.select_timing_queue_items(snapshot, resolved)
        snapshot["planner_payload_item_count"] = len(payload_items)
        planner_payloads = []
        for item in payload_items:
            path = [int(pin) for pin in item.get("path_pins", ())]
            endpoint_slack = float(item.get("slack_ps", 0.0))
            root_net_id = int(item["root_net_id"])
            payload = context.native_current_tree_payload(
                root_net_id,
                endpoint_slack_ps=endpoint_slack,
                load_slack_override=_path_load_slack_override(
                    timing_model.active_pydb, path, endpoint_slack
                ),
            )
            if payload.get("status") != "ok":
                continue
            # ``reachable_net_ids`` are active carrier partitions, unlike the
            # snapshot's logical ``net_ids``.  C++ issues one focused request
            # per physical branch; all requests share one current-tree
            # payload, and the root request keeps the complete VGDP bundle.
            focus_net_ids = [int(net_id) for net_id in payload.get("reachable_net_ids", ())]
            item["focus_net_ids"] = focus_net_ids or [root_net_id]
            for focus_net_id in item["focus_net_ids"]:
                planner_payloads.append(
                    {
                        "endpoint_pin": int(item["endpoint_pin"]),
                        "root_net_id": root_net_id,
                        "net_id": focus_net_id,
                        "focus_net_id": focus_net_id,
                        # Resolved by C++ only for the rows it visits.
                        "payload": lambda payload=payload, focus=focus_net_id, root=root_net_id: {
                            **payload,
                            "current_tree": payload["current_tree"](),
                            "focus_net_id": -1 if focus == root else focus,
                        },
                    }
                )
        snapshot["planner_payloads"] = planner_payloads
        return snapshot

    def native_apply(actions):
        timing_model.bind_overlay(session.overlay)
        if timing_model.current_graph() is None:
            timing_model.force_full_overlay_runtime()
            if timing_model.current_graph() is None:
                timing_model.evaluate(session.base_pydb, session.overlay)
        raw = {"actions": list(actions)}
        if any("source_parent_node_id" in action for action in raw["actions"]):
            return session.apply(context.map_native_tree_actions(raw))
        return session.apply(context.map_native_actions(raw))

    def restore_and_resync(checkpoint):
        session.restore(checkpoint)
        return session.evaluate()

    result = dict(
        rebuffer_cpp.run_repair_timing_native_snapshot_operator(
            timing_snapshot,
            lambda label: session.checkpoint(label),
            native_apply,
            evaluate,
            restore_and_resync,
            resolved,
        )
    )
    result["accepted_actions"] = session.export_actions()
    result["accepted_count"] = len(result["accepted_actions"])
    result["artifact"] = ARTIFACT
    result["artifact_version"] = ARTIFACT_VERSION
    result["config"] = resolved
    result["native_path_stats"] = dict(timing_model.native_setup_path_statistics())
    result["carrier_rebuild_statistics"] = dict(timing_model.carrier_rebuild_statistics())
    return result


__all__ = ["plan_timing_repair"]
