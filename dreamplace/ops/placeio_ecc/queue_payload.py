"""Attach lazy current-tree payloads to either native queue producer."""


def attach_planner_payloads(snapshot, context, timing_model, config, *, load_slack_override):
    from dreamplace.ops.rebuffer import rebuffer_cpp

    # Use the queue's native endpoint window so payload coverage and
    # scheduling include exactly the same path-driver records.
    payload_items = rebuffer_cpp.select_timing_queue_items(snapshot, config)
    snapshot["planner_payload_item_count"] = len(payload_items)
    planner_payloads = []
    focus_updates = []
    for item in payload_items:
        path = [int(pin) for pin in item.get("path_pins", ())]
        endpoint_slack = float(item.get("slack_ps", 0.0))
        root_net_id = int(item["root_net_id"])
        payload = context.native_current_tree_payload(
            root_net_id,
            endpoint_slack_ps=endpoint_slack,
            load_slack_override=load_slack_override(timing_model.active_pydb, path, endpoint_slack),
        )
        if payload.get("status") != "ok":
            continue
        # ``reachable_net_ids`` are active carrier partitions, unlike the
        # snapshot's logical ``net_ids``.  C++ issues one focused request
        # per physical branch; all requests share one current-tree
        # payload, and the root request keeps the complete VGDP bundle.
        focus_net_ids = [int(net_id) for net_id in payload.get("reachable_net_ids", ())]
        item["focus_net_ids"] = focus_net_ids or [root_net_id]
        focus_updates.append(
            {
                "endpoint_pin": int(item["endpoint_pin"]),
                "root_net_id": root_net_id,
                "tree_key": str(item.get("tree_key", "")),
                "focus_net_ids": item["focus_net_ids"],
            }
        )
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
    update_focus = getattr(rebuffer_cpp, "update_native_timing_queue_focus", None)
    if focus_updates and callable(update_focus):
        update_focus(snapshot, focus_updates)
    snapshot["planner_payloads"] = planner_payloads
    return snapshot
