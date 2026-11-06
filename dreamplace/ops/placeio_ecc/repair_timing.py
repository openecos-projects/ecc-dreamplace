"""Bridge for the native ``repair_timing`` operator.

C++ owns endpoint/driver queue ordering, current-tree VGDP, candidate frontier
selection, acceptance, restore and stopping.  This module only transports the
current DreamPlace ``TimingPropagation`` state and placeio_ecc overlay into
the callbacks of ``rebuffer_cpp.run_repair_timing_native_snapshot_operator``.
"""

import os

import numpy as np

from .queue_payload import attach_planner_payloads

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


def _snapshot_array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).reshape(-1).copy()


def _net_pin_sets(pydb):
    starts = _snapshot_array(getattr(pydb, "flat_net2pin_start_map", None))
    flat = _snapshot_array(getattr(pydb, "flat_net2pin_map", None))
    if starts is not None and flat is not None and starts.size >= 1:
        return {
            net: tuple(int(pin) for pin in flat[int(starts[net]) : int(starts[net + 1])])
            for net in range(max(0, starts.size - 1))
        }
    pin2net = _snapshot_array(getattr(pydb, "pin2net_map", None))
    result = {}
    if pin2net is not None:
        for pin, net in enumerate(pin2net):
            result.setdefault(int(net), []).append(pin)
    return {net: tuple(pins) for net, pins in result.items()}


def _timing_snapshot_state(timing_model):
    model = timing_model.model
    pydb = timing_model.active_pydb
    changed_hint = getattr(model, "last_incremental_changed_pins", None)
    if changed_hint is not None:
        if hasattr(changed_hint, "detach"):
            changed_hint = changed_hint.detach()
        if hasattr(changed_hint, "cpu"):
            changed_hint = changed_hint.cpu()
        changed_hint = frozenset(int(pin) for pin in np.asarray(changed_hint).reshape(-1))
        return {
            "pin_names": tuple(str(name) for name in pydb.pin_names),
            "pin2net": _snapshot_array(getattr(pydb, "pin2net_map", None)),
            "net_drivers": _snapshot_array(getattr(pydb, "net2driver_pin_map", None)),
            "net_pins": None,
            "arrays": None,
            "delta_pins": changed_hint,
        }
    fields = (
        "pin_rAAT_live_snapshot",
        "pin_fAAT_live_snapshot",
        "pin_rRAT_live_snapshot",
        "pin_fRAT_live_snapshot",
        "pin_rtran_live",
        "pin_ftran_live",
        "pin_slack_live_snapshot",
        "pin_net_cap_rise_live",
        "pin_net_cap_fall_live",
        "pin_net_delay_rise_live",
        "pin_net_delay_fall_live",
        "pin_net_impulse_rise_live",
        "pin_net_impulse_fall_live",
    )
    return {
        "pin_names": tuple(str(name) for name in pydb.pin_names),
        "pin2net": _snapshot_array(getattr(pydb, "pin2net_map", None)),
        "net_drivers": _snapshot_array(getattr(pydb, "net2driver_pin_map", None)),
        "net_pins": _net_pin_sets(pydb),
        "arrays": {name: _snapshot_array(getattr(model, name, None)) for name in fields},
        "delta_pins": None,
    }


def _changed_snapshot_pins(previous, current):
    """Return the pins whose timing or local net topology can affect a path."""
    if previous is None:
        return None
    old_names, new_names = previous["pin_names"], current["pin_names"]
    common = min(len(old_names), len(new_names))
    if old_names[:common] != new_names[:common] or len(new_names) < len(old_names):
        return None
    if current.get("delta_pins") is not None:
        changed = set(current["delta_pins"])
        affected_nets = set()
        for kind, old_values, new_values in (
            ("pin2net", previous.get("pin2net"), current.get("pin2net")),
            ("net_drivers", previous.get("net_drivers"), current.get("net_drivers")),
        ):
            if old_values is None or new_values is None:
                if old_values is not None or new_values is not None:
                    return None
                continue
            limit = min(old_values.size, new_values.size)
            different = np.flatnonzero(old_values[:limit] != new_values[:limit])
            if kind == "pin2net":
                affected_nets.update(int(value) for value in old_values[different])
                affected_nets.update(int(value) for value in new_values[different])
                changed.update(int(value) for value in different)
            else:
                affected_nets.update(int(value) for value in different)
                changed.update(int(value) for value in old_values[different])
                changed.update(int(value) for value in new_values[different])
            if old_values.size != new_values.size:
                if kind == "pin2net":
                    changed.update(range(min(old_values.size, new_values.size), new_values.size))
                else:
                    affected_nets.update(range(limit, max(old_values.size, new_values.size)))
        if affected_nets:
            for values in (previous.get("pin2net"), current.get("pin2net")):
                if values is not None:
                    changed.update(
                        int(pin)
                        for pin in np.flatnonzero(np.isin(values, list(affected_nets)))
                    )
        changed.update(range(common, len(new_names)))
        return changed
    if previous.get("arrays") is None or current.get("arrays") is None:
        return None
    changed = set(range(common, len(new_names)))
    count = len(new_names)
    for name, current_values in current["arrays"].items():
        previous_values = previous["arrays"].get(name)
        if current_values is None or previous_values is None:
            if current_values is not None or previous_values is not None:
                return None
            continue
        limit = min(count, current_values.size, previous_values.size)
        if limit:
            different = (
                current_values[:limit] != previous_values[:limit]
            ) | np.isnan(current_values[:limit]) | np.isnan(previous_values[:limit])
            changed.update(int(pin) for pin in np.flatnonzero(different))
        if current_values.size > previous_values.size:
            changed.update(range(previous_values.size, min(count, current_values.size)))

    affected_nets = set()
    for kind, old_values, new_values in (
        ("pin2net", previous.get("pin2net"), current.get("pin2net")),
        ("net_drivers", previous.get("net_drivers"), current.get("net_drivers")),
    ):
        if old_values is None or new_values is None:
            if old_values is not None or new_values is not None:
                return None
            continue
        limit = min(old_values.size, new_values.size)
        different = np.flatnonzero(old_values[:limit] != new_values[:limit])
        if kind == "pin2net":
            affected_nets.update(int(net) for net in old_values[different])
            affected_nets.update(int(net) for net in new_values[different])
            affected_nets.update(int(net) for net in old_values[limit:])
            affected_nets.update(int(net) for net in new_values[limit:])
        else:
            affected_nets.update(int(net) for net in different)
            affected_nets.update(range(limit, max(old_values.size, new_values.size)))
    for net in affected_nets:
        changed.update(previous["net_pins"].get(net, ()))
        changed.update(current["net_pins"].get(net, ()))
    return changed


_QUEUE_ORACLE_FIELDS = (
    "endpoint_pin",
    "endpoint_pin_name",
    "path_pins",
    "root_net_id",
    "driver_pin_id",
    "net_ids",
    "tree_key",
)


def _queue_item_oracle_value(item, field):
    value = item.get(field)
    if field in {"path_pins", "net_ids"}:
        return tuple(int(entry) for entry in (value or ()))
    if field in {"endpoint_pin", "root_net_id", "driver_pin_id"}:
        return None if value is None else int(value)
    return "" if value is None else str(value)


def _queue_items_for_oracle(snapshot, selector):
    """Read the canonical native item order from either queue representation."""

    if snapshot.get("_native_queue_state") is not None:
        # State-only snapshots intentionally omit Python ``items``.  Selecting
        # an unbounded window exposes the same sorted C++ items without
        # materializing the state on the Python side.
        return list(
            selector(
                snapshot,
                {
                    "native_tree_round_robin": False,
                    "native_tree_round_robin_endpoint_window": 0,
                    "retired_endpoints": [],
                },
            )
        )
    return list(snapshot.get("items", ()))


def _assert_queue_snapshot_oracle(actual, reference, selector, config):
    """Compare local queue construction with a full native reconstruction.

    The comparison deliberately stops at the queue contract.  ``focus_net_ids``
    is assigned after this check while the planner expands a selected tree, so
    it is not part of the snapshot identity here.
    """

    actual_items = _queue_items_for_oracle(actual, selector)
    reference_items = _queue_items_for_oracle(reference, selector)
    if len(actual_items) != len(reference_items):
        raise RuntimeError(
            "native timing snapshot oracle item-count mismatch: "
            f"actual={len(actual_items)} reference={len(reference_items)}"
        )
    for index, (actual_item, reference_item) in enumerate(
        zip(actual_items, reference_items, strict=True)
    ):
        for field in _QUEUE_ORACLE_FIELDS:
            actual_value = _queue_item_oracle_value(actual_item, field)
            reference_value = _queue_item_oracle_value(reference_item, field)
            if actual_value != reference_value:
                raise RuntimeError(
                    "native timing snapshot oracle item mismatch at "
                    f"index={index} field={field}: "
                    f"actual={actual_value!r} reference={reference_value!r}"
                )
        actual_slack = float(actual_item.get("slack_ps", 0.0))
        reference_slack = float(reference_item.get("slack_ps", 0.0))
        if not np.isclose(actual_slack, reference_slack, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(
                "native timing snapshot oracle slack mismatch at "
                f"index={index}: actual={actual_slack!r} reference={reference_slack!r}"
            )

    selected_actual = list(selector(actual, config))
    selected_reference = list(selector(reference, config))
    if len(selected_actual) != len(selected_reference):
        raise RuntimeError(
            "native timing snapshot oracle window-count mismatch: "
            f"actual={len(selected_actual)} reference={len(selected_reference)}"
        )
    for index, (actual_item, reference_item) in enumerate(
        zip(selected_actual, selected_reference, strict=True)
    ):
        for field in _QUEUE_ORACLE_FIELDS:
            actual_value = _queue_item_oracle_value(actual_item, field)
            reference_value = _queue_item_oracle_value(reference_item, field)
            if actual_value != reference_value:
                raise RuntimeError(
                    "native timing snapshot oracle window mismatch at "
                    f"index={index} field={field}: "
                    f"actual={actual_value!r} reference={reference_value!r}"
                )
        if not np.isclose(
            float(actual_item.get("slack_ps", 0.0)),
            float(reference_item.get("slack_ps", 0.0)),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError(
                "native timing snapshot oracle window slack mismatch at "
                f"index={index}"
            )


def plan_timing_repair(session, timing_model, *, max_passes, config, context=None):
    """Run native ``repair_timing`` on an open ``PlaceIOBufferSession``."""

    if session.state != "open":
        raise RuntimeError("session.begin() must be called before planning")
    if config.get("timing_scheduler") == "openroad_legacy":
        from .native_legacy_timing import plan_legacy_timing_repair

        return plan_legacy_timing_repair(
            session, timing_model, max_passes=max_passes, config=config, context=context
        )
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
    local_snapshot_cache = {
        "state": None,
        "records": {},
        "native_snapshot": None,
        "active_endpoint_pins": set(),
    }
    local_snapshot_statistics = {
        "enabled": os.environ.get("REBUFFER_LOCAL_SNAPSHOT", "") == "1"
        or bool(config.get("repair_local_snapshot", False)),
        "oracle_enabled": os.environ.get("REBUFFER_SNAPSHOT_ORACLE", "") == "1"
        or bool(config.get("repair_snapshot_oracle", False)),
        "native_update_enabled": (
            os.environ.get("REBUFFER_NATIVE_LOCAL_SNAPSHOT_UPDATE", "") == "1"
            or bool(config.get("repair_native_local_snapshot_update", False))
        ),
        "native_context_reuse_enabled": (
            os.environ.get("REBUFFER_NATIVE_QUEUE_CONTEXT_REUSE", "") == "1"
            or bool(config.get("repair_native_queue_context_reuse", False))
        ),
        "snapshot_count": 0,
        "native_graph_queue_count": 0,
        "oracle_check_count": 0,
        "changed_pin_counts": [],
        "reused_record_counts": [],
        "rebuilt_record_counts": [],
        "queue_context_reuse_count": 0,
        "queue_context_rebuild_count": 0,
        "queue_design_reuse_count": 0,
        "queue_liberty_reuse_count": 0,
        "gradient_ordering_enabled": (
            os.environ.get("REBUFFER_GRADIENT_ORDERING", "") == "1"
            or os.environ.get("REBUFFER_GRADIENT_WINDOW_TIEBREAK", "") == "1"
            or bool(config.get("repair_gradient_ordering", False))
            or bool(config.get("repair_gradient_window_tiebreak", False))
        ),
        "gradient_score_status": "disabled",
        "gradient_score_failures": [],
        "gradient_ranking_oracle_enabled": (
            os.environ.get("REBUFFER_GRADIENT_RANKING_ORACLE", "") == "1"
            or bool(config.get("repair_gradient_ranking_oracle", False))
        ),
        "gradient_ranking_oracle_max_paths": int(
            config.get(
                "repair_gradient_ranking_oracle_max_paths",
                os.environ.get("REBUFFER_GRADIENT_RANKING_ORACLE_MAX_PATHS", "32")
                or 32,
            )
        ),
        "gradient_ranking_oracle_status": "disabled",
        "gradient_ranking_oracle_path_count": 0,
        "gradient_ranking_oracle_top1_matches": 0,
        "gradient_ranking_oracle_common_drivers": 0,
        "gradient_ranking_oracle_pairwise_comparisons": 0,
        "gradient_ranking_oracle_pairwise_agreements": 0,
        "gradient_ranking_oracle_failures": [],
    }
    gradient_ordering_enabled = local_snapshot_statistics["gradient_ordering_enabled"]
    gradient_ranking_oracle_enabled = local_snapshot_statistics[
        "gradient_ranking_oracle_enabled"
    ]
    gradient_probe_enabled = gradient_ordering_enabled or gradient_ranking_oracle_enabled
    if gradient_ordering_enabled:
        resolved["gradient_ordering"] = True
    gradient_window_tiebreak = (
        os.environ.get("REBUFFER_GRADIENT_WINDOW_TIEBREAK", "") == "1"
        or bool(config.get("repair_gradient_window_tiebreak", False))
    )
    if gradient_window_tiebreak:
        resolved["gradient_window_tiebreak"] = True
    gradient_reject_limit = int(
        config.get(
            "repair_gradient_reject_limit",
            os.environ.get("REBUFFER_GRADIENT_REJECT_LIMIT", "0") or 0,
        )
    )
    if gradient_ordering_enabled and gradient_reject_limit > 0:
        resolved["gradient_reject_limit"] = gradient_reject_limit
    native_queue_enabled = bool(config.get("use_native_timing_queue", False))
    native_evaluation_enabled = bool(
        getattr(timing_model, "use_native_timing_evaluation", False)
    )
    native_planning_enabled = bool(config.get("use_native_timing_planning", False))
    if config.get("use_native_timing_path_context", False) and not native_queue_enabled:
        raise ValueError("native path context requires the native timing queue")
    planning_snapshot = None
    if native_planning_enabled and (not native_evaluation_enabled or not native_queue_enabled):
        raise ValueError("native planning requires native timing evaluation and queue")
    if native_evaluation_enabled and not native_queue_enabled:
        raise ValueError("native timing evaluation requires the native timing queue")
    if native_queue_enabled:
        if not getattr(timing_model, "use_native_timing_graph", False):
            raise ValueError("native timing queue requires a native timing campaign")
        native_gradient = (
            native_planning_enabled and gradient_window_tiebreak
            and not gradient_ranking_oracle_enabled
        )
        if (gradient_probe_enabled and not native_gradient) or any(
            local_snapshot_statistics[name]
            for name in ("enabled", "native_update_enabled", "native_context_reuse_enabled")
        ):
            raise ValueError("native timing queue does not yet support local/gradient policies")
        if native_gradient:
            if gradient_reject_limit not in (0, 1):
                raise ValueError("native gradient window requires gradient_reject_limit=1")
            resolved["gradient_reject_limit"] = 1
            resolved["gradient_slack_window_ps"] = float(
                config.get("repair_gradient_slack_window_ps", 5.0)
            )
            local_snapshot_statistics["gradient_slack_window_ps"] = resolved[
                "gradient_slack_window_ps"
            ]
    if native_planning_enabled:
        session.overlay.native_timing_bridge.initialize_planning(context)
    legacy_queue_builder = rebuffer_cpp.build_native_timing_queue_snapshot
    state_queue_builder = (
        getattr(rebuffer_cpp, "build_native_timing_queue_state", None)
        if os.environ.get("REBUFFER_NATIVE_QUEUE_STATE_ONLY", "") == "1"
        else None
    )

    def build_queue_snapshot(design, timing, liberty, records, net_map):
        if callable(state_queue_builder):
            try:
                return dict(
                    state_queue_builder(design, timing, liberty, records, net_map)
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # Lightweight bridge doubles and older native extensions only
                # implement the legacy materialized snapshot contract.
                pass
        return dict(legacy_queue_builder(design, timing, liberty, records, net_map))

    def evaluate():
        observation = dict(session.evaluate() or {})
        if native_evaluation_enabled:
            observation.update(
                session.overlay.native_timing_bridge.observation(
                    max_fanout=resolved.get("max_fanout", 32),
                    cap_margin=resolved.get("cap_margin", 0.0),
                    slew_margin=resolved.get("slew_margin", 0.0),
                )
            )
            return observation
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
        nonlocal planning_snapshot

        if native_planning_enabled:
            if gradient_ordering_enabled:
                from .gradient_order import native_timing_gradient_payload

                payload = native_timing_gradient_payload(
                    timing_model, session.overlay.native_timing_bridge, session.overlay
                )
                resolved.update(payload)
                local_snapshot_statistics["gradient_score_status"] = "ok"
                local_snapshot_statistics["gradient_score_epoch"] = payload[
                    "gradient_score_epoch"
                ]
                local_snapshot_statistics.setdefault("gradient_score_epochs", []).append(
                    payload["gradient_score_epoch"]
                )
                local_snapshot_statistics["gradient_steiner_provider"] = payload[
                    "gradient_steiner_provider"
                ]
                local_snapshot_statistics["gradient_driver_score_count"] = len(
                    payload["gradient_score_by_driver_name"]
                )
            planning_snapshot = session.overlay.native_timing_bridge.planning_snapshot(
                session.overlay, resolved
            )
            local_snapshot_statistics["snapshot_count"] += 1
            local_snapshot_statistics["native_graph_queue_count"] += 1
            snapshot = dict(planning_snapshot.queue_state())
            if gradient_ordering_enabled:
                if (
                    "gradient_scored_item_count" not in snapshot
                    or snapshot.get("gradient_slack_window_ps")
                    != resolved["gradient_slack_window_ps"]
                ):
                    raise RuntimeError(
                        "native gradient-score transport requires a rebuilt rebuffer op"
                    )
                local_snapshot_statistics.setdefault("gradient_scored_item_counts", []).append(
                    int(snapshot["gradient_scored_item_count"])
                )
            return snapshot
        # Materialization may replace the TP carrier and pin IDs. Capture
        # endpoints, paths and design views only after that transition.
        views = context.native_queue_views()
        if views.get("status") != "ok":
            return views
        if native_queue_enabled:
            bridge = session.overlay.native_timing_bridge
            if config.get("use_native_timing_path_context", False):
                snapshot = bridge.queue_state(
                    session.overlay, timing_model.active_pydb, use_path_context=True
                )
            else:
                snapshot = bridge.queue_state(session.overlay, timing_model.active_pydb)
            local_snapshot_statistics["snapshot_count"] += 1
            local_snapshot_statistics["native_graph_queue_count"] += 1
            snapshot = attach_planner_payloads(
                snapshot, context, timing_model, resolved,
                load_slack_override=_path_load_slack_override,
            )
            snapshot["local_snapshot_enabled"] = False
            return snapshot
        model = timing_model.model
        slack = _slack_vector(timing_model)
        local_enabled = local_snapshot_statistics["enabled"]
        local_snapshot_statistics["snapshot_count"] += 1
        current_state = _timing_snapshot_state(timing_model) if local_enabled else None
        changed_pins = (
            _changed_snapshot_pins(local_snapshot_cache["state"], current_state)
            if local_enabled
            else None
        )
        previous_records = local_snapshot_cache["records"] if local_enabled else {}
        endpoint_values = [
            (int(endpoint), float(slack[int(endpoint)]))
            for endpoint in model.end_points.detach().cpu().tolist()
        ]
        rebuild_endpoints = []
        if not local_enabled or changed_pins is None:
            rebuild_endpoints = [endpoint for endpoint, value in endpoint_values if value < 0.0]
        else:
            for endpoint, endpoint_slack in endpoint_values:
                previous = previous_records.get(endpoint)
                if endpoint_slack >= 0.0:
                    continue
                if (
                    previous is None
                    or float(previous.get("slack_ps", 0.0)) != endpoint_slack
                    or str(previous.get("endpoint_pin_name", ""))
                    != str(timing_model.active_pydb.pin_names[endpoint])
                    or bool(set(previous.get("path_pins", ())) & changed_pins)
                ):
                    rebuild_endpoints.append(endpoint)
        if local_enabled and rebuild_endpoints:
            try:
                details = timing_model.native_setup_path_details(
                    endpoint_pins=rebuild_endpoints
                ) or {}
            except TypeError:
                # Keep lightweight bridge test doubles and older timing-model
                # adapters on their full-detail compatibility path.
                details = timing_model.native_setup_path_details() or {}
        elif not local_enabled:
            details = timing_model.native_setup_path_details() or {}
        else:
            details = {}
        records = []
        changed_records = []
        changed_endpoint_pins = set()
        current_records = {}
        reused_record_count = 0
        rebuilt_record_count = 0
        for endpoint, endpoint_slack in endpoint_values:
            if endpoint_slack >= 0.0:
                continue
            previous = previous_records.get(endpoint) if local_enabled else None
            if local_enabled and endpoint not in rebuild_endpoints and previous is not None:
                record = dict(previous)
                records.append(record)
                current_records[endpoint] = record
                reused_record_count += 1
                continue
            path = [int(pin) for pin in (details.get(endpoint) or {}).get("path", ())]
            if not path:
                paths = model.get_critical_paths([endpoint], K=1)
                path = [int(pin) for pin in (paths[0] if paths else ())]
            if not path:
                continue
            path_details = details if endpoint in details else None
            if local_enabled and path_details is not None:
                try:
                    path_timing = context.native_path_context(
                        path, path_details=path_details
                    )
                except TypeError:
                    path_timing = context.native_path_context(path)
            else:
                path_timing = context.native_path_context(path)
            record = {
                "endpoint_pin": endpoint,
                "endpoint_pin_name": str(timing_model.active_pydb.pin_names[endpoint]),
                "slack_ps": endpoint_slack,
                "path_pins": path,
                "path_timing": path_timing,
            }
            records.append(record)
            changed_records.append(record)
            changed_endpoint_pins.add(endpoint)
            current_records[endpoint] = record
            rebuilt_record_count += 1
        if gradient_probe_enabled:
            gradient_method = getattr(context, "timing_gradient_scores", None)
            gradient_result = (
                gradient_method()
                if callable(gradient_method)
                else {
                    "status": "unsupported",
                    "reason": "timing_gradient_context_missing",
                }
            )
            if gradient_result.get("status") == "ok":
                score_by_net = {
                    int(net): float(score)
                    for net, score in gradient_result.get("net_scores", {}).items()
                }
                local_snapshot_statistics["gradient_score_status"] = "ok"
                local_snapshot_statistics["gradient_loss"] = gradient_result.get("loss")
                if gradient_ordering_enabled:
                    for record in records:
                        record["gradient_score_by_net"] = dict(score_by_net)
            else:
                if gradient_ordering_enabled:
                    local_snapshot_statistics["gradient_score_status"] = "unsupported"
                reason = str(gradient_result.get("reason", "unknown"))
                if reason not in local_snapshot_statistics["gradient_score_failures"]:
                    local_snapshot_statistics["gradient_score_failures"].append(reason)
                if gradient_ordering_enabled:
                    for record in records:
                        record.pop("gradient_score_by_net", None)

            if gradient_ranking_oracle_enabled:
                oracle_method = getattr(context, "timing_gradient_ranking_oracle", None)
                if not callable(oracle_method):
                    local_snapshot_statistics["gradient_ranking_oracle_status"] = (
                        "unsupported"
                    )
                    local_snapshot_statistics["gradient_ranking_oracle_failures"].append(
                        "timing_gradient_ranking_oracle_missing"
                    )
                elif gradient_result.get("status") != "ok":
                    local_snapshot_statistics["gradient_ranking_oracle_status"] = (
                        "unsupported"
                    )
                else:
                    max_paths = max(
                        0,
                        int(
                            local_snapshot_statistics[
                                "gradient_ranking_oracle_max_paths"
                            ]
                        ),
                    )
                    oracle_records = records[:max_paths] if max_paths else []
                    for record in oracle_records:
                        path = [int(pin) for pin in record.get("path_pins", ())]
                        if not path:
                            continue
                        endpoint = int(record.get("endpoint_pin", -1))
                        path_detail = details.get(endpoint)
                        try:
                            oracle = oracle_method(
                                path,
                                path_details=path_detail,
                                gradient_result=gradient_result,
                            )
                        except (
                            AttributeError,
                            KeyError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                        ) as error:
                            reason = f"oracle_failed:{type(error).__name__}"
                            if reason not in local_snapshot_statistics[
                                "gradient_ranking_oracle_failures"
                            ]:
                                local_snapshot_statistics[
                                    "gradient_ranking_oracle_failures"
                                ].append(reason)
                            continue
                        if oracle.get("status") != "ok":
                            reason = str(oracle.get("reason", "oracle_unsupported"))
                            if reason not in local_snapshot_statistics[
                                "gradient_ranking_oracle_failures"
                            ]:
                                local_snapshot_statistics[
                                    "gradient_ranking_oracle_failures"
                                ].append(reason)
                            continue
                        local_snapshot_statistics["gradient_ranking_oracle_path_count"] += 1
                        local_snapshot_statistics[
                            "gradient_ranking_oracle_top1_matches"
                        ] += int(bool(oracle.get("top1_match", False)))
                        local_snapshot_statistics[
                            "gradient_ranking_oracle_common_drivers"
                        ] += int(oracle.get("common_driver_count", 0))
                        local_snapshot_statistics[
                            "gradient_ranking_oracle_pairwise_comparisons"
                        ] += int(oracle.get("pairwise_comparisons", 0))
                        local_snapshot_statistics[
                            "gradient_ranking_oracle_pairwise_agreements"
                        ] += int(oracle.get("pairwise_agreements", 0))
                    if local_snapshot_statistics["gradient_ranking_oracle_path_count"]:
                        local_snapshot_statistics["gradient_ranking_oracle_status"] = "ok"
                    else:
                        local_snapshot_statistics[
                            "gradient_ranking_oracle_status"
                        ] = "unsupported"
        if local_enabled:
            active_endpoint_pins = {
                endpoint for endpoint, value in endpoint_values if value < 0.0
            }
            changed_endpoint_pins.update(
                local_snapshot_cache["active_endpoint_pins"] - active_endpoint_pins
            )
            local_snapshot_cache["state"] = current_state
            local_snapshot_cache["records"] = current_records
            update_snapshot = getattr(
                rebuffer_cpp, "update_native_timing_queue_snapshot", None
            )
            update_state = getattr(
                rebuffer_cpp, "update_native_timing_queue_state", None
            )
            if gradient_ordering_enabled:
                # Gradient scores are state-wide and can change even when an
                # endpoint record is reusable. Reparse the native items so a
                # context/update path cannot retain stale scores.
                snapshot = dict(
                    build_queue_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        records,
                        views["original_net_by_net"],
                    )
                )
            elif (
                local_snapshot_cache["native_snapshot"] is not None
                and local_snapshot_statistics["native_update_enabled"]
                and os.environ.get("REBUFFER_NATIVE_QUEUE_STATE_UPDATE", "") == "1"
                and callable(update_state)
            ):
                snapshot = dict(
                    update_state(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        local_snapshot_cache["native_snapshot"],
                        changed_records,
                        sorted(active_endpoint_pins),
                        sorted(changed_endpoint_pins),
                        views["original_net_by_net"],
                    )
                )
            elif (
                local_snapshot_cache["native_snapshot"] is not None
                and local_snapshot_statistics["native_context_reuse_enabled"]
                and callable(update_snapshot)
            ):
                # Rebuild the current queue records, but keep the immutable
                # native design/Liberty context from the previous snapshot.
                # This is intentionally separate from local record reuse:
                # BM64 often changes every critical path, while the context
                # itself remains valid across the timing state.
                sorted_active_endpoints = sorted(active_endpoint_pins)
                snapshot = dict(
                    update_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        local_snapshot_cache["native_snapshot"],
                        records,
                        sorted_active_endpoints,
                        sorted_active_endpoints,
                        views["original_net_by_net"],
                    )
                )
            elif (
                local_snapshot_cache["native_snapshot"] is not None
                and local_snapshot_statistics["native_update_enabled"]
                and callable(update_snapshot)
            ):
                snapshot = dict(
                    update_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        local_snapshot_cache["native_snapshot"],
                        changed_records,
                        sorted(active_endpoint_pins),
                        sorted(changed_endpoint_pins),
                        views["original_net_by_net"],
                    )
                )
            else:
                snapshot = dict(
                    build_queue_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        records,
                        views["original_net_by_net"],
                    )
                )
            local_snapshot_cache["active_endpoint_pins"] = active_endpoint_pins
            local_snapshot_cache["native_snapshot"] = snapshot
        else:
            update_snapshot = getattr(
                rebuffer_cpp, "update_native_timing_queue_snapshot", None
            )
            active_endpoint_pins = sorted(
                endpoint for endpoint, value in endpoint_values if value < 0.0
            )
            if gradient_ordering_enabled:
                snapshot = dict(
                    build_queue_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        records,
                        views["original_net_by_net"],
                    )
                )
            elif (
                local_snapshot_cache["native_snapshot"] is not None
                and local_snapshot_statistics["native_context_reuse_enabled"]
                and callable(update_snapshot)
            ):
                snapshot = dict(
                    update_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        local_snapshot_cache["native_snapshot"],
                        records,
                        active_endpoint_pins,
                        active_endpoint_pins,
                        views["original_net_by_net"],
                    )
                )
            else:
                snapshot = dict(
                    build_queue_snapshot(
                        views["design"],
                        views["timing"],
                        views["liberty"],
                        records,
                        views["original_net_by_net"],
                    )
                )
            local_snapshot_cache["native_snapshot"] = snapshot

        if bool(snapshot.get("queue_context_reused", False)):
            local_snapshot_statistics["queue_context_reuse_count"] += 1
        elif "queue_context_reused" in snapshot:
            local_snapshot_statistics["queue_context_rebuild_count"] += 1
        if bool(snapshot.get("queue_design_reused", False)):
            local_snapshot_statistics["queue_design_reuse_count"] += 1
        if bool(snapshot.get("queue_liberty_reused", False)):
            local_snapshot_statistics["queue_liberty_reuse_count"] += 1

        if local_snapshot_statistics["oracle_enabled"]:
            # The local/update path must be checked against a reconstruction
            # from the complete current record list.  Keep this behind an
            # explicit switch: the reference intentionally repeats the native
            # queue work and is a semantic diagnostic, not a production path.
            reference = dict(
                legacy_queue_builder(
                    views["design"],
                    views["timing"],
                    views["liberty"],
                    records,
                    views["original_net_by_net"],
                )
            )
            _assert_queue_snapshot_oracle(
                snapshot,
                reference,
                rebuffer_cpp.select_timing_queue_items,
                resolved,
            )
            local_snapshot_statistics["oracle_check_count"] += 1

        snapshot = attach_planner_payloads(
            snapshot, context, timing_model, resolved,
            load_slack_override=_path_load_slack_override,
        )
        snapshot["local_snapshot_enabled"] = local_enabled
        snapshot["local_changed_pin_count"] = (
            None if changed_pins is None else len(changed_pins)
        )
        snapshot["local_reused_record_count"] = reused_record_count
        snapshot["local_rebuilt_record_count"] = rebuilt_record_count
        local_snapshot_statistics["changed_pin_counts"].append(
            None if changed_pins is None else len(changed_pins)
        )
        local_snapshot_statistics["reused_record_counts"].append(reused_record_count)
        local_snapshot_statistics["rebuilt_record_counts"].append(rebuilt_record_count)
        return snapshot

    def native_apply(actions):
        timing_model.bind_overlay(session.overlay)
        raw = {"actions": list(actions)}
        if native_planning_enabled:
            for net in planning_snapshot.source_nets(raw["actions"]):
                state = session.overlay._native_state(str(session.base_pydb.net_names[int(net)]))
                session.overlay.native_timing_bridge.attach_state(state)
            return session.apply(
                list(planning_snapshot.project_actions(raw["actions"])),
                transactional=False,
            )
        if any("source_parent_node_id" in action for action in raw["actions"]):
            return session.apply(
                context.map_native_tree_actions(raw), transactional=False
            )
        return session.apply(context.map_native_actions(raw), transactional=False)

    def restore_and_resync(checkpoint):
        session.restore(checkpoint)
        return session.evaluate()

    def release_checkpoint(checkpoint):
        return session.release_checkpoint(checkpoint)

    operator_kwargs = {}
    if resolved.get("native_checkpoint_release", False):
        operator_kwargs["release_checkpoint"] = release_checkpoint
    result = dict(
        rebuffer_cpp.run_repair_timing_native_snapshot_operator(
            timing_snapshot,
            lambda label: session.checkpoint(label),
            native_apply,
            evaluate,
            restore_and_resync,
            resolved,
            **operator_kwargs,
        )
    )
    result["accepted_actions"] = session.export_actions()
    result["accepted_count"] = len(result["accepted_actions"])
    result["artifact"] = ARTIFACT
    result["artifact_version"] = ARTIFACT_VERSION
    result["config"] = resolved
    result["native_path_stats"] = dict(timing_model.native_setup_path_statistics())
    result["carrier_rebuild_statistics"] = dict(timing_model.carrier_rebuild_statistics())
    result["incremental_profile"] = dict(
        getattr(timing_model.model, "last_incremental_profile", {})
    )
    result["persistent_timing_statistics"] = dict(
        getattr(timing_model, "_persistent_timing_statistics", {})
    )
    result["local_snapshot_statistics"] = local_snapshot_statistics
    if getattr(timing_model, "use_native_timing_graph", False):
        result["native_timing_graph_statistics"] = (
            session.overlay.native_timing_bridge.statistics()
        )
    return result


__all__ = ["plan_timing_repair"]
