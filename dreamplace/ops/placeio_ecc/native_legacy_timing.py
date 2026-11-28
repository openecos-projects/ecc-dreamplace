"""Transport placeio_ecc state to the native LEGACY endpoint/pass scheduler."""

import os


def resolve_legacy_timing_config(config):
    resolved = dict(config)
    native_flags = (
        "use_native_timing_graph",
        "use_native_timing_queue",
        "use_native_timing_evaluation",
        "use_native_timing_planning",
    )
    if not all(resolved.get(name, False) for name in native_flags):
        raise ValueError("openroad_legacy requires all four native timing consumers")
    if any(
        resolved.get(name, False)
        for name in (
            "repair_gradient_ordering",
            "repair_gradient_window_tiebreak",
            "repair_gradient_ranking_oracle",
            "gradient_ordering",
        )
    ) or any(
        os.environ.get(name) == "1"
        for name in (
            "REBUFFER_GRADIENT_ORDERING",
            "REBUFFER_GRADIENT_WINDOW_TIEBREAK",
        )
    ):
        raise ValueError("openroad_legacy requires the canonical path-driver ordering")
    if int(resolved.get("max_repairs_per_pass", 1)) != 1:
        raise ValueError("openroad_legacy currently requires max_repairs_per_pass=1")
    if float(resolved.get("setup_slack_margin_ps", 0.0)) > 0:
        raise ValueError("openroad_legacy currently requires a nonpositive setup slack margin")
    resolved.update(
        timing_scheduler="openroad_legacy",
        use_native_timing_path_context=True,
        native_tree_queue=True,
        native_tree_per_driver=True,
        native_tree_branch_queue=False,
        native_tree_round_robin=False,
        native_tree_stop_at_existing_buffers=True,
        native_skip_top_level_output_ports=True,
        native_tree_candidate_variants=1,
        python_timing_candidate_top_k=1,
        max_repairs_per_pass=1,
        acceptance="openroad_legacy",
        allow_decreasing_slack=True,
        native_openroad_decreasing_slack_journal=True,
        openroad_journal_keep_nonimproving=True,
    )
    resolved.setdefault("resize_drivers", False)
    resolved.setdefault("legacy_max_passes_per_endpoint", 50)
    resolved.setdefault("repair_tns_end_percent", 1.0)
    resolved.setdefault("native_metadata_journal", True)
    resolved.setdefault("native_timing_local_views", True)
    resolved.setdefault("native_checkpoint_release", True)
    return resolved


def plan_legacy_timing_repair(session, timing_model, *, max_passes, config, context=None):
    from ..rebuffer import rebuffer_cpp
    from ..rebuffer.rebuffer_timing import RebufferPlanContext

    resolved = {**resolve_legacy_timing_config(config), "max_passes": int(max_passes)}
    if not timing_model.use_native_timing_evaluation:
        raise ValueError("openroad_legacy requires the native timing evaluation owner")
    if context is None:
        context = RebufferPlanContext(session.base_pydb, timing_model, config=resolved)
    elif any(
        context.config.get(name) != resolved[name]
        for name in (
            "timing_scheduler",
            "use_native_timing_path_context",
            "native_tree_per_driver",
            "native_tree_branch_queue",
            "native_tree_round_robin",
            "python_timing_candidate_top_k",
        )
    ):
        raise ValueError("LEGACY context must use the resolved scheduler configuration")
    if context.status != "ok":
        return {
            "status": "unsupported",
            "unsupported_reasons": list(context.reasons),
            "accepted_actions": [],
            "accepted_count": 0,
            "passes": [],
        }
    bridge = session.overlay.native_timing_bridge
    if resolved["native_metadata_journal"]:
        session.overlay.enable_metadata_journal()
    bridge.initialize_planning(context)
    snapshot = None

    def evaluate():
        observation = dict(session.evaluate())
        observation.update(
            bridge.observation(
                max_fanout=resolved.get("max_fanout", 32),
                cap_margin=resolved.get("cap_margin", 0.0),
                slew_margin=resolved.get("slew_margin", 0.0),
            )
        )
        return observation

    def endpoint_state():
        bridge.evaluate(session.overlay)
        return bridge.endpoint_state()

    def timing_snapshot():
        nonlocal snapshot
        snapshot = bridge.planning_snapshot(session.overlay, resolved)
        return dict(snapshot.queue_state())

    def apply(actions):
        for net in snapshot.source_nets(actions):
            state = session.overlay._native_state(str(session.base_pydb.net_names[int(net)]))
            bridge.attach_state(state)
        # The native scheduler already owns a trial checkpoint and restores it
        # after a failed candidate. Avoid taking a second full graph snapshot
        # inside BufferOverlay.apply; ordinary session.apply keeps its atomic
        # Python transaction by default.
        return session.apply(
            list(snapshot.project_actions(actions)), transactional=False
        )

    def restore(checkpoint):
        session.restore(checkpoint)
        return session.evaluate()

    def release_checkpoint(checkpoint):
        return session.release_checkpoint(checkpoint)

    result = dict(
        rebuffer_cpp.run_repair_timing_native_snapshot_operator(
            timing_snapshot,
            lambda label: session.checkpoint(label),
            apply,
            evaluate,
            restore,
            resolved,
            endpoint_state,
            release_checkpoint,
        )
    )
    result.update(
        accepted_actions=session.export_actions(),
        artifact="placeio_ecc_timing_repair",
        artifact_version=1,
        config=resolved,
        native_timing_graph_statistics=bridge.statistics(),
        native_path_stats={},
        carrier_rebuild_statistics=dict(timing_model.carrier_rebuild_statistics()),
        persistent_timing_statistics={},
    )
    result["accepted_count"] = len(result["accepted_actions"])
    return result
