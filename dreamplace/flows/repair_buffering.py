"""Placement-stage repair buffering: plan, commit, refresh, rebuild.

The stage is deliberately thin.  Native ``repair_design`` and
``repair_timing`` planners own every electrical decision; this module only
sequences the existing ECC commit/refresh primitives and invalidates the
placement runtime so the caller can rebuild ``NonLinearPlace`` from the new
database.  The default ``repair_timing`` candidate consumer is the native C++
timing graph, queue, propagation, and planning path.  Python
``TimingPropagation`` remains the fresh verification authority after commit.
"""

import json
from pathlib import Path

from dreamplace.ops.placeio_ecc.repair_profile import measure_phase

ARTIFACT = "repair_buffering_stage"
ARTIFACT_VERSION = 2

MODES = ("repair_design", "repair_timing", "repair_design_then_timing")
TIMING_EVALUATOR = "dreamplace_timing_propagation_after_native_commit"

# Native ``repair_timing -setup -sequence buffer`` configuration.  These are
# the values of the verified BM64 runs; explicit caller keys override them.
REPAIR_TIMING_DEFAULTS = {
    # The production repair_timing consumer is native C++.  Python
    # TimingPropagation remains the post-commit fresh verification path;
    # callers can explicitly disable these flags for compatibility/oracle
    # runs.
    "use_persistent_graph": False,
    "use_native_timing_graph": True,
    "use_native_timing_queue": True,
    "use_native_timing_evaluation": True,
    "use_native_timing_planning": True,
    # Queue: current driver trees from a 32-endpoint window, interleaved in
    # quanta of 8 attempts, with branch-focused complete-tree requests.
    "native_tree_queue": True,
    "native_tree_per_driver": True,
    "native_tree_branch_queue": True,
    "native_tree_round_robin": True,
    "native_tree_round_robin_endpoint_window": 32,
    "native_tree_round_robin_endpoint_quantum": 8,
    "native_tree_max_attempts_per_endpoint": 24,
    "native_tree_candidate_variants": 8,
    "native_tree_branch_candidate_variants": 2,
    "native_tree_branch_global_acceptance": True,
    # Each request evaluates the top two VGDP trees through the selected
    # native consumer.  The fresh Python TP remains the post-commit oracle.
    "python_timing_candidate_top_k": 2,
    # Candidate model: OpenROAD BufferGenerator eligibility and fast set.
    "buffer_set": "fast",
    "enforce_timing_buffer_move_fanout": True,
    "timing_max_fanout": 33,
    "use_downstream_slew_delay_bonus": True,
    "resize_drivers": False,
    # Acceptance: global WNS/TNS and per-pin electrical non-regression.
    "acceptance": "endpoint_journal",
    "allow_decreasing_slack": False,
    "native_openroad_decreasing_slack_journal": False,
    "decreasing_slack_max_passes": 5,
    "openroad_journal_keep_nonimproving": True,
    # Physical commit is re-verified in chunks with a fresh Python TP.
    "commit_verify_chunk": 8,
    "commit_verify_wns_margin_ps": 0.0,
    "commit_verify_tns_margin_ps": 0.0,
}


def resolve_repair_timing_config(config):
    """Return the native repair_timing config with explicit overrides."""

    resolved = {**REPAIR_TIMING_DEFAULTS, **dict(config or {})}
    if int(resolved["commit_verify_chunk"]) <= 0:
        raise ValueError("repair_timing requires commit_verify_chunk > 0")
    if int(resolved["python_timing_candidate_top_k"]) < 1:
        raise ValueError("python_timing_candidate_top_k must be a positive integer")
    scheduler = resolved.get("timing_scheduler", "queue")
    if scheduler not in ("queue", "openroad_legacy"):
        raise ValueError(f"unknown timing scheduler: {scheduler}")
    if scheduler == "openroad_legacy":
        from dreamplace.ops.placeio_ecc.native_legacy_timing import resolve_legacy_timing_config

        resolved = resolve_legacy_timing_config(resolved)
    return resolved


def _rawdb(placedb):
    raw_db = getattr(placedb, "rawdb", None)
    if raw_db is None or not bool(getattr(raw_db, "timing_enabled", False)):
        raise RuntimeError("repair buffering requires a timing-enabled ECC placement backend")
    return raw_db


def buffer_histogram(actions):
    """Return the committed buffer-master histogram."""

    histogram = {}
    for action in actions:
        master = str(action.get("buffer_master_name", "") or "")
        if master:
            histogram[master] = histogram.get(master, 0) + 1
    return dict(sorted(histogram.items()))


def _write_stage_artifact(stage, artifact_dir):
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "repair_buffering_stage.json"
    stage["artifact_path"] = str(path)
    path.write_text(json.dumps(stage, indent=2, sort_keys=True, default=str) + "\n")


def _rebuild_placement(placedb, raw_db):
    from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction

    rebuild = placedb.rebuild_from_pydb(raw_db.pydb, rebuild_mode="all")
    placedb.backend_caps = PlaceIOFunction.backend_caps(raw_db)
    return dict(rebuild)


def _run_repair_design(
    placedb, *, config, profile=None, defer_rebuild=False, skip_size_only_rebuild=False
):
    from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
    from dreamplace.ops.placeio_ecc.timing_carrier import (
        apply_inplace_carrier_sizing,
        refresh_python_timing_carrier,
    )
    from dreamplace.ops.rebuffer.repair_design import plan_repair_design_from_pydb
    from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
    from dreamplace.ops.rebuffer.timing_profile import placement_timing_kwargs

    raw_db = _rawdb(placedb)
    # The placement view is scaled/shifted; repair planners and buffer
    # coordinates use the original ECC dbu database.
    pydb = raw_db.pydb
    profile_kwargs = {"profile": profile} if profile is not None else {}
    if getattr(raw_db, "module", None) is not None and (
        bool(getattr(pydb, "native_state_dirty", False))
        or not bool(getattr(raw_db, "_carrier_positions_synced", False))
    ):
        with measure_phase(profile, "input_carrier_refresh_ms"):
            pydb, _ = refresh_python_timing_carrier(raw_db)
            try:
                raw_db._carrier_positions_synced = True
            except AttributeError:
                pass
    provider = str(config.get("steiner_provider", "ecc"))
    # The selected PD profile retains the physical-pin geometry of its
    # previous planner input; the legacy profile uses instance origins.
    timing_kwargs = placement_timing_kwargs(config, use_pin_offset=provider == "openroad_stt")
    with measure_phase(profile, "python_timing_model_build_ms"):
        timing_model = RebufferTimingModel(pydb, **timing_kwargs)
    summary = {}
    if profile is not None:
        from dreamplace.ops.placeio_ecc.electrical_observation import electrical_observation

        with profile.measure("initial_electrical_observation_ms"):
            summary["timing_before_electrical"] = electrical_observation(
                timing_model,
                max_fanout=config.get("max_fanout", 32),
                cap_margin=config.get("cap_margin", 0.0),
                slew_margin=config.get("slew_margin", 0.0),
            )
    with measure_phase(profile, "repair_plan_ms"):
        plan = plan_repair_design_from_pydb(pydb, timing_model, config=config)
    if profile is not None:
        profile.plans["repair_design"] = plan
    summary |= {
        "status": plan.get("status"),
        "timing_before": plan.get("baseline"),
        "inserted_count": plan.get("inserted_count"),
        "resized_count": plan.get("resized_count"),
        "buffer_histogram": buffer_histogram(plan.get("actions", ())),
        "plan": plan,
    }
    if profile is not None:
        summary["timing_profile"] = {
            "steiner_provider": provider, "use_pin_offset": bool(timing_model.use_pin_offset),
            "r_unit": timing_model.r_unit, "c_unit": timing_model.c_unit,
        }
    if plan.get("status") != "ok":
        return summary
    if not plan.get("actions") and not plan.get("resizes"):
        summary.update({"status": "noop", "commit": {"status": "skipped"}})
        return summary

    commit = PlaceIOFunction.apply_repair_design(raw_db, plan, refresh=False, **profile_kwargs)
    summary["commit"] = dict(commit)
    if str(commit.get("status") or "") != "committed":
        summary["status"] = "commit_failed"
        return summary

    inplace = (
        apply_inplace_carrier_sizing(raw_db, plan.get("resizes"), **profile_kwargs)
        if not plan.get("actions") and plan.get("resizes")
        else None
    )
    if inplace is not None:
        refreshed, refresh = inplace
    else:
        refreshed, refresh = refresh_python_timing_carrier(raw_db, **profile_kwargs)
    summary["refresh"] = dict(refresh)
    size_only_inplace = bool(
        skip_size_only_rebuild and not plan.get("actions") and plan.get("resizes")
    )
    if size_only_inplace:
        placedb.pydb = raw_db.pydb
        placedb.backend_caps = PlaceIOFunction.backend_caps(raw_db)
        summary["rebuild"] = {"status": "skipped_inplace_sizing"}
    elif not defer_rebuild:
        with measure_phase(profile, "placedb_rebuild_ms"):
            summary["rebuild"] = _rebuild_placement(placedb, raw_db)
    with measure_phase(profile, "python_timing_model_build_ms"):
        after = RebufferTimingModel(refreshed, **timing_kwargs)
    summary["timing_after"] = after.baseline.as_dict()
    if profile is not None:
        from dreamplace.ops.placeio_ecc.electrical_observation import electrical_observation

        with profile.measure("fresh_electrical_observation_ms"):
            summary["timing_after_electrical"] = electrical_observation(
                after,
                max_fanout=config.get("max_fanout", 32),
                cap_margin=config.get("cap_margin", 0.0),
                slew_margin=config.get("slew_margin", 0.0),
            )
    if provider == "openroad_stt":
        summary["timing_after_steiner_input_identity"] = dict(after.steiner_input_identity)
    summary["status"] = "ok"
    return summary


def _run_repair_timing(placedb, *, max_passes, config, profile=None):
    from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
    from dreamplace.ops.placeio_ecc.commit_verify import (
        commit_buffer_actions_verified,
        timing_reader,
    )
    from dreamplace.ops.placeio_ecc.repair_timing import plan_timing_repair
    from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
    from dreamplace.ops.rebuffer.timing_profile import placement_timing_kwargs

    raw_db = _rawdb(placedb)
    # In repair_design_then_timing the design stage already refreshed
    # raw_db.pydb to the dbu database.  The session and timing model must
    # share that exact object; PlaceIOBufferSession rejects a second export.
    pydb = raw_db.pydb
    with measure_phase(profile, "python_timing_model_build_ms"):
        timing_model = RebufferTimingModel(
            pydb,
            use_persistent_graph=bool(config.get("use_persistent_graph", False)),
            use_native_timing_graph=bool(config.get("use_native_timing_graph", False)),
            use_native_timing_evaluation=bool(config.get("use_native_timing_evaluation", False)),
            **placement_timing_kwargs(config),
        )
    session = PlaceIOBufferSession(raw_db, timing_model=timing_model)
    session.begin()
    with measure_phase(profile, "repair_plan_ms"):
        plan = plan_timing_repair(session, timing_model, max_passes=max_passes, config=config)
    if profile is not None:
        profile.plans["repair_timing"] = plan
    accepted = list(plan.get("accepted_actions", ()) or ())
    summary = {
        "status": plan.get("status"),
        "accepted_count": len(accepted),
        "timing_before": plan.get("baseline"),
        "timing_overlay_final": plan.get("final"),
        "buffer_histogram": buffer_histogram(accepted),
        "plan": plan,
    }
    if profile is not None:
        summary["timing_profile"] = {
            "steiner_provider": str(config.get("steiner_provider", "ecc")),
            "use_pin_offset": bool(timing_model.use_pin_offset),
            "r_unit": timing_model.r_unit, "c_unit": timing_model.c_unit,
        }
    if plan.get("status") != "ok":
        session.rollback()
        summary["status"] = "plan_failed"
        return summary
    if not accepted:
        session.rollback()
        summary.update({"status": "noop", "commit": {"status": "skipped"}})
        return summary

    # Commit to the native IDB in chunks; each chunk rebuilds timing indices
    # from the committed connectivity and re-evaluates it with Python TP.
    with measure_phase(profile, "repair_timing_verified_commit_ms"):
        commit = commit_buffer_actions_verified(
            raw_db,
            accepted,
            chunk_size=int(config["commit_verify_chunk"]),
            wns_margin_ps=float(config["commit_verify_wns_margin_ps"]),
            tns_margin_ps=float(config["commit_verify_tns_margin_ps"]),
            reader=lambda current: timing_reader(current, config=config),
            **({"profile": profile} if profile is not None else {}),
        )
    session.rollback()
    summary["commit"] = dict(commit)
    summary["verified_kept_count"] = int(commit.get("kept_count", 0) or 0)
    summary["verified_reverted_count"] = int(commit.get("reverted_count", 0) or 0)
    if str(commit.get("status") or "") != "ok":
        summary["status"] = "commit_failed"
        return summary

    with measure_phase(profile, "placedb_rebuild_ms"):
        summary["rebuild"] = _rebuild_placement(placedb, raw_db)
    summary["timing_after"] = dict(commit["timing_after"])
    summary["timing_after_electrical"] = dict(commit.get("timing_after_electrical") or {})
    summary["status"] = "ok"
    return summary


def run_repair_buffering_stage(
    placedb,
    *,
    mode="repair_design_then_timing",
    max_passes=3000,
    config=None,
    artifact_dir=None,
    placement_snapshot_digest=None,
    topology_generation=None,
    profile=None,
    skip_size_only_rebuild=False,
):
    """Run one repair-buffering stage on ``placedb``.

    ``repair_design`` and ``repair_timing`` each commit once and rebuild the
    placement database before returning.  The combined mode runs the two
    stages in order, so the timing stage sees the post-design database.
    The timing stage is buffer-only: it never resizes drivers or runs an
    implicit electrical prepass.
    """

    if mode not in MODES:
        raise ValueError(f"unsupported repair buffering mode: {mode!r}")
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    config = dict(config or {})
    summary = {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "mode": mode,
        "status": "ok",
        "placement_snapshot_digest": placement_snapshot_digest,
        "topology_generation": topology_generation,
        "config": config,
        "timing_evaluator": TIMING_EVALUATOR,
        "external_sta_run": False,
        "stages": {},
    }
    design_rebuild_deferred = False
    if mode in {"repair_design", "repair_design_then_timing"}:
        before = dict(profile.timings_ms) if profile is not None else {}
        stage = _run_repair_design(
            placedb,
            config=config,
            profile=profile,
            defer_rebuild=(mode == "repair_design_then_timing"),
            skip_size_only_rebuild=skip_size_only_rebuild,
        )
        if profile is not None:
            stage["phase_timings_ms"] = {
                key: value - before.get(key, 0.0) for key, value in profile.timings_ms.items()
                if value != before.get(key, 0.0)
            }
        summary["stages"]["repair_design"] = stage
        design_rebuild_deferred = (
            mode == "repair_design_then_timing"
            and stage.get("status") == "ok"
            and "rebuild" not in stage
        )
        if stage.get("status") not in {"ok", "noop"}:
            summary["status"] = "failed"
        elif (
            mode == "repair_design_then_timing" and artifact_dir is not None and profile is not None
        ):
            from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction

            path = artifact_dir / "repair_design_committed.def"
            with profile.measure("design_boundary_export_ms"):
                PlaceIOFunction.write_def(_rawdb(placedb), str(path))
            summary["design_committed_def"] = str(path)
    if summary["status"] == "ok" and mode in {"repair_timing", "repair_design_then_timing"}:
        try:
            timing_config = resolve_repair_timing_config(config)
        except ValueError as error:
            if design_rebuild_deferred:
                with measure_phase(profile, "placedb_rebuild_ms"):
                    summary["stages"]["repair_design"]["rebuild"] = _rebuild_placement(
                        placedb, _rawdb(placedb)
                    )
            summary.update(status="invalid_config", failure_reason=str(error))
            _write_stage_artifact(summary, artifact_dir)
            return summary
        summary["repair_timing_config"] = timing_config
        before = dict(profile.timings_ms) if profile is not None else {}
        stage = _run_repair_timing(
            placedb, max_passes=max_passes, config=timing_config, profile=profile
        )
        if design_rebuild_deferred and stage.get("status") != "ok":
            with measure_phase(profile, "placedb_rebuild_ms"):
                summary["stages"]["repair_design"]["rebuild"] = _rebuild_placement(
                    placedb, _rawdb(placedb)
                )
        if profile is not None:
            stage["phase_timings_ms"] = {
                key: value - before.get(key, 0.0) for key, value in profile.timings_ms.items()
                if value != before.get(key, 0.0)
            }
        summary["stages"]["repair_timing"] = stage
        if stage.get("status") not in {"ok", "noop"}:
            summary["status"] = "failed"
    _write_stage_artifact(summary, artifact_dir)
    if profile is not None and artifact_dir is not None:
        (artifact_dir / "repair_buffering_profile.json").write_text(
            json.dumps(profile.timings_ms, indent=2, sort_keys=True) + "\n"
        )
    return summary


def reenter_after_repair_buffering(placer, params, placedb):
    """Run the stage and re-enter legalization with a fresh placer.

    The old ``NonLinearPlace`` tensors describe the pre-commit topology and
    must not be reused.  The caller reaches this helper after global
    placement, so the fresh placer is configured for legalization only.
    """

    import copy
    import hashlib
    import os

    from dreamplace import NonLinearPlace
    from dreamplace.ops.placeio_ecc.timing_carrier import refresh_python_timing_carrier
    from dreamplace.ops.routability.gpugr_context import write_back_movable_lpos

    write_back_movable_lpos(placer.pos[0], params, placedb)
    snapshot = placer.pos[0].detach().cpu().numpy().tobytes()
    stage = run_repair_buffering_stage(
        placedb,
        mode=str(getattr(params, "repair_buffering_mode", "repair_design_then_timing")),
        max_passes=int(getattr(params, "repair_buffering_max_passes", 3000) or 3000),
        config=getattr(params, "repair_buffering_config", None),
        artifact_dir=getattr(params, "result_dir", None),
        placement_snapshot_digest=hashlib.sha256(snapshot).hexdigest(),
        topology_generation=int(getattr(placedb, "topology_generation", 0) or 0),
    )
    if stage.get("status") != "ok":
        raise RuntimeError(f"repair buffering stage failed: {stage}")

    terminal_params = copy.copy(params)
    terminal_params.repair_buffering_enable = 0
    terminal_params.global_place_flag = 0
    terminal_params.legalize_flag = 1
    terminal_params.timing_opt_enabled = 0
    # The fresh placer is legalization-only; carry-over real_size sizing is
    # invalid without a size/joint/timing flow.
    terminal_params.sizing_parameterization = "logits"
    terminal_params.routability_opt_flag = 0
    terminal_params.l_shape_routability_flag = 0
    terminal_params.enable_relaxed_buffer_timing = False
    terminal_params.enable_fillers = 0
    terminal_params.result_dir = (
        str(getattr(params, "result_dir", "")) + "_repair_buffering_terminal"
    )
    os.makedirs(terminal_params.result_dir, exist_ok=True)

    new_placer = NonLinearPlace.NonLinearPlace(terminal_params, placedb, None)
    result = new_placer(terminal_params, placedb)
    # Make the Python timing carrier match coordinates written by legalizer.
    raw_db = getattr(placedb, "rawdb", None)
    if raw_db is not None and bool(getattr(raw_db, "timing_enabled", False)):
        refreshed, _ = refresh_python_timing_carrier(raw_db)
        placedb.rebuild_from_pydb(refreshed, rebuild_mode="all")
    legalize = {
        "status": "ok",
        "metrics": result[2] if isinstance(result, tuple) and len(result) == 3 else result,
    }
    if hasattr(new_placer, "run_sta_only"):
        try:
            _, _, timing_metrics = new_placer.run_sta_only(terminal_params, placedb)
        except Exception as error:
            legalize.update({"sta_status": "failed", "error": str(error)})
        else:
            sta = ((timing_metrics or {}).get("timing_stage_summary") or {}).get("sta", {})
            legalize.update({"sta_status": "ok", "wns": sta.get("wns"), "tns": sta.get("tns")})
    stage["legalize"] = legalize
    _write_stage_artifact(stage, getattr(params, "result_dir", None))
    # The repair stage already committed the terminal topology.  The outer
    # Placer.run() must not finalize the stale inflation window (its
    # terminal_request was consumed by this stage).
    params.timing_opt_enabled = 0
    placer.inflation_s5b1 = None
    placer.alternating_discrete = None
    if isinstance(result, tuple) and len(result) == 3:
        metrics = result[2]
        if isinstance(metrics, dict):
            metrics["repair_buffering_stage"] = stage
        return result
    return (
        float("nan"),
        float("nan"),
        {"repair_buffering_stage": stage, "legalization": result},
    )


__all__ = [
    "ARTIFACT",
    "ARTIFACT_VERSION",
    "MODES",
    "REPAIR_TIMING_DEFAULTS",
    "buffer_histogram",
    "reenter_after_repair_buffering",
    "resolve_repair_timing_config",
    "run_repair_buffering_stage",
]
