"""OpenROAD topology handoff transaction orchestration."""

import copy
from dataclasses import dataclass
from numbers import Integral
from typing import Callable


HANDOFF_COMMAND_METADATA_FIELDS = (
    "buffer_insertion_strategy",
    "buffer_insertion_command",
    "buffer_insertion_strategy_profile",
    "buffer_insertion_strategy_kind",
    "buffer_insertion_strategy_experimental",
    "buffer_insertion_strategy_validation",
    "buffer_insertion_strategy_non_buffer_mutation_free",
)


@dataclass
class HandoffTransactionContext:
    """Explicit callbacks for runtime counts, reports and restart ownership."""

    placement_counts: Callable
    mutation_payload: Callable
    failure_payload: Callable
    restart_after_sync: Callable
    restart_policy: Callable


def execute_openroad_handoff_transaction(
    context,
    params,
    placedb,
    pos,
    run_event,
    trigger_decision,
    handoff_event,
    model=None,
):
    session = placedb.handoff_session
    if session is None:
        session = placedb.create_or_reset_handoff_session()
    elif session.status == "failed":
        raise RuntimeError("openroad handoff session is failed")

    pre_counts = placedb.current_topology_counts()
    pre_snapshot_fingerprint = placedb.current_snapshot_fingerprint(pre_counts)
    source_placement_counts = context.placement_counts(placedb)
    openroad_session_identity = placedb._get_openroad_session_identity()
    try:
        handoff_seq = session.begin_handoff(
            run_event=handoff_event,
            trigger_decision=trigger_decision,
            openroad_session_identity=openroad_session_identity,
            pre_counts=pre_counts,
            pre_snapshot_fingerprint=pre_snapshot_fingerprint,
        )
    except Exception as exc:
        if session.status == "active":
            session.fail_handoff_start(
                run_event=handoff_event,
                trigger_decision=trigger_decision,
                openroad_session_identity=openroad_session_identity,
                pre_counts=pre_counts,
                pre_snapshot_fingerprint=pre_snapshot_fingerprint,
                failure_stage="refresh",
                error_summary=str(exc),
            )
        raise

    try:
        mutation_result = placedb.execute_openroad_handoff(params, pos, handoff_event)
        mutation_result["_source_placement_counts"] = source_placement_counts
    except Exception as exc:
        failure_stage = getattr(exc, "failure_stage", "mutation")
        session.fail_current_handoff(
            handoff_seq,
            failure_stage,
            str(exc),
            failure_payload=getattr(exc, "handoff_payload", None),
        )
        raise

    try:
        session.record_mutation_result(
            handoff_seq,
            context.mutation_payload(mutation_result),
        )
    except Exception as exc:
        session.fail_current_handoff(
            handoff_seq,
            "mutation",
            str(exc),
            failure_payload=context.failure_payload(mutation_result),
        )
        raise

    try:
        rebuild_result = context.restart_after_sync(
            params,
            placedb,
            mutation_result,
            pos=pos,
            handoff_seq=handoff_seq,
            model=model,
        )
        continuation_result = {
            "runtimedb_rebuild_performed": rebuild_result is not None,
            "restart_policy": context.restart_policy(params),
            "restart_probes": [],
            "restart_probe_target_iterations": [0, 50],
        }
        if rebuild_result is not None:
            continuation_result["runtimedb_generation_after"] = (
                session.runtimedb_generation + 1
            )
        session.commit_continuation(handoff_seq, continuation_result)
    except Exception as exc:
        failure_stage = (
            "rebuild"
            if mutation_result.get("requires_runtimedb_rebuild")
            else "refresh"
        )
        session.fail_current_handoff(handoff_seq, failure_stage, str(exc))
        raise

    if rebuild_result is not None and callable(rebuild_result):
        continuation_summary = session.active_topology_summary or {}
        placedb.runtimedb_generation = session.runtimedb_generation
        continuation_pre_counts = copy.deepcopy(
            continuation_summary.get("counts", mutation_result.get("post_counts"))
        )
        continuation_pre_snapshot_fingerprint = continuation_summary.get(
            "snapshot_fingerprint",
            mutation_result.get("post_snapshot_fingerprint"),
        )
        try:
            return rebuild_result(params, placedb)
        except Exception as exc:
            if session.status == "active":
                session.fail_handoff_start(
                    run_event=handoff_event,
                    trigger_decision=trigger_decision,
                    openroad_session_identity=openroad_session_identity,
                    pre_counts=continuation_pre_counts,
                    pre_snapshot_fingerprint=continuation_pre_snapshot_fingerprint,
                    failure_stage="rebuild",
                    error_summary=str(exc),
                    parent_handoff_seq=handoff_seq,
                )
            raise
    return rebuild_result


def build_session_mutation_result(handoff_result):
    mutation_kind = handoff_result.get("mutation_kind")

    def _buffer_churn_count(field_name):
        if field_name in handoff_result:
            value = handoff_result[field_name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise RuntimeError(
                    "topology_changed handoff invalid buffer churn count: %s"
                    % field_name
                )
            return int(value)
        if mutation_kind == "topology_changed":
            raise RuntimeError(
                "topology_changed handoff missing buffer churn count: %s" % field_name
            )
        return 0

    return {
        "mutation_kind": mutation_kind,
        "requires_runtimedb_rebuild": handoff_result.get(
            "requires_runtimedb_rebuild", False
        ),
        "post_counts": copy.deepcopy(handoff_result.get("post_counts")),
        "post_snapshot_fingerprint": handoff_result.get("post_snapshot_fingerprint"),
        "identity_summary": copy.deepcopy(handoff_result.get("identity_summary", {})),
        "added_names": copy.deepcopy(handoff_result.get("added_names", {})),
        "removed_names": copy.deepcopy(handoff_result.get("removed_names", {})),
        "added_buffer_count": _buffer_churn_count("added_buffer_count"),
        "removed_buffer_count": _buffer_churn_count("removed_buffer_count"),
        "surviving_buffer_count": _buffer_churn_count("surviving_buffer_count"),
        **{name: handoff_result.get(name) for name in HANDOFF_COMMAND_METADATA_FIELDS},
        "buffer_only_policy": copy.deepcopy(handoff_result.get("buffer_only_policy")),
        "runtimedb_rebuild_owner": handoff_result.get("runtimedb_rebuild_owner"),
    }


def build_handoff_command_failure_payload(handoff_result):
    payload = {}
    for field_name in HANDOFF_COMMAND_METADATA_FIELDS:
        if field_name in handoff_result:
            payload[field_name] = handoff_result[field_name]
    return payload
