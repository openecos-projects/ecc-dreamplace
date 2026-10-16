from copy import deepcopy
from numbers import Integral


class PlacementHandoffSession:
    _ALLOWED_MUTATION_KINDS = {
        "topology_changed",
        "geometry_changed",
        "placement_only",
        "no_mutation",
    }
    _REBUILD_REQUIRED_MUTATION_KINDS = {
        "topology_changed",
        "geometry_changed",
    }
    _ALLOWED_FAILURE_STAGES = {"sync", "mutation", "refresh", "rebuild"}
    _RESERVED_KEYS = {
        "handoff_seq",
        "parent_handoff_seq",
        "status",
        "continuation_status",
        "failure_stage",
        "error_summary",
        "absolute_iteration",
        "phase",
        "mutation_recorded",
        "post_topology_epoch",
        "topology_epoch",
        "runtimedb_generation",
        "runtimedb_generation_before",
        "openroad_session_identity",
        "metric_snapshot",
    }
    _RESERVED_PREFIXES = ("pre_", "trigger_")
    _MUTATION_OWNED_FIELDS = {
        "mutation_kind",
        "post_counts",
        "post_snapshot_fingerprint",
        "requires_runtimedb_rebuild",
        "identity_summary",
        "added_names",
        "removed_names",
        "added_buffer_count",
        "removed_buffer_count",
        "surviving_buffer_count",
        "buffer_insertion_strategy",
        "buffer_insertion_command",
        "buffer_insertion_strategy_profile",
        "buffer_insertion_strategy_kind",
        "buffer_insertion_strategy_experimental",
        "buffer_insertion_strategy_validation",
        "buffer_insertion_strategy_non_buffer_mutation_free",
        "buffer_only_policy",
        "runtimedb_rebuild_owner",
    }
    _BUFFER_CHURN_FIELDS = (
        "added_buffer_count",
        "removed_buffer_count",
        "surviving_buffer_count",
    )
    _CONTINUATION_OWNED_FIELDS = {
        "runtimedb_rebuild_performed",
        "runtimedb_generation_after",
        "continuation_status",
        "failure_stage",
        "error_summary",
    }

    def __init__(self, session_id):
        self.session_id = session_id
        self.status = "active"
        self.handoff_seq_counter = 0
        self.topology_epoch = 0
        self.runtimedb_generation = 0
        self.active_topology_summary = None
        self.event_history = []

    def begin_handoff(self, run_event, trigger_decision, openroad_session_identity, pre_counts, pre_snapshot_fingerprint):
        if self.status != "active":
            raise RuntimeError("handoff session is not active")
        if self.active_topology_summary is not None:
            previous_identity = self.event_history[-1].get("openroad_session_identity")
            if openroad_session_identity != previous_identity:
                raise RuntimeError("openroad session identity does not match active handoff state")
            if pre_counts != self.active_topology_summary.get("counts"):
                raise RuntimeError("pre_counts do not match active topology summary")
            if pre_snapshot_fingerprint != self.active_topology_summary.get("snapshot_fingerprint"):
                raise RuntimeError("pre_snapshot_fingerprint does not match active topology summary")
        self.handoff_seq_counter += 1
        record = {
            "handoff_seq": self.handoff_seq_counter,
            "absolute_iteration": run_event.get("absolute_iteration"),
            "phase": run_event.get("phase"),
            "metric_snapshot": deepcopy(run_event.get("metric_snapshot")),
            "trigger_mode": trigger_decision.get("trigger_mode"),
            "trigger_reason": trigger_decision.get("trigger_reason"),
            "openroad_session_identity": openroad_session_identity,
            "pre_counts": deepcopy(pre_counts),
            "pre_topology_epoch": self.topology_epoch,
            "pre_snapshot_fingerprint": pre_snapshot_fingerprint,
            "runtimedb_generation_before": self.runtimedb_generation,
            "status": "in_progress",
        }
        self.event_history.append(record)
        self.status = "handoff_in_progress"
        return record["handoff_seq"]

    def fail_handoff_start(
        self,
        run_event,
        trigger_decision,
        openroad_session_identity,
        pre_counts,
        pre_snapshot_fingerprint,
        failure_stage,
        error_summary,
        parent_handoff_seq=None,
    ):
        if self.status != "active":
            raise RuntimeError("handoff session is not active")
        if failure_stage not in self._ALLOWED_FAILURE_STAGES:
            raise RuntimeError("unknown failure stage")
        self.handoff_seq_counter += 1
        record = {
            "handoff_seq": self.handoff_seq_counter,
            "absolute_iteration": run_event.get("absolute_iteration"),
            "phase": run_event.get("phase"),
            "metric_snapshot": deepcopy(run_event.get("metric_snapshot")),
            "trigger_mode": trigger_decision.get("trigger_mode"),
            "trigger_reason": trigger_decision.get("trigger_reason"),
            "openroad_session_identity": openroad_session_identity,
            "pre_counts": deepcopy(pre_counts),
            "pre_topology_epoch": self.topology_epoch,
            "pre_snapshot_fingerprint": pre_snapshot_fingerprint,
            "runtimedb_generation_before": self.runtimedb_generation,
            "continuation_status": "failed_before_continue",
            "failure_stage": failure_stage,
            "error_summary": error_summary,
            "status": "%s_failed" % failure_stage,
        }
        if parent_handoff_seq is not None:
            record["parent_handoff_seq"] = parent_handoff_seq
        self.event_history.append(record)
        self.status = "failed"
        return record["handoff_seq"]

    def record_mutation_result(self, handoff_seq, mutation_result):
        self._require_pending_handoff()
        record = self._get_record(handoff_seq)
        if record.get("mutation_recorded"):
            raise RuntimeError("mutation result already recorded for this handoff")
        mutation_kind = mutation_result.get("mutation_kind")
        if mutation_kind not in self._ALLOWED_MUTATION_KINDS:
            raise RuntimeError("unknown mutation kind")
        if mutation_kind in self._REBUILD_REQUIRED_MUTATION_KINDS and not mutation_result.get(
            "requires_runtimedb_rebuild"
        ):
            raise RuntimeError("%s mutation requires runtimedb rebuild" % mutation_kind)
        if mutation_kind == "topology_changed":
            self._validate_buffer_churn_counts(mutation_result)
        self._reject_reserved_keys(mutation_result, extra_reserved=self._CONTINUATION_OWNED_FIELDS)
        record.update(deepcopy(mutation_result))
        record["mutation_recorded"] = True
        if mutation_result.get("requires_runtimedb_rebuild"):
            self.status = "rebuild_in_progress"

    def commit_continuation(self, handoff_seq, continuation_result):
        self._require_pending_handoff()
        record = self._get_record(handoff_seq)
        if not record.get("mutation_recorded"):
            raise RuntimeError("mutation result must be recorded before commit")
        self._reject_reserved_keys(continuation_result, extra_reserved=self._MUTATION_OWNED_FIELDS)
        mutation_kind = record.get("mutation_kind")
        post_counts = record.get("post_counts")
        if mutation_kind == "topology_changed" and post_counts is None:
            raise RuntimeError("post_counts is required after topology-changing mutation")
        requires_runtimedb_rebuild = bool(record.get("requires_runtimedb_rebuild"))
        if mutation_kind in self._REBUILD_REQUIRED_MUTATION_KINDS and not requires_runtimedb_rebuild:
            raise RuntimeError("%s mutation requires runtimedb rebuild" % mutation_kind)
        runtimedb_rebuild_performed = bool(continuation_result.get("runtimedb_rebuild_performed", False))
        if requires_runtimedb_rebuild and not runtimedb_rebuild_performed:
            raise RuntimeError("required runtimedb rebuild was not performed")
        if runtimedb_rebuild_performed and "runtimedb_generation_after" not in continuation_result:
            raise RuntimeError("runtimedb_generation_after is required when rebuild is performed")
        if runtimedb_rebuild_performed:
            generation_after = continuation_result["runtimedb_generation_after"]
            if generation_after is None:
                raise RuntimeError("runtimedb_generation_after must not be None")
            if isinstance(generation_after, bool) or not isinstance(generation_after, Integral):
                raise RuntimeError("runtimedb_generation_after must be an integer")
            if generation_after <= self.runtimedb_generation:
                raise RuntimeError("runtimedb_generation_after must advance")
        record.update(deepcopy(continuation_result))
        record["continuation_status"] = "continued"
        if mutation_kind == "topology_changed":
            self.topology_epoch += 1
        if runtimedb_rebuild_performed:
            self.runtimedb_generation = generation_after
        record["post_topology_epoch"] = self.topology_epoch
        record["status"] = "success"
        if mutation_kind == "topology_changed":
            counts = post_counts
        elif self.active_topology_summary is not None:
            counts = self.active_topology_summary.get("counts")
        else:
            counts = record.get("pre_counts")
        snapshot_fingerprint = record.get("post_snapshot_fingerprint")
        if snapshot_fingerprint is None:
            snapshot_fingerprint = record.get("pre_snapshot_fingerprint")
        self.active_topology_summary = {
            "counts": deepcopy(counts),
            "snapshot_fingerprint": snapshot_fingerprint,
            "topology_epoch": self.topology_epoch,
            "runtimedb_generation": self.runtimedb_generation,
        }
        self.status = "active"

    def fail_current_handoff(
        self,
        handoff_seq,
        failure_stage,
        error_summary,
        failure_payload=None,
    ):
        self._require_pending_handoff()
        if failure_stage not in self._ALLOWED_FAILURE_STAGES:
            raise RuntimeError("unknown failure stage")
        record = self._get_record(handoff_seq)
        if failure_stage == "rebuild":
            if self.status != "rebuild_in_progress" or not record.get("requires_runtimedb_rebuild"):
                raise RuntimeError("rebuild failure stage requires a rebuild-in-progress handoff")
        failure_payload = failure_payload or {}
        self._reject_reserved_keys(
            failure_payload,
            extra_reserved=self._CONTINUATION_OWNED_FIELDS,
        )
        record.update(deepcopy(failure_payload))
        record["continuation_status"] = "failed_before_continue"
        record["failure_stage"] = failure_stage
        record["error_summary"] = error_summary
        record["status"] = "%s_failed" % failure_stage
        self.status = "failed"

    def _require_pending_handoff(self):
        if self.status not in ("handoff_in_progress", "rebuild_in_progress"):
            raise RuntimeError("handoff session is not accepting continuation work")

    def _reject_reserved_keys(self, payload, extra_reserved=None):
        extra_reserved = extra_reserved or ()
        for key in payload:
            if key in self._RESERVED_KEYS or key in extra_reserved or key.startswith(self._RESERVED_PREFIXES):
                raise RuntimeError("payload may not overwrite reserved handoff fields")

    def _validate_buffer_churn_counts(self, payload):
        for field_name in self._BUFFER_CHURN_FIELDS:
            if field_name not in payload:
                raise RuntimeError(
                    "topology_changed mutation missing buffer churn count: %s"
                    % field_name
                )
            value = payload[field_name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise RuntimeError(
                    "topology_changed mutation invalid buffer churn count: %s"
                    % field_name
                )

    def _get_record(self, handoff_seq):
        if not self.event_history:
            raise RuntimeError("no handoff record is available")
        record = self.event_history[-1]
        if record["handoff_seq"] != handoff_seq:
            raise RuntimeError("handoff sequence out of order")
        return record
