"""Join the placement overlay to one native timing/campaign journal.

Inputs come from the immutable Python TimingPropagation owner and PyPlaceDB.
The default graph mode retains the Python timing consumer. An additional
opt-in evaluates candidates in the native graph and materializes Python TP
only when a carrier consumer requests it. Verified commit keeps fresh TP.
"""

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

from .current_timing import PersistentGraphBridge


@dataclass(frozen=True)
class NativeTimingCheckpoint:
    campaign: object
    evaluated_epoch: int
    evaluation: dict


class NativeTimingBridge(PersistentGraphBridge):
    """A campaign owns topology, RC and timing, including the empty root."""

    def __init__(self, owner, base_pydb):
        if owner.use_pin_offset:
            raise ValueError("native timing graph requires use_pin_offset=False")
        super().__init__(owner, base_pydb)
        from dreamplace.ops.rebuffer import rebuffer_cpp
        from dreamplace.ops.rebuffer.liberty import build_liberty_view_from_pydb
        from dreamplace.ops.rebuffer.propagation_payload import persistent_propagation_payload
        from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb
        from dreamplace.ops.rebuffer.timing_model_payload import persistent_timing_state_payload

        # A model can bind a second overlay after evaluating the first. Import
        # its immutable base operator, never that first overlay's live state.
        inputs = SimpleNamespace(pydb=base_pydb, model=owner._base_operator)
        self.graph.initialize_timing_state(persistent_timing_state_payload(inputs))
        self.graph.initialize_propagation(persistent_propagation_payload(inputs))
        self.graph.initialize_mutations()
        design = build_rebuffer_design_from_pydb(base_pydb)
        if design["status"] != "ok":
            raise ValueError("native timing graph requires a complete PyPlaceDB design")
        self.graph.initialize_queue(
            build_liberty_view_from_pydb(base_pydb), design["node_is_buffer"].tolist()
        )
        self.campaign = rebuffer_cpp.create_repair_timing_campaign_session(self.graph)
        self.propagation_count = 0
        self.evaluation_counts = {"full_seed": 0, "events": 0}
        self.full_fallbacks = []
        self.evaluation_cache_hits = 0
        self.restore_count = 0
        self.queue_snapshot_count = 0
        self.python_materialization_count = 0
        self.native_observation_count = 0
        self.planning_snapshot_count = 0
        self._planning_config = None
        self._last_planning_snapshot = None
        self._planning_completed = {
            "planned_requests": 0,
            "imported_tree_count": 0,
            "local_filled_pin_count": 0,
            "local_filled_vertex_count": 0,
        }
        self._planning_completed_ms = {}
        self._remember_evaluation(self.graph.propagate())
        self._initial_checkpoint = self.checkpoint()

    def attach_state(self, state):
        if state is None:
            raise ValueError("native timing bridge cannot attach a null topology")
        # The campaign/graph own the pointer map. A Python binding cache would
        # retain states that a root restore has detached from the journal.
        self.campaign.attach_topology(int(state.net_id), state)

    def apply_buffer_split(self, state, action):
        self.attach_state(state)
        return dict(self.campaign.apply_candidate(int(state.net_id), action))

    def remove_buffer(self, state, action_id):
        self.attach_state(state)
        return dict(self.campaign.remove_candidate(int(state.net_id), int(action_id)))

    def checkpoint(self):
        return NativeTimingCheckpoint(
            self.campaign.checkpoint(), self._evaluated_epoch, self._evaluation
        )

    def restore(self, checkpoint, overlay=None):
        """Restore native state; campaign.summary() explicitly exports diagnostics."""
        if not isinstance(checkpoint, NativeTimingCheckpoint):
            raise ValueError("native timing bridge requires its campaign checkpoint")
        self.campaign.restore_state(checkpoint.campaign)
        self._evaluated_epoch = checkpoint.evaluated_epoch
        self._evaluation = checkpoint.evaluation
        self.restore_count += 1
        return {"status": "ok", "epoch": int(self.graph.epoch)}

    def reset(self):
        return self.restore(self._initial_checkpoint)

    def _remember_evaluation(self, result, *, elapsed_ms=0.0):
        self._evaluated_epoch = int(self.graph.epoch)
        mode = result["mode"]
        if mode == "full_seed" and self.propagation_count:
            self.full_fallbacks.append(
                {
                    "reason": result["fallback_reason"],
                    "epoch": self._evaluated_epoch,
                    "phase": "native_propagation",
                    "elapsed_ms": elapsed_ms,
                }
            )
        self.propagation_count += 1
        self.evaluation_counts[mode] = self.evaluation_counts.get(mode, 0) + 1
        # Retain no NumPy state views across a mutation/checkpoint. The native
        # graph owns the arrays; these are small metrics and coverage records.
        self._evaluation = {
            "status": "ok",
            "mode": result["mode"],
            "epoch": self._evaluated_epoch,
            "metrics": dict(result["metrics"]),
            "coverage": dict(result["coverage"]),
            "changed_pin_count": len(result["changed_pins"]),
            "changed_endpoint_count": len(result["changed_endpoints"]),
        }
        if "fallback_reason" in result:
            self._evaluation["fallback_reason"] = result["fallback_reason"]

    def evaluate(self, overlay):
        self.attach_overlay(overlay)
        if self._evaluated_epoch == int(self.graph.epoch):
            self.evaluation_cache_hits += 1
        else:
            started = perf_counter()
            result = self.graph.propagate_incremental()
            self._remember_evaluation(result, elapsed_ms=(perf_counter() - started) * 1000.0)
        return {
            **self._evaluation,
            "metrics": dict(self._evaluation["metrics"]),
            "coverage": dict(self._evaluation["coverage"]),
        }

    def statistics(self):
        planning = dict(self._planning_completed)
        phase_timings = dict(self._planning_completed_ms)
        if self._last_planning_snapshot is not None:
            current = self._last_planning_snapshot.summary()
            for key in planning:
                planning[key] += int(current.get(key, 0))
            for name, elapsed in current["phase_timings_ms"].items():
                phase_timings[name] = phase_timings.get(name, 0.0) + elapsed
        planning["phase_timings_ms"] = phase_timings
        return {
            "mode": (
                "native_campaign_with_native_timing_consumer"
                if self.owner.use_native_timing_evaluation
                else "native_campaign_with_python_timing_consumer"
            ),
            "propagation_count": self.propagation_count,
            "evaluation_counts": dict(self.evaluation_counts),
            "full_fallback_count": len(self.full_fallbacks),
            "full_fallbacks": [dict(entry) for entry in self.full_fallbacks],
            "evaluation_cache_hits": self.evaluation_cache_hits,
            "restore_count": self.restore_count,
            "queue_snapshot_count": self.queue_snapshot_count,
            "python_materialization_count": self.python_materialization_count,
            "native_observation_count": self.native_observation_count,
            "planning_snapshot_count": self.planning_snapshot_count,
            "native_planning": planning,
            "mutation_statistics": dict(self.graph.mutation_statistics()),
            "state_view_statistics": dict(self.graph.state_view_statistics()),
            "epoch": int(self.graph.epoch),
            "evaluation": {
                **self._evaluation,
                "metrics": dict(self._evaluation["metrics"]),
                "coverage": dict(self._evaluation["coverage"]),
            },
        }

    def materialize(self, overlay, *, gradient_probe=False):
        if self._planning_config is not None and not gradient_probe:
            raise RuntimeError("native planning cannot materialize a Python timing carrier")
        from .current_timing import evaluate_current_overlay

        result = evaluate_current_overlay(self.owner, self.base_pydb, overlay)
        self.python_materialization_count += 1
        return result

    def initialize_planning(self, context):
        if self._planning_config is not None:
            if context.config != self._planning_config:
                raise ValueError("native planning inputs require the same campaign configuration")
            return
        self.graph.initialize_planning(
            context.design,
            context.characterization,
            self._base_steiner,
            context.config,
            self.owner.r_unit,
            self.owner.c_unit,
        )
        self._planning_config = dict(context.config)

    def planning_snapshot(self, overlay, config):
        self.evaluate(overlay)
        builder = (
            self.graph.local_planning_snapshot
            if config.get("native_timing_local_views", False)
            else self.graph.planning_snapshot
        )
        snapshot = builder(list(overlay.actions), config)
        if self._last_planning_snapshot is not None:
            previous = self._last_planning_snapshot.summary()
            for key in self._planning_completed:
                self._planning_completed[key] += int(previous.get(key, 0))
            for name, elapsed in previous["phase_timings_ms"].items():
                self._planning_completed_ms[name] = (
                    self._planning_completed_ms.get(name, 0.0) + elapsed
                )
        self._last_planning_snapshot = snapshot
        self.planning_snapshot_count += 1
        self.queue_snapshot_count += 1
        return snapshot

    def observation(self, *, max_fanout, cap_margin, slew_margin):
        result = dict(self.graph.evaluation_observation(max_fanout, cap_margin, slew_margin))
        self.native_observation_count += 1
        return result

    def endpoint_state(self):
        return self.graph.timing_endpoint_state()

    def queue_state(self, overlay, carrier, *, use_path_context=False):
        self.evaluate(overlay)
        result = dict(
            self.graph.timing_queue_state_for_carrier(
                list(range(len(self.base_pydb.net_names))), carrier, use_path_context
            )
        )
        self.queue_snapshot_count += 1
        return result
