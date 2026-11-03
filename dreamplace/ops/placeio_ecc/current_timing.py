"""Transport native current partitions into the existing Python timing ops."""

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .buffer_overlay import action_digest
from .incremental_timing import IncrementalTimingState
from .persistent_timing import _PersistentTimingOwner
from .timing_carrier import CommittedTimingPyDB


@dataclass
class CurrentTimingGraph:
    model: object
    payload: dict
    key: tuple


class PersistentGraphBridge:
    """Guarded bridge from an overlay transaction to the native graph handle.

    The bridge owns the persistent C++ graph and shares each overlay's native
    topology state with it.  Python still constructs the exported carrier and
    runs ``TimingPropagation``; this keeps the timing source and verified
    commit boundary unchanged while making the 4a callback contract testable.
    """

    def __init__(self, owner, base_pydb):
        from dreamplace.ops.rebuffer import rebuffer_cpp
        from dreamplace.ops.rebuffer.payloads import steiner_payload

        self.owner = owner
        self.base_pydb = base_pydb
        self.graph = rebuffer_cpp.create_persistent_timing_graph(base_pydb)
        self._states = {}
        self._initial_checkpoint = self.graph.checkpoint()
        self._base_steiner = steiner_payload(base_pydb, owner)

    def attach_state(self, state):
        if state is None:
            raise ValueError("persistent graph bridge cannot attach a null topology")
        net_id = int(state.net_id)
        previous = self._states.get(net_id)
        if previous is not None and previous is not state:
            raise ValueError("persistent graph bridge received a replacement topology state")
        if previous is None:
            self.graph.attach_net_topology_state(net_id, state)
            self._states[net_id] = state

    def attach_overlay(self, overlay):
        for name in sorted(overlay.native_topologies):
            self.attach_state(overlay.native_topologies[name])

    def checkpoint(self):
        # BufferOverlay already checkpoints each mutable IncrementalTopologyState
        # before applying a candidate.  The bridge only attaches those shared
        # states; it does not mutate PersistentTimingGraph's timing-state
        # journal on the production path.  Avoid copying every topology a
        # second time for every candidate.  Restore clears the graph's binding
        # map and reattaches the overlay's already-restored state objects.
        if callable(getattr(self.graph, "clear_net_topologies", None)):
            return {
                "mode": "bindings_only",
                "attached_net_ids": tuple(sorted(self._states)),
            }
        return {
            "mode": "full",
            "graph": self.graph.checkpoint(),
            "attached_net_ids": tuple(sorted(self._states)),
        }

    def restore(self, checkpoint, overlay=None):
        if checkpoint is None:
            raise ValueError("persistent graph bridge restore requires a checkpoint")
        binding_only = (
            isinstance(checkpoint, dict)
            and checkpoint.get("mode") == "bindings_only"
        )
        graph_checkpoint = (
            checkpoint.get("graph") if isinstance(checkpoint, dict) else checkpoint
        )
        if binding_only:
            result = dict(self.graph.clear_net_topologies() or {})
        elif graph_checkpoint is not None:
            result = dict(self.graph.restore(graph_checkpoint) or {})
        else:
            result = {"status": "ok", "net_count": 0}
        attached_net_ids = set(
            int(net_id)
            for net_id in (
                checkpoint.get("attached_net_ids", ())
                if isinstance(checkpoint, dict)
                else ()
            )
        )
        self._states = {}
        if overlay is not None:
            for name in sorted(overlay.native_topologies):
                state = overlay.native_topologies[name]
                net_id = int(state.net_id)
                if binding_only or net_id not in attached_net_ids:
                    self.graph.attach_net_topology_state(net_id, state)
                self._states[net_id] = state
        return result

    def reset(self):
        result = dict(self.graph.restore(self._initial_checkpoint) or {})
        self._states = {}
        return result

    def apply_buffer_split(self, state, action):
        self.attach_state(state)
        return dict(self.graph.apply_buffer_split(int(state.net_id), action) or {})

    def remove_buffer(self, state, action_id):
        self.attach_state(state)
        return dict(self.graph.remove_buffer(int(state.net_id), int(action_id)) or {})

    def finalize_actions(self, actions, overlay=None):
        # Commit-time projection may be reached before the last evaluate
        # callback (for example after a native no-op). Attach the live
        # topology states here so the graph-owned finalizer sees the same
        # transaction state as the overlay.
        if overlay is not None:
            self.attach_overlay(overlay)
        return list(self.graph.finalize_actions(list(actions)))

    def export(self, overlay):
        self.attach_overlay(overlay)
        return dict(
            self.graph.export_carrier(
                self._base_steiner,
                list(overlay.actions),
                float(self.owner.r_unit),
                float(self.owner.c_unit),
            )
        )


def _evaluate_persistent_graph_overlay(owner, base_pydb, overlay):
    from types import SimpleNamespace

    from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel

    key = overlay_key(overlay)
    history = owner._overlay_runtime_history
    cached = history.pop(key, None)
    if cached is not None:
        if (
            getattr(owner, "_overlay_runtime", None) is None
            or owner._overlay_runtime.key != key
        ):
            if hasattr(owner, "invalidate_native_setup_path_cache"):
                owner.invalidate_native_setup_path_cache()
        owner._overlay_runtime = cached
        owner.model = cached.model.model
        if getattr(owner, "use_persistent_graph", False):
            history[key] = cached
        return {
            "status": "ok",
            "action_count": len(overlay.actions),
            "net_count": len(overlay.native_topologies),
            "topology_source": "persistent_graph_bridge_cached",
            **cached.model.baseline.as_dict(),
        }

    bridge = getattr(owner, "_persistent_graph_bridge", None)
    if bridge is None or bridge.base_pydb is not base_pydb:
        bridge = PersistentGraphBridge(owner, base_pydb)
        owner._persistent_graph_bridge = bridge
    exported = bridge.export(overlay)
    carrier = CommittedTimingPyDB(
        base_pydb,
        SimpleNamespace(**exported["physical"]),
        validate_schema=False,
    )
    steiner_view = (
        exported["steiner"]
        if os.environ.get("REBUFFER_USE_OVERLAY_STEINER", "") == "1"
        else None
    )
    model = RebufferTimingModel(
        carrier,
        r_unit=owner.r_unit,
        c_unit=owner.c_unit,
        scale_factor=owner.scale_factor,
        use_pin_offset=False,
        arcs_info=getattr(owner, "_arcs_info", None),
        # The graph bridge controls physical carrier export. Reusing its
        # native Steiner view is a separate opt-in experiment because it can
        # change the Python FLUTE path and therefore the accepted trajectory.
        # Keep the canonical Python Steiner path unless that experiment is
        # explicitly enabled.
        steiner_view=steiner_view,
        steiner_provider=owner.steiner_provider if steiner_view is None else "ecc",
    )
    owner.model = model.model
    owner._overlay_runtime = CurrentTimingGraph(model, exported, key)
    # Replacing the carrier also replaces the timing graph used by native
    # critical-path extraction. Do not let a baseline path snapshot survive
    # the overlay transition; that would omit newly inserted synthetic pins
    # from the next queue snapshot and change the repair trajectory.
    if hasattr(owner, "invalidate_native_setup_path_cache"):
        owner.invalidate_native_setup_path_cache()
    owner._persistent_timing_runtime = None
    history[key] = owner._overlay_runtime
    while len(history) > owner._overlay_runtime_history_limit:
        history.popitem(last=False)
    return {
        "status": "ok",
        "action_count": len(overlay.actions),
        "net_count": len(overlay.native_topologies),
        "topology_source": "persistent_graph_bridge",
        **model.baseline.as_dict(),
    }


def _persistent_timing_statistics(owner):
    statistics = getattr(owner, "_persistent_timing_statistics", None)
    if statistics is None:
        statistics = {
            "delta_attempts": 0,
            "delta_successes": 0,
            "delta_failures": 0,
            "delta_parent_key_misses": 0,
            "delta_history_hits": 0,
            "delta_state_shape_skips": 0,
            "delta_key_examples": [],
            "same_net_attempts": 0,
            "same_net_successes": 0,
            "same_net_failures": 0,
            "schedule_cache_requests": 0,
            "schedule_cache_hits": 0,
            "full_overlay_fallbacks": 0,
            "last_error": None,
        }
        owner._persistent_timing_statistics = statistics
    return statistics


def overlay_key(overlay):
    # ``epoch`` is an execution counter and changes when a native checkpoint
    # is restored.  It must not invalidate a semantic snapshot for the same
    # action list; the action digest already includes the resolved native
    # split and physical load partition.
    digest = getattr(overlay, "actions_digest", None)
    return digest() if callable(digest) else action_digest(overlay.actions)


def evaluate_current_overlay(owner, base_pydb, overlay):
    from dreamplace.ops.rebuffer import rebuffer_cpp
    from dreamplace.ops.rebuffer.payloads import steiner_payload
    from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel

    if base_pydb is not owner.pydb:
        return {"status": "unsupported", "reason": "base_pydb_mismatch"}
    persistent_statistics = _persistent_timing_statistics(owner)
    if not overlay.actions and not getattr(
        owner, "_force_materialize_empty_overlay", False
    ):
        if hasattr(owner, "invalidate_native_setup_path_cache"):
            owner.invalidate_native_setup_path_cache()
        owner.model = owner._base_operator
        owner._overlay_runtime = None
        # The immutable base operator and its Python TimingPropagation
        # snapshot are unchanged by a shadow overlay. Re-running a full
        # forward on every checkpoint restore only repeats the same baseline
        # evaluation and dominates BM64 rollback timing. Keep the forward
        # fallback for lightweight test doubles and pre-baseline owners.
        if hasattr(owner, "_baseline"):
            return {"status": "ok", "action_count": 0, **owner.baseline.as_dict()}
        snapshot = owner._forward()
        return {"status": "ok", "action_count": 0, **snapshot.as_dict()}
    if owner.use_pin_offset:
        raise ValueError("native partition timing currently requires use_pin_offset=False")

    key = overlay_key(overlay)
    history = owner._overlay_runtime_history
    cached = history.pop(key, None)
    if cached is not None:
        if (
            (getattr(owner, "_overlay_runtime", None) is None or owner._overlay_runtime.key != key)
            and hasattr(owner, "invalidate_native_setup_path_cache")
        ):
            owner.invalidate_native_setup_path_cache()
        owner._overlay_runtime = cached
        owner.model = cached.model.model
        if getattr(owner, "use_persistent_graph", False):
            history[key] = cached
        return {
            "status": "ok",
            "action_count": len(overlay.actions),
            "net_count": len(overlay.native_topologies),
            "topology_source": "native_current_partitions_cached_snapshot",
            **cached.model.baseline.as_dict(),
        }

    runtime = owner._overlay_runtime
    if runtime is not None and runtime.key == key:
        source = (
            "native_current_partitions"
            if "steiner" in runtime.payload
            else "native_session_patch_handle"
        )
        return {
            "status": "ok",
            "action_count": len(overlay.actions),
            "net_count": len(overlay.native_topologies),
            "topology_source": source,
            **runtime.model.baseline.as_dict(),
        }

    # A checkpointed repair candidate normally contains one active action, but
    # a same-net chain may contain several active buffers. Keep one native
    # graph/RC/timing owner across both cases; cross-net parent deltas retain
    # their separate one-new-net path below.
    persistent = getattr(owner, "_persistent_timing_runtime", None)
    if persistent is not None and runtime is not None and len(overlay.actions) > 1:
        persistent_statistics["delta_attempts"] += 1
        parent_key = action_digest(list(overlay.actions[:-1]))
        parent_runtime = runtime
        if runtime.key != parent_key:
            parent_runtime = history.get(parent_key)
            if parent_runtime is not None:
                persistent_statistics["delta_history_hits"] += 1
        action = overlay.actions[-1]
        state = overlay.native_topologies.get(str(action.get("net_name", "")))
        state_buffers = ()
        if state is not None:
            from dreamplace.ops.rebuffer import rebuffer_cpp

            state_buffers = rebuffer_cpp.incremental_topology_view(state).get(
                "buffers", ()
            )
        if (
            parent_runtime is not None
            and state is not None
            and len(state_buffers) == 1
        ):
            try:
                snapshot, current = persistent.evaluate_delta(
                    parent_runtime, overlay, action, state
                )
            except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
                owner._persistent_timing_error = repr(error)
                persistent_statistics["delta_failures"] += 1
                persistent_statistics["last_error"] = repr(error)
            else:
                persistent_statistics["delta_successes"] += 1
                if hasattr(owner, "invalidate_native_setup_path_cache"):
                    owner.invalidate_native_setup_path_cache()
                history[parent_runtime.key] = parent_runtime
                key = overlay_key(overlay)
                runtime_model = _PersistentTimingOwner(
                    parent_runtime.model,
                    current,
                    snapshot,
                    dict(persistent.payload),
                )
                owner.model = current.model
                owner._overlay_runtime = CurrentTimingGraph(
                    runtime_model,
                    dict(persistent.payload),
                    key,
                )
                return {
                    "status": "ok",
                    "action_count": len(overlay.actions),
                    "net_count": len(overlay.native_topologies),
                    "topology_source": "native_session_parent_delta_patch",
                    **snapshot.as_dict(),
                }
        elif parent_runtime is None:
            persistent_statistics["delta_parent_key_misses"] += 1
            examples = persistent_statistics["delta_key_examples"]
            if len(examples) < 5:
                examples.append(
                    {
                        "action_count": len(overlay.actions),
                        "runtime_key": runtime.key,
                        "parent_key": parent_key,
                    }
                )
        else:
            persistent_statistics["delta_state_shape_skips"] += 1
    same_net_persistent = False
    if persistent is not None and overlay.actions:
        same_net_persistent = (
            len({str(action.get("net_name", "")) for action in overlay.actions}) == 1
            and len(overlay.native_topologies) == 1
        )
    if persistent is not None and same_net_persistent:
        persistent_statistics["same_net_attempts"] += 1
        try:
            snapshot = persistent.evaluate(overlay)
        except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
            # Unsupported graph shape or Liberty contract must retain the
            # exact carrier path as the correctness fallback.
            owner._persistent_timing_error = repr(error)
            persistent_statistics["same_net_failures"] += 1
            persistent_statistics["last_error"] = repr(error)
            persistent = None
        else:
            persistent_statistics["same_net_successes"] += 1
            if hasattr(owner, "invalidate_native_setup_path_cache"):
                owner.invalidate_native_setup_path_cache()
            key = overlay_key(overlay)
            owner.model = persistent.slot_model
            runtime_model = _PersistentTimingOwner(
                owner,
                persistent.current,
                snapshot,
                dict(persistent.payload),
            )
            owner._overlay_runtime = CurrentTimingGraph(
                runtime_model,
                persistent.payload,
                key,
            )
            return {
                "status": "ok",
                "action_count": len(overlay.actions),
                "net_count": len(overlay.native_topologies),
                "topology_source": "native_session_patch_handle",
                **snapshot.as_dict(),
            }
    if runtime is None or runtime.key != key:
        persistent_statistics["full_overlay_fallbacks"] += 1
        if hasattr(owner, "invalidate_native_setup_path_cache"):
            owner.invalidate_native_setup_path_cache()
        if runtime is not None and "steiner" in runtime.payload:
            history[runtime.key] = runtime
        states = [overlay.native_topologies[name] for name in sorted(overlay.native_topologies)]
        started = time.perf_counter()
        # The graph bridge owns the first physical export for the overlay.
        # Once a runtime exists, keep the canonical exporter as the carrier
        # predecessor for insertion-monotonic reconstruction; replacing that
        # predecessor with a new bridge wrapper increases full fallbacks.
        use_bridge_export = (
            getattr(owner, "use_persistent_graph", False) and runtime is None
        )
        if use_bridge_export:
            bridge = getattr(owner, "_persistent_graph_bridge", None)
            if bridge is None or bridge.base_pydb is not base_pydb:
                bridge = PersistentGraphBridge(owner, base_pydb)
                owner._persistent_graph_bridge = bridge
            payload = bridge.export(overlay)
        else:
            payload = rebuffer_cpp.export_overlay_carrier(
                base_pydb,
                steiner_payload(base_pydb, owner),
                states,
                list(overlay.actions),
                owner.r_unit,
                owner.c_unit,
            )
        exported = time.perf_counter()
        previous_pydb = None if runtime is None else runtime.model.pydb
        if previous_pydb is not None and (
            not hasattr(previous_pydb, "node_names")
            or bool(getattr(previous_pydb, "_compact_timing_graph", False))
        ):
            # The persistent one-slot handle exposes a compact graph view;
            # it is not a CommittedTimingPyDB and cannot be used as the
            # insertion-monotonic carrier predecessor.
            previous_pydb = None
        # ``rebuild_timing_carrier_incremental`` matches timing pins and
        # instances by stable names. It supports both adding and removing
        # synthetic buffer nodes, so a current-tree replacement can retain
        # the accepted carrier as its predecessor even when the replacement
        # tree has fewer buffers. Falling back merely because the new
        # physical view is smaller turns every replacement into a full
        # carrier rebuild and prevents the OpenROAD-style loop from reaching
        # its iteration budget.
        carrier = CommittedTimingPyDB(
            base_pydb,
            SimpleNamespace(**payload["physical"]),
            previous_pydb=previous_pydb,
            validate_schema=False,
        )
        carrier_mode = "incremental" if previous_pydb is not None else "full"
        carrier_stats = getattr(owner, "_carrier_rebuild_stats", None)
        if carrier_stats is None:
            carrier_stats = {"incremental": 0, "full": 0, "last_mode": None}
            owner._carrier_rebuild_stats = carrier_stats
        carrier_stats[carrier_mode] = int(carrier_stats.get(carrier_mode, 0)) + 1
        carrier_stats["last_mode"] = carrier_mode
        rebuilt = time.perf_counter()
        previous_timing_owner = None
        if (
            runtime is not None
            and hasattr(runtime.model, "loads")
            and not bool(getattr(runtime.model.pydb, "_compact_timing_graph", False))
        ):
            previous_timing_owner = runtime.model
        steiner_view = (
            payload.get("steiner")
            if os.environ.get("REBUFFER_USE_OVERLAY_STEINER", "") == "1"
            else None
        )
        model = RebufferTimingModel(
            carrier,
            r_unit=owner.r_unit,
            c_unit=owner.c_unit,
            scale_factor=owner.scale_factor,
            use_pin_offset=owner.use_pin_offset,
            arcs_info=getattr(owner, "_arcs_info", None),
            # The exported partition tree preserves native action identity for
            # projection. Keep the fresh Python Steiner build as the default
            # compatibility path; the opt-in view reuses the native current
            # Steiner/RC topology already produced for this overlay.
            steiner_view=steiner_view,
            steiner_provider=owner.steiner_provider if steiner_view is None else "ecc",
            incremental_state=(
                None
                if runtime is not None and previous_timing_owner is None
                else IncrementalTimingState(
                    owner if previous_timing_owner is None else previous_timing_owner
                )
            ),
        )
        profile = getattr(model.model, "last_incremental_profile", {})
        model.model.last_incremental_profile = dict(profile)
        model.model.last_incremental_profile.update(
            export_carrier_ms=(exported - started) * 1000,
            rebuild_carrier_ms=(rebuilt - exported) * 1000,
            build_model_ms=(time.perf_counter() - rebuilt) * 1000,
            carrier_rebuild_mode=carrier_mode,
        )
        runtime = CurrentTimingGraph(model, payload, key)
        owner._overlay_runtime = runtime
        if getattr(owner, "_persistent_timing_runtime", None) is None:
            from .persistent_timing import PersistentTimingRuntime

            owner._persistent_timing_runtime = PersistentTimingRuntime(owner, runtime)
        history[key] = runtime
        while len(history) > owner._overlay_runtime_history_limit:
            history.popitem(last=False)
    owner.model = runtime.model.model
    result = {
        "status": "ok",
        "action_count": len(overlay.actions),
        "net_count": len(overlay.native_topologies),
        "topology_source": "native_current_partitions",
        **runtime.model.baseline.as_dict(),
    }
    if getattr(owner, "_persistent_timing_error", None):
        result["persistent_timing_fallback"] = owner._persistent_timing_error
    return result
