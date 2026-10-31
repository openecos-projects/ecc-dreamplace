"""Transport native current partitions into the existing Python timing ops."""

import time
from dataclasses import dataclass
from types import SimpleNamespace

from .buffer_overlay import action_digest
from .incremental_timing import IncrementalTimingState
from .timing_carrier import CommittedTimingPyDB


@dataclass
class CurrentTimingGraph:
    model: object
    payload: dict
    key: tuple


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
        parent_key = action_digest(list(overlay.actions[:-1]))
        action = overlay.actions[-1]
        state = overlay.native_topologies.get(str(action.get("net_name", "")))
        state_buffers = ()
        if state is not None:
            from dreamplace.ops.rebuffer import rebuffer_cpp

            state_buffers = rebuffer_cpp.incremental_topology_view(state).get(
                "buffers", ()
            )
        if (
            runtime.key == parent_key
            and state is not None
            and len(state_buffers) == 1
            and not bool(getattr(runtime.model.pydb, "_compact_timing_graph", False))
        ):
            try:
                snapshot, current = persistent.evaluate_delta(
                    runtime, overlay, action, state
                )
            except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
                owner._persistent_timing_error = repr(error)
            else:
                if hasattr(owner, "invalidate_native_setup_path_cache"):
                    owner.invalidate_native_setup_path_cache()
                history[runtime.key] = runtime
                key = overlay_key(overlay)
                owner.model = current.model
                runtime_model = SimpleNamespace(
                    model=current.model,
                    pydb=current.pydb,
                    baseline=snapshot,
                    delays=current.delays,
                    loads=current.loads,
                    impulses=current.impulses,
                    lib_pin_caps=current.lib_pin_caps,
                    pin_max_slew=current.pin_max_slew,
                    pin_max_cap=current.pin_max_cap,
                )
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
    same_net_persistent = False
    if persistent is not None and overlay.actions:
        same_net_persistent = (
            len({str(action.get("net_name", "")) for action in overlay.actions}) == 1
            and len(overlay.native_topologies) == 1
        )
    if persistent is not None and same_net_persistent:
        try:
            snapshot = persistent.evaluate(overlay)
        except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
            # Unsupported graph shape or Liberty contract must retain the
            # exact carrier path as the correctness fallback.
            owner._persistent_timing_error = repr(error)
            persistent = None
        else:
            if hasattr(owner, "invalidate_native_setup_path_cache"):
                owner.invalidate_native_setup_path_cache()
            key = overlay_key(overlay)
            owner.model = persistent.slot_model
            owner._overlay_runtime = CurrentTimingGraph(
                SimpleNamespace(
                    model=persistent.slot_model,
                    pydb=persistent.current.pydb,
                    baseline=snapshot,
                    delays=persistent.current.delays,
                    loads=persistent.current.loads,
                    impulses=persistent.current.impulses,
                    lib_pin_caps=persistent.current.lib_pin_caps,
                    pin_max_slew=persistent.current.pin_max_slew,
                    pin_max_cap=persistent.current.pin_max_cap,
                ),
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
        if hasattr(owner, "invalidate_native_setup_path_cache"):
            owner.invalidate_native_setup_path_cache()
        if runtime is not None and "steiner" in runtime.payload:
            history[runtime.key] = runtime
        states = [overlay.native_topologies[name] for name in sorted(overlay.native_topologies)]
        started = time.perf_counter()
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
        model = RebufferTimingModel(
            carrier,
            r_unit=owner.r_unit,
            c_unit=owner.c_unit,
            scale_factor=owner.scale_factor,
            use_pin_offset=owner.use_pin_offset,
            # The exported partition tree preserves native action identity for
            # projection. Timing uses a fresh ECC Steiner/RC build over the
            # physical overlay, matching the post-commit PyPlaceDB path.
            steiner_view=None,
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
