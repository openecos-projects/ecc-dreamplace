"""Shadow buffer transaction for the ECC placement backend.

The session owns a copy-on-write buffer overlay.  It never mutates the native
ECC database until :meth:`PlaceIOBufferSession.commit`; commit uses the
existing batch ``apply_buffer_actions`` primitive followed by one full
refresh.  Tree search, candidate selection, and accept/rollback decisions stay
in the native rebuffer planner.
"""

import hashlib
import json
import os
from collections.abc import Mapping

from .overlay_metadata import (
    FrozenRecord,
    FrozenValues,
    NativeOverlayMetadata,
    RecordList,
    freeze_record,
    frozen_mutation_arguments,
)

ARTIFACT = "placeio_ecc_buffer_session"
ARTIFACT_VERSION = 1


def action_digest(actions):
    """Return the canonical digest used by the native batch transaction."""

    serialized = json.dumps(
        list(actions),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Gate-only: recompute the digest on every cache hit and require equality.
_CHECK_CACHED_DIGEST = os.environ.get("REBUFFER_CHECK_OVERLAY_DIGEST", "") == "1"


class _ActionList(list):
    """Action list that caches its canonical digest until it is mutated.

    ``overlay_key`` is queried for every planner payload; recomputing the JSON
    digest of all actions each time dominated repair snapshots. Appended
    actions are not edited in place, so list-level mutation is the only
    invalidation event.
    """

    __slots__ = ("_digest", "_freeze")

    def __init__(self, values=(), *, freeze=False, already_frozen=False, digest=None):
        self._freeze = freeze
        if freeze and not already_frozen:
            super().__init__(freeze_record(value) for value in values)
        else:
            super().__init__(values)
        self._digest = digest

    def digest(self):
        if self._digest is None:
            self._digest = action_digest(self)
        elif _CHECK_CACHED_DIGEST:
            fresh = action_digest(self)
            if fresh != self._digest:
                raise AssertionError(
                    "cached overlay digest is stale: an action was edited in place"
                )
        return self._digest


def _invalidating(name):
    method = getattr(list, name)

    def wrapper(self, *args, **kwargs):
        self._digest = None
        if self._freeze:
            args = frozen_mutation_arguments(name, args)
        return method(self, *args, **kwargs)

    wrapper.__name__ = name
    return wrapper


for _name in (
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "__setitem__", "__delitem__", "__iadd__", "__imul__",
):
    setattr(_ActionList, _name, _invalidating(_name))


def _fast_json_safe(value):
    t = type(value)
    if t is str or t is int or t is bool or value is None:
        return value
    if t is float:
        if value != value or value == float("inf") or value == float("-inf"):
            raise ValueError("Out of range float values are not JSON compliant")
        return value
    if t is dict or t is FrozenRecord:
        for k in value:
            if type(k) is not str:
                raise TypeError("non-string dict key")
        return {k: _fast_json_safe(value[k]) for k in sorted(value)}
    if t is list or t is tuple or t is FrozenValues:
        return [_fast_json_safe(item) for item in value]
    raise TypeError("fallback")


def _json_safe(value):
    try:
        return _fast_json_safe(value)
    except TypeError:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True))


def _names(value):
    if value is None:
        return ()
    return tuple(str(item) for item in list(value))


class BufferOverlay(NativeOverlayMetadata):
    """Copy-on-write representation of buffer insertion actions.

    The overlay groups actions by net and assigns a session-local buffer id to
    every inserted buffer.  The base PyPlaceDB is never modified.
    """

    def __init__(self, base_pydb, *, timing_model=None):
        super().__init__()
        if base_pydb is None:
            raise ValueError("buffer overlay requires a base pydb")
        self.base_pydb = base_pydb
        self.timing_model = timing_model
        self._actions = _ActionList()
        self.net_patches = {}
        self.epoch = 0
        self._next_buffer_id = 0
        self.removed_buffer_ids = set()
        self.base_removals = []
        self._checkpoints = {}
        self._next_checkpoint_id = 0
        self.native_topologies = {}
        self._synced_topology_count = 0
        self.native_repair_sessions = {}
        self.native_campaign_session = None
        self.native_timing_bridge = None
        self.native_topology_events = []
        self.projection_audit = {}
        if getattr(timing_model, "use_native_timing_graph", False):
            timing_model.bind_overlay(self)

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, values):
        self._actions = _ActionList(values, freeze=self._metadata_journal_enabled)

    def actions_digest(self):
        """Canonical digest of the current action list (cached until mutation)."""

        return self._actions.digest()

    def _native_module(self):
        if self.timing_model is None or not hasattr(self.timing_model, "_net_record"):
            return None
        try:
            from dreamplace.ops.rebuffer import rebuffer_cpp
        except ImportError:
            return None
        return rebuffer_cpp

    def _native_state(self, net_name):
        state = self.native_topologies.get(str(net_name))
        if state is not None:
            return state
        module = self._native_module()
        if module is None:
            return None
        create_overlay = getattr(self.timing_model, "create_net_overlay", None)
        if create_overlay is not None:
            state = create_overlay(str(net_name))
            self.native_topologies[str(net_name)] = state
            if self.native_timing_bridge is not None and state is not None:
                self.native_timing_bridge.attach_state(state)
                if getattr(self, "_synced_bridge_id", None) == id(self.native_timing_bridge):
                    self._synced_topology_count = len(self.native_topologies)
            return state
        try:
            net_id = int(self.timing_model._net_id(str(net_name)))
            record = self.timing_model._net_record(net_id)
            tree = record["rc_tree"]
            children = tree["children_by_node"]
            edges = []
            nodes = {int(record["driver_pin_id"])}
            for parent, children_for_node in children.items():
                parent = int(parent)
                nodes.add(parent)
                for child in children_for_node:
                    child = int(child)
                    nodes.add(child)
                    rc = tree["edge_rc"].get((parent, child), {})
                    edges.append(
                        {
                            "parent": parent,
                            "child": child,
                            "r": float(rc.get("r", 0.0)),
                            "c": float(rc.get("c", 0.0)),
                        }
                    )
            node_loads = {
                int(node): [int(node)]
                for node in tree.get("sink_nodes", ())
                if int(node) != int(record["driver_pin_id"])
            }
            state = module.create_incremental_topology(
                net_id,
                int(record["driver_pin_id"]),
                max(nodes, default=0) + 1,
                {
                    "edges": edges,
                    "node_loads": node_loads,
                    "coordinates": record.get("coordinates", {}),
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            self.native_topology_events.append(
                {
                    "status": "unsupported",
                    "net_name": str(net_name),
                    "reason": str(error),
                }
            )
            return None
        self.native_topologies[str(net_name)] = state
        if self.native_timing_bridge is not None and state is not None:
            self.native_timing_bridge.attach_state(state)
            if getattr(self, "_synced_bridge_id", None) == id(self.native_timing_bridge):
                self._synced_topology_count = len(self.native_topologies)
        return state

    def _native_repair_session(self, net_name):
        """Return the C++ transaction boundary for one active net tree."""

        name = str(net_name)
        session = self.native_repair_sessions.get(name)
        if session is not None:
            return session
        state = self._native_state(name)
        module = self._native_module()
        if state is None or module is None:
            return None
        factory = getattr(module, "create_repair_timing_session", None)
        if factory is None:
            return None
        session = factory(state)
        self.native_repair_sessions[name] = session
        return session

    def _native_campaign(self):
        """Return the cross-net C++ transaction session, if available."""

        if self.native_campaign_session is not None:
            return self.native_campaign_session
        module = self._native_module()
        if module is None:
            return None
        factory = getattr(module, "create_repair_timing_campaign_session", None)
        if factory is None:
            return None
        self.native_campaign_session = factory()
        return self.native_campaign_session

    def record_native_candidate(self, net_id, observation, state):
        """Record a Python-TimingPropagation candidate in the native ledger."""

        campaign = self._native_campaign()
        if campaign is None:
            return {"status": "unsupported", "reason": "native_campaign_missing"}
        return dict(
            campaign.record_candidate(
                int(net_id),
                _json_safe(dict(observation)),
                str(state),
            )
            or {}
        )

    def record_native_action_state(self, net_id, action_id, state):
        """Record an action lifecycle transition in the native ledger."""

        campaign = self._native_campaign()
        if campaign is None:
            return {"status": "unsupported", "reason": "native_campaign_missing"}
        return dict(
            campaign.record_action_state(
                int(net_id),
                int(action_id),
                str(state),
            )
            or {}
        )

    def replace_session_tree(self, net_name, actions):
        """Atomically replace all session buffers rooted at one net."""

        name = str(net_name)
        snapshot = self._snapshot()
        old_actions = [
            action for action in self.actions if str(action.get("net_name")) == name
        ]
        try:
            campaign = self._native_campaign()
            state = self._native_state(name)
            module = self._native_module()
            replace = (
                getattr(campaign, "replace_candidate_tree", None)
                if campaign is not None
                else None
            )
            if state is None or module is None or not callable(replace):
                self.remove_session_buffers(name)
                result = dict(self.apply(actions) or {})
                result["status"] = result.get("status", "ok")
                result["replacement_mode"] = "overlay_fallback"
                return result

            next_id = int(self._next_buffer_id)
            normalized_actions = []
            native_actions = []
            for action in actions or ():
                candidate = dict(action)
                candidate["session_buffer_id"] = next_id
                normalized = self._normalize_action(
                    candidate, preserve_session_id=True
                )
                native_action = self._native_action(normalized)
                if native_action is None:
                    raise ValueError("replacement action cannot cross native boundary")
                normalized_actions.append(normalized)
                native_actions.append(native_action)
                next_id += 1

            campaign.attach_topology(int(state.net_id), state)
            result = dict(
                replace(int(state.net_id), native_actions) or {}
            )
            applied = list(result.get("applied", ()) or ())
            for normalized, applied_result in zip(
                normalized_actions, applied, strict=True
            ):
                for field in (
                    "parent_node_id",
                    "child_node_id",
                    "split_ratio",
                    "x_dbu",
                    "y_dbu",
                    "resolution_source",
                    "projection_distance_dbu",
                ):
                    if field in applied_result:
                        normalized[field] = applied_result[field]
                if "x_dbu" in applied_result:
                    normalized["candidate_location_x_dbu"] = int(
                        round(applied_result["x_dbu"])
                    )
                if "y_dbu" in applied_result:
                    normalized["candidate_location_y_dbu"] = int(
                        round(applied_result["y_dbu"])
                    )

            self.actions = [
                action
                for action in self.actions
                if str(action.get("net_name")) != name
            ] + normalized_actions
            self.removed_buffer_ids.update(
                int(action["session_buffer_id"]) for action in old_actions
            )
            self._next_buffer_id = next_id
            self._rebuild_patches()
            self.epoch += 1
            result.update(
                {
                    "replacement_mode": "native_campaign",
                    "net_name": name,
                    "session_buffer_ids": [
                        int(action["session_buffer_id"])
                        for action in normalized_actions
                    ],
                    "removed_session_buffer_ids": [
                        int(action["session_buffer_id"]) for action in old_actions
                    ],
                    "epoch": self.epoch,
                }
            )
            return result
        except Exception:
            self._restore_snapshot(snapshot)
            raise

    def _native_action(self, action):
        parent = action.get("parent_node_id")
        child = action.get("child_node_id")
        native_geometry = hasattr(self.timing_model, "create_net_overlay")
        if not native_geometry and (parent is None or child is None):
            return None
        ratio = action.get("split_ratio")
        if not native_geometry and ratio is None:
            return None
        if not action.get("downstream_pin_names"):
            return None
        pin_ids = [
            int(self.timing_model._pin_id(name)) for name in action.get("downstream_pin_names", ())
        ]
        result = {
            "action_id": int(action["session_buffer_id"]),
            "buffer_master_name": str(action.get("buffer_master_name", "")),
            "downstream_pin_ids": pin_ids,
            "x_dbu": action["candidate_location_x_dbu"],
            "y_dbu": action["candidate_location_y_dbu"],
        }
        if parent is not None and child is not None:
            result.update(parent_node_id=int(parent), child_node_id=int(child))
        if ratio is not None:
            result["split_ratio"] = float(ratio)
        return result

    def _apply_native_insert(self, action):
        state = self._native_state(action.get("net_name", ""))
        native_action = self._native_action(action)
        if state is None or native_action is None:
            if hasattr(self.timing_model, "create_net_overlay"):
                raise ValueError("current native topology is required for a repair action")
            self.native_topology_events.append(
                {
                    "status": "unsupported",
                    "net_name": action.get("net_name", ""),
                    "action_id": action.get("session_buffer_id"),
                    "reason": "explicit_active_edge_required",
                }
            )
            return {"status": "unsupported", "reason": "explicit_active_edge_required"}
        from dreamplace.ops.rebuffer import rebuffer_cpp

        native_campaign = self._native_campaign()
        if native_campaign is not None:
            native_campaign.attach_topology(int(state.net_id), state)
            result = dict(
                native_campaign.apply_candidate(int(state.net_id), native_action)
                or {}
            )
            for field in (
                "parent_node_id",
                "child_node_id",
                "split_ratio",
                "x_dbu",
                "y_dbu",
                "resolution_source",
                "projection_distance_dbu",
            ):
                if field not in result:
                    continue
                action[field] = result[field]
            if "x_dbu" in result:
                action["candidate_location_x_dbu"] = int(round(result["x_dbu"]))
            if "y_dbu" in result:
                action["candidate_location_y_dbu"] = int(round(result["y_dbu"]))
            result["net_name"] = action.get("net_name", "")
            result["action_id"] = action.get("session_buffer_id")
            self.native_topology_events.append(result)
            return result

        native_session = self._native_repair_session(action.get("net_name", ""))
        if native_session is not None:
            result = dict(native_session.apply_candidate(native_action) or {})
            for field in (
                "parent_node_id",
                "child_node_id",
                "split_ratio",
                "x_dbu",
                "y_dbu",
                "resolution_source",
                "projection_distance_dbu",
            ):
                if field not in result:
                    continue
                action[field] = result[field]
            if "x_dbu" in result:
                action["candidate_location_x_dbu"] = int(round(result["x_dbu"]))
            if "y_dbu" in result:
                action["candidate_location_y_dbu"] = int(round(result["y_dbu"]))
        else:
            if hasattr(self.timing_model, "create_net_overlay"):
                native_action = rebuffer_cpp.resolve_overlay_split(state, native_action)
                for field in (
                    "parent_node_id",
                    "child_node_id",
                    "split_ratio",
                    "x_dbu",
                    "y_dbu",
                    "resolution_source",
                    "projection_distance_dbu",
                ):
                    if field not in native_action:
                        continue
                    action[field] = native_action[field]
                if "x_dbu" in native_action:
                    action["candidate_location_x_dbu"] = int(round(native_action["x_dbu"]))
                if "y_dbu" in native_action:
                    action["candidate_location_y_dbu"] = int(round(native_action["y_dbu"]))
            result = dict(rebuffer_cpp.apply_incremental_buffer_split(state, native_action) or {})
        result["net_name"] = action.get("net_name", "")
        result["action_id"] = action.get("session_buffer_id")
        self.native_topology_events.append(result)
        return result

    def _apply_native_resize(self, action):
        bridge = self.native_timing_bridge or getattr(
            self.timing_model, "_persistent_graph_bridge", None
        )
        if bridge is None or not hasattr(bridge, "graph"):
            raise ValueError("resize_cell requires a native timing graph bridge")
        if hasattr(bridge, "attach_overlay"):
            bridge.attach_overlay(self)
        result = dict(
            bridge.graph.resize_cell(
                str(action["instance_name"]),
                str(action["target_master_name"]),
            )
            or {}
        )
        result["action_kind"] = "resize_cell"
        result["net_name"] = action.get("net_name", "")
        result["node_id"] = int(action["node_id"])
        self.native_topology_events.append(result)
        return result

    def _collapse_resize_actions(self, resize_actions):
        by_node = {}
        order = []
        for action in resize_actions:
            item = _json_safe(dict(action))
            node_id = int(item["node_id"])
            if node_id not in by_node:
                by_node[node_id] = item
                order.append(node_id)
            else:
                prev = by_node[node_id]
                merged = dict(item)
                if "previous_cell_index" in prev:
                    merged["previous_cell_index"] = prev["previous_cell_index"]
                if "previous_master_name" in prev:
                    merged["previous_master_name"] = prev["previous_master_name"]
                by_node[node_id] = merged
        collapsed = []
        for node_id in order:
            item = by_node[node_id]
            if (
                "previous_cell_index" in item
                and int(item["target_cell_index"]) == int(item["previous_cell_index"])
            ):
                continue
            collapsed.append(item)
        return collapsed

    def commit_actions(self):
        """Export final tree depths and physical load sets from native state."""
        resize_actions = [
            action for action in self.actions if action.get("action_kind") == "resize_cell"
        ]
        buffer_actions = [
            action for action in self.actions if action.get("action_kind") != "resize_cell"
        ]
        collapsed_resizes = self._collapse_resize_actions(resize_actions)
        if not buffer_actions or not hasattr(self.timing_model, "create_net_overlay"):
            is_resize_only = bool(resize_actions and not buffer_actions)
            self.projection_audit = {
                "status": "ok" if is_resize_only else "unsupported",
                "reason": "resize_only" if is_resize_only else "native_projection_unavailable",
                "projected_action_ids": [
                    int(action["session_buffer_id"]) for action in buffer_actions
                ],
                "resized_count": len(collapsed_resizes),
            }
            return collapsed_resizes + _json_safe(buffer_actions)
        from dreamplace.ops.rebuffer import rebuffer_cpp

        states = [self.native_topologies[name] for name in sorted(self.native_topologies)]
        projection_source = "native_overlay"
        projection_error = None
        bridge = self.native_timing_bridge or getattr(
            self.timing_model, "_persistent_graph_bridge", None
        )
        use_graph_bridge = bool(
            getattr(self.timing_model, "use_persistent_graph", False)
            and os.environ.get("REBUFFER_PERSISTENT_GRAPH_FINALIZE", "") == "1"
            and bridge is not None
            and callable(getattr(bridge, "finalize_actions", None))
        )
        if self.native_timing_bridge is not None:
            projected = list(bridge.finalize_actions(buffer_actions, overlay=self))
            projection_source = "native_timing_graph"
        elif use_graph_bridge:
            try:
                projected = list(bridge.finalize_actions(buffer_actions, overlay=self))
                projection_source = "persistent_graph"
            except (
                AttributeError,
                KeyError,
                LookupError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                # The opt-in graph is a guarded projection accelerator. A
                # graph-owned state mismatch retains the canonical native
                # finalizer so it cannot weaken commit correctness.
                projection_error = repr(error)
                projected = list(
                    rebuffer_cpp.finalize_overlay_actions(
                        self.base_pydb, states, buffer_actions
                    )
                )
        else:
            projected = list(
                rebuffer_cpp.finalize_overlay_actions(
                    self.base_pydb, states, buffer_actions
                )
            )
        native_action_ids = []
        for state in states:
            view = rebuffer_cpp.incremental_topology_view(state)
            native_action_ids.extend(
                int(buffer["action_id"])
                for buffer in view.get("buffers", ())
            )
        projected_action_ids = [
            int(action["session_buffer_id"]) for action in projected
        ]
        campaign = self.native_campaign_session
        campaign_summary = (
            dict(campaign.summary()) if campaign is not None else {}
        )
        self.projection_audit = {
            "status": "ok",
            "projection_source": projection_source,
            "native_topology_action_ids": sorted(native_action_ids),
            "projected_action_ids": projected_action_ids,
            "native_active_count": len(native_action_ids),
            "projected_count": len(projected_action_ids),
            "resized_count": len(collapsed_resizes),
            "campaign_ledger_history_count": len(
                campaign_summary.get("ledger_history", ())
            ),
            "campaign_action_ledger": dict(
                campaign_summary.get("action_ledger", {})
            ),
        }
        if projection_error is not None:
            self.projection_audit = {
                **self.projection_audit, "persistent_graph_fallback": projection_error
            }
        return collapsed_resizes + projected

    def _remove_native_insert(self, action):
        state = self.native_topologies.get(str(action.get("net_name", "")))
        if state is None:
            return {"status": "unsupported", "reason": "native_state_missing"}
        from dreamplace.ops.rebuffer import rebuffer_cpp

        native_campaign = self._native_campaign()
        if native_campaign is not None:
            native_campaign.attach_topology(int(state.net_id), state)
            result = dict(
                native_campaign.remove_candidate(
                    int(state.net_id), int(action["session_buffer_id"])
                )
                or {}
            )
            result["net_name"] = action.get("net_name", "")
            result["action_id"] = action.get("session_buffer_id")
            self.native_topology_events.append(result)
            return result

        native_session = self._native_repair_session(action.get("net_name", ""))
        if native_session is not None:
            result = dict(
                native_session.remove_candidate(int(action["session_buffer_id"])) or {}
            )
        else:
            result = dict(
                rebuffer_cpp.remove_incremental_buffer(
                    state, int(action["session_buffer_id"])
                )
                or {}
            )
        result["net_name"] = action.get("net_name", "")
        result["action_id"] = action.get("session_buffer_id")
        self.native_topology_events.append(result)
        return result

    def _net_names(self):
        return _names(getattr(self.base_pydb, "net_names", ()))

    def _known_net(self, net_name):
        name_set = getattr(self, "_net_name_set", None)
        if name_set is None:
            name_set = frozenset(self._net_names())
            self._net_name_set = name_set
        return not name_set or net_name in name_set

    def _normalize_action(self, action, *, preserve_session_id=False):
        if not isinstance(action, Mapping):
            raise TypeError("buffer action must be a mapping")
        normalized = _json_safe(dict(action))
        kind = normalized.get("action_kind")
        if kind == "buffer_remove":
            if normalized.get("session_buffer_id") is None:
                raise ValueError("buffer_remove requires session_buffer_id")
            normalized["session_buffer_id"] = int(normalized["session_buffer_id"])
            return normalized
        if kind == "resize_cell":
            net_name = str(normalized.get("net_name", "") or "")
            if not net_name or not self._known_net(net_name):
                raise ValueError(f"unknown net_name {net_name!r}")
            if normalized.get("node_id") is None or int(normalized["node_id"]) < 0:
                raise ValueError("resize_cell requires nonnegative node_id")
            if not str(normalized.get("instance_name", "") or ""):
                raise ValueError("resize_cell requires instance_name")
            target_idx = normalized.get("target_cell_index")
            if target_idx is None or int(target_idx) < 0:
                raise ValueError("resize_cell requires nonnegative target_cell_index")
            if not str(normalized.get("target_master_name", "") or ""):
                raise ValueError("resize_cell requires target_master_name")
            normalized["node_id"] = int(normalized["node_id"])
            normalized["target_cell_index"] = int(target_idx)
            if normalized.get("previous_cell_index") is not None:
                normalized["previous_cell_index"] = int(normalized["previous_cell_index"])
            normalized.setdefault("commit_order", len(self.actions))
            return normalized
        if kind != "buffer_insert":
            raise ValueError(
                "the shadow buffer session accepts buffer_insert, buffer_remove, or resize_cell"
            )
        net_name = str(normalized.get("net_name", "") or "")
        if not net_name:
            raise ValueError("buffer action requires net_name")
        if not self._known_net(net_name):
            raise ValueError(f"unknown net_name {net_name!r}")
        if not str(normalized.get("buffer_master_name", "") or ""):
            raise ValueError("buffer action requires buffer_master_name")
        if not (
            normalized.get("candidate_location_x_dbu") is not None
            and normalized.get("candidate_location_y_dbu") is not None
        ):
            raise ValueError("buffer action requires candidate_location_x/y_dbu")
        if not (
            normalized.get("downstream_pin_names")
            or normalized.get("downstream_pin_ids")
            or normalized.get("load_pin_name")
        ):
            raise ValueError("buffer action requires a downstream load partition")
        normalized.setdefault("parent_action_id", -1)
        normalized.setdefault("commit_order", len(self.actions))
        if preserve_session_id and normalized.get("session_buffer_id") is not None:
            normalized["session_buffer_id"] = int(normalized["session_buffer_id"])
        else:
            normalized["session_buffer_id"] = int(self._next_buffer_id)
        return normalized

    def seed(self, actions):
        """Rebuild a shadow topology from already verified insertions.

        Seeding never touches the native IDB.  It exists for the next planner
        round after a verified native commit, whose PyPlaceDB refresh is
        intentionally deferred while Python ``TimingPropagation`` is the
        timing authority.
        """

        if self.actions:
            raise ValueError("buffer overlay can only be seeded once")
        snapshot = self._snapshot()
        try:
            for action in actions or ():
                normalized = self._normalize_action(action, preserve_session_id=True)
                if normalized["action_kind"] != "buffer_insert":
                    raise ValueError("seed requires buffer_insert actions")
                if any(
                    item["session_buffer_id"] == normalized["session_buffer_id"]
                    for item in self.actions
                ):
                    raise ValueError("seeded actions contain duplicate session_buffer_id")
                self._apply_native_insert(normalized)
                self.actions.append(normalized)
                self._next_buffer_id = max(
                    self._next_buffer_id, normalized["session_buffer_id"] + 1
                )
        except Exception:
            self._restore_snapshot(snapshot)
            raise
        self._rebuild_patches()
        if self.actions:
            self.epoch += 1
        return {
            "status": "ok",
            "seeded_count": len(self.actions),
            "epoch": self.epoch,
        }

    def _rebuild_patches(self):
        patches = {}
        for action in self.actions:
            net_name = action["net_name"]
            patch = patches.setdefault(
                net_name,
                {
                    "inserted_buffer_ids": [],
                    "load_pin_names": [],
                    "action_count": 0,
                },
            )
            if action.get("action_kind") != "resize_cell":
                patch["inserted_buffer_ids"].append(action["session_buffer_id"])
                patch["load_pin_names"].extend(
                    str(value) for value in action.get("downstream_pin_names", ()) or ()
                )
            patch["action_count"] += 1
        self.net_patches = patches

    def _append_patch_record(self, action):
        """Publish one immutable patch update without freezing all actions.

        LEGACY enables the metadata journal so checkpoints can retain immutable
        roots.  Rebuilding the complete ``net_patches`` mapping after every
        accepted candidate turned that guarantee into an O(history) JSON
        serialization.  An insertion changes only one net, so copy the outer
        root and the affected patch while sharing every other frozen record.
        """

        if not self._metadata_journal_enabled:
            return False
        net_name = action["net_name"]
        current = self._net_patches
        old = current.get(net_name)
        inserted = list(old.get("inserted_buffer_ids", ())) if old is not None else []
        loads = list(old.get("load_pin_names", ())) if old is not None else []
        if action.get("action_kind") != "resize_cell":
            inserted.append(action["session_buffer_id"])
            loads.extend(str(value) for value in action.get("downstream_pin_names", ()) or ())
        patch = FrozenRecord(
            (
                ("inserted_buffer_ids", FrozenValues(inserted)),
                ("load_pin_names", FrozenValues(loads)),
                ("action_count", (int(old["action_count"]) if old is not None else 0) + 1),
            )
        )
        next_patches = dict(current)
        next_patches[net_name] = patch
        self._net_patches = FrozenRecord(next_patches)
        return True

    def apply(self, actions, *, transactional=True):
        if not isinstance(actions, (list, tuple)):
            raise TypeError("actions must be a list or tuple")
        snapshot = self._snapshot() if transactional else None
        try:
            return self._apply_actions(actions)
        except Exception:
            if snapshot is not None:
                self._restore_snapshot(snapshot)
            raise

    def _apply_actions(self, actions):
        added = []
        removed = []
        rebuild_patches = False
        for action in actions:
            normalized = self._normalize_action(action)
            if normalized.get("action_kind") == "buffer_remove":
                target = int(normalized["session_buffer_id"])
                index = next(
                    (
                        position
                        for position, item in enumerate(self.actions)
                        if int(item.get("session_buffer_id", -1)) == target
                    ),
                    None,
                )
                if index is None:
                    raise ValueError(f"unknown session_buffer_id {target}")
                existing = self.actions[index]
                native_result = self._remove_native_insert(existing)
                if native_result.get("status") not in {"ok", "unsupported"}:
                    raise ValueError(native_result.get("reason", "native_remove_failed"))
                self.actions.pop(index)
                self.removed_buffer_ids.add(target)
                removed.append(target)
                rebuild_patches = True
                continue
            if normalized.get("action_kind") == "resize_cell":
                try:
                    self._apply_native_resize(normalized)
                except Exception as error:
                    raise ValueError(
                        f"native resize action failed: {normalized}; reason={error}"
                    ) from error
                self.actions.append(normalized)
                added.append(normalized)
                if not self._append_patch_record(normalized):
                    rebuild_patches = True
                continue
            try:
                self._apply_native_insert(normalized)
            except Exception as error:
                identity = {
                    key: normalized.get(key)
                    for key in (
                        "net_name",
                        "session_buffer_id",
                        "parent_node_id",
                        "child_node_id",
                        "split_ratio",
                        "candidate_location_x_dbu",
                        "candidate_location_y_dbu",
                        "downstream_pin_names",
                        "segment_tree_depth",
                        "source_parent_node_id",
                        "source_child_node_id",
                        "planning_partition_id",
                        "planning_original_net_id",
                        "resolved_state_net_id",
                        "projection_had_active_state",
                        "projection_simulated_bundle_state",
                    )
                }
                raise ValueError(
                    f"native buffer action failed: {identity}; reason={error}"
                ) from error
            self.actions.append(normalized)
            added.append(normalized)
            self._next_buffer_id += 1
            if not self._append_patch_record(normalized):
                rebuild_patches = True
        if rebuild_patches:
            self._rebuild_patches()
        if added or removed:
            self.epoch += 1
        return {
            "status": "ok",
            "applied_count": len(added),
            "removed_count": len(removed),
            "epoch": self.epoch,
            "session_buffer_ids": [
                item["session_buffer_id"]
                for item in added
                if "session_buffer_id" in item
            ],
            "removed_session_buffer_ids": removed,
        }

    def remove_base_buffers(self, instance_names, *, net_name=None):
        """Plan removal of buffers already present in the base PyPlaceDB.

        The native removal transaction is removal-only.  Commit this overlay
        before planning inserts, then refresh and start a new session for the
        replacement tree.
        """

        if self.actions:
            raise ValueError(
                "base buffer removal must be planned before inserts; "
                "commit removals, refresh, then start a new session"
            )
        if isinstance(instance_names, str):
            instance_names = [instance_names]
        names = []
        seen = set()
        for value in instance_names or ():
            name = str(value)
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        if not names:
            return {
                "status": "skipped",
                "removed_count": 0,
                "base_remove_count": len(self.base_removals),
                "epoch": self.epoch,
            }
        if self.native_timing_bridge is not None:
            raise ValueError("native timing campaign does not support base buffer removal")
        for name in names:
            action = {
                "action_kind": "buffer_remove",
                "instance_name": name,
            }
            if net_name:
                action["net_name"] = str(net_name)
            self.base_removals.append(action)
        self.epoch += 1
        return {
            "status": "ok",
            "removed_count": len(names),
            "base_remove_count": len(self.base_removals),
            "epoch": self.epoch,
        }

    def remove_session_buffers(self, net_name=None):
        """Remove session-inserted buffers, optionally restricted to one net."""

        targets = [
            int(action["session_buffer_id"])
            for action in self.actions
            if net_name is None or action.get("net_name") == net_name
        ]
        targets.reverse()
        if not targets:
            return {
                "status": "skipped",
                "removed_count": 0,
                "epoch": self.epoch,
            }
        return self.apply(
            [
                {
                    "action_kind": "buffer_remove",
                    "session_buffer_id": target,
                }
                for target in targets
            ]
        )

    def _snapshot(self):
        module = self._native_module()
        bridge = getattr(self.timing_model, "_persistent_graph_bridge", None)
        metadata = self.metadata_checkpoint() if self._metadata_journal_enabled else {
            "actions": _json_safe(self.actions),
            "base_removals": _json_safe(self.base_removals),
            "native_topology_events": _json_safe(self.native_topology_events),
            "projection_audit": _json_safe(self.projection_audit),
        }
        return {
            **metadata,
            "next_buffer_id": self._next_buffer_id,
            "removed_buffer_ids": set(self.removed_buffer_ids),
            "epoch": self.epoch,
            "native_topologies": dict(self.native_topologies),
            "native_checkpoints": (
                {
                    net_name: module.checkpoint_incremental_topology(state)
                    for net_name, state in self.native_topologies.items()
                }
                if module is not None and self.native_timing_bridge is None
                else {}
            ),
            "native_campaign_checkpoint": (
                self.native_campaign_session.checkpoint()
                if self.native_campaign_session is not None
                and self.native_timing_bridge is None
                else None
            ),
            "native_timing_checkpoint": (
                self.native_timing_bridge.checkpoint()
                if self.native_timing_bridge is not None
                else None
            ),
            "persistent_graph_checkpoint": (
                bridge.checkpoint()
                if getattr(self.timing_model, "use_persistent_graph", False)
                and self.native_timing_bridge is None
                and bridge is not None
                and callable(getattr(bridge, "checkpoint", None))
                else None
            ),
        }

    def checkpoint(self, label=""):
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        self._checkpoints[checkpoint_id] = self._snapshot()
        return {
            "status": "ok",
            "checkpoint_id": checkpoint_id,
            "epoch": self.epoch,
            "action_count": len(self.actions),
        }

    def release_checkpoint(self, checkpoint_id):
        """Drop a native scheduler checkpoint after its restore window closes."""

        checkpoint_id = int(checkpoint_id)
        existed = checkpoint_id in self._checkpoints
        self._checkpoints.pop(checkpoint_id, None)
        return {
            "status": "released" if existed else "already_released",
            "checkpoint_id": checkpoint_id,
        }

    def _restore_snapshot(self, checkpoint):
        shared_metadata = checkpoint.get("metadata_journal", False)
        clone = (lambda value: value) if shared_metadata else _json_safe
        native_timing_checkpoint = checkpoint.get("native_timing_checkpoint")
        if native_timing_checkpoint is not None:
            if self.native_timing_bridge is None:
                raise ValueError("native timing checkpoint has no campaign owner")
            self.native_timing_bridge.restore(native_timing_checkpoint)
        if shared_metadata and self._metadata_journal_enabled:
            self._actions = _ActionList(
                checkpoint["actions"],
                freeze=True,
                already_frozen=True,
                digest=checkpoint.get("actions_digest"),
            )
            self._base_removals = RecordList(
                checkpoint["base_removals"], already_frozen=True
            )
        else:
            self.actions = clone(checkpoint["actions"])
            self.base_removals = clone(checkpoint["base_removals"])
        self._next_buffer_id = checkpoint["next_buffer_id"]
        self.removed_buffer_ids = set(checkpoint["removed_buffer_ids"])
        self.native_topologies = dict(checkpoint["native_topologies"])
        if native_timing_checkpoint is not None and self.native_timing_bridge is not None:
            self._synced_bridge_id = id(self.native_timing_bridge)
            self._synced_topology_count = len(self.native_topologies)
        else:
            self._synced_topology_count = -1
        self.native_repair_sessions = {
            name: session
            for name, session in self.native_repair_sessions.items()
            if name in self.native_topologies
        }
        # Native checkpoints now use journal metadata and restore the state
        # object in place.  Restore those topology objects before the campaign
        # ledger, because the campaign checkpoint retains the same shared
        # topology pointers; a prior remove may otherwise make the checkpoint
        # appear to come from a future state.
        native_checkpoints = checkpoint.get("native_checkpoints", {})
        if native_checkpoints:
            from dreamplace.ops.rebuffer import rebuffer_cpp

            for net_name, native_checkpoint in native_checkpoints.items():
                state = self.native_topologies.get(net_name)
                if state is not None:
                    rebuffer_cpp.restore_incremental_topology(
                        state, native_checkpoint
                    )
        bridge = getattr(self.timing_model, "_persistent_graph_bridge", None)
        persistent_graph_checkpoint = checkpoint.get("persistent_graph_checkpoint")
        if (
            persistent_graph_checkpoint is not None
            and bridge is not None
            and callable(getattr(bridge, "restore", None))
        ):
            bridge.restore(persistent_graph_checkpoint, overlay=self)
        campaign_checkpoint = checkpoint.get("native_campaign_checkpoint")
        if self.native_campaign_session is not None and campaign_checkpoint is not None:
            self.native_campaign_session.restore(campaign_checkpoint)
        elif campaign_checkpoint is None and self.native_timing_bridge is None:
            self.native_campaign_session = None
        if shared_metadata and self._metadata_journal_enabled:
            self._native_topology_events = RecordList(
                checkpoint["native_topology_events"], already_frozen=True
            )
            self._projection_audit = checkpoint.get("projection_audit", FrozenRecord())
            self._net_patches = checkpoint["net_patches"]
        else:
            self.native_topology_events = clone(checkpoint["native_topology_events"])
            self.projection_audit = clone(
                checkpoint.get("projection_audit", {})
            )
            if shared_metadata:
                self.net_patches = checkpoint["net_patches"]
            else:
                self._rebuild_patches()
        self.epoch = checkpoint["epoch"]

    def restore(self, checkpoint_id):
        checkpoint = self._checkpoints.get(int(checkpoint_id))
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint_id {checkpoint_id}")
        self._restore_snapshot(checkpoint)
        return {
            "status": "ok",
            "checkpoint_id": int(checkpoint_id),
            "epoch": self.epoch,
            "action_count": len(self.actions),
        }

    def rollback(self):
        bridge = self.native_timing_bridge or getattr(
            self.timing_model, "_persistent_graph_bridge", None
        )
        if (
            getattr(self.timing_model, "use_persistent_graph", False)
            and bridge is not None
            and callable(getattr(bridge, "reset", None))
        ):
            bridge.reset()
        self.actions = []
        self.net_patches = {}
        self._next_buffer_id = 0
        self.removed_buffer_ids = set()
        self.base_removals = []
        self._checkpoints = {}
        self._next_checkpoint_id = 0
        self.native_topologies = {}
        self._synced_topology_count = 0
        self.native_repair_sessions = {}
        self.native_campaign_session = (
            self.native_timing_bridge.campaign
            if self.native_timing_bridge is not None
            else None
        )
        self.native_topology_events = []
        self.projection_audit = {}
        self.epoch += 1
        return {
            "status": "rolled_back",
            "epoch": self.epoch,
            "action_count": 0,
        }

    def summary(self):
        module = self._native_module()
        return {
            "epoch": int(self.epoch),
            "action_count": len(self.actions),
            "buffer_count": len(self.actions),
            "allocated_buffer_count": int(self._next_buffer_id),
            "removed_buffer_count": len(self.removed_buffer_ids),
            "base_remove_count": len(self.base_removals),
            "base_removals": _json_safe(self.base_removals),
            "net_count": len(self.net_patches),
            "net_patches": _json_safe(self.net_patches),
            "action_digest": action_digest(self.actions),
            "native_topology_events": _json_safe(self.native_topology_events),
            "projection_audit": _json_safe(self.projection_audit),
            "native_topologies": (
                {
                    net_name: module.incremental_topology_view(state)
                    for net_name, state in self.native_topologies.items()
                }
                if module is not None
                else {}
            ),
        }
