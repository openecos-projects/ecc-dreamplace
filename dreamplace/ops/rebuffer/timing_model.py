"""Python STA bridge for the rebuffer overlay gate.

The module only marshals ``PyPlaceDB`` arrays into the existing ECC Steiner,
RC and timing-propagation operators.  Tree search, candidate selection and
accept/rollback remain native.  ``evaluate_action`` is intentionally limited to
one ``buffer_insert`` action so the P5.0b overlay/native gate has a narrow,
auditable contract.
"""

from collections import OrderedDict

import numpy as np

from .buffer_lut import LibertyBufferLut
from .timing_model_payload import (
    TimingSnapshot,
    timing_pin_slack,
)

ARTIFACT = "rebuffer_timing_model"
ARTIFACT_VERSION = 1


class RebufferTimingModel:
    """Placement-RC timing model used by the rebuffer overlay gate."""

    def __init__(
        self,
        pydb,
        *,
        r_unit=0.914,
        c_unit=1.89e-4,
        scale_factor=1.0,
        dbu=None,
        use_pin_offset=True,
        sparse_timing_view=None,
        steiner_view=None,
        incremental_state=None,
    ):
        if pydb is None:
            raise ValueError("timing model requires a pydb")
        self.pydb = pydb
        self.dbu = float(dbu if dbu is not None else getattr(pydb, "dbu", 0) or 0)
        if self.dbu <= 0:
            raise ValueError("timing model requires a positive dbu")
        self.r_unit = float(r_unit)
        self.c_unit = float(c_unit)
        self.scale_factor = float(scale_factor)
        self.use_pin_offset = bool(use_pin_offset)
        self.steiner_view = steiner_view
        self._incremental_state = incremental_state
        from dreamplace.ops.placeio_ecc.timing_schema import validate_timing_schema

        schema_view = validate_timing_schema(pydb)
        self.sparse_timing_view = (
            sparse_timing_view if sparse_timing_view is not None else schema_view
        )
        self._build()
        self._base_operator = self.model
        self._overlay_runtime = None
        # A small bounded history makes checkpoint/restore reuse the exact
        # Python TimingPropagation slot order and state. Older checkpoints
        # fall back to native carrier reconstruction instead of retaining an
        # unbounded graph chain.
        self._overlay_runtime_history = OrderedDict()
        self._overlay_runtime_history_limit = 4
        self._carrier_rebuild_stats = {
            "incremental": 0,
            "full": 0,
            "last_mode": None,
        }
        self._bound_overlay = None
        self._native_setup_path_cache = None
        self._native_setup_path_cache_epoch = None
        self._native_setup_path_stats = {
            "request_count": 0,
            "cache_hit_count": 0,
            "unsupported_count": 0,
            "failure_count": 0,
            "extraction_count": 0,
            "last_error_type": None,
            "last_extraction": None,
        }

    def _build(self):
        from .timing_model_build import build_timing_model

        build_timing_model(self)

    def _forward(self, *, dynamic_net_provider=None):
        # TimingPropagation consumes the subtree load at each driver pin.
        # This is the same contract as PlaceObj's production call, which
        # passes RCTiming.loads rather than the local node capacitance.
        runtime = getattr(self, "_overlay_runtime", None)
        active = runtime.model if runtime is not None else self
        wns, tns, worst, _ = self.model(
            {"rise": active.delays["rise"], "fall": active.delays["fall"]},
            {"rise": active.impulses["rise"], "fall": active.impulses["fall"]},
            {"rise": active.loads["rise"], "fall": active.loads["fall"]},
            surrogate_mode="lut_only",
            dynamic_net_provider=dynamic_net_provider,
        )
        return self._snapshot_from_outputs(wns, tns, worst)

    def _snapshot_from_outputs(self, wns, tns, worst):
        # A new forward invalidates the native predecessor/path snapshot.  The
        # repair planner may request it once for the current timing state and
        # reuse the resulting endpoint map across all nets on that state.
        self.invalidate_native_setup_path_cache()
        if not np.isfinite([float(wns), float(tns), float(worst)]).all():
            raise ValueError("TimingPropagation returned nonfinite repair timing")
        slack = timing_pin_slack(self.model).cpu().numpy()
        endpoints = self.model.end_points.detach().cpu().numpy()
        endpoint_slack = slack[endpoints]
        return TimingSnapshot(
            wns_ps=float(wns),
            tns_ps=float(tns),
            worst_slack_ps=float(worst),
            endpoint_count=int(np.count_nonzero(np.isfinite(endpoint_slack))),
            violating_endpoint_count=int(np.count_nonzero(endpoint_slack < 0.0)),
        )

    def native_setup_path_details(self):
        """Return native transition/arc-aware setup paths for the current state.

        The timing propagation operator already owns a C++ predecessor
        extractor.  Capture one hard timing snapshot for the current model and
        index its failing rise/fall states by endpoint.  Python only selects
        the endpoint entry and transports the path pins; it does not rebuild a
        predecessor path from combined rise/fall slack.
        """

        model = self.model
        if self._native_setup_path_cache is not None:
            current_epoch = getattr(model, "critical_path_topology_epoch", None)
            if (
                current_epoch is not None
                and self._native_setup_path_cache_epoch is not None
                and int(current_epoch) != int(self._native_setup_path_cache_epoch)
            ):
                self.invalidate_native_setup_path_cache()
            else:
                self._native_setup_path_stats["cache_hit_count"] += 1
                return self._native_setup_path_cache
        required = (
            getattr(model, "request_critical_path_snapshot", None),
            getattr(model, "extract_setup_critical_paths", None),
        )
        if any(not callable(value) for value in required):
            self._native_setup_path_stats["unsupported_count"] += 1
            return None
        self._native_setup_path_stats["request_count"] += 1
        try:
            # The current state was just evaluated; packing it avoids a second
            # full forward. Older operators without retained state re-run it.
            capture = getattr(model, "capture_critical_path_snapshot_from_state", None)
            if not (callable(capture) and capture()):
                model.request_critical_path_snapshot()
                self._forward()
            batch = model.extract_setup_critical_paths(
                global_k=0,
                max_depth=0,
                residual_tolerance_ps=1.0e-3,
            )
        except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
            clear_request = getattr(model, "clear_critical_path_snapshot_request", None)
            if callable(clear_request):
                clear_request()
            self._native_setup_path_stats["failure_count"] += 1
            self._native_setup_path_stats["last_error_type"] = type(error).__name__
            self.invalidate_native_setup_path_cache()
            return None

        # The native extractor identifies the winning transition and arc, but
        # OpenROAD's root correction also needs the current arrival at the
        # driver's input and output pins.  These values are already produced
        # by Python TimingPropagation; copy them into each path record so the
        # C++ rebuffer DP can apply the correction without an STA callback.
        def arrival_snapshot(name):
            value = getattr(model, name, None)
            if value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            return np.asarray(value, dtype=np.float64)

        rise_aat = arrival_snapshot("pin_rAAT_live_snapshot")
        fall_aat = arrival_snapshot("pin_fAAT_live_snapshot")

        offsets = batch.path_offsets.detach().cpu().tolist()
        pins = batch.path_pins.detach().cpu().tolist()
        transitions = batch.path_transitions.detach().cpu().tolist()
        arc_ids = batch.path_arc_ids.detach().cpu().tolist()
        endpoints = batch.endpoint_pins.detach().cpu().tolist()
        endpoint_transitions = batch.endpoint_transitions.detach().cpu().tolist()
        endpoint_slacks = batch.endpoint_slacks.detach().cpu().tolist()
        valid = batch.path_valid.detach().cpu().tolist()
        path_delays_tensor = getattr(batch, "path_delays", None)
        path_delays = (
            None
            if path_delays_tensor is None
            else path_delays_tensor.detach().cpu().tolist()
        )
        details = {}
        arc_cursor = 0
        for index, endpoint in enumerate(endpoints):
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            arc_begin = arc_cursor
            arc_end = arc_begin + max(0, end - begin - 1)
            arc_cursor = arc_end
            candidate = {
                "path": [int(value) for value in pins[begin:end]],
                "transitions": [int(value) for value in transitions[begin:end]],
                "arc_ids": [int(value) for value in arc_ids[arc_begin:arc_end]],
                "endpoint_transition": int(endpoint_transitions[index]),
                "endpoint_slack": float(endpoint_slacks[index]),
                "valid": bool(valid[index]),
                "topology_epoch": int(batch.topology_epoch),
            }
            if path_delays is not None:
                candidate["path_delays"] = [
                    float(value) for value in path_delays[arc_begin:arc_end]
                ]
            if (
                rise_aat is not None
                and fall_aat is not None
                and len(candidate["path"]) == len(candidate["transitions"])
                and all(0 <= pin < len(rise_aat) for pin in candidate["path"])
            ):
                candidate["path_arrivals"] = [
                    float((rise_aat if transition == 0 else fall_aat)[pin])
                    for pin, transition in zip(
                        candidate["path"], candidate["transitions"], strict=True
                    )
                ]
            previous = details.get(int(endpoint))
            if previous is None or candidate["endpoint_slack"] < previous["endpoint_slack"]:
                details[int(endpoint)] = candidate
        self._native_setup_path_cache = details
        self._native_setup_path_cache_epoch = int(batch.topology_epoch)
        self._native_setup_path_stats["extraction_count"] += 1
        extraction_stats = getattr(model, "last_critical_path_extraction_stats", None)
        self._native_setup_path_stats["last_extraction"] = (
            dict(extraction_stats) if isinstance(extraction_stats, dict) else None
        )
        return details

    def invalidate_native_setup_path_cache(self):
        """Invalidate path details after a topology or timing-state change."""

        if not hasattr(self, "_native_setup_path_cache"):
            self._native_setup_path_cache = None
        if not hasattr(self, "_native_setup_path_cache_epoch"):
            self._native_setup_path_cache_epoch = None
        if not hasattr(self, "_native_setup_path_stats"):
            self._native_setup_path_stats = {
                "request_count": 0,
                "cache_hit_count": 0,
                "unsupported_count": 0,
                "failure_count": 0,
                "extraction_count": 0,
                "invalidation_count": 0,
                "last_error_type": None,
                "last_extraction": None,
            }
        if self._native_setup_path_cache is not None:
            self._native_setup_path_stats["invalidation_count"] = (
                self._native_setup_path_stats.get("invalidation_count", 0) + 1
            )
        self._native_setup_path_cache = None
        self._native_setup_path_cache_epoch = None

    def native_setup_path_statistics(self):
        """Return JSON-safe statistics for the native path adapter."""

        stats = dict(self._native_setup_path_stats)
        if isinstance(stats.get("last_extraction"), dict):
            stats["last_extraction"] = dict(stats["last_extraction"])
        return stats

    def carrier_rebuild_statistics(self):
        """Return carrier rebuild provenance for the current overlay campaign."""

        return dict(getattr(self, "_carrier_rebuild_stats", {}))

    @property
    def baseline(self):
        return self._baseline

    def create_net_overlay(self, net_name):
        """Import the existing Steiner arrays through the native boundary."""

        from . import rebuffer_cpp
        from .payloads import steiner_payload

        net_id = self._net_id(net_name)
        return rebuffer_cpp.incremental_topology_from_steiner(
            steiner_payload(self.pydb, self),
            net_id,
            int(self.pydb.net2driver_pin_map[net_id]),
            self.r_unit,
            self.c_unit,
        )

    def _ensure_name_maps(self):
        if hasattr(self, "_pin_name_to_id"):
            return
        self._pin_name_to_id = {}
        for index, name in enumerate(self.pydb.pin_names):
            normalized = str(name).replace("/", ":").replace("\\", ":")
            self._pin_name_to_id.setdefault(normalized, int(index))
        self._net_name_to_id = {}
        for index, name in enumerate(getattr(self.pydb, "net_names", ()) or ()):
            self._net_name_to_id.setdefault(str(name), int(index))
        self._net_record_cache = {}

    def _net_record(self, net_id):
        self._ensure_name_maps()
        cached = self._net_record_cache.get(int(net_id))
        if cached is not None:
            return cached
        pydb = self.pydb
        start = int(self.steiner.net_flat_topo_sort_start[net_id])
        end = int(self.steiner.net_flat_topo_sort_start[net_id + 1])
        nodes = [int(value) for value in self.steiner.net_flat_topo_sort[start:end]]
        node_set = set(nodes)
        children = {node: [] for node in nodes}
        for node in nodes:
            begin = int(self.steiner.flat_pin_to_start[node])
            finish = int(self.steiner.flat_pin_to_start[node + 1])
            for child in self.steiner.flat_pin_to[begin:finish]:
                child = int(child)
                if child in node_set:
                    children[node].append(child)
        edge_rc = {}
        for node in nodes:
            for child in children[node]:
                resistance = float(self.rc.flat_pin_to_res[child])
                edge_rc[(node, child)] = {
                    "r": resistance,
                    "c": resistance / self.r_unit * self.c_unit,
                }
        driver = int(np.asarray(pydb.net2driver_pin_map)[net_id])
        sinks = [node for node in nodes if node < len(pydb.pin_names) and node != driver]
        record = {
            "net_id": int(net_id),
            "driver_pin_id": driver,
            "rc_tree": {
                "root_node_id": driver,
                "children_by_node": children,
                "edge_rc": edge_rc,
                "node_cap": {node: float(self.rc.net_cap[node]) for node in nodes},
                "sink_nodes": sinks,
            },
            "coordinates": {
                node: (float(self.new_x[node]), float(self.new_y[node])) for node in nodes
            },
        }
        self._net_record_cache[int(net_id)] = record
        return record

    def _pin_id(self, name):
        self._ensure_name_maps()
        normalized = str(name).replace("/", ":").replace("\\", ":")
        pin_id = self._pin_name_to_id.get(normalized)
        if pin_id is None:
            raise ValueError(f"unknown pin {name!r}")
        return pin_id

    def _net_id(self, name):
        self._ensure_name_maps()
        net_id = self._net_name_to_id.get(str(name))
        if net_id is None:
            raise ValueError(f"unknown net {name!r}")
        return net_id

    @property
    def active_pydb(self):
        return self.pydb if self._overlay_runtime is None else self._overlay_runtime.model.pydb

    def bind_overlay(self, overlay):
        self._bound_overlay = overlay

    def current_graph(self):
        if self._bound_overlay is not None:
            from dreamplace.ops.placeio_ecc.current_timing import overlay_key

            overlay = self._bound_overlay
            if (self._overlay_runtime is not None or overlay.actions) and (
                self._overlay_runtime is None or self._overlay_runtime.key != overlay_key(overlay)
            ):
                self.evaluate(self.pydb, overlay)
        return self._overlay_runtime

    def force_full_overlay_runtime(self):
        """Materialize the bound overlay before native planning needs its carrier."""

        overlay = self._bound_overlay
        if overlay is None:
            return self._overlay_runtime
        # The one-slot patch has no complete physical carrier or partition
        # mapping. Drop it before evaluate() so current_timing takes the
        # canonical carrier path and rebuilds the current ECC Steiner/RC.
        self._persistent_timing_runtime = None
        self._overlay_runtime = None
        self._force_materialize_empty_overlay = True
        try:
            self.evaluate(self.pydb, overlay)
        finally:
            self._force_materialize_empty_overlay = False
        return self._overlay_runtime

    def sync_overlay_state(self, overlay):
        return self.evaluate(self.pydb, overlay)

    def evaluate(self, base_pydb, overlay):
        from dreamplace.ops.placeio_ecc.current_timing import evaluate_current_overlay

        self._bound_overlay = overlay
        return evaluate_current_overlay(self, base_pydb, overlay)

    def evaluate_action(self, action):
        from dreamplace.ops.placeio_ecc.buffer_overlay import BufferOverlay

        overlay = BufferOverlay(self.pydb, timing_model=self)
        overlay.apply([action])
        result = self.evaluate(self.pydb, overlay)
        return TimingSnapshot(
            **{name: result[name] for name in TimingSnapshot.__dataclass_fields__}
        )


__all__ = ["LibertyBufferLut", "RebufferTimingModel", "TimingSnapshot"]
