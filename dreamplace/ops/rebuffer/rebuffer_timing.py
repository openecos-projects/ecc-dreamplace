"""Native ``repair_timing`` payload context over ``placedb.pydb``.

The native side owns path-driver ranking, current-tree VGDP and candidate
choice; this module only marshals the active Python TimingPropagation views
and maps native ids back to ECC names.
"""

from types import SimpleNamespace

from .characterization import characterize_buffers_from_pydb
from .liberty import build_liberty_view_from_pydb
from .payloads import steiner_payload, timing_payload
from .pydb_adapter import build_rebuffer_design_from_pydb
from .repair_design import DEFAULT_CONFIG as REPAIR_DESIGN_CONFIG
from .repair_design import map_buffer_actions
from .timing_model import RebufferTimingModel


def _unsupported(reasons):
    return {"status": "unsupported", "unsupported_reasons": list(reasons), "actions": []}


def _select_buffer_set(characterization, config):
    """Apply the requested native buffer candidate set.

    ``fast`` is the OpenROAD ``Rebuffer::init`` set.  ``filtered`` keeps the
    larger repair_design set; it is useful as a QoR control because the
    smaller fast set changes the accepted trajectory on BM64.
    """

    if str(config.get("buffer_set", "fast")) != "filtered":
        return characterization
    selected = dict(characterization)
    selected["fast_buffers"] = []
    return selected


def _net_id(pydb, net):
    if isinstance(net, str):
        net_names = [str(name) for name in pydb.net_names]
        return net_names.index(net) if net in net_names else None
    net_id = int(net)
    return net_id if 0 <= net_id < len(pydb.net_names) else None


def _base_runtime_view(pydb, timing_model):
    """Return a read-only runtime identity for the unmodified carrier.

    Native projection needs the same provenance fields as an overlay runtime,
    but the first candidate should not require exporting an empty full carrier.
    The base TimingPropagation Steiner view already contains stable physical
    node identities, so construct only those transport arrays here.
    """

    steiner = steiner_payload(pydb, timing_model)
    starts = [int(value) for value in steiner["net_flat_topo_sort_start"]]
    topo = [int(value) for value in steiner["net_flat_topo_sort"]]
    vertex_count = len(steiner["node_x"])
    state_node_by_vertex = [-1] * vertex_count
    state_net_by_vertex = [-1] * vertex_count
    for net_id in range(len(starts) - 1):
        for node in topo[starts[net_id] : starts[net_id + 1]]:
            if 0 <= node < vertex_count:
                state_node_by_vertex[node] = node
                state_net_by_vertex[node] = net_id
    return SimpleNamespace(
        model=timing_model,
        payload={
            "steiner": steiner,
            "original_net_by_net": list(range(len(pydb.net_names))),
            "state_node_by_vertex": state_node_by_vertex,
            "state_net_by_vertex": state_net_by_vertex,
        },
    )


class RebufferPlanContext:
    """Cached native payloads for repeated current-tree planning.

    Library data and the base graph are cached. During a session, native
    current partitions provide the physical graph and existing Steiner view
    used for every subsequent DP call.
    """

    def __init__(self, pydb, timing_model=None, *, config=None):
        self.pydb = pydb
        self.model = timing_model if timing_model is not None else RebufferTimingModel(pydb)
        self._base_runtime = _base_runtime_view(pydb, self.model)
        # A complete tree may require a chain or several branch buffers, so
        # current-tree planning keeps the repair_design action budget.
        self.config = {**REPAIR_DESIGN_CONFIG, **dict(config or {})}
        self.design = build_rebuffer_design_from_pydb(pydb)
        self.liberty = build_liberty_view_from_pydb(pydb)
        wire_rc = {
            key: float(
                config[key]
                if config and key in config
                else getattr(self.model, key, self.config[key])
            )
            for key in ("r_unit", "c_unit")
        }
        self.characterization = _select_buffer_set(
            characterize_buffers_from_pydb(pydb, **wire_rc),
            self.config,
        )
        self._current_design = None
        self._current_graph = None
        self._buffer_links = None
        self._path_detail = None
        self.status = "ok"
        self.reasons = []
        for name, payload in (
            ("design_view_not_ok", self.design),
            ("liberty_view_not_ok", self.liberty),
            ("characterization_not_ok", self.characterization),
        ):
            if payload.get("status") != "ok":
                self.status = "unsupported"
                self.reasons.extend(payload.get("unsupported_reasons", [name]))

    def _runtime(self, required_key="steiner"):
        """Return the materialized overlay runtime, forcing a full rebuild
        when its payload lacks ``required_key``."""

        runtime = self.model.current_graph()
        if runtime is None:
            runtime = self._base_runtime
        if runtime is not None and required_key not in runtime.payload:
            runtime = self.model.force_full_overlay_runtime()
        return runtime

    def _active(self, runtime):
        """Return ``(timing model, pydb, design)`` for the active carrier."""

        if runtime is None:
            return self.model, self.pydb, self.design
        if self._current_graph is not runtime:
            self._current_design = build_rebuffer_design_from_pydb(runtime.model.pydb)
            self._current_graph = runtime
            self._buffer_links = None
        return runtime.model, runtime.model.pydb, self._current_design

    def _driver_tree_links(self, pydb, design):
        """Return buffer input/output net links for the active carrier.

        The links depend only on carrier connectivity, so they are computed
        once per carrier instead of once per current-tree payload.
        """

        from dreamplace.ops.placeio_ecc.current_driver_tree import buffer_net_links

        key = (id(pydb), id(design))
        if self._buffer_links is None or self._buffer_links[0] != key:
            self._buffer_links = (key, buffer_net_links(pydb, design.get("node_is_buffer")))
        return self._buffer_links[1]

    def set_path_context(self, path, path_details=None):
        """Bind one native critical path for path-specific scoring."""

        self._path_detail = None
        if not path:
            return
        snapshot = (
            path_details
            if path_details is not None
            else self.model.native_setup_path_details()
        )
        if not isinstance(snapshot, dict):
            return
        candidate = snapshot.get(int(path[-1]))
        if candidate is None or not candidate.get("valid"):
            return
        if [int(pin) for pin in candidate.get("path", ())] == [int(pin) for pin in path]:
            self._path_detail = dict(candidate)

    def _path_rank_context(self, pydb, path):
        """Build the native path-edge timing payload for driver ranking."""

        detail = self._path_detail
        if not detail:
            return None
        path = [int(pin) for pin in path]
        if path != [int(pin) for pin in detail.get("path", ())]:
            return None
        transitions = [int(value) for value in detail.get("transitions", ())]
        delays = [float(value) for value in detail.get("path_delays", ())]
        graph_arcs = [int(value) for value in detail.get("arc_ids", ())]
        if len(transitions) != len(path) or len(delays) != max(0, len(path) - 1):
            return None
        flat_arcs = getattr(pydb, "flat_inst_arcs_by_level", None)
        flat_lib_arcs = getattr(pydb, "flat_libarc_info", None)
        if flat_arcs is None or flat_lib_arcs is None:
            return None
        local_ids = [-1] * len(delays)
        for edge, graph_arc in enumerate(graph_arcs):
            if graph_arc < 0 or graph_arc >= len(flat_arcs):
                continue
            row = flat_arcs[graph_arc]
            if hasattr(row, "detach"):
                row = row.detach().cpu().tolist()
            if len(row) <= 3:
                continue
            global_arc = int(row[3])
            if 0 <= global_arc < len(flat_lib_arcs):
                local_ids[edge] = int(flat_lib_arcs[global_arc][3])
        return {
            "path_arc_delays": delays,
            "path_arc_local_ids": local_ids,
            "path_transitions": transitions,
        }

    def native_path_context(self, path, path_details=None):
        """Export path arc data for the C++ queue ranking policy."""

        runtime = self._runtime()
        active_pydb = self.pydb if runtime is None else runtime.model.pydb
        self.set_path_context(path, path_details=path_details)
        return self._path_rank_context(active_pydb, path) or {}

    def native_queue_views(self):
        """Return the current native views used by the C++ timing queue."""

        runtime = self._runtime()
        active, active_pydb, design = self._active(runtime)
        if design is None or design.get("status") != "ok":
            return _unsupported(
                (design or {}).get("unsupported_reasons", ["current_design_not_ok"])
            )
        original_net_by_net = list(range(len(active_pydb.net_names)))
        if runtime is not None:
            values = runtime.payload.get("original_net_by_net", ())
            if hasattr(values, "detach"):
                values = values.detach().cpu()
            if hasattr(values, "tolist"):
                values = values.tolist()
            original_net_by_net = [int(value) for value in list(values)]
        return {
            "status": "ok",
            "design": design,
            "timing": timing_payload(active_pydb, active),
            "liberty": self.liberty,
            "original_net_by_net": original_net_by_net,
        }

    def timing_gradient_scores(self):
        """Return opt-in Python-TP scores for native queue ordering."""

        from dreamplace.ops.placeio_ecc.gradient_order import (
            compute_timing_gradient_scores,
        )

        # A stale-gradient reject limit intentionally requests a fresh score
        # at the same overlay state. Do not content-cache this opt-in probe.
        return compute_timing_gradient_scores(self.model)

    def timing_gradient_ranking_oracle(self, path, *, path_details=None, gradient_result=None):
        """Compare native path-driver order with the Python TP score map."""

        from dreamplace.ops.placeio_ecc.gradient_order import (
            compare_path_driver_ranking,
        )

        return compare_path_driver_ranking(
            self,
            path,
            path_details=path_details,
            gradient_result=gradient_result,
        )

    def native_current_tree_payload(
        self,
        net,
        *,
        endpoint_slack_ps=None,
        load_slack_override=None,
    ):
        """Export one native current-tree VGDP input without planning it."""

        from dreamplace.ops.placeio_ecc.current_driver_tree import (
            current_driver_tree_reachable_net_ids,
            current_driver_tree_root_net_id,
        )

        from . import rebuffer_cpp
        from .payloads import current_tree_topology_payload

        if self.status != "ok":
            return _unsupported(self.reasons)

        runtime = self._runtime()
        active, active_pydb, design = self._active(runtime)
        net_id = _net_id(active_pydb, net)
        if net_id is None:
            return _unsupported([f"unknown_net:{net}"])
        if design is None or design.get("status") != "ok":
            return _unsupported(
                (design or {}).get("unsupported_reasons", ["current_design_not_ok"])
            )

        links = self._driver_tree_links(active_pydb, design)
        root_net_id = (
            int(net_id)
            if bool(self.config.get("native_tree_per_driver", False))
            else current_driver_tree_root_net_id(active_pydb, net_id, links=links)
        )
        reachable_net_ids = current_driver_tree_reachable_net_ids(
            active_pydb, root_net_id, links=links
        )
        topology = current_tree_topology_payload(
            active_pydb,
            active,
            reachable_net_ids,
            # The projector resolves source vertex IDs through this carrier's
            # provenance. A Python Steiner rebuild may reuse those integers
            # for other junctions, even when the physical tree is unchanged.
            steiner_view=runtime.payload["steiner"],
        )
        if topology.get("status") != "ok":
            return _unsupported(
                topology.get("unsupported_reasons", ["current_tree_topology_not_ok"])
            )
        tree_cache = []

        def current_tree():
            # Importing is pure in (design, topology, root); defer it until
            # the native queue actually requests this payload.
            if not tree_cache:
                tree_cache.append(
                    rebuffer_cpp.import_current_driver_tree(design, topology, int(root_net_id))
                )
            return tree_cache[0]

        timing = timing_payload(active_pydb, active)
        if endpoint_slack_ps is not None and load_slack_override:
            pin_slack = timing["pin_slack"].copy()
            for pin, value in load_slack_override.items():
                index = int(pin)
                if 0 <= index < len(pin_slack):
                    pin_slack[index] = float(value)
            timing["pin_slack"] = pin_slack
        return {
            "status": "ok",
            "net_id": int(net_id),
            "root_net_id": int(root_net_id),
            "net_name": str(active_pydb.net_names[root_net_id]),
            "reachable_net_ids": reachable_net_ids,
            "current_tree": current_tree,
            "topology_source": topology["topology_source"],
            "design": design,
            "timing": timing,
            "liberty": self.liberty,
            "characterization": self.characterization,
            "config": dict(self.config),
        }

    def _project(self, runtime, original_net_ids, actions):
        """Project mapped actions through the native placeio_ecc carrier."""

        from . import rebuffer_cpp

        overlay = self.model._bound_overlay
        original_net_by_net = runtime.payload["original_net_by_net"]
        for current_net in original_net_ids:
            if 0 <= current_net < len(original_net_by_net):
                original_net = int(original_net_by_net[current_net])
                if 0 <= original_net < len(self.pydb.net_names):
                    overlay._native_state(str(self.pydb.net_names[original_net]))
        states = [overlay.native_topologies[name] for name in sorted(overlay.native_topologies)]
        return list(
            rebuffer_cpp.project_overlay_actions(self.pydb, runtime.payload, states, actions)
        )

    def map_native_actions(self, values):
        """Map raw single-partition C++ actions through the current carrier."""

        runtime = self._runtime("original_net_by_net")
        if runtime is None:
            return map_buffer_actions(values, self.pydb)[0]
        raw_actions = list(values.get("actions", ()) or ())
        return self._project(
            runtime, [int(action["net_id"]) for action in raw_actions], raw_actions
        )

    def map_native_tree_actions(self, values):
        """Map raw native VGDP actions through the physical carrier."""

        from dreamplace.ops.placeio_ecc.current_driver_tree import (
            current_driver_tree_reachable_net_ids,
            current_driver_tree_root_net_id,
            map_current_driver_tree_actions,
        )

        runtime = self._runtime("original_net_by_net")
        if runtime is None:
            runtime = self.model.force_full_overlay_runtime()
        if runtime is None:
            raise RuntimeError("native tree action mapping requires a materialized overlay runtime")
        _, active_pydb, active_design = self._active(runtime)
        raw_actions = list(values.get("actions", ()) or ())
        mapped = map_current_driver_tree_actions(raw_actions, active_pydb)
        # The same carrier already served this tree's payload; reuse its
        # design view and buffer links instead of re-exporting them by name.
        links = self._driver_tree_links(active_pydb, active_design)
        reachable = []
        for raw_action in raw_actions:
            partition = int(raw_action["net_id"])
            if partition < 0 or partition >= len(active_pydb.net_names):
                continue
            root = current_driver_tree_root_net_id(active_pydb, partition, links=links)
            reachable.extend(current_driver_tree_reachable_net_ids(active_pydb, root, links=links))
        return self._project(runtime, reachable, mapped)


__all__ = ["RebufferPlanContext"]
