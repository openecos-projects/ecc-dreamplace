"""placeio_ecc boundary for native current driver-tree import/export."""

from collections.abc import Mapping

from dreamplace.ops.rebuffer.pydb_adapter import build_rebuffer_design_from_pydb

ARTIFACT = "placeio_ecc_current_driver_tree"
ARTIFACT_VERSION = 1


def _native_module():
    from dreamplace.ops.rebuffer import rebuffer_cpp

    return rebuffer_cpp


def buffer_net_links(pydb, node_is_buffer):
    """Return ``(input_net_by_output, output_nets_by_input)`` buffer links."""

    return _buffer_net_links(pydb, node_is_buffer)


def _buffer_net_links(pydb, node_is_buffer):
    if node_is_buffer is None:
        design = build_rebuffer_design_from_pydb(
            pydb,
            require_buffer_metadata=True,
            require_names=False,
        )
        if design.get("status") != "ok":
            raise ValueError(
                "current driver-tree metadata unsupported: "
                + ";".join(design.get("unsupported_reasons", []))
            )
        node_is_buffer = design["node_is_buffer"]

    pin2node = [int(value) for value in pydb.pin2node_map]
    pin2net = [int(value) for value in pydb.pin2net_map]
    net2driver = [int(value) for value in pydb.net2driver_pin_map]
    node_pins = [[] for _ in range(len(pydb.node_names))]
    for pin, node in enumerate(pin2node):
        if 0 <= node < len(node_pins):
            node_pins[node].append(pin)

    input_net_by_output = {}
    output_nets_by_input = {}
    for instance, is_buffer in enumerate(node_is_buffer):
        if not bool(is_buffer) or instance >= len(node_pins):
            continue
        pins = node_pins[instance]
        outputs = [
            pin for pin in pins if 0 <= pin < len(pin2net) and net2driver[pin2net[pin]] == pin
        ]
        inputs = [
            pin for pin in pins if 0 <= pin < len(pin2net) and net2driver[pin2net[pin]] != pin
        ]
        if len(outputs) == 1 and len(inputs) == 1:
            input_net = pin2net[inputs[0]]
            output_net = pin2net[outputs[0]]
            input_net_by_output[output_net] = input_net
            output_nets_by_input.setdefault(input_net, []).append(output_net)
    return input_net_by_output, output_nets_by_input


def current_driver_tree_root_net_id(pydb, net_id, node_is_buffer=None, *, links=None):
    """Return the non-buffer-driver root net for a possibly buffered net."""

    input_net_by_output, _ = links or _buffer_net_links(pydb, node_is_buffer)
    current = int(net_id)
    seen = set()
    while current in input_net_by_output:
        if current in seen:
            raise ValueError("buffer net graph contains a cycle")
        seen.add(current)
        current = int(input_net_by_output[current])
    return current


def current_driver_tree_reachable_net_ids(pydb, root_net_id, node_is_buffer=None, *, links=None):
    """Return the root net and all buffer-output nets reachable from it."""

    _, output_nets_by_input = links or _buffer_net_links(pydb, node_is_buffer)

    root_net_id = int(root_net_id)
    if root_net_id < 0 or root_net_id >= len(pydb.net2driver_pin_map):
        raise ValueError("current driver-tree root net is out of range")
    reachable = {root_net_id}
    pending = [root_net_id]
    while pending:
        net_id = pending.pop()
        for output_net in output_nets_by_input.get(net_id, ()):
            if output_net not in reachable:
                reachable.add(output_net)
                pending.append(output_net)
    return sorted(reachable)


def map_current_driver_tree_actions(raw_actions, pydb):
    """Map native tree actions to physical source ids and stable pydb names."""

    from dreamplace.ops.rebuffer.repair_design import map_buffer_actions

    mapped, _ = map_buffer_actions({"actions": raw_actions}, pydb)
    for raw, action in zip(raw_actions, mapped, strict=True):
        if "source_parent_node_id" not in raw or "source_child_node_id" not in raw:
            raise ValueError("native tree action is missing physical source identity")
        action["parent_node_id"] = int(raw["source_parent_node_id"])
        action["child_node_id"] = int(raw["source_child_node_id"])
        action["tree_candidate_parent_graph_id"] = int(raw["parent_node_id"])
        action["tree_candidate_child_graph_id"] = int(raw["child_node_id"])
    return mapped


class CurrentDriverTreeSession:
    """Thin placeio_ecc wrapper around the native tree transaction."""

    def __init__(self, native_tree):
        self._native = _native_module()
        self._transaction = self._native.create_current_driver_tree_transaction(native_tree)
        self._next_checkpoint_id = 0
        self._last_checkpoint_id = -1

    def checkpoint(self):
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        self._last_checkpoint_id = checkpoint_id
        return {
            "checkpoint": self._transaction.checkpoint(),
            "checkpoint_id": checkpoint_id,
            "source_tree_digest": self.summary()["source_tree_digest"],
        }

    def apply_mutation(self, mutation):
        return dict(self._transaction.apply_mutation(mutation))

    def area_recovery_candidates(self, design, liberty):
        """Enumerate native TreeAreaRecovery candidates for the active tree."""

        return list(self._transaction.area_recovery_candidates(design, liberty))

    def run_area_recovery(
        self,
        design,
        liberty,
        physical_checkpoint,
        physical_apply,
        evaluate,
        physical_restore,
        config=None,
    ):
        """Run the C++ TreeAreaRecovery transaction through bridge callbacks."""

        return dict(
            self._native.run_tree_area_recovery(
                self._transaction,
                design,
                liberty,
                physical_checkpoint,
                physical_apply,
                evaluate,
                physical_restore,
                dict(config or {}),
            )
        )

    def apply_candidate_tree(self, candidate, *, source_tree_digest=None):
        """Apply a native complete candidate tree after identity checking."""

        if source_tree_digest is None:
            source_tree_digest = self.summary()["source_tree_digest"]
        return dict(
            self._transaction.apply_candidate_tree(
                candidate,
                str(source_tree_digest),
            )
        )

    def restore(self, checkpoint):
        native_checkpoint = (
            checkpoint.get("checkpoint", checkpoint)
            if isinstance(checkpoint, Mapping)
            else checkpoint
        )
        result = dict(self._transaction.restore(native_checkpoint))
        if isinstance(checkpoint, Mapping) and "checkpoint_id" in checkpoint:
            result["checkpoint_id"] = int(checkpoint["checkpoint_id"])
        return result

    def summary(self):
        result = dict(self._transaction.summary())
        result["artifact"] = ARTIFACT
        result["artifact_version"] = ARTIFACT_VERSION
        result["checkpoint_id"] = self._last_checkpoint_id
        result["transaction_identity"] = {
            "checkpoint_id": self._last_checkpoint_id,
            "source_tree_digest": result["source_tree_digest"],
            "candidate_tree_digest": result["active_tree_digest"],
            "carrier_epoch": -1,
            "timing_epoch": -1,
            "transaction_state": result["transaction_state"],
        }
        return result

    def active_tree(self):
        """Return the native-owned active tree view for a TP callback."""

        return dict(self._transaction.active_tree())

    def evaluate_candidate(
        self,
        evaluator,
        *,
        design_identity=None,
        affected_instance_ids=(),
        affected_net_ids=(),
        endpoint_scope=(),
        timing_epoch=0,
        carrier_epoch=-1,
        requested_metrics=(),
    ):
        """Evaluate one complete native tree candidate through Python TP.

        The callback receives a read-only transaction view.  It returns timing
        observations only; native C++ records the result and owns the
        subsequent accept/restore state transition.
        """

        summary = self.summary()
        if summary["transaction_state"] != "APPLIED":
            raise RuntimeError(
                "current tree candidate must be APPLIED before Python timing evaluation"
            )
        tree = self.active_tree()
        tree_identity = dict(tree["tree_identity"])
        transaction_identity = {
            "checkpoint_id": int(self._last_checkpoint_id),
            "source_tree_digest": summary["source_tree_digest"],
            "candidate_tree_digest": summary["active_tree_digest"],
            "carrier_epoch": int(carrier_epoch),
            "timing_epoch": int(timing_epoch),
            "transaction_state": "APPLIED",
        }
        callback_input = {
            "design_identity": dict(design_identity or {}),
            "transaction_identity": transaction_identity,
            "source_tree_digest": summary["source_tree_digest"],
            "candidate_tree_digest": summary["active_tree_digest"],
            "tree_identity": tree_identity,
            "tree": tree,
            "affected_instance_ids": [int(value) for value in affected_instance_ids],
            "affected_net_ids": [int(value) for value in affected_net_ids],
            "endpoint_scope": [int(value) for value in endpoint_scope],
            "timing_epoch": int(timing_epoch),
            "requested_metrics": [str(value) for value in requested_metrics],
        }
        callback = getattr(evaluator, "evaluate_tree_candidate", evaluator)
        if not callable(callback):
            raise TypeError("candidate evaluator must be callable")
        result = callback(callback_input)
        if not isinstance(result, Mapping):
            raise TypeError("candidate evaluator must return a mapping")
        result = dict(result)
        required = {
            "status",
            "timing_epoch",
            "endpoint_scope_digest",
            "wns_ps",
            "tns_ps",
            "worst_slack_ps",
            "violating_endpoint_count",
            "max_slew_violation",
            "max_cap_violation",
            "area",
            "buffer_count",
            "score_digest",
            "failure_reason",
        }
        missing = sorted(required.difference(result))
        if missing:
            raise ValueError("candidate evaluator result is missing: " + ", ".join(missing))
        if int(result["timing_epoch"]) != int(timing_epoch):
            raise ValueError("timing_epoch_mismatch")
        result.update(
            {
                "source_tree_digest": summary["source_tree_digest"],
                "candidate_tree_digest": summary["active_tree_digest"],
                "timing_epoch": int(timing_epoch),
            }
        )
        return dict(self._transaction.evaluate_candidate(result))

    def accept_candidate(self):
        """Commit native candidate state after a successful TP evaluation."""

        return dict(self._transaction.accept_candidate())


def begin_current_driver_tree_session(pydb, topology, *, root_net_id):
    """Create a native tree transaction from the canonical placeio_ecc view."""

    design = build_rebuffer_design_from_pydb(
        pydb,
        require_buffer_metadata=True,
        require_names=False,
    )
    if design.get("status") != "ok":
        raise ValueError(
            "current driver-tree design view unsupported: "
            + ";".join(design.get("unsupported_reasons", []))
        )
    if not isinstance(topology, dict) or topology.get("status") != "ok":
        raise ValueError("current driver-tree topology view unsupported")
    tree = _native_module().import_current_driver_tree(
        design,
        topology,
        int(root_net_id),
    )
    return CurrentDriverTreeSession(tree)


def import_current_driver_tree(pydb, topology, *, root_net_id):
    """Import one complete current driver tree in native C++.

    Python only normalizes the canonical PyPlaceDB boundary and performs the
    export round-trip probe. Tree traversal, buffer bridges, rerooting,
    identity validation, and digest construction remain native.
    """

    design = build_rebuffer_design_from_pydb(
        pydb,
        require_buffer_metadata=True,
        require_names=False,
    )
    if design.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                design.get("unsupported_reasons", ["design_status_not_ok"])
            ),
            "design_summary": dict(design.get("summary", {})),
        }
    if not isinstance(topology, dict) or topology.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["topology_status_not_ok"],
            "design_summary": dict(design.get("summary", {})),
        }

    native = _native_module()
    tree = native.import_current_driver_tree(
        design,
        topology,
        int(root_net_id),
    )
    exported = dict(native.export_current_driver_tree(tree))
    round_trip = dict(native.validate_current_driver_tree_export(exported))
    if round_trip.get("status") != "ok":
        raise RuntimeError(
            "native current driver-tree export failed round-trip: "
            + str(round_trip.get("reason", "unknown"))
        )
    if round_trip.get("tree_digest") != exported.get("tree_digest"):
        raise RuntimeError("native current driver-tree digest changed on round-trip")

    exported.update(
        {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "design_source": "placedb.pydb",
            "topology_source": topology.get("topology_source", "steiner"),
            "design_summary": dict(design.get("summary", {})),
            "round_trip": round_trip,
        }
    )
    return exported


__all__ = [
    "CurrentDriverTreeSession",
    "begin_current_driver_tree_session",
    "current_driver_tree_reachable_net_ids",
    "current_driver_tree_root_net_id",
    "import_current_driver_tree",
    "map_current_driver_tree_actions",
]
