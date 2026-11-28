"""Native rebuffer forest skeleton over the canonical PyPlaceDB view."""

from __future__ import annotations

import numpy as np

from .pydb_adapter import build_rebuffer_design_from_pydb

ARTIFACT = "rebuffer_forest"
ARTIFACT_VERSION = 1

_DESIGN_ARRAYS = (
    "flat_net2pin_map",
    "flat_net2pin_start_map",
    "net2driver_pin_map",
    "pin2node_map",
    "pin2net_map",
    "pin_offset_x",
    "pin_offset_y",
    "node_x",
    "node_y",
)
_TOPOLOGY_ARRAYS = (
    "net_ids",
    "net_flat_topo_sort_start",
    "net_flat_topo_sort",
    "pin_fa",
)


def _extension():
    try:
        import dreamplace.ops.rebuffer.rebuffer_cpp as native
    except ImportError as exc:
        raise RuntimeError(
            "rebuffer_cpp is not built; the native forest skeleton is unavailable"
        ) from exc
    return native


def _normalized_design(design):
    if not isinstance(design, dict):
        raise TypeError("design must be a mapping")
    if design.get("status") != "ok":
        return None
    normalized = dict(design)
    for name in _DESIGN_ARRAYS:
        normalized[name] = np.ascontiguousarray(np.asarray(design[name]))
    return normalized


def _normalized_topology(topology):
    if not isinstance(topology, dict):
        raise TypeError("topology must be a mapping")
    if topology.get("status") != "ok":
        return None
    normalized = dict(topology)
    for name in _TOPOLOGY_ARRAYS:
        normalized[name] = np.ascontiguousarray(np.asarray(topology[name]))
    return normalized


def build_rebuffer_forest(design, topology):
    """Build the native forest from an adapter view and Steiner topology."""

    normalized_design = _normalized_design(design)
    if normalized_design is None:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["design_status_not_ok"],
        }
    normalized_topology = _normalized_topology(topology)
    if normalized_topology is None:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["topology_status_not_ok"],
        }

    result = dict(_extension().build_forest(normalized_design, normalized_topology))
    result.update(
        {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "design_source": design.get("source", "pydb"),
            "topology_source": topology.get("topology_source", "steiner"),
        }
    )
    return result


def build_rebuffer_forest_from_pydb(pydb, *, topology=None, max_nets=None):
    """Build the design view and forest from ``placedb.pydb``."""

    design = build_rebuffer_design_from_pydb(pydb)
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
    if topology is None:
        from dreamplace.ops.steiner_topo.pydb_topology import (
            build_local_steiner_topology_from_pydb,
        )

        topology = build_local_steiner_topology_from_pydb(
            pydb,
            max_nets=max_nets,
        )
    forest = build_rebuffer_forest(design, topology)
    forest["design_summary"] = dict(design.get("summary", {}))
    forest["topology_summary"] = {
        "status": topology.get("status"),
        "net_count": len(topology.get("net_ids", ())),
        "node_count": len(topology.get("pin_fa", ())),
        "steiner_node_base": topology.get("steiner_node_base"),
        "skipped_reasons": dict(topology.get("skipped_reasons", {})),
    }
    return forest
