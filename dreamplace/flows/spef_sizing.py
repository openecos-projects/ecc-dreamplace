"""Prepare immutable post-route SPEF RC trees for standalone sizing/STA.

Reuses ``RouteSnapshot`` and ``GRParasiticsOp`` (`placedb.gr_sizing`) so
``PlaceObj``, ``HoldRC``, ``DiscreteTimingOwner``, and ``run_sta_only`` consume
extracted post-route parasitics without special-case conditionals.
"""

import json
import time
from pathlib import Path

import torch

from dreamplace.flows.gr_sizing import (
    GRSizingSession,
    _file_hash,
    build_signal_pin_mapping,
    compute_placedb_geometry_sha256,
)
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.route_snapshot import RouteIdentity


def resolve_spef_input(params) -> str:
    direct = str(getattr(params, "spef_input", "") or "").strip()
    if direct:
        return direct
    design_inputs = dict(getattr(params, "design_inputs", {}) or {})
    for key in ("spef_input", "spef", "spef_path"):
        candidate = str(design_inputs.get(key) or "").strip()
        if candidate:
            return candidate
    return ""


def prepare_spef_sizing(params, placedb) -> None:
    started = time.perf_counter()
    spef_raw = resolve_spef_input(params)
    if not spef_raw:
        raise ValueError("timing_rc_mode=spef requires spef_input")
    spef_path = Path(spef_raw)
    if not spef_path.is_file():
        raise FileNotFoundError(f"SPEF file does not exist: {spef_path}")

    coupling_cap_scale = float(getattr(params, "spef_coupling_cap_scale", 1.0))
    result_dir = Path(params.result_dir) / "spef_parasitics"
    result_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_signal_pin_mapping(params, placedb)
    identity = RouteIdentity(
        _file_hash(spef_path),
        compute_placedb_geometry_sha256(placedb),
        1,
    )
    op = GRParasiticsOp.prepare_spef(
        spef_path,
        mapping,
        identity,
        coupling_cap_scale=coupling_cap_scale,
        dtype=getattr(torch, params.dtype),
    )
    tree = op.snapshot
    report = {
        "timing_rc_mode": "spef",
        "spef_input": str(spef_path),
        **tree.model_report,
        "input_sha256": identity.input_sha256,
        "geometry_sha256": identity.geometry_sha256,
        "router_calls": 0,
        "prepare_seconds": time.perf_counter() - started,
        "vertices": tree.parent.numel(),
        "edges": tree.edge_parent.numel(),
        "covered_pins": tree.valid_pin_ids.numel(),
        "eligible_nets": tree.net_ids.numel(),
        "filtered_nets": tree.filtered_net_count,
        "wire_cap_pf": float(tree.wire_cap.sum()),
        "threads": params.num_threads,
    }
    (result_dir / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    placedb.gr_sizing = GRSizingSession(op, report, mapping.pin_names)


__all__ = ["prepare_spef_sizing", "resolve_spef_input"]
