"""Initial and terminal routing for frozen-GR standalone sizing/STA.

The optimizer references one immutable snapshot. Only this module invokes the
router; the terminal invocation parses the committed DEF into a new router DB.
"""

import copy
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from dreamplace.ops.buffer_insertion.net_eligibility import clock_net_ids
from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity
from dreamplace.ops.routability.gpugr_context import compute_route_grid_like_xplace


def _names(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class GRSizingSession:
    op: GRParasiticsOp
    report: dict
    pin_names: dict[str, int]


def prepare_gr_sizing(params, placedb):
    started = time.perf_counter()
    lefs = list(params.lef_input)
    if not lefs:
        inputs = params.design_inputs
        lefs = [inputs["tech_lef"], *inputs["lef"]]
        params.lef_input = lefs
    result_dir = Path(params.result_dir) / "gr_parasitics"
    router = XplaceGPUGR(params, placedb)
    route_xsize, route_ysize = compute_route_grid_like_xplace(params, placedb)
    routed = router.run_gpugr(
        input_def=params.def_input,
        out_dir=str(result_dir),
        design_name=params.design_name(),
        threads=params.num_threads,
        route_xsize=route_xsize,
        route_ysize=route_ysize,
        rrr_iters=int(getattr(params, "gr_sizing_rrr_iters", 0)),
        backend=params.gpugr_backend,
        keep_temp_def=True,
        include_timing_route_pack=True,
        profile_enabled=True,
    )
    routing_seconds = time.perf_counter() - started
    pack = routed["timing_route_pack"]
    pin_names, net_names = _names(placedb.pin_names), _names(placedb.net_names)
    if len(set(pin_names)) != len(pin_names) or len(set(net_names)) != len(net_names):
        raise ValueError("GR timing requires unique physical pin and net names")
    eligible = torch.ones(len(net_names), dtype=torch.bool)
    excluded = sorted(clock_net_ids(placedb))
    eligible[excluded] = False
    mapping = PinMapping(
        dict(zip(pin_names, range(len(pin_names)), strict=True)),
        dict(zip(net_names, range(len(net_names)), strict=True)),
        torch.as_tensor(placedb.pin2net_map),
        torch.as_tensor(placedb.net2driver_pin_map),
        eligible,
    )
    geometry_hash = hashlib.sha256()
    for values in (
        placedb.node_x,
        placedb.node_y,
        placedb.node_size_x,
        placedb.node_size_y,
        placedb.pin_offset_x,
        placedb.pin_offset_y,
        placedb.inst_libcell_offset,
    ):
        geometry_hash.update(np.ascontiguousarray(values).tobytes())
    geometry_hash.update("\n".join(_names(placedb.node_names)).encode())
    identity = RouteIdentity(_file_hash(params.def_input), geometry_hash.hexdigest(), 1)
    rc = RCParameters.from_lefs(lefs, pack["layer_names"])
    # Native prepare validates routing, estimates local access and records the
    # electrical loop reduction. Missing connectivity/RC still fails.
    try:
        op = GRParasiticsOp.prepare(pack, rc, mapping, identity, dtype=getattr(torch, params.dtype))
    except RuntimeError as exc:
        np.savez(
            result_dir / "unsupported-route.npz",
            **{key: value for key, value in pack.items() if isinstance(value, np.ndarray)},
        )
        (result_dir / "unsupported-route.json").write_text(
            json.dumps(
                {
                    "reason": str(exc),
                    **{
                        key: value
                        for key, value in pack.items()
                        if not isinstance(value, np.ndarray)
                    },
                },
                indent=2,
            )
            + "\n"
        )
        raise
    tree = op.snapshot
    report = {
        "timing_rc_mode": "gr",
        **tree.model_report,
        "native_pin_access_model": pack["pin_access_model"],
        "via_cap_model": rc.via_cap_model,
        "rc_sources": rc.sources,
        "input_sha256": identity.input_sha256,
        "geometry_sha256": identity.geometry_sha256,
        "router_calls": 1,
        "routing_seconds": routing_seconds,
        "prepare_seconds": time.perf_counter() - started,
        "native_route_stats": routed["metrics"].get("native_route_stats", {}),
        "vertices": tree.parent.numel(),
        "edges": tree.edge_parent.numel(),
        "covered_pins": tree.valid_pin_ids.numel(),
        "eligible_nets": tree.net_ids.numel(),
        "filtered_nets": tree.filtered_net_count,
        "wire_cap_pf": float(tree.wire_cap.sum()),
        "parser_cache_hit": bool(routed["metrics"]["parser_cache_hit"]),
        "threads": params.num_threads,
        "backend": routed["metrics"]["gpugr_backend"],
    }
    (result_dir / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    placedb.gr_sizing = GRSizingSession(op, report, mapping.pin_names)


def finalize_gr_sizing(engine, result):
    """Evaluate the final legal native output using a fresh PyDB and GR parse."""
    if result["status"] != "ok":
        return result
    params = engine.params
    session = engine.placedb.gr_sizing
    output_dir = Path(params.result_dir) / "gr_parasitics"
    final_def, final_verilog = output_dir / "final.def", output_dir / "final.v"
    engine.write_back(str(final_def))
    engine.write_verilog(str(final_verilog))
    from dreamplace.flows.flow_config import apply_flow_defaults

    final_params = copy.deepcopy(params)
    final_params.flow_kind = "sta"
    final_params.def_input = str(final_def)
    final_params.verilog_input = str(final_verilog)
    final_params.result_dir = str(output_dir / "terminal_sta")
    final_params.design_inputs["def"] = str(final_def)
    final_params.design_inputs["verilog"] = str(final_verilog)
    # A materialized S50 config can carry an explicit dynamics flag that was
    # changed by warmup. Normalize the read-only lane before generic validation.
    from dreamplace.flows.gr_sizing_config import configure_gr_sizing

    configure_gr_sizing(final_params)
    apply_flow_defaults(final_params)
    terminal = type(engine)(final_params)
    terminal.setup_rawdb(engine.data_manager)
    terminal_result = terminal.run_sta_only()
    terminal_session = terminal.placedb.gr_sizing
    timing = terminal.metrics["timing_stage_summary"]["sta"]
    # Use the fresh native master/arc state for both LUT reports, remapping
    # physical pin IDs by name. Share the frozen graph; do not reroute again.
    order = torch.tensor(
        [session.pin_names[name] for name in _names(terminal.placedb.pin_names)],
        dtype=torch.int64,
    )
    tree = session.op.snapshot
    pin_vertex = tree.pin_to_vertex[order]
    valid = pin_vertex >= 0
    valid_ids = valid.nonzero().flatten()
    frozen_op = GRParasiticsOp(
        replace(
            tree,
            pin_to_vertex=pin_vertex,
            pin_grid_vertex=tree.pin_grid_vertex[order],
            valid_pin_mask=valid,
            valid_pin_ids=valid_ids,
            valid_vertex_ids=pin_vertex[valid_ids],
        )
    )
    model = terminal.placer.last_global_place_model
    model.op_collections.elmore_delay_op = frozen_op
    try:
        with torch.no_grad():
            wns, tns, _, _ = model.timing_obj(terminal.placer.pos[0], surrogate_mode="lut_only")
        timing_op = model.op_collections.timing_propagation_op
        frozen_timing = {
            "wns": float(wns),
            "tns": float(tns),
            "slew_violation": float(timing_op.last_total_slew_violation),
            "cap_violation": float(timing_op.last_total_cap_violation),
            "master_state": "fresh_native_final",
        }
    finally:
        model.op_collections.elmore_delay_op = terminal_session.op
    report = {
        "status": terminal_result["status"],
        "initial_snapshot": session.report,
        "terminal_snapshot": terminal_session.report,
        "router_calls": session.report["router_calls"] + terminal_session.report["router_calls"],
        "optimization_rc_calls": session.op.forward_calls,
        "optimization_rc_seconds": session.op.forward_seconds,
        "terminal_timing": timing,
        "frozen_snapshot_lut_timing": frozen_timing,
        "def": str(final_def),
        "verilog": str(final_verilog),
        "def_sha256": _file_hash(final_def),
        "verilog_sha256": _file_hash(final_verilog),
        "backend_caps": engine.placedb.rawdb.backend_caps.to_dict(),
    }
    report_path = output_dir / "sizing.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {**result, "gr_sizing": report, "gr_sizing_artifact": str(report_path)}
