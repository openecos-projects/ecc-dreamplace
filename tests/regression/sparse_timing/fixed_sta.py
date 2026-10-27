"""Run native sparse propagation on a preflight export with frozen loads.

This diagnostic uses zero wire delay/impulse, matching the no-SPEF preflight.
The exported native path cache is diagnostic, not the golden reference.
Use compare_opensta.py for the OpenSTA endpoint qualification gate.
"""

import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dreamplace.ops.placeio_ecc.sparse_timing_schema import read_sparse_timing_view
from dreamplace.ops.timing_propagation.lut_entry_indexed import lut_entry_indexed
from dreamplace.ops.timing_propagation.sparse_timing import SparseTimingSchedule


def production_timing(view, base, device, *, backend=None):
    """Use the real production entry on the same frozen native export.

    Legacy scalar seed/RAT fields are intentionally empty: schema v3 supplies
    those via the validated sparse view. RC remains frozen and zero-wire here.
    """
    from dreamplace.flows.timing_factory import build_timing_propagation_op
    from dreamplace.ops.placeio_ecc.timing_generation import timing_generation_lease
    from dreamplace.ops.timing_propagation.timing_propagation import (
        ARCS_INFO,
        LUTS_INFO,
    )

    def ids(values):
        return torch.as_tensor(values, device=device, dtype=torch.int64)

    infos = {}
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        info = lut_info(base, prefix, device)
        infos[f"{prefix}_luts"] = LUTS_INFO(
            info.flat_luts_values,
            info.flat_luts_trans_table,
            info.flat_luts_cap_table,
            torch.stack((info.trans_dims_actual, info.cap_dims_actual), dim=1),
        )
    pin_net = ids(base["pin2net_map"])
    net_arcs = np.asarray(base["net_flat_arcs"])
    net_ids = base["pin2net_map"][net_arcs[:, 1]]
    order = np.argsort(net_ids, kind="stable")
    offsets = np.r_[
        0, np.cumsum(np.bincount(net_ids, minlength=len(base["net2driver_pin_map"])))
    ]
    arrays = view.arrays
    seeds = arrays["state_pin"][arrays["state_seed"].any(axis=1)]
    empty = torch.empty(0, device=device, dtype=torch.float64)
    arguments = dict(
        inrdelays=empty,
        infdelays=empty,
        inrtrans=empty,
        inftrans=empty,
        outcaps=empty,
        pin_net=pin_net,
        start_points=ids(np.unique(seeds)),
        end_points=ids(base["end_points"]),
        clock_pins=ids(base["clock_pins"]),
        FF_ids=ids([]),
        clk_pin_rtran=empty,
        clk_pin_ftran=empty,
        net_flat_arcs_start=ids(offsets),
        net_flat_arcs=ids(net_arcs[order]),
        net2driver_pin_map=ids(base["net2driver_pin_map"]),
        arcs_info=ARCS_INFO(**infos),
        inst_flat_arcs_start=ids([0]),
        inst_flat_arcs=ids(base["flat_inst_arcs_by_level"]),
        endpoints_constraint_arcs=ids([]).reshape(0, 8),
        endpoints_timing_check_arcs=ids(base["endpoints_timing_check_arcs"]).reshape(-1, 8),
        flat_inst_arcs_by_level=ids(base["flat_inst_arcs_by_level"]),
        flat_inst_arcs_by_level_start=ids(base["flat_inst_arcs_by_level_start"]),
        endpoints_rRAT=empty,
        endpoints_fRAT=empty,
        flat_pin_to_graph=ids([]),
        flat_pin_to_graph_start=ids([0] * (len(pin_net) + 1)),
        flat_pin_to_graph_reverse=ids([]),
        flat_pin_to_graph_start_reverse=ids([0] * (len(pin_net) + 1)),
        timing_propagation_device=device,
        timing_aggregation_mode="hard",
        sparse_timing_view=view,
        production_fast_loop=True,
    )
    params = SimpleNamespace(
        timing_propagation_device=arguments.pop("timing_propagation_device"),
        timing_aggregation_mode=arguments.pop("timing_aggregation_mode"),
        production_fast_loop=arguments.pop("production_fast_loop"),
    )
    placedb = SimpleNamespace(
        endpoints_rRAT=arguments.pop("endpoints_rRAT"),
        endpoints_fRAT=arguments.pop("endpoints_fRAT"),
        sparse_timing_view=arguments.pop("sparse_timing_view"),
        hold_timing_data=None,
        hold_timing_corner=None,
        rawdb=backend,
        pydb=backend.pydb if backend is not None else None,
    )
    arguments["pin2net_map"] = arguments.pop("pin_net")
    arguments["timing_model_lease"] = timing_generation_lease(placedb)
    for name in (
        "pin_pair_arc_keys",
        "flat_pin_pair_arc_start",
        "flat_pin_pair_arc_indices",
        "pin2node_map",
        "inst_main_id",
    ):
        arguments[name] = None
    timing = build_timing_propagation_op(params, placedb, SimpleNamespace(**arguments), None)
    if timing.sparse_timing_view is not view:
        raise RuntimeError("production timing factory lost the native sparse model")
    return timing


def lut_info(base, prefix, device):
    def tensor(suffix, dtype):
        return torch.as_tensor(
            base[f"{prefix}_flat_luts_{suffix}"], dtype=dtype, device=device
        ).contiguous()

    dims = tensor("dim", torch.int64)
    return SimpleNamespace(
        flat_luts_values=tensor("values", torch.float64),
        flat_luts_trans_table=tensor("trans_table", torch.float64),
        flat_luts_cap_table=tensor("cap_table", torch.float64),
        trans_dims_actual=dims[:, 0].contiguous(),
        cap_dims_actual=dims[:, 1].contiguous(),
    )


def metrics(slack):
    return {
        "wns_ns": float(min(slack.min(initial=0), 0) / 1000),
        "tns_ns": float(np.minimum(slack, 0).sum() / 1000),
        "endpoint_count": len(slack),
        "violating_endpoint_count": int(np.count_nonzero(slack < 0)),
    }


def run(args):
    torch.set_num_threads(args.threads)
    arrays = dict(np.load(args.input / "sparse_arrays.npz"))
    metadata = json.loads((args.input / "sparse_metadata.json").read_text())
    base = dict(np.load(args.input / "base_arrays.npz"))
    pydb = SimpleNamespace(**base, timing_sparse={**metadata, "arrays": arrays})
    view = read_sparse_timing_view(pydb)
    view.require_supported()
    started = time.perf_counter()
    model = SparseTimingSchedule(
        view, base["flat_inst_arcs_by_level"], device=args.device, dtype=torch.float64
    )
    tables = []
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans"):
        info = lut_info(base, prefix, args.device)
        tables.extend(
            (
                info.flat_luts_values,
                info.flat_luts_trans_table,
                info.flat_luts_cap_table,
                info.trans_dims_actual,
                info.cap_dims_actual,
            )
        )
    caps = torch.as_tensor(
        arrays["pin_reference_cap"], dtype=torch.float64, device=args.device
    )
    zeros = torch.zeros(len(base["pin_names"]), dtype=torch.float64, device=args.device)
    check_infos = tuple(
        lut_info(base, prefix, args.device) for prefix in ("r_check", "f_check")
    )
    endpoint_arcs = torch.as_tensor(
        base["endpoints_timing_check_arcs"], dtype=torch.int64, device=args.device
    ).reshape(-1, 8)
    dynamic = model.legal & (model.check_index[:, 2] >= 0)
    indices = model.check_index[dynamic]
    lib_arc = endpoint_arcs[indices[:, 2], 3]
    clock_slew = model.check_values[dynamic, 2]
    production = (
        production_timing(view, base, args.device)
        if args.entry == "production"
        else None
    )
    if production is not None:
        production.production_fast_loop = not args.full_rat
    setup_seconds = time.perf_counter() - started

    def forward():
        if production is not None:
            production(
                {"rise": zeros, "fall": zeros},
                {"rise": zeros, "fall": zeros},
                {"rise": caps[:, 0], "fall": caps[:, 1]},
                surrogate_mode="lut_only",
            )
            return production._sparse_runtime.last_result
        lanes = model.propagate(
            tables, (caps[:, 0], caps[:, 1]), (zeros, zeros), (zeros, zeros)
        )
        checks = model.check_values[:, 0].new_zeros(model.check_values.size(0))
        slew = torch.stack(lanes[2:], dim=1)[indices[:, 0], indices[:, 3]]
        queries = tuple(
            lut_entry_indexed(clock_slew, slew, lib_arc, info) for info in check_infos
        )
        checks[dynamic] = torch.where(indices[:, 3] == 0, queries[0], queries[1])
        return model.reduce_endpoints(lanes, checks)

    samples = []
    with torch.no_grad():
        result = forward()
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = forward()
            if args.device == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    check_index = arrays["check_index"]
    legal = check_index[:, 6] == 1
    source = check_index[legal, 0]
    transition = check_index[legal, 3]
    reference_check_slack = (
        arrays["check_values"][legal, 3] - arrays["state_reference"][source, transition]
    )
    reference = np.full(len(result.endpoint_pins), np.inf)
    np.minimum.at(
        reference, model.check_to_endpoint.cpu().numpy(), reference_check_slack
    )
    actual = result.endpoint_slack.cpu().numpy()
    state_arrivals = torch.stack(result.lanes[:2], dim=1).cpu().numpy()
    state_error = abs(state_arrivals - arrays["state_reference"][:, :2])
    required = result.required.cpu().numpy()
    tolerance = np.maximum(
        2.0, 0.03 * np.maximum(1.0, abs(arrays["check_values"][:, 3]))
    )
    failures = legal & (abs(required - arrays["check_values"][:, 3]) > tolerance)
    summary = {
        "qualification": "diagnostic_frozen_sta",
        "entry": args.entry,
        "input": str(args.input.resolve()),
        "device": args.device,
        "threads": args.threads,
        "setup_seconds": setup_seconds,
        "forward_seconds": samples,
        "forward_median_seconds": float(np.median(samples)),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else 0,
        "native_reference": metrics(reference),
        "sparse_forward": metrics(actual),
        "state_aat_max_error_ps": float(state_error.max(initial=0)),
        "required_rule_failures": int(failures.sum()),
        "required_max_error_ps": float(
            abs(required[legal] - arrays["check_values"][legal, 3]).max(initial=0)
        ),
        "endpoint_slack_max_error_ps": float(abs(actual - reference).max(initial=0)),
        "semantic_digest": view.semantic_digest,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if production is not None and args.paths:
        from dreamplace.ops.timing_propagation.timing_path_report import (
            build_critical_path_rows,
            write_critical_path_report,
        )

        production.request_critical_path_snapshot()
        with torch.no_grad():
            forward()
        paths = production.extract_pin2pin_critical_paths(global_k=args.paths)
        rows = build_critical_path_rows(
            batch=paths,
            timing_op=production,
            pin_names=base["pin_names"],
            flat_inst_arcs_by_level=production.flat_inst_arcs_by_level,
        )
        write_critical_path_report(
            args.output / "paths.csv",
            args.output / "paths.json",
            batch=paths,
            rows=rows,
        )
        summary["paths"] = production.last_critical_path_extraction_stats
    if production is not None:
        (args.output / "production_profile.json").write_text(
            json.dumps(production.last_profile_payload, indent=2)
        )
    np.savez_compressed(
        args.output / "timing.npz",
        state_aat=state_arrivals,
        state_slew=torch.stack(result.lanes[2:], dim=1).cpu().numpy(),
        required=required,
        check_slack=result.check_slack.cpu().numpy(),
        endpoint_pins=result.endpoint_pins.cpu().numpy(),
        endpoint_slack=actual,
        reference_endpoint_slack=reference,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--entry", choices=("schedule", "production"), default="schedule"
    )
    parser.add_argument("--full-rat", action="store_true")
    parser.add_argument("--paths", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
