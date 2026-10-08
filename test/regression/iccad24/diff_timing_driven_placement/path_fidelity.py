#!/usr/bin/env python3
"""Export and compare AutoDMP critical paths on a frozen placement state."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
if str(AUTODMP_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTODMP_ROOT))


SCHEMA_VERSION = 1
OPENTIMER_GOLDEN_SCHEMA_VERSION = 3
TRANSITION_NAMES = {0: "rise", 1: "fall"}


def sha256_bytes(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def def_components_sha256(path: Path | str) -> str:
    in_components = False
    component_lines = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if stripped.startswith("COMPONENTS "):
                in_components = True
            if in_components:
                component_lines.append(stripped)
            if in_components and stripped == "END COMPONENTS":
                break
    if not component_lines or component_lines[-1] != "END COMPONENTS":
        raise ValueError("DEF does not contain a complete COMPONENTS section")
    return sha256_bytes("\n".join(component_lines) + "\n")


def repository_provenance(path: Path | str) -> dict[str, Any]:
    root = Path(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "head": head,
        "dirty": bool(diff),
        "dirty_diff_sha256": sha256_bytes(diff),
    }


def _rc_parameters_from_placedb(placedb) -> dict[str, float]:
    resistance = float(placedb.r_unit)
    capacitance_pf = float(placedb.c_unit)
    if not math.isfinite(resistance) or not math.isfinite(capacitance_pf):
        raise ValueError("placement database contains non-finite RC parameters")
    return {
        "wire_capacitance_per_micron": capacitance_pf * 1.0e-12,
        "wire_resistance_per_micron": resistance,
    }


def _rc_parameters_match(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    keys = ("wire_capacitance_per_micron", "wire_resistance_per_micron")
    return all(
        key in lhs
        and key in rhs
        and math.isclose(
            float(lhs[key]),
            float(rhs[key]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-24,
        )
        for key in keys
    )


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _semantic_identifier(value: Any) -> str:
    """Normalize OpenDB's syntactic DEF bus escaping."""
    return _decode_name(value).replace(r"\[", "[").replace(r"\]", "]")


def build_canonical_pin_map(placedb) -> tuple[list[dict[str, Any]], list[str]]:
    """Build structural pin identities without suffix or fuzzy matching."""
    raw_pin_names = list(placedb.pin_names)
    raw_node_names = list(placedb.node_names)
    pin_to_node = list(placedb.pin2node_map)
    if len(raw_pin_names) != len(pin_to_node):
        raise ValueError("pin_names and pin2node_map length mismatch")

    records = []
    canonical_names = []
    canonical_to_id: dict[str, int] = {}
    for pin_id, (raw_pin_value, node_id_value) in enumerate(
        zip(raw_pin_names, pin_to_node)
    ):
        raw_pin_storage = _decode_name(raw_pin_value)
        raw_pin = _semantic_identifier(raw_pin_value)
        node_id = int(node_id_value)
        if node_id < 0 or node_id >= len(raw_node_names):
            raise ValueError(f"pin {pin_id} has out-of-range node id {node_id}")
        node_name_storage = _decode_name(raw_node_names[node_id])
        node_name = _semantic_identifier(raw_node_names[node_id])
        if raw_pin == node_name:
            canonical = f"port:{raw_pin}"
            kind = "port"
            local_pin = None
        else:
            prefix = node_name + ":"
            if not raw_pin.startswith(prefix) or len(raw_pin) == len(prefix):
                raise ValueError(
                    "pin identity is not structurally tied to its node: "
                    f"pin_id={pin_id} raw_pin={raw_pin!r} node={node_name!r}"
                )
            local_pin = raw_pin[len(prefix) :]
            canonical = f"inst:{node_name}/pin:{local_pin}"
            kind = "instance_pin"
        previous = canonical_to_id.setdefault(canonical, pin_id)
        if previous != pin_id:
            raise ValueError(
                f"ambiguous canonical pin identity {canonical!r}: "
                f"pin ids {previous} and {pin_id}"
            )
        canonical_names.append(canonical)
        records.append(
            {
                "autodmp_pin_id": pin_id,
                "canonical_pin": canonical,
                "kind": kind,
                "local_pin": local_pin,
                "node_id": node_id,
                "node_name": node_name,
                "node_name_storage": node_name_storage,
                "raw_pin": raw_pin,
                "raw_pin_storage": raw_pin_storage,
                "timing_graph_member": True,
            }
        )
    semantic_node_names = {
        _semantic_identifier(value) for value in raw_node_names
    }
    raw_clock_names = getattr(placedb, "clk_pin_names", None)
    clock_names = [] if raw_clock_names is None else list(raw_clock_names)
    for clock_pin_id, raw_clock_value in enumerate(clock_names):
        raw_clock_storage = _decode_name(raw_clock_value)
        raw_clock = _semantic_identifier(raw_clock_value)
        if ":" not in raw_clock:
            raise ValueError(f"clock pin has no instance delimiter: {raw_clock!r}")
        instance_name, local_pin = raw_clock.rsplit(":", 1)
        if instance_name not in semantic_node_names or not local_pin:
            raise ValueError(
                "clock pin identity is not structurally tied to an instance: "
                f"clock_pin_id={clock_pin_id} raw_pin={raw_clock!r}"
            )
        canonical = f"inst:{instance_name}/pin:{local_pin}"
        if canonical in canonical_to_id and canonical_to_id[canonical] >= 0:
            # OpenROAD may classify an asynchronous check pin as both a
            # graph pin and a clock/check pin. The graph record is canonical.
            continue
        mapping_id = -1 - clock_pin_id
        previous = canonical_to_id.setdefault(canonical, mapping_id)
        if previous != mapping_id:
            raise ValueError(f"ambiguous canonical clock pin identity {canonical!r}")
        records.append(
            {
                "autodmp_pin_id": None,
                "canonical_pin": canonical,
                "clock_pin_id": clock_pin_id,
                "kind": "clock_pin",
                "local_pin": local_pin,
                "node_id": None,
                "node_name": instance_name,
                "raw_pin": raw_clock,
                "raw_pin_storage": raw_clock_storage,
                "timing_graph_member": False,
            }
        )
    return records, canonical_names


def _json_line(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    content = "".join(_json_line(record) + "\n" for record in records)
    path.write_text(content, encoding="ascii")
    return sha256_bytes(content)


def _path_sort_key(record: dict[str, Any]):
    return (
        float(record["slack_ps"]),
        record["endpoint_pin"],
        record["endpoint_transition"],
        record["startpoint_pin"],
        tuple(
            (point["pin"], point["transition"])
            for point in record["points"]
        ),
    )


def _rank_paths(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(records, key=_path_sort_key)
    for rank, record in enumerate(ranked):
        record["rank"] = rank
    return ranked


def _derive_pairs(
    paths: list[dict[str, Any]],
    pin_to_node: list[int],
    canonical_to_id: dict[str, int],
    *,
    wns_ps: float,
    min_weight: float,
    max_weight: float,
    accumulate_weight: float,
) -> list[dict[str, Any]]:
    if not math.isfinite(wns_ps) or wns_ps >= 0.0:
        raise ValueError("pair derivation requires finite negative WNS")
    state: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        points = path["points"]
        scale = float(path["slack_ps"]) / wns_ps
        for src, dst in zip(points, points[1:]):
            src_pin = src["pin"]
            dst_pin = dst["pin"]
            src_id = canonical_to_id[src_pin]
            dst_id = canonical_to_id[dst_pin]
            if int(pin_to_node[src_id]) == int(pin_to_node[dst_id]):
                continue
            key = (src_pin, dst_pin)
            record = state.get(key)
            if record is None:
                state[key] = {
                    "clamped": False,
                    "dst_pin": dst_pin,
                    "first_path_rank": int(path["rank"]),
                    "occurrence_count": 1,
                    "src_pin": src_pin,
                    "weight": float(min_weight),
                }
                continue
            record["occurrence_count"] += 1
            unclamped = float(record["weight"]) + accumulate_weight * scale
            record["clamped"] = bool(record["clamped"] or unclamped > max_weight)
            record["weight"] = float(min(max_weight, unclamped))
    return [state[key] for key in sorted(state)]


def _new_paths_from_batch(
    batch,
    snapshot: dict[str, Any],
    canonical_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    offsets = batch.path_offsets.tolist()
    pins = batch.path_pins.tolist()
    transitions = batch.path_transitions.tolist()
    endpoint_pins = batch.endpoint_pins.tolist()
    endpoint_test_ids = batch.endpoint_test_ids.tolist()
    endpoint_transitions = batch.endpoint_transitions.tolist()
    endpoint_slacks = batch.endpoint_slacks.tolist()
    valid = batch.path_valid.tolist()
    invalid_reason = batch.invalid_reason.tolist()
    residual = batch.max_residual_ps.tolist()
    rise_aat = snapshot["pin_rise_aat"].tolist()
    fall_aat = snapshot["pin_fall_aat"].tolist()

    paths = []
    invalid = []
    for path_id in range(len(endpoint_pins)):
        begin = int(offsets[path_id])
        end = int(offsets[path_id + 1])
        if not bool(valid[path_id]):
            invalid.append(
                {
                    "endpoint_pin": canonical_names[int(endpoint_pins[path_id])],
                    "endpoint_test_id": int(endpoint_test_ids[path_id]),
                    "endpoint_transition": TRANSITION_NAMES[
                        int(endpoint_transitions[path_id])
                    ],
                    "invalid_reason": int(invalid_reason[path_id]),
                    "max_residual_ps": float(residual[path_id]),
                    "path_id": path_id,
                }
            )
            continue
        points = []
        for pin_id, transition_id in zip(pins[begin:end], transitions[begin:end]):
            pin_id = int(pin_id)
            transition_id = int(transition_id)
            points.append(
                {
                    "point_gba_aat_ps": float(
                        rise_aat[pin_id]
                        if transition_id == 0
                        else fall_aat[pin_id]
                    ),
                    "pin": canonical_names[pin_id],
                    "raw_pin_id": pin_id,
                    "transition": TRANSITION_NAMES[transition_id],
                }
            )
        if not points:
            raise ValueError("native extractor marked an empty path valid")
        paths.append(
            {
                "endpoint_pin": canonical_names[int(endpoint_pins[path_id])],
                "endpoint_test_id": int(endpoint_test_ids[path_id]),
                "endpoint_transition": TRANSITION_NAMES[
                    int(endpoint_transitions[path_id])
                ],
                "max_residual_ps": float(residual[path_id]),
                "points": points,
                "slack_ps": float(endpoint_slacks[path_id]),
                "split": "max",
                "startpoint_pin": points[0]["pin"],
            }
        )
    return _rank_paths(paths), invalid


def _old_paths_from_live_state(
    timing_prop,
    canonical_names: list[str],
    *,
    global_k: int,
) -> list[dict[str, Any]]:
    endpoint_ids = timing_prop.end_points.detach().cpu().to(torch.int64)
    pin_slack = timing_prop.pin_slack_live_snapshot.detach().cpu().to(torch.float64)
    endpoint_records = [
        (float(pin_slack[int(pin_id)]), int(pin_id))
        for pin_id in endpoint_ids.tolist()
        if float(pin_slack[int(pin_id)]) < 0.0
    ]
    endpoint_records.sort(key=lambda item: (item[0], item[1]))
    if global_k > 0:
        endpoint_records = endpoint_records[:global_k]
    selected_ids = [pin_id for _, pin_id in endpoint_records]
    raw_paths = timing_prop.get_critical_paths(selected_ids, K=1)
    records = []
    for (slack_ps, endpoint_pin), raw_path in zip(endpoint_records, raw_paths):
        if not raw_path:
            continue
        points = [
            {
                "pin": canonical_names[int(pin_id)],
                "raw_pin_id": int(pin_id),
                "transition": "unknown",
            }
            for pin_id in raw_path
        ]
        records.append(
            {
                "endpoint_pin": canonical_names[endpoint_pin],
                "endpoint_transition": "unknown",
                "points": points,
                "slack_ps": slack_ps,
                "split": "max",
                "startpoint_pin": points[0]["pin"],
            }
        )
    return _rank_paths(records)


def _all_pysta_endpoint_states(
    snapshot: dict[str, Any],
    canonical_names: list[str],
) -> list[dict[str, Any]]:
    records = []
    endpoint_pins = snapshot["endpoint_pins"].tolist()
    endpoint_test_ids = snapshot["endpoint_test_ids"].tolist()
    lanes = (
        (
            "rise",
            snapshot["endpoint_rise_slack"].tolist(),
            snapshot["pin_rise_aat"].tolist(),
        ),
        (
            "fall",
            snapshot["endpoint_fall_slack"].tolist(),
            snapshot["pin_fall_aat"].tolist(),
        ),
    )
    for transition, slacks, pin_aats in lanes:
        for pin_id, test_id, slack_ps in zip(
            endpoint_pins, endpoint_test_ids, slacks
        ):
            aat_ps = float(pin_aats[int(pin_id)])
            records.append(
                {
                    "aat_ps": aat_ps,
                    "endpoint_pin": canonical_names[int(pin_id)],
                    "endpoint_test_id": int(test_id),
                    "endpoint_transition": transition,
                    "rat_ps": aat_ps + float(slack_ps),
                    "slack_ps": float(slack_ps),
                    "split": "max",
                    "violating": bool(math.isfinite(slack_ps) and slack_ps < 0.0),
                }
            )
    return sorted(
        records,
        key=lambda record: (
            record["endpoint_pin"],
            record["endpoint_transition"],
            record["endpoint_test_id"],
        ),
    )


def _pysta_gba_points_for_golden(
    snapshot: dict[str, Any],
    canonical_to_id: dict[str, int],
    golden_paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    point_keys = sorted(
        {
            (str(point["pin"]), str(point["transition"]))
            for path in golden_paths
            for point in path["points"]
        }
    )
    rise_aat = snapshot["pin_rise_aat"].tolist()
    fall_aat = snapshot["pin_fall_aat"].tolist()
    records = []
    for canonical_pin, transition in point_keys:
        if canonical_pin not in canonical_to_id:
            records.append(
                {
                    "autodmp_pin_id": None,
                    "availability_reason": "not_in_pysta_timing_graph",
                    "available": False,
                    "pin": canonical_pin,
                    "point_gba_aat_ps": None,
                    "transition": transition,
                }
            )
            continue
        pin_id = canonical_to_id[canonical_pin]
        if transition == "rise":
            aat_ps = rise_aat[pin_id]
        elif transition == "fall":
            aat_ps = fall_aat[pin_id]
        else:
            raise ValueError(f"invalid golden point transition: {transition}")
        reachable = math.isfinite(float(aat_ps)) and float(aat_ps) > -1.0e7
        records.append(
            {
                "autodmp_pin_id": pin_id,
                "availability_reason": (
                    None if reachable else "unreachable_transition_sentinel"
                ),
                "available": reachable,
                "pin": canonical_pin,
                "point_gba_aat_ps": float(aat_ps) if reachable else None,
                "transition": transition,
            }
        )
    return records


def _pysta_sequential_launch_arcs_for_golden(
    placedb,
    snapshot: dict[str, Any],
    canonical_to_id: dict[str, int],
    golden_paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_transitions: dict[str, set[str]] = {}
    for path in golden_paths:
        for source, target in zip(path["points"], path["points"][1:]):
            source_inst, source_pin = _pin_identity_parts(str(source["pin"]))
            target_inst, target_pin = _pin_identity_parts(str(target["pin"]))
            if (
                source_inst is not None
                and source_inst == target_inst
                and source_pin in {"CLK", "RESET", "SET"}
                and target_pin in {"Q", "QN"}
            ):
                target_transitions.setdefault(str(target["pin"]), set()).add(
                    str(target["transition"])
                )

    flat_arcs = torch.as_tensor(placedb.flat_inst_arcs_by_level).cpu()
    level_starts = torch.as_tensor(
        placedb.flat_inst_arcs_by_level_start
    ).cpu()
    if level_starts.numel() < 2:
        return []
    level_start = int(level_starts[0].item())
    level_end = int(level_starts[1].item())
    clock_names = [_semantic_identifier(name) for name in placedb.clk_pin_names]
    lib_cell_names = [
        _semantic_identifier(name) for name in placedb.flat_libcell_names
    ]
    rise_slew = [float(value) for value in placedb.clk_pin_rtran]
    fall_slew = [float(value) for value in placedb.clk_pin_ftran]
    rise_aat = snapshot["pin_rise_aat"].tolist()
    fall_aat = snapshot["pin_fall_aat"].tolist()
    target_ids = {
        canonical_to_id[name]: (name, transitions)
        for name, transitions in target_transitions.items()
        if name in canonical_to_id
    }

    records = []
    for relative_index, row in enumerate(flat_arcs[level_start:level_end].tolist()):
        source_id, target_id, lib_cell_id, lib_arc_id = map(int, row[:4])
        if target_id not in target_ids:
            continue
        if source_id < 0 or source_id >= len(clock_names):
            raise ValueError(
                f"level-0 source id {source_id} is outside clock namespace"
            )
        target_name, transitions = target_ids[target_id]
        source_name = clock_names[source_id]
        source_inst, source_pin = _pin_identity_parts(
            "inst:" + source_name.replace(":", "/pin:", 1)
        )
        records.append(
            {
                "absolute_arc_index": level_start + relative_index,
                "lib_arc_id": lib_arc_id,
                "lib_cell_id": lib_cell_id,
                "lib_cell_name": (
                    lib_cell_names[lib_cell_id]
                    if 0 <= lib_cell_id < len(lib_cell_names)
                    else None
                ),
                "source_fall_slew_ps": fall_slew[source_id],
                "source_namespace_id": source_id,
                "source_pin": source_pin,
                "source_pin_name": source_name,
                "source_rise_slew_ps": rise_slew[source_id],
                "target_fall_gba_aat_ps": float(fall_aat[target_id]),
                "target_pin": target_name,
                "target_pin_id": target_id,
                "target_rise_gba_aat_ps": float(rise_aat[target_id]),
                "target_transitions_in_golden": sorted(transitions),
                "timing_sense": int(row[4]),
                "timing_type": int(row[5]),
            }
        )
    return sorted(
        records,
        key=lambda record: (
            record["target_pin"],
            record["source_pin_name"],
            record["lib_arc_id"],
            record["absolute_arc_index"],
        ),
    )


def _write_backend_artifacts(
    output_dir: Path,
    backend: str,
    paths: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    invalid_paths: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prefix = backend.replace("-", "_")
    path_sha = _write_jsonl(output_dir / f"{prefix}_paths.jsonl", paths)
    pair_sha = _write_jsonl(output_dir / f"{prefix}_pairs.jsonl", pairs)
    result = {
        "path_count": len(paths),
        "paths_sha256": path_sha,
        "pair_count": len(pairs),
        "pairs_sha256": pair_sha,
    }
    if invalid_paths is not None:
        result["invalid_path_count"] = len(invalid_paths)
        result["invalid_paths_sha256"] = _write_jsonl(
            output_dir / f"{prefix}_invalid_paths.jsonl", invalid_paths
        )
    return result


def export_autodmp_fidelity(
    engine,
    *,
    output_dir: Path,
    frame_id: str,
    input_def: Path,
    global_k: int,
    min_weight: float,
    max_weight: float,
    accumulate_weight: float,
    expected_manifest: Path | None = None,
) -> dict[str, Any]:
    placer = engine.placer
    model = placer.last_global_place_model
    timing_prop = placer.op_collections.timing_propagation_op
    pos = placer.data_collections.pos[0]
    if model is None or timing_prop is None:
        raise RuntimeError("STA flow did not retain its timing model")

    timing_prop.request_critical_path_snapshot()
    try:
        with torch.no_grad():
            if hasattr(placer, "_refresh_live_timing_topology"):
                placer._refresh_live_timing_topology(pos)
            wns, tns, _, _ = model.timing_obj(pos)
        snapshot = timing_prop._critical_path_snapshot
        if snapshot is None:
            raise RuntimeError("timing forward did not capture a path snapshot")
        old_paths = _old_paths_from_live_state(
            timing_prop,
            build_canonical_pin_map(engine.placedb)[1],
            global_k=global_k,
        )
        new_batch = timing_prop.extract_setup_critical_paths(
            global_k=global_k,
            residual_tolerance_ps=1.0e-2,
        )
    finally:
        timing_prop.clear_critical_path_snapshot_request()

    pin_map, canonical_names = build_canonical_pin_map(engine.placedb)
    canonical_to_id = {
        record["canonical_pin"]: int(record["autodmp_pin_id"])
        for record in pin_map
        if record["autodmp_pin_id"] is not None
    }
    pin_to_node = [int(value) for value in engine.placedb.pin2node_map]
    wns_ps = float(wns.detach().cpu().item() if torch.is_tensor(wns) else wns)
    tns_ps = float(tns.detach().cpu().item() if torch.is_tensor(tns) else tns)
    new_paths, invalid_paths = _new_paths_from_batch(
        new_batch,
        snapshot,
        canonical_names,
    )
    new_pairs = _derive_pairs(
        new_paths,
        pin_to_node,
        canonical_to_id,
        wns_ps=wns_ps,
        min_weight=min_weight,
        max_weight=max_weight,
        accumulate_weight=accumulate_weight,
    )
    old_pairs = _derive_pairs(
        old_paths,
        pin_to_node,
        canonical_to_id,
        wns_ps=wns_ps,
        min_weight=min_weight,
        max_weight=max_weight,
        accumulate_weight=accumulate_weight,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_endpoint_states = _all_pysta_endpoint_states(
        snapshot,
        canonical_names,
    )
    all_endpoint_states_sha = _write_jsonl(
        output_dir / "pysta_endpoint_states.jsonl",
        all_endpoint_states,
    )
    pin_map_sha = _write_jsonl(
        output_dir / "canonical_pin_map.jsonl",
        sorted(pin_map, key=lambda record: record["canonical_pin"]),
    )
    artifacts = {
        "new": _write_backend_artifacts(
            output_dir,
            "new",
            new_paths,
            new_pairs,
            invalid_paths,
        ),
        "old": _write_backend_artifacts(
            output_dir,
            "old",
            old_paths,
            old_pairs,
        ),
        "canonical_pin_map_sha256": pin_map_sha,
        "pysta_endpoint_state_count": len(all_endpoint_states),
        "pysta_endpoint_states_sha256": all_endpoint_states_sha,
    }

    input_def_sha = sha256_file(input_def)
    coordinate_sha = def_components_sha256(input_def)
    params = engine.params
    design_inputs = getattr(params, "design_inputs", {}) or {}
    raw_liberty = design_inputs.get("lib") or getattr(params, "lib_input", ()) or ()
    if isinstance(raw_liberty, (str, Path)):
        raw_liberty = [raw_liberty]
    liberty_sha = [sha256_file(path) for path in raw_liberty]
    effective_sdc_value = design_inputs.get("sdc") or getattr(
        params, "sdc_input", None
    )
    if not effective_sdc_value:
        raise ValueError("effective SDC input is unavailable")
    effective_sdc = Path(effective_sdc_value)
    effective_sdc_sha = sha256_file(effective_sdc)
    expected = None
    if expected_manifest is not None:
        expected = json.loads(expected_manifest.read_text(encoding="utf-8"))
        if expected.get("input_def_sha256") != input_def_sha:
            raise ValueError("OpenTimer and AutoDMP input DEF SHA mismatch")
        if expected.get("coordinate_sha256") != coordinate_sha:
            raise ValueError("OpenTimer and AutoDMP coordinate SHA mismatch")
        if expected.get("liberty_sha256") != liberty_sha:
            raise ValueError("OpenTimer and AutoDMP Liberty input SHA mismatch")
        if expected.get("effective_sdc_sha256") != effective_sdc_sha:
            raise ValueError("OpenTimer and AutoDMP effective SDC SHA mismatch")
        expected_rc = expected.get("rc") or {}
        actual_rc = _rc_parameters_from_placedb(engine.placedb)
        if not _rc_parameters_match(expected_rc, actual_rc):
            raise ValueError("OpenTimer and AutoDMP RC parameter mismatch")
        expected_analysis = expected.get("analysis") or {}
        if expected_analysis != {
            "ideal_clock": True,
            "setup_only": True,
            "split": "max",
        }:
            raise ValueError("OpenTimer golden analysis mode is incompatible")
        expected_metadata = expected.get("opentimer_metadata") or {}
        if int(expected_metadata.get("schema_version", -1)) != (
            OPENTIMER_GOLDEN_SCHEMA_VERSION
        ):
            raise ValueError("unsupported OpenTimer golden schema version")
        golden_paths_path = expected_manifest.parent / "golden_paths.jsonl"
        golden_paths_for_probe = _load_jsonl(golden_paths_path)
        pysta_gba_points = _pysta_gba_points_for_golden(
            snapshot,
            canonical_to_id,
            golden_paths_for_probe,
        )
        artifacts["pysta_gba_point_count"] = len(pysta_gba_points)
        artifacts["pysta_gba_points_sha256"] = _write_jsonl(
            output_dir / "pysta_gba_points.jsonl",
            pysta_gba_points,
        )
        sequential_launch_arcs = _pysta_sequential_launch_arcs_for_golden(
            engine.placedb,
            snapshot,
            canonical_to_id,
            golden_paths_for_probe,
        )
        artifacts["pysta_sequential_launch_arc_count"] = len(
            sequential_launch_arcs
        )
        artifacts["pysta_sequential_launch_arcs_sha256"] = _write_jsonl(
            output_dir / "pysta_sequential_launch_arcs.jsonl",
            sequential_launch_arcs,
        )

    manifest = {
        "analysis": {
            "ideal_clock": True,
            "setup_only": True,
            "split": "max",
        },
        "artifacts": artifacts,
        "autodmp": repository_provenance(AUTODMP_ROOT),
        "coordinate_sha256": coordinate_sha,
        "effective_sdc_sha256": effective_sdc_sha,
        "frame_id": frame_id,
        "global_k": int(global_k),
        "input_def_sha256": input_def_sha,
        "liberty_sha256": liberty_sha,
        "new_extractor": dict(timing_prop.last_critical_path_extraction_stats or {}),
        "pair_profile": {
            "accumulate_weight": float(accumulate_weight),
            "max_weight": float(max_weight),
            "min_weight": float(min_weight),
            "state_policy": "empty_frame_local",
        },
        "pysta": {
            "tns_ps": tns_ps,
            "wns_ps": wns_ps,
        },
        "rc": _rc_parameters_from_placedb(engine.placedb),
        "schema_version": SCHEMA_VERSION,
    }
    if expected is not None:
        manifest["opentimer_golden_manifest_sha256"] = sha256_file(
            expected_manifest
        )
        manifest["sdc_role"] = expected.get("sdc_role")
    manifest_text = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ) + "\n"
    (output_dir / "autodmp_manifest.json").write_text(
        manifest_text,
        encoding="ascii",
    )
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def _path_state(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["split"]),
        str(record["endpoint_pin"]),
        str(record["endpoint_transition"]),
    )


def _path_sequence(record: dict[str, Any]):
    return tuple(
        (str(point["pin"]), str(point["transition"]))
        for point in record["points"]
    )


def summarize_endpoint_gba_alignment(
    golden_paths: list[dict[str, Any]],
    pysta_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare selected-state slack using endpoint GBA, never recovered PBA."""
    required_golden_fields = (
        "endpoint_gba_slack_ps",
        "endpoint_gba_aat_ps",
        "endpoint_gba_rat_ps",
        "endpoint_cppr_credit_ps",
    )
    golden_by_state: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in golden_paths:
        missing = [field for field in required_golden_fields if field not in path]
        if missing:
            raise ValueError(f"OpenTimer path lacks {missing[0]}")
        golden_by_state.setdefault(_path_state(path), []).append(path)
    pysta_by_state: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for state in pysta_states:
        pysta_by_state.setdefault(_path_state(state), []).append(state)

    matched = []
    for state_key, golden_records in golden_by_state.items():
        pysta_records = pysta_by_state.get(state_key, ())
        for golden_record, pysta_record in zip(
            sorted(
                golden_records,
                key=lambda item: float(item["endpoint_gba_slack_ps"]),
            ),
            sorted(pysta_records, key=lambda item: float(item["slack_ps"])),
        ):
            matched.append((golden_record, pysta_record))

    if not matched:
        return {
            "matched_state_count": 0,
            "sign_mismatch_count": 0,
        }

    golden_values = [float(item[0]["endpoint_gba_slack_ps"]) for item in matched]
    pysta_values = [float(item[1]["slack_ps"]) for item in matched]
    deltas = [
        float(pysta["slack_ps"]) - float(golden["endpoint_gba_slack_ps"])
        for golden, pysta in matched
    ]
    aat_deltas = [
        float(pysta["aat_ps"]) - float(golden["endpoint_gba_aat_ps"])
        for golden, pysta in matched
    ]
    rat_deltas = [
        float(pysta["rat_ps"]) - float(golden["endpoint_gba_rat_ps"])
        for golden, pysta in matched
    ]
    cppr_values = [
        float(golden["endpoint_cppr_credit_ps"]) for golden, _ in matched
    ]
    sign_mismatches = [
        (golden, pysta)
        for golden, pysta in matched
        if float(golden["endpoint_gba_slack_ps"])
        < 0.0
        <= float(pysta["slack_ps"])
    ]
    summary = {
        "golden_cppr_credit_median_ps": statistics.median(cppr_values),
        "golden_endpoint_gba_slack_min_ps": min(golden_values),
        "golden_endpoint_gba_slack_median_ps": statistics.median(golden_values),
        "matched_state_count": len(matched),
        "pysta_minus_golden_aat_median_ps": statistics.median(aat_deltas),
        "pysta_minus_golden_median_ps": statistics.median(deltas),
        "pysta_minus_golden_rat_median_ps": statistics.median(rat_deltas),
        "pysta_slack_median_ps": statistics.median(pysta_values),
        "sign_mismatch_count": len(sign_mismatches),
    }
    if sign_mismatches:
        summary.update(
            {
                "sign_mismatch_golden_endpoint_gba_slack_median_ps":
                    statistics.median(
                        float(item[0]["endpoint_gba_slack_ps"])
                        for item in sign_mismatches
                    ),
                "sign_mismatch_pysta_minus_golden_aat_median_ps":
                    statistics.median(
                        float(item[1]["aat_ps"])
                        - float(item[0]["endpoint_gba_aat_ps"])
                        for item in sign_mismatches
                    ),
                "sign_mismatch_pysta_minus_golden_median_ps":
                    statistics.median(
                        float(item[1]["slack_ps"])
                        - float(item[0]["endpoint_gba_slack_ps"])
                        for item in sign_mismatches
                    ),
                "sign_mismatch_pysta_minus_golden_rat_median_ps":
                    statistics.median(
                        float(item[1]["rat_ps"])
                        - float(item[0]["endpoint_gba_rat_ps"])
                        for item in sign_mismatches
                    ),
                "sign_mismatch_pysta_slack_median_ps":
                    statistics.median(
                        float(item[1]["slack_ps"]) for item in sign_mismatches
                    ),
            }
        )
    return summary


def _pin_identity_parts(canonical_pin: str) -> tuple[str | None, str | None]:
    if not canonical_pin.startswith("inst:") or "/pin:" not in canonical_pin:
        return None, None
    instance, local_pin = canonical_pin[5:].rsplit("/pin:", 1)
    return instance, local_pin


def _launch_point_role(
    points: list[dict[str, Any]],
    point_index: int,
) -> str:
    if point_index == 0:
        return "launch_source"
    if point_index == len(points) - 1:
        return "endpoint"
    previous_instance, previous_pin = _pin_identity_parts(
        str(points[point_index - 1]["pin"])
    )
    current_instance, current_pin = _pin_identity_parts(
        str(points[point_index]["pin"])
    )
    if (
        previous_instance is not None
        and previous_instance == current_instance
        and previous_pin in {"CLK", "RESET", "SET"}
        and current_pin in {"Q", "QN"}
    ):
        return "sequential_control_to_output"
    if point_index >= 2:
        prior_role = _launch_point_role(points, point_index - 1)
        if prior_role == "sequential_control_to_output":
            return "first_post_sequential_net_sink"
    return "data_path"


def build_launch_stage_probe(
    golden_paths: list[dict[str, Any]],
    pysta_states: list[dict[str, Any]],
    pysta_gba_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Locate where PySTA and OpenTimer GBA arrival first diverge."""
    golden_by_state: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in golden_paths:
        golden_by_state.setdefault(_path_state(path), []).append(path)
    pysta_by_state: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for state in pysta_states:
        pysta_by_state.setdefault(_path_state(state), []).append(state)

    matched = []
    for state_key, golden_records in golden_by_state.items():
        pysta_records = pysta_by_state.get(state_key, ())
        for golden, pysta in zip(
            sorted(
                golden_records,
                key=lambda item: float(item["endpoint_gba_slack_ps"]),
            ),
            sorted(pysta_records, key=lambda item: float(item["slack_ps"])),
        ):
            matched.append((golden, pysta))

    if not matched:
        return {"representative_count": 0, "representatives": []}

    point_lookup = {
        (str(record["pin"]), str(record["transition"])): record
        for record in pysta_gba_points
    }
    choices = []
    chosen_paths = set()

    def choose(category, candidates, *, middle=False):
        ordered = sorted(
            candidates,
            key=lambda item: float(item[0]["endpoint_gba_slack_ps"]),
        )
        if not ordered:
            return
        if middle:
            center = len(ordered) // 2
            search_order = sorted(
                range(len(ordered)), key=lambda index: abs(index - center)
            )
        else:
            search_order = range(len(ordered))
        selected = next(
            (
                ordered[index]
                for index in search_order
                if id(ordered[index][0]) not in chosen_paths
            ),
            ordered[next(iter(search_order))],
        )
        choices.append((category, selected))
        chosen_paths.add(id(selected[0]))

    common_failing = [item for item in matched if float(item[1]["slack_ps"]) < 0.0]
    choose("worst_common_failing", common_failing)
    sign_mismatches = [
        item
        for item in matched
        if float(item[0]["endpoint_gba_slack_ps"]) < 0.0
        <= float(item[1]["slack_ps"])
    ]
    choose("median_sign_mismatch", sign_mismatches, middle=True)
    normal_clock_launch = [
        item
        for item in matched
        if any(
            _pin_identity_parts(str(first["pin"]))[0]
            == _pin_identity_parts(str(second["pin"]))[0]
            and _pin_identity_parts(str(first["pin"]))[1] == "CLK"
            and _pin_identity_parts(str(second["pin"]))[1] in {"Q", "QN"}
            for first, second in zip(
                item[0]["points"], item[0]["points"][1:]
            )
        )
    ]
    choose("worst_normal_clock_launch", normal_clock_launch)
    async_prefix = [
        item
        for item in matched
        if any(
            _pin_identity_parts(str(point["pin"]))[1] in {"RESET", "SET"}
            for point in item[0]["points"][:-1]
        )
    ]
    choose("worst_async_control_prefix", async_prefix)

    representatives = []
    for category, (golden, pysta) in choices:
        probe_points = []
        for point_index, point in enumerate(golden["points"]):
            key = (str(point["pin"]), str(point["transition"]))
            if key not in point_lookup:
                raise ValueError(f"PySTA GBA probe point is missing: {key}")
            pysta_point = point_lookup[key]
            opentimer_gba = float(point["point_gba_aat_ps"])
            pysta_gba_value = pysta_point.get("point_gba_aat_ps")
            if pysta_gba_value is not None and float(pysta_gba_value) <= -1.0e7:
                pysta_gba_value = None
                pysta_point = {
                    **pysta_point,
                    "availability_reason": "unreachable_transition_sentinel",
                    "available": False,
                }
            pysta_gba = (
                None if pysta_gba_value is None else float(pysta_gba_value)
            )
            probe_points.append(
                {
                    "availability_reason": pysta_point.get("availability_reason"),
                    "available_in_pysta_timing_graph": bool(
                        pysta_point.get("available", False)
                    ),
                    "index": point_index,
                    "opentimer_gba_aat_ps": opentimer_gba,
                    "opentimer_recovered_path_aat_ps": float(
                        point["recovered_path_aat_ps"]
                    ),
                    "pin": key[0],
                    "pysta_gba_aat_ps": pysta_gba,
                    "pysta_minus_opentimer_gba_aat_ps": (
                        None if pysta_gba is None else pysta_gba - opentimer_gba
                    ),
                    "role": _launch_point_role(golden["points"], point_index),
                    "transition": key[1],
                }
            )
        first_nontrivial = next(
            (
                point["index"]
                for point in probe_points
                if point["pysta_minus_opentimer_gba_aat_ps"] is not None
                and abs(point["pysta_minus_opentimer_gba_aat_ps"]) > 1.0
            ),
            None,
        )
        first_large = next(
            (
                point["index"]
                for point in probe_points
                if point["pysta_minus_opentimer_gba_aat_ps"] is not None
                and abs(point["pysta_minus_opentimer_gba_aat_ps"]) > 50.0
            ),
            None,
        )
        representatives.append(
            {
                "category": category,
                "endpoint_pin": golden["endpoint_pin"],
                "endpoint_transition": golden["endpoint_transition"],
                "first_large_delta_index": first_large,
                "first_nontrivial_delta_index": first_nontrivial,
                "opentimer_endpoint_gba_slack_ps": float(
                    golden["endpoint_gba_slack_ps"]
                ),
                "points": probe_points,
                "pysta_endpoint_slack_ps": float(pysta["slack_ps"]),
                "pysta_endpoint_test_id": int(pysta["endpoint_test_id"]),
            }
        )
    return {
        "large_delta_threshold_ps": 50.0,
        "nontrivial_delta_threshold_ps": 1.0,
        "representative_count": len(representatives),
        "representatives": representatives,
        "schema_version": 1,
    }


def _pair_occurrences(records: list[dict[str, Any]]) -> Counter:
    return Counter(
        {
            (str(record["src_pin"]), str(record["dst_pin"])): int(
                record["occurrence_count"]
            )
            for record in records
        }
    )


def _multiset_overlap(lhs: Counter, rhs: Counter) -> tuple[int, int, float]:
    keys = set(lhs) | set(rhs)
    intersection = sum(min(lhs[key], rhs[key]) for key in keys)
    union = sum(max(lhs[key], rhs[key]) for key in keys)
    return intersection, union, 1.0 if union == 0 else intersection / union


def summarize_old_new_endpoint_domain(
    old_paths: list[dict[str, Any]],
    new_paths: list[dict[str, Any]],
    pysta_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose endpoint-domain changes separately from backtrace fidelity."""
    old_pins = {str(path["endpoint_pin"]) for path in old_paths}
    new_pins = {str(path["endpoint_pin"]) for path in new_paths}
    new_states = {
        (str(path["endpoint_pin"]), str(path["endpoint_transition"]))
        for path in new_paths
    }
    pysta_domain = {str(state["endpoint_pin"]) for state in pysta_states}
    pysta_failing_domain = {
        str(state["endpoint_pin"])
        for state in pysta_states
        if bool(state.get("violating", False))
    }
    old_only = sorted(old_pins - new_pins)
    old_by_pin = {
        str(path["endpoint_pin"]): path
        for path in sorted(
            old_paths,
            key=lambda path: float(path["slack_ps"]),
            reverse=True,
        )
    }
    local_pin_counts = Counter()
    for canonical_pin in old_only:
        _, local_pin = _pin_identity_parts(canonical_pin)
        local_pin_counts[local_pin or "<port>"] += 1

    return {
        "common_endpoint_pin_count": len(old_pins & new_pins),
        "new_duplicate_test_count": len(new_paths) - len(new_states),
        "new_endpoint_pin_count": len(new_pins),
        "new_endpoint_transition_count": len(new_states),
        "new_only_endpoint_pin_count": len(new_pins - old_pins),
        "new_path_count": len(new_paths),
        "old_endpoint_pin_count": len(old_pins),
        "old_only_absent_from_pysta_state_domain_count": sum(
            pin not in pysta_domain for pin in old_only
        ),
        "old_only_endpoint_pin_count": len(old_only),
        "old_only_local_pin_counts": dict(sorted(local_pin_counts.items())),
        "old_only_present_but_nonviolating_count": sum(
            pin in pysta_domain and pin not in pysta_failing_domain
            for pin in old_only
        ),
        "old_only_slack_sum_ps": sum(
            float(old_by_pin[pin]["slack_ps"]) for pin in old_only
        ),
        "old_path_count": len(old_paths),
    }


def _suffix_match_count(
    golden_paths: list[dict[str, Any]],
    candidate_paths: list[dict[str, Any]],
) -> int:
    golden_by_state: dict[tuple[str, str, str], list[tuple]] = {}
    for path in golden_paths:
        golden_by_state.setdefault(_path_state(path), []).append(
            _path_sequence(path)
        )
    candidate_by_state: dict[tuple[str, str, str], list[tuple]] = {}
    for path in candidate_paths:
        candidate_by_state.setdefault(_path_state(path), []).append(
            _path_sequence(path)
        )

    matched = 0
    for state, candidate_sequences in candidate_by_state.items():
        golden_sequences = list(golden_by_state.get(state, ()))
        used = set()
        for candidate in candidate_sequences:
            for index, golden in enumerate(golden_sequences):
                if index in used or len(candidate) > len(golden):
                    continue
                if candidate == golden[-len(candidate) :]:
                    used.add(index)
                    matched += 1
                    break
    return matched


def compare_backend(
    golden_paths: list[dict[str, Any]],
    golden_pairs: list[dict[str, Any]],
    candidate_paths: list[dict[str, Any]],
    candidate_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    golden_states = Counter(_path_state(path) for path in golden_paths)
    candidate_states = Counter(_path_state(path) for path in candidate_paths)
    matched_states = sum(
        min(count, candidate_states[state])
        for state, count in golden_states.items()
    )
    exact_golden = Counter(
        (_path_state(path), _path_sequence(path)) for path in golden_paths
    )
    exact_candidate = Counter(
        (_path_state(path), _path_sequence(path)) for path in candidate_paths
    )
    exact_matches = sum(
        min(count, exact_candidate[key])
        for key, count in exact_golden.items()
    )
    pair_intersection, pair_union, pair_overlap = _multiset_overlap(
        _pair_occurrences(golden_pairs),
        _pair_occurrences(candidate_pairs),
    )
    golden_weights = {
        (record["src_pin"], record["dst_pin"]): float(record["weight"])
        for record in golden_pairs
    }
    candidate_weights = {
        (record["src_pin"], record["dst_pin"]): float(record["weight"])
        for record in candidate_pairs
    }
    pair_keys = set(golden_weights) | set(candidate_weights)
    weighted_min = sum(
        min(golden_weights.get(key, 0.0), candidate_weights.get(key, 0.0))
        for key in pair_keys
    )
    weighted_max = sum(
        max(golden_weights.get(key, 0.0), candidate_weights.get(key, 0.0))
        for key in pair_keys
    )
    weight_abs_error = sum(
        abs(golden_weights.get(key, 0.0) - candidate_weights.get(key, 0.0))
        for key in pair_keys
    )
    golden_weight_sum = sum(abs(weight) for weight in golden_weights.values())
    golden_count = sum(golden_states.values())
    candidate_count = sum(candidate_states.values())
    golden_endpoint_pins = Counter(
        str(path["endpoint_pin"]) for path in golden_paths
    )
    candidate_endpoint_pins = Counter(
        str(path["endpoint_pin"]) for path in candidate_paths
    )
    matched_endpoint_pins = sum(
        min(count, candidate_endpoint_pins[pin])
        for pin, count in golden_endpoint_pins.items()
    )
    suffix_matches = _suffix_match_count(golden_paths, candidate_paths)
    return {
        "candidate_path_count": len(candidate_paths),
        "candidate_duplicate_state_count": candidate_count - len(candidate_states),
        "candidate_unique_state_count": len(candidate_states),
        "candidate_unique_pair_count": len(candidate_pairs),
        "endpoint_pin_match_count": matched_endpoint_pins,
        "endpoint_pin_recall": (
            1.0 if not golden_paths else matched_endpoint_pins / len(golden_paths)
        ),
        "exact_path_denominator": len(golden_paths),
        "exact_path_match_count": exact_matches,
        "exact_path_match_rate": (
            1.0 if not golden_paths else exact_matches / len(golden_paths)
        ),
        "golden_path_count": len(golden_paths),
        "golden_duplicate_state_count": golden_count - len(golden_states),
        "golden_unique_state_count": len(golden_states),
        "golden_unique_pair_count": len(golden_pairs),
        "normalized_pair_weight_error": (
            weight_abs_error / max(golden_weight_sum, 1.0e-12)
        ),
        "ordered_pair_intersection": pair_intersection,
        "ordered_pair_overlap": pair_overlap,
        "ordered_pair_union": pair_union,
        "selected_state_denominator": golden_count,
        "selected_state_match_count": matched_states,
        "selected_state_precision": (
            1.0 if candidate_count == 0 else matched_states / candidate_count
        ),
        "selected_state_recall": (
            1.0 if golden_count == 0 else matched_states / golden_count
        ),
        "suffix_path_match_count": suffix_matches,
        "suffix_path_match_rate": (
            1.0 if not golden_paths else suffix_matches / len(golden_paths)
        ),
        "unknown_transition_path_count": sum(
            path["endpoint_transition"] == "unknown"
            for path in candidate_paths
        ),
        "weighted_pair_jaccard": (
            1.0 if weighted_max == 0.0 else weighted_min / weighted_max
        ),
    }


def compare_frame(
    autodmp_dir: Path,
    golden_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    autodmp_manifest_path = autodmp_dir / "autodmp_manifest.json"
    golden_manifest_path = golden_dir / "golden_manifest.json"
    autodmp_manifest = json.loads(autodmp_manifest_path.read_text(encoding="ascii"))
    golden_manifest = json.loads(golden_manifest_path.read_text(encoding="ascii"))
    for field in ("frame_id", "input_def_sha256", "coordinate_sha256"):
        if autodmp_manifest.get(field) != golden_manifest.get(field):
            raise ValueError(f"manifest field mismatch: {field}")
    if autodmp_manifest.get("analysis") != golden_manifest.get("analysis"):
        raise ValueError("analysis mode mismatch")
    if not _rc_parameters_match(
        autodmp_manifest.get("rc") or {}, golden_manifest.get("rc") or {}
    ):
        raise ValueError("RC parameter mismatch")
    if autodmp_manifest.get("effective_sdc_sha256") != golden_manifest.get(
        "effective_sdc_sha256"
    ):
        raise ValueError("effective fidelity SDC mismatch")
    if autodmp_manifest.get("liberty_sha256") != golden_manifest.get(
        "liberty_sha256"
    ):
        raise ValueError("Liberty input mismatch")

    golden_metadata = golden_manifest.get("opentimer_metadata") or {}
    if int(golden_metadata.get("schema_version", -1)) != (
        OPENTIMER_GOLDEN_SCHEMA_VERSION
    ):
        raise ValueError("unsupported OpenTimer golden schema version")
    if golden_metadata.get("selection_slack_field") != "endpoint_gba_slack_ps":
        raise ValueError("OpenTimer selected-state slack is not endpoint GBA")
    if golden_metadata.get("pair_weight_slack_field") != (
        "recovered_path_slack_ps"
    ):
        raise ValueError("OpenTimer pair weight does not use recovered path slack")

    golden_paths = _load_jsonl(golden_dir / "golden_paths.jsonl")
    golden_pairs = _load_jsonl(golden_dir / "golden_pairs.jsonl")
    golden_used_pins = {
        point["pin"] for path in golden_paths for point in path["points"]
    }
    autodmp_pin_records = _load_jsonl(autodmp_dir / "canonical_pin_map.jsonl")
    canonical_counts = Counter(
        record["canonical_pin"] for record in autodmp_pin_records
    )
    missing = sorted(pin for pin in golden_used_pins if canonical_counts[pin] == 0)
    ambiguous = sorted(pin for pin in golden_used_pins if canonical_counts[pin] > 1)
    if missing or ambiguous:
        raise ValueError(
            "canonical pin mapping failed: "
            f"missing={len(missing)} ambiguous={len(ambiguous)}"
        )

    pysta_all_states = _load_jsonl(autodmp_dir / "pysta_endpoint_states.jsonl")
    new_paths = _load_jsonl(autodmp_dir / "new_paths.jsonl")
    old_paths = _load_jsonl(autodmp_dir / "old_paths.jsonl")
    new_pairs = _load_jsonl(autodmp_dir / "new_pairs.jsonl")
    old_pairs = _load_jsonl(autodmp_dir / "old_pairs.jsonl")
    golden_state_counter = Counter(_path_state(path) for path in golden_paths)
    pysta_all_state_counter = Counter(
        (
            str(record["split"]),
            str(record["endpoint_pin"]),
            str(record["endpoint_transition"]),
        )
        for record in pysta_all_states
    )
    pysta_failing_state_counter = Counter(
        (
            str(record["split"]),
            str(record["endpoint_pin"]),
            str(record["endpoint_transition"]),
        )
        for record in pysta_all_states
        if record["violating"]
    )
    all_state_match_count = sum(
        min(count, pysta_all_state_counter[state])
        for state, count in golden_state_counter.items()
    )
    failing_state_match_count = sum(
        min(count, pysta_failing_state_counter[state])
        for state, count in golden_state_counter.items()
    )
    pysta_gba_points = _load_jsonl(autodmp_dir / "pysta_gba_points.jsonl")
    launch_stage_probe = build_launch_stage_probe(
        golden_paths,
        pysta_all_states,
        pysta_gba_points,
    )
    launch_stage_probe_path = output_path.with_name(
        output_path.stem + "_launch_stage_probe.json"
    )
    launch_stage_probe_path.parent.mkdir(parents=True, exist_ok=True)
    launch_stage_probe_path.write_text(
        json.dumps(launch_stage_probe, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    comparison = {
        "autodmp_manifest_sha256": sha256_file(autodmp_manifest_path),
        "canonical_pin_mapping": {
            "ambiguous_count": len(ambiguous),
            "golden_used_pin_count": len(golden_used_pins),
            "missing_count": len(missing),
        },
        "frame_id": autodmp_manifest["frame_id"],
        "golden_manifest_sha256": sha256_file(golden_manifest_path),
        "old_new_endpoint_domain": summarize_old_new_endpoint_domain(
            old_paths,
            new_paths,
            pysta_all_states,
        ),
        "launch_stage_probe": {
            "artifact": launch_stage_probe_path.name,
            "categories": [
                record["category"]
                for record in launch_stage_probe["representatives"]
            ],
            "representative_count": launch_stage_probe["representative_count"],
            "sha256": sha256_file(launch_stage_probe_path),
        },
        "pysta_state_domain": {
            "all_state_count": sum(pysta_all_state_counter.values()),
            "failing_state_count": sum(pysta_failing_state_counter.values()),
            "golden_state_denominator": sum(golden_state_counter.values()),
            "golden_state_present_count": all_state_match_count,
            "golden_state_present_rate": (
                all_state_match_count / max(1, sum(golden_state_counter.values()))
            ),
            "golden_state_sign_match_count": failing_state_match_count,
            "golden_state_sign_mismatch_count": (
                all_state_match_count - failing_state_match_count
            ),
            "endpoint_gba_slack_alignment": summarize_endpoint_gba_alignment(
                golden_paths,
                pysta_all_states,
            ),
        },
        "new": compare_backend(
            golden_paths,
            golden_pairs,
            new_paths,
            new_pairs,
        ),
        "old": compare_backend(
            golden_paths,
            golden_pairs,
            old_paths,
            old_pairs,
        ),
        "schema_version": SCHEMA_VERSION,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return comparison


def _run_export(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--golden-manifest", type=Path)
    parser.add_argument("--global-k", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=10.0)
    parser.add_argument("--max-weight", type=float, default=50.0)
    parser.add_argument("--accumulate-weight", type=float, default=0.2)
    args, placer_argv = parser.parse_known_args(argv)
    if not placer_argv:
        parser.error("missing Placer.py arguments")

    from dreamplace import Placer
    from dreamplace.flows.optimization_flow import run_optimization_flow

    placer_parser = Placer.build_arg_parser()
    placer_args = placer_parser.parse_args(placer_argv)
    params, launch = Placer.build_effective_params_from_args(placer_args)
    if str(getattr(params, "flow_kind", "")) != "sta":
        raise ValueError("path fidelity export requires --flow-kind sta")
    if not bool(getattr(params, "with_sta", False)):
        raise ValueError("path fidelity export requires --with-sta")
    result = run_optimization_flow(params, launch, engine_cls=Placer.PlacementEngine)
    export_autodmp_fidelity(
        result.engine,
        output_dir=args.output_dir,
        frame_id=args.frame_id,
        input_def=Path(params.def_input),
        global_k=max(0, args.global_k),
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        accumulate_weight=args.accumulate_weight,
        expected_manifest=args.golden_manifest,
    )
    return 0


def _run_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autodmp-dir", required=True, type=Path)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    compare_frame(args.autodmp_dir, args.golden_dir, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("export", "compare"):
        raise SystemExit("usage: path_fidelity.py {export|compare} ...")
    command = argv.pop(0)
    return _run_export(argv) if command == "export" else _run_compare(argv)


if __name__ == "__main__":
    raise SystemExit(main())
