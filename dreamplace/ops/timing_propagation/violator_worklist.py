import json
import re
from pathlib import Path


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def parse_report_check_types_violators(text):
    records = []
    for line in str(text or "").splitlines():
        if "(VIOLATED)" not in line:
            continue
        parts = line.split()
        if len(parts) < 4 or "/" not in parts[0]:
            continue
        try:
            limit = float(parts[1])
            value = float(parts[2])
            slack = float(parts[3])
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "pin_name": parts[0],
                "limit": limit,
                "value": value,
                "slack": slack,
            }
        )
    return records


def _report_pin_aliases(pin_name):
    pin_name = str(pin_name)
    aliases = [pin_name]
    if "/" in pin_name:
        inst, port = pin_name.rsplit("/", 1)
        aliases.append(f"{inst}:{port}")
    if ":" in pin_name:
        inst, port = pin_name.rsplit(":", 1)
        aliases.append(f"{inst}/{port}")
    return aliases


def map_violator_pins_to_nets(pydb, violator_records, *, check_type):
    pin_names = [str(name) for name in _as_list(getattr(pydb, "pin_names", []))]
    pin_name_to_id = {}
    for pin_id, pin_name in enumerate(pin_names):
        for alias in _report_pin_aliases(pin_name):
            pin_name_to_id.setdefault(alias, int(pin_id))

    pin2net = [int(value) for value in _as_list(getattr(pydb, "pin2net_map", []))]
    net_names = [str(name) for name in _as_list(getattr(pydb, "net_names", []))]
    mapped = []
    for record in violator_records or []:
        pin_id = None
        for alias in _report_pin_aliases(record.get("pin_name", "")):
            if alias in pin_name_to_id:
                pin_id = pin_name_to_id[alias]
                break
        if pin_id is None or pin_id < 0 or pin_id >= len(pin2net):
            continue
        net_id = int(pin2net[pin_id])
        if net_id < 0 or net_id >= len(net_names):
            continue
        item = dict(record)
        item.update(
            {
                "check_type": str(check_type),
                "pin_id": int(pin_id),
                "net_id": int(net_id),
                "net_name": str(net_names[net_id]),
            }
        )
        mapped.append(item)
    return mapped


def summarize_violator_nets(mapped_records, *, top_n=50):
    by_net = {}
    for record in mapped_records or []:
        net_id = int(record.get("net_id", -1))
        if net_id < 0:
            continue
        entry = by_net.setdefault(
            net_id,
            {
                "net_id": net_id,
                "net_name": str(record.get("net_name", "")),
                "violator_pin_count": 0,
                "worst_slack": float(record.get("slack", 0.0)),
                "check_types": set(),
                "violator_pins": [],
            },
        )
        slack = float(record.get("slack", 0.0))
        entry["violator_pin_count"] += 1
        entry["worst_slack"] = min(float(entry["worst_slack"]), slack)
        entry["check_types"].add(str(record.get("check_type", "")))
        entry["violator_pins"].append(str(record.get("pin_name", "")))

    top = sorted(
        by_net.values(),
        key=lambda item: (
            float(item["worst_slack"]),
            -int(item["violator_pin_count"]),
            int(item["net_id"]),
        ),
    )[: int(top_n)]
    for item in top:
        item["check_types"] = sorted(value for value in item["check_types"] if value)
    return {
        "artifact": "buffering_opensta_violator_net_summary",
        "artifact_version": 1,
        "violator_pin_count": len(list(mapped_records or [])),
        "unique_violator_net_count": len(by_net),
        "top_violator_nets": top,
    }


def discover_opensta_violator_nets(raw_db, pydb, *, output_dir, top_n=50):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    commands = {
        "max_slew": "report_check_types -max_slew -violators",
        "max_capacitance": "report_check_types -max_capacitance -violators",
    }
    report_paths = {}
    for check_type, command in commands.items():
        report_path = output_dir / f"{check_type}.rpt"
        raw_db.eval_tcl_string(f"{command} > {{{report_path}}}")
        report_paths[check_type] = str(report_path)
        records = parse_report_check_types_violators(report_path.read_text())
        all_records.extend(
            map_violator_pins_to_nets(pydb, records, check_type=check_type)
        )

    summary = summarize_violator_nets(all_records, top_n=top_n)
    summary["status"] = "ok"
    summary["report_paths"] = report_paths
    summary_path = output_dir / "buffering_opensta_violator_net_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary
