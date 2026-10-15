import json
import re
from pathlib import Path


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_endpoint_name(line):
    match = re.match(r"\s*Endpoint\s*:\s*(\S+)", line)
    if match:
        return match.group(1)
    match = re.match(r"\s*Endpoint\s+Pin\s*:\s*(\S+)", line, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _parse_slack_value(line):
    if not re.search(r"\bslack\b", line, flags=re.IGNORECASE):
        return None
    values = _FLOAT_RE.findall(line)
    if not values:
        return None
    if re.match(r"\s*slack\b", line, flags=re.IGNORECASE):
        return float(values[-1])
    return float(values[0])


def _parse_path_pin(line):
    match = re.search(r"(?:\^|v)\s+(\S+/\S+)\s+\(", line)
    if match:
        return match.group(1)
    return None


def parse_report_checks_setup_endpoints(text, *, max_endpoints=None):
    by_pin = {}
    current_endpoint = None
    current_endpoint_pin = None
    data_arrival_seen = False
    for line in str(text or "").splitlines():
        endpoint = _parse_endpoint_name(line)
        if endpoint:
            current_endpoint = endpoint
            current_endpoint_pin = None
            data_arrival_seen = False
            continue
        if "data arrival time" in line:
            data_arrival_seen = True
            continue
        path_pin = _parse_path_pin(line)
        if path_pin and current_endpoint is not None and not data_arrival_seen:
            current_endpoint_pin = path_pin
        slack = _parse_slack_value(line)
        if slack is None or current_endpoint is None:
            continue
        pin_name = current_endpoint_pin or current_endpoint
        entry = by_pin.setdefault(
            pin_name,
            {"pin_name": pin_name, "slack": float(slack), "npath": 0},
        )
        entry["slack"] = min(float(entry["slack"]), float(slack))
        entry["npath"] = int(entry["npath"]) + 1
        current_endpoint = None
        current_endpoint_pin = None
        data_arrival_seen = False

    records = sorted(
        by_pin.values(),
        key=lambda item: (float(item["slack"]), str(item["pin_name"])),
    )
    if max_endpoints is not None:
        records = records[: int(max_endpoints)]
    return [
        {
            "pin_name": str(item["pin_name"]),
            "slack": float(item["slack"]),
            "npath": int(item["npath"]),
        }
        for item in records
    ]


def write_setup_endpoint_worklist(records, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        {
            "pin_name": str(record["pin_name"]),
            "slack": float(record.get("slack", -1.0)),
            "npath": max(1, int(record.get("npath", 1))),
        }
        for record in records
    ]
    path.write_text(
        json.dumps({"setup_endpoints": normalized}, indent=2, sort_keys=True) + "\n"
    )
    slacks = [float(record["slack"]) for record in normalized]
    return {
        "artifact": "setup_endpoint_worklist",
        "artifact_version": 1,
        "setup_endpoint_count": len(normalized),
        "setup_endpoint_json_path": str(path),
        "worst_slack": min(slacks) if slacks else None,
        "best_slack": max(slacks) if slacks else None,
        "total_npath": sum(int(record["npath"]) for record in normalized),
    }


def collect_opensta_setup_endpoint_worklist(
    raw_db,
    *,
    output_dir,
    group_count=200,
    endpoint_count=1,
    max_endpoints=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "setup_report_checks.rpt"
    command = (
        "report_checks -path_delay max "
        f"-group_count {int(group_count)} "
        f"-endpoint_count {int(endpoint_count)} "
        "-digits 6"
    )
    raw_db.eval_tcl_string(f"{command} > {{{report_path}}}")
    records = parse_report_checks_setup_endpoints(
        report_path.read_text(),
        max_endpoints=max_endpoints,
    )
    setup_endpoint_json_path = output_dir / "setup_endpoints.json"
    summary = write_setup_endpoint_worklist(records, setup_endpoint_json_path)
    summary.update(
        {
            "status": "ok",
            "report_checks_command": command,
            "report_path": str(report_path),
            "group_count": int(group_count),
            "endpoint_count": int(endpoint_count),
            "max_endpoints": None if max_endpoints is None else int(max_endpoints),
        }
    )
    summary_path = output_dir / "setup_endpoint_worklist_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary
