"""Enforce the frozen wall/RSS budget of one native structure preflight.

The child owns a new process group. RSS samples include that group's children;
budget termination never targets a pre-existing ECC or another user's process.
State/edge limits are checked by structure_scale.py after model materialization.
"""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def group_rss_bytes(group):
    rows = subprocess.check_output(["ps", "-eo", "pgid=,rss="], text=True)
    return sum(
        int(rss) * 1024
        for pgid, rss in (row.split() for row in rows.splitlines())
        if int(pgid) == group
    )


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    # A child may survive even when its parent has already exited.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run(command, *, limits_path, log_path, report_path):
    frozen = limits_path.read_bytes()
    limits = json.loads(frozen)
    if any(
        limits[name] <= 0 for name in ("max_wall_s", "max_rss_bytes", "max_states", "max_edges")
    ):
        raise ValueError("resource limits must be frozen positive numbers")
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.monotonic()
    interval = min(0.5, limits["max_wall_s"] / 10)
    peak_rss = 0
    violation = None
    with log_path.open("x") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, group_rss_bytes(process.pid))
                if time.monotonic() - started > limits["max_wall_s"]:
                    violation = "max_wall_s"
                elif peak_rss > limits["max_rss_bytes"]:
                    violation = "max_rss_bytes"
                if violation:
                    terminate_group(process)
                    break
                time.sleep(interval)
        except BaseException:
            terminate_group(process)
            raise
        returncode = process.wait()
    elapsed = time.monotonic() - started
    if violation is None and elapsed > limits["max_wall_s"]:
        violation = "max_wall_s"
    report = {
        "status": "budget_exceeded" if violation else "completed" if returncode == 0 else "failed",
        "command": command,
        "process_group": process.pid,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "sampled_group_peak_rss_bytes": peak_rss,
        "rss_sample_interval_seconds": interval,
        "limits": limits,
        "limits_sha256": hashlib.sha256(frozen).hexdigest(),
        "limit_exceeded": violation,
        "log": str(log_path),
    }
    if limits_path.read_bytes() != frozen:
        report["status"] = "failed"
        report["error"] = "resource limits changed during execution"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("limits", "log", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    report = run(command, limits_path=args.limits, log_path=args.log, report_path=args.report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
