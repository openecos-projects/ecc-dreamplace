import argparse
import contextlib
import ctypes
import json
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_BASE_TECH_LEF = Path(
    "/nfs/share/home/qiming/icsprout55-pdk/prtech/techLEF/N551P6M.lef"
)
DEFAULT_CAP_SOURCE_LEF = Path("/nfs/share/home/qiming/0924/N551P6M_cmax.lef")
DEFAULT_MERGED_TECH_LEF = (
    THIS_DIR / "artifacts" / "tech_lef" / "N551P6M_with_cmax_cap.lef"
)
DEFAULT_ARTIFACT_BASE = THIS_DIR / "artifacts" / "margin_search"


def top_layer_name(stripped):
    parts = stripped.split()
    if len(parts) == 2 and parts[0] == "LAYER":
        return parts[1]
    return None


def extract_cap_lines(cap_source_lef):
    cap_by_layer = {}
    current = None
    with open(cap_source_lef, "r") as stream:
        for line in stream:
            stripped = line.strip()
            maybe_layer = top_layer_name(stripped)
            if current is None and maybe_layer:
                current = maybe_layer
                continue
            if current and (
                stripped.startswith("CAPACITANCE ")
                or stripped.startswith("EDGECAPACITANCE ")
            ):
                cap_by_layer.setdefault(current, []).append(line)
                continue
            if current and stripped == "END %s" % current:
                current = None
    return cap_by_layer


def write_merged_tech_lef(base_tech_lef, cap_source_lef, output_lef):
    cap_by_layer = extract_cap_lines(cap_source_lef)
    output_lef.parent.mkdir(parents=True, exist_ok=True)
    inserted = []
    current = None
    layer_has_cap = False
    with open(base_tech_lef, "r") as source, open(output_lef, "w") as output:
        for line in source:
            stripped = line.strip()
            maybe_layer = top_layer_name(stripped)
            if current is None and maybe_layer:
                current = maybe_layer
                layer_has_cap = False
                output.write(line)
                continue
            if current and (
                stripped.startswith("CAPACITANCE ")
                or stripped.startswith("EDGECAPACITANCE ")
            ):
                layer_has_cap = True
            if (
                current
                and stripped.startswith("RESISTANCE ")
                and not layer_has_cap
                and current in cap_by_layer
            ):
                output.writelines(cap_by_layer[current])
                inserted.append(current)
                layer_has_cap = True
            output.write(line)
            if current and stripped == "END %s" % current:
                current = None
                layer_has_cap = False
    return inserted


def load_matrix_runner():
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import cx55_ablation_iter100_matrix as matrix

    return matrix


def install_margin_command_patch(matrix, margin_state):
    original_prepare = matrix.prepare_runtime_modules

    def patched_prepare_runtime_modules(mpl_config_dir):
        result = original_prepare(mpl_config_dir)
        import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

        margin = margin_state["margin"]
        command = (
            "repair_timing -setup "
            "-setup_margin %.12g "
            '-sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap"
        ) % margin
        for alias in ("buffer-only", "buffer_only"):
            placeio_openroad._BUFFER_INSERTION_STRATEGY_ALIASES[alias][
                "command"
            ] = command
        return result

    matrix.prepare_runtime_modules = patched_prepare_runtime_modules


def success_from_summary(summary, success_mode):
    session = summary.get("session", {}) or {}
    churn = session.get("buffer_churn_totals", {}) or {}
    mutations = session.get("mutation_kind_counts", {}) or {}
    if success_mode == "topology":
        return int(mutations.get("topology_changed", 0) or 0) > 0
    return int(churn.get("added_buffer_count", 0) or 0) > 0


def format_attempt(step, margin, summary, success, bracket_low, bracket_high):
    session = summary.get("session", {}) or {}
    churn = session.get("buffer_churn_totals", {}) or {}
    mutations = session.get("mutation_kind_counts", {}) or {}
    diagnostics = summary.get("openroad_diagnostic_evidence", {}) or {}
    labels = ",".join(diagnostics.get("diagnostic_labels", []) or [])
    return (
        "step=%d margin=%.12g success=%s failure=%s "
        "added=%s removed=%s topology=%s geometry=%s no_setup=%s "
        "bracket=[%.12g,%.12g] run_dir=%s"
        % (
            step,
            margin,
            int(success),
            summary.get("failure"),
            churn.get("added_buffer_count"),
            churn.get("removed_buffer_count"),
            mutations.get("topology_changed"),
            mutations.get("geometry_changed"),
            int("insufficient_timing_evidence" in labels),
            bracket_low,
            bracket_high,
            summary.get("run_dir"),
        )
    )


@contextlib.contextmanager
def redirected_output_to(sink):
    sink.flush()
    libc = ctypes.CDLL(None)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        libc.fflush(None)
        sink.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def run_attempt(matrix, margin_state, args, margin, suppressed_log):
    margin_state["margin"] = margin
    with open(suppressed_log, "a") as sink:
        sink.write("\n=== margin %.12g ===\n" % margin)
        sink.flush()
        with redirected_output_to(sink):
            return matrix.run_one(
                args.design,
                "buffer-only",
                artifact_base=str(args.artifact_base),
                target_iter=args.target_iter,
                trigger_period=args.trigger_period,
                plot_interval=args.plot_interval,
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="APU", choices=("APU", "PPU", "BM64"))
    parser.add_argument("--target-iter", type=int, default=100)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--high", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trigger-period", type=int)
    parser.add_argument("--plot-interval", type=int)
    parser.add_argument(
        "--success-mode",
        choices=("added-buffer", "topology"),
        default="added-buffer",
    )
    parser.add_argument("--artifact-base", type=Path, default=DEFAULT_ARTIFACT_BASE)
    parser.add_argument("--base-tech-lef", type=Path, default=DEFAULT_BASE_TECH_LEF)
    parser.add_argument("--cap-source-lef", type=Path, default=DEFAULT_CAP_SOURCE_LEF)
    parser.add_argument("--merged-tech-lef", type=Path, default=DEFAULT_MERGED_TECH_LEF)
    parser.add_argument("--skip-merged-lef-generation", action="store_true")
    parser.add_argument("--no-probe-high-first", action="store_true")
    args = parser.parse_args(argv)

    if args.high <= args.low:
        raise RuntimeError("--high must be greater than --low")
    if args.trigger_period is not None and args.trigger_period <= 0:
        raise RuntimeError("--trigger-period must be positive")
    if args.plot_interval is not None and args.plot_interval <= 0:
        raise RuntimeError("--plot-interval must be positive")

    args.artifact_base.mkdir(parents=True, exist_ok=True)
    if not args.skip_merged_lef_generation:
        inserted = write_merged_tech_lef(
            args.base_tech_lef,
            args.cap_source_lef,
            args.merged_tech_lef,
        )
        print(
            "merged_lef=%s inserted_layers=%s"
            % (args.merged_tech_lef, ",".join(inserted))
        )
    else:
        print("merged_lef=%s generation=skipped" % args.merged_tech_lef)

    matrix = load_matrix_runner()
    matrix.TECH_LEF = str(args.merged_tech_lef)
    margin_state = {"margin": None}
    install_margin_command_patch(matrix, margin_state)

    suppressed_log = args.artifact_base / "margin_search_suppressed_stdout.log"
    low = args.low
    high = args.high
    attempts = []
    best = None
    step = 0

    if not args.no_probe_high_first:
        step += 1
        margin = high
        summary = run_attempt(matrix, margin_state, args, margin, suppressed_log)
        success = success_from_summary(summary, args.success_mode)
        attempts.append(
            {
                "step": step,
                "margin": margin,
                "success": success,
                "summary_path": summary.get("summary_path"),
                "run_dir": summary.get("run_dir"),
                "failure": summary.get("failure"),
                "session": summary.get("session"),
                "diagnostics": summary.get("openroad_diagnostic_evidence"),
            }
        )
        if success:
            best = attempts[-1]
        else:
            low = margin
        print(format_attempt(step, margin, summary, success, low, high), flush=True)
        if not success:
            result = {
                "design": args.design,
                "target_iter": args.target_iter,
                "trigger_period": args.trigger_period,
                "plot_interval": args.plot_interval,
                "success_mode": args.success_mode,
                "initial_low": args.low,
                "initial_high": args.high,
                "final_low": low,
                "final_high": high,
                "best": best,
                "attempts": attempts,
                "suppressed_log": str(suppressed_log),
                "bracketed": False,
            }
            result_path = args.artifact_base / "margin_search_summary.json"
            with open(result_path, "w") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
            print("result_json=%s bracketed=0" % result_path)
            return 0

    for _ in range(args.steps):
        step += 1
        margin = (low + high) / 2.0
        summary = run_attempt(matrix, margin_state, args, margin, suppressed_log)
        success = success_from_summary(summary, args.success_mode)
        attempts.append(
            {
                "step": step,
                "margin": margin,
                "success": success,
                "summary_path": summary.get("summary_path"),
                "run_dir": summary.get("run_dir"),
                "failure": summary.get("failure"),
                "session": summary.get("session"),
                "diagnostics": summary.get("openroad_diagnostic_evidence"),
            }
        )
        if success:
            best = attempts[-1]
            high = margin
        else:
            low = margin
        print(format_attempt(step, margin, summary, success, low, high), flush=True)

    result = {
        "design": args.design,
        "target_iter": args.target_iter,
        "trigger_period": args.trigger_period,
        "plot_interval": args.plot_interval,
        "success_mode": args.success_mode,
        "initial_low": args.low,
        "initial_high": args.high,
        "final_low": low,
        "final_high": high,
        "best": best,
        "attempts": attempts,
        "suppressed_log": str(suppressed_log),
        "bracketed": best is not None,
    }
    result_path = args.artifact_base / "margin_search_summary.json"
    with open(result_path, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print("result_json=%s" % result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
