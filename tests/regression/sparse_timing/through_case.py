"""Native three-cell reconvergence, four action partials and fresh OpenSTA.

The base manifest supplies the frozen CX55 technology/LEF/Liberty files.
This diagnostic creates its own DEF, Verilog and SDC; it never rewrites the
base design. Wire RC is zero and Liberty/SDC pin loads remain live.
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import preflight
import torch
from hold_action_gradients import verify
from native_hold_case import NativeHoldCase


def fixture(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_manifest.read_text())
    (root / "design.v").write_text("""module through_case(input in, output out);
wire hit, bypass;
BUFX2H7R a (.A(in), .Y(hit));
BUFX4H7R b (.A(in), .Y(bypass));
XOR2X1H7R merge (.A(hit), .B(bypass), .Y(out));
endmodule
""")
    (root / "design.def").write_text("""VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN through_case ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 30000 6000 ) ;
ROW ROW_0 core7 1000 1000 N DO 140 BY 1 STEP 200 0 ;
COMPONENTS 3 ;
- a BUFX2H7R + PLACED ( 2000 1000 ) N ;
- b BUFX4H7R + PLACED ( 6000 1000 ) N ;
- merge XOR2X1H7R + PLACED ( 12000 1000 ) N ;
END COMPONENTS
PINS 2 ;
- in + NET input + DIRECTION INPUT + USE SIGNAL
  + LAYER M3 ( -50 -50 ) ( 50 50 ) + FIXED ( 0 1400 ) N ;
- out + NET output + DIRECTION OUTPUT + USE SIGNAL
  + LAYER M3 ( -50 -50 ) ( 50 50 ) + FIXED ( 30000 1400 ) N ;
END PINS
NETS 4 ;
- input ( PIN in ) ( a A ) ( b A ) ;
- hit ( a Y ) ( merge A ) ;
- bypass ( b Y ) ( merge B ) ;
- output ( merge Y ) ( PIN out ) ;
END NETS
END DESIGN
""")
    selector = "-rise_through" if args.rise_through else "-through"
    sdc = """create_clock -name clock -period 1
set_input_delay -clock clock 0 [get_ports in]
set_input_transition 0.02 [get_ports in]
set_output_delay -clock clock -max 0 [get_ports out]
set_output_delay -clock clock -min -0.2 [get_ports out]
set_load 0.005 [get_ports out]
"""
    if not args.hold_only:
        sdc += f"set_max_delay 0.03 {selector} [get_pins a/Y]\n"
    sdc += f"set_false_path -hold {selector} [get_pins a/Y]\n"
    (root / "constraints.sdc").write_text(sdc)
    base.update(
        {"def": str(root / "design.def"), "sdc": str(root / "constraints.sdc"), "sources": {}}
    )
    manifest = root / "inputs.json"
    manifest.write_text(json.dumps(base, indent=2))
    return root, manifest, base


def golden(root, base, binary):
    lines = [f"read_liberty {{{path}}}" for path in base["libs"]]
    lines += [
        f"read_verilog {{{root / 'design.v'}}}",
        "link_design through_case",
        f"read_sdc {{{root / 'constraints.sdc'}}}",
    ]
    lines.append(f"set output [open {{{root / 'opensta.tsv'}}} w]")
    lines.append('puts $output "analysis\ttransition\tslack_ps"')
    lines.append("""foreach analysis {max min} {
  foreach transition {rise fall} {
    set paths [find_timing_paths -path_delay $analysis -${transition}_to [get_ports out] \
      -endpoint_path_count 1 -group_path_count 100]
    set worst Inf
    foreach path $paths {
      if {![$path is_unconstrained]} {
        set slack [expr {[$path slack] * 1e12}]
        set worst [expr {min($worst, $slack)}]
      }
    }
    puts $output "$analysis\t$transition\t$worst"
  }
}
close $output
exit
""")
    script = root / "opensta.tcl"
    script.write_text("\n".join(lines))
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    with (root / "opensta.log").open("w") as stream:
        subprocess.run(
            [str(binary), "-exit", str(script)],
            check=True,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    with (root / "opensta.tsv").open() as stream:
        return {
            (row["analysis"], row["transition"]): float(row["slack_ps"])
            for row in csv.DictReader(stream, delimiter="\t")
        }


def run(args):
    torch.set_num_threads(args.threads)
    root, manifest, base = fixture(args)
    actual = {}

    def export(pydb, output):
        view = preflight.save_export(pydb, output)
        case = NativeHoldCase(
            pydb, view, dict(np.load(output / "base_arrays.npz")), device=args.device
        )
        verify(case, output, margin_ps=args.margin_ps, samples=1, epsilon=1e-5)
        beta = torch.zeros_like(case.context.beta)
        result = case.forward(beta)
        for edge, name in enumerate(("rise", "fall")):
            rows = [
                i
                for i, context in enumerate(case.hold.contexts)
                if context.data_transition == (1 if edge == 0 else -1)
            ]
            actual["min", name] = float(result.context_slack[rows].min())
        case.timing._active_discrete_action_context = None
        caps = torch.as_tensor(
            view.arrays["pin_reference_cap"], dtype=torch.float64, device=args.device
        )
        zero = case.zeros
        case.timing(
            {"rise": zero, "fall": zero},
            {"rise": zero, "fall": zero},
            {"rise": caps[:, 0], "fall": caps[:, 1]},
            surrogate_mode="lut_only",
        )
        maximum = case.timing._sparse_runtime.last_result
        transitions = case.timing._sparse_runtime.schedule.check_transition
        for edge, name in enumerate(("rise", "fall")):
            actual["max", name] = float(maximum.check_slack[transitions == edge].min())
        min_view = pydb.timing_min_sparse
        np.savez_compressed(output / "min_sparse_arrays.npz", **min_view["arrays"])
        (output / "min_sparse_metadata.json").write_text(
            json.dumps(
                {name: value for name, value in min_view.items() if name != "arrays"}, indent=2
            )
        )
        return view

    args.manifest = manifest
    args.output = root / "ecc"
    preflight.run(args, export=export)
    reference = golden(root, base, args.opensta)
    rows = []
    for key, value in actual.items():
        error = abs(value - reference[key])
        tolerance = max(2.0, 0.03 * abs(reference[key]))
        rows.append(
            {
                "analysis": key[0],
                "transition": key[1],
                "ecc_ps": value,
                "opensta_ps": reference[key],
                "error_ps": error,
                "tolerance_ps": tolerance,
                "passed": bool(np.isfinite(error) and error <= tolerance),
            }
        )
    report = {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "timing_mode": args.timing_mode,
        "device": args.device,
        "rise_through": args.rise_through,
        "hold_only": args.hold_only,
        "rows": rows,
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("native through timing disagrees with OpenSTA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-manifest", "output", "ecc-root", "native-module", "opensta"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timing-mode", choices=("complete", "model_only"), default="model_only")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--margin-ps", type=float, required=True)
    parser.add_argument("--rise-through", action="store_true")
    parser.add_argument("--hold-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
