import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy

# matplotlib < 3.8 references numpy.Inf, which numpy >= 2 removed; the plotting
# tests cannot run on that combination (environment capability, not product).
PLOTTING_STACK_USABLE = not (
    numpy.__version__ >= "2"
    and tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) < (3, 8)
)
PLOTTING_STACK_REASON = (
    "matplotlib %s cannot run against numpy %s; install matplotlib>=3.8"
    % (matplotlib.__version__, numpy.__version__)
)

SCRIPT = Path(__file__).with_name("plot_iccad24_def_compare.py")


def write_def(path, diearea=(0, 0, 1000, 800), placed=None, fixed=None):
    placed = placed or []
    fixed = fixed or []
    lines = [
        "VERSION 5.8 ;",
        "DIVIDERCHAR \"/\" ;",
        "BUSBITCHARS \"[]\" ;",
        "DESIGN tiny ;",
        "UNITS DISTANCE MICRONS 1000 ;",
        "DIEAREA ( %d %d ) ( %d %d ) ;" % diearea,
        "COMPONENTS %d ;" % (len(placed) + len(fixed)),
    ]
    for idx, item in enumerate(placed):
        if len(item) == 2:
            x, y = item
            master = "INV_X1"
        else:
            x, y, master = item
        lines.append("- u%d %s + PLACED ( %d %d ) N ;" % (idx, master, x, y))
    for idx, item in enumerate(fixed):
        if len(item) == 2:
            x, y = item
            master = "TAPCELL"
        else:
            x, y, master = item
        lines.append("- f%d %s + FIXED ( %d %d ) N ;" % (idx, master, x, y))
    lines.extend(["END COMPONENTS", "END DESIGN"])
    path.write_text("\n".join(lines) + "\n")


def write_lef(path):
    path.write_text(
        "\n".join(
            [
                "VERSION 5.8 ;",
                "MACRO INV_X1",
                "  CLASS CORE ;",
                "  SIZE 1.000 BY 1.000 ;",
                "END INV_X1",
                "MACRO SRAM_BIG",
                "  CLASS BLOCK ;",
                "  SIZE 120.000 BY 80.000 ;",
                "END SRAM_BIG",
            ]
        )
        + "\n"
    )


@unittest.skipUnless(PLOTTING_STACK_USABLE, PLOTTING_STACK_REASON)
class PlotIccad24DefCompareTest(unittest.TestCase):
    def test_cli_generates_plot_and_summary_from_row_final_def(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design_dir = root / "design" / "tiny"
            design_dir.mkdir(parents=True)
            original_def = design_dir / "tiny.def"
            final_def = root / "runs" / "tiny_final.def"
            final_def.parent.mkdir()
            write_def(original_def, placed=[(100, 100), (300, 200)], fixed=[(0, 0)])
            write_def(final_def, placed=[(150, 120), (350, 220)], fixed=[(0, 0)])
            summary_path = root / "run_sta_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "design": "tiny",
                                "mode_config": "buffer_only",
                                "status": "sta_completed",
                                "final_def": str(final_def),
                            }
                        ]
                    }
                )
            )
            output_dir = root / "plots"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--summary",
                    str(summary_path),
                    "--benchmark-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=SCRIPT.parents[3],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            plot_path = output_dir / "tiny__buffer_only__original_vs_ours.png"
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)
            output_summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(len(output_summary["plots"]), 1)
            entry = output_summary["plots"][0]
            self.assertEqual(entry["design"], "tiny")
            self.assertEqual(entry["mode_config"], "buffer_only")
            self.assertEqual(entry["original"]["movable_count"], 2)
            self.assertEqual(entry["final"]["movable_count"], 2)
            self.assertEqual(entry["original"]["diearea"], [0, 0, 1000, 800])
            self.assertEqual(entry["plot_path"], str(plot_path))
            self.assertEqual(entry["compare_plot_path"], str(plot_path))
            self.assertEqual(entry["plot_style"]["dpi"], 240)
            self.assertEqual(entry["plot_style"]["bins"], [160, 160])
            self.assertIn("count / bin", entry["plot_style"]["colorbar_label"])
            self.assertTrue(entry["plot_style"]["shared_colorbar"])
            self.assertTrue(entry["plot_style"]["shared_color_scale"])
            self.assertGreaterEqual(entry["shared_color_vmax"], 1.0)

    def test_cli_draws_macro_rectangles_from_lef_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design_dir = root / "design" / "macrocase"
            design_dir.mkdir(parents=True)
            lef = root / "macrocase.lef"
            write_lef(lef)
            original_def = design_dir / "macrocase.def"
            final_def = root / "runs" / "macrocase_final.def"
            final_def.parent.mkdir()
            placed = [(100, 100), (200, 200), (300, 300)]
            fixed = [(400, 250, "SRAM_BIG")]
            write_def(original_def, placed=placed, fixed=fixed)
            write_def(final_def, placed=placed, fixed=fixed)
            summary_path = root / "run_sta_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "design": "macrocase",
                                "mode_config": "buffer_only",
                                "status": "sta_completed",
                                "final_def": str(final_def),
                            }
                        ],
                        "placement_summaries": [
                            {
                                "design": "macrocase",
                                "mode_config": "buffer_only",
                                "design_inputs": {"lef": [str(lef)]},
                            }
                        ],
                    }
                )
            )
            output_dir = root / "plots"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--summary",
                    str(summary_path),
                    "--benchmark-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=SCRIPT.parents[3],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_summary = json.loads((output_dir / "summary.json").read_text())
            entry = output_summary["plots"][0]
            self.assertEqual(entry["original"]["macro_count"], 1)
            self.assertEqual(entry["final"]["macro_count"], 1)
            self.assertEqual(entry["original"]["macro_bbox"], [400, 250, 120400, 80250])
            self.assertGreater(
                (output_dir / "macrocase__buffer_only__original_vs_ours.png")
                .stat()
                .st_size,
                0,
            )

    def test_cli_uses_placement_summary_fallback_and_skips_missing_final_def(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design_dir = root / "design" / "fallback"
            design_dir.mkdir(parents=True)
            original_def = design_dir / "fallback.def"
            final_def = root / "runs" / "fallback_final.def"
            final_def.parent.mkdir()
            write_def(original_def, placed=[(20, 30)])
            write_def(final_def, placed=[(40, 60)])
            summary_path = root / "run_sta_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "design": "fallback",
                                "mode_config": "no_handoff",
                                "status": "skipped_unvalidated_final_def",
                            },
                            {
                                "design": "missing",
                                "mode_config": "no_handoff",
                                "status": "skipped_no_final_def",
                            },
                        ],
                        "placement_summaries": [
                            {
                                "design": "fallback",
                                "final_def": str(final_def),
                                "final_def_exists": True,
                            }
                        ],
                    }
                )
            )
            output_dir = root / "plots"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--summary",
                    str(summary_path),
                    "--benchmark-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=SCRIPT.parents[3],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(
                (output_dir / "fallback__no_handoff__original_vs_ours.png").exists()
            )
            self.assertFalse(
                (output_dir / "missing__no_handoff__original_vs_ours.png").exists()
            )
            output_summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                [entry["design"] for entry in output_summary["plots"]],
                ["fallback"],
            )
            self.assertEqual(len(output_summary["skipped"]), 1)
            self.assertEqual(output_summary["skipped"][0]["design"], "missing")


if __name__ == "__main__":
    unittest.main()
