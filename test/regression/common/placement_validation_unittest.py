#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import placement_validation as validation


def _row(
    *,
    master: str,
    is_block: int,
    bbox: tuple[int, int, int, int],
    keepout: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    x0, y0, x1, y1 = bbox
    kx0, ky0, kx1, ky1 = keepout or bbox
    return {
        "master": master,
        "is_block": is_block,
        "area_um2": float((x1 - x0) * (y1 - y0)),
        "x_dbu": x0,
        "y_dbu": y0,
        "orient": "R0",
        "placement_status": "FIXED" if is_block else "PLACED",
        "bbox_x_min_dbu": x0,
        "bbox_y_min_dbu": y0,
        "bbox_x_max_dbu": x1,
        "bbox_y_max_dbu": y1,
        "keepout_x_min_dbu": kx0,
        "keepout_y_min_dbu": ky0,
        "keepout_x_max_dbu": kx1,
        "keepout_y_max_dbu": ky1,
    }


def _write_report(path: Path, category: str, sources: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "DPL": {
                    "category": {
                        category: {
                            "violations": [
                                {
                                    "sources": [
                                        {"type": "inst", "name": source}
                                        for source in sources
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class PlacementValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {
            "mem": _row(
                master="SRAM",
                is_block=1,
                bbox=(100, 100, 200, 200),
                keepout=(90, 90, 210, 210),
            ),
            "u0": _row(master="INVx1", is_block=0, bbox=(0, 0, 10, 10)),
        }

    def test_full_core_detailed_placement_is_explicit_and_unique(self):
        tcl = "\n".join(
            validation.detailed_placement_tcl_lines(search_window="full_core")
        )
        self.assertEqual(tcl.count("detailed_placement -max_displacement"), 1)
        self.assertIn("getCoreArea", tcl)
        self.assertIn("ceil", tcl)
        self.assertIn(
            "[list $dpl_max_displacement_x_um $dpl_max_displacement_y_um]", tcl
        )

    def test_unknown_detailed_placement_search_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported detailed-placement"):
            validation.detailed_placement_tcl_lines(search_window="case_override")

    def test_native_opendp_pass_still_requires_macro_invariants(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "missing.json"
            result = validation.qualify_placement(
                raw_placement_valid=1,
                reference_inventory=self.reference,
                evaluated_inventory=dict(self.reference),
                dpl_report_path=report,
            )
        self.assertEqual(result["effective_placement_valid"], 1)
        self.assertEqual(result["mode"], "native_opendp_pass")

    def test_macro_only_padding_grid_false_positive_is_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "dpl.json"
            _write_report(report, "Padding_failures", ["mem"])
            result = validation.qualify_placement(
                raw_placement_valid=0,
                reference_inventory=self.reference,
                evaluated_inventory=dict(self.reference),
                dpl_report_path=report,
            )
        self.assertEqual(result["effective_placement_valid"], 1)
        self.assertEqual(
            result["mode"], "qualified_fixed_macro_grid_padding_false_positive"
        )
        self.assertEqual(result["qualified_padding_failure_count"], 1)

    def test_non_macro_or_real_keepout_failure_remains_invalid(self):
        evaluated = dict(self.reference)
        evaluated["u0"] = _row(master="INVx1", is_block=0, bbox=(95, 95, 105, 105))
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "dpl.json"
            _write_report(report, "Padding_failures", ["mem"])
            overlap_result = validation.qualify_placement(
                raw_placement_valid=0,
                reference_inventory=self.reference,
                evaluated_inventory=evaluated,
                dpl_report_path=report,
            )
            _write_report(report, "Overlap_failures", ["u0"])
            category_result = validation.qualify_placement(
                raw_placement_valid=0,
                reference_inventory=self.reference,
                evaluated_inventory=dict(self.reference),
                dpl_report_path=report,
            )
        self.assertEqual(overlap_result["effective_placement_valid"], 0)
        self.assertFalse(overlap_result["macro_geometry_ok"])
        self.assertEqual(category_result["effective_placement_valid"], 0)


if __name__ == "__main__":
    unittest.main()
