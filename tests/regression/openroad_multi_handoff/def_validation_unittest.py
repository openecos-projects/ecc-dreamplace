import tempfile
import unittest
from pathlib import Path

from def_validation import (
    DefCoordinateValidationError,
    parse_def_placement_summary,
    validate_final_def_coordinates,
)


def write_def(path, body):
    path.write_text(
        "\n".join(
            [
                "VERSION 5.8 ;",
                "DIVIDERCHAR \"/\" ;",
                "BUSBITCHARS \"[]\" ;",
                body,
                "END DESIGN",
                "",
            ]
        )
    )


class DefValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_summary_counts_placements_and_movable_bounds(self):
        def_path = self.root / "mixed.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "DIEAREA ( 0 0 ) ( 1000 800 ) ;",
                    "COMPONENTS 4 ;",
                    "  - u_movable_a NAND2_X1 + PLACED ( 10 20 ) N ;",
                    "  - u_fixed_a MACRO + FIXED ( 900 700 ) N ;",
                    "  - u_movable_b INV_X1 + PLACED ( 1100 100 ) N ;",
                    "  - u_unplaced BUF_X1 ;",
                    "END COMPONENTS",
                ]
            ),
        )

        summary = parse_def_placement_summary(def_path)

        self.assertEqual(summary["def_path"], str(def_path))
        self.assertEqual(summary["diearea"], (0, 0, 1000, 800))
        self.assertEqual(summary["component_count"], 4)
        self.assertEqual(summary["movable_count"], 2)
        self.assertEqual(summary["fixed_count"], 1)
        self.assertEqual(summary["movable_bbox"], (10, 20, 1100, 100))
        self.assertEqual(summary["fixed_bbox"], (900, 700, 900, 700))
        self.assertEqual(summary["movable_outside_die_count"], 1)
        self.assertEqual(summary["fixed_outside_die_count"], 0)
        self.assertEqual(summary["tolerance_dbu"], 0)

    def test_parse_summary_accepts_string_def_path(self):
        def_path = self.root / "string_path.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "DIEAREA ( 100 200 ) ( 300 400 ) ;",
                    "COMPONENTS 1 ;",
                    "  - u0 NAND2_X1 + PLACED ( 150 250 ) N ;",
                    "END COMPONENTS",
                ]
            ),
        )

        summary = parse_def_placement_summary(str(def_path))

        self.assertEqual(summary["def_path"], str(def_path))
        self.assertEqual(summary["diearea"], (100, 200, 300, 400))
        self.assertEqual(summary["component_count"], 1)

    def test_validate_rejects_all_movable_components_outside_diearea(self):
        def_path = self.root / "outside.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 2 ;",
                    "  - u0 NAND2_X1 + PLACED ( 100000 100000 ) N ;",
                    "  - u1 INV_X1 + PLACED ( 200000 200000 ) N ;",
                    "END COMPONENTS",
                ]
            ),
        )

        with self.assertRaisesRegex(DefCoordinateValidationError, "outside DIEAREA"):
            validate_final_def_coordinates(def_path)

    def test_validate_rejects_missing_diearea(self):
        def_path = self.root / "missing_diearea.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "COMPONENTS 1 ;",
                    "  - u0 NAND2_X1 + PLACED ( 10 20 ) N ;",
                    "END COMPONENTS",
                ]
            ),
        )

        with self.assertRaisesRegex(DefCoordinateValidationError, "missing DIEAREA"):
            validate_final_def_coordinates(def_path)

    def test_validate_rejects_zero_movable_placed_components(self):
        def_path = self.root / "zero_movable.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 2 ;",
                    "  - u_fixed MACRO + FIXED ( 500 500 ) N ;",
                    "  - u_unplaced BUF_X1 ;",
                    "END COMPONENTS",
                ]
            ),
        )

        with self.assertRaisesRegex(DefCoordinateValidationError, "zero movable"):
            validate_final_def_coordinates(def_path)

    def test_validate_allows_small_boundary_tolerance(self):
        def_path = self.root / "boundary_tolerance.def"
        write_def(
            def_path,
            "\n".join(
                [
                    "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                    "COMPONENTS 2 ;",
                    "  - u0 NAND2_X1 + PLACED ( -5 500 ) N ;",
                    "  - u1 INV_X1 + PLACED ( 1005 1000 ) N ;",
                    "END COMPONENTS",
                ]
            ),
        )

        summary = validate_final_def_coordinates(def_path, tolerance_dbu=5)

        self.assertEqual(summary["movable_outside_die_count"], 0)
        self.assertEqual(summary["movable_bbox"], (-5, 500, 1005, 1000))
        self.assertEqual(summary["tolerance_dbu"], 5)


if __name__ == "__main__":
    unittest.main()
