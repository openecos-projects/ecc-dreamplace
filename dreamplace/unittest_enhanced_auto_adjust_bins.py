import os
import sys
import unittest


AIEDA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (AIEDA_ROOT, AUTODMP_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dreamplace.macroPlaceDB import _compute_enhanced_auto_adjust_bins  # noqa: E402


class EnhancedAutoAdjustBinsTest(unittest.TestCase):
    def test_caps_y_by_rows_and_scales_x_from_square_preset(self):
        self.assertEqual(
            _compute_enhanced_auto_adjust_bins(
                preset_num_bins_x=512,
                preset_num_bins_y=512,
                layout_height=1320.0,
                row_height=10.0,
            ),
            (128, 128),
        )

    def test_keeps_preset_when_y_does_not_exceed_row_count(self):
        self.assertEqual(
            _compute_enhanced_auto_adjust_bins(
                preset_num_bins_x=1024,
                preset_num_bins_y=1024,
                layout_height=20646.0,
                row_height=9.0,
            ),
            (1024, 1024),
        )

    def test_preserves_preset_bin_ratio_when_row_cap_applies(self):
        self.assertEqual(
            _compute_enhanced_auto_adjust_bins(
                preset_num_bins_x=1024,
                preset_num_bins_y=512,
                layout_height=2000.0,
                row_height=10.0,
            ),
            (256, 128),
        )


if __name__ == "__main__":
    unittest.main()
