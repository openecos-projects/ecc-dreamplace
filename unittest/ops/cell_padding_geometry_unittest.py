import unittest
from types import SimpleNamespace

import numpy as np

from dreamplace.macroPlaceDB import MacroPlaceDB, compute_auto_bin_counts


class CellPaddingGeometryTest(unittest.TestCase):
    def test_bm64_padded_area_selects_64_bins(self):
        self.assertEqual(
            compute_auto_bin_counts(978502, 11465, 0.913443, 1046, 1050),
            (64, 64),
        )

    def test_bins_follow_layout_aspect_ratio(self):
        self.assertEqual(compute_auto_bin_counts(10000, 1000, 0.5, 200, 100), (32, 16))
        self.assertEqual(compute_auto_bin_counts(10000, 1000, 0.5, 100, 200), (16, 32))

    def test_large_design_is_capped_at_512(self):
        self.assertEqual(
            compute_auto_bin_counts(1000000, 1000000, 0.8, 1000, 1000),
            (512, 512),
        )

    def test_konder_padded_area_reaches_cap(self):
        self.assertEqual(
            compute_auto_bin_counts(6.91784e7, 844534, 0.367778, 14206, 14210),
            (512, 512),
        )

    def test_site_aligned_padding_updates_movable_geometry_and_pins(self):
        placedb = MacroPlaceDB(None)
        placedb.num_physical_nodes = 2
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.movable_slice = slice(0, 2)
        placedb.fixed_slice = slice(2, 2)
        placedb.node_names = np.array([b"cell0", b"cell1"])
        placedb.node_size_x = np.array([10.0, 20.0])
        placedb.node_size_y = np.array([2.0, 2.0])
        placedb.node_x = np.array([0.0, 10.0])
        placedb.node_y = np.array([0.0, 0.0])
        placedb.pin_offset_x = np.array([1.0, 2.0])
        placedb.pin_offset_y = np.array([0.0, 0.0])
        placedb.pin2node_map = np.array([0, 1], dtype=np.int32)
        placedb.site_width = 2.0
        placedb.row_height = 2.0
        placedb.total_space_area = 2000.0
        params = SimpleNamespace(
            cell_padding_x=5.0,
            macro_only=False,
            macro_halo_x=0.0,
            macro_halo_y=0.0,
            macro_pin_halo_x=0.0,
            macro_pin_halo_y=0.0,
            bndry_padding_x=0.0,
            bndry_padding_y=0.0,
            result_dir=None,
        )
        unpadded_area = float(np.sum(placedb.node_size_x * placedb.node_size_y))

        placedb.update_macros(params)
        placedb._apply_cell_padding(params)

        self.assertEqual((params.cell_padding_x, placedb.cell_padding_x), (6.0, 6.0))
        np.testing.assert_array_equal(placedb.node_size_x, [22.0, 32.0])
        np.testing.assert_array_equal(placedb.node_x, [-6.0, 4.0])
        np.testing.assert_array_equal(placedb.pin_offset_x, [7.0, 8.0])
        padded_area = float(np.sum(placedb.node_size_x * placedb.node_size_y))
        self.assertEqual(
            compute_auto_bin_counts(unpadded_area, 2, 0.5, 40, 40), (4, 4)
        )
        self.assertEqual(
            compute_auto_bin_counts(padded_area, 2, 0.5, 40, 40), (2, 2)
        )

    def test_padding_is_reduced_to_fit_placeable_area(self):
        placedb = MacroPlaceDB(None)
        placedb.num_physical_nodes = 2
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.movable_slice = slice(0, 2)
        placedb.fixed_slice = slice(2, 2)
        placedb.node_size_x = np.array([10.0, 20.0])
        placedb.node_size_y = np.array([2.0, 2.0])
        placedb.node_x = np.array([0.0, 10.0])
        placedb.node_y = np.array([0.0, 0.0])
        placedb.pin_offset_x = np.array([1.0, 2.0])
        placedb.pin_offset_y = np.array([0.0, 0.0])
        placedb.pin2node_map = np.array([0, 1], dtype=np.int32)
        placedb.site_width = 1.0
        placedb.row_height = 2.0
        placedb.total_space_area = 100.0
        params = SimpleNamespace(cell_padding_x=20.0)

        placedb._apply_cell_padding(params)

        self.assertEqual(params.cell_padding_x, 4.0)
        self.assertEqual(placedb.cell_padding_x, 4.0)
        np.testing.assert_array_equal(placedb.node_size_x, [18.0, 28.0])
        np.testing.assert_array_equal(placedb.node_x, [-4.0, 6.0])
        np.testing.assert_array_equal(placedb.pin_offset_x, [5.0, 6.0])
        padded_area = float(np.sum(placedb.node_size_x * placedb.node_size_y))
        self.assertLessEqual(padded_area, 0.99 * placedb.total_space_area)


if __name__ == "__main__":
    unittest.main()
