import unittest
from types import SimpleNamespace

import numpy as np

from dreamplace.macroPlaceDB import MacroPlaceDB


class CellPaddingGeometryTest(unittest.TestCase):
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

        placedb.update_macros(params)

        self.assertEqual((params.cell_padding_x, placedb.cell_padding_x), (6.0, 6.0))
        np.testing.assert_array_equal(placedb.node_size_x, [22.0, 32.0])
        np.testing.assert_array_equal(placedb.node_x, [-6.0, 4.0])
        np.testing.assert_array_equal(placedb.pin_offset_x, [7.0, 8.0])


if __name__ == "__main__":
    unittest.main()
