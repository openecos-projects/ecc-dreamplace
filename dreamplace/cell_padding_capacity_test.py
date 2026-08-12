import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreamplace.macroPlaceDB import MAX_MOVABLE_UTILIZATION, MacroPlaceDB


class CellPaddingCapacityTest(unittest.TestCase):
    def make_placedb(self, total_space_area=100.0):
        placedb = MacroPlaceDB(None)
        placedb.num_physical_nodes = 2
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.node_size_x = np.array([10.0, 20.0])
        placedb.node_size_y = np.array([2.0, 2.0])
        placedb.node_x = np.array([0.0, 10.0])
        placedb.pin_offset_x = np.array([1.0, 2.0])
        placedb.pin2node_map = np.array([0, 1], dtype=np.int32)
        placedb.site_width = 1.0
        placedb.total_space_area = total_space_area
        return placedb

    def test_cell_padding_is_capped_by_placeable_area(self):
        placedb = self.make_placedb()
        params = SimpleNamespace(cell_padding_x=20.0)

        placedb._apply_cell_padding(params)

        self.assertEqual(params.cell_padding_x, 4.0)
        self.assertEqual(placedb.cell_padding_x, 4.0)
        np.testing.assert_array_equal(placedb.node_size_x, [18.0, 28.0])
        np.testing.assert_array_equal(placedb.node_x, [-4.0, 6.0])
        np.testing.assert_array_equal(placedb.pin_offset_x, [5.0, 6.0])
        movable_area = np.sum(placedb.node_size_x * placedb.node_size_y)
        self.assertLessEqual(movable_area, MAX_MOVABLE_UTILIZATION * placedb.total_space_area)

    def test_cell_padding_is_unchanged_when_area_fits(self):
        placedb = self.make_placedb(total_space_area=200.0)
        params = SimpleNamespace(cell_padding_x=5.0)

        placedb._apply_cell_padding(params)

        self.assertEqual(params.cell_padding_x, 5.0)
        np.testing.assert_array_equal(placedb.node_size_x, [20.0, 30.0])
        np.testing.assert_array_equal(placedb.node_x, [-5.0, 5.0])
        np.testing.assert_array_equal(placedb.pin_offset_x, [6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
