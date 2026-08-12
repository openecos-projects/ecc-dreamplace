import os
import sys
import types
import unittest

import numpy as np


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)


def _install_ieda_stubs():
    tools = types.ModuleType("tools")
    ieda = types.ModuleType("tools.iEDA")
    data = types.ModuleType("tools.iEDA.data")
    design = types.ModuleType("tools.iEDA.data.design")
    module = types.ModuleType("tools.iEDA.module")
    io = types.ModuleType("tools.iEDA.module.io")

    class IEDADesign:
        pass

    class IEDAIO:
        pass

    design.IEDADesign = IEDADesign
    io.IEDAIO = IEDAIO
    sys.modules.setdefault("tools", tools)
    sys.modules.setdefault("tools.iEDA", ieda)
    sys.modules.setdefault("tools.iEDA.data", data)
    sys.modules.setdefault("tools.iEDA.data.design", design)
    sys.modules.setdefault("tools.iEDA.module", module)
    sys.modules.setdefault("tools.iEDA.module.io", io)


_install_ieda_stubs()

from dreamplace.macroPlaceDB import MacroPlaceDB  # noqa: E402


class MacroPlaceDBFixedMacroMaskTest(unittest.TestCase):
    def _make_params(self):
        return types.SimpleNamespace(
            result_dir=None,
            bndry_padding_x=0.0,
            bndry_padding_y=0.0,
            macro_halo_x=0.0,
            macro_halo_y=0.0,
            macro_pin_halo_x=0.0,
            macro_pin_halo_y=0.0,
            cell_padding_x=0.0,
        )

    def _make_minimal_placedb(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.dtype = np.float32
        placedb.num_physical_nodes = 5
        placedb.num_terminals = 3
        placedb.num_terminal_NIs = 0
        placedb.num_place_blockages = 2
        placedb.movable_slice = slice(0, 2)
        placedb.fixed_slice = slice(2, 5)
        placedb.node_names = np.array(
            [b"cell0", b"cell1", b"real_fixed_macro", b"blockage0", b"blockage1"]
        )
        placedb.node_x = np.zeros(5, dtype=np.float32)
        placedb.node_y = np.zeros(5, dtype=np.float32)
        placedb.node_size_x = np.array([1.0, 1.0, 20.0, 20.0, 20.0], dtype=np.float32)
        placedb.node_size_y = np.array([1.0, 1.0, 5.0, 5.0, 5.0], dtype=np.float32)
        placedb.pin2node_map = np.array([0, 1], dtype=np.int32)
        placedb.node2pin_map = [np.array([0]), np.array([1]), np.array([]), np.array([]), np.array([])]
        placedb.pin_offset_x = np.zeros(2, dtype=np.float32)
        placedb.pin_offset_y = np.zeros(2, dtype=np.float32)
        placedb.site_width = 1.0
        placedb.row_height = 1.0
        return placedb

    def test_update_macros_excludes_ieda_placement_blockages_from_fixed_macro_mask(self):
        placedb = self._make_minimal_placedb()

        placedb.update_macros(self._make_params())

        self.assertEqual(placedb.fixed_macro_mask.tolist(), [True, False, False])
        self.assertEqual(placedb.fixed_macro_idx.tolist(), [2])
        self.assertEqual(placedb.num_fixed_macros, 1)


if __name__ == "__main__":
    unittest.main()
