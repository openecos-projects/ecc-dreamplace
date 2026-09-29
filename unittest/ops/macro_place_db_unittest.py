import os
import sys
import types
import unittest

import numpy as np


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
            macro_only=False,
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


class MacroPlaceDBFixedAreaTest(unittest.TestCase):
    def test_initialize_preserves_native_union_area(self):
        placedb = MacroPlaceDB(None)
        placedb.dtype = np.float32
        placedb.num_physical_nodes = 4
        placedb.num_terminals = 2
        placedb.num_terminal_NIs = 0
        placedb.num_movable_macros = 0
        placedb.num_fixed_macros = 0
        placedb.movable_slice = slice(0, 2)
        placedb.fixed_slice = slice(2, 4)
        placedb.io_slice = slice(4, 4)
        placedb.node_size_x = np.array([2.0, 2.0, 10.0, 10.0], dtype=np.float32)
        placedb.node_size_y = np.array([1.0, 1.0, 10.0, 10.0], dtype=np.float32)
        placedb.node_x = np.zeros(4, dtype=np.float32)
        placedb.node_y = np.zeros(4, dtype=np.float32)
        placedb.node_names = np.array([b"cell0", b"cell1", b"fixed0", b"fixed1"])
        placedb.net_names = np.array([], dtype=np.bytes_)
        placedb.pin2node_map = np.array([], dtype=np.int32)
        placedb.pin2net_map = np.array([], dtype=np.int32)
        placedb.regions = []
        placedb.flat_region_boxes = np.zeros((0, 4), dtype=np.float32)
        placedb.rows = np.zeros((0, 4), dtype=np.float32)
        placedb.xl = 0.0
        placedb.yl = 0.0
        placedb.xh = 100.0
        placedb.yh = 100.0
        placedb.row_height = 1.0
        placedb.site_width = 1.0
        placedb.num_movable_pins = 0
        placedb.movable_macro_mask = np.zeros(2, dtype=bool)
        placedb.cell_padding_x = 0.0
        placedb.bndry_padding_x = 0.0
        placedb.bndry_padding_y = 0.0
        placedb.total_fixed_node_area = 35.0
        placedb.total_space_area = 65.0

        placedb.update_macros = lambda params: None
        placedb.pin_density_inflation = lambda *args: None
        placedb.set_routing_info = lambda route_file: None
        placedb.scale = lambda shift_factor, scale_factor: None

        params = types.SimpleNamespace(
            risa_weights=0,
            pin_density=0.0,
            route_info_input="unused.route_info",
            shift_factor=[0.0, 0.0],
            scale_factor=1.0,
            macro_halo_x=0.0,
            macro_halo_y=0.0,
            macro_pin_halo_x=0.0,
            macro_pin_halo_y=0.0,
            cell_padding_x=0.0,
            target_density=0.5,
            macro_place_flag=True,
            enable_fillers=False,
            routability_opt_flag=False,
            auto_adjust_bins=False,
            enhanced_auto_adjust_bins=False,
            num_bins_x=4,
            num_bins_y=4,
        )

        placedb.initialize(params)

        self.assertEqual(placedb.total_fixed_node_area, 35.0)
        self.assertEqual(placedb.total_space_area, 65.0)


class MacroPlaceDBWritebackTest(unittest.TestCase):
    def test_macro_only_apply_uses_native_selective_writeback(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_physical_nodes = 2
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.node_x = np.zeros(2, dtype=np.float32)
        placedb.node_y = np.zeros(2, dtype=np.float32)
        placedb.ecc_db = object()
        placedb.ecc_module = types.SimpleNamespace(
            write_placement_back=lambda *args: self.fail("dense writeback used")
        )
        calls = []
        placedb.pydb = types.SimpleNamespace(
            write_macro_placement_back=lambda node_x, node_y: calls.append(
                (node_x.copy(), node_y.copy())
            )
        )

        placedb.apply(
            types.SimpleNamespace(macro_only=1, scale_factor=2.0, shift_factor=[10.0, 20.0]),
            np.array([2.0, 4.0], dtype=np.float32),
            np.array([8.0, 10.0], dtype=np.float32),
        )

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], [11.0, 12.0])
        np.testing.assert_array_equal(calls[0][1], [24.0, 25.0])


if __name__ == "__main__":
    unittest.main()
