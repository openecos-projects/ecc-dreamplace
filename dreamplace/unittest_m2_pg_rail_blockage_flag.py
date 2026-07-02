import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

BENCHMARK_ROOT = Path(__file__).resolve().parents[4]


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

from dreamplace import macroPlaceDB as macro_place_db_module  # noqa: E402
from dreamplace.macroPlaceDB import MacroPlaceDB  # noqa: E402


class FakeIEDAIO:
    calls = []

    def __init__(self, workspace):
        self.workspace = workspace

    def get_dmInst_ptr(self):
        return "dm-inst-ptr"

    def pydb(self, *args):
        self.calls.append(args)
        return object()


class M2PgRailBlockageFlagTest(unittest.TestCase):
    def setUp(self):
        FakeIEDAIO.calls = []

    def _make_params(self, **overrides):
        params = {
            "dtype": "float32",
            "route_num_bins_x": 7,
            "route_num_bins_y": 11,
            "routability_opt_flag": 1,
            "with_sta": 0,
        }
        params.update(overrides)
        return types.SimpleNamespace(**params)

    def _run_setup_rawdb(self, params):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.data_manager = types.SimpleNamespace(dir_workspace="/tmp/workspace")
        placedb.pydb = None
        with mock.patch.object(macro_place_db_module, "IEDAIO", FakeIEDAIO):
            placedb.setup_rawdb(params)
        return placedb

    def test_schema_defaults_disable_hard_blockage_and_enable_soft_density(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertEqual(params["ieda_m2_pg_rail_blockage_flag"]["default"], 0)
        self.assertIn(
            "synthetic placement blockages",
            params["ieda_m2_pg_rail_blockage_flag"]["description"].lower(),
        )
        self.assertEqual(params["m2_pg_rail_density_weight"]["default"], 1.0)
        self.assertIn(
            "soft density",
            params["m2_pg_rail_density_weight"]["description"].lower(),
        )

    def test_setup_rawdb_passes_hard_flag_off_and_soft_density_on_by_default(self):
        self._run_setup_rawdb(
            self._make_params(ieda_m2_pg_rail_blockage_flag=0)
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False, True),
        )

    def test_setup_rawdb_defaults_missing_hard_flag_to_disabled_with_soft_density_on(self):
        self._run_setup_rawdb(self._make_params())

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False, True),
        )

    def test_setup_rawdb_hard_flag_on_disables_soft_density_collection(self):
        self._run_setup_rawdb(
            self._make_params(ieda_m2_pg_rail_blockage_flag=1)
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, True, False),
        )

    def test_setup_rawdb_zero_soft_density_weight_disables_density_collection(self):
        self._run_setup_rawdb(
            self._make_params(
                ieda_m2_pg_rail_blockage_flag=0,
                m2_pg_rail_density_weight=0.0,
            )
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False, False),
        )

    def test_iopin_density_weight_does_not_affect_m2_pg_rail_flag(self):
        self._run_setup_rawdb(
            self._make_params(ieda_m2_pg_rail_blockage_flag=0, iopin_density_weight=3.0)
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False, True),
        )

    def test_flag_resolver_accepts_numeric_and_text_boolean_values(self):
        cases = [
            (0, False),
            (1, True),
            ("0", False),
            ("1", True),
            ("false", False),
            ("true", True),
            ("off", False),
            ("on", True),
        ]
        for raw_value, expected in cases:
            params = self._make_params(ieda_m2_pg_rail_blockage_flag=raw_value)
            with self.subTest(raw_value=raw_value):
                self.assertIs(
                    MacroPlaceDB._resolve_ieda_m2_pg_rail_blockage_flag(params),
                    expected,
                )

    def test_flag_resolver_rejects_unsupported_text_values(self):
        params = self._make_params(ieda_m2_pg_rail_blockage_flag="maybe")

        with self.assertRaises(ValueError):
            MacroPlaceDB._resolve_ieda_m2_pg_rail_blockage_flag(params)

    def test_ieda_direct_pydb_defaults_hard_blockage_off_and_soft_density_on(self):
        header_path = (
            BENCHMARK_ROOT
            / "AiEDA/third_party/iEDA/src/interface/python/py_imp/"
            "idb_to_imp_db/PyPlaceDB.h"
        )
        register_path = (
            BENCHMARK_ROOT
            / "AiEDA/third_party/iEDA/src/interface/python/py_imp/"
            "py_register_imp.cpp"
        )

        header = header_path.read_text(encoding="utf-8")
        register = register_path.read_text(encoding="utf-8")

        self.assertIn("bool include_m2_pg_rail_blockage = false", header)
        self.assertIn("bool include_m2_pg_rail_density = true", header)
        self.assertIn(
            'py::arg("include_m2_pg_rail_blockage") = false',
            register,
        )
        self.assertIn(
            'py::arg("include_m2_pg_rail_density") = true',
            register,
        )

    def test_place_blockage_count_must_fit_trailing_fixed_terminals(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_terminals = 1
        placedb.num_place_blockages = 2

        with self.assertRaises(RuntimeError):
            placedb._validate_place_blockage_bookkeeping()

    def test_hard_blockage_logging_preserves_original_blockage_message(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(
            m2_pg_rail_blockage_rects=13,
        )

        with self.assertLogs(level="INFO") as logs:
            placedb._log_ieda_m2_pg_rail_blockage_effect(True, False, pydb)

        self.assertIn(
            "PyPlaceDB M2 PG rail blockage rectangles added before union: 13",
            "\n".join(logs.output),
        )

    def test_disabled_effect_logging_records_skipped_soft_density_collection(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(
            m2_pg_rail_blockage_rects=13,
            m2_pg_rail_density_boxes=[[0, 0, 10, 1]],
        )

        with self.assertLogs(level="INFO") as logs:
            placedb._log_ieda_m2_pg_rail_blockage_effect(False, True, pydb)

        self.assertIn(
            "PyPlaceDB M2 PG rail hard blockage conversion skipped",
            "\n".join(logs.output),
        )
        self.assertIn(
            "PyPlaceDB M2 PG rail density boxes exported: 1",
            "\n".join(logs.output),
        )
        self.assertIn(
            "PyPlaceDB M2 PG rail soft density collection enabled",
            "\n".join(logs.output),
        )

    def test_soft_density_weight_zero_logging_records_collection_skipped(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(
            m2_pg_rail_blockage_rects=0,
            m2_pg_rail_density_boxes=[],
        )

        with self.assertLogs(level="INFO") as logs:
            placedb._log_ieda_m2_pg_rail_blockage_effect(False, False, pydb)

        self.assertIn(
            "PyPlaceDB M2 PG rail soft density collection skipped",
            "\n".join(logs.output),
        )

    def test_import_present_empty_rail_density_boxes_is_valid(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(m2_pg_rail_density_boxes=[])

        boxes = placedb._import_m2_pg_rail_density_boxes(pydb, include_m2_pg_rail_density=True)

        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(boxes.dtype, np.float32)

    def test_import_malformed_rail_density_boxes_fails(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(m2_pg_rail_density_boxes=[[0, 0, 0, 1]])

        with self.assertRaises(ValueError):
            placedb._import_m2_pg_rail_density_boxes(pydb, include_m2_pg_rail_density=True)

    def test_flag_off_missing_rail_density_box_field_warns_and_uses_empty(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace()

        with self.assertLogs(level="WARNING") as logs:
            boxes = placedb._import_m2_pg_rail_density_boxes(pydb, include_m2_pg_rail_density=False)

        self.assertEqual(boxes.shape, (0, 4))
        self.assertIn("m2_pg_rail_density_boxes", "\n".join(logs.output))

    def test_flag_on_missing_rail_density_box_field_fails_as_stale_pybind(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace()

        with self.assertRaises(RuntimeError):
            placedb._import_m2_pg_rail_density_boxes(pydb, include_m2_pg_rail_density=True)

    def test_rail_box_scaling_matches_old_hard_node_width_semantics(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.node_x = np.zeros(0, dtype=np.float32)
        placedb.node_y = np.zeros(0, dtype=np.float32)
        placedb.node_size_x = np.zeros(0, dtype=np.float32)
        placedb.node_size_y = np.zeros(0, dtype=np.float32)
        placedb.pin_offset_x = np.zeros(0, dtype=np.float32)
        placedb.pin_offset_y = np.zeros(0, dtype=np.float32)
        placedb.xl = 0.0
        placedb.yl = 0.0
        placedb.xh = 100000.0
        placedb.yh = 100000.0
        placedb.row_height = 1000.0
        placedb.site_width = 100.0
        placedb.min_wire_widths = None
        placedb.min_wire_spacings = None
        placedb.routing_grid_xl = 0.0
        placedb.routing_grid_yl = 0.0
        placedb.routing_grid_xh = 100000.0
        placedb.routing_grid_yh = 100000.0
        placedb.routing_V = 0.0
        placedb.routing_H = 0.0
        placedb.macro_util_V = np.zeros(0, dtype=np.float32)
        placedb.macro_util_H = np.zeros(0, dtype=np.float32)
        placedb.rows = np.zeros((0, 4), dtype=np.float32)
        placedb.total_space_area = 1.0
        placedb.flat_region_boxes = np.zeros((0, 4), dtype=np.float32)
        placedb.regions = []
        placedb.m2_pg_rail_density_boxes = np.array(
            [[81855.0, 0.0, 82145.0, 171000.0]], dtype=np.float32
        )

        original = placedb.m2_pg_rail_density_boxes.copy()
        scale_factor = 0.01
        old_node_x = (original[:, 0] - 0.0) * scale_factor
        old_node_w = (original[:, 2] - original[:, 0]) * scale_factor
        old_hard_node_xh = old_node_x + old_node_w
        independent_scaled_xh = (original[:, 2] - 0.0) * scale_factor
        self.assertFalse(np.array_equal(old_hard_node_xh, independent_scaled_xh))

        placedb.scale(np.array([0.0, 0.0], dtype=np.float32), scale_factor)

        self.assertTrue(
            np.array_equal(placedb.m2_pg_rail_density_boxes[:, 2], old_hard_node_xh)
        )

    def test_native_def_placement_blockage_collection_stays_outside_flag_guard(self):
        source_path = (
            BENCHMARK_ROOT
            / "AiEDA/third_party/iEDA/src/interface/python/py_imp/"
            "idb_to_imp_db/PyPlaceDB.cpp"
        )
        source = source_path.read_text(encoding="utf-8")

        native_blockage_pos = source.index(
            "for (auto blockage : db->get_idb_design()->get_blockage_list()->get_blockage_list())"
        )
        flag_guard_pos = source.index("if (include_m2_pg_rail_blockage)")
        append_pos = source.index("blockage_ps_list.get_rectangles(vRect)")

        self.assertLess(native_blockage_pos, flag_guard_pos)
        self.assertLess(flag_guard_pos, append_pos)
        self.assertIn("blockage_ps_list += ps;", source[native_blockage_pos:flag_guard_pos])
        self.assertIn("blockage_ps_list += rail_ps;", source[flag_guard_pos:append_pos])
        self.assertIn("rail_density_ps", source[flag_guard_pos:append_pos])
        self.assertIn("m2_pg_rail_density_boxes", source[flag_guard_pos:])
        self.assertIn("addNode(\"R0\", block_name, box, true);", source[append_pos:])


if __name__ == "__main__":
    unittest.main()
