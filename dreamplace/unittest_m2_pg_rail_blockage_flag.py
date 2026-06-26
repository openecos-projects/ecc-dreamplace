import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


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

    def test_schema_default_enables_m2_pg_rail_blockages(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertEqual(params["ieda_m2_pg_rail_blockage_flag"]["default"], 1)

    def test_setup_rawdb_passes_disabled_flag_to_ieda_pydb(self):
        self._run_setup_rawdb(
            self._make_params(ieda_m2_pg_rail_blockage_flag=0)
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False),
        )

    def test_setup_rawdb_defaults_missing_flag_to_enabled_for_compatibility(self):
        self._run_setup_rawdb(self._make_params())

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, True),
        )

    def test_iopin_density_weight_does_not_affect_m2_pg_rail_flag(self):
        self._run_setup_rawdb(
            self._make_params(ieda_m2_pg_rail_blockage_flag=0, iopin_density_weight=3.0)
        )

        self.assertEqual(len(FakeIEDAIO.calls), 1)
        self.assertEqual(
            FakeIEDAIO.calls[0],
            ("dm-inst-ptr", 7, 11, 1, 0, False),
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

    def test_place_blockage_count_must_fit_trailing_fixed_terminals(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_terminals = 1
        placedb.num_place_blockages = 2

        with self.assertRaises(RuntimeError):
            placedb._validate_place_blockage_bookkeeping()

    def test_enabled_effect_logging_records_rectangle_count(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(m2_pg_rail_blockage_rects=13)

        with self.assertLogs(level="INFO") as logs:
            placedb._log_ieda_m2_pg_rail_blockage_effect(True, pydb)

        self.assertIn(
            "PyPlaceDB M2 PG rail blockage rectangles added before union: 13",
            "\n".join(logs.output),
        )

    def test_disabled_effect_logging_records_skipped_conversion(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        pydb = types.SimpleNamespace(m2_pg_rail_blockage_rects=13)

        with self.assertLogs(level="INFO") as logs:
            placedb._log_ieda_m2_pg_rail_blockage_effect(False, pydb)

        self.assertIn(
            "PyPlaceDB M2 PG rail blockage conversion skipped",
            "\n".join(logs.output),
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
        self.assertIn("addNode(\"R0\", block_name, box, true);", source[append_pos:])


if __name__ == "__main__":
    unittest.main()
