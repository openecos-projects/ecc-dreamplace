#!/usr/bin/env python3

import sys
import types
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DREAMPLACE_ROOT = REPO_ROOT / "dreamplace"
IEDA_ROOT = REPO_ROOT.parent / "iEDA"
sys.path.append(str(REPO_ROOT))


class PlaceIOIEDAContractTest(unittest.TestCase):
    def test_legacy_ieda_interface_stays_removed(self):
        self.assertFalse((DREAMPLACE_ROOT / "ops" / "ieda_interface").exists())
        cmake_text = (DREAMPLACE_ROOT / "ops" / "CMakeLists.txt").read_text()
        self.assertNotIn("ieda_interface", cmake_text)

    @unittest.skipUnless(
        IEDA_ROOT.is_dir(),
        "requires the iEDA source tree next to the repository (%s)" % IEDA_ROOT,
    )
    def test_ieda_pydb_exports_full_timing_check_arc_metadata(self):
        pydb_header = (
            IEDA_ROOT
            / "src"
            / "interface"
            / "python"
            / "py_imp"
            / "idb_to_imp_db"
            / "PyPlaceDB.h"
        ).read_text()
        pydb_binding = (
            IEDA_ROOT
            / "src"
            / "interface"
            / "python"
            / "py_imp"
            / "py_register_imp.cpp"
        ).read_text()
        timing_source = (
            IEDA_ROOT
            / "src"
            / "interface"
            / "python"
            / "py_imp"
            / "idb_to_imp_db"
            / "PyPlaceDBTiming.cpp"
        ).read_text()

        self.assertIn("endpoints_timing_check_arcs", pydb_header)
        self.assertIn("backend_endpoint_min_rAAT", pydb_header)
        self.assertIn("backend_endpoint_min_fRAT", pydb_header)
        self.assertIn('def_readwrite("endpoints_timing_check_arcs"', pydb_binding)
        self.assertIn('def_readwrite("backend_endpoint_min_rAAT"', pydb_binding)
        self.assertIn('def_readwrite("backend_endpoint_min_fRAT"', pydb_binding)
        self.assertIn("append_timing_check_arc", timing_source)
        self.assertIn("lib_arc_timing_check_class_to_int", timing_source)
        self.assertIn("kSetupRising", timing_source)
        self.assertIn("kSetupFalling", timing_source)
        self.assertIn("kRecoveryRising", timing_source)
        self.assertIn("kRecoveryFalling", timing_source)
        self.assertIn("kHoldRising", timing_source)
        self.assertIn("kHoldFalling", timing_source)
        self.assertIn("kRemovalRising", timing_source)
        self.assertIn("kRemovalFalling", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kSetupRising:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kHoldRising:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kRecoveryRising:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kRemovalRising:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kSetupFalling:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kHoldFalling:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kRecoveryFalling:", timing_source)
        self.assertIn("case ista::LibArc::TimingType::kRemovalFalling:", timing_source)
        self.assertIn("info2timing_check_arcs", timing_source)
        self.assertIn("is_timing_check_export_metadata_type", timing_source)
        self.assertIn("emplace_back(-1, -1, arc)", timing_source)
        self.assertIn("AnalysisMode::kMin", timing_source)
        self.assertIn("backend_endpoint_min_rAAT.append", timing_source)
        self.assertIn("backend_endpoint_min_fRAT.append", timing_source)

    def test_live_ieda_imports_are_concentrated_in_placeio_ieda(self):
        allowed = {
            DREAMPLACE_ROOT / "ops" / "placeio_ieda" / "place_io.py",
            DREAMPLACE_ROOT / "Placer.py",
        }
        offenders = []
        for source_path in DREAMPLACE_ROOT.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            text = source_path.read_text()
            if "tools.iEDA" not in text:
                continue
            if source_path not in allowed:
                offenders.append(str(source_path.relative_to(REPO_ROOT)))
        self.assertEqual([], offenders)

    def test_read_builds_ieda_backend_and_pydb_with_process_node(self):
        from dreamplace.ops.placeio_ieda import place_io

        calls = {}

        class FakeIEDAModule:
            @staticmethod
            def pydb(dm_inst, with_sta, vt_config, process_node):
                calls["pydb"] = (dm_inst, with_sta, vt_config, process_node)
                return "pydb"

        class FakeIEDAIO:
            def __init__(self, workspace):
                calls["workspace"] = workspace
                self.ieda = FakeIEDAModule()

            def read_def(self, input_def=""):
                calls["read_def"] = input_def

            def get_dmInst_ptr(self):
                calls["get_dmInst_ptr"] = True
                return "dm-inst"

        workspace = "/tmp/ieda-workspace"
        with patch.object(place_io, "IEDAIO", FakeIEDAIO), patch.object(
            place_io, "_read_process_node_from_workspace", return_value="asap7"
        ):
            backend = place_io.PlaceIOFunction.read(
                types.SimpleNamespace(
                    with_sta=False,
                    design_inputs={"def": "/tmp/current.def"},
                ),
                workspace,
            )

        self.assertEqual(workspace, calls["workspace"])
        self.assertEqual("/tmp/current.def", calls["read_def"])
        self.assertTrue(calls["get_dmInst_ptr"])
        self.assertEqual("pydb", backend.pydb)
        self.assertEqual("dm-inst", backend.dm_inst)
        self.assertEqual(
            ("dm-inst", False, place_io.DEFAULT_VT_CONFIG, "asap7"),
            calls["pydb"],
        )
        self.assertEqual(
            {
                "backend": "ieda",
                "has_sta": True,
                "supports_sta_pydb": True,
                "has_libcell_timing": True,
                "has_diff_sizing_metadata": True,
                "has_buffer_optimizer_metadata": False,
                "has_diff_optimizer_metadata": False,
                "supports_apply_placement": True,
                "supports_apply_sizing": True,
                "supports_buffer_commit": False,
                "supports_committed_refresh": False,
                "supports_tcl_save": True,
                "coordinate_source": "ieda_dm",
                "sta_reference_role": "diagnostic_exporter",
                "parasitics_initialization": "unverified",
                "sta_state_status": "diagnostic_unverified",
            },
            place_io.PlaceIOFunction.backend_caps(backend),
        )

    def test_read_allows_sta_enabled_pydb(self):
        from dreamplace.ops.placeio_ieda import place_io

        calls = {}

        class FakeIEDAModule:
            @staticmethod
            def pydb(dm_inst, with_sta, vt_config, process_node):
                calls["pydb"] = (dm_inst, with_sta, vt_config, process_node)
                return "pydb"

        class FakeIEDAIO:
            def __init__(self, workspace):
                self.ieda = FakeIEDAModule()

            def read_def(self, input_def=""):
                calls["read_def"] = input_def

            def get_dmInst_ptr(self):
                return "dm-inst"

        with patch.object(place_io, "IEDAIO", FakeIEDAIO), patch.object(
            place_io, "_read_process_node_from_workspace", return_value="asap7"
        ):
            backend = place_io.PlaceIOFunction.read(
                types.SimpleNamespace(with_sta=True),
                "/tmp/ieda-workspace",
            )

        self.assertEqual("pydb", backend.pydb)
        self.assertEqual(
            ("dm-inst", True, place_io.DEFAULT_VT_CONFIG, "asap7"),
            calls["pydb"],
        )
        self.assertEqual(
            "diagnostic_exporter",
            place_io.PlaceIOFunction.backend_caps(backend)["sta_reference_role"],
        )
        self.assertEqual(
            "unverified",
            place_io.PlaceIOFunction.backend_caps(backend)["parasitics_initialization"],
        )
        self.assertEqual(
            "diagnostic_unverified",
            place_io.PlaceIOFunction.backend_caps(backend)["sta_state_status"],
        )

    def test_read_falls_back_to_legacy_pydb_signature(self):
        from dreamplace.ops.placeio_ieda import place_io

        calls = {}

        class FakeIEDAModule:
            @staticmethod
            def pydb(*args):
                calls.setdefault("pydb_calls", []).append(args)
                if len(args) == 4:
                    raise TypeError("incompatible function arguments")
                return "legacy-pydb"

        class FakeIEDAIO:
            def __init__(self, workspace):
                self.ieda = FakeIEDAModule()

            def read_def(self, input_def=""):
                calls["read_def"] = input_def

            def get_dmInst_ptr(self):
                return "dm-inst"

        with patch.object(place_io, "IEDAIO", FakeIEDAIO), patch.object(
            place_io, "_read_process_node_from_workspace", return_value="legacy-node"
        ):
            backend = place_io.PlaceIOFunction.read(
                types.SimpleNamespace(with_sta=False),
                "/tmp/ieda-workspace",
            )

        self.assertEqual("legacy-pydb", backend.pydb)
        self.assertEqual("", calls["read_def"])
        self.assertEqual(2, len(calls["pydb_calls"]))
        self.assertEqual(("dm-inst", False, place_io.DEFAULT_VT_CONFIG), calls["pydb_calls"][1])

    def test_read_converts_ieda_loader_system_exit_to_runtime_error(self):
        from dreamplace.ops.placeio_ieda import place_io

        class ExitingIEDAIO:
            def __init__(self, workspace):
                raise SystemExit(0)

        with patch.object(place_io, "IEDAIO", ExitingIEDAIO):
            with self.assertRaisesRegex(RuntimeError, "failed to initialize iEDA backend"):
                place_io.PlaceIOFunction.read(
                    types.SimpleNamespace(with_sta=False),
                    "/tmp/ieda-workspace",
                )

    def test_write_and_sta_helpers_delegate_to_ieda_backend(self):
        from dreamplace.ops.placeio_ieda import place_io

        calls = {}

        class FakeIEDAIO:
            def __init__(self, workspace):
                calls.setdefault("io_workspaces", []).append(workspace)

            def def_save(self, filename):
                calls.setdefault("def_save", []).append(filename)

            def tcl_save(self, filename):
                calls.setdefault("tcl_save", []).append(filename)

            def write_placement_back(self, dm_inst, node_x, node_y):
                calls["write_placement_back"] = (dm_inst, node_x, node_y)
                return "placement-result"

            def write_sizing_back(self, dm_inst, cell_ids, cell_master_names):
                calls["write_sizing_back"] = (dm_inst, cell_ids, cell_master_names)
                return {"ok": True}

        class FakeIEDASta:
            def __init__(self, workspace):
                calls["sta_workspace"] = workspace

            def init_sta(self):
                calls["init_sta"] = True

        backend = place_io.IEDAPlaceIOBackend(
            workspace="/tmp/ieda-workspace",
            io=FakeIEDAIO("/tmp/ieda-workspace"),
            dm_inst="dm-inst",
            pydb="pydb",
        )
        with patch.object(place_io, "IEDAIO", FakeIEDAIO), patch.object(
            place_io, "IEDASta", FakeIEDASta
        ):
            place_io.PlaceIOFunction.write_def(backend, "out.def")
            place_io.PlaceIOFunction.write_tcl(backend, "out.tcl")
            place_io.PlaceIOFunction.write_def("/tmp/ieda-workspace", "workspace.def")
            place_io.PlaceIOFunction.write_tcl("/tmp/ieda-workspace", "workspace.tcl")
            place_io.PlaceIOFunction.apply(backend, "x", "y")
            sizing_result = place_io.PlaceIOFunction.apply_sizing(
                backend,
                [1, 2],
                ["A", "B"],
            )
            place_io.PlaceIOFunction.init_sta("/tmp/ieda-workspace")

        self.assertEqual(["out.def", "workspace.def"], calls["def_save"])
        self.assertEqual(["out.tcl", "workspace.tcl"], calls["tcl_save"])
        self.assertIn("/tmp/ieda-workspace", calls["io_workspaces"])
        self.assertEqual(("dm-inst", "x", "y"), calls["write_placement_back"])
        self.assertEqual(("dm-inst", [1, 2], ["A", "B"]), calls["write_sizing_back"])
        self.assertEqual({"ok": True}, sizing_result)
        self.assertEqual("/tmp/ieda-workspace", calls["sta_workspace"])
        self.assertTrue(calls["init_sta"])

    def test_make_io_and_module_binding_are_adapter_owned(self):
        from dreamplace.ops.placeio_ieda import place_io

        calls = {}

        class FakeIEDAIO:
            def __init__(self, workspace, *args, **kwargs):
                calls["make_io"] = (workspace, args, kwargs)

        fake_base = types.SimpleNamespace(ieda=None)
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tools.iEDA.utility" and fromlist == ("base",):
                return types.SimpleNamespace(base=fake_base)
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(place_io, "IEDAIO", FakeIEDAIO), patch(
            "builtins.__import__",
            side_effect=fake_import,
        ):
            io_obj = place_io.PlaceIOFunction.make_io(
                "/tmp/ieda-workspace",
                input_def="design.def",
            )
            returned = place_io.bind_ieda_module("real-ieda")

        self.assertIsInstance(io_obj, FakeIEDAIO)
        self.assertEqual(
            ("/tmp/ieda-workspace", (), {"input_def": "design.def"}),
            calls["make_io"],
        )
        self.assertEqual("real-ieda", returned)
        self.assertEqual("real-ieda", fake_base.ieda)

    def test_macroplacedb_preserves_sta_pydb_for_ieda(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("MacroPlaceDB import requires torch")

        from dreamplace import macroPlaceDB

        calls = {}

        class FakePlaceIOFunction:
            @staticmethod
            def read(params, workspace):
                calls["read"] = (params.with_sta, workspace)
                return "rawdb"

            @staticmethod
            def dm_inst(raw_db):
                return "dm-inst"

            @staticmethod
            def pydb(raw_db):
                return "pydb"

            @staticmethod
            def backend_caps(raw_db):
                return {"backend": "ieda"}

        fake_placeio = types.SimpleNamespace(PlaceIOFunction=FakePlaceIOFunction)
        placedb = macroPlaceDB.MacroPlaceDB.__new__(macroPlaceDB.MacroPlaceDB)
        placedb.pydb = None
        placedb.data_manager = types.SimpleNamespace(dir_workspace="/tmp/ieda-workspace")
        placedb.backend_caps = {}
        placedb.rawdb = None
        placedb.openroad_bridge = None

        with patch.object(macroPlaceDB, "_load_placeio_ieda", return_value=fake_placeio):
            placedb.setup_rawdb(
                types.SimpleNamespace(
                    dtype="float32",
                    place_io_engine="ieda",
                    with_sta=True,
                )
            )

        self.assertEqual((True, "/tmp/ieda-workspace"), calls["read"])
        self.assertEqual("rawdb", placedb.rawdb)
        self.assertEqual("dm-inst", placedb.get_dmInst_ptr)
        self.assertEqual("pydb", placedb.pydb)
        self.assertEqual({"backend": "ieda"}, placedb.backend_caps)

    def test_macroplacedb_diff_optimizer_metadata_gate_uses_backend_caps(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("MacroPlaceDB import requires torch")

        from dreamplace import macroPlaceDB

        placedb = macroPlaceDB.MacroPlaceDB.__new__(macroPlaceDB.MacroPlaceDB)
        placedb.backend_caps = {
            "backend": "ieda",
            "has_diff_optimizer_metadata": False,
            "has_diff_sizing_metadata": True,
        }
        self.assertTrue(placedb._supports_diff_optimizer_metadata())

        placedb.backend_caps = {"backend": "ieda", "has_diff_sizing_metadata": False}
        self.assertFalse(placedb._supports_diff_optimizer_metadata())

        placedb.backend_caps = {"backend": "openroad", "has_diff_optimizer_metadata": True}
        self.assertTrue(placedb._supports_diff_optimizer_metadata())


if __name__ == "__main__":
    unittest.main()
