import importlib.util
import sys
import types
import unittest
from pathlib import Path


AUTODMP_ROOT = Path(__file__).resolve().parents[3]
PLACE_IO_PATH = AUTODMP_ROOT / "dreamplace" / "ops" / "placeio_openroad" / "place_io.py"

# Removed test_cpp_exports_timing_scalars_in_python_units_and_wire_rc: it pinned
# C++ implementation text (read_text + assertIn on openroad_place_io.cpp symbols),
# which the repo forbids; those symbols now live in openroad_pyplacedb_export_impl.cpp.


def load_place_io_with_fake_cpp(fake_cpp):
    module_name = "place_io_under_test"
    package_names = [
        "dreamplace",
        "dreamplace.ops",
        "dreamplace.ops.placeio_openroad",
        "dreamplace.ops.placeio_openroad.placeio_openroad_cpp",
    ]
    original_modules = {name: sys.modules.get(name) for name in package_names}
    try:
        sys.modules["dreamplace"] = types.ModuleType("dreamplace")
        sys.modules["dreamplace.ops"] = types.ModuleType("dreamplace.ops")
        sys.modules["dreamplace.ops.placeio_openroad"] = types.ModuleType(
            "dreamplace.ops.placeio_openroad"
        )
        sys.modules["dreamplace.ops.placeio_openroad.placeio_openroad_cpp"] = fake_cpp
        spec = importlib.util.spec_from_file_location(module_name, PLACE_IO_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _FakeRawDB:
    def __init__(self):
        self.commands = []

    def eval_tcl_string(self, command):
        self.commands.append(command)
        return ""


class OpenRoadPlaceIOTimingUnitsTest(unittest.TestCase):
    def _read_with_design_inputs(self, design_inputs, **params_extra):
        raw_db = _FakeRawDB()
        calls = []

        def fake_forward(**kwargs):
            calls.append(kwargs)
            return raw_db

        place_io = load_place_io_with_fake_cpp(
            types.SimpleNamespace(forward=fake_forward)
        )
        params = types.SimpleNamespace(
            design_inputs=design_inputs, place_io_engine="openroad", **params_extra
        )
        returned = place_io.PlaceIOFunction.read(params)
        return returned, raw_db, calls

    def test_read_sources_rc_tcl_before_pydb_export(self):
        returned, raw_db, calls = self._read_with_design_inputs(
            {
                "tech_lef": ["/tmp/tech.lef"],
                "def": "/tmp/design.def",
                "lib": ["/tmp/lib.lib"],
                "sdc": "/tmp/design.sdc",
                "rc_tcl": "/tmp/setRC.tcl",
            }
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(returned, raw_db)
        self.assertEqual(
            raw_db.commands,
            [
                "set_ideal_network [all_clocks]",
                "source {/tmp/setRC.tcl}",
                "estimate_parasitics -placement",
            ],
        )

    def test_read_passes_path_arguments_as_paths(self):
        # Path objects must reach the native keyword interface unchanged:
        # no string joining and no whitespace splitting on design paths.
        tech_lef = Path("/tmp/pdk dir/tech.lef")
        design_def = Path("/tmp/design dir/top level.def")
        design_lib = Path("/tmp/pdk dir/slow.lib")
        design_sdc = Path("/tmp/design dir/top level.sdc")

        _returned, _raw_db, calls = self._read_with_design_inputs(
            {
                "tech_lef": [tech_lef],
                "def": design_def,
                "lib": [design_lib],
                "sdc": design_sdc,
            },
            openroad_vt_suffixes=["HVT"],
        )

        self.assertEqual(
            calls,
            [
                {
                    "def_input": design_def,
                    "lef_input": [tech_lef],
                    "lib_input": [design_lib],
                    "sdc_input": design_sdc,
                    "vt_suffixes": ["HVT"],
                    "num_threads": 8,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
