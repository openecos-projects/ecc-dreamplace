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


class OpenRoadPlaceIOTimingUnitsTest(unittest.TestCase):
    def test_read_sources_rc_tcl_before_pydb_export(self):
        class FakeRawDB:
            def __init__(self):
                self.commands = []

            def eval_tcl_string(self, command):
                self.commands.append(command)
                return ""

        raw_db = FakeRawDB()
        fake_cpp = types.SimpleNamespace(forward=lambda args: raw_db)
        place_io = load_place_io_with_fake_cpp(fake_cpp)
        params = types.SimpleNamespace(
            design_inputs={
                "tech_lef": ["/tmp/tech.lef"],
                "def": "/tmp/design.def",
                "lib": ["/tmp/lib.lib"],
                "sdc": "/tmp/design.sdc",
                "rc_tcl": "/tmp/setRC.tcl",
            },
            place_io_engine="openroad",
        )

        returned = place_io.PlaceIOFunction.read(params)

        self.assertIs(returned, raw_db)
        self.assertEqual(
            raw_db.commands,
            [
                "set_ideal_network [all_clocks]",
                "source {/tmp/setRC.tcl}",
                "estimate_parasitics -placement",
            ],
        )


if __name__ == "__main__":
    unittest.main()
