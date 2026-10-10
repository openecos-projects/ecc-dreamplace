#!/usr/bin/env python3

import os
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np


_MISSING = object()


@contextmanager
def restored_sys_path():
    previous_path = list(sys.path)
    previous_path_object = sys.path
    try:
        yield
    finally:
        if sys.path is previous_path_object:
            sys.path[:] = previous_path
        else:
            sys.path = previous_path


def _parent_attribute(module_name):
    parent_name, _, child_name = module_name.rpartition(".")
    if not parent_name:
        return None
    parent = sys.modules.get(parent_name)
    if parent is None:
        return None
    return parent, child_name, getattr(parent, child_name, _MISSING)


@contextmanager
def temporary_modules(modules):
    previous_modules = {
        module_name: sys.modules.get(module_name, _MISSING) for module_name in modules
    }
    previous_attrs = {
        module_name: _parent_attribute(module_name) for module_name in modules
    }

    for module_name, module in modules.items():
        if module is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module
    for module_name, module in modules.items():
        if module is _MISSING:
            continue
        parent_attr = _parent_attribute(module_name)
        if parent_attr is not None:
            parent, child_name, _previous_value = parent_attr
            setattr(parent, child_name, module)

    try:
        yield
    finally:
        for module_name, parent_attr in reversed(previous_attrs.items()):
            if parent_attr is None:
                continue
            parent, child_name, previous_value = parent_attr
            if previous_value is _MISSING:
                if hasattr(parent, child_name):
                    delattr(parent, child_name)
            else:
                setattr(parent, child_name, previous_value)

        for module_name, previous_module in previous_modules.items():
            if previous_module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def _stub_module(module_name, **attributes):
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _placer_import_stubs():
    tools = _stub_module("tools")
    ieda = _stub_module("tools.iEDA")
    module = _stub_module("tools.iEDA.module")
    sta = _stub_module("tools.iEDA.module.sta", IEDASta=object)
    io = _stub_module("tools.iEDA.module.io", IEDAIO=object)
    tools.iEDA = ieda
    ieda.module = module
    module.sta = sta
    module.io = io

    return {
        "dreamplace.macroPlaceDB": _stub_module(
            "dreamplace.macroPlaceDB", MacroPlaceDB=object
        ),
        "dreamplace.NonLinearPlace": _stub_module("dreamplace.NonLinearPlace"),
        "tools": tools,
        "tools.iEDA": ieda,
        "tools.iEDA.module": module,
        "tools.iEDA.module.sta": sta,
        "tools.iEDA.module.io": io,
    }


def load_placement_engine():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    modules = _placer_import_stubs()
    modules["dreamplace.Placer"] = _MISSING

    with restored_sys_path(), temporary_modules(modules):
        sys.path.append(repo_root)
        from dreamplace.Placer import PlacementEngine

        return PlacementEngine


class OpenRoadWriteBackUnitsTest(unittest.TestCase):
    def test_openroad_write_back_exports_macroplacedb_units_unscaled(self):
        recorded = {}

        class FakePlaceIOFunction:
            @staticmethod
            def write(openroad_bridge, def_file, solution_format, node_x, node_y):
                recorded["openroad_bridge"] = openroad_bridge
                recorded["def_file"] = def_file
                recorded["solution_format"] = solution_format
                recorded["node_x"] = np.asarray(node_x)
                recorded["node_y"] = np.asarray(node_y)

        fake_place_io = _stub_module(
            "dreamplace.ops.placeio_openroad.place_io",
            PlaceIOFunction=FakePlaceIOFunction,
            SolutionFileFormat=SimpleNamespace(DEF="DEF"),
        )
        fake_placeio_openroad = _stub_module(
            "dreamplace.ops.placeio_openroad", place_io=fake_place_io
        )
        fake_handoff_launcher = _stub_module(
            "dreamplace.ops.openroad_handoff.launcher"
        )
        fake_openroad_handoff = _stub_module(
            "dreamplace.ops.openroad_handoff", launcher=fake_handoff_launcher
        )
        fake_ops = _stub_module(
            "dreamplace.ops",
            placeio_openroad=fake_placeio_openroad,
            openroad_handoff=fake_openroad_handoff,
        )

        with temporary_modules(
            {
                "dreamplace.ops": fake_ops,
                "dreamplace.ops.placeio_openroad": fake_placeio_openroad,
                "dreamplace.ops.placeio_openroad.place_io": fake_place_io,
                "dreamplace.ops.openroad_handoff": fake_openroad_handoff,
                "dreamplace.ops.openroad_handoff.launcher": fake_handoff_launcher,
            }
        ):
            PlacementEngine = load_placement_engine()
            engine = PlacementEngine.__new__(PlacementEngine)
            engine.params = SimpleNamespace(
                place_io_engine="openroad",
                shift_factor=[0.0, 0.0],
                scale_factor=1.0 / 54.0,
            )

            # Mirrors MacroPlaceDB.unscale_pl_positions: the database stores
            # shifted/scaled coordinates, the DEF export needs DB units.
            def unscale_pl_positions(node_x, node_y):
                unscale_factor = 1.0 / engine.params.scale_factor
                return (
                    np.asarray(node_x) * unscale_factor + engine.params.shift_factor[0],
                    np.asarray(node_y) * unscale_factor + engine.params.shift_factor[1],
                )

            engine.placedb = SimpleNamespace(
                openroad_bridge=object(),
                num_movable_nodes=2,
                node_x=np.array([5400.0, 10800.0, 300.0]),
                node_y=np.array([2700.0, 5400.0, 400.0]),
                unscale_pl_positions=unscale_pl_positions,
            )

            engine.write_back("out.def")

            unscale_factor = 1.0 / engine.params.scale_factor
            np.testing.assert_allclose(recorded["node_x"], np.array([5400.0, 10800.0]) * unscale_factor)
            np.testing.assert_allclose(recorded["node_y"], np.array([2700.0, 5400.0]) * unscale_factor)


if __name__ == "__main__":
    unittest.main()
