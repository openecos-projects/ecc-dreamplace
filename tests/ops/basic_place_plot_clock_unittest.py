import os
import sys
import types
import unittest
from pathlib import Path


def _install_basicplace_import_stubs():
    module_names = [
        "dreamplace.ops",
        "dreamplace.ops.move_boundary",
        "dreamplace.ops.move_boundary.move_boundary",
        "dreamplace.ops.hpwl",
        "dreamplace.ops.hpwl.hpwl",
        "dreamplace.ops.rmst_wl",
        "dreamplace.ops.rmst_wl.rmst_wl",
        "dreamplace.ops.macro_legalize",
        "dreamplace.ops.macro_legalize.macro_legalize",
        "dreamplace.ops.greedy_legalize",
        "dreamplace.ops.greedy_legalize.greedy_legalize",
        "dreamplace.ops.abacus_legalize",
        "dreamplace.ops.abacus_legalize.abacus_legalize",
        "dreamplace.ops.legality_check",
        "dreamplace.ops.legality_check.legality_check",
        "dreamplace.ops.draw_place",
        "dreamplace.ops.draw_place.draw_place",
        "dreamplace.ops.pin_pos",
        "dreamplace.ops.pin_pos.pin_pos",
        "dreamplace.ops.global_swap",
        "dreamplace.ops.global_swap.global_swap",
        "dreamplace.ops.k_reorder",
        "dreamplace.ops.k_reorder.k_reorder",
        "dreamplace.ops.independent_set_matching",
        "dreamplace.ops.independent_set_matching.independent_set_matching",
        "dreamplace.ops.steiner_topo",
        "dreamplace.ops.steiner_topo.steiner_topo",
        "dreamplace.ops.timing_propagation",
        "dreamplace.ops.timing_propagation.timing_propagation",
        "dreamplace.ops.pin_weight_sum",
        "dreamplace.ops.pin_weight_sum.pin_weight_sum",
        "dreamplace.ops.irt_egr",
        "dreamplace.ops.irt_egr.irt_egr",
        "dreamplace.ops.cell_modeling",
        "dreamplace.ops.cell_modeling.cell_modeling",
        "dreamplace.ops.gate_projection",
        "dreamplace.ops.gate_projection.gate_projection",
        "dreamplace.ops.timing_propagation.crash_stage_marker",
    ]
    package_names = {
        name
        for name in module_names
        if name.count(".") <= 2 or name.endswith(("move_boundary", "draw_place"))
    }
    previous = {}
    for name in module_names:
        previous[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        if name in package_names:
            module.__path__ = []
        sys.modules[name] = module
    sys.modules[
        "dreamplace.ops.timing_propagation.timing_propagation"
    ].ARCS_INFO = object()
    sys.modules[
        "dreamplace.ops.timing_propagation.timing_propagation"
    ].LUTS_INFO = object()
    sys.modules[
        "dreamplace.ops.timing_propagation.crash_stage_marker"
    ].write_crash_stage_marker = lambda *args, **kwargs: None
    sys.modules["dreamplace.ops.cell_modeling.cell_modeling"].CellModeling = object
    gate_projection = sys.modules["dreamplace.ops.gate_projection.gate_projection"]
    for symbol in (
        "ArgminProjectionResolver",
        "GateProjectionOp",
        "NearestSizeProjectionResolver",
        "StableCurrentCellResolver",
        "VectorizedMainIdCandidateProvider",
    ):
        setattr(gate_projection, symbol, object)
    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_basicplace_module():
    previous = _install_basicplace_import_stubs()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        import dreamplace.BasicPlace as BasicPlaceModule
    finally:
        sys.path.pop(0)
        _restore_modules(previous)
    return BasicPlaceModule


class BasicPlacePlotClockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.BasicPlaceModule = _load_basicplace_module()

    def test_default_plot_name_uses_global_generation_and_local_iteration(self):
        placedb = types.SimpleNamespace()

        first = self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=0,
        )
        second = self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=30,
        )

        self.assertEqual(
            first,
            "/tmp/result/design/plot/iter000000_g00_local0000.png",
        )
        self.assertEqual(
            second,
            "/tmp/result/design/plot/iter000030_g00_local0030.png",
        )

    def test_generation_change_avoids_restart_overwrite(self):
        placedb = types.SimpleNamespace(runtimedb_generation=0)
        self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=0,
        )
        placedb.runtimedb_generation = 1

        restart = self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=0,
        )

        self.assertEqual(
            restart,
            "/tmp/result/design/plot/iter000001_g01_local0000.png",
        )

    def test_repeated_sentinel_frames_get_stable_tags_and_unique_global_numbers(self):
        placedb = types.SimpleNamespace(runtimedb_generation=0)

        first = self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=9999,
            tag="final",
        )
        second = self.BasicPlaceModule.next_plot_filename(
            "/tmp/result/design",
            placedb,
            local_iteration=9999,
            tag="final",
        )

        self.assertEqual(
            first,
            "/tmp/result/design/plot/iter000000_g00_local9999_final.png",
        )
        self.assertEqual(
            second,
            "/tmp/result/design/plot/iter000001_g00_local9999_final.png",
        )

    def test_plot_uses_clock_filename_but_keeps_local_iteration_argument(self):
        import tempfile
        import numpy as np

        calls = []
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)

        class FakeParams:
            result_dir = tmpdir.name

            def design_name(self):
                return "design"

        placedb = types.SimpleNamespace(runtimedb_generation=1)
        engine = object.__new__(self.BasicPlaceModule.BasicPlace)
        engine.op_collections = types.SimpleNamespace(
            draw_place_op=lambda pos, figname: calls.append(figname)
        )

        engine.plot(FakeParams(), placedb, 0, np.array([1.0, 2.0]))

        self.assertEqual(len(calls), 1)
        self.assertTrue(
            calls[0].endswith("design/plot/iter000000_g01_local0000.png"),
            calls[0],
        )


if __name__ == "__main__":
    unittest.main()
