import os
import sys
import types
import unittest


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTODMP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
AIEDA_ROOT = os.path.abspath(os.path.join(AUTODMP_ROOT, "..", ".."))
for path in (AUTODMP_ROOT, AIEDA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dreamplace.Params import Params
from dreamplace import Placer


class PlacementEnginePlotFlagTest(unittest.TestCase):
    def test_place_preserves_disabled_plot_flag(self):
        params = Params()
        params.plot_flag = 0
        params.timing_opt_flag = 0
        params.num_threads = 1
        params.gpu = 0
        params.evaluate_pl = 0
        params.random_seed = 1
        params.deterministic_flag = 0

        engine = Placer.PlacementEngine(params)
        observed = {}

        class FakeNonLinearPlace:
            def __init__(self, params, placedb):
                observed["constructor_plot_flag"] = params.plot_flag

            def __call__(self, params, placedb):
                observed["call_plot_flag"] = params.plot_flag
                metrics = {
                    "objective": [1.0],
                    "hpwl": [2.0],
                    "overflow": [0.1],
                    "density": [0.8],
                }
                return 0.0, 2.0, metrics

        module_name = "dreamplace.NonLinearPlace"
        original_non_linear_place = sys.modules.get(module_name)
        sys.modules[module_name] = types.SimpleNamespace(NonLinearPlace=FakeNonLinearPlace)
        try:
            engine.place()
        finally:
            if original_non_linear_place is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_non_linear_place

        self.assertEqual(observed["constructor_plot_flag"], 0)
        self.assertEqual(observed["call_plot_flag"], 0)
        self.assertEqual(engine.params.plot_flag, 0)


if __name__ == "__main__":
    unittest.main()
