import json
import importlib.util
import os
import sys
import unittest


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.Params import Params  # noqa: E402


def load_enhanced_inflation_controller():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "ops",
        "routability",
        "enhanced_inflation_controller.py",
    )
    spec = importlib.util.spec_from_file_location(
        "enhanced_inflation_controller", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnhancedInflationParamsTest(unittest.TestCase):
    def test_schema_uses_enhanced_inflation_names(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("enhanced_inflation_flag", params)
        self.assertNotIn("enhanced_inflation_max_rounds", params)
        self.assertNotIn("enhanced_inflation_use_target_area", params)
        self.assertNotIn("enhanced_inflation_dynamic_target_density_flag", params)
        self.assertNotIn("xplace_style_inflation_flag", params)
        self.assertNotIn("xplace_inflation_max_rounds", params)

    def test_enhanced_flag_controls_enhanced_inflation(self):
        controller = load_enhanced_inflation_controller()
        params = Params()
        params.fromJson({"routability_opt_flag": 1, "enhanced_inflation_flag": 1})

        self.assertTrue(controller.is_enhanced_inflation_enabled(params))

    def test_legacy_xplace_flag_does_not_enable_enhanced_inflation(self):
        controller = load_enhanced_inflation_controller()
        params = Params()
        params.fromJson({"routability_opt_flag": 1, "xplace_style_inflation_flag": 1})

        self.assertFalse(controller.is_enhanced_inflation_enabled(params))

    def test_enhanced_round_limit_uses_standard_area_adjust_rounds(self):
        controller = load_enhanced_inflation_controller()
        params = Params()
        params.fromJson(
            {
                "routability_opt_flag": 1,
                "enhanced_inflation_flag": 1,
                "max_num_area_adjust": 3,
                "enhanced_inflation_max_rounds": 9,
            }
        )

        self.assertEqual(controller.get_inflation_round_limit(params), 3)


if __name__ == "__main__":
    unittest.main()
