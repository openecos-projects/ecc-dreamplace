import json
import os
import sys
import unittest

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.Params import Params  # noqa: E402
from dreamplace.ops.routability import (  # noqa: E402
    enhanced_inflation_controller as _enhanced_inflation_controller,
)


def load_enhanced_inflation_controller():
    return _enhanced_inflation_controller


class EnhancedInflationParamsTest(unittest.TestCase):
    def test_schema_uses_enhanced_inflation_names(self):
        params_path = os.path.join(os.path.dirname(__file__), "..", "..", "dreamplace", "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("enhanced_inflation_flag", params)
        self.assertNotIn("enhanced_inflation_max_rounds", params)
        self.assertNotIn("enhanced_inflation_use_target_area", params)
        self.assertNotIn("enhanced_inflation_dynamic_target_density_flag", params)
        self.assertNotIn("xplace_style_inflation_flag", params)
        self.assertNotIn("xplace_inflation_max_rounds", params)

    def test_inflation_area_budget_ratio_schema_and_runtime_default(self):
        params_path = os.path.join(os.path.dirname(__file__), "..", "..", "dreamplace", "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self.assertEqual(schema["inflation_area_budget_ratio"]["default"], 0.1)
        params = Params()
        params.fromJson({})
        self.assertEqual(params.inflation_area_budget_ratio, 0.1)

    def test_gpugr_area_adjust_defaults_to_directional_max_overflow(self):
        params_path = os.path.join(os.path.dirname(__file__), "..", "..", "dreamplace", "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self.assertEqual(
            schema["gpugr_area_adjust_congestion_mode"]["default"],
            "max_hv",
        )
        params = Params()
        params.fromJson({})
        self.assertEqual(params.gpugr_area_adjust_congestion_mode, "max_hv")

    def test_inflation_area_budget_ratio_is_normalized(self):
        params = Params()
        params.fromJson({"inflation_area_budget_ratio": "0.25"})
        self.assertEqual(params.inflation_area_budget_ratio, 0.25)
        self.assertIsInstance(params.inflation_area_budget_ratio, float)

        for invalid_value in (-0.1, "invalid", float("inf")):
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError, "inflation_area_budget_ratio"
                ):
                    Params().fromJson(
                        {"inflation_area_budget_ratio": invalid_value}
                    )

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
