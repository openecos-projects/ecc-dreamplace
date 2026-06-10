import os
import json
import sys
import unittest


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.Params import Params  # noqa: E402


class LShapePresetParamsTest(unittest.TestCase):
    def test_xplace_weight_schedule_schema_default_is_off(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        self.assertEqual(params["l_shape_use_xplace_weight_schedule"]["default"], 0)

    def test_l_shape_flag_applies_conservative_hard_ggr_defaults(self):
        params = Params()
        params.fromJson({"l_shape_routability_flag": 1})

        self.assertEqual(params.l_direction_use_gpugr, 1)
        self.assertEqual(params.l_shape_use_ggr_topology, 1)
        self.assertEqual(params.l_shape_capacity_al_enable, 1)
        self.assertEqual(params.soft_l_assignment, 0)
        self.assertEqual(params.l_shape_grad_target_ratio, 0.1)
        self.assertEqual(params.l_shape_grad_target_ratio_max, 0.1)
        self.assertEqual(params.l_shape_overflow_threshold, 0.2)
        self.assertEqual(params.l_shape_keep_during_inflation, 1)
        self.assertEqual(params.l_shape_use_xplace_weight_schedule, 0)
        self.assertFalse(hasattr(params, "l_shape_update_interval"))

    def test_l_shape_preset_does_not_override_explicit_values(self):
        params = Params()
        params.fromJson(
            {
                "l_shape_routability_flag": 1,
                "l_direction_use_gpugr": 0,
                "l_shape_grad_target_ratio": 0.2,
                "l_shape_grad_target_ratio_max": 0.2,
                "l_shape_overflow_threshold": 0.3,
                "l_shape_keep_during_inflation": 0,
                "l_shape_use_xplace_weight_schedule": 1,
            }
        )

        self.assertEqual(params.l_direction_use_gpugr, 0)
        self.assertEqual(params.l_shape_grad_target_ratio, 0.2)
        self.assertEqual(params.l_shape_grad_target_ratio_max, 0.2)
        self.assertEqual(params.l_shape_overflow_threshold, 0.3)
        self.assertEqual(params.l_shape_keep_during_inflation, 0)
        self.assertEqual(params.l_shape_use_xplace_weight_schedule, 1)
        self.assertEqual(params.l_shape_use_ggr_topology, 1)
        self.assertEqual(params.l_shape_capacity_al_enable, 1)
        self.assertEqual(params.soft_l_assignment, 0)

    def test_l_shape_preset_is_inactive_when_flag_is_off(self):
        params = Params()
        params.fromJson({"l_shape_routability_flag": 0})

        self.assertFalse(hasattr(params, "l_direction_use_gpugr"))
        self.assertFalse(hasattr(params, "l_shape_use_ggr_topology"))
        self.assertFalse(hasattr(params, "l_shape_capacity_al_enable"))
        self.assertFalse(hasattr(params, "l_shape_grad_target_ratio"))
        self.assertFalse(hasattr(params, "l_shape_keep_during_inflation"))
        self.assertFalse(hasattr(params, "l_shape_use_xplace_weight_schedule"))

    def test_l_shape_preset_handles_command_line_string_flags(self):
        disabled = Params()
        disabled.fromCmdLine(["--l_shape_routability_flag=0"])
        self.assertFalse(hasattr(disabled, "l_direction_use_gpugr"))

        enabled = Params()
        enabled.fromCmdLine(["--l_shape_routability_flag=1"])
        self.assertEqual(enabled.l_direction_use_gpugr, 1)


if __name__ == "__main__":
    unittest.main()
