import json
import os
import sys
import unittest


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.Params import Params  # noqa: E402


class IOPinDensityWeightParamsTest(unittest.TestCase):
    def test_schema_default_enables_xplace_iopin_density_weight(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.assertIn("iopin_density_weight", params)
        self.assertEqual(params["iopin_density_weight"]["default"], 3.0)
        self.assertIn("m2_pg_rail_density_weight", params)
        self.assertEqual(params["m2_pg_rail_density_weight"]["default"], 1.0)

    def test_missing_runtime_value_defaults_to_xplace_iopin_density_weight(self):
        params = Params()
        params.fromJson({})

        self.assertEqual(params.iopin_density_weight, 3.0)
        self.assertEqual(params.m2_pg_rail_density_weight, 1.0)

    def test_json_value_is_cast_to_float(self):
        params = Params()
        params.fromJson({"iopin_density_weight": 3, "m2_pg_rail_density_weight": 2})

        self.assertEqual(params.iopin_density_weight, 3.0)
        self.assertIsInstance(params.iopin_density_weight, float)
        self.assertEqual(params.m2_pg_rail_density_weight, 2.0)
        self.assertIsInstance(params.m2_pg_rail_density_weight, float)

    def test_command_line_value_is_cast_to_float(self):
        params = Params()
        params.fromCmdLine(["--iopin_density_weight=3.0", "--m2_pg_rail_density_weight=2.0"])

        self.assertEqual(params.iopin_density_weight, 3.0)
        self.assertIsInstance(params.iopin_density_weight, float)
        self.assertEqual(params.m2_pg_rail_density_weight, 2.0)
        self.assertIsInstance(params.m2_pg_rail_density_weight, float)

    def test_negative_value_is_rejected(self):
        params = Params()

        with self.assertRaisesRegex(ValueError, "iopin_density_weight"):
            params.fromJson({"iopin_density_weight": -0.1})

        params = Params()
        with self.assertRaisesRegex(ValueError, "m2_pg_rail_density_weight"):
            params.fromJson({"m2_pg_rail_density_weight": -0.1})

    def test_non_numeric_value_is_rejected(self):
        params = Params()

        with self.assertRaisesRegex(ValueError, "iopin_density_weight"):
            params.fromCmdLine(["--iopin_density_weight=abc"])

        params = Params()
        with self.assertRaisesRegex(ValueError, "m2_pg_rail_density_weight"):
            params.fromCmdLine(["--m2_pg_rail_density_weight=abc"])


if __name__ == "__main__":
    unittest.main()
