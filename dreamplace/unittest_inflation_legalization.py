import json
import os
import unittest
from types import SimpleNamespace

import torch

from dreamplace import inflation_legalization
from dreamplace.Params import Params


class InflationLegalizationParamsTest(unittest.TestCase):
    def test_schema_defaults_to_disabled(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as stream:
            schema = json.load(stream)

        self.assertEqual(
            schema["legalize_before_each_inflation_flag"]["default"], 0
        )

    def test_flag_is_normalized(self):
        params = Params()
        params.fromJson({"legalize_before_each_inflation_flag": "yes"})
        self.assertEqual(params.legalize_before_each_inflation_flag, 1)

        params.fromJson({"legalize_before_each_inflation_flag": "false"})
        self.assertEqual(params.legalize_before_each_inflation_flag, 0)

    def test_new_mode_uses_stop_overflow_only(self):
        legacy = SimpleNamespace(
            legalize_before_each_inflation_flag=0,
            node_area_adjust_overflow=0.15,
            stop_overflow=0.05,
        )
        scheduled = SimpleNamespace(
            legalize_before_each_inflation_flag=1,
            node_area_adjust_overflow=0.15,
            stop_overflow=0.05,
        )

        self.assertTrue(
            inflation_legalization.should_trigger_legacy_inflation(
                legacy, 0, 3, 0.10
            )
        )
        self.assertFalse(
            inflation_legalization.should_trigger_legacy_inflation(
                scheduled, 0, 3, 0.10
            )
        )
        self.assertTrue(
            inflation_legalization.should_trigger_legacy_inflation(
                scheduled, 0, 3, 0.049
            )
        )

    def test_new_mode_rejects_enhanced_inflation(self):
        params = SimpleNamespace(
            legalize_before_each_inflation_flag=1,
            routability_opt_flag=1,
            legalize_flag=1,
            enhanced_inflation_flag=1,
        )
        with self.assertRaisesRegex(ValueError, "ordinary inflation only"):
            inflation_legalization.validate_params(params)


class InflationGeometryTest(unittest.TestCase):
    def test_physical_round_trip_preserves_centers_and_inflation(self):
        data = SimpleNamespace(
            node_size_x=torch.tensor([4.0, 6.0, 8.0]),
            node_size_y=torch.tensor([10.0, 12.0, 14.0]),
            original_node_size_x=torch.tensor([2.0, 4.0, 6.0]),
            original_node_size_y=torch.tensor([8.0, 10.0, 12.0]),
            pin_offset_x=torch.tensor([3.0, 4.0]),
            pin_offset_y=torch.tensor([5.0, 6.0]),
            original_pin_offset_x=torch.tensor([2.0, 3.0]),
            original_pin_offset_y=torch.tensor([4.0, 5.0]),
        )
        pos = torch.tensor([8.0, 17.0, 26.0, 15.0, 34.0, 53.0])
        inflated_center_x = pos[:3] + data.node_size_x * 0.5
        inflated_center_y = pos[3:] + data.node_size_y * 0.5

        backup = inflation_legalization.use_physical_geometry(pos, data)

        torch.testing.assert_close(
            pos[:3] + data.node_size_x * 0.5, inflated_center_x
        )
        torch.testing.assert_close(
            pos[3:] + data.node_size_y * 0.5, inflated_center_y
        )
        torch.testing.assert_close(data.node_size_x, data.original_node_size_x)
        torch.testing.assert_close(data.node_size_y, data.original_node_size_y)
        torch.testing.assert_close(data.pin_offset_x, data.original_pin_offset_x)
        torch.testing.assert_close(data.pin_offset_y, data.original_pin_offset_y)

        pos[:3].add_(torch.tensor([1.0, 2.0, 3.0]))
        pos[3:].add_(torch.tensor([4.0, 5.0, 6.0]))
        legalized_center_x = pos[:3] + data.node_size_x * 0.5
        legalized_center_y = pos[3:] + data.node_size_y * 0.5

        inflation_legalization.restore_inflated_geometry(pos, data, backup)

        torch.testing.assert_close(
            pos[:3] + data.node_size_x * 0.5, legalized_center_x
        )
        torch.testing.assert_close(
            pos[3:] + data.node_size_y * 0.5, legalized_center_y
        )
        torch.testing.assert_close(data.node_size_x, backup.node_size_x)
        torch.testing.assert_close(data.node_size_y, backup.node_size_y)
        torch.testing.assert_close(data.pin_offset_x, backup.pin_offset_x)
        torch.testing.assert_close(data.pin_offset_y, backup.pin_offset_y)


if __name__ == "__main__":
    unittest.main()
