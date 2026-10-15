import unittest
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.openroad_handoff import OpenRoadHandoffController
sys.path.pop()


class OpenRoadHandoffTriggerTest(unittest.TestCase):
    def test_interval_trigger_waits_for_overflow_guard(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "interval",
                    "interval": 50,
                    "guards": [{"name": "overflow", "op": "<=", "value": 0.2}],
                },
            }
        )

        blocked, blocked_decision = controller.should_handoff(
            {
                "absolute_iteration": 50,
                "metric": SimpleNamespace(overflow=0.3),
            }
        )
        allowed, allowed_decision = controller.should_handoff(
            {
                "absolute_iteration": 100,
                "metric": SimpleNamespace(overflow=0.2),
            }
        )

        self.assertFalse(blocked)
        self.assertIsNone(blocked_decision)
        self.assertTrue(allowed)
        self.assertEqual(allowed_decision["trigger_mode"], "interval")
        self.assertIn("overflow<=0.2", allowed_decision["trigger_reason"])

    def test_interval_trigger_dedupes_by_absolute_iteration(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "interval",
                    "interval": 50,
                    "dedupe_by": "absolute_iteration",
                },
            }
        )
        event = {
            "absolute_iteration": 50,
            "metric": SimpleNamespace(overflow=0.1),
        }

        first, first_decision = controller.should_handoff(event)
        second, second_decision = controller.should_handoff(event)

        self.assertTrue(first)
        self.assertIsNotNone(first_decision)
        self.assertFalse(second)
        self.assertIsNone(second_decision)

    def test_dedupe_state_survives_controller_recreation(self):
        config = {
            "enabled": True,
            "trigger": {
                "mode": "interval",
                "interval": 50,
                "dedupe_by": "absolute_iteration",
            },
        }
        params = SimpleNamespace(openroad_handoff=config)
        first_controller = OpenRoadHandoffController.from_params(params)
        second_controller = OpenRoadHandoffController.from_params(params)
        event = {
            "absolute_iteration": 50,
            "metric": SimpleNamespace(overflow=0.1),
        }

        first, first_decision = first_controller.should_handoff(event)
        second, second_decision = second_controller.should_handoff(event)

        self.assertTrue(first)
        self.assertIsNotNone(first_decision)
        self.assertFalse(second)
        self.assertIsNone(second_decision)

    def test_missing_guard_metric_blocks_trigger(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "interval",
                    "interval": 50,
                    "guards": [{"name": "overflow", "op": "<=", "value": 0.2}],
                },
            }
        )

        should_handoff, decision = controller.should_handoff(
            {
                "absolute_iteration": 50,
                "metric": SimpleNamespace(),
            }
        )

        self.assertFalse(should_handoff)
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
