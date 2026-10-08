import unittest


class PhysicalActionRealizationPolicyTest(unittest.TestCase):
    def test_realization_policy_is_canonical_physical_action_api(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
            realize_one_net_buffer_action,
        )

        self.assertTrue(callable(realize_one_net_buffer_action))
        self.assertTrue(callable(realize_coordinate_buffer_action))

    def test_coordinate_realization_can_reject_tns_regression(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
        )

        class FakeBridge:
            def run_coordinate_buffer_insert(self, action, config):
                return {
                    "status": "ok",
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": -0.5,
                    "before_metrics": {"wns": -1.0, "tns": -10.0},
                    "after_metrics": {"wns": -1.0, "tns": -10.5},
                }

        result = realize_coordinate_buffer_action(
            FakeBridge(),
            {"action_id": 7, "action_kind": "buffer_insert", "net_name": "n7"},
            config={
                "enable_realization": True,
                "disposable_design": True,
                "max_wns_degradation": 0.0,
                "max_tns_degradation": 0.0,
            },
        )

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reject_reason"], "tns_regression")
        self.assertEqual(result["actual_delta_tns"], -0.5)
        self.assertEqual(result["accept_policy"], "reject_wns_tns_regression")

    def test_coordinate_realization_default_policy_keeps_legacy_tns_behavior(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
        )

        class FakeBridge:
            def run_coordinate_buffer_insert(self, action, config):
                return {
                    "status": "ok",
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": -0.5,
                    "before_metrics": {"wns": -1.0, "tns": -10.0},
                    "after_metrics": {"wns": -1.0, "tns": -10.5},
                }

        result = realize_coordinate_buffer_action(
            FakeBridge(),
            {"action_id": 7, "action_kind": "buffer_insert", "net_name": "n7"},
            config={
                "enable_realization": True,
                "disposable_design": True,
                "max_wns_degradation": 0.0,
            },
        )

        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["accept_policy"], "reject_wns_regression")

    def test_coordinate_realization_passes_deferred_timing_update_to_backend(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
        )

        class FakeBridge:
            def __init__(self):
                self.configs = []

            def run_coordinate_buffer_insert(self, action, config):
                self.configs.append(dict(config))
                return {
                    "status": "ok",
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": 0.0,
                    "before_metrics": {},
                    "after_metrics": {},
                    "source_net_name": "n7",
                    "resolved_target_net_name": "n7_buffer_coord_1",
                    "target_net_remapped": True,
                    "resolved_driver_pin_name": "buffer_coord_buf_1_n7:Y",
                    "driver_pin_remapped": True,
                }

        bridge = FakeBridge()
        result = realize_coordinate_buffer_action(
            bridge,
            {"action_id": 7, "action_kind": "buffer_insert", "net_name": "n7"},
            config={
                "enable_realization": True,
                "disposable_design": True,
                "defer_timing_update": True,
            },
        )

        self.assertEqual(result["status"], "accepted")
        self.assertTrue(bridge.configs[0]["defer_timing_update"])
        self.assertEqual(result["source_net_name"], "n7")
        self.assertEqual(result["resolved_target_net_name"], "n7_buffer_coord_1")
        self.assertTrue(result["target_net_remapped"])
        self.assertEqual(result["resolved_driver_pin_name"], "buffer_coord_buf_1_n7:Y")
        self.assertTrue(result["driver_pin_remapped"])

    def test_coordinate_realization_balanced_policy_accepts_drv_improvement(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
        )

        class FakeBridge:
            def run_coordinate_buffer_insert(self, action, config):
                return {
                    "status": "ok",
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": -10.0,
                    "slew_violation_count_delta": -2.0,
                    "cap_violation_count_delta": 0.0,
                    "inserted_buffer_count_delta": 1,
                    "before_metrics": {
                        "wns": -1.0,
                        "tns": -100.0,
                        "slew_vio_count": 10,
                        "cap_vio_count": 3,
                    },
                    "after_metrics": {
                        "wns": -1.0,
                        "tns": -110.0,
                        "slew_vio_count": 8,
                        "cap_vio_count": 3,
                    },
                }

        result = realize_coordinate_buffer_action(
            FakeBridge(),
            {"action_id": 7, "action_kind": "buffer_insert", "net_name": "n7"},
            config={
                "enable_realization": True,
                "disposable_design": True,
                "max_wns_degradation": 0.0,
                "accept_policy": "balanced_setup_slew_cap",
                "balanced_setup_weight": 1.0,
                "balanced_slew_weight": 20.0,
                "balanced_cap_weight": 20.0,
                "balanced_buffer_weight": 1.0,
                "balanced_min_score": 0.0,
            },
        )

        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["accept_policy"], "balanced_setup_slew_cap")
        self.assertEqual(result["actual_delta_tns"], -10.0)
        self.assertEqual(result["slew_violation_count_delta"], -2.0)
        self.assertGreater(result["balanced_acceptance_score"], 0.0)
        self.assertEqual(
            result["balanced_acceptance_breakdown"]["slew_violation_reduction"],
            2.0,
        )

    def test_coordinate_realization_balanced_policy_rejects_low_score(self):
        from dreamplace.ops.physical_action.realization_policy import (
            realize_coordinate_buffer_action,
        )

        class FakeBridge:
            def run_coordinate_buffer_insert(self, action, config):
                return {
                    "status": "ok",
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": -10.0,
                    "slew_violation_count_delta": 0.0,
                    "cap_violation_count_delta": 0.0,
                    "inserted_buffer_count_delta": 1,
                    "before_metrics": {"wns": -1.0, "tns": -100.0},
                    "after_metrics": {"wns": -1.0, "tns": -110.0},
                }

        result = realize_coordinate_buffer_action(
            FakeBridge(),
            {"action_id": 7, "action_kind": "buffer_insert", "net_name": "n7"},
            config={
                "enable_realization": True,
                "disposable_design": True,
                "max_wns_degradation": 0.0,
                "accept_policy": "balanced_setup_slew_cap",
                "balanced_setup_weight": 1.0,
                "balanced_slew_weight": 20.0,
                "balanced_cap_weight": 20.0,
                "balanced_buffer_weight": 1.0,
                "balanced_min_score": 0.0,
            },
        )

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reject_reason"], "balanced_score_below_threshold")
        self.assertLess(result["balanced_acceptance_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
