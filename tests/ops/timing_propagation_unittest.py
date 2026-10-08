import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
from dreamplace.PlaceObj import (
    _backend_min_slack_from_endpoint_values,
    _endpoint_check_class,
    _endpoint_backend_check_slack,
    _endpoint_pin_role,
    _endpoint_timing_check_bucket,
    _summarize_endpoint_compare_buckets,
)
sys.path.pop()


class TimingPropagationTest(unittest.TestCase):
    def test_endpoint_check_class_labels_setup_and_async_control(self):
        self.assertEqual(_endpoint_pin_role("u_reg:D"), "data")
        self.assertEqual(_endpoint_pin_role("u_reg:RESET"), "async_control")
        self.assertEqual(_endpoint_check_class("u_reg:D", 1), "setup_exported")
        self.assertEqual(_endpoint_check_class("u_reg:RESET", 0), "async_control_unmodeled")
        self.assertEqual(_endpoint_check_class("u_block:EN", 0), "unsupported_unmodeled")

    def test_endpoint_timing_check_bucket_summary_keeps_async_visible(self):
        self.assertEqual(_endpoint_timing_check_bucket([1, 2]), "setup")
        self.assertEqual(_endpoint_timing_check_bucket([3, 4]), "async_recovery_removal")
        self.assertEqual(_endpoint_timing_check_bucket([2]), "hold")
        self.assertEqual(_endpoint_timing_check_bucket([]), "unmodeled")

        rows = [
            {
                "timing_check_bucket": "setup",
                "timing_check_classes_list": [1],
                "py_slack": -2.0,
                "cpp_slack": -1.0,
            },
            {
                "timing_check_bucket": "async_recovery_removal",
                "timing_check_classes_list": [3, 4],
                "py_slack": -5.0,
                "cpp_slack": -3.0,
            },
        ]

        summary = _summarize_endpoint_compare_buckets(rows)

        self.assertEqual(summary["all"]["count"], 2)
        self.assertEqual(summary["all"]["python_tns"], -7.0)
        self.assertEqual(summary["all"]["backend_tns"], -4.0)
        self.assertEqual(summary["check_bucket:setup"]["count"], 1)
        self.assertEqual(summary["check_bucket:async_recovery_removal"]["count"], 1)
        self.assertEqual(summary["check_class:4"]["backend_tns"], -3.0)

    def test_backend_check_slack_uses_min_side_for_removal(self):
        recovery_only = {
            "timing_check_bucket": "async_recovery_removal",
            "timing_check_classes_list": [3],
            "cpp_slack": -2.0,
            "cpp_min_slack": -8.0,
        }
        removal_only = {
            "timing_check_bucket": "async_recovery_removal",
            "timing_check_classes_list": [4],
            "cpp_slack": -2.0,
            "cpp_min_slack": -8.0,
        }
        recovery_removal = {
            "timing_check_bucket": "async_recovery_removal",
            "timing_check_classes_list": [3, 4],
            "cpp_slack": -2.0,
            "cpp_min_slack": -8.0,
        }

        self.assertEqual(_endpoint_backend_check_slack(recovery_only), -2.0)
        self.assertEqual(_endpoint_backend_check_slack(removal_only), -8.0)
        self.assertEqual(_endpoint_backend_check_slack(recovery_removal), -8.0)

    def test_backend_min_slack_ignores_opensta_sentinel_values(self):
        self.assertTrue(
            torch.isnan(
                torch.tensor(
                    _backend_min_slack_from_endpoint_values(
                        -9.0e7, -9.0e7, 9.0e7, 9.0e7
                    )
                )
            )
        )
        self.assertEqual(
            _backend_min_slack_from_endpoint_values(12.0, 10.0, 5.0, 20.0),
            -10.0,
        )

    def test_empty_endpoint_constraint_arcs_skip_setup_rat_update(self):
        op = TimingPropagation.__new__(TimingPropagation)
        op.endpoints_constraint_arcs = torch.empty(0, dtype=torch.long)
        op.endpoints_timing_check_arcs = torch.tensor(
            [[0, 1, 0, -1, 1, 0, 3, -1], [0, 1, 0, -1, 1, 0, 4, -1]],
            dtype=torch.long,
        )
        pin_rRAT = torch.tensor([1.0, 2.0])
        pin_fRAT = torch.tensor([3.0, 4.0])

        out_rRAT, out_fRAT = op.calculate_setup_rat(
            pin_rRAT,
            pin_fRAT,
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

        self.assertIs(out_rRAT, pin_rRAT)
        self.assertIs(out_fRAT, pin_fRAT)
        self.assertTrue(torch.equal(out_rRAT, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(out_fRAT, torch.tensor([3.0, 4.0])))

    def test_setup_rat_aggregates_multiple_constraints_by_min_required_time(self):
        op = TimingPropagation.__new__(TimingPropagation)
        op.device = torch.device("cpu")
        op.dtype = torch.float32
        op.timing_aggregation_mode = "hard"
        op.timing_aggregation_tau_ps = 1.0
        op.endpoints_constraint_arcs = torch.tensor(
            [
                [0, 2, 0, 0, 1],
                [1, 2, 0, 1, 1],
            ],
            dtype=torch.long,
        )

        def r_setup_entry(_lib_cell_idxs, _clk_pin_rtrans, _data_pin_trans, lib_arc_idxs):
            return torch.where(
                lib_arc_idxs == 0,
                torch.tensor(30.0),
                torch.tensor(10.0),
            )

        def f_setup_entry(_lib_cell_idxs, _clk_pin_ftrans, _data_pin_trans, lib_arc_idxs):
            return torch.where(
                lib_arc_idxs == 0,
                torch.tensor(25.0),
                torch.tensor(12.0),
            )

        op.r_setup_entry = r_setup_entry
        op.f_setup_entry = f_setup_entry
        pin_rRAT = torch.tensor([1.0e8, 1.0e8, 100.0])
        pin_fRAT = torch.tensor([1.0e8, 1.0e8, 100.0])

        out_rRAT, out_fRAT = op.calculate_setup_rat(
            pin_rRAT,
            pin_fRAT,
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
        )

        self.assertAlmostEqual(float(out_rRAT[2]), 70.0)
        self.assertAlmostEqual(float(out_fRAT[2]), 75.0)

    def test_setup_endpoint_mask_excludes_unmodeled_async_endpoint(self):
        op = TimingPropagation.__new__(TimingPropagation)
        op.end_points = torch.tensor([10, 20, 30], dtype=torch.long)
        op.endpoints_constraint_arcs = torch.tensor(
            [
                [0, 10, 0, 0, 1],
                [1, 30, 0, 1, 1],
            ],
            dtype=torch.long,
        )

        mask = op._setup_endpoint_objective_mask(torch.device("cpu"))

        self.assertTrue(torch.equal(mask, torch.tensor([True, False, True])))


if __name__ == "__main__":
    unittest.main()
