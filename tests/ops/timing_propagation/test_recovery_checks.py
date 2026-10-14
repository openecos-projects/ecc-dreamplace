"""Native recovery reference slews preserve timing values and live gradients."""

import unittest
from types import SimpleNamespace

import torch
from dreamplace.ops.timing_propagation.recovery_checks import calculate_recovery_rat
from dreamplace.ops.timing_propagation.timing_propagation import scatter_timing_min


class RecoveryChecksTest(unittest.TestCase):
    @staticmethod
    def _recovery_fixture(
        *, native_r=True, native_f=True, timing_metric_scope="setup_plus_recovery"
    ):
        op = SimpleNamespace(
            endpoints_timing_check_arcs=torch.tensor([[0, 1, 0, 0, 1, 1, 3, 0]], dtype=torch.long),
            endpoints_timing_check_max_valid=torch.tensor([[True, True]]),
            num_pins=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            end_points=torch.tensor([1], dtype=torch.long),
            clock_pins=torch.tensor([0], dtype=torch.long),
            clk_pin_rtran=torch.tensor([3.0]),
            clk_pin_ftran=torch.tensor([4.0]),
            timing_aggregation_mode="hard",
            timing_aggregation_tau_ps=2.0,
            timing_metric_scope=timing_metric_scope,
            backend_endpoint_rSlew=torch.tensor([2.0]) if native_r else None,
            backend_endpoint_fSlew=torch.tensor([2.5]) if native_f else None,
            arcs_info=SimpleNamespace(r_check_luts=None, f_check_luts=None),
        )
        op.lut_entry_vectorized = lambda _cells, data, clock, _arcs, _luts: data + 0.1 * clock
        return op

    def test_recovery_requires_native_endpoint_slews_for_recovery_metric(self):
        op = self._recovery_fixture(native_r=False, native_f=False)
        with self.assertRaisesRegex(ValueError, "native endpoint slew tensors"):
            calculate_recovery_rat(
                op,
                torch.tensor([100.0, 100.0]),
                torch.tensor([100.0, 100.0]),
                torch.zeros(2),
                torch.zeros(2),
                scatter_timing_min,
            )

    def test_recovery_rejects_misaligned_native_endpoint_slews(self):
        op = self._recovery_fixture(native_r=True, native_f=True)
        op.backend_endpoint_fSlew = torch.empty(0)
        with self.assertRaisesRegex(ValueError, "one native rise/fall slew"):
            calculate_recovery_rat(
                op,
                torch.tensor([100.0, 100.0]),
                torch.tensor([100.0, 100.0]),
                torch.zeros(2),
                torch.zeros(2),
                scatter_timing_min,
            )

    def test_recovery_native_slews_preserve_live_slew_gradients(self):
        op = self._recovery_fixture()
        pin_rtran = torch.zeros(2, requires_grad=True)
        pin_ftran = torch.zeros(2, requires_grad=True)
        pin_rRAT, pin_fRAT = calculate_recovery_rat(
            op,
            torch.tensor([100.0, 100.0]),
            torch.tensor([100.0, 100.0]),
            pin_rtran,
            pin_ftran,
            scatter_timing_min,
        )
        rise_grad, fall_grad = torch.autograd.grad(
            pin_rRAT[1] + pin_fRAT[1], (pin_rtran, pin_ftran)
        )
        self.assertAlmostEqual(float(rise_grad[1]), 1.0)
        self.assertAlmostEqual(float(fall_grad[1]), 1.0)

    def test_setup_only_accepts_recovery_arcs_without_native_slews(self):
        op = self._recovery_fixture(
            native_r=False, native_f=False, timing_metric_scope="setup_only"
        )
        required = (torch.tensor([100.0, 100.0]), torch.tensor([100.0, 100.0]))
        result = calculate_recovery_rat(
            op, *required, torch.zeros(2), torch.zeros(2), scatter_timing_min
        )
        torch.testing.assert_close(result, required)

    def test_no_valid_recovery_arcs_preserve_required_times_without_native_slews(self):
        for recovery_class, valid in ((1, True), (3, False)):
            with self.subTest(recovery_class=recovery_class, valid=valid):
                op = self._recovery_fixture(native_r=False, native_f=False)
                op.endpoints_timing_check_arcs[:, 6] = recovery_class
                op.endpoints_timing_check_max_valid.fill_(valid)
                required = (torch.tensor([100.0, 100.0]), torch.tensor([100.0, 100.0]))
                result = calculate_recovery_rat(
                    op, *required, torch.zeros(2), torch.zeros(2), scatter_timing_min
                )
                torch.testing.assert_close(result, required)

    def test_empty_recovery_arcs_preserve_required_times_without_native_slews(self):
        op = self._recovery_fixture(native_r=False, native_f=False)
        op.endpoints_timing_check_arcs = torch.empty((0, 8), dtype=torch.long)
        required = (torch.tensor([100.0, 100.0]), torch.tensor([100.0, 100.0]))
        result = calculate_recovery_rat(
            op, *required, torch.zeros(2), torch.zeros(2), scatter_timing_min
        )
        torch.testing.assert_close(result, required)
