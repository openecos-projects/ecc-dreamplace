import math
import os
import sys
import types
import unittest

import torch


_tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _tests_dir)
from _import_stubs import auto_stub, register_stub_modules
sys.path.remove(_tests_dir)

# The stubs below isolate the heavy op layer. auto_stub fabricates whatever the
# product imports from dreamplace.ops, so a new op import in the product cannot
# rot this list again (it used to fail with "'dreamplace.ops' is not a package").
_OPS_STUBS = auto_stub("dreamplace.ops")


def _install_nonlinearplace_import_stubs():
    _OPS_STUBS.__enter__()
    modules = {
        "dreamplace.BasicPlace": types.ModuleType("dreamplace.BasicPlace"),
        "dreamplace.PlaceObj": types.ModuleType("dreamplace.PlaceObj"),
        "dreamplace.NesterovAcceleratedGradientOptimizer": types.ModuleType(
            "dreamplace.NesterovAcceleratedGradientOptimizer"
        ),
        "dreamplace.EvalMetrics": types.ModuleType("dreamplace.EvalMetrics"),
        "dreamplace.ops.fence_region": types.ModuleType("dreamplace.ops.fence_region"),
        "dreamplace.ops.fence_region.fence_region": types.ModuleType(
            "dreamplace.ops.fence_region.fence_region"
        ),
        "dreamplace.ops.timing_propagation": types.ModuleType(
            "dreamplace.ops.timing_propagation"
        ),
        "dreamplace.ops.timing_propagation.crash_stage_marker": types.ModuleType(
            "dreamplace.ops.timing_propagation.crash_stage_marker"
        ),
        "dreamplace.ops.timing_net_weighting": types.ModuleType(
            "dreamplace.ops.timing_net_weighting"
        ),
        "dreamplace.ops.timing_net_weighting.net_weighting": types.ModuleType(
            "dreamplace.ops.timing_net_weighting.net_weighting"
        ),
        "dreamplace.ops.gate_projection": types.ModuleType(
            "dreamplace.ops.gate_projection"
        ),
        "dreamplace.ops.gate_projection.gate_projection": types.ModuleType(
            "dreamplace.ops.gate_projection.gate_projection"
        ),
        "dreamplace.ops.discrete_gradient_topk": types.ModuleType(
            "dreamplace.ops.discrete_gradient_topk"
        ),
        "dreamplace.ops.buffer_insertion": types.ModuleType(
            "dreamplace.ops.buffer_insertion"
        ),
        "dreamplace.ops.buffer_insertion.optimization_state": types.ModuleType(
            "dreamplace.ops.buffer_insertion.optimization_state"
        ),
        "dreamplace.ops.size_interpolated_pin.sizing_limit_utils": types.ModuleType(
            "dreamplace.ops.size_interpolated_pin.sizing_limit_utils"
        ),
        "torch_optimizer": types.ModuleType("torch_optimizer"),
        "ncg_optimizer": types.ModuleType("ncg_optimizer"),
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
    }
    modules["dreamplace.BasicPlace"].BasicPlace = type("BasicPlace", (), {})
    modules[
        "dreamplace.ops.timing_propagation.crash_stage_marker"
    ].write_crash_stage_marker = lambda *args, **kwargs: None
    modules[
        "dreamplace.ops.timing_net_weighting.net_weighting"
    ].update_net_weights_from_tp = lambda *args, **kwargs: None
    modules["dreamplace.ops.gate_projection.gate_projection"].GateProjectionOp = object
    modules["dreamplace.ops.gate_projection.gate_projection"].ProjectionContext = object
    modules[
        "dreamplace.ops.discrete_gradient_topk"
    ].apply_discrete_gradient_topk_update = lambda *args, **kwargs: None
    modules[
        "dreamplace.ops.discrete_gradient_topk"
    ].build_discrete_gradient_topk_candidate_cache = lambda *args, **kwargs: None
    modules[
        "dreamplace.ops.buffer_insertion.optimization_state"
    ].buffer_optimization_grad_stats = lambda *args, **kwargs: {}
    modules[
        "dreamplace.ops.buffer_insertion.optimization_state"
    ].capture_buffer_optimization_snapshot = lambda *args, **kwargs: {}
    modules[
        "dreamplace.ops.buffer_insertion.optimization_state"
    ].restore_buffer_optimization_snapshot = lambda *args, **kwargs: None
    sizing_limit_utils = modules["dreamplace.ops.size_interpolated_pin.sizing_limit_utils"]
    for name in (
        "build_size_interpolated_pin_property_debug",
        "compute_current_pin2libpin_flat_ids",
        "compute_size_interpolated_pin_properties",
    ):
        setattr(sizing_limit_utils, name, lambda *args, **kwargs: None)
    return register_stub_modules(modules)


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    _OPS_STUBS.__exit__(None, None, None)


def _load_nonlinearplace_class():
    previous = _install_nonlinearplace_import_stubs()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.NonLinearPlace import NonLinearPlace
    finally:
        sys.path.pop()
        _restore_modules(previous)
    return NonLinearPlace


class NonLinearPlaceNumericTest(unittest.TestCase):
    def test_nonfinite_metric_detects_tensor_and_python_scalars(self):
        NonLinearPlace = _load_nonlinearplace_class()

        finite_metric = types.SimpleNamespace(
            objective=torch.tensor(1.0),
            hpwl=torch.tensor(2.0),
        )
        self.assertFalse(NonLinearPlace._metric_has_nonfinite_value(finite_metric))

        inf_objective_metric = types.SimpleNamespace(
            objective=torch.tensor(float("inf")),
            hpwl=torch.tensor(2.0),
        )
        self.assertTrue(
            NonLinearPlace._metric_has_nonfinite_value(inf_objective_metric)
        )

        nan_hpwl_metric = types.SimpleNamespace(objective=1.0, hpwl=math.nan)
        self.assertTrue(NonLinearPlace._metric_has_nonfinite_value(nan_hpwl_metric))

        nan_goverflow_metric = types.SimpleNamespace(
            objective=torch.tensor(1.0),
            hpwl=torch.tensor(2.0),
            goverflow=torch.tensor(float("nan")),
        )
        self.assertTrue(
            NonLinearPlace._metric_has_nonfinite_value(nan_goverflow_metric)
        )

    def test_gradient_nan_assertion_detection_is_narrow(self):
        NonLinearPlace = _load_nonlinearplace_class()

        self.assertTrue(
            NonLinearPlace._is_gradient_nan_assertion(
                AssertionError("Gradient contains NaN")
            )
        )
        self.assertFalse(
            NonLinearPlace._is_gradient_nan_assertion(
                AssertionError("different assertion")
            )
        )
        self.assertFalse(
            NonLinearPlace._is_gradient_nan_assertion(
                AssertionError("Gradient contains NaN", "extra context")
            )
        )
        self.assertFalse(
            NonLinearPlace._is_gradient_nan_assertion(
                RuntimeError("Gradient contains NaN")
            )
        )

    def test_gradient_nan_assertion_guard_only_handles_exact_message(self):
        NonLinearPlace = _load_nonlinearplace_class()

        def _guard(exc):
            try:
                raise exc
            except AssertionError as caught:
                if not NonLinearPlace._is_gradient_nan_assertion(caught):
                    raise
                return "handled"

        self.assertEqual(_guard(AssertionError("Gradient contains NaN")), "handled")

        with self.assertRaisesRegex(AssertionError, "different assertion"):
            _guard(AssertionError("different assertion"))

    def test_mark_metric_gradient_nan_failure_sets_consumable_nan_objective(self):
        NonLinearPlace = _load_nonlinearplace_class()
        metric = types.SimpleNamespace(
            hpwl=torch.tensor(2.0, dtype=torch.float64),
            objective=None,
        )

        NonLinearPlace._mark_metric_gradient_nan_failure(metric)

        self.assertIsInstance(metric.objective, torch.Tensor)
        self.assertEqual(metric.objective.dtype, torch.float64)
        self.assertTrue(torch.isnan(metric.objective).item())

    def test_copy_timing_metrics_to_eval_metric_skips_none_snapshot_values(self):
        NonLinearPlace = _load_nonlinearplace_class()
        model = types.SimpleNamespace(
            timing_metric_snapshot=lambda: {
                "wns": -0.1,
                "tns": None,
                "ws": -0.2,
                "ts": None,
                "max_slew_violation": 0.03,
                "max_load_cap_violation": None,
            }
        )
        metric = types.SimpleNamespace(
            wns=9.0,
            tns=8.0,
            ws=7.0,
            ts=6.0,
            max_slew_violation=5.0,
            max_load_cap_violation=4.0,
        )

        NonLinearPlace._copy_timing_metrics_to_eval_metric(model, metric)

        self.assertEqual(metric.wns, -0.1)
        self.assertEqual(metric.tns, 8.0)
        self.assertEqual(metric.ws, -0.2)
        self.assertEqual(metric.ts, 6.0)
        self.assertEqual(metric.max_slew_violation, 0.03)
        self.assertEqual(metric.max_load_cap_violation, 4.0)

    def test_refresh_timing_metrics_updates_model_fields_from_return_tuple(self):
        NonLinearPlace = _load_nonlinearplace_class()

        class FakeModel:
            def __init__(self):
                self.seen_positions = []

            def timing_obj(self, pos):
                self.seen_positions.append(pos)
                return (
                    torch.tensor(-0.11),
                    torch.tensor(-1.2),
                    torch.tensor(-0.09),
                    torch.tensor(0.0),
                )

        model = FakeModel()
        pos = torch.tensor([1.0, 2.0])

        refreshed = NonLinearPlace._refresh_timing_metrics(model, pos)

        self.assertTrue(refreshed)
        self.assertEqual(model.seen_positions, [pos])
        self.assertAlmostEqual(model.wns.item(), -0.11, places=6)
        self.assertAlmostEqual(model.tns.item(), -1.2, places=6)
        self.assertAlmostEqual(model.ws.item(), -0.09, places=6)
        self.assertAlmostEqual(model.ts.item(), 0.0, places=6)

    def test_should_skip_post_global_placement_for_nan_and_stop_reason(self):
        NonLinearPlace = _load_nonlinearplace_class()
        finite_metric = types.SimpleNamespace(
            objective=torch.tensor(1.0),
            hpwl=torch.tensor(2.0),
        )
        nonfinite_metric = types.SimpleNamespace(
            objective=torch.tensor(float("nan")),
            hpwl=torch.tensor(2.0),
        )

        self.assertTrue(
            NonLinearPlace._should_skip_post_global_placement(
                "gradient_nan", finite_metric
            )
        )
        self.assertTrue(
            NonLinearPlace._should_skip_post_global_placement(
                None, nonfinite_metric
            )
        )
        self.assertFalse(
            NonLinearPlace._should_skip_post_global_placement(None, finite_metric)
        )

    def test_should_skip_post_global_placement_respects_stop_reason_and_metric_state(self):
        NonLinearPlace = _load_nonlinearplace_class()
        finite_metric = types.SimpleNamespace(
            objective=torch.tensor(1.0),
            hpwl=torch.tensor(2.0),
            overflow=torch.tensor(0.1),
        )
        nonfinite_metric = types.SimpleNamespace(
            objective=torch.tensor(float("nan")),
            hpwl=torch.tensor(2.0),
            overflow=torch.tensor(0.1),
        )

        self.assertTrue(
            NonLinearPlace._should_skip_post_global_placement(
                "gradient_nan", finite_metric
            )
        )
        self.assertTrue(
            NonLinearPlace._should_skip_post_global_placement(
                "nonfinite_metric", finite_metric
            )
        )
        self.assertTrue(
            NonLinearPlace._should_skip_post_global_placement(
                None, nonfinite_metric
            )
        )
        self.assertFalse(
            NonLinearPlace._should_skip_post_global_placement(None, finite_metric)
        )

    def test_last_metric_from_metrics_handles_zero_iteration_empty_metrics(self):
        NonLinearPlace = _load_nonlinearplace_class()
        metric = types.SimpleNamespace(iteration=7)

        self.assertIsNone(NonLinearPlace._last_metric_from_metrics([]))
        self.assertIsNone(NonLinearPlace._last_metric_from_metrics([[]]))
        self.assertIs(
            NonLinearPlace._last_metric_from_metrics([[[metric]]]),
            metric,
        )


if __name__ == "__main__":
    unittest.main()
