"""Complete/incomplete optimizer history conversion at the metric owner."""
from types import SimpleNamespace

import torch

from dreamplace.EvalMetrics import EvalMetrics
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.flows import metric_state


def _metric(value):
    return SimpleNamespace(
        objective=None if value is None else torch.tensor(value),
        hpwl=torch.tensor(2.0),
        overflow=torch.tensor(0.3),
        max_density=torch.tensor(1.2),
    )


def test_metric_record_includes_l_shape_raw_wirelength_gradient():
    metric = EvalMetrics(1)
    metric.l_shape_grad_ratio = 0.2
    metric.l_shape_target_ratio = 0.2
    metric.l_shape_weight = 3.0e-12
    metric.l_shape_base_grad_norm = 100.0
    metric.l_shape_wirelength_grad_norm = 80.0
    metric.l_shape_grad_raw_norm = 2.0e12
    metric.l_shape_grad_norm = 2.0e12
    metric.l_shape_cap_active = True

    record = metric_state.metric_to_record(metric)

    assert record["l_shape_grad_ratio"] == 0.2
    assert record["l_shape_target_ratio"] == 0.2
    assert record["l_shape_weight"] == 3.0e-12
    assert record["l_shape_base_grad_norm"] == 100.0
    assert record["l_shape_wirelength_grad_norm"] == 80.0
    assert record["l_shape_grad_raw_norm"] == 2.0e12
    assert record["l_shape_grad_norm"] == 2.0e12
    assert record["l_shape_cap_active"] is True


def test_processed_metrics_preserve_complete_history_after_early_stop_frame():
    placer = object.__new__(NonLinearPlace)

    result = placer._processed_global_place_metrics(
        [[[_metric(1.0), _metric(2.0), _metric(None)]]]
    )

    assert result == {
        "objective": [1.0, 2.0],
        "hpwl": [2.0, 2.0],
        "overflow": [0.30000001192092896, 0.30000001192092896],
        "density": [1.2000000476837158, 1.2000000476837158],
        "incomplete_metric_count": 1,
    }


def test_incomplete_terminal_metric_is_not_misclassified_as_divergence():
    assert not NonLinearPlace._metric_objective_is_nonfinite(_metric(None))
    assert not NonLinearPlace._metric_objective_is_nonfinite(_metric(1.0))
    assert NonLinearPlace._metric_objective_is_nonfinite(_metric(float("inf")))
    assert NonLinearPlace._metric_objective_is_nonfinite(_metric(float("nan")))


def test_last_metric_recurses_into_nested_optimizer_history():
    complete = _metric(1.0)
    milestone_terminal = _metric(None)

    assert (
        NonLinearPlace._last_metric_from_metrics([[complete, milestone_terminal]])
        is milestone_terminal
    )
