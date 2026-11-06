from types import SimpleNamespace

import torch

from dreamplace.ops.timing_propagation.action_cell_aat import (
    apply_action_cell_aat,
)
from dreamplace.ops.timing_propagation.cell_aat_native import _CellAat
from dreamplace.ops.timing_propagation.timing_propagation import (
    scatter_timing_max,
)


def _tables(dtype=torch.float64):
    def one(values):
        values = torch.tensor(values, dtype=dtype)
        return (
            values,
            torch.tensor([[1.0, 2.0]] * 2, dtype=dtype),
            torch.tensor([[1.0, 2.0]] * 2, dtype=dtype),
            torch.tensor([2, 2], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        )

    # Four LUTs, each with two rows.  Row 1 is a deliberately larger delay so
    # the action delta is easy to distinguish from the current query.
    base = one([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]])
    return [tensor for _ in range(4) for tensor in base]


def _query_inputs():
    return (
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )


def test_beta_zero_is_bitwise_equal_to_fixed_master_query():
    arcs = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long)
    pins = _query_inputs()
    tables = _tables()
    expected = _CellAat.apply(arcs, *pins, tables, False, 0.5)
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)

    actual = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.ones((1, 4), dtype=torch.long),
        valid_arcs=torch.ones((1, 4), dtype=torch.bool),
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_beta_gradient_is_the_target_minus_current_loss_delta():
    arcs = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long)
    pins = _query_inputs()
    tables = _tables()
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, False, True, False]])

    updates = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.ones((1, 4), dtype=torch.long),
        valid_arcs=valid,
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
    )
    loss = updates[:, 0].sum()
    grad = torch.autograd.grad(loss, beta)[0]

    assert torch.isfinite(grad).all()
    assert grad[0, 0].item() > 0.0
    assert grad[0, 1].item() == 0.0
    assert grad[0, 2].item() > 0.0
    assert grad[0, 3].item() == 0.0


def test_all_invalid_actions_still_return_a_finite_connected_loss():
    arcs = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long)
    pins = _query_inputs()
    tables = _tables()
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)

    updates = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.ones((1, 4), dtype=torch.long),
        valid_arcs=torch.zeros((1, 4), dtype=torch.bool),
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
    )
    loss = updates[:, :4].sum()
    grad = torch.autograd.grad(loss, beta, allow_unused=True)[0]

    assert torch.isfinite(loss)
    assert grad is not None
    assert torch.equal(grad, torch.zeros_like(beta))


def test_context_probe_uses_values_only_current_with_bitwise_parity():
    arcs = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long)
    pins = _query_inputs()
    tables = _tables()
    expected = _CellAat.apply(arcs, *pins, tables, False, 0.5)
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    context = SimpleNamespace(
        native_values_used=False,
        action_cell_aat_calls=0,
    )

    actual = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.ones((1, 4), dtype=torch.long),
        valid_arcs=torch.ones((1, 4), dtype=torch.bool),
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
        context=context,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert context.action_cell_aat_calls == 1
    assert context.native_values_used


def test_multi_arc_instance_gradient_aggregates_all_arcs():
    arcs = torch.tensor(
        [[0, 1, 0, 0, 1], [1, 2, 0, 0, 1]],
        dtype=torch.long,
    )
    pins = (
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
    )
    tables = _tables()
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    context = SimpleNamespace(
        native_values_used=False,
        action_cell_aat_calls=0,
    )

    updates = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((2, 4), dtype=torch.long),
        target_senses=torch.ones((2, 4), dtype=torch.long),
        valid_arcs=torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        ),
        inst_ids=torch.zeros(2, dtype=torch.long),
        beta=beta,
        context=context,
    )
    loss = updates[:, 0].sum()
    grad = torch.autograd.grad(loss, beta)[0]

    # Each arc's target table row 1 is 10 larger than row 0.
    assert grad[0, 0].item() == 20.0
    assert grad[0, 1].item() == 0.0


def test_native_action_path_matches_python_fallback():
    arcs = torch.tensor(
        [[0, 1, 0, 0, 1], [1, 2, 0, 0, 0]],
        dtype=torch.long,
    )
    pins = (
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64),
        torch.tensor([1.0, 1.2, 0.8], dtype=torch.float64),
        torch.tensor([1.1, 0.9, 1.0], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
    )
    tables = _tables()
    target_rows = torch.tensor(
        [[1, 0, 1, 0], [1, 1, 0, 0]],
        dtype=torch.long,
    )
    target_senses = torch.tensor(
        [[1, 0, -1, 1], [0, 1, 1, -1]],
        dtype=torch.long,
    )
    valid = torch.tensor(
        [[True, True, False, True], [True, False, True, False]]
    )
    beta_values = torch.tensor(
        [[0.3, -0.2, 0.1, 0.05], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    def run(use_native, smooth):
        beta = beta_values.clone().requires_grad_(True)
        context = SimpleNamespace(
            native_values_used=False,
            native_action_values_used=False,
            action_cell_aat_calls=0,
        )
        updates = apply_action_cell_aat(
            arcs,
            pins,
            tables,
            smooth,
            0.5,
            target_arc_ids=target_rows,
            target_senses=target_senses,
            valid_arcs=valid,
            inst_ids=torch.tensor([0, 1], dtype=torch.long),
            beta=beta,
            context=context,
            use_native_action=use_native,
        )
        grad = torch.autograd.grad(updates[:, :4].sum(), beta)[0]
        return updates.detach(), grad.detach(), context

    for smooth in (False, True):
        native_updates, native_grad, native_context = run(True, smooth)
        python_updates, python_grad, python_context = run(False, smooth)
        assert native_context.native_action_values_used
        assert not python_context.native_action_values_used
        torch.testing.assert_close(
            native_updates, python_updates, rtol=0.0, atol=1e-12
        )
        torch.testing.assert_close(native_grad, python_grad, rtol=0.0, atol=1e-12)


def test_native_action_values_without_delay_lanes_match_slew_and_arrival():
    import torch as _torch

    from dreamplace.ops.timing_propagation import lut_entry_2d_op

    native = lut_entry_2d_op._select_module(_torch.zeros(1))
    if not hasattr(native, "cell_aat_action_values"):
        return
    arcs = torch.tensor(
        [[0, 1, 0, 0, 1], [1, 2, 0, 0, 0]],
        dtype=torch.long,
    )
    pins = (
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64),
        torch.tensor([1.0, 1.2, 0.8], dtype=torch.float64),
        torch.tensor([1.1, 0.9, 1.0], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
    )
    tables = _tables()
    target_rows = torch.tensor(
        [[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.long
    )
    target_senses = torch.tensor(
        [[1, 0, -1, 1], [0, 1, 1, -1]], dtype=torch.long
    )
    valid = torch.ones((2, 4), dtype=torch.bool)
    full = native.cell_aat_action_values(
        arcs, target_rows, target_senses, valid, *pins, tables, True, False, 0.5
    )
    light = native.cell_aat_action_values(
        arcs, target_rows, target_senses, valid, *pins, tables, False, False, 0.5
    )
    torch.testing.assert_close(light[0][:, :4], full[0][:, :4])
    assert torch.equal(light[0][:, 4:], torch.zeros_like(light[0][:, 4:]))
    assert torch.equal(light[1][..., 4:], torch.zeros_like(light[1][..., 4:]))
    torch.testing.assert_close(light[1][..., :4], full[1][..., :4])


def test_timing_obj_clears_action_context_on_exception():
    import pytest

    from dreamplace.PlaceObj import PlaceObj

    timing_op = SimpleNamespace(
        _active_discrete_action_context="stale",
        _sparse_runtime=SimpleNamespace(
            timing=SimpleNamespace(_active_discrete_action_context="stale")
        ),
    )

    class Dummy:
        _active_discrete_action_context = "stale"
        op_collections = SimpleNamespace(timing_propagation_op=timing_op)

        def _timing_obj_impl(self, *args, **kwargs):
            raise RuntimeError("boom")

    dummy = Dummy()
    with pytest.raises(RuntimeError, match="boom"):
        PlaceObj.timing_obj(dummy, torch.zeros(1), discrete_action_context=object())

    assert dummy._active_discrete_action_context is None
    assert timing_op._active_discrete_action_context is None
    assert timing_op._sparse_runtime.timing._active_discrete_action_context is None


def test_deferred_action_backward_matches_eager_action_values():
    arcs = torch.tensor(
        [[0, 1, 0, 0, 1], [1, 2, 0, 0, 0]],
        dtype=torch.long,
    )
    pins = (
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64),
        torch.tensor([1.0, 1.2, 0.8], dtype=torch.float64),
        torch.tensor([1.1, 0.9, 1.0], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
        torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64),
    )
    tables = _tables()
    target_rows = torch.tensor(
        [[1, 0, 1, 0], [1, 1, 0, 0]],
        dtype=torch.long,
    )
    target_senses = torch.tensor(
        [[1, 0, -1, 1], [0, 1, 1, -1]],
        dtype=torch.long,
    )
    valid = torch.tensor(
        [[True, True, False, True], [True, False, True, False]]
    )
    beta_values = torch.zeros((2, 4), dtype=torch.float64)

    def run(defer):
        beta = beta_values.clone().requires_grad_(True)
        context = SimpleNamespace(
            native_values_used=False,
            native_action_values_used=False,
            native_action_backward_used=False,
            action_cell_aat_calls=0,
            valid_arcs=valid.clone(),
        )
        updates = apply_action_cell_aat(
            arcs,
            pins,
            tables,
            True,
            0.5,
            target_arc_ids=target_rows,
            target_senses=target_senses,
            valid_arcs=valid,
            inst_ids=torch.tensor([0, 1], dtype=torch.long),
            beta=beta,
            context=context,
            global_rows=torch.arange(2, dtype=torch.long),
            defer_target_queries=defer,
        )
        grad = torch.autograd.grad(updates[:, :4].sum(), beta)[0]
        return updates.detach(), grad.detach(), context

    eager_updates, eager_grad, eager_context = run(False)
    deferred_updates, deferred_grad, deferred_context = run(True)
    assert eager_context.native_action_values_used
    assert deferred_context.native_action_backward_used
    torch.testing.assert_close(
        deferred_updates, eager_updates, rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        deferred_grad, eager_grad, rtol=0.0, atol=1e-12
    )
    assert torch.equal(deferred_context.valid_arcs, valid)


def test_base_point_action_delta_uses_one_sided_target_difference():
    arcs = torch.tensor([[0, 1, 0, 0, 1]], dtype=torch.long)
    tables = _tables()
    pins = _query_inputs()
    beta = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    context = SimpleNamespace(
        native_values_used=False,
        native_action_values_used=False,
        action_cell_aat_calls=0,
    )

    updates = apply_action_cell_aat(
        arcs,
        pins,
        tables,
        False,
        0.5,
        target_arc_ids=torch.ones((1, 4), dtype=torch.long),
        target_senses=torch.ones((1, 4), dtype=torch.long),
        valid_arcs=torch.ones((1, 4), dtype=torch.bool),
        inst_ids=torch.zeros(1, dtype=torch.long),
        beta=beta,
        context=context,
    )
    loss = updates[:, 0].sum()
    grad = torch.autograd.grad(loss, beta)[0]

    # Row 1 has arrival 11 instead of 1, so the base-point delta is +10.
    torch.testing.assert_close(
        grad,
        torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_avesmooth_scatter_max_averages_only_near_max_gradients():
    dest = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    src = torch.tensor(
        [10.0, 9.5, 1.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.zeros(3, dtype=torch.long)

    value = scatter_timing_max(
        dest,
        index,
        src,
        include_self=False,
        mode="avesmooth",
        tau_ps=1.0,
    )
    value.sum().backward()

    assert value.item() == 10.0
    assert dest.grad.item() == 0.0
    torch.testing.assert_close(
        src.grad,
        torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64),
    )


def test_avesmooth_scatter_max_includes_near_max_self_value():
    dest = torch.tensor([9.8], dtype=torch.float64, requires_grad=True)
    src = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    index = torch.zeros(1, dtype=torch.long)

    value = scatter_timing_max(
        dest,
        index,
        src,
        include_self=True,
        mode="avesmooth",
        tau_ps=1.0,
    )
    value.sum().backward()

    assert value.item() == 10.0
    assert src.grad.item() == 0.5
    assert dest.grad.item() == 0.5
