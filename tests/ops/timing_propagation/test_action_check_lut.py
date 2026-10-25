from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import ACTION_NAMES
from dreamplace.ops.timing_propagation.action_check_lut import action_check_values
from dreamplace.ops.timing_propagation.discrete_actions import build_discrete_action_context
from dreamplace.ops.timing_propagation.lut_entry_indexed import lut_entry_indexed


def test_context_maps_check_arcs_and_propagation_arcs_together():
    data = SimpleNamespace(
        flat_inst_arcs_by_level=torch.tensor([[0, 2, 0, 0, 1, 0, 0, 0]]),
        endpoints_timing_check_arcs=torch.tensor([[0, 1, 0, 1, 1, 1, 0, 1]]),
        inst_cell_id=torch.tensor([0]),
        pin2node_map=torch.zeros(3, dtype=torch.long),
        cell_id_2_arc_id_start=torch.tensor([0, 2, 4]),
        cell_id_2_libpin_id_start=torch.tensor([0, 3, 6]),
        flat_libarc_info=torch.tensor(
            [
                [0, 2, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 1],
                [0, 2, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1],
            ]
        ),
        inst_is_sizeable=torch.ones(1, dtype=torch.bool),
        inst_vt_mask=torch.ones((1, 1), dtype=torch.bool),
        flat_libcell_info=torch.tensor([[0.0, 10.0, 1.0, 0.0], [1.0, 10.0, 2.0, 0.0]]),
    )
    data.endpoints_constraint_arcs = data.endpoints_timing_check_arcs.clone()
    candidates = SimpleNamespace(target_cell_ids=torch.tensor([[1, -1, -1, -1], [-1, 0, -1, -1]]))
    context = build_discrete_action_context(
        data,
        candidates,
        action_names=ACTION_NAMES,
        pin_names=("CLK", "D", "Q") * 2,
    )
    assert {
        "cell": context.target_arc_ids.tolist(),
        "check": context.check_target_arc_ids.tolist(),
        "constraint": context.constraint_target_arc_ids.tolist(),
        "valid": context.valid_instances.tolist(),
    } == {
        "cell": [[2, -1, -1, -1]],
        "check": [[3, -1, -1, -1]],
        "constraint": [[3, -1, -1, -1]],
        "valid": [[True, False, False, False]],
    }


@pytest.mark.parametrize("transition", [0, 1])
@pytest.mark.parametrize("frontend", ["sparse", "legacy"])
def test_check_master_and_data_slew_share_the_same_beta(transition, frontend):
    dtype = torch.float64
    beta = torch.zeros((1, 4), dtype=dtype, requires_grad=True)
    intercept = torch.tensor([10.0, 7.0, 14.0, 12.0, 8.0], dtype=dtype)
    slopes = torch.tensor([2.0, 3.0, 4.0, 1.0, 5.0], dtype=dtype)
    info = SimpleNamespace(
        flat_luts_values=torch.stack((intercept, intercept + slopes), dim=1),
        flat_luts_trans_table=torch.zeros((5, 1), dtype=dtype),
        flat_luts_cap_table=torch.tensor([[0.0, 1.0]] * 5, dtype=dtype),
        trans_dims_actual=torch.ones(5, dtype=torch.long),
        cap_dims_actual=torch.full((5,), 2),
    )
    fall = SimpleNamespace(**vars(info))
    fall.flat_luts_values = 1.5 * info.flat_luts_values
    timing = SimpleNamespace(
        endpoints_timing_check_arcs=torch.tensor([[0, 1, 0, 0, 1]]),
        arcs_info=SimpleNamespace(r_check_luts=info, f_check_luts=fall),
        lut_entry_vectorized=lambda cell, clock, data, rows, lut: lut_entry_indexed(
            clock, data, rows, lut
        ),
    )
    context = SimpleNamespace(
        beta=beta,
        check_inst_ids=torch.zeros(1, dtype=torch.long),
        check_target_arc_ids=torch.tensor([[1, 2, 3, 4]]),
        valid_instances=torch.ones((1, 4), dtype=torch.bool),
    )
    slew_delta = torch.tensor([[0.2, -0.1, -0.05, 0.1]], dtype=dtype)
    if frontend == "sparse":
        value = action_check_values(
            timing,
            context,
            torch.tensor([0]),
            torch.tensor([transition]),
            torch.zeros(1, dtype=dtype),
            0.4 + (slew_delta * beta).sum(1),
        )
    else:
        from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation

        context.constraint_target_arc_ids = context.check_target_arc_ids
        context.constraint_inst_ids = context.check_inst_ids
        timing._active_discrete_action_context = context
        timing.arcs_info.r_delay_luts = info
        timing.arcs_info.f_delay_luts = fall
        entry = (
            TimingPropagation.r_setup_entry if transition == 0 else TimingPropagation.f_setup_entry
        )
        value = entry(
            timing,
            torch.tensor([0]),
            torch.zeros(1, dtype=dtype),
            0.4 + (slew_delta * beta).sum(1),
            torch.tensor([0]),
        )
    gradient = torch.autograd.grad(-value.sum(), beta)[0]

    def forward(b):
        data_slew = 0.4 + (slew_delta * b).sum()
        values = intercept + slopes * data_slew
        mixed = values[0] + (b[0] * (values[1:] - values[0])).sum()
        # setup/recovery enters required time with a negative coefficient.
        return -mixed * (1.0 if transition == 0 else 1.5)

    numerical = torch.empty_like(beta)
    for a in range(4):
        step = torch.zeros_like(beta)
        step[0, a] = 1e-6
        numerical[0, a] = (forward(step) - forward(-step)) / 2e-6
    torch.testing.assert_close(gradient, numerical, atol=1e-8, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_check_values_moves_cpu_rows_to_lut_device():
    device = torch.device("cuda")
    timing = SimpleNamespace(
        endpoints_timing_check_arcs=torch.tensor(
            [[0, 1, 0, 0, 1]], device=device
        ),
        lut_entry_vectorized=lambda cell, clock, data, rows, lut: rows.to(
            dtype=data.dtype
        ) + data * 0.0,
        arcs_info=SimpleNamespace(
            r_check_luts=object(),
            f_check_luts=object(),
        ),
    )
    context = SimpleNamespace(
        beta=torch.zeros((1, 4), device=device),
        check_inst_ids=torch.zeros(1, dtype=torch.long, device=device),
        check_target_arc_ids=torch.tensor(
            [[1, 2, 3, 4]], dtype=torch.long, device=device
        ),
        valid_instances=torch.ones((1, 4), dtype=torch.bool, device=device),
    )

    result = action_check_values(
        timing,
        context,
        torch.tensor([0], dtype=torch.long),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.zeros(1, device=device),
        torch.ones(1, device=device),
    )

    assert result.device == torch.device("cuda", torch.cuda.current_device())
