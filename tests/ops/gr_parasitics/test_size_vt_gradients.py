"""GR RC chain through native size/VT cap and piecewise clock-to-Q arithmetic.

This is numerical qualification on a synthetic arc fixture. Actual Liberty
clock-arc identification and native DB continuation belong to Gate C.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from dreamplace.ops.cell_modeling import cell_modeling_op
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.size_interpolated_pin import size_interpolated_pin
from torch.nn import functional as F


def piecewise_model(root):
    # Reuse the existing metadata fixture; do not implement another LUT library.
    path = root / "tests/ops/cell_modeling_piecewise_size_op_unittest.py"
    spec = importlib.util.spec_from_file_location("piecewise_size_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model, state = (
        fixture.CellModelingPiecewiseSizeIntegrationTest()._build_piecewise_interp_model_and_state(
            torch.device("cpu")
        )
    )
    lookup = {
        k: v.double() if v.is_floating_point() else v
        for k, v in model._lut_runtime_lookup_cache["f_delay"].items()
    }
    state = {
        k: v.repeat(1, 1, 2, 1).double()
        if k == "size_table"
        else v.repeat(1, 1, 2, 1)
        if k == "arc_index_table"
        else v.repeat(1, 1, 2)
        for k, v in state.items()
    }
    state["arc_index_table"][:, :, 1] = state["arc_index_table"][:, :, 0].roll(1, -1)
    state["vt_available"] = torch.ones(state["size_count"].shape, dtype=torch.bool)
    model.vt_order_tensor = torch.tensor([0.0, 1.0], dtype=torch.float64)
    model.piecewise_state = {"r_delay": state, "r_trans": state}
    model._lut_runtime_lookup_cache = {
        "r_delay": lookup,
        "r_trans": {**lookup, "values_3d": lookup["values_3d"] * 5 + 0.5},
    }
    model.piecewise_gradient_mode = "native_piecewise"
    return model


def test_size_vt_slew_cap_clock_to_q_and_combined_fd(snapshot_inputs):
    op = GRParasiticsOp.prepare(*snapshot_inputs, dtype=torch.float64)
    model = piecewise_model(Path(__file__).resolve().parents[3])
    cell_modeling_op.reset_piecewise_size_backend_profile()
    dtype = torch.float64
    # Two size masters per VT. Root is the FF output and has zero local pin C.
    sizes = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]] * 3, dtype=dtype)
    pin_ids = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    dims = torch.full((3, 2), 2, dtype=torch.int32)
    caps = torch.tensor([0.01, 0.015, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    limit_slew = torch.tensor([2.0, 2.5, 1.8, 2.3, 2.0, 2.5, 1.8, 2.3], dtype=dtype)
    limit_cap = torch.tensor([0.025, 0.03, 0.022, 0.027, 0.025, 0.03, 0.022, 0.027], dtype=dtype)
    properties = torch.stack([caps, caps * 1.2, caps * 0.8, limit_slew, limit_cap])
    current = torch.zeros((5, 3), dtype=dtype)
    params = torch.tensor([1.35, 1.45, 1.25, 0.2, 0.3, 0.25], dtype=dtype, requires_grad=True)
    zero_id = torch.tensor([0], dtype=torch.long)
    clock_slew = torch.tensor([0.23], dtype=dtype)

    def evaluate(values):
        s, vt = values[:3], values[3:]
        probabilities = torch.stack([1 - vt, vt], dim=1)
        live = size_interpolated_pin(s, probabilities, sizes, pin_ids, dims, properties, current)
        rc = op(live[0], live[1], live[2])
        ff_args = (zero_id, zero_id, vt[2:3], s[2:3], clock_slew, rc[1]["rise"][2:3])
        clk_q = model._piecewise_forward_dataset("r_delay", *ff_args).sum()
        output_slew = model._piecewise_forward_dataset("r_trans", *ff_args).sum()
        sink_slew = torch.sqrt(output_slew.square() + rc[5]["rise"][:2])
        slew_loss = F.relu(sink_slew - live[3, :2]).sum()
        cap_loss = F.relu(rc[1]["rise"][2] - live[4, 2])
        timing = F.relu(clk_q + rc[2]["rise"][1] - 0.3)
        return {
            "delay": rc[2]["rise"][1],
            "impulse": rc[5]["rise"][1],
            "clk_q": clk_q,
            "slew": slew_loss,
            "cap": cap_loss,
            "combined": timing + slew_loss + 1000 * cap_loss,
        }

    metrics = evaluate(params)
    assert all(torch.isfinite(value) and value > 0 for value in metrics.values())
    for name, value in metrics.items():
        gradient = torch.autograd.grad(value, params, retain_graph=True)[0]
        for index in range(len(params)):
            plus, minus = params.detach().clone(), params.detach().clone()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            fd = (evaluate(plus)[name] - evaluate(minus)[name]) / 2e-4
            assert float(gradient[index]) == pytest.approx(float(fd), rel=1e-3, abs=1e-6), (
                name,
                index,
                gradient[index],
                fd,
            )
    clock_grad = torch.autograd.grad(metrics["clk_q"], params)[0]
    assert clock_grad[2] != 0 and clock_grad[5] != 0
    profile = cell_modeling_op.get_piecewise_size_backend_profile()
    assert profile["piecewise_size_forward_backend"] == "cpp"
    assert profile["piecewise_size_backward_backend"] == "cpp"
