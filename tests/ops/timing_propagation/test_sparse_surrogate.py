"""Mixed clock-LUT and combinational-surrogate propagation at one level."""

import pytest
import torch
from dreamplace.ops.timing_propagation.cell_aat_native import _CellAat, _tables
from dreamplace.ops.timing_propagation.sparse_runtime import SparseTimingRuntime
from sparse_fixtures import production_timing


@pytest.mark.parametrize("need_delays", [False, True])
def test_mixed_level_preserves_lanes_and_electrical_gradients(native_device, need_delays):
    timing = production_timing(native_device)
    timing.flat_inst_arcs_by_level = timing.flat_inst_arcs_by_level.repeat(2, 1)
    timing.flat_inst_arcs_by_level_start[1:] = torch.tensor([1, 2], device=native_device)
    timing.sparse_timing_view.arrays["cell_edges"][1, 0] = 1
    timing.pin2node_map = torch.arange(4, device=native_device)
    timing.inst_main_id = torch.zeros(4, dtype=torch.long, device=native_device)
    timing.size_var_getter = lambda: torch.ones(4, dtype=torch.float64, device=native_device)
    timing.vt_var_getter = lambda: torch.ones((4, 1), dtype=torch.float64, device=native_device)
    timing.cell_modeling_op = lambda main, arc, vt, size, slew, cap, kind: (
        2.0 + 0.5 * slew + 8.0 * cap + 0.1 * kind.to(slew.dtype)
    )
    runtime = SparseTimingRuntime(timing)
    runtime._bind_masters()
    runtime._tables = _tables(timing)
    runtime._need_delays = need_delays
    runtime._surrogate_flags = False, True
    level = next(level for level in runtime.schedule.levels if level.cell.numel())
    values = tuple(
        torch.linspace(0.1, 0.4, len(level.pins), dtype=torch.float64, device=native_device)
        .clone()
        .requires_grad_()
        for _ in range(6)
    )
    actual = runtime._cell_updates(level, values[:4], values[4:])

    arcs = runtime._cell_arcs[id(level)]
    ra, fa, rs, fs, rc, fc = values
    source, sink = arcs[1, :2]
    delay_r = 2.1 + 0.5 * rs[source] + 8.0 * rc[sink]
    delay_f = 2.0 + 0.5 * fs[source] + 8.0 * fc[sink]
    surrogate = torch.stack(
        (
            ra[source] + delay_r,
            fa[source] + delay_f,
            delay_r + 0.2,
            delay_f + 0.2,
        )
    )
    if need_delays:
        surrogate = torch.cat(
            (
                surrogate,
                torch.stack(
                    (
                        delay_r,
                        delay_r.new_tensor(-torch.inf),
                        delay_f.new_tensor(-torch.inf),
                        delay_f,
                    )
                ),
            )
        )
    native = _CellAat.apply(arcs[:1], *values, runtime._tables, False, 10.0)  # noqa: FBT003 -- Function.apply takes positional arguments
    expected = torch.cat((native[:, : surrogate.numel()], surrogate.unsqueeze(0)))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(actual[:, :4].sum(), values, retain_graph=True)
    expected_gradient = torch.autograd.grad(expected[:, :4].sum(), values)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
