import os
import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dreamplace.ops.cell_modeling import cell_modeling as cell_modeling_module
from dreamplace.ops.cell_modeling.cell_modeling import CellModeling


def _lut2d_python(input_slew, out_cap, trans_tables, cap_tables, values, trans_dims, cap_dims, arc_idx):
    t = trans_tables[arc_idx]
    c = cap_tables[arc_idx]
    v = values[arc_idx].reshape(-1)
    td = int(trans_dims[arc_idx].item())
    cd = int(cap_dims[arc_idx].item())
    if td < 2 or cd < 2:
        return input_slew.new_tensor(0.0)
    x = torch.minimum(torch.maximum(input_slew, t[0]), t[td - 1])
    y = torch.minimum(torch.maximum(out_cap, c[0]), c[cd - 1])
    ti = torch.searchsorted(t[:td], x, right=True).clamp(min=1, max=td - 1)
    ci = torch.searchsorted(c[:cd], y, right=True).clamp(min=1, max=cd - 1)
    tl = ti - 1
    cl = ci - 1
    t0 = t[tl]
    t1 = t[ti]
    c0 = c[cl]
    c1 = c[ci]
    idx00 = tl * cd + cl
    idx01 = tl * cd + ci
    idx10 = ti * cd + cl
    idx11 = ti * cd + ci
    v00 = v[idx00]
    v01 = v[idx01]
    v10 = v[idx10]
    v11 = v[idx11]
    tii = t1 - t0
    cii = c1 - c0
    if torch.abs(tii) < 1e-12 and torch.abs(cii) < 1e-12:
        return v00
    if torch.abs(tii) < 1e-12:
        return torch.lerp(v00, v01, (y - c0) / cii)
    if torch.abs(cii) < 1e-12:
        return torch.lerp(v00, v10, (x - t0) / tii)
    return (
        v00 * (t1 - x) * (c1 - y)
        + v01 * (t1 - x) * (y - c0)
        + v10 * (x - t0) * (c1 - y)
        + v11 * (x - t0) * (y - c0)
    ) / (tii * cii)


def _piecewise_python(size_table, arc_table, size_count, trans_tables, cap_tables, values, trans_dims, cap_dims, size, slew, cap):
    out = []
    for i in range(size.numel()):
        count = int(size_count[i].item())
        if count <= 0:
            out.append(size.new_tensor(0.0))
            continue
        if count <= 1:
            arc = int(arc_table[i, 0].item())
            out.append(_lut2d_python(slew[i], cap[i], trans_tables, cap_tables, values, trans_dims, cap_dims, arc))
            continue
        sizes = size_table[i]
        max_idx = count - 1
        s_clamped = torch.minimum(torch.maximum(size[i], sizes[0]), sizes[max_idx])
        hi = torch.searchsorted(sizes[:count], s_clamped, right=True).clamp(min=1, max=max_idx)
        lo = hi - 1
        s_low = sizes[lo]
        s_high = sizes[hi]
        arc_low = int(arc_table[i, lo].item())
        arc_high = int(arc_table[i, hi].item())
        v_low = _lut2d_python(slew[i], cap[i], trans_tables, cap_tables, values, trans_dims, cap_dims, arc_low)
        if arc_low == arc_high:
            v_high = v_low
        else:
            v_high = _lut2d_python(slew[i], cap[i], trans_tables, cap_tables, values, trans_dims, cap_dims, arc_high)
        denom = s_high - s_low
        if (not torch.isfinite(denom)) or torch.abs(denom) < 1e-12:
            out.append(v_low)
        else:
            out.append(torch.lerp(v_low, v_high, (s_clamped - s_low) / denom))
    return torch.stack(out)


class CellModelingPiecewiseSizeOpTest(unittest.TestCase):
    def test_piecewise_backend_profile_records_cpu_forward_backward(self):
        from dreamplace.ops.cell_modeling import cell_modeling_op

        cell_modeling_op.reset_piecewise_size_backend_profile()
        self.addCleanup(cell_modeling_op.reset_piecewise_size_backend_profile)

        dtype = torch.float32
        device = torch.device("cpu")
        size_table = torch.tensor([[1.0, 2.0]], dtype=dtype, device=device)
        arc_table = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        size_count = torch.tensor([2], dtype=torch.long, device=device)
        trans_tables = torch.tensor(
            [[0.1, 0.3], [0.1, 0.3]], dtype=dtype, device=device
        )
        cap_tables = torch.tensor(
            [[0.2, 0.5], [0.2, 0.5]], dtype=dtype, device=device
        )
        values = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        trans_dims = torch.tensor([2, 2], dtype=torch.long, device=device)
        cap_dims = torch.tensor([2, 2], dtype=torch.long, device=device)
        size = torch.tensor([1.5], dtype=dtype, device=device, requires_grad=True)
        slew = torch.tensor([0.2], dtype=dtype, device=device, requires_grad=True)
        cap = torch.tensor([0.3], dtype=dtype, device=device, requires_grad=True)

        out = cell_modeling_op.piecewise_size_forward(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
            size,
            slew,
            cap,
        )
        out.sum().backward()

        profile = cell_modeling_op.get_piecewise_size_backend_profile()
        self.assertEqual(profile["piecewise_size_forward_backend"], "cpp")
        self.assertEqual(profile["piecewise_size_backward_backend"], "cpp")
        self.assertEqual(profile["piecewise_size_forward_calls"], 1)
        self.assertEqual(profile["piecewise_size_backward_calls"], 1)
        self.assertEqual(profile["piecewise_size_native_op"], "default_on")

    def _run_case(self, device):
        from dreamplace.ops.cell_modeling import cell_modeling_op

        dtype = torch.float32
        size_table = torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, float("inf"), float("inf")], [1.0, 3.0, float("inf")]],
            dtype=dtype,
            device=device,
        )
        arc_table = torch.tensor([[0, 1, 2], [3, -1, -1], [1, 2, -1]], dtype=torch.long, device=device)
        size_count = torch.tensor([3, 1, 2], dtype=torch.long, device=device)
        trans_tables = torch.tensor(
            [[0.1, 0.3, 1.0], [0.1, 0.4, 1.0], [0.2, 0.6, 1.2], [0.1, 0.5, 1.5]],
            dtype=dtype,
            device=device,
        )
        cap_tables = torch.tensor(
            [[0.2, 0.5, 1.0], [0.2, 0.6, 1.2], [0.1, 0.7, 1.4], [0.3, 0.8, 1.6]],
            dtype=dtype,
            device=device,
        )
        values = torch.tensor(
            [
                [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0],
                [0.5, 1.0, 2.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0],
                [2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0],
                [1.2, 1.4, 1.6, 2.2, 2.4, 2.6, 3.2, 3.4, 3.6],
            ],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        trans_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)
        cap_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)
        size = torch.tensor([1.5, 0.7, 3.5], dtype=dtype, device=device, requires_grad=True)
        slew = torch.tensor([0.15, 0.4, 0.9], dtype=dtype, device=device, requires_grad=True)
        cap = torch.tensor([0.25, 0.9, 1.1], dtype=dtype, device=device, requires_grad=True)

        actual = cell_modeling_op.piecewise_size_forward(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
            size,
            slew,
            cap,
        )
        expected = _piecewise_python(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
            size,
            slew,
            cap,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        grad = torch.tensor([1.0, -0.5, 0.25], dtype=dtype, device=device)
        actual_grads = torch.autograd.grad(actual, (size, slew, cap, values), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (size, slew, cap, values), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_cpu_forward_backward_matches_python(self):
        self._run_case(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_matches_python(self):
        self._run_case(torch.device("cuda"))


class CellModelingPiecewiseSizeIntegrationTest(unittest.TestCase):
    def test_lut_runtime_cache_skips_coeff_table_when_gate_disabled(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        luts = mock.Mock()
        luts.flat_luts_values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        luts.flat_luts_trans_table = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        luts.flat_luts_cap_table = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        luts.flat_luts_dim = torch.tensor([[2, 2]], dtype=torch.long)
        model.data_collections = mock.Mock()
        model.data_collections.arcs_info.f_delay_luts = luts

        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_COEFF_OP": "0"}), mock.patch.object(
            cell_modeling_module.lut_entry_2d_op,
            "build_2d_coefficients",
            wraps=cell_modeling_module.lut_entry_2d_op.build_2d_coefficients,
        ) as build_coeff:
            lookup_state = model._ensure_lut_runtime_lookup_cache("f_delay")

        build_coeff.assert_not_called()
        self.assertIsNone(lookup_state["coeff_table"])

    def _build_piecewise_interp_model_and_state(self, device):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = device
        model._lut_runtime_lookup_cache = {
            "f_delay": {
                "trans_dims": torch.tensor([3, 3, 3], dtype=torch.long, device=device),
                "cap_dims": torch.tensor([3, 3, 3], dtype=torch.long, device=device),
                "values_3d": torch.tensor(
                    [
                        [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [4.0, 4.5, 5.0]],
                        [[0.5, 1.0, 2.0], [1.5, 2.0, 3.0], [3.5, 4.0, 5.0]],
                        [[2.0, 2.5, 3.0], [3.0, 3.5, 4.0], [4.0, 4.5, 5.0]],
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                "trans_tables": torch.tensor(
                    [[0.1, 0.3, 1.0], [0.1, 0.4, 1.0], [0.2, 0.6, 1.2]],
                    dtype=torch.float32,
                    device=device,
                ),
                "cap_tables": torch.tensor(
                    [[0.2, 0.5, 1.0], [0.2, 0.6, 1.2], [0.1, 0.7, 1.4]],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        }
        state = {
            "size_table": torch.tensor(
                [[[[1.0, 2.0, 4.0]]], [[[1.0, 3.0, float("inf")]]]],
                dtype=torch.float32,
                device=device,
            ),
            "arc_index_table": torch.tensor(
                [[[[0, 1, 2]]], [[[1, 2, -1]]]], dtype=torch.long, device=device
            ),
            "size_count": torch.tensor([[[3]], [[2]]], dtype=torch.long, device=device),
        }
        return model, state

    def test_piecewise_size_interp_uses_fused_op_by_default(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model._lut_runtime_lookup_cache = {
            "f_delay": {
                "trans_dims": torch.tensor([3, 3], dtype=torch.long),
                "cap_dims": torch.tensor([3, 3], dtype=torch.long),
                "values_3d": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
                "trans_tables": torch.tensor(
                    [[0.1, 0.3, 1.0], [0.2, 0.4, 1.2]], dtype=torch.float32
                ),
                "cap_tables": torch.tensor(
                    [[0.2, 0.5, 1.0], [0.1, 0.6, 1.4]], dtype=torch.float32
                ),
            }
        }
        state = {
            "size_table": torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32),
            "arc_index_table": torch.tensor([[[[0, 1]]]], dtype=torch.long),
            "size_count": torch.tensor([[[2]]], dtype=torch.long),
        }
        sentinel = torch.tensor([123.0], dtype=torch.float32)

        with mock.patch.object(
            cell_modeling_module.cell_modeling_op,
            "piecewise_size_forward",
            return_value=sentinel,
        ) as fused:
            result = model._piecewise_size_interp(
                "f_delay",
                state,
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.5], dtype=torch.float32),
                torch.tensor([0.2], dtype=torch.float32),
                torch.tensor([0.3], dtype=torch.float32),
            )

        self.assertIs(result, sentinel)
        fused.assert_called_once()

    def test_piecewise_size_interp_caches_dataset_all_2d_fast_path(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.reset_forward_profile(enabled=True)
        model._lut_runtime_lookup_cache = {
            "f_delay": {
                "trans_dims": torch.tensor([3, 3], dtype=torch.long),
                "cap_dims": torch.tensor([3, 3], dtype=torch.long),
                "values_3d": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
                "trans_tables": torch.tensor(
                    [[0.1, 0.3, 1.0], [0.2, 0.4, 1.2]], dtype=torch.float32
                ),
                "cap_tables": torch.tensor(
                    [[0.2, 0.5, 1.0], [0.1, 0.6, 1.4]], dtype=torch.float32
                ),
            }
        }
        state = {
            "size_table": torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32),
            "arc_index_table": torch.tensor([[[[0, 1]]]], dtype=torch.long),
            "size_count": torch.tensor([[[2]]], dtype=torch.long),
        }
        sentinel = torch.tensor([123.0], dtype=torch.float32)

        with mock.patch.object(
            cell_modeling_module.cell_modeling_op,
            "piecewise_size_forward",
            return_value=sentinel,
        ) as fused:
            for _ in range(2):
                result = model._piecewise_size_interp(
                    "f_delay",
                    state,
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([1.5], dtype=torch.float32),
                    torch.tensor([0.2], dtype=torch.float32),
                    torch.tensor([0.3], dtype=torch.float32),
                )
                self.assertIs(result, sentinel)

        self.assertTrue(state["_all_luts_2d_only"])
        self.assertEqual(fused.call_count, 2)
        profile = model.get_forward_profile()
        self.assertEqual(profile["piecewise_size_interp_calls"], 2)
        self.assertEqual(profile["piecewise_size_interp_fused_calls"], 2)
        self.assertEqual(profile["piecewise_size_interp_python_calls"], 0)
        self.assertEqual(profile["piecewise_size_interp_calls_by_dataset"], {"f_delay": 2})
        self.assertEqual(profile["piecewise_size_interp_arcs"], 2)
        self.assertEqual(profile["piecewise_size_interp_fused_arcs"], 2)

    def test_forward_profile_reset_and_dataset_counters(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.reset_forward_profile(enabled=True)

        model._profile_increment("cell_model_forward_calls")
        model._profile_increment("cell_model_forward_total_ms", 2.5)
        model._profile_increment("cell_model_forward_total_arcs", 10)
        model._profile_increment("cell_model_forward_dataset_calls")
        model._profile_increment_dataset(
            "cell_model_forward_dataset_calls_by_dataset",
            "r_delay",
        )

        profile = model.get_forward_profile()
        self.assertEqual(profile["cell_model_forward_calls"], 1)
        self.assertEqual(profile["cell_model_forward_total_arcs"], 10)
        self.assertEqual(profile["cell_model_forward_dataset_calls"], 1)
        self.assertEqual(profile["cell_model_forward_dataset_calls_by_dataset"], {"r_delay": 1})
        self.assertAlmostEqual(profile["cell_model_forward_avg_ms"], 2.5)
        self.assertAlmostEqual(profile["cell_model_forward_avg_us_per_arc"], 250.0)

    def _run_piecewise_size_interp_matches_python_path(self, device):
        model, state = self._build_piecewise_interp_model_and_state(device)
        mt = torch.tensor([0, 1], dtype=torch.long, device=device)
        ao = torch.tensor([0, 0], dtype=torch.long, device=device)
        vt_slot = torch.tensor([0, 0], dtype=torch.long, device=device)
        size = torch.tensor([1.5, 2.5], dtype=torch.float32, device=device, requires_grad=True)
        slew = torch.tensor([0.15, 0.9], dtype=torch.float32, device=device, requires_grad=True)
        cap = torch.tensor([0.25, 1.1], dtype=torch.float32, device=device, requires_grad=True)

        expected = _piecewise_python(
            state["size_table"][mt, ao, vt_slot],
            state["arc_index_table"][mt, ao, vt_slot],
            state["size_count"][mt, ao, vt_slot],
            model._lut_runtime_lookup_cache["f_delay"]["trans_tables"],
            model._lut_runtime_lookup_cache["f_delay"]["cap_tables"],
            model._lut_runtime_lookup_cache["f_delay"]["values_3d"].reshape(3, -1),
            model._lut_runtime_lookup_cache["f_delay"]["trans_dims"],
            model._lut_runtime_lookup_cache["f_delay"]["cap_dims"],
            size,
            slew,
            cap,
        )
        actual = model._piecewise_size_interp("f_delay", state, mt, ao, vt_slot, size, slew, cap)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        grad = torch.tensor([1.0, -0.25], dtype=torch.float32, device=device)
        actual_grads = torch.autograd.grad(actual, (size, slew, cap), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (size, slew, cap), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_piecewise_size_interp_fused_path_matches_python_path(self):
        self._run_piecewise_size_interp_matches_python_path(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_piecewise_size_interp_fused_cuda_path_matches_python_path(self):
        self._run_piecewise_size_interp_matches_python_path(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
