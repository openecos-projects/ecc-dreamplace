import os
import sys
import unittest

import torch
from unittest import mock

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _python_lut_2d(input_trans, output_caps, trans_tables, cap_tables, values, trans_dims, cap_dims):
    from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation

    model = TimingPropagation.__new__(TimingPropagation)
    return model._lut_entry_2d_vectorized(
        input_trans,
        output_caps,
        trans_tables,
        cap_tables,
        values,
        trans_dims,
        cap_dims,
    )


class LutEntry2DOpTest(unittest.TestCase):
    def _run_forward_backward_case(self, device):
        from dreamplace.ops.timing_propagation import lut_entry_2d_op

        dtype = torch.float32
        input_trans = torch.tensor([0.05, 0.15, 0.7, 1.2], dtype=dtype, device=device, requires_grad=True)
        output_caps = torch.tensor([0.1, 0.25, 0.8, 1.3], dtype=dtype, device=device, requires_grad=True)
        trans_tables = torch.tensor(
            [[0.1, 0.3, 1.0], [0.0, 0.4, 1.0], [0.2, 0.6, 1.2], [0.2, 0.6, 1.2]],
            dtype=dtype,
            device=device,
        )
        cap_tables = torch.tensor(
            [[0.2, 0.5, 1.0], [0.1, 0.7, 1.4], [0.2, 0.6, 1.0], [0.2, 0.6, 1.0]],
            dtype=dtype,
            device=device,
        )
        values = torch.tensor(
            [
                [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0],
                [0.5, 1.0, 2.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0],
                [2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0],
                [1.0, 1.2, 1.4, 2.0, 2.2, 2.4, 3.0, 3.2, 3.4],
            ],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        trans_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)
        cap_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)

        actual = lut_entry_2d_op.lut_entry_2d(
            input_trans,
            output_caps,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
        )
        expected = _python_lut_2d(
            input_trans,
            output_caps,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        grad = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=dtype, device=device)
        actual_grads = torch.autograd.grad(actual, (input_trans, output_caps, values), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (input_trans, output_caps, values), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_cpu_forward_backward_matches_python_lut(self):
        self._run_forward_backward_case(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_matches_python_lut(self):
        self._run_forward_backward_case(torch.device("cuda"))

    def _run_coeff_forward_backward_case(self, device):
        from dreamplace.ops.timing_propagation import lut_entry_2d_op

        dtype = torch.float32
        input_trans = torch.tensor([0.05, 0.15, 0.7, 1.2], dtype=dtype, device=device, requires_grad=True)
        output_caps = torch.tensor([0.1, 0.25, 0.8, 1.3], dtype=dtype, device=device, requires_grad=True)
        trans_tables = torch.tensor(
            [[0.1, 0.3, 1.0], [0.0, 0.4, 1.0], [0.2, 0.6, 1.2], [0.2, 0.6, 1.2]],
            dtype=dtype,
            device=device,
        )
        cap_tables = torch.tensor(
            [[0.2, 0.5, 1.0], [0.1, 0.7, 1.4], [0.2, 0.6, 1.0], [0.2, 0.6, 1.0]],
            dtype=dtype,
            device=device,
        )
        values = torch.tensor(
            [
                [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0],
                [0.5, 1.0, 2.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0],
                [2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0],
                [1.0, 1.2, 1.4, 2.0, 2.2, 2.4, 3.0, 3.2, 3.4],
            ],
            dtype=dtype,
            device=device,
        )
        trans_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)
        cap_dims = torch.tensor([3, 3, 3, 3], dtype=torch.long, device=device)

        coeff = lut_entry_2d_op.build_2d_coefficients(
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
        )
        actual = lut_entry_2d_op.lut_entry_2d_coeff(
            input_trans,
            output_caps,
            trans_tables,
            cap_tables,
            coeff,
            trans_dims,
            cap_dims,
        )
        expected = _python_lut_2d(
            input_trans,
            output_caps,
            trans_tables,
            cap_tables,
            values,
            trans_dims,
            cap_dims,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        grad = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=dtype, device=device)
        actual_grads = torch.autograd.grad(actual, (input_trans, output_caps), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (input_trans, output_caps), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_cpu_coeff_forward_backward_matches_python_lut(self):
        self._run_coeff_forward_backward_case(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_coeff_forward_backward_matches_python_lut(self):
        self._run_coeff_forward_backward_case(torch.device("cuda"))

    def test_cell_modeling_lut_gate_uses_fused_op(self):
        from dreamplace.ops.cell_modeling import cell_modeling

        model = cell_modeling.CellModeling.__new__(cell_modeling.CellModeling)
        input_trans = torch.tensor([0.2], dtype=torch.float32)
        output_caps = torch.tensor([0.4], dtype=torch.float32)
        trans_tables = torch.tensor([[0.1, 0.3]], dtype=torch.float32)
        cap_tables = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        trans_dims = torch.tensor([2], dtype=torch.long)
        cap_dims = torch.tensor([2], dtype=torch.long)

        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_OP": "1"}), mock.patch.object(
            cell_modeling.lut_entry_2d_op,
            "lut_entry_2d",
            return_value=torch.tensor([7.0]),
        ) as fused:
            result = model._lut_entry_2d_vectorized(
                input_trans,
                output_caps,
                trans_tables,
                cap_tables,
                values,
                trans_dims,
                cap_dims,
            )

        fused.assert_called_once()
        self.assertTrue(torch.equal(result, torch.tensor([7.0])))

    def test_cell_modeling_lut_gate_uses_coeff_op_when_enabled(self):
        from dreamplace.ops.cell_modeling import cell_modeling

        model = cell_modeling.CellModeling.__new__(cell_modeling.CellModeling)
        input_trans = torch.tensor([0.2], dtype=torch.float32)
        output_caps = torch.tensor([0.4], dtype=torch.float32)
        trans_tables = torch.tensor([[0.1, 0.3]], dtype=torch.float32)
        cap_tables = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        trans_dims = torch.tensor([2], dtype=torch.long)
        cap_dims = torch.tensor([2], dtype=torch.long)
        coeff = torch.zeros((1, 1, 1, 4), dtype=torch.float32)

        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_COEFF_OP": "1"}), mock.patch.object(
            cell_modeling.lut_entry_2d_op,
            "lut_entry_2d_coeff",
            return_value=torch.tensor([9.0]),
        ) as fused:
            result = model._lut_entry_2d_vectorized(
                input_trans,
                output_caps,
                trans_tables,
                cap_tables,
                values,
                trans_dims,
                cap_dims,
                coeff_table_batch=coeff,
            )

        fused.assert_called_once()
        self.assertTrue(torch.equal(result, torch.tensor([9.0])))


if __name__ == "__main__":
    unittest.main()
