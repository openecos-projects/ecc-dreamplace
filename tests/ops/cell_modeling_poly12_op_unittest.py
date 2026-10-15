import os
import sys
import unittest
from unittest import mock

import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _python_poly12(coeff, main_id, arc_offset, vt, size, slew, cap):
    eps = 1e-9
    inv_size = 1.0 / (size.float() + eps)
    features = torch.stack(
        [
            slew.float(),
            cap.float(),
            inv_size,
            cap.float() * inv_size,
            vt.float(),
            slew.float() * cap.float(),
            slew.float() * inv_size,
            slew.float() * vt.float(),
            cap.float() * vt.float(),
            inv_size * vt.float(),
            inv_size.square(),
            torch.ones_like(slew.float()),
        ],
        dim=1,
    )
    w = coeff[main_id.long(), arc_offset.long()]
    return (w * features).sum(dim=1)


class CellModelingPoly12OpTest(unittest.TestCase):
    def _run_forward_backward_case(self, device):
        from dreamplace.ops.cell_modeling import cell_modeling_op

        torch.manual_seed(7)
        coeff = torch.randn(4, 5, 12, dtype=torch.float32, device=device)
        main_id = torch.tensor([0, 1, 3, 9], dtype=torch.long, device=device)
        arc_offset = torch.tensor([0, 4, 1, 2], dtype=torch.long, device=device)
        vt = torch.tensor([0.0, 1.0, 0.5, 1.5], dtype=torch.float32, device=device, requires_grad=True)
        size = torch.tensor([1.0, 2.0, 3.5, 4.0], dtype=torch.float32, device=device, requires_grad=True)
        slew = torch.tensor([0.02, 0.17, 0.33, 0.51], dtype=torch.float32, device=device, requires_grad=True)
        cap = torch.tensor([0.001, 0.04, 0.08, 0.12], dtype=torch.float32, device=device, requires_grad=True)

        actual = cell_modeling_op.poly12_forward(coeff, main_id, arc_offset, vt, size, slew, cap)
        expected = _python_poly12(
            coeff,
            main_id.clamp(max=coeff.shape[0] - 1),
            arc_offset.clamp(max=coeff.shape[1] - 1),
            vt,
            size,
            slew,
            cap,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        grad = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.float32, device=device)
        actual_grads = torch.autograd.grad(actual, (vt, size, slew, cap), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (vt, size, slew, cap), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_cpu_forward_backward_matches_python_poly12(self):
        self._run_forward_backward_case(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_matches_python_poly12(self):
        self._run_forward_backward_case(torch.device("cuda"))

    def test_cell_modeling_poly12_gate_uses_fused_op(self):
        from dreamplace.ops.cell_modeling import cell_modeling

        model = cell_modeling.CellModeling.__new__(cell_modeling.CellModeling)
        model.cell_model_schema = cell_modeling.POLY12_SCHEMA
        model.use_poly12_op = True
        coeff = torch.randn(2, 3, 12, dtype=torch.float32)
        main_id = torch.tensor([0, 1], dtype=torch.long)
        arc_offset = torch.tensor([1, 2], dtype=torch.long)
        vt = torch.tensor([0.0, 1.0], dtype=torch.float32)
        size = torch.tensor([1.0, 2.0], dtype=torch.float32)
        slew = torch.tensor([0.1, 0.2], dtype=torch.float32)
        cap = torch.tensor([0.01, 0.02], dtype=torch.float32)

        with mock.patch.object(cell_modeling.cell_modeling_op, "poly12_forward", return_value=torch.ones(2)) as fused:
            result = model._linear_forward_dataset(coeff, main_id, arc_offset, vt, size, slew, cap)

        fused.assert_called_once()
        self.assertTrue(torch.equal(result, torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
