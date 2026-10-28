import unittest

import torch

from dreamplace.ops.buffer_insertion.buffer_interpolation import (
    interpolate_buffer_values_by_bsu_index,
)


class BufferInterpolationTest(unittest.TestCase):
    def test_bsu_index_interpolation_has_gradient_inside_legal_range(self):
        bsu_index = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.float64))
        per_size_input_cap = torch.tensor([[0.10, 0.20, 0.50, 0.90]], dtype=torch.float64)
        per_size_delay = torch.tensor([[10.0, 8.0, 4.0, 2.0]], dtype=torch.float64)
        per_size_output_slew = torch.tensor([[1.0, 0.8, 0.5, 0.4]], dtype=torch.float64)

        result = interpolate_buffer_values_by_bsu_index(
            bsu_index,
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )
        loss = result["buffer_delay"].sum()
        loss.backward()

        self.assertAlmostEqual(float(result["buffer_delay"][0]), 7.0)
        self.assertIsNotNone(bsu_index.grad)
        self.assertAlmostEqual(float(bsu_index.grad[0]), -4.0)
        self.assertEqual(result["bsu_index_status"]["clamp"], "none")
        self.assertEqual(result["bsu_index_status"]["coordinate_source"], "current_relaxed")

    def test_bsu_index_interpolation_reports_guard_clamp(self):
        bsu_index = torch.nn.Parameter(torch.tensor([9.0], dtype=torch.float32))
        per_size = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        result = interpolate_buffer_values_by_bsu_index(
            bsu_index,
            per_size_input_cap=per_size,
            per_size_delay=per_size,
            per_size_output_slew=per_size,
        )

        self.assertTrue(result["bsu_index_status"]["clamped"])
        self.assertEqual(result["bsu_index_status"]["clamp"], "high")
        self.assertAlmostEqual(float(result["buffer_delay"][0]), 2.0)
