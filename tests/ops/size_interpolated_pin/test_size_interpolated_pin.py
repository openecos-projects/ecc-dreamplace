import unittest

import torch

import dreamplace.ops.size_interpolated_pin.sizing_limit_utils as sizing_limit_utils
from dreamplace.ops.size_interpolated_pin.sizing_limit_utils import compute_size_interpolated_pin_properties
from dreamplace.ops.size_interpolated_pin.size_interpolated_pin_op import (
    size_interpolated_pin,
)


def _reference(
    local_sizes,
    vt_probs,
    candidate_sizes,
    candidate_libpin_ids,
    actual_dims,
    flat_pin_values,
    current_values_active,
):
    q_count = flat_pin_values.size(0)
    pin_count = local_sizes.numel()
    vt_count = vt_probs.size(1)
    out = current_values_active.clone()
    for q in range(q_count):
        for p in range(pin_count):
            weighted_sum = local_sizes.new_tensor(0.0)
            total_weight = local_sizes.new_tensor(0.0)
            for vt in range(vt_count):
                dim = int(actual_dims[p, vt].item())
                if dim <= 0:
                    continue
                weight = vt_probs[p, vt]
                xs = candidate_sizes[p, vt, :dim]
                ids = candidate_libpin_ids[p, vt, :dim].long()
                ys = flat_pin_values[q, ids]
                if dim == 1:
                    interp = ys[0]
                else:
                    x = torch.clamp(local_sizes[p], min=xs[0], max=xs[-1])
                    high = torch.searchsorted(xs.contiguous(), x.unsqueeze(0), right=True)[0]
                    high = torch.clamp(high, min=1, max=dim - 1)
                    low = high - 1
                    x0 = xs[low]
                    x1 = xs[high]
                    y0 = ys[low]
                    y1 = ys[high]
                    denom = x1 - x0
                    interp = torch.where(
                        denom.abs() < 1e-12,
                        y0,
                        y0 + (x - x0) / denom * (y1 - y0),
                    )
                weighted_sum = weighted_sum + weight * interp
                total_weight = total_weight + weight
            if total_weight > 0:
                out[q, p] = weighted_sum / total_weight.clamp_min(1e-12)
    return out


def _fixture(device="cpu", dtype=torch.float64):
    local_sizes = torch.tensor([1.5, 4.0, 0.5], device=device, dtype=dtype)
    vt_probs = torch.tensor(
        [
            [0.25, 0.75],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    candidate_sizes = torch.tensor(
        [
            [[1.0, 2.0, 4.0], [1.0, 3.0, 0.0]],
            [[1.0, 2.0, 4.0], [2.0, 0.0, 0.0]],
            [[1.0, 2.0, 4.0], [1.0, 3.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    candidate_libpin_ids = torch.tensor(
        [
            [[0, 1, 2], [3, 4, -1]],
            [[5, 6, 7], [8, -1, -1]],
            [[0, 1, 2], [3, 4, -1]],
        ],
        device=device,
        dtype=torch.long,
    )
    actual_dims = torch.tensor(
        [[3, 2], [3, 1], [3, 2]],
        device=device,
        dtype=torch.int32,
    )
    flat_pin_values = torch.tensor(
        [
            [10.0, 20.0, 40.0, 30.0, 60.0, 100.0, 110.0, 140.0, 200.0],
            [1.0, 2.0, 4.0, 3.0, 6.0, 10.0, 11.0, 14.0, 20.0],
        ],
        device=device,
        dtype=dtype,
    )
    current_values_active = torch.tensor(
        [[9.0, 90.0, 7.0], [0.9, 9.0, 0.7]],
        device=device,
        dtype=dtype,
    )
    return (
        local_sizes,
        vt_probs,
        candidate_sizes,
        candidate_libpin_ids,
        actual_dims,
        flat_pin_values,
        current_values_active,
    )


class SizeInterpolatedPinTest(unittest.TestCase):
    def test_cpu_forward_matches_reference_for_two_properties(self):
        tensors = _fixture()
        result = size_interpolated_pin(*tensors)
        expected = _reference(*tensors)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_cpu_forward_clamps_out_of_range_sizes(self):
        tensors = list(_fixture())
        tensors[0] = torch.tensor([0.25, 8.0, 1.5], dtype=tensors[0].dtype)
        tensors[1] = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            dtype=tensors[1].dtype,
        )
        result = size_interpolated_pin(*tensors)
        expected = _reference(*tensors)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
        self.assertEqual(result[0, 0].item(), 10.0)
        self.assertEqual(result[0, 1].item(), 140.0)

    def test_cpu_forward_uses_single_candidate_value(self):
        tensors = list(_fixture())
        tensors[1] = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=tensors[1].dtype,
        )
        result = size_interpolated_pin(*tensors)
        expected = _reference(*tensors)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
        self.assertEqual(result[0, 1].item(), 200.0)

    def test_cpu_gradcheck_local_size_and_vt_prob(self):
        tensors = list(_fixture())
        tensors[0] = torch.tensor([1.5, 3.0, 1.5], dtype=tensors[0].dtype)
        tensors[1] = torch.tensor(
            [[0.25, 0.75], [1.0, 0.2], [0.4, 0.6]],
            dtype=tensors[1].dtype,
        )
        tensors[0] = tensors[0].clone().requires_grad_(True)
        tensors[1] = tensors[1].clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda local_sizes, vt_probs: size_interpolated_pin(
                    local_sizes,
                    vt_probs,
                    tensors[2],
                    tensors[3],
                    tensors[4],
                    tensors[5],
                    tensors[6],
                ),
                (tensors[0], tensors[1]),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-4,
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_matches_cpu_reference(self):
        cpu_tensors = _fixture()
        cuda_tensors = tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in cpu_tensors)
        result = size_interpolated_pin(*cuda_tensors).cpu()
        expected = _reference(*cpu_tensors)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_gradcheck_local_size_and_vt_prob(self):
        tensors = list(_fixture(device="cuda"))
        tensors[0] = torch.tensor([1.5, 3.0, 1.5], device="cuda", dtype=tensors[0].dtype)
        tensors[1] = torch.tensor(
            [[0.25, 0.75], [1.0, 0.2], [0.4, 0.6]],
            device="cuda",
            dtype=tensors[1].dtype,
        )
        tensors[0] = tensors[0].clone().requires_grad_(True)
        tensors[1] = tensors[1].clone().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda local_sizes, vt_probs: size_interpolated_pin(
                    local_sizes,
                    vt_probs,
                    tensors[2],
                    tensors[3],
                    tensors[4],
                    tensors[5],
                    tensors[6],
                ),
                (tensors[0], tensors[1]),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_compute_size_interpolated_pin_properties_matches_legacy(self):
        class DataCollections:
            pass

        data = DataCollections()
        data.pin2node_map = torch.tensor([0, 1, 2], dtype=torch.long)
        data.inst_main_id = torch.tensor([0, 0, 0], dtype=torch.long)
        data.inst_libcell_offset = torch.tensor([0, 2, 1], dtype=torch.long)
        data.inst_is_sizeable = torch.tensor([True, True, True])
        data.pin_2_libpin_offset = torch.tensor([0, 0, 0], dtype=torch.long)
        data.main_id_2_cell_id_start = torch.tensor([0, 4], dtype=torch.long)
        data.cell_id_2_libpin_id_start = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        data.flat_libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 3.0, 1.0],
            ],
            dtype=torch.float32,
        )
        size_var = torch.tensor([1.5, 2.5, 0.8], dtype=torch.float32, requires_grad=True)
        vt_var = torch.tensor(
            [[0.25, 0.75], [0.8, 0.2], [1.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        data.get_size_var = lambda: size_var
        data.get_vt_var = lambda: vt_var
        flat_a = torch.tensor([10.0, 20.0, 30.0, 60.0], dtype=torch.float32)
        flat_b = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float32)
        native = compute_size_interpolated_pin_properties(data, [flat_a, flat_b])
        legacy = sizing_limit_utils._compute_size_interpolated_pin_properties_legacy(
            data,
            [flat_a, flat_b],
        )
        self.assertEqual(len(native), 2)
        for native_prop, legacy_prop in zip(native, legacy):
            torch.testing.assert_close(native_prop, legacy_prop, atol=1e-5, rtol=1e-5)

    def test_native_mode_off_uses_legacy_path(self):
        class DataCollections:
            pass

        data = DataCollections()
        data.size_interpolated_pin_native_op = "off"
        flat = torch.tensor([1.0], dtype=torch.float32)

        result = sizing_limit_utils._compute_size_interpolated_pin_properties_native(
            data,
            [flat],
        )

        self.assertIsNone(result)

    def test_native_mode_on_requires_extension(self):
        class DataCollections:
            pass

        data = DataCollections()
        data.size_interpolated_pin_native_op = "on"
        flat = torch.tensor([1.0], dtype=torch.float32)
        original = sizing_limit_utils.size_interpolated_pin
        try:
            sizing_limit_utils.size_interpolated_pin = None
            with self.assertRaises(RuntimeError):
                sizing_limit_utils._compute_size_interpolated_pin_properties_native(
                    data,
                    [flat],
                )
        finally:
            sizing_limit_utils.size_interpolated_pin = original


if __name__ == "__main__":
    unittest.main()
