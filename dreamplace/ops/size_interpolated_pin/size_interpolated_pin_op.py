import importlib
import importlib.util
from pathlib import Path

import torch
from torch.autograd import Function

import dreamplace.configure as configure


def _load_extension(module_name):
    try:
        return importlib.import_module(f"dreamplace.ops.size_interpolated_pin.{module_name}")
    except ImportError as exc:
        import_error = exc

    autodmp_root = Path(__file__).resolve().parents[3]
    for build_name in ("build", "build_codex"):
        module_dir = autodmp_root / build_name / "dreamplace" / "ops" / "size_interpolated_pin"
        candidates = sorted(module_dir.glob(f"{module_name}*.so"))
        if candidates:
            spec = importlib.util.spec_from_file_location(
                f"dreamplace.ops.size_interpolated_pin.{module_name}",
                candidates[0],
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise import_error


size_interpolated_pin_cpp = _load_extension("size_interpolated_pin_cpp")

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    try:
        size_interpolated_pin_cuda = _load_extension("size_interpolated_pin_cuda")
    except ImportError:
        size_interpolated_pin_cuda = None
else:
    size_interpolated_pin_cuda = None


def _select_module(tensor):
    if tensor.is_cuda:
        if size_interpolated_pin_cuda is None:
            raise RuntimeError("size_interpolated_pin_cuda extension is not available")
        return size_interpolated_pin_cuda
    return size_interpolated_pin_cpp


class _SizeInterpolatedPinFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_sizes,
        vt_probs,
        candidate_sizes,
        candidate_libpin_ids,
        actual_dims,
        flat_pin_values,
        current_values_active,
    ):
        local_sizes = local_sizes.contiguous()
        vt_probs = vt_probs.contiguous()
        candidate_sizes = candidate_sizes.contiguous()
        candidate_libpin_ids = candidate_libpin_ids.to(dtype=torch.long).contiguous()
        actual_dims = actual_dims.to(dtype=torch.int32).contiguous()
        flat_pin_values = flat_pin_values.contiguous()
        current_values_active = current_values_active.contiguous()
        output = _select_module(local_sizes).forward(
            local_sizes,
            vt_probs,
            candidate_sizes,
            candidate_libpin_ids,
            actual_dims,
            flat_pin_values,
            current_values_active,
        )
        ctx.save_for_backward(
            local_sizes,
            vt_probs,
            candidate_sizes,
            candidate_libpin_ids,
            actual_dims,
            flat_pin_values,
            current_values_active,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        local_sizes = tensors[0]
        grads = _select_module(local_sizes).backward(grad_output.contiguous(), *tensors)
        return tuple(grads)


def size_interpolated_pin(
    local_sizes,
    vt_probs,
    candidate_sizes,
    candidate_libpin_ids,
    actual_dims,
    flat_pin_values,
    current_values_active,
):
    return _SizeInterpolatedPinFunction.apply(
        local_sizes,
        vt_probs,
        candidate_sizes,
        candidate_libpin_ids,
        actual_dims,
        flat_pin_values,
        current_values_active,
    )


def size_interpolated_pin_forward(
    local_sizes,
    vt_probs,
    candidate_sizes,
    candidate_libpin_ids,
    actual_dims,
    flat_pin_values,
    current_values_active,
):
    return size_interpolated_pin(
        local_sizes,
        vt_probs,
        candidate_sizes,
        candidate_libpin_ids,
        actual_dims,
        flat_pin_values,
        current_values_active,
    )
