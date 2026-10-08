import importlib
import importlib.util
from pathlib import Path

import torch
from torch.autograd import Function

import dreamplace.configure as configure


def _load_extension(module_name):
    try:
        return importlib.import_module(f"dreamplace.ops.timing_propagation.{module_name}")
    except ImportError as exc:
        import_error = exc

    autodmp_root = Path(__file__).resolve().parents[3]
    for build_name in ("build", "build_codex"):
        module_dir = autodmp_root / build_name / "dreamplace" / "ops" / "timing_propagation"
        candidates = sorted(module_dir.glob(f"{module_name}*.so"))
        if candidates:
            spec = importlib.util.spec_from_file_location(
                f"dreamplace.ops.timing_propagation.{module_name}",
                candidates[0],
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise import_error


lut_entry_2d_cpp = _load_extension("lut_entry_2d_cpp")

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    try:
        lut_entry_2d_cuda = _load_extension("lut_entry_2d_cuda")
    except ImportError:
        lut_entry_2d_cuda = None
else:
    lut_entry_2d_cuda = None


class _LutEntry2DFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_trans,
        output_caps,
        trans_tables_batch,
        cap_tables_batch,
        lut_values_batch,
        trans_dims_actual,
        cap_dims_actual,
    ):
        input_trans = input_trans.contiguous()
        output_caps = output_caps.contiguous()
        trans_tables_batch = trans_tables_batch.contiguous()
        cap_tables_batch = cap_tables_batch.contiguous()
        lut_values_batch = lut_values_batch.contiguous()
        trans_dims_actual = trans_dims_actual.to(dtype=torch.int32).contiguous()
        cap_dims_actual = cap_dims_actual.to(dtype=torch.int32).contiguous()
        if input_trans.is_cuda:
            if lut_entry_2d_cuda is None:
                raise RuntimeError("lut_entry_2d_cuda extension is not available")
            func = lut_entry_2d_cuda.forward
        else:
            func = lut_entry_2d_cpp.forward
        output = func(
            input_trans,
            output_caps,
            trans_tables_batch,
            cap_tables_batch,
            lut_values_batch,
            trans_dims_actual,
            cap_dims_actual,
        )
        ctx.save_for_backward(
            input_trans,
            output_caps,
            trans_tables_batch,
            cap_tables_batch,
            lut_values_batch,
            trans_dims_actual,
            cap_dims_actual,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        input_trans = tensors[0]
        if input_trans.is_cuda:
            if lut_entry_2d_cuda is None:
                raise RuntimeError("lut_entry_2d_cuda extension is not available")
            func = lut_entry_2d_cuda.backward
        else:
            func = lut_entry_2d_cpp.backward
        return tuple(func(grad_output.contiguous(), *tensors))


def lut_entry_2d(
    input_trans,
    output_caps,
    trans_tables_batch,
    cap_tables_batch,
    lut_values_batch,
    trans_dims_actual,
    cap_dims_actual,
):
    return _LutEntry2DFunction.apply(
        input_trans,
        output_caps,
        trans_tables_batch,
        cap_tables_batch,
        lut_values_batch,
        trans_dims_actual,
        cap_dims_actual,
    )


def _select_module(tensor):
    if tensor.is_cuda:
        if lut_entry_2d_cuda is None:
            raise RuntimeError("lut_entry_2d_cuda extension is not available")
        return lut_entry_2d_cuda
    return lut_entry_2d_cpp


def build_2d_coefficients(
    trans_tables_batch,
    cap_tables_batch,
    lut_values_batch,
    trans_dims_actual,
    cap_dims_actual,
):
    trans_tables_batch = trans_tables_batch.contiguous()
    cap_tables_batch = cap_tables_batch.contiguous()
    lut_values_batch = lut_values_batch.contiguous()
    trans_dims_actual = trans_dims_actual.to(dtype=torch.int32).contiguous()
    cap_dims_actual = cap_dims_actual.to(dtype=torch.int32).contiguous()
    return _select_module(trans_tables_batch).build_coefficients(
        trans_tables_batch,
        cap_tables_batch,
        lut_values_batch,
        trans_dims_actual,
        cap_dims_actual,
    )


class _LutEntry2DCoeffFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_trans,
        output_caps,
        trans_tables_batch,
        cap_tables_batch,
        coeff_batch,
        trans_dims_actual,
        cap_dims_actual,
    ):
        input_trans = input_trans.contiguous()
        output_caps = output_caps.contiguous()
        trans_tables_batch = trans_tables_batch.contiguous()
        cap_tables_batch = cap_tables_batch.contiguous()
        coeff_batch = coeff_batch.contiguous()
        trans_dims_actual = trans_dims_actual.to(dtype=torch.int32).contiguous()
        cap_dims_actual = cap_dims_actual.to(dtype=torch.int32).contiguous()
        output = _select_module(input_trans).coeff_forward(
            input_trans,
            output_caps,
            trans_tables_batch,
            cap_tables_batch,
            coeff_batch,
            trans_dims_actual,
            cap_dims_actual,
        )
        ctx.save_for_backward(
            input_trans,
            output_caps,
            trans_tables_batch,
            cap_tables_batch,
            coeff_batch,
            trans_dims_actual,
            cap_dims_actual,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        input_trans = tensors[0]
        grads = _select_module(input_trans).coeff_backward(grad_output.contiguous(), *tensors)
        return tuple(grads)


def lut_entry_2d_coeff(
    input_trans,
    output_caps,
    trans_tables_batch,
    cap_tables_batch,
    coeff_batch,
    trans_dims_actual,
    cap_dims_actual,
):
    return _LutEntry2DCoeffFunction.apply(
        input_trans,
        output_caps,
        trans_tables_batch,
        cap_tables_batch,
        coeff_batch,
        trans_dims_actual,
        cap_dims_actual,
    )
