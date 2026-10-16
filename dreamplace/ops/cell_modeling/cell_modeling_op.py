import importlib
import importlib.util
from pathlib import Path

import torch
from torch.autograd import Function

import dreamplace.configure as configure


def _load_extension(module_name):
    try:
        return importlib.import_module(f"dreamplace.ops.cell_modeling.{module_name}")
    except ImportError as exc:
        import_error = exc

    autodmp_root = Path(__file__).resolve().parents[3]
    candidates = []
    for build_name in ("build", "build_codex"):
        candidates.extend(
            (autodmp_root / build_name / "dreamplace" / "ops" / "cell_modeling").glob(
                f"{module_name}*.so"
            )
        )
    if candidates:
        path = candidates[0]
        spec = importlib.util.spec_from_file_location(
            f"dreamplace.ops.cell_modeling.{module_name}",
            path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise import_error


cell_modeling_cpp = _load_extension("cell_modeling_cpp")

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    try:
        cell_modeling_cuda = _load_extension("cell_modeling_cuda")
    except ImportError:
        cell_modeling_cuda = None
else:
    cell_modeling_cuda = None


_piecewise_size_backend_profile = {}

LUT_BOUNDARY_MODE_CLAMP = "clamp"
LUT_BOUNDARY_MODE_EXTRAPOLATE = "extrapolate"
_LUT_BOUNDARY_MODE_TO_ID = {
    LUT_BOUNDARY_MODE_CLAMP: 0,
    LUT_BOUNDARY_MODE_EXTRAPOLATE: 1,
}


def _lut_boundary_mode_id(boundary_mode):
    mode = str(boundary_mode or LUT_BOUNDARY_MODE_EXTRAPOLATE).strip().lower()
    if mode not in _LUT_BOUNDARY_MODE_TO_ID:
        raise ValueError(
            f"unsupported cell_model_lut_boundary_mode: {mode!r}; "
            f"expected one of {tuple(_LUT_BOUNDARY_MODE_TO_ID)}"
        )
    return _LUT_BOUNDARY_MODE_TO_ID[mode]


def reset_piecewise_size_backend_profile():
    _piecewise_size_backend_profile.clear()
    _piecewise_size_backend_profile.update(
        {
            "piecewise_size_native_op": "default_on",
            "piecewise_size_forward_backend": "not_called",
            "piecewise_size_backward_backend": "not_called",
            "piecewise_size_forward_calls": 0,
            "piecewise_size_backward_calls": 0,
        }
    )


def _record_piecewise_size_backend(direction, backend):
    if not _piecewise_size_backend_profile:
        reset_piecewise_size_backend_profile()
    backend_key = f"piecewise_size_{direction}_backend"
    calls_key = f"piecewise_size_{direction}_calls"
    _piecewise_size_backend_profile[backend_key] = backend
    _piecewise_size_backend_profile[calls_key] = (
        int(_piecewise_size_backend_profile.get(calls_key, 0)) + 1
    )


def get_piecewise_size_backend_profile():
    if not _piecewise_size_backend_profile:
        reset_piecewise_size_backend_profile()
    return dict(_piecewise_size_backend_profile)


class _Poly12Function(Function):
    @staticmethod
    def forward(ctx, coeff, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
        coeff = coeff.contiguous()
        libcell_main_id = libcell_main_id.long().contiguous()
        arc_offset = arc_offset.long().contiguous()
        vt = vt.contiguous()
        size = size.contiguous()
        input_slew = input_slew.contiguous()
        out_cap = out_cap.contiguous()

        if coeff.is_cuda:
            if cell_modeling_cuda is None:
                raise RuntimeError("cell_modeling_cuda extension is not available")
            func = cell_modeling_cuda.forward
        else:
            func = cell_modeling_cpp.forward
        output = func(coeff, libcell_main_id, arc_offset, vt, size, input_slew, out_cap)
        ctx.save_for_backward(coeff, libcell_main_id, arc_offset, vt, size, input_slew, out_cap)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff, libcell_main_id, arc_offset, vt, size, input_slew, out_cap = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if coeff.is_cuda:
            if cell_modeling_cuda is None:
                raise RuntimeError("cell_modeling_cuda extension is not available")
            func = cell_modeling_cuda.backward
        else:
            func = cell_modeling_cpp.backward
        grad_vt, grad_size, grad_slew, grad_cap = func(
            grad_output,
            coeff,
            libcell_main_id,
            arc_offset,
            vt,
            size,
            input_slew,
            out_cap,
        )
        return None, None, None, grad_vt, grad_size, grad_slew, grad_cap


def poly12_forward(coeff, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
    return _Poly12Function.apply(
        coeff,
        libcell_main_id,
        arc_offset,
        vt,
        size,
        input_slew,
        out_cap,
    )


class _PiecewiseSizeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        size_table,
        arc_table,
        size_count,
        trans_tables,
        cap_tables,
        lut_values,
        trans_dims,
        cap_dims,
        size,
        input_slew,
        out_cap,
        boundary_mode=LUT_BOUNDARY_MODE_EXTRAPOLATE,
    ):
        boundary_mode_id = int(_lut_boundary_mode_id(boundary_mode))
        size_table = size_table.contiguous()
        arc_table = arc_table.long().contiguous()
        size_count = size_count.long().contiguous()
        trans_tables = trans_tables.contiguous()
        cap_tables = cap_tables.contiguous()
        lut_values = lut_values.contiguous()
        trans_dims = trans_dims.long().contiguous()
        cap_dims = cap_dims.long().contiguous()
        size = size.contiguous()
        input_slew = input_slew.contiguous()
        out_cap = out_cap.contiguous()
        if size.is_cuda:
            if cell_modeling_cuda is None:
                raise RuntimeError("cell_modeling_cuda extension is not available")
            backend = "cuda"
            func = cell_modeling_cuda.piecewise_size_forward
        else:
            backend = "cpp"
            func = cell_modeling_cpp.piecewise_size_forward
        _record_piecewise_size_backend("forward", backend)
        output = func(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            lut_values,
            trans_dims,
            cap_dims,
            size,
            input_slew,
            out_cap,
            boundary_mode_id,
        )
        ctx.save_for_backward(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            lut_values,
            trans_dims,
            cap_dims,
            size,
            input_slew,
            out_cap,
        )
        ctx.boundary_mode_id = boundary_mode_id
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        size = tensors[8]
        if size.is_cuda:
            if cell_modeling_cuda is None:
                raise RuntimeError("cell_modeling_cuda extension is not available")
            backend = "cuda"
            func = cell_modeling_cuda.piecewise_size_backward
        else:
            backend = "cpp"
            func = cell_modeling_cpp.piecewise_size_backward
        _record_piecewise_size_backend("backward", backend)
        grad_size, grad_slew, grad_cap, grad_lut_values = func(
            grad_output.contiguous(),
            *tensors,
            int(ctx.boundary_mode_id),
        )
        return (
            None,
            None,
            None,
            None,
            None,
            grad_lut_values,
            None,
            None,
            grad_size,
            grad_slew,
            grad_cap,
            None,
        )


def piecewise_size_forward(
    size_table,
    arc_table,
    size_count,
    trans_tables,
    cap_tables,
    lut_values,
    trans_dims,
    cap_dims,
    size,
    input_slew,
    out_cap,
    boundary_mode=LUT_BOUNDARY_MODE_EXTRAPOLATE,
):
    return _PiecewiseSizeFunction.apply(
        size_table,
        arc_table,
        size_count,
        trans_tables,
        cap_tables,
        lut_values,
        trans_dims,
        cap_dims,
        size,
        input_slew,
        out_cap,
        boundary_mode,
    )
