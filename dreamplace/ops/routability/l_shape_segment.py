##
# @file   l_shape_segment.py
# @brief  Build L-shape segments from Steiner tree edges with EGR L-direction
#         Each L-shape edge is split into two segments (horizontal + vertical)
#         This enables accurate density modeling for routing congestion
#

import torch
import logging
import hashlib
import os
import glob
import importlib.util
import sys
import sysconfig
from contextlib import contextmanager
from torch.autograd import Function

logger = logging.getLogger(__name__)

def _load_local_extension(module_name):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    matches = glob.glob(os.path.join(module_dir, f"{module_name}*.so"))
    if not matches:
        return None
    # Editable installs can leave extensions for more than one interpreter
    # ABI in the package directory.  Loading the first glob result may pick a
    # stale cpython-310 module when running under cpython-311.
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    compatible = [path for path in matches if path.endswith(extension_suffix)]
    selected = sorted(compatible or matches)[0]
    spec = importlib.util.spec_from_file_location(module_name, selected)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.modules.pop(module_name, None)
        return None
    sys.modules[f"dreamplace.ops.routability.{module_name}"] = module
    return module


segment_compaction_cpp = _load_local_extension("segment_compaction_cpp")
segment_compaction_cuda = _load_local_extension("segment_compaction_cuda")
_segment_compaction_logged_forward_backends = set()
_segment_compaction_logged_backward_backends = set()

# L方向常量 (与steiner_topo.py保持一致)
H_FIRST = 0       # 先水平后垂直, 拐点在 (x2, y1)
V_FIRST = 1       # 先垂直后水平, 拐点在 (x1, y2)
STRAIGHT = 2      # 直线（水平或垂直）
FAKE_STRAIGHT = 3 # 伪直线（在gcell下只有一条wire）
UNKNOWN = -1      # 未知，segment阶段跳过


def _stable_argsort(values):
    try:
        return torch.argsort(values, stable=True)
    except TypeError:
        return torch.argsort(values)


def _build_gather_plan(indices):
    indices = indices.to(dtype=torch.long).contiguous()
    if indices.numel() == 0:
        empty_long = torch.tensor([], dtype=torch.long, device=indices.device)
        return {
            "indices": indices,
            "order": empty_long,
            "unique": empty_long,
            "counts": empty_long,
        }

    order = _stable_argsort(indices)
    sorted_indices = indices.index_select(0, order)
    unique, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
    return {
        "indices": indices,
        "order": order,
        "unique": unique,
        "counts": counts.to(dtype=torch.long),
    }


def _move_gather_plan(plan, device):
    if plan["indices"].device == device:
        return plan
    return {key: value.to(device) for key, value in plan.items()}


class _DeterministicGather1DFunction(Function):
    @staticmethod
    def forward(ctx, values, indices, order, unique_indices, counts):
        ctx.input_numel = int(values.numel())
        ctx.save_for_backward(order, unique_indices, counts)
        return values.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[0]:
            order, unique_indices, counts = ctx.saved_tensors
            grad_values = grad_output.new_zeros(ctx.input_numel)
            if unique_indices.numel() > 0 and grad_output.numel() > 0:
                sorted_grad = grad_output.contiguous().index_select(0, order)
                counts = counts.to(device=sorted_grad.device)
                if hasattr(torch, "segment_reduce"):
                    reduced = torch.segment_reduce(sorted_grad, "sum", lengths=counts)
                else:
                    pieces = []
                    start = 0
                    for count in counts.cpu().tolist():
                        end = start + int(count)
                        pieces.append(sorted_grad[start:end].sum(dim=0))
                        start = end
                    reduced = torch.stack(pieces) if pieces else sorted_grad[:0]
                grad_values.index_copy_(0, unique_indices, reduced)
        return grad_values, None, None, None, None


def _deterministic_gather_1d(values, plan):
    plan = _move_gather_plan(plan, values.device)
    return _DeterministicGather1DFunction.apply(
        values,
        plan["indices"],
        plan["order"],
        plan["unique"],
        plan["counts"],
    )


def _env_flag_enabled(name):
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _env_any_flag_enabled(*names):
    return any(_env_flag_enabled(name) for name in names)


@contextmanager
def _temporary_env_flag(name, enabled):
    old_value = os.environ.get(name)
    os.environ[name] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _segment_compaction_extension_for_device(device):
    if device.type == "cuda" and segment_compaction_cuda is not None:
        return segment_compaction_cuda
    if device.type == "cpu" and segment_compaction_cpp is not None:
        return segment_compaction_cpp
    return None


def _segment_compaction_backend_name(extension, device):
    if extension is segment_compaction_cuda:
        return "cuda"
    if extension is segment_compaction_cpp:
        return "cpp"
    return f"python_fallback_{device.type}"


class _SegmentPositionCompactionFunction(Function):
    @staticmethod
    def forward(ctx, seg1_llx, seg1_lly, seg2_llx, seg2_lly, seg1_indices, seg2_indices):
        ctx.seg1_numel = int(seg1_llx.numel())
        ctx.seg2_numel = int(seg2_llx.numel())
        ctx.save_for_backward(seg1_indices, seg2_indices)
        ctx.extension = _segment_compaction_extension_for_device(seg1_llx.device)
        ctx.backend_name = _segment_compaction_backend_name(ctx.extension, seg1_llx.device)

        if ctx.extension is not None:
            if ctx.backend_name not in _segment_compaction_logged_forward_backends:
                _segment_compaction_logged_forward_backends.add(ctx.backend_name)
                logger.info(
                    "LShapeSegmentCompaction forward backend=%s device=%s seg1_numel=%d seg2_numel=%d",
                    ctx.backend_name,
                    seg1_llx.device,
                    ctx.seg1_numel,
                    ctx.seg2_numel,
                )
            return tuple(ctx.extension.forward(
                seg1_llx.contiguous(),
                seg1_lly.contiguous(),
                seg2_llx.contiguous(),
                seg2_lly.contiguous(),
                seg1_indices.contiguous(),
                seg2_indices.contiguous(),
            ))

        seg1_out_llx = seg1_llx.index_select(0, seg1_indices)
        seg1_out_lly = seg1_lly.index_select(0, seg1_indices)
        seg2_out_llx = seg2_llx.index_select(0, seg2_indices)
        seg2_out_lly = seg2_lly.index_select(0, seg2_indices)
        return (
            torch.cat((seg1_out_llx, seg2_out_llx), dim=0),
            torch.cat((seg1_out_lly, seg2_out_lly), dim=0),
        )

    @staticmethod
    def backward(ctx, grad_segment_llx, grad_segment_lly):
        seg1_indices, seg2_indices = ctx.saved_tensors
        seg1_count = int(seg1_indices.numel())
        seg2_count = int(seg2_indices.numel())

        if grad_segment_llx is not None and grad_segment_lly is not None and ctx.extension is not None:
            if ctx.backend_name not in _segment_compaction_logged_backward_backends:
                _segment_compaction_logged_backward_backends.add(ctx.backend_name)
                logger.info(
                    "LShapeSegmentCompaction backward backend=%s device=%s seg1_numel=%d seg2_numel=%d",
                    ctx.backend_name,
                    grad_segment_llx.device,
                    ctx.seg1_numel,
                    ctx.seg2_numel,
                )
            grad_segment_llx = grad_segment_llx.contiguous()
            grad_segment_lly = grad_segment_lly.contiguous()
            grad_seg1_llx, grad_seg1_lly, grad_seg2_llx, grad_seg2_lly = ctx.extension.backward(
                grad_segment_llx,
                grad_segment_lly,
                seg1_indices.contiguous(),
                seg2_indices.contiguous(),
                ctx.seg1_numel,
                ctx.seg2_numel,
            )
            if not ctx.needs_input_grad[0]:
                grad_seg1_llx = None
            if not ctx.needs_input_grad[1]:
                grad_seg1_lly = None
            if not ctx.needs_input_grad[2]:
                grad_seg2_llx = None
            if not ctx.needs_input_grad[3]:
                grad_seg2_lly = None
            return grad_seg1_llx, grad_seg1_lly, grad_seg2_llx, grad_seg2_lly, None, None

        grad_seg1_llx = grad_seg1_lly = grad_seg2_llx = grad_seg2_lly = None
        if grad_segment_llx is not None:
            if ctx.needs_input_grad[0]:
                grad_seg1_llx = grad_segment_llx.new_zeros(ctx.seg1_numel)
                if seg1_count > 0:
                    grad_seg1_llx.index_copy_(0, seg1_indices, grad_segment_llx[:seg1_count])
            if ctx.needs_input_grad[2]:
                grad_seg2_llx = grad_segment_llx.new_zeros(ctx.seg2_numel)
                if seg2_count > 0:
                    grad_seg2_llx.index_copy_(0, seg2_indices, grad_segment_llx[seg1_count:])

        if grad_segment_lly is not None:
            if ctx.needs_input_grad[1]:
                grad_seg1_lly = grad_segment_lly.new_zeros(ctx.seg1_numel)
                if seg1_count > 0:
                    grad_seg1_lly.index_copy_(0, seg1_indices, grad_segment_lly[:seg1_count])
            if ctx.needs_input_grad[3]:
                grad_seg2_lly = grad_segment_lly.new_zeros(ctx.seg2_numel)
                if seg2_count > 0:
                    grad_seg2_lly.index_copy_(0, seg2_indices, grad_segment_lly[seg1_count:])

        return grad_seg1_llx, grad_seg1_lly, grad_seg2_llx, grad_seg2_lly, None, None


def _compact_segment_positions(seg1_llx, seg1_lly, seg2_llx, seg2_lly, seg1_indices, seg2_indices):
    return _SegmentPositionCompactionFunction.apply(
        seg1_llx,
        seg1_lly,
        seg2_llx,
        seg2_lly,
        seg1_indices,
        seg2_indices,
    )


def _compact_hard_segment_positions(seg1_llx, seg1_lly, seg2_llx, seg2_lly, seg1_indices, seg2_indices):
    return _compact_segment_positions(seg1_llx, seg1_lly, seg2_llx, seg2_lly, seg1_indices, seg2_indices)


def _clone_snapshot_tensor(tensor, *, requires_grad=False):
    cloned = tensor.detach().clone()
    if requires_grad:
        cloned.requires_grad_(True)
    return cloned


def _run_segment_snapshot(snapshot, *, reference):
    with _temporary_env_flag("DREAMPLACE_L_SHAPE_SEGMENT_COMPACTION_REFERENCE", reference):
        op = LShapeSegmentOp(
            wire_width=float(snapshot.get("wire_width", 0.0)),
            wire_width_h=snapshot.get("wire_width_h", None),
            wire_width_v=snapshot.get("wire_width_v", None),
            deterministic_backward=bool(snapshot.get("deterministic_backward", True)),
        )
        newx = _clone_snapshot_tensor(snapshot["newx"], requires_grad=True)
        newy = _clone_snapshot_tensor(snapshot["newy"], requires_grad=True)
        result = op(
            newx,
            newy,
            _clone_snapshot_tensor(snapshot["flat_from"]).long(),
            _clone_snapshot_tensor(snapshot["flat_to"]).long(),
            _clone_snapshot_tensor(snapshot["l_directions"]).long(),
        )
    return result, newx, newy


def _run_hard_segment_snapshot(snapshot, *, reference):
    return _run_segment_snapshot(snapshot, reference=reference)


def _safe_relative_norm(diff, reference):
    reference_norm = reference.norm(p=2)
    denom = reference_norm.clamp_min(torch.finfo(reference.dtype).eps)
    return float((diff.norm(p=2) / denom).detach().cpu().item())


def save_segment_snapshot(
    path,
    newx,
    newy,
    flat_from,
    flat_to,
    l_directions,
    *,
    wire_width,
    wire_width_h=None,
    wire_width_v=None,
    deterministic_backward=True,
):
    snapshot = {
        "newx": newx.detach().cpu(),
        "newy": newy.detach().cpu(),
        "flat_from": flat_from.detach().cpu().long(),
        "flat_to": flat_to.detach().cpu().long(),
        "l_directions": l_directions.detach().cpu().long(),
        "wire_width": float(wire_width),
        "wire_width_h": None if wire_width_h is None else float(wire_width_h),
        "wire_width_v": None if wire_width_v is None else float(wire_width_v),
        "deterministic_backward": bool(deterministic_backward),
    }
    torch.save(snapshot, path)


def save_hard_segment_snapshot(*args, **kwargs):
    return save_segment_snapshot(*args, **kwargs)


def maybe_save_segment_snapshot(
    path,
    snapshot_iter,
    current_iter,
    soft_l_assignment,
    already_saved,
    newx,
    newy,
    flat_from,
    flat_to,
    l_directions,
    *,
    wire_width,
    wire_width_h=None,
    wire_width_v=None,
    deterministic_backward=True,
):
    if not path or already_saved or soft_l_assignment:
        return False
    if snapshot_iter:
        if current_iter is None:
            return False
        if int(snapshot_iter) != int(current_iter):
            return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_segment_snapshot(
        path,
        newx,
        newy,
        flat_from,
        flat_to,
        l_directions,
        wire_width=wire_width,
        wire_width_h=wire_width_h,
        wire_width_v=wire_width_v,
        deterministic_backward=deterministic_backward,
    )
    return True


def maybe_save_hard_segment_snapshot(*args, **kwargs):
    return maybe_save_segment_snapshot(*args, **kwargs)


def replay_segment_snapshot_vjp(path):
    snapshot = torch.load(path, map_location="cpu")
    reference, ref_newx, ref_newy = _run_segment_snapshot(snapshot, reference=True)
    optimized, opt_newx, opt_newy = _run_segment_snapshot(snapshot, reference=False)

    forward_max_abs = 0.0
    for key in ("segment_llx", "segment_lly", "segment_size_x", "segment_size_y", "segment_weight"):
        diff = optimized[key].detach() - reference[key].detach()
        if diff.numel() > 0:
            forward_max_abs = max(forward_max_abs, float(diff.abs().max().cpu().item()))
    for key in ("segment_edge_idx", "segment_is_horizontal"):
        if not torch.equal(optimized[key].detach().cpu(), reference[key].detach().cpu()):
            raise AssertionError(f"{key} mismatch in segment snapshot replay")

    num_segments = int(optimized["num_segments"])
    upstream_llx = torch.linspace(
        -0.75,
        0.85,
        num_segments,
        dtype=optimized["segment_llx"].dtype,
        device=optimized["segment_llx"].device,
    )
    upstream_lly = torch.linspace(
        0.45,
        -0.65,
        num_segments,
        dtype=optimized["segment_lly"].dtype,
        device=optimized["segment_lly"].device,
    )
    opt_cost = (
        optimized["segment_llx"].mul(upstream_llx).sum()
        + optimized["segment_lly"].mul(upstream_lly).sum()
    )
    ref_cost = (
        reference["segment_llx"].mul(upstream_llx).sum()
        + reference["segment_lly"].mul(upstream_lly).sum()
    )
    opt_cost.backward()
    ref_cost.backward()

    grad_newx_diff = opt_newx.grad - ref_newx.grad
    grad_newy_diff = opt_newy.grad - ref_newy.grad
    grad_diff = torch.cat((grad_newx_diff, grad_newy_diff), dim=0)
    ref_grad = torch.cat((ref_newx.grad, ref_newy.grad), dim=0)
    opt_grad = torch.cat((opt_newx.grad, opt_newy.grad), dim=0)
    cosine = torch.nn.functional.cosine_similarity(
        opt_grad.reshape(1, -1),
        ref_grad.reshape(1, -1),
        dim=1,
        eps=torch.finfo(opt_grad.dtype).eps,
    )

    return {
        "num_segments": num_segments,
        "forward_max_abs": forward_max_abs,
        "grad_newx_max_abs": float(grad_newx_diff.abs().max().cpu().item()),
        "grad_newy_max_abs": float(grad_newy_diff.abs().max().cpu().item()),
        "grad_relative_norm": _safe_relative_norm(grad_diff, ref_grad),
        "grad_cosine_similarity": float(cosine.detach().cpu().item()),
    }


def replay_hard_segment_snapshot_vjp(path):
    return replay_segment_snapshot_vjp(path)


def _compact_segments_reference(
    seg1_llx,
    seg1_lly,
    seg1_size_x,
    seg1_size_y,
    seg1_is_h,
    seg2_llx,
    seg2_lly,
    seg2_size_x,
    seg2_size_y,
    seg2_is_h,
    valid_edge_idx,
    seg1_valid,
    seg2_valid,
    final_valid,
):
    segment_llx = torch.cat((seg1_llx[seg1_valid], seg2_llx[seg2_valid]), dim=0)
    segment_lly = torch.cat((seg1_lly[seg1_valid], seg2_lly[seg2_valid]), dim=0)
    segment_size_x = torch.cat((seg1_size_x[seg1_valid], seg2_size_x[seg2_valid]), dim=0)
    segment_size_y = torch.cat((seg1_size_y[seg1_valid], seg2_size_y[seg2_valid]), dim=0)
    segment_edge_idx = torch.cat((valid_edge_idx[seg1_valid], valid_edge_idx[seg2_valid]), dim=0)
    segment_is_horizontal = torch.cat((seg1_is_h[seg1_valid], seg2_is_h[seg2_valid]), dim=0)
    segment_weight = torch.cat(
        (
            torch.ones_like(seg1_size_x[seg1_valid]),
            torch.ones_like(seg2_size_x[seg2_valid]),
        ),
        dim=0,
    )

    return {
        "segment_llx": segment_llx[final_valid],
        "segment_lly": segment_lly[final_valid],
        "segment_size_x": segment_size_x[final_valid],
        "segment_size_y": segment_size_y[final_valid],
        "segment_edge_idx": segment_edge_idx[final_valid],
        "segment_is_horizontal": segment_is_horizontal[final_valid],
        "segment_weight": segment_weight[final_valid],
    }


def _compact_hard_segments_reference(*args, **kwargs):
    return _compact_segments_reference(*args, **kwargs)


class LShapeSegmentBuilder:
    """
    将Steiner树的边根据L方向拆分为segments
    
    核心思想：
    - H_FIRST: p1 -> corner(p2.x, p1.y) -> p2 (先水平后垂直)
    - V_FIRST: p1 -> corner(p1.x, p2.y) -> p2 (先垂直后水平)
    - STRAIGHT: p1 -> p2 (直线)
    
    每个segment是一个矩形，用于后续密度计算
    """
    
    def __init__(self, wire_width=0.0, wire_width_h=None, wire_width_v=None):
        """
        Args:
            wire_width: 线宽，用于给segment增加宽度
        """
        self.wire_width = wire_width
        self.wire_width_h = float(wire_width if wire_width_h is None else wire_width_h)
        self.wire_width_v = float(wire_width if wire_width_v is None else wire_width_v)
        self.use_directional_widths = wire_width_h is not None or wire_width_v is not None
    
    def build_segments(self, newx, newy, flat_from, flat_to, l_directions):
        """
        根据L方向将边拆分为segments
        
        Args:
            newx: [num_vertices] 所有顶点的x坐标 (pins + Steiner points)
            newy: [num_vertices] 所有顶点的y坐标
            flat_from: [num_edges] 边的起点索引
            flat_to: [num_edges] 边的终点索引
            l_directions: [num_edges] 每条边的L方向 (H_FIRST/V_FIRST/STRAIGHT/UNKNOWN)
            
        Returns:
            dict with:
                segment_llx: [num_segments] segment左下角x
                segment_lly: [num_segments] segment左下角y
                segment_size_x: [num_segments] segment宽度
                segment_size_y: [num_segments] segment高度
                segment_edge_idx: [num_segments] 每个segment对应的原始边索引
                segment_is_horizontal: [num_segments] 是否是水平segment
        """
        device = newx.device
        dtype = newx.dtype
        
        num_edges = flat_from.numel()
        
        # 预分配（最多每条边拆分为2个segment）
        max_segments = num_edges * 2
        
        segment_llx_list = []
        segment_lly_list = []
        segment_size_x_list = []
        segment_size_y_list = []
        segment_edge_idx_list = []
        segment_is_horizontal_list = []
        
        # 获取numpy用于循环（但保持tensor用于梯度）
        flat_from_np = flat_from.cpu().numpy() if flat_from.is_cuda else flat_from.numpy()
        flat_to_np = flat_to.cpu().numpy() if flat_to.is_cuda else flat_to.numpy()
        l_dir_np = l_directions.cpu().numpy() if l_directions.is_cuda else l_directions.numpy()
        
        for edge_idx in range(num_edges):
            from_idx = flat_from_np[edge_idx]
            to_idx = flat_to_np[edge_idx]
            
            if from_idx < 0 or to_idx < 0:
                continue
            if from_idx >= len(newx) or to_idx >= len(newx):
                continue
            
            # 获取端点坐标（保持tensor以保持梯度）
            x1, y1 = newx[from_idx], newy[from_idx]
            x2, y2 = newx[to_idx], newy[to_idx]
            
            l_dir = l_dir_np[edge_idx]
            
            # 判断是否是直线（水平或垂直）
            is_horizontal_line = torch.abs(y1 - y2) < 1e-4
            is_vertical_line = torch.abs(x1 - x2) < 1e-4
            # 如果l_dir标记为STRAIGHT但几何上是斜线，强制按L形处理
            is_straight = (is_horizontal_line or is_vertical_line) or (
                l_dir == STRAIGHT and (is_horizontal_line or is_vertical_line)
            )
            
            if is_straight:
                # 直线段：创建一个segment
                seg_llx, seg_lly, seg_sx, seg_sy, seg_is_h = self._create_segment(
                    x1, y1, x2, y2
                )
                segment_llx_list.append(seg_llx)
                segment_lly_list.append(seg_lly)
                segment_size_x_list.append(seg_sx)
                segment_size_y_list.append(seg_sy)
                segment_edge_idx_list.append(edge_idx)
                segment_is_horizontal_list.append(seg_is_h)
            else:
                # L形：根据方向确定拐点，拆分为两个segment
                # STRAIGHT但为斜线时，默认按H_FIRST处理
                if l_dir == STRAIGHT:
                    l_dir = H_FIRST
                if l_dir == H_FIRST:
                    # 先水平后垂直，拐点在 (x2, y1)
                    corner_x, corner_y = x2, y1
                elif l_dir == V_FIRST:
                    # 先垂直后水平，拐点在 (x1, y2)
                    corner_x, corner_y = x1, y2
                elif l_dir == FAKE_STRAIGHT:
                    # 残余 FAKE_STRAIGHT 仍按 H_FIRST fallback 处理
                    corner_x, corner_y = x2, y1
                elif l_dir == UNKNOWN:
                    # 解析后仍未知的边不参与 L-shape segment 构建
                    continue
                else:
                    # 非法方向值不参与 L-shape segment 构建
                    continue
                
                # Segment 1: p1 -> corner
                seg1_llx, seg1_lly, seg1_sx, seg1_sy, seg1_is_h = self._create_segment(
                    x1, y1, corner_x, corner_y
                )
                if seg1_sx > 1e-6 or seg1_sy > 1e-6:  # 过滤零尺寸segment
                    segment_llx_list.append(seg1_llx)
                    segment_lly_list.append(seg1_lly)
                    segment_size_x_list.append(seg1_sx)
                    segment_size_y_list.append(seg1_sy)
                    segment_edge_idx_list.append(edge_idx)
                    segment_is_horizontal_list.append(seg1_is_h)
                
                # Segment 2: corner -> p2
                seg2_llx, seg2_lly, seg2_sx, seg2_sy, seg2_is_h = self._create_segment(
                    corner_x, corner_y, x2, y2
                )
                if seg2_sx > 1e-6 or seg2_sy > 1e-6:  # 过滤零尺寸segment
                    segment_llx_list.append(seg2_llx)
                    segment_lly_list.append(seg2_lly)
                    segment_size_x_list.append(seg2_sx)
                    segment_size_y_list.append(seg2_sy)
                    segment_edge_idx_list.append(edge_idx)
                    segment_is_horizontal_list.append(seg2_is_h)
        
        # 转换为tensor
        if len(segment_llx_list) == 0:
            # 没有有效segment
            empty = torch.tensor([], dtype=dtype, device=device)
            return {
                'segment_llx': empty,
                'segment_lly': empty,
                'segment_size_x': empty,
                'segment_size_y': empty,
                'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
                'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
                'num_segments': 0
            }
        
        segment_llx = torch.stack(segment_llx_list)
        segment_lly = torch.stack(segment_lly_list)
        segment_size_x = torch.stack(segment_size_x_list)
        segment_size_y = torch.stack(segment_size_y_list)
        segment_edge_idx = torch.tensor(segment_edge_idx_list, dtype=torch.long, device=device)
        segment_is_horizontal = torch.tensor(segment_is_horizontal_list, dtype=torch.bool, device=device)
        
        num_segments = len(segment_llx_list)
        logger.info(f"Built {num_segments} segments from {num_edges} edges")
        
        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'num_segments': num_segments
        }
    
    def _create_segment(self, x1, y1, x2, y2):
        """
        创建一个segment（矩形）
        
        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
            half_width: 半线宽
            
        Returns:
            llx, lly: 左下角坐标
            size_x, size_y: 尺寸
            is_horizontal: 是否是水平segment
        """
        min_x = torch.minimum(x1, x2)
        max_x = torch.maximum(x1, x2)
        min_y = torch.minimum(y1, y2)
        max_y = torch.maximum(y1, y2)
        
        # 判断是水平还是垂直
        dx = torch.abs(x2 - x1)
        dy = torch.abs(y2 - y1)
        is_horizontal = dx >= dy
        
        if self.use_directional_widths:
            half_width_h = self.wire_width_h / 2.0
            half_width_v = self.wire_width_v / 2.0
            width_h = torch.full_like(dx, self.wire_width_h)
            width_v = torch.full_like(dx, self.wire_width_v)
            llx = torch.where(
                is_horizontal,
                min_x,
                min_x - half_width_v,
            )
            lly = torch.where(
                is_horizontal,
                min_y - half_width_h,
                min_y,
            )
            size_x = torch.where(
                is_horizontal,
                max_x - min_x,
                width_v,
            )
            size_y = torch.where(
                is_horizontal,
                width_h,
                max_y - min_y,
            )
        else:
            half_width = self.wire_width / 2.0
            llx = min_x - half_width
            lly = min_y - half_width
            size_x = (max_x - min_x) + 2 * half_width
            size_y = (max_y - min_y) + 2 * half_width

            min_size = half_width * 2 if half_width > 0 else 1e-6
            size_x = torch.maximum(size_x, torch.tensor(min_size, dtype=size_x.dtype, device=size_x.device))
            size_y = torch.maximum(size_y, torch.tensor(min_size, dtype=size_y.dtype, device=size_y.device))
        
        return llx, lly, size_x, size_y, is_horizontal


def build_l_shape_segments_vectorized(newx, newy, flat_from, flat_to, l_directions, wire_width=0.0):
    """
    向量化版本的L形segment构建（更快但需要更多内存）
    
    Args:
        newx: [num_vertices] 所有顶点的x坐标
        newy: [num_vertices] 所有顶点的y坐标
        flat_from: [num_edges] 边的起点索引
        flat_to: [num_edges] 边的终点索引
        l_directions: [num_edges] 每条边的L方向
        wire_width: 线宽
        
    Returns:
        dict with segment information
    """
    device = newx.device
    dtype = newx.dtype
    
    num_edges = flat_from.numel()
    half_width = wire_width / 2.0
    
    # 确保所有tensor在同一设备上
    if flat_from.device != device:
        flat_from = flat_from.to(device)
    if flat_to.device != device:
        flat_to = flat_to.to(device)
    if l_directions.device != device:
        l_directions = l_directions.to(device)
    
    # 过滤无效边
    valid_mask = (flat_from >= 0) & (flat_to >= 0) & (flat_from < len(newx)) & (flat_to < len(newx))
    valid_from = flat_from[valid_mask]
    valid_to = flat_to[valid_mask]
    valid_l_dir = l_directions[valid_mask]
    valid_edge_idx = torch.arange(num_edges, device=device)[valid_mask]
    
    num_valid = valid_from.numel()
    if num_valid == 0:
        empty = torch.tensor([], dtype=dtype, device=device)
        return {
            'segment_llx': empty,
            'segment_lly': empty,
            'segment_size_x': empty,
            'segment_size_y': empty,
            'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
            'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
            'num_segments': 0
        }
    
    # 获取端点坐标
    x1 = newx[valid_from]
    y1 = newy[valid_from]
    x2 = newx[valid_to]
    y2 = newy[valid_to]
    
    # 判断边的类型
    is_horizontal_line = torch.abs(y1 - y2) < 1e-4
    is_vertical_line = torch.abs(x1 - x2) < 1e-4
    # 如果l_dir标记为STRAIGHT但几何上为斜线，强制按L形处理
    straight_by_dir = (valid_l_dir == STRAIGHT)
    diag_straight = straight_by_dir & ~(is_horizontal_line | is_vertical_line)
    is_straight = (is_horizontal_line | is_vertical_line) | (straight_by_dir & ~diag_straight)
    # 对斜线但被标记为STRAIGHT的边，默认按H_FIRST处理。
    # 残余 UNKNOWN 在 segment 阶段跳过，不生成 L-shape。
    is_upper_l = (~is_straight) & (
        (valid_l_dir == H_FIRST) | (valid_l_dir == FAKE_STRAIGHT) | diag_straight
    )
    is_lower_l = (~is_straight) & (valid_l_dir == V_FIRST)
    
    # 计算拐点坐标
    # H_FIRST: corner = (x2, y1)
    # V_FIRST: corner = (x1, y2)
    corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
    corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))
    
    # ===== 构建所有segments =====
    # 对于直线边：1个segment (p1 -> p2)
    # 对于L形边：2个segments (p1 -> corner, corner -> p2)
    
    # Segment类型1: 直线边 或 L形边的第一段
    seg1_x1 = x1
    seg1_y1 = y1
    seg1_x2 = torch.where(is_straight, x2, corner_x)
    seg1_y2 = torch.where(is_straight, y2, corner_y)
    
    seg1_min_x = torch.minimum(seg1_x1, seg1_x2)
    seg1_max_x = torch.maximum(seg1_x1, seg1_x2)
    seg1_min_y = torch.minimum(seg1_y1, seg1_y2)
    seg1_max_y = torch.maximum(seg1_y1, seg1_y2)
    
    seg1_llx = seg1_min_x - half_width
    seg1_lly = seg1_min_y - half_width
    seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
    seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
    seg1_is_h = torch.abs(seg1_x2 - seg1_x1) >= torch.abs(seg1_y2 - seg1_y1)
    
    # Segment类型2: L形边的第二段 (corner -> p2)
    # 只对L形边有效
    seg2_x1 = corner_x
    seg2_y1 = corner_y
    seg2_x2 = x2
    seg2_y2 = y2
    
    seg2_min_x = torch.minimum(seg2_x1, seg2_x2)
    seg2_max_x = torch.maximum(seg2_x1, seg2_x2)
    seg2_min_y = torch.minimum(seg2_y1, seg2_y2)
    seg2_max_y = torch.maximum(seg2_y1, seg2_y2)
    
    seg2_llx = seg2_min_x - half_width
    seg2_lly = seg2_min_y - half_width
    seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
    seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width
    seg2_is_h = torch.abs(seg2_x2 - seg2_x1) >= torch.abs(seg2_y2 - seg2_y1)
    
    # 过滤有效segment2（只有L形边有第二段）
    is_l_shape = is_upper_l | is_lower_l
    seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
    seg1_valid = is_straight | is_l_shape
    
    # 合并所有segments
    # Segment 1 (直线边或L形边的第一段)
    all_llx = [seg1_llx[seg1_valid]]
    all_lly = [seg1_lly[seg1_valid]]
    all_size_x = [seg1_size_x[seg1_valid]]
    all_size_y = [seg1_size_y[seg1_valid]]
    all_edge_idx = [valid_edge_idx[seg1_valid]]
    all_is_h = [seg1_is_h[seg1_valid]]
    
    # Segment 2 (只有L形边)
    if seg2_valid.any():
        all_llx.append(seg2_llx[seg2_valid])
        all_lly.append(seg2_lly[seg2_valid])
        all_size_x.append(seg2_size_x[seg2_valid])
        all_size_y.append(seg2_size_y[seg2_valid])
        all_edge_idx.append(valid_edge_idx[seg2_valid])
        all_is_h.append(seg2_is_h[seg2_valid])
    
    segment_llx = torch.cat(all_llx)
    segment_lly = torch.cat(all_lly)
    segment_size_x = torch.cat(all_size_x)
    segment_size_y = torch.cat(all_size_y)
    segment_edge_idx = torch.cat(all_edge_idx)
    segment_is_horizontal = torch.cat(all_is_h)
    
    # 过滤零尺寸segment
    min_size = max(half_width * 2, 1e-6)
    valid_seg = (segment_size_x > min_size) | (segment_size_y > min_size)
    
    segment_llx = segment_llx[valid_seg]
    segment_lly = segment_lly[valid_seg]
    segment_size_x = segment_size_x[valid_seg]
    segment_size_y = segment_size_y[valid_seg]
    segment_edge_idx = segment_edge_idx[valid_seg]
    segment_is_horizontal = segment_is_horizontal[valid_seg]
    
    num_segments = segment_llx.numel()
    logger.info(f"Built {num_segments} segments from {num_edges} edges (vectorized)")
    
    return {
        'segment_llx': segment_llx,
        'segment_lly': segment_lly,
        'segment_size_x': segment_size_x,
        'segment_size_y': segment_size_y,
        'segment_edge_idx': segment_edge_idx,
        'segment_is_horizontal': segment_is_horizontal,
        'num_segments': num_segments
    }


def build_segment_pos_tensor(segment_llx, segment_lly):
    """
    将segment坐标转换为pos tensor格式 (与BBoxElectricPotential兼容)
    
    Args:
        segment_llx: [num_segments]
        segment_lly: [num_segments]
        
    Returns:
        pos: [num_segments * 2] 格式为 [llx1, llx2, ..., lly1, lly2, ...]
    """
    return torch.cat([segment_llx, segment_lly])


class LShapeSegmentOp:
    """
    可微的L形segment操作
    
    使用方式：
        op = LShapeSegmentOp(wire_width=100.0)
        result = op(newx, newy, flat_from, flat_to, l_directions)
        segment_pos = result['segment_pos']  # 用于密度计算
        
    优化：预计算拓扑结构，只在坐标更新时重新计算segment位置
    """
    
    def __init__(
        self,
        wire_width=0.0,
        wire_width_h=None,
        wire_width_v=None,
        use_vectorized=True,
        soft_min_weight=0.0,
        deterministic_backward=False,
        log_verbose=0,
    ):
        self.wire_width = wire_width
        self.wire_width_h = float(wire_width if wire_width_h is None else wire_width_h)
        self.wire_width_v = float(wire_width if wire_width_v is None else wire_width_v)
        self.use_directional_widths = wire_width_h is not None or wire_width_v is not None
        self.use_vectorized = use_vectorized
        self.soft_min_weight = float(soft_min_weight)
        self.deterministic_backward = bool(deterministic_backward)
        self.log_verbose = int(log_verbose)
        self.segment_compaction_reference = _env_any_flag_enabled(
            "DREAMPLACE_L_SHAPE_SEGMENT_COMPACTION_REFERENCE",
            "DREAMPLACE_L_SHAPE_HARD_SEGMENT_COMPACTION_REFERENCE",
        )
        if self.segment_compaction_reference:
            logger.info("Use reference PyTorch segment compaction via environment override")
        self.builder = LShapeSegmentBuilder(
            wire_width,
            wire_width_h=wire_width_h,
            wire_width_v=wire_width_v,
        )
        
        # 缓存拓扑结构（不随pos变化）
        self._cached_topology = None
        # 缓存输入签名，用于检测EGR/GPUGR拓扑更新
        self._cached_input_key = None
        self._cached_input_hash = None
    
    def reset_cache(self):
        """重置拓扑缓存，在EGR重新运行后调用"""
        self._cached_topology = None
        self._cached_input_key = None
        self._cached_input_hash = None
        if self.log_verbose >= 2:
            logger.info("LShapeSegmentOp cache reset")

    def _tensor_fast_key(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return ("none", 0, 0, 0, 0)
        values = tensor.detach()
        return (
            str(values.dtype),
            str(values.device),
            int(values.numel()),
            int(values.data_ptr()) if values.numel() > 0 else 0,
            int(getattr(tensor, "_version", 0)),
        )

    def _compute_input_key(self, flat_from, flat_to, l_directions):
        """Compute a cheap exact-enough cache key for stable tensor topology objects."""
        return (
            self._tensor_fast_key(flat_from),
            self._tensor_fast_key(flat_to),
            self._tensor_fast_key(l_directions),
        )
    
    def _tensor_content_digest(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return ("none", 0, "none")
        values = tensor.detach()
        if values.device.type != "cpu":
            values = values.cpu()
        values = values.contiguous()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(int(dim) for dim in values.shape)).encode("ascii"))
        if values.numel() > 0:
            digest.update(values.view(torch.uint8).numpy().tobytes())
        return (str(values.dtype), int(values.numel()), digest.hexdigest())

    def _compute_input_hash(self, flat_from, flat_to, l_directions):
        """Compute an exact topology signature for cache invalidation."""
        return (
            self._tensor_content_digest(flat_from),
            self._tensor_content_digest(flat_to),
            self._tensor_content_digest(l_directions),
        )

    def _empty_segment_result(self, dtype, device):
        empty = torch.tensor([], dtype=dtype, device=device)
        return {
            'segment_llx': empty,
            'segment_lly': empty,
            'segment_size_x': empty,
            'segment_size_y': empty,
            'segment_edge_idx': torch.tensor([], dtype=torch.long, device=device),
            'segment_is_horizontal': torch.tensor([], dtype=torch.bool, device=device),
            'segment_weight': empty,
            'num_segments': 0
        }

    def _create_segment_batch(self, x1, y1, x2, y2):
        min_x = torch.minimum(x1, x2)
        max_x = torch.maximum(x1, x2)
        min_y = torch.minimum(y1, y2)
        max_y = torch.maximum(y1, y2)

        is_horizontal = torch.abs(x2 - x1) >= torch.abs(y2 - y1)
        if self.use_directional_widths:
            half_width_h = self.wire_width_h / 2.0
            half_width_v = self.wire_width_v / 2.0
            width_h = torch.full_like(min_x, self.wire_width_h)
            width_v = torch.full_like(min_x, self.wire_width_v)
            llx = torch.where(is_horizontal, min_x, min_x - half_width_v)
            lly = torch.where(is_horizontal, min_y - half_width_h, min_y)
            size_x = torch.where(is_horizontal, max_x - min_x, width_v)
            size_y = torch.where(is_horizontal, width_h, max_y - min_y)
        else:
            half_width = self.wire_width / 2.0
            llx = min_x - half_width
            lly = min_y - half_width
            size_x = (max_x - min_x) + 2 * half_width
            size_y = (max_y - min_y) + 2 * half_width
        return llx, lly, size_x, size_y, is_horizontal
    
    def _compute_topology(self, flat_from, flat_to, l_directions, num_vertices, device):
        """
        预计算拓扑结构（只需要计算一次）
        
        Args:
            flat_from, flat_to: 边的端点索引
            l_directions: L方向
            num_vertices: 顶点数量
            device: 目标设备（应与newx/newy一致）
        
        Returns:
            dict with precomputed topology info
        """
        num_edges = flat_from.numel()
        # 确保所有输入在同一设备上
        if flat_from.device != device:
            flat_from = flat_from.to(device)
        if flat_to.device != device:
            flat_to = flat_to.to(device)
        if l_directions.device != device:
            l_directions = l_directions.to(device)
        
        # 过滤无效边
        valid_mask = (flat_from >= 0) & (flat_to >= 0) & (flat_from < num_vertices) & (flat_to < num_vertices)
        valid_from = flat_from[valid_mask]
        valid_to = flat_to[valid_mask]
        valid_l_dir = l_directions[valid_mask]
        valid_edge_idx = torch.arange(num_edges, device=device)[valid_mask]
        
        num_valid = valid_from.numel()
        
        if num_valid == 0:
            return {
                'valid': False,
                'num_valid': 0
            }

        return {
            'valid': True,
            'num_valid': num_valid,
            'valid_mask': valid_mask,
            'valid_from': valid_from,
            'valid_to': valid_to,
            'valid_from_gather_plan': _build_gather_plan(valid_from),
            'valid_to_gather_plan': _build_gather_plan(valid_to),
            'valid_edge_idx': valid_edge_idx,
            'num_edges': num_edges
        }

    def _gather_vertices(self, values, topo, key):
        indices = topo[key]
        if indices.device != values.device:
            indices = indices.to(values.device)
        if not self.deterministic_backward or not values.requires_grad:
            return values[indices]

        plan = topo.get(f"{key}_gather_plan")
        if plan is None:
            plan = _build_gather_plan(indices)
        return _deterministic_gather_1d(values, plan)

    def _compute_segments_discrete(self, newx, newy, topo, l_directions):
        """
        使用离散 L-direction 计算 segment
        """
        device = newx.device
        dtype = newx.dtype
        if not topo['valid']:
            return self._empty_segment_result(dtype, device)

        # 确保索引在同一设备上
        valid_from = topo['valid_from']
        valid_to = topo['valid_to']
        valid_edge_idx = topo['valid_edge_idx']
        # 关键：将索引移到与newx相同的设备
        if valid_from.device != device:
            valid_from = valid_from.to(device)
            valid_to = valid_to.to(device)
            valid_edge_idx = valid_edge_idx.to(device)
        valid_mask = topo['valid_mask']
        if valid_mask.device != device:
            valid_mask = valid_mask.to(device)
        if l_directions.device != device:
            l_directions = l_directions.to(device)
        valid_l_dir = l_directions[valid_mask]
        is_h_first = (valid_l_dir == H_FIRST) | (valid_l_dir == FAKE_STRAIGHT)
        is_v_first = (valid_l_dir == V_FIRST)
        is_straight_by_dir = (valid_l_dir == STRAIGHT)

        # 获取端点坐标
        x1 = self._gather_vertices(newx, topo, 'valid_from')
        y1 = self._gather_vertices(newy, topo, 'valid_from')
        x2 = self._gather_vertices(newx, topo, 'valid_to')
        y2 = self._gather_vertices(newy, topo, 'valid_to')
        
        # 判断几何上的直线
        is_horizontal_line = torch.abs(y1 - y2) < 1e-4
        is_vertical_line = torch.abs(x1 - x2) < 1e-4
        # 如果l_dir标记为STRAIGHT但几何上为斜线，强制按L形处理
        diag_straight = is_straight_by_dir & ~(is_horizontal_line | is_vertical_line)
        is_straight = (is_horizontal_line | is_vertical_line) | (is_straight_by_dir & ~diag_straight)
        
        # 对斜线但被标记为STRAIGHT的边，默认按H_FIRST处理。
        # 残余 UNKNOWN 在 segment 阶段跳过，不生成 L-shape。
        is_upper_l = (~is_straight) & (is_h_first | diag_straight)
        is_lower_l = (~is_straight) & is_v_first
        
        # 计算拐点坐标
        corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
        corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))
        
        # Segment 1: 所有边都有 (p1 -> corner/p2)
        seg1_x2 = torch.where(is_straight, x2, corner_x)
        seg1_y2 = torch.where(is_straight, y2, corner_y)
        
        seg1_min_x = torch.minimum(x1, seg1_x2)
        seg1_max_x = torch.maximum(x1, seg1_x2)
        seg1_min_y = torch.minimum(y1, seg1_y2)
        seg1_max_y = torch.maximum(y1, seg1_y2)
        
        seg1_is_h = torch.abs(seg1_x2 - x1) >= torch.abs(seg1_y2 - y1)
        if self.use_directional_widths:
            half_width_h = self.wire_width_h / 2.0
            half_width_v = self.wire_width_v / 2.0
            width_h = torch.full_like(seg1_min_x, self.wire_width_h)
            width_v = torch.full_like(seg1_min_x, self.wire_width_v)
            seg1_llx = torch.where(seg1_is_h, seg1_min_x, seg1_min_x - half_width_v)
            seg1_lly = torch.where(seg1_is_h, seg1_min_y - half_width_h, seg1_min_y)
            seg1_size_x = torch.where(seg1_is_h, seg1_max_x - seg1_min_x, width_v)
            seg1_size_y = torch.where(seg1_is_h, width_h, seg1_max_y - seg1_min_y)
        else:
            half_width = self.wire_width / 2.0
            seg1_llx = seg1_min_x - half_width
            seg1_lly = seg1_min_y - half_width
            seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
            seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
        
        # Segment 2: 只有L形边有 (corner -> p2)
        is_l_shape = is_upper_l | is_lower_l
        seg1_valid = is_straight | is_l_shape
        
        seg2_min_x = torch.minimum(corner_x, x2)
        seg2_max_x = torch.maximum(corner_x, x2)
        seg2_min_y = torch.minimum(corner_y, y2)
        seg2_max_y = torch.maximum(corner_y, y2)
        
        seg2_is_h = torch.abs(x2 - corner_x) >= torch.abs(y2 - corner_y)
        if self.use_directional_widths:
            half_width_h = self.wire_width_h / 2.0
            half_width_v = self.wire_width_v / 2.0
            width_h = torch.full_like(seg2_min_x, self.wire_width_h)
            width_v = torch.full_like(seg2_min_x, self.wire_width_v)
            seg2_llx = torch.where(seg2_is_h, seg2_min_x, seg2_min_x - half_width_v)
            seg2_lly = torch.where(seg2_is_h, seg2_min_y - half_width_h, seg2_min_y)
            seg2_size_x = torch.where(seg2_is_h, seg2_max_x - seg2_min_x, width_v)
            seg2_size_y = torch.where(seg2_is_h, width_h, seg2_max_y - seg2_min_y)
        else:
            half_width = self.wire_width / 2.0
            seg2_llx = seg2_min_x - half_width
            seg2_lly = seg2_min_y - half_width
            seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
            seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width

        seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
        
        if self.use_directional_widths:
            seg1_final_valid = seg1_valid & (seg1_size_x > 1e-6) & (seg1_size_y > 1e-6)
            seg2_final_valid = seg2_valid & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
        else:
            half_width = self.wire_width / 2.0
            min_size = max(half_width * 2, 1e-6)
            seg1_final_valid = seg1_valid & (
                (seg1_size_x > min_size) | (seg1_size_y > min_size)
            )
            seg2_final_valid = seg2_valid & (
                (seg2_size_x > min_size) | (seg2_size_y > min_size)
            )

        if self.segment_compaction_reference:
            reference_final_valid = torch.cat(
                (
                    seg1_final_valid.index_select(0, torch.nonzero(seg1_valid, as_tuple=False).flatten()),
                    seg2_final_valid.index_select(0, torch.nonzero(seg2_valid, as_tuple=False).flatten()),
                ),
                dim=0,
            )
            compacted = _compact_segments_reference(
                seg1_llx,
                seg1_lly,
                seg1_size_x,
                seg1_size_y,
                seg1_is_h,
                seg2_llx,
                seg2_lly,
                seg2_size_x,
                seg2_size_y,
                seg2_is_h,
                valid_edge_idx,
                seg1_valid,
                seg2_valid,
                reference_final_valid,
            )
            segment_llx = compacted["segment_llx"]
            segment_lly = compacted["segment_lly"]
            segment_size_x = compacted["segment_size_x"]
            segment_size_y = compacted["segment_size_y"]
            segment_edge_idx = compacted["segment_edge_idx"]
            segment_is_horizontal = compacted["segment_is_horizontal"]
            segment_weight = compacted["segment_weight"]
            num_segments = segment_llx.numel()
            return {
                'segment_llx': segment_llx,
                'segment_lly': segment_lly,
                'segment_size_x': segment_size_x,
                'segment_size_y': segment_size_y,
                'segment_edge_idx': segment_edge_idx,
                'segment_is_horizontal': segment_is_horizontal,
                'segment_weight': segment_weight,
                'num_segments': num_segments
            }

        seg1_indices = torch.nonzero(seg1_final_valid, as_tuple=False).flatten()
        seg2_indices = torch.nonzero(seg2_final_valid, as_tuple=False).flatten()

        seg1_size_x_forward = seg1_size_x.detach()
        seg1_size_y_forward = seg1_size_y.detach()
        seg2_size_x_forward = seg2_size_x.detach()
        seg2_size_y_forward = seg2_size_y.detach()

        segment_llx, segment_lly = _compact_segment_positions(
            seg1_llx,
            seg1_lly,
            seg2_llx,
            seg2_lly,
            seg1_indices,
            seg2_indices,
        )

        segment_size_x = torch.cat(
            (
                seg1_size_x_forward.index_select(0, seg1_indices),
                seg2_size_x_forward.index_select(0, seg2_indices),
            ),
            dim=0,
        )
        segment_size_y = torch.cat(
            (
                seg1_size_y_forward.index_select(0, seg1_indices),
                seg2_size_y_forward.index_select(0, seg2_indices),
            ),
            dim=0,
        )
        segment_edge_idx = torch.cat(
            (
                valid_edge_idx.index_select(0, seg1_indices),
                valid_edge_idx.index_select(0, seg2_indices),
            ),
            dim=0,
        )
        segment_is_horizontal = torch.cat(
            (
                seg1_is_h.index_select(0, seg1_indices),
                seg2_is_h.index_select(0, seg2_indices),
            ),
            dim=0,
        )
        segment_weight = torch.ones_like(segment_size_x)
        
        num_segments = segment_llx.numel()
        
        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'segment_weight': segment_weight,
            'num_segments': num_segments
        }

    def _compute_segments_soft(self, newx, newy, topo, soft_l_weights):
        """
        使用 soft H/V 权重生成候选 segment
        """
        device = newx.device
        dtype = newx.dtype
        if not topo['valid']:
            return self._empty_segment_result(dtype, device)

        valid_from = topo['valid_from']
        valid_to = topo['valid_to']
        valid_edge_idx = topo['valid_edge_idx']
        valid_mask = topo['valid_mask']
        if valid_from.device != device:
            valid_from = valid_from.to(device)
            valid_to = valid_to.to(device)
            valid_edge_idx = valid_edge_idx.to(device)
        if valid_mask.device != device:
            valid_mask = valid_mask.to(device)
        if soft_l_weights.device != device:
            soft_l_weights = soft_l_weights.to(device)

        valid_soft = soft_l_weights[valid_mask].to(dtype=dtype)
        if valid_soft.dim() != 2 or valid_soft.size(1) != 2:
            raise ValueError("soft_l_weights must have shape [num_edges, 2]")

        valid_soft = torch.clamp(valid_soft, min=0.0)
        weight_sum = valid_soft.sum(dim=1, keepdim=True)
        fallback = torch.full_like(valid_soft, 0.5)
        valid_soft = torch.where(weight_sum > 1e-12, valid_soft / weight_sum.clamp_min(1e-12), fallback)

        x1 = self._gather_vertices(newx, topo, 'valid_from')
        y1 = self._gather_vertices(newy, topo, 'valid_from')
        x2 = self._gather_vertices(newx, topo, 'valid_to')
        y2 = self._gather_vertices(newy, topo, 'valid_to')

        is_horizontal_line = torch.abs(y1 - y2) < 1e-4
        is_vertical_line = torch.abs(x1 - x2) < 1e-4
        is_straight = is_horizontal_line | is_vertical_line
        is_diagonal = ~is_straight

        all_llx = []
        all_lly = []
        all_size_x = []
        all_size_y = []
        all_edge_idx = []
        all_is_h = []
        all_weight = []

        def append_group(llx, lly, size_x, size_y, edge_idx, is_h, weight, valid):
            if valid.any():
                all_llx.append(llx[valid])
                all_lly.append(lly[valid])
                all_size_x.append(size_x[valid])
                all_size_y.append(size_y[valid])
                all_edge_idx.append(edge_idx[valid])
                all_is_h.append(is_h[valid])
                all_weight.append(weight[valid])

        straight_llx, straight_lly, straight_size_x, straight_size_y, straight_is_h = self._create_segment_batch(
            x1, y1, x2, y2
        )
        append_group(
            straight_llx,
            straight_lly,
            straight_size_x,
            straight_size_y,
            valid_edge_idx,
            straight_is_h,
            torch.ones_like(straight_size_x),
            is_straight,
        )

        h_weight = valid_soft[:, 0]
        v_weight = valid_soft[:, 1]
        min_weight = max(self.soft_min_weight, 0.0)

        h_corner_x = x2
        h_corner_y = y1
        h_seg1_llx, h_seg1_lly, h_seg1_size_x, h_seg1_size_y, h_seg1_is_h = self._create_segment_batch(
            x1, y1, h_corner_x, h_corner_y
        )
        h_seg2_llx, h_seg2_lly, h_seg2_size_x, h_seg2_size_y, h_seg2_is_h = self._create_segment_batch(
            h_corner_x, h_corner_y, x2, y2
        )
        h_valid = is_diagonal & (h_weight > min_weight)
        append_group(h_seg1_llx, h_seg1_lly, h_seg1_size_x, h_seg1_size_y, valid_edge_idx, h_seg1_is_h, h_weight, h_valid)
        append_group(h_seg2_llx, h_seg2_lly, h_seg2_size_x, h_seg2_size_y, valid_edge_idx, h_seg2_is_h, h_weight, h_valid)

        v_corner_x = x1
        v_corner_y = y2
        v_seg1_llx, v_seg1_lly, v_seg1_size_x, v_seg1_size_y, v_seg1_is_h = self._create_segment_batch(
            x1, y1, v_corner_x, v_corner_y
        )
        v_seg2_llx, v_seg2_lly, v_seg2_size_x, v_seg2_size_y, v_seg2_is_h = self._create_segment_batch(
            v_corner_x, v_corner_y, x2, y2
        )
        v_valid = is_diagonal & (v_weight > min_weight)
        append_group(v_seg1_llx, v_seg1_lly, v_seg1_size_x, v_seg1_size_y, valid_edge_idx, v_seg1_is_h, v_weight, v_valid)
        append_group(v_seg2_llx, v_seg2_lly, v_seg2_size_x, v_seg2_size_y, valid_edge_idx, v_seg2_is_h, v_weight, v_valid)

        if not all_llx:
            return self._empty_segment_result(dtype, device)

        segment_llx = torch.cat(all_llx)
        segment_lly = torch.cat(all_lly)
        segment_size_x = torch.cat(all_size_x)
        segment_size_y = torch.cat(all_size_y)
        segment_edge_idx = torch.cat(all_edge_idx)
        segment_is_horizontal = torch.cat(all_is_h)
        segment_weight = torch.cat(all_weight)

        if self.use_directional_widths:
            valid_seg = (segment_size_x > 1e-6) & (segment_size_y > 1e-6) & (segment_weight > min_weight)
        else:
            half_width = self.wire_width / 2.0
            min_size = max(half_width * 2, 1e-6)
            valid_seg = ((segment_size_x > min_size) | (segment_size_y > min_size)) & (segment_weight > min_weight)

        segment_llx = segment_llx[valid_seg]
        segment_lly = segment_lly[valid_seg]
        segment_size_x = segment_size_x[valid_seg]
        segment_size_y = segment_size_y[valid_seg]
        segment_edge_idx = segment_edge_idx[valid_seg]
        segment_is_horizontal = segment_is_horizontal[valid_seg]
        segment_weight = segment_weight[valid_seg]

        return {
            'segment_llx': segment_llx,
            'segment_lly': segment_lly,
            'segment_size_x': segment_size_x,
            'segment_size_y': segment_size_y,
            'segment_edge_idx': segment_edge_idx,
            'segment_is_horizontal': segment_is_horizontal,
            'segment_weight': segment_weight,
            'num_segments': segment_llx.numel()
        }

    def __call__(self, newx, newy, flat_from, flat_to, l_directions, soft_l_weights=None):
        """
        构建L形segments
        
        Returns:
            dict with:
                segment_pos: [num_segments * 2] 位置tensor
                segment_size_x: [num_segments]
                segment_size_y: [num_segments]
                ...
        """
        device = newx.device
        
        # 计算输入签名，检测EGR/GPUGR是否重新运行。内容hash只在cache miss时计算，
        # 避免每次forward重复扫描大规模edge tensor。
        current_key = self._compute_input_key(flat_from, flat_to, l_directions)
        
        # 检查是否需要重新计算拓扑
        need_recompute_topo = (
            self._cached_topology is None or
            self._cached_input_key != current_key
        )
        
        if need_recompute_topo:
            current_hash = self._compute_input_hash(flat_from, flat_to, l_directions)
            # 预计算拓扑（只需一次，或EGR更新后重新计算）
            self._cached_topology = self._compute_topology(
                flat_from, flat_to, l_directions, len(newx), device
            )
            self._cached_input_key = current_key
            self._cached_input_hash = current_hash
            if self.log_verbose >= 2:
                logger.info(
                    "Computed topology: %d valid edges (hash=%s)",
                    self._cached_topology["num_valid"],
                    current_hash[0][2] if current_hash and current_hash[0] else "none",
                )

        # 快速计算segment坐标
        if soft_l_weights is None:
            result = self._compute_segments_discrete(newx, newy, self._cached_topology, l_directions)
            mode_name = "discrete"
        else:
            result = self._compute_segments_soft(newx, newy, self._cached_topology, soft_l_weights)
            mode_name = "soft"

        # 只在第一次打印详细日志
        if self.log_verbose >= 2 and need_recompute_topo and result['num_segments'] > 0:
            logger.info(
                f"Built {result['num_segments']} segments from {flat_from.numel()} edges (vectorized, mode={mode_name})"
            )

        # 添加pos tensor
        if result['num_segments'] > 0:
            result['segment_pos'] = build_segment_pos_tensor(
                result['segment_llx'], result['segment_lly']
            )
        else:
            result['segment_pos'] = torch.tensor([], dtype=newx.dtype, device=newx.device)

        return result
