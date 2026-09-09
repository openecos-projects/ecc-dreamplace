import math

import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def build_smoothed_overflow_map(
    horizontal_overflow,
    vertical_overflow,
    smooth_kernel=3,
):
    horizontal = torch.as_tensor(horizontal_overflow).detach().to(
        device="cpu", dtype=torch.float32
    )
    vertical = torch.as_tensor(vertical_overflow).detach().to(
        device="cpu", dtype=torch.float32
    )
    if horizontal.dim() != 2 or vertical.shape != horizontal.shape:
        raise ValueError("directional overflow maps must have the same 2D shape")

    congestion = torch.maximum(horizontal.clamp(min=0), vertical.clamp(min=0))
    kernel = int(smooth_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("smooth_kernel must be a positive odd integer")
    if kernel == 1:
        return congestion.contiguous()

    # The gpugr tensors use [x, y]. Pooling is isotropic, so preserve that
    # convention and only add batch/channel dimensions.
    return F.avg_pool2d(
        congestion.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
        count_include_pad=False,
    ).squeeze(0).squeeze(0).contiguous()


def score_cells_from_overflow(
    pos,
    node_size_x,
    node_size_y,
    num_nodes,
    num_movable_nodes,
    overflow_xy,
    grid_xl,
    grid_yl,
    grid_xh,
    grid_yh,
):
    overflow = _as_numpy(overflow_xy)
    if overflow.ndim != 2:
        raise ValueError("overflow_xy must be a 2D [x, y] map")
    num_bins_x, num_bins_y = overflow.shape
    if num_bins_x <= 0 or num_bins_y <= 0:
        raise ValueError("overflow_xy must not be empty")

    die_w = float(grid_xh - grid_xl)
    die_h = float(grid_yh - grid_yl)
    if die_w <= 0 or die_h <= 0:
        raise ValueError("invalid routing-grid bounds")
    bin_w = die_w / num_bins_x
    bin_h = die_h / num_bins_y

    pos_np = _as_numpy(pos)
    size_x = _as_numpy(node_size_x)
    size_y = _as_numpy(node_size_y)
    node_x = pos_np[:num_movable_nodes]
    node_y = pos_np[num_nodes : num_nodes + num_movable_nodes]
    scores = np.zeros(num_movable_nodes, dtype=np.float32)

    for node_id in range(num_movable_nodes):
        width = float(size_x[node_id])
        height = float(size_y[node_id])
        if width <= 0 or height <= 0:
            continue
        xl = float(node_x[node_id])
        yl = float(node_y[node_id])
        xh = xl + width
        yh = yl + height
        bin_xl = int(math.floor((xl - grid_xl) / bin_w))
        bin_yl = int(math.floor((yl - grid_yl) / bin_h))
        bin_xh = int(math.ceil((xh - grid_xl) / bin_w) - 1)
        bin_yh = int(math.ceil((yh - grid_yl) / bin_h) - 1)
        bin_xl = min(max(bin_xl, 0), num_bins_x - 1)
        bin_xh = min(max(bin_xh, 0), num_bins_x - 1)
        bin_yl = min(max(bin_yl, 0), num_bins_y - 1)
        bin_yh = min(max(bin_yh, 0), num_bins_y - 1)
        if bin_xh < bin_xl or bin_yh < bin_yl:
            continue
        scores[node_id] = float(
            overflow[bin_xl : bin_xh + 1, bin_yl : bin_yh + 1].max()
        )
    return scores


def _merge_intervals(intervals, lower, upper):
    clipped = []
    for interval_xl, interval_xh in intervals:
        interval_xl = max(float(lower), float(interval_xl))
        interval_xh = min(float(upper), float(interval_xh))
        if interval_xh > interval_xl:
            clipped.append((interval_xl, interval_xh))
    clipped.sort()
    merged = []
    for interval_xl, interval_xh in clipped:
        if not merged or interval_xl > merged[-1][1]:
            merged.append([interval_xl, interval_xh])
        else:
            merged[-1][1] = max(merged[-1][1], interval_xh)
    return [(interval_xl, interval_xh) for interval_xl, interval_xh in merged]


def _segments_from_blockages(blockages, xl, xh):
    segments = []
    cursor = float(xl)
    for blockage_xl, blockage_xh in _merge_intervals(blockages, xl, xh):
        if blockage_xl > cursor:
            segments.append((cursor, blockage_xl))
        cursor = max(cursor, blockage_xh)
    if cursor < xh:
        segments.append((cursor, float(xh)))
    return segments


def allocate_padding_sites(
    scores,
    pos,
    node_size_x,
    node_size_y,
    num_nodes,
    num_movable_nodes,
    num_physical_nodes,
    xl,
    yl,
    xh,
    yh,
    site_width,
    row_height,
    hot_cell_ratio=0.2,
    row_free_ratio=0.5,
    max_padding_sites=1,
    eligible_mask=None,
):
    if not 0 <= hot_cell_ratio <= 1:
        raise ValueError("hot_cell_ratio must be in [0, 1]")
    if not 0 <= row_free_ratio <= 1:
        raise ValueError("row_free_ratio must be in [0, 1]")
    max_padding_sites = int(max_padding_sites)
    if max_padding_sites < 0:
        raise ValueError("max_padding_sites must be non-negative")
    if site_width <= 0 or row_height <= 0:
        raise ValueError("site_width and row_height must be positive")

    score_np = np.asarray(scores, dtype=np.float64)
    if score_np.shape != (num_movable_nodes,):
        raise ValueError("scores must have one entry per movable node")
    pos_np = _as_numpy(pos)
    size_x = _as_numpy(node_size_x)
    size_y = _as_numpy(node_size_y)
    node_x = pos_np[:num_nodes]
    node_y = pos_np[num_nodes : 2 * num_nodes]

    if eligible_mask is None:
        eligible = np.ones(num_movable_nodes, dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool).copy()
        if eligible.shape != (num_movable_nodes,):
            raise ValueError("eligible_mask must have one entry per movable node")
    eligible &= size_x[:num_movable_nodes] > 0
    eligible &= size_y[:num_movable_nodes] > 0
    eligible &= size_y[:num_movable_nodes] <= row_height * 1.01

    num_rows = max(1, int(math.ceil((yh - yl) / row_height)))
    fixed_intervals = [[] for _ in range(num_rows)]
    fixed_end = min(int(num_physical_nodes), int(num_nodes))
    for node_id in range(num_movable_nodes, fixed_end):
        width = float(size_x[node_id])
        height = float(size_y[node_id])
        if width <= 0 or height <= 0:
            continue
        row_begin = max(
            0, int(math.floor((float(node_y[node_id]) - yl) / row_height))
        )
        row_end = min(
            num_rows - 1,
            int(
                math.ceil(
                    (float(node_y[node_id]) + height - yl) / row_height
                )
                - 1
            ),
        )
        for row_id in range(row_begin, row_end + 1):
            fixed_intervals[row_id].append(
                (float(node_x[node_id]), float(node_x[node_id]) + width)
            )

    row_segments = [
        _segments_from_blockages(intervals, xl, xh)
        for intervals in fixed_intervals
    ]
    segment_used_sites = {}
    node_segment_keys = [None] * num_movable_nodes

    for node_id in range(num_movable_nodes):
        width = float(size_x[node_id])
        height = float(size_y[node_id])
        if width <= 0 or height <= 0:
            continue
        center_x = float(node_x[node_id]) + width * 0.5
        row_begin = max(
            0, int(math.floor((float(node_y[node_id]) - yl) / row_height))
        )
        row_end = min(
            num_rows - 1,
            int(
                math.ceil(
                    (float(node_y[node_id]) + height - yl) / row_height
                )
                - 1
            ),
        )
        base_key = None
        used_sites = int(math.ceil(width / site_width - 1.0e-9))
        for row_id in range(row_begin, row_end + 1):
            segment_id = None
            for candidate_id, (segment_xl, segment_xh) in enumerate(
                row_segments[row_id]
            ):
                if segment_xl <= center_x <= segment_xh:
                    segment_id = candidate_id
                    break
            if segment_id is None:
                continue
            key = (row_id, segment_id)
            if base_key is None:
                base_key = key
            segment_used_sites[key] = segment_used_sites.get(key, 0) + used_sites
        if base_key is None:
            eligible[node_id] = False
            continue
        node_segment_keys[node_id] = base_key

    segment_budgets = {}
    total_free_sites = 0
    total_budget_sites = 0
    for row_id, segments in enumerate(row_segments):
        for segment_id, (segment_xl, segment_xh) in enumerate(segments):
            key = (row_id, segment_id)
            capacity_sites = int(
                math.floor((segment_xh - segment_xl) / site_width + 1.0e-9)
            )
            free_sites = max(
                0, capacity_sites - segment_used_sites.get(key, 0)
            )
            budget_sites = int(math.floor(free_sites * row_free_ratio))
            segment_budgets[key] = budget_sites
            total_free_sites += free_sites
            total_budget_sites += budget_sites

    eligible_ids = np.flatnonzero(eligible)
    positive_ids = eligible_ids[score_np[eligible_ids] > 0]
    requested_hot_count = int(math.ceil(len(eligible_ids) * hot_cell_ratio))
    requested_hot_count = min(requested_hot_count, len(positive_ids))
    if requested_hot_count > 0:
        ranked_ids = positive_ids[
            np.argsort(-score_np[positive_ids], kind="stable")
        ][:requested_hot_count]
    else:
        ranked_ids = np.zeros(0, dtype=np.int64)

    padding_sites = np.zeros(num_movable_nodes, dtype=np.int32)
    allocated_ids = []
    for node_id in ranked_ids:
        key = node_segment_keys[int(node_id)]
        if key is None:
            continue
        available_sites = segment_budgets.get(key, 0)
        per_side_sites = min(max_padding_sites, available_sites // 2)
        if per_side_sites <= 0:
            continue
        padding_sites[int(node_id)] = per_side_sites
        segment_budgets[key] -= 2 * per_side_sites
        allocated_ids.append(int(node_id))

    return {
        "padding_sites": padding_sites,
        "scores": score_np.astype(np.float32),
        "ranked_candidate_ids": np.asarray(ranked_ids, dtype=np.int64),
        "allocated_ids": np.asarray(allocated_ids, dtype=np.int64),
        "eligible_count": int(len(eligible_ids)),
        "positive_score_count": int(len(positive_ids)),
        "requested_hot_count": int(requested_hot_count),
        "allocated_count": int(len(allocated_ids)),
        "total_added_sites": int(2 * padding_sites.sum()),
        "total_free_sites": int(total_free_sites),
        "total_budget_sites": int(total_budget_sites),
        "max_score": float(score_np.max()) if score_np.size else 0.0,
        "min_allocated_score": (
            float(score_np[allocated_ids].min()) if allocated_ids else 0.0
        ),
    }


def compute_cell_box_overlap_stats(
    pos,
    node_size_x,
    node_size_y,
    boxes,
    num_nodes,
    num_movable_nodes,
    chunk_size=4096,
):
    boxes_cpu = torch.as_tensor(boxes).detach().to(
        device="cpu", dtype=torch.float64
    ).reshape(-1, 4)
    if boxes_cpu.numel() == 0:
        return {"overlap_count": 0, "overlap_area": 0.0}
    pos_cpu = torch.as_tensor(pos).detach().to(device="cpu", dtype=torch.float64)
    size_x = torch.as_tensor(node_size_x).detach().to(
        device="cpu", dtype=torch.float64
    )
    size_y = torch.as_tensor(node_size_y).detach().to(
        device="cpu", dtype=torch.float64
    )
    x = pos_cpu[:num_movable_nodes]
    y = pos_cpu[num_nodes : num_nodes + num_movable_nodes]
    overlap_count = 0
    overlap_area = 0.0
    for begin in range(0, num_movable_nodes, int(chunk_size)):
        end = min(num_movable_nodes, begin + int(chunk_size))
        cell_xl = x[begin:end].unsqueeze(1)
        cell_yl = y[begin:end].unsqueeze(1)
        cell_xh = cell_xl + size_x[begin:end].unsqueeze(1)
        cell_yh = cell_yl + size_y[begin:end].unsqueeze(1)
        overlap_w = (
            torch.minimum(cell_xh, boxes_cpu[:, 2])
            - torch.maximum(cell_xl, boxes_cpu[:, 0])
        ).clamp(min=0)
        overlap_h = (
            torch.minimum(cell_yh, boxes_cpu[:, 3])
            - torch.maximum(cell_yl, boxes_cpu[:, 1])
        ).clamp(min=0)
        overlap = overlap_w * overlap_h
        overlap_count += int((overlap > 0).sum().item())
        overlap_area += float(overlap.sum().item())
    return {"overlap_count": overlap_count, "overlap_area": overlap_area}
