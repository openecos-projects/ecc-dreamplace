"""Safe row-segment repair for standard-cell legalization fallback."""

import math

import torch


def repair_row_segments(pos, placedb, data_collections):
    """Pack single-row movable cells around fixed row obstacles.

    The native Greedy legalizer can leave a small number of cells unplaced
    when fixed boundary/endcap cells fragment otherwise sufficient capacity.
    This fallback performs a deterministic row/segment assignment in Python,
    then lets the normal legality check decide whether the candidate is usable.
    It deliberately declines multi-row cells, fillers, and fence regions.
    """
    if len(placedb.regions) > 0 or placedb.num_filler_nodes > 0:
        return None

    num_movable = int(placedb.num_movable_nodes)
    if pos.ndim == 1:
        if pos.numel() % 2 != 0:
            return None
        num_nodes = int(pos.numel() // 2)
    elif pos.ndim == 2 and pos.shape[0] == 2:
        num_nodes = int(pos.shape[1])
    else:
        return None
    if num_movable <= 0 or num_nodes <= num_movable:
        return None

    pos_cpu = pos.detach().to(device="cpu", dtype=torch.float64)
    size_x = data_collections.node_size_x.detach().to(
        device="cpu", dtype=torch.float64
    )
    size_y = data_collections.node_size_y.detach().to(
        device="cpu", dtype=torch.float64
    )
    if pos_cpu.ndim == 1:
        x_cpu = pos_cpu[:num_nodes]
        y_cpu = pos_cpu[num_nodes:]
    else:
        x_cpu = pos_cpu[0]
        y_cpu = pos_cpu[1]
    if size_x.numel() < num_nodes or size_y.numel() < num_nodes:
        return None
    if not bool(
        torch.isfinite(x_cpu[:num_movable]).all()
        and torch.isfinite(y_cpu[:num_movable]).all()
    ):
        return None

    xl = float(placedb.xl)
    yl = float(placedb.yl)
    xh = float(placedb.xh)
    yh = float(placedb.yh)
    site_width = float(placedb.site_width)
    row_height = float(placedb.row_height)
    if site_width <= 0 or row_height <= 0 or xh <= xl or yh <= yl:
        return None

    movable_height = size_y[:num_movable]
    if not bool(
        torch.allclose(
            movable_height,
            movable_height.new_full(movable_height.shape, row_height),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        return None

    row_count = max(1, int(math.ceil((yh - yl) / row_height - 1e-9)))
    fixed_end = num_nodes
    intervals = [[] for _ in range(row_count)]
    for node_id in range(num_movable, fixed_end):
        fy = float(y_cpu[node_id])
        fh = float(size_y[node_id])
        if fy + fh <= yl or fy >= yh:
            continue
        row_first = max(0, int(math.floor((fy - yl) / row_height)))
        row_last = min(
            row_count - 1,
            int(math.ceil((fy + fh - yl) / row_height) - 1),
        )
        fx = float(x_cpu[node_id])
        fw = float(size_x[node_id])
        for row_id in range(row_first, row_last + 1):
            intervals[row_id].append((max(xl, fx), min(xh, fx + fw)))

    segments = []
    for row_id, row_intervals in enumerate(intervals):
        merged = []
        for left, right in sorted(row_intervals):
            if right <= left:
                continue
            if merged and left <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        cursor = xl
        free_intervals = []
        for left, right in merged:
            if left > cursor + 1e-6:
                free_intervals.append((cursor, left))
            cursor = max(cursor, right)
        if cursor < xh - 1e-6:
            free_intervals.append((cursor, xh))
        for left, right in free_intervals:
            left = math.ceil((left - xl) / site_width - 1e-9) * site_width + xl
            right = math.floor((right - xl) / site_width + 1e-9) * site_width + xl
            if right - left >= site_width - 1e-6:
                segments.append(
                    {
                        "row": row_id,
                        "left": left,
                        "right": right,
                        "cursor": left,
                    }
                )

    if not segments:
        return None

    def assign_cells(compact):
        row_segments = {}
        for segment in segments:
            row_segments.setdefault(segment["row"], []).append(
                {
                    "row": segment["row"],
                    "left": segment["left"],
                    "right": segment["right"],
                    "cursor": segment["left"],
                }
            )

        row_cells = [[] for _ in range(row_count)]
        for node_id in range(num_movable):
            preferred_row = int(round((float(y_cpu[node_id]) - yl) / row_height))
            preferred_row = max(0, min(row_count - 1, preferred_row))
            row_cells[preferred_row].append(node_id)

        assignments = {}
        pending = []
        ordered_rows = range(row_count)
        for row_id in ordered_rows:
            cell_ids = row_cells[row_id]
            if compact:
                cell_ids.sort(
                    key=lambda node_id: (
                        -math.ceil(float(size_x[node_id]) / site_width - 1e-9),
                        float(x_cpu[node_id]),
                    )
                )
            else:
                cell_ids.sort(key=lambda node_id: (float(x_cpu[node_id]), node_id))
            for node_id in cell_ids:
                width = math.ceil(
                    float(size_x[node_id]) / site_width - 1e-9
                ) * site_width
                current_x = float(x_cpu[node_id])
                candidates = []
                for segment in row_segments.get(row_id, []):
                    if segment["right"] - segment["cursor"] < width - 1e-6:
                        continue
                    desired_x = (
                        math.floor((current_x - xl) / site_width + 1e-9)
                        * site_width
                        + xl
                    )
                    desired_x = max(
                        segment["left"],
                        min(desired_x, segment["right"] - width),
                    )
                    assigned_x = (
                        segment["cursor"]
                        if compact
                        else max(segment["cursor"], desired_x)
                    )
                    if assigned_x + width > segment["right"] + 1e-6:
                        continue
                    candidates.append(
                        (abs(assigned_x - current_x), segment, assigned_x)
                    )
                if not candidates:
                    pending.append(node_id)
                    continue
                _, segment, assigned_x = min(
                    candidates,
                    key=lambda item: (item[0], item[1]["right"] - item[1]["cursor"]),
                )
                assignments[node_id] = (
                    assigned_x,
                    yl + row_id * row_height,
                )
                segment["cursor"] = assigned_x + width

        pending.sort(
            key=lambda node_id: (
                -math.ceil(float(size_x[node_id]) / site_width - 1e-9),
                abs(float(y_cpu[node_id]) - yl),
                float(x_cpu[node_id]),
            )
        )
        if pending:
            all_segments = [
                segment
                for row_segments_for_row in row_segments.values()
                for segment in row_segments_for_row
            ]
            for node_id in pending:
                width = math.ceil(
                    float(size_x[node_id]) / site_width - 1e-9
                ) * site_width
                current_x = float(x_cpu[node_id])
                current_y = float(y_cpu[node_id])
                candidates = []
                for segment in all_segments:
                    if segment["right"] - segment["cursor"] < width - 1e-6:
                        continue
                    desired_x = (
                        math.floor((current_x - xl) / site_width + 1e-9)
                        * site_width
                        + xl
                    )
                    desired_x = max(
                        segment["left"],
                        min(desired_x, segment["right"] - width),
                    )
                    assigned_x = (
                        segment["cursor"]
                        if compact
                        else max(segment["cursor"], desired_x)
                    )
                    if assigned_x + width > segment["right"] + 1e-6:
                        continue
                    assigned_y = yl + segment["row"] * row_height
                    candidates.append(
                        (
                            abs(assigned_y - current_y),
                            abs(assigned_x - current_x),
                            segment,
                            assigned_x,
                        )
                    )
                if not candidates:
                    continue
                _, _, segment, assigned_x = min(
                    candidates,
                    key=lambda item: (item[0], item[1]),
                )
                assignments[node_id] = (
                    assigned_x,
                    yl + segment["row"] * row_height,
                )
                segment["cursor"] = assigned_x + width

        if len(assignments) != num_movable:
            return None
        return assignments

    assignments = assign_cells(compact=False)
    if assignments is None:
        assignments = assign_cells(compact=True)
    if assignments is None:
        return None

    repaired = pos_cpu.clone()
    if repaired.ndim == 1:
        repaired_x = repaired[:num_nodes]
        repaired_y = repaired[num_nodes:]
    else:
        repaired_x = repaired[0]
        repaired_y = repaired[1]
    for node_id, (assigned_x, assigned_y) in assignments.items():
        repaired_x[node_id] = assigned_x
        repaired_y[node_id] = assigned_y
    return repaired.to(device=pos.device, dtype=pos.dtype)
