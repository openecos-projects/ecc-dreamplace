"""Row gaps as fixed blockages in legalization.

PlaceDB hands the placer one box, the union of the DEF ROWs, with one site width
and one row height, and the greedy and Abacus legalizers and the legality check
treat every row as spanning that box.  In superblue4 / 5 / 18 some ROWs start
later or end earlier than the box; the legalizer filled the part of the box those
rows do not cover with cells, outside every row site, which the ICCAD 2015
checker rejects (ILLEGAL_TYPE 4) and the legality check did not see.

Each run of consecutive rows with the same uncovered left or right part becomes
one fixed, pin-less node, only inside legalization: the legalizers cut their row
blanks at it and the legality check reports a movable cell that overlaps it.  A
node keeps a quarter row inside its run in y and a quarter site off the row edge
in x, so float rounding of the scaled coordinates cannot move a cut by a site.
The nodes go after the fixed cells, before the IO pins and fillers.  A design
whose rows all span the box gets none and its legalization is unchanged.
"""

import logging

import numpy as np
import torch

NO_FENCE_REGION = 2147483647  # node2fence_region_map of a node outside every fence region
MARGIN = 0.25  # of a row height (y) or a site width (x)


def row_gap_boxes(rows, xl, yl, xh, yh, site_width, row_height):
    """(K, 4) gap boxes (xl, yl, xh, yh) for placedb.rows ((R, 4) xl, yl, xh, yh)
    inside the row box (xl, yl, xh, yh).

    Supported: one row per row-height slot of the box, stacked without holes, each
    starting and ending on the box's site grid.  The rows are float32 in scaled
    units, so the slot and site counts are rounded (a tenth of a site or row is far
    above that rounding and far below a half) and the edges are rebuilt from the
    box, the legalizers' own grid.  Any other layout is logged and gets no boxes
    (legalization as before).
    """
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if rows.shape[0] == 0:
        return np.zeros((0, 4))
    rows = rows[np.argsort(rows[:, 1], kind="stable")]
    counts = [
        (rows[:, 1] - yl) / row_height,
        (rows[:, 0] - xl) / site_width,
        (xh - rows[:, 2]) / site_width,
        np.asarray([(yh - yl) / row_height]),
    ]
    rounded = [np.rint(v) for v in counts]
    slots, left, right, height = rounded
    if (
        any(np.abs(v - r).max() > 0.1 for v, r in zip(counts, rounded))
        or int(height[0]) != len(rows)
        or not np.array_equal(slots, np.arange(len(rows)))
        or left.min() < 0
        or right.min() < 0
    ):
        logging.warning("rows are not one row per slot on the site grid; row gaps not modeled")
        return np.zeros((0, 4))
    boxes, start = [], 0
    for end in range(1, len(rows) + 1):
        if end < len(rows) and left[end] == left[start] and right[end] == right[start]:
            continue
        box_yl = yl + (start + MARGIN) * row_height
        box_yh = yl + (end - MARGIN) * row_height
        if left[start] > 0:
            boxes.append((xl - MARGIN * site_width, box_yl, xl + (left[start] - MARGIN) * site_width, box_yh))
        if right[start] > 0:
            boxes.append((xh - (right[start] - MARGIN) * site_width, box_yl, xh + MARGIN * site_width, box_yh))
        start = end
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


class RowGapBlockages:
    """Legalization tensors with the gap boxes inserted as fixed nodes, and the
    position layout change in and out of them."""

    def __init__(self, boxes, placedb, data_collections):
        size_x = data_collections.node_size_x
        boxes = torch.as_tensor(np.asarray(boxes), dtype=size_x.dtype, device=size_x.device)
        self.count = int(boxes.shape[0])
        self.index = placedb.num_movable_nodes + placedb.num_terminals
        self.num_nodes = placedb.num_nodes  # entries per axis in a position tensor
        self.num_terminals = placedb.num_terminals + self.count
        self.x, self.y = boxes[:, 0], boxes[:, 1]
        self.node_size_x = self.insert(size_x, boxes[:, 2] - boxes[:, 0])
        self.node_size_y = self.insert(data_collections.node_size_y, boxes[:, 3] - boxes[:, 1])
        self.num_pins_in_nodes = self.insert(data_collections.num_pins_in_nodes, torch.zeros_like(self.x))
        fence = data_collections.node2fence_region_map
        if fence.numel() >= self.index:  # covers the fixed range (not only the movable cells)
            fence = self.insert(
                fence,
                torch.full((self.count,), NO_FENCE_REGION, dtype=fence.dtype, device=fence.device),
            )
        self.node2fence_region_map = fence

    def insert(self, values, extra):
        return torch.cat((values[: self.index], extra.to(values), values[self.index :]))

    def extend(self, pos):
        x, y, i = pos[: self.num_nodes], pos[self.num_nodes :], self.index
        return torch.cat((x[:i], self.x.to(pos), x[i:], y[:i], self.y.to(pos), y[i:]))

    def strip(self, pos):
        n, i, k = self.num_nodes + self.count, self.index, self.count
        x, y = pos[:n], pos[n:]
        return torch.cat((x[:i], x[i + k :], y[:i], y[i + k :]))


def row_gap_blockages(placedb, data_collections):
    """The design's RowGapBlockages, None if every row spans the row box (built once per data collection)."""
    cached = getattr(data_collections, "row_gap_blockages", False)
    if cached is not False:
        return cached
    rows = getattr(placedb, "rows", None)
    gaps = None
    if rows is not None and len(rows):
        boxes = row_gap_boxes(
            rows,
            placedb.xl,
            placedb.yl,
            placedb.xh,
            placedb.yh,
            placedb.site_width,
            placedb.row_height,
        )
        gaps = RowGapBlockages(boxes, placedb, data_collections) if len(boxes) else None
    if gaps is not None:
        logging.info("row gaps: %d fixed blockage node(s) during legalization", gaps.count)
    data_collections.row_gap_blockages = gaps
    return gaps
