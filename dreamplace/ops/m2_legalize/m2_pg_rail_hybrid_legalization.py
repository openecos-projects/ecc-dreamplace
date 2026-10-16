from dataclasses import dataclass
import logging
import math

import numpy as np
import torch


def _positive_overlap(a_l, a_h, b_l, b_h):
    return max(0.0, min(a_h, b_h) - max(a_l, b_l))


def _normalize_rail_index(value, name, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be an integer" % name)
    minimum = 0 if allow_zero else 1
    if (
        not math.isfinite(numeric_value)
        or not numeric_value.is_integer()
        or numeric_value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be a %s integer" % (name, qualifier))
    return int(numeric_value)


def _node_name(placedb, node_id):
    names = getattr(placedb, "node_names", None)
    if names is None or node_id >= len(names):
        return str(node_id)
    name = names[node_id]
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def _subtract_rectangle(box, cut, tolerance):
    xl, yl, xh, yh = box
    cut_xl = max(xl, cut[0])
    cut_yl = max(yl, cut[1])
    cut_xh = min(xh, cut[2])
    cut_yh = min(yh, cut[3])
    if cut_xh <= cut_xl + tolerance or cut_yh <= cut_yl + tolerance:
        return [box]

    pieces = (
        (xl, yl, cut_xl, yh),
        (cut_xh, yl, xh, yh),
        (cut_xl, yl, cut_xh, cut_yl),
        (cut_xl, cut_yh, cut_xh, yh),
    )
    return [
        piece
        for piece in pieces
        if piece[2] > piece[0] + tolerance
        and piece[3] > piece[1] + tolerance
    ]


def _subtract_rectangles(boxes, cuts, tolerance):
    fragments = [tuple(float(value) for value in box) for box in boxes]
    for cut in cuts:
        next_fragments = []
        for fragment in fragments:
            next_fragments.extend(
                _subtract_rectangle(fragment, cut, tolerance)
            )
        fragments = next_fragments
    return fragments


@dataclass
class M2PgRailHybridLegalizationProblem:
    original_num_nodes: int
    num_nodes: int
    num_movable_nodes: int
    num_terminals: int
    num_terminal_NIs: int
    num_filler_nodes: int
    node_size_x: torch.Tensor
    node_size_y: torch.Tensor
    node_weights: torch.Tensor
    node2fence_region_map: torch.Tensor
    packed_pos: torch.Tensor
    remaining_movable_ids: torch.Tensor
    reserved_movable_ids: torch.Tensor
    reserved_packed_begin: int
    reserved_positions: torch.Tensor
    reserved_overlap_areas: torch.Tensor
    rail_boxes: torch.Tensor

    def restore_original_order(self, original_pos, legalized_pos):
        if original_pos.numel() != self.original_num_nodes * 2:
            raise ValueError(
                "Expected original position tensor with %d entries, got %d"
                % (self.original_num_nodes * 2, original_pos.numel())
            )
        if legalized_pos.numel() != self.num_nodes * 2:
            raise ValueError(
                "Expected hybrid position tensor with %d entries, got %d"
                % (self.num_nodes * 2, legalized_pos.numel())
            )

        restored = original_pos.detach().clone()
        original_x = restored[: self.original_num_nodes]
        original_y = restored[self.original_num_nodes :]
        legalized_x = legalized_pos[: self.num_nodes]
        legalized_y = legalized_pos[self.num_nodes :]
        remaining = self.remaining_movable_ids.to(original_pos.device)
        reserved = self.reserved_movable_ids.to(original_pos.device)

        original_x[remaining] = legalized_x[: self.num_movable_nodes].detach()
        original_y[remaining] = legalized_y[: self.num_movable_nodes].detach()
        if reserved.numel():
            begin = self.reserved_packed_begin
            end = begin + reserved.numel()
            original_x[reserved] = legalized_x[begin:end].detach()
            original_y[reserved] = legalized_y[begin:end].detach()
        return restored


@dataclass
class M2PgRailHybridLegalizationView:
    original_num_nodes: int
    original_num_movable_nodes: int
    original_num_terminals: int
    num_terminal_NIs: int
    num_filler_nodes: int
    rail_boxes: torch.Tensor
    full_rail_boxes: torch.Tensor
    max_rail_free_width: float
    xl: float
    yl: float
    xh: float
    yh: float
    site_width: float
    row_height: float
    movable_node_names: tuple
    legalization_mode: str
    hard_rail_start: int
    hard_rail_end: int

    @classmethod
    def create(
        cls,
        placedb,
        data_collections,
        hard_rail_start=None,
        hard_rail_end=None,
        legalization_mode="hybrid_hard",
    ):
        if legalization_mode not in ("hybrid_hard", "subset_hard"):
            raise ValueError(
                "legalization_mode must be 'hybrid_hard' or 'subset_hard'"
            )
        rail_boxes = data_collections.m2_pg_rail_density_boxes.detach().clone()
        if rail_boxes.numel() == 0:
            return None

        rail_boxes_np = rail_boxes.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        rail_centers_x = rail_boxes_np[:, 0] + rail_boxes_np[:, 2]
        rail_order = np.lexsort(
            (
                np.arange(rail_boxes_np.shape[0], dtype=np.int64),
                rail_boxes_np[:, 0],
                rail_boxes_np[:, 2],
                rail_centers_x,
            )
        )
        sorted_rail_boxes_np = rail_boxes_np[rail_order]
        full_rail_boxes = torch.as_tensor(
            sorted_rail_boxes_np,
            dtype=rail_boxes.dtype,
            device=rail_boxes.device,
        ).reshape(-1, 4)
        rail_count = int(full_rail_boxes.size(0))
        if (
            legalization_mode == "subset_hard"
            and hard_rail_start is None
            and hard_rail_end is None
        ):
            raise ValueError(
                "subset_hard legalization requires hard_rail_start and "
                "hard_rail_end"
            )
        if hard_rail_start is None and hard_rail_end is None:
            hard_rail_start = 1
            hard_rail_end = rail_count
        elif hard_rail_start is None or hard_rail_end is None:
            raise ValueError(
                "hard_rail_start and hard_rail_end must be provided together"
            )
        else:
            hard_rail_start = _normalize_rail_index(
                hard_rail_start, "m2_pg_rail_legalization_hard_rail_start"
            )
            hard_rail_end = _normalize_rail_index(
                hard_rail_end,
                "m2_pg_rail_legalization_hard_rail_end",
                allow_zero=True,
            )
        if hard_rail_start < 1 or hard_rail_end > rail_count:
            raise ValueError(
                "M2 hard rail range %d-%d is outside the %d available "
                "left-to-right rails"
                % (hard_rail_start, hard_rail_end, rail_count)
            )
        if hard_rail_end < hard_rail_start:
            raise ValueError(
                "M2 hard rail range end must be >= start, got %d-%d"
                % (hard_rail_start, hard_rail_end)
            )
        rail_boxes = full_rail_boxes[hard_rail_start - 1 : hard_rail_end].clone()

        num_nodes = int(placedb.num_nodes)
        num_movable = int(placedb.num_movable_nodes)
        num_terminals = int(placedb.num_terminals)
        num_terminal_NIs = int(placedb.num_terminal_NIs)
        num_fillers = int(placedb.num_filler_nodes)
        filler_begin = num_nodes - num_fillers
        if num_movable + num_terminals + num_terminal_NIs != filler_begin:
            raise RuntimeError(
                "Unexpected node ordering while building hybrid M2 "
                "legalization view"
            )

        site_width = float(placedb.site_width)
        row_height = float(placedb.row_height)
        if site_width <= 0 or row_height <= 0:
            raise ValueError("site width and row height must be positive")

        rail_boxes_np = rail_boxes.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        max_free_width = cls._max_site_aligned_rail_free_width(
            rail_boxes_np,
            float(placedb.xl),
            float(placedb.yl),
            float(placedb.xh),
            float(placedb.yh),
            site_width,
            row_height,
        )
        logging.info(
            "M2 %s rail selection: full_rail_boxes=%d hard_rail_boxes=%d "
            "hard_rail_range=%d-%d",
            legalization_mode.replace("_", "-"),
            rail_count,
            int(rail_boxes.size(0)),
            hard_rail_start,
            hard_rail_end,
        )
        logging.info(
            "M2 %s rail-only width telemetry: hard_rail_boxes=%d "
            "max_site_aligned_rail_free_width=%.6g",
            legalization_mode.replace("_", "-"),
            int(rail_boxes.size(0)),
            max_free_width,
        )

        return cls(
            original_num_nodes=num_nodes,
            original_num_movable_nodes=num_movable,
            original_num_terminals=num_terminals,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_fillers,
            rail_boxes=rail_boxes,
            full_rail_boxes=full_rail_boxes,
            max_rail_free_width=max_free_width,
            xl=float(placedb.xl),
            yl=float(placedb.yl),
            xh=float(placedb.xh),
            yh=float(placedb.yh),
            site_width=site_width,
            row_height=row_height,
            movable_node_names=tuple(
                _node_name(placedb, node_id)
                for node_id in range(num_movable)
            ),
            legalization_mode=legalization_mode,
            hard_rail_start=hard_rail_start,
            hard_rail_end=hard_rail_end,
        )

    @staticmethod
    def _coordinate_tolerance(xl, yl, xh, yh):
        return max(1.0, abs(float(xl)), abs(float(yl)), abs(float(xh)), abs(float(yh))) * 1e-7

    @classmethod
    def _max_site_aligned_rail_free_width(
        cls, rail_boxes, xl, yl, xh, yh, site_width, row_height
    ):
        tolerance = cls._coordinate_tolerance(xl, yl, xh, yh)
        num_rows = int(
            math.floor((yh - yl - row_height) / row_height + tolerance)
        ) + 1
        if num_rows <= 0:
            return 0.0

        max_width = 0.0
        for row_id in range(num_rows):
            row_yl = yl + row_id * row_height
            row_yh = row_yl + row_height
            intervals = []
            for rail_xl, rail_yl, rail_xh, rail_yh in rail_boxes:
                if _positive_overlap(row_yl, row_yh, rail_yl, rail_yh) <= tolerance:
                    continue
                clipped_xl = max(xl, float(rail_xl))
                clipped_xh = min(xh, float(rail_xh))
                if clipped_xh > clipped_xl + tolerance:
                    intervals.append((clipped_xl, clipped_xh))
            intervals.sort()

            merged = []
            for begin, end in intervals:
                if not merged or begin > merged[-1][1] + tolerance:
                    merged.append([begin, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)

            gap_begin = xl
            for begin, end in merged + [[xh, xh]]:
                aligned_begin = xl + math.ceil(
                    (gap_begin - xl) / site_width - tolerance
                ) * site_width
                max_width = max(max_width, begin - aligned_begin)
                gap_begin = max(gap_begin, end)
        return max(0.0, max_width)

    def prepare(self, pos, data_collections):
        if pos.numel() != self.original_num_nodes * 2:
            raise ValueError(
                "Expected original position tensor with %d entries, got %d"
                % (self.original_num_nodes * 2, pos.numel())
            )

        node_size_x = data_collections.node_size_x.detach()
        node_size_y = data_collections.node_size_y.detach()
        node_weights = data_collections.num_pins_in_nodes.detach()
        x = pos[: self.original_num_nodes].detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        y = pos[self.original_num_nodes :].detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        size_x = node_size_x.detach().cpu().numpy().astype(np.float64, copy=False)
        size_y = node_size_y.detach().cpu().numpy().astype(np.float64, copy=False)
        rail_boxes = self.rail_boxes.detach().cpu().numpy().astype(
            np.float64, copy=False
        )

        fixed_begin = self.original_num_movable_nodes
        fixed_end = fixed_begin + self.original_num_terminals
        fixed_boxes = [
            (x[node_id], y[node_id], x[node_id] + size_x[node_id], y[node_id] + size_y[node_id])
            for node_id in range(fixed_begin, fixed_end)
            if size_x[node_id] > 0 and size_y[node_id] > 0
        ]
        combined_hard_obstacles = np.concatenate(
            (
                rail_boxes,
                np.asarray(fixed_boxes, dtype=np.float64).reshape(-1, 4),
            ),
            axis=0,
        )
        max_hard_obstacle_free_width = (
            self._max_site_aligned_rail_free_width(
                combined_hard_obstacles,
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.site_width,
                self.row_height,
            )
        )
        tolerance = self._coordinate_tolerance(
            self.xl, self.yl, self.xh, self.yh
        )
        unavoidable_node_ids = tuple(
            node_id
            for node_id in range(self.original_num_movable_nodes)
            if size_y[node_id] <= self.row_height + tolerance
            and size_x[node_id] > max_hard_obstacle_free_width + tolerance
        )
        logging.info(
            "M2 %s width precheck: hard_rail_boxes=%d fixed_boxes=%d "
            "max_site_aligned_hard_obstacle_free_width=%.6g "
            "unavoidable_cells=%d",
            self.legalization_mode.replace("_", "-"),
            len(rail_boxes),
            len(fixed_boxes),
            max_hard_obstacle_free_width,
            len(unavoidable_node_ids),
        )
        for node_id in unavoidable_node_ids:
            logging.info(
                "M2 %s unavoidable cell: node_id=%d name=%s "
                "width=%.6g sites=%.6g",
                self.legalization_mode.replace("_", "-"),
                node_id,
                _node_name_from_view(self, node_id),
                float(size_x[node_id]),
                float(size_x[node_id]) / self.site_width,
            )
        reserved_boxes = []
        reserved_positions = []
        reserved_overlap_areas = []
        ordered_unavoidable = sorted(
            unavoidable_node_ids,
            key=lambda node_id: (-size_x[node_id], node_id),
        )
        for node_id in ordered_unavoidable:
            selected_x, selected_y, overlap_area = self._select_reservation(
                node_id=node_id,
                gp_x=x[node_id],
                gp_y=y[node_id],
                width=size_x[node_id],
                height=size_y[node_id],
                rail_boxes=rail_boxes,
                obstacle_boxes=fixed_boxes + reserved_boxes,
            )
            reserved_positions.append((selected_x, selected_y))
            reserved_overlap_areas.append(overlap_area)
            reserved_boxes.append(
                (
                    selected_x,
                    selected_y,
                    selected_x + size_x[node_id],
                    selected_y + size_y[node_id],
                )
            )
            logging.info(
                "M2 %s reservation: node_id=%d name=%s "
                "x=%.6g y=%.6g overlap_area=%.6g displacement=%.6g",
                self.legalization_mode.replace("_", "-"),
                node_id,
                _node_name_from_view(self, node_id),
                selected_x,
                selected_y,
                overlap_area,
                abs(selected_x - x[node_id]) + abs(selected_y - y[node_id]),
            )

        rail_fragments = _subtract_rectangles(
            rail_boxes, reserved_boxes, tolerance
        )
        reserved_set = set(ordered_unavoidable)
        remaining_ids_list = [
            node_id
            for node_id in range(self.original_num_movable_nodes)
            if node_id not in reserved_set
        ]
        terminal_ids_list = list(
            range(
                self.original_num_movable_nodes,
                self.original_num_movable_nodes + self.original_num_terminals,
            )
        )
        terminal_ni_begin = (
            self.original_num_movable_nodes + self.original_num_terminals
        )
        terminal_ni_ids_list = list(
            range(terminal_ni_begin, terminal_ni_begin + self.num_terminal_NIs)
        )
        filler_begin = self.original_num_nodes - self.num_filler_nodes
        filler_ids_list = list(range(filler_begin, self.original_num_nodes))

        device = pos.device
        remaining_ids = torch.as_tensor(
            remaining_ids_list, dtype=torch.long, device=device
        )
        terminal_ids = torch.as_tensor(
            terminal_ids_list, dtype=torch.long, device=device
        )
        reserved_ids = torch.as_tensor(
            ordered_unavoidable, dtype=torch.long, device=device
        )
        terminal_ni_ids = torch.as_tensor(
            terminal_ni_ids_list, dtype=torch.long, device=device
        )
        filler_ids = torch.as_tensor(
            filler_ids_list, dtype=torch.long, device=device
        )
        original_ids = torch.cat(
            (
                remaining_ids,
                terminal_ids,
                reserved_ids,
                terminal_ni_ids,
                filler_ids,
            )
        )
        rail_tensor = torch.as_tensor(
            rail_fragments,
            dtype=pos.dtype,
            device=device,
        ).reshape(-1, 4)
        rail_sizes_x = rail_tensor[:, 2] - rail_tensor[:, 0]
        rail_sizes_y = rail_tensor[:, 3] - rail_tensor[:, 1]
        num_rail_fragments = int(rail_tensor.size(0))
        rail_insert_index = (
            len(remaining_ids_list)
            + len(terminal_ids_list)
            + len(ordered_unavoidable)
        )

        packed_size_x = torch.cat(
            (
                node_size_x[original_ids[:rail_insert_index]],
                rail_sizes_x,
                node_size_x[original_ids[rail_insert_index:]],
            )
        )
        packed_size_y = torch.cat(
            (
                node_size_y[original_ids[:rail_insert_index]],
                rail_sizes_y,
                node_size_y[original_ids[rail_insert_index:]],
            )
        )
        rail_weights = node_weights.new_zeros(num_rail_fragments)
        packed_weights = torch.cat(
            (
                node_weights[original_ids[:rail_insert_index]],
                rail_weights,
                node_weights[original_ids[rail_insert_index:]],
            )
        )

        original_x = pos[: self.original_num_nodes].detach()
        original_y = pos[self.original_num_nodes :].detach()
        reserved_positions_tensor = torch.as_tensor(
            reserved_positions, dtype=pos.dtype, device=device
        ).reshape(-1, 2)
        packed_x_before_rails = original_x[original_ids[:rail_insert_index]].clone()
        packed_y_before_rails = original_y[original_ids[:rail_insert_index]].clone()
        reserved_packed_begin = len(remaining_ids_list) + len(terminal_ids_list)
        if reserved_ids.numel():
            reserved_packed_end = reserved_packed_begin + reserved_ids.numel()
            packed_x_before_rails[reserved_packed_begin:reserved_packed_end] = (
                reserved_positions_tensor[:, 0]
            )
            packed_y_before_rails[reserved_packed_begin:reserved_packed_end] = (
                reserved_positions_tensor[:, 1]
            )
        packed_x = torch.cat(
            (
                packed_x_before_rails,
                rail_tensor[:, 0],
                original_x[original_ids[rail_insert_index:]],
            )
        )
        packed_y = torch.cat(
            (
                packed_y_before_rails,
                rail_tensor[:, 1],
                original_y[original_ids[rail_insert_index:]],
            )
        )
        packed_pos = torch.cat((packed_x, packed_y))
        num_nodes = self.original_num_nodes + num_rail_fragments
        node2fence_region_map = data_collections.node2fence_region_map.new_zeros(
            num_nodes
        )
        logging.info(
            "M2 %s temporary view: original_nodes=%d "
            "legalizer_nodes=%d movable=%d->%d reserved=%d "
            "hard_rail_boxes=%d full_rail_boxes=%d rail_fragments=%d",
            self.legalization_mode.replace("_", "-"),
            self.original_num_nodes,
            num_nodes,
            self.original_num_movable_nodes,
            len(remaining_ids_list),
            len(ordered_unavoidable),
            int(self.rail_boxes.size(0)),
            int(self.full_rail_boxes.size(0)),
            num_rail_fragments,
        )

        return M2PgRailHybridLegalizationProblem(
            original_num_nodes=self.original_num_nodes,
            num_nodes=num_nodes,
            num_movable_nodes=len(remaining_ids_list),
            num_terminals=(
                self.original_num_terminals
                + len(ordered_unavoidable)
                + num_rail_fragments
            ),
            num_terminal_NIs=self.num_terminal_NIs,
            num_filler_nodes=self.num_filler_nodes,
            node_size_x=packed_size_x,
            node_size_y=packed_size_y,
            node_weights=packed_weights,
            node2fence_region_map=node2fence_region_map,
            packed_pos=packed_pos,
            remaining_movable_ids=remaining_ids,
            reserved_movable_ids=reserved_ids,
            reserved_packed_begin=reserved_packed_begin,
            reserved_positions=reserved_positions_tensor,
            reserved_overlap_areas=torch.as_tensor(
                reserved_overlap_areas, dtype=pos.dtype, device=device
            ),
            rail_boxes=rail_tensor,
        )

    def _select_reservation(
        self,
        node_id,
        gp_x,
        gp_y,
        width,
        height,
        rail_boxes,
        obstacle_boxes,
    ):
        tolerance = self._coordinate_tolerance(
            self.xl, self.yl, self.xh, self.yh
        )
        num_x = int(
            math.floor(
                (self.xh - self.xl - width) / self.site_width + tolerance
            )
        ) + 1
        num_y = int(
            math.floor(
                (self.yh - self.yl - height) / self.row_height + tolerance
            )
        ) + 1
        if num_x <= 0 or num_y <= 0:
            raise RuntimeError(
                "Cannot reserve unavoidable M2-overlap cell %d: cell does "
                "not fit inside the placement boundary" % node_id
            )

        x_starts = self.xl + np.arange(num_x, dtype=np.float64) * self.site_width
        y_starts = self.yl + np.arange(num_y, dtype=np.float64) * self.row_height
        overlap_area = np.zeros((num_y, num_x), dtype=np.float64)
        for rail_xl, rail_yl, rail_xh, rail_yh in rail_boxes:
            x_overlap = np.maximum(
                0.0,
                np.minimum(x_starts + width, rail_xh)
                - np.maximum(x_starts, rail_xl),
            )
            if not np.any(x_overlap > tolerance):
                continue
            y_overlap = np.maximum(
                0.0,
                np.minimum(y_starts + height, rail_yh)
                - np.maximum(y_starts, rail_yl),
            )
            if np.any(y_overlap > tolerance):
                overlap_area += y_overlap[:, None] * x_overlap[None, :]

        valid = np.ones((num_y, num_x), dtype=np.bool_)
        for obs_xl, obs_yl, obs_xh, obs_yh in obstacle_boxes:
            x_overlap = (x_starts < obs_xh - tolerance) & (
                x_starts + width > obs_xl + tolerance
            )
            y_overlap = (y_starts < obs_yh - tolerance) & (
                y_starts + height > obs_yl + tolerance
            )
            if np.any(x_overlap) and np.any(y_overlap):
                valid[np.ix_(y_overlap, x_overlap)] = False

        if not np.any(valid):
            raise RuntimeError(
                "Cannot reserve unavoidable M2-overlap cell %d: no "
                "boundary-, site-, row-, and fixed-obstacle-legal position"
                % node_id
            )
        minimum_overlap = float(np.min(overlap_area[valid]))
        area_tolerance = max(1.0, abs(minimum_overlap)) * 1e-9
        best_overlap = valid & (overlap_area <= minimum_overlap + area_tolerance)
        displacement = (
            np.abs(y_starts[:, None] - gp_y)
            + np.abs(x_starts[None, :] - gp_x)
        )
        displacement[~best_overlap] = np.inf
        flat_index = int(np.argmin(displacement))
        row_id, site_id = np.unravel_index(flat_index, displacement.shape)
        if minimum_overlap <= tolerance:
            raise RuntimeError(
                "Hybrid M2 width precheck marked cell %d unavoidable, but "
                "reservation found a zero-overlap position" % node_id
            )
        return (
            float(x_starts[site_id]),
            float(y_starts[row_id]),
            minimum_overlap,
        )

    def audit(self, pos, node_size_x, node_size_y, problem):
        x = pos[: self.original_num_nodes].detach().cpu().numpy().astype(
            np.float64, copy=False
        )[: self.original_num_movable_nodes]
        y = pos[self.original_num_nodes :].detach().cpu().numpy().astype(
            np.float64, copy=False
        )[: self.original_num_movable_nodes]
        width = node_size_x.detach().cpu().numpy().astype(
            np.float64, copy=False
        )[: self.original_num_movable_nodes]
        height = node_size_y.detach().cpu().numpy().astype(
            np.float64, copy=False
        )[: self.original_num_movable_nodes]
        def compute_overlap_area(rails):
            rails = rails.detach().cpu().numpy().astype(np.float64, copy=False)
            overlap_area = np.zeros(
                (self.original_num_movable_nodes, len(rails)), dtype=np.float64
            )
            for rail_id, (rail_xl, rail_yl, rail_xh, rail_yh) in enumerate(rails):
                x_overlap = np.maximum(
                    0.0, np.minimum(x + width, rail_xh) - np.maximum(x, rail_xl)
                )
                y_overlap = np.maximum(
                    0.0, np.minimum(y + height, rail_yh) - np.maximum(y, rail_yl)
                )
                overlap_area[:, rail_id] = x_overlap * y_overlap
            return overlap_area

        overlap_area = compute_overlap_area(self.rail_boxes)
        full_overlap_area = compute_overlap_area(self.full_rail_boxes)

        tolerance = self._coordinate_tolerance(
            self.xl, self.yl, self.xh, self.yh
        )
        overlap_mask = overlap_area > tolerance
        exempt = np.zeros(self.original_num_movable_nodes, dtype=np.bool_)
        reserved_ids = problem.reserved_movable_ids.detach().cpu().numpy()
        exempt[reserved_ids] = True
        non_exempt_mask = overlap_mask[~exempt]
        reserved_actual = overlap_area[reserved_ids].sum(axis=1)
        reserved_expected = (
            problem.reserved_overlap_areas.detach().cpu().numpy().astype(np.float64)
        )
        reserved_delta = np.abs(reserved_actual - reserved_expected)
        full_overlap_mask = full_overlap_area > tolerance
        full_non_exempt_mask = full_overlap_mask[~exempt]
        return {
            "non_exempt_overlap_count": int(np.count_nonzero(non_exempt_mask)),
            "non_exempt_overlap_area": float(overlap_area[~exempt].sum()),
            "full_non_exempt_overlap_count": int(
                np.count_nonzero(full_non_exempt_mask)
            ),
            "full_non_exempt_overlap_area": float(
                full_overlap_area[~exempt].sum()
            ),
            "reserved_overlap_count": int(np.count_nonzero(overlap_mask[exempt])),
            "reserved_overlap_area": float(overlap_area[exempt].sum()),
            "reserved_actual_areas": reserved_actual,
            "reserved_expected_areas": reserved_expected,
            "reserved_max_area_delta": (
                float(reserved_delta.max()) if reserved_delta.size else 0.0
            ),
        }


def _node_name_from_view(view, node_id):
    if node_id < 0 or node_id >= len(view.movable_node_names):
        return str(node_id)
    return view.movable_node_names[node_id]
