import torch

import dreamplace.ops.m2_soft_legalize.m2_soft_legalize_cpp as m2_soft_legalize_cpp


STAT_FIELDS = (
    "overlap_count_before",
    "overlap_area_before",
    "overlap_count_after",
    "overlap_area_after",
    "moved_count",
    "total_displacement",
    "max_displacement",
    "move_events",
    "skipped_illegal_rows",
    "accepted_candidates",
    "rejected_candidates",
    "passes_executed",
)


class M2SoftLegalize:
    """Reduce cell-rail overlap while preserving an ordinary legal placement."""

    def __init__(
        self,
        node_size_x,
        node_size_y,
        rail_boxes,
        xl,
        yl,
        xh,
        yh,
        site_width,
        row_height,
        num_movable_nodes,
        num_terminals,
        displacement_weight=0.01,
    ):
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.rail_boxes = rail_boxes
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.site_width = site_width
        self.row_height = row_height
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.displacement_weight = displacement_weight

    @staticmethod
    def _cpu_like(tensor, reference):
        return tensor.detach().to(device="cpu", dtype=reference.dtype).contiguous()

    def __call__(self, pos):
        with torch.no_grad():
            cpu_pos = pos.detach().to(device="cpu").contiguous()
            output, raw_stats = m2_soft_legalize_cpp.forward(
                cpu_pos,
                self._cpu_like(self.node_size_x, cpu_pos),
                self._cpu_like(self.node_size_y, cpu_pos),
                self._cpu_like(self.rail_boxes, cpu_pos),
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.site_width,
                self.row_height,
                self.num_movable_nodes,
                self.num_terminals,
                self.displacement_weight,
            )
            stats = {
                name: raw_stats[index].item()
                for index, name in enumerate(STAT_FIELDS)
            }
            return output.to(device=pos.device, dtype=pos.dtype), stats
