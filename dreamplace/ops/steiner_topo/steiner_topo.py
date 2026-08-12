# @file   steiner_topo.py
# @author
# @date   Mar 2025
# @brief  Get steiner tree topology & steiner node locations
#

import torch
from torch import nn
from torch.autograd import Function
import logging
import bisect
from pathlib import Path

import dreamplace.ops.steiner_topo.steiner_topo_cpp as steiner_topo_cpp
import dreamplace.configure as configure
# if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
#     import dreamplace.ops.steiner_topo.steiner_topo_cuda as steiner_topo_cuda
#     import dreamplace.ops.steiner_topo.steiner_topo_cuda_segment as steiner_topo_cuda_segment

logger = logging.getLogger(__name__)

_FLUTE_LUT_DIR = Path(__file__).resolve().parents[3] / "thirdparty" / "flute" / "lut.ICCAD2015"
_FLUTE_POWV_FILE = _FLUTE_LUT_DIR / "POWV9.dat"
_FLUTE_POST_FILE = _FLUTE_LUT_DIR / "POST9.dat"


class SteinerTopoFunction(Function):
    @staticmethod
    def forward(ctx, pos, pin_relate_x, pin_relate_y,
                net_vertex_start, num_vertices, deterministic_flag=False):

        updated_newx, updated_newy = steiner_topo_cpp.forward(
            pos,
            pin_relate_x.contiguous(),
            pin_relate_y.contiguous(),
            num_vertices,
            bool(deterministic_flag)
        )

        ctx.save_for_backward(pos,
                              net_vertex_start.contiguous(),
                              pin_relate_x.contiguous(),
                              pin_relate_y.contiguous()
                              )

        return updated_newx, updated_newy

    @staticmethod
    def backward(ctx, grad_newx, grad_newy):

        pos, net_vertex_start, pin_relate_x, pin_relate_y = ctx.saved_tensors
        grad_pos = steiner_topo_cpp.backward(
            grad_newx,
            grad_newy,
            pos,
            pin_relate_x,
            pin_relate_y
        )

        return grad_pos, None, None, None, None, None


class SteinerTopoDeterministicFunction(Function):
    @staticmethod
    def forward(ctx, pos, pin_relate_x, pin_relate_y, num_vertices):
        num_pins = pos.numel() // 2
        relate_x = pin_relate_x[:num_vertices].contiguous()
        relate_y = pin_relate_y[:num_vertices].contiguous()
        idx_x = relate_x.to(dtype=torch.long)
        idx_y = relate_y.to(dtype=torch.long)

        updated_newx = pos[:num_pins].index_select(0, idx_x).contiguous()
        updated_newy = pos[num_pins:].index_select(0, idx_y).contiguous()

        ctx.save_for_backward(pos, relate_x, relate_y)
        return updated_newx, updated_newy

    @staticmethod
    def backward(ctx, grad_newx, grad_newy):
        pos, pin_relate_x, pin_relate_y = ctx.saved_tensors
        grad_pos = steiner_topo_cpp.backward(
            grad_newx.contiguous(),
            grad_newy.contiguous(),
            pos,
            pin_relate_x,
            pin_relate_y
        )
        return grad_pos, None, None, None


class SteinerTopo(nn.Module):
    
    # L方向常量
    H_FIRST = 0       # 先水平后垂直 (Horizontal First)
    V_FIRST = 1       # 先垂直后水平 (Vertical First)
    STRAIGHT = 2      # 直线（水平或垂直）
    FAKE_STRAIGHT = 3 # 伪直线（在gcell下只有一条wire）
    UNKNOWN = -1      # 未知
    
    def __init__(self,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 ignore_net_degree=None,
                 algorithm="FLUTE",
                 deterministic_flag=False,
                 collect_edge_geometry_stats=False):
        super(SteinerTopo, self).__init__()
        # Register buffers
        self.register_buffer('flat_net2pin_map', flat_net2pin_map.contiguous())
        self.register_buffer('flat_net2pin_start_map',
                             flat_net2pin_start_map.contiguous())

        # Set ignore degree threshold
        self.ignore_net_degree = ignore_net_degree if ignore_net_degree is not None else flat_net2pin_map.numel()

        self.newx = None
        self.newy = None
        self.pin_relate_x = None
        self.pin_relate_y = None
        self.net_vertex_start = None
        self.net_steiner_start = None
        self.pin_fa = None
        self.flat_pin_to = None
        self.flat_pin_from = None
        self.flat_pin_to_start = None
        self.net_flat_topo_sort = None
        self.net_flat_topo_sort_start = None
        self.num_vertices = None

        self.algorithm = algorithm
        self.deterministic_flag = bool(deterministic_flag)
        self.collect_edge_geometry_stats = bool(collect_edge_geometry_stats)
        self.last_edge_geometry_stats = None
        
        # L方向相关
        self.edge_l_directions = None  # 每条边的L方向
        self.l_direction_resolver = None  # EGR L方向解析器

    def forward(self, pos):

        if self.pin_relate_x is None or self.pin_relate_y is None \
           or self.net_vertex_start is None \
           or self.num_vertices is None:
            raise RuntimeError(
                "SteinerTopo topology not initialized. Call rebuild_tree and update_topology first.")

        if self.deterministic_flag:
            return SteinerTopoDeterministicFunction.apply(
                pos,
                self.pin_relate_x,
                self.pin_relate_y,
                self.num_vertices
            )

        updated_newx, updated_newy = SteinerTopoFunction.apply(
            pos,
            self.pin_relate_x,
            self.pin_relate_y,
            self.net_vertex_start,
            self.num_vertices,
            self.deterministic_flag
        )
        # outputs = (
        #     updated_newx,
        #     updated_newy,
        #     self.net_flat_topo_sort,
        #     self.net_flat_topo_sort_start,
        #     self.pin_fa,
        #     self.flat_pin_to,
        #     self.flat_pin_to_start,
        #     self.flat_pin_from
        # )
        return updated_newx, updated_newy

    def update_cache(self, build_tree_outputs, sanitize_pin_relate=True):

        (self.newx, self.newy, self.pin_relate_x, self.pin_relate_y,
            self.net_vertex_start, self.net_steiner_start,
            self.pin_fa, self.flat_pin_to, self.flat_pin_from, self.flat_pin_to_start,
            self.net_flat_topo_sort, self.net_flat_topo_sort_start) = build_tree_outputs

        self.num_vertices = self.newx.numel()
        self.pin_relate_x = self.pin_relate_x.contiguous()
        self.pin_relate_y = self.pin_relate_y.contiguous()
        self.net_vertex_start = self.net_vertex_start.contiguous()
        self.net_steiner_start = self.net_steiner_start.contiguous()
        self.pin_fa = self.pin_fa.contiguous()
        self.flat_pin_to = self.flat_pin_to.contiguous()
        self.flat_pin_from = self.flat_pin_from.contiguous()
        self.flat_pin_to_start = self.flat_pin_to_start.contiguous()
        self.net_flat_topo_sort = self.net_flat_topo_sort.contiguous()
        self.net_flat_topo_sort_start = self.net_flat_topo_sort_start.contiguous()
        if sanitize_pin_relate:
            self._sanitize_pin_relate_indices()

    def load_ggr_topology_pack(self, pack, pin_pos):
        from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
            build_steiner_cache_from_ggr_pack,
        )

        cache_tuple, edge_l_directions, metadata = build_steiner_cache_from_ggr_pack(
            pack,
            pin_pos,
        )
        self.update_cache(cache_tuple, sanitize_pin_relate=False)
        self.edge_l_directions = edge_l_directions.contiguous()
        if self.collect_edge_geometry_stats:
            self.last_edge_geometry_stats = self._collect_edge_geometry_stats(
                self.flat_pin_from,
                self.flat_pin_to,
                self.newx,
                self.newy,
            )
        else:
            self.last_edge_geometry_stats = None
        logger.info(
            "Loaded GGR L-shape topology pack: nets=%d pins=%d vertices=%d edges=%d",
            int(metadata["num_nets"]),
            int(metadata["num_pins"]),
            int(metadata["num_vertices"]),
            int(metadata["num_edges"]),
        )
        return self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from

    def _sanitize_pin_relate_indices(self):
        if (
            self.pin_relate_x is None
            or self.pin_relate_y is None
            or self.newx is None
            or self.newy is None
            or self.net_steiner_start is None
            or self.flat_net2pin_map is None
            or self.flat_net2pin_start_map is None
        ):
            return

        if self.net_steiner_start.numel() == 0:
            return

        num_pins = int(self.net_steiner_start[0].item())
        num_vertices = int(self.num_vertices or self.pin_relate_x.numel())
        if num_pins <= 0 or num_vertices <= 0:
            return

        net_steiner_start = self.net_steiner_start.detach().cpu().to(torch.long).tolist()
        flat_net2pin = self.flat_net2pin_map.detach().cpu().to(torch.long).tolist()
        flat_net2pin_start = self.flat_net2pin_start_map.detach().cpu().to(torch.long).tolist()
        newx = self.newx.detach().cpu().tolist()
        newy = self.newy.detach().cpu().tolist()

        def fallback_pin(vertex_id, axis):
            vertex_id = int(vertex_id)
            if 0 <= vertex_id < num_pins:
                return vertex_id

            net_id = bisect.bisect_right(net_steiner_start, vertex_id) - 1
            if (
                net_id < 0
                or net_id + 1 >= len(net_steiner_start)
                or vertex_id < net_steiner_start[net_id]
                or vertex_id >= net_steiner_start[net_id + 1]
                or net_id + 1 >= len(flat_net2pin_start)
            ):
                return 0

            pin_begin = int(flat_net2pin_start[net_id])
            pin_end = int(flat_net2pin_start[net_id + 1])
            pins = [
                int(pin_id)
                for pin_id in flat_net2pin[pin_begin:pin_end]
                if 0 <= int(pin_id) < num_pins
            ]
            if not pins:
                return 0

            primary = newx if axis == "x" else newy
            secondary = newy if axis == "x" else newx
            target_primary = primary[vertex_id] if 0 <= vertex_id < len(primary) else 0.0
            target_secondary = secondary[vertex_id] if 0 <= vertex_id < len(secondary) else 0.0
            return min(
                pins,
                key=lambda pin_id: (
                    abs(primary[pin_id] - target_primary),
                    abs(secondary[pin_id] - target_secondary),
                    pin_id,
                ),
            )

        def sanitize_one(relate_tensor, axis):
            relate_cpu = relate_tensor.detach().cpu().to(torch.long)
            invalid_mask = (relate_cpu < 0) | (relate_cpu >= num_pins)
            invalid_indices = invalid_mask.nonzero(as_tuple=False).reshape(-1).tolist()
            if not invalid_indices:
                return 0

            relate_values = relate_cpu.tolist()
            fixed_values = []
            for vertex_id in invalid_indices:
                value = int(relate_values[vertex_id])
                seen = set()
                while num_pins <= value < num_vertices and value not in seen:
                    seen.add(value)
                    value = int(relate_values[value])
                if not (0 <= value < num_pins):
                    value = fallback_pin(vertex_id, axis)
                fixed_values.append(value)

            index_tensor = torch.tensor(
                invalid_indices, dtype=torch.long, device=relate_tensor.device
            )
            value_tensor = torch.tensor(
                fixed_values, dtype=relate_tensor.dtype, device=relate_tensor.device
            )
            relate_tensor[index_tensor] = value_tensor
            return len(invalid_indices)

        fixed_x = sanitize_one(self.pin_relate_x, "x")
        fixed_y = sanitize_one(self.pin_relate_y, "y")
        if fixed_x or fixed_y:
            logger.warning(
                "Sanitized invalid Steiner relate indices: x=%d y=%d num_pins=%d num_vertices=%d",
                fixed_x,
                fixed_y,
                num_pins,
                num_vertices,
            )

    def _collect_edge_geometry_stats(self, edge_from, edge_to, x_coords, y_coords, eps=1e-4):
        def _empty_stats(total_edges=0):
            return {
                "total_edges": int(total_edges),
                "invalid_edges": int(total_edges),
                "valid_edges": 0,
                "diagonal_edges": 0,
                "straight_edges": 0,
                "diagonal_ratio": 0.0,
                "mapped_edges": 0,
                "unmapped_edges": 0,
                "total_nets": 0,
                "two_pin_nets": 0,
                "non_two_pin_nets": 0,
                "two_pin_net_ratio": 0.0,
                "mapped_nets": 0,
                "diagonal_nets": 0,
                "net_diagonal_ratio": 0.0,
                "two_pin_mapped_nets": 0,
                "two_pin_diagonal_nets": 0,
                "two_pin_net_diagonal_ratio": 0.0,
                "non_two_pin_mapped_nets": 0,
                "non_two_pin_diagonal_nets": 0,
                "non_two_pin_net_diagonal_ratio": 0.0,
                "two_pin_edges": 0,
                "two_pin_diagonal_edges": 0,
                "two_pin_diagonal_ratio": 0.0,
                "non_two_pin_edges": 0,
                "non_two_pin_diagonal_edges": 0,
                "non_two_pin_diagonal_ratio": 0.0,
            }

        if edge_from is None or edge_to is None or x_coords is None or y_coords is None:
            return _empty_stats()

        total_edges = int(edge_from.numel())
        num_vertices = min(int(x_coords.numel()), int(y_coords.numel()))
        if num_vertices <= 0:
            return _empty_stats(total_edges)

        valid_mask = (
            (edge_from >= 0)
            & (edge_to >= 0)
            & (edge_from < num_vertices)
            & (edge_to < num_vertices)
        )
        valid_edges = int(valid_mask.sum().item())
        if valid_edges == 0:
            stats = _empty_stats(total_edges)
            stats["invalid_edges"] = total_edges
            return stats

        valid_from = edge_from[valid_mask]
        valid_to = edge_to[valid_mask]
        x1 = x_coords[valid_from]
        y1 = y_coords[valid_from]
        x2 = x_coords[valid_to]
        y2 = y_coords[valid_to]
        is_diagonal = (torch.abs(x1 - x2) >= eps) & (torch.abs(y1 - y2) >= eps)
        diagonal_edges = int(is_diagonal.sum().item())
        straight_edges = valid_edges - diagonal_edges

        stats = {
            "total_edges": total_edges,
            "invalid_edges": total_edges - valid_edges,
            "valid_edges": valid_edges,
            "diagonal_edges": diagonal_edges,
            "straight_edges": straight_edges,
            "diagonal_ratio": diagonal_edges / max(valid_edges, 1),
            "mapped_edges": 0,
            "unmapped_edges": 0,
            "total_nets": 0,
            "two_pin_nets": 0,
            "non_two_pin_nets": 0,
            "two_pin_net_ratio": 0.0,
            "mapped_nets": 0,
            "diagonal_nets": 0,
            "net_diagonal_ratio": 0.0,
            "two_pin_mapped_nets": 0,
            "two_pin_diagonal_nets": 0,
            "two_pin_net_diagonal_ratio": 0.0,
            "non_two_pin_mapped_nets": 0,
            "non_two_pin_diagonal_nets": 0,
            "non_two_pin_net_diagonal_ratio": 0.0,
            "two_pin_edges": 0,
            "two_pin_diagonal_edges": 0,
            "two_pin_diagonal_ratio": 0.0,
            "non_two_pin_edges": 0,
            "non_two_pin_diagonal_edges": 0,
            "non_two_pin_diagonal_ratio": 0.0,
        }

        if self.flat_net2pin_start_map is None:
            return stats

        if self.flat_net2pin_start_map.numel() < 2:
            return stats

        num_nets = int(self.flat_net2pin_start_map.numel()) - 1
        if num_nets <= 0:
            return stats

        net_degrees = (
            self.flat_net2pin_start_map[1:num_nets + 1]
            - self.flat_net2pin_start_map[:num_nets]
        )
        two_pin_net_mask = net_degrees == 2
        non_two_pin_net_mask = ~two_pin_net_mask
        total_nets = int(num_nets)
        two_pin_nets = int(two_pin_net_mask.sum().item())
        non_two_pin_nets = int(non_two_pin_net_mask.sum().item())
        stats.update({
            "total_nets": total_nets,
            "two_pin_nets": two_pin_nets,
            "non_two_pin_nets": non_two_pin_nets,
            "two_pin_net_ratio": two_pin_nets / max(total_nets, 1),
        })

        # Build pin->net map from flat_net2pin structures.
        pin_ids = self.flat_net2pin_map[:self.flat_net2pin_start_map[num_nets]].long()
        if pin_ids.numel() == 0:
            return stats

        num_pins = int(pin_ids.max().item()) + 1
        if self.net_steiner_start is not None and self.net_steiner_start.numel() > 0:
            num_pins = max(num_pins, int(self.net_steiner_start[0].item()))

        pin2net = torch.full((num_pins,), -1, device=pin_ids.device, dtype=torch.long)
        net_ids = torch.arange(num_nets, device=pin_ids.device, dtype=torch.long)
        pin_net_ids = torch.repeat_interleave(net_ids, net_degrees.long())
        pin2net[pin_ids] = pin_net_ids

        vertex_to_net = torch.full((num_vertices,), -1, device=pin_ids.device, dtype=torch.long)
        pin_span = min(num_vertices, pin2net.numel())
        if pin_span > 0:
            vertex_to_net[:pin_span] = pin2net[:pin_span]

        # Fill Steiner ranges directly to avoid bucketize ambiguity on repeated boundaries.
        if self.net_steiner_start is not None and self.net_steiner_start.numel() >= num_nets + 1:
            steiner_starts = self.net_steiner_start[:num_nets].long()
            steiner_ends = self.net_steiner_start[1:num_nets + 1].long()
            for net_id in range(num_nets):
                start = int(steiner_starts[net_id].item())
                end = int(steiner_ends[net_id].item())
                if end <= start:
                    continue
                if start >= num_vertices:
                    continue
                start = max(start, 0)
                end = min(end, num_vertices)
                if end > start:
                    vertex_to_net[start:end] = net_id

        edge_net_ids = vertex_to_net[valid_from]

        mapped_edge_mask = edge_net_ids >= 0
        if not mapped_edge_mask.any():
            stats.update({
                "unmapped_edges": valid_edges,
            })
            return stats

        edge_degrees = net_degrees[edge_net_ids[mapped_edge_mask]]
        mapped_diagonal = is_diagonal[mapped_edge_mask]
        two_pin_mask = edge_degrees == 2
        non_two_pin_mask = ~two_pin_mask

        two_pin_edges = int(two_pin_mask.sum().item())
        two_pin_diag = int((mapped_diagonal & two_pin_mask).sum().item())
        non_two_pin_edges = int(non_two_pin_mask.sum().item())
        non_two_pin_diag = int((mapped_diagonal & non_two_pin_mask).sum().item())

        mapped_net_ids = edge_net_ids[mapped_edge_mask]
        net_edge_counts = torch.bincount(mapped_net_ids, minlength=num_nets)
        diagonal_net_ids = mapped_net_ids[mapped_diagonal]
        diagonal_net_counts = torch.bincount(diagonal_net_ids, minlength=num_nets)

        mapped_net_mask = net_edge_counts > 0
        diagonal_net_mask = diagonal_net_counts > 0
        mapped_nets = int(mapped_net_mask.sum().item())
        diagonal_nets = int(diagonal_net_mask.sum().item())

        two_pin_mapped_nets = int((mapped_net_mask & two_pin_net_mask).sum().item())
        two_pin_diagonal_nets = int((diagonal_net_mask & two_pin_net_mask).sum().item())
        non_two_pin_mapped_nets = int((mapped_net_mask & non_two_pin_net_mask).sum().item())
        non_two_pin_diagonal_nets = int((diagonal_net_mask & non_two_pin_net_mask).sum().item())

        stats.update({
            "mapped_edges": int(mapped_edge_mask.sum().item()),
            "unmapped_edges": int((~mapped_edge_mask).sum().item()),
            "mapped_nets": mapped_nets,
            "diagonal_nets": diagonal_nets,
            "net_diagonal_ratio": diagonal_nets / max(mapped_nets, 1),
            "two_pin_mapped_nets": two_pin_mapped_nets,
            "two_pin_diagonal_nets": two_pin_diagonal_nets,
            "two_pin_net_diagonal_ratio": two_pin_diagonal_nets / max(two_pin_mapped_nets, 1),
            "non_two_pin_mapped_nets": non_two_pin_mapped_nets,
            "non_two_pin_diagonal_nets": non_two_pin_diagonal_nets,
            "non_two_pin_net_diagonal_ratio": non_two_pin_diagonal_nets / max(non_two_pin_mapped_nets, 1),
            "two_pin_edges": two_pin_edges,
            "two_pin_diagonal_edges": two_pin_diag,
            "two_pin_diagonal_ratio": two_pin_diag / max(two_pin_edges, 1),
            "non_two_pin_edges": non_two_pin_edges,
            "non_two_pin_diagonal_edges": non_two_pin_diag,
            "non_two_pin_diagonal_ratio": non_two_pin_diag / max(non_two_pin_edges, 1),
        })
        return stats

    def rebuild_tree(self, pos):

        new_outputs_tuple = steiner_topo_cpp.build_tree(
            pos,
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            self.ignore_net_degree,
            self.deterministic_flag,
            str(_FLUTE_POWV_FILE),
            str(_FLUTE_POST_FILE),
        )

        self.update_cache(new_outputs_tuple)
        if self.collect_edge_geometry_stats:
            self.last_edge_geometry_stats = self._collect_edge_geometry_stats(
                self.flat_pin_from,
                self.flat_pin_to,
                self.newx,
                self.newy,
            )
            logger.info(
                "FLUTE edge geometry: diagonal_edges=%d/%d (%.2f%%), straight_edges=%d, invalid_edges=%d, "
                "mapped_edges=%d, unmapped_edges=%d | 2pin diagonal=%d/%d (%.2f%%) | "
                "non2pin diagonal=%d/%d (%.2f%%) || net_ratio: 2pin=%d/%d (%.2f%%), "
                "diag_nets=%d/%d (%.2f%%), 2pin_diag_nets=%d/%d (%.2f%%), non2pin_diag_nets=%d/%d (%.2f%%)",
                self.last_edge_geometry_stats["diagonal_edges"],
                self.last_edge_geometry_stats["valid_edges"],
                self.last_edge_geometry_stats["diagonal_ratio"] * 100.0,
                self.last_edge_geometry_stats["straight_edges"],
                self.last_edge_geometry_stats["invalid_edges"],
                self.last_edge_geometry_stats["mapped_edges"],
                self.last_edge_geometry_stats["unmapped_edges"],
                self.last_edge_geometry_stats["two_pin_diagonal_edges"],
                self.last_edge_geometry_stats["two_pin_edges"],
                self.last_edge_geometry_stats["two_pin_diagonal_ratio"] * 100.0,
                self.last_edge_geometry_stats["non_two_pin_diagonal_edges"],
                self.last_edge_geometry_stats["non_two_pin_edges"],
                self.last_edge_geometry_stats["non_two_pin_diagonal_ratio"] * 100.0,
                self.last_edge_geometry_stats["two_pin_nets"],
                self.last_edge_geometry_stats["total_nets"],
                self.last_edge_geometry_stats["two_pin_net_ratio"] * 100.0,
                self.last_edge_geometry_stats["diagonal_nets"],
                self.last_edge_geometry_stats["mapped_nets"],
                self.last_edge_geometry_stats["net_diagonal_ratio"] * 100.0,
                self.last_edge_geometry_stats["two_pin_diagonal_nets"],
                self.last_edge_geometry_stats["two_pin_mapped_nets"],
                self.last_edge_geometry_stats["two_pin_net_diagonal_ratio"] * 100.0,
                self.last_edge_geometry_stats["non_two_pin_diagonal_nets"],
                self.last_edge_geometry_stats["non_two_pin_mapped_nets"],
                self.last_edge_geometry_stats["non_two_pin_net_diagonal_ratio"] * 100.0,
            )
        else:
            self.last_edge_geometry_stats = None
        return self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from

    def init_l_direction_resolver(self, placedb, params):
        """
        初始化L方向解析器
        
        Args:
            placedb: DREAMPlace placement database
            params: 参数对象
        """
        from dreamplace.ops.steiner_topo.egr_l_direction import EGRLDirectionResolver
        self.l_direction_resolver = EGRLDirectionResolver(placedb, params)
        logger.info("L direction resolver initialized")
    
    def resolve_l_directions_from_egr(self, guide_path):
        """
        从EGR guide文件解析每条边的L方向
        
        Args:
            guide_path: route_planar.guide文件路径
            
        Returns:
            torch.Tensor: shape=(num_edges,), 每条边的L方向
                H_FIRST(0): 先水平后垂直
                V_FIRST(1): 先垂直后水平
                STRAIGHT(2): 直线
                UNKNOWN(-1): 未知
        """
        if self.l_direction_resolver is None:
            logger.error("L direction resolver not initialized, call init_l_direction_resolver first")
            return None
        
        # 解析EGR guide
        self.l_direction_resolver.parse_egr_guide(guide_path)
        
        # 解析L方向
        self.edge_l_directions = self.l_direction_resolver.resolve_l_directions(self, guide_path)
        
        return self.edge_l_directions

    def resolve_l_directions_from_gpugr(self, route_entries):
        """
        从 gpugr 导出的 route_entries 解析每条边的 L 方向。

        Args:
            route_entries: gpugr.RouteForce.route_entries() 的返回结果

        Returns:
            torch.Tensor: shape=(num_edges,), 每条边的L方向
        """
        if self.l_direction_resolver is None:
            logger.error("L direction resolver not initialized, call init_l_direction_resolver first")
            return None

        self.l_direction_resolver.parse_gpugr_route_entries(route_entries)
        self.edge_l_directions = self.l_direction_resolver.resolve_l_directions(self)
        return self.edge_l_directions
    
    def get_edge_l_direction(self, edge_idx):
        """
        获取指定边的L方向
        
        Args:
            edge_idx: 边索引（对应flat_pin_from/flat_pin_to的索引）
            
        Returns:
            L方向: H_FIRST(0), V_FIRST(1), STRAIGHT(2), UNKNOWN(-1)
        """
        if self.edge_l_directions is None:
            return self.UNKNOWN
        if edge_idx < 0 or edge_idx >= len(self.edge_l_directions):
            return self.UNKNOWN
        return int(self.edge_l_directions[edge_idx])
    
    def get_net_edges_with_l_direction(self, net_id, placedb):
        """
        获取指定net的所有边及其L方向
        
        Args:
            net_id: net索引
            placedb: placement database
            
        Returns:
            list of tuples: [(from_idx, to_idx, l_direction), ...]
        """
        if self.flat_pin_from is None or self.flat_pin_to is None:
            return []
        
        flat_pin_from = self.flat_pin_from.cpu().numpy()
        flat_pin_to = self.flat_pin_to.cpu().numpy()
        net_steiner_start = self.net_steiner_start.cpu().numpy()
        
        num_pins = placedb.num_pins
        pin2net = placedb.pin2net_map
        if hasattr(pin2net, 'cpu'):
            pin2net = pin2net.cpu().numpy()
        
        edges = []
        for edge_idx in range(len(flat_pin_from)):
            from_idx = flat_pin_from[edge_idx]
            to_idx = flat_pin_to[edge_idx]
            
            if from_idx == -1 or to_idx == -1:
                continue
            
            # 检查是否属于该net
            edge_net_id = -1
            if from_idx < num_pins:
                edge_net_id = int(pin2net[from_idx])
            else:
                for nid in range(len(net_steiner_start) - 1):
                    if net_steiner_start[nid] <= from_idx < net_steiner_start[nid + 1]:
                        edge_net_id = nid
                        break
            
            if edge_net_id == net_id:
                l_dir = self.get_edge_l_direction(edge_idx)
                edges.append((from_idx, to_idx, l_dir))
        
        return edges
    
    @staticmethod
    def l_direction_name(direction):
        """将L方向常量转换为可读名称"""
        names = {
            SteinerTopo.H_FIRST: "upper_L",
            SteinerTopo.V_FIRST: "lower_L",
            SteinerTopo.STRAIGHT: "straight",
            SteinerTopo.UNKNOWN: "unknown"
        }
        return names.get(direction, "invalid")
    
    # ==================== EGR Steiner Builder ====================
    
    def init_egr_steiner_builder(self, placedb, params):
        """
        初始化EGR Steiner构建器
        
        Args:
            placedb: DREAMPlace placement database
            params: 参数对象
        """
        from dreamplace.ops.steiner_topo.egr_steiner_builder import EGRSteinerBuilder
        self.egr_steiner_builder = EGRSteinerBuilder(placedb, params)
        self.placedb = placedb
        self.params = params
        logger.info("EGR Steiner builder initialized")
    
    def rebuild_tree_from_egr(self, pos, guide_path, gcell_info_path=None):
        """
        使用EGR guide替代FLUTE构建Steiner树
        
        这会：
        1. 从EGR解析拓扑结构和L方向
        2. 用pin坐标计算Steiner点位置（保持可微性）
        3. 记录所有Steiner点的详细信息
        4. 生成边列表和L方向（用于绘图）
        
        Args:
            pos: pin坐标tensor
            guide_path: EGR route_planar.guide文件路径
            gcell_info_path: gcell.info文件路径（可选，用于记录gcell中心）
            
        Returns:
            dict with EGR建树结果
        """
        if not hasattr(self, 'egr_steiner_builder') or self.egr_steiner_builder is None:
            logger.error("EGR Steiner builder not initialized, call init_egr_steiner_builder first")
            return None
        
        # 使用EGR构建
        result = self.egr_steiner_builder.build_all_nets(pos, guide_path, gcell_info_path)
        
        # 更新relate关系
        self.pin_relate_x = result['pin_relate_x'].contiguous()
        self.pin_relate_y = result['pin_relate_y'].contiguous()
        self.net_steiner_start = result['net_steiner_start'].contiguous()
        self.num_vertices = self.placedb.num_pins + result['num_steiner']
        
        # 保存Steiner点记录
        self.egr_steiner_points = result['steiner_points']
        
        # 保存EGR边列表（用于绘图）
        self.egr_flat_pin_from = result['flat_pin_from'].contiguous()
        self.egr_flat_pin_to = result['flat_pin_to'].contiguous()
        self.egr_edge_l_directions = result['edge_l_directions'].contiguous()
        
        logger.info(f"Built Steiner tree from EGR: {result['num_steiner']} Steiner points, "
                   f"{len(self.egr_flat_pin_from)} edges")
        
        return result
    
    def update_steiner_hanan_coords(self, pos):
        """
        更新Steiner点的Hanan坐标（根据当前pin位置）
        
        在forward之后调用，记录Steiner点的实际坐标
        """
        if not hasattr(self, 'egr_steiner_points') or not self.egr_steiner_points:
            return
        
        # 获取pin坐标
        if pos.is_cuda:
            pos_np = pos.cpu().numpy()
        else:
            pos_np = pos.numpy()
        
        num_nodes = self.placedb.num_nodes
        pin_pos_x = pos_np[:num_nodes]  # 简化处理，实际需要pin_pos_op
        pin_pos_y = pos_np[num_nodes:]
        
        # 更新每个Steiner点的Hanan坐标
        for sp in self.egr_steiner_points:
            if sp.relate_x_pin_id is not None and sp.relate_x_pin_id < len(pin_pos_x):
                # 需要通过pin_pos_op计算，这里简化
                pass
            if sp.relate_y_pin_id is not None and sp.relate_y_pin_id < len(pin_pos_y):
                pass
    
    def get_steiner_points(self):
        """
        获取所有Steiner点的记录
        
        Returns:
            list of SteinerPointInfo
        """
        if hasattr(self, 'egr_steiner_points'):
            return self.egr_steiner_points
        return []
    
    def get_net_steiner_points(self, net_id):
        """
        获取指定net的Steiner点记录
        
        Args:
            net_id: net ID
            
        Returns:
            list of SteinerPointInfo
        """
        if hasattr(self, 'egr_steiner_builder') and self.egr_steiner_builder:
            return self.egr_steiner_builder.net_steiner_points.get(net_id, [])
        return []
    
    def export_steiner_points(self, output_path):
        """
        导出Steiner点信息到CSV
        
        Args:
            output_path: 输出文件路径
        """
        if hasattr(self, 'egr_steiner_builder') and self.egr_steiner_builder:
            self.egr_steiner_builder.export_steiner_points_csv(output_path)
        else:
            logger.warning("No EGR Steiner builder, cannot export")
    
    def print_steiner_info(self, net_name=None):
        """
        打印Steiner点信息
        
        Args:
            net_name: 如果指定，只打印该net的信息；否则打印统计摘要
        """
        if not hasattr(self, 'egr_steiner_builder') or not self.egr_steiner_builder:
            print("No EGR Steiner builder")
            return
        
        if net_name:
            self.egr_steiner_builder.print_net_steiner_info(net_name)
        else:
            steiner_pts = self.get_steiner_points()
            print(f"\n=== Steiner Points Summary ===")
            print(f"Total Steiner points: {len(steiner_pts)}")
            
            # 按net统计
            net_counts = {}
            for sp in steiner_pts:
                net_counts[sp.net_name] = net_counts.get(sp.net_name, 0) + 1
            
            print(f"Nets with Steiner points: {len(net_counts)}")
            
            # 打印前10个net
            sorted_nets = sorted(net_counts.items(), key=lambda x: -x[1])[:10]
            print("Top 10 nets by Steiner point count:")
            for net_name, count in sorted_nets:
                print(f"  {net_name}: {count}")


'''
self.net_flat_topo_sort, self.net_flat_topo_sort_start, self.pin_fa, \
            self.flat_pin_to, self.flat_pin_to_start, self.flat_pin_from

'''
