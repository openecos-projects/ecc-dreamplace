# @file   egr_l_direction.py
# @brief  Parse EGR route guide and determine L-shape direction for Steiner tree edges
#
import torch
import numpy as np
import logging
import time
from collections import Counter, defaultdict
from dreamplace.ops.steiner_topo import steiner_topo_cpp

logger = logging.getLogger(__name__)


class EGRLDirectionResolver:
    """
    从EGR的route_planar.guide中解析L形走线方向，
    
    H_FIRST (水平优先): 先水平后垂直, 拐点在 (x2, y1)
    V_FIRST (垂直优先): 先垂直后水平, 拐点在 (x1, y2)
    """
    
    # L方向常量
    H_FIRST = 0   # 先水平后垂直 (Horizontal First)
    V_FIRST = 1   # 先垂直后水平 (Vertical First)
    STRAIGHT = 2  # 直线（水平或垂直）
    FAKE_STRAIGHT = 3 # 伪直线（在gcell下只有一条wire）
    UNKNOWN = -1  # 未知
    
    
    def __init__(self, placedb, params):
        """
        Args:
            placedb: DREAMPlace的placement database
            params: 参数对象，包含scale_factor, shift_factor等
        """
        self.placedb = placedb
        self.params = params
        
        # 建立net名字到id的映射
        self.net_name2id = self._build_net_name_to_id_map()
        
        # EGR解析结果
        self.egr_net_data = {}  # {net_name: {'wires': [...], 'pins': [...]}}
        self._route_grid_x_um = None
        self._route_grid_y_um = None
        self._route_pitch_x_um = None
        self._route_pitch_y_um = None

        # L方向结果: edge_idx -> direction
        self.edge_l_directions = None
        self._native_matcher = None
        
    def _build_net_name_to_id_map(self):
        """建立 net_name -> net_id 的映射"""
        net_name2id = {}
        for net_id in range(self.placedb.num_nets):
            net_name = self.placedb.net_names[net_id]
            # 处理可能的bytes类型
            if isinstance(net_name, bytes):
                net_name = net_name.decode('utf-8')
            net_name2id[net_name] = net_id
        logger.info(f"Built net name to id map with {len(net_name2id)} nets")
        return net_name2id
    
    def parse_egr_guide(self, guide_path):
        """
        解析EGR guide文件
        
        格式:
            guide net_name
            pin grid_x grid_y real_x real_y layer energy name
            wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer
            via grid_x grid_y real_x real_y layer1 layer2
        
        Args:
            guide_path: route_planar.guide文件路径
            
        Returns:
            dict: {net_name: {'wires': [...], 'pins': [...]}}
        """
        net_data = defaultdict(lambda: {'wires': [], 'pins': []})
        current_net = None
        self._route_grid_x_um = None
        self._route_grid_y_um = None
        self._route_pitch_x_um = None
        self._route_pitch_y_um = None
        self._native_matcher = None
        route_x_coords = []
        route_y_coords = []
        
        # 头部说明行的关键词（用于跳过）
        header_keywords = {'net_name', 'grid_x', 'grid_y', 'grid1_x', 'grid2_x', 
                          'real_x', 'real_y', 'real1_x', 'real2_x', 'layer1', 'layer2'}
        
        try:
            with open(guide_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    # 跳过头部说明行（包含 grid_x, net_name 等关键词）
                    if len(parts) > 1 and parts[1] in header_keywords:
                        continue
                    
                    if parts[0] == 'guide':
                        # guide net_name
                        current_net = parts[1]
                        
                    elif parts[0] == 'pin' and current_net:
                        # pin grid_x grid_y real_x real_y layer energy name
                        # idx:  0     1      2      3      4     5      6     7
                        try:
                            grid_x = int(parts[1])
                            grid_y = int(parts[2])
                            real_x = float(parts[3])
                            real_y = float(parts[4])
                            layer = parts[5]
                            energy = parts[6]  # 'load' or 'driven'
                            pin_name = parts[7] if len(parts) > 7 else ""
                            net_data[current_net]['pins'].append({
                                'grid': (grid_x, grid_y),
                                'real': (real_x, real_y),
                                'layer': layer,
                                'energy': energy,
                                'name': pin_name
                            })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Failed to parse pin line: {line}, error: {e}")
                        
                    elif parts[0] == 'wire' and current_net:
                        # wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer
                        # idx:  0       1       2       3       4       5       6       7       8    9
                        try:
                            grid1_x = int(parts[1])
                            grid1_y = int(parts[2])
                            grid2_x = int(parts[3])
                            grid2_y = int(parts[4])
                            real1_x = float(parts[5])
                            real1_y = float(parts[6])
                            real2_x = float(parts[7])
                            real2_y = float(parts[8])
                            layer = parts[9] if len(parts) > 9 else ""
                            
                            # 判断wire方向
                            is_horizontal = (grid1_y == grid2_y)
                            is_vertical = (grid1_x == grid2_x)
                            
                            net_data[current_net]['wires'].append({
                                'grid1': (grid1_x, grid1_y),
                                'grid2': (grid2_x, grid2_y),
                                'real1': (real1_x, real1_y),
                                'real2': (real2_x, real2_y),
                                'layer': layer,
                                'is_horizontal': is_horizontal,
                                'is_vertical': is_vertical
                            })
                            route_x_coords.extend((real1_x, real2_x))
                            route_y_coords.extend((real1_y, real2_y))
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Failed to parse wire line: {line}, error: {e}")
                            
        except FileNotFoundError:
            logger.error(f"EGR guide file not found: {guide_path}")
            return {}

        self.egr_net_data = dict(net_data)
        self._route_grid_x_um = self._build_route_coord_axis(route_x_coords)
        self._route_grid_y_um = self._build_route_coord_axis(route_y_coords)
        self._route_pitch_x_um = self._estimate_route_pitch(self._route_grid_x_um)
        self._route_pitch_y_um = self._estimate_route_pitch(self._route_grid_y_um)

        logger.info(f"Parsed EGR guide: {len(self.egr_net_data)} nets")

        return self.egr_net_data

    def parse_gpugr_route_entries(self, route_entries):
        """
        解析 gpugr.route_entries() 返回的有序 route entries。

        Args:
            route_entries: list of
                {
                    "net_name": str,
                    "route_failed": bool,
                    "entries": [
                        {
                            "type": "wire" / "via",
                            "grid_x1/grid_y1/grid_x2/grid_y2": int,
                            "dbu_center_x1/dbu_center_y1/dbu_center_x2/dbu_center_y2": int,
                            ...
                        }
                    ]
                }

        Returns:
            dict: 与 parse_egr_guide 相同的 net_data 结构
        """
        net_data = {}
        dbu = float(self.placedb.dbu)
        route_x_coords = []
        route_y_coords = []
        self._native_matcher = None

        for net_route in route_entries or []:
            net_name = net_route.get("net_name", "") or f"net_{net_route.get('net_id', -1)}"
            entries = net_route.get("entries", []) or []

            data = {
                "wires": [],
                "pins": [],
                "vias": [],
                "route_failed": bool(net_route.get("route_failed", False)),
                "source": "gpugr",
            }

            for entry in entries:
                entry_type = entry.get("type", "")
                grid1 = (int(entry.get("grid_x1", 0)), int(entry.get("grid_y1", 0)))
                grid2 = (int(entry.get("grid_x2", 0)), int(entry.get("grid_y2", 0)))
                real1 = (
                    float(entry.get("dbu_center_x1", entry.get("dbu_lx", 0))) / dbu,
                    float(entry.get("dbu_center_y1", entry.get("dbu_ly", 0))) / dbu,
                )
                real2 = (
                    float(entry.get("dbu_center_x2", entry.get("dbu_hx", 0))) / dbu,
                    float(entry.get("dbu_center_y2", entry.get("dbu_hy", 0))) / dbu,
                )

                if entry_type == "wire":
                    orientation = entry.get("orientation", "")
                    is_horizontal = orientation == "H" or grid1[1] == grid2[1]
                    is_vertical = orientation == "V" or grid1[0] == grid2[0]
                    route_x_coords.extend((real1[0], real2[0]))
                    route_y_coords.extend((real1[1], real2[1]))
                    data["wires"].append(
                        {
                            "grid1": grid1,
                            "grid2": grid2,
                            "real1": real1,
                            "real2": real2,
                            "layer": entry.get("layer_name", ""),
                            "is_horizontal": is_horizontal,
                            "is_vertical": is_vertical,
                            "order": int(entry.get("order", len(data["wires"]))),
                        }
                    )
                elif entry_type == "via":
                    route_x_coords.append(real1[0])
                    route_y_coords.append(real1[1])
                    data["vias"].append(
                        {
                            "grid": grid1,
                            "real": real1,
                            "layer": entry.get("layer_name", ""),
                            "order": int(entry.get("order", len(data["vias"]))),
                        }
                    )

            net_data[net_name] = data

        self.egr_net_data = net_data
        self._route_grid_x_um = self._build_route_coord_axis(route_x_coords)
        self._route_grid_y_um = self._build_route_coord_axis(route_y_coords)
        self._route_pitch_x_um = self._estimate_route_pitch(self._route_grid_x_um)
        self._route_pitch_y_um = self._estimate_route_pitch(self._route_grid_y_um)
        logger.info(f"Parsed gpugr route entries: {len(self.egr_net_data)} nets")
        return self.egr_net_data

    def _build_route_coord_axis(self, coords, eps=1e-6):
        if not coords:
            return None
        coords = np.sort(np.asarray(coords, dtype=np.float64))
        uniq = [coords[0]]
        for value in coords[1:]:
            if abs(value - uniq[-1]) > eps:
                uniq.append(value)
        return np.asarray(uniq, dtype=np.float64)

    def _estimate_route_pitch(self, coords, eps=1e-6):
        if coords is None or len(coords) < 2:
            return None
        diffs = np.diff(coords)
        diffs = diffs[diffs > eps]
        if diffs.size == 0:
            return None
        return float(np.median(diffs))

    def _candidate_match_tolerances(self, net_data):
        if net_data.get("source") != "gpugr":
            return [1.0]

        pitches = [pitch for pitch in (self._route_pitch_x_um, self._route_pitch_y_um) if pitch is not None and pitch > 0]
        if not pitches:
            return [1.0, 2.0]

        base_pitch = min(pitches)
        tolerances = [
            max(base_pitch * 0.10, 0.10),
            max(base_pitch * 0.25, 0.25),
            max(base_pitch * 0.50, 0.50),
        ]
        return sorted({round(float(value), 6) for value in tolerances if value > 0}) or [1.0]

    def _endpoint_match_tolerance(self, net_data, segment_tolerance):
        if net_data.get("source") != "gpugr":
            return float(segment_tolerance)

        pitches = [
            pitch
            for pitch in (self._route_pitch_x_um, self._route_pitch_y_um)
            if pitch is not None and pitch > 0
        ]
        if not pitches:
            return min(float(segment_tolerance), 1.0)

        base_pitch = min(pitches)
        return min(float(segment_tolerance), max(base_pitch * 0.10, 0.10))

    def _log_edge_geometry_statistics(self, flat_pin_from, flat_pin_to, newx_um, newy_um, l_directions, valid_indices, eps=1e-5):
        if valid_indices is None or len(valid_indices) == 0:
            return

        valid_from = flat_pin_from[valid_indices]
        valid_to = flat_pin_to[valid_indices]
        dx = np.abs(newx_um[valid_from] - newx_um[valid_to])
        dy = np.abs(newy_um[valid_from] - newy_um[valid_to])
        lengths = dx + dy

        is_horizontal = dy < eps
        is_vertical = dx < eps
        geo_straight_mask = is_horizontal | is_vertical
        label_straight_mask = l_directions[valid_indices] == self.STRAIGHT
        nonstraight_label_mask = ~label_straight_mask

        geo_straight = int(geo_straight_mask.sum())
        geo_diagonal = int((~geo_straight_mask).sum())
        label_straight = int(label_straight_mask.sum())
        straight_but_diag = int((label_straight_mask & ~geo_straight_mask).sum())
        nonstraight_but_geo_straight = int((nonstraight_label_mask & geo_straight_mask).sum())
        horizontal_straight = int((is_horizontal & ~is_vertical).sum())
        vertical_straight = int((is_vertical & ~is_horizontal).sum())
        degenerate = int((is_horizontal & is_vertical).sum())

        logger.info(
            "L-direction geometry stats: valid_edges=%d geo_straight=%d (horizontal=%d vertical=%d degenerate=%d) "
            "geo_diagonal=%d label_straight=%d mismatches[straight_but_diag=%d nonstraight_but_geo_straight=%d]",
            len(valid_indices),
            geo_straight,
            horizontal_straight,
            vertical_straight,
            degenerate,
            geo_diagonal,
            label_straight,
            straight_but_diag,
            nonstraight_but_geo_straight,
        )

        straight_lengths = lengths[label_straight_mask]
        if straight_lengths.size == 0:
            return

        p50 = float(np.percentile(straight_lengths, 50))
        p90 = float(np.percentile(straight_lengths, 90))
        p99 = float(np.percentile(straight_lengths, 99))
        mean_len = float(np.mean(straight_lengths))
        max_len = float(np.max(straight_lengths))

        logger.info(
            "L-direction straight length stats (um): count=%d mean=%.3f p50=%.3f p90=%.3f p99=%.3f max=%.3f",
            straight_lengths.size,
            mean_len,
            p50,
            p90,
            p99,
            max_len,
        )

        pitches = [
            pitch
            for pitch in (self._route_pitch_x_um, self._route_pitch_y_um)
            if pitch is not None and pitch > 0
        ]
        if not pitches:
            return

        base_pitch = min(pitches)
        le_1x = int((straight_lengths <= base_pitch + eps).sum())
        le_2x = int((straight_lengths <= base_pitch * 2 + eps).sum())
        le_4x = int((straight_lengths <= base_pitch * 4 + eps).sum())
        gt_4x = int((straight_lengths > base_pitch * 4 + eps).sum())
        logger.info(
            "L-direction straight pitch histogram: base_pitch=%.3fum <=1x=%d <=2x=%d <=4x=%d >4x=%d",
            base_pitch,
            le_1x,
            le_2x,
            le_4x,
            gt_4x,
        )

    def _coord_dp_to_micron(self, x_dp, y_dp):
        """
        DREAMPlace内部坐标转换为micron
        
        Args:
            x_dp, y_dp: DREAMPlace内部坐标
            
        Returns:
            (x_um, y_um): micron坐标
        """
        # DREAMPlace坐标 -> 原始DBU坐标
        x_dbu = x_dp / self.params.scale_factor + self.params.shift_factor[0]
        y_dbu = y_dp / self.params.scale_factor + self.params.shift_factor[1]
        
        # DBU -> micron 
        x_um = x_dbu / self.placedb.dbu
        y_um = y_dbu / self.placedb.dbu
        
        return x_um, y_um

    def resolve_l_directions(self, steiner_topo_op, guide_path=None):
        """
        批量调用 native matcher，保留当前路由反馈并消费最新 Steiner 坐标。
        
        Args:
            steiner_topo_op: SteinerTopo对象
            guide_path: EGR guide文件路径（如果之前没有parse过）
            
        Returns:
            torch.Tensor: shape=(num_edges,), 每条边的L方向
        """
        if guide_path and not hasattr(self, 'egr_net_data'):
            self.parse_egr_guide(guide_path)
        
        if not hasattr(self, 'egr_net_data') or not self.egr_net_data:
            logger.warning("No EGR data parsed, returning all UNKNOWN")
            num_edges = steiner_topo_op.flat_pin_from.numel()
            return torch.full((num_edges,), self.UNKNOWN, dtype=torch.int32)
        
        # 获取Steiner树数据
        flat_pin_from = steiner_topo_op.flat_pin_from.cpu().numpy()
        flat_pin_to = steiner_topo_op.flat_pin_to.cpu().numpy()
        newx = steiner_topo_op.newx.cpu().numpy()
        newy = steiner_topo_op.newy.cpu().numpy()
        net_steiner_start = steiner_topo_op.net_steiner_start.cpu().numpy()
        
        num_pins = self.placedb.num_pins
        
        # ========== 优化1: 预计算所有顶点的net_id ==========
        vertex_to_net = self._precompute_vertex_to_net(num_pins, net_steiner_start, len(newx))
        
        # ========== 优化2: 向量化坐标转换 ==========
        # DREAMPlace坐标 -> micron (一次性转换所有顶点)
        scale = self.params.scale_factor
        shift_x, shift_y = self.params.shift_factor
        dbu = self.placedb.dbu
        
        newx_um = (newx / scale + shift_x) / dbu
        newy_um = (newy / scale + shift_y) / dbu
        
        # Route data changes only at feedback publication; tree geometry changes
        # at every rebuild. Native matching keeps those lifetimes separate.
        preparation_start = time.perf_counter()
        if self._native_matcher is None:
            names = [name.decode() if isinstance(name, bytes) else str(name)
                     for name in self.placedb.net_names]
            source = {"source": "gpugr"}
            self._native_matcher = steiner_topo_cpp.LDirectionMatcher(
                names, self.egr_net_data,
                [] if self._route_grid_x_um is None else self._route_grid_x_um.tolist(),
                [] if self._route_grid_y_um is None else self._route_grid_y_um.tolist(),
                self._candidate_match_tolerances(source),
                self._endpoint_match_tolerance(source, float("inf")),
            )
        preparation_ms = (time.perf_counter() - preparation_start) * 1000.0
        valid_mask = (flat_pin_from >= 0) & (flat_pin_to >= 0)
        valid_indices = np.where(valid_mask)[0]
        l_directions, edge_pass_ms, fallback_ms, resolved, downgraded = self._native_matcher.resolve(
            flat_pin_from, flat_pin_to, vertex_to_net, newx_um, newy_um,
            newx_um.dtype == np.float32,
        )
        stats = Counter(l_directions.tolist())
        if resolved or downgraded:
            logger.info(
                "Known-path map fallback resolved %d unresolved edges and downgraded %d FAKE_STRAIGHT ties to UNKNOWN.",
                resolved, downgraded,
            )
        logger.info(
            "L-direction resolve timing: valid_edges=%d edge_pass=%.2fms fallback=%.2fms preparation=%.2fms backend=cpp",
            len(valid_indices),
            edge_pass_ms,
            fallback_ms,
            preparation_ms,
        )
        
        logger.info(f"L direction resolution: h_first={stats[self.H_FIRST]}, "
                   f"v_first={stats[self.V_FIRST]}, straight={stats[self.STRAIGHT]}, "
                   f"fake_straight={stats[self.FAKE_STRAIGHT]}, unknown={stats[self.UNKNOWN]}")
        self._log_edge_geometry_statistics(
            flat_pin_from,
            flat_pin_to,
            newx_um,
            newy_um,
            l_directions,
            valid_indices,
        )
        
        self.edge_l_directions = torch.from_numpy(l_directions)
        if steiner_topo_op is not None:
            steiner_topo_op.edge_l_directions = self.edge_l_directions
        return self.edge_l_directions
    
    def _precompute_vertex_to_net(self, num_pins, net_steiner_start, num_vertices):
        """
        预计算所有顶点到net_id的映射
        
        Args:
            num_pins: pin总数
            net_steiner_start: 每个net的Steiner点起始索引
            num_vertices: 顶点总数
            
        Returns:
            numpy array: vertex_idx -> net_id
        """
        vertex_to_net = np.full(num_vertices, -1, dtype=np.int32)
        
        # Pin -> net (使用 pin2net_map)
        pin2net = self.placedb.pin2net_map
        if hasattr(pin2net, 'cpu'):
            pin2net = pin2net.cpu().numpy()
        elif hasattr(pin2net, '__iter__'):
            pin2net = np.array(pin2net)
        
        vertex_to_net[:num_pins] = pin2net[:num_pins]
        
        # Steiner点 -> net (使用 net_steiner_start)
        num_nets = len(net_steiner_start) - 1
        for net_id in range(num_nets):
            start = net_steiner_start[net_id]
            end = net_steiner_start[net_id + 1]
            if start < num_vertices and end <= num_vertices:
                vertex_to_net[start:end] = net_id
        
        return vertex_to_net
    
    def _find_net_for_vertex(self, vertex_idx, num_pins, net_steiner_start):
        """
        通过vertex_idx找到对应的net_id
        
        Args:
            vertex_idx: 顶点索引
            num_pins: pin总数
            net_steiner_start: 每个net的Steiner点起始索引
            
        Returns:
            net_id, 如果找不到返回-1
        """
        if vertex_idx < num_pins:
            # 是pin，通过pin2net_map查找
            pin2net = self.placedb.pin2net_map
            if hasattr(pin2net, 'cpu'):
                pin2net = pin2net.cpu().numpy()
            elif hasattr(pin2net, '__iter__'):
                pin2net = np.array(pin2net)
            return int(pin2net[vertex_idx])
        else:
            # 是Steiner点，通过net_steiner_start定位
            for net_id in range(len(net_steiner_start) - 1):
                if net_steiner_start[net_id] <= vertex_idx < net_steiner_start[net_id + 1]:
                    return net_id
        return -1
    
    def update_steiner_relate(self, steiner_topo_op):
        """
        根据L方向更新SteinerTopo的pin_relate_x和pin_relate_y
        
        这会影响Steiner点的坐标计算方式：
        - H_FIRST: x来自水平方向的邻居，y来自垂直方向的邻居
        - V_FIRST: x来自垂直方向的邻居，y来自水平方向的邻居
        
        核心原理：
        - pin_relate_x[vtx_id] = 哪个pin的x坐标用于该顶点
        - pin_relate_y[vtx_id] = 哪个pin的y坐标用于该顶点
        - 对于Steiner点，根据L方向决定x/y分别来自哪个方向的pin
        
        Args:
            steiner_topo_op: SteinerTopo对象
        """
        if self.edge_l_directions is None:
            logger.warning("L directions not resolved yet, call resolve_l_directions first")
            return
        
        # 获取必要的数据
        flat_pin_from = steiner_topo_op.flat_pin_from.cpu().numpy()
        flat_pin_to = steiner_topo_op.flat_pin_to.cpu().numpy()
        pin_relate_x = steiner_topo_op.pin_relate_x.cpu().numpy().copy()
        pin_relate_y = steiner_topo_op.pin_relate_y.cpu().numpy().copy()
        l_directions = self.edge_l_directions.cpu().numpy()
        
        num_pins = self.placedb.num_pins
        num_edges = len(flat_pin_from)
        
        # 统计更新次数
        update_count = 0
        conflict_count = 0
        x_proposals = defaultdict(list)
        y_proposals = defaultdict(list)

        def collect_proposal(vertex_idx, new_relate_x, new_relate_y):
            x_proposals[int(vertex_idx)].append(int(new_relate_x))
            y_proposals[int(vertex_idx)].append(int(new_relate_y))

        def resolve_axis_proposal(vertex_idx, axis_name, proposals, current_value):
            if not proposals:
                return int(current_value), False

            counter = Counter(int(value) for value in proposals)
            if len(counter) == 1:
                return int(next(iter(counter))), False

            most_common = counter.most_common()
            top_value, top_count = most_common[0]
            second_count = most_common[1][1] if len(most_common) > 1 else 0
            if top_count > second_count:
                logger.debug(
                    "Steiner vertex %d resolved conflicting %s proposals by majority: %s -> %d",
                    vertex_idx,
                    axis_name,
                    dict(counter),
                    top_value,
                )
                return int(top_value), False

            logger.debug(
                "Steiner vertex %d has tied conflicting %s proposals: %s; keep current=%d",
                vertex_idx,
                axis_name,
                dict(counter),
                int(current_value),
            )
            return int(current_value), True
        
        # 遍历所有边
        for edge_idx in range(num_edges):
            from_idx = flat_pin_from[edge_idx]
            to_idx = flat_pin_to[edge_idx]
            l_dir = l_directions[edge_idx]
            
            if from_idx < 0 or to_idx < 0:
                continue
            
            # 跳过直线和未知方向
            if l_dir == self.STRAIGHT or l_dir == self.UNKNOWN or l_dir == self.FAKE_STRAIGHT:
                continue
            
            # 确定哪个是Steiner点
            from_is_steiner = from_idx >= num_pins
            to_is_steiner = to_idx >= num_pins
            
            # 如果边连接了Steiner点，需要更新
            if from_is_steiner or to_is_steiner:
                # 找到这两个顶点对应的原始pin索引
                # relate_x/y存储的是pin索引
                from_pin_x = pin_relate_x[from_idx]
                from_pin_y = pin_relate_y[from_idx]
                to_pin_x = pin_relate_x[to_idx]
                to_pin_y = pin_relate_y[to_idx]
                
                if from_is_steiner:
                    # 更新from_idx的relate
                    if l_dir == self.H_FIRST:
                        # 水平优先：corner在(x2, y1)
                        # Steiner点的x来自to方向（水平），y来自自己方向（垂直）
                        new_relate_x = to_pin_x
                        new_relate_y = from_pin_y
                    elif l_dir == self.V_FIRST:
                        # 垂直优先：corner在(x1, y2)
                        # Steiner点的x来自自己方向（垂直），y来自to方向（水平）
                        new_relate_x = from_pin_x
                        new_relate_y = to_pin_y
                    else:
                        continue
                    collect_proposal(from_idx, new_relate_x, new_relate_y)
                
                if to_is_steiner:
                    # 更新to_idx的relate
                    if l_dir == self.H_FIRST:
                        # 水平优先：corner在(x2, y1)
                        # Steiner点的x来自from方向（水平），y来自自己方向（垂直）
                        new_relate_x = from_pin_x
                        new_relate_y = to_pin_y
                    elif l_dir == self.V_FIRST:
                        # 垂直优先：corner在(x1, y2)
                        # Steiner点的x来自自己方向（垂直），y来自from方向（水平）
                        new_relate_x = to_pin_x
                        new_relate_y = from_pin_y
                    else:
                        continue

                    collect_proposal(to_idx, new_relate_x, new_relate_y)

        all_vertices = sorted(set(x_proposals.keys()) | set(y_proposals.keys()))
        for vertex_idx in all_vertices:
            current_x = int(pin_relate_x[vertex_idx])
            current_y = int(pin_relate_y[vertex_idx])
            resolved_x, x_conflict = resolve_axis_proposal(
                vertex_idx,
                "x",
                x_proposals.get(vertex_idx, []),
                current_x,
            )
            resolved_y, y_conflict = resolve_axis_proposal(
                vertex_idx,
                "y",
                y_proposals.get(vertex_idx, []),
                current_y,
            )
            if x_conflict or y_conflict:
                conflict_count += 1

            if current_x != resolved_x or current_y != resolved_y:
                pin_relate_x[vertex_idx] = resolved_x
                pin_relate_y[vertex_idx] = resolved_y
                update_count += 1
        
        # 更新回steiner_topo_op
        steiner_topo_op.pin_relate_x = torch.from_numpy(pin_relate_x).to(
            steiner_topo_op.pin_relate_x.device)
        steiner_topo_op.pin_relate_y = torch.from_numpy(pin_relate_y).to(
            steiner_topo_op.pin_relate_y.device)
        
        logger.info(
            "update_steiner_relate: updated %d Steiner point relates (conflicted_vertices=%d)",
            update_count,
            conflict_count,
        )
        return update_count
    
    def get_l_direction(self, edge_idx):
        """
        获取指定边的L方向
        
        Args:
            edge_idx: 边索引
            
        Returns:
            L方向常量 (H_FIRST, V_FIRST, STRAIGHT, UNKNOWN)
        """
        if self.edge_l_directions is None:
            return self.UNKNOWN
        if edge_idx < 0 or edge_idx >= len(self.edge_l_directions):
            return self.UNKNOWN
        return int(self.edge_l_directions[edge_idx])
    
    def get_l_direction_name(self, direction):
        """将L方向常量转换为可读名称"""
        names = {
            self.H_FIRST: "h_first",
            self.V_FIRST: "vl_first", 
            self.STRAIGHT: "straight",
            self.UNKNOWN: "unknown"
        }
        return names.get(direction, "invalid")


    def debug_print_net_routing(self, net_name):
        """
        打印某个net的EGR routing信息，用于调试
        
        Args:
            net_name: net名称
        """
        if net_name not in self.egr_net_data:
            print(f"Net '{net_name}' not found in EGR data")
            return
        
        net_data = self.egr_net_data[net_name]
        print(f"\n=== Net: {net_name} ===")
        
        print("Pins:")
        for pin in net_data.get('pins', []):
            energy = pin.get('energy', '')
            name = pin.get('name', '')
            grid = pin.get('grid', (0, 0))
            real = pin.get('real', (0, 0))
            print(f"  [{energy}] {name}: grid{grid} real{real}")
        
        print("Wires:")
        for i, wire in enumerate(net_data.get('wires', [])):
            direction = "H" if wire['is_horizontal'] else ("V" if wire['is_vertical'] else "?")
            grid1 = wire.get('grid1', (0, 0))
            grid2 = wire.get('grid2', (0, 0))
            real1 = wire.get('real1', (0, 0))
            real2 = wire.get('real2', (0, 0))
            print(f"  [{i}] {direction}: grid{grid1}->{grid2}  real({real1[0]:.2f},{real1[1]:.2f})->({real2[0]:.2f},{real2[1]:.2f})")
        
        # 尝试重建路径
        print("Path reconstruction:")
        driven_pin = None
        for pin in net_data.get('pins', []):
            if pin.get('energy') == 'driven':
                driven_pin = pin
                break
        
        if driven_pin:
            self._trace_path_from_pin(net_data, driven_pin)
    
    def _trace_path_from_pin(self, net_data, start_pin, tolerance=1.0):
        """
        从起始pin开始追踪wire路径
        """
        wires = net_data.get('wires', [])
        if not wires:
            print("  No wires to trace")
            return
        
        start_x, start_y = start_pin.get('real', (0, 0))
        print(f"  Start from driven pin: ({start_x:.2f}, {start_y:.2f})")
        
        visited = set()
        current_pos = (start_x, start_y)
        path = []
        
        for _ in range(len(wires) + 1):  # 最多遍历wire数量次
            found = False
            for i, wire in enumerate(wires):
                if i in visited:
                    continue
                
                rx1, ry1 = wire['real1']
                rx2, ry2 = wire['real2']
                
                # 检查wire是否从当前位置出发
                at_start = (abs(rx1 - current_pos[0]) < tolerance and abs(ry1 - current_pos[1]) < tolerance)
                at_end = (abs(rx2 - current_pos[0]) < tolerance and abs(ry2 - current_pos[1]) < tolerance)
                
                if at_start:
                    direction = "H" if wire['is_horizontal'] else "V"
                    path.append(f"wire[{i}]({direction}): ({rx1:.1f},{ry1:.1f})->({rx2:.1f},{ry2:.1f})")
                    current_pos = (rx2, ry2)
                    visited.add(i)
                    found = True
                    break
                elif at_end:
                    direction = "H" if wire['is_horizontal'] else "V"
                    path.append(f"wire[{i}]({direction}): ({rx2:.1f},{ry2:.1f})->({rx1:.1f},{ry1:.1f})")
                    current_pos = (rx1, ry1)
                    visited.add(i)
                    found = True
                    break
            
            if not found:
                break
        
        for step in path:
            print(f"    -> {step}")
        
        if path:
            # 判断第一段的方向
            first_step = path[0]
            if "(H)" in first_step:
                print("  => First segment is HORIZONTAL -> H_FIRST")
            elif "(V)" in first_step:
                print("  => First segment is VERTICAL -> V_FIRST")


def create_l_direction_resolver(placedb, params):
    """
    工厂函数，创建EGRLDirectionResolver实例
    
    Args:
        placedb: DREAMPlace placement database
        params: 参数对象
        
    Returns:
        EGRLDirectionResolver实例
    """
    return EGRLDirectionResolver(placedb, params)
