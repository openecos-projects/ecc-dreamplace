# @file   egr_l_direction.py
# @brief  Parse EGR route guide and determine L-shape direction for Steiner tree edges
#

import torch
import numpy as np
import logging
from collections import defaultdict

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
        
        # L方向结果: edge_idx -> direction
        self.edge_l_directions = None
        
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
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Failed to parse wire line: {line}, error: {e}")
                            
        except FileNotFoundError:
            logger.error(f"EGR guide file not found: {guide_path}")
            return {}
        
        self.egr_net_data = dict(net_data)
        
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
        logger.info(f"Parsed gpugr route entries: {len(self.egr_net_data)} nets")
        return self.egr_net_data
    
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
    
    def _find_wire_at_point(self, net_data, x, y, tolerance=1.0):
        """
        找到经过给定点的wire
        
        Args:
            net_data: net的数据 {'wires': [...], 'pins': [...]}
            x, y: 点坐标 (micron)
            tolerance: 坐标匹配容差 (micron)
            
        Returns:
            list of wires that pass through or start/end at the point
        """
        matching_wires = []
        for wire in net_data.get('wires', []):
            rx1, ry1 = wire['real1']
            rx2, ry2 = wire['real2']
            
            # 检查点是否在wire的端点
            at_start = (abs(rx1 - x) < tolerance and abs(ry1 - y) < tolerance)
            at_end = (abs(rx2 - x) < tolerance and abs(ry2 - y) < tolerance)
            
            if at_start or at_end:
                matching_wires.append({
                    **wire,
                    'at_start': at_start,
                    'at_end': at_end
                })
        
        return matching_wires
    
    def _find_first_wire_direction_from_point(self, net_data, start_x, start_y, tolerance=1.0):
        """
        从给定起点出发，找到第一段wire的方向
        
        Args:
            net_data: net的数据
            start_x, start_y: 起点坐标 (micron)
            tolerance: 坐标匹配容差 (micron)
            
        Returns:
            'horizontal', 'vertical', or None
        """
        matching_wires = self._find_wire_at_point(net_data, start_x, start_y, tolerance)
        
        for wire in matching_wires:
            if wire['is_horizontal']:
                return 'horizontal'
            elif wire['is_vertical']:
                return 'vertical'
        
        return None
    
    def _determine_l_direction_for_edge(self, net_data, p1_um, p2_um, eps=1e-5):
        """
        判断从p1到p2的边应该走H_FIRST还是V_FIRST
        
        通过查找EGR中的wire序列来判断：
        - 如果从p1出发的第一段wire是水平的 -> H_FIRST (先水平后垂直)
        - 如果从p1出发的第一段wire是垂直的 -> V_FIRST (先垂直后水平)
        
        Args:
            net_data: net的数据 {'wires': [...], 'pins': [...]}
            p1_um, p2_um: 边的两个端点 (micron)
            tolerance: 坐标匹配容差 (micron)
            
        Returns:
            H_FIRST, V_FIRST, STRAIGHT, or UNKNOWN
        """
        x1, y1 = p1_um
        x2, y2 = p2_um
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        # 几乎重合
        if dx < eps and dy < eps:
            return self.UNKNOWN
        
        if dx < eps and dy > eps:
            return self.STRAIGHT  # 垂直线
        if dy < eps and dx > eps:
            return self.STRAIGHT  # 水平线
        
        # H_FIRST: (x1,y1) -> (x2,y1) -> (x2,y2), 拐点在 (x2, y1)
        # V_FIRST: (x1,y1) -> (x1,y2) -> (x2,y2), 拐点在 (x1, y2)
        h_first_corner = (x2, y1)
        v_first_corner = (x1, y2)
        
        wires = net_data.get('wires', [])

        if len(wires) == 1:
            return self.FAKE_STRAIGHT
        
        point_count = defaultdict(int)

        for wire in wires:
            rx1, ry1 = wire['real1']
            rx2, ry2 = wire['real2']

            point_count[(rx1, ry1)] += 1
            point_count[(rx2, ry2)] += 1
            
        corners = [point for point, count in point_count.items() if count > 1]
        
        # 检查每个拐点距离 h_first_corner 还是 v_first_corner 更近
        for corner in corners:
            cx, cy = corner
            dist_h = abs(cx - h_first_corner[0]) + abs(cy - h_first_corner[1])
            dist_v = abs(cx - v_first_corner[0]) + abs(cy - v_first_corner[1])
            
            if dist_h < dist_v:
                return self.H_FIRST
            else:
                return self.V_FIRST
        
        return self.UNKNOWN
    
    def resolve_l_directions(self, steiner_topo_op, guide_path=None):
        """
        解析所有Steiner树边的L方向（优化版本）
        
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
        
        num_edges = len(flat_pin_from)
        num_pins = self.placedb.num_pins
        num_nets = self.placedb.num_nets
        
        # ========== 优化1: 预计算所有顶点的net_id ==========
        vertex_to_net = self._precompute_vertex_to_net(num_pins, net_steiner_start, len(newx))
        
        # ========== 优化2: 向量化坐标转换 ==========
        # DREAMPlace坐标 -> micron (一次性转换所有顶点)
        scale = self.params.scale_factor
        shift_x, shift_y = self.params.shift_factor
        dbu = self.placedb.dbu
        
        newx_um = (newx / scale + shift_x) / dbu
        newy_um = (newy / scale + shift_y) / dbu
        
        # ========== 优化3: 预构建 net_name -> net_data 的快速查找 ==========
        # 将net_id直接映射到net_data（避免字符串查找）
        net_id_to_data = {}
        for net_id in range(num_nets):
            net_name = self.placedb.net_names[net_id]
            if isinstance(net_name, bytes):
                net_name = net_name.decode('utf-8')
            if net_name in self.egr_net_data:
                net_id_to_data[net_id] = self.egr_net_data[net_name]
        
        # ========== 优化4: 向量化过滤无效边 ==========
        valid_mask = (flat_pin_from >= 0) & (flat_pin_to >= 0)
        valid_indices = np.where(valid_mask)[0]
        
        # 结果数组
        l_directions = np.full(num_edges, self.UNKNOWN, dtype=np.int32)
        
        # 统计
        stats = {self.H_FIRST: 0, self.V_FIRST: 0, self.STRAIGHT: 0, self.FAKE_STRAIGHT: 0, self.UNKNOWN: 0}
        
        # ========== 优化5: 只遍历有效边 ==========
        for edge_idx in valid_indices:
            from_idx = flat_pin_from[edge_idx]
            to_idx = flat_pin_to[edge_idx]
            
            # 直接使用预计算的坐标（已转换为micron）
            p1_um = (newx_um[from_idx], newy_um[from_idx])
            p2_um = (newx_um[to_idx], newy_um[to_idx])
            
            # 使用预计算的 vertex -> net_id 映射
            net_id = vertex_to_net[from_idx]
            if net_id < 0 or net_id >= num_nets:
                continue
            
            # 使用预构建的 net_id -> net_data 映射（避免字符串查找）
            if net_id not in net_id_to_data:
                continue
            
            net_data = net_id_to_data[net_id]
            
            # 判断L方向
            direction = self._determine_l_direction_for_edge(net_data, p1_um, p2_um)
            l_directions[edge_idx] = direction
            stats[direction] += 1
        
        logger.info(f"L direction resolution: h_first={stats[self.H_FIRST]}, "
                   f"v_first={stats[self.V_FIRST]}, straight={stats[self.STRAIGHT]}, "
                   f"fake_straight={stats[self.FAKE_STRAIGHT]}, unknown={stats[self.UNKNOWN]}")
        
        self.edge_l_directions = torch.from_numpy(l_directions)
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
        newx = steiner_topo_op.newx.cpu().numpy()
        newy = steiner_topo_op.newy.cpu().numpy()
        l_directions = self.edge_l_directions.cpu().numpy()
        
        num_pins = self.placedb.num_pins
        num_vertices = len(pin_relate_x)
        num_edges = len(flat_pin_from)
        
        # 统计更新次数
        update_count = 0
        
        # 遍历所有边
        for edge_idx in range(num_edges):
            from_idx = flat_pin_from[edge_idx]
            to_idx = flat_pin_to[edge_idx]
            l_dir = l_directions[edge_idx]
            
            if from_idx < 0 or to_idx < 0:
                continue
            
            # 跳过直线和未知方向
            if l_dir == self.STRAIGHT or l_dir == self.UNKNOWN:
                continue
            
            # 确定哪个是Steiner点
            from_is_steiner = from_idx >= num_pins
            to_is_steiner = to_idx >= num_pins
            
            # 如果边连接了Steiner点，需要更新
            if from_is_steiner or to_is_steiner:
                # 获取两端点坐标
                x1, y1 = newx[from_idx], newy[from_idx]
                x2, y2 = newx[to_idx], newy[to_idx]
                
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
                    elif l_dir == self.V_FIRST or l_dir == self.FAKE_STRAIGHT:
                        # 垂直优先：corner在(x1, y2)
                        # Steiner点的x来自自己方向（垂直），y来自to方向（水平）
                        new_relate_x = from_pin_x
                        new_relate_y = to_pin_y
                    else:
                        continue
                    
                    if pin_relate_x[from_idx] != new_relate_x or pin_relate_y[from_idx] != new_relate_y:
                        pin_relate_x[from_idx] = new_relate_x
                        pin_relate_y[from_idx] = new_relate_y
                        update_count += 1
                
                if to_is_steiner:
                    # 更新to_idx的relate
                    if l_dir == self.H_FIRST:
                        # 水平优先：corner在(x2, y1)
                        # Steiner点的x来自from方向（水平），y来自自己方向（垂直）
                        new_relate_x = from_pin_x
                        new_relate_y = to_pin_y
                    elif l_dir == self.V_FIRST or l_dir == self.FAKE_STRAIGHT:
                        # 垂直优先：corner在(x1, y2)
                        # Steiner点的x来自自己方向（垂直），y来自from方向（水平）
                        new_relate_x = to_pin_x
                        new_relate_y = from_pin_y
                    else:
                        continue
                    
                    if pin_relate_x[to_idx] != new_relate_x or pin_relate_y[to_idx] != new_relate_y:
                        pin_relate_x[to_idx] = new_relate_x
                        pin_relate_y[to_idx] = new_relate_y
                        update_count += 1
        
        # 更新回steiner_topo_op
        steiner_topo_op.pin_relate_x = torch.from_numpy(pin_relate_x).to(
            steiner_topo_op.pin_relate_x.device)
        steiner_topo_op.pin_relate_y = torch.from_numpy(pin_relate_y).to(
            steiner_topo_op.pin_relate_y.device)
        
        logger.info(f"update_steiner_relate: updated {update_count} Steiner point relates")
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
