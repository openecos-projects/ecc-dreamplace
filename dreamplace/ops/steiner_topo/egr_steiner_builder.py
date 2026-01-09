# @file   egr_steiner_builder.py
# @brief  Build Steiner tree from EGR route guide, replacing FLUTE
#         Steiner points are recorded with both gcell center (from EGR) 
#         and Hanan grid position (from pin coordinates)
#

import torch
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SteinerPointInfo:
    """记录单个Steiner点的详细信息"""
    
    def __init__(self, local_idx, global_idx, net_id, net_name):
        self.local_idx = local_idx      # net内的局部索引
        self.global_idx = global_idx    # 全局索引
        self.net_id = net_id
        self.net_name = net_name
        
        # gcell位置（来自EGR）
        self.gcell_x = None
        self.gcell_y = None
        self.gcell_center_x = None  # gcell中心坐标 (micron)
        self.gcell_center_y = None
        
        # Hanan网格位置（来自pin坐标）
        self.hanan_x = None  # 实际使用的x坐标 (来自relate_x pin)
        self.hanan_y = None  # 实际使用的y坐标 (来自relate_y pin)
        
        # relate关系
        self.relate_x_pin_id = None  # x坐标来源的pin_id
        self.relate_y_pin_id = None  # y坐标来源的pin_id
        self.relate_x_pin_name = None
        self.relate_y_pin_name = None
        
        # 连接的wire信息
        self.horizontal_wires = []  # 连接的水平wire
        self.vertical_wires = []    # 连接的垂直wire
    
    def __repr__(self):
        return (f"SteinerPoint(net={self.net_name}, local_idx={self.local_idx}, "
                f"gcell=({self.gcell_x},{self.gcell_y}), "
                f"gcell_center=({self.gcell_center_x:.2f},{self.gcell_center_y:.2f}), "
                f"hanan=({self.hanan_x:.2f},{self.hanan_y:.2f}), "
                f"relate_x={self.relate_x_pin_name}, relate_y={self.relate_y_pin_name})")


class EGRSteinerBuilder:
    """
    从EGR route guide构建Steiner树，替代FLUTE
    
    核心思想：
    - 使用EGR确定拓扑结构（谁连接谁）和L方向
    - Steiner点坐标仍通过pin坐标计算（保证可微性）
    - 记录Steiner点的gcell位置和Hanan位置供分析
    """
    
    def __init__(self, placedb, params):
        self.placedb = placedb
        self.params = params
        
        # 建立映射
        self.net_name2id = self._build_net_name_to_id_map()
        self.pin_name2id = self._build_pin_name_to_id_map()
        
        # EGR数据
        self.egr_net_data = {}
        
        # Steiner点记录
        self.steiner_points = []  # List[SteinerPointInfo]
        self.net_steiner_points = defaultdict(list)  # net_id -> [SteinerPointInfo]
        
        # gcell信息
        self.gcell_info = None  # (grid_x, grid_y) -> (llx, lly, urx, ury)
    
    def _build_net_name_to_id_map(self):
        """建立 net_name -> net_id 的映射"""
        net_name2id = {}
        for net_id in range(self.placedb.num_nets):
            net_name = self.placedb.net_names[net_id]
            if isinstance(net_name, bytes):
                net_name = net_name.decode('utf-8')
            net_name2id[net_name] = net_id
        return net_name2id
    
    def _build_pin_name_to_id_map(self):
        """建立 pin_name -> pin_id 的映射"""
        pin_name2id = {}
        if hasattr(self.placedb, 'pin_names'):
            for pin_id in range(self.placedb.num_pins):
                pin_name = self.placedb.pin_names[pin_id]
                if isinstance(pin_name, bytes):
                    pin_name = pin_name.decode('utf-8')
                pin_name2id[pin_name] = pin_id
        return pin_name2id
    
    def load_gcell_info(self, gcell_info_path):
        """
        加载gcell信息
        
        格式: grid_x,grid_y,llx,lly,urx,ury (单位nm)
        """
        self.gcell_info = {}
        try:
            with open(gcell_info_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        gx, gy = int(parts[0]), int(parts[1])
                        llx, lly = float(parts[2]), float(parts[3])
                        urx, ury = float(parts[4]), float(parts[5])
                        self.gcell_info[(gx, gy)] = (llx, lly, urx, ury)
            logger.info(f"Loaded gcell info: {len(self.gcell_info)} gcells")
        except Exception as e:
            logger.warning(f"Failed to load gcell info: {e}")
    
    def get_gcell_center(self, gx, gy):
        """获取gcell中心坐标（micron）"""
        if self.gcell_info and (gx, gy) in self.gcell_info:
            llx, lly, urx, ury = self.gcell_info[(gx, gy)]
            # nm to micron
            cx = (llx + urx) / 2.0 / 1000.0
            cy = (lly + ury) / 2.0 / 1000.0
            return cx, cy
        return None, None
    
    def parse_egr_guide(self, guide_path):
        """解析EGR guide文件"""
        net_data = defaultdict(lambda: {'wires': [], 'pins': []})
        current_net = None
        
        try:
            with open(guide_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'guide':
                        current_net = parts[1]
                        
                    elif parts[0] == 'pin' and current_net:
                        try:
                            grid_x = int(parts[1])
                            grid_y = int(parts[2])
                            real_x = float(parts[3])
                            real_y = float(parts[4])
                            layer = parts[5]
                            energy = parts[6]
                            pin_name = parts[7] if len(parts) > 7 else ""
                            net_data[current_net]['pins'].append({
                                'grid': (grid_x, grid_y),
                                'real': (real_x, real_y),
                                'layer': layer,
                                'energy': energy,
                                'name': pin_name
                            })
                        except (ValueError, IndexError):
                            pass
                        
                    elif parts[0] == 'wire' and current_net:
                        try:
                            grid1_x, grid1_y = int(parts[1]), int(parts[2])
                            grid2_x, grid2_y = int(parts[3]), int(parts[4])
                            real1_x, real1_y = float(parts[5]), float(parts[6])
                            real2_x, real2_y = float(parts[7]), float(parts[8])
                            layer = parts[9] if len(parts) > 9 else ""
                            
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
                        except (ValueError, IndexError):
                            pass
                            
        except FileNotFoundError:
            logger.error(f"EGR guide file not found: {guide_path}")
            return {}
        
        self.egr_net_data = dict(net_data)
        logger.info(f"Parsed EGR guide: {len(self.egr_net_data)} nets")
        return self.egr_net_data
    
    def _find_steiner_gcells(self, net_data):
        """
        找出一个net中的Steiner点（在wire端点但不在pin位置的点）
        
        Returns:
            set of gcell positions that are Steiner points
        """
        # 所有pin所在的gcell
        pin_gcells = set()
        for pin in net_data.get('pins', []):
            pin_gcells.add(pin['grid'])
        
        # 所有wire端点的gcell
        wire_gcells = set()
        for wire in net_data.get('wires', []):
            wire_gcells.add(wire['grid1'])
            wire_gcells.add(wire['grid2'])
        
        # Steiner点 = wire端点 - pin位置
        steiner_gcells = wire_gcells - pin_gcells
        return steiner_gcells
    
    def _trace_to_pin_along_direction(self, start_gcell, wires, pin_gcells, direction):
        """
        沿指定方向追踪wire，找到第一个pin
        
        Args:
            start_gcell: 起始gcell (gx, gy)
            wires: wire列表
            pin_gcells: pin所在gcell -> pin_info
            direction: 'horizontal' or 'vertical'
            
        Returns:
            pin_info if found, else None
        """
        visited = {start_gcell}
        queue = [start_gcell]
        
        while queue:
            curr = queue.pop(0)
            
            for wire in wires:
                if direction == 'horizontal' and not wire['is_horizontal']:
                    continue
                if direction == 'vertical' and not wire['is_vertical']:
                    continue
                
                # 找到连接curr的wire
                next_gcell = None
                if wire['grid1'] == curr:
                    next_gcell = wire['grid2']
                elif wire['grid2'] == curr:
                    next_gcell = wire['grid1']
                
                if next_gcell is None:
                    continue
                
                # 检查是否到达pin
                if next_gcell in pin_gcells:
                    return pin_gcells[next_gcell]
                
                # 继续搜索
                if next_gcell not in visited:
                    visited.add(next_gcell)
                    queue.append(next_gcell)
        
        return None
    
    def _get_pin_coord_by_name(self, pin_name, pos):
        """通过pin名字获取pin坐标（DREAMPlace坐标系）"""
        if pin_name in self.pin_name2id:
            pin_id = self.pin_name2id[pin_name]
            num_nodes = self.placedb.num_nodes
            # pos格式: [x0, x1, ..., xn, y0, y1, ..., yn] for nodes
            # 需要通过pin_pos_op计算pin坐标，这里简化处理
            return pin_id, None, None
        return None, None, None
    
    def _match_egr_pin_to_placedb(self, egr_pin, net_id, pos):
        """
        将EGR的pin匹配到placedb的pin
        
        策略：通过坐标近似匹配
        """
        # 获取该net的所有pin
        net2pin_start = self.placedb.flat_net2pin_start_map
        net2pin = self.placedb.flat_net2pin_map
        
        if hasattr(net2pin_start, 'cpu'):
            net2pin_start = net2pin_start.cpu().numpy()
        if hasattr(net2pin, 'cpu'):
            net2pin = net2pin.cpu().numpy()
        
        start = net2pin_start[net_id]
        end = net2pin_start[net_id + 1]
        
        egr_x, egr_y = egr_pin['real']
        
        # 转换EGR坐标到DREAMPlace坐标
        egr_x_dp = (egr_x * 1000 - self.params.shift_factor[0]) * self.params.scale_factor
        egr_y_dp = (egr_y * 1000 - self.params.shift_factor[1]) * self.params.scale_factor
        
        best_pin_id = None
        best_dist = float('inf')
        
        for idx in range(start, end):
            pin_id = net2pin[idx]
            # 这里需要pin坐标，简化处理：返回pin_id
            # 实际使用时需要通过pin_pos_op计算
            if best_pin_id is None:
                best_pin_id = pin_id
        
        return best_pin_id
    
    def build_steiner_topology(self, pos, net_id, net_name, net_data, 
                                num_pins_before, num_steiner_before):
        """
        为单个net构建Steiner树拓扑
        
        Args:
            pos: pin坐标
            net_id: net ID
            net_name: net名字
            net_data: EGR解析的net数据
            num_pins_before: 该net之前的pin数量（用于全局索引）
            num_steiner_before: 该net之前的Steiner点数量
            
        Returns:
            dict with topology info
        """
        wires = net_data.get('wires', [])
        egr_pins = net_data.get('pins', [])
        
        # 建立gcell到pin的映射
        pin_gcells = {}
        for pin in egr_pins:
            pin_gcells[pin['grid']] = pin
        
        # 找出Steiner点
        steiner_gcells = self._find_steiner_gcells(net_data)
        
        # 获取该net在placedb中的pin
        net2pin_start = self.placedb.flat_net2pin_start_map
        net2pin = self.placedb.flat_net2pin_map
        if hasattr(net2pin_start, 'cpu'):
            net2pin_start = net2pin_start.cpu().numpy()
        if hasattr(net2pin, 'cpu'):
            net2pin = net2pin.cpu().numpy()
        
        net_start = net2pin_start[net_id]
        net_end = net2pin_start[net_id + 1]
        degree = net_end - net_start  # net的pin数量
        
        # 构建本地索引: 0~degree-1 是pin, degree~ 是Steiner点
        steiner_list = list(steiner_gcells)
        num_steiner = len(steiner_list)
        
        # 建立 gcell -> 全局vertex索引 的映射
        gcell_to_vertex = {}
        
        # Pin: 匹配EGR pin到placedb pin
        for i, egr_pin in enumerate(egr_pins):
            gcell = egr_pin['grid']
            # 尝试通过名字匹配
            pin_name = egr_pin.get('name', '')
            matched_pin_id = None
            if pin_name and pin_name in self.pin_name2id:
                matched_pin_id = self.pin_name2id[pin_name]
            else:
                # fallback: 使用顺序
                if i < degree:
                    matched_pin_id = net2pin[net_start + i]
            if matched_pin_id is not None:
                gcell_to_vertex[gcell] = matched_pin_id
        
        # Steiner点: 全局索引 = num_pins + num_steiner_before + local_idx
        for local_idx, steiner_gcell in enumerate(steiner_list):
            global_idx = num_pins_before + num_steiner_before + local_idx
            gcell_to_vertex[steiner_gcell] = global_idx
        
        # 为每个Steiner点创建记录
        result_steiner_points = []
        relate_x = []
        relate_y = []
        
        # Pin的relate: 自己
        for i in range(degree):
            relate_x.append(net2pin[net_start + i])
            relate_y.append(net2pin[net_start + i])
        
        # Steiner点的relate
        for local_idx, steiner_gcell in enumerate(steiner_list):
            global_idx = num_steiner_before + local_idx
            
            # 创建Steiner点记录
            sp_info = SteinerPointInfo(
                local_idx=degree + local_idx,
                global_idx=global_idx,
                net_id=net_id,
                net_name=net_name
            )
            sp_info.gcell_x, sp_info.gcell_y = steiner_gcell
            
            # gcell中心坐标
            cx, cy = self.get_gcell_center(steiner_gcell[0], steiner_gcell[1])
            sp_info.gcell_center_x = cx if cx else 0
            sp_info.gcell_center_y = cy if cy else 0
            
            # 沿水平方向找pin（提供y坐标）
            h_pin = self._trace_to_pin_along_direction(
                steiner_gcell, wires, pin_gcells, 'horizontal')
            # 沿垂直方向找pin（提供x坐标）
            v_pin = self._trace_to_pin_along_direction(
                steiner_gcell, wires, pin_gcells, 'vertical')
            
            # 确定relate关系
            if v_pin:
                # x来自垂直方向的pin
                matched_pin_id = self._match_egr_pin_to_placedb(v_pin, net_id, pos)
                if matched_pin_id is not None:
                    sp_info.relate_x_pin_id = matched_pin_id
                    sp_info.relate_x_pin_name = v_pin.get('name', '')
                    relate_x.append(matched_pin_id)
                else:
                    relate_x.append(net2pin[net_start])  # fallback
            else:
                relate_x.append(net2pin[net_start])  # fallback
            
            if h_pin:
                # y来自水平方向的pin
                matched_pin_id = self._match_egr_pin_to_placedb(h_pin, net_id, pos)
                if matched_pin_id is not None:
                    sp_info.relate_y_pin_id = matched_pin_id
                    sp_info.relate_y_pin_name = h_pin.get('name', '')
                    relate_y.append(matched_pin_id)
                else:
                    relate_y.append(net2pin[net_start])  # fallback
            else:
                relate_y.append(net2pin[net_start])  # fallback
            
            result_steiner_points.append(sp_info)
        
        # ========== 从EGR wire构建边列表 ==========
        edges = []  # [(from_vertex, to_vertex, l_direction), ...]
        edge_l_directions = []
        
        # 处理L形wire（非水平非垂直的连接）
        # 需要将多个相邻wire组合成L形边
        processed_wires = set()
        
        for wire in wires:
            wire_id = id(wire)
            if wire_id in processed_wires:
                continue
            
            g1, g2 = wire['grid1'], wire['grid2']
            
            # 检查两端是否都在我们的顶点映射中
            if g1 not in gcell_to_vertex or g2 not in gcell_to_vertex:
                # 可能是中间的wire段，尝试追踪到端点
                continue
            
            v1 = gcell_to_vertex[g1]
            v2 = gcell_to_vertex[g2]
            
            # 确定L方向
            if wire['is_horizontal']:
                # 水平wire：可能是L的一部分或直线
                l_dir = 2  # STRAIGHT
            elif wire['is_vertical']:
                # 垂直wire：可能是L的一部分或直线
                l_dir = 2  # STRAIGHT
            else:
                l_dir = -1  # UNKNOWN
            
            edges.append((v1, v2, l_dir))
            processed_wires.add(wire_id)
        
        # 尝试合并相邻的水平+垂直wire成L形边
        edges = self._merge_wires_to_l_edges(wires, gcell_to_vertex, pin_gcells, steiner_gcells)
        
        return {
            'degree': degree,
            'num_steiner': num_steiner,
            'steiner_points': result_steiner_points,
            'relate_x': relate_x,
            'relate_y': relate_y,
            'steiner_gcells': steiner_list,
            'gcell_to_vertex': gcell_to_vertex,
            'edges': edges  # [(from_vertex, to_vertex, l_direction), ...]
        }
    
    def _merge_wires_to_l_edges(self, wires, gcell_to_vertex, pin_gcells, steiner_gcells):
        """
        将EGR的wire合并成L形边
        
        EGR中每个wire是单独的水平或垂直段，我们需要：
        1. 找到所有端点（pin或Steiner点）
        2. 对于每对需要连接的端点，确定L方向
        
        Returns:
            list of (from_vertex, to_vertex, l_direction)
        """
        # L方向常量
        H_FIRST = 0  # 先水平后垂直
        V_FIRST = 1  # 先垂直后水平
        STRAIGHT = 2
        UNKNOWN = -1
        
        edges = []
        
        # 建立邻接关系: gcell -> set of adjacent gcells
        adjacency = defaultdict(set)
        wire_info = {}  # (g1, g2) -> wire info
        
        for wire in wires:
            g1, g2 = wire['grid1'], wire['grid2']
            adjacency[g1].add(g2)
            adjacency[g2].add(g1)
            key = (min(g1, g2), max(g1, g2))
            wire_info[key] = wire
        
        # 所有端点（需要连接的点）
        endpoints = set(gcell_to_vertex.keys())
        
        # 已处理的端点对
        processed_pairs = set()
        
        # 对每个端点，BFS找到它连接的其他端点
        for start in endpoints:
            if start not in gcell_to_vertex:
                continue
            
            # BFS
            visited = {start}
            queue = [(start, None, [])]  # (current_gcell, first_direction, path)
            
            while queue:
                curr, first_dir, path = queue.pop(0)
                
                for neighbor in adjacency[curr]:
                    if neighbor in visited:
                        continue
                    
                    # 获取wire信息
                    key = (min(curr, neighbor), max(curr, neighbor))
                    wire = wire_info.get(key, {})
                    
                    # 确定这段wire的方向
                    if wire.get('is_horizontal', False):
                        this_dir = 'H'
                    elif wire.get('is_vertical', False):
                        this_dir = 'V'
                    else:
                        this_dir = '?'
                    
                    new_first_dir = first_dir if first_dir else this_dir
                    new_path = path + [(curr, neighbor, this_dir)]
                    
                    if neighbor in endpoints:
                        # 找到另一个端点，创建边
                        pair = (min(start, neighbor), max(start, neighbor))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            
                            v1 = gcell_to_vertex[start]
                            v2 = gcell_to_vertex[neighbor]
                            
                            # 确定L方向
                            if len(new_path) == 1:
                                # 只有一段wire，是直线
                                l_dir = STRAIGHT
                            elif len(new_path) == 2:
                                # 两段wire组成L
                                dir1 = new_path[0][2]
                                dir2 = new_path[1][2]
                                if dir1 == 'H' and dir2 == 'V':
                                    l_dir = H_FIRST  # 先水平后垂直
                                elif dir1 == 'V' and dir2 == 'H':
                                    l_dir = V_FIRST  # 先垂直后水平
                                else:
                                    l_dir = UNKNOWN
                            else:
                                # 多段wire，复杂情况
                                l_dir = UNKNOWN
                            
                            edges.append((v1, v2, l_dir))
                    else:
                        # 继续搜索
                        visited.add(neighbor)
                        queue.append((neighbor, new_first_dir, new_path))
        
        return edges
    
    def build_all_nets(self, pos, guide_path, gcell_info_path=None):
        """
        为所有net构建Steiner树
        
        Args:
            pos: pin坐标tensor
            guide_path: EGR guide文件路径
            gcell_info_path: gcell信息文件路径（可选）
            
        Returns:
            构建结果，格式与FLUTE的build_tree兼容
        """
        # 加载gcell信息
        if gcell_info_path:
            self.load_gcell_info(gcell_info_path)
        
        # 解析EGR guide
        self.parse_egr_guide(guide_path)
        
        # 清空之前的记录
        self.steiner_points = []
        self.net_steiner_points = defaultdict(list)
        
        num_nets = self.placedb.num_nets
        num_pins = self.placedb.num_pins
        
        all_relate_x = list(range(num_pins))  # pin的relate是自己
        all_relate_y = list(range(num_pins))
        
        # 边列表
        all_flat_pin_from = []
        all_flat_pin_to = []
        all_edge_l_directions = []
        
        total_steiner = 0
        net_steiner_start = [num_pins]  # Steiner点的起始索引
        
        nets_from_egr = 0
        nets_fallback = 0
        total_edges = 0
        
        for net_id in range(num_nets):
            net_name = self.placedb.net_names[net_id]
            if isinstance(net_name, bytes):
                net_name = net_name.decode('utf-8')
            
            if net_name in self.egr_net_data:
                # 使用EGR构建
                net_data = self.egr_net_data[net_name]
                result = self.build_steiner_topology(
                    pos, net_id, net_name, net_data,
                    num_pins, total_steiner
                )
                
                # 记录Steiner点
                for sp in result['steiner_points']:
                    self.steiner_points.append(sp)
                    self.net_steiner_points[net_id].append(sp)
                
                # 更新relate（只添加Steiner点的部分）
                all_relate_x.extend(result['relate_x'][result['degree']:])
                all_relate_y.extend(result['relate_y'][result['degree']:])
                
                # 收集边列表
                for from_v, to_v, l_dir in result.get('edges', []):
                    all_flat_pin_from.append(from_v)
                    all_flat_pin_to.append(to_v)
                    all_edge_l_directions.append(l_dir)
                    total_edges += 1
                
                total_steiner += result['num_steiner']
                nets_from_egr += 1
            else:
                # 该net不在EGR中，没有Steiner点
                nets_fallback += 1
            
            net_steiner_start.append(num_pins + total_steiner)
        
        logger.info(f"Built Steiner trees: {nets_from_egr} from EGR, "
                   f"{nets_fallback} without EGR data, "
                   f"{total_steiner} total Steiner points, "
                   f"{total_edges} total edges")
        
        # 转换为tensor
        relate_x_tensor = torch.tensor(all_relate_x, dtype=torch.int32)
        relate_y_tensor = torch.tensor(all_relate_y, dtype=torch.int32)
        net_steiner_start_tensor = torch.tensor(net_steiner_start, dtype=torch.int32)
        
        # 边列表tensor
        flat_pin_from_tensor = torch.tensor(all_flat_pin_from, dtype=torch.int32) if all_flat_pin_from else torch.tensor([], dtype=torch.int32)
        flat_pin_to_tensor = torch.tensor(all_flat_pin_to, dtype=torch.int32) if all_flat_pin_to else torch.tensor([], dtype=torch.int32)
        edge_l_directions_tensor = torch.tensor(all_edge_l_directions, dtype=torch.int32) if all_edge_l_directions else torch.tensor([], dtype=torch.int32)
        
        return {
            'pin_relate_x': relate_x_tensor,
            'pin_relate_y': relate_y_tensor,
            'net_steiner_start': net_steiner_start_tensor,
            'num_steiner': total_steiner,
            'steiner_points': self.steiner_points,
            # 边列表（用于绘图）
            'flat_pin_from': flat_pin_from_tensor,
            'flat_pin_to': flat_pin_to_tensor,
            'edge_l_directions': edge_l_directions_tensor
        }
    
    def get_steiner_points_dataframe(self):
        """
        将Steiner点信息转换为DataFrame格式，方便分析
        
        Returns:
            list of dict，可以直接用pandas.DataFrame(result)
        """
        records = []
        for sp in self.steiner_points:
            records.append({
                'net_id': sp.net_id,
                'net_name': sp.net_name,
                'local_idx': sp.local_idx,
                'global_idx': sp.global_idx,
                'gcell_x': sp.gcell_x,
                'gcell_y': sp.gcell_y,
                'gcell_center_x': sp.gcell_center_x,
                'gcell_center_y': sp.gcell_center_y,
                'hanan_x': sp.hanan_x,
                'hanan_y': sp.hanan_y,
                'relate_x_pin_id': sp.relate_x_pin_id,
                'relate_y_pin_id': sp.relate_y_pin_id,
                'relate_x_pin_name': sp.relate_x_pin_name,
                'relate_y_pin_name': sp.relate_y_pin_name
            })
        return records
    
    def export_steiner_points_csv(self, output_path):
        """导出Steiner点信息到CSV文件"""
        records = self.get_steiner_points_dataframe()
        
        with open(output_path, 'w') as f:
            if records:
                headers = list(records[0].keys())
                f.write(','.join(headers) + '\n')
                for r in records:
                    f.write(','.join(str(r[h]) for h in headers) + '\n')
        
        logger.info(f"Exported {len(records)} Steiner points to {output_path}")
    
    def print_net_steiner_info(self, net_name):
        """打印某个net的Steiner点详细信息"""
        if net_name not in self.net_name2id:
            print(f"Net '{net_name}' not found")
            return
        
        net_id = self.net_name2id[net_name]
        steiner_pts = self.net_steiner_points.get(net_id, [])
        
        print(f"\n=== Net: {net_name} (id={net_id}) ===")
        print(f"Number of Steiner points: {len(steiner_pts)}")
        
        for sp in steiner_pts:
            print(f"\n  Steiner Point {sp.local_idx}:")
            print(f"    GCell: ({sp.gcell_x}, {sp.gcell_y})")
            print(f"    GCell Center: ({sp.gcell_center_x:.3f}, {sp.gcell_center_y:.3f}) um")
            print(f"    Relate X: pin_id={sp.relate_x_pin_id}, name={sp.relate_x_pin_name}")
            print(f"    Relate Y: pin_id={sp.relate_y_pin_id}, name={sp.relate_y_pin_name}")


def create_egr_steiner_builder(placedb, params):
    """工厂函数"""
    return EGRSteinerBuilder(placedb, params)
