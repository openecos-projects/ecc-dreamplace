"""Original Python matching oracle; production uses the C++ batch matcher."""

import logging
import time
from collections import Counter, defaultdict

import numpy as np
from dreamplace.ops.steiner_topo.egr_l_direction import EGRLDirectionResolver

logger = logging.getLogger(__name__)


class PythonLDirectionReference(EGRLDirectionResolver):
    def __init__(self, placedb, params):
        super().__init__(placedb, params)
        self._reset_route_lookup_caches()

    def _reset_route_lookup_caches(self):
        self._route_snap_cache_x = {}
        self._route_snap_cache_y = {}
        self._route_axis_index_cache_x = {}
        self._route_axis_index_cache_y = {}

    def _point_cache_key(self, x, y, scale=1e6):
        return (int(round(float(x) * scale)), int(round(float(y) * scale)))

    def _ensure_net_lookup_cache(self, net_data):
        cache = net_data.get("_resolver_cache")
        if cache is not None:
            return cache

        endpoint_index = defaultdict(list)
        point_count = Counter()
        point_value = {}
        for wire in net_data.get("wires", []):
            p1 = wire["real1"]
            p2 = wire["real2"]
            key1 = self._point_cache_key(*p1)
            key2 = self._point_cache_key(*p2)
            endpoint_index[key1].append(wire)
            endpoint_index[key2].append(wire)
            point_count[key1] += 1
            point_count[key2] += 1
            point_value.setdefault(key1, (float(p1[0]), float(p1[1])))
            point_value.setdefault(key2, (float(p2[0]), float(p2[1])))

        cache = {
            "endpoint_index": dict(endpoint_index),
            "shared_points": [point_value[key] for key, count in point_count.items() if count > 1],
        }
        net_data["_resolver_cache"] = cache
        return cache

    def _snap_value_to_axis(self, value, coords):
        if coords is None or len(coords) == 0:
            return float(value)
        cache = None
        if coords is self._route_grid_x_um:
            cache = self._route_snap_cache_x
        elif coords is self._route_grid_y_um:
            cache = self._route_snap_cache_y
        key = float(value)
        if cache is not None and key in cache:
            return cache[key]
        idx = int(np.searchsorted(coords, value))
        candidates = []
        if idx < len(coords):
            candidates.append(coords[idx])
        if idx > 0:
            candidates.append(coords[idx - 1])
        snapped = (
            float(min(candidates, key=lambda candidate: abs(candidate - value)))
            if candidates
            else float(value)
        )
        if cache is not None:
            cache[key] = snapped
        return snapped

    def _snap_point_to_route_grid(self, net_data, point_um):
        if net_data.get("source") != "gpugr":
            return (float(point_um[0]), float(point_um[1]))
        return (
            self._snap_value_to_axis(point_um[0], self._route_grid_x_um),
            self._snap_value_to_axis(point_um[1], self._route_grid_y_um),
        )

    def _axis_index(self, coords, value):
        if coords is None or len(coords) == 0:
            return None
        cache = None
        if coords is self._route_grid_x_um:
            cache = self._route_axis_index_cache_x
        elif coords is self._route_grid_y_um:
            cache = self._route_axis_index_cache_y
        key = float(value)
        if cache is not None and key in cache:
            return cache[key]
        idx = int(np.searchsorted(coords, value))
        candidates = []
        if idx < len(coords):
            candidates.append((abs(coords[idx] - value), idx))
        if idx > 0:
            candidates.append((abs(coords[idx - 1] - value), idx - 1))
        result = min(candidates)[1] if candidates else None
        if cache is not None:
            cache[key] = result
        return result

    def _path_axis_indices(self, p1_um, p2_um):
        return (
            self._axis_index(self._route_grid_x_um, p1_um[0]),
            self._axis_index(self._route_grid_y_um, p1_um[1]),
            self._axis_index(self._route_grid_x_um, p2_um[0]),
            self._axis_index(self._route_grid_y_um, p2_um[1]),
        )

    def _add_horizontal_usage_indices(self, usage_map, x_idx1, x_idx2, y_idx):
        if usage_map is None or x_idx1 is None or x_idx2 is None or y_idx is None:
            return
        lo, hi = sorted((x_idx1, x_idx2))
        usage_map[lo : hi + 1, y_idx] += 1.0

    def _add_vertical_usage_indices(self, usage_map, x_idx, y_idx1, y_idx2):
        if usage_map is None or x_idx is None or y_idx1 is None or y_idx2 is None:
            return
        lo, hi = sorted((y_idx1, y_idx2))
        usage_map[x_idx, lo : hi + 1] += 1.0

    def _accumulate_path_usage_indices(self, usage_h, usage_v, axis_indices, direction):
        x_idx1, y_idx1, x_idx2, y_idx2 = axis_indices
        if direction == self.H_FIRST:
            self._add_horizontal_usage_indices(usage_h, x_idx1, x_idx2, y_idx1)
            self._add_vertical_usage_indices(usage_v, x_idx2, y_idx1, y_idx2)
        elif direction == self.V_FIRST:
            self._add_vertical_usage_indices(usage_v, x_idx1, y_idx1, y_idx2)
            self._add_horizontal_usage_indices(usage_h, x_idx1, x_idx2, y_idx2)
        elif direction == self.STRAIGHT:
            if y_idx1 == y_idx2 and y_idx1 is not None:
                self._add_horizontal_usage_indices(usage_h, x_idx1, x_idx2, y_idx1)
            elif x_idx1 == x_idx2 and x_idx1 is not None:
                self._add_vertical_usage_indices(usage_v, x_idx1, y_idx1, y_idx2)

    def _score_candidate_path_indices(self, usage_h, usage_v, axis_indices, direction):
        x_idx1, y_idx1, x_idx2, y_idx2 = axis_indices
        if None in axis_indices:
            return 0.0

        cost = 0.0
        if direction == self.H_FIRST:
            x_lo, x_hi = sorted((x_idx1, x_idx2))
            y_lo, y_hi = sorted((y_idx1, y_idx2))
            cost += float(usage_h[x_lo : x_hi + 1, y_idx1].sum())
            cost += float(usage_v[x_idx2, y_lo : y_hi + 1].sum())
        elif direction == self.V_FIRST:
            x_lo, x_hi = sorted((x_idx1, x_idx2))
            y_lo, y_hi = sorted((y_idx1, y_idx2))
            cost += float(usage_v[x_idx1, y_lo : y_hi + 1].sum())
            cost += float(usage_h[x_lo : x_hi + 1, y_idx2].sum())
        elif direction == self.STRAIGHT:
            if y_idx1 == y_idx2:
                x_lo, x_hi = sorted((x_idx1, x_idx2))
                cost += float(usage_h[x_lo : x_hi + 1, y_idx1].sum())
            elif x_idx1 == x_idx2:
                y_lo, y_hi = sorted((y_idx1, y_idx2))
                cost += float(usage_v[x_idx1, y_lo : y_hi + 1].sum())
        return cost

    def _fallback_unresolved_with_path_maps(self, edge_records, l_directions, tie_tol=1e-6):
        if self._route_grid_x_um is None or self._route_grid_y_um is None:
            return 0

        usage_h = np.zeros(
            (len(self._route_grid_x_um), len(self._route_grid_y_um)), dtype=np.float32
        )
        usage_v = np.zeros_like(usage_h)
        usage_build_start = time.perf_counter()

        for record in edge_records:
            direction = int(l_directions[record["edge_idx"]])
            if direction == self.UNKNOWN or direction == self.FAKE_STRAIGHT:
                continue
            self._accumulate_path_usage_indices(
                usage_h, usage_v, record["path_axis_indices"], direction
            )

        fallback_count = 0
        fake_straight_to_unknown = 0
        unresolved_count = 0
        scoring_start = time.perf_counter()
        for record in edge_records:
            edge_idx = record["edge_idx"]
            current_direction = int(l_directions[edge_idx])
            if current_direction not in (self.UNKNOWN, self.FAKE_STRAIGHT):
                continue
            unresolved_count += 1

            cost_h = self._score_candidate_path_indices(
                usage_h, usage_v, record["path_axis_indices"], self.H_FIRST
            )
            cost_v = self._score_candidate_path_indices(
                usage_h, usage_v, record["path_axis_indices"], self.V_FIRST
            )

            if abs(cost_h - cost_v) <= tie_tol:
                if current_direction == self.FAKE_STRAIGHT:
                    l_directions[edge_idx] = self.UNKNOWN
                    fake_straight_to_unknown += 1
                continue

            l_directions[edge_idx] = self.H_FIRST if cost_h < cost_v else self.V_FIRST
            fallback_count += 1

        if fallback_count > 0 or fake_straight_to_unknown > 0:
            logger.info(
                "Known-path map fallback resolved %d unresolved edges and "
                "downgraded %d FAKE_STRAIGHT ties to UNKNOWN.",
                fallback_count,
                fake_straight_to_unknown,
            )
        logger.info(
            "Known-path map fallback timing: unresolved=%d usage_build=%.2fms scoring=%.2fms",
            unresolved_count,
            (scoring_start - usage_build_start) * 1000.0,
            (time.perf_counter() - scoring_start) * 1000.0,
        )
        return fallback_count + fake_straight_to_unknown

    def _wire_distance_metrics(self, wire, x, y):
        rx1, ry1 = wire["real1"]
        rx2, ry2 = wire["real2"]
        endpoint_distance = min(
            float(np.hypot(rx1 - x, ry1 - y)),
            float(np.hypot(rx2 - x, ry2 - y)),
        )

        if wire.get("is_horizontal", False):
            lo, hi = sorted((rx1, rx2))
            closest_x = min(max(x, lo), hi)
            closest_y = ry1
        elif wire.get("is_vertical", False):
            lo, hi = sorted((ry1, ry2))
            closest_x = rx1
            closest_y = min(max(y, lo), hi)
        else:
            candidates = [(rx1, ry1), (rx2, ry2)]
            closest_x, closest_y = min(
                candidates,
                key=lambda point: float(np.hypot(point[0] - x, point[1] - y)),
            )

        segment_distance = float(np.hypot(closest_x - x, closest_y - y))
        return endpoint_distance, segment_distance

    def _wire_progress_score(self, wire, start_x, start_y, target_point=None):
        if target_point is None:
            return 0.0

        target_x, target_y = target_point
        current_distance = abs(target_x - start_x) + abs(target_y - start_y)
        rx1, ry1 = wire["real1"]
        rx2, ry2 = wire["real2"]

        candidates = []
        if wire.get("is_horizontal", False):
            lo, hi = sorted((rx1, rx2))
            candidates.append((min(max(target_x, lo), hi), ry1))
        elif wire.get("is_vertical", False):
            lo, hi = sorted((ry1, ry2))
            candidates.append((rx1, min(max(target_y, lo), hi)))
        else:
            candidates.extend([(rx1, ry1), (rx2, ry2)])

        best_next_distance = min(abs(target_x - cx) + abs(target_y - cy) for cx, cy in candidates)
        return float(current_distance - best_next_distance)

    def _find_wire_at_point(
        self,
        net_data,
        x,
        y,
        *,
        endpoint_tolerance=1.0,
        segment_tolerance=1.0,
        target_point=None,
        endpoint_only=False,
    ):
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
        candidate_wires = net_data.get("wires", [])
        if endpoint_only:
            cache = self._ensure_net_lookup_cache(net_data)
            indexed_wires = cache["endpoint_index"].get(self._point_cache_key(x, y))
            if indexed_wires:
                candidate_wires = indexed_wires

        for wire in candidate_wires:
            rx1, ry1 = wire["real1"]
            rx2, ry2 = wire["real2"]

            # 先检查端点命中，再检查是否落在线段内部。
            at_start = abs(rx1 - x) <= endpoint_tolerance and abs(ry1 - y) <= endpoint_tolerance
            at_end = abs(rx2 - x) <= endpoint_tolerance and abs(ry2 - y) <= endpoint_tolerance
            min_x = min(rx1, rx2) - segment_tolerance
            max_x = max(rx1, rx2) + segment_tolerance
            min_y = min(ry1, ry2) - segment_tolerance
            max_y = max(ry1, ry2) + segment_tolerance

            if wire.get("is_horizontal", False):
                on_segment = abs(ry1 - y) <= segment_tolerance and min_x <= x <= max_x
            elif wire.get("is_vertical", False):
                on_segment = abs(rx1 - x) <= segment_tolerance and min_y <= y <= max_y
            else:
                on_segment = min_x <= x <= max_x and min_y <= y <= max_y

            matched = at_start or at_end if endpoint_only else at_start or at_end or on_segment

            if matched:
                endpoint_distance, segment_distance = self._wire_distance_metrics(wire, x, y)
                matching_wires.append(
                    {
                        **wire,
                        "at_start": at_start,
                        "at_end": at_end,
                        "on_segment": on_segment,
                        "endpoint_distance": endpoint_distance,
                        "segment_distance": segment_distance,
                        "progress_score": self._wire_progress_score(wire, x, y, target_point),
                    }
                )

        matching_wires.sort(
            key=lambda wire: (
                0 if (wire["at_start"] or wire["at_end"]) else 1,
                -wire["progress_score"],
                wire["segment_distance"],
                wire["endpoint_distance"],
                int(wire.get("order", 0)),
            )
        )

        return matching_wires

    def _find_first_wire_direction_from_point(
        self,
        net_data,
        start_x,
        start_y,
        tolerance=1.0,
        target_point=None,
    ):
        """
        从给定起点出发，找到第一段wire的方向

        Args:
            net_data: net的数据
            start_x, start_y: 起点坐标 (micron)
            tolerance: 坐标匹配容差 (micron)

        Returns:
            'horizontal', 'vertical', or None
        """
        endpoint_tolerance = self._endpoint_match_tolerance(net_data, tolerance)

        for endpoint_only in (True, False):
            matching_wires = self._find_wire_at_point(
                net_data,
                start_x,
                start_y,
                endpoint_tolerance=endpoint_tolerance,
                segment_tolerance=tolerance,
                target_point=target_point,
                endpoint_only=endpoint_only,
            )

            for wire in matching_wires:
                if wire["is_horizontal"]:
                    return "horizontal"
                if wire["is_vertical"]:
                    return "vertical"

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

        wires = net_data.get("wires", [])

        if len(wires) == 1:
            return self.FAKE_STRAIGHT

        match_p1_um = self._snap_point_to_route_grid(net_data, p1_um)
        match_p2_um = self._snap_point_to_route_grid(net_data, p2_um)

        for tolerance in self._candidate_match_tolerances(net_data):
            first_dir_from_p1 = self._find_first_wire_direction_from_point(
                net_data,
                match_p1_um[0],
                match_p1_um[1],
                tolerance=tolerance,
                target_point=match_p2_um,
            )
            if first_dir_from_p1 == "horizontal":
                return self.H_FIRST
            if first_dir_from_p1 == "vertical":
                return self.V_FIRST

            # 从 p2 反推时，末段方向与 p1 出发的首段方向互补。
            first_dir_from_p2 = self._find_first_wire_direction_from_point(
                net_data,
                match_p2_um[0],
                match_p2_um[1],
                tolerance=tolerance,
                target_point=match_p1_um,
            )
            if first_dir_from_p2 == "horizontal":
                return self.V_FIRST
            if first_dir_from_p2 == "vertical":
                return self.H_FIRST

        corners = self._ensure_net_lookup_cache(net_data)["shared_points"]

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
