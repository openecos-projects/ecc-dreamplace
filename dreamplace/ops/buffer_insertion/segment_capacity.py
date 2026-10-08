"""Static per-bin resource control for fixed-placement segment buffering.

The controller deliberately models only the added area of a relaxed segment
repeater count.  It does not create virtual placement nodes or modify the
timing graph.  Geometry preparation runs once; optimizer steps use packed
device tensors and scatter-add reductions.
"""

# TODO(segment-capacity): This controller remains experimental. Do not promote
# it to the default or use it as paper-main QoR evidence until an in-memory,
# projection-aware integer state closes the final-only round_clip gap, the
# projected/committed DEF bin replay is validated, and multi-design QoR/runtime
# gates are closed.

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch


RHO_BUFFER = 0.01
PHR_MU = 1.0
INITIALIZATION_MARGIN = 1.0e-6
RELAXED_FEASIBILITY_TOLERANCE = 1.0e-4
SUPPORTED_GRID_FACTORS = (4, 8)
_DEF_COMPONENT_PLACEMENT_RE = re.compile(
    r"^\s*-\s+(?P<name>\S+)\s+(?P<master>\S+).*?"
    r"\+\s+(?:PLACED|FIXED|COVER)\s+\(\s*(?P<x>-?\d+)\s+(?P<y>-?\d+)\s*\)\s+"
    r"(?P<orientation>\S+)",
    flags=re.MULTILINE,
)


def _finite_positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _json_dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    return abs(
        sum(
            float(vertices[index][0]) * float(vertices[(index + 1) % len(vertices)][1])
            - float(vertices[(index + 1) % len(vertices)][0]) * float(vertices[index][1])
            for index in range(len(vertices))
        )
        * 0.5
    )


def _clip_polygon(vertices, *, inside, intersect):
    if not vertices:
        return []
    result = []
    previous = vertices[-1]
    previous_inside = inside(previous)
    for current in vertices:
        current_inside = inside(current)
        if current_inside != previous_inside:
            result.append(intersect(previous, current))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _clip_polygon_to_rect(vertices, xl, yl, xh, yh):
    """Sutherland-Hodgman clipping against an axis-aligned rectangle."""

    def vertical_intersection(x_value, first, second):
        dx = float(second[0]) - float(first[0])
        if abs(dx) <= 1.0e-30:
            return float(x_value), float(first[1])
        ratio = (float(x_value) - float(first[0])) / dx
        return float(x_value), float(first[1]) + ratio * (float(second[1]) - float(first[1]))

    def horizontal_intersection(y_value, first, second):
        dy = float(second[1]) - float(first[1])
        if abs(dy) <= 1.0e-30:
            return float(first[0]), float(y_value)
        ratio = (float(y_value) - float(first[1])) / dy
        return float(first[0]) + ratio * (float(second[0]) - float(first[0])), float(y_value)

    clipped = list(vertices)
    clipped = _clip_polygon(
        clipped,
        inside=lambda point: float(point[0]) >= float(xl),
        intersect=lambda first, second: vertical_intersection(xl, first, second),
    )
    clipped = _clip_polygon(
        clipped,
        inside=lambda point: float(point[0]) <= float(xh),
        intersect=lambda first, second: vertical_intersection(xh, first, second),
    )
    clipped = _clip_polygon(
        clipped,
        inside=lambda point: float(point[1]) >= float(yl),
        intersect=lambda first, second: horizontal_intersection(yl, first, second),
    )
    return _clip_polygon(
        clipped,
        inside=lambda point: float(point[1]) <= float(yh),
        intersect=lambda first, second: horizontal_intersection(yh, first, second),
    )


def _swept_polygon(parent, child, row_height):
    dx = float(child[0]) - float(parent[0])
    dy = float(child[1]) - float(parent[1])
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("segment footprint requires nonzero finite length")
    normal_x = -dy / length
    normal_y = dx / length
    half_height = 0.5 * float(row_height)
    return (
        (float(parent[0]) + half_height * normal_x, float(parent[1]) + half_height * normal_y),
        (float(child[0]) + half_height * normal_x, float(child[1]) + half_height * normal_y),
        (float(child[0]) - half_height * normal_x, float(child[1]) - half_height * normal_y),
        (float(parent[0]) - half_height * normal_x, float(parent[1]) - half_height * normal_y),
    )


def _interval_bin_range(edges, low, high):
    if high <= edges[0] or low >= edges[-1]:
        return range(0)
    low = max(float(low), float(edges[0]))
    high = min(float(high), float(edges[-1]))
    first = max(0, bisect_right(edges, low) - 1)
    last = min(len(edges) - 2, bisect_left(edges, high) - 1)
    return range(first, last + 1) if last >= first else range(0)


def _raw_to_normalized(raw_coordinate, shift, scale):
    return (
        (float(raw_coordinate[0]) - float(shift[0])) * float(scale),
        (float(raw_coordinate[1]) - float(shift[1])) * float(scale),
    )


def _segment_raw_coordinates(segment_row):
    return (
        (float(segment_row["parent_x_dbu"]), float(segment_row["parent_y_dbu"])),
        (float(segment_row["child_x_dbu"]), float(segment_row["child_y_dbu"])),
    )


def _group_base_capacity(base_capacity, *, xl, yl, bin_size_x, bin_size_y, factor):
    if base_capacity.dim() != 2:
        raise ValueError("base_capacity must have shape [num_bins_x, num_bins_y]")
    base_x, base_y = (int(value) for value in base_capacity.shape)
    if base_x <= 0 or base_y <= 0:
        raise ValueError("base_capacity must have positive grid dimensions")
    factor = int(factor)
    if factor not in SUPPORTED_GRID_FACTORS:
        raise ValueError(f"unsupported capacity grid factor {factor}")
    x_edges = tuple(
        float(xl) + float(min(index * factor, base_x)) * float(bin_size_x)
        for index in range((base_x + factor - 1) // factor + 1)
    )
    y_edges = tuple(
        float(yl) + float(min(index * factor, base_y)) * float(bin_size_y)
        for index in range((base_y + factor - 1) // factor + 1)
    )
    groups = []
    for x_index in range(len(x_edges) - 1):
        x_begin = x_index * factor
        x_end = min(base_x, x_begin + factor)
        for y_index in range(len(y_edges) - 1):
            y_begin = y_index * factor
            y_end = min(base_y, y_begin + factor)
            groups.append(base_capacity[x_begin:x_end, y_begin:y_end].sum())
    return x_edges, y_edges, torch.stack(groups)


@dataclass(frozen=True)
class SegmentCapacityGrid:
    factor: int
    x_edges: tuple
    y_edges: tuple
    capacity: torch.Tensor
    base_num_bins_x: int
    base_num_bins_y: int
    base_bin_size_x: float
    base_bin_size_y: float
    fixed_area_source: str

    @property
    def num_bins_x(self):
        return len(self.x_edges) - 1

    @property
    def num_bins_y(self):
        return len(self.y_edges) - 1

    @property
    def num_bins(self):
        return int(self.capacity.numel())

    def bin_bounds(self, bin_index):
        bin_index = int(bin_index)
        x_index = bin_index // self.num_bins_y
        y_index = bin_index % self.num_bins_y
        return (
            self.x_edges[x_index],
            self.y_edges[y_index],
            self.x_edges[x_index + 1],
            self.y_edges[y_index + 1],
        )

    def to_metadata(self):
        return {
            "factor": int(self.factor),
            "num_bins_x": int(self.num_bins_x),
            "num_bins_y": int(self.num_bins_y),
            "base_num_bins_x": int(self.base_num_bins_x),
            "base_num_bins_y": int(self.base_num_bins_y),
            "base_bin_size_x": float(self.base_bin_size_x),
            "base_bin_size_y": float(self.base_bin_size_y),
            "x_edges": [float(value) for value in self.x_edges],
            "y_edges": [float(value) for value in self.y_edges],
            "fixed_area_source": str(self.fixed_area_source),
        }


@dataclass
class SegmentCapacityController:
    state: object
    grid: SegmentCapacityGrid
    segment_index: torch.Tensor
    bin_index: torch.Tensor
    normalized_weight: torch.Tensor
    buffer_area: float
    buffer_width: float
    buffer_height: float
    row_height: float
    rho: float
    raw_shift: tuple
    raw_scale: float
    fixed_cell_id: int
    fixed_master_name: str
    dual_state: torch.Tensor
    geometry_summary: dict
    initialization_summary: dict = field(default_factory=dict)
    timing_scale: float | None = None
    trace: list = field(default_factory=list)
    projection_fidelity: dict | None = None

    def __post_init__(self):
        if int(self.segment_index.numel()) != int(self.bin_index.numel()):
            raise ValueError("segment_index and bin_index must have equal length")
        if int(self.segment_index.numel()) != int(self.normalized_weight.numel()):
            raise ValueError("segment_index and normalized_weight must have equal length")
        if int(self.dual_state.numel()) != int(self.grid.num_bins):
            raise ValueError("dual_state size must equal capacity grid size")

    @property
    def allowance(self):
        return self.grid.capacity * float(self.rho)

    def added_area(self, z_value):
        z_value = z_value.reshape(-1)
        if int(z_value.numel()) != len(self.state.segment_rows):
            raise ValueError("z value count does not match segment geometry")
        area = torch.zeros_like(self.grid.capacity)
        contribution = (
            z_value[self.segment_index]
            * self.normalized_weight
            * float(self.buffer_area)
        )
        return area.scatter_add(0, self.bin_index, contribution)

    def capacity_state(self, z_value):
        area = self.added_area(z_value)
        allowance = self.allowance
        positive = allowance > 0.0
        if not bool(positive.any()):
            raise ValueError("segment capacity grid has no positive-capacity bins")
        ratio = torch.zeros_like(area)
        ratio[positive] = area[positive] / allowance[positive]
        return area, ratio, ratio - 1.0, positive

    def initial_ratio(self):
        with torch.no_grad():
            _, ratio, _, positive = self.capacity_state(self.state.z_value())
            return ratio[positive].detach()

    def apply_initial_scale_(self, alpha, *, shared_grid_factors):
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError(f"capacity initialization alpha must be in (0, 1], got {alpha}")
        with torch.no_grad():
            self.state.z_param.mul_(alpha)
            _, ratio, _, positive = self.capacity_state(self.state.z_value())
            positive_ratio = ratio[positive]
        self.initialization_summary = {
            "status": "ok",
            "alpha": float(alpha),
            "initialization_margin": float(INITIALIZATION_MARGIN),
            "shared_grid_factors": [int(value) for value in shared_grid_factors],
            "r_init_max_after_scale": float(positive_ratio.max().cpu().item()),
            "r_init_p99_after_scale": float(
                torch.quantile(positive_ratio.float(), 0.99).cpu().item()
            ),
            "z_initial_sum": float(self.state.z_value().sum().detach().cpu().item()),
            "z_initial_mean": float(self.state.z_value().mean().detach().cpu().item()),
        }
        return dict(self.initialization_summary)

    def compose_objective(self, timing_loss):
        if self.timing_scale is None:
            self.timing_scale = max(abs(_scalar(timing_loss.detach())), 1.0)
        z_value = self.state.z_value()
        area, ratio, violation, positive = self.capacity_state(z_value)
        shifted = torch.relu(self.dual_state + float(PHR_MU) * violation)
        phi = (shifted.square() - self.dual_state.square()) / (2.0 * float(PHR_MU))
        phi = torch.where(positive, phi, torch.zeros_like(phi))
        capacity_term = float(self.timing_scale) * phi.sum()
        loss = timing_loss + capacity_term
        return loss, self._metric_snapshot(area, ratio, violation, capacity_term)

    def update_dual_after_step(self, iteration, *, objective_metrics=None):
        with torch.no_grad():
            area, ratio, violation, positive = self.capacity_state(self.state.z_value())
            self.dual_state[positive] = torch.relu(
                self.dual_state[positive] + float(PHR_MU) * violation[positive]
            )
            self.dual_state[~positive] = 0.0
            metrics = self._metric_snapshot(area, ratio, violation, None)
            metrics["capacity_iteration"] = int(iteration)
            if objective_metrics is not None:
                for source, target in (
                    ("timing_loss", "timing_loss_pre_step"),
                    ("tns", "tns_pre_step"),
                    ("wns", "wns_pre_step"),
                ):
                    value = objective_metrics.get(source)
                    if value is not None:
                        metrics[target] = _scalar(value)
            z_value = self.state.z_value()
            metrics["z_value_distribution"] = {
                "count": int(z_value.numel()),
                "min": float(z_value.min().detach().cpu().item()),
                "mean": float(z_value.mean().detach().cpu().item()),
                "max": float(z_value.max().detach().cpu().item()),
                "sum": float(z_value.sum().detach().cpu().item()),
                "ge_0p5_count": int((z_value >= 0.5).sum().detach().cpu().item()),
                "ge_1p0_count": int((z_value >= 1.0).sum().detach().cpu().item()),
            }
            self.trace.append(dict(metrics))
        return metrics

    def _metric_snapshot(self, area, ratio, violation, capacity_term):
        positive = self.allowance > 0.0
        positive_ratio = ratio[positive]
        dual = self.dual_state[positive]
        result = {
            "segment_capacity_enabled": True,
            "segment_capacity_grid_factor": int(self.grid.factor),
            "segment_capacity_total_virtual_area": float(area.sum().detach().cpu().item()),
            "segment_capacity_total_allowance": float(self.allowance.sum().detach().cpu().item()),
            "segment_capacity_max_ratio": float(positive_ratio.max().detach().cpu().item()),
            "segment_capacity_p50_ratio": float(torch.quantile(positive_ratio.float(), 0.50).detach().cpu().item()),
            "segment_capacity_p90_ratio": float(torch.quantile(positive_ratio.float(), 0.90).detach().cpu().item()),
            "segment_capacity_p99_ratio": float(torch.quantile(positive_ratio.float(), 0.99).detach().cpu().item()),
            "segment_capacity_max_violation": float(violation[positive].max().detach().cpu().item()),
            "segment_capacity_violating_bin_count": int((violation[positive] > 0.0).sum().detach().cpu().item()),
            "segment_capacity_positive_bin_count": int(positive.sum().detach().cpu().item()),
            "segment_capacity_dual_max": float(dual.max().detach().cpu().item()),
            "segment_capacity_dual_mean": float(dual.mean().detach().cpu().item()),
            "segment_capacity_timing_scale": self.timing_scale,
            "segment_capacity_mu": float(PHR_MU),
        }
        if capacity_term is not None:
            result["segment_capacity_objective_term"] = float(
                capacity_term.detach().cpu().item()
            )
        return result

    def projection_map(self, actions):
        output = torch.zeros_like(self.grid.capacity)
        outside_area = 0.0
        for action in list(actions or []):
            raw_x = float(action.get("candidate_location_x_dbu", action.get("x_dbu", 0.0)))
            raw_y = float(action.get("candidate_location_y_dbu", action.get("y_dbu", 0.0)))
            x, y = _raw_to_normalized((raw_x, raw_y), self.raw_shift, self.raw_scale)
            for bin_index, overlap in _rectangle_bin_overlaps(
                x,
                y,
                x + float(self.buffer_width),
                y + float(self.buffer_height),
                self.grid,
            ):
                output[bin_index] += float(overlap)
            covered = sum(
                overlap
                for _, overlap in _rectangle_bin_overlaps(
                    x,
                    y,
                    x + float(self.buffer_width),
                    y + float(self.buffer_height),
                    self.grid,
                )
            )
            outside_area += max(0.0, float(self.buffer_area) - float(covered))
        return output, outside_area

    def close_projection_fidelity(self, actions):
        with torch.no_grad():
            relaxed, _, _, positive = self.capacity_state(self.state.z_value())
            projected, outside_area = self.projection_map(actions)
            allowance = self.allowance
            difference = (projected - relaxed).abs()
            normalized = torch.zeros_like(difference)
            normalized[positive] = difference[positive] / allowance[positive]
            ratio_mask = positive & (relaxed > 0.0)
            projected_to_relaxed = projected[ratio_mask] / relaxed[ratio_mask]
            overshoot = torch.zeros_like(projected)
            overshoot[positive] = torch.relu(projected[positive] - allowance[positive])
            relaxed_feasible_projected_infeasible = (relaxed <= allowance) & (projected > allowance)
            one_quantum_pass = bool((overshoot[positive] <= float(self.buffer_area) + 1.0e-7).all())
            payload = {
                "artifact": "segment_capacity_projection_fidelity",
                "artifact_version": 1,
                "status": "pass" if one_quantum_pass else "fidelity_fail",
                "grid_factor": int(self.grid.factor),
                "buffer_area": float(self.buffer_area),
                "projected_action_count": int(len(list(actions or []))),
                "e_inf": float(normalized[positive].max().cpu().item()),
                "projected_to_relaxed_ratio_p50": _quantile_or_none(projected_to_relaxed, 0.50),
                "projected_to_relaxed_ratio_p90": _quantile_or_none(projected_to_relaxed, 0.90),
                "projected_to_relaxed_ratio_p99": _quantile_or_none(projected_to_relaxed, 0.99),
                "relaxed_feasible_projected_infeasible_bin_count": int(
                    relaxed_feasible_projected_infeasible.sum().cpu().item()
                ),
                "projected_max_overshoot_in_buffer_quanta": float(
                    (overshoot[positive] / float(self.buffer_area)).max().cpu().item()
                ),
                "projected_outside_core_area": float(outside_area),
                "one_buffer_quantum_gate": bool(one_quantum_pass),
                "committed_status": "pending_openroad_legalized_map",
                "relaxed_area_by_bin": relaxed.detach().cpu().tolist(),
                "projected_area_by_bin": projected.detach().cpu().tolist(),
                "allowance_by_bin": allowance.detach().cpu().tolist(),
            }
        self.projection_fidelity = payload
        return dict(payload)

    def artifact_payloads(self):
        geometry = dict(self.geometry_summary)
        geometry["grid"] = self.grid.to_metadata()
        geometry["capacity_by_bin"] = self.grid.capacity.detach().cpu().tolist()
        return {
            "segment_capacity_config.json": {
                "artifact": "segment_capacity_config",
                "artifact_version": 1,
                "rho_buffer": float(self.rho),
                "phr_mu": float(PHR_MU),
                "timing_scale": self.timing_scale,
                "fixed_cell_id": int(self.fixed_cell_id),
                "fixed_master_name": str(self.fixed_master_name),
                "buffer_width": float(self.buffer_width),
                "buffer_height": float(self.buffer_height),
                "buffer_area": float(self.buffer_area),
                "row_height": float(self.row_height),
                "raw_shift": [float(value) for value in self.raw_shift],
                "raw_scale": float(self.raw_scale),
                "periodic_integer_projection_interval": 0,
                "grid": self.grid.to_metadata(),
            },
            "segment_capacity_geometry.json": geometry,
            "segment_capacity_initialization.json": dict(self.initialization_summary),
            "segment_capacity_trace.jsonl": list(self.trace),
            "segment_capacity_projection_fidelity.json": dict(
                self.projection_fidelity
                or {
                    "artifact": "segment_capacity_projection_fidelity",
                    "status": "pending_final_projection",
                }
            ),
        }

    def write_artifacts(self, output_dir):
        root = Path(output_dir)
        payloads = self.artifact_payloads()
        for name, payload in payloads.items():
            path = root / name
            if name.endswith(".jsonl"):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as stream:
                    for row in payload:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                _json_dump(path, payload)
        return {name: str(root / name) for name in payloads}


@dataclass
class SegmentCapacityBundle:
    controllers: dict
    selected_grid_factor: int
    grid_failures: dict
    shared_alpha: float
    shared_initialization: dict

    @property
    def selected(self):
        return self.controllers[int(self.selected_grid_factor)]

    def to_summary(self):
        return {
            "segment_capacity_enabled": True,
            "segment_capacity_selected_grid_factor": int(self.selected_grid_factor),
            "segment_capacity_available_grid_factors": sorted(int(key) for key in self.controllers),
            "segment_capacity_grid_failures": dict(self.grid_failures),
            "segment_capacity_shared_alpha": float(self.shared_alpha),
            "segment_capacity_shared_initialization": dict(self.shared_initialization),
        }


def _quantile_or_none(values, quantile):
    if int(values.numel()) == 0:
        return None
    return float(torch.quantile(values.float(), float(quantile)).cpu().item())


def _rectangle_bin_overlaps(xl, yl, xh, yh, grid):
    if xh <= xl or yh <= yl:
        return []
    overlaps = []
    for x_index in _interval_bin_range(grid.x_edges, xl, xh):
        for y_index in _interval_bin_range(grid.y_edges, yl, yh):
            bin_xl = grid.x_edges[x_index]
            bin_yl = grid.y_edges[y_index]
            bin_xh = grid.x_edges[x_index + 1]
            bin_yh = grid.y_edges[y_index + 1]
            overlap_x = max(0.0, min(float(xh), bin_xh) - max(float(xl), bin_xl))
            overlap_y = max(0.0, min(float(yh), bin_yh) - max(float(yl), bin_yl))
            area = overlap_x * overlap_y
            if area > 0.0:
                overlaps.append((x_index * grid.num_bins_y + y_index, area))
    return overlaps


def _def_component_placements(path):
    """Read placed component origins from a finalized DEF without OpenDB."""

    path = Path(path)
    placements = {}
    for match in _DEF_COMPONENT_PLACEMENT_RE.finditer(
        path.read_text(encoding="utf-8", errors="ignore")
    ):
        name = match.group("name")
        if name in placements:
            raise ValueError(f"DEF contains duplicate component placement for {name}")
        placements[name] = {
            "master": match.group("master"),
            "x_dbu": int(match.group("x")),
            "y_dbu": int(match.group("y")),
            "orientation": match.group("orientation"),
        }
    return placements


def _grid_from_capacity_artifacts(config, projection):
    metadata = dict(config.get("grid") or {})
    x_edges = tuple(float(value) for value in metadata.get("x_edges", ()))
    y_edges = tuple(float(value) for value in metadata.get("y_edges", ()))
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise ValueError("segment capacity config is missing coarse-grid edges")
    allowance = torch.as_tensor(
        projection.get("allowance_by_bin", ()),
        dtype=torch.float64,
    ).reshape(-1)
    expected_bins = (len(x_edges) - 1) * (len(y_edges) - 1)
    if int(allowance.numel()) != expected_bins:
        raise ValueError(
            "segment capacity allowance does not match coarse-grid dimensions"
        )
    rho = _finite_positive(config.get("rho_buffer"), "artifact rho_buffer")
    return SegmentCapacityGrid(
        factor=int(metadata.get("factor", 0)),
        x_edges=x_edges,
        y_edges=y_edges,
        capacity=allowance / rho,
        base_num_bins_x=int(metadata.get("base_num_bins_x", 0)),
        base_num_bins_y=int(metadata.get("base_num_bins_y", 0)),
        base_bin_size_x=float(metadata.get("base_bin_size_x", 0.0)),
        base_bin_size_y=float(metadata.get("base_bin_size_y", 0.0)),
        fixed_area_source=str(metadata.get("fixed_area_source", "artifact")),
    )


def _ratio_percentiles(numerator, denominator, mask):
    values = numerator[mask] / denominator[mask]
    return {
        "p50": _quantile_or_none(values, 0.50),
        "p90": _quantile_or_none(values, 0.90),
        "p99": _quantile_or_none(values, 0.99),
    }


def close_committed_def_fidelity(
    *,
    projection_path,
    config_path,
    committed_def_path,
    committed_instance_names,
):
    """Append the committed/legalized buffer map to a final-projection artifact.

    The OpenROAD replay DEF is authoritative for physical buffer origins.  This
    post-processing is deliberately outside the relaxed timing loop: it only
    compares the final committed state with the already-recorded relaxed and
    round-clip projected maps.
    """

    projection_path = Path(projection_path)
    config_path = Path(config_path)
    committed_def_path = Path(committed_def_path)
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    grid = _grid_from_capacity_artifacts(config, projection)
    expected_names = tuple(sorted({str(value) for value in committed_instance_names}))
    if not expected_names:
        raise ValueError("committed fidelity requires at least one added buffer instance")
    placements = _def_component_placements(committed_def_path)
    missing = [name for name in expected_names if name not in placements]
    if missing:
        raise ValueError(
            "committed DEF is missing added buffer placement(s): " + ", ".join(missing[:5])
        )

    buffer_width = _finite_positive(config.get("buffer_width"), "artifact buffer_width")
    buffer_height = _finite_positive(config.get("buffer_height"), "artifact buffer_height")
    buffer_area = _finite_positive(config.get("buffer_area"), "artifact buffer_area")
    raw_shift = tuple(float(value) for value in config.get("raw_shift", ()))
    if len(raw_shift) != 2:
        raise ValueError("segment capacity config raw_shift must contain two values")
    raw_scale = _finite_positive(config.get("raw_scale"), "artifact raw_scale")

    committed = torch.zeros_like(grid.capacity)
    outside_area = 0.0
    for name in expected_names:
        placement = placements[name]
        x, y = _raw_to_normalized(
            (placement["x_dbu"], placement["y_dbu"]),
            raw_shift,
            raw_scale,
        )
        overlaps = _rectangle_bin_overlaps(
            x,
            y,
            x + buffer_width,
            y + buffer_height,
            grid,
        )
        for bin_index, overlap in overlaps:
            committed[bin_index] += float(overlap)
        outside_area += max(0.0, buffer_area - sum(overlap for _, overlap in overlaps))

    allowance = torch.as_tensor(projection["allowance_by_bin"], dtype=torch.float64)
    relaxed = torch.as_tensor(projection["relaxed_area_by_bin"], dtype=torch.float64)
    projected = torch.as_tensor(projection["projected_area_by_bin"], dtype=torch.float64)
    if not (
        int(allowance.numel())
        == int(relaxed.numel())
        == int(projected.numel())
        == int(committed.numel())
    ):
        raise ValueError("segment capacity artifacts have inconsistent bin-map lengths")
    positive = allowance > 0.0
    overshoot = torch.zeros_like(committed)
    overshoot[positive] = torch.relu(committed[positive] - allowance[positive])
    committed_quantum_pass = bool(
        (overshoot[positive] <= buffer_area + 1.0e-7).all()
    )
    committed_relaxed_error = torch.zeros_like(committed)
    committed_projected_error = torch.zeros_like(committed)
    committed_relaxed_error[positive] = (
        (committed[positive] - relaxed[positive]).abs() / allowance[positive]
    )
    committed_projected_error[positive] = (
        (committed[positive] - projected[positive]).abs() / allowance[positive]
    )
    relaxed_mask = positive & (relaxed > 0.0)
    projected_mask = positive & (projected > 0.0)
    relaxed_feasible_committed_infeasible = (
        (relaxed <= allowance) & (committed > allowance)
    )
    projected_feasible_committed_infeasible = (
        (projected <= allowance) & (committed > allowance)
    )
    exact_count = int(projection.get("projected_action_count", -1)) == len(expected_names)
    status = "pass" if (
        projection.get("status") == "pass"
        and bool(projection.get("one_buffer_quantum_gate"))
        and committed_quantum_pass
        and exact_count
    ) else "fidelity_fail"
    projection.update(
        {
            "status": status,
            "committed_status": status,
            "committed_def": str(committed_def_path),
            "committed_instance_count": len(expected_names),
            "committed_instance_names": list(expected_names),
            "committed_exact_action_count_match": bool(exact_count),
            "committed_area_by_bin": committed.tolist(),
            "committed_total_area": float(committed.sum().item()),
            "committed_outside_core_area": float(outside_area),
            "committed_e_inf_vs_relaxed": float(
                committed_relaxed_error[positive].max().item()
            ),
            "committed_e_inf_vs_projected": float(
                committed_projected_error[positive].max().item()
            ),
            "committed_to_relaxed_ratio": _ratio_percentiles(
                committed,
                relaxed,
                relaxed_mask,
            ),
            "committed_to_projected_ratio": _ratio_percentiles(
                committed,
                projected,
                projected_mask,
            ),
            "committed_max_overshoot_in_buffer_quanta": float(
                (overshoot[positive] / buffer_area).max().item()
            ),
            "committed_one_buffer_quantum_gate": bool(committed_quantum_pass),
            "relaxed_feasible_committed_infeasible_bin_count": int(
                relaxed_feasible_committed_infeasible.sum().item()
            ),
            "projected_feasible_committed_infeasible_bin_count": int(
                projected_feasible_committed_infeasible.sum().item()
            ),
        }
    )
    _json_dump(projection_path, projection)
    return projection


def capacity_grid_from_base_map(
    base_capacity,
    *,
    xl,
    yl,
    bin_size_x,
    bin_size_y,
    factor,
    fixed_area_source="density_op_initial_fixed_map",
):
    """Build a static coarse capacity grid from already normalized base bins."""

    _finite_positive(bin_size_x, "base bin_size_x")
    _finite_positive(bin_size_y, "base bin_size_y")
    if not torch.is_tensor(base_capacity):
        base_capacity = torch.as_tensor(base_capacity, dtype=torch.float32)
    if not bool(torch.isfinite(base_capacity).all()):
        raise ValueError("base capacity map contains non-finite values")
    base_capacity = base_capacity.clamp_min(0.0)
    x_edges, y_edges, capacity = _group_base_capacity(
        base_capacity,
        xl=float(xl),
        yl=float(yl),
        bin_size_x=float(bin_size_x),
        bin_size_y=float(bin_size_y),
        factor=int(factor),
    )
    return SegmentCapacityGrid(
        factor=int(factor),
        x_edges=x_edges,
        y_edges=y_edges,
        capacity=capacity.reshape(-1),
        base_num_bins_x=int(base_capacity.shape[0]),
        base_num_bins_y=int(base_capacity.shape[1]),
        base_bin_size_x=float(bin_size_x),
        base_bin_size_y=float(bin_size_y),
        fixed_area_source=str(fixed_area_source),
    )


def build_segment_capacity_controller(
    state,
    *,
    grid,
    row_height,
    buffer_width,
    buffer_height,
    fixed_cell_id,
    fixed_master_name,
    raw_shift=(0.0, 0.0),
    raw_scale=1.0,
    rho=RHO_BUFFER,
):
    """Prepare one static swept-footprint controller for one coarse grid."""

    row_height = _finite_positive(row_height, "row_height")
    buffer_width = _finite_positive(buffer_width, "buffer_width")
    buffer_height = _finite_positive(buffer_height, "buffer_height")
    raw_scale = _finite_positive(raw_scale, "raw_scale")
    rho = _finite_positive(rho, "rho")
    if not isinstance(grid, SegmentCapacityGrid):
        raise TypeError("grid must be a SegmentCapacityGrid")
    if abs(buffer_height - row_height) > 1.0e-6 * max(1.0, row_height):
        raise ValueError(
            "fixed buffer master must be one row high for the segment capacity profile"
        )
    if int(state.z_param.numel()) != len(state.segment_rows):
        raise ValueError("segment state tensor count does not match segment rows")

    segment_index = []
    bin_index = []
    normalized_weight = []
    invalid_rows = []
    zero_capacity_crossings = []
    footprint_area_sum = 0.0
    in_core_footprint_area_sum = 0.0
    core_clipped_segment_count = 0
    core_xl, core_xh = grid.x_edges[0], grid.x_edges[-1]
    core_yl, core_yh = grid.y_edges[0], grid.y_edges[-1]
    positive_capacity = grid.capacity.detach().cpu()

    for row_index, row in enumerate(state.segment_rows):
        try:
            raw_parent, raw_child = _segment_raw_coordinates(row)
            parent = _raw_to_normalized(raw_parent, raw_shift, raw_scale)
            child = _raw_to_normalized(raw_child, raw_shift, raw_scale)
            polygon = _swept_polygon(parent, child, row_height)
            polygon_area = _polygon_area(polygon)
            expected_area = math.hypot(
                float(child[0]) - float(parent[0]),
                float(child[1]) - float(parent[1]),
            ) * row_height
            if abs(polygon_area - expected_area) > 1.0e-6 * max(1.0, expected_area):
                raise ValueError("swept polygon area is inconsistent with length times row height")
            in_core_polygon = _clip_polygon_to_rect(
                polygon,
                core_xl,
                core_yl,
                core_xh,
                core_yh,
            )
            in_core_area = _polygon_area(in_core_polygon)
            if in_core_area <= 1.0e-12 * max(1.0, polygon_area):
                raise ValueError(
                    "segment swept footprint has no positive in-core area "
                    f"(core=({core_xl}, {core_yl}, {core_xh}, {core_yh}))"
                )
            x_values = [float(point[0]) for point in in_core_polygon]
            y_values = [float(point[1]) for point in in_core_polygon]
            local = []
            for x_index in _interval_bin_range(grid.x_edges, min(x_values), max(x_values)):
                for y_index in _interval_bin_range(grid.y_edges, min(y_values), max(y_values)):
                    bin_id = x_index * grid.num_bins_y + y_index
                    clipped = _clip_polygon_to_rect(
                        in_core_polygon,
                        grid.x_edges[x_index],
                        grid.y_edges[y_index],
                        grid.x_edges[x_index + 1],
                        grid.y_edges[y_index + 1],
                    )
                    overlap = _polygon_area(clipped)
                    if overlap > 0.0:
                        local.append((bin_id, overlap / in_core_area))
            weight_sum = sum(value for _, value in local)
            if abs(weight_sum - 1.0) > 1.0e-5:
                raise ValueError(
                    "in-core segment swept footprint is not fully covered by the capacity grid "
                    f"(weight_sum={weight_sum:.8f}, core=({core_xl}, {core_yl}, {core_xh}, {core_yh}))"
                )
            for bin_id, weight in local:
                if float(positive_capacity[bin_id].item()) <= 0.0:
                    zero_capacity_crossings.append(
                        {"segment_id": int(row["segment_id"]), "bin_id": int(bin_id)}
                    )
                segment_index.append(int(row_index))
                bin_index.append(int(bin_id))
                normalized_weight.append(float(weight))
            footprint_area_sum += polygon_area
            in_core_footprint_area_sum += in_core_area
            if in_core_area < polygon_area * (1.0 - 1.0e-6):
                core_clipped_segment_count += 1
        except (KeyError, TypeError, ValueError) as exc:
            invalid_rows.append(
                {"segment_id": int(row.get("segment_id", row_index)), "reason": str(exc)}
            )

    if invalid_rows:
        preview = "; ".join(
            f"segment {item['segment_id']}: {item['reason']}" for item in invalid_rows[:3]
        )
        raise ValueError(
            "segment capacity profile requires a valid footprint for every "
            f"SegmentCountState row; {len(invalid_rows)} invalid row(s): {preview}"
        )
    if zero_capacity_crossings:
        preview = ", ".join(
            f"segment {item['segment_id']} -> bin {item['bin_id']}"
            for item in zero_capacity_crossings[:4]
        )
        raise ValueError(
            "segment capacity profile does not allow zero-capacity crossings; "
            f"count={len(zero_capacity_crossings)} ({preview})"
        )
    if len(segment_index) == 0:
        raise ValueError("segment capacity profile produced no segment/bin overlaps")

    device = state.z_param.device
    dtype = state.z_param.dtype
    segment_tensor = torch.as_tensor(segment_index, dtype=torch.long, device=device)
    bin_tensor = torch.as_tensor(bin_index, dtype=torch.long, device=device)
    weight_tensor = torch.as_tensor(normalized_weight, dtype=dtype, device=device)
    weight_sums = torch.zeros(len(state.segment_rows), dtype=dtype, device=device).scatter_add(
        0, segment_tensor, weight_tensor
    )
    if not bool(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1.0e-5, rtol=1.0e-5)):
        raise ValueError("packed segment capacity weights do not sum to one per segment")
    if grid.capacity.device != device or grid.capacity.dtype != dtype:
        grid = SegmentCapacityGrid(
            factor=grid.factor,
            x_edges=grid.x_edges,
            y_edges=grid.y_edges,
            capacity=grid.capacity.to(device=device, dtype=dtype),
            base_num_bins_x=grid.base_num_bins_x,
            base_num_bins_y=grid.base_num_bins_y,
            base_bin_size_x=grid.base_bin_size_x,
            base_bin_size_y=grid.base_bin_size_y,
            fixed_area_source=grid.fixed_area_source,
        )
    buffer_area = float(buffer_width) * float(buffer_height)
    geometry_summary = {
        "artifact": "segment_capacity_geometry",
        "artifact_version": 1,
        "status": "ok",
        "state_segment_count": int(len(state.segment_rows)),
        "resource_segment_count": int(len(state.segment_rows)),
        "segment_bin_nnz": int(len(segment_index)),
        "row_height": float(row_height),
        "swept_footprint": "row_height_oriented_parallelogram_no_longitudinal_extension",
        "coordinate_frame": "normalized_placedb_from_raw_dbu_shift_scale",
        "raw_shift": [float(value) for value in raw_shift],
        "raw_scale": float(raw_scale),
        "swept_polygon_area_total": float(footprint_area_sum),
        "swept_footprint_in_core_area_total": float(in_core_footprint_area_sum),
        "swept_footprint_outside_core_area_total": float(
            footprint_area_sum - in_core_footprint_area_sum
        ),
        "core_clipped_segment_count": int(core_clipped_segment_count),
        "core_clipping_policy": "intersect_legal_core_then_renormalize",
        "weight_sum_min": float(weight_sums.min().detach().cpu().item()),
        "weight_sum_max": float(weight_sums.max().detach().cpu().item()),
        "zero_capacity_crossing_count": 0,
        "invalid_segment_count": 0,
    }
    return SegmentCapacityController(
        state=state,
        grid=grid,
        segment_index=segment_tensor,
        bin_index=bin_tensor,
        normalized_weight=weight_tensor,
        buffer_area=buffer_area,
        buffer_width=float(buffer_width),
        buffer_height=float(buffer_height),
        row_height=float(row_height),
        rho=float(rho),
        raw_shift=(float(raw_shift[0]), float(raw_shift[1])),
        raw_scale=float(raw_scale),
        fixed_cell_id=int(fixed_cell_id),
        fixed_master_name=str(fixed_master_name),
        dual_state=torch.zeros_like(grid.capacity),
        geometry_summary=geometry_summary,
    )


def _fixed_capacity_base_map(model):
    density_op = getattr(getattr(model, "op_collections", None), "density_op", None)
    data_collections = getattr(model, "data_collections", None)
    positions = getattr(data_collections, "pos", None)
    if density_op is None or positions is None or not positions:
        raise ValueError("segment capacity requires the model density_op and placement positions")
    pos = positions[0]
    with torch.no_grad():
        if getattr(density_op, "initial_density_map", None) is None:
            density_op.compute_initial_density_map(pos)
        initial_density = density_op.initial_density_map.detach()
    target_density = _scalar(getattr(density_op, "target_density", 1.0))
    _finite_positive(target_density, "density_op target_density")
    bin_size_x = _finite_positive(getattr(density_op, "bin_size_x", 0.0), "density_op bin_size_x")
    bin_size_y = _finite_positive(getattr(density_op, "bin_size_y", 0.0), "density_op bin_size_y")
    bin_area = bin_size_x * bin_size_y
    fixed_area = initial_density / float(target_density)
    base_capacity = (float(bin_area) - fixed_area).clamp_min(0.0)
    return {
        "base_capacity": base_capacity,
        "xl": _scalar(getattr(density_op, "xl")),
        "yl": _scalar(getattr(density_op, "yl")),
        "bin_size_x": bin_size_x,
        "bin_size_y": bin_size_y,
        "source": "PlaceObj.density_op.initial_density_map_div_target_density",
    }


def _fixed_master_geometry(data_collections, legal_table, fixed_bsu_index):
    if fixed_bsu_index is None:
        raise ValueError("segment capacity profile requires buffering_fixed_bsu_index")
    legal_cell_ids = [int(value) for value in list((legal_table or {}).get("legal_cell_ids", []) or [])]
    legal_names = [str(value) for value in list((legal_table or {}).get("legal_master_names", []) or [])]
    fixed_bsu_index = int(fixed_bsu_index)
    if not 0 <= fixed_bsu_index < len(legal_cell_ids):
        raise ValueError("fixed_bsu_index is outside the legal buffer table")
    if len(legal_names) != len(legal_cell_ids):
        raise ValueError("legal buffer table is missing master names")
    width_table = getattr(data_collections, "flat_libcell_width", None)
    height_table = getattr(data_collections, "flat_libcell_height", None)
    if width_table is None or height_table is None:
        raise ValueError("segment capacity profile requires normalized flat libcell geometry")
    cell_id = int(legal_cell_ids[fixed_bsu_index])
    if not 0 <= cell_id < int(width_table.numel()) or not 0 <= cell_id < int(height_table.numel()):
        raise ValueError("fixed buffer cell id is outside flat libcell geometry")
    return {
        "fixed_bsu_index": fixed_bsu_index,
        "cell_id": cell_id,
        "master_name": legal_names[fixed_bsu_index],
        "width": _scalar(width_table[cell_id]),
        "height": _scalar(height_table[cell_id]),
    }


def build_segment_capacity_bundle_for_model(
    state,
    *,
    model,
    placedb,
    params,
    legal_table,
    fixed_bsu_index,
    selected_grid_factor="auto",
    grid_factors=SUPPORTED_GRID_FACTORS,
):
    """Build both admissible grids once and apply the shared feasible init."""

    if state is None or not hasattr(state, "segment_rows"):
        raise ValueError("segment capacity requires a SegmentCountState")
    fixed_master = _fixed_master_geometry(
        model.data_collections,
        legal_table,
        fixed_bsu_index,
    )
    row_height = _finite_positive(getattr(placedb, "row_height", 0.0), "placedb row_height")
    raw_shift = tuple(getattr(params, "shift_factor", (0.0, 0.0)) or (0.0, 0.0))
    if len(raw_shift) != 2:
        raise ValueError("params.shift_factor must contain exactly two coordinates")
    raw_scale = _finite_positive(getattr(params, "scale_factor", 0.0), "params scale_factor")
    base = _fixed_capacity_base_map(model)
    controllers = {}
    failures = {}
    for factor in tuple(int(value) for value in grid_factors):
        try:
            grid = capacity_grid_from_base_map(
                base["base_capacity"],
                xl=base["xl"],
                yl=base["yl"],
                bin_size_x=base["bin_size_x"],
                bin_size_y=base["bin_size_y"],
                factor=factor,
                fixed_area_source=base["source"],
            )
            controllers[factor] = build_segment_capacity_controller(
                state,
                grid=grid,
                row_height=row_height,
                buffer_width=fixed_master["width"],
                buffer_height=fixed_master["height"],
                fixed_cell_id=fixed_master["cell_id"],
                fixed_master_name=fixed_master["master_name"],
                raw_shift=raw_shift,
                raw_scale=raw_scale,
            )
        except ValueError as exc:
            failures[factor] = str(exc)
    if not controllers:
        raise ValueError(
            "segment capacity profile has no admissible grid: "
            + "; ".join(f"{factor}x{factor}: {reason}" for factor, reason in failures.items())
        )
    requested = str(selected_grid_factor).strip().lower()
    if requested in ("", "auto"):
        selected = 4 if 4 in controllers else min(controllers)
    else:
        selected = int(requested)
        if selected not in controllers:
            detail = failures.get(selected, "not constructed")
            raise ValueError(f"requested segment capacity grid {selected} is not admissible: {detail}")
    initial_max = {
        factor: float(controller.initial_ratio().max().cpu().item())
        for factor, controller in controllers.items()
    }
    maximum = max(initial_max.values())
    alpha = 1.0 if maximum <= 1.0 else (1.0 - INITIALIZATION_MARGIN) / maximum
    selected_controller = controllers[selected]
    selected_controller.apply_initial_scale_(
        alpha,
        shared_grid_factors=sorted(controllers),
    )
    initialization = dict(selected_controller.initialization_summary)
    initialization.update(
        {
            "r_init_max_before_scale_by_grid": {str(key): value for key, value in initial_max.items()},
            "selected_grid_factor": int(selected),
            "admissible_grid_factors": sorted(int(key) for key in controllers),
            "grid_failures": {str(key): value for key, value in failures.items()},
        }
    )
    selected_controller.initialization_summary = initialization
    return SegmentCapacityBundle(
        controllers=controllers,
        selected_grid_factor=int(selected),
        grid_failures={str(key): value for key, value in failures.items()},
        shared_alpha=float(alpha),
        shared_initialization=initialization,
    )
