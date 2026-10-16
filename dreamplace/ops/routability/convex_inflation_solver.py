from dataclasses import dataclass
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class ActiveBinInfo:
    flat_bin_ids: torch.Tensor
    bin_x: torch.Tensor
    bin_y: torch.Tensor
    demand_values: torch.Tensor
    capacity_values: torch.Tensor
    overflow_values: torch.Tensor
    num_bins_x: int
    num_bins_y: int


@dataclass
class ClusterDemandRecords:
    cluster_ids: torch.Tensor
    bin_ids: torch.Tensor
    demand_values: torch.Tensor
    num_clusters: int
    num_bins: int


@dataclass
class ConvexInflationResult:
    cluster_inflation: torch.Tensor
    inverse_inflation: torch.Tensor
    dual_variables: torch.Tensor
    active_violation: torch.Tensor
    cluster_total_demand: torch.Tensor
    converged: bool
    fallback_used: bool
    fallback_reason: str
    num_iters: int
    max_violation: float


@dataclass
class ConvexInflationRunResult:
    active_bins: ActiveBinInfo
    demand_records: ClusterDemandRecords
    cluster_wirelength: torch.Tensor
    solve_result: ConvexInflationResult
    node_inflation: torch.Tensor


def _as_float_tensor(values, device=None, dtype=None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    return torch.as_tensor(values, device=device, dtype=dtype)


def _as_long_tensor(values, device=None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.to(dtype=torch.int64)
    return torch.as_tensor(values, device=device, dtype=torch.int64)


def _to_numpy_float(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _to_numpy_int(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64)


def _routing_grid_geometry(placedb, num_bins_x: int, num_bins_y: int) -> Tuple[float, float, float, float, float, float]:
    xl = float(getattr(placedb, "routing_grid_xl", getattr(placedb, "xl", 0.0)))
    yl = float(getattr(placedb, "routing_grid_yl", getattr(placedb, "yl", 0.0)))
    xh = float(getattr(placedb, "routing_grid_xh", getattr(placedb, "xh", 0.0)))
    yh = float(getattr(placedb, "routing_grid_yh", getattr(placedb, "yh", 0.0)))
    bin_size_x = (xh - xl) / max(int(num_bins_x), 1)
    bin_size_y = (yh - yl) / max(int(num_bins_y), 1)
    return xl, yl, xh, yh, bin_size_x, bin_size_y


def _compute_overlap_records(
    placedb,
    pos,
    node_size_x,
    node_size_y,
    selected_nodes,
    num_bins_x: int,
    num_bins_y: int,
    active_bin_lookup: Optional[Dict[int, int]] = None,
    node_weights=None,
):
    pos = _as_float_tensor(pos)
    node_size_x = _as_float_tensor(node_size_x, device=pos.device, dtype=pos.dtype)
    node_size_y = _as_float_tensor(node_size_y, device=pos.device, dtype=pos.dtype)
    selected_nodes = _as_long_tensor(selected_nodes, device=pos.device)
    if node_weights is not None:
        node_weights = _as_float_tensor(node_weights, device=pos.device, dtype=pos.dtype)

    num_nodes = int(pos.numel() // 2)
    xl, yl, xh, yh, bin_size_x, bin_size_y = _routing_grid_geometry(
        placedb,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    eps = max(min(bin_size_x, bin_size_y) * 1e-9, 1e-12)

    pos_x = _to_numpy_float(pos[:num_nodes])
    pos_y = _to_numpy_float(pos[num_nodes:])
    size_x_np = _to_numpy_float(node_size_x)
    size_y_np = _to_numpy_float(node_size_y)
    selected_nodes_np = _to_numpy_int(selected_nodes)
    node_weights_np = _to_numpy_float(node_weights) if node_weights is not None else None

    overlap_mass: Dict[Tuple[int, int], float] = {}
    for node_id in selected_nodes_np:
        if node_id < 0 or node_id >= int(size_x_np.size):
            continue
        width = float(size_x_np[node_id])
        height = float(size_y_np[node_id])
        if width <= 0 or height <= 0:
            continue

        x0 = float(pos_x[node_id])
        y0 = float(pos_y[node_id])
        x1 = x0 + width
        y1 = y0 + height

        bx_lo = max(int(math.floor((x0 - xl) / bin_size_x)), 0)
        by_lo = max(int(math.floor((y0 - yl) / bin_size_y)), 0)
        bx_hi = min(int(math.floor((max(x1 - eps, x0) - xl) / bin_size_x)), int(num_bins_x) - 1)
        by_hi = min(int(math.floor((max(y1 - eps, y0) - yl) / bin_size_y)), int(num_bins_y) - 1)
        if bx_hi < bx_lo or by_hi < by_lo:
            continue

        node_weight = float(node_weights_np[node_id]) if node_weights_np is not None else 1.0
        for bx in range(bx_lo, bx_hi + 1):
            bin_x0 = xl + bx * bin_size_x
            bin_x1 = min(bin_x0 + bin_size_x, xh)
            overlap_x = max(0.0, min(x1, bin_x1) - max(x0, bin_x0))
            if overlap_x <= 0:
                continue
            for by in range(by_lo, by_hi + 1):
                flat_bin_id = bx * int(num_bins_y) + by
                if active_bin_lookup is not None and flat_bin_id not in active_bin_lookup:
                    continue
                bin_y0 = yl + by * bin_size_y
                bin_y1 = min(bin_y0 + bin_size_y, yh)
                overlap_y = max(0.0, min(y1, bin_y1) - max(y0, bin_y0))
                if overlap_y <= 0:
                    continue
                overlap = overlap_x * overlap_y * node_weight
                if overlap <= 0:
                    continue
                overlap_mass[(int(node_id), int(flat_bin_id))] = overlap_mass.get((int(node_id), int(flat_bin_id)), 0.0) + overlap
    return overlap_mass


def collect_active_bins(demand_map, capacity_map, overflow_eps=1e-6) -> ActiveBinInfo:
    demand_map = _as_float_tensor(demand_map)
    capacity_map = _as_float_tensor(capacity_map, device=demand_map.device, dtype=demand_map.dtype)
    if demand_map.dim() != 2 or capacity_map.dim() != 2:
        raise ValueError("collect_active_bins expects 2D demand/capacity maps")
    if demand_map.shape != capacity_map.shape:
        raise ValueError(
            "demand/capacity map shape mismatch: %s vs %s"
            % (tuple(demand_map.shape), tuple(capacity_map.shape))
        )

    demand_map = torch.nan_to_num(demand_map, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    capacity_map = torch.nan_to_num(capacity_map, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    overflow_map = demand_map - capacity_map
    # Only overflowing bins can become binding constraints because inverse inflation is clamped to v <= 1.
    active_mask = overflow_map > float(overflow_eps)
    active_indices = active_mask.nonzero(as_tuple=False)
    if active_indices.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.int64, device=demand_map.device)
        empty_float = torch.empty(0, dtype=demand_map.dtype, device=demand_map.device)
        return ActiveBinInfo(
            flat_bin_ids=empty_long,
            bin_x=empty_long,
            bin_y=empty_long,
            demand_values=empty_float,
            capacity_values=empty_float,
            overflow_values=empty_float,
            num_bins_x=int(demand_map.size(0)),
            num_bins_y=int(demand_map.size(1)),
        )

    bin_x = active_indices[:, 0].to(dtype=torch.int64)
    bin_y = active_indices[:, 1].to(dtype=torch.int64)
    flat_bin_ids = bin_x * int(demand_map.size(1)) + bin_y
    info = ActiveBinInfo(
        flat_bin_ids=flat_bin_ids,
        bin_x=bin_x,
        bin_y=bin_y,
        demand_values=demand_map[active_mask].contiguous(),
        capacity_values=capacity_map[active_mask].contiguous(),
        overflow_values=overflow_map[active_mask].contiguous(),
        num_bins_x=int(demand_map.size(0)),
        num_bins_y=int(demand_map.size(1)),
    )
    logger.info(
        "Collected active bins for convex inflation: active=%d total=%d overflow_max=%.4f demand_sum=%.4f capacity_sum=%.4f",
        int(info.flat_bin_ids.numel()),
        int(demand_map.numel()),
        float(info.overflow_values.max().item()) if info.overflow_values.numel() else 0.0,
        float(info.demand_values.sum().item()) if info.demand_values.numel() else 0.0,
        float(info.capacity_values.sum().item()) if info.capacity_values.numel() else 0.0,
    )
    return info


def build_cluster_demand_records(
    placedb,
    pos,
    node_size_x,
    node_size_y,
    cluster_ids,
    active_bins: ActiveBinInfo,
    node_weights=None,
    eps=1e-12,
) -> ClusterDemandRecords:
    cluster_ids = _as_long_tensor(cluster_ids)
    device = cluster_ids.device
    num_clusters = int(cluster_ids.max().item()) + 1 if cluster_ids.numel() else 0
    num_active_bins = int(active_bins.flat_bin_ids.numel())
    empty_long = torch.empty(0, dtype=torch.int64, device=device)
    empty_float = torch.empty(0, dtype=_as_float_tensor(pos).dtype, device=device)
    if num_clusters == 0 or num_active_bins == 0:
        return ClusterDemandRecords(
            cluster_ids=empty_long,
            bin_ids=empty_long,
            demand_values=empty_float,
            num_clusters=num_clusters,
            num_bins=num_active_bins,
        )

    active_lookup = {
        int(flat_id): idx for idx, flat_id in enumerate(_to_numpy_int(active_bins.flat_bin_ids))
    }
    overlap_by_node_bin = _compute_overlap_records(
        placedb=placedb,
        pos=pos,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        selected_nodes=torch.arange(cluster_ids.numel(), device=device, dtype=torch.int64),
        num_bins_x=active_bins.num_bins_x,
        num_bins_y=active_bins.num_bins_y,
        active_bin_lookup=active_lookup,
        node_weights=node_weights,
    )

    total_mass_by_active_bin = np.zeros(num_active_bins, dtype=np.float64)
    cluster_mass: Dict[Tuple[int, int], float] = {}
    cluster_ids_np = _to_numpy_int(cluster_ids)
    for (node_id, flat_bin_id), overlap_mass in overlap_by_node_bin.items():
        active_idx = active_lookup.get(int(flat_bin_id))
        if active_idx is None:
            continue
        cluster_id = int(cluster_ids_np[node_id])
        total_mass_by_active_bin[active_idx] += float(overlap_mass)
        key = (cluster_id, active_idx)
        cluster_mass[key] = cluster_mass.get(key, 0.0) + float(overlap_mass)

    demand_np = _to_numpy_float(active_bins.demand_values)
    records = []
    covered_demand = 0.0
    for (cluster_id, active_idx), mass in sorted(cluster_mass.items()):
        total_mass = total_mass_by_active_bin[active_idx]
        if total_mass <= float(eps):
            continue
        demand_value = float(demand_np[active_idx]) * float(mass) / float(total_mass)
        if demand_value <= float(eps):
            continue
        covered_demand += demand_value
        records.append((cluster_id, active_idx, demand_value))

    uncovered_bins = int(np.sum((total_mass_by_active_bin <= float(eps)) & (demand_np > float(eps))))
    uncovered_demand = float(demand_np[(total_mass_by_active_bin <= float(eps))].sum()) if uncovered_bins > 0 else 0.0
    if records:
        record_cluster_ids = torch.tensor([item[0] for item in records], dtype=torch.int64, device=device)
        record_bin_ids = torch.tensor([item[1] for item in records], dtype=torch.int64, device=device)
        record_demands = torch.tensor([item[2] for item in records], dtype=_as_float_tensor(pos).dtype, device=device)
    else:
        record_cluster_ids = empty_long
        record_bin_ids = empty_long
        record_demands = empty_float

    logger.info(
        "Built sparse modularity demand records: records=%d clusters=%d active_bins=%d covered_demand=%.4f total_demand=%.4f uncovered_bins=%d uncovered_demand=%.4f",
        len(records),
        num_clusters,
        num_active_bins,
        covered_demand,
        float(active_bins.demand_values.sum().item()) if active_bins.demand_values.numel() else 0.0,
        uncovered_bins,
        uncovered_demand,
    )
    return ClusterDemandRecords(
        cluster_ids=record_cluster_ids,
        bin_ids=record_bin_ids,
        demand_values=record_demands,
        num_clusters=num_clusters,
        num_bins=num_active_bins,
    )


def compute_cluster_wirelength(
    placedb,
    pos,
    node_size_x,
    node_size_y,
    cluster_ids,
    eps=1e-12,
) -> torch.Tensor:
    cluster_ids = _as_long_tensor(cluster_ids)
    num_clusters = int(cluster_ids.max().item()) + 1 if cluster_ids.numel() else 0
    pos = _as_float_tensor(pos)
    dtype = pos.dtype
    device = pos.device
    if num_clusters == 0:
        return torch.empty(0, dtype=dtype, device=device)

    num_nodes = int(pos.numel() // 2)
    num_movable_nodes = int(cluster_ids.numel())
    center_x = _to_numpy_float(pos[:num_movable_nodes]) + 0.5 * _to_numpy_float(node_size_x[:num_movable_nodes])
    center_y = _to_numpy_float(pos[num_nodes:num_nodes + num_movable_nodes]) + 0.5 * _to_numpy_float(node_size_y[:num_movable_nodes])
    cluster_ids_np = _to_numpy_int(cluster_ids)
    flat_net2pin_map = _to_numpy_int(placedb.flat_net2pin_map)
    flat_net2pin_start_map = _to_numpy_int(placedb.flat_net2pin_start_map)
    pin2node_map = _to_numpy_int(placedb.pin2node_map)
    net_weights = _to_numpy_float(getattr(placedb, "net_weights", np.ones(int(placedb.num_nets), dtype=np.float64)))

    cluster_wl = np.zeros(num_clusters, dtype=np.float64)
    for net_id in range(int(placedb.num_nets)):
        start = int(flat_net2pin_start_map[net_id])
        end = int(flat_net2pin_start_map[net_id + 1])
        if end - start < 2:
            continue
        nodes = pin2node_map[flat_net2pin_map[start:end]]
        movable_nodes = np.unique(nodes[nodes < num_movable_nodes])
        if movable_nodes.size < 2:
            continue
        clusters = cluster_ids_np[movable_nodes]
        for cluster_id in np.unique(clusters):
            members = movable_nodes[clusters == cluster_id]
            if members.size < 2:
                continue
            hpwl = (
                float(center_x[members].max() - center_x[members].min()) +
                float(center_y[members].max() - center_y[members].min())
            )
            if hpwl > float(eps):
                cluster_wl[int(cluster_id)] += hpwl * float(net_weights[net_id])

    output = torch.as_tensor(cluster_wl, dtype=dtype, device=device)
    logger.info(
        "Computed cluster wirelength proxy: clusters=%d wl_sum=%.4f wl_max=%.4f",
        num_clusters,
        float(output.sum().item()) if output.numel() else 0.0,
        float(output.max().item()) if output.numel() else 0.0,
    )
    return output


def solve_convex_inflation_sparse(
    cluster_demand_records: ClusterDemandRecords,
    active_bin_capacity,
    cluster_wirelength,
    max_inflation=4.0,
    min_inflation=1.0,
    max_iters=50,
    rho_init=1.0,
    tolerance=1e-4,
    eps=1e-12,
    stagnation_iters=8,
) -> ConvexInflationResult:
    cluster_wirelength = _as_float_tensor(cluster_wirelength).flatten()
    device = cluster_wirelength.device
    dtype = cluster_wirelength.dtype
    active_bin_capacity = _as_float_tensor(active_bin_capacity, device=device, dtype=dtype).flatten()

    num_clusters = max(int(cluster_demand_records.num_clusters), int(cluster_wirelength.numel()))
    num_bins = int(cluster_demand_records.num_bins)
    if active_bin_capacity.numel() != num_bins:
        raise ValueError(
            "active_bin_capacity length mismatch: expected %d, got %d"
            % (num_bins, int(active_bin_capacity.numel()))
        )
    if cluster_wirelength.numel() < num_clusters:
        padded = torch.zeros(num_clusters, dtype=dtype, device=device)
        padded[: cluster_wirelength.numel()] = cluster_wirelength
        cluster_wirelength = padded
    elif cluster_wirelength.numel() > num_clusters:
        cluster_wirelength = cluster_wirelength[:num_clusters]

    cluster_ids = _as_long_tensor(cluster_demand_records.cluster_ids, device=device)
    bin_ids = _as_long_tensor(cluster_demand_records.bin_ids, device=device)
    demand_values = _as_float_tensor(cluster_demand_records.demand_values, device=device, dtype=dtype)
    cluster_total_demand = torch.zeros(num_clusters, dtype=dtype, device=device)
    if demand_values.numel():
        cluster_total_demand.scatter_add_(0, cluster_ids, demand_values)

    ones = torch.ones(num_clusters, dtype=dtype, device=device)
    empty_dual = torch.zeros(num_bins, dtype=dtype, device=device)
    empty_violation = torch.zeros(num_bins, dtype=dtype, device=device)
    if num_clusters == 0:
        return ConvexInflationResult(
            cluster_inflation=ones,
            inverse_inflation=ones,
            dual_variables=empty_dual,
            active_violation=empty_violation,
            cluster_total_demand=cluster_total_demand,
            converged=True,
            fallback_used=False,
            fallback_reason="",
            num_iters=0,
            max_violation=0.0,
        )
    if num_bins == 0 or demand_values.numel() == 0:
        return ConvexInflationResult(
            cluster_inflation=ones,
            inverse_inflation=ones,
            dual_variables=empty_dual,
            active_violation=empty_violation,
            cluster_total_demand=cluster_total_demand,
            converged=True,
            fallback_used=False,
            fallback_reason="",
            num_iters=0,
            max_violation=0.0,
        )

    max_inflation = max(float(max_inflation), 1.0)
    min_inflation = max(float(min_inflation), 1.0)
    v_min = 1.0 / max_inflation
    v_max = 1.0 / min_inflation
    rho = float(rho_init)

    initial_predicted_demand = torch.zeros(num_bins, dtype=dtype, device=device)
    initial_predicted_demand.scatter_add_(0, bin_ids, demand_values)
    initial_violation = initial_predicted_demand - active_bin_capacity
    initial_max_violation = (
        float(torch.clamp(initial_violation.max(), min=0).item()) if initial_violation.numel() else 0.0
    )

    dual = torch.clamp(initial_violation, min=0)
    best_v = ones.clone()
    best_dual = dual.clone()
    best_violation = initial_violation.clone()
    best_max_violation = initial_max_violation
    best_iter = 0
    converged = False
    fallback_reason = ""
    stale_count = 0

    positive_cluster_wl = cluster_wirelength[cluster_wirelength > float(eps)]
    wl_scale = (
        positive_cluster_wl.mean().clamp_min(float(eps))
        if positive_cluster_wl.numel()
        else torch.tensor(1.0, dtype=dtype, device=device)
    )
    safe_cluster_wl = (cluster_wirelength / wl_scale).clamp_min(float(eps))
    inactive_cluster_mask = (cluster_wirelength <= float(eps)) & (cluster_total_demand <= float(eps))
    used_best_effort = False

    for iter_idx in range(int(max_iters)):
        eta = torch.zeros(num_clusters, dtype=dtype, device=device)
        eta.scatter_add_(0, cluster_ids, dual[bin_ids] * demand_values)
        safe_eta = eta.clamp_min(float(eps))
        inverse_inflation = torch.sqrt(safe_cluster_wl / safe_eta).clamp(min=v_min, max=v_max)
        inverse_inflation[inactive_cluster_mask] = v_max

        predicted_demand = torch.zeros(num_bins, dtype=dtype, device=device)
        predicted_demand.scatter_add_(0, bin_ids, demand_values * inverse_inflation[cluster_ids])
        violation = predicted_demand - active_bin_capacity

        if not torch.isfinite(inverse_inflation).all() or not torch.isfinite(violation).all():
            fallback_reason = "nan_inf"
            break

        max_violation = float(torch.clamp(violation.max(), min=0).item()) if violation.numel() else 0.0
        if max_violation + float(tolerance) < best_max_violation:
            best_max_violation = max_violation
            best_v = inverse_inflation.clone()
            best_dual = dual.clone()
            best_violation = violation.clone()
            best_iter = iter_idx + 1
            stale_count = 0
        else:
            stale_count += 1

        if max_violation <= float(tolerance):
            converged = True
            break

        dual = torch.clamp(dual + rho * violation, min=0)
        if stale_count > 0 and stale_count % 4 == 0:
            rho = min(rho * 2.0, 128.0)
        if stale_count >= int(stagnation_iters):
            fallback_reason = "stagnation"
            break

    if not converged and not fallback_reason:
        fallback_reason = "max_iters"

    if converged:
        final_inverse = best_v
        final_dual = best_dual
        final_violation = best_violation
        fallback_used = False
    elif best_iter > 0 and best_max_violation + float(tolerance) < initial_max_violation:
        final_inverse = best_v
        final_dual = best_dual
        final_violation = best_violation
        fallback_used = False
        used_best_effort = True
    else:
        final_inverse = ones
        final_dual = empty_dual
        final_violation = empty_violation
        fallback_used = True

    cluster_inflation = torch.clamp(1.0 / final_inverse.clamp_min(float(eps)), min=min_inflation, max=max_inflation)
    cluster_inflation[inactive_cluster_mask] = 1.0
    final_inverse = 1.0 / cluster_inflation
    logger.info(
        "Sparse convex inflation solver: clusters=%d active_bins=%d records=%d converged=%s fallback=%s best_effort=%s reason=%s iter=%d initial_max_violation=%.6f max_violation=%.6f inflation_max=%.4f",
        num_clusters,
        num_bins,
        int(demand_values.numel()),
        str(converged),
        str(fallback_used),
        str(used_best_effort),
        fallback_reason,
        best_iter if best_iter > 0 else 0,
        initial_max_violation,
        best_max_violation if math.isfinite(best_max_violation) else 0.0,
        float(cluster_inflation.max().item()) if cluster_inflation.numel() else 1.0,
    )
    return ConvexInflationResult(
        cluster_inflation=cluster_inflation,
        inverse_inflation=final_inverse,
        dual_variables=final_dual,
        active_violation=final_violation,
        cluster_total_demand=cluster_total_demand,
        converged=converged,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason if fallback_used else "",
        num_iters=best_iter if best_iter > 0 else 0,
        max_violation=best_max_violation if math.isfinite(best_max_violation) else 0.0,
    )


def broadcast_cluster_inflation_to_cells(cluster_ids, cluster_inflation, num_movable_nodes=None) -> torch.Tensor:
    cluster_ids = _as_long_tensor(cluster_ids)
    cluster_inflation = _as_float_tensor(cluster_inflation, device=cluster_ids.device)
    if num_movable_nodes is None:
        num_movable_nodes = int(cluster_ids.numel())
    if cluster_ids.numel() != int(num_movable_nodes):
        raise ValueError(
            "cluster_ids length mismatch: expected %d, got %d"
            % (int(num_movable_nodes), int(cluster_ids.numel()))
        )
    if cluster_inflation.numel() == 0:
        return torch.ones(int(num_movable_nodes), dtype=torch.float32, device=cluster_ids.device)
    return cluster_inflation[cluster_ids].contiguous()


def active_values_to_full_map(active_bins: ActiveBinInfo, active_values, fill_value=0.0) -> torch.Tensor:
    active_values = _as_float_tensor(active_values)
    if int(active_values.numel()) != int(active_bins.flat_bin_ids.numel()):
        raise ValueError(
            "active_values length mismatch: expected %d, got %d"
            % (int(active_bins.flat_bin_ids.numel()), int(active_values.numel()))
        )
    output = torch.full(
        (int(active_bins.num_bins_x), int(active_bins.num_bins_y)),
        float(fill_value),
        dtype=active_values.dtype,
        device=active_values.device,
    )
    if active_values.numel():
        output[active_bins.bin_x.to(device=active_values.device), active_bins.bin_y.to(device=active_values.device)] = active_values
    return output


def aggregate_active_bin_inflation(
    cluster_demand_records: ClusterDemandRecords,
    cluster_inflation,
    eps=1e-12,
) -> torch.Tensor:
    cluster_inflation = _as_float_tensor(cluster_inflation).flatten()
    num_bins = int(cluster_demand_records.num_bins)
    if num_bins <= 0:
        return torch.empty(0, dtype=cluster_inflation.dtype, device=cluster_inflation.device)

    bin_ids = _as_long_tensor(cluster_demand_records.bin_ids, device=cluster_inflation.device)
    cluster_ids = _as_long_tensor(cluster_demand_records.cluster_ids, device=cluster_inflation.device)
    demand_values = _as_float_tensor(
        cluster_demand_records.demand_values,
        device=cluster_inflation.device,
        dtype=cluster_inflation.dtype,
    )
    weighted = torch.zeros(num_bins, dtype=cluster_inflation.dtype, device=cluster_inflation.device)
    total = torch.zeros(num_bins, dtype=cluster_inflation.dtype, device=cluster_inflation.device)
    if demand_values.numel():
        weighted.scatter_add_(0, bin_ids, demand_values * cluster_inflation[cluster_ids])
        total.scatter_add_(0, bin_ids, demand_values)
    output = torch.ones(num_bins, dtype=cluster_inflation.dtype, device=cluster_inflation.device)
    valid = total > float(eps)
    output[valid] = weighted[valid] / total[valid]
    return output


def local_gcell_correction(
    prev_inflation_map,
    local_demand_map,
    global_demand_map,
    capacity_map,
    gamma=0.2,
    max_inflation=4.0,
    eps=1e-12,
) -> torch.Tensor:
    prev_inflation_map = _as_float_tensor(prev_inflation_map)
    local_demand_map = _as_float_tensor(local_demand_map, device=prev_inflation_map.device, dtype=prev_inflation_map.dtype)
    global_demand_map = _as_float_tensor(global_demand_map, device=prev_inflation_map.device, dtype=prev_inflation_map.dtype)
    capacity_map = _as_float_tensor(capacity_map, device=prev_inflation_map.device, dtype=prev_inflation_map.dtype)
    if (
        prev_inflation_map.shape != local_demand_map.shape
        or prev_inflation_map.shape != global_demand_map.shape
        or prev_inflation_map.shape != capacity_map.shape
    ):
        raise ValueError("local_gcell_correction expects same-shape 2D maps")

    safe_remain = torch.clamp(capacity_map - global_demand_map, min=float(eps))
    local_target = torch.maximum(
        torch.ones_like(prev_inflation_map),
        local_demand_map / safe_remain,
    ).clamp(max=float(max_inflation))
    corrected = (1.0 - float(gamma)) * prev_inflation_map + float(gamma) * local_target
    return corrected.clamp(min=1.0, max=float(max_inflation))


def iterative_local_gcell_correction(
    prev_inflation_map,
    local_demand_map,
    global_demand_map,
    capacity_map,
    gamma=0.2,
    max_inflation=4.0,
    num_iters=3,
    eps=1e-12,
) -> torch.Tensor:
    corrected = _as_float_tensor(prev_inflation_map)
    for _ in range(max(int(num_iters), 0)):
        corrected = local_gcell_correction(
            corrected,
            local_demand_map=local_demand_map,
            global_demand_map=global_demand_map,
            capacity_map=capacity_map,
            gamma=gamma,
            max_inflation=max_inflation,
            eps=eps,
        )
    return corrected


def sample_node_inflation_from_bin_map(
    placedb,
    pos,
    node_size_x,
    node_size_y,
    bin_inflation_map,
    reduction="max",
    eps=1e-12,
) -> torch.Tensor:
    bin_inflation_map = _as_float_tensor(bin_inflation_map)
    if bin_inflation_map.dim() != 2:
        raise ValueError("sample_node_inflation_from_bin_map expects a 2D inflation map")

    num_bins_x, num_bins_y = int(bin_inflation_map.size(0)), int(bin_inflation_map.size(1))
    overlaps = _compute_overlap_records(
        placedb=placedb,
        pos=pos,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        selected_nodes=torch.arange(int(placedb.num_movable_nodes), dtype=torch.int64, device=_as_float_tensor(pos).device),
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        active_bin_lookup=None,
        node_weights=None,
    )
    num_movable_nodes = int(placedb.num_movable_nodes)
    output = torch.ones(num_movable_nodes, dtype=bin_inflation_map.dtype, device=bin_inflation_map.device)
    bin_map_np = _to_numpy_float(bin_inflation_map)
    accum = np.ones(num_movable_nodes, dtype=np.float64) if reduction == "max" else np.zeros(num_movable_nodes, dtype=np.float64)
    weight_sum = np.zeros(num_movable_nodes, dtype=np.float64)
    for (node_id, flat_bin_id), overlap in overlaps.items():
        bx = int(flat_bin_id) // num_bins_y
        by = int(flat_bin_id) % num_bins_y
        value = float(bin_map_np[bx, by])
        if reduction == "max":
            accum[node_id] = max(accum[node_id], value)
        elif reduction == "area_weighted_mean":
            accum[node_id] += value * float(overlap)
            weight_sum[node_id] += float(overlap)
        else:
            raise ValueError("Unsupported reduction %s" % reduction)
    if reduction == "area_weighted_mean":
        valid = weight_sum > float(eps)
        accum[valid] = accum[valid] / weight_sum[valid]
        accum[~valid] = 1.0
    output.copy_(torch.as_tensor(accum, dtype=output.dtype, device=output.device))
    return output.clamp(min=1.0)


def run_convex_inflation(
    placedb,
    params,
    pos,
    node_size_x,
    node_size_y,
    cluster_ids,
    demand_map,
    capacity_map,
    node_weights=None,
) -> ConvexInflationRunResult:
    active_bins = collect_active_bins(
        demand_map,
        capacity_map,
        overflow_eps=getattr(params, "modularity_active_bin_overflow_eps", 1e-6),
    )
    demand_records = build_cluster_demand_records(
        placedb=placedb,
        pos=pos,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        cluster_ids=cluster_ids,
        active_bins=active_bins,
        node_weights=node_weights,
    )
    cluster_wirelength = compute_cluster_wirelength(
        placedb=placedb,
        pos=pos,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        cluster_ids=cluster_ids,
    )
    solve_result = solve_convex_inflation_sparse(
        demand_records,
        active_bin_capacity=active_bins.capacity_values,
        cluster_wirelength=cluster_wirelength,
        max_inflation=getattr(params, "modularity_max_inflation", 4.0),
        max_iters=getattr(params, "modularity_convex_max_iters", 50),
        rho_init=getattr(params, "modularity_convex_rho_init", 1.0),
    )
    node_inflation = broadcast_cluster_inflation_to_cells(
        cluster_ids,
        solve_result.cluster_inflation,
    )
    return ConvexInflationRunResult(
        active_bins=active_bins,
        demand_records=demand_records,
        cluster_wirelength=cluster_wirelength,
        solve_result=solve_result,
        node_inflation=node_inflation,
    )


__all__ = [
    "ActiveBinInfo",
    "ClusterDemandRecords",
    "ConvexInflationResult",
    "ConvexInflationRunResult",
    "aggregate_active_bin_inflation",
    "active_values_to_full_map",
    "broadcast_cluster_inflation_to_cells",
    "build_cluster_demand_records",
    "collect_active_bins",
    "compute_cluster_wirelength",
    "iterative_local_gcell_correction",
    "local_gcell_correction",
    "run_convex_inflation",
    "sample_node_inflation_from_bin_map",
    "solve_convex_inflation_sparse",
]
