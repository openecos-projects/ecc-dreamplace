import math
import os
import torch
from torch import nn
import torch.nn.functional as F
import logging
import pdb
import numpy as np

import dreamplace.ops.adjust_node_area.adjust_node_area_cpp as adjust_node_area_cpp
import dreamplace.ops.adjust_node_area.update_pin_offset_cpp as update_pin_offset_cpp
import dreamplace.ops.routability.convex_inflation_solver as convex_inflation_solver
try:
    import dreamplace.ops.adjust_node_area.adjust_node_area_cuda as adjust_node_area_cuda
    import dreamplace.ops.adjust_node_area.update_pin_offset_cuda as update_pin_offset_cuda
except:
    pass

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

class ComputeNodeAreaFromRouteMap(nn.Module):
    def __init__(self, xl, yl, xh, yh, num_movable_nodes, num_bins_x,
                 num_bins_y):
        super(ComputeNodeAreaFromRouteMap, self).__init__()
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_movable_nodes = num_movable_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y

    def forward(self, pos, node_size_x, node_size_y, utilization_map):
        if pos.is_cuda:
            func = adjust_node_area_cuda.forward
        else:
            func = adjust_node_area_cpp.forward
        output = func(pos, node_size_x, node_size_y, utilization_map,
                      self.bin_size_x, self.bin_size_y, self.xl, self.yl,
                      self.xh, self.yh, self.num_movable_nodes,
                      self.num_bins_x, self.num_bins_y)
        return output


class ComputeNodeAreaFromPinMap(ComputeNodeAreaFromRouteMap):
    def __init__(self, pin_weights, flat_node2pin_start_map, xl, yl, xh, yh,
                 num_movable_nodes, num_bins_x, num_bins_y, unit_pin_capacity):
        super(ComputeNodeAreaFromPinMap,
              self).__init__(xl, yl, xh, yh, num_movable_nodes, num_bins_x,
                             num_bins_y)
        bin_area = (xh - xl) / num_bins_x * (yh - yl) / num_bins_y
        self.unit_pin_capacity = unit_pin_capacity
        # for each physical node, we use the pin counts as the weights
        if pin_weights is not None:
            self.pin_weights = pin_weights
        elif flat_node2pin_start_map is not None:
            self.pin_weights = flat_node2pin_start_map[
                1:self.num_movable_nodes +
                1] - flat_node2pin_start_map[:self.num_movable_nodes]
        else:
            assert "either pin_weights or flat_node2pin_start_map is required"

    def forward(self, pos, node_size_x, node_size_y, utilization_map):
        output = super(ComputeNodeAreaFromPinMap,
                       self).forward(pos, node_size_x, node_size_y,
                                     utilization_map)
        #output.mul_(self.pin_weights[:self.num_movable_nodes].to(node_size_x.dtype) / (node_size_x[:self.num_movable_nodes] * node_size_y[:self.num_movable_nodes] * self.unit_pin_capacity))
        return output


class AdjustNodeArea(nn.Module):
    def __init__(
        self,
        flat_node2pin_map,
        flat_node2pin_start_map,
        pin_weights,  # only one of them needed
        xl,
        yl,
        xh,
        yh,
        num_movable_nodes,
        num_filler_nodes,
        route_num_bins_x,
        route_num_bins_y,
        pin_num_bins_x,
        pin_num_bins_y,
        total_place_area,  # total placement area excluding fixed cells
        total_whitespace_area,  # total white space area excluding movable and fixed cells
        max_route_opt_adjust_rate,
        route_opt_adjust_exponent=2.5,
        max_pin_opt_adjust_rate=2.5,
        inflation_area_budget_ratio=0.1,
        area_adjust_stop_ratio=0.01,
        route_area_adjust_stop_ratio=0.01,
        pin_area_adjust_stop_ratio=0.05,
        unit_pin_capacity=0.0,
        modularity_config=None,
        params=None):
        super(AdjustNodeArea, self).__init__()
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.flat_node2pin_map = flat_node2pin_map
        self.pin_weights = pin_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh

        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes

        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate
        # exponent for adjusting the utilization map
        self.route_opt_adjust_exponent = route_opt_adjust_exponent
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_pin_opt_adjust_rate = max_pin_opt_adjust_rate
        self.min_pin_opt_adjust_rate = 1.0 / max_pin_opt_adjust_rate
        self.inflation_area_budget_ratio = float(inflation_area_budget_ratio)
        if (
            not math.isfinite(self.inflation_area_budget_ratio)
            or self.inflation_area_budget_ratio < 0
        ):
            raise ValueError(
                "inflation_area_budget_ratio must be finite and non-negative"
            )

        # stop ratio
        self.area_adjust_stop_ratio = area_adjust_stop_ratio
        self.route_area_adjust_stop_ratio = route_area_adjust_stop_ratio
        self.pin_area_adjust_stop_ratio = pin_area_adjust_stop_ratio

        self.compute_node_area_route = ComputeNodeAreaFromRouteMap(
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            num_movable_nodes=self.num_movable_nodes,
            num_bins_x=route_num_bins_x,
            num_bins_y=route_num_bins_y)
        self.compute_node_area_pin = ComputeNodeAreaFromPinMap(
            pin_weights=self.pin_weights,
            flat_node2pin_start_map=self.flat_node2pin_start_map,
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            num_movable_nodes=self.num_movable_nodes,
            num_bins_x=pin_num_bins_x,
            num_bins_y=pin_num_bins_y,
            unit_pin_capacity=unit_pin_capacity)

        # placement area excluding fixed cells
        self.total_place_area = total_place_area
        # placement area excluding movable and fixed cells
        self.total_whitespace_area = total_whitespace_area
        self.modularity_config = modularity_config or {}
        self.modularity_enabled = bool(self.modularity_config.get("enabled", False))
        self.params = params or self.modularity_config.get("params")
        self.last_modularity_summary = None

    def _maybe_plot_inflation_cells(
        self,
        pos,
        old_movable_area,
        old_node_size_x_movable,
        old_node_size_y_movable,
        actual_area_increment,
        inflation_round,
        color_limits=None,
    ):
        params = self.params
        if params is None or not getattr(params, "modularity_plot_flag", False):
            return
        if actual_area_increment is None or actual_area_increment.numel() == 0:
            return

        result_dir = getattr(params, "result_dir", None)
        if not result_dir:
            return

        try:
            design_name = params.design_name()
        except Exception:
            design_name = "design"

        inflation_mode = "modularity" if self.modularity_enabled else "standard"
        level_idx = -1
        if self.modularity_enabled and isinstance(self.last_modularity_summary, dict):
            level_idx = int(self.last_modularity_summary.get("level_idx", -1))

        output_dir = os.path.join(
            result_dir,
            design_name,
            "plot",
            "%s_inflation" % inflation_mode,
        )
        os.makedirs(output_dir, exist_ok=True)
        if self.modularity_enabled:
            output_name = "modularity_cell_inflation_round%d_level%d.png" % (
                int(inflation_round),
                level_idx,
            )
        else:
            output_name = "standard_cell_inflation_round%d.png" % int(inflation_round)
        output_path = os.path.join(output_dir, output_name)
        npz_path = output_path[:-4] + ".npz"

        with torch.no_grad():
            num_nodes = int(pos.numel() / 2)
            center_x = (
                pos[: self.num_movable_nodes] + old_node_size_x_movable * 0.5
            ).detach().cpu().numpy()
            center_y = (
                pos[num_nodes : num_nodes + self.num_movable_nodes]
                + old_node_size_y_movable * 0.5
            ).detach().cpu().numpy()
            area_increment_np = actual_area_increment.detach().cpu().numpy()
            inflated_mask = area_increment_np > 0
            old_movable_area_np = old_movable_area.detach().cpu().numpy()
            inflation_ratio_np = np.ones_like(area_increment_np)
            positive_area_mask = old_movable_area_np > 0
            inflation_ratio_np[positive_area_mask] = (
                1.0 + area_increment_np[positive_area_mask] / old_movable_area_np[positive_area_mask]
            )

        fig, ax = plt.subplots(figsize=(10, 10))
        if (~inflated_mask).any():
            ax.scatter(
                center_x[~inflated_mask],
                center_y[~inflated_mask],
                c="#bdbdbd",
                s=1.0,
                marker="s",
                linewidths=0,
                alpha=0.35,
                rasterized=True,
                label="No inflation",
            )
        if inflated_mask.any():
            inflated_ratios = inflation_ratio_np[inflated_mask]
            if color_limits is None:
                ratio_min = float(inflated_ratios.min())
                ratio_max = float(inflated_ratios.max())
            else:
                ratio_min, ratio_max = color_limits
            if math.isclose(ratio_min, ratio_max):
                ratio_max = ratio_min + 1e-6
            norm = plt.Normalize(
                vmin=ratio_min,
                vmax=ratio_max,
            )
            scatter = ax.scatter(
                center_x[inflated_mask],
                center_y[inflated_mask],
                c=inflated_ratios,
                s=2.0,
                marker="s",
                linewidths=0,
                alpha=0.9,
                cmap="turbo",
                norm=norm,
                rasterized=True,
                label="Inflated (colored by ratio)",
            )
            cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
            cbar.set_label("Inflation Ratio")
        ax.set_xlim(float(self.xl), float(self.xh))
        ax.set_ylim(float(self.yl), float(self.yh))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        inflated_ratio_stats = inflation_ratio_np[inflated_mask]
        if self.modularity_enabled:
            title = (
                "Modularity Inflation Cells R%d L%d | inflated=%d/%d (%.2f%%) | ratio %.4f/%.4f/%.4f"
                % (
                    int(inflation_round),
                    level_idx,
                    int(inflated_mask.sum()),
                    int(inflated_mask.size),
                    100.0 * float(inflated_mask.mean()) if inflated_mask.size else 0.0,
                    float(inflated_ratio_stats.min()) if inflated_ratio_stats.size else 1.0,
                    float(inflated_ratio_stats.mean()) if inflated_ratio_stats.size else 1.0,
                    float(inflated_ratio_stats.max()) if inflated_ratio_stats.size else 1.0,
                )
            )
        else:
            title = (
                "Standard Inflation Cells R%d | inflated=%d/%d (%.2f%%) | ratio %.4f/%.4f/%.4f"
                % (
                    int(inflation_round),
                    int(inflated_mask.sum()),
                    int(inflated_mask.size),
                    100.0 * float(inflated_mask.mean()) if inflated_mask.size else 0.0,
                    float(inflated_ratio_stats.min()) if inflated_ratio_stats.size else 1.0,
                    float(inflated_ratio_stats.mean()) if inflated_ratio_stats.size else 1.0,
                    float(inflated_ratio_stats.max()) if inflated_ratio_stats.size else 1.0,
                )
            )
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=False, markerscale=4)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            npz_path,
            center_x=center_x,
            center_y=center_y,
            inflated_mask=inflated_mask.astype(np.uint8),
            area_increment=area_increment_np,
            inflation_ratio=inflation_ratio_np,
            inflation_round=np.asarray([int(inflation_round)], dtype=np.int64),
            level_idx=np.asarray([level_idx], dtype=np.int64),
            inflation_mode=np.asarray([inflation_mode]),
        )
        logger.info(
            "Saved %s cell inflation plot: round=%d level=%d inflated=%d/%d ratio[min/mean/max]=%.4f/%.4f/%.4f path=%s",
            inflation_mode,
            int(inflation_round),
            level_idx,
            int(inflated_mask.sum()),
            int(inflated_mask.size),
            float(inflated_ratio_stats.min()) if inflated_ratio_stats.size else 1.0,
            float(inflated_ratio_stats.mean()) if inflated_ratio_stats.size else 1.0,
            float(inflated_ratio_stats.max()) if inflated_ratio_stats.size else 1.0,
            output_path,
        )

    def _maybe_plot_route_inflation_maps(
        self,
        route_power_map,
        route_effective_map,
        scale_factor,
        old_movable_area,
        actual_area_increment,
        inflation_round,
    ):
        params = self.params
        if params is None or not getattr(params, "plot_flag", False):
            return None
        if route_power_map is None or route_effective_map is None:
            return None

        try:
            design_name = params.design_name()
        except Exception:
            design_name = "design"
        output_dir = os.path.join(
            getattr(params, "result_dir", "."),
            design_name,
            "plot",
        )
        os.makedirs(output_dir, exist_ok=True)

        applied_scale = min(max(float(scale_factor), 0.0), 1.0)
        with torch.no_grad():
            raw_np = route_power_map.detach().cpu().numpy()
            effective_np = route_effective_map.detach().cpu().numpy()
            applied_np = 1.0 + applied_scale * np.maximum(effective_np - 1.0, 0.0)

            old_area_np = old_movable_area.detach().cpu().numpy()
            increment_np = actual_area_increment.detach().cpu().numpy()
            cell_ratio_np = np.ones_like(increment_np)
            positive_area_mask = old_area_np > 0
            cell_ratio_np[positive_area_mask] += (
                increment_np[positive_area_mask] / old_area_np[positive_area_mask]
            )

        finite_raw = raw_np[np.isfinite(raw_np)]
        raw_vmin = float(finite_raw.min()) if finite_raw.size else 0.0
        raw_vmax = (
            float(np.percentile(finite_raw, 99.5)) if finite_raw.size else raw_vmin
        )
        if not math.isfinite(raw_vmax) or raw_vmax <= raw_vmin:
            raw_vmax = raw_vmin + 1e-6

        shared_vmin = 1.0
        shared_vmax = max(
            float(np.nanmax(applied_np)) if applied_np.size else shared_vmin,
            float(np.nanmax(cell_ratio_np)) if cell_ratio_np.size else shared_vmin,
            shared_vmin + 1e-6,
        )
        extent = (float(self.xl), float(self.xh), float(self.yl), float(self.yh))

        def save_map(data, filename, title, colorbar_label, vmin, vmax, cmap):
            fig, ax = plt.subplots(figsize=(10, 10))
            image = ax.imshow(
                data.T,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(title)
            colorbar = fig.colorbar(
                image,
                ax=ax,
                orientation="vertical",
                fraction=0.035,
                pad=0.02,
            )
            colorbar.set_label(colorbar_label)
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            return output_path

        round_idx = int(inflation_round)
        raw_path = save_map(
            raw_np,
            "route%d_raw.png" % round_idx,
            "Raw Powered Route Signal R%d | display <= P99.5 %.4f | max %.4f"
            % (round_idx, raw_vmax, float(np.nanmax(raw_np))),
            "Powered Route Signal",
            raw_vmin,
            raw_vmax,
            "magma",
        )
        effective_path = save_map(
            effective_np,
            "route%d_effective.png" % round_idx,
            "Effective Route Inflation Map R%d | visible range 1.0000 to %.4f"
            % (round_idx, self.max_route_opt_adjust_rate),
            "Pre-budget Inflation Factor",
            1.0,
            self.max_route_opt_adjust_rate,
            "turbo",
        )
        applied_path = save_map(
            applied_np,
            "route%d.png" % round_idx,
            "Applied Route Inflation Map R%d | area scale %.6f"
            % (round_idx, applied_scale),
            "Applied Inflation Ratio",
            shared_vmin,
            shared_vmax,
            "turbo",
        )
        npz_path = os.path.join(output_dir, "route%d_maps.npz" % round_idx)
        np.savez_compressed(
            npz_path,
            raw_power_map=raw_np,
            effective_map=effective_np,
            applied_map=applied_np,
            applied_scale=np.asarray([applied_scale], dtype=np.float64),
            shared_color_limits=np.asarray(
                [shared_vmin, shared_vmax], dtype=np.float64
            ),
        )
        logger.info(
            "Saved route inflation maps: round=%d raw=%s effective=%s applied=%s "
            "npz=%s raw[min/p99.5/max]=%.4f/%.4f/%.4f "
            "effective[min/max]=%.4f/%.4f applied[min/max]=%.4f/%.4f scale=%.6f",
            round_idx,
            raw_path,
            effective_path,
            applied_path,
            npz_path,
            raw_vmin,
            raw_vmax,
            float(np.nanmax(raw_np)),
            float(np.nanmin(effective_np)),
            float(np.nanmax(effective_np)),
            float(np.nanmin(applied_np)),
            float(np.nanmax(applied_np)),
            applied_scale,
        )
        return shared_vmin, shared_vmax

    def _select_modularity_cluster_ids(self, inflation_round):
        if not self.modularity_enabled:
            raise RuntimeError("modularity solver requested while modularity is disabled")
        data_collections = self.modularity_config.get("data_collections")
        params = self.modularity_config.get("params")
        cluster_ids_by_level = getattr(data_collections, "modularity_cluster_ids_by_level", [])
        if not cluster_ids_by_level:
            raise RuntimeError("modularity_inflation_flag=1 but no active modularity clusters are available")
        schedule = getattr(params, "modularity_cluster_level_schedule", "coarse_to_fine")
        round_idx = max(int(inflation_round), 0)
        if schedule == "coarse_to_fine":
            level_idx = min(round_idx, len(cluster_ids_by_level) - 1)
        elif schedule == "fine_to_coarse":
            level_idx = max(len(cluster_ids_by_level) - 1 - round_idx, 0)
        else:
            raise RuntimeError("Unsupported modularity_cluster_level_schedule=%s" % schedule)
        return level_idx, cluster_ids_by_level[level_idx]

    def _compute_modularity_route_opt_area(
        self,
        pos,
        node_size_x,
        node_size_y,
        old_movable_area,
        modularity_maps,
        inflation_round,
    ):
        params = self.modularity_config.get("params")
        placedb = self.modularity_config.get("placedb")
        node_weights = self.modularity_config.get("node_weights")
        solver_demand_map = modularity_maps.get("solver_demand_map")
        solver_capacity_map = modularity_maps.get("solver_capacity_map")
        if solver_demand_map is None or solver_capacity_map is None:
            raise RuntimeError("modularity_inflation_flag=1 requires solver_demand_map and solver_capacity_map")

        level_idx, cluster_ids = self._select_modularity_cluster_ids(inflation_round)
        solver_bundle = convex_inflation_solver.run_convex_inflation(
            placedb=placedb,
            params=params,
            pos=pos,
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            cluster_ids=cluster_ids,
            demand_map=solver_demand_map,
            capacity_map=solver_capacity_map,
            node_weights=node_weights,
        )
        solve_result = solver_bundle.solve_result
        self.last_modularity_summary = {
            "level_idx": int(level_idx),
            "num_active_bins": int(solver_bundle.active_bins.flat_bin_ids.numel()),
            "num_records": int(solver_bundle.demand_records.demand_values.numel()),
            "converged": bool(solve_result.converged),
            "fallback_used": bool(solve_result.fallback_used),
            "fallback_reason": solve_result.fallback_reason,
            "max_violation": float(solve_result.max_violation),
            "cluster_inflation_max": float(solve_result.cluster_inflation.max().item()) if solve_result.cluster_inflation.numel() else 1.0,
        }
        if solve_result.fallback_used:
            logger.warning(
                "Modularity convex inflation fallback at round %d level %d: reason=%s active_bins=%d records=%d",
                int(inflation_round),
                int(level_idx),
                solve_result.fallback_reason,
                int(solver_bundle.active_bins.flat_bin_ids.numel()),
                int(solver_bundle.demand_records.demand_values.numel()),
            )
            return None

        node_inflation = solver_bundle.node_inflation.clamp(
            min=1.0,
            max=float(getattr(params, "modularity_max_inflation", self.max_route_opt_adjust_rate)),
        )
        local_demand_map = modularity_maps.get("local_demand_map")
        global_demand_map = modularity_maps.get("global_demand_map")
        full_capacity_map = modularity_maps.get("capacity_map")
        if (
            isinstance(local_demand_map, torch.Tensor)
            and isinstance(global_demand_map, torch.Tensor)
            and isinstance(full_capacity_map, torch.Tensor)
            and solver_bundle.active_bins.flat_bin_ids.numel() > 0
        ):
            active_bin_inflation = convex_inflation_solver.aggregate_active_bin_inflation(
                solver_bundle.demand_records,
                solve_result.cluster_inflation,
            )
            bin_inflation_map = convex_inflation_solver.active_values_to_full_map(
                solver_bundle.active_bins,
                active_bin_inflation,
                fill_value=1.0,
            )
            corrected_bin_map = convex_inflation_solver.iterative_local_gcell_correction(
                bin_inflation_map,
                local_demand_map=local_demand_map,
                global_demand_map=global_demand_map,
                capacity_map=full_capacity_map,
                gamma=getattr(params, "modularity_local_gamma", 0.2),
                max_inflation=getattr(params, "modularity_max_inflation", self.max_route_opt_adjust_rate),
                num_iters=getattr(params, "modularity_local_max_iters", 3),
            )
            local_node_inflation = convex_inflation_solver.sample_node_inflation_from_bin_map(
                placedb=placedb,
                pos=pos,
                node_size_x=node_size_x,
                node_size_y=node_size_y,
                bin_inflation_map=corrected_bin_map,
                reduction="max",
            )
            node_inflation = torch.max(node_inflation, local_node_inflation[: self.num_movable_nodes])

        logger.info(
            "Modularity convex inflation round %d level %d: active_bins=%d records=%d max_violation=%.6f node_inflation avg/max=%.4f/%.4f",
            int(inflation_round),
            int(level_idx),
            int(solver_bundle.active_bins.flat_bin_ids.numel()),
            int(solver_bundle.demand_records.demand_values.numel()),
            float(solve_result.max_violation),
            float(node_inflation.mean().item()) if node_inflation.numel() else 1.0,
            float(node_inflation.max().item()) if node_inflation.numel() else 1.0,
        )
        self.last_modularity_summary["node_inflation_max"] = float(node_inflation.max().item()) if node_inflation.numel() else 1.0
        return old_movable_area * node_inflation

    def forward(self, pos, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, target_density, route_utilization_map,
                pin_utilization_map, modularity_maps=None, inflation_round=0,
                fixed_target_area=None):

        with torch.no_grad():
            adjust_area_flag = True
            adjust_route_area_flag = route_utilization_map is not None
            adjust_pin_area_flag = pin_utilization_map is not None

            if not (adjust_pin_area_flag or adjust_route_area_flag):
                return False, False, False

            # compute old areas of movable nodes
            node_size_x_movable = node_size_x[:self.num_movable_nodes]
            node_size_y_movable = node_size_y[:self.num_movable_nodes]
            old_node_size_x_movable = node_size_x_movable.clone()
            old_node_size_y_movable = node_size_y_movable.clone()
            
            old_movable_area = node_size_x_movable * node_size_y_movable
            old_movable_area_sum = old_movable_area.sum()
            # compute old areas of filler nodes
            if self.num_filler_nodes > 0:
                node_size_x_filler = node_size_x[-self.num_filler_nodes:]
                node_size_y_filler = node_size_y[-self.num_filler_nodes:]
            else:
                node_size_x_filler = node_size_x[:0]
                node_size_y_filler = node_size_y[:0]
            old_filler_area_sum = (node_size_x_filler *
                                   node_size_y_filler).sum()
            fixed_target_area_tensor = None
            if fixed_target_area is not None:
                if isinstance(fixed_target_area, torch.Tensor):
                    fixed_target_area_value = float(
                        fixed_target_area.detach().cpu().reshape(-1)[0].item()
                    )
                else:
                    fixed_target_area_value = float(fixed_target_area)
                fixed_target_area_value = max(
                    0.0,
                    min(fixed_target_area_value, float(self.total_place_area)),
                )
                fixed_target_area_tensor = old_movable_area_sum.new_tensor(
                    fixed_target_area_value
                )

            # compute routability optimized area
            route_power_map = None
            route_effective_map = None
            if adjust_route_area_flag:
                route_opt_area = None
                if self.modularity_enabled:
                    if modularity_maps is None:
                        raise RuntimeError(
                            "modularity_inflation_flag=1 requires modularity_maps when adjust_route_area_flag is true"
                        )
                    route_opt_area = self._compute_modularity_route_opt_area(
                        pos=pos,
                        node_size_x=node_size_x,
                        node_size_y=node_size_y,
                        old_movable_area=old_movable_area,
                        modularity_maps=modularity_maps,
                        inflation_round=inflation_round,
                    )
                if route_opt_area is None:
                    route_power_map = route_utilization_map.pow(
                        self.route_opt_adjust_exponent
                    )
                    route_effective_map = route_power_map.clamp(
                        min=self.min_route_opt_adjust_rate,
                        max=self.max_route_opt_adjust_rate,
                    )
                    route_opt_area = self.compute_node_area_route(
                        pos, node_size_x, node_size_y, route_effective_map)
            # compute pin density optimized area
            if adjust_pin_area_flag:
                pin_opt_area = self.compute_node_area_pin(
                    pos,
                    node_size_x,
                    node_size_y,
                    # clamp the pin utilization map
                    pin_utilization_map.clamp(
                        min=self.min_pin_opt_adjust_rate,
                        max=self.max_pin_opt_adjust_rate))

            # compute the extra area max(route_opt_area, pin_opt_area) over the base area for each movable node
            if adjust_route_area_flag and adjust_pin_area_flag:
                area_increment = F.relu(
                    torch.max(route_opt_area, pin_opt_area) - old_movable_area)
            elif adjust_route_area_flag:
                area_increment = F.relu(route_opt_area - old_movable_area)
            else:
                area_increment = F.relu(pin_opt_area - old_movable_area)
            area_increment_sum = area_increment.sum()

            # plot
            # 将area_increment大于0的标准单元在图中打印出来，并根据数值给予亮度。 
            
            # try:
            #     import matplotlib.patches as patches
            #     # 找到面积增加的单元
            #     inflated_indices = torch.where(area_increment > 0)[0]
            #     if inflated_indices.numel() > 0:
            #         # 获取这些单元的位置、大小和面积增加量
            #         inflated_pos_x = pos.data[:self.num_movable_nodes][inflated_indices].cpu().numpy()
            #         num_nodes = pos.numel() // 2
            #         inflated_pos_y = pos.data[num_nodes:num_nodes + self.num_movable_nodes][inflated_indices].cpu().numpy()
            #         inflated_size_x = old_node_size_x_movable[inflated_indices].cpu().numpy()
            #         inflated_size_y = old_node_size_y_movable[inflated_indices].cpu().numpy()
            #         inflated_values = area_increment[inflated_indices].cpu().numpy()

            #         # 创建图像
            #         fig, ax = plt.subplots(figsize=(10, 10))
            #         ax.set_xlim(self.xl, self.xh)
            #         ax.set_ylim(self.yl, self.yh)
            #         ax.set_aspect('equal', adjustable='box')

            #         # 归一化颜色
            #         norm = plt.Normalize(vmin=inflated_values.min(), vmax=inflated_values.max())
            #         cmap = plt.get_cmap('viridis')

            #         # 绘制每个被放大单元的矩形
            #         for i in range(len(inflated_indices)):
            #             rect = patches.Rectangle(
            #                 (inflated_pos_x[i], inflated_pos_y[i]),
            #                 inflated_size_x[i],
            #                 inflated_size_y[i],
            #                 linewidth=0,
            #                 edgecolor='none',
            #                 facecolor=cmap(norm(inflated_values[i]))
            #             )
            #             ax.add_patch(rect)
                    
            #         # 添加颜色条
            #         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            #         sm.set_array([])
            #         fig.colorbar(sm, ax=ax, orientation='vertical', label='Area Increment')

            #         plt.title('Inflated Movable Nodes Visualization')
            #         plt.xlabel('X coordinate')
            #         plt.ylabel('Y coordinate')
            #         plt.savefig("inflated_nodes.png", dpi=300)
            #         plt.close(fig)
            # except Exception as e:
            #     logger.error(f"Failed to plot inflated nodes: {e}")
            ### end plot
            
            # check whether the total area is larger than the max area requirement
            # If yes, scale the extra area to meet the requirement
            # We assume the total base area is no greater than the max area requirement
            total_whitespace_area_tensor = torch.as_tensor(
                self.total_whitespace_area,
                device=old_movable_area_sum.device,
                dtype=old_movable_area_sum.dtype,
            )
            total_place_area_tensor = torch.as_tensor(
                self.total_place_area,
                device=old_movable_area_sum.device,
                dtype=old_movable_area_sum.dtype,
            )
            whitespace_area_budget = (
                self.inflation_area_budget_ratio * total_whitespace_area_tensor
            )
            # Inflation may consume up to the configured share of baseline
            # whitespace without exceeding the remaining physical place area.
            remaining_area_budget = total_place_area_tensor - old_movable_area_sum
            area_budget = torch.minimum(
                whitespace_area_budget,
                remaining_area_budget,
            )
            logger.info(
                "inflation area budget %.3E: whitespace cap %.3E (ratio %.6g), "
                "physical cap %.3E, old movable %.3E, old filler %.3E",
                area_budget,
                whitespace_area_budget,
                self.inflation_area_budget_ratio,
                remaining_area_budget,
                old_movable_area_sum,
                old_filler_area_sum,
            )
            if area_increment_sum.data.item() <= 0:
                scale_factor = 0
            else:
                scale_factor = (area_budget / area_increment_sum).item()

            # set the new_movable_area as base_area + scaled area increment
            if scale_factor <= 0:
                new_movable_area = old_movable_area
                area_increment_sum = 0
            elif scale_factor >= 1:
                new_movable_area = old_movable_area + area_increment
            else:
                new_movable_area = old_movable_area + area_increment * scale_factor
                area_increment_sum *= scale_factor
            actual_area_increment = new_movable_area - old_movable_area
            route_color_limits = self._maybe_plot_route_inflation_maps(
                route_power_map=route_power_map,
                route_effective_map=route_effective_map,
                scale_factor=scale_factor,
                old_movable_area=old_movable_area,
                actual_area_increment=actual_area_increment,
                inflation_round=inflation_round,
            )
            self._maybe_plot_inflation_cells(
                pos=pos,
                old_movable_area=old_movable_area,
                old_node_size_x_movable=old_node_size_x_movable,
                old_node_size_y_movable=old_node_size_y_movable,
                actual_area_increment=actual_area_increment,
                inflation_round=inflation_round,
                color_limits=route_color_limits,
            )
            new_movable_area_sum = old_movable_area_sum + area_increment_sum
            area_increment_ratio = area_increment_sum / old_movable_area_sum
            logger.info(
                "area_increment = %E, area_increment / movable = %g, area_adjust_stop_ratio = %g"
                % (area_increment_sum, area_increment_ratio,
                   self.area_adjust_stop_ratio))
            logger.info(
                "area_increment / total_place_area = %g, area_increment / filler = %g, area_increment / total_whitespace_area = %g"
                % (area_increment_sum / self.total_place_area,
                   area_increment_sum / old_filler_area_sum,
                   area_increment_sum / self.total_whitespace_area))

            # compute the adjusted area increase ratio
            # disable some of the area adjustment if the condition holds
            if adjust_route_area_flag:
                route_area_increment_ratio = F.relu(
                    route_opt_area -
                    old_movable_area).sum() / old_movable_area_sum
                adjust_route_area_flag = route_area_increment_ratio.data.item(
                ) > self.route_area_adjust_stop_ratio
                
                # route_opt_area
                
                logger.info(
                    "route_area_increment_ratio = %g, route_area_adjust_stop_ratio = %g"
                    % (route_area_increment_ratio,
                       self.route_area_adjust_stop_ratio))
            if adjust_pin_area_flag:
                pin_area_increment_ratio = F.relu(
                    pin_opt_area -
                    old_movable_area).sum() / old_movable_area_sum
                adjust_pin_area_flag = pin_area_increment_ratio.data.item(
                ) > self.pin_area_adjust_stop_ratio
                logger.info(
                    "pin_area_increment_ratio = %g, pin_area_adjust_stop_ratio = %g"
                    % (pin_area_increment_ratio,
                       self.pin_area_adjust_stop_ratio))
            adjust_area_flag = (
                area_increment_ratio.data.item() > self.area_adjust_stop_ratio
            ) and (adjust_route_area_flag or adjust_pin_area_flag)

            if not adjust_area_flag:
                return adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag

            num_nodes = int(pos.numel() / 2)
            # adjust the size and positions of movable nodes
            # each movable node have its own inflation ratio, the shape of movable_nodes_ratio is (num_movable_nodes)
            # we keep the centers the same
            movable_nodes_ratio = new_movable_area / old_movable_area
            logger.info(
                "inflation ratio for movable nodes: avg/max %g/%g" %
                (movable_nodes_ratio.mean(), movable_nodes_ratio.max()))
            
            movable_nodes_ratio.sqrt_()
            # convert positions to centers
            pos.data[:self.num_movable_nodes] += node_size_x_movable * 0.5
            pos.data[num_nodes:num_nodes +
                     self.num_movable_nodes] += node_size_y_movable * 0.5
            # scale size
            node_size_x_movable *= movable_nodes_ratio
            node_size_y_movable *= movable_nodes_ratio
            # convert back to lower left corners
            pos.data[:self.num_movable_nodes] -= node_size_x_movable * 0.5
            pos.data[num_nodes:num_nodes +
                     self.num_movable_nodes] -= node_size_y_movable * 0.5

            # finally scale the filler instance areas to match the active area budget
            # all the filler nodes share the same deflation ratio, filler_nodes_ratio is a scalar
            # we keep the centers the same
            if fixed_target_area_tensor is not None:
                if self.num_filler_nodes > 0 and old_filler_area_sum > 0:
                    # Enhanced inflation consumes filler and out-of-target
                    # whitespace at the same rate instead of exhausting filler first.
                    old_whitespace_area_sum = F.relu(remaining_area_budget)
                    whitespace_consumption_ratio = torch.clamp(
                        area_increment_sum / old_whitespace_area_sum,
                        min=0.0,
                        max=1.0,
                    )
                    new_filler_area_sum = old_filler_area_sum * (
                        1.0 - whitespace_consumption_ratio
                    )
                    filler_nodes_ratio = new_filler_area_sum / old_filler_area_sum
                    old_non_filler_whitespace_area_sum = F.relu(
                        old_whitespace_area_sum - old_filler_area_sum
                    )
                    logger.info(
                        "proportional whitespace consumption: ratio %g, "
                        "filler %.3E -> %.3E, non-filler whitespace %.3E -> %.3E"
                        % (
                            whitespace_consumption_ratio,
                            old_filler_area_sum,
                            new_filler_area_sum,
                            old_non_filler_whitespace_area_sum,
                            old_non_filler_whitespace_area_sum
                            * (1.0 - whitespace_consumption_ratio),
                        )
                    )
                    logger.info("inflation ratio for filler nodes: %g" %
                                (filler_nodes_ratio))
                    filler_nodes_ratio.sqrt_()
                    # convert positions to centers
                    pos.data[num_nodes - self.num_filler_nodes:
                             num_nodes] += node_size_x_filler * 0.5
                    pos.data[-self.num_filler_nodes:] += node_size_y_filler * 0.5
                    # scale size
                    node_size_x_filler *= filler_nodes_ratio
                    node_size_y_filler *= filler_nodes_ratio
                    # convert back to lower left corners
                    pos.data[num_nodes - self.num_filler_nodes:
                             num_nodes] -= node_size_x_filler * 0.5
                    pos.data[-self.num_filler_nodes:] -= node_size_y_filler * 0.5
                else:
                    new_filler_area_sum = old_filler_area_sum
            elif (
                self.num_filler_nodes > 0
                and old_filler_area_sum > 0
                and new_movable_area_sum + old_filler_area_sum > self.total_place_area
            ):
                new_filler_area_sum = F.relu(self.total_place_area -
                                             new_movable_area_sum)
                filler_nodes_ratio = new_filler_area_sum / old_filler_area_sum
                logger.info("inflation ratio for filler nodes: %g" %
                            (filler_nodes_ratio))
                filler_nodes_ratio.sqrt_()
                # convert positions to centers
                pos.data[num_nodes - self.num_filler_nodes:
                         num_nodes] += node_size_x_filler * 0.5
                pos.data[-self.num_filler_nodes:] += node_size_y_filler * 0.5
                # scale size
                node_size_x_filler *= filler_nodes_ratio
                node_size_y_filler *= filler_nodes_ratio
                # convert back to lower left corners
                pos.data[num_nodes - self.num_filler_nodes:
                         num_nodes] -= node_size_x_filler * 0.5
                pos.data[-self.num_filler_nodes:] -= node_size_y_filler * 0.5
            else:
                new_filler_area_sum = old_filler_area_sum

            logger.info(
                "old total movable nodes area %.3E, filler area %.3E, total movable + filler area %.3E, total_place_area %.3E"
                % (old_movable_area_sum, old_filler_area_sum,
                   old_movable_area_sum + old_filler_area_sum,
                   self.total_place_area))
            logger.info(
                "new total movable nodes area %.3E, filler area %.3E, total movable + filler area %.3E, total_place_area %.3E"
                % (new_movable_area_sum, new_filler_area_sum,
                   new_movable_area_sum + new_filler_area_sum,
                   self.total_place_area))
            target_density.copy_(
                (new_movable_area_sum + new_filler_area_sum) /
                self.total_place_area)
            logger.info("new target_density %g" % (target_density))

            if pos.is_cuda:
                func = update_pin_offset_cuda.forward
            else:
                func = update_pin_offset_cpp.forward
            # update_pin_offset requires node_size before adjustment
            # update_pin_offset makes sure the absolute pin locations remain the same after inflation
            func(old_node_size_x_movable , old_node_size_y_movable , self.flat_node2pin_start_map,
                 self.flat_node2pin_map, movable_nodes_ratio,
                 self.num_movable_nodes, pin_offset_x, pin_offset_y)
            return adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag
