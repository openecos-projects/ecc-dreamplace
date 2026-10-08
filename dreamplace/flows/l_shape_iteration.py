"""L-shape activation and periodic feedback before objective evaluation.

Context is call-scoped: it references the live data/ops without owning copies.
The current model remains the objective owner; this stage never steps it.
"""
from dataclasses import dataclass
import os
import time
import logging
import torch
from dreamplace.ops.routability.profile_timing import l_shape_log_verbose, profile_scope
from dreamplace.ops.routability.l_shape_inputs import (
    load_l_shape_topology_pack_from_gpugr as _load_l_shape_topology_pack_from_gpugr,
    prepare_l_shape_inputs_from_egr as _prepare_l_shape_inputs_from_egr,
    prepare_l_shape_inputs_from_gpugr as _prepare_l_shape_inputs_from_gpugr,
    resolve_l_directions_for_l_shape as _resolve_l_directions_for_l_shape,
    should_skip_resolver_l_direction_for_soft as _should_skip_resolver_l_direction_for_soft,
)
from dreamplace.ops.steiner_topo.ggr_l_shape_topology import use_ggr_l_shape_topology
from dreamplace.ops.routability.l_shape_overflow_update import update_overflow
from dreamplace.ops.routability.l_shape_iteration_plots import plot_initial_state, plot_updated_state

@dataclass(frozen=True)
class LShapeIterationContext:
    data_collections: object
    op_collections: object
    topology_prepared: bool = False


def prepare_flute_topology(context, pos):
    """Keep a prepared/frozen tree; fresh feedback supplies directions afterward."""
    if context.topology_prepared:
        return
    topo = context.op_collections.steiner_topo_op
    with torch.no_grad():
        pin_pos = context.op_collections.pin_pos_op(pos).detach().cpu().contiguous()
        if topo.topology_frozen:
            topo(pin_pos)
        else:
            topo.rebuild_tree(pin_pos, resolve_l_directions=False)
    data = context.data_collections
    data.net_flat_topo_sort = topo.net_flat_topo_sort
    data.net_flat_topo_sort_start = topo.net_flat_topo_sort_start
    data.pin_fa = topo.pin_fa
    data.flat_pin_to = topo.flat_pin_to
    data.flat_pin_to_start = topo.flat_pin_to_start
    data.flat_pin_from = topo.flat_pin_from

def update_iteration(context, params, placedb, model, pos, iteration, cur_metric, l_shape_policy):

    l_shape_policy.maybe_reenable(
        model,
        float(cur_metric.overflow[-1]),
    )

        # # 条件2: 也可以根据iteration启用
        # if iteration >= getattr(params, 'l_shape_start_iteration', 100):
        #     model.enable_l_shape_routability = True

    if model.enable_l_shape_routability and not model.use_l_shape_routability:
        # 首次启用L形routability
        t_l_shape_init = time.time()

        L_shape_num_bins_x = params.num_bins_x
        L_shape_num_bins_y = params.num_bins_y

        ggr_topology_mode = use_ggr_l_shape_topology(params)
        gpugr_inputs = None
        l_shape_inputs = None
        if ggr_topology_mode:
            gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                params,
                placedb,
                pos,
                model=model,
            )
            l_shape_inputs = gpugr_inputs
            l_directions = _load_l_shape_topology_pack_from_gpugr(
                params,
                pos,
                context.op_collections.pin_pos_op,
                context.op_collections.steiner_topo_op,
                context.data_collections,
                l_shape_inputs,
                "l_shape_init.load_ggr_topology_pack",
                iteration,
            )
        else:
            with profile_scope(params, "l_shape_init.rebuild_tree", tensor=pos, iteration=iteration):
                prepare_flute_topology(context, pos)
            if getattr(params, "l_direction_use_gpugr", False):
                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                    params,
                    placedb,
                    pos,
                    model=model,
                )
                l_shape_inputs = gpugr_inputs
            else:
                l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                    params,
                    placedb,
                    pos,
                    model=model,
                )

        supply_map = l_shape_inputs["supply_map"]
        demand_map = l_shape_inputs["demand_map"]
        wire_width = l_shape_inputs["wire_width"]

        # Resolve L directions from the selected topology source.
        steiner_topo_op = context.op_collections.steiner_topo_op
        if ggr_topology_mode:
            l_directions = steiner_topo_op.edge_l_directions
        elif _should_skip_resolver_l_direction_for_soft(params):
            steiner_topo_op.edge_l_directions = None
            l_directions = None
            if l_shape_log_verbose(params) >= 2:
                logging.info(
                    "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                    "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                    "will be refreshed."
                )
        else:
            l_directions = _resolve_l_directions_for_l_shape(
                params,
                placedb,
                pos,
                steiner_topo_op,
                gpugr_route_entries=(
                    gpugr_inputs["route_entries"]
                    if getattr(params, "l_direction_use_gpugr", False)
                    else None
                ),
                gpugr_metrics=(
                    gpugr_inputs["metrics"]
                    if getattr(params, "l_direction_use_gpugr", False)
                    else None
                ),
                gpugr_route_grid=(
                    (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                    if getattr(params, "l_direction_use_gpugr", False)
                    else None
                ),
            )

        # # ========== Plot edges with L-shape by l_direction ==========
        # import matplotlib.pyplot as plt
        # import matplotlib.collections as mc

        # # 获取坐标和边信息
        # newx = steiner_topo_op.newx.cpu().numpy()
        # newy = steiner_topo_op.newy.cpu().numpy()
        # flat_pin_from = context.data_collections.flat_pin_from.cpu().numpy()
        # flat_pin_to = context.data_collections.flat_pin_to.cpu().numpy()
        # l_dirs = l_directions.cpu().numpy()

        # # 颜色映射: H_FIRST=0(红), V_FIRST=1(蓝), STRAIGHT=2(绿), FAKE_STRAIGHT=3(橙)
        # color_map = {
        #     0: 'red',      # H_FIRST: 先水平后垂直
        #     1: 'blue',     # V_FIRST: 先垂直后水平
        #     2: 'green',    # STRAIGHT: 直线
        #     3: 'orange'    # FAKE_STRAIGHT: 伪直线
        # }
        # label_map = {
        #     0: 'H_FIRST (H→V)',
        #     1: 'V_FIRST (V→H)',
        #     2: 'STRAIGHT',
        #     3: 'FAKE_STRAIGHT'
        # }

        # # 按l_direction分组收集线段
        # # L形边变成两段，直线保持一段
        # edges_by_dir = {0: [], 1: [], 2: [], 3: []}
        # edge_count_by_dir = {0: 0, 1: 0, 2: 0, 3: 0}

        # for i in range(len(l_dirs)):
        #     from_idx = flat_pin_from[i]
        #     to_idx = flat_pin_to[i]
        #     if from_idx == -1 or to_idx == -1:
        #         continue
        #     x1, y1 = newx[from_idx], newy[from_idx]
        #     x2, y2 = newx[to_idx], newy[to_idx]
        #     direction = int(l_dirs[i])
        #     if direction not in edges_by_dir:
        #         direction = 1  # 默认用V_FIRST

        #     edge_count_by_dir[direction] += 1

        #     # 根据方向生成路径
        #     if direction == 0:  # H_FIRST: 水平优先 (x1,y1) -> (x2,y1) -> (x2,y2)
        #         corner = (x2, y1)
        #         edges_by_dir[direction].append([(x1, y1), corner])
        #         edges_by_dir[direction].append([corner, (x2, y2)])
        #     elif direction == 2:  # STRAIGHT: 直线
        #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
        #     elif direction == 3:  # FAKE_STRAIGHT: 伪直线（画成直线）
        #         edges_by_dir[direction].append([(x1, y1), (x2, y2)])
        #     elif direction == 1:  # V_FIRST: 垂直优先 (x1,y1) -> (x1,y2) -> (x2,y2)
        #         # (x1,y1) -> (x1,y2) -> (x2,y2)
        #         corner = (x1, y2)
        #         edges_by_dir[direction].append([(x1, y1), corner])
        #         edges_by_dir[direction].append([corner, (x2, y2)])

        # # 绘图
        # fig, ax = plt.subplots(figsize=(12, 10))

        # for direction, edges in edges_by_dir.items():
        #     if len(edges) > 0:
        #         lc = mc.LineCollection(edges, colors=color_map[direction], 
        #                               linewidths=0.5, alpha=0.7,
        #                               label=f'{label_map[direction]} ({edge_count_by_dir[direction]})')
        #         ax.add_collection(lc)

        # ax.autoscale()
        # ax.set_aspect('equal')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_title('Steiner Tree L-Shape Edges')
        # ax.legend(loc='upper right')

        # # 保存图片
        # plot_path = os.path.join(params.result_dir, 'l_direction_edges.png')
        # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        # plt.close()
        # logging.info(f"L-direction edge plot saved to {plot_path}")

        # exit(0)

        # Initialize the L-shape routability operator.

        with profile_scope(params, "l_shape_init.construct_op", tensor=pos, iteration=iteration):
            model.init_l_shape_routability(
                wire_width=wire_width,
                num_bins_x=L_shape_num_bins_x,
                num_bins_y=L_shape_num_bins_y,
                target_density=supply_map,
                target_demand=demand_map,
                raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                supply_original=l_shape_inputs.get("supply_original"),
                target_density_h=l_shape_inputs.get("supply_map_h"),
                target_density_v=l_shape_inputs.get("supply_map_v"),
                target_demand_h=l_shape_inputs.get("demand_map_h"),
                target_demand_v=l_shape_inputs.get("demand_map_v"),
                raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                supply_original_h=l_shape_inputs.get("supply_original_h"),
                supply_original_v=l_shape_inputs.get("supply_original_v"),
                fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
            )
        if model.l_shape_routability_op is not None:
            model.l_shape_routability_op.update_same_net_topology(
                topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                topo_stats=l_shape_inputs.get("same_net_topo_stats"),
            )
        if hasattr(model, "start_l_shape_weight_controller"):
            model.start_l_shape_weight_controller(iteration)
        # 初始化基于L-shape overflow的外环状态
        l_shape_policy.reset_overflow_state(model)

        if l_shape_log_verbose(params) >= 1:
            logging.info(f"L-shape routability enabled at iteration {iteration}, "
                    f"overflow={cur_metric.overflow[-1]:.4f}, "
                        f"threshold={float(getattr(model, '_l_shape_reenable_threshold', getattr(params, 'l_shape_overflow_threshold', 0.2))):.4f}, "
                        f"descend_streak={int(getattr(model, '_l_shape_reenable_descend_streak', 0))}, "
                        f"init time={((time.time() - t_l_shape_init) * 1000):.2f}ms")
        l_shape_policy.reset_reenable_progress(model)

        # ========== 梯度正确性检查 (可选) ==========
        if getattr(params, 'l_shape_gradient_check', False):
            # 1. 运行梯度链诊断
            logging.info("Running L-shape gradient chain diagnosis...")
            model.diagnose_l_shape_gradient_chain(pos)

            # 2. 运行梯度问题诊断（检查边界问题）
            logging.info("Running L-shape gradient issues diagnosis...")
            model.diagnose_l_shape_gradient_issues(pos)

            # 3. 运行梯度方向检查（更实用）
            logging.info("Running L-shape gradient direction check...")
            direction_results = model.check_l_shape_gradient_direction(
                pos, 
                step_sizes=[0.1, 1.0, 10.0, 100.0]
            )

            # 4. 可选：运行数值梯度检查（对bin-based函数可能失败）
            logging.info("Running L-shape gradient numerical check...")
            logging.info("Note: Numerical check may fail for bin-based density functions")
            grad_check_results = model.check_l_shape_gradient_numerical(
                pos, 
                num_check=200,  # 检查200个位置
                eps=1e-3,       # 有限差分步长
                check_movable_only=True,
                verbose=True
            )

            # 保存结果到文件
            import json
            results_to_save = {
                'direction_check': direction_results,
                'numerical_check': grad_check_results
            }
            grad_check_path = os.path.join(params.result_dir, "l_shape_grad_check.json")
            with open(grad_check_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            logging.info(f"Gradient check results saved to {grad_check_path}")

            exit(0)
        # =============================================

        # 可视化L形密度图和segments
        plot_initial_state(params, model, pos, iteration)
        return True

        # exit(0)
    # 定期更新Steiner树和L方向（每N次迭代）
    elif model.use_l_shape_routability and (iteration % params.l_shape_update_interval == 0):
        t_l_shape_update = time.time()

        # 重置L形segment缓存（EGR将重新运行）
        if model.l_shape_routability_op is not None:
            model.l_shape_routability_op.segment_builder.reset_cache()

        ggr_topology_mode = use_ggr_l_shape_topology(params)
        gpugr_inputs = None
        l_shape_inputs = None
        if ggr_topology_mode:
            gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                params,
                placedb,
                pos,
                model=model,
            )
            l_shape_inputs = gpugr_inputs
            l_directions = _load_l_shape_topology_pack_from_gpugr(
                params,
                pos,
                context.op_collections.pin_pos_op,
                context.op_collections.steiner_topo_op,
                context.data_collections,
                l_shape_inputs,
                "l_shape_update.load_ggr_topology_pack",
                iteration,
            )
        else:
            with profile_scope(params, "l_shape_update.rebuild_tree", tensor=pos, iteration=iteration):
                prepare_flute_topology(context, pos)
            if not getattr(params, "l_direction_use_gpugr", False):
                l_shape_inputs = _prepare_l_shape_inputs_from_egr(
                    params,
                    placedb,
                    pos,
                    model=model,
                )
            else:
                gpugr_inputs = _prepare_l_shape_inputs_from_gpugr(
                    params,
                    placedb,
                    pos,
                    model=model,
                )
                l_shape_inputs = gpugr_inputs

        steiner_topo_op = context.op_collections.steiner_topo_op
        if ggr_topology_mode:
            l_directions = steiner_topo_op.edge_l_directions
        elif _should_skip_resolver_l_direction_for_soft(params):
            steiner_topo_op.edge_l_directions = None
            l_directions = None
            if l_shape_log_verbose(params) >= 2:
                logging.info(
                    "Skip resolver L-direction parsing because soft_l_assignment is enabled "
                    "and soft_l_use_resolver_prior is disabled; only routing supply/demand maps "
                    "will be refreshed."
                )
        else:
            l_directions = _resolve_l_directions_for_l_shape(
                params,
                placedb,
                pos,
                steiner_topo_op,
                gpugr_route_entries=(
                    gpugr_inputs["route_entries"]
                    if gpugr_inputs is not None
                    else None
                ),
                gpugr_metrics=(
                    gpugr_inputs["metrics"]
                    if gpugr_inputs is not None
                    else None
                ),
                gpugr_route_grid=(
                    (gpugr_inputs["route_xsize"], gpugr_inputs["route_ysize"])
                    if gpugr_inputs is not None
                    else None
                ),
            )
        if model.l_shape_routability_op is not None and l_shape_inputs is not None:
            with profile_scope(params, "l_shape_update.update_targets", tensor=pos, iteration=iteration):
                model.l_shape_routability_op.update_targets(
                    target_density=l_shape_inputs["supply_map"],
                    target_demand=l_shape_inputs["demand_map"],
                    raw_wire_demand_map=l_shape_inputs.get("raw_wire_demand_map"),
                    target_density_h=l_shape_inputs.get("supply_map_h"),
                    target_density_v=l_shape_inputs.get("supply_map_v"),
                    target_demand_h=l_shape_inputs.get("demand_map_h"),
                    target_demand_v=l_shape_inputs.get("demand_map_v"),
                    raw_wire_demand_map_h=l_shape_inputs.get("raw_wire_demand_map_h"),
                    raw_wire_demand_map_v=l_shape_inputs.get("raw_wire_demand_map_v"),
                    supply_original=l_shape_inputs.get("supply_original"),
                    supply_original_h=l_shape_inputs.get("supply_original_h"),
                    supply_original_v=l_shape_inputs.get("supply_original_v"),
                    fix_usage_map=l_shape_inputs.get("fix_usage_map"),
                    fix_usage_map_h=l_shape_inputs.get("fix_usage_map_h"),
                    fix_usage_map_v=l_shape_inputs.get("fix_usage_map_v"),
                )
                model.l_shape_routability_op.update_same_net_topology(
                    topo_cache=l_shape_inputs.get("same_net_topo_cache"),
                    topo_stats=l_shape_inputs.get("same_net_topo_stats"),
                )
            if (
                getattr(model.l_shape_routability_op, "wire_width_h", None) is None
                and getattr(model.l_shape_routability_op, "wire_width_v", None) is None
            ):
                updated_wire_width = float(l_shape_inputs["wire_width"])
                current_wire_width = float(model.l_shape_routability_op.wire_width)
                if abs(updated_wire_width - current_wire_width) > 1e-6:
                    logging.info(
                        "L-shape periodic target update kept existing wire_width %.4f while refreshed maps imply %.4f. "
                        "Segment width is not updated online after initialization.",
                        current_wire_width,
                        updated_wire_width,
                    )
        density_map = None

        # 基于L-shape overflow变化更新target_ratio（外环慢速更新）
        density_map = update_overflow(params, model, pos, iteration, l_shape_policy, cur_metric)

        logging.debug(f"L-shape routability updated at iteration {iteration}, "
                     f"time={((time.time() - t_l_shape_update) * 1000):.2f}ms")

        # 定期可视化L形密度图和segments
        plot_updated_state(params, model, pos, iteration, density_map)
        return True
    return False
