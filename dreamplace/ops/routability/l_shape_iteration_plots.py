"""Existing L-shape iteration diagnostics; no optimizer or native mutations."""
import os
import logging

def plot_initial_state(params, model, pos, iteration, density_map=None):
    if params.l_shape_plot_flag:
        try:
            from dreamplace.ops.routability.l_shape_density_plots import (
                plot_l_shape_electric_overflow_map,
                plot_l_shape_initial_density_map,
                plot_l_shape_supply_maps,
            )
            from dreamplace.ops.routability.l_shape_source_plots import (
                plot_l_shape_macro_source_maps,
                plot_l_shape_electric_potential_map,
                plot_l_shape_true_source_maps,
                plot_segment_density_map,
            )
            from dreamplace.ops.routability.l_shape_segment_plots import (
                plot_soft_l_intermediate,
                plot_soft_l_scoring_maps,
            )

            # 获取密度图
            forward_source_snapshot = model.snapshot_l_shape_forward_state()
            density_map = model.get_l_shape_density_map(pos, use_l_direction=True)
            model.restore_l_shape_forward_state(forward_source_snapshot)
            if density_map is not None:
                density_plot_path = os.path.join(
                    params.result_dir, f"l_shape_density_iter{iteration}.png"
                )
                plot_segment_density_map(
                    density_map, density_plot_path,
                    title=f"L-shape Density (iter={iteration})",
                    colormap="binary"
                )
                logging.info(f"L-shape density plot saved to {density_plot_path}")

            # 绘制基于 electric potential 的 overflow map
            if model.l_shape_routability_op is not None and \
               model.l_shape_routability_op.cached_segments is not None:
                l_shape_op = model.l_shape_routability_op
                overflow_plot_path = os.path.join(
                    params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                )
                plot_l_shape_electric_overflow_map(
                    l_shape_op,
                    output_path=overflow_plot_path,
                    title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                )
                logging.info(f"L-shape electric overflow plot saved to {overflow_plot_path}")
                supply_plot_path = os.path.join(
                    params.result_dir, f"l_shape_supply_iter{iteration}.png"
                )
                plot_l_shape_supply_maps(
                    l_shape_op,
                    output_path=supply_plot_path,
                    title_prefix=f"L-shape Supply Debug (iter={iteration})",
                )
                logging.info(f"L-shape supply debug plot saved to {supply_plot_path}")
                initial_density_plot_path = os.path.join(
                    params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                )
                plot_l_shape_initial_density_map(
                    l_shape_op,
                    output_path=initial_density_plot_path,
                    title_prefix=f"L-shape Initial Density (iter={iteration})",
                )
                logging.info(
                    f"L-shape initial density plot saved to {initial_density_plot_path}"
                )
                source_plot_path = os.path.join(
                    params.result_dir, f"l_shape_source_iter{iteration}.png"
                )
                plot_l_shape_true_source_maps(
                    l_shape_op,
                    output_path=source_plot_path,
                    title_prefix=f"L-shape True Source (iter={iteration})",
                )
                logging.info(f"L-shape true source plot saved to {source_plot_path}")
                macro_source_plot_path = os.path.join(
                    params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                )
                plot_l_shape_macro_source_maps(
                    l_shape_op,
                    output_path=macro_source_plot_path,
                    title_prefix=f"L-shape Macro Source (iter={iteration})",
                )
                logging.info(
                    f"L-shape macro source plot saved to {macro_source_plot_path}"
                )
                potential_plot_path = os.path.join(
                    params.result_dir, f"l_shape_potential_iter{iteration}.png"
                )
                plot_l_shape_electric_potential_map(
                    l_shape_op,
                    output_path=potential_plot_path,
                    title_prefix=f"L-shape Electric Potential (iter={iteration})",
                )
                logging.info(f"L-shape electric potential plot saved to {potential_plot_path}")
                if getattr(l_shape_op, "soft_l_assignment", False) and \
                   'soft_l_weights' in l_shape_op.cached_segments:
                    soft_plot_path = os.path.join(
                        params.result_dir, f"l_shape_soft_iter{iteration}.png"
                    )
                    plot_soft_l_intermediate(
                        l_shape_op.cached_segments,
                        output_path=soft_plot_path,
                    )
                    logging.info(f"Soft L-shape plot saved to {soft_plot_path}")
                    if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                        soft_scoring_path = os.path.join(
                            params.result_dir,
                            f"l_shape_soft_scoring_iter{iteration}.png",
                        )
                        plot_soft_l_scoring_maps(
                            l_shape_op.cached_soft_debug,
                            output_path=soft_scoring_path,
                            title_prefix=f"Soft L Scoring (iter={iteration})",
                        )
                        logging.info(
                            f"Soft L-shape scoring plot saved to {soft_scoring_path}"
                        )
        except Exception as e:
            logging.warning(f"Failed to plot L-shape density/segments: {e}")


def plot_updated_state(params, model, pos, iteration, density_map=None):
    if params.l_shape_plot_flag:
        try:
            from dreamplace.ops.routability.l_shape_density_plots import (
                plot_l_shape_electric_overflow_map,
                plot_l_shape_initial_density_map,
                plot_l_shape_supply_maps,
            )
            from dreamplace.ops.routability.l_shape_source_plots import (
                plot_l_shape_macro_source_maps,
                plot_l_shape_electric_potential_map,
                plot_l_shape_true_source_maps,
                plot_segment_density_map,
            )
            from dreamplace.ops.routability.l_shape_segment_plots import (
                plot_soft_l_intermediate,
                plot_soft_l_scoring_maps,
            )

            if density_map is None:
                forward_source_snapshot = model.snapshot_l_shape_forward_state()
                density_map = model.get_l_shape_density_map(
                    pos, use_l_direction=True
                )
                model.restore_l_shape_forward_state(forward_source_snapshot)
            if density_map is not None:
                density_plot_path = os.path.join(
                    params.result_dir, f"l_shape_density_iter{iteration}.png"
                )
                plot_segment_density_map(
                    density_map, density_plot_path,
                    title=f"L-shape Density (iter={iteration})",
                    colormap="binary"
                )

            # 绘制基于 electric potential 的 overflow map
            if model.l_shape_routability_op is not None and \
               model.l_shape_routability_op.cached_segments is not None:
                l_shape_op = model.l_shape_routability_op
                overflow_plot_path = os.path.join(
                    params.result_dir, f"l_shape_overflow_iter{iteration}.png"
                )
                plot_l_shape_electric_overflow_map(
                    l_shape_op,
                    output_path=overflow_plot_path,
                    title_prefix=f"L-shape Electric Overflow (iter={iteration})",
                )
                supply_plot_path = os.path.join(
                    params.result_dir, f"l_shape_supply_iter{iteration}.png"
                )
                plot_l_shape_supply_maps(
                    l_shape_op,
                    output_path=supply_plot_path,
                    title_prefix=f"L-shape Supply Debug (iter={iteration})",
                )
                initial_density_plot_path = os.path.join(
                    params.result_dir, f"l_shape_initial_density_iter{iteration}.png"
                )
                plot_l_shape_initial_density_map(
                    l_shape_op,
                    output_path=initial_density_plot_path,
                    title_prefix=f"L-shape Initial Density (iter={iteration})",
                )
                source_plot_path = os.path.join(
                    params.result_dir, f"l_shape_source_iter{iteration}.png"
                )
                plot_l_shape_true_source_maps(
                    l_shape_op,
                    output_path=source_plot_path,
                    title_prefix=f"L-shape True Source (iter={iteration})",
                )
                macro_source_plot_path = os.path.join(
                    params.result_dir, f"l_shape_macro_source_iter{iteration}.png"
                )
                plot_l_shape_macro_source_maps(
                    l_shape_op,
                    output_path=macro_source_plot_path,
                    title_prefix=f"L-shape Macro Source (iter={iteration})",
                )
                potential_plot_path = os.path.join(
                    params.result_dir, f"l_shape_potential_iter{iteration}.png"
                )
                plot_l_shape_electric_potential_map(
                    l_shape_op,
                    output_path=potential_plot_path,
                    title_prefix=f"L-shape Electric Potential (iter={iteration})",
                )
                if getattr(l_shape_op, "soft_l_assignment", False) and \
                   'soft_l_weights' in l_shape_op.cached_segments:
                    soft_plot_path = os.path.join(
                        params.result_dir, f"l_shape_soft_iter{iteration}.png"
                    )
                    plot_soft_l_intermediate(
                        l_shape_op.cached_segments,
                        output_path=soft_plot_path,
                    )
                    if getattr(l_shape_op, "cached_soft_debug", None) is not None:
                        soft_scoring_path = os.path.join(
                            params.result_dir,
                            f"l_shape_soft_scoring_iter{iteration}.png",
                        )
                        plot_soft_l_scoring_maps(
                            l_shape_op.cached_soft_debug,
                            output_path=soft_scoring_path,
                            title_prefix=f"Soft L Scoring (iter={iteration})",
                        )
        except Exception as e:
            logging.warning(f"Failed to plot L-shape density/segments: {e}")


