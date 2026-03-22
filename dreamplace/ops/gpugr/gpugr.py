import os
import logging

import torch
import torch.nn.functional as F

from tools.iEDA.module.gpugr import IEDAGPUGR


logger = logging.getLogger(__name__)


class GPUGR(object):
    def __init__(self, params, placedb):
        self.params = params
        self.placedb = placedb
        self._gpugr_op = None
        self.last_result = None
        self.last_metrics = None
        self.last_route_grid = None

    def __call__(self, pos):
        return self.forward(pos)

    def _get_gpugr_op(self):
        if self._gpugr_op is None:
            self._gpugr_op = IEDAGPUGR(dir_workspace=self.placedb.data_manager.dir_workspace)
        return self._gpugr_op

    def _write_back_pos_to_ieda(self, pos):
        if pos.is_cuda:
            pos_cpu = pos.detach().cpu().numpy().copy()
        else:
            pos_cpu = pos.detach().numpy().copy()

        node_x = pos_cpu[: self.placedb.num_movable_nodes]
        node_y = pos_cpu[
            self.placedb.num_nodes : self.placedb.num_nodes + self.placedb.num_movable_nodes
        ]
        if self.params.cell_padding_x >= 0:
            node_x += self.params.cell_padding_x

        unscale_factor = 1.0 / self.params.scale_factor
        node_x = node_x * unscale_factor + self.params.shift_factor[0]
        node_y = node_y * unscale_factor + self.params.shift_factor[1]
        self.placedb.write_placement_back(node_x, node_y)

    @staticmethod
    def _resample_xy_map(map_xy, target_x, target_y):
        if map_xy.shape == (target_x, target_y):
            return map_xy.contiguous()
        image_yx = map_xy.t().unsqueeze(0).unsqueeze(0)
        image_yx = F.interpolate(
            image_yx,
            size=(target_y, target_x),
            mode="bilinear",
            align_corners=False,
        )
        return image_yx.squeeze(0).squeeze(0).t().contiguous()

    def forward(self, pos):
        route_xsize = int(self.placedb.num_routing_grids_x)
        route_ysize = int(self.placedb.num_routing_grids_y)
        if route_xsize <= 0 or route_ysize <= 0:
            raise RuntimeError(
                f"Invalid gpugr routing grid ({route_xsize}, {route_ysize}). "
                "Sync AutoDMP route_num_bins_x/y before calling the gpugr congestion map op."
            )

        self._write_back_pos_to_ieda(pos)
        gpugr_op = self._get_gpugr_op()
        result_dir = os.path.join(self.params.result_dir, "gpugr_area_adjust")
        result = gpugr_op.run_gpugr(
            out_dir=result_dir,
            design_name=self.params.design_name(),
            gpu=getattr(self.params, "gpu_id", 0),
            threads=self.params.num_threads,
            route_xsize=route_xsize,
            route_ysize=route_ysize,
            rrr_iters=int(getattr(self.params, "gpugr_area_adjust_rrr_iters", 0)),
            skip_m1_route=bool(getattr(self.params, "gpugr_area_adjust_skip_m1_route", 1)),
            verbose_parser_log=bool(getattr(self.params, "gpugr_area_adjust_verbose_parser_log", 0)),
            cpp_log_level=int(getattr(self.params, "gpugr_area_adjust_cpp_log_level", 2)),
            keep_temp_def=bool(getattr(self.params, "gpugr_area_adjust_keep_temp_def", 0)),
            save_artifacts=bool(getattr(self.params, "gpugr_area_adjust_save_artifacts", 0)),
        )
        self.last_result = result
        self.last_route_grid = (route_xsize, route_ysize)

        overflow_xy = result["maps"]["cg_map_union_overflow"].detach().to(
            device=pos.device,
            dtype=pos.dtype,
        )
        overflow_xy = self._resample_xy_map(overflow_xy, route_xsize, route_ysize)
        route_utilization_map = (overflow_xy + 1.0).contiguous()

        metrics = result["metrics"]
        self.last_metrics = dict(metrics)
        logger.info(
            "gpugr congestion map for inflation: grid=%dx%d ovfl_max=%.4f ovfl_mean=%.4f #OvflNets=%d EstShorts=%.0f",
            route_xsize,
            route_ysize,
            overflow_xy.max().item(),
            overflow_xy.mean().item(),
            metrics["num_overflow_nets"],
            metrics["gr_est_shorts"],
        )
        return route_utilization_map
