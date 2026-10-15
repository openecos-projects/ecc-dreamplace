import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dreamplace.ops.routability import l_shape_inputs


class LShapeInputsTest(unittest.TestCase):
    def test_gpugr_writeback_precedes_route_and_preserves_directional_maps(self):
        trace = []
        capacity = torch.tensor(
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ]
        )
        demand = torch.ones_like(capacity)
        maps = {
            "capacity_map": capacity,
            "raw_wire_demand_map": demand,
            "fix_usage_map": torch.zeros_like(capacity),
            "mov_usage_map": demand,
            "wire_demand_map": demand,
            "via_demand_map": torch.zeros_like(capacity),
        }
        metrics = {
            "cg_map_h_raw_max": 0.1,
            "cg_map_h_raw_top1pct_mean": 0.1,
            "cg_map_h_raw_overflow_bin_ratio": 0.0,
            "cg_map_v_raw_max": 0.1,
            "cg_map_v_raw_top1pct_mean": 0.1,
            "cg_map_v_raw_overflow_bin_ratio": 0.0,
            "num_overflow_nets": 0,
            "gr_est_shorts": 0,
        }

        class Backend:
            cache_hit = False

            def parser_cache_would_hit(self, **kwargs):
                trace.append("cache_check")
                return self.cache_hit

            def run_gpugr(self, **kwargs):
                trace.append("route")
                self.kwargs = kwargs
                if self.cache_hit:
                    kwargs["parser_cache_fallback_before_export"]()
                return {"maps": maps, "metrics": metrics}

        backend = Backend()
        params = SimpleNamespace(
            result_dir="/tmp",
            design_name=lambda: "TINY",
            route_num_bins_x=2,
            route_num_bins_y=2,
            auto_adjust_bins=0,
            num_bins_x=2,
            num_bins_y=2,
            num_threads=1,
            gpugr_parser_cache_enable=False,
        )
        placedb = SimpleNamespace(
            xl=0.0,
            yl=0.0,
            xh=2.0,
            yh=2.0,
            num_bins_x=2,
            num_bins_y=2,
            num_nodes=1,
            num_movable_nodes=1,
            net_names=[b"N1"],
            pin_names=[b"U1:A"],
            flat_net2pin_map=np.asarray([0], dtype=np.int32),
            flat_net2pin_start_map=np.asarray([0, 1], dtype=np.int32),
            num_nets=1,
            unit_horizontal_capacities=[1.0, 0.0],
            unit_vertical_capacities=[0.0, 1.0],
            min_wire_widths=[0.1, 0.1],
            write_placement_back=lambda x, y, **kwargs: trace.append("writeback"),
        )
        pos = torch.tensor([0.0, 0.0])
        params.cell_padding_x = -1
        params.place_io_engine = "ecc"
        params.scale_factor = 1.0
        params.shift_factor = (0.0, 0.0)

        with mock.patch.object(l_shape_inputs, "get_cached_gpugr_operator", return_value=backend):
            result = l_shape_inputs.prepare_l_shape_inputs_from_gpugr(
                params, placedb, pos
            )

        self.assertEqual(trace, ["writeback", "route"])
        self.assertEqual((backend.kwargs["route_xsize"], backend.kwargs["route_ysize"]), (2, 2))
        self.assertEqual(backend.kwargs["topology_net_name_to_id"], {"N1": 0})
        self.assertEqual(backend.kwargs["topology_pin_name_to_id"], {"U1:A": 0})
        self.assertTrue(backend.kwargs["keep_temp_def"])
        torch.testing.assert_close(result["supply_map"], capacity.sum(dim=0))
        torch.testing.assert_close(result["supply_map_h"], capacity[0])
        torch.testing.assert_close(result["supply_map_v"], capacity[1])
        torch.testing.assert_close(result["demand_map"], demand.sum(dim=0))

        trace.clear()
        backend.cache_hit = True
        params.gpugr_parser_cache_enable = True
        placedb.node_names = [b"U1"]
        with mock.patch.object(l_shape_inputs, "get_cached_gpugr_operator", return_value=backend):
            l_shape_inputs.prepare_l_shape_inputs_from_gpugr(params, placedb, pos)

        self.assertEqual(trace, ["cache_check", "route", "writeback"])
        self.assertEqual(backend.kwargs["parser_cache_node_names"], ("U1",))


if __name__ == "__main__":
    unittest.main()
