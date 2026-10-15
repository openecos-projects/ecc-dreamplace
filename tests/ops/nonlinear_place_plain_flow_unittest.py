"""Native placement-loop regressions without ECC or a private PDK."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dreamplace import configure
from dreamplace.macroPlaceDB import MacroPlaceDB
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.Params import Params


def make_placement_database(params):
    writeback = SimpleNamespace(write_placement_back=lambda *args: None)
    db = MacroPlaceDB(writeback)
    db.params = params
    db.write_placement_back = lambda *args, **kwargs: None
    db.ecc_db = object()
    db.pydb = SimpleNamespace(write_macro_placement_back=lambda *args: 1)
    db.dtype = np.float32
    db.dbu = 1.0
    db.num_physical_nodes = 6
    db.num_terminals = 1
    db.num_terminal_NIs = 0
    db.node_names = np.array([b"U0", b"U1", b"U2", b"U3", b"MOVABLE_MEM", b"FIXED_MEM"])
    db.node_x = np.array([1, 13, 1, 13, 3, 7], dtype=np.float32)
    db.node_y = np.array([1, 1, 12, 12, 3, 7], dtype=np.float32)
    db.node_size_x = np.array([1, 1, 1, 1, 2, 2], dtype=np.float32)
    db.node_size_y = db.node_size_x.copy()
    db.node_orient = np.array([b"N"] * 6)
    db.node_is_hard_macro = np.array([False] * 4 + [True, True])
    db.macro_writeback_candidate = np.array([False] * 4 + [True, False])
    db.net_names = np.array([b"N0", b"N1", b"N2", b"N3"])
    db.net_weights = np.ones(4, dtype=np.float32)
    db.pin2node_map = np.array([0, 4, 4, 1, 2, 5, 5, 3], dtype=np.int32)
    db.pin2net_map = np.repeat(np.arange(4, dtype=np.int32), 2)
    db.pin_names = np.array([b"Y", b"A"] * 4)
    db.pin_direct = np.array([b"OUTPUT", b"INPUT"] * 4)
    db.pin_offset_x = np.array([0.8, 0.2, 1.8, 0.2, 0.8, 0.2, 1.8, 0.2], dtype=np.float32)
    db.pin_offset_y = np.array([0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.5], dtype=np.float32)
    db.net2pin_map = [np.array([i, i + 1], dtype=np.int32) for i in range(0, 8, 2)]
    db.flat_net2pin_map = np.arange(8, dtype=np.int32)
    db.flat_net2pin_start_map = np.arange(0, 9, 2, dtype=np.int32)
    db.node2pin_map = [np.flatnonzero(db.pin2node_map == i).astype(np.int32) for i in range(6)]
    db.flat_node2pin_map = np.concatenate(db.node2pin_map)
    db.flat_node2pin_start_map = np.array([0, 1, 2, 3, 4, 6, 8], dtype=np.int32)
    db.regions = []
    db.flat_region_boxes = np.empty((0, 4), dtype=np.float32)
    db.flat_region_boxes_start = np.zeros(1, dtype=np.int32)
    db.node2fence_region_map = np.full(6, np.iinfo(np.int32).max, dtype=np.int32)
    db.rows = np.array([[0, y, 16, y + 1] for y in range(16)], dtype=np.float32)
    db.xl = db.yl = 0.0
    db.xh = db.yh = 16.0
    db.site_width = db.row_height = 1.0
    db.total_fixed_node_area = 4.0
    db.total_space_area = 252.0
    db.routing_grid_xl = db.routing_grid_yl = 0.0
    db.routing_grid_xh = db.routing_grid_yh = 16.0
    db.num_routing_grids_x = db.num_routing_grids_y = 8
    db.num_routing_layers = 2
    db.unit_horizontal_capacity = db.unit_vertical_capacity = 1.0
    db.unit_horizontal_capacities = np.array([1, 0], dtype=np.float32)
    db.unit_vertical_capacities = np.array([0, 1], dtype=np.float32)
    db.min_wire_widths = db.min_wire_spacings = np.array([0.2, 0.2], dtype=np.float32)
    db.initialize(params)
    return db


class PlainPlacementFlowTest(unittest.TestCase):
    def test_auto_bins_use_site_aligned_padded_geometry(self):
        params = Params()
        schema = json.loads((Path(configure.__file__).parent / "params.json").read_text())
        params.fromJson({key: value["default"] for key, value in schema.items()})
        params.aux_input = ""
        params.pin_density = 0
        params.macro_halo_x = params.macro_halo_y = 0
        params.auto_adjust_bins = 1
        params.target_density = 0.2

        with tempfile.TemporaryDirectory(prefix="auto-placement-bins-") as directory:
            params.result_dir = directory
            params.cell_padding_x = 0
            unpadded = make_placement_database(params)
            params.cell_padding_x = 1
            padded = make_placement_database(params)

        self.assertEqual((unpadded.num_bins_x, unpadded.num_bins_y), (4, 4))
        self.assertEqual((padded.num_bins_x, padded.num_bins_y), (2, 2))
        self.assertGreater(padded.total_movable_node_area, unpadded.total_movable_node_area)

    def _run(self, gpu, macro_only):
        params = Params()
        schema = json.loads((Path(configure.__file__).parent / "params.json").read_text())
        params.fromJson({key: value["default"] for key, value in schema.items()})
        params.gpu = gpu
        params.num_threads = 1
        params.num_bins_x = params.num_bins_y = 8
        params.auto_adjust_bins = 0
        params.global_place_stages = [
            {
                "iteration": 3,
                "num_bins_x": 8,
                "num_bins_y": 8,
                "learning_rate": 1.0,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }
        ]
        params.aux_input = ""
        params.routability_opt_flag = 0
        params.get_congestion_map = 0
        params.with_sta = False
        params.timing_opt_flag = 0
        params.plot_flag = 0
        params.pin_density = 0
        params.cell_padding_x = 0
        params.macro_halo_x = params.macro_halo_y = 0
        params.macro_only = macro_only
        params.macro_place_flag = macro_only
        params.two_stage_flag = 0
        with tempfile.TemporaryDirectory(prefix="plain-placement-loop-") as directory:
            params.result_dir = directory
            db = make_placement_database(params)
            engine = NonLinearPlace(params, db, timer=None)
            rsmt, hpwl, metrics = engine(params, db)
            self.assertTrue(math.isfinite(float(hpwl)))
            self.assertTrue(math.isfinite(float(rsmt)))
            self.assertTrue(metrics)
            self.assertTrue(engine.op_collections.legality_check_op(engine.pos[0]))

    def test_cpu_standard_placement_without_routability(self):
        self._run(0, 0)

    def test_cpu_macro_placement_without_routability(self):
        self._run(0, 1)

    @unittest.skipUnless(
        torch.cuda.is_available() and configure.compile_configurations["CUDA_FOUND"] == "TRUE",
        "CUDA placement runtime is unavailable",
    )
    def test_cuda_macro_placement_without_routability(self):
        self._run(1, 1)


if __name__ == "__main__":
    unittest.main()
