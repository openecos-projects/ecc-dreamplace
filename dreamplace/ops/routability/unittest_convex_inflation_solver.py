#!/usr/bin/python3

import importlib.util
import os
import unittest

import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(CURRENT_DIR, "convex_inflation_solver.py")
SPEC = importlib.util.spec_from_file_location("convex_inflation_solver", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load convex_inflation_solver.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakePlaceDB(object):
    def __init__(self):
        self.routing_grid_xl = 0.0
        self.routing_grid_yl = 0.0
        self.routing_grid_xh = 4.0
        self.routing_grid_yh = 1.0
        self.num_routing_grids_x = 2
        self.num_routing_grids_y = 1
        self.num_movable_nodes = 3
        self.num_nodes = 3
        self.num_nets = 1
        self.flat_net2pin_map = np.array([0, 1], dtype=np.int64)
        self.flat_net2pin_start_map = np.array([0, 2], dtype=np.int64)
        self.pin2node_map = np.array([0, 1], dtype=np.int64)
        self.net_weights = np.array([1.0], dtype=np.float64)


class TestConvexInflationSolver(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.placedb = FakePlaceDB()
        self.pos = torch.tensor(
            [
                0.0,
                0.8,
                2.5,
                0.0,
                0.0,
                0.0,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.node_size_x = torch.tensor([0.8, 0.8, 0.8], dtype=self.dtype, device=self.device)
        self.node_size_y = torch.tensor([1.0, 1.0, 1.0], dtype=self.dtype, device=self.device)
        self.cluster_ids = torch.tensor([0, 0, 1], dtype=torch.int64, device=self.device)

    def test_collect_active_bins(self):
        demand_map = torch.tensor([[2.0], [0.5]], dtype=self.dtype, device=self.device)
        capacity_map = torch.tensor([[1.0], [1.0]], dtype=self.dtype, device=self.device)
        active_bins = MODULE.collect_active_bins(demand_map, capacity_map)
        self.assertEqual(int(active_bins.flat_bin_ids.numel()), 1)
        self.assertTrue(torch.allclose(active_bins.demand_values, torch.tensor([2.0], dtype=self.dtype)))

    def test_build_records_and_cluster_wirelength(self):
        demand_map = torch.tensor([[2.0], [0.5]], dtype=self.dtype, device=self.device)
        capacity_map = torch.tensor([[1.0], [1.0]], dtype=self.dtype, device=self.device)
        active_bins = MODULE.collect_active_bins(demand_map, capacity_map)
        records = MODULE.build_cluster_demand_records(
            placedb=self.placedb,
            pos=self.pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            cluster_ids=self.cluster_ids,
            active_bins=active_bins,
        )
        self.assertEqual(records.num_clusters, 2)
        self.assertEqual(records.num_bins, 1)
        self.assertEqual(int(records.demand_values.numel()), 1)
        self.assertTrue(torch.allclose(records.demand_values, torch.tensor([2.0], dtype=self.dtype)))

        cluster_wl = MODULE.compute_cluster_wirelength(
            placedb=self.placedb,
            pos=self.pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            cluster_ids=self.cluster_ids,
        )
        self.assertEqual(tuple(cluster_wl.shape), (2,))
        self.assertGreater(float(cluster_wl[0].item()), 0.0)
        self.assertEqual(float(cluster_wl[1].item()), 0.0)

    def test_solver_converges_on_simple_case(self):
        records = MODULE.ClusterDemandRecords(
            cluster_ids=torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            bin_ids=torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            demand_values=torch.tensor([2.0, 0.5], dtype=self.dtype, device=self.device),
            num_clusters=2,
            num_bins=2,
        )
        result = MODULE.solve_convex_inflation_sparse(
            cluster_demand_records=records,
            active_bin_capacity=torch.tensor([1.0, 1.0], dtype=self.dtype, device=self.device),
            cluster_wirelength=torch.tensor([1.0, 1.0], dtype=self.dtype, device=self.device),
            max_inflation=4.0,
            max_iters=80,
            rho_init=1.0,
            tolerance=1e-5,
        )
        self.assertTrue(result.converged)
        self.assertFalse(result.fallback_used)
        self.assertLessEqual(result.max_violation, 1e-4)
        self.assertAlmostEqual(float(result.cluster_inflation[0].item()), 2.0, places=2)
        self.assertAlmostEqual(float(result.cluster_inflation[1].item()), 1.0, places=2)

    def test_local_correction_uses_safe_remain(self):
        corrected = MODULE.local_gcell_correction(
            prev_inflation_map=torch.tensor([[1.2]], dtype=self.dtype, device=self.device),
            local_demand_map=torch.tensor([[1.5]], dtype=self.dtype, device=self.device),
            global_demand_map=torch.tensor([[1.4]], dtype=self.dtype, device=self.device),
            capacity_map=torch.tensor([[1.0]], dtype=self.dtype, device=self.device),
            gamma=0.2,
            max_inflation=4.0,
            eps=1e-6,
        )
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertGreaterEqual(float(corrected.item()), 1.0)
        self.assertLessEqual(float(corrected.item()), 4.0)

    def test_run_convex_inflation_end_to_end(self):
        class Params(object):
            modularity_active_bin_overflow_eps = 1e-6
            modularity_max_inflation = 4.0
            modularity_convex_max_iters = 80
            modularity_convex_rho_init = 1.0

        result = MODULE.run_convex_inflation(
            placedb=self.placedb,
            params=Params(),
            pos=self.pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            cluster_ids=self.cluster_ids,
            demand_map=torch.tensor([[2.0], [0.5]], dtype=self.dtype, device=self.device),
            capacity_map=torch.tensor([[1.0], [1.0]], dtype=self.dtype, device=self.device),
        )
        self.assertEqual(int(result.node_inflation.numel()), 3)
        self.assertGreater(float(result.node_inflation[0].item()), 1.0)
        self.assertGreater(float(result.node_inflation[1].item()), 1.0)
        self.assertAlmostEqual(float(result.node_inflation[2].item()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
