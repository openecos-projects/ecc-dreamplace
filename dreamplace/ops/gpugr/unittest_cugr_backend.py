import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dreamplace.ops.gpugr.cugr_backend import CugrGPUGR
from dreamplace.ops.gpugr.result_maps import validate_map_contract, validate_map_planes


def _native_topology():
    # The same two-pin net is represented by two layer trees. The repeated
    # pin access must collapse to one ECC pin vertex rather than create two
    # parents for that vertex.
    return {
        "routing_layer_begin": 0,
        "routing_layer_end": 2,
        "nets": [
            {
                "name": "n0",
                "routed": True,
                "pin_access": [(0, 4, 2), (0, 2, 3)],
                "nodes": [
                    (2, 2, 3, -1, -1),
                    (1, 2, 3, -1, 0),
                    (0, 2, 3, 1, 1),
                    (2, 4, 3, -1, 0),
                    (1, 4, 3, -1, 3),
                    (1, 4, 2, -1, 4),
                    (0, 4, 2, 1, 5),
                ],
            }
        ],
    }


class CugrBackendTest(unittest.TestCase):
    def test_qualification_rejects_dirty_native_source(self):
        operator = CugrGPUGR(
            SimpleNamespace(cugr_require_clean_source=True),
            None,
        )
        operator._import_cugr = lambda: SimpleNamespace(
            source_dirty=lambda: True,
        )
        with self.assertRaisesRegex(RuntimeError, "source metadata is dirty"):
            operator.run_gpugr(
                route_xsize=8,
                route_ysize=8,
                threads=1,
                rrr_iters=0,
                backend="cugr",
            )

    def test_l_shape_pack_collapses_repeated_layer_pin_access(self):
        native = _native_topology()
        pack = CugrGPUGR._l_shape_pack(
            native,
            flat_net2pin=[0, 1],
            flat_net2pin_start=[0, 2],
            num_pins=2,
            num_nets=1,
            net_name_to_id={"n0": 0},
        )

        self.assertEqual(pack["metadata"]["num_pins"], 2)
        self.assertEqual(pack["metadata"]["num_vertices"], len(pack["pin_fa"]))
        self.assertEqual(pack["metadata"]["num_edges"], len(pack["flat_pin_from"]))
        self.assertEqual(pack["net_steiner_start"].tolist()[0], 2)
        self.assertEqual(pack["net_steiner_start"].tolist()[-1], pack["metadata"]["num_vertices"])
        self.assertEqual(pack["flat_pin_to_start"].tolist()[-1], pack["metadata"]["num_edges"])
        self.assertEqual(pack["net_vertex_start"].tolist(), [0, 2])
        self.assertEqual(pack["pin_relate_x"].tolist()[:2], [0, 1])
        self.assertEqual(pack["pin_relate_y"].tolist()[:2], [0, 1])

    def test_l_shape_pack_rejects_missing_native_net(self):
        with self.assertRaisesRegex(RuntimeError, "missing ECC nets"):
            CugrGPUGR._l_shape_pack(
                {"nets": [{"name": "n0", "routed": False, "nodes": []}]},
                flat_net2pin=[0, 1, 2, 3],
                flat_net2pin_start=[0, 2, 4],
                num_pins=4,
                num_nets=2,
                net_name_to_id={"n0": 0, "n1": 1},
            )

    def test_l_shape_pack_preserves_original_net_vertex_offsets(self):
        native = {
            "nets": [
                {"name": "n0", "routed": False, "nodes": []},
                {"name": "n1", "routed": False, "nodes": []},
            ]
        }
        pack = CugrGPUGR._l_shape_pack(
            native,
            flat_net2pin=[0, 1, 2, 3],
            flat_net2pin_start=[0, 2, 4],
            num_pins=4,
            num_nets=2,
            net_name_to_id={"n0": 0, "n1": 1},
        )
        self.assertEqual(pack["net_vertex_start"].tolist(), [0, 2, 4])
        self.assertEqual(pack["net_steiner_start"].tolist(), [4, 4, 4])

    def test_l_shape_pack_covers_multipin_duplicate_location_io_and_unroutable(self):
        native = {
            "nets": [
                {
                    "name": "multipin",
                    "routed": True,
                    "pin_access": [(0, 10, 10), (0, 12, 10), (0, 10, 12)],
                    "nodes": [
                        (0, 10, 10, 0, -1),
                        (0, 12, 10, 1, 0),
                        (0, 10, 12, 2, 0),
                    ],
                },
                {
                    "name": "duplicate_io",
                    "routed": False,
                    "pin_access": [(-1, 20, 20), (-1, 20, 20)],
                    "nodes": [],
                },
            ]
        }
        pack = CugrGPUGR._l_shape_pack(
            native,
            flat_net2pin=[0, 1, 2, 3, 4],
            flat_net2pin_start=[0, 3, 5],
            num_pins=5,
            num_nets=2,
            net_name_to_id={"multipin": 0, "duplicate_io": 1},
        )

        self.assertEqual(pack["metadata"]["route_failed_count"], 1)
        self.assertEqual(pack["metadata"]["num_edges"], 2)
        self.assertEqual(pack["net_vertex_start"].tolist(), [0, 3, 5])
        self.assertEqual(pack["pin_relate_x"].tolist()[:5], [0, 1, 2, 3, 4])
        self.assertEqual(pack["pin_relate_y"].tolist()[:5], [0, 1, 2, 3, 4])

    def test_route_entries_preserve_explicit_failed_reason(self):
        entries = CugrGPUGR._route_entries(
            {
                "nets": [
                    {
                        "name": "failed_net",
                        "routed": False,
                        "route_failed": True,
                        "route_failed_reason": "pin has no legal route-grid access",
                        "entries": [],
                    }
                ]
            },
            {"failed_net": 0},
        )

        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["route_failed"])
        self.assertEqual(entries[0]["route_failed_reason"], "pin has no legal route-grid access")

    def test_route_validation_rejects_diagonal_and_out_of_grid_entries(self):
        native = {
            "num_layers": 2,
            "grid_x": 4,
            "grid_y": 4,
            "routing_layer_begin": 0,
            "routing_layer_end": 1,
            "nets": [],
        }
        with self.assertRaisesRegex(RuntimeError, "diagonal"):
            CugrGPUGR._validate_route_layer_window(
                native,
                [{
                    "net_name": "n0",
                    "entries": [{
                        "type": "wire",
                        "layer_idx": 0,
                        "grid_x1": 0,
                        "grid_y1": 0,
                        "grid_x2": 1,
                        "grid_y2": 1,
                    }],
                }],
            )
        with self.assertRaisesRegex(RuntimeError, "outside the route grid"):
            CugrGPUGR._validate_route_layer_window(
                native,
                [{
                    "net_name": "n0",
                    "entries": [{
                        "type": "wire",
                        "layer_idx": 0,
                        "grid_x1": 0,
                        "grid_y1": 0,
                        "grid_x2": 4,
                        "grid_y2": 0,
                    }],
                }],
            )

    def test_primitive_maps_zero_layers_outside_window(self):
        native = {
            "num_layers": 3,
            "grid_x": 2,
            "grid_y": 2,
            "layer_directions": [0, 1, 0],
            "layer_edge_offsets": [0, 2, 4, 6],
            "capacity": [1, 1, 2, 2, 3, 3],
            "fixed_usage": [0, 0, 0, 0, 0, 0],
            "movable_usage": [0, 0, 1, 1, 0, 0],
            "wire_usage": [1, 1, 1, 1, 1, 1],
            "via_usage": np.ones(12, dtype=np.float32),
            "routing_layer_begin": 1,
            "routing_layer_end": 1,
        }
        maps, _ = CugrGPUGR._primitive_maps(native, skip_m1_route=False)
        self.assertTrue(torch.equal(maps["capacity_map"][0], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(maps["capacity_map"][2], torch.zeros((2, 2))))
        self.assertEqual(float(maps["capacity_map"][1].sum()), 4.0)
        self.assertEqual(float(maps["mov_usage_map"][1].sum()), 2.0)
        self.assertEqual(float(maps["wire_demand_map"][1].sum()), 4.0)
        self.assertTrue(torch.isfinite(maps["cg_map_union_raw"]).all())

    def test_map_contract_rejects_missing_movable_plane(self):
        with self.assertRaisesRegex(RuntimeError, "movable_usage"):
            CugrGPUGR._primitive_maps(
                {
                    "num_layers": 1,
                    "grid_x": 2,
                    "grid_y": 2,
                    "layer_directions": [0],
                    "layer_edge_offsets": [0, 2],
                    "capacity": [1, 1],
                    "fixed_usage": [0, 0],
                    "wire_usage": [0, 0],
                    "via_usage": [0, 0, 0, 0],
                    "routing_layer_begin": 0,
                    "routing_layer_end": 0,
                },
                skip_m1_route=False,
            )

    def test_shared_map_validator_accepts_normalized_planes(self):
        plane = torch.zeros((2, 2, 2), dtype=torch.float32)
        validate_map_planes(
            {
                "dmd_map": plane,
                "wire_demand_map": plane,
                "via_demand_map": plane,
                "fix_usage_map": plane,
                "mov_usage_map": plane,
                "capacity_map": plane,
            },
            expected_layers=2,
        )

    def test_map_contract_rejects_wrong_directional_shape(self):
        maps = {
            "dmd_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "wire_demand_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "via_demand_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "fix_usage_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "mov_usage_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "capacity_map": torch.zeros((2, 4, 4), dtype=torch.float32),
            "cg_map_h_raw": torch.zeros((2, 4, 4), dtype=torch.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "cg_map_h_raw"):
            validate_map_contract(maps, expected_layers=2)


if __name__ == "__main__":
    unittest.main()
