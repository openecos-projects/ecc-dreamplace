#!/usr/bin/python3

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTODMP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo  # noqa: E402
from dreamplace.ops.routability.l_shape_segment import (  # noqa: E402
    LShapeSegmentBuilder,
)


def tiny_pack():
    return {
        "pin_relate_x": torch.tensor([0, 1, 2, 1], dtype=torch.int32),
        "pin_relate_y": torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        "net_vertex_start": torch.tensor([0, 4], dtype=torch.int32),
        "net_steiner_start": torch.tensor([3, 4], dtype=torch.int32),
        "pin_fa": torch.tensor([-1, 3, -1, 0], dtype=torch.int32),
        "flat_pin_to": torch.tensor([3, 1], dtype=torch.int32),
        "flat_pin_from": torch.tensor([0, 3], dtype=torch.int32),
        "flat_pin_to_start": torch.tensor([0, 1, 1, 1, 2], dtype=torch.int32),
        "net_flat_topo_sort": torch.tensor([0, 3, 1, 2], dtype=torch.int32),
        "net_flat_topo_sort_start": torch.tensor([0, 4], dtype=torch.int32),
        "edge_l_directions": torch.tensor([2, 2], dtype=torch.int32),
        "metadata": {
            "schema_version": 1,
            "num_pins": 3,
            "num_vertices": 4,
            "num_edges": 2,
            "num_nets": 1,
        },
    }


class TestGGRLShapeTopologyPack(unittest.TestCase):
    def make_topo(self):
        return SteinerTopo(
            flat_net2pin_map=torch.tensor([0, 1, 2], dtype=torch.int32),
            flat_net2pin_start_map=torch.tensor([0, 3], dtype=torch.int32),
            deterministic_flag=True,
        )

    def test_param_validation_rejects_incompatible_modes(self):
        from dreamplace.ops.steiner_topo.ggr_l_shape_topology import (
            validate_ggr_l_shape_topology_params,
        )

        validate_ggr_l_shape_topology_params(
            SimpleNamespace(
                l_shape_use_ggr_topology=1,
                l_direction_use_gpugr=1,
                soft_l_assignment=0,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "l_direction_use_gpugr"):
            validate_ggr_l_shape_topology_params(
                SimpleNamespace(
                    l_shape_use_ggr_topology=1,
                    l_direction_use_gpugr=0,
                    soft_l_assignment=0,
                )
            )

        with self.assertRaisesRegex(RuntimeError, "soft_l_assignment"):
            validate_ggr_l_shape_topology_params(
                SimpleNamespace(
                    l_shape_use_ggr_topology=1,
                    l_direction_use_gpugr=1,
                    soft_l_assignment=1,
                )
            )

    def test_loader_rejects_missing_fields_and_invalid_directions(self):
        topo = self.make_topo()
        pos = torch.tensor([0.0, 10.0, 20.0, 0.0, 10.0, 0.0], dtype=torch.float32)

        missing = tiny_pack()
        missing.pop("flat_pin_from")
        with self.assertRaisesRegex(RuntimeError, "missing field.*flat_pin_from"):
            topo.load_ggr_topology_pack(missing, pos)

        invalid_direction = tiny_pack()
        invalid_direction["edge_l_directions"] = torch.tensor([2, -1], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "invalid direction"):
            topo.load_ggr_topology_pack(invalid_direction, pos)

    def test_loader_keeps_segment_gradients_connected_to_pin_pos(self):
        topo = self.make_topo()
        pos = torch.tensor(
            [0.0, 10.0, 20.0, 0.0, 10.0, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        topo.load_ggr_topology_pack(tiny_pack(), pos)
        newx, newy = topo(pos)
        self.assertTrue(newx.requires_grad)
        self.assertTrue(newy.requires_grad)
        self.assertAlmostEqual(float(newx[3].item()), 10.0)
        self.assertAlmostEqual(float(newy[3].item()), 0.0)

        segments = LShapeSegmentBuilder(wire_width=0.0).build_segments(
            newx,
            newy,
            topo.flat_pin_from,
            topo.flat_pin_to,
            topo.edge_l_directions,
        )
        cost = segments["segment_size_x"].sum() + segments["segment_size_y"].sum()
        cost.backward()

        self.assertIsNotNone(pos.grad)
        self.assertLess(float(pos.grad[0].item()), 0.0)
        self.assertGreater(float(pos.grad[1].item()), 0.0)
        self.assertLess(float(pos.grad[3].item()), 0.0)
        self.assertGreater(float(pos.grad[4].item()), 0.0)

    def test_ggr_loader_skips_redundant_sanitize(self):
        topo = self.make_topo()
        pos = torch.tensor([0.0, 10.0, 20.0, 0.0, 10.0, 0.0], dtype=torch.float32)

        with mock.patch.object(topo, "_sanitize_pin_relate_indices") as sanitize:
            topo.load_ggr_topology_pack(tiny_pack(), pos)

        sanitize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
