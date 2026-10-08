import unittest

from dreamplace.ops.buffer_insertion.real_design_adapter import (
    build_edge_rc_by_net_from_rc_timing_topology as legacy_build_edge_rc_by_net_from_rc_timing_topology,
)
from dreamplace.ops.steiner_topo import build_edge_rc_by_net_from_rc_timing_topology


class PydbTopologyTest(unittest.TestCase):
    def test_builds_edge_rc_from_rc_timing_topology_formula(self):
        topology = {
            "topology_source": "steiner",
            "net_ids": [10],
            "net_flat_topo_sort": [0, 1, 2],
            "net_flat_topo_sort_start": [0, 3],
            "pin_fa": [-1, 0, 1],
            "node_x": {0: 0, 1: 3000, 2: 3000},
            "node_y": {0: 0, 1: 0, 2: 4000},
        }

        edge_rc_by_net, summary = build_edge_rc_by_net_from_rc_timing_topology(
            topology,
            dbu=1000,
            scale_factor=1.0,
            r_unit=2.0,
            c_unit=0.25,
        )

        self.assertEqual(summary["rc_source"], "rc_timing_topology_formula")
        self.assertEqual(summary["edge_rc_map_net_count"], 1)
        self.assertEqual(summary["length_denominator"], 1000.0)
        self.assertEqual(
            edge_rc_by_net,
            {
                10: {
                    (0, 1): {"r": 6.0, "c": 0.75, "length_um": 3.0},
                    (1, 2): {"r": 8.0, "c": 1.0, "length_um": 4.0},
                }
            },
        )

    def test_builds_edge_rc_from_scaled_live_topology_coordinates(self):
        topology = {
            "topology_source": "live_timing_topology",
            "net_ids": [10],
            "net_flat_topo_sort": [0, 1],
            "net_flat_topo_sort_start": [0, 2],
            "pin_fa": [-1, 0],
            "node_x": {0: 0, 1: 54},
            "node_y": {0: 0, 1: 0},
        }

        edge_rc_by_net, summary = build_edge_rc_by_net_from_rc_timing_topology(
            topology,
            dbu=1000,
            scale_factor=1.0 / 54.0,
            r_unit=2.0,
            c_unit=0.25,
        )

        self.assertAlmostEqual(summary["length_denominator"], 1000.0 / 54.0)
        edge = edge_rc_by_net[10][(0, 1)]
        self.assertAlmostEqual(edge["length_um"], 2.916)
        self.assertAlmostEqual(edge["r"], 5.832)
        self.assertAlmostEqual(edge["c"], 0.729)

    def test_legacy_real_design_adapter_reexports_edge_rc_builder(self):
        self.assertIs(
            legacy_build_edge_rc_by_net_from_rc_timing_topology,
            build_edge_rc_by_net_from_rc_timing_topology,
        )


if __name__ == "__main__":
    unittest.main()
