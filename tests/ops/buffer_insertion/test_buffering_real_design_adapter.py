import unittest
from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.real_design_adapter import (
    _driver_rooted_live_topology_view,
    build_buffering_nets_from_pydb,
    build_local_steiner_topology_from_pydb,
)
from dreamplace.ops.steiner_topo import (
    build_local_steiner_topology_from_pydb as build_local_steiner_topology_from_pydb_canonical,
    build_timing_rooted_tree_view,
)


def _fake_pydb():
    return SimpleNamespace(
        dbu=1000,
        flat_net2pin_map=[2, 0, 1, 3, 4],
        flat_net2pin_start_map=[0, 3, 5],
        net2driver_pin_map=[0, 3],
        pin2node_map=[0, 1, 2, 3, 4],
        node_x=[10_000, 30_000, 50_000, 70_000, 90_000],
        node_y=[1_000, 1_000, 1_000, 2_000, 2_000],
        pin_offset_x=[100, 200, 300, 0, 0],
        pin_offset_y=[10, 20, 30, 0, 0],
        net_names=["net0", "net1"],
        pin_names=["U0/Y", "U1/A", "U2/A", "U3/Y", "U4/A"],
        num_movable_nodes=5,
        num_terminals=0,
        num_terminal_NIs=0,
        inst_main_id=[0, 1, 1, 0, 1],
        inst_libcell_offset=[0, 0, 0, 0, 0],
        main_id_2_cell_id_start=[0, 1],
        cell_id_2_libpin_id_start=[0, 1],
        pin_2_libpin_offset=[0, 0, 0, 0, 0],
        flat_lib_pin_cap=[0.0, 0.42],
    )


class DuRealDesignAdapterTest(unittest.TestCase):
    def test_clock_net_is_excluded_before_candidate_generation(self):
        pydb = _fake_pydb()
        pydb.clock_net_names = ["net0"]
        original_drivers = list(pydb.net2driver_pin_map)
        nets, summary = build_buffering_nets_from_pydb(pydb)
        self.assertEqual([net["net_id"] for net in nets], [1])
        self.assertEqual(summary["skipped_reasons"], {"clock_net": 1})
        self.assertEqual(pydb.net2driver_pin_map, original_drivers)

    def test_driver_rooted_live_topology_view_matches_legacy_contract(self):
        net_pin_ids = [2, 0, 1]
        directed_edges = [(0, 2), (0, 1)]
        coordinates = {0: (0, 0), 1: (10, 0), 2: (20, 0)}
        legacy = build_timing_rooted_tree_view(
            net_id=0,
            net_pin_ids=net_pin_ids,
            driver_pin_id=0,
            undirected_edges=directed_edges,
            coordinates=coordinates,
            flat_first_pin_id=2,
        )
        direct = _driver_rooted_live_topology_view(
            net_id=0,
            net_pin_ids=net_pin_ids,
            driver_pin_id=0,
            directed_edges=directed_edges,
            coordinates=coordinates,
            flat_first_pin_id=2,
        )

        self.assertIsNotNone(direct)
        for key in (
            "status",
            "root_node_id",
            "flat_first_pin_id",
            "driver_is_flat_first_pin",
            "reroot_applied",
            "sink_pin_ids",
            "node_ids",
            "edge_pairs",
            "children_by_node",
            "coordinates",
            "is_connected",
            "has_cycle",
        ):
            self.assertEqual(direct[key], legacy[key])

    def test_live_topology_driver_root_uses_direct_tree_view(self):
        topology = {
            "topology_source": "live_timing_topology",
            "net_ids": [0],
            "net_flat_topo_sort": [0, 1, 2],
            "net_flat_topo_sort_start": [0, 3],
            "pin_fa": [-1, 0, 0],
            "node_x": {0: 0, 1: 1000, 2: 2000},
            "node_y": {0: 0, 1: 0, 2: 0},
        }
        runtime_profile = {}

        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
            edge_rc_by_net={
                0: {
                    (0, 1): {"r": 1.25, "c": 0.5},
                    (0, 2): {"r": 2.5, "c": 0.25},
                }
            },
            runtime_profile=runtime_profile,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(runtime_profile["adapter_topology_direct_rooted_view_count"], 1)
        self.assertNotIn("adapter_topology_legacy_reroot_count", runtime_profile)
        self.assertEqual(
            nets[0]["rc_tree"]["children_by_node"],
            {0: [1, 2], 1: [], 2: []},
        )

    def test_runtime_profile_accounts_adapter_stages(self):
        runtime_profile = {}

        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            topology_source="flat_net_star",
            runtime_profile=runtime_profile,
        )

        self.assertEqual(summary["built_net_count"], 2)
        self.assertEqual(runtime_profile["adapter_processed_net_count"], 2)
        self.assertEqual(runtime_profile["adapter_built_net_count"], 2)
        self.assertGreaterEqual(
            runtime_profile["adapter_total_ms"],
            runtime_profile["adapter_accounted_ms"],
        )
        self.assertGreaterEqual(runtime_profile["adapter_unattributed_ms"], 0.0)
        self.assertGreaterEqual(
            runtime_profile["adapter_net_topology_and_coordinates_ms"],
            0.0,
        )
        self.assertGreaterEqual(
            runtime_profile["adapter_net_node_cap_build_ms"],
            0.0,
        )
        self.assertGreaterEqual(
            runtime_profile["adapter_net_rc_tree_and_record_build_ms"],
            0.0,
        )
        self.assertGreaterEqual(
            runtime_profile["adapter_net_fallback_star_setup_ms"],
            0.0,
        )
        self.assertEqual(len(nets), 2)

    def test_builds_timing_rooted_star_nets_from_pydb(self):
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["topology_source"], "flat_net_star")
        self.assertEqual(summary["built_net_count"], 1)
        self.assertEqual(nets[0]["net_id"], 0)
        self.assertEqual(nets[0]["net_name"], "net0")
        self.assertEqual(nets[0]["net_pin_ids"], [2, 0, 1])
        self.assertEqual(nets[0]["driver_pin_id"], 0)
        self.assertEqual(nets[0]["driver_pin_name"], "U0/Y")
        self.assertEqual(
            nets[0]["pin_name_by_id"],
            {2: "U2/A", 0: "U0/Y", 1: "U1/A"},
        )
        self.assertEqual(nets[0]["flat_first_pin_id"], 2)
        self.assertEqual(set(nets[0]["undirected_edges"]), {(0, 2), (0, 1)})
        self.assertEqual(nets[0]["coordinates"][0], (10_100, 1_010))
        self.assertEqual(nets[0]["coordinates"][1], (30_200, 1_020))
        self.assertEqual(nets[0]["coordinates"][2], (50_300, 1_030))
        self.assertEqual(nets[0]["sink_slack_by_pin"], {2: 0.0, 1: 0.0})
        self.assertEqual(nets[0]["npath_by_pin"], {2: 1, 1: 1})
        self.assertEqual(nets[0]["rc_tree"]["children_by_node"], {0: [2, 1], 2: [], 1: []})

    def test_default_net_limits_do_not_filter_real_design_nets(self):
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            topology_source="flat_net_star",
        )

        self.assertEqual(summary["status"], "ok")
        self.assertIsNone(summary["max_nets"])
        self.assertIsNone(summary["max_net_degree"])
        self.assertEqual(summary["built_net_count"], 2)
        self.assertEqual([net["net_id"] for net in nets], [0, 1])
        self.assertNotIn("degree_over_limit", summary["skipped_reasons"])

    def test_explicit_net_degree_limit_still_throttles_smoke_nets(self):
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            topology_source="flat_net_star",
            max_net_degree=2,
        )

        self.assertEqual(summary["built_net_count"], 1)
        self.assertEqual([net["net_id"] for net in nets], [1])
        self.assertEqual(summary["skipped_reasons"]["degree_over_limit"], 1)

    def test_uses_external_sink_slack_and_npath_maps(self):
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
            default_sink_slack=-0.5,
            default_npath=3,
            sink_slack_by_pin={1: -12.0},
            npath_by_pin={2: 5},
        )

        self.assertEqual(summary["criticality_source"], "external_maps")
        self.assertEqual(nets[0]["sink_slack_by_pin"], {2: -0.5, 1: -12.0})
        self.assertEqual(nets[0]["npath_by_pin"], {2: 5, 1: 3})

    def test_uses_openroad_libpin_cap_for_sink_node_cap(self):
        pydb = _fake_pydb()
        pydb.inst_main_id = [0, 1, 1, 0, 0]
        pydb.inst_libcell_offset = [0, 0, 0, 0, 0]
        pydb.main_id_2_cell_id_start = [0, 1, 2]
        pydb.cell_id_2_libpin_id_start = [0, 2, 4]
        pydb.pin_2_libpin_offset = [1, 0, 1, 1, 0]
        pydb.flat_lib_pin_cap = [0.0, 0.0, 0.42, 0.84]

        nets, summary = build_buffering_nets_from_pydb(
            pydb,
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
        )

        self.assertEqual(summary["node_cap_source"], "libpin_cap")
        self.assertEqual(summary["node_cap_nondefault_count"], 2)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][1], 0.42)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][2], 0.84)

    def test_prefers_current_pin_cap_for_sink_node_cap(self):
        pydb = _fake_pydb()
        pydb.inst_main_id = [0, 1, 1, 0, 0]
        pydb.inst_libcell_offset = [0, 0, 0, 0, 0]
        pydb.main_id_2_cell_id_start = [0, 1, 2]
        pydb.cell_id_2_libpin_id_start = [0, 2, 4]
        pydb.pin_2_libpin_offset = [1, 0, 1, 1, 0]
        pydb.flat_lib_pin_cap = [0.0, 0.0, 0.42, 0.84]

        nets, summary = build_buffering_nets_from_pydb(
            pydb,
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
            pin_cap_by_pin_id={1: 1.25, 2: 2.5},
            pin_cap_source="unit_current_pin_caps",
        )

        self.assertEqual(summary["node_cap_libpin_cap_source"], "unit_current_pin_caps")
        self.assertEqual(summary["node_cap_external_non_sink_count"], 0)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][0], 0.0)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][1], 1.25)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][2], 2.5)

    def test_current_pin_cap_can_set_driver_node_cap(self):
        pydb = _fake_pydb()

        nets, summary = build_buffering_nets_from_pydb(
            pydb,
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
            pin_cap_by_pin_id={0: 0.75, 1: 1.25, 2: 2.5},
            pin_cap_source="unit_current_pin_caps",
        )

        self.assertEqual(summary["node_cap_external_non_sink_count"], 1)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][0], 0.75)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][1], 1.25)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][2], 2.5)

    def test_rejects_missing_sink_pin_cap_instead_of_using_unit_fallback(self):
        pydb = _fake_pydb()
        pydb.flat_lib_pin_cap = []

        with self.assertRaisesRegex(AssertionError, "missing sink pin capacitance"):
            build_buffering_nets_from_pydb(
                pydb,
                net_ids=[0],
                topology_source="flat_net_star",
                max_net_degree=8,
            )

    def test_uses_zero_cap_for_io_sink_without_libpin_cap(self):
        pydb = _fake_pydb()
        pydb.num_movable_nodes = 4
        pydb.num_terminals = 0
        pydb.num_terminal_NIs = 1
        pydb.pin2node_map = [0, 1, 2, 3, 4]
        pydb.flat_lib_pin_cap = [0.0, 0.42]

        nets, summary = build_buffering_nets_from_pydb(
            pydb,
            net_ids=[1],
            topology_source="flat_net_star",
            max_net_degree=8,
        )

        self.assertEqual(summary["node_cap_source"], "io_zero_cap")
        self.assertEqual(summary["node_cap_io_sink_count"], 1)
        self.assertEqual(nets[0]["rc_tree"]["node_cap"][4], 0.0)

    def test_builds_criticality_maps_from_timing_outputs(self):
        from dreamplace.ops.buffer_insertion.real_design_adapter import (
            build_criticality_maps_from_timing_outputs,
        )

        maps, summary = build_criticality_maps_from_timing_outputs(
            pin_slack=[0.0, -4.5, 2.0],
            endpoint_incidence_result=SimpleNamespace(
                pin_endpoint_incidence_count=[0, 3, 1],
                active_endpoint_count=2,
            ),
        )

        self.assertEqual(summary["criticality_source"], "timing_propagation")
        self.assertEqual(summary["sink_slack_map_entry_count"], 3)
        self.assertEqual(summary["npath_map_entry_count"], 3)
        self.assertEqual(summary["active_endpoint_count"], 2)
        self.assertEqual(maps["sink_slack_by_pin"], {0: 0.0, 1: -4.5, 2: 2.0})
        self.assertEqual(maps["npath_by_pin"], {0: 0, 1: 3, 2: 1})

    def test_builds_criticality_maps_from_timing_op(self):
        from dreamplace.ops.buffer_insertion.real_design_adapter import (
            build_criticality_maps_from_timing_op,
        )

        timing_op = SimpleNamespace(
            get_pin_slack=lambda: [1.0, -2.0],
            compute_active_endpoint_incidence=lambda: SimpleNamespace(
                pin_endpoint_incidence_count=[1, 4],
                active_endpoint_count=4,
            ),
        )

        maps, summary = build_criticality_maps_from_timing_op(timing_op)

        self.assertEqual(summary["criticality_source"], "timing_propagation")
        self.assertEqual(maps["sink_slack_by_pin"], {0: 1.0, 1: -2.0})
        self.assertEqual(maps["npath_by_pin"], {0: 1, 1: 4})

    def test_builds_criticality_maps_derives_active_endpoints_from_negative_slack(self):
        from dreamplace.ops.buffer_insertion.real_design_adapter import (
            build_criticality_maps_from_timing_op,
        )

        calls = []

        def compute_active_endpoint_incidence(active_endpoint_ids=None):
            calls.append(list(active_endpoint_ids or []))
            return SimpleNamespace(
                pin_endpoint_incidence_count=[0, 2, 1, 2],
                active_endpoint_count=len(active_endpoint_ids or []),
            )

        timing_op = SimpleNamespace(
            get_pin_slack=lambda: [1.0, -2.0, 3.0, -0.5],
            last_endpoint_ids_tensor=[1, 3],
            last_endpoint_slack_tensor=[-2.0, -0.5],
            last_active_endpoint_ids=[],
            compute_active_endpoint_incidence=compute_active_endpoint_incidence,
        )

        maps, summary = build_criticality_maps_from_timing_op(timing_op)

        self.assertEqual(calls, [[1, 3]])
        self.assertEqual(summary["npath_map_entry_count"], 4)
        self.assertEqual(summary["active_endpoint_count"], 2)
        self.assertEqual(maps["npath_by_pin"], {0: 0, 1: 2, 2: 1, 3: 2})

    def test_uses_external_edge_rc_map_for_net_edges(self):
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology_source="flat_net_star",
            max_net_degree=8,
            edge_rc_by_net={
                0: {
                    (0, 2): {"r": 1.25, "c": 0.5},
                    (0, 1): {"r": 2.5, "c": 0.25},
                }
            },
        )

        self.assertEqual(summary["rc_source"], "external_edge_rc")
        self.assertEqual(
            nets[0]["rc_tree"]["edge_rc"],
            {
                (0, 2): {"r": 1.25, "c": 0.5},
                (0, 1): {"r": 2.5, "c": 0.25},
            },
        )

    def test_topology_external_edge_rc_can_be_rekeyed_after_driver_rooting(self):
        topology = {
            "topology_source": "live_timing_topology",
            "net_ids": [0],
            "net_flat_topo_sort": [0, 1, 2],
            "net_flat_topo_sort_start": [0, 3],
            "pin_fa": [1, -1, 1],
            "node_x": {0: 0, 1: 1000, 2: 2000},
            "node_y": {0: 0, 1: 0, 2: 0},
        }

        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
            max_net_degree=8,
            edge_rc_by_net={
                0: {
                    (1, 0): {"r": 1.25, "c": 0.5},
                    (1, 2): {"r": 2.5, "c": 0.25},
                }
            },
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["external_edge_rc_missing_count"], 0)
        self.assertEqual(summary["skipped_reasons"], {})
        self.assertEqual(
            nets[0]["rc_tree"]["children_by_node"],
            {0: [1], 1: [2], 2: []},
        )
        self.assertEqual(
            nets[0]["rc_tree"]["edge_rc"],
            {
                (0, 1): {"r": 1.25, "c": 0.5},
                (1, 2): {"r": 2.5, "c": 0.25},
            },
        )

    def test_topology_coordinates_override_pydb_pin_coordinates(self):
        topology = {
            "topology_source": "live_timing_topology",
            "net_ids": [0],
            "net_flat_topo_sort": [0, 1, 2],
            "net_flat_topo_sort_start": [0, 3],
            "pin_fa": [-1, 0, 0],
            "node_x": {0: 10, 1: 30, 2: 70},
            "node_y": {0: 20, 1: 40, 2: 80},
            "node_x_dbu": {0: 10_100, 1: 30_200, 2: 50_300},
            "node_y_dbu": {0: 1_010, 1: 1_020, 2: 1_030},
        }

        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
            max_net_degree=8,
            edge_rc_by_net={
                0: {
                    (0, 1): {"r": 0.75, "c": 0.25},
                    (0, 2): {"r": 1.25, "c": 0.5},
                }
            },
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(nets[0]["coordinates"][0], (10, 20))
        self.assertEqual(nets[0]["coordinates"][2], (70, 80))
        self.assertEqual(nets[0]["coordinates_dbu"][0], (10_100, 1_010))
        self.assertEqual(nets[0]["coordinates_dbu"][2], (50_300, 1_030))

    def test_builds_edge_rc_from_rc_timing_topology_formula(self):
        from dreamplace.ops.buffer_insertion.real_design_adapter import (
            build_edge_rc_by_net_from_rc_timing_topology,
        )

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
        self.assertEqual(
            edge_rc_by_net,
            {
                10: {
                    (0, 1): {"r": 6.0, "c": 0.75, "length_um": 3.0},
                    (1, 2): {"r": 8.0, "c": 1.0, "length_um": 4.0},
                }
            },
        )

    def test_parses_signal_wire_rc_from_setrc_tcl(self):
        import tempfile
        from pathlib import Path
        from dreamplace.ops.buffer_insertion.real_design_adapter import (
            parse_signal_wire_rc_from_setrc_tcl,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "setRC.tcl"
            path.write_text(
                "\n".join(
                    [
                        "# Liberty units are fF,kOhm",
                        "set_layer_rc -layer M2 -resistance 0.2 -capacitance 0.3",
                        "set_layer_rc -layer M3 -resistance 0.024222 -capacitance 0.12918",
                        "set_wire_rc -signal -layer M3",
                    ]
                )
                + "\n"
            )

            parsed = parse_signal_wire_rc_from_setrc_tcl(path)

        self.assertEqual(parsed["rc_parameter_source"], "set_rc_tcl_signal_layer_raw")
        self.assertEqual(parsed["signal_layer"], "M3")
        self.assertEqual(parsed["wire_resistance_per_micron"], 0.024222)
        self.assertEqual(parsed["wire_capacitance_per_micron"], 0.12918)
        self.assertEqual(parsed["unit_note"], "Liberty units are fF,kOhm")

    def test_skips_nets_without_driver_pin(self):
        pydb = _fake_pydb()
        pydb.net2driver_pin_map = [-1, 3]
        nets, summary = build_buffering_nets_from_pydb(pydb, net_ids=[0])

        self.assertEqual(nets, [])
        self.assertEqual(summary["built_net_count"], 0)
        self.assertEqual(summary["skipped_reasons"]["invalid_driver_pin"], 1)

    def test_consumes_steiner_topology_when_provided(self):
        topology = {
            "topology_source": "steiner",
            "net_flat_topo_sort": [0, 5, 1, 2],
            "net_flat_topo_sort_start": [0, 4],
            "pin_fa": [-1, 5, 5, -1, -1, 0],
            "node_x": {
                0: 10_100,
                1: 30_200,
                2: 50_300,
                5: 20_000,
            },
            "node_y": {
                0: 1_010,
                1: 1_020,
                2: 1_030,
                5: 1_000,
            },
        }
        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
        )

        self.assertEqual(summary["topology_source"], "steiner")
        self.assertEqual(set(nets[0]["undirected_edges"]), {(0, 5), (5, 1), (5, 2)})
        self.assertEqual(nets[0]["coordinates"][5], (20_000, 1_000))

    def test_rebranching_summary_is_exported(self):
        topology = {
            "topology_source": "steiner",
            "net_flat_topo_sort": [0, 5, 1, 2],
            "net_flat_topo_sort_start": [0, 4],
            "pin_fa": [-1, 5, 5, -1, -1, 0],
            "node_x": {
                0: 10_100,
                1: 30_200,
                2: 50_300,
                5: 20_000,
            },
            "node_y": {
                0: 1_010,
                1: 1_020,
                2: 1_030,
                5: 1_000,
            },
        }

        nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
            sink_slack_by_pin={1: -1.0, 2: -10.0},
            npath_by_pin={1: 1, 2: 3},
            enable_rebranching=True,
            max_rebranched_sink_ratio=0.5,
        )

        self.assertEqual(summary["rebranching_enabled"], True)
        self.assertEqual(summary["rebranching_status"], "ok")
        self.assertEqual(summary["rebranched_sink_count"], 1)
        self.assertEqual(summary["candidate_set_skeleton_source"], "timing_aware_rebranched_skeleton")
        self.assertEqual(nets[0]["rebranching_summary"]["rebranching_status"], "ok")
        self.assertIn((0, 2), set(nets[0]["undirected_edges"]))
        self.assertEqual(sorted(nets[0]["rc_tree"]["children_by_node"][0]), [2, 5])

    def test_rebranching_wirelength_guard_count_is_exported(self):
        topology = {
            "topology_source": "steiner",
            "net_flat_topo_sort": [0, 5, 1, 2],
            "net_flat_topo_sort_start": [0, 4],
            "pin_fa": [-1, 5, 5, -1, -1, 0],
            "node_x": {
                0: 0,
                1: 0,
                2: 0,
                5: 30_200,
            },
            "node_y": {
                0: 0,
                1: 0,
                2: 0,
                5: 1_020,
            },
        }

        _nets, summary = build_buffering_nets_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            topology=topology,
            sink_slack_by_pin={1: -10.0, 2: 0.0},
            npath_by_pin={1: 1, 2: 1},
            enable_rebranching=True,
            max_rebranched_sink_ratio=1.0,
            rebranching_max_wirelength_delta_ratio=0.1,
        )

        self.assertEqual(summary["rebranching_status"], "skipped_with_reason")
        self.assertEqual(summary["rebranched_sink_count"], 0)
        self.assertEqual(summary["wirelength_guard_triggered_count"], 1)
        self.assertEqual(
            summary["rebranching_skip_reason_counts"],
            {"wirelength_delta_exceeds_limit": 1},
        )

    def test_builds_local_steiner_topology_with_original_pin_mapping(self):
        topology = build_local_steiner_topology_from_pydb(
            _fake_pydb(),
            net_ids=[0],
            max_net_degree=8,
        )

        self.assertEqual(topology["topology_source"], "steiner")
        self.assertEqual(topology["net_ids"], [0])
        self.assertEqual(topology["net_flat_topo_sort_start"][0], 0)
        self.assertEqual(topology["net_flat_topo_sort_start"][-1], len(topology["net_flat_topo_sort"]))
        self.assertIn(0, topology["net_flat_topo_sort"])
        self.assertIn(1, topology["net_flat_topo_sort"])
        self.assertIn(2, topology["net_flat_topo_sort"])
        self.assertEqual(topology["node_x"][0], 10_100)
        self.assertEqual(topology["node_y"][0], 1_010)

    def test_steiner_topology_builder_is_canonical_topology_entry(self):
        self.assertIs(
            build_local_steiner_topology_from_pydb,
            build_local_steiner_topology_from_pydb_canonical,
        )


if __name__ == "__main__":
    unittest.main()
