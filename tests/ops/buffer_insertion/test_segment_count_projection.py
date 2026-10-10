import unittest

import torch

from dreamplace.ops.buffer_insertion.coordinate_backend import (
    build_coordinate_backend_contract,
)
from dreamplace.ops.buffer_insertion.segment_count_projection import (
    project_segment_count_state_to_candidates,
    segment_count_projection_to_virtual_state,
)
from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)


class SegmentCountProjectionTest(unittest.TestCase):
    def _line_net(self):
        return {
            "net_id": 41,
            "net_name": "line",
            "driver_pin_id": 0,
            "driver_pin_name": "drv/Y",
            "load_pin_id": 1,
            "load_pin_name": "sink/A",
            "downstream_pin_ids": [1],
            "downstream_pin_names": ["sink/A"],
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 10.0, "c": 0.0}},
                "node_cap": {0: 0.0, 1: 6.0},
                "sink_nodes": [1],
            },
        }

    def _branch_net_without_top_level_load_context(self):
        return {
            "net_id": 42,
            "net_name": "branch",
            "driver_pin_id": 0,
            "driver_pin_name": "drv/Y",
            "pin_name_by_id": {
                0: "drv/Y",
                2: "sink_a/A",
                3: "sink_b/A",
            },
            "sink_slack_by_pin": {
                2: -2.0,
                3: 1.0,
            },
            "npath_by_pin": {
                2: 3,
                3: 5,
            },
            "coordinates": {
                0: (0, 0),
                1: (100, 0),
                2: (150, 50),
                3: (150, -50),
            },
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: [2, 3], 2: [], 3: []},
                "edge_rc": {
                    (0, 1): {"r": 10.0, "c": 0.0},
                    (1, 2): {"r": 5.0, "c": 0.0},
                    (1, 3): {"r": 5.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 0.0, 2: 3.0, 3: 4.0},
                "sink_nodes": [2, 3],
            },
        }

    def _criticality_sort_net(self):
        return {
            "net_id": 43,
            "net_name": "sort_net",
            "driver_pin_id": 0,
            "driver_pin_name": "drv/Y",
            "pin_name_by_id": {
                0: "drv/Y",
                1: "weak/A",
                2: "strong/A",
            },
            "sink_slack_by_pin": {
                1: -1.0,
                2: -10.0,
            },
            "npath_by_pin": {
                1: 1,
                2: 10,
            },
            "coordinates": {
                0: (0, 0),
                1: (100, 0),
                2: (200, 0),
            },
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [], 2: []},
                "edge_rc": {
                    (0, 1): {"r": 1.0, "c": 0.0},
                    (0, 2): {"r": 1.0, "c": 0.0},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 1.0},
                "sink_nodes": [1, 2],
            },
        }

    def test_projection_uses_optimized_global_state_and_shared_segment_bsu(self):
        nets = [self._line_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.0,
            initial_bsu_index=0.0,
        )
        with torch.no_grad():
            state.z_param.fill_(2.4)
            state.bsu_index_param.fill_(2.6)

        artifact = project_segment_count_state_to_candidates(nets, state)
        candidates = artifact["candidates"]

        self.assertEqual(artifact["projection_policy"], "round_clip_optimized_global_state")
        self.assertFalse(artifact["local_ranking_used"])
        self.assertEqual(artifact["projected_candidate_count"], 2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate["x_dbu"] for candidate in candidates], [33, 67])
        self.assertEqual([candidate["projected_repeater_count"] for candidate in candidates], [2, 2])
        self.assertEqual([candidate["segment_split_index"] for candidate in candidates], [0, 1])
        for candidate in candidates:
            self.assertEqual(candidate["net_id"], 41)
            self.assertEqual(candidate["segment_id"], 0)
            self.assertEqual(candidate["bsu"], 3)
            self.assertEqual(candidate["projected_bsu_index"], 3)
            self.assertEqual(candidate["buffer_main_type_index"], 7)
            self.assertEqual(candidate["bsu_sharing"], "per_segment")
            self.assertEqual(candidate["max_repeater_count"], 3)
            self.assertIn("segment_count_projection", candidate["source_flags"])
            self.assertIn("equal_spaced_segment_count", candidate["source_flags"])
            self.assertNotIn("selection_score", candidate)
            self.assertNotIn("local_transfer_score", candidate)

    def test_projection_clips_count_and_bsu_to_legal_range(self):
        nets = [self._line_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=0.0,
            initial_bsu_index=0.0,
        )
        with torch.no_grad():
            state.z_param.fill_(10.0)
            state.bsu_index_param.fill_(-2.0)

        artifact = project_segment_count_state_to_candidates(nets, state)
        candidates = artifact["candidates"]

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["projected_repeater_count"] == 2 for candidate in candidates))
        self.assertTrue(all(candidate["projected_bsu_index"] == 0 for candidate in candidates))

    def test_projection_top_k_limits_segment_count_actions_after_sort(self):
        net = self._criticality_sort_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=1.0,
            initial_bsu_index=1.0,
        )

        artifact = project_segment_count_state_to_candidates([net], state, top_k=1)
        candidates = artifact["candidates"]

        self.assertEqual(artifact["projected_candidate_count_before_topk"], 2)
        self.assertEqual(artifact["projected_candidate_count"], 1)
        self.assertEqual(artifact["top_k"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["load_pin_name"], "strong/A")

    def test_projection_uses_commit_dbu_coordinates_when_available(self):
        net = self._line_net()
        net["coordinates"] = {0: (0, 0), 1: (100, 0)}
        net["coordinates_dbu"] = {0: (1000, 2000), 1: (3000, 2000)}
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=1.0,
            initial_bsu_index=1.0,
        )

        artifact = project_segment_count_state_to_candidates([net], state)
        candidate = artifact["candidates"][0]

        self.assertEqual(candidate["x_dbu"], 2000)
        self.assertEqual(candidate["y_dbu"], 2000)

    def test_joint_projection_uses_final_live_geometry_and_records_provenance(self):
        net = self._line_net()
        net["coordinates_dbu"] = {0: (1000, 2000), 1: (3000, 2000)}
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=2.0,
            initial_bsu_index=1.0,
        )
        context = {
            "coordinate_source": "final_live_geometry",
            "placement_snapshot_identity": "sha256:final-pos",
            "placement_snapshot_iteration": 123,
            "topology_generation": 4,
            "frozen_topology_generation": 4,
            "expected_frozen_topology_generation": 4,
            "node_x_dbu": torch.tensor([2000.0, 5003.0]),
            "node_y_dbu": torch.tensor([3000.0, 3000.0]),
        }

        artifact = project_segment_count_state_to_candidates(
            [net],
            state,
            live_projection_context=context,
        )

        candidates = artifact["candidates"]
        self.assertEqual([candidate["x_dbu"] for candidate in candidates], [3001, 4002])
        self.assertEqual([candidate["y_dbu"] for candidate in candidates], [3000, 3000])
        self.assertAlmostEqual(
            candidates[0]["candidate_location_x_dbu_analytical"],
            3001.0,
        )
        self.assertAlmostEqual(
            candidates[1]["candidate_location_x_dbu_analytical"],
            4002.0,
        )
        self.assertEqual(candidates[0]["segment_parent_x_dbu"], 2000.0)
        self.assertEqual(candidates[0]["segment_child_x_dbu"], 5003.0)
        self.assertEqual(candidates[0]["coordinate_source"], "final_live_geometry")
        self.assertEqual(
            candidates[0]["placement_snapshot_identity"],
            "sha256:final-pos",
        )
        self.assertEqual(candidates[0]["topology_generation"], 4)
        self.assertEqual(
            artifact["projection_provenance"]["coordinate_rounding"],
            "round_to_nearest_dbu",
        )
        from dreamplace.ops.physical_action import build_commit_action_candidates

        action_artifact = build_commit_action_candidates(
            segment_count_projection_to_virtual_state(artifact),
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
        )
        action = action_artifact["actions"][0]
        self.assertEqual(action["coordinate_source"], "final_live_geometry")
        self.assertEqual(action["placement_snapshot_identity"], "sha256:final-pos")
        self.assertEqual(action["topology_generation"], 4)
        self.assertAlmostEqual(
            action["candidate_location_x_dbu_analytical"],
            3001.0,
        )

    def test_joint_projection_coordinates_co_move_with_final_segment(self):
        net = self._line_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=3.0,
            initial_bsu_index=1.0,
        )

        def project(node_x, node_y, snapshot):
            return project_segment_count_state_to_candidates(
                [net],
                state,
                live_projection_context={
                    "coordinate_source": "final_live_geometry",
                    "placement_snapshot_identity": snapshot,
                    "topology_generation": 2,
                    "frozen_topology_generation": 2,
                    "expected_frozen_topology_generation": 2,
                    "node_x_dbu": torch.tensor(node_x, dtype=torch.float64),
                    "node_y_dbu": torch.tensor(node_y, dtype=torch.float64),
                },
            )["candidates"]

        base = project([0.0, 100.0], [10.0, 10.0], "base")
        translated = project([50.0, 150.0], [30.0, 30.0], "translated")
        stretched = project([0.0, 200.0], [10.0, 10.0], "stretched")

        self.assertEqual(
            [item["x_dbu"] for item in translated],
            [item["x_dbu"] + 50 for item in base],
        )
        self.assertEqual(
            [item["y_dbu"] for item in translated],
            [item["y_dbu"] + 20 for item in base],
        )
        self.assertEqual([item["x_dbu"] for item in stretched], [50, 100, 150])
        self.assertEqual(
            [item["segment_split_ratio"] for item in stretched],
            [0.25, 0.5, 0.75],
        )

    def test_joint_projection_places_diagonal_edge_sites_on_x_then_y_path(self):
        net = self._line_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=3.0,
            initial_bsu_index=1.0,
        )

        candidates = project_segment_count_state_to_candidates(
            [net],
            state,
            live_projection_context={
                "coordinate_source": "final_live_geometry",
                "placement_snapshot_identity": "diagonal",
                "topology_generation": 2,
                "frozen_topology_generation": 2,
                "expected_frozen_topology_generation": 2,
                "rectilinear_path_policy": "x_then_y",
                "node_x_dbu": torch.tensor([0.0, 100.0]),
                "node_y_dbu": torch.tensor([0.0, 100.0]),
            },
        )["candidates"]

        self.assertEqual(
            [(item["x_dbu"], item["y_dbu"]) for item in candidates],
            [(50, 0), (100, 0), (100, 50)],
        )
        self.assertEqual(
            [item["rectilinear_path_policy"] for item in candidates],
            ["x_then_y"] * 3,
        )

    def test_joint_projection_rejects_snapshot_generation_mismatch(self):
        net = self._line_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=1,
            initial_z=1.0,
            initial_bsu_index=1.0,
        )
        with self.assertRaisesRegex(ValueError, "generation mismatch"):
            project_segment_count_state_to_candidates(
                [net],
                state,
                live_projection_context={
                    "coordinate_source": "final_live_geometry",
                    "placement_snapshot_identity": "mismatch",
                    "topology_generation": 3,
                    "frozen_topology_generation": 3,
                    "expected_frozen_topology_generation": 2,
                    "node_x_dbu": torch.tensor([0.0, 100.0]),
                    "node_y_dbu": torch.tensor([0.0, 0.0]),
                },
            )

    def test_projection_carries_forced_z_prediction_to_commit_action(self):
        from dreamplace.ops.physical_action import build_commit_action_candidates

        nets = [self._line_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=1.0,
            initial_bsu_index=2.0,
        )
        state.segment_forced_z_prediction_by_segment_id = {
            0: {
                "predicted_delta_tns_vs_z0": 12.5,
                "predicted_delta_wns_vs_z0": 0.25,
                "predicted_delta_loss_vs_z0": -12.5,
            }
        }

        projection = project_segment_count_state_to_candidates(nets, state)
        candidate = projection["candidates"][0]
        virtual_state = segment_count_projection_to_virtual_state(projection)
        action_artifact = build_commit_action_candidates(
            virtual_state,
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
        )
        action = action_artifact["actions"][0]

        self.assertEqual(candidate["predicted_delta_tns"], 12.5)
        self.assertEqual(candidate["predicted_delta_wns"], 0.25)
        self.assertEqual(candidate["predicted_delta_obj"], -12.5)
        self.assertEqual(candidate["selection_score"], 12.5)
        self.assertEqual(action["predicted_delta_tns"], 12.5)
        self.assertEqual(action["predicted_delta_wns"], 0.25)
        self.assertEqual(action["predicted_delta_obj"], -12.5)

    def test_projection_skips_zero_projected_count(self):
        nets = [self._line_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=0.0,
            initial_bsu_index=1.0,
        )

        artifact = project_segment_count_state_to_candidates(nets, state)

        self.assertEqual(artifact["projected_candidate_count"], 0)
        self.assertEqual(artifact["candidates"], [])

    def test_projection_candidates_feed_existing_commit_action_schema(self):
        from dreamplace.ops.physical_action import build_commit_action_candidates

        nets = [self._line_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=1.0,
            initial_bsu_index=2.0,
        )

        projection = project_segment_count_state_to_candidates(nets, state)
        virtual_state = segment_count_projection_to_virtual_state(projection)
        action_artifact = build_commit_action_candidates(
            virtual_state,
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
        )
        contract = build_coordinate_backend_contract(action_artifact["actions"])

        self.assertTrue(action_artifact["schema_compatible_with_compact_action"])
        self.assertEqual(action_artifact["candidate_commit_action_count"], 1)
        self.assertEqual(contract["payload_ready_count"], 1)
        self.assertEqual(contract["status"], "contract_ready_backend_not_implemented")
        action = action_artifact["actions"][0]
        self.assertEqual(action["action_kind"], "buffer_insert")
        self.assertEqual(action["net_name"], "line")
        self.assertEqual(action["driver_pin_name"], "drv/Y")
        self.assertEqual(action["load_pin_name"], "sink/A")
        self.assertEqual(action["target_size_idx"], 2)
        self.assertEqual(action["segment_id"], 0)
        self.assertAlmostEqual(action["z_value"], 1.0)
        self.assertAlmostEqual(action["bsu_value"], 2.0)
        self.assertEqual(action["projected_bsu_index"], 2)
        self.assertEqual(action["projected_repeater_count"], 1)
        self.assertEqual(action["projection_policy"], "round_clip_optimized_global_state")
        self.assertEqual(action["bsu_sharing"], "per_segment")
        self.assertEqual(action["max_repeater_count"], 2)
        self.assertIn("segment_count_projection", action["source_flags"])
        self.assertIn("equal_spaced_segment_count", action["source_flags"])
        self.assertEqual(action["coordinate_action_source"], "segment_count_projection")
        self.assertEqual(action["selected_action_source"], "segment_count_projection")
        self.assertEqual(action["selection_result_class"], "segment_count_projection")
        self.assertEqual(action["result_class"], "segment_count_projection")

    def test_projection_orders_candidates_by_relaxed_count_and_criticality(self):
        nets = [self._criticality_sort_net()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=0.0,
            initial_bsu_index=2.0,
        )
        with torch.no_grad():
            state.z_param[0] = 1.4
            state.z_param[1] = 0.6

        projection = project_segment_count_state_to_candidates(nets, state)
        candidates = projection["candidates"]

        self.assertEqual([candidate["load_pin_name"] for candidate in candidates], ["strong/A", "weak/A"])
        self.assertGreater(
            candidates[0]["projection_selection_score"],
            candidates[1]["projection_selection_score"],
        )
        self.assertAlmostEqual(candidates[0]["z_value"], 0.6, places=6)
        self.assertEqual(candidates[0]["downstream_sink_criticality_sum"], 100.0)
        self.assertEqual(candidates[0]["projection_order"], 0)
        self.assertEqual(candidates[1]["projection_order"], 1)
        self.assertEqual(
            projection["selected_candidate_diagnostics"][0]["load_pin_name"],
            "strong/A",
        )
        self.assertEqual(
            projection["top_candidate_diagnostics"][0]["segment_id"],
            candidates[0]["segment_id"],
        )
        self.assertEqual(
            projection["top_candidate_diagnostics"][0][
                "downstream_sink_criticality_sum"
            ],
            100.0,
        )

    def test_projection_can_filter_zero_criticality_candidates(self):
        net = self._criticality_sort_net()
        net["sink_slack_by_pin"] = {1: 2.0, 2: 3.0}
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=2,
            initial_z=1.0,
            initial_bsu_index=1.0,
        )

        unfiltered = project_segment_count_state_to_candidates([net], state)
        filtered = project_segment_count_state_to_candidates(
            [net],
            state,
            require_setup_criticality=True,
        )

        self.assertEqual(unfiltered["projected_candidate_count"], 2)
        self.assertFalse(unfiltered["require_setup_criticality"])
        self.assertEqual(filtered["projected_candidate_count_before_topk"], 2)
        self.assertEqual(filtered["projected_candidate_count"], 0)
        self.assertTrue(filtered["require_setup_criticality"])
        self.assertEqual(filtered["criticality_filtered_candidate_count"], 2)
        self.assertEqual(filtered["candidates"], [])

    def test_projection_derives_downstream_context_from_child_subtree(self):
        from dreamplace.ops.physical_action import build_commit_action_candidates

        nets = [self._branch_net_without_top_level_load_context()]
        state = build_segment_count_state(
            nets,
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=1,
            initial_z=0.0,
            initial_bsu_index=2.0,
        )
        with torch.no_grad():
            state.z_param.zero_()
            state.z_param[0] = 1.0

        projection = project_segment_count_state_to_candidates(nets, state)
        virtual_state = segment_count_projection_to_virtual_state(projection)
        action_artifact = build_commit_action_candidates(
            virtual_state,
            legal_table={
                "legal_cell_ids": [10, 11, 12, 13],
                "legal_master_names": ["BUF_X1", "BUF_X2", "BUF_X4", "BUF_X8"],
                "legal_size_values": [1.0, 2.0, 4.0, 8.0],
            },
        )
        contract = build_coordinate_backend_contract(action_artifact["actions"])

        self.assertEqual(projection["projected_candidate_count"], 1)
        candidate = projection["candidates"][0]
        self.assertEqual(candidate["load_pin_name"], "sink_a/A")
        self.assertEqual(candidate["downstream_pin_names"], ["sink_a/A", "sink_b/A"])
        self.assertEqual(candidate["downstream_pin_ids"], [2, 3])
        self.assertEqual(candidate["downstream_pin_count"], 2)
        self.assertEqual(candidate["load_partition_source"], "segment_count_child_subtree")
        self.assertEqual(candidate["source_sink_slack"], -2.0)
        self.assertEqual(candidate["source_sink_npath"], 3)
        self.assertEqual(candidate["source_sink_criticality"], 6.0)
        self.assertEqual(candidate["downstream_negative_slack_sink_count"], 1)
        self.assertEqual(candidate["total_negative_slack_sink_count"], 1)
        self.assertEqual(candidate["downstream_worst_sink_slack"], -2.0)
        self.assertEqual(candidate["total_worst_sink_slack"], -2.0)
        self.assertEqual(candidate["downstream_sink_criticality_sum"], 6.0)
        self.assertEqual(candidate["total_sink_criticality_sum"], 6.0)
        self.assertEqual(candidate["downstream_sink_criticality_ratio"], 1.0)
        self.assertEqual(candidate["downstream_sink_criticality_mean"], 3.0)
        self.assertEqual(candidate["downstream_sink_criticality_max"], 6.0)
        self.assertTrue(candidate["moved_branch_has_setup_criticality"])
        self.assertEqual(contract["payload_ready_count"], 1)
        self.assertEqual(action_artifact["actions"][0]["load_pin_name"], "sink_a/A")
        self.assertEqual(
            action_artifact["actions"][0]["downstream_pin_names"],
            ["sink_a/A", "sink_b/A"],
        )
        self.assertEqual(
            action_artifact["actions"][0]["downstream_sink_criticality_sum"],
            6.0,
        )
        self.assertEqual(action_artifact["actions"][0]["source_sink_criticality"], 6.0)


if __name__ == "__main__":
    unittest.main()
