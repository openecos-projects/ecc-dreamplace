import unittest

from dreamplace.ops.physical_action import (
    VirtualBufferState,
    build_commit_action_candidates,
    select_best_insert_candidate,
    select_best_insert_per_net,
)


class PhysicalActionVirtualSchemaTest(unittest.TestCase):
    def _candidate(self, candidate_id=0, predicted_delta_obj=-5.0):
        return {
            "candidate_id": candidate_id,
            "net_id": 12,
            "net_name": "net12",
            "driver_pin_id": 100,
            "driver_pin_name": "U_DRV/Y",
            "load_pin_id": 101,
            "load_pin_name": "U_SINK/A",
            "downstream_pin_ids": [101, 102],
            "downstream_pin_names": ["U_SINK/A", "U_SINK2/A"],
            "downstream_pin_count": 2,
            "moved_branch_sink_cap_sum": 4.0,
            "upstream_sibling_sink_cap_sum": 6.0,
            "moved_branch_criticality_sum": 3.0,
            "upstream_sibling_criticality_sum": 9.0,
            "upstream_sibling_negative_slack_sink_count": 1,
            "moved_branch_has_setup_criticality": True,
            "sibling_has_setup_criticality": True,
            "moved_noncritical_branch_with_critical_sibling": False,
            "buffer_input_cap_minus_moved_branch_cap": 1.0,
            "buffer_input_cap_to_moved_branch_cap_ratio": 1.25,
            "buffer_input_cap_exceeds_moved_branch_cap": True,
            "noncritical_moved_branch_replaced_by_larger_buffer_input_cap": False,
            "load_partition_source": "timing_rooted_tree_subtree",
            "parent_node_id": 100,
            "child_node_ids": [101, 102],
            "x_dbu": 20_000,
            "y_dbu": 30_000,
            "x_um": 20.0,
            "y_um": 30.0,
            "buffer_main_type_index": 7,
            "predicted_delta_obj": predicted_delta_obj,
            "predicted_improvement": -predicted_delta_obj,
            "source_sink_slack": -2.0,
            "source_sink_npath": 4,
            "source_sink_criticality": 8.0,
        }

    def _legal_table(self):
        return {
            "legal_cell_ids": [40, 41],
            "legal_master_names": ["BUF_X1", "BUF_X2"],
            "legal_size_values": [1.0, 2.0],
        }

    def test_selects_most_negative_predicted_delta_obj(self):
        best = select_best_insert_candidate([
            self._candidate(1, predicted_delta_obj=2.0),
            self._candidate(2, predicted_delta_obj=-1.0),
            self._candidate(3, predicted_delta_obj=-3.0),
        ])

        self.assertEqual(best["candidate_id"], 3)

    def test_selects_at_most_one_insert_per_net(self):
        candidates = [
            dict(self._candidate(1, predicted_delta_obj=-1.0), net_id=1),
            dict(self._candidate(2, predicted_delta_obj=-3.0), net_id=1),
            dict(self._candidate(3, predicted_delta_obj=-2.0), net_id=2),
        ]

        selected = select_best_insert_per_net(candidates)

        self.assertEqual([item["candidate_id"] for item in selected], [2, 3])

    def test_virtual_insert_resize_and_remove_invariants(self):
        state = VirtualBufferState()
        candidate = self._candidate()

        with self.assertRaises(ValueError):
            state.resize(candidate, target_bsu=1)

        inserted = state.insert(candidate, bsu=0)
        self.assertEqual(inserted["bu"], 1)
        self.assertEqual(inserted["bsu"], 0)
        self.assertEqual(state.committed_buffer_count, 0)
        self.assertEqual(state.virtual_buffer_count, 1)

        resized = state.resize(candidate, target_bsu=1)
        self.assertEqual(resized["action"], "upsize")
        self.assertEqual(state.entries[0]["bsu"], 1)

        removed = state.remove(candidate, allow_remove=True)
        self.assertEqual(removed["action"], "remove")
        self.assertEqual(state.virtual_buffer_count, 0)

    def test_remove_before_warmup_is_rejected(self):
        state = VirtualBufferState()
        candidate = self._candidate()

        state.insert(candidate, bsu=0)

        with self.assertRaises(ValueError):
            state.remove(candidate, allow_remove=False)

    def test_compact_artifact_is_schema_compatible_but_not_executable_v1(self):
        state = VirtualBufferState()
        state.insert(self._candidate(), bsu=1)

        artifact = build_commit_action_candidates(
            state,
            legal_table=self._legal_table(),
            executable_by_compact_batch_v1=False,
        )

        self.assertTrue(artifact["schema_compatible_with_compact_action"])
        self.assertFalse(artifact["executable_by_compact_batch_v1"])
        self.assertEqual(artifact["committed_buffer_count"], 0)
        self.assertTrue(artifact["realization_boundary"]["realization_ready"])
        self.assertEqual(
            artifact["realization_boundary"]["backend_probe"]["openroad_one_net"]["status"],
            "available_scoped_repair",
        )
        self.assertTrue(
            artifact["realization_boundary"]["backend_probe"]["openroad_one_net"][
                "supports_buffer_insert"
            ]
        )
        self.assertEqual(
            artifact["realization_boundary"]["realization_mode"],
            "ready_for_openroad_one_net",
        )
        self.assertEqual(
            artifact["realization_boundary"]["supported_realization_backends"],
            ["openroad_one_net"],
        )
        self.assertIn(
            "compact_batch_buffer_insert_not_supported",
            artifact["realization_boundary"]["unsupported_realization_reasons"],
        )
        action = artifact["actions"][0]
        self.assertEqual(action["action_kind"], "buffer_insert")
        self.assertEqual(action["affected_net_name"], "net12")
        self.assertEqual(action["net_name"], "net12")
        self.assertEqual(action["driver_pin_name"], "U_DRV/Y")
        self.assertEqual(action["load_pin_name"], "U_SINK/A")
        self.assertEqual(action["downstream_pin_ids"], [101, 102])
        self.assertEqual(action["downstream_pin_names"], ["U_SINK/A", "U_SINK2/A"])
        self.assertEqual(action["downstream_pin_count"], 2)
        self.assertEqual(action["moved_branch_sink_cap_sum"], 4.0)
        self.assertEqual(action["upstream_sibling_sink_cap_sum"], 6.0)
        self.assertEqual(action["moved_branch_criticality_sum"], 3.0)
        self.assertEqual(action["upstream_sibling_criticality_sum"], 9.0)
        self.assertEqual(action["upstream_sibling_negative_slack_sink_count"], 1)
        self.assertTrue(action["moved_branch_has_setup_criticality"])
        self.assertTrue(action["sibling_has_setup_criticality"])
        self.assertFalse(action["moved_noncritical_branch_with_critical_sibling"])
        self.assertEqual(action["buffer_input_cap_minus_moved_branch_cap"], 1.0)
        self.assertEqual(action["buffer_input_cap_to_moved_branch_cap_ratio"], 1.25)
        self.assertTrue(action["buffer_input_cap_exceeds_moved_branch_cap"])
        self.assertFalse(
            action["noncritical_moved_branch_replaced_by_larger_buffer_input_cap"]
        )
        self.assertEqual(action["load_partition_source"], "timing_rooted_tree_subtree")
        self.assertEqual(action["parent_node_id"], 100)
        self.assertEqual(action["child_node_ids"], [101, 102])
        self.assertEqual(action["bsu"], 1)
        self.assertEqual(action["buffer_cell_id"], 41)
        self.assertEqual(action["buffer_master_id"], 41)
        self.assertEqual(action["buffer_master_name"], "BUF_X2")
        self.assertEqual(action["candidate_location_x_dbu"], 20_000.0)
        self.assertEqual(action["candidate_location_y_dbu"], 30_000.0)
        self.assertEqual(action["candidate_location_x_um"], 20.0)
        self.assertEqual(action["candidate_location_y_um"], 30.0)
        self.assertEqual(action["legal_size_value"], 2.0)
        self.assertEqual(action["source_sink_slack"], -2.0)
        self.assertEqual(action["source_sink_npath"], 4)
        self.assertEqual(action["source_sink_criticality"], 8.0)
        self.assertFalse(action["debug_for_realization_smoke"])
        self.assertEqual(action["legal_cell_id_candidates"], [40, 41])
        self.assertEqual(action["legal_master_candidates"], ["BUF_X1", "BUF_X2"])



if __name__ == "__main__":
    unittest.main()
