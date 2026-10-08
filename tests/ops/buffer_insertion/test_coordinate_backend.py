import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.coordinate_backend import (
    build_coordinate_measured_results_artifact,
    build_coordinate_batch_measured_results_artifact,
    build_coordinate_backend_contract,
    build_measured_coordinate_action_result,
    commit_coordinate_buffer_actions,
    _commit_summary_from_artifact,
    summarize_coordinate_action_readiness,
)
from dreamplace.ops.physical_action import (
    VirtualBufferState,
    build_commit_action_candidates,
)


class DuCoordinateBackendContractTest(unittest.TestCase):
    def _action(self, **updates):
        action = {
            "action_id": 3,
            "action_kind": "buffer_insert",
            "net_name": "net3",
            "buffer_main_type_index": 7,
            "bsu": 1,
            "buffer_master_id": 11,
            "buffer_master_name": "BUF_X2",
            "candidate_location_x_dbu": 1200.0,
            "candidate_location_y_dbu": 3400.0,
            "driver_pin_name": "drv/Y",
            "load_pin_name": "u0/A",
        }
        action.update(updates)
        return action

    def test_minimal_coordinate_seed_payload_is_contract_ready(self):
        summary = summarize_coordinate_action_readiness(self._action())

        self.assertTrue(summary["payload_ready"])
        self.assertEqual(summary["payload_status"], "ready_minimal_coordinate_seed")
        self.assertEqual(summary["partition_status"], "singleton_load_partition_seed")
        self.assertIn("explicit_downstream_partition_missing", summary["warnings"])

    def test_explicit_downstream_partition_is_contract_ready(self):
        summary = summarize_coordinate_action_readiness(
            self._action(downstream_pin_names=["u0/A", "u1/A"])
        )

        self.assertTrue(summary["payload_ready"])
        self.assertEqual(summary["partition_status"], "explicit_downstream_partition")
        self.assertNotIn("explicit_downstream_partition_missing", summary["warnings"])

    def test_missing_core_coordinate_payload_is_blocked(self):
        summary = summarize_coordinate_action_readiness(
            self._action(
                buffer_main_type_index=-1,
                candidate_location_x_dbu=None,
                candidate_location_y_dbu=None,
            )
        )

        self.assertFalse(summary["payload_ready"])
        self.assertIn("missing_buffer_main_type_index", summary["blockers"])
        self.assertIn("missing_candidate_coordinate", summary["blockers"])

    def test_fractional_relaxed_variables_are_blocked_at_coordinate_boundary(self):
        summary = summarize_coordinate_action_readiness(
            self._action(
                bu=0.5,
                bsu_index_param=1.75,
            )
        )

        self.assertFalse(summary["payload_ready"])
        self.assertIn("fractional_bu_must_not_reach_coordinate_backend", summary["blockers"])
        self.assertIn(
            "continuous_bsu_index_must_not_reach_coordinate_backend",
            summary["blockers"],
        )

    def test_contract_records_backend_as_not_implemented(self):
        contract = build_coordinate_backend_contract(
            [self._action()],
            openroad_source_root="/openroad",
        )

        self.assertEqual(contract["artifact"], "buffering_coordinate_backend_contract")
        self.assertEqual(contract["status"], "contract_ready_backend_not_implemented")
        self.assertEqual(contract["implementation_status"], "not_implemented")
        self.assertFalse(contract["qor_evidence"])
        self.assertEqual(contract["payload_ready_count"], 1)
        self.assertEqual(contract["singleton_seed_count"], 1)
        self.assertIn(
            "requires_cxx_openroad_db_net_split_and_pin_reconnect",
            contract["unsupported_realization_reasons"],
        )
        self.assertTrue(contract["openroad_source_evidence"])

    def test_measured_action_result_schema_flattens_opensta_before_after(self):
        measured = build_measured_coordinate_action_result(
            self._action(
                segment_parent_node_id=10,
                segment_child_node_id=11,
                segment_tree_depth=3,
                segment_split_index=1,
                segment_split_count_on_edge=2,
                segment_split_ratio=2.0 / 3.0,
                downstream_pin_names=["u0/A", "u1/A"],
                commit_input_order=8,
                commit_topology_order=4,
            ),
            {
                "status": "accepted",
                "backend_status": "ok",
                "accepted": True,
                "before_metrics": {"wns": -1.5, "tns": -20.0},
                "after_metrics": {"wns": -1.0, "tns": -12.0},
                "actual_delta_wns": 0.5,
                "actual_delta_tns": 8.0,
                "rollback_status": "not_needed_disposable_design",
                "source_net_name": "net3",
                "resolved_target_net_name": "net3_buffer_coord_2",
                "target_net_remapped": True,
                "resolved_driver_pin_name": "buffer_coord_buf_2_net3:Y",
                "driver_pin_remapped": True,
                "requested_downstream_pin_count": 2,
                "moved_load_pin_names": ["u0/A", "u1/A"],
            },
            disposable_design_path="/tmp/buffering-disposable",
        )

        self.assertEqual(measured["action_id"], 3)
        self.assertEqual(measured["net_name"], "net3")
        self.assertEqual(measured["pre_wns"], -1.5)
        self.assertEqual(measured["pre_tns"], -20.0)
        self.assertEqual(measured["post_wns"], -1.0)
        self.assertEqual(measured["post_tns"], -12.0)
        self.assertTrue(measured["accepted"])
        self.assertEqual(measured["reject_reason"], "")
        self.assertEqual(measured["db_commit_status"], "committed_to_disposable_design")
        self.assertEqual(measured["rollback_status"], "not_needed_disposable_design")
        self.assertEqual(measured["disposable_design_path"], "/tmp/buffering-disposable")
        self.assertEqual(measured["segment_parent_node_id"], 10)
        self.assertEqual(measured["segment_child_node_id"], 11)
        self.assertEqual(measured["segment_tree_depth"], 3)
        self.assertEqual(measured["segment_split_index"], 1)
        self.assertEqual(measured["segment_split_count_on_edge"], 2)
        self.assertAlmostEqual(measured["segment_split_ratio"], 2.0 / 3.0)
        self.assertEqual(measured["downstream_pin_names"], ["u0/A", "u1/A"])
        self.assertEqual(measured["commit_input_order"], 8)
        self.assertEqual(measured["commit_topology_order"], 4)
        self.assertEqual(measured["source_net_name"], "net3")
        self.assertEqual(measured["resolved_target_net_name"], "net3_buffer_coord_2")
        self.assertTrue(measured["target_net_remapped"])
        self.assertEqual(measured["resolved_driver_pin_name"], "buffer_coord_buf_2_net3:Y")
        self.assertTrue(measured["driver_pin_remapped"])
        self.assertEqual(measured["requested_downstream_pin_count"], 2)
        self.assertEqual(measured["moved_load_pin_names"], ["u0/A", "u1/A"])

    def test_measured_action_result_preserves_probe_rank_metadata(self):
        measured = build_measured_coordinate_action_result(
            self._action(
                measured_label_probe_rank_source="safe_group_autograd_grad_bu",
                measured_label_probe_rank_sources=[
                    "safe_group_autograd_grad_bu",
                    "finite_difference_delta_obj",
                ],
                measured_label_probe_rank_value=-4.0,
                measured_label_probe_rank_values={
                    "safe_group_autograd_grad_bu": -4.0,
                    "finite_difference_delta_obj": -7.0,
                },
                measured_label_probe_rank_direction="ascending",
                measured_label_probe_rank_directions={
                    "safe_group_autograd_grad_bu": "ascending",
                    "finite_difference_delta_obj": "ascending",
                },
            ),
            {
                "status": "accepted",
                "backend_status": "ok",
                "accepted": True,
                "before_metrics": {"wns": -1.5, "tns": -20.0},
                "after_metrics": {"wns": -1.0, "tns": -12.0},
                "actual_delta_wns": 0.5,
                "actual_delta_tns": 8.0,
            },
        )

        self.assertEqual(
            measured["measured_label_probe_rank_source"],
            "safe_group_autograd_grad_bu",
        )
        self.assertEqual(
            measured["measured_label_probe_rank_sources"],
            [
                "safe_group_autograd_grad_bu",
                "finite_difference_delta_obj",
            ],
        )
        self.assertEqual(
            measured["measured_label_probe_rank_values"],
            {
                "safe_group_autograd_grad_bu": -4.0,
                "finite_difference_delta_obj": -7.0,
            },
        )
        self.assertEqual(
            measured["measured_label_probe_rank_directions"],
            {
                "safe_group_autograd_grad_bu": "ascending",
                "finite_difference_delta_obj": "ascending",
            },
        )

    def test_measured_action_result_preserves_hard_commit_diagnostic(self):
        measured = build_measured_coordinate_action_result(
            self._action(),
            {
                "status": "accepted",
                "backend_status": "ok",
                "accepted": True,
                "before_metrics": {"wns": -1.5, "tns": -20.0},
                "after_metrics": {"wns": -1.0, "tns": -12.0},
                "actual_delta_wns": 0.5,
                "actual_delta_tns": 8.0,
                "hard_commit_diagnostic": {
                    "schema_name": "buffering_hard_commit_diagnostic",
                    "schema_version": 1,
                    "status": "ok",
                    "topology_delta": {
                        "instance_count_delta": 1,
                        "net_count_delta": 1,
                        "buffer_count_delta": 1,
                    },
                    "load_partition": {
                        "moved_load_count": 2,
                        "moved_load_pin_names": ["u0/A", "u1/A"],
                    },
                    "pin_timing_samples": {
                        "status": "not_sampled",
                        "reason": "backend_did_not_export_pin_samples",
                    },
                    "cell_arc_delay_samples": {
                        "schema_name": "buffering_hard_commit_cell_arc_delay_samples",
                        "schema_version": 1,
                        "status": "ok",
                        "liberty_arc_sample_count": 2,
                    },
                },
            },
        )

        self.assertEqual(
            measured["hard_commit_diagnostic"]["schema_name"],
            "buffering_hard_commit_diagnostic",
        )
        self.assertEqual(measured["hard_commit_diagnostic_status"], "ok")
        self.assertEqual(
            measured["hard_commit_topology_delta"]["instance_count_delta"],
            1,
        )
        self.assertEqual(
            measured["hard_commit_load_partition"]["moved_load_pin_names"],
            ["u0/A", "u1/A"],
        )
        self.assertEqual(
            measured["hard_commit_cell_arc_delay_samples"][
                "liberty_arc_sample_count"
            ],
            2,
        )

    def test_measured_results_artifact_counts_before_after_records(self):
        artifact = build_coordinate_measured_results_artifact(
            [self._action()],
            [
                {
                    "status": "rejected",
                    "backend_status": "ok",
                    "reject_reason": "wns_regression",
                    "before_metrics": {"wns": -1.5, "tns": -20.0},
                    "after_metrics": {"wns": -1.7, "tns": -22.0},
                    "rollback_status": "not_needed_disposable_design",
                }
            ],
            disposable_design_path="/tmp/buffering-disposable",
        )

        self.assertEqual(artifact["artifact"], "buffering_coordinate_commit_results")
        self.assertEqual(artifact["coordinate_commit_attempt_count"], 1)
        self.assertEqual(artifact["measured_before_after_count"], 1)
        self.assertEqual(artifact["accepted_action_count"], 0)
        self.assertEqual(artifact["rejected_action_count"], 1)
        self.assertEqual(artifact["tns_improved_action_count"], 0)
        self.assertEqual(artifact["tns_regressed_action_count"], 1)
        self.assertEqual(artifact["accepted_tns_regressed_action_count"], 0)
        self.assertEqual(artifact["results"][0]["reject_reason"], "wns_regression")

    def test_measured_results_artifact_flags_accepted_tns_regression(self):
        artifact = build_coordinate_measured_results_artifact(
            [self._action()],
            [
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "actual_delta_wns": 0.0,
                    "actual_delta_tns": -4.0,
                    "before_metrics": {"wns": -1.5, "tns": -20.0},
                    "after_metrics": {"wns": -1.5, "tns": -24.0},
                }
            ],
            disposable_design_path="/tmp/buffering-disposable",
        )

        self.assertEqual(artifact["accepted_action_count"], 1)
        self.assertEqual(artifact["tns_improved_action_count"], 0)
        self.assertEqual(artifact["tns_regressed_action_count"], 1)
        self.assertEqual(artifact["accepted_tns_regressed_action_count"], 1)
        self.assertEqual(artifact["accepted_tns_delta_sum"], -4.0)
        self.assertEqual(artifact["qor_status"], "fail_tns_nonpositive")
        self.assertFalse(artifact["qor_pass"])

        summary = _commit_summary_from_artifact(
            artifact,
            commit_enabled=True,
            artifact_path="/tmp/commit.json",
        )

        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(summary["transaction_status"], "accepted")
        self.assertEqual(summary["qor_status"], "fail_tns_nonpositive")
        self.assertFalse(summary["qor_pass"])
        self.assertEqual(summary["accepted_tns_regressed_action_count"], 1)

    def test_measured_action_result_preserves_balanced_acceptance_score(self):
        measured = build_measured_coordinate_action_result(
            self._action(),
            {
                "status": "accepted",
                "backend_status": "ok",
                "accepted": True,
                "accept_policy": "balanced_setup_slew_cap",
                "balanced_acceptance_score": 29.0,
                "balanced_acceptance_min_score": 0.0,
                "balanced_acceptance_breakdown": {
                    "setup_tns_delta": -10.0,
                    "slew_violation_reduction": 2.0,
                    "cap_violation_reduction": 0.0,
                    "inserted_buffer_count_delta": 1.0,
                },
                "before_metrics": {"wns": -1.5, "tns": -20.0},
                "after_metrics": {"wns": -1.5, "tns": -30.0},
            },
        )

        self.assertEqual(measured["accept_policy"], "balanced_setup_slew_cap")
        self.assertEqual(measured["balanced_acceptance_score"], 29.0)
        self.assertEqual(measured["balanced_acceptance_min_score"], 0.0)
        self.assertEqual(
            measured["balanced_acceptance_breakdown"]["slew_violation_reduction"],
            2.0,
        )

    def test_measured_action_result_preserves_drv_violation_deltas(self):
        measured = build_measured_coordinate_action_result(
            self._action(),
            {
                "status": "accepted",
                "backend_status": "ok",
                "accepted": True,
                "before_metrics": {
                    "wns": 0.0,
                    "tns": 0.0,
                    "slew_violation_count": 276,
                    "cap_violation_count": 3,
                },
                "after_metrics": {
                    "wns": 0.0,
                    "tns": 0.0,
                    "slew_violation_count": 270,
                    "cap_violation_count": 4,
                },
                "slew_violation_count_delta": -6.0,
                "cap_violation_count_delta": 1.0,
            },
        )

        self.assertEqual(measured["pre_slew_violation_count"], 276.0)
        self.assertEqual(measured["post_slew_violation_count"], 270.0)
        self.assertEqual(measured["actual_delta_slew_violation_count"], -6.0)
        self.assertEqual(measured["pre_cap_violation_count"], 3.0)
        self.assertEqual(measured["post_cap_violation_count"], 4.0)
        self.assertEqual(measured["actual_delta_cap_violation_count"], 1.0)

    def test_measured_results_artifact_summarizes_drv_violation_deltas(self):
        artifact = build_coordinate_measured_results_artifact(
            [self._action(), self._action(action_id=4)],
            [
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "before_metrics": {
                        "wns": 0.0,
                        "tns": 0.0,
                        "slew_violation_count": 10,
                        "cap_violation_count": 1,
                    },
                    "after_metrics": {
                        "wns": 0.0,
                        "tns": 0.0,
                        "slew_violation_count": 7,
                        "cap_violation_count": 1,
                    },
                    "slew_violation_count_delta": -3.0,
                    "cap_violation_count_delta": 0.0,
                },
                {
                    "status": "rejected",
                    "backend_status": "ok",
                    "reject_reason": "wns_regression",
                    "before_metrics": {
                        "wns": 0.0,
                        "tns": 0.0,
                        "slew_violation_count": 7,
                        "cap_violation_count": 1,
                    },
                    "after_metrics": {
                        "wns": -0.1,
                        "tns": -1.0,
                        "slew_violation_count": 8,
                        "cap_violation_count": 0,
                    },
                    "slew_violation_count_delta": 1.0,
                    "cap_violation_count_delta": -1.0,
                },
            ],
        )

        self.assertEqual(artifact["slew_violation_improved_action_count"], 1)
        self.assertEqual(artifact["slew_violation_regressed_action_count"], 1)
        self.assertEqual(artifact["cap_violation_improved_action_count"], 1)
        self.assertEqual(artifact["cap_violation_regressed_action_count"], 0)
        self.assertEqual(artifact["accepted_slew_violation_delta_sum"], -3.0)
        self.assertEqual(artifact["accepted_cap_violation_delta_sum"], 0.0)

    def test_batch_measured_results_artifact_uses_single_batch_before_after(self):
        artifact = build_coordinate_batch_measured_results_artifact(
            [self._action(action_id=3), self._action(action_id=4)],
            [
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "inserted_buffer_count_delta": 1,
                },
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "inserted_buffer_count_delta": 1,
                },
            ],
            before_metrics={"wns": -1.5, "tns": -20.0},
            after_metrics={"wns": -1.4, "tns": -17.0},
            disposable_design_path="/tmp/buffering-batch",
        )

        self.assertEqual(artifact["artifact"], "buffering_coordinate_commit_results")
        self.assertEqual(artifact["coordinate_commit_measurement_mode"], "single_batch_before_after")
        self.assertEqual(artifact["coordinate_commit_attempt_count"], 2)
        self.assertEqual(artifact["measured_before_after_count"], 1)
        self.assertEqual(artifact["accepted_action_count"], 2)
        self.assertEqual(artifact["pre_wns"], -1.5)
        self.assertEqual(artifact["post_wns"], -1.4)
        self.assertEqual(artifact["actual_delta_tns"], 3.0)
        self.assertEqual(artifact["batch_before_metrics"]["tns"], -20.0)
        self.assertEqual(artifact["batch_after_metrics"]["tns"], -17.0)
        self.assertFalse(artifact["per_action_sta_loop"])
        self.assertFalse(artifact["results"][0]["measured_before_after"])
        self.assertEqual(artifact["results"][0]["measurement_source"], "batch_commit")

    def test_batch_measured_results_artifact_fails_qor_on_wns_regression(self):
        artifact = build_coordinate_batch_measured_results_artifact(
            [self._action(action_id=3), self._action(action_id=4)],
            [
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "inserted_buffer_count_delta": 1,
                },
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "inserted_buffer_count_delta": 1,
                },
            ],
            before_metrics={"wns": -1.5, "tns": -20.0},
            after_metrics={"wns": -1.7, "tns": -17.0},
        )

        self.assertEqual(artifact["accepted_action_count"], 2)
        self.assertEqual(artifact["actual_delta_tns"], 3.0)
        self.assertAlmostEqual(artifact["actual_delta_wns"], -0.2)
        self.assertEqual(artifact["qor_status"], "fail_wns_regression")
        self.assertFalse(artifact["qor_pass"])

        summary = _commit_summary_from_artifact(
            artifact,
            commit_enabled=True,
            artifact_path="/tmp/commit.json",
        )
        self.assertEqual(summary["transaction_status"], "accepted")
        self.assertEqual(summary["qor_status"], "fail_wns_regression")
        self.assertFalse(summary["qor_pass"])

    def test_commit_coordinate_actions_defaults_to_single_batch_before_after(self):
        class FakeBridge:
            def __init__(self):
                self.metrics_calls = 0
                self.insert_configs = []
                self.tcl_commands = []

            def query_diff_guided_batch_timing_metrics(self):
                self.metrics_calls += 1
                if self.metrics_calls == 1:
                    return {"status": "ok", "wns": -1.5, "tns": -20.0}
                return {"status": "ok", "wns": -1.25, "tns": -11.0}

            def run_coordinate_buffer_insert(self, action, config):
                self.insert_configs.append(dict(config))
                return {
                    "status": "ok",
                    "inserted_buffer_count_delta": 1,
                    "instance_count_delta": 1,
                    "net_count_delta": 1,
                }

            def eval_tcl_string(self, command):
                self.tcl_commands.append(command)
                return ""

        bridge = FakeBridge()
        with self.subTest("commit_summary"):
            summary = commit_coordinate_buffer_actions(
                [self._action(action_id=3), self._action(action_id=4)],
                openroad_bridge=bridge,
            )

            self.assertEqual(summary["status"], "accepted")
            self.assertEqual(summary["transaction_status"], "accepted")
            self.assertEqual(summary["qor_status"], "pass")
            self.assertEqual(summary["accepted_action_count"], 2)
            self.assertEqual(summary["delta_tns"], 9.0)

        self.assertEqual(bridge.metrics_calls, 2)
        self.assertEqual(bridge.tcl_commands, ["estimate_parasitics -placement"])
        self.assertEqual(len(bridge.insert_configs), 2)
        self.assertTrue(all(config["defer_timing_update"] for config in bridge.insert_configs))
        self.assertTrue(
            all(config["load_partition_mode"] == "branch_downstream" for config in bridge.insert_configs)
        )

    def test_commit_coordinate_actions_orders_same_net_segments_root_to_leaf(self):
        class FakeBridge:
            def __init__(self):
                self.metrics_calls = 0
                self.action_ids = []

            def query_diff_guided_batch_timing_metrics(self):
                self.metrics_calls += 1
                return {"status": "ok", "wns": -1.0, "tns": -10.0 + self.metrics_calls}

            def run_coordinate_buffer_insert(self, action, config):
                self.action_ids.append(
                    (
                        action["action_id"],
                        action["commit_input_order"],
                        action["commit_topology_order"],
                    )
                )
                return {
                    "status": "ok",
                    "inserted_buffer_count_delta": 1,
                    "instance_count_delta": 1,
                    "net_count_delta": 1,
                }

            def eval_tcl_string(self, command):
                return ""

        bridge = FakeBridge()
        leaf = self._action(action_id=9)
        leaf.update(
            {
                "net_name": "tree_net",
                "segment_parent_node_id": 4,
                "segment_child_node_id": 5,
                "segment_tree_depth": 2,
                "segment_split_index": 0,
                "segment_split_ratio": 0.5,
            }
        )
        root = self._action(action_id=3)
        root.update(
            {
                "net_name": "tree_net",
                "segment_parent_node_id": 1,
                "segment_child_node_id": 4,
                "segment_tree_depth": 0,
                "segment_split_index": 0,
                "segment_split_ratio": 0.5,
            }
        )

        summary = commit_coordinate_buffer_actions(
            [leaf, root],
            openroad_bridge=bridge,
        )

        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(bridge.action_ids, [(3, 1, 0), (9, 0, 1)])

    def test_commit_coordinate_actions_orders_candidate_superset_before_subset(self):
        class FakeBridge:
            def __init__(self):
                self.action_ids = []

            def query_diff_guided_batch_timing_metrics(self):
                return {"status": "ok", "wns": -1.0, "tns": -10.0}

            def run_coordinate_buffer_insert(self, action, config):
                self.action_ids.append(
                    (
                        action["action_id"],
                        action["commit_input_order"],
                        action["commit_topology_order"],
                    )
                )
                return {
                    "status": "ok",
                    "inserted_buffer_count_delta": 1,
                    "instance_count_delta": 1,
                    "net_count_delta": 1,
                }

            def eval_tcl_string(self, command):
                return ""

        subset = self._action(
            action_id=3,
            net_name="tree_net",
            downstream_pin_names=["u0/A", "u1/A"],
        )
        superset = self._action(
            action_id=9,
            net_name="tree_net",
            downstream_pin_names=["u0/A", "u1/A", "u2/A"],
        )
        bridge = FakeBridge()

        summary = commit_coordinate_buffer_actions(
            [subset, superset],
            openroad_bridge=bridge,
        )

        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(bridge.action_ids, [(9, 1, 0), (3, 0, 1)])

    def test_commit_coordinate_actions_keeps_explicit_per_action_diagnostic_mode(self):
        class FakeBridge:
            def __init__(self):
                self.metrics_calls = 0
                self.insert_configs = []
                self.tcl_commands = []

            def query_diff_guided_batch_timing_metrics(self):
                self.metrics_calls += 1
                return {"status": "ok", "wns": 0.0, "tns": 0.0}

            def run_coordinate_buffer_insert(self, action, config):
                self.insert_configs.append(dict(config))
                return {
                    "status": "ok",
                    "before_metrics": {"wns": -2.0, "tns": -30.0},
                    "after_metrics": {"wns": -1.9, "tns": -25.0},
                    "actual_delta_wns": 0.1,
                    "actual_delta_tns": 5.0,
                    "inserted_buffer_count_delta": 1,
                }

            def eval_tcl_string(self, command):
                self.tcl_commands.append(command)
                return ""

        bridge = FakeBridge()
        summary = commit_coordinate_buffer_actions(
            [self._action(action_id=3)],
            openroad_bridge=bridge,
            realization_config={"per_action_sta_loop": True},
        )

        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(summary["qor_status"], "pass")
        self.assertEqual(summary["delta_tns"], 5.0)
        self.assertEqual(bridge.metrics_calls, 0)
        self.assertEqual(bridge.tcl_commands, [])
        self.assertEqual(len(bridge.insert_configs), 1)
        self.assertFalse(bridge.insert_configs[0]["defer_timing_update"])

    def test_commit_summary_uses_full_sequential_measured_chain(self):
        artifact = build_coordinate_measured_results_artifact(
            [self._action(action_id=3), self._action(action_id=4)],
            [
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "before_metrics": {"wns": -2.0, "tns": -100.0},
                    "after_metrics": {"wns": -1.8, "tns": -70.0},
                },
                {
                    "status": "accepted",
                    "backend_status": "ok",
                    "accepted": True,
                    "before_metrics": {"wns": -1.8, "tns": -70.0},
                    "after_metrics": {"wns": -1.5, "tns": -40.0},
                },
            ],
        )

        summary = _commit_summary_from_artifact(
            artifact,
            commit_enabled=True,
            artifact_path="/tmp/commit.json",
        )

        self.assertEqual(summary["before_tns"], -100.0)
        self.assertEqual(summary["after_tns"], -40.0)
        self.assertEqual(summary["delta_tns"], 60.0)
        self.assertEqual(summary["before_wns"], -2.0)
        self.assertEqual(summary["after_wns"], -1.5)
        self.assertAlmostEqual(summary["delta_wns"], 0.5)
        self.assertEqual(summary["qor_status"], "pass")
        self.assertTrue(summary["qor_pass"])

    def test_commit_action_payload_carries_buffer_main_type_index(self):
        state = VirtualBufferState()
        state.insert(
            {
                "candidate_id": 4,
                "action_kind": "buffer_insert",
                "net_name": "net4",
                "buffer_main_type_index": 9,
                "driver_pin_name": "drv/Y",
                "load_pin_name": "u0/A",
                "x_dbu": 100.0,
                "y_dbu": 200.0,
                "x_um": 0.1,
                "y_um": 0.2,
            },
            bsu=0,
        )

        artifact = build_commit_action_candidates(
            state,
            legal_table={
                "legal_cell_ids": [21],
                "legal_master_names": ["BUF_X1"],
                "legal_size_values": [1.0],
            },
        )

        action = artifact["actions"][0]
        self.assertEqual(action["buffer_main_type_index"], 9)
        contract = build_coordinate_backend_contract([action])
        self.assertEqual(contract["payload_ready_count"], 1)

    def test_commit_action_payload_carries_native_physical_slew_and_load_deltas(self):
        state = VirtualBufferState()
        state.insert(
            {
                "candidate_id": 4,
                "action_kind": "buffer_insert",
                "net_name": "net4",
                "buffer_main_type_index": 9,
                "driver_pin_name": "drv/Y",
                "load_pin_name": "u0/A",
                "x_dbu": 100.0,
                "y_dbu": 200.0,
                "physical_slew_delta_obj": -1.25,
                "physical_load_delta_obj": -2.5,
                "baseline_sink_slew_obj": 3.0,
                "trial_sink_slew_obj": 1.75,
                "baseline_sink_load_obj": 7.0,
                "trial_sink_load_obj": 4.5,
            },
            bsu=0,
        )

        artifact = build_commit_action_candidates(
            state,
            legal_table={
                "legal_cell_ids": [21],
                "legal_master_names": ["BUF_X1"],
                "legal_size_values": [1.0],
            },
        )

        action = artifact["actions"][0]
        self.assertEqual(action["physical_slew_delta_obj"], -1.25)
        self.assertEqual(action["physical_load_delta_obj"], -2.5)
        self.assertEqual(action["baseline_sink_slew_obj"], 3.0)
        self.assertEqual(action["trial_sink_slew_obj"], 1.75)
        self.assertEqual(action["baseline_sink_load_obj"], 7.0)
        self.assertEqual(action["trial_sink_load_obj"], 4.5)

    def test_measured_result_carries_predicted_timing_deltas(self):
        action = dict(
            self._action(),
            predicted_delta_obj=-12.5,
            predicted_delta_tns=12.5,
            predicted_delta_wns=0.25,
            selection_score=12.5,
        )

        result = build_measured_coordinate_action_result(
            action,
            {
                "status": "accepted",
                "before_metrics": {"wns": -1.5, "tns": -20.0},
                "after_metrics": {"wns": -1.3, "tns": -11.0},
            },
        )

        self.assertEqual(result["proxy_predicted_delta_obj"], -12.5)
        self.assertEqual(result["proxy_predicted_delta_tns"], 12.5)
        self.assertEqual(result["proxy_predicted_delta_wns"], 0.25)
        self.assertEqual(result["proxy_selection_score"], 12.5)
        self.assertEqual(result["actual_delta_tns"], 9.0)

    def test_commit_action_payload_carries_native_gradient_chain_provenance(self):
        state = VirtualBufferState()
        state.insert(
            {
                "candidate_id": 4,
                "action_kind": "buffer_insert",
                "net_name": "net4",
                "buffer_main_type_index": 9,
                "driver_pin_name": "drv/Y",
                "load_pin_name": "u0/A",
                "x_dbu": 100.0,
                "y_dbu": 200.0,
                "scoring_node_id": 17,
                "scoring_node_source": "tree_node",
                "scoring_location_model": "exact_tree_node",
                "scoring_location_is_exact": True,
                "candidate_input_slew": 1.25,
                "candidate_output_cap": 4.5,
                "buffer_input_cap": 0.5,
                "buffer_delay": 1.0,
                "buffer_output_slew": 0.2,
                "delay_source": "lut_bilinear_interpolation",
                "transition_source": "lut_bilinear_interpolation",
                "delay_lut_status": {
                    "source": "unit_delay_lut",
                    "input_slew_clamped": True,
                    "output_cap_clamped": False,
                },
                "transition_lut_status": {
                    "source": "unit_transition_lut",
                    "input_slew_clamped": False,
                    "output_cap_clamped": True,
                },
                "baseline_obj": 10.0,
                "trial_obj": 8.0,
                "gradient_chain": {
                    "baseline_provider": "net_subgraph_timing_cpp",
                    "coordinate_source": "baseline_native_slew_in_and_lout",
                    "surrogate_provider": "buffer_surrogate",
                    "trial_provider": "net_subgraph_timing_cpp",
                    "objective": "max_sink_arrival_delta",
                },
            },
            bsu=0,
        )

        artifact = build_commit_action_candidates(
            state,
            legal_table={
                "legal_cell_ids": [21],
                "legal_master_names": ["BUF_X1"],
                "legal_size_values": [1.0],
            },
        )

        action = artifact["actions"][0]
        self.assertEqual(action["scoring_node_id"], 17)
        self.assertEqual(action["scoring_location_model"], "exact_tree_node")
        self.assertTrue(action["scoring_location_is_exact"])
        self.assertEqual(action["candidate_input_slew"], 1.25)
        self.assertEqual(action["candidate_output_cap"], 4.5)
        self.assertEqual(action["buffer_delay"], 1.0)
        self.assertEqual(action["delay_lut_status"]["source"], "unit_delay_lut")
        self.assertTrue(action["delay_lut_status"]["input_slew_clamped"])
        self.assertEqual(
            action["transition_lut_status"]["source"],
            "unit_transition_lut",
        )
        self.assertTrue(action["transition_lut_status"]["output_cap_clamped"])
        self.assertEqual(action["baseline_obj"], 10.0)
        self.assertEqual(action["trial_obj"], 8.0)
        self.assertEqual(
            action["gradient_chain"]["baseline_provider"],
            "net_subgraph_timing_cpp",
        )
        self.assertEqual(
            action["gradient_chain"]["surrogate_provider"],
            "buffer_surrogate",
        )

    def test_placeio_wrapper_calls_coordinate_buffer_insert_bridge(self):
        from dreamplace.ops.placeio_openroad.place_io import PlaceIOFunction

        class FakeRawDb:
            def __init__(self):
                self.calls = []

            def run_coordinate_buffer_insert(self, action, config):
                self.calls.append((dict(action), dict(config)))
                return {
                    "status": "unsupported",
                    "realization_backend": "openroad_coordinate_buffer_insert",
                }

        raw_db = FakeRawDb()
        result = PlaceIOFunction.run_coordinate_buffer_insert(
            raw_db,
            self._action(),
            {"allow_singleton_load_partition": True},
        )

        self.assertEqual(result["realization_backend"], "openroad_coordinate_buffer_insert")
        self.assertEqual(raw_db.calls[0][0]["net_name"], "net3")
        self.assertTrue(raw_db.calls[0][1]["allow_singleton_load_partition"])

    def test_real_openroad_coordinate_buffer_insert_wire9_smoke(self):
        if os.environ.get("DREAMPLACE_BUFFERING_REAL_COORDINATE_BUFFER_SMOKE") != "1":
            self.skipTest(
                "set DREAMPLACE_BUFFERING_REAL_COORDINATE_BUFFER_SMOKE=1 to run real coordinate smoke"
            )

        from dreamplace.ops.placeio_openroad.place_io import PlaceIOFunction

        openroad_root = Path(
            "/nfs/share/home/zhaoxueyan/ord-buffer-benchmark/"
            "AiEDA/third_party/AutoDMP/thirdparty/OpenROAD"
        )
        required = [
            openroad_root / "test/Nangate45/Nangate45.lef",
            openroad_root / "test/Nangate45/Nangate45_typ.lib",
            openroad_root / "test/Nangate45/Nangate45.rc",
            openroad_root / "src/rsz/test/repair_wire9.def",
        ]
        for path in required:
            if not path.exists():
                self.skipTest("missing OpenROAD repair_wire9 smoke input: %s" % path)

        raw_db = PlaceIOFunction.read(
            SimpleNamespace(
                design_inputs={
                    "lef": [str(openroad_root / "test/Nangate45/Nangate45.lef")],
                    "lib": [str(openroad_root / "test/Nangate45/Nangate45_typ.lib")],
                    "def": str(openroad_root / "src/rsz/test/repair_wire9.def"),
                },
                place_io_engine="openroad",
            )
        )
        raw_db.eval_tcl_string(
            "source {%s}" % (openroad_root / "test/Nangate45/Nangate45.rc")
        )
        raw_db.eval_tcl_string("set_wire_rc -layer metal3")
        raw_db.eval_tcl_string("estimate_parasitics -placement")

        result = PlaceIOFunction.run_coordinate_buffer_insert(
            raw_db,
            {
                "action_id": 91,
                "action_kind": "buffer_insert",
                "net_name": "out1",
                "driver_pin_name": "u1/Z",
                "load_pin_name": "out1",
                "buffer_main_type_index": 0,
                "bsu": 0,
                "buffer_master_name": "BUF_X1",
                "candidate_location_x_dbu": 1000,
                "candidate_location_y_dbu": 1000,
            },
            {"allow_singleton_load_partition": True},
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["realization_backend"], "openroad_coordinate_buffer_insert")
        self.assertEqual(result["implementation_scope"], "flat_singleton_load_partition")
        self.assertEqual(result["net_name"], "out1")
        self.assertEqual(result["buffer_master_name"], "BUF_X1")
        self.assertGreater(result["inserted_buffer_count_delta"], 0)
        self.assertGreater(result["instance_count_delta"], 0)
        self.assertGreater(result["net_count_delta"], 0)
        self.assertTrue(result["requires_sync_back"])
        self.assertTrue(result["requires_runtimedb_rebuild"])
        self.assertIn("inserted_buffer_name", result)
        self.assertIn("downstream_net_name", result)
        self.assertIn("before_metrics", result)
        self.assertIn("after_metrics", result)

    def test_real_openroad_coordinate_buffer_insert_remaps_split_load_net(self):
        if os.environ.get("DREAMPLACE_BUFFERING_REAL_COORDINATE_BUFFER_SMOKE") != "1":
            self.skipTest(
                "set DREAMPLACE_BUFFERING_REAL_COORDINATE_BUFFER_SMOKE=1 to run real coordinate smoke"
            )

        from dreamplace.ops.placeio_openroad.place_io import PlaceIOFunction

        openroad_root = Path(
            "/nfs/share/home/zhaoxueyan/ord-buffer-benchmark/"
            "AiEDA/third_party/AutoDMP/thirdparty/OpenROAD"
        )
        required = [
            openroad_root / "test/Nangate45/Nangate45.lef",
            openroad_root / "test/Nangate45/Nangate45_typ.lib",
            openroad_root / "test/Nangate45/Nangate45.rc",
            openroad_root / "src/rsz/test/repair_wire9.def",
        ]
        for path in required:
            if not path.exists():
                self.skipTest("missing OpenROAD repair_wire9 smoke input: %s" % path)

        raw_db = PlaceIOFunction.read(
            SimpleNamespace(
                design_inputs={
                    "lef": [str(openroad_root / "test/Nangate45/Nangate45.lef")],
                    "lib": [str(openroad_root / "test/Nangate45/Nangate45_typ.lib")],
                    "def": str(openroad_root / "src/rsz/test/repair_wire9.def"),
                },
                place_io_engine="openroad",
            )
        )
        raw_db.eval_tcl_string("source {%s}" % (openroad_root / "test/Nangate45/Nangate45.rc"))
        raw_db.eval_tcl_string("set_wire_rc -layer metal3")
        raw_db.eval_tcl_string("estimate_parasitics -placement")
        action = {
            "action_kind": "buffer_insert",
            "net_name": "out1",
            "driver_pin_name": "u1/Z",
            "load_pin_name": "out1",
            "buffer_main_type_index": 0,
            "bsu": 0,
            "buffer_master_name": "BUF_X1",
            "candidate_location_x_dbu": 1000,
            "candidate_location_y_dbu": 1000,
        }
        first = PlaceIOFunction.run_coordinate_buffer_insert(
            raw_db,
            {**action, "action_id": 91},
            {"allow_singleton_load_partition": True},
        )
        second = PlaceIOFunction.run_coordinate_buffer_insert(
            raw_db,
            {**action, "action_id": 92, "candidate_location_x_dbu": 2000},
            {"allow_singleton_load_partition": True},
        )

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertTrue(second["target_net_remapped"])
        self.assertTrue(second["driver_pin_remapped"])


if __name__ == "__main__":
    unittest.main()
