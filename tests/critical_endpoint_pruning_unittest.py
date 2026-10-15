#!/usr/bin/env python

import unittest
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dreamplace.ops.timing_propagation.critical_endpoint_pruning import (
    DynamicCriticalEndpointSelector,
)
from dreamplace.ops.timing_propagation import timing_propagation as timing_propagation_module
from dreamplace.ops.timing_propagation.timing_propagation import (
    TimingPropagation,
)


class FakeTraversalPruningResult:
    def __init__(
        self,
        kept_flat_arc_indices,
        kept_level_offsets,
        kept_counts_by_level,
        active_inst_count=0,
        preparation_runtime_ms=0.25,
    ):
        self.kept_flat_arc_indices = torch.tensor(
            kept_flat_arc_indices,
            dtype=torch.long,
        )
        self.kept_level_offsets = torch.tensor(
            kept_level_offsets,
            dtype=torch.long,
        )
        self.kept_counts_by_level = torch.tensor(
            kept_counts_by_level,
            dtype=torch.long,
        )
        self.active_inst_count = active_inst_count
        self.active_arc_count = len(kept_flat_arc_indices)
        self.dropped_arc_count = 3 - len(kept_flat_arc_indices)
        self.preparation_runtime_ms = preparation_runtime_ms


class FakeTraversalPruner:
    def __init__(self):
        self.calls = []

    def refresh(self, active_endpoint_ids):
        self.calls.append(active_endpoint_ids.detach().cpu().tolist())
        return FakeTraversalPruningResult(
            kept_flat_arc_indices=[0, 2],
            kept_level_offsets=[0, 1, 2],
            kept_counts_by_level=[1, 1],
            active_inst_count=2,
        )


class RecordingTraversalPruner:
    instances = []

    def __init__(self, *args):
        self.arg_count = len(args)
        RecordingTraversalPruner.instances.append(self)

    def refresh(self, active_endpoint_ids):
        return FakeTraversalPruningResult(
            kept_flat_arc_indices=[0],
            kept_level_offsets=[0, 1],
            kept_counts_by_level=[1],
            active_inst_count=1,
        )


class FakeTimingPropagationCpp:
    CriticalEndpointTraversalPruner = RecordingTraversalPruner


class DynamicCriticalEndpointSelectorTest(unittest.TestCase):
    def test_selects_top_k_worst_slack_endpoints(self):
        selector = DynamicCriticalEndpointSelector(
            top_k=2,
            slack_window_ps=0.0,
            refresh_interval=1,
            hysteresis_interval=0,
            full_refresh_interval=10,
        )

        active, stats = selector.refresh(
            iteration=1,
            endpoint_ids=[10, 11, 12, 13],
            endpoint_slack_ps=[-5.0, -20.0, -1.0, -10.0],
        )

        self.assertEqual(active, {11, 13})
        self.assertEqual(stats["active_endpoint_count"], 2)
        self.assertEqual(stats["selected_by_top_k_count"], 2)

    def test_slack_window_is_union_with_top_k(self):
        selector = DynamicCriticalEndpointSelector(
            top_k=1,
            slack_window_ps=4.0,
            refresh_interval=1,
            hysteresis_interval=0,
            full_refresh_interval=10,
        )

        active, stats = selector.refresh(
            iteration=1,
            endpoint_ids=[20, 21, 22, 23],
            endpoint_slack_ps=[-10.0, -8.0, -6.0, -1.0],
        )

        self.assertEqual(active, {20, 21, 22})
        self.assertEqual(stats["selected_by_top_k_count"], 1)
        self.assertEqual(stats["selected_by_slack_window_count"], 3)
        self.assertAlmostEqual(stats["full_endpoint_wns_ps"], -10.0)
        self.assertAlmostEqual(stats["full_endpoint_tns_ps"], -25.0)

    def test_hysteresis_retains_recently_active_endpoints(self):
        selector = DynamicCriticalEndpointSelector(
            top_k=1,
            slack_window_ps=0.0,
            refresh_interval=1,
            hysteresis_interval=2,
            full_refresh_interval=10,
        )

        active1, stats1 = selector.refresh(
            iteration=1,
            endpoint_ids=[30, 31],
            endpoint_slack_ps=[-10.0, -2.0],
        )
        active2, stats2 = selector.refresh(
            iteration=2,
            endpoint_ids=[30, 31],
            endpoint_slack_ps=[-1.0, -9.0],
        )
        active4, stats4 = selector.refresh(
            iteration=4,
            endpoint_ids=[30, 31],
            endpoint_slack_ps=[-1.0, -9.0],
        )

        self.assertEqual(active1, {30})
        self.assertEqual(active2, {30, 31})
        self.assertEqual(stats2["selected_by_hysteresis_count"], 1)
        self.assertEqual(active4, {31})
        self.assertEqual(stats4["selected_by_hysteresis_count"], 0)

    def test_overlap_stats_are_deterministic(self):
        selector = DynamicCriticalEndpointSelector(
            top_k=2,
            slack_window_ps=3.0,
            refresh_interval=1,
            hysteresis_interval=0,
            full_refresh_interval=10,
        )

        active1, stats1 = selector.refresh(
            iteration=1,
            endpoint_ids=[40, 41, 42, 43],
            endpoint_slack_ps=[-10.0, -8.0, -7.0, -1.0],
        )
        active2, stats2 = selector.refresh(
            iteration=2,
            endpoint_ids=[40, 41, 42, 43],
            endpoint_slack_ps=[-9.0, -2.0, -8.0, -1.0],
        )

        self.assertEqual(active1, {40, 41, 42})
        self.assertEqual(active2, {40, 42})
        self.assertEqual(stats1["previous_active_endpoint_count"], 0)
        self.assertEqual(stats2["previous_active_endpoint_count"], 3)
        self.assertEqual(stats2["active_overlap_count"], 2)
        self.assertAlmostEqual(stats2["active_overlap_fraction"], 2.0 / 3.0)


class TimingPropagationCriticalEndpointPruningTest(unittest.TestCase):
    def _make_shell(self, mode="dynamic"):
        shell = object.__new__(TimingPropagation)
        shell.critical_endpoint_pruning_mode = mode
        shell.critical_endpoint_top_k = 2
        shell.critical_endpoint_slack_window_ps = 0.0
        shell.critical_endpoint_refresh_interval = 10
        shell.critical_endpoint_hysteresis_interval = 0
        shell.critical_endpoint_full_refresh_interval = 0
        shell.critical_endpoint_selector = None
        shell.end_points = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        shell.last_endpoint_slack_tensor = torch.tensor([-5.0, -20.0, -1.0, -10.0])
        shell.last_endpoint_ids_tensor = shell.end_points.detach().clone()
        shell.last_active_endpoint_ids = []
        shell.last_critical_endpoint_pruning_stats = {}
        shell.traversal_pruning_refresh_interval = 10
        shell._traversal_pruner = None
        shell._traversal_pruner_topology_source = None
        shell.last_traversal_pruning_result = None
        shell.last_traversal_pruning_iteration = None
        shell.last_traversal_pruning_stats = shell._empty_traversal_pruning_stats(
            iteration=None,
            reason="not_run",
        )
        return shell

    def _install_static_pruner_inputs(self, shell):
        shell.flat_inst_arcs_by_level = torch.tensor(
            [[0, 1, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        shell.flat_inst_arcs_by_level_start = torch.tensor([0, 1], dtype=torch.long)
        shell.start_points = torch.tensor([0], dtype=torch.long)
        shell.pin2node_map = torch.tensor([0, 0], dtype=torch.long)
        shell.flat_pin_to_graph_reverse = torch.tensor([0], dtype=torch.long)
        shell.flat_pin_to_graph_start_reverse = torch.tensor([0, 0, 1], dtype=torch.long)
        shell.pin_pair_arc_keys = torch.tensor([[0, 1]], dtype=torch.long)
        shell.flat_pin_pair_arc_start = torch.tensor([0, 1], dtype=torch.long)
        shell.flat_pin_pair_arc_indices = torch.tensor([0], dtype=torch.long)

    def test_ensure_traversal_pruner_prefers_compact_topology(self):
        shell = self._make_shell()
        self._install_static_pruner_inputs(shell)
        shell.pin_pred_start = torch.tensor([0, 0, 1], dtype=torch.long)
        shell.pin_pred_pin = torch.tensor([0], dtype=torch.long)
        shell.pin_pred_arc_id = torch.tensor([0], dtype=torch.long)
        RecordingTraversalPruner.instances = []
        old_cpp = timing_propagation_module._tp_cpp
        timing_propagation_module._tp_cpp = FakeTimingPropagationCpp
        try:
            pruner = shell._ensure_traversal_pruner()
        finally:
            timing_propagation_module._tp_cpp = old_cpp

        self.assertIs(pruner, RecordingTraversalPruner.instances[-1])
        self.assertEqual(pruner.arg_count, 7)
        self.assertEqual(shell._traversal_pruner_topology_source, "compact_csr")

    def test_ensure_traversal_pruner_falls_back_to_pair_key_compat(self):
        shell = self._make_shell()
        self._install_static_pruner_inputs(shell)
        shell.pin_pred_start = None
        shell.pin_pred_pin = None
        shell.pin_pred_arc_id = None
        RecordingTraversalPruner.instances = []
        old_cpp = timing_propagation_module._tp_cpp
        timing_propagation_module._tp_cpp = FakeTimingPropagationCpp
        try:
            pruner = shell._ensure_traversal_pruner()
        finally:
            timing_propagation_module._tp_cpp = old_cpp

        self.assertIs(pruner, RecordingTraversalPruner.instances[-1])
        self.assertEqual(pruner.arg_count, 9)
        self.assertEqual(shell._traversal_pruner_topology_source, "pair_key_compat")

    def test_timing_propagation_refreshes_active_endpoints_from_endpoint_slack(self):
        shell = self._make_shell()

        shell.update_critical_endpoint_pruning_state(iteration=10)

        self.assertEqual(shell.last_active_endpoint_ids, [11, 13])
        self.assertEqual(
            shell.last_critical_endpoint_pruning_stats["selected_by_top_k_count"],
            2,
        )
        self.assertTrue(shell.last_critical_endpoint_pruning_stats["refreshed"])

    def test_timing_propagation_pruning_keeps_unmodeled_endpoint_slack_in_objective_set(self):
        shell = self._make_shell()
        shell.endpoints_constraint_arcs = torch.tensor(
            [
                [0, 10, 0, 0, 1],
                [1, 12, 0, 1, 1],
            ],
            dtype=torch.long,
        )

        shell.update_critical_endpoint_pruning_state(iteration=10)

        self.assertEqual(shell.last_active_endpoint_ids, [11, 13])


    def test_timing_propagation_builds_pruning_artifact_payload(self):
        shell = self._make_shell()
        shell.update_critical_endpoint_pruning_state(iteration=10)

        payload = shell.build_critical_endpoint_pruning_artifact(iteration=10)

        self.assertEqual(payload["artifact"], "critical_endpoint_pruning_latest")
        self.assertEqual(payload["iteration"], 10)
        self.assertEqual(payload["mode"], "dynamic")
        self.assertEqual(payload["active_endpoint_count"], 2)
        self.assertEqual(payload["active_endpoint_ids"], [11, 13])
        self.assertEqual(payload["full_endpoint_count"], 4)
        self.assertAlmostEqual(payload["selected_endpoint_wns_ps"], -20.0)
        self.assertAlmostEqual(payload["selected_endpoint_tns_ps"], -30.0)
        self.assertAlmostEqual(payload["full_endpoint_wns_ps"], -20.0)
        self.assertAlmostEqual(payload["full_endpoint_tns_ps"], -36.0)
        self.assertEqual(payload["refresh_interval"], 10)
        self.assertEqual(payload["refresh_decision"], "refreshed")
        self.assertEqual(payload["selection_provenance"]["owner"], "TimingPropagation")
        self.assertEqual(payload["selection_provenance"]["source"], "last_endpoint_slack_tensor")
        self.assertEqual(payload["selection_provenance"]["top_k"], 2)
        self.assertEqual(payload["selection_provenance"]["active_endpoint_id_limit"], 1024)
        self.assertEqual(payload["selector_stats"]["full_endpoint_count"], 4)
        self.assertTrue(payload["traversal_pruning_enabled"])
        self.assertEqual(payload["traversal_refresh_interval"], 10)
        self.assertEqual(payload["traversal_ordering"], "original_flat_topological_order")

    def test_timing_propagation_pruning_artifact_reports_reused_refresh(self):
        shell = self._make_shell()
        shell.update_critical_endpoint_pruning_state(iteration=10)
        shell.update_critical_endpoint_pruning_state(iteration=11)

        payload = shell.build_critical_endpoint_pruning_artifact(iteration=11)

        self.assertEqual(payload["refresh_decision"], "reused")
        self.assertEqual(payload["active_endpoint_count"], 2)
        self.assertAlmostEqual(payload["selected_endpoint_tns_ps"], -30.0)

    def test_timing_summary_filters_non_finite_endpoint_slack(self):
        shell = self._make_shell()
        shell.last_endpoint_ids_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        shell.last_endpoint_slack_tensor = torch.tensor(
            [float("nan"), -5.0, float("inf"), -2.0]
        )

        summary = shell._build_endpoint_pruning_timing_summary([2, 3])

        self.assertEqual(summary["full_endpoint_count"], 2)
        self.assertAlmostEqual(summary["full_endpoint_wns_ps"], -5.0)
        self.assertAlmostEqual(summary["full_endpoint_tns_ps"], -7.0)
        self.assertAlmostEqual(summary["selected_endpoint_wns_ps"], -5.0)
        self.assertAlmostEqual(summary["selected_endpoint_tns_ps"], -5.0)

    def test_timing_summary_returns_none_for_all_non_finite_endpoint_slack(self):
        shell = self._make_shell()
        shell.last_endpoint_ids_tensor = torch.tensor([1, 2], dtype=torch.long)
        shell.last_endpoint_slack_tensor = torch.tensor([float("nan"), float("inf")])

        summary = shell._build_endpoint_pruning_timing_summary([1, 2])

        self.assertEqual(summary["full_endpoint_count"], 0)
        self.assertIsNone(summary["full_endpoint_wns_ps"])
        self.assertIsNone(summary["full_endpoint_tns_ps"])
        self.assertIsNone(summary["selected_endpoint_wns_ps"])
        self.assertIsNone(summary["selected_endpoint_tns_ps"])

    def test_pruning_artifact_filters_non_finite_endpoint_slack(self):
        shell = self._make_shell()
        shell.last_endpoint_ids_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
        shell.last_endpoint_slack_tensor = torch.tensor([float("nan"), -5.0, float("inf")])
        shell.last_active_endpoint_ids = [2, 3]
        shell.last_critical_endpoint_pruning_stats = {
            "refreshed": True,
            "full_endpoint_count": 1,
            "full_endpoint_wns_ps": -5.0,
            "full_endpoint_tns_ps": -5.0,
        }

        payload = shell.build_critical_endpoint_pruning_artifact(iteration=12)

        self.assertEqual(payload["full_endpoint_count"], 1)
        self.assertAlmostEqual(payload["full_endpoint_wns_ps"], -5.0)
        self.assertAlmostEqual(payload["full_endpoint_tns_ps"], -5.0)
        self.assertAlmostEqual(payload["selected_endpoint_wns_ps"], -5.0)
        self.assertAlmostEqual(payload["selected_endpoint_tns_ps"], -5.0)

    def test_traversal_refresh_uses_cached_pruner_and_reports_metadata(self):
        shell = self._make_shell()
        fake_pruner = FakeTraversalPruner()
        shell._traversal_pruner = fake_pruner
        shell.last_active_endpoint_ids = [11, 13]
        shell.last_critical_endpoint_pruning_stats = {"refreshed": True}

        result = shell._refresh_traversal_pruning_state(iteration=10)

        self.assertIs(result, shell.last_traversal_pruning_result)
        self.assertEqual(fake_pruner.calls, [[11, 13]])
        self.assertTrue(shell.last_traversal_pruning_stats["enabled"])
        self.assertTrue(shell.last_traversal_pruning_stats["applied"])
        self.assertTrue(shell.last_traversal_pruning_stats["refreshed_this_iteration"])
        self.assertEqual(shell.last_traversal_pruning_stats["active_endpoint_count"], 2)
        self.assertEqual(shell.last_traversal_pruning_stats["active_inst_count"], 2)
        self.assertEqual(shell.last_traversal_pruning_stats["active_arc_count"], 2)
        self.assertEqual(shell.last_traversal_pruning_stats["dropped_arc_count"], 1)
        self.assertEqual(shell.last_traversal_pruning_stats["per_level_kept_counts"], [1, 1])

        shell.last_critical_endpoint_pruning_stats = {"refreshed": False}
        reused = shell._refresh_traversal_pruning_state(iteration=11)

        self.assertIs(reused, result)
        self.assertEqual(fake_pruner.calls, [[11, 13]])
        self.assertFalse(shell.last_traversal_pruning_stats["refreshed_this_iteration"])
        self.assertEqual(shell.last_traversal_pruning_stats["reason"], "reused")

    def test_traversal_working_view_uses_absolute_arc_indices(self):
        shell = self._make_shell()
        shell.flat_inst_arcs_by_level = torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 0, 0, 1],
                [2, 3, 0, 2, 0, 0, 2],
            ],
            dtype=torch.long,
        )
        shell.flat_inst_arcs_by_level_start = torch.tensor([0, 2, 3], dtype=torch.long)
        shell.last_traversal_pruning_result = FakeTraversalPruningResult(
            kept_flat_arc_indices=[0, 2],
            kept_level_offsets=[0, 1, 2],
            kept_counts_by_level=[1, 1],
        )
        shell.last_traversal_pruning_stats = {
            "enabled": True,
            "applied": True,
        }

        working_arcs, offsets, arc_indices, active = shell._build_traversal_pruning_working_view(
            torch.device("cpu")
        )

        self.assertTrue(active)
        self.assertEqual(arc_indices.tolist(), [0, 2])
        self.assertEqual(offsets.tolist(), [0, 1, 2])
        self.assertEqual(working_arcs[:, 1].tolist(), [1, 3])

    def test_traversal_pruning_disabled_keeps_full_working_view(self):
        shell = self._make_shell(mode="off")
        shell.flat_inst_arcs_by_level = torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 0, 0, 1],
            ],
            dtype=torch.long,
        )
        shell.flat_inst_arcs_by_level_start = torch.tensor([0, 2], dtype=torch.long)

        shell._refresh_traversal_pruning_state(iteration=3)
        working_arcs, offsets, arc_indices, active = shell._build_traversal_pruning_working_view(
            torch.device("cpu")
        )

        self.assertFalse(active)
        self.assertFalse(shell.last_traversal_pruning_stats["enabled"])
        self.assertEqual(shell.last_traversal_pruning_stats["reason"], "disabled")
        self.assertEqual(arc_indices.tolist(), [0, 1])
        self.assertEqual(offsets.tolist(), [0, 2])
        self.assertEqual(working_arcs.tolist(), shell.flat_inst_arcs_by_level.tolist())


if __name__ == "__main__":
    unittest.main()
