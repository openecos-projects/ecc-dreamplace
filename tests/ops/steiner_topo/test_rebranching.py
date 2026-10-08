import unittest

from dreamplace.ops.steiner_topo import (
    build_rebranching_dry_run,
    build_timing_aware_skeleton,
    build_timing_rooted_tree_view,
)


class RebranchingTest(unittest.TestCase):
    def test_rebranching_prioritizes_highest_sink_criticality(self):
        tree = build_timing_rooted_tree_view(
            net_id=9,
            net_pin_ids=[0, 2, 3],
            driver_pin_id=0,
            undirected_edges=[(0, 1), (1, 2), (1, 3)],
            coordinates={0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (10, 10)},
        )

        artifact = build_rebranching_dry_run(
            tree,
            sink_slack_by_pin={2: -10.0, 3: -1.0},
            npath_by_pin={2: 3, 3: 1},
            max_rebranched_sink_ratio=0.5,
        )

        self.assertEqual(artifact["status"], "ok")
        self.assertEqual(artifact["rebranching_status"], "ok")
        self.assertEqual(artifact["topology_source"], "timing_rooted_steiner")
        self.assertEqual(artifact["rebranching_enabled"], True)
        self.assertEqual(artifact["rebranching_sink_criticality_source"], "slack_times_npath")
        self.assertEqual(artifact["candidate_set_skeleton_source"], "timing_aware_rebranched_skeleton")
        self.assertEqual(artifact["rebranched_sink_count"], 1)
        self.assertEqual(artifact["rebranching_candidate_sink_count"], 2)
        self.assertEqual(artifact["rebranching_max_sink_fraction"], 0.5)
        self.assertAlmostEqual(artifact["rebranching_ratio"], 0.5)
        self.assertEqual(artifact["rebranching_changed_edge_count"], 1)
        self.assertEqual(artifact["rebranching_skip_reason"], "")
        self.assertEqual(artifact["rebranched_sinks"][0]["sink_pin_id"], 2)
        self.assertEqual(artifact["rebranched_sinks"][0]["old_parent_node_id"], 1)
        self.assertEqual(artifact["rebranched_sinks"][0]["new_parent_node_id"], 0)
        self.assertEqual(artifact["rebranched_sinks"][0]["npath"], 3)
        self.assertEqual(artifact["rebranched_sinks"][0]["ki"], 2)
        self.assertEqual(artifact["rebranched_sinks"][0]["jump_level"], 2)
        self.assertEqual(
            artifact["rebranched_sinks"][0]["normalized_criticality"],
            1.0,
        )
        self.assertTrue(artifact["is_connected"])
        self.assertFalse(artifact["has_cycle"])

    def test_non_negative_slack_has_zero_criticality(self):
        tree = build_timing_rooted_tree_view(
            net_id=10,
            net_pin_ids=[0, 1],
            driver_pin_id=0,
            undirected_edges=[(0, 1)],
            coordinates={0: (0, 0), 1: (1, 0)},
        )

        artifact = build_rebranching_dry_run(
            tree,
            sink_slack_by_pin={1: 5.0},
            npath_by_pin={1: 100},
            max_rebranched_sink_ratio=1.0,
        )

        self.assertEqual(artifact["sink_criticality"][0]["criticality"], 0.0)
        self.assertEqual(artifact["rebranched_sink_count"], 0)
        self.assertEqual(artifact["candidate_set_skeleton_source"], "fallback_timing_rooted_steiner")

    def test_rebranching_limit_does_not_round_small_fanout_above_ratio(self):
        tree = build_timing_rooted_tree_view(
            net_id=13,
            net_pin_ids=[0, 2, 3],
            driver_pin_id=0,
            undirected_edges=[(0, 1), (1, 2), (1, 3)],
            coordinates={0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (10, 10)},
        )

        artifact = build_rebranching_dry_run(
            tree,
            sink_slack_by_pin={2: -10.0, 3: -1.0},
            npath_by_pin={2: 3, 3: 1},
            max_rebranched_sink_ratio=0.2,
        )

        self.assertEqual(artifact["rebranch_limit"], 0)
        self.assertEqual(artifact["rebranched_sink_count"], 0)
        self.assertEqual(artifact["candidate_set_skeleton_source"], "fallback_timing_rooted_steiner")
        self.assertEqual(artifact["rebranching_skip_reason"], "no_positive_rebranching_move")

    def test_rebranching_wirelength_guard_falls_back_to_original_skeleton(self):
        tree = build_timing_rooted_tree_view(
            net_id=14,
            net_pin_ids=[0, 3],
            driver_pin_id=0,
            undirected_edges=[(0, 1), (1, 2), (2, 3)],
            coordinates={
                0: (0, 0),
                1: (10, 0),
                2: (10, 100),
                3: (20, 100),
            },
        )

        artifact = build_rebranching_dry_run(
            tree,
            sink_slack_by_pin={3: -10.0},
            npath_by_pin={3: 1},
            max_rebranched_sink_ratio=1.0,
            max_wirelength_delta_ratio=0.1,
        )

        self.assertEqual(artifact["rebranching_status"], "skipped_with_reason")
        self.assertEqual(artifact["rebranched_sink_count"], 0)
        self.assertEqual(artifact["candidate_set_skeleton_source"], "fallback_timing_rooted_steiner")
        self.assertEqual(artifact["rebranching_skip_reason"], "wirelength_delta_exceeds_limit")
        self.assertTrue(artifact["wirelength_guard_triggered"])
        self.assertGreater(artifact["rebranching_wirelength_delta"], 0.0)

    def test_missing_criticality_maps_skip_with_explicit_reason(self):
        tree = build_timing_rooted_tree_view(
            net_id=11,
            net_pin_ids=[0, 1, 2],
            driver_pin_id=0,
            undirected_edges=[(0, 1), (1, 2)],
            coordinates={0: (0, 0), 1: (1, 0), 2: (2, 0)},
        )

        artifact = build_rebranching_dry_run(
            tree,
            sink_slack_by_pin={1: -1.0},
            npath_by_pin={1: 1},
        )

        self.assertEqual(artifact["status"], "skipped_with_reason")
        self.assertEqual(artifact["rebranching_status"], "skipped_with_reason")
        self.assertEqual(artifact["candidate_set_skeleton_source"], "fallback_timing_rooted_steiner")
        self.assertIn("missing_sink_slack_or_npath", artifact["rebranching_skip_reason"])
        self.assertEqual(artifact["rebranched_sink_count"], 0)

    def test_build_timing_aware_skeleton_returns_rebranched_tree(self):
        tree = build_timing_rooted_tree_view(
            net_id=12,
            net_pin_ids=[0, 2, 3],
            driver_pin_id=0,
            undirected_edges=[(0, 1), (1, 2), (1, 3)],
            coordinates={0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (10, 10)},
        )

        skeleton, summary = build_timing_aware_skeleton(
            tree,
            sink_slack_by_pin={2: -10.0, 3: -1.0},
            npath_by_pin={2: 3, 3: 1},
            max_sink_fraction=0.5,
        )

        self.assertEqual(summary["rebranching_status"], "ok")
        self.assertEqual(summary["candidate_set_skeleton_source"], "timing_aware_rebranched_skeleton")
        self.assertIn((0, 2), skeleton["edge_pairs"])
        self.assertEqual(skeleton["parent_by_node"][2], 0)
        self.assertEqual(sorted(skeleton["children_by_node"][0]), [1, 2])

    def test_unsupported_tree_propagates_failure(self):
        artifact = build_rebranching_dry_run(
            {"status": "unsupported", "unsupported_reasons": ["driver_not_in_tree"]},
            sink_slack_by_pin={},
            npath_by_pin={},
        )

        self.assertEqual(artifact["status"], "unsupported")
        self.assertEqual(artifact["rebranching_status"], "skipped_with_reason")
        self.assertEqual(artifact["candidate_set_skeleton_source"], "fallback_unsupported")
        self.assertIn("driver_not_in_tree", artifact["unsupported_reasons"])


if __name__ == "__main__":
    unittest.main()
