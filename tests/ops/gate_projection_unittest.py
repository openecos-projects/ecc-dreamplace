import csv
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.gate_projection.gate_projection import (
    GateProjectionOp,
    NearestSizeProjectionResolver,
    ProjectionScore,
    ProjectionScorer,
    StableCurrentCellResolver,
    TimingAwareProjectionScorer,
)
sys.path.pop()


class PreferLargestCellScorer(ProjectionScorer):
    def score(self, request, candidates, context=None):
        score = -candidates.candidate_sizes.masked_fill(~candidates.candidate_legal_mask, 0.0)
        score = score.masked_fill(~candidates.candidate_legal_mask, float("inf"))
        return ProjectionScore(total_score=score, terms={"prefer_largest": score})


class FlatScoreScorer(ProjectionScorer):
    def score(self, request, candidates, context=None):
        score = torch.zeros_like(candidates.candidate_sizes)
        score = score.masked_fill(~candidates.candidate_legal_mask, float("inf"))
        return ProjectionScore(total_score=score, terms={"flat_score": score})


class GateProjectionOpTest(unittest.TestCase):
    def _build_data(self):
        size_var = torch.tensor([2.2, 7.8, 1.1], dtype=torch.float32)
        vt_var = torch.tensor(
            [
                [0.05, 0.9, 0.05],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )

        return SimpleNamespace(
            device=torch.device("cpu"),
            main_id_2_cell_id_start=torch.tensor([0, 3, 5], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [10.0, 0.0, 1.0, 0.0],
                    [11.0, 0.0, 2.0, 1.0],
                    [12.0, 0.0, 4.0, 2.0],
                    [20.0, 1.0, 8.0, 1.0],
                    [21.0, 1.0, 16.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            flat_libcell_leakage=torch.tensor([3.0, 2.0, 0.5, 4.0, 5.0], dtype=torch.float32),
            inst_main_id=torch.tensor([0, 1, 0], dtype=torch.int64),
            inst_cell_id=torch.tensor([1, 3, 0], dtype=torch.int64),
            inst_libcell_offset=torch.tensor([1, 0, 0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True, False, True], dtype=torch.bool),
            inst_size_lower=torch.tensor([1.0, 8.0, 1.0], dtype=torch.float32),
            inst_size_upper=torch.tensor([4.0, 8.0, 2.0], dtype=torch.float32),
            inst_vt_mask=torch.tensor(
                [
                    [True, True, True],
                    [False, True, False],
                    [True, True, False],
                ],
                dtype=torch.bool,
            ),
            pin_offset_x=torch.tensor([0.1, 0.3, 0.5], dtype=torch.float32),
            pin_offset_y=torch.tensor([1.0, 1.2, 1.4], dtype=torch.float32),
            flat_node2pin_map=torch.tensor([0, 1, 2], dtype=torch.int64),
            flat_node2pin_start_map=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 0, 0], dtype=torch.int64),
            flat_lib_pin_offset_x=torch.tensor([0.1, 0.25, 0.6, 0.3, 0.9], dtype=torch.float32),
            flat_lib_pin_offset_y=torch.tensor([1.0, 1.5, 1.8, 1.2, 2.2], dtype=torch.float32),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

    def test_default_projection_uses_size_vt_and_legality(self):
        data = self._build_data()
        op = GateProjectionOp(data)

        result = op()

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [1, 3, 0])
        self.assertEqual(result.projected_libcell_offset.tolist(), [1, 0, 0])
        self.assertEqual(result.projected_vt.tolist(), [1, 1, 0])
        self.assertEqual(result.current_leakage.tolist(), [2.0, 4.0, 3.0])
        self.assertEqual(result.projected_leakage.tolist(), [2.0, 4.0, 3.0])
        self.assertIn("size_distance", result.score_terms)
        self.assertIn("vt_mismatch", result.score_terms)

    def test_custom_scorer_can_override_selection_policy(self):
        data = self._build_data()
        op = GateProjectionOp(data, scorer=PreferLargestCellScorer())

        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [2])
        self.assertEqual(result.projected_libcell_offset.tolist(), [2])
        self.assertIn("prefer_largest", result.score_terms)

    def test_stable_current_cell_resolver_keeps_current_choice_on_tie(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=FlatScoreScorer(),
            resolver=StableCurrentCellResolver(),
        )

        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [1])
        self.assertEqual(result.projected_libcell_offset.tolist(), [1])
        self.assertEqual(result.resolution_metadata["resolver"], "stable_current_cell")
        self.assertTrue(result.resolution_metadata["preserved_current_cell_mask"].item())

    def test_nearest_size_round_resolver_ignores_score_and_uses_closest_legal_size(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=PreferLargestCellScorer(),
            resolver=NearestSizeProjectionResolver(),
        )

        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [1])
        self.assertEqual(result.projected_libcell_offset.tolist(), [1])
        self.assertEqual(result.resolution_metadata["resolver"], "nearest_size_round")

    def test_nearest_size_round_resolver_preserves_current_cell_on_exact_size_tie(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=FlatScoreScorer(),
            resolver=NearestSizeProjectionResolver(),
        )

        result = op(inst_ids=torch.tensor([2], dtype=torch.int64))

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [0])
        self.assertEqual(result.projected_libcell_offset.tolist(), [0])
        self.assertTrue(result.resolution_metadata["preserved_current_cell_mask"].item())

    def test_timing_aware_scorer_respects_named_candidate_terms(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=TimingAwareProjectionScorer(
                size_weight=0.0,
                vt_weight=0.0,
                offset_weight=0.0,
                timing_weight=1.0,
                leakage_weight=1.0,
            ),
        )
        timing_penalty = torch.tensor([[3.0, 0.2, 0.4]], dtype=torch.float32)
        leakage_penalty = torch.tensor([[5.0, 3.0, 0.1]], dtype=torch.float32)

        result = op(
            inst_ids=torch.tensor([0], dtype=torch.int64),
            context=SimpleNamespace(
                candidate_terms={
                    "timing_penalty": timing_penalty,
                    "leakage_penalty": leakage_penalty,
                },
                term_weights={},
                metadata={},
            ),
        )

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [2])
        self.assertEqual(result.projected_leakage.tolist(), [0.5])
        self.assertEqual(result.projected_libcell_offset.tolist(), [2])
        self.assertIn("timing_penalty", result.score_terms)
        self.assertIn("leakage_penalty", result.score_terms)
        self.assertIn("area", result.validation.metrics)
        self.assertIn("density", result.validation.metrics)
        self.assertIn("leakage", result.validation.metrics)
        self.assertIn("timing", result.validation.metrics)

    def test_write_artifacts_emits_projection_rows_and_summary(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=FlatScoreScorer(),
            resolver=StableCurrentCellResolver(),
        )
        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_paths = op.write_artifacts(
                result,
                result_dir=tmpdir,
                design_name="toy",
                metadata={"artifact_scope": "post_optimization"},
            )

            self.assertTrue(os.path.exists(artifact_paths["jsonl"]))
            self.assertTrue(os.path.exists(artifact_paths["csv"]))
            self.assertTrue(os.path.exists(artifact_paths["summary"]))

            with open(artifact_paths["summary"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["num_instances"], 1)
            self.assertEqual(summary["num_changed_cells"], 0)
            self.assertEqual(summary["total_current_leakage"], 2.0)
            self.assertEqual(summary["total_projected_leakage"], 2.0)
            self.assertEqual(summary["total_leakage_delta"], 0.0)
            self.assertEqual(
                summary["resolution_metadata"]["resolver"],
                "stable_current_cell",
            )
            self.assertEqual(
                summary["metadata"]["artifact_scope"],
                "post_optimization",
            )

            with open(artifact_paths["csv"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["projected_cell_id"], "1")
            self.assertEqual(rows[0]["changed_cell"], "False")
            self.assertEqual(rows[0]["metadata_preserved_current_cell_mask"], "True")

            with open(artifact_paths["jsonl"], "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["projected_cell_id"], 1)
            self.assertFalse(records[0]["changed_cell"])

    def test_build_artifact_summary_emits_objective_term_breakdown(self):
        data = self._build_data()
        op = GateProjectionOp(data, scorer=PreferLargestCellScorer())
        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        summary = op.build_artifact_summary(
            result,
            metadata={"projection_term_weights": {"prefer_largest": 2.0}},
        )

        breakdown = summary["objective_term_breakdown"]
        self.assertEqual(breakdown["all_legal"]["num_instances"], 1)
        self.assertEqual(breakdown["changed_cells"]["num_instances"], 1)
        self.assertEqual(
            breakdown["all_legal"]["terms"]["prefer_largest"]["weight"],
            2.0,
        )
        self.assertAlmostEqual(
            breakdown["all_legal"]["terms"]["prefer_largest"]["sum"],
            -4.0,
            places=6,
        )
        self.assertAlmostEqual(
            breakdown["all_legal"]["terms"]["prefer_largest"]["weighted_sum"],
            -8.0,
            places=6,
        )
        self.assertEqual(
            breakdown["changed_cells"]["dominant_terms_by_weighted_sum"][0]["term"],
            "prefer_largest",
        )

    def test_build_artifact_summary_uses_base_scorer_weights_for_core_terms(self):
        data = self._build_data()
        op = GateProjectionOp(
            data,
            scorer=TimingAwareProjectionScorer(
                size_weight=2.0,
                vt_weight=3.0,
                offset_weight=0.25,
                timing_weight=4.0,
                area_weight=5.0,
                density_weight=6.0,
                leakage_weight=7.0,
            ),
        )
        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        summary = op.build_artifact_summary(result)

        terms = summary["objective_term_breakdown"]["all_legal"]["terms"]
        self.assertEqual(terms["size_distance"]["weight"], 2.0)
        self.assertEqual(terms["vt_mismatch"]["weight"], 3.0)
        self.assertEqual(terms["offset_distance"]["weight"], 0.25)

    def test_projection_result_exports_current_candidate_counterfactual_score(self):
        data = self._build_data()
        op = GateProjectionOp(data, scorer=PreferLargestCellScorer())

        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        self.assertTrue(result.validation.is_valid)
        self.assertEqual(result.projected_cell_id.tolist(), [2])
        self.assertEqual(
            result.resolution_metadata["current_candidate_present"].tolist(),
            [True],
        )
        self.assertAlmostEqual(
            result.resolution_metadata["current_candidate_total_score"].item(),
            -2.0,
            places=6,
        )
        self.assertAlmostEqual(
            result.resolution_metadata["current_candidate_total_score_delta_vs_selected"].item(),
            2.0,
            places=6,
        )

    def test_write_artifacts_emits_pin_offset_projection_reports(self):
        data = self._build_data()
        op = GateProjectionOp(data, scorer=PreferLargestCellScorer())
        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_paths = op.write_artifacts(
                result,
                result_dir=tmpdir,
                design_name="toy",
                metadata={"artifact_scope": "post_optimization"},
            )

            self.assertTrue(os.path.exists(artifact_paths["pin_offset_jsonl"]))
            self.assertTrue(os.path.exists(artifact_paths["pin_offset_csv"]))
            self.assertTrue(os.path.exists(artifact_paths["pin_offset_summary"]))

            with open(artifact_paths["pin_offset_summary"], "r", encoding="utf-8") as f:
                pin_summary = json.load(f)
            self.assertEqual(pin_summary["num_pin_records"], 1)
            self.assertEqual(pin_summary["num_pins_with_projected_offset"], 1)
            self.assertEqual(pin_summary["num_changed_pin_offsets"], 1)

            with open(artifact_paths["pin_offset_csv"], "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["inst_id"], "0")
            self.assertEqual(rows[0]["pin_id"], "0")
            self.assertEqual(rows[0]["projected_cell_id"], "2")
            self.assertEqual(rows[0]["changed_pin_offset"], "True")

            with open(artifact_paths["summary"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(
                summary["pin_offset_consistency"]["num_changed_pin_offsets"],
                1,
            )
            self.assertEqual(summary["total_current_leakage"], 2.0)
            self.assertEqual(summary["total_projected_leakage"], 0.5)
            self.assertEqual(summary["num_leakage_improved_cells"], 1)

    def test_compute_projected_pin_offsets_returns_runtime_targets(self):
        data = self._build_data()
        op = GateProjectionOp(data, scorer=PreferLargestCellScorer())
        result = op(inst_ids=torch.tensor([0], dtype=torch.int64))

        pin_offsets = op.compute_projected_pin_offsets(result)

        self.assertEqual(pin_offsets["inst_ids"].tolist(), [0])
        self.assertEqual(pin_offsets["pin_ids"].tolist(), [0])
        self.assertEqual(pin_offsets["current_cell_id"].tolist(), [1])
        self.assertEqual(pin_offsets["projected_cell_id"].tolist(), [2])
        self.assertTrue(pin_offsets["has_projected_pin_offset"].item())
        self.assertTrue(pin_offsets["changed_pin_offset"].item())
        self.assertAlmostEqual(pin_offsets["projected_pin_offset_x"].item(), 0.6, places=6)
        self.assertAlmostEqual(pin_offsets["projected_pin_offset_y"].item(), 1.8, places=6)


if __name__ == "__main__":
    unittest.main()
