import json
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)


import dreamplace.BasicPlace as basic_place_module  # noqa: E402
from dreamplace.BasicPlace import BasicPlace  # noqa: E402
from dreamplace.Params import Params  # noqa: E402
from dreamplace.ops.m2_legalize.m2_pg_rail_hybrid_legalization import (  # noqa: E402
    M2PgRailHybridLegalizationView,
)


def _make_params(**overrides):
    values = {
        "m2_pg_rail_legalization_blockage_flag": 1,
        "m2_pg_rail_legalization_mode": "soft",
        "ieda_m2_pg_rail_blockage_flag": 0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _make_placedb(**overrides):
    values = {
        "num_nodes": 4,
        "num_movable_nodes": 2,
        "num_terminals": 1,
        "num_terminal_NIs": 1,
        "num_filler_nodes": 0,
        "regions": [],
        "xl": 0.0,
        "yl": 0.0,
        "xh": 20.0,
        "yh": 4.0,
        "site_width": 1.0,
        "row_height": 1.0,
        "num_bins_x": 8,
        "num_bins_y": 8,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _make_data():
    return types.SimpleNamespace(
        node_size_x=torch.tensor([2.0, 2.0, 4.0, 0.0]),
        node_size_y=torch.tensor([1.0, 1.0, 2.0, 0.0]),
        num_pins_in_nodes=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        flat_region_boxes=torch.empty((0, 4)),
        flat_region_boxes_start=torch.tensor([0], dtype=torch.int32),
        node2fence_region_map=torch.zeros(3, dtype=torch.int32),
    )


class M2PgRailSoftLegalizationFlowTest(unittest.TestCase):
    def test_schema_defaults_to_disabled_soft_mode(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as stream:
            params = json.load(stream)

        entry = params["m2_pg_rail_legalization_blockage_flag"]
        self.assertEqual(entry["default"], 0)
        self.assertEqual(
            params["m2_pg_rail_legalization_mode"]["default"], "soft"
        )

    def _build_with_fakes(self, greedy_legal=True, abacus_legal=True):
        constructed = []
        calls = []
        legality_results = iter((greedy_legal, abacus_legal))

        class FakeLegalizer:
            def __init__(self, name, **kwargs):
                self.name = name
                self.kwargs = kwargs
                constructed.append(self)

            def __call__(self, init_pos, pos):
                calls.append(self.name)
                return pos + 1.0

        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        soft_calls = []

        def soft_op(pos):
            soft_calls.append(pos.clone())
            return pos + 10.0, {"legal": True}

        place.op_collections = types.SimpleNamespace(
            legality_check_op=lambda pos: next(legality_results),
            m2_soft_legalize_op=soft_op,
        )
        patches = (
            mock.patch.object(
                basic_place_module.macro_legalize,
                "MacroLegalize",
                side_effect=lambda **kwargs: FakeLegalizer("macro", **kwargs),
            ),
            mock.patch.object(
                basic_place_module.greedy_legalize,
                "GreedyLegalize",
                side_effect=lambda **kwargs: FakeLegalizer("greedy", **kwargs),
            ),
            mock.patch.object(
                basic_place_module.abacus_legalize,
                "AbacusLegalize",
                side_effect=lambda **kwargs: FakeLegalizer("abacus", **kwargs),
            ),
        )
        return place, constructed, calls, soft_calls, patches

    def test_soft_stage_consumes_ordinary_legal_result(self):
        place, constructed, calls, soft_calls, patches = self._build_with_fakes()
        original = torch.tensor([0.0, 4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 0.0])

        with patches[0], patches[1], patches[2]:
            legalize = place.build_legalization(
                _make_params(), _make_placedb(), _make_data(), torch.device("cpu")
            )
            result = legalize(original)

        self.assertEqual(
            [op.name for op in constructed], ["macro", "greedy", "abacus"]
        )
        self.assertEqual(calls, ["macro", "greedy", "abacus"])
        self.assertEqual(len(soft_calls), 1)
        self.assertTrue(torch.equal(soft_calls[0], original + 3.0))
        self.assertTrue(torch.equal(result, original + 13.0))

    def test_abacus_failure_uses_legal_greedy_result_before_soft_stage(self):
        place, _, calls, soft_calls, patches = self._build_with_fakes(
            greedy_legal=True, abacus_legal=False
        )
        original = torch.zeros(8)

        with patches[0], patches[1], patches[2]:
            legalize = place.build_legalization(
                _make_params(), _make_placedb(), _make_data(), torch.device("cpu")
            )
            result = legalize(original)

        self.assertEqual(calls, ["macro", "greedy", "abacus"])
        self.assertTrue(torch.equal(soft_calls[0], original + 2.0))
        self.assertTrue(torch.equal(result, original + 12.0))

    def test_ordinary_legality_failure_is_not_published(self):
        place, _, calls, soft_calls, patches = self._build_with_fakes(
            greedy_legal=False
        )

        with patches[0], patches[1], patches[2]:
            legalize = place.build_legalization(
                _make_params(), _make_placedb(), _make_data(), torch.device("cpu")
            )
            with self.assertRaisesRegex(RuntimeError, "ordinary legalization failed"):
                legalize(torch.zeros(8))

        self.assertEqual(calls, ["macro", "greedy"])
        self.assertEqual(soft_calls, [])


def _make_hybrid_data(node_size_x, node_size_y, rail_boxes):
    num_nodes = len(node_size_x)
    return types.SimpleNamespace(
        node_size_x=torch.tensor(node_size_x, dtype=torch.float32),
        node_size_y=torch.tensor(node_size_y, dtype=torch.float32),
        num_pins_in_nodes=torch.arange(
            1, num_nodes + 1, dtype=torch.float32
        ),
        m2_pg_rail_density_boxes=torch.tensor(
            rail_boxes, dtype=torch.float32
        ).reshape(-1, 4),
        flat_region_boxes=torch.empty((0, 4)),
        flat_region_boxes_start=torch.tensor([0], dtype=torch.int32),
        node2fence_region_map=torch.zeros(num_nodes, dtype=torch.int32),
        fp_info=types.SimpleNamespace(
            xl=0.0,
            yl=0.0,
            xh=10.0,
            yh=2.0,
            site_width=1.0,
            row_height=1.0,
            scale_factor=1.0,
        ),
    )


def _make_pos(x, y):
    return torch.tensor(list(x) + list(y), dtype=torch.float32)


def _boxes_overlap(lhs, rhs, tolerance=1e-6):
    return (
        min(lhs[2], rhs[2]) - max(lhs[0], rhs[0]) > tolerance
        and min(lhs[3], rhs[3]) - max(lhs[1], rhs[1]) > tolerance
    )


class M2PgRailHybridHardViewTest(unittest.TestCase):
    def test_mode_normalization(self):
        params = Params()
        params.m2_pg_rail_legalization_mode = " HYBRID_HARD "
        params.normalize_m2_pg_rail_legalization_mode()
        self.assertEqual(params.m2_pg_rail_legalization_mode, "hybrid_hard")

        params.m2_pg_rail_legalization_mode = " SUBSET_HARD "
        params.normalize_m2_pg_rail_legalization_mode()
        self.assertEqual(params.m2_pg_rail_legalization_mode, "subset_hard")

        params.m2_pg_rail_legalization_mode = "budget"
        with self.assertRaisesRegex(ValueError, "soft.*hybrid_hard.*subset_hard"):
            params.normalize_m2_pg_rail_legalization_mode()

    def test_subset_hard_selects_left_to_right_inclusive_rail_range(self):
        placedb = _make_placedb(
            num_nodes=4,
            num_movable_nodes=2,
            num_terminals=1,
            num_terminal_NIs=1,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [
                [7.0, 0.0, 8.0, 2.0],
                [1.0, 0.0, 2.0, 2.0],
                [5.0, 0.0, 6.0, 2.0],
                [3.0, 0.0, 4.0, 2.0],
            ],
        )

        view = M2PgRailHybridLegalizationView.create(
            placedb,
            data,
            hard_rail_start=2,
            hard_rail_end=3,
            legalization_mode="subset_hard",
        )

        self.assertEqual(view.legalization_mode, "subset_hard")
        self.assertEqual((view.hard_rail_start, view.hard_rail_end), (2, 3))
        self.assertEqual(view.full_rail_boxes[:, 0].tolist(), [1.0, 3.0, 5.0, 7.0])
        self.assertEqual(view.rail_boxes[:, 0].tolist(), [3.0, 5.0])

    def test_subset_hard_rejects_range_outside_available_rails(self):
        placedb = _make_placedb()
        data = _make_hybrid_data(
            [2.0, 2.0, 4.0, 0.0],
            [1.0, 1.0, 2.0, 0.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )

        with self.assertRaisesRegex(ValueError, "outside.*2 available"):
            M2PgRailHybridLegalizationView.create(
                placedb,
                data,
                hard_rail_start=1,
                hard_rail_end=3,
                legalization_mode="subset_hard",
            )

    def test_only_cell_wider_than_every_rail_free_gap_is_unavoidable(self):
        placedb = _make_placedb(
            num_nodes=3,
            num_movable_nodes=3,
            num_terminals=0,
            num_terminal_NIs=0,
            node_names=np.array([b"wide", b"fits", b"exact"]),
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        problem = view.prepare(
            _make_pos([0.0, 4.0, 4.0], [0.0, 0.0, 1.0]), data
        )

        self.assertEqual(view.max_rail_free_width, 3.0)
        self.assertEqual(problem.reserved_movable_ids.tolist(), [0])
        self.assertEqual(view.movable_node_names[0], "wide")

    def test_soft_and_hybrid_modes_build_only_their_own_stage(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        placedb = _make_placedb(
            num_nodes=1,
            num_movable_nodes=1,
            num_terminals=0,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0],
            [1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )

        self.assertIsNone(
            place.build_m2_soft_legalization(
                _make_params(m2_pg_rail_legalization_mode="hybrid_hard"),
                placedb,
                data,
            )
        )
        self.assertIsNone(
            place.build_m2_pg_rail_hybrid_legalization_view(
                _make_params(m2_pg_rail_legalization_mode="soft"),
                placedb,
                data,
            )
        )
        self.assertIsNotNone(
            place.build_m2_pg_rail_hybrid_legalization_view(
                _make_params(m2_pg_rail_legalization_mode="hybrid_hard"),
                placedb,
                data,
            )
        )
        subset_view = place.build_m2_pg_rail_hybrid_legalization_view(
            _make_params(
                m2_pg_rail_legalization_mode="subset_hard",
                m2_pg_rail_legalization_hard_rail_start=1,
                m2_pg_rail_legalization_hard_rail_end=1,
            ),
            placedb,
            data,
        )
        self.assertIsNotNone(subset_view)
        self.assertEqual(subset_view.rail_boxes.size(0), 1)
        self.assertEqual(subset_view.full_rail_boxes.size(0), 2)
        self.assertIsNotNone(
            place.build_m2_pg_rail_hybrid_legalization_view(
                _make_params(
                    m2_pg_rail_legalization_mode="subset_hard",
                    m2_pg_rail_legalization_hard_rail_start=1,
                    m2_pg_rail_legalization_hard_rail_end=1,
                    detailed_place_flag=1,
                ),
                placedb,
                data,
            )
        )

    def test_params_normalize_subset_hard_rail_range(self):
        params = Params()
        params.m2_pg_rail_legalization_mode = "subset_hard"
        params.m2_pg_rail_legalization_hard_rail_start = "2"
        params.m2_pg_rail_legalization_hard_rail_end = 4.0
        params.normalize_params()
        self.assertEqual(params.m2_pg_rail_legalization_hard_rail_start, 2)
        self.assertEqual(params.m2_pg_rail_legalization_hard_rail_end, 4)

        params.m2_pg_rail_legalization_hard_rail_end = 1
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            params.normalize_params()

    def test_hybrid_mode_rejects_detailed_placement(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        placedb = _make_placedb(
            num_nodes=1,
            num_movable_nodes=1,
            num_terminals=0,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0],
            [1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        with self.assertRaisesRegex(ValueError, "detailed_place_flag=0"):
            place.build_m2_pg_rail_hybrid_legalization_view(
                _make_params(
                    m2_pg_rail_legalization_mode="hybrid_hard",
                    detailed_place_flag=1,
                ),
                placedb,
                data,
            )

    def test_reservation_avoids_fixed_obstacle_and_minimizes_overlap(self):
        placedb = _make_placedb(
            num_nodes=2,
            num_movable_nodes=1,
            num_terminals=1,
            num_terminal_NIs=0,
            node_names=np.array([b"wide", b"fixed"]),
            xh=10.0,
            yh=2.0,
        )
        rails = [
            [2.0, 0.0, 3.0, 2.0],
            [5.0, 0.0, 6.0, 2.0],
            [8.0, 0.0, 9.0, 2.0],
        ]
        data = _make_hybrid_data([4.0, 4.0], [1.0, 1.0], rails)
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        pos = _make_pos([0.0, 0.0], [0.0, 0.0])
        problem = view.prepare(pos, data)

        self.assertEqual(problem.reserved_movable_ids.tolist(), [0])
        self.assertTrue(
            torch.equal(
                problem.reserved_positions,
                torch.tensor([[0.0, 1.0]]),
            )
        )
        self.assertEqual(problem.reserved_overlap_areas.tolist(), [1.0])
        reserved_box = [0.0, 1.0, 4.0, 2.0]
        fixed_box = [0.0, 0.0, 4.0, 1.0]
        self.assertFalse(_boxes_overlap(reserved_box, fixed_box))
        for fragment in problem.rail_boxes.tolist():
            self.assertFalse(_boxes_overlap(reserved_box, fragment))

    def test_fixed_macros_can_close_rail_only_wide_gaps(self):
        placedb = _make_placedb(
            num_nodes=3,
            num_movable_nodes=1,
            num_terminals=2,
            num_terminal_NIs=0,
            node_names=np.array([b"wide", b"fixed_left", b"fixed_right"]),
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 2.0, 2.0],
            [1.0, 2.0, 2.0],
            [[4.0, 0.0, 5.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        self.assertEqual(view.max_rail_free_width, 5.0)

        problem = view.prepare(
            _make_pos([2.0, 0.0, 8.0], [0.0, 0.0, 0.0]), data
        )
        self.assertEqual(problem.reserved_movable_ids.tolist(), [0])
        self.assertEqual(problem.reserved_positions.tolist(), [[2.0, 0.0]])
        self.assertEqual(problem.reserved_overlap_areas.tolist(), [1.0])

    def test_packing_and_copy_back_preserve_original_node_ids(self):
        placedb = _make_placedb(
            num_nodes=6,
            num_movable_nodes=3,
            num_terminals=1,
            num_terminal_NIs=1,
            num_filler_nodes=1,
            node_names=np.array(
                [b"left", b"wide", b"right", b"fixed", b"io", b"filler"]
            ),
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [1.0, 4.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        original = _make_pos(
            [0.0, 0.0, 5.0, 9.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        )
        problem = view.prepare(original, data)

        self.assertEqual(problem.remaining_movable_ids.tolist(), [0, 2])
        self.assertEqual(problem.reserved_movable_ids.tolist(), [1])
        self.assertEqual(problem.num_movable_nodes, 2)
        self.assertEqual(
            problem.num_terminals,
            1 + 1 + problem.rail_boxes.size(0),
        )

        legalized = problem.packed_pos.clone()
        legalized[: problem.num_movable_nodes] = torch.tensor([1.0, 6.0])
        legalized[
            problem.num_nodes : problem.num_nodes + problem.num_movable_nodes
        ] = torch.tensor([0.0, 0.0])
        restored = problem.restore_original_order(original, legalized)
        self.assertEqual(restored[:6].tolist(), [1.0, 0.0, 6.0, 9.0, 0.0, 1.0])
        self.assertEqual(restored[6:].tolist(), [0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    def test_audit_rejects_non_exempt_overlap(self):
        placedb = _make_placedb(
            num_nodes=2,
            num_movable_nodes=2,
            num_terminals=0,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 1.0],
            [1.0, 1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        original = _make_pos([0.0, 4.0], [0.0, 0.0])
        problem = view.prepare(original, data)
        legal = problem.restore_original_order(original, problem.packed_pos)
        audit = view.audit(legal, data.node_size_x, data.node_size_y, problem)
        self.assertEqual(audit["non_exempt_overlap_count"], 0)
        self.assertEqual(audit["reserved_overlap_area"], 1.0)
        self.assertEqual(audit["reserved_max_area_delta"], 0.0)

        illegal = legal.clone()
        illegal[1] = 3.0
        audit = view.audit(illegal, data.node_size_x, data.node_size_y, problem)
        self.assertEqual(audit["non_exempt_overlap_count"], 1)

    def test_subset_audit_reports_full_rail_overlap_separately(self):
        placedb = _make_placedb(
            num_nodes=4,
            num_movable_nodes=2,
            num_terminals=1,
            num_terminal_NIs=1,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [[2.0, 0.0, 3.0, 2.0], [5.0, 0.0, 6.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(
            placedb,
            data,
            hard_rail_start=2,
            hard_rail_end=2,
            legalization_mode="subset_hard",
        )
        original = _make_pos([2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        problem = view.prepare(original, data)
        audit = view.audit(
            original, data.node_size_x, data.node_size_y, problem
        )

        self.assertEqual(audit["non_exempt_overlap_count"], 0)
        self.assertEqual(audit["full_non_exempt_overlap_count"], 1)
        self.assertEqual(audit["full_non_exempt_overlap_area"], 1.0)

    def test_prepare_and_copy_back_detach_autograd_parameter(self):
        placedb = _make_placedb(
            num_nodes=2,
            num_movable_nodes=2,
            num_terminals=0,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 1.0],
            [1.0, 1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        original = torch.nn.Parameter(_make_pos([0.0, 4.0], [0.0, 0.0]))
        problem = view.prepare(original, data)
        restored = problem.restore_original_order(original, problem.packed_pos)

        self.assertFalse(problem.packed_pos.requires_grad)
        self.assertFalse(restored.requires_grad)
        self.assertTrue(
            torch.equal(restored, _make_pos([0.0, 4.0], [0.0, 0.0]))
        )

    def test_impossible_reservation_fails_clearly(self):
        placedb = _make_placedb(
            num_nodes=2,
            num_movable_nodes=1,
            num_terminals=1,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 10.0],
            [1.0, 2.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        with self.assertRaisesRegex(RuntimeError, "no .*legal position"):
            view.prepare(_make_pos([0.0, 0.0], [0.0, 0.0]), data)

    def test_hybrid_flow_legalizes_only_non_reserved_movable_nodes(self):
        placedb = _make_placedb(
            num_nodes=2,
            num_movable_nodes=2,
            num_terminals=0,
            num_terminal_NIs=0,
            xh=10.0,
            yh=2.0,
        )
        data = _make_hybrid_data(
            [4.0, 1.0],
            [1.0, 1.0],
            [[3.0, 0.0, 4.0, 2.0], [7.0, 0.0, 8.0, 2.0]],
        )
        view = M2PgRailHybridLegalizationView.create(placedb, data)
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        place.m2_pg_rail_hybrid_legalization_view = view
        ordinary_checks = []
        place.op_collections = types.SimpleNamespace(
            legality_check_op=lambda pos: ordinary_checks.append(pos.clone()) or True,
            m2_soft_legalize_op=None,
        )
        constructed = []
        calls = []

        class FakeLegalizer:
            def __init__(self, name, **kwargs):
                self.name = name
                self.kwargs = kwargs
                constructed.append(self)

            def __call__(self, init_pos, pos):
                calls.append((self.name, self.kwargs["num_movable_nodes"]))
                return pos

        with mock.patch.object(
            basic_place_module.macro_legalize,
            "MacroLegalize",
            side_effect=lambda **kwargs: FakeLegalizer("macro", **kwargs),
        ), mock.patch.object(
            basic_place_module.greedy_legalize,
            "GreedyLegalize",
            side_effect=lambda **kwargs: FakeLegalizer("greedy", **kwargs),
        ), mock.patch.object(
            basic_place_module.abacus_legalize,
            "AbacusLegalize",
            side_effect=lambda **kwargs: FakeLegalizer("abacus", **kwargs),
        ), mock.patch.object(
            basic_place_module.legality_check,
            "LegalityCheck",
            return_value=lambda pos: True,
        ):
            legalize = place.build_legalization(
                _make_params(m2_pg_rail_legalization_mode="hybrid_hard"),
                placedb,
                data,
                torch.device("cpu"),
            )
            result = legalize(_make_pos([0.0, 4.0], [0.0, 0.0]))

        self.assertEqual(len(constructed), 6)
        self.assertEqual(
            calls,
            [("macro", 1), ("greedy", 1), ("abacus", 1)],
        )
        self.assertEqual(len(ordinary_checks), 1)
        self.assertTrue(torch.equal(result, _make_pos([0.0, 4.0], [0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
