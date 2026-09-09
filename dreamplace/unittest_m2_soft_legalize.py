import json
import os
import sys
import types
import unittest
from unittest import mock

import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)


from dreamplace.BasicPlace import BasicPlace
from dreamplace.Params import Params
from dreamplace.ops.m2_soft_legalize.m2_soft_legalize import (
    M2SoftLegalize,
    STAT_FIELDS,
)


def _make_op(
    widths,
    rail_boxes,
    *,
    heights=None,
    num_movable_nodes=None,
    num_terminals=0,
    displacement_weight=0.01,
    xh=12,
):
    if heights is None:
        heights = [1.0] * len(widths)
    if num_movable_nodes is None:
        num_movable_nodes = len(widths) - num_terminals
    return M2SoftLegalize(
        node_size_x=torch.tensor(widths),
        node_size_y=torch.tensor(heights),
        rail_boxes=torch.tensor(rail_boxes).reshape(-1, 4),
        xl=0,
        yl=0,
        xh=xh,
        yh=4,
        site_width=1,
        row_height=1,
        num_movable_nodes=num_movable_nodes,
        num_terminals=num_terminals,
        displacement_weight=displacement_weight,
    )


def _run(op, xs, ys=None):
    if ys is None:
        ys = [0.0] * len(xs)
    return op(torch.tensor(xs + ys, dtype=torch.float32))


class M2SoftLegalizeKernelTest(unittest.TestCase):
    def test_empty_rail_geometry_is_a_noop(self):
        output, stats = _run(_make_op([2.0], []), [4.0])

        self.assertTrue(torch.equal(output, torch.tensor([4.0, 0.0])))
        self.assertTrue(all(stats[name] == 0 for name in STAT_FIELDS))

    def test_moves_cell_out_of_rail_when_whitespace_exists(self):
        output, stats = _run(
            _make_op([2.0, 2.0, 2.0], [[8.0, 0.0, 10.0, 1.0]]),
            [0.0, 4.0, 8.0],
        )

        self.assertEqual(output[:3].tolist(), [0.0, 4.0, 6.0])
        self.assertEqual(stats["overlap_count_before"], 1)
        self.assertEqual(stats["overlap_count_after"], 0)
        self.assertEqual(stats["moved_count"], 1)
        self.assertEqual(stats["total_displacement"], 2)

    def test_chain_shift_preserves_cell_legality(self):
        output, stats = _run(
            _make_op([2.0, 2.0, 2.0], [[2.0, 0.0, 4.0, 1.0]], xh=8),
            [0.0, 2.0, 6.0],
        )

        self.assertEqual(output[:3].tolist(), [0.0, 4.0, 6.0])
        self.assertEqual(stats["overlap_count_after"], 0)

    def test_insufficient_capacity_keeps_residual_rail_overlap(self):
        output, stats = _run(
            _make_op([5.0, 5.0], [[4.0, 0.0, 6.0, 1.0]], xh=10),
            [0.0, 5.0],
        )

        self.assertEqual(output[:2].tolist(), [0.0, 5.0])
        self.assertEqual(stats["overlap_count_before"], 2)
        self.assertEqual(stats["overlap_count_after"], 2)
        self.assertEqual(stats["overlap_area_after"], 2)
        self.assertEqual(stats["moved_count"], 0)

    def test_displacement_weight_rejects_expensive_move(self):
        output, stats = _run(
            _make_op(
                [2.0, 2.0],
                [[8.0, 0.0, 10.0, 1.0]],
                displacement_weight=2.0,
                xh=10,
            ),
            [0.0, 8.0],
        )

        self.assertEqual(output[:2].tolist(), [0.0, 8.0])
        self.assertEqual(stats["overlap_count_after"], 1)
        self.assertEqual(stats["moved_count"], 0)

    def test_does_not_trade_one_rail_overlap_for_another(self):
        output, stats = _run(
            _make_op(
                [2.0, 2.0],
                [[2.0, 0.0, 4.0, 1.0], [4.0, 0.0, 6.0, 1.0]],
                num_movable_nodes=1,
                num_terminals=1,
                xh=8,
            ),
            [4.0, 6.0],
        )

        self.assertEqual(output[:2].tolist(), [4.0, 6.0])
        self.assertEqual(stats["overlap_area_before"], 2)
        self.assertEqual(stats["overlap_area_after"], 2)


class M2SoftLegalizeIntegrationTest(unittest.TestCase):
    @staticmethod
    def _params(**overrides):
        values = {
            "m2_pg_rail_legalization_blockage_flag": 1,
            "m2_pg_rail_legalization_displacement_weight": 0.01,
            "ieda_m2_pg_rail_blockage_flag": 0,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    @staticmethod
    def _placedb(**overrides):
        values = {
            "regions": [],
            "xl": 0.0,
            "yl": 0.0,
            "xh": 20.0,
            "yh": 2.0,
            "site_width": 1.0,
            "row_height": 1.0,
            "num_movable_nodes": 1,
            "num_terminals": 0,
            "total_movable_node_area": 2.0,
            "total_space_area": 20.0,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    @staticmethod
    def _data(rail_boxes=None):
        if rail_boxes is None:
            rail_boxes = [[4.0, 0.0, 6.0, 1.0]]
        return types.SimpleNamespace(
            node_size_x=torch.tensor([2.0]),
            node_size_y=torch.tensor([1.0]),
            m2_pg_rail_density_boxes=torch.tensor(rail_boxes).reshape(-1, 4),
        )

    def _place(self, legal=True):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        place.op_collections = types.SimpleNamespace(
            hpwl_op=lambda pos: pos.sum(),
            legality_check_op=lambda pos: legal,
        )
        return place

    def test_disabled_flag_does_not_build_op(self):
        op = self._place().build_m2_soft_legalization(
            self._params(m2_pg_rail_legalization_blockage_flag=0),
            self._placedb(),
            self._data(),
        )

        self.assertIsNone(op)

    def test_insufficient_hard_capacity_is_advisory_not_fallback(self):
        with mock.patch(
            "dreamplace.m2_rail_legalization.m2_soft_legalize.M2SoftLegalize"
        ) as constructor, self.assertLogs(level="INFO") as logs:
            op = self._place().build_m2_soft_legalization(
                self._params(),
                self._placedb(
                    total_movable_node_area=9.0,
                    total_space_area=10.0,
                    xh=10.0,
                    yh=1.0,
                ),
                self._data([[0.0, 0.0, 3.0, 1.0]]),
            )

        self.assertTrue(callable(op))
        constructor.assert_called_once()
        self.assertIn("advisory_only=1", "\n".join(logs.output))

    def test_illegal_soft_result_is_rolled_back(self):
        class FakeSoftLegalize:
            def __init__(self, **kwargs):
                pass

            def __call__(self, pos):
                stats = {name: 0.0 for name in STAT_FIELDS}
                stats.update(
                    {
                        "overlap_count_before": 1.0,
                        "overlap_area_before": 2.0,
                        "moved_count": 1.0,
                    }
                )
                return pos + 2.0, stats

        original = torch.tensor([4.0, 0.0])
        with mock.patch(
            "dreamplace.m2_rail_legalization.m2_soft_legalize.M2SoftLegalize",
            FakeSoftLegalize,
        ):
            op = self._place(legal=False).build_m2_soft_legalization(
                self._params(), self._placedb(), self._data()
            )
            output, stats = op(original)

        self.assertTrue(torch.equal(output, original))
        self.assertTrue(stats["rollback"])
        self.assertFalse(stats["legal"])

    def test_schema_and_parameter_normalization(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as stream:
            schema = json.load(stream)
        self.assertEqual(
            schema["m2_pg_rail_legalization_displacement_weight"]["default"],
            0.01,
        )

        params = Params()
        params.fromJson(
            {"m2_pg_rail_legalization_displacement_weight": "0.25"}
        )
        self.assertEqual(
            params.m2_pg_rail_legalization_displacement_weight, 0.25
        )
        for value in (-1, float("inf"), "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Params().fromJson(
                    {"m2_pg_rail_legalization_displacement_weight": value}
                )


if __name__ == "__main__":
    unittest.main()
