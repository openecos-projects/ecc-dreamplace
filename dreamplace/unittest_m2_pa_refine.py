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
from dreamplace.ops.m2_pa_refine.m2_pa_refine import M2PARefine, STAT_FIELDS


def _make_op(
    widths,
    rail_boxes,
    *,
    heights=None,
    num_movable_nodes=None,
    num_terminals=0,
    max_neighbors=5,
    max_displacement_sites=50,
    xh=30,
):
    if heights is None:
        heights = [1.0] * len(widths)
    if num_movable_nodes is None:
        num_movable_nodes = len(widths) - num_terminals
    return M2PARefine(
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
        max_neighbors=max_neighbors,
        max_displacement_sites=max_displacement_sites,
    )


def _run(op, xs, ys=None):
    if ys is None:
        ys = [0.0] * len(xs)
    return op(torch.tensor(xs + ys, dtype=torch.float32))


class M2PARefineKernelTest(unittest.TestCase):
    def test_empty_rail_geometry_is_a_noop(self):
        output, stats = _run(_make_op([2.0], []), [4.0])

        self.assertTrue(torch.equal(output, torch.tensor([4.0, 0.0])))
        self.assertTrue(all(stats[name] == 0 for name in STAT_FIELDS))

    def test_prefers_lower_displacement_left_move(self):
        output, stats = _run(
            _make_op([2.0, 2.0, 2.0], [[8.0, 0.0, 10.0, 1.0]]),
            [0.0, 4.0, 8.0],
        )

        self.assertEqual(output[:3].tolist(), [0.0, 4.0, 6.0])
        self.assertEqual(stats["overlap_count_before"], 1)
        self.assertEqual(stats["overlap_count_after"], 0)
        self.assertEqual(stats["moved_count"], 1)
        self.assertEqual(stats["total_displacement"], 2)

    def test_fixed_left_obstacle_forces_right_move(self):
        output, stats = _run(
            _make_op(
                [2.0, 4.0],
                [[4.0, 0.0, 6.0, 1.0]],
                num_movable_nodes=1,
                num_terminals=1,
                xh=10,
            ),
            [4.0, 0.0],
        )

        self.assertEqual(output[:2].tolist(), [6.0, 0.0])
        self.assertEqual(stats["overlap_count_after"], 0)

    def test_fixed_right_obstacle_forces_left_move(self):
        output, stats = _run(
            _make_op(
                [2.0, 2.0],
                [[4.0, 0.0, 6.0, 1.0]],
                num_movable_nodes=1,
                num_terminals=1,
                xh=10,
            ),
            [4.0, 6.0],
        )

        self.assertEqual(output[:2].tolist(), [2.0, 6.0])
        self.assertEqual(stats["overlap_count_after"], 0)

    def test_neighbor_limit_controls_chain_shift(self):
        xs = [0.0, 4.0, 6.0]
        rails = [[6.0, 0.0, 8.0, 1.0]]
        blocked, blocked_stats = _run(
            _make_op(
                [2.0, 2.0, 2.0],
                rails,
                max_neighbors=1,
                xh=8,
            ),
            xs,
        )
        shifted, shifted_stats = _run(
            _make_op(
                [2.0, 2.0, 2.0],
                rails,
                max_neighbors=2,
                xh=8,
            ),
            xs,
        )

        self.assertEqual(blocked[:3].tolist(), xs)
        self.assertEqual(blocked_stats["overlap_count_after"], 1)
        self.assertEqual(shifted[:3].tolist(), [0.0, 2.0, 4.0])
        self.assertEqual(shifted_stats["overlap_count_after"], 0)

    def test_displacement_cap_rejects_large_move(self):
        output, stats = _run(
            _make_op(
                [2.0, 2.0, 2.0],
                [[8.0, 0.0, 10.0, 1.0]],
                max_displacement_sites=1,
            ),
            [0.0, 4.0, 8.0],
        )

        self.assertEqual(output[:3].tolist(), [0.0, 4.0, 8.0])
        self.assertEqual(stats["moved_count"], 0)
        self.assertEqual(stats["overlap_count_after"], 1)

    def test_multirow_movable_is_an_obstacle_not_a_move_candidate(self):
        output, stats = _run(
            _make_op(
                [2.0],
                [[4.0, 0.0, 6.0, 2.0]],
                heights=[2.0],
            ),
            [4.0],
        )

        self.assertEqual(output[0].item(), 4.0)
        self.assertEqual(stats["overlap_count_before"], 0)
        self.assertEqual(stats["moved_count"], 0)


class M2PARefineIntegrationTest(unittest.TestCase):
    @staticmethod
    def _params(**overrides):
        values = {
            "m2_pa_refine_flag": 1,
            "ieda_m2_pg_rail_blockage_flag": 0,
            "m2_pg_rail_legalization_blockage_flag": 0,
            "m2_pa_refine_max_neighbors": 5,
            "m2_pa_refine_max_displacement_sites": 50.0,
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
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    @staticmethod
    def _data():
        return types.SimpleNamespace(
            node_size_x=torch.tensor([2.0]),
            node_size_y=torch.tensor([1.0]),
            m2_pg_rail_boxes=torch.tensor([[4.0, 0.0, 6.0, 1.0]]),
        )

    def test_disabled_flag_does_not_build_op(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)

        result = place.build_m2_pa_refine(
            self._params(m2_pa_refine_flag=0), self._placedb(), self._data()
        )

        self.assertIsNone(result)

    def test_illegal_result_is_rolled_back(self):
        class FakeRefine:
            def __init__(self, **kwargs):
                pass

            def __call__(self, pos):
                stats = {name: 0.0 for name in STAT_FIELDS}
                stats.update(
                    {
                        "overlap_count_before": 1.0,
                        "moved_count": 1.0,
                        "total_displacement": 2.0,
                        "max_displacement": 2.0,
                        "move_events": 1.0,
                    }
                )
                return pos + 2.0, stats

        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)
        place.op_collections = types.SimpleNamespace(
            hpwl_op=lambda pos: pos.sum(),
            legality_check_op=lambda pos: False,
        )
        original = torch.tensor([4.0, 0.0])
        with mock.patch(
            "dreamplace.m2_rail_legalization.m2_pa_refine.M2PARefine",
            FakeRefine,
        ):
            op = place.build_m2_pa_refine(
                self._params(), self._placedb(), self._data()
            )
            output, stats = op(original)

        self.assertTrue(torch.equal(output, original))
        self.assertTrue(stats["rollback"])
        self.assertFalse(stats["legal"])
        self.assertEqual(stats["hpwl_delta"], 0)

    def test_global_hard_m2_mode_is_supported(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)

        op = place.build_m2_pa_refine(
            self._params(ieda_m2_pg_rail_blockage_flag=1),
            self._placedb(),
            self._data(),
        )

        self.assertTrue(callable(op))

    def test_fences_fail_clearly(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)

        with self.assertRaisesRegex(ValueError, "fence"):
            place.build_m2_pa_refine(
                self._params(),
                self._placedb(regions=[[[0, 0, 1, 1]]]),
                self._data(),
            )

    def test_legalization_only_m2_mode_is_supported(self):
        place = BasicPlace.__new__(BasicPlace)
        torch.nn.Module.__init__(place)

        op = place.build_m2_pa_refine(
            self._params(m2_pg_rail_legalization_blockage_flag=1),
            self._placedb(),
            self._data(),
        )

        self.assertTrue(callable(op))

    def test_schema_and_parameter_normalization(self):
        params_path = os.path.join(os.path.dirname(__file__), "params.json")
        with open(params_path, "r", encoding="utf-8") as stream:
            schema = json.load(stream)
        self.assertEqual(schema["m2_pa_refine_flag"]["default"], 0)
        self.assertEqual(schema["m2_pa_refine_max_neighbors"]["default"], 5)
        self.assertEqual(
            schema["m2_pa_refine_max_displacement_sites"]["default"], 50
        )

        params = Params()
        params.fromJson(
            {
                "m2_pa_refine_max_neighbors": "7",
                "m2_pa_refine_max_displacement_sites": "12.5",
            }
        )
        self.assertEqual(params.m2_pa_refine_max_neighbors, 7)
        self.assertEqual(params.m2_pa_refine_max_displacement_sites, 12.5)

        for key, value in (
            ("m2_pa_refine_max_neighbors", 0),
            ("m2_pa_refine_max_neighbors", 1.5),
            ("m2_pa_refine_max_displacement_sites", -1),
            ("m2_pa_refine_max_displacement_sites", float("inf")),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                Params().fromJson({key: value})


if __name__ == "__main__":
    unittest.main()
