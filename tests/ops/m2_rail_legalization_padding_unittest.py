import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from dreamplace.ops.m2_legalize import m2_rail_legalization


class DirectAbacusPaddingTest(unittest.TestCase):
    def setUp(self):
        self.placedb = SimpleNamespace(
            regions=[], num_movable_nodes=2, num_nodes=2,
            num_terminals=0, num_terminal_NIs=0, num_filler_nodes=0,
            xl=0.0, yl=0.0, xh=12.0, yh=1.0,
            site_width=1.0, row_height=1.0, num_bins_x=1, num_bins_y=1,
        )
        data = SimpleNamespace(
            node_size_x=torch.ones(2), node_size_y=torch.ones(2),
            num_pins_in_nodes=torch.ones(2), flat_region_boxes=torch.empty(0),
            flat_region_boxes_start=torch.tensor([0]),
            node2fence_region_map=torch.full((2,), -1), fp_info=object(),
        )
        self.model = SimpleNamespace(
            data_collections=data,
            op_collections=SimpleNamespace(legality_check_op=lambda pos: True),
        )
        self.pos = torch.tensor([2.0, 6.0, 0.0, 0.0])

    def test_direct_abacus_skips_global_greedy_when_padded_result_is_legal(self):
        abacus = mock.Mock(return_value=lambda initial, current: current.clone())
        greedy = mock.Mock()
        macro = mock.Mock()
        with (
            mock.patch.object(m2_rail_legalization.legality_check, "LegalityCheck",
                              return_value=lambda pos: True),
            mock.patch.object(m2_rail_legalization.abacus_legalize,
                              "AbacusLegalize", abacus),
            mock.patch.object(m2_rail_legalization.greedy_legalize,
                              "GreedyLegalize", greedy),
            mock.patch.object(m2_rail_legalization.macro_legalize,
                              "MacroLegalize", macro),
        ):
            result, stats = m2_rail_legalization.run_adaptive_padding_legalization(
                self.model, self.placedb, self.pos,
                padding_sites=np.array([1, 0]), scores=np.array([2.0, 0.0]),
                max_retries=0, prefer_direct_abacus=True,
            )
        torch.testing.assert_close(result, self.pos)
        self.assertFalse(stats["rollback"])
        abacus.assert_called_once()
        greedy.assert_not_called()
        macro.assert_not_called()

    def test_illegal_direct_abacus_uses_existing_greedy_path(self):
        abacus = mock.Mock(return_value=lambda initial, current: current.clone())
        greedy = mock.Mock(return_value=lambda initial, current: current.clone())
        macro = mock.Mock(return_value=lambda initial, current: current.clone())
        padded_legality = iter((False, True, True))
        with (
            mock.patch.object(m2_rail_legalization.legality_check, "LegalityCheck",
                              return_value=lambda pos: next(padded_legality)),
            mock.patch.object(m2_rail_legalization.abacus_legalize,
                              "AbacusLegalize", abacus),
            mock.patch.object(m2_rail_legalization.greedy_legalize,
                              "GreedyLegalize", greedy),
            mock.patch.object(m2_rail_legalization.macro_legalize,
                              "MacroLegalize", macro),
        ):
            _, stats = m2_rail_legalization.run_adaptive_padding_legalization(
                self.model, self.placedb, self.pos,
                padding_sites=np.array([1, 0]), scores=np.array([2.0, 0.0]),
                max_retries=0, prefer_direct_abacus=True,
            )
        self.assertFalse(stats["rollback"])
        self.assertEqual(abacus.call_count, 1)
        greedy.assert_called_once()
        macro.assert_called_once()


if __name__ == "__main__":
    unittest.main()
