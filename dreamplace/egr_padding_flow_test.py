import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreamplace.NonLinearPlace import NonLinearPlace


class EGRPaddingFlowTest(unittest.TestCase):
    def test_egr_padding_runs_irt_after_initial_legalization_and_padded_legalization_sees_padding(self):
        calls = []

        inst = object.__new__(NonLinearPlace)
        inst._egr_padding_state = None
        inst.pos = [torch.tensor([0.0, 10.0, 0.0, 0.0])]
        inst.data_collections = SimpleNamespace(
            node_size_x=torch.tensor([2.0, 2.0]),
            node_size_y=torch.tensor([1.0, 1.0]),
            pin_offset_x=torch.tensor([0.0, 0.0]),
            pin2node_map=torch.tensor([0, 1]),
            movable_macro_mask=torch.tensor([False, False]),
            net_weights=torch.tensor([2.0]),
        )

        def legalize_op(pos):
            calls.append(
                (
                    "legalize",
                    inst.data_collections.node_size_x.clone().tolist(),
                    pos.clone().tolist(),
                )
            )
            return pos + 1.0

        def congestion_map_op(pos, stage, resolve_congestion):
            calls.append(("irt", pos.clone().tolist(), stage, resolve_congestion))
            return torch.tensor([[5.0, 5.0], [5.0, 5.0]])

        class Metric:
            def __init__(self, iteration):
                calls.append(("metric_init", iteration))

            def evaluate(self, placedb, ops, pos):
                calls.append(("metric_eval", pos.clone().tolist()))

        inst.op_collections = SimpleNamespace(
            legalize_op=legalize_op,
            irt_egr_congestion_map_op=congestion_map_op,
            hpwl_op=lambda *args, **kwargs: 0,
            rsmt_wl_op=lambda *args, **kwargs: torch.tensor(0.0),
        )

        placedb = SimpleNamespace(
            num_movable_nodes=2,
            num_nodes=2,
            routing_grid_xl=0.0,
            routing_grid_yl=0.0,
            routing_grid_size_x=10.0,
            routing_grid_size_y=10.0,
            site_width=1.0,
            dbu=1.0,
            apply=lambda params, x, y: calls.append(("apply", x.tolist(), y.tolist())),
        )
        params = SimpleNamespace(
            global_place_flag=0,
            legalize_flag=1,
            macro_place_flag=0,
            egr_padding_flag=1,
            cell_padding_x=-1,
            plot_flag=0,
            dump_legalize_solution_flag=0,
            detailed_place_flag=0,
            with_sta=False,
        )

        import dreamplace.NonLinearPlace as nonlinear_module

        original_metric = nonlinear_module.EvalMetrics.EvalMetrics
        nonlinear_module.EvalMetrics.EvalMetrics = Metric
        try:
            inst(params, placedb)
        finally:
            nonlinear_module.EvalMetrics.EvalMetrics = original_metric

        legalize_calls = [call for call in calls if call[0] == "legalize"]
        irt_call = next(call for call in calls if call[0] == "irt")
        self.assertEqual(len(legalize_calls), 2)
        self.assertLess(calls.index(legalize_calls[0]), calls.index(irt_call))
        self.assertLess(calls.index(irt_call), calls.index(legalize_calls[1]))
        self.assertEqual(legalize_calls[0][1], [2.0, 2.0])
        self.assertTrue(any(size > 2.0 for size in legalize_calls[1][1]))
        self.assertEqual(inst.data_collections.node_size_x.tolist(), [2.0, 2.0])
        self.assertIsNone(inst._egr_padding_state)


if __name__ == "__main__":
    unittest.main()
