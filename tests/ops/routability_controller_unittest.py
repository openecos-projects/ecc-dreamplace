import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from dreamplace.EvalMetrics import EvalMetrics
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.PlaceObj import PlaceObj
from dreamplace.ops.routability.gpugr_context import sync_route_grid_to_autodmp
from dreamplace.ops.routability.l_shape_policy import LShapePolicy
from dreamplace.ops.routability.routability_controller import RoutabilityController


class RoutabilityControllerTest(unittest.TestCase):
    def test_l_shape_overflow_target_updates_follow_ema_deadband_and_bounds(self):
        params = SimpleNamespace(
            l_shape_overflow_ema_beta=0.5,
            l_shape_overflow_deadband=0.01,
            l_shape_overflow_update_k=2.0,
            l_shape_grad_target_ratio_min=0.05,
            l_shape_grad_target_ratio_max=0.2,
            l_shape_auto_disable_flag=0,
        )
        policy = LShapePolicy(params)
        model = SimpleNamespace(l_shape_grad_target_ratio=0.1)

        policy.update_overflow_target(model, 0, 3.0, 0.1)
        self.assertEqual(
            (model._l_shape_overflow_ema, model._l_shape_overflow_last,
             model.l_shape_grad_target_ratio),
            (0.1, 0.1, 0.1),
        )
        policy.update_overflow_target(model, 1, 4.0, 0.11)
        self.assertEqual(model.l_shape_grad_target_ratio, 0.1)
        policy.update_overflow_target(model, 2, 5.0, 1.0)
        self.assertEqual(model.l_shape_grad_target_ratio, 0.2)
        self.assertAlmostEqual(model._l_shape_overflow_ema, 0.5525)
        policy.update_overflow_target(model, 3, 1.0, 0.0)
        self.assertAlmostEqual(model.l_shape_grad_target_ratio, 0.2 * math.exp(-0.5525))

    def test_flow_orders_legalization_padding_writeback_and_final_route(self):
        import dreamplace.NonLinearPlace as nonlinear_module

        trace = []
        engine = object.__new__(NonLinearPlace)
        torch.nn.Module.__init__(engine)
        engine.size_params = torch.nn.ParameterDict()
        engine.vt_params = torch.nn.ParameterDict()
        engine.pos = [torch.tensor([0.0, 1.0, 0.0, 0.0])]
        engine.data_collections = SimpleNamespace(
            net_weights=torch.ones(1), get_size_var=lambda: None, get_vt_var=lambda: None
        )
        engine.op_collections = SimpleNamespace(
            legalize_op=lambda pos: trace.append("legalize") or pos,
            hpwl_op=lambda pos: torch.tensor(1.0),
            rsmt_wl_op=lambda pos: torch.tensor(1.0),
            m2_pa_refine_op=None,
        )
        placedb = SimpleNamespace(
            num_movable_nodes=2,
            num_nodes=2,
            dbu=1.0,
            apply=lambda *args: trace.append("apply"),
        )
        params = SimpleNamespace(
            global_place_flag=0,
            stop_overflow=0.1,
            routability_opt_flag=1,
            legalize_flag=1,
            macro_place_flag=0,
            egr_padding_flag=0,
            cell_padding_x=-1,
            plot_flag=0,
            dump_legalize_solution_flag=0,
            detailed_place_flag=0,
            with_sta=False,
            timing_opt_flag=0,
            result_dir=self.enterContext(tempfile.TemporaryDirectory()),
            design_name=lambda: "lifecycle",
        )

        class Metric(EvalMetrics):
            def evaluate(self, *args):
                pass

        with (
            mock.patch.object(
                nonlinear_module,
                "RoutabilityController",
                side_effect=lambda config: RoutabilityController(
                    config, event_sink=lambda name, payload: trace.append(name)
                ),
            ),
            mock.patch.object(nonlinear_module.EvalMetrics, "EvalMetrics", Metric),
            mock.patch.object(
                nonlinear_module.route_evaluation,
                "run_post_legalization_adaptive_padding",
                side_effect=lambda *args: (trace.append("adaptive_padding") or args[2], None),
            ),
            mock.patch.object(
                nonlinear_module.route_evaluation,
                "run_gpugr_final_eval",
                side_effect=lambda *args: trace.append("final_gpugr"),
            ),
        ):
            engine(params, placedb)

        self.assertEqual(
            trace,
            [
                "before_legalization",
                "legalize",
                "after_legalization",
                "adaptive_padding",
                "apply",
                "finalize",
                "final_gpugr",
            ],
        )

    def test_fake_backend_trace_preserves_lifecycle_order(self):
        trace = []

        def record(name, payload):
            trace.append(name)

        controller = RoutabilityController(
            SimpleNamespace(routability_opt_flag=1), event_sink=record
        )
        model = object()
        position = object()
        metrics = object()

        controller.before_stage(model=model, stage_idx=0)
        controller.before_iteration(model=model, iteration=7, position=position)
        controller.after_iteration(model=model, iteration=7, metrics=metrics)
        controller.before_area_adjust(model=model, position=position, maps={})
        controller.after_geometry_change(model=model, position=position)
        controller.before_legalization(model=model, position=position)
        controller.after_legalization(model=model, position=position)
        controller.finalize(model=model, position=position)

        self.assertEqual(
            trace,
            [
                "before_stage",
                "before_iteration",
                "after_iteration",
                "before_area_adjust",
                "after_geometry_change",
                "before_legalization",
                "after_legalization",
                "finalize",
            ],
        )

    def test_plain_path_still_accepts_hooks_without_routability(self):
        trace = []
        controller = RoutabilityController(
            SimpleNamespace(routability_opt_flag=0),
            event_sink=lambda name, payload: trace.append(name),
        )

        controller.before_stage(model=None, stage_idx=0)
        controller.before_iteration(model=None, iteration=0, position=None)

        self.assertFalse(controller.enabled)
        self.assertEqual(trace, [])

    def test_place_obj_owns_l_shape_metric_projection(self):
        model = object.__new__(PlaceObj)
        model.use_l_shape_routability = True
        model.l_shape_last_cost = 1.5
        model.l_shape_last_grad_norm = 2.0
        model.l_shape_last_weighted_cost = 0.75
        model.l_shape_last_weight = 0.5
        model.l_shape_last_target_weight = 0.4
        model.l_shape_last_weight_candidate = 0.6
        model.l_shape_last_cap_active = True
        model.l_shape_last_base_grad_norm = 3.0
        model.l_shape_last_grad_raw_norm = 2.5
        model.l_shape_last_grad_ratio = 0.8
        model.l_shape_fast_mode = False
        model.l_shape_energy_valid = True
        model.l_shape_grad_target_ratio = 0.2
        model._l_shape_overflow_ema = 0.1
        model.l_shape_capacity_al_last_summary = {"g_h_max": 3.0}
        model.l_shape_macro_exclusion_last_summary = {}
        model.soft_l_last_summary = {"tau": 0.25}
        model.params = SimpleNamespace()

        metric = EvalMetrics(0)
        model.collect_l_shape_telemetry(metric)

        self.assertEqual(metric.l_shape_cost, 1.5)
        self.assertEqual(metric.l_shape_grad_ratio, 0.8)
        self.assertEqual(metric.l_shape_capacity_al_g_h_max, 3.0)
        self.assertEqual(metric.soft_l_tau, 0.25)

    def test_non_linear_place_keeps_backend_implementations_out_of_flow_module(self):
        source = (
            Path(__file__).parents[2] / "dreamplace" / "NonLinearPlace.py"
        ).read_text()

        self.assertNotIn("gpugr_op.run_gpugr(", source)
        self.assertNotIn("def _build_gpugr_", source)
        self.assertNotIn("def _prepare_l_shape_inputs_from_", source)
        self.assertIn("RoutabilityController", source)

    def test_place_obj_refresh_facade_resets_geometry_dependent_operators(self):
        class Resettable:
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        model = object.__new__(PlaceObj)
        density = Resettable()
        overflow = Resettable()
        pin = Resettable()
        model.op_collections = SimpleNamespace(
            density_op=density,
            density_overflow_op=overflow,
            pin_utilization_map_op=pin,
        )
        model._virtual_cell_density_view = SimpleNamespace(_density_ops={1: object()})
        model._timing_geometry_cache = object()

        model.refresh_after_geometry_change()

        self.assertEqual(
            [density.reset_count, overflow.reset_count, pin.reset_count],
            [1, 1, 1],
        )
        self.assertEqual(model._virtual_cell_density_view._density_ops, {})
        self.assertIsNone(model._timing_geometry_cache)

    def test_grid_sync_rebuilds_objective_owned_operators_only_when_grid_changes(self):
        model = object.__new__(PlaceObj)
        model.params = SimpleNamespace(
            route_num_bins_x=8,
            route_num_bins_y=8,
            auto_adjust_bins=0,
        )
        model.placedb = SimpleNamespace(num_routing_grids_x=4, num_routing_grids_y=4)
        model.data_collections = object()
        model.op_collections = SimpleNamespace()
        pin_operator = object()
        adjust_operator = object()

        with (
            mock.patch.object(PlaceObj, "build_pin_utilization_map", return_value=pin_operator) as build_pin,
            mock.patch.object(PlaceObj, "build_adjust_node_area", return_value=adjust_operator) as build_adjust,
        ):
            sync_route_grid_to_autodmp(model.params, model.placedb, model)
            sync_route_grid_to_autodmp(model.params, model.placedb, model)

        self.assertEqual(
            (model.placedb.num_routing_grids_x, model.placedb.num_routing_grids_y),
            (8, 8),
        )
        self.assertIs(model.op_collections.pin_utilization_map_op, pin_operator)
        self.assertIs(model.op_collections.adjust_node_area_op, adjust_operator)
        build_pin.assert_called_once_with(model.params, model.placedb, model.data_collections)
        build_adjust.assert_called_once_with(model.params, model.placedb, model.data_collections)

    def test_l_shape_policy_disables_and_lowers_reenable_threshold_after_inflation(self):
        model = SimpleNamespace(
            use_l_shape_routability=True,
            enable_l_shape_routability=True,
            reset_l_shape_weight_state=lambda: None,
        )
        policy = LShapePolicy(
            SimpleNamespace(l_shape_overflow_threshold=0.2)
        )
        policy.reset_reenable_state(model)
        policy.disable_for_recovery(
            model,
            iteration=10,
            reason="inflation",
            update_threshold=True,
            inflation_round=1,
        )

        self.assertFalse(model.use_l_shape_routability)
        self.assertFalse(model.enable_l_shape_routability)
        self.assertTrue(model._l_shape_inflation_guard_disabled)
        self.assertEqual(model._l_shape_reenable_count, 1)
        self.assertAlmostEqual(model._l_shape_reenable_threshold, 0.14)

    def test_l_shape_policy_reenables_on_low_overflow_without_guard(self):
        model = SimpleNamespace(
            enable_l_shape_routability=False,
            _l_shape_auto_disabled=False,
            _l_shape_ratio_guard_disabled=False,
            _l_shape_inflation_guard_disabled=False,
            _l_shape_reenable_threshold=0.2,
            _l_shape_reenable_count=0,
        )
        LShapePolicy(SimpleNamespace(l_shape_overflow_threshold=0.2)).maybe_reenable(
            model, 0.1
        )

        self.assertTrue(model.enable_l_shape_routability)


if __name__ == "__main__":
    unittest.main()
