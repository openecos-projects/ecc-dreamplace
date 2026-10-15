#!/usr/bin/env python3

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


class FlowConfigTest(unittest.TestCase):
    def test_infer_flow_kind_from_existing_params(self):
        from dreamplace.flows.flow_config import FlowKind, infer_flow_kind

        self.assertEqual(FlowKind.PLACEMENT, infer_flow_kind(types.SimpleNamespace()))
        self.assertEqual(
            FlowKind.SIZING,
            infer_flow_kind(types.SimpleNamespace(placement_sizing_mode="size_only")),
        )
        self.assertEqual(
            FlowKind.JOINT,
            infer_flow_kind(types.SimpleNamespace(placement_sizing_mode="joint")),
        )
        self.assertEqual(
            FlowKind.BUFFERING,
            infer_flow_kind(
                types.SimpleNamespace(buffering_continuous_relaxed_optimization=1)
            ),
        )
        self.assertEqual(
            FlowKind.STA,
            infer_flow_kind(types.SimpleNamespace(flow_kind="sta")),
        )

    def test_explicit_flow_kind_overrides_inference(self):
        from dreamplace.flows.flow_config import FlowKind, infer_flow_kind

        self.assertEqual(
            FlowKind.BUFFERING,
            infer_flow_kind(
                types.SimpleNamespace(
                    flow_kind="buffering",
                    placement_sizing_mode="size_only",
                )
            ),
        )

    def test_timing_driven_placement_is_a_placement_profile(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="placement",
            diff_timing_driven_placement=1,
        )

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.PLACEMENT, flow_kind)
        self.assertEqual("placement", params.flow_kind)
        self.assertEqual(3000, params.global_place_stages[0]["iteration"])
        self.assertEqual(256, params.global_place_stages[0]["num_bins_x"])
        self.assertEqual(256, params.global_place_stages[0]["num_bins_y"])
        self.assertEqual(3000, params.random_seed)
        self.assertEqual(0.8, params.target_density)
        self.assertEqual(0.1, params.stop_overflow)
        self.assertEqual(1, params.diff_timing_driven_placement)
        self.assertEqual(1, params.with_sta)
        self.assertEqual(15, params.timing_topology_refresh_interval)
        self.assertEqual(0.35, params.timing_topology_enable_overflow_threshold)
        self.assertEqual(0.0005, params.pin2pin_weight)
        self.assertEqual(10.0, params.pin2pin_min_weight)
        self.assertEqual(50.0, params.pin2pin_max_weight)
        self.assertEqual(0.2, params.pin2pin_accumulate_weight)

    def test_resolve_flow_config_copies_input(self):
        from dreamplace.flows.flow_config import resolve_flow_config

        config = {
            "flow_kind": "placement",
            "diff_timing_driven_placement": 1,
        }

        resolved = resolve_flow_config(config)

        self.assertEqual(1, resolved["diff_timing_driven_placement"])
        self.assertEqual(15, resolved["timing_topology_refresh_interval"])
        self.assertEqual(
            {"flow_kind": "placement", "diff_timing_driven_placement": 1},
            config,
        )

    def test_pin2pin_timing_placement_disables_differentiable_objective(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="placement",
            enable_net_weighting=1,
            net_weighting_scheme="pin2pin",
        )

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.PLACEMENT, flow_kind)
        self.assertEqual(1, params.with_sta)
        self.assertEqual(1, params.timing_eval_flag)
        self.assertEqual(0, params.diff_timing_driven_placement)
        self.assertEqual(0, params.differentiable_timing_obj)
        self.assertEqual(1, params.pin2pin_net_weighting)
        self.assertEqual(15, params.timing_topology_refresh_interval)

    def test_pin2pin_timing_placement_rejects_differentiable_objective(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="placement",
            enable_net_weighting=1,
            net_weighting_scheme="pin2pin",
            differentiable_timing_obj=1,
            _differentiable_timing_obj_explicit=True,
        )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            apply_flow_defaults(params)

    def test_apply_sizing_flow_defaults_sets_recommended_entry_defaults(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(flow_kind="sizing")

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.SIZING, flow_kind)
        self.assertEqual("size_only", params.placement_sizing_mode)
        self.assertEqual(100, params.global_place_stages[0]["iteration"])
        self.assertEqual("discrete_gradient_topk", params.continuous_size_dynamics_mode)
        self.assertEqual("surrogate_only", params.timing_surrogate_mode)
        self.assertEqual("nearest_size_round", params.projection_resolver)
        self.assertEqual(
            "main_id_arc_offset_piecewise_linear", params.cell_model_schema
        )
        self.assertEqual("off", params.critical_endpoint_pruning_mode)
        self.assertTrue(params.production_fast_loop)
        self.assertEqual("auto", params.timing_lut_2d_native_op)
        self.assertEqual("auto", params.size_interpolated_pin_native_op)
        self.assertEqual("real_size_taylor", params.discrete_gradient_topk_ranking_mode)
        self.assertEqual(30.0, params.discrete_gradient_topk_up_percent)
        self.assertEqual(0.0, params.discrete_gradient_topk_down_percent)
        self.assertEqual(10.0, params.discrete_gradient_topk_vt_percent)
        self.assertEqual(1, params.discrete_gradient_topk_shared_budget)
        self.assertEqual(1.0, params.discrete_gradient_topk_shared_budget_percent)
        self.assertTrue(params.early_stop_restore_best)

    def test_openroad_vt_suffixes_have_configurable_default_and_normalization(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="sizing",
            openroad_vt_suffixes=" H7H, H7R;H7L,H7R ",
        )

        apply_flow_defaults(params)

        self.assertEqual(("H7H", "H7R", "H7L"), params.openroad_vt_suffixes)

    def test_openroad_vt_suffixes_default_is_ics55_configuration(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(flow_kind="sizing")
        apply_flow_defaults(params)

        self.assertEqual(("H7H", "H7R", "H7L"), params.openroad_vt_suffixes)

    def test_openroad_vt_suffixes_custom_value_survives_explicit_placement_flow(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="placement",
            openroad_vt_suffixes=["SVT", "LVT"],
        )
        apply_flow_defaults(params)

        self.assertEqual(("SVT", "LVT"), params.openroad_vt_suffixes)

    def test_apply_sizing_flow_defaults_preserves_user_overrides(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            placement_sizing_mode="size_only",
            continuous_size_dynamics_mode="custom_mode",
            timing_surrogate_mode="custom_surrogate",
            projection_resolver="custom_projection",
            critical_endpoint_pruning_mode="dynamic",
            production_fast_loop=False,
            timing_lut_2d_native_op="off",
            size_interpolated_pin_native_op="off",
            global_place_stages=[{"iteration": 12}],
            _global_place_stages_explicit=True,
            discrete_gradient_topk_up_percent=12.5,
            discrete_gradient_topk_down_percent=2.0,
            discrete_gradient_topk_vt_percent=7.5,
            discrete_gradient_topk_shared_budget=0,
            discrete_gradient_topk_shared_budget_percent=2.5,
            _discrete_gradient_topk_shared_budget_explicit=True,
            _discrete_gradient_topk_shared_budget_percent_explicit=True,
            discrete_gradient_topk_ranking_mode="raw_gradient",
            early_stop_restore_best=False,
        )

        apply_flow_defaults(params)

        self.assertEqual("custom_mode", params.continuous_size_dynamics_mode)
        self.assertEqual(12, params.global_place_stages[0]["iteration"])
        self.assertEqual("custom_surrogate", params.timing_surrogate_mode)
        self.assertEqual("custom_projection", params.projection_resolver)
        self.assertEqual("dynamic", params.critical_endpoint_pruning_mode)
        self.assertFalse(params.production_fast_loop)
        self.assertEqual("off", params.timing_lut_2d_native_op)
        self.assertEqual("off", params.size_interpolated_pin_native_op)
        self.assertEqual(12.5, params.discrete_gradient_topk_up_percent)
        self.assertEqual(2.0, params.discrete_gradient_topk_down_percent)
        self.assertEqual(7.5, params.discrete_gradient_topk_vt_percent)
        self.assertEqual(0, params.discrete_gradient_topk_shared_budget)
        self.assertEqual(2.5, params.discrete_gradient_topk_shared_budget_percent)

    def test_apply_sta_flow_defaults_enables_timing_without_optimization_loop(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="sta", global_place_stages=[{"iteration": 20}]
        )

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.STA, flow_kind)
        self.assertEqual("sta", params.flow_kind)
        self.assertEqual("size_only", params.placement_sizing_mode)
        self.assertEqual(1, params.with_sta)
        self.assertEqual("lut_only", params.timing_surrogate_mode)
        self.assertEqual("auto", params.timing_lut_2d_native_op)
        self.assertEqual("auto", params.size_interpolated_pin_native_op)
        self.assertEqual(0, params.global_place_stages[0]["iteration"])

    def test_apply_buffering_flow_defaults_sets_recommended_dry_run_profile(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(flow_kind="buffering")

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.BUFFERING, flow_kind)
        self.assertEqual("buffering", params.flow_kind)
        self.assertEqual(1, params.buffering_continuous_relaxed_optimization)
        self.assertEqual(1, params.buffering_segment_count_tns_gradient)
        self.assertEqual("segment_only", params.buffering_candidate_policy)
        self.assertEqual(3, params.buffering_max_repeaters_per_segment)
        self.assertEqual(0, params.buffering_include_tree_node_candidates)
        self.assertEqual(
            "dynamic_net_provider",
            params.buffering_relaxed_timing_integration_mode,
        )
        self.assertEqual(0.05, params.buffering_continuous_initial_relaxed_bu)
        self.assertEqual(0.1, params.buffering_segment_count_z_init)
        self.assertEqual(
            "cpp_cuda_segment_transfer_explicit_autograd",
            params.buffering_segment_count_timing_backend,
        )
        self.assertFalse(hasattr(params, "buffering_enable_real_coordinate_commit"))

    def test_apply_buffering_flow_defaults_honors_candidate_mode_profile(self):
        from dreamplace.flows.flow_config import FlowKind, apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="buffering", buffering_mode="candidate"
        )

        flow_kind = apply_flow_defaults(params)

        self.assertEqual(FlowKind.BUFFERING, flow_kind)
        self.assertEqual("candidate", params.buffering_mode)
        self.assertEqual(1, params.buffering_continuous_relaxed_optimization)
        self.assertEqual(0, params.buffering_segment_count_tns_gradient)

    def test_apply_physical_eco_flow_kind_is_explicitly_rejected(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(flow_kind="physical_eco")

        with self.assertRaisesRegex(ValueError, "physical_eco.*not supported"):
            apply_flow_defaults(params)

    def test_apply_physical_eco_candidate_mode_is_explicitly_rejected(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="physical_eco",
            buffering_mode="candidate",
        )

        with self.assertRaisesRegex(ValueError, "physical_eco.*not supported"):
            apply_flow_defaults(params)

    def test_apply_buffering_flow_defaults_preserves_explicit_user_controls(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="buffering",
            buffering_continuous_steps=7,
            _buffering_continuous_steps_explicit=True,
            buffering_continuous_lr=0.2,
            _buffering_continuous_lr_explicit=True,
            buffering_max_selected_actions=9,
            _buffering_max_selected_actions_explicit=True,
            buffering_segment_count_max_repeater_count=6,
            _buffering_segment_count_max_repeater_count_explicit=True,
            buffering_max_repeaters_per_segment=5,
            _buffering_max_repeaters_per_segment_explicit=True,
        )

        apply_flow_defaults(params)

        self.assertEqual(7, params.buffering_continuous_steps)
        self.assertEqual(0.2, params.buffering_continuous_lr)
        self.assertEqual(9, params.buffering_max_selected_actions)
        self.assertEqual(6, params.buffering_segment_count_max_repeater_count)
        self.assertEqual(5, params.buffering_max_repeaters_per_segment)

    def test_explicit_sizing_flow_kind_overrides_stale_json_defaults(self):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(
            flow_kind="sizing",
            continuous_size_dynamics_mode="discrete_gradient_topk",
            timing_surrogate_mode="mixed",
            projection_resolver="argmin",
            critical_endpoint_pruning_mode="dynamic",
            production_fast_loop=False,
            timing_lut_2d_native_op="off",
            size_interpolated_pin_native_op="off",
            global_place_stages=[{"iteration": 20}],
            discrete_gradient_topk_up_percent=12.5,
            discrete_gradient_topk_down_percent=2.0,
        )

        apply_flow_defaults(params)

        self.assertEqual("discrete_gradient_topk", params.continuous_size_dynamics_mode)
        self.assertEqual("surrogate_only", params.timing_surrogate_mode)
        self.assertEqual("nearest_size_round", params.projection_resolver)
        self.assertEqual("off", params.critical_endpoint_pruning_mode)
        self.assertTrue(params.production_fast_loop)
        self.assertEqual("auto", params.timing_lut_2d_native_op)
        self.assertEqual("auto", params.size_interpolated_pin_native_op)
        self.assertEqual(100, params.global_place_stages[0]["iteration"])
        self.assertEqual("real_size_taylor", params.discrete_gradient_topk_ranking_mode)
        self.assertEqual(30.0, params.discrete_gradient_topk_up_percent)
        self.assertEqual(0.0, params.discrete_gradient_topk_down_percent)
        self.assertTrue(params.early_stop_restore_best)

    def test_nonlinear_place_sizing_fallbacks_match_canonical_entry(self):
        from dreamplace.NonLinearPlace import NonLinearPlace

        placer = NonLinearPlace.__new__(NonLinearPlace)
        config = placer._continuous_size_dynamics_config(types.SimpleNamespace())

        self.assertEqual(
            "real_size_taylor", config["discrete_gradient_topk_ranking_mode"]
        )
        self.assertEqual(30.0, config["discrete_gradient_topk_up_percent"])
        self.assertEqual(0.0, config["discrete_gradient_topk_down_percent"])

    def test_continuous_early_stop_restore_tracks_timing_loss_before_objective(self):
        from dreamplace.NonLinearPlace import NonLinearPlace

        placer = NonLinearPlace.__new__(NonLinearPlace)
        metric = types.SimpleNamespace(objective=1.0, combined_timing_loss=3.0)

        loss, source = placer._continuous_early_stop_loss_and_source(
            types.SimpleNamespace(), metric
        )

        self.assertEqual(3.0, loss)
        self.assertEqual("combined_timing_loss", source)

    def test_continuous_early_stop_restore_can_run_without_patience(self):
        from dreamplace.NonLinearPlace import NonLinearPlace

        placer = NonLinearPlace.__new__(NonLinearPlace)
        state = placer._build_continuous_early_stop_state(
            types.SimpleNamespace(early_stop_restore_best=True)
        )

        self.assertTrue(state["enabled"])
        self.assertTrue(state["restore_best"])
        self.assertIsNone(state["patience"])

    def test_timing_lut_2d_native_op_auto_targets_gpu_fast_loop(self):
        from dreamplace.ops.timing_propagation.timing_propagation import (
            TimingPropagation,
        )

        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        model.timing_lut_2d_native_op = model._resolve_timing_lut_2d_native_op_mode(
            "auto"
        )

        self.assertTrue(
            model._use_lut_2d_native_op_for_tensor(types.SimpleNamespace(is_cuda=True))
        )
        self.assertFalse(
            model._use_lut_2d_native_op_for_tensor(types.SimpleNamespace(is_cuda=False))
        )

        model.production_fast_loop = False
        self.assertFalse(
            model._use_lut_2d_native_op_for_tensor(types.SimpleNamespace(is_cuda=True))
        )

    def test_timing_lut_2d_native_op_explicit_modes(self):
        from dreamplace.ops.timing_propagation.timing_propagation import (
            TimingPropagation,
        )

        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = False
        tensor = types.SimpleNamespace(is_cuda=False)

        model.timing_lut_2d_native_op = model._resolve_timing_lut_2d_native_op_mode(
            "on"
        )
        self.assertTrue(model._use_lut_2d_native_op_for_tensor(tensor))

        model.timing_lut_2d_native_op = model._resolve_timing_lut_2d_native_op_mode(
            "off"
        )
        self.assertFalse(model._use_lut_2d_native_op_for_tensor(tensor))

    def test_timing_lut_2d_native_op_rejects_unknown_mode(self):
        from dreamplace.ops.timing_propagation.timing_propagation import (
            TimingPropagation,
        )

        model = TimingPropagation.__new__(TimingPropagation)

        with self.assertRaises(ValueError):
            model._resolve_timing_lut_2d_native_op_mode("maybe")


if __name__ == "__main__":
    unittest.main()
