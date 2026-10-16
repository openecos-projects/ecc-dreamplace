import os
import sys
import json
import csv
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.size_interpolated_pin import sizing_limit_utils
from dreamplace.macroPlaceDB import MacroPlaceDB
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.PlaceObj import PlaceObj
from dreamplace.flows.joint_proximal import JointProximalObjective
from dreamplace.ops.timing_propagation.timing_propagation import (
    ARCS_INFO,
    LUTS_INFO,
    TimingPropagation,
    build_pin_violation_detail_rows,
)
sys.path.pop()


class SizingModeTest(unittest.TestCase):
    def _make_size_only_place_obj(
        self,
        timing_lane="timing_only",
        slew_weight=1.0,
        cap_weight=1.0,
        leakage_weight=0.0,
    ):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            macro_overlap_flag=False,
            enable_net_weighting=False,
            timing_objective_lane=timing_lane,
            timing_slew_weight=slew_weight,
            timing_cap_weight=cap_weight,
            timing_leakage_weight=leakage_weight,
        )
        obj.timing_wns_coeff = 0.01
        obj.timing_tns_coeff = 0.0001
        obj.use_timing_obj = True
        obj.placedb = SimpleNamespace(num_movable_nodes=0, regions=[])
        obj.density_factor = 1.0
        obj.density_weight = torch.tensor([1.0], dtype=torch.float32)
        obj.init_density = None
        obj.quad_penalty = False
        obj.density_quad_coeff = 0.0
        obj.size_density_area_weight = 0.0
        obj.op_collections = SimpleNamespace(
            wirelength_op=lambda pos: torch.tensor(0.0, dtype=pos.dtype),
            density_op=lambda pos: torch.tensor(0.0, dtype=pos.dtype),
            timing_propagation_op=SimpleNamespace(
                last_total_slew_violation_tensor=torch.tensor(4.0, dtype=torch.float32),
                last_total_cap_violation_tensor=torch.tensor(5.0, dtype=torch.float32),
                last_total_leakage_tensor=torch.tensor(0.0, dtype=torch.float32),
                last_total_slew_violation=0.004,
                last_total_cap_violation=5000.0,
                last_total_leakage=0.0,
            ),
        )
        obj.data_collections = SimpleNamespace(
            size_logits=torch.tensor([], dtype=torch.float32, requires_grad=True),
            vt_logits=None,
        )
        obj.timing_obj = lambda pos: (
            torch.tensor(-1.0, dtype=pos.dtype),
            torch.tensor(-10.0, dtype=pos.dtype),
            torch.tensor(0.0, dtype=pos.dtype),
            torch.tensor(0.0, dtype=pos.dtype),
        )
        return obj

    def test_constructor_keeps_placedb_for_endpoint_pin_name_mapping(self):
        placedb = SimpleNamespace(
            pin_names=np.array(
                [b"u_reg[17]:D", b"u_logic:Y"],
                dtype=np.bytes_,
            )
        )

        def fake_basic_place_init(self, _params, _placedb, _timer):
            nn.Module.__init__(self)

        with mock.patch(
            "dreamplace.BasicPlace.BasicPlace.__init__",
            new=fake_basic_place_init,
        ):
            placer = NonLinearPlace(SimpleNamespace(), placedb, None)

        self.assertIs(placer.placedb, placedb)
        self.assertEqual(
            placer._endpoint_pin_name_to_id_map(),
            {
                "u_reg[17]:D": 0,
                "u_logic:Y": 1,
            },
        )

    def _make_metric_record(
        self,
        iteration,
        wns,
        tns,
        timing_objective,
        slew_violation,
        cap_violation,
        leakage,
    ):
        return SimpleNamespace(
            iteration=iteration,
            detailed_step=None,
            objective=None,
            wirelength=None,
            density=None,
            density_weight=None,
            hpwl=None,
            overflow=None,
            max_density=None,
            gamma=None,
            wns=wns,
            tns=tns,
            timing_objective=timing_objective,
            slew_violation=slew_violation,
            cap_violation=cap_violation,
            leakage=leakage,
            eval_time=0.1,
            size_min=None,
            size_mean=None,
            size_max=None,
            vt_class_proportions=None,
        )

    def test_build_pin_violation_detail_rows_keeps_checked_pins_and_violation_formula(self):
        rows = build_pin_violation_detail_rows(
            pin_names=[b"inst0/A", b"inst0/Y", b"inst1/A"],
            pin_mask=torch.tensor([True, False, True], dtype=torch.bool),
            rise_values=torch.tensor([120.0, 400.0, 75.0], dtype=torch.float32),
            fall_values=torch.tensor([90.0, 500.0, 95.0], dtype=torch.float32),
            limits=torch.tensor([100.0, 300.0, 80.0], dtype=torch.float32),
            check_type="slew",
            unit="ps",
            output_pin_mask=torch.tensor([False, True, False], dtype=torch.bool),
        )

        self.assertEqual([row["pin_name"] for row in rows], ["inst0/A", "inst1/A"])
        self.assertEqual(rows[0]["check_type"], "slew")
        self.assertEqual(rows[0]["is_output_pin"], 0)
        self.assertAlmostEqual(rows[0]["limit"], 100.0)
        self.assertAlmostEqual(rows[0]["rise_value"], 120.0)
        self.assertAlmostEqual(rows[0]["fall_value"], 90.0)
        self.assertAlmostEqual(rows[0]["rise_violation"], 20.0)
        self.assertAlmostEqual(rows[0]["fall_violation"], 0.0)
        self.assertAlmostEqual(rows[0]["total_violation"], 20.0)
        self.assertEqual(rows[0]["unit"], "ps")
        self.assertAlmostEqual(rows[1]["rise_violation"], 0.0)
        self.assertAlmostEqual(rows[1]["fall_violation"], 15.0)
        self.assertAlmostEqual(rows[1]["total_violation"], 15.0)

    def test_update_metric_timing_refreshes_violation_observability_to_match_detail_aggregates(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        slew_rows = build_pin_violation_detail_rows(
            pin_names=[b"u0/A", b"u0/Y"],
            pin_mask=torch.tensor([True, True], dtype=torch.bool),
            rise_values=torch.tensor([330.0, 140.0], dtype=torch.float32),
            fall_values=torch.tensor([300.0, 160.0], dtype=torch.float32),
            limits=torch.tensor([320.0, 150.0], dtype=torch.float32),
            check_type="slew",
            unit="ps",
            output_pin_mask=torch.tensor([False, True], dtype=torch.bool),
        )
        cap_rows = build_pin_violation_detail_rows(
            pin_names=[b"u0/A", b"u0/Y"],
            pin_mask=torch.tensor([False, True], dtype=torch.bool),
            rise_values=torch.tensor([0.0, 0.060], dtype=torch.float32),
            fall_values=torch.tensor([0.0, 0.052], dtype=torch.float32),
            limits=torch.tensor([0.0, 0.050], dtype=torch.float32),
            check_type="cap",
            unit="pF",
            output_pin_mask=torch.tensor([False, True], dtype=torch.bool),
        )
        expected_slew_ns = sum(
            max(row["rise_violation"], row["fall_violation"]) for row in slew_rows
        ) / 1000.0
        expected_cap_ff = (
            sum(max(row["rise_violation"], row["fall_violation"]) for row in cap_rows)
            * 1000.0
        )

        placer.op_collections = SimpleNamespace(
            timing_propagation_op=SimpleNamespace(
                last_total_slew_violation=expected_slew_ns,
                last_total_cap_violation=expected_cap_ff,
                last_total_leakage=0.123,
            )
        )
        metric = SimpleNamespace(
            wns=None,
            tns=None,
            timing_objective=None,
            slew_violation=-1.0,
            cap_violation=-1.0,
            leakage=-1.0,
        )

        placer._record_timing_metrics(-0.45, -12.0, -0.33, None)
        placer._update_metric_timing(metric, -0.45, -12.0, -0.33)

        self.assertAlmostEqual(metric.wns, -0.45)
        self.assertAlmostEqual(metric.tns, -12.0)
        self.assertAlmostEqual(metric.timing_objective, -0.33)
        self.assertAlmostEqual(metric.slew_violation, expected_slew_ns)
        self.assertAlmostEqual(metric.cap_violation, expected_cap_ff)
        self.assertAlmostEqual(metric.leakage, 0.123)

    def test_validate_sizing_mode_rejects_missing_sizing_tensors(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.size_params = nn.ParameterList()
        placer.vt_params = nn.ParameterList()

        with self.assertRaisesRegex(ValueError, "requires size/vt parameters"):
            placer._validate_sizing_mode(SimpleNamespace(placement_sizing_mode="size_only"))

    def test_optimizer_parameter_groups_follow_mode(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.pos = nn.ParameterList([nn.Parameter(torch.tensor([0.0]))])
        placer.size_params = nn.ParameterList([nn.Parameter(torch.tensor([1.0]))])
        placer.vt_params = nn.ParameterList([nn.Parameter(torch.tensor([2.0, -2.0]))])

        place_groups = placer._optimizer_parameters(SimpleNamespace(placement_sizing_mode="place_only"))
        size_groups = placer._optimizer_parameters(SimpleNamespace(placement_sizing_mode="size_only"))
        joint_groups = placer._optimizer_parameters(SimpleNamespace(placement_sizing_mode="joint"))

        self.assertEqual([g["group_name"] for g in place_groups], ["placement"])
        self.assertEqual([g["group_name"] for g in size_groups], ["sizing"])
        self.assertEqual([g["group_name"] for g in joint_groups], ["placement", "sizing"])

    def test_default_mode_is_place_only_for_backward_compatibility(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        self.assertEqual(
            placer._placement_sizing_mode(SimpleNamespace()),
            "place_only",
        )

    def test_ieda_backend_allows_diff_sizing_metadata(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placedb = SimpleNamespace(
            backend_caps={
                "backend": "ieda",
                "has_diff_optimizer_metadata": False,
                "has_diff_sizing_metadata": True,
                "has_buffer_optimizer_metadata": False,
            }
        )

        placer._validate_backend_capabilities(
            SimpleNamespace(placement_sizing_mode="size_only"),
            placedb,
        )

    def test_ieda_backend_rejects_buffering_without_metadata(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placedb = SimpleNamespace(
            backend_caps={
                "backend": "ieda",
                "has_diff_sizing_metadata": True,
                "has_buffer_optimizer_metadata": False,
                "supports_buffer_commit": False,
                "supports_committed_refresh": False,
            }
        )

        with self.assertRaisesRegex(RuntimeError, "flow_kind=buffering requires"):
            placer._validate_backend_capabilities(
                SimpleNamespace(
                    placement_sizing_mode="place_only",
                    flow_kind="buffering",
                ),
                placedb,
            )

    def test_backend_capability_artifact_records_backend_contract(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        params = SimpleNamespace(
            result_dir=None,
            place_io_engine="ieda",
            placement_sizing_mode="place_only",
            flow_kind="buffering",
            buffering_mode="segment",
            design_name=lambda: "toy",
        )
        placedb = SimpleNamespace(
            backend_caps={
                "backend": "ieda",
                "has_diff_optimizer_metadata": False,
                "has_diff_sizing_metadata": True,
                "has_buffer_optimizer_metadata": False,
                "supports_sta_pydb": True,
                "sta_reference_role": "diagnostic_exporter",
                "parasitics_initialization": "unverified",
                "sta_state_status": "diagnostic_unverified",
                "supports_buffer_commit": False,
                "supports_committed_refresh": False,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            params.result_dir = tmpdir
            output_path = placer._write_backend_capability_artifact(params, placedb)
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual("toy", payload["design_name"])
        self.assertEqual("ieda", payload["place_io_engine"])
        self.assertEqual("place_only", payload["placement_sizing_mode"])
        self.assertEqual("buffering", payload["flow_kind"])
        self.assertEqual("segment", payload["buffering_mode"])
        self.assertEqual(placedb.backend_caps, payload["backend_caps"])
        self.assertTrue(
            payload["capability_interpretation"]["diff_sizing_metadata_available"]
        )
        self.assertFalse(
            payload["capability_interpretation"]["buffer_optimizer_metadata_available"]
        )
        self.assertTrue(payload["capability_interpretation"]["sta_pydb_available"])
        self.assertEqual(
            "diagnostic_exporter",
            payload["capability_interpretation"]["sta_reference_role"],
        )
        self.assertFalse(payload["capability_interpretation"]["sta_is_golden"])
        self.assertEqual(
            "unverified",
            payload["capability_interpretation"]["parasitics_initialization"],
        )
        self.assertEqual(
            "diagnostic_unverified",
            payload["capability_interpretation"]["sta_state_status"],
        )
        self.assertFalse(payload["capability_interpretation"]["buffer_commit_available"])
        self.assertFalse(
            payload["capability_interpretation"]["committed_refresh_available"]
        )

    def test_timing_backward_component_profile_is_disabled_by_default(self):
        model = PlaceObj.__new__(PlaceObj)
        model.params = SimpleNamespace(placement_sizing_mode="size_only")
        model.op_collections = SimpleNamespace()
        model.joint_proximal_objective = JointProximalObjective(model.params)
        model.data_collections = SimpleNamespace(
            size_logits=torch.tensor([1.0], requires_grad=True),
            vt_logits=None,
        )
        model.obj_fn = lambda pos: (model.data_collections.size_logits ** 2).sum()

        old_flag = os.environ.pop("AIMP_TIMING_BACKWARD_PROFILE", None)
        try:
            model.obj_and_grad_fn(torch.tensor([0.0], requires_grad=True))
        finally:
            if old_flag is not None:
                os.environ["AIMP_TIMING_BACKWARD_PROFILE"] = old_flag

        self.assertNotIn(
            "timing_backward_component_profile",
            model.debug_obj_and_grad_profile,
        )

    def test_timing_backward_component_profile_records_component_grad_times(self):
        model = PlaceObj.__new__(PlaceObj)
        model.params = SimpleNamespace(placement_sizing_mode="size_only")
        model.op_collections = SimpleNamespace()
        model.joint_proximal_objective = JointProximalObjective(model.params)
        size_logits = torch.tensor([1.0, 2.0], requires_grad=True)
        model.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=None,
        )

        def fake_obj_fn(_pos):
            timing = (size_logits ** 2).sum()
            slew = (size_logits * 3.0).sum()
            cap = (size_logits * -2.0).sum()
            model._last_timing_objective_component_tensors = {
                "timing": timing,
                "slew": slew,
                "cap": cap,
            }
            return timing + slew + cap

        model.obj_fn = fake_obj_fn
        old_flag = os.environ.get("AIMP_TIMING_BACKWARD_PROFILE")
        os.environ["AIMP_TIMING_BACKWARD_PROFILE"] = "1"
        try:
            model.obj_and_grad_fn(torch.tensor([0.0], requires_grad=True))
        finally:
            if old_flag is None:
                os.environ.pop("AIMP_TIMING_BACKWARD_PROFILE", None)
            else:
                os.environ["AIMP_TIMING_BACKWARD_PROFILE"] = old_flag

        profile = model.debug_obj_and_grad_profile["timing_backward_component_profile"]
        self.assertEqual(sorted(profile.keys()), ["cap", "slew", "timing"])
        for component in profile.values():
            self.assertGreaterEqual(component["grad_ms"], 0.0)
            self.assertGreater(component["grad_norm"], 0.0)

    def test_fast_loop_detaches_zero_slew_violation_objective_term(self):
        model = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(model)
        model.params = SimpleNamespace(
            production_fast_loop=True,
            timing_objective_lane="timing_slew_cap",
        )
        model.timing_wns_coeff = 1.0
        model.timing_tns_coeff = 0.0
        size_logits = torch.tensor([1.0], requires_grad=True)
        zero_slew_with_graph = (size_logits * 0.0).sum()
        timing_op = SimpleNamespace(
            last_total_slew_violation_tensor=zero_slew_with_graph,
            last_total_slew_violation=0.0,
            last_total_cap_violation_tensor=None,
            last_total_cap_violation=None,
            last_total_leakage_tensor=None,
            last_total_leakage=None,
        )
        model.op_collections = SimpleNamespace(timing_propagation_op=timing_op)

        loss = model._timing_objective_term_components(
            -torch.sigmoid(size_logits).sum(),
            torch.tensor(0.0),
        )

        self.assertTrue(loss.requires_grad)
        self.assertFalse(
            model._last_timing_objective_component_tensors["slew"].requires_grad
        )
        self.assertEqual(
            model.last_timing_objective_terms["detached_zero_terms"],
            ["slew"],
        )

    def test_surrogate_fast_loop_detaches_slew_inputs_unless_disabled_by_env(self):
        source = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)

        debug_model = TimingPropagation.__new__(TimingPropagation)
        debug_model.production_fast_loop = False
        debug_value = debug_model._surrogate_slew_input_tensor(source)
        self.assertIs(debug_value, source)
        self.assertTrue(debug_value.requires_grad)

        fast_model = TimingPropagation.__new__(TimingPropagation)
        fast_model.production_fast_loop = True
        with mock.patch.dict(os.environ, {}, clear=True):
            fast_detached = fast_model._surrogate_slew_input_tensor(source)
        self.assertIsNot(fast_detached, source)
        self.assertFalse(fast_detached.requires_grad)
        torch.testing.assert_close(fast_detached, source.detach())

        with mock.patch.dict(
            os.environ, {"AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS": "1"}
        ):
            fast_kept = fast_model._surrogate_slew_input_tensor(source)
        self.assertIs(fast_kept, source)
        self.assertTrue(fast_kept.requires_grad)

    def test_surrogate_fast_loop_detaches_cap_inputs_unless_disabled_by_env(self):
        source = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)

        debug_model = TimingPropagation.__new__(TimingPropagation)
        debug_model.production_fast_loop = False
        debug_value = debug_model._surrogate_cap_input_tensor(source)
        self.assertIs(debug_value, source)
        self.assertTrue(debug_value.requires_grad)

        fast_model = TimingPropagation.__new__(TimingPropagation)
        fast_model.production_fast_loop = True
        with mock.patch.dict(os.environ, {}, clear=True):
            fast_detached = fast_model._surrogate_cap_input_tensor(source)
        self.assertIsNot(fast_detached, source)
        self.assertFalse(fast_detached.requires_grad)
        torch.testing.assert_close(fast_detached, source.detach())

        with mock.patch.dict(
            os.environ, {"AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS": "1"}
        ):
            fast_kept = fast_model._surrogate_cap_input_tensor(source)
        self.assertIs(fast_kept, source)
        self.assertTrue(fast_kept.requires_grad)

    def test_fast_loop_can_limit_cap_limit_interpolation_to_output_pins_by_env(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        output_pin_mask = torch.tensor([False, True, False], dtype=torch.bool)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(obj._cap_limit_interpolation_pin_mask(None, output_pin_mask))

        with mock.patch.dict(os.environ, {"AIMP_LIMIT_CAP_LIMIT_TO_OUTPUT_PINS": "1"}):
            selected = obj._cap_limit_interpolation_pin_mask(None, output_pin_mask)

        self.assertIs(selected, output_pin_mask)

    def test_fast_loop_cap_limit_interpolation_keeps_active_cone_mask_precedence(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        aux_cap_mask = torch.tensor([True, False, False], dtype=torch.bool)
        output_pin_mask = torch.tensor([False, True, False], dtype=torch.bool)

        with mock.patch.dict(os.environ, {"AIMP_LIMIT_CAP_LIMIT_TO_OUTPUT_PINS": "1"}):
            selected = obj._cap_limit_interpolation_pin_mask(aux_cap_mask, output_pin_mask)

        self.assertIs(selected, aux_cap_mask)

    def test_reused_zero_slew_cap_limit_interpolation_uses_dense_default(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        output_pin_mask = torch.tensor([False, True, False], dtype=torch.bool)

        with mock.patch.dict(os.environ, {"AIMP_LIMIT_CAP_LIMIT_TO_OUTPUT_PINS": "1"}):
            selected = obj._zero_slew_reuse_cap_limit_interpolation_pin_mask(
                None,
                output_pin_mask,
            )

        self.assertIsNone(selected)

    def test_size_only_skips_position_randomization_and_noise(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        place_only = SimpleNamespace(
            global_place_flag=1,
            random_center_init_flag=1,
            gp_noise_ratio=0.025,
            placement_sizing_mode="place_only",
        )
        size_only = SimpleNamespace(
            global_place_flag=1,
            random_center_init_flag=1,
            gp_noise_ratio=0.025,
            placement_sizing_mode="size_only",
        )
        joint = SimpleNamespace(
            global_place_flag=1,
            random_center_init_flag=1,
            gp_noise_ratio=0.025,
            placement_sizing_mode="joint",
        )

        self.assertTrue(placer._overflow_blocks_placement_success(place_only))
        self.assertFalse(placer._should_randomize_init_pos(size_only))
        self.assertFalse(placer._should_apply_gp_noise(size_only))
        self.assertFalse(placer._overflow_blocks_placement_success(size_only))
        self.assertTrue(placer._should_randomize_init_pos(joint))
        self.assertTrue(placer._should_apply_gp_noise(joint))
        self.assertFalse(placer._overflow_blocks_placement_success(joint))

    def test_joint_uses_live_timing_loss_when_enabled(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        place_only = SimpleNamespace(
            placement_sizing_mode="place_only",
            with_sta=True,
            differentiable_timing_obj=1,
        )
        joint = SimpleNamespace(
            placement_sizing_mode="joint",
            with_sta=True,
            differentiable_timing_obj=1,
        )
        disabled = SimpleNamespace(
            placement_sizing_mode="joint",
            with_sta=True,
            differentiable_timing_obj=0,
        )

        self.assertFalse(placer._should_use_live_timing_loss(place_only))
        self.assertFalse(placer._should_use_live_timing_loss(disabled))

        # Joint direct-loss placement only consumes live timing while the
        # diff-timing-driven placement step reports the objective as active.
        placer._diff_tdp_step_status = {"timing_objective_active": False}
        self.assertFalse(placer._should_use_live_timing_loss(joint))

        placer._diff_tdp_step_status = {"timing_objective_active": True}
        self.assertTrue(placer._should_use_live_timing_loss(joint))

    def test_size_only_topology_refresh_skip_is_env_gated_after_initialization(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        size_only_fast = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=True,
        )
        joint_fast = SimpleNamespace(
            placement_sizing_mode="joint",
            production_fast_loop=True,
        )
        size_only_debug = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=False,
        )

        placer._live_timing_topology_initialized = True
        with mock.patch.dict(os.environ, {}, clear=True):
            # The skip is the fast-loop default and requires prior initialization.
            self.assertTrue(placer._should_skip_live_timing_topology_refresh(size_only_fast))

            placer._live_timing_topology_initialized = False
            self.assertFalse(placer._should_skip_live_timing_topology_refresh(size_only_fast))

            placer._live_timing_topology_initialized = True
            self.assertFalse(placer._should_skip_live_timing_topology_refresh(joint_fast))
            self.assertFalse(placer._should_skip_live_timing_topology_refresh(size_only_debug))

        with mock.patch.dict(
            os.environ, {"AIMP_DISABLE_SIZE_ONLY_TOPOLOGY_REFRESH_SKIP": "1"}
        ):
            placer._live_timing_topology_initialized = True
            self.assertFalse(placer._should_skip_live_timing_topology_refresh(size_only_fast))

    def test_discrete_gradient_topk_artifact_write_is_skipped_in_fast_loop_unless_disabled(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        params = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=True,
            continuous_size_dynamics_mode="discrete_gradient_topk",
        )
        debug_params = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=False,
            continuous_size_dynamics_mode="discrete_gradient_topk",
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            # The skip is the fast-loop default for discrete_gradient_topk.
            self.assertTrue(placer._should_skip_discrete_gradient_topk_artifact_write(params))
            self.assertFalse(placer._should_skip_discrete_gradient_topk_artifact_write(debug_params))

        with mock.patch.dict(
            os.environ, {"AIMP_DISABLE_DISCRETE_TOPK_ARTIFACT_SKIP": "1"}
        ):
            self.assertFalse(placer._should_skip_discrete_gradient_topk_artifact_write(params))

    def test_metrics_stage_name_preserves_place_trace(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        place = SimpleNamespace(global_place_flag=1, legalize_flag=1)
        legalization = SimpleNamespace(global_place_flag=0, legalize_flag=1)

        self.assertEqual(placer._metrics_stage_name(place), "place")
        self.assertEqual(placer._metrics_stage_name(legalization), "legalization")

    def test_sizing_driven_pin_caps_produce_gradients(self):
        obj = PlaceObj.__new__(PlaceObj)
        data = SimpleNamespace()
        data.pin2node_map = torch.tensor([0, 1], dtype=torch.int64)
        data.inst_main_id = torch.tensor([0, 0], dtype=torch.int64)
        data.inst_is_sizeable = torch.tensor([True, True], dtype=torch.bool)
        data.pin_2_libpin_offset = torch.tensor([0, 0], dtype=torch.int64)
        data.main_id_2_cell_id_start = torch.tensor([0, 2], dtype=torch.int64)
        data.flat_libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        data.cell_id_2_libpin_id_start = torch.tensor([0, 1], dtype=torch.int64)
        data.flat_lib_pin_cap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_rcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_fcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        size_logits = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32, requires_grad=True)
        data.size_logits = size_logits
        data.vt_logits = vt_logits
        data.inst_size_lower = torch.tensor([1.0, 1.0], dtype=torch.float32)
        data.inst_size_upper = torch.tensor([4.0, 4.0], dtype=torch.float32)
        data.inst_vt_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
        data.inst_vt_init = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

        def get_size_var():
            size_norm = torch.sigmoid(data.size_logits)
            return data.inst_size_lower + size_norm * (data.inst_size_upper - data.inst_size_lower)

        def get_vt_var():
            return torch.softmax(data.vt_logits, dim=1)

        data.get_size_var = get_size_var
        data.get_vt_var = get_vt_var

        base_cap = torch.zeros(2, dtype=torch.float32)
        base_rcap = torch.zeros(2, dtype=torch.float32)
        base_fcap = torch.zeros(2, dtype=torch.float32)
        pin_cap, pin_rcap, pin_fcap = obj._apply_sizing_driven_pin_caps(
            base_cap, base_rcap, base_fcap, data
        )
        loss = pin_cap.sum() + pin_rcap.sum() + pin_fcap.sum()
        loss.backward()

        self.assertGreater(pin_cap[0].item(), 1.0)
        self.assertLess(pin_cap[0].item(), 5.0)
        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(vt_logits.grad.abs().sum().item(), 0.0)

    def test_sizing_driven_pin_caps_reuse_static_cache_with_dynamic_size(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        data = SimpleNamespace()
        data.pin2node_map = torch.tensor([0, 1], dtype=torch.int64)
        data.inst_main_id = torch.tensor([0, 0], dtype=torch.int64)
        data.inst_is_sizeable = torch.tensor([True, True], dtype=torch.bool)
        data.pin_2_libpin_offset = torch.tensor([0, 0], dtype=torch.int64)
        data.main_id_2_cell_id_start = torch.tensor([0, 2], dtype=torch.int64)
        data.flat_libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        data.cell_id_2_libpin_id_start = torch.tensor([0, 1], dtype=torch.int64)
        data.flat_lib_pin_cap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_rcap = torch.tensor([2.0, 6.0], dtype=torch.float32)
        data.flat_lib_pin_fcap = torch.tensor([3.0, 7.0], dtype=torch.float32)
        data.inst_size_lower = torch.tensor([1.0, 1.0], dtype=torch.float32)
        data.inst_size_upper = torch.tensor([4.0, 4.0], dtype=torch.float32)
        data.inst_vt_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
        data.inst_vt_init = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        data.size_logits = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
        data.vt_logits = torch.tensor([[3.0, -3.0], [-3.0, 3.0]], dtype=torch.float32, requires_grad=True)

        def get_size_var():
            size_norm = torch.sigmoid(data.size_logits)
            return data.inst_size_lower + size_norm * (data.inst_size_upper - data.inst_size_lower)

        def get_vt_var():
            return torch.softmax(data.vt_logits, dim=1)

        data.get_size_var = get_size_var
        data.get_vt_var = get_vt_var

        first = obj._apply_sizing_driven_pin_caps(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            data,
        )
        cache = getattr(data, "_sizing_pin_cap_static_cache", None)
        self.assertIsNotNone(cache)

        with torch.no_grad():
            data.size_logits.add_(2.0)

        second = obj._apply_sizing_driven_pin_caps(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            data,
        )

        self.assertIs(getattr(data, "_sizing_pin_cap_static_cache", None), cache)
        self.assertFalse(torch.allclose(first[0], second[0]))
        loss = second[0].sum() + second[1].sum() + second[2].sum()
        loss.backward()
        self.assertGreater(data.size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(data.vt_logits.grad.abs().sum().item(), 0.0)

    def test_sizing_driven_pin_caps_reach_timing_delay_gradients(self):
        obj = PlaceObj.__new__(PlaceObj)
        data = SimpleNamespace()
        data.pin2node_map = torch.tensor([0], dtype=torch.int64)
        data.inst_main_id = torch.tensor([0], dtype=torch.int64)
        data.inst_is_sizeable = torch.tensor([True], dtype=torch.bool)
        data.pin_2_libpin_offset = torch.tensor([0], dtype=torch.int64)
        data.main_id_2_cell_id_start = torch.tensor([0, 2], dtype=torch.int64)
        data.flat_libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        )
        data.cell_id_2_libpin_id_start = torch.tensor([0, 1], dtype=torch.int64)
        data.flat_lib_pin_cap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_rcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_fcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        size_logits = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        data.size_logits = size_logits
        data.vt_logits = vt_logits
        data.inst_size_lower = torch.tensor([1.0], dtype=torch.float32)
        data.inst_size_upper = torch.tensor([4.0], dtype=torch.float32)
        data.inst_vt_mask = torch.tensor([[True]], dtype=torch.bool)
        data.inst_vt_init = torch.tensor([[1.0]], dtype=torch.float32)

        def get_size_var():
            size_norm = torch.sigmoid(data.size_logits)
            return data.inst_size_lower + size_norm * (
                data.inst_size_upper - data.inst_size_lower
            )

        def get_vt_var():
            return torch.softmax(data.vt_logits, dim=1)

        data.get_size_var = get_size_var
        data.get_vt_var = get_vt_var

        pin_cap, _, _ = obj._apply_sizing_driven_pin_caps(
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            data,
        )

        luts = LUTS_INFO(
            flat_luts_values=torch.tensor(
                [[1.0, 2.0, 3.0, 5.0]],
                dtype=torch.float32,
            ),
            flat_luts_trans_table=torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[1.0, 5.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[2, 2]], dtype=torch.long),
        )
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.arcs_info = ARCS_INFO(
            f_delay_luts=luts,
            r_delay_luts=luts,
            f_trans_luts=luts,
            r_trans_luts=luts,
        )
        tp.cell_modeling_op = None
        tp.pin2node_map = None
        tp.inst_main_id = None
        tp.inst_is_sizeable = None
        tp.size_var_getter = None
        tp.vt_var_getter = None

        delay = tp.r_delay_entry(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0.5], dtype=torch.float32),
            pin_cap,
            torch.tensor([0], dtype=torch.int64),
            use_surrogate=False,
        )
        delay.sum().backward()

        self.assertGreater(delay.item(), 0.0)
        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)

    def test_lut_only_skips_sizing_driven_pin_cap_weighting(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(timing_surrogate_mode="lut_only")
        data = SimpleNamespace()
        data.pin2node_map = torch.tensor([0], dtype=torch.int64)
        data.inst_main_id = torch.tensor([0], dtype=torch.int64)
        data.inst_is_sizeable = torch.tensor([True], dtype=torch.bool)
        data.pin_2_libpin_offset = torch.tensor([0], dtype=torch.int64)
        data.main_id_2_cell_id_start = torch.tensor([0, 2], dtype=torch.int64)
        data.flat_libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        )
        data.cell_id_2_libpin_id_start = torch.tensor([0, 1], dtype=torch.int64)
        data.flat_lib_pin_cap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_rcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.flat_lib_pin_fcap = torch.tensor([1.0, 5.0], dtype=torch.float32)
        data.size_logits = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        data.vt_logits = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        data.inst_size_lower = torch.tensor([1.0], dtype=torch.float32)
        data.inst_size_upper = torch.tensor([4.0], dtype=torch.float32)
        data.inst_vt_mask = torch.tensor([[True]], dtype=torch.bool)
        data.inst_vt_init = torch.tensor([[1.0]], dtype=torch.float32)

        def get_size_var():
            size_norm = torch.sigmoid(data.size_logits)
            return data.inst_size_lower + size_norm * (
                data.inst_size_upper - data.inst_size_lower
            )

        def get_vt_var():
            return torch.softmax(data.vt_logits, dim=1)

        data.get_size_var = get_size_var
        data.get_vt_var = get_vt_var

        pin_cap, pin_rcap, pin_fcap = obj._apply_sizing_driven_pin_caps(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.5], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            data,
        )

        self.assertAlmostEqual(pin_cap.item(), 1.0)
        self.assertAlmostEqual(pin_rcap.item(), 1.5)
        self.assertAlmostEqual(pin_fcap.item(), 2.0)

    def test_1d_lut_entry_remains_valid_in_timing_propagation(self):
        luts = LUTS_INFO(
            flat_luts_values=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[2, 0]], dtype=torch.long),
        )
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.arcs_info = ARCS_INFO(
            f_delay_luts=luts,
            r_delay_luts=luts,
            f_trans_luts=luts,
            r_trans_luts=luts,
        )
        tp.cell_modeling_op = None
        tp.pin2node_map = None
        tp.inst_main_id = None
        tp.inst_is_sizeable = None
        tp.size_var_getter = None
        tp.vt_var_getter = None

        delay = tp.r_delay_entry(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0], dtype=torch.int64),
            use_surrogate=False,
        )

        self.assertGreater(delay.item(), 10.0)
        self.assertLess(delay.item(), 20.0)

    def test_endpoint_timing_compare_scales_cpp_ns_to_ps(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.placedb = SimpleNamespace(
            pin_names=[b"ep0"],
            backend_caps={
                "backend": "openroad",
                "sta_reference_role": "golden_opensta",
                "parasitics_initialization": "placement",
                "sta_state_status": "validated_golden",
            },
        )
        obj.data_collections = SimpleNamespace(end_points=torch.tensor([0], dtype=torch.int64))
        obj.op_collections = SimpleNamespace(
            timing_propagation_op=SimpleNamespace(
                pin_rAAT=torch.tensor([1500.0], dtype=torch.float32),
                pin_fAAT=torch.tensor([1600.0], dtype=torch.float32),
                pin_rRAT=torch.tensor([4000.0], dtype=torch.float32),
                pin_fRAT=torch.tensor([4200.0], dtype=torch.float32),
            )
        )

        rows, summary = obj._build_endpoint_timing_compare(
            at_late_cpp=[[1.0, 1.2]],
            rt_late_cpp=[[3.5, 3.7]],
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["cpp_r_aat"], 1000.0)
        self.assertAlmostEqual(rows[0]["cpp_f_aat"], 1200.0)
        self.assertAlmostEqual(rows[0]["cpp_r_rat"], 3500.0)
        self.assertAlmostEqual(rows[0]["cpp_f_rat"], 3700.0)
        self.assertAlmostEqual(rows[0]["cpp_slack"], 2500.0)
        self.assertAlmostEqual(summary["ieda_wns"], 2500.0)
        self.assertEqual("openroad", summary["backend_metadata"]["backend"])
        self.assertEqual("golden_opensta", summary["backend_metadata"]["sta_reference_role"])
        self.assertTrue(summary["backend_metadata"]["sta_is_golden"])
        self.assertEqual("placement", summary["backend_metadata"]["parasitics_initialization"])
        self.assertEqual("validated_golden", summary["backend_metadata"]["sta_state_status"])

    def test_write_arc_all_uses_level_arc_indices(self):
        obj = PlaceObj.__new__(PlaceObj)
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "toy",
            )
            obj.placedb = SimpleNamespace(
                node_name2id_map={"inst0": 0},
                pin_names=[b"inst0:A", b"inst0:Y"],
            )
            obj.data_collections = SimpleNamespace(
                inst_flat_arcs=torch.tensor([1], dtype=torch.int64),
                inst_flat_arcs_start=torch.tensor([0, 1], dtype=torch.int64),
                flat_inst_arcs_by_level=torch.tensor(
                    [
                        [0, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 0],
                    ],
                    dtype=torch.int64,
                ),
            )
            obj.op_collections = SimpleNamespace(
                timing_propagation_op=SimpleNamespace(
                    cell_arc_rr_delays=torch.tensor([-float("inf"), 123.0], dtype=torch.float32),
                    cell_arc_ff_delays=torch.tensor([-float("inf"), 234.0], dtype=torch.float32),
                    cell_arc_rf_delays=torch.tensor([-float("inf"), 345.0], dtype=torch.float32),
                    cell_arc_fr_delays=torch.tensor([-float("inf"), 456.0], dtype=torch.float32),
                    pin_rtran=torch.tensor([11.0, 22.0], dtype=torch.float32),
                    pin_ftran=torch.tensor([33.0, 44.0], dtype=torch.float32),
                ),
                elmore_delay_op=SimpleNamespace(
                    loads={
                        "rise": torch.tensor([1.0, 2.0], dtype=torch.float32),
                        "fall": torch.tensor([3.0, 4.0], dtype=torch.float32),
                    }
                ),
            )

            obj.write_arc_all(
                [
                    {
                        "inst_name": "inst0",
                        "from_pin": "inst0:A",
                        "to_pin": "inst0:Y",
                        "transition": "Rise",
                        "arc_sense": 1,
                        "delay_ns": 0.123,
                        "in_slew_ns": 0.011,
                        "load_cap": 2.0,
                    }
                ]
            )

            csv_path = os.path.join(tmpdir, "toy_cell_arc_delay_report.csv")
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["Py Delay (ps)"]), 123.0)

    def test_density_area_proxy_produces_size_gradients(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            macro_overlap_flag=False,
            enable_net_weighting=False,
        )
        obj.placedb = SimpleNamespace(num_movable_nodes=2, regions=[])
        obj.op_collections = SimpleNamespace(
            wirelength_op=lambda pos: torch.sum(pos * 0.0),
            density_op=lambda pos: pos.pow(2).sum() + torch.tensor(1.0, dtype=pos.dtype),
        )
        obj.data_collections = SimpleNamespace()
        obj.data_collections.node_areas = torch.tensor([2.0, 1.0], dtype=torch.float32)
        obj.data_collections.inst_size_init = torch.tensor([1.0, 2.0], dtype=torch.float32)
        obj.data_collections.inst_is_sizeable = torch.tensor([True, True], dtype=torch.bool)
        size_logits = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
        obj.data_collections.size_logits = size_logits
        obj.data_collections.inst_size_lower = torch.tensor([1.0, 2.0], dtype=torch.float32)
        obj.data_collections.inst_size_upper = torch.tensor([3.0, 4.0], dtype=torch.float32)
        obj.data_collections.get_size_var = lambda: (
            obj.data_collections.inst_size_lower
            + torch.sigmoid(obj.data_collections.size_logits)
            * (obj.data_collections.inst_size_upper - obj.data_collections.inst_size_lower)
        )
        obj.density_weight = torch.tensor([1.0], dtype=torch.float32)
        obj.density_factor = 1.0
        obj.quad_penalty = False
        obj.density_quad_coeff = 0.0
        obj.init_density = None
        obj.use_timing_obj = False
        obj.size_density_area_weight = 0.5

        pos = torch.tensor([0.25, -0.5], dtype=torch.float32, requires_grad=True)
        loss = obj.obj_fn(pos)
        loss.backward()

        self.assertIsNotNone(obj.size_density_area_penalty)
        self.assertGreater(obj.size_density_area_penalty.item(), 0.0)
        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)

    def test_size_only_obj_fn_excludes_wirelength_and_main_density_terms(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            macro_overlap_flag=False,
            enable_net_weighting=False,
        )
        obj.timing_wns_coeff = 0.01
        obj.timing_tns_coeff = 0.0001
        obj.timing_ws_coeff = 0.01
        obj.use_timing_obj = True
        obj.placedb = SimpleNamespace(num_movable_nodes=2, regions=[])
        obj.density_factor = 1.0
        obj.density_weight = torch.tensor([3.0], dtype=torch.float32)
        obj.init_density = None
        obj.quad_penalty = False
        obj.density_quad_coeff = 0.0
        obj.size_density_area_weight = 0.5

        size_logits = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
        obj.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=None,
            node_areas=torch.tensor([1.0, 1.0], dtype=torch.float32),
            inst_size_init=torch.tensor([1.0, 1.0], dtype=torch.float32),
            inst_is_sizeable=torch.tensor([True, True], dtype=torch.bool),
            inst_size_lower=torch.tensor([1.0, 1.0], dtype=torch.float32),
            inst_size_upper=torch.tensor([2.0, 2.0], dtype=torch.float32),
        )
        obj.data_collections.get_size_var = lambda: (
            obj.data_collections.inst_size_lower
            + torch.sigmoid(obj.data_collections.size_logits)
            * (obj.data_collections.inst_size_upper - obj.data_collections.inst_size_lower)
        )
        obj.op_collections = SimpleNamespace(
            wirelength_op=lambda pos: torch.tensor(1000.0, dtype=pos.dtype) + pos.square().sum(),
            density_op=lambda pos: torch.tensor(2.0, dtype=pos.dtype) + pos.sum() * 0.0,
        )

        def timing_obj(pos):
            size_term = torch.sigmoid(obj.data_collections.size_logits).sum()
            zero = pos.sum() * 0.0
            return size_term, zero, zero, zero

        obj.timing_obj = timing_obj

        pos = torch.tensor([3.0, -4.0], dtype=torch.float32, requires_grad=True)
        loss = obj.obj_fn(pos)

        expected_timing_loss = -0.01
        expected_area_penalty = 0.25
        self.assertAlmostEqual(loss.item(), expected_timing_loss + expected_area_penalty, places=6)

    def test_size_only_obj_and_grad_fn_blocks_pos_gradients_but_keeps_sizing_gradients(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            macro_overlap_flag=False,
            enable_net_weighting=False,
        )
        obj.timing_wns_coeff = 0.01
        obj.timing_tns_coeff = 0.0001
        obj.timing_ws_coeff = 0.01
        obj.use_timing_obj = True
        obj.placedb = SimpleNamespace(num_movable_nodes=1, regions=[])
        obj.density_factor = 1.0
        obj.density_weight = torch.tensor([2.0], dtype=torch.float32)
        obj.init_density = None
        obj.quad_penalty = False
        obj.density_quad_coeff = 0.0
        obj.size_density_area_weight = 0.0
        obj.update_mask = None
        obj.fix_nodes_mask = None

        size_logits = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[0.0, 0.5]], dtype=torch.float32, requires_grad=True)
        obj.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
        )
        obj.op_collections = SimpleNamespace(
            wirelength_op=lambda pos: torch.tensor(500.0, dtype=pos.dtype) + (pos.square()).sum(),
            density_op=lambda pos: torch.tensor(7.0, dtype=pos.dtype) + pos.sum() * 0.0,
            precondition_op=lambda *args, **kwargs: None,
        )
        obj.joint_proximal_objective = JointProximalObjective(obj.params)

        def timing_obj(pos):
            size_term = torch.sigmoid(obj.data_collections.size_logits).sum()
            vt_term = torch.softmax(obj.data_collections.vt_logits, dim=1)[:, 1].sum()
            shared = pos.sum() + size_term + vt_term
            zero = pos.sum() * 0.0
            return shared, zero, shared, zero

        obj.timing_obj = timing_obj

        pos = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        _, grad = obj.obj_and_grad_fn(pos)

        self.assertEqual(grad.shape, pos.shape)
        self.assertAlmostEqual(grad.abs().sum().item(), 0.0, places=8)
        if pos.grad is not None:
            self.assertAlmostEqual(pos.grad.abs().sum().item(), 0.0, places=8)
        self.assertGreater(obj.data_collections.size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(obj.data_collections.vt_logits.grad.abs().sum().item(), 0.0)

    def test_size_only_fast_loop_uses_targeted_sizing_backward(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=True,
            targeted_sizing_backward=True,
        )
        obj.op_collections = SimpleNamespace()
        obj.joint_proximal_objective = JointProximalObjective(obj.params)

        size_logits = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[0.0, 0.5]], dtype=torch.float32, requires_grad=True)
        obj.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
        )

        def fake_obj_fn(pos):
            size_term = torch.sigmoid(size_logits).sum()
            vt_term = torch.softmax(vt_logits, dim=1)[:, 1].sum()
            return pos.sum() + size_term + vt_term

        obj.obj_fn = fake_obj_fn

        pos = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        _, grad = obj.obj_and_grad_fn(pos)

        self.assertEqual(grad.shape, pos.shape)
        self.assertAlmostEqual(grad.abs().sum().item(), 0.0, places=8)
        self.assertIsNone(pos.grad)
        torch.testing.assert_close(
            size_logits.grad,
            torch.autograd.grad(torch.sigmoid(size_logits).sum(), size_logits)[0],
        )
        torch.testing.assert_close(
            vt_logits.grad,
            torch.autograd.grad(torch.softmax(vt_logits, dim=1)[:, 1].sum(), vt_logits)[0],
        )
        self.assertEqual(
            obj.debug_obj_and_grad_profile.get("objective_backward_mode"),
            "targeted_sizing_autograd_grad",
        )

    def test_size_only_fast_loop_defaults_to_full_backward(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="size_only",
            production_fast_loop=True,
        )
        obj.op_collections = SimpleNamespace()
        obj.joint_proximal_objective = JointProximalObjective(obj.params)
        size_logits = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
        obj.data_collections = SimpleNamespace(size_logits=size_logits, vt_logits=None)
        obj.obj_fn = lambda pos: pos.sum() + torch.sigmoid(size_logits).sum()

        old_flag = os.environ.pop("AIMP_TARGETED_SIZING_BACKWARD", None)
        try:
            pos = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            obj.obj_and_grad_fn(pos)
        finally:
            if old_flag is not None:
                os.environ["AIMP_TARGETED_SIZING_BACKWARD"] = old_flag

        self.assertEqual(
            obj.debug_obj_and_grad_profile.get("objective_backward_mode"),
            "full_backward",
        )
        self.assertAlmostEqual(pos.grad.abs().sum().item(), 0.0, places=8)
        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)

    def test_timing_obj_profile_exposes_pruning_stats_for_runtime_diagnosis(self):
        obj = PlaceObj.__new__(PlaceObj)
        timing_op = SimpleNamespace(
            last_critical_endpoint_pruning_stats={
                "mode": "dynamic",
                "applied": True,
                "active_endpoint_count": 64,
                "selected_endpoint_tns_ps": -1234.0,
            },
            last_traversal_pruning_stats={
                "enabled": True,
                "applied": True,
                "active_endpoint_count": 64,
                "active_inst_count": 512,
                "active_arc_count": 2048,
                "dropped_arc_count": 4096,
                "parallel_task_count": 7,
                "preparation_runtime_ms": 1.25,
                "per_level_kept_counts": [3, 5, 8],
            },
        )

        fields = obj._timing_pruning_profile_fields(timing_op)

        self.assertEqual(fields["active_endpoint_count"], 64)
        self.assertEqual(fields["active_inst_count"], 512)
        self.assertEqual(fields["active_arc_count"], 2048)
        self.assertEqual(fields["dropped_arc_count"], 4096)
        self.assertEqual(
            fields["critical_endpoint_pruning"]["selected_endpoint_tns_ps"],
            -1234.0,
        )
        self.assertEqual(
            fields["traversal_pruning"]["per_level_kept_counts"],
            [3, 5, 8],
        )

    def test_timing_obj_profile_embeds_timing_propagation_breakdown(self):
        obj = PlaceObj.__new__(PlaceObj)
        timing_op = SimpleNamespace(
            last_critical_endpoint_pruning_stats={},
            last_traversal_pruning_stats={},
            last_profile_payload={
                "stage_runtime_ms": {
                    "cell_aat_levels": 123.0,
                    "cell_rat_levels": 45.0,
                },
                "cell_aat_levels_detail": {
                    "total_ms": 123.0,
                    "surrogate_forward_ms": 30.0,
                    "net_aat_ms": 20.0,
                    "top_levels": [{"level": 7, "query_eval_ms": 9.0}],
                },
            },
        )

        fields = obj._timing_pruning_profile_fields(timing_op)

        self.assertEqual(
            fields["timing_propagation_stage_runtime_ms"]["cell_aat_levels"],
            123.0,
        )
        self.assertEqual(
            fields["timing_propagation_cell_aat_detail"]["surrogate_forward_ms"],
            30.0,
        )
        self.assertEqual(
            fields["timing_propagation_cell_aat_detail"]["top_levels"],
            [{"level": 7, "query_eval_ms": 9.0}],
        )

    def test_active_cone_aux_pin_masks_use_traversal_kept_arcs(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.inst_pins_mask = torch.tensor([True, True, True, True, True, True])
        obj.output_pin_mask = torch.tensor([False, True, False, True, False, True])
        timing_op = SimpleNamespace(
            last_traversal_pruning_stats={"applied": True},
            last_traversal_pruning_result=SimpleNamespace(
                kept_flat_arc_indices=torch.tensor([1, 3], dtype=torch.long)
            ),
            flat_inst_arcs_by_level=torch.tensor(
                [
                    [0, 1, 0, 0, 1, 0],
                    [2, 3, 0, 0, 1, 0],
                    [1, 5, 0, 0, 1, 0],
                    [4, 5, 0, 0, 1, 0],
                ],
                dtype=torch.long,
            ),
        )

        slew_mask, cap_mask, metadata = obj._active_cone_aux_pin_masks(timing_op)

        torch.testing.assert_close(
            slew_mask,
            torch.tensor([False, False, True, True, True, True]),
        )
        torch.testing.assert_close(
            cap_mask,
            torch.tensor([False, False, False, True, False, True]),
        )
        self.assertEqual(metadata["active_aux_pin_count"], 4)
        self.assertEqual(metadata["active_aux_output_pin_count"], 2)

    def test_fast_loop_reuse_zero_slew_aux_skips_slew_limit_query(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        timing_op = SimpleNamespace(
            last_total_slew_violation=0.0,
            last_total_slew_violation_tensor=torch.tensor(0.0, requires_grad=True),
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            reuse = obj._should_reuse_zero_slew_aux(timing_op)

        self.assertTrue(reuse)
        term = obj._reused_zero_slew_aux_tensor(timing_op)
        self.assertIsNotNone(term)
        self.assertFalse(term.requires_grad)
        self.assertEqual(float(term.item()), 0.0)

    def test_fast_loop_reuse_zero_slew_aux_is_default_unless_disabled_by_env(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(production_fast_loop=True)
        timing_op = SimpleNamespace(last_total_slew_violation=0.0)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(obj._should_reuse_zero_slew_aux(timing_op))

        with mock.patch.dict(os.environ, {"AIMP_DISABLE_REUSE_ZERO_SLEW_AUX": "1"}):
            self.assertFalse(obj._should_reuse_zero_slew_aux(timing_op))

    def test_fast_loop_pin_detail_does_not_force_pin_slack_copy(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(production_fast_loop=True)

        self.assertFalse(obj._should_copy_pin_slack_after_timing(False))

    def test_pin_violation_detail_export_uses_live_fast_loop_tensors(self):
        obj = PlaceObj.__new__(PlaceObj)
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.params = SimpleNamespace(
                export_pin_violation_csv=True,
                result_dir=tmpdir,
                design_name=lambda: "toy",
            )
            obj.invoke_timing_count = 7
            obj.placedb = SimpleNamespace(pin_names=["u0/A", "u0/Y"])
            obj.inst_pins_mask = torch.tensor([True, True], dtype=torch.bool)
            obj.output_pin_mask = torch.tensor([False, True], dtype=torch.bool)
            timing_op = SimpleNamespace(
                pin_rtran=None,
                pin_ftran=None,
                pin_net_cap_rise=None,
                pin_net_cap_fall=None,
                pin_rtran_live=torch.tensor([350.0, 120.0], dtype=torch.float32),
                pin_ftran_live=torch.tensor([310.0, 140.0], dtype=torch.float32),
                pin_net_cap_rise_live=torch.tensor([0.0, 0.060], dtype=torch.float32),
                pin_net_cap_fall_live=torch.tensor([0.0, 0.052], dtype=torch.float32),
            )

            obj._log_violation_details(
                timing_op,
                slew_limits=torch.tensor([320.0, 320.0], dtype=torch.float32),
                cap_limits=torch.tensor([0.0, 0.050], dtype=torch.float32),
            )

            slew_path = os.path.join(tmpdir, "toy_slew_pin_detail.csv")
            cap_path = os.path.join(tmpdir, "toy_cap_pin_detail.csv")
            iter_slew_path = os.path.join(tmpdir, "toy_slew_pin_detail_iter0007.csv")
            iter_cap_path = os.path.join(tmpdir, "toy_cap_pin_detail_iter0007.csv")
            self.assertTrue(os.path.isfile(slew_path))
            self.assertTrue(os.path.isfile(cap_path))
            self.assertTrue(os.path.isfile(iter_slew_path))
            self.assertTrue(os.path.isfile(iter_cap_path))
            with open(slew_path, newline="", encoding="utf-8") as csv_file:
                slew_rows = list(csv.DictReader(csv_file))
            with open(cap_path, newline="", encoding="utf-8") as csv_file:
                cap_rows = list(csv.DictReader(csv_file))

        self.assertEqual([row["pin_name"] for row in slew_rows], ["u0/A", "u0/Y"])
        self.assertAlmostEqual(float(slew_rows[0]["rise_violation"]), 30.0)
        self.assertEqual([row["pin_name"] for row in cap_rows], ["u0/Y"])
        self.assertAlmostEqual(float(cap_rows[0]["rise_violation"]), 0.010, places=6)

    def test_first_level_numeric_alignment_export_records_live_topology_and_limits(self):
        obj = PlaceObj.__new__(PlaceObj)
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "toy",
            )
            obj.invoke_timing_count = 3
            obj.placedb = SimpleNamespace(
                pin_names=np.array(
                    [b"drv0:Q", b"sink0:A", b"g44249__69005:B2", b"g65266:Y"],
                    dtype=np.bytes_,
                ),
                pin_name2id_map={
                    "drv0:Q": 0,
                    "sink0:A": 1,
                    "g44249__69005:B2": 2,
                    "g65266:Y": 3,
                },
                net_names=np.array([b"net0", b"n_24291"], dtype=np.bytes_),
                pin2net_map=np.array([0, 0, 1, 1], dtype=np.int32),
                net2driver_pin_map=np.array([0, 3], dtype=np.int32),
                flat_net2pin_start_map=np.array([0, 2, 4], dtype=np.int32),
                flat_net2pin_map=np.array([0, 1, 3, 2], dtype=np.int32),
            )
            obj.data_collections = SimpleNamespace(
                start_points=torch.tensor([0], dtype=torch.int64),
            )
            obj.pin2libpin_flat_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
            obj.inst_pins_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
            obj.output_pin_mask = torch.tensor([True, False, False, True], dtype=torch.bool)
            obj.op_collections = SimpleNamespace(
                timing_propagation_op=SimpleNamespace(
                    pin_rAAT=torch.tensor([0.0, 20.0, 31.0, 25.0], dtype=torch.float32),
                    pin_fAAT=torch.tensor([0.0, 22.0, 33.0, 27.0], dtype=torch.float32),
                    pin_rRAT=torch.tensor([100.0, 90.0, 80.0, 85.0], dtype=torch.float32),
                    pin_fRAT=torch.tensor([100.0, 88.0, 78.0, 83.0], dtype=torch.float32),
                    pin_rtran=torch.tensor([10.0, 12.0, 40.0, 35.0], dtype=torch.float32),
                    pin_ftran=torch.tensor([11.0, 13.0, 44.0, 36.0], dtype=torch.float32),
                    pin_rtran_live=torch.tensor([14.0, 16.0, 50.0, 45.0], dtype=torch.float32),
                    pin_ftran_live=torch.tensor([15.0, 17.0, 54.0, 46.0], dtype=torch.float32),
                    pin_net_cap_rise=torch.tensor([0.010, 0.020, 0.030, 0.040], dtype=torch.float32),
                    pin_net_cap_fall=torch.tensor([0.011, 0.021, 0.031, 0.041], dtype=torch.float32),
                    pin_net_cap_rise_live=torch.tensor([0.012, 0.022, 0.032, 0.042], dtype=torch.float32),
                    pin_net_cap_fall_live=torch.tensor([0.013, 0.023, 0.033, 0.043], dtype=torch.float32),
                )
            )
            obj.data_collections.flat_lib_pin_slew_limit = torch.tensor(
                [80.0, 81.0, 82.0, 83.0],
                dtype=torch.float32,
            )
            obj.data_collections.flat_lib_pin_cap_limit = torch.tensor(
                [0.090, 0.091, 0.092, 0.093],
                dtype=torch.float32,
            )

            path = obj.write_first_level_numeric_alignment()

            self.assertEqual(path, os.path.join(tmpdir, "toy_first_level_numeric_alignment.csv"))
            with open(path, newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            [row["pin_name"] for row in rows],
            ["drv0:Q", "sink0:A", "g44249__69005:B2", "g65266:Y"],
        )
        sink_row = rows[1]
        self.assertEqual(sink_row["row_role"], "first_fanout_sink")
        self.assertEqual(sink_row["driver_pin_name"], "drv0:Q")
        self.assertEqual(sink_row["net_name"], "net0")
        self.assertEqual(sink_row["is_output_pin"], "0")
        self.assertAlmostEqual(float(sink_row["py_r_slew_ps"]), 16.0, places=6)
        self.assertAlmostEqual(float(sink_row["py_f_slew_ps"]), 17.0, places=6)
        self.assertAlmostEqual(float(sink_row["py_r_cap_pf"]), 0.022, places=6)
        self.assertAlmostEqual(float(sink_row["py_f_cap_pf"]), 0.023, places=6)
        self.assertAlmostEqual(float(sink_row["slew_limit_ps"]), 81.0, places=6)
        self.assertAlmostEqual(float(sink_row["cap_limit_pf"]), 0.091, places=6)
        probe_row = rows[2]
        self.assertEqual(probe_row["row_role"], "top_mismatch_anchor")
        self.assertEqual(probe_row["driver_pin_name"], "g65266:Y")
        self.assertEqual(probe_row["reachable_static_first_fanout"], "false")
        self.assertEqual(rows[3]["row_role"], "top_mismatch_anchor")

    def test_fast_loop_can_detach_timing_rc_cap_inputs_by_env(self):
        source = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)

        debug_obj = PlaceObj.__new__(PlaceObj)
        debug_obj.params = SimpleNamespace(production_fast_loop=False)
        with mock.patch.dict(os.environ, {"AIMP_DETACH_TIMING_RC_CAP_INPUTS": "1"}):
            debug_value = debug_obj._timing_rc_cap_input_tensor(source)
        self.assertIs(debug_value, source)
        self.assertTrue(debug_value.requires_grad)

        fast_obj = PlaceObj.__new__(PlaceObj)
        fast_obj.params = SimpleNamespace(production_fast_loop=True)
        fast_default = fast_obj._timing_rc_cap_input_tensor(source)
        self.assertIs(fast_default, source)
        self.assertTrue(fast_default.requires_grad)

        fast_obj.params.timing_objective_lane = "timing_only"
        with mock.patch.dict(os.environ, {"AIMP_DETACH_TIMING_RC_CAP_INPUTS": "1"}):
            fast_detached = fast_obj._timing_rc_cap_input_tensor(source)
        self.assertIsNot(fast_detached, source)
        self.assertFalse(fast_detached.requires_grad)
        torch.testing.assert_close(fast_detached, source.detach())

    def test_fast_loop_keeps_timing_rc_cap_grad_when_cap_objective_is_enabled(self):
        source = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(
            production_fast_loop=True,
            timing_objective_lane="timing_slew_cap",
        )

        with mock.patch.dict(os.environ, {"AIMP_DETACH_TIMING_RC_CAP_INPUTS": "1"}):
            value = obj._timing_rc_cap_input_tensor(source)

        self.assertIs(value, source)
        self.assertTrue(value.requires_grad)

    def test_size_only_timing_loss_ignores_positive_ws_when_wns_tns_are_zero(self):
        obj = PlaceObj.__new__(PlaceObj)
        obj.params = SimpleNamespace(placement_sizing_mode="size_only")
        obj.timing_wns_coeff = 0.01
        obj.timing_tns_coeff = 0.0001
        obj.timing_ws_coeff = 0.01

        wns = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        tns = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        ws = torch.tensor(10.0, dtype=torch.float32, requires_grad=True)

        loss = obj._timing_loss(wns, tns, ws, ts=0.0)
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertIsNone(ws.grad)
        self.assertAlmostEqual(wns.grad.item(), -obj.timing_wns_coeff, places=6)
        self.assertAlmostEqual(tns.grad.item(), -obj.timing_tns_coeff, places=6)

    def test_endpoint_timing_compare_summarizes_python_vs_ieda_mismatch(self):
        obj = PlaceObj.__new__(PlaceObj)
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
            )
            obj.placedb = SimpleNamespace(
                pin_names=np.array(["p0", "p1", "p2"], dtype=object),
                backend_caps={
                    "backend": "ieda",
                    "sta_reference_role": "diagnostic_exporter",
                    "parasitics_initialization": "unverified",
                    "sta_state_status": "diagnostic_unverified",
                },
            )
            obj.data_collections = SimpleNamespace(
                end_points=torch.tensor([1, 2], dtype=torch.int64)
            )
            obj.op_collections = SimpleNamespace(
                timing_propagation_op=SimpleNamespace(
                    pin_rAAT=torch.tensor([0.0, -1.0e8, 50.0], dtype=torch.float32),
                    pin_fAAT=torch.tensor([0.0, -1.0e8, 60.0], dtype=torch.float32),
                    pin_rRAT=torch.tensor([100.0, 10000.0, 40.0], dtype=torch.float32),
                    pin_fRAT=torch.tensor([100.0, 10000.0, 45.0], dtype=torch.float32),
                )
            )
            at_late_cpp = [
                [0.0, 0.0],
                [10.0, 12.0],
                [55.0, 58.0],
            ]
            rt_late_cpp = [
                [100.0, 100.0],
                [5.0, 4.0],
                [40.0, 43.0],
            ]

            rows, summary = obj.write_endpoint_timing_compare(at_late_cpp, rt_late_cpp)

            self.assertEqual(summary["num_endpoints"], 2)
            self.assertEqual(summary["num_python_negative_slack"], 1)
            self.assertEqual(summary["num_backend_negative_slack"], 2)
            self.assertEqual(summary["num_ieda_negative_slack"], 2)
            self.assertEqual(summary["num_python_arrival_sentinel"], 1)
            self.assertEqual(summary["num_python_required_sentinel"], 0)
            self.assertEqual(rows[0]["pin_name"], "p1")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv")))

            with open(os.path.join(tmpdir, "FFT_endpoint_timing_compare_summary.json"), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["num_backend_negative_slack"], 2)
            self.assertEqual(saved["num_ieda_negative_slack"], 2)
            self.assertEqual("ieda", saved["backend_metadata"]["backend"])
            self.assertEqual("diagnostic_exporter", saved["backend_metadata"]["sta_reference_role"])
            self.assertFalse(saved["backend_metadata"]["sta_is_golden"])
            self.assertEqual("unverified", saved["backend_metadata"]["parasitics_initialization"])
            self.assertEqual("diagnostic_unverified", saved["backend_metadata"]["sta_state_status"])

    def test_joint_placeobj_path_produces_pos_size_and_vt_gradients(self):
        obj = PlaceObj.__new__(PlaceObj)
        nn.Module.__init__(obj)
        obj.params = SimpleNamespace(
            placement_sizing_mode="joint",
            macro_overlap_flag=False,
            enable_net_weighting=False,
        )
        obj.timing_wns_coeff = 0.01
        obj.timing_tns_coeff = 0.0001
        obj.timing_ws_coeff = 0.01
        obj.use_timing_obj = True
        obj.placedb = SimpleNamespace(regions=[])
        obj.density_factor = 1.0
        obj.density_weight = torch.tensor([0.0], dtype=torch.float32)
        obj.init_density = None
        obj.quad_penalty = False
        obj.density_quad_coeff = 0.0
        obj.update_mask = None
        obj.fix_nodes_mask = None
        obj.use_l_shape_routability = False
        obj.l_shape_routability_op = None

        size_logits = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[0.0, 0.5]], dtype=torch.float32, requires_grad=True)
        obj.data_collections = SimpleNamespace(
            size_logits=size_logits,
            vt_logits=vt_logits,
        )

        obj.op_collections = SimpleNamespace(
            wirelength_op=lambda pos: (pos.square()).sum(),
            density_op=lambda pos: pos.sum() * 0.0,
            precondition_op=lambda *args, **kwargs: None,
        )
        obj.joint_proximal_objective = JointProximalObjective(obj.params)

        def timing_obj(pos):
            size_term = torch.sigmoid(obj.data_collections.size_logits).sum()
            vt_term = torch.softmax(obj.data_collections.vt_logits, dim=1)[:, 1].sum()
            shared = pos.sum() + size_term + vt_term
            zero = pos.sum() * 0.0
            return shared, zero, shared, zero

        obj.timing_obj = timing_obj

        pos = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        obj.obj_and_grad_fn(pos)

        self.assertGreater(pos.grad.abs().sum().item(), 0.0)
        self.assertGreater(obj.data_collections.size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(obj.data_collections.vt_logits.grad.abs().sum().item(), 0.0)






    def test_final_check_log_temporarily_restores_full_timing_state_when_fast_loop_enabled(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.op_collections = SimpleNamespace()
        placer._record_timing_metrics = mock.Mock()

        params = SimpleNamespace(production_fast_loop=True)
        timing_op = SimpleNamespace(production_fast_loop=True)
        observed = {}

        def timing_obj(pos):
            observed["params_during_timing"] = params.production_fast_loop
            observed["op_during_timing"] = timing_op.production_fast_loop
            return (
                torch.tensor(-1.0),
                torch.tensor(-2.0),
                torch.tensor(-1.0),
                torch.tensor(-2.0),
            )

        def check_log(wns, tns, ws, ts):
            observed["params_during_check_log"] = params.production_fast_loop
            observed["op_during_check_log"] = timing_op.production_fast_loop

        model = SimpleNamespace(
            params=params,
            op_collections=SimpleNamespace(timing_propagation_op=timing_op),
            timing_obj=timing_obj,
            check_log=check_log,
        )

        placer._run_final_check_log_after_writeback(model)

        self.assertFalse(observed["params_during_timing"])
        self.assertFalse(observed["op_during_timing"])
        self.assertFalse(observed["params_during_check_log"])
        self.assertFalse(observed["op_during_check_log"])
        self.assertTrue(params.production_fast_loop)
        self.assertTrue(timing_op.production_fast_loop)

    def test_final_check_timing_record_updates_post_legalization_stage(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.last_final_check_timing_record = {
            "wns": -10.0,
            "tns": -20.0,
            "timing_objective": -10.0,
            "slew_violation": 0.0,
            "cap_violation": 0.0,
            "leakage": 1.25,
            "stage_note": "post_runtime_projection_full_timing_check",
        }

        stale_summary = {
            "pre_projection": {
                "wns": -100.0,
                "tns": -200.0,
                "timing_objective": -100.0,
                "combined_timing_loss": 1.2,
                "slew_violation": 0.0,
                "cap_violation": 180.0,
                "leakage": 1.0,
            },
            "post_projection": {
                "wns": -90.0,
                "tns": -190.0,
                "timing_objective": -90.0,
                "combined_timing_loss": 1.1,
                "slew_violation": 0.0,
                "cap_violation": 183.0,
                "leakage": 1.1,
            },
            "post_legalization": {
                "wns": -95.0,
                "tns": -195.0,
                "timing_objective": -95.0,
                "combined_timing_loss": 1.15,
                "slew_violation": 0.0,
                "cap_violation": 183.0,
                "leakage": 1.1,
            },
            "delta": {},
        }

        refreshed = placer._apply_final_check_timing_stage_record(stale_summary)

        self.assertEqual(refreshed["post_legalization"]["cap_violation"], 0.0)
        self.assertEqual(refreshed["post_legalization"]["tns"], -20.0)
        self.assertEqual(
            refreshed["post_legalization"]["stage_note"],
            "post_runtime_projection_full_timing_check",
        )
        self.assertEqual(refreshed["delta"]["cap_violation"], -180.0)
        self.assertEqual(refreshed["delta"]["tns"], 180.0)

    def test_batched_size_interpolated_pin_properties_match_single_property_api(self):
        batch_builder = getattr(
            sizing_limit_utils,
            "compute_size_interpolated_pin_properties",
            None,
        )
        if not callable(batch_builder):
            self.fail("expected batched size-interpolated pin property API")

        size_var = torch.tensor([1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [80.0, 100.0, 80.0, 120.0],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [0.040, 0.050, 0.040, 0.070],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        batched = batch_builder(
            data_collections,
            [
                data_collections.flat_lib_pin_slew_limit,
                data_collections.flat_lib_pin_cap_limit,
            ],
        )
        single_slew = sizing_limit_utils.compute_size_interpolated_pin_property(
            data_collections,
            data_collections.flat_lib_pin_slew_limit,
        )
        single_cap = sizing_limit_utils.compute_size_interpolated_pin_property(
            data_collections,
            data_collections.flat_lib_pin_cap_limit,
        )

        self.assertEqual(len(batched), 2)
        torch.testing.assert_close(batched[0], single_slew)
        torch.testing.assert_close(batched[1], single_cap)

    def test_batched_size_interpolated_pin_properties_reuse_static_cache_with_dynamic_size(self):
        size_var = torch.tensor([1.0], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            # Force the legacy path so the static property cache is populated.
            size_interpolated_pin_native_op="off",
            pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [80.0, 100.0, 80.0, 120.0],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [0.040, 0.050, 0.040, 0.070],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        first = sizing_limit_utils.compute_size_interpolated_pin_properties(
            data_collections,
            [
                data_collections.flat_lib_pin_slew_limit,
                data_collections.flat_lib_pin_cap_limit,
            ],
        )
        cache = getattr(data_collections, "_size_interpolated_pin_property_cache", None)
        self.assertIsNotNone(cache)
        self.assertEqual(cache.get("num_main_groups"), 1)

        size_var.fill_(2.0)
        second = sizing_limit_utils.compute_size_interpolated_pin_properties(
            data_collections,
            [
                data_collections.flat_lib_pin_slew_limit,
                data_collections.flat_lib_pin_cap_limit,
            ],
        )
        self.assertIs(cache, getattr(data_collections, "_size_interpolated_pin_property_cache", None))
        self.assertAlmostEqual(float(first[1][1].item()), 0.050, places=6)
        self.assertAlmostEqual(float(second[1][1].item()), 0.070, places=6)

    def test_batched_size_interpolated_pin_properties_can_limit_interpolation_to_pin_mask(self):
        size_var = torch.tensor([1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [10.0, 100.0, 20.0, 120.0],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        masked = sizing_limit_utils.compute_size_interpolated_pin_properties(
            data_collections,
            [data_collections.flat_lib_pin_slew_limit],
            pin_mask=torch.tensor([False, True], dtype=torch.bool),
        )[0]

        self.assertAlmostEqual(float(masked[0].item()), 10.0, places=6)
        self.assertAlmostEqual(float(masked[1].item()), 110.0, places=6)

    def test_batched_size_interpolated_pin_properties_mask_handles_interleaved_main_groups(self):
        size_var = torch.tensor([1.5, 1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            inst_main_id=torch.tensor([0, 1], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True, True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0, 0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [10.0, 20.0, 100.0, 200.0],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        masked = sizing_limit_utils.compute_size_interpolated_pin_properties(
            data_collections,
            [data_collections.flat_lib_pin_cap_limit],
            pin_mask=torch.tensor([False, True, False, True], dtype=torch.bool),
        )[0]

        torch.testing.assert_close(
            masked,
            torch.tensor([10.0, 150.0, 10.0, 150.0], dtype=torch.float32),
        )

    def test_batched_size_interpolated_pin_properties_pin_mask_limits_vt_rows(self):
        size_var = torch.tensor([1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 3, 6], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1, 2], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [10.0, 100.0, 1000.0, 20.0, 120.0, 2000.0],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )
        captured_node_ids = []
        original_vt_rows = sizing_limit_utils._vt_probability_rows

        def capture_vt_rows(vt_value, node_ids, *args, **kwargs):
            captured_node_ids.append(node_ids.detach().cpu().tolist())
            return original_vt_rows(vt_value, node_ids, *args, **kwargs)

        with mock.patch.object(
            sizing_limit_utils,
            "_vt_probability_rows",
            side_effect=capture_vt_rows,
        ):
            masked = sizing_limit_utils.compute_size_interpolated_pin_properties(
                data_collections,
                [data_collections.flat_lib_pin_slew_limit],
                pin_mask=torch.tensor([False, True, False], dtype=torch.bool),
            )[0]

        self.assertEqual(captured_node_ids, [[0]])
        self.assertAlmostEqual(float(masked[1].item()), 110.0, places=6)

    def test_size_limit_debug_builder_emits_legal_points_midpoints_and_units(self):
        builder = getattr(sizing_limit_utils, "build_size_interpolated_pin_property_debug", None)
        if not callable(builder):
            self.fail("expected size-limit sampled debug builder for AC-1 artifacts")

        size_var = torch.tensor([1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [80.0, 100.0, 80.0, 120.0],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [0.040, 0.050, 0.040, 0.070],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        artifact = builder(
            data_collections,
            pin_ids=torch.tensor([1], dtype=torch.int64),
            property_specs={
                "slew_limit": {
                    "flat_pin_values": data_collections.flat_lib_pin_slew_limit,
                    "unit": "ps",
                },
                "cap_limit": {
                    "flat_pin_values": data_collections.flat_lib_pin_cap_limit,
                    "unit": "pF",
                },
            },
        )

        self.assertEqual(len(artifact["samples"]), 1)
        sample = artifact["samples"][0]
        self.assertEqual(sample["pin_id"], 1)
        self.assertEqual(sample["pin_offset"], 1)
        self.assertEqual(sample["units"]["slew_limit"], "ps")
        self.assertEqual(sample["units"]["cap_limit"], "pF")
        self.assertEqual(
            [row["size"] for row in sample["legal_size_points"]],
            [1.0, 2.0],
        )
        self.assertAlmostEqual(
            sample["legal_size_points"][0]["properties"]["slew_limit"],
            100.0,
            places=6,
        )
        self.assertAlmostEqual(
            sample["legal_size_points"][1]["properties"]["cap_limit"],
            0.070,
            places=6,
        )
        self.assertEqual(len(sample["continuous_samples"]), 1)
        self.assertAlmostEqual(sample["continuous_samples"][0]["size"], 1.5, places=6)
        self.assertAlmostEqual(
            sample["continuous_samples"][0]["properties"]["slew_limit"],
            110.0,
            places=6,
        )
        self.assertAlmostEqual(
            sample["continuous_samples"][0]["properties"]["cap_limit"],
            0.060,
            places=6,
        )

    def test_size_only_obj_fn_can_add_continuous_slew_cap_terms(self):
        obj = self._make_size_only_place_obj(
            timing_lane="timing_slew_cap",
            slew_weight=2.0,
            cap_weight=3.0,
        )

        pos = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        loss = obj.obj_fn(pos)

        self.assertAlmostEqual(loss.item(), 23.011, places=6)

    def test_timing_propagation_slew_violation_prefers_live_tensor(self):
        op = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(op)
        x = torch.tensor([6.0], dtype=torch.float32, requires_grad=True)
        op.pin_rtran_live = x
        op.pin_ftran_live = x * 0.5
        op.pin_rtran = torch.tensor([999.0], dtype=torch.float32)
        op.pin_ftran = torch.tensor([999.0], dtype=torch.float32)
        total = op.get_total_slew_violation_tensor(
            torch.tensor([True]),
            torch.tensor([2.0], dtype=torch.float32),
        )
        total.backward()
        self.assertAlmostEqual(total.item(), 0.004, places=6)
        self.assertAlmostEqual(x.grad.item(), 0.001, places=6)

    def test_timing_propagation_cap_violation_prefers_live_tensor(self):
        op = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(op)
        x = torch.tensor([0.020], dtype=torch.float32, requires_grad=True)
        op.pin_net_cap_rise_live = x
        op.pin_net_cap_fall_live = x * 0.5
        op.pin_net_cap_rise = torch.tensor([9.0], dtype=torch.float32)
        op.pin_net_cap_fall = torch.tensor([9.0], dtype=torch.float32)
        total = op.get_total_cap_violation_tensor(
            torch.tensor([True]),
            torch.tensor([0.005], dtype=torch.float32),
        )
        total.backward()
        self.assertAlmostEqual(total.item(), 0.015, places=6)
        self.assertAlmostEqual(x.grad.item(), 1.0, places=6)

    def test_size_only_obj_fn_records_objective_term_breakdown_and_lane_identity(self):
        obj = self._make_size_only_place_obj(
            timing_lane="timing_slew",
            slew_weight=1.5,
            cap_weight=9.0,
        )

        pos = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        obj.obj_fn(pos)

        terms = getattr(obj, "last_timing_objective_terms", None)
        self.assertIsInstance(terms, dict)
        self.assertEqual(terms["lane"], "timing_slew")
        self.assertEqual(terms["enabled_terms"], ["timing", "slew"])
        self.assertAlmostEqual(terms["raw_terms"]["slew"], 4.0, places=6)
        self.assertAlmostEqual(terms["weighted_terms"]["slew"], 6.0, places=6)
        self.assertAlmostEqual(terms["raw_terms"]["cap"], 5.0, places=6)
        self.assertAlmostEqual(terms["weighted_terms"]["cap"], 0.0, places=6)
        self.assertEqual(terms["units"]["slew"], "ns")
        self.assertEqual(terms["units"]["cap"], "pF")
        self.assertAlmostEqual(terms["report_aligned_terms"]["cap_violation_fF"], 5000.0, places=6)

    def test_timing_lane_slew_term_changes_gradient_when_enabled(self):
        obj = self._make_size_only_place_obj(timing_lane="timing_slew", slew_weight=2.0)
        x = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
        obj.op_collections.timing_propagation_op.last_total_slew_violation_tensor = x * 4.0
        obj.op_collections.timing_propagation_op.last_total_cap_violation_tensor = x * 0.0
        loss = obj._timing_loss(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 8.0, places=6)

    def test_timing_lane_cap_term_changes_gradient_when_enabled(self):
        obj = self._make_size_only_place_obj(timing_lane="timing_cap", cap_weight=3.0)
        x = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
        obj.op_collections.timing_propagation_op.last_total_slew_violation_tensor = x * 0.0
        obj.op_collections.timing_propagation_op.last_total_cap_violation_tensor = x * 5.0
        loss = obj._timing_loss(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 15.0, places=6)

    def test_timing_only_lane_blocks_slew_cap_gradients(self):
        obj = self._make_size_only_place_obj(timing_lane="timing_only", slew_weight=2.0, cap_weight=3.0)
        x = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        obj.op_collections.timing_propagation_op.last_total_slew_violation_tensor = x * 4.0
        obj.op_collections.timing_propagation_op.last_total_cap_violation_tensor = x * 5.0
        loss = obj._timing_loss(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.assertFalse(loss.requires_grad)

    def test_leakage_tensor_preserves_gradient(self):
        obj = self._make_size_only_place_obj(
            timing_lane="timing_slew_cap_leakage",
            leakage_weight=7.0,
        )
        x = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
        obj.op_collections.timing_propagation_op.last_total_slew_violation_tensor = x * 0.0
        obj.op_collections.timing_propagation_op.last_total_cap_violation_tensor = x * 0.0
        obj.op_collections.timing_propagation_op.last_total_leakage_tensor = x * 2.0
        loss = obj._timing_loss(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 14.0, places=6)

    def test_write_size_limit_samples_artifact_exports_fixed_path_json(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
            )
            size_var = torch.tensor([1.5], dtype=torch.float32)
            vt_var = torch.tensor([[1.0]], dtype=torch.float32)
            placer.data_collections = SimpleNamespace(
                pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
                inst_main_id=torch.tensor([0], dtype=torch.int64),
                inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
                inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
                main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
                cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
                pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 2.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_lib_pin_slew_limit=torch.tensor(
                    [80.0, 100.0, 80.0, 120.0],
                    dtype=torch.float32,
                ),
                flat_lib_pin_cap_limit=torch.tensor(
                    [0.040, 0.050, 0.040, 0.070],
                    dtype=torch.float32,
                ),
                get_size_var=lambda: size_var,
                get_vt_var=lambda: vt_var,
            )

            writer = getattr(placer, "_write_size_limit_samples_artifact", None)
            if not callable(writer):
                self.fail("expected fixed-path size-limit sampled artifact writer")

            artifact = writer(params, pin_ids=torch.tensor([1], dtype=torch.int64))

            self.assertTrue(os.path.exists(artifact["path"]))
            with open(artifact["path"], encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["samples"][0]["units"]["slew_limit"], "ps")
            self.assertEqual(payload["samples"][0]["units"]["cap_limit"], "pF")

    def test_write_timing_objective_terms_artifact_exports_lane_and_terms(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
            )
            model = SimpleNamespace(
                last_timing_objective_terms={
                    "lane": "timing_slew_cap",
                    "enabled_terms": ["timing", "slew", "cap"],
                    "units": {
                        "timing": "loss_proxy",
                        "slew": "ns",
                        "cap": "pF",
                        "leakage": "mW",
                    },
                    "weights": {
                        "timing": 1.0,
                        "slew": 2.0,
                        "cap": 3.0,
                        "leakage": 0.0,
                    },
                    "raw_terms": {
                        "timing": 0.011,
                        "slew": 4.0,
                        "cap": 5.0,
                        "leakage": None,
                    },
                    "weighted_terms": {
                        "timing": 0.011,
                        "slew": 8.0,
                        "cap": 15.0,
                        "leakage": 0.0,
                    },
                    "report_aligned_terms": {
                        "slew_violation_ns": 4.0,
                        "cap_violation_fF": 5000.0,
                        "leakage_mW": None,
                    },
                    "total_loss": 23.011,
                }
            )

            writer = getattr(placer, "_write_timing_objective_terms_artifact", None)
            if not callable(writer):
                self.fail("expected fixed-path timing-objective terms artifact writer")

            artifact = writer(params, model)

            self.assertTrue(os.path.exists(artifact["path"]))
            with open(artifact["path"], encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["lane"], "timing_slew_cap")
            self.assertEqual(payload["enabled_terms"], ["timing", "slew", "cap"])
            self.assertEqual(payload["units"]["cap"], "pF")
            self.assertAlmostEqual(payload["weighted_terms"]["cap"], 15.0, places=6)
            self.assertAlmostEqual(payload["report_aligned_terms"]["cap_violation_fF"], 5000.0, places=6)





    def test_endpoint_stage_payload_prefers_backend_summary_fields(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
            )
            summary_path = os.path.join(
                tmpdir,
                "FFT_endpoint_timing_compare_summary.json",
            )
            csv_path = os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backend_wns": -80.0,
                        "backend_tns": -700.0,
                        "num_backend_negative_slack": 1,
                        "backend_metadata": {
                            "backend": "ieda",
                            "sta_reference_role": "diagnostic_exporter",
                            "sta_is_golden": False,
                            "parasitics_initialization": "unverified",
                            "sta_state_status": "diagnostic_unverified",
                        },
                        "ieda_wns": -999.0,
                        "ieda_tns": -999.0,
                        "num_ieda_negative_slack": 9,
                        "python_wns": -75.0,
                        "python_tns": -650.0,
                    },
                    f,
                )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "u0/D",
                        "cpp_slack": -80.0,
                        "py_slack": -75.0,
                        "slack_delta": 5.0,
                    }
                )

            primary, top_rows = placer._build_endpoint_stage_payload(
                params,
                "post_legalization",
            )

            self.assertEqual(primary["source"], "backend_endpoint_debug")
            self.assertAlmostEqual(primary["wns"], -0.08)
            self.assertAlmostEqual(primary["tns"], -0.7)
            self.assertEqual(primary["num_negative_endpoints"], 1)
            self.assertEqual(primary["sta_reference_role"], "diagnostic_exporter")
            self.assertFalse(primary["sta_is_golden"])
            self.assertEqual(primary["parasitics_initialization"], "unverified")
            self.assertEqual(primary["sta_state_status"], "diagnostic_unverified")
            self.assertEqual(top_rows[0]["backend_slack_ns"], -0.08)
            self.assertEqual(top_rows[0]["ieda_slack_ns"], -0.08)


    def test_openroad_aimp_db_summary_records_endpoint_contract(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.data_manager = SimpleNamespace(dir_workspace="/tmp/ws")
        placedb.num_physical_nodes = 2
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.pin_names = np.array(
            [
                b"drv0:Q",
                b"cmac_a2csb_resp_pd[33]",
                b"u1:D",
                b"g44249__69005:B2",
            ],
            dtype=np.bytes_,
        )
        placedb.net_names = np.array([b"cmac_a2csb_resp_pd[33]"], dtype=np.bytes_)
        placedb.pin2net_map = np.array([0, 0, 0, 0], dtype=np.int32)
        placedb.flat_net2pin_map = np.array([0, 1, 2], dtype=np.int32)
        placedb.flat_net2pin_start_map = np.array([0, 3], dtype=np.int32)
        placedb.net2pin_map = [np.array([0, 1, 2], dtype=np.int32)]
        placedb.net2driver_pin_map = np.array([0], dtype=np.int32)
        placedb.pin_name2id_map = {"cmac_a2csb_resp_pd[33]": 1}
        placedb.end_points = np.array([1, 2], dtype=np.int32)
        placedb.start_points = np.array([0], dtype=np.int32)
        placedb.clock_pins = np.array([], dtype=np.int32)
        placedb.endpoints_rRAT = np.array([400.0, 90000000.0], dtype=np.float32)
        placedb.endpoints_fRAT = np.array([400.0, 90000000.0], dtype=np.float32)
        placedb.endpoints_constraint_arcs = np.array([0, 2, 0, 0, 0, 0], dtype=np.int32)
        placedb.flat_libcell_info = np.zeros((0, 4), dtype=np.float32)
        placedb.flat_libarc_info = np.zeros((0, 6), dtype=np.int32)
        placedb.flat_lib_pin_cap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_rcap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_fcap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_cap_limit = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_slew_limit = np.array([], dtype=np.float32)
        placedb.flat_inst_arcs_by_level = np.array([], dtype=np.int32)
        placedb.flat_inst_arcs_by_level_start = np.array([0], dtype=np.int32)
        placedb.net_flat_arcs = np.array([[0, 1], [0, 2]], dtype=np.int32)
        placedb.pin_pred_start = np.array([0, 0, 1, 2, 3], dtype=np.int32)
        placedb.pin_pred_pin = np.array([0, 0, 2], dtype=np.int32)
        placedb.pin_pred_arc_id = np.array([-1, -1, 7], dtype=np.int32)
        placedb.pin_succ_start = np.array([0, 2, 2, 3, 3], dtype=np.int32)
        placedb.pin_succ_pin = np.array([1, 2], dtype=np.int32)
        placedb.pin_succ_arc_id = np.array([-1, -1], dtype=np.int32)
        placedb.arc_src_pin = np.array([0] * 8, dtype=np.int32)
        placedb.arc_dst_pin = np.array([0] * 7 + [3], dtype=np.int32)
        placedb.pin_pair_arc_keys = np.empty((0, 2), dtype=np.int32)
        placedb.main_id_2_cell_id_start = np.array([0], dtype=np.int32)
        placedb.inst_is_sizeable = np.array([False, False], dtype=np.bool_)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = placedb._write_openroad_aimp_db_summary(
                SimpleNamespace(
                    result_dir=tmpdir,
                    design_name=lambda: "FFT",
                    timing_data_source="openroad",
                    openroad_aimp_debug_artifacts=True,
                    design_inputs={"def": "input.def"},
                )
            )
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
            audit_path = os.path.join(tmpdir, "FFT_openroad_first_level_graph_audit.json")
            self.assertTrue(os.path.exists(audit_path))
            with open(audit_path, encoding="utf-8") as f:
                audit = json.load(f)
            alignment_csv_path = os.path.join(tmpdir, "FFT_first_level_alignment_debug.csv")
            self.assertTrue(os.path.exists(alignment_csv_path))
            with open(alignment_csv_path, newline="", encoding="utf-8") as f:
                alignment_rows = list(csv.DictReader(f))

        endpoint_contract = summary["endpoint_contract"]
        self.assertEqual(endpoint_contract["num_top_level_output_endpoints"], 1)
        self.assertEqual(endpoint_contract["num_top_level_output_r_rat_sentinel"], 0)
        self.assertEqual(
            endpoint_contract["probe_pins"]["cmac_a2csb_resp_pd[33]"]["pin_id"],
            1,
        )
        self.assertEqual(
            endpoint_contract["sample_top_level_output_endpoints"][0]["driver_pin_name"],
            "drv0:Q",
        )
        first_level = summary["first_level_graph_contract"]
        self.assertEqual(first_level["graph_shape"]["pins"], 4)
        self.assertEqual(first_level["graph_shape"]["nets"], 1)
        self.assertEqual(first_level["graph_shape"]["net_arcs"], 2)
        self.assertEqual(first_level["net_driver_contract"]["checked_nets"], 1)
        self.assertEqual(first_level["net_driver_contract"]["driver_first_failures"], 0)
        self.assertEqual(first_level["net_driver_contract"]["net_arc_source_failures"], 0)
        self.assertEqual(first_level["startpoint_contract"]["num_start_points"], 1)
        self.assertEqual(first_level["startpoint_contract"]["startpoints_missing_net"], 0)
        self.assertEqual(first_level["first_fanout_samples"][0]["sink_pin_names"], ["cmac_a2csb_resp_pd[33]", "u1:D"])
        self.assertEqual(audit["net_driver_contract"]["checked_nets"], 1)
        self.assertEqual(audit["first_fanout_samples"][0]["driver_pin_name"], "drv0:Q")
        probe = audit["probe_pin_backward_traces"]["g44249__69005:B2"]
        self.assertEqual(probe["pin_id"], 3)
        self.assertEqual(probe["predecessors"][0]["pred_pin_name"], "u1:D")
        self.assertEqual(probe["predecessors"][0]["edge_type"], "cell_arc")
        self.assertEqual(
            summary["paths"]["first_level_alignment_debug"],
            alignment_csv_path,
        )
        self.assertEqual(len(alignment_rows), 2)
        self.assertEqual(alignment_rows[0]["start_pin_name"], "drv0:Q")
        self.assertEqual(alignment_rows[0]["sink_pin_name"], "cmac_a2csb_resp_pd[33]")
        self.assertEqual(alignment_rows[0]["has_first_level_cell_successor"], "false")

    def test_openroad_aimp_db_debug_artifacts_are_disabled_by_default(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.data_manager = SimpleNamespace(dir_workspace="/tmp/ws")
        placedb.num_physical_nodes = 1
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.pin_names = np.array([b"drv0:Q", b"u1:D"], dtype=np.bytes_)
        placedb.net_names = np.array([b"n0"], dtype=np.bytes_)
        placedb.pin2net_map = np.array([0, 0], dtype=np.int32)
        placedb.flat_net2pin_map = np.array([0, 1], dtype=np.int32)
        placedb.flat_net2pin_start_map = np.array([0, 2], dtype=np.int32)
        placedb.net2pin_map = [np.array([0, 1], dtype=np.int32)]
        placedb.net2driver_pin_map = np.array([0], dtype=np.int32)
        placedb.pin_name2id_map = {}
        placedb.end_points = np.array([1], dtype=np.int32)
        placedb.start_points = np.array([0], dtype=np.int32)
        placedb.clock_pins = np.array([], dtype=np.int32)
        placedb.endpoints_rRAT = np.array([400.0], dtype=np.float32)
        placedb.endpoints_fRAT = np.array([400.0], dtype=np.float32)
        placedb.endpoints_constraint_arcs = np.array([], dtype=np.int32)
        placedb.flat_libcell_info = np.zeros((0, 4), dtype=np.float32)
        placedb.flat_libarc_info = np.zeros((0, 6), dtype=np.int32)
        placedb.flat_lib_pin_cap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_rcap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_fcap = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_cap_limit = np.array([], dtype=np.float32)
        placedb.flat_lib_pin_slew_limit = np.array([], dtype=np.float32)
        placedb.flat_inst_arcs_by_level = np.array([], dtype=np.int32)
        placedb.flat_inst_arcs_by_level_start = np.array([0], dtype=np.int32)
        placedb.net_flat_arcs = np.array([[0, 1]], dtype=np.int32)
        placedb.pin_pred_start = np.array([0, 0, 1], dtype=np.int32)
        placedb.pin_pred_pin = np.array([0], dtype=np.int32)
        placedb.pin_pred_arc_id = np.array([-1], dtype=np.int32)
        placedb.pin_succ_start = np.array([0, 1, 1], dtype=np.int32)
        placedb.pin_succ_pin = np.array([1], dtype=np.int32)
        placedb.pin_succ_arc_id = np.array([-1], dtype=np.int32)
        placedb.arc_src_pin = np.array([], dtype=np.int32)
        placedb.arc_dst_pin = np.array([], dtype=np.int32)
        placedb.pin_pair_arc_keys = np.empty((0, 2), dtype=np.int32)
        placedb.main_id_2_cell_id_start = np.array([0], dtype=np.int32)
        placedb.inst_is_sizeable = np.array([False], dtype=np.bool_)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = placedb._write_openroad_aimp_db_summary(
                SimpleNamespace(
                    result_dir=tmpdir,
                    design_name=lambda: "FFT",
                    timing_data_source="openroad",
                    design_inputs={"def": "input.def"},
                )
            )
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)

            self.assertFalse(os.path.exists(os.path.join(tmpdir, "FFT_openroad_first_level_graph_audit.json")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "FFT_first_level_alignment_debug.csv")))
            self.assertIsNone(summary["paths"]["first_level_graph_audit"])
            self.assertIsNone(summary["paths"]["first_level_alignment_debug"])

    def test_update_continuous_early_stop_state_triggers_after_patience_rounds(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)

        state = placer._build_continuous_early_stop_state(
            SimpleNamespace(
                early_stop_patience=2,
                early_stop_min_delta=0.01,
            )
        )

        should_stop = placer._update_continuous_early_stop_state(
            state,
            combined_timing_loss=10.0,
            iteration=0,
        )
        self.assertFalse(should_stop)
        self.assertEqual(state["best_iteration"], 0)
        self.assertEqual(state["no_improve_rounds"], 0)

        should_stop = placer._update_continuous_early_stop_state(
            state,
            combined_timing_loss=9.995,
            iteration=1,
        )
        self.assertFalse(should_stop)
        self.assertEqual(state["no_improve_rounds"], 1)

        should_stop = placer._update_continuous_early_stop_state(
            state,
            combined_timing_loss=9.994,
            iteration=2,
        )
        self.assertTrue(should_stop)
        self.assertEqual(state["trigger_iteration"], 2)

    def test_openroad_backed_continuous_early_stop_uses_objective_loss(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        params = SimpleNamespace(
            openroad_backed_aimp_db=True,
            early_stop_patience=2,
            early_stop_min_delta=0.01,
            early_stop_restore_best=False,
        )
        state = placer._build_continuous_early_stop_state(params)

        first_loss, first_source = placer._continuous_early_stop_loss_and_source(
            params,
            SimpleNamespace(objective=torch.tensor(10.0), combined_timing_loss=None),
        )
        should_stop = placer._update_continuous_early_stop_state(
            state,
            combined_timing_loss=first_loss,
            iteration=0,
            loss_source=first_source,
        )
        self.assertFalse(should_stop)
        self.assertEqual(state["best_iteration"], 0)

        improved_loss, improved_source = placer._continuous_early_stop_loss_and_source(
            params,
            SimpleNamespace(objective=torch.tensor(9.0), combined_timing_loss=None),
        )
        should_stop = placer._update_continuous_early_stop_state(
            state,
            combined_timing_loss=improved_loss,
            iteration=1,
            loss_source=improved_source,
        )
        self.assertFalse(should_stop)
        self.assertEqual(state["best_iteration"], 1)
        self.assertEqual(state["no_improve_rounds"], 0)
        self.assertEqual(state["loss_source"], "objective")


if __name__ == "__main__":
    unittest.main()
