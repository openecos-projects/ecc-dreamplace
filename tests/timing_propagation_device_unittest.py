#!/usr/bin/env python

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dreamplace.PlaceObj import PlaceObj
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


class _FakeParityOp:
    def __init__(self, wns, tns, endpoint_slack):
        self.wns = torch.tensor(float(wns))
        self.tns = torch.tensor(float(tns))
        self.last_endpoint_slack_tensor = torch.tensor(endpoint_slack, dtype=torch.float32)
        self.materialized_devices = []

    def _materialize_timing_device_tensors(self):
        self.materialized_devices.append(str(self.device))

    def __call__(self, *args, surrogate_mode=None):
        return self.wns, self.tns, torch.tensor(0.0), torch.tensor(0.0)


class TimingPropagationDeviceContractTest(unittest.TestCase):
    def _make_op(self, **kwargs):
        tensor = torch.zeros(1)
        empty_long = torch.zeros(0, dtype=torch.long)
        empty_arcs = torch.zeros((0, 5), dtype=torch.long)
        return TimingPropagation(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            empty_long,
            empty_long,
            empty_long,
            empty_long,
            torch.zeros(1, dtype=torch.long),
            empty_arcs,
            torch.zeros(1, dtype=torch.long),
            empty_arcs,
            None,
            torch.zeros(1, dtype=torch.long),
            empty_arcs,
            torch.zeros((0, 5), dtype=torch.long),
            empty_arcs,
            torch.zeros(1, dtype=torch.long),
            tensor,
            tensor,
            empty_long,
            torch.zeros(1, dtype=torch.long),
            empty_long,
            torch.zeros(1, dtype=torch.long),
            **kwargs,
        )

    def test_inherit_cpu_records_ok_contract(self):
        op = self._make_op(
            timing_propagation_profile=True,
            timing_propagation_device="inherit",
            timing_propagation_global_gpu=0,
            timing_propagation_global_gpu_id=0,
        )

        self.assertEqual(op.resolved_timing_propagation_device, torch.device("cpu"))
        self.assertEqual(op.last_device_contract["requested_device"], "inherit")
        self.assertEqual(op.last_device_contract["resolved_device"], "cpu")
        self.assertEqual(op.last_device_contract["resolved_device_type"], "cpu")
        self.assertEqual(op.last_device_contract["mismatch_count"], 0)

    def test_explicit_cpu_override_wins_under_global_gpu(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            op = self._make_op(
                timing_propagation_device="cpu",
                timing_propagation_global_gpu=1,
                timing_propagation_global_gpu_id=2,
            )

        self.assertEqual(op.resolved_timing_propagation_device, torch.device("cpu"))
        self.assertEqual(op.last_device_contract["requested_device"], "cpu")
        self.assertEqual(op.last_device_contract["resolved_device"], "cpu")
        self.assertEqual(op.inrdelays.device, torch.device("cpu"))
        self.assertEqual(op.last_device_contract["mismatch_count"], 0)

    def test_inherited_cuda_honors_gpu_id_when_cuda_is_available(self):
        shell = object.__new__(TimingPropagation)
        shell.timing_propagation_device = "inherit"
        shell.timing_propagation_global_gpu = 1
        shell.timing_propagation_global_gpu_id = 2

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=4
        ):
            self.assertEqual(shell._resolve_timing_device(), torch.device("cuda:2"))

    def test_explicit_cuda_honors_gpu_id_when_cuda_is_available(self):
        shell = object.__new__(TimingPropagation)
        shell.timing_propagation_device = "cuda"
        shell.timing_propagation_global_gpu = 0
        shell.timing_propagation_global_gpu_id = 1

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=2
        ):
            self.assertEqual(shell._resolve_timing_device(), torch.device("cuda:1"))

    def test_explicit_cuda_rejects_out_of_range_gpu_id(self):
        shell = object.__new__(TimingPropagation)
        shell.timing_propagation_device = "cuda"
        shell.timing_propagation_global_gpu = 1
        shell.timing_propagation_global_gpu_id = 3

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=2
        ):
            with self.assertRaisesRegex(RuntimeError, "gpu_id 3"):
                shell._resolve_timing_device()

    def test_explicit_cuda_fails_when_cuda_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                self._make_op(timing_propagation_device="cuda")

    def test_device_contract_reports_mismatched_tensor(self):
        shell = object.__new__(TimingPropagation)
        shell.resolved_timing_propagation_device = torch.device("cuda:1")
        shell.timing_propagation_device = "cuda"
        shell.timing_propagation_global_gpu = 1
        shell.timing_propagation_global_gpu_id = 1

        contract = shell._build_device_contract(
            {"cpu_tensor": torch.zeros(1)},
            scope="unit",
        )

        self.assertEqual(contract["status"], "mismatch")
        self.assertEqual(contract["mismatch_count"], 1)
        self.assertEqual(contract["mismatches"][0]["name"], "cpu_tensor")
        self.assertEqual(contract["mismatches"][0]["expected_device"], "cuda:1")

    def test_profile_payload_schema_and_artifact_gate(self):
        op = self._make_op(
            timing_propagation_profile=True,
            timing_propagation_device="inherit",
        )
        op.last_critical_endpoint_pruning_stats = {
            "active_endpoint_count": 2,
            "applied": True,
        }
        op.last_profile_payload = op._build_profile_payload(
            total_runtime_ms=1.5,
            stage_runtime_ms={"input_clone": 0.1},
            surrogate_mode="surrogate_only",
            device_contract=op.last_device_contract,
            cell_aat_profile={
                "enabled": True,
                "num_levels": 2,
                "num_arcs": 10,
                "query_count": 8,
                "total_ms": 0.75,
                "query_build_ms": 0.1,
                "query_eval_ms": 0.2,
                "query_pack_ms": 0.01,
                "surrogate_batch_ms": 0.12,
                "fallback_lut_ms": 0.02,
                "fallback_lut_select_ms": 0.01,
                "fallback_lut_eval_ms": 0.01,
                "result_split_ms": 0.03,
                "result_all_valid_check_ms": 0.004,
                "result_clone_ms": 0.005,
                "result_fallback_mask_ms": 0.006,
                "result_fallback_assign_ms": 0.007,
                "result_split_non_lut_ms": 0.008,
                "surrogate_state_ms": 0.04,
                "surrogate_index_ms": 0.05,
                "surrogate_support_ms": 0.06,
                "surrogate_forward_ms": 0.07,
                "surrogate_scatter_ms": 0.08,
                "cell_update_ms": 0.3,
                "unique_net_ms": 0.05,
                "arc_delay_scatter_ms": 0.1,
                "net_aat_ms": 0.0,
                "surrogate_candidate_count": 20,
                "surrogate_supported_count": 18,
                "surrogate_unsupported_count": 2,
                "surrogate_non_sizeable_candidate_count": 0,
                "surrogate_fixed_non_sizeable_count": 0,
                "surrogate_invalid_main_id_count": 0,
                "surrogate_non_sizeable_count": 0,
                "surrogate_support_missing_count": 2,
                "fallback_lut_query_count": 2,
                "fallback_lut_entry_count": 2,
                "fallback_lut_scalar_entry_count": 1,
                "fallback_lut_trans_1d_entry_count": 0,
                "fallback_lut_cap_1d_entry_count": 0,
                "fallback_lut_2d_entry_count": 1,
                "fallback_lut_f_delay_entry_count": 0,
                "fallback_lut_r_delay_entry_count": 1,
                "fallback_lut_f_trans_entry_count": 1,
                "fallback_lut_r_trans_entry_count": 0,
                "direct_lut_query_count": 0,
                "direct_lut_entry_count": 0,
                "surrogate_support_missing_unique_key_count": 1,
                "surrogate_support_missing_top_keys": [
                    {
                        "main_id": 7,
                        "arc_offset": 11,
                        "arc_type": 2,
                        "count": 2,
                    }
                ],
                "surrogate_unsupported_unique_key_count": 1,
                "surrogate_unsupported_top_keys": [
                    {
                        "main_id": 7,
                        "arc_offset": 11,
                        "arc_type": 2,
                        "count": 2,
                    }
                ],
                "top_levels": [],
            },
        )

        payload = op.build_timing_propagation_profile_artifact()

        self.assertEqual(payload["artifact"], "timing_propagation_profile_latest")
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["requested_device"], "inherit")
        self.assertEqual(payload["resolved_device"], "cpu")
        self.assertEqual(payload["resolved_device_type"], "cpu")
        self.assertEqual(payload["device_contract"]["mismatch_count"], 0)
        self.assertIn("checked_tensors", payload["device_contract"])
        self.assertEqual(payload["stage_runtime_ms"]["input_clone"], 0.1)
        self.assertIn("cell_aat_levels_detail", payload)
        self.assertTrue(payload["cell_aat_levels_detail"]["enabled"])
        self.assertEqual(payload["cell_aat_levels_detail"]["num_levels"], 2)
        self.assertEqual(payload["cell_aat_levels_detail"]["query_count"], 8)
        self.assertEqual(payload["cell_aat_levels_detail"]["query_build_ms"], 0.1)
        self.assertEqual(payload["cell_aat_levels_detail"]["query_eval_ms"], 0.2)
        self.assertEqual(payload["cell_aat_levels_detail"]["query_pack_ms"], 0.01)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_batch_ms"], 0.12)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_ms"], 0.02)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_select_ms"], 0.01)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_eval_ms"], 0.01)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_split_ms"], 0.03)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_all_valid_check_ms"], 0.004)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_clone_ms"], 0.005)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_fallback_mask_ms"], 0.006)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_fallback_assign_ms"], 0.007)
        self.assertEqual(payload["cell_aat_levels_detail"]["result_split_non_lut_ms"], 0.008)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_state_ms"], 0.04)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_index_ms"], 0.05)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_support_ms"], 0.06)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_forward_ms"], 0.07)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_scatter_ms"], 0.08)
        self.assertEqual(payload["cell_aat_levels_detail"]["cell_update_ms"], 0.3)
        self.assertEqual(payload["cell_aat_levels_detail"]["unique_net_ms"], 0.05)
        self.assertEqual(payload["cell_aat_levels_detail"]["arc_delay_scatter_ms"], 0.1)
        self.assertEqual(payload["cell_aat_levels_detail"]["net_aat_ms"], 0.0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_candidate_count"], 20)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_supported_count"], 18)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_unsupported_count"], 2)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_non_sizeable_candidate_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_fixed_non_sizeable_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_invalid_main_id_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_non_sizeable_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_support_missing_count"], 2)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_query_count"], 2)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_entry_count"], 2)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_scalar_entry_count"], 1)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_trans_1d_entry_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_cap_1d_entry_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_2d_entry_count"], 1)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_f_delay_entry_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_r_delay_entry_count"], 1)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_f_trans_entry_count"], 1)
        self.assertEqual(payload["cell_aat_levels_detail"]["fallback_lut_r_trans_entry_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["direct_lut_query_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["direct_lut_entry_count"], 0)
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_support_missing_unique_key_count"], 1)
        self.assertEqual(
            payload["cell_aat_levels_detail"]["surrogate_support_missing_top_keys"],
            [
                {
                    "main_id": 7,
                    "arc_offset": 11,
                    "arc_type": 2,
                    "count": 2,
                }
            ],
        )
        self.assertEqual(payload["cell_aat_levels_detail"]["surrogate_unsupported_unique_key_count"], 1)
        self.assertEqual(
            payload["cell_aat_levels_detail"]["surrogate_unsupported_top_keys"],
            [
                {
                    "main_id": 7,
                    "arc_offset": 11,
                    "arc_type": 2,
                    "count": 2,
                }
            ],
        )
        self.assertEqual(payload["post_timing_boundary"]["core_device"], "cpu")
        self.assertTrue(payload["post_timing_boundary"]["excluded_from_runtime_ms"])
        self.assertFalse(payload["parity"]["checked"])
        self.assertEqual(payload["parity"]["reason"], "not_run")

    def test_forward_profile_payload_carries_cell_aat_detail(self):
        op = self._make_op(
            timing_propagation_profile=True,
            timing_propagation_device="inherit",
        )
        op.num_pins = 1
        op.dtype = torch.float32
        op.flat_inst_arcs_by_level = torch.zeros((1, 6), dtype=torch.long)
        op.flat_inst_arcs_by_level_start = torch.tensor([0, 0, 1], dtype=torch.long)
        working_arcs = torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.long)
        working_offsets = torch.tensor([0, 0, 1], dtype=torch.long)
        working_indices = torch.tensor([0], dtype=torch.long)

        op._validate_runtime_indices = lambda: None
        op._resolve_surrogate_mode = lambda surrogate_mode=None: (False, False)
        op.calculate_clk2q_aat = (
            lambda pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *_args, **_kwargs: (
                torch.tensor([0], dtype=torch.long),
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
            )
        )
        op.calculate_net_aat_level = (
            lambda _curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *_args: (
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
            )
        )
        op._build_traversal_pruning_working_view = (
            lambda _device: (working_arcs, working_offsets, working_indices, False)
        )

        def fake_cell_aat_level(
            _inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            *_args,
            **_kwargs,
        ):
            op._record_cell_aat_level_profile(
                {
                    "level": 1,
                    "num_arcs": 1,
                    "query_count": 4,
                    "query_build_ms": 1.0,
                    "query_eval_ms": 2.0,
                    "cell_update_ms": 3.0,
                    "unique_net_ms": 4.0,
                    "surrogate_candidate_count": 6,
                    "surrogate_supported_count": 5,
                    "surrogate_unsupported_count": 1,
                    "surrogate_non_sizeable_candidate_count": 0,
                    "surrogate_fixed_non_sizeable_count": 0,
                    "surrogate_invalid_main_id_count": 0,
                    "surrogate_non_sizeable_count": 0,
                    "surrogate_support_missing_count": 1,
                    "fallback_lut_query_count": 1,
                    "fallback_lut_entry_count": 1,
                    "fallback_lut_2d_entry_count": 1,
                    "fallback_lut_r_trans_entry_count": 1,
                    "surrogate_support_missing_unique_key_count": 1,
                    "surrogate_support_missing_top_keys": [
                        {
                            "main_id": 3,
                            "arc_offset": 9,
                            "arc_type": 3,
                            "count": 1,
                        }
                    ],
                    "surrogate_unsupported_unique_key_count": 1,
                    "surrogate_unsupported_top_keys": [
                        {
                            "main_id": 3,
                            "arc_offset": 9,
                            "arc_type": 3,
                            "count": 1,
                        }
                    ],
                }
            )
            delay = torch.zeros(1, dtype=torch.float32)
            return (
                torch.tensor([0], dtype=torch.long),
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                delay,
                delay,
                delay,
                delay,
            )

        op.calculate_cell_aat_level = fake_cell_aat_level
        op.calculate_setup_rat = lambda pin_rRAT, pin_fRAT, *_args: (pin_rRAT, pin_fRAT)
        op.calculate_net_rat_level = (
            lambda _cur_endpoint, pin_rRAT, pin_fRAT, *_args: (pin_rRAT, pin_fRAT)
        )
        op.calculate_cell_rat_level = (
            lambda _inst_arcs, pin_rRAT, pin_fRAT, *_args: (
                torch.tensor([0], dtype=torch.long),
                pin_rRAT,
                pin_fRAT,
            )
        )
        op.update_critical_endpoint_pruning_state = lambda iteration: None
        op._refresh_traversal_pruning_state = lambda iteration: None
        op._try_run_cpu_cuda_parity = lambda *_args, **_kwargs: op.last_parity_payload

        captured_payload_kwargs = {}
        original_build_profile_payload = op._build_profile_payload

        def capture_build_profile_payload(**kwargs):
            captured_payload_kwargs.update(kwargs)
            return original_build_profile_payload(**kwargs)

        op._build_profile_payload = capture_build_profile_payload

        zeros = torch.zeros(1, dtype=torch.float32)
        op.forward(
            {"rise": zeros, "fall": zeros},
            {"rise": zeros, "fall": zeros},
            {"rise": zeros, "fall": zeros},
            surrogate_mode="surrogate_only",
        )

        self.assertIn("cell_aat_profile", captured_payload_kwargs)
        detail = captured_payload_kwargs["cell_aat_profile"]
        self.assertTrue(detail["enabled"])
        self.assertEqual(detail["num_levels"], 1)
        self.assertEqual(detail["num_arcs"], 1)
        self.assertEqual(detail["query_count"], 4)
        self.assertEqual(detail["query_build_ms"], 1.0)
        self.assertEqual(detail["query_eval_ms"], 2.0)
        self.assertEqual(detail["query_pack_ms"], 0.0)
        self.assertEqual(detail["surrogate_batch_ms"], 0.0)
        self.assertEqual(detail["fallback_lut_ms"], 0.0)
        self.assertEqual(detail["result_split_ms"], 0.0)
        self.assertEqual(detail["surrogate_state_ms"], 0.0)
        self.assertEqual(detail["surrogate_index_ms"], 0.0)
        self.assertEqual(detail["surrogate_support_ms"], 0.0)
        self.assertEqual(detail["surrogate_forward_ms"], 0.0)
        self.assertEqual(detail["surrogate_scatter_ms"], 0.0)
        self.assertEqual(detail["cell_update_ms"], 3.0)
        self.assertEqual(detail["unique_net_ms"], 4.0)
        self.assertEqual(detail["surrogate_candidate_count"], 6)
        self.assertEqual(detail["surrogate_supported_count"], 5)
        self.assertEqual(detail["surrogate_unsupported_count"], 1)
        self.assertEqual(detail["surrogate_non_sizeable_candidate_count"], 0)
        self.assertEqual(detail["surrogate_fixed_non_sizeable_count"], 0)
        self.assertEqual(detail["surrogate_invalid_main_id_count"], 0)
        self.assertEqual(detail["surrogate_non_sizeable_count"], 0)
        self.assertEqual(detail["surrogate_support_missing_count"], 1)
        self.assertEqual(detail["fallback_lut_query_count"], 1)
        self.assertEqual(detail["fallback_lut_entry_count"], 1)
        self.assertEqual(detail["fallback_lut_scalar_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_trans_1d_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_cap_1d_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_2d_entry_count"], 1)
        self.assertEqual(detail["fallback_lut_f_delay_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_r_delay_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_f_trans_entry_count"], 0)
        self.assertEqual(detail["fallback_lut_r_trans_entry_count"], 1)
        self.assertEqual(detail["direct_lut_query_count"], 0)
        self.assertEqual(detail["direct_lut_entry_count"], 0)
        self.assertEqual(detail["surrogate_support_missing_unique_key_count"], 1)
        self.assertEqual(
            detail["surrogate_support_missing_top_keys"],
            [
                {
                    "main_id": 3,
                    "arc_offset": 9,
                    "arc_type": 3,
                    "count": 1,
                }
            ],
        )
        self.assertEqual(detail["surrogate_unsupported_unique_key_count"], 1)
        self.assertEqual(
            detail["surrogate_unsupported_top_keys"],
            [
                {
                    "main_id": 3,
                    "arc_offset": 9,
                    "arc_type": 3,
                    "count": 1,
                }
            ],
        )
        self.assertEqual(detail["top_levels"][0]["level"], 1)

    def test_profile_artifact_disabled_when_profile_off(self):
        op = self._make_op(timing_propagation_profile=False)
        op.last_profile_payload = {"artifact": "timing_propagation_profile_latest"}

        self.assertIsNone(op.build_timing_propagation_profile_artifact())

    def test_parity_payload_records_cuda_unavailable_skip(self):
        op = self._make_op(timing_propagation_parity_check=True)
        dummy = {
            "rise": torch.zeros(1),
            "fall": torch.zeros(1),
        }

        with mock.patch("torch.cuda.is_available", return_value=False):
            payload = op._try_run_cpu_cuda_parity(
                dummy,
                dummy,
                dummy,
                "lut_only",
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.zeros(1),
            )

        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["checked"])
        self.assertEqual(payload["reason"], "cuda_unavailable")
        self.assertFalse(payload["cuda_timing_safe"])

    def test_cell_modeling_parity_skip_records_narrow_blocker(self):
        op = self._make_op(timing_propagation_parity_check=True)
        dummy = {
            "rise": torch.zeros(1),
            "fall": torch.zeros(1),
        }

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch.object(
            op,
            "_cell_modeling_enabled",
            return_value=True,
        ):
            payload = op._try_run_cpu_cuda_parity(
                dummy,
                dummy,
                dummy,
                "surrogate_only",
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.zeros(1),
            )

        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["checked"])
        self.assertIsNone(payload["passed"])
        self.assertFalse(payload["cuda_timing_safe"])
        self.assertNotEqual(payload["reason"], "cell_modeling_surrogate_parity_not_supported")
        self.assertEqual(payload["blocker_function"], "TimingPropagation._run_cpu_cuda_parity")
        self.assertIn("timing_propagation.py", payload["blocker_file"])
        self.assertIn("cell_modeling_op", payload["blocker_object"])
        self.assertIn("CPU/CUDA", payload["next_fix"])

    def test_parity_check_reuses_first_result(self):
        op = self._make_op(timing_propagation_parity_check=True)
        first_payload = {
            "artifact": "timing_propagation_parity_latest",
            "enabled": True,
            "checked": False,
            "reason": "cuda_unavailable",
        }
        op.last_parity_payload = first_payload
        dummy = {
            "rise": torch.zeros(1),
            "fall": torch.zeros(1),
        }

        with mock.patch.object(op, "_run_cpu_cuda_parity") as run_parity:
            payload = op._try_run_cpu_cuda_parity(
                dummy,
                dummy,
                dummy,
                "lut_only",
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.zeros(1),
            )

        self.assertIs(payload, first_payload)
        run_parity.assert_not_called()

    def test_run_cpu_cuda_parity_reports_passed_payload(self):
        op = self._make_op(
            timing_propagation_parity_check=True,
            timing_propagation_global_gpu_id=0,
        )
        cpu_op = _FakeParityOp(-1.0, -3.0, [-1.0, -2.0])
        cuda_op = _FakeParityOp(-1.0, -3.0, [-1.0, -2.0])
        dummy = {
            "rise": torch.zeros(1),
            "fall": torch.zeros(1),
        }

        with mock.patch(
            "dreamplace.ops.timing_propagation.timing_propagation.copy.deepcopy",
            side_effect=[cpu_op, cuda_op],
        ), mock.patch.object(
            op,
            "_move_forward_inputs",
            side_effect=lambda delays, impulses, caps, device: (delays, impulses, caps),
        ):
            payload = op._run_cpu_cuda_parity(
                dummy,
                dummy,
                dummy,
                "lut_only",
                torch.tensor(-1.0),
                torch.tensor(-3.0),
                torch.tensor([-1.0, -2.0]),
            )

        self.assertTrue(payload["checked"])
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["cuda_timing_safe"])
        self.assertEqual(payload["reason"], "passed")
        self.assertEqual(payload["wns_abs_diff_ps"], 0.0)
        self.assertEqual(payload["tns_abs_diff_ps"], 0.0)
        self.assertEqual(payload["max_endpoint_slack_abs_diff_ps"], 0.0)
        self.assertEqual(cpu_op.materialized_devices, ["cpu"])
        self.assertEqual(cuda_op.materialized_devices, ["cuda:0"])

    def test_run_cpu_cuda_parity_reports_failed_payload(self):
        op = self._make_op(
            timing_propagation_parity_check=True,
            timing_propagation_global_gpu_id=0,
            timing_propagation_parity_atol_ps=0.01,
            timing_propagation_parity_rtol=0.0,
        )
        cpu_op = _FakeParityOp(-1.0, -3.0, [-1.0, -2.0])
        cuda_op = _FakeParityOp(-1.5, -3.5, [-1.0, -2.5])
        dummy = {
            "rise": torch.zeros(1),
            "fall": torch.zeros(1),
        }

        with mock.patch(
            "dreamplace.ops.timing_propagation.timing_propagation.copy.deepcopy",
            side_effect=[cpu_op, cuda_op],
        ), mock.patch.object(
            op,
            "_move_forward_inputs",
            side_effect=lambda delays, impulses, caps, device: (delays, impulses, caps),
        ):
            payload = op._run_cpu_cuda_parity(
                dummy,
                dummy,
                dummy,
                "lut_only",
                torch.tensor(-1.0),
                torch.tensor(-3.0),
                torch.tensor([-1.0, -2.0]),
            )

        self.assertTrue(payload["checked"])
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["cuda_timing_safe"])
        self.assertEqual(payload["reason"], "diff_exceeds_tolerance")
        self.assertAlmostEqual(payload["wns_abs_diff_ps"], 0.5)
        self.assertAlmostEqual(payload["tns_abs_diff_ps"], 0.5)
        self.assertAlmostEqual(payload["max_endpoint_slack_abs_diff_ps"], 0.5)

    def test_parity_artifact_gate(self):
        op = self._make_op(timing_propagation_parity_check=True)
        op.last_parity_payload = {"artifact": "timing_propagation_parity_latest"}

        self.assertEqual(
            op.build_timing_propagation_parity_artifact(),
            {"artifact": "timing_propagation_parity_latest"},
        )

        disabled = self._make_op(timing_propagation_parity_check=False)
        disabled.last_parity_payload = {"artifact": "timing_propagation_parity_latest"}
        self.assertIsNone(disabled.build_timing_propagation_parity_artifact())


class PlaceObjTimingPropagationProfileArtifactTest(unittest.TestCase):
    def test_placeobj_writes_profile_artifact_from_timing_op_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            place_obj = object.__new__(PlaceObj)
            place_obj.params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "unit_design",
            )
            payload = {
                "artifact": "timing_propagation_profile_latest",
                "enabled": True,
                "runtime_ms": 1.0,
            }
            timing_op = SimpleNamespace(
                build_timing_propagation_profile_artifact=lambda: payload
            )
            place_obj.op_collections = SimpleNamespace(timing_propagation_op=timing_op)

            artifact_path = place_obj._write_timing_propagation_profile_artifact()

            self.assertEqual(
                Path(artifact_path).name,
                "unit_design_timing_propagation_profile_latest.json",
            )
            self.assertEqual(
                json.loads(Path(artifact_path).read_text(encoding="utf-8")),
                payload,
            )

    def test_placeobj_writes_parity_artifact_from_timing_op_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            place_obj = object.__new__(PlaceObj)
            place_obj.params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "unit_design",
            )
            payload = {
                "artifact": "timing_propagation_parity_latest",
                "enabled": True,
                "checked": False,
                "reason": "cuda_unavailable",
            }
            timing_op = SimpleNamespace(
                build_timing_propagation_parity_artifact=lambda: payload
            )
            place_obj.op_collections = SimpleNamespace(timing_propagation_op=timing_op)

            artifact_path = place_obj._write_timing_propagation_parity_artifact()

            self.assertEqual(
                Path(artifact_path).name,
                "unit_design_timing_propagation_parity_latest.json",
            )
            self.assertEqual(
                json.loads(Path(artifact_path).read_text(encoding="utf-8")),
                payload,
            )


if __name__ == "__main__":
    unittest.main()
