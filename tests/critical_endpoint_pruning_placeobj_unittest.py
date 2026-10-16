#!/usr/bin/env python

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT / "AiEDA"))
sys.path.append(str(REPO_ROOT / "AiEDA" / "third_party" / "AutoDMP"))

from dreamplace.PlaceObj import PlaceObj


class CriticalEndpointPruningPlaceObjTest(unittest.TestCase):
    def _make_shell(self, pruning_mode="off", active_endpoint_ids=None, artifact_payload=None):
        shell = object.__new__(PlaceObj)
        shell.params = SimpleNamespace(
            timing_objective_lane="timing_only",
            critical_endpoint_pruning_mode=pruning_mode,
        )
        shell.timing_wns_coeff = 1.0
        shell.timing_tns_coeff = 1.0
        shell.pos = [torch.zeros(1)]
        def select_critical_endpoint_timing(full_wns, full_tns):
            metadata = {
                "mode": pruning_mode,
                "applied": pruning_mode == "dynamic",
                "active_endpoint_count": len(active_endpoint_ids or []),
                "selected_endpoint_wns_ps": -20.0 if pruning_mode == "dynamic" else None,
                "selected_endpoint_tns_ps": -30.0 if pruning_mode == "dynamic" else None,
                "full_endpoint_wns_ps": float(full_wns.detach().item()),
                "full_endpoint_tns_ps": float(full_tns.detach().item()),
            }
            if pruning_mode == "dynamic":
                return torch.tensor(-20.0), torch.tensor(-30.0), metadata
            return full_wns, full_tns, metadata

        def build_critical_endpoint_pruning_artifact(iteration):
            if artifact_payload is not None:
                payload = dict(artifact_payload)
                payload["iteration"] = int(iteration)
                return payload
            return {
                "artifact": "critical_endpoint_pruning_latest",
                "iteration": int(iteration),
                "mode": pruning_mode,
                "active_endpoint_count": len(active_endpoint_ids or []),
                "full_endpoint_count": 4,
                "selected_endpoint_wns_ps": -20.0 if pruning_mode == "dynamic" else None,
                "selected_endpoint_tns_ps": -30.0 if pruning_mode == "dynamic" else None,
                "full_endpoint_wns_ps": -20.0,
                "full_endpoint_tns_ps": -36.0,
                "refresh_interval": 10,
                "refresh_decision": "refreshed",
                "selection_provenance": {
                    "owner": "TimingPropagation",
                    "source": "last_endpoint_slack_tensor",
                    "top_k": 2,
                },
                "active_endpoint_ids": active_endpoint_ids or [],
                "active_endpoint_ids_truncated": False,
                "selector_stats": {"active_endpoint_count": len(active_endpoint_ids or [])},
            }

        timing_op = SimpleNamespace(
            last_endpoint_slack_tensor=torch.tensor([-5.0, -20.0, -1.0, -10.0]),
            select_critical_endpoint_timing=select_critical_endpoint_timing,
            build_critical_endpoint_pruning_artifact=build_critical_endpoint_pruning_artifact,
        )
        shell.op_collections = SimpleNamespace(timing_propagation_op=timing_op)
        shell.data_collections = SimpleNamespace(
            end_points=torch.tensor([10, 11, 12, 13], dtype=torch.long)
        )
        return shell

    def test_off_mode_uses_full_wns_tns_unchanged(self):
        shell = self._make_shell(pruning_mode="off", active_endpoint_ids=[11, 13])

        loss = shell._timing_objective_term_components(
            torch.tensor(-20.0),
            torch.tensor(-36.0),
        )

        self.assertAlmostEqual(loss.item(), 56.0)
        self.assertEqual(
            shell.last_timing_objective_terms["critical_endpoint_pruning"]["mode"],
            "off",
        )
        self.assertFalse(
            shell.last_timing_objective_terms["critical_endpoint_pruning"]["applied"]
        )

    def test_dynamic_mode_uses_selected_endpoint_wns_tns_for_timing_term(self):
        shell = self._make_shell(pruning_mode="dynamic", active_endpoint_ids=[11, 13])

        loss = shell._timing_objective_term_components(
            torch.tensor(-20.0),
            torch.tensor(-36.0),
        )

        self.assertAlmostEqual(loss.item(), 50.0)
        pruning_terms = shell.last_timing_objective_terms["critical_endpoint_pruning"]
        self.assertEqual(pruning_terms["mode"], "dynamic")
        self.assertTrue(pruning_terms["applied"])
        self.assertEqual(pruning_terms["active_endpoint_count"], 2)
        self.assertAlmostEqual(pruning_terms["selected_endpoint_wns_ps"], -20.0)
        self.assertAlmostEqual(pruning_terms["selected_endpoint_tns_ps"], -30.0)
        self.assertAlmostEqual(pruning_terms["full_endpoint_wns_ps"], -20.0)
        self.assertAlmostEqual(pruning_terms["full_endpoint_tns_ps"], -36.0)

    def test_writes_latest_pruning_artifact(self):
        shell = self._make_shell(pruning_mode="dynamic", active_endpoint_ids=[11, 13])
        with tempfile.TemporaryDirectory() as tmpdir:
            shell.params.result_dir = tmpdir
            shell.params.design_name = lambda: "NV_NVDLA_partition_m"

            artifact_path = shell._write_critical_endpoint_pruning_artifact(
                iteration=7,
            )

            self.assertEqual(
                Path(artifact_path),
                Path(tmpdir) / "NV_NVDLA_partition_m_critical_endpoint_pruning_latest.json",
            )
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["iteration"], 7)
            self.assertEqual(payload["mode"], "dynamic")
            self.assertEqual(payload["active_endpoint_count"], 2)
            self.assertEqual(payload["active_endpoint_ids"], [11, 13])
            self.assertEqual(payload["full_endpoint_count"], 4)
            self.assertAlmostEqual(payload["selected_endpoint_wns_ps"], -20.0)
            self.assertAlmostEqual(payload["selected_endpoint_tns_ps"], -30.0)
            self.assertAlmostEqual(payload["full_endpoint_wns_ps"], -20.0)
            self.assertAlmostEqual(payload["full_endpoint_tns_ps"], -36.0)
            self.assertEqual(payload["refresh_interval"], 10)
            self.assertEqual(payload["refresh_decision"], "refreshed")
            self.assertEqual(payload["selection_provenance"]["owner"], "TimingPropagation")

    def test_writes_latest_pruning_artifact_without_nan_timing_fields(self):
        owner_payload = {
            "artifact": "critical_endpoint_pruning_latest",
            "iteration": 0,
            "mode": "dynamic",
            "active_endpoint_count": 1,
            "full_endpoint_count": 1,
            "selected_endpoint_wns_ps": -5.0,
            "selected_endpoint_tns_ps": -5.0,
            "full_endpoint_wns_ps": -5.0,
            "full_endpoint_tns_ps": -5.0,
            "refresh_interval": 10,
            "refresh_decision": "refreshed",
            "selection_provenance": {"owner": "TimingPropagation"},
            "active_endpoint_ids": [2],
            "active_endpoint_ids_truncated": False,
            "selector_stats": {"full_endpoint_count": 1},
        }
        shell = self._make_shell(
            pruning_mode="dynamic",
            active_endpoint_ids=[2],
            artifact_payload=owner_payload,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shell.params.result_dir = tmpdir
            shell.params.design_name = lambda: "NV_NVDLA_partition_m"

            artifact_path = shell._write_critical_endpoint_pruning_artifact(iteration=13)

            payload_text = Path(artifact_path).read_text(encoding="utf-8")
            self.assertNotIn("NaN", payload_text)
            payload = json.loads(payload_text)
            self.assertEqual(payload["iteration"], 13)
            self.assertEqual(payload["full_endpoint_count"], 1)
            self.assertAlmostEqual(payload["full_endpoint_wns_ps"], -5.0)
            self.assertAlmostEqual(payload["selected_endpoint_tns_ps"], -5.0)


class FakeTimingPropagationOp:
    def __init__(self):
        self.pin_slack = torch.tensor([-1.0], dtype=torch.float32)

    def __call__(
        self,
        delays,
        impulses,
        loads,
        surrogate_mode=None,
        dynamic_net_arc_inputs=None,
        dynamic_net_subgraph_inputs=None,
        dynamic_net_provider=None,
    ):
        return (
            torch.tensor(-1.0, dtype=torch.float32),
            torch.tensor(-2.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

    def get_pin_slack(self):
        return self.pin_slack

    def get_total_slew_violation_tensor(self, inst_pins_mask, slew_limits):
        return torch.tensor(0.25, dtype=torch.float32)

    def get_total_cap_violation_tensor(self, output_pin_mask, cap_limits):
        return torch.tensor(0.125, dtype=torch.float32)


class TimingObjViolationCacheTest(unittest.TestCase):
    def test_timing_obj_caches_violation_metrics_on_timing_op(self):
        shell = object.__new__(PlaceObj)
        shell.params = SimpleNamespace(
            timing_surrogate_mode="surrogate_only",
            critical_endpoint_pruning_mode="off",
            export_pin_violation_csv=False,
        )
        shell.invoke_timing_count = 0
        shell.inst_pins_mask = torch.tensor([True], dtype=torch.bool)
        shell.output_pin_mask = torch.tensor([True], dtype=torch.bool)
        shell.pin2libpin_flat_ids = torch.tensor([0], dtype=torch.long)
        shell._log_violation_details = lambda timing_op, slew_limits, cap_limits: None
        shell.get_total_leakage_tensor = lambda: torch.tensor(0.5, dtype=torch.float32)

        timing_op = FakeTimingPropagationOp()
        shell.op_collections = SimpleNamespace(
            pin_pos_op=lambda pos: pos,
            steiner_topo_op=lambda pin_pos: (pin_pos, pin_pos),
            elmore_delay_op=lambda *args: (
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
            ),
            timing_propagation_op=timing_op,
        )
        shell.pin_caps_op = lambda inst_libcell_offset, data_collections: (
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )
        shell.data_collections = SimpleNamespace(
            inst_libcell_offset=torch.tensor([0], dtype=torch.long),
            net_flat_topo_sort=torch.tensor([], dtype=torch.long),
            net_flat_topo_sort_start=torch.tensor([0], dtype=torch.long),
            pin_fa=torch.tensor([], dtype=torch.long),
            flat_pin_to_start=torch.tensor([0], dtype=torch.long),
            flat_pin_to=torch.tensor([], dtype=torch.long),
            flat_pin_from=torch.tensor([], dtype=torch.long),
            flat_lib_pin_slew_limit=torch.tensor([1.0], dtype=torch.float32),
            flat_lib_pin_cap_limit=torch.tensor([1.0], dtype=torch.float32),
        )

        wns, tns, _, _ = shell.timing_obj(torch.tensor([0.0], dtype=torch.float32))

        self.assertAlmostEqual(float(wns.item()), -1.0)
        self.assertAlmostEqual(float(tns.item()), -2.0)
        self.assertAlmostEqual(float(timing_op.last_total_slew_violation), 0.25)
        self.assertAlmostEqual(float(timing_op.last_total_cap_violation), 125.0)
        self.assertAlmostEqual(float(timing_op.last_total_leakage), 0.5)
        self.assertGreaterEqual(timing_op.last_runtime_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
