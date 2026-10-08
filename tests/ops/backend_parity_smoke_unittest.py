#!/usr/bin/env python3

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Current smoke scope:
# `backend_contract`, `backend_parity_smoke`, `placeio_ieda`,
# `timing_propagation`, and the minimal flow-layer tests pass together under
# the PlaceOPT Python environment. This verifies backend governance and data
# export contracts only. It does not claim full OpenSTA report_tns parity,
# because remaining no-op TNS differences are driven by parasitic/RC
# initialization policy.


def make_pydb(pin_order, net_starts=None, pin2net=None):
    return types.SimpleNamespace(
        node_names=["u0", "u1"],
        pin_names=["u0/A", "u1/Y"],
        net_names=["n0"],
        flat_net2pin_map=pin_order,
        flat_net2pin_start_map=net_starts or [0, 2],
        pin2node_map=[0, 1],
        pin2net_map=pin2net or [0, 0],
        num_nodes=2,
    )


def make_openroad_pydb_with_buffer_metadata():
    pydb = make_pydb([0, 1])
    pydb.buffer_main_type_index = 3
    pydb.buffer_main_type_status = "ok"
    pydb.buffer_main_type_candidate_indices = [3]
    pydb.node_is_buffer = [False, True]
    return pydb


class BackendParitySmokeTest(unittest.TestCase):
    def test_run_smoke_writes_passing_report(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(),
                workspace="/tmp/ieda-workspace",
                output=str(output),
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_pydb([0, 1]),
            )

            report = json.loads(output.read_text())

        self.assertEqual(0, exit_code)
        self.assertTrue(report["passed"])
        self.assertEqual("ieda", report["lhs_label"])
        self.assertEqual("openroad", report["rhs_label"])
        self.assertEqual([], report["mismatches"])

    def test_run_smoke_returns_nonzero_for_failed_parity(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(),
                workspace="/tmp/ieda-workspace",
                output=str(output),
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: types.SimpleNamespace(
                    node_names=["u0", "u1"],
                    pin_names=["u0/A", "u1/Y"],
                    net_names=["n0", "n1"],
                    flat_net2pin_map=[0, 1],
                    flat_net2pin_start_map=[0, 2, 2],
                    pin2node_map=[0, 1],
                    pin2net_map=[0, 0],
                    num_nodes=2,
                ),
            )

            report = json.loads(output.read_text())

        self.assertEqual(1, exit_code)
        self.assertFalse(report["passed"])
        self.assertIn("net_connectivity_hash", report["mismatches"])

    def test_core_topology_profile_passes_when_only_metadata_differs(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(),
                workspace="/tmp/ieda-workspace",
                output=str(output),
                profile="core_topology",
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_openroad_pydb_with_buffer_metadata(),
            )

            report = json.loads(output.read_text())

        self.assertEqual(0, exit_code)
        self.assertFalse(report["passed"])
        self.assertTrue(report["core_topology_passed"])
        self.assertFalse(report["metadata_passed"])

    def test_default_profile_is_core_topology_for_governance_smoke(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(),
                workspace="/tmp/ieda-workspace",
                output=str(output),
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_openroad_pydb_with_buffer_metadata(),
            )

            report = json.loads(output.read_text())

        self.assertEqual(0, exit_code)
        self.assertFalse(report["passed"])
        self.assertTrue(report["core_topology_passed"])
        self.assertFalse(report["metadata_passed"])

    def test_report_records_parasitic_state_audit_without_changing_core_topology_exit(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "config").mkdir(parents=True)
            (workspace / "config" / "design_path.json").write_text(
                json.dumps({"spef_path": ""})
            )
            rc_tcl = Path(tmpdir) / "setRC.tcl"
            rc_tcl.write_text(
                "set_layer_rc -layer M3 -resistance 1.0 -capacitance 2.0\n"
                "set_wire_rc -signal -layer M3\n"
            )
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(design_inputs={"rc_tcl": str(rc_tcl)}),
                workspace=str(workspace),
                output=str(output),
                profile="core_topology",
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_pydb([0, 1]),
            )

            report = json.loads(output.read_text())

        self.assertEqual(0, exit_code)
        self.assertTrue(report["core_topology_passed"])
        self.assertFalse(report["parasitic_state_passed"])
        self.assertEqual(
            "calibrate_ieda_placement_parasitic_estimator_against_opensta",
            report["parasitic_state"]["recommended_next_step"],
        )
        self.assertTrue(report["parasitic_state"]["openroad"]["rc_tcl_exists"])
        self.assertEqual(0, report["parasitic_state"]["ieda"]["discovered_spef_count"])
        self.assertEqual(
            "openroad_opensta",
            report["parasitic_state"]["comparison_policy"]["golden_backend"],
        )

    def test_full_profile_fails_when_parasitic_state_is_unverified_even_if_pydb_matches(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "config").mkdir(parents=True)
            (workspace / "config" / "design_path.json").write_text(
                json.dumps({"spef_path": ""})
            )
            rc_tcl = Path(tmpdir) / "setRC.tcl"
            rc_tcl.write_text(
                "set_layer_rc -layer M3 -resistance 1.0 -capacitance 2.0\n"
                "set_wire_rc -signal -layer M3\n"
            )
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(design_inputs={"rc_tcl": str(rc_tcl)}),
                workspace=str(workspace),
                output=str(output),
                profile="full",
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_pydb([0, 1]),
            )

            report = json.loads(output.read_text())

        self.assertEqual(1, exit_code)
        self.assertTrue(report["passed"])
        self.assertFalse(report["parasitic_state_passed"])
        self.assertFalse(report["full_sta_parity_passed"])
        self.assertEqual([], report["mismatches"])

    def test_full_profile_fails_when_only_metadata_differs(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "parity.json"
            exit_code = backend_parity_smoke.run_backend_parity_smoke(
                params=types.SimpleNamespace(),
                workspace="/tmp/ieda-workspace",
                output=str(output),
                profile="full",
                ieda_reader=lambda params, workspace: make_pydb([0, 1]),
                openroad_reader=lambda params: make_openroad_pydb_with_buffer_metadata(),
            )

            report = json.loads(output.read_text())

        self.assertEqual(1, exit_code)
        self.assertFalse(report["passed"])
        self.assertFalse(report["full_sta_parity_passed"])
        self.assertTrue(report["core_topology_passed"])
        self.assertFalse(report["metadata_passed"])

    def test_main_defaults_to_core_topology_profile(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = Path(tmpdir) / "params.json"
            output_path = Path(tmpdir) / "report.json"
            params_path.write_text(
                json.dumps(
                    {
                        "design_inputs": {"def": "design.def"},
                        "with_sta": True,
                    }
                )
            )

            captured = {}

            def fake_run(params, workspace, output, profile="full"):
                captured["params"] = params
                captured["workspace"] = workspace
                captured["output"] = output
                captured["profile"] = profile
                return 7

            with patch.object(
                backend_parity_smoke,
                "run_backend_parity_smoke",
                side_effect=fake_run,
            ):
                exit_code = backend_parity_smoke.main(
                    [
                        "--params",
                        str(params_path),
                        "--workspace",
                        "/tmp/ieda-workspace",
                        "--output",
                        str(output_path),
                    ]
                )

        self.assertEqual(7, exit_code)
        self.assertEqual("/tmp/ieda-workspace", captured["workspace"])
        self.assertEqual(str(output_path), captured["output"])
        self.assertTrue(captured["params"].with_sta)
        self.assertEqual({"def": "design.def"}, captured["params"].design_inputs)
        self.assertEqual("core_topology", captured["profile"])

    def test_main_accepts_explicit_full_profile(self):
        from dreamplace.ops.placeio_common import backend_parity_smoke

        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = Path(tmpdir) / "params.json"
            output_path = Path(tmpdir) / "report.json"
            params_path.write_text(json.dumps({"design_inputs": {"def": "design.def"}}))

            captured = {}

            def fake_run(params, workspace, output, profile="core_topology"):
                captured["profile"] = profile
                return 0

            with patch.object(
                backend_parity_smoke,
                "run_backend_parity_smoke",
                side_effect=fake_run,
            ):
                exit_code = backend_parity_smoke.main(
                    [
                        "--params",
                        str(params_path),
                        "--workspace",
                        "/tmp/ieda-workspace",
                        "--output",
                        str(output_path),
                        "--profile",
                        "full",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual("full", captured["profile"])


if __name__ == "__main__":
    unittest.main()
