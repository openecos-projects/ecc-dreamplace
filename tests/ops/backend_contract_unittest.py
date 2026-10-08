#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


class BackendContractTest(unittest.TestCase):
    def test_backend_caps_have_stable_shared_keys(self):
        from dreamplace.ops.placeio_common.backend_contract import (
            BackendCaps,
            ecc_backend_caps,
            openroad_backend_caps,
        )

        ieda_caps = BackendCaps.ieda().to_dict()
        openroad_caps = openroad_backend_caps().to_dict()

        self.assertEqual(set(ieda_caps), set(openroad_caps))
        self.assertEqual("ieda", ieda_caps["backend"])
        self.assertEqual("openroad", openroad_caps["backend"])
        self.assertTrue(ieda_caps["supports_sta_pydb"])
        self.assertTrue(openroad_caps["supports_sta_pydb"])
        self.assertFalse(ieda_caps["supports_buffer_commit"])
        self.assertTrue(openroad_caps["supports_buffer_commit"])
        self.assertTrue(ieda_caps["has_diff_sizing_metadata"])
        self.assertTrue(openroad_caps["has_diff_sizing_metadata"])
        self.assertFalse(ieda_caps["has_buffer_optimizer_metadata"])
        self.assertTrue(openroad_caps["has_buffer_optimizer_metadata"])
        self.assertFalse(ieda_caps["has_diff_optimizer_metadata"])
        self.assertTrue(openroad_caps["has_diff_optimizer_metadata"])
        self.assertEqual("diagnostic_exporter", ieda_caps["sta_reference_role"])
        self.assertEqual("golden_opensta", openroad_caps["sta_reference_role"])
        self.assertEqual("unverified", ieda_caps["parasitics_initialization"])
        self.assertEqual("placement", openroad_caps["parasitics_initialization"])
        self.assertEqual("diagnostic_unverified", ieda_caps["sta_state_status"])
        self.assertEqual("validated_golden", openroad_caps["sta_state_status"])

        ecc_caps = ecc_backend_caps().to_dict()
        self.assertEqual(set(ieda_caps), set(ecc_caps))
        self.assertEqual("ecc", ecc_caps["backend"])
        self.assertFalse(ecc_caps["has_sta"])
        self.assertFalse(ecc_caps["supports_sta_pydb"])
        self.assertTrue(ecc_caps["supports_apply_placement"])
        self.assertFalse(ecc_caps["supports_apply_sizing"])
        self.assertFalse(ecc_caps["supports_buffer_commit"])
        self.assertEqual("ecc_idb", ecc_caps["coordinate_source"])

    def test_parasitic_state_audit_recommends_ieda_calibration_without_spef(self):
        from docs.backend_governance.audit_backend_parasitic_state import (
            audit_backend_parasitic_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(os.path.join(workspace, "config"))
            with open(os.path.join(workspace, "config", "workspace.json"), "w", encoding="utf-8") as f:
                json.dump({"workspace": {"design": "FFT"}}, f)
            with open(os.path.join(workspace, "config", "design_path.json"), "w", encoding="utf-8") as f:
                json.dump({"spef_path": ""}, f)
            rc_tcl = os.path.join(tmpdir, "setRC.tcl")
            with open(rc_tcl, "w", encoding="utf-8") as f:
                f.write("set_layer_rc -layer M3 -resistance 1.0 -capacitance 2.0\n")
                f.write("set_wire_rc -signal -layer M3\n")

            report = audit_backend_parasitic_state(workspace, rc_tcl)

        self.assertEqual("openroad_opensta", report["comparison_policy"]["golden_backend"])
        self.assertEqual(
            "diagnostic_until_parasitics_calibrated",
            report["comparison_policy"]["ieda_role"],
        )
        self.assertTrue(report["openroad"]["requires_estimate_parasitics_placement"])
        self.assertEqual(0, report["ieda"]["discovered_spef_count"])
        self.assertEqual("unverified", report["ieda"]["current_pydb_parasitics_initialization"])
        self.assertEqual(
            "calibrate_ieda_placement_parasitic_estimator_against_opensta",
            report["recommended_next_step"],
        )

    def test_parasitic_state_audit_prefers_same_spef_when_available(self):
        from docs.backend_governance.audit_backend_parasitic_state import (
            audit_backend_parasitic_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "workspace")
            os.makedirs(os.path.join(workspace, "config"))
            spef = os.path.join(workspace, "design.spef")
            with open(spef, "w", encoding="utf-8") as f:
                f.write("*SPEF \"IEEE 1481-1998\"\n")
            with open(os.path.join(workspace, "config", "design_path.json"), "w", encoding="utf-8") as f:
                json.dump({"spef_path": spef}, f)

            report = audit_backend_parasitic_state(workspace, "")

        self.assertTrue(report["ieda"]["workspace_spef_exists"])
        self.assertEqual(1, report["ieda"]["discovered_spef_count"])
        self.assertEqual(
            "run_same_spef_ieda_openroad_endpoint_compare",
            report["recommended_next_step"],
        )

    def test_parity_audit_compares_counts_names_and_connectivity(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            num_nodes=2,
        )
        pydb_b = types.SimpleNamespace(
            node_names=[b"u0", b"u1"],
            pin_names=[b"u0/A", b"u1/Y"],
            net_names=[b"n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            num_nodes=2,
        )

        report = audit_pydb_parity(pydb_a, pydb_b, lhs_label="ieda", rhs_label="openroad")

        self.assertTrue(report["passed"])
        self.assertTrue(report["core_topology_passed"])
        self.assertTrue(report["metadata_passed"])
        self.assertEqual([], report["mismatches"])
        self.assertEqual("ieda", report["lhs_label"])
        self.assertEqual("openroad", report["rhs_label"])
        self.assertEqual(2, report["lhs"]["counts"]["nodes"])
        self.assertEqual(report["lhs"]["hashes"], report["rhs"]["hashes"])

    def test_parity_audit_ignores_backend_local_ordering(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
        )
        pydb_b = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[1, 0],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
        )

        report = audit_pydb_parity(pydb_a, pydb_b)

        self.assertTrue(report["passed"])
        self.assertEqual([], report["mismatches"])

    def test_parity_audit_reports_mismatched_connectivity(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0", "n1"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 1, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 1],
        )
        pydb_b = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0", "n1"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
        )

        report = audit_pydb_parity(pydb_a, pydb_b)

        self.assertFalse(report["passed"])
        self.assertFalse(report["core_topology_passed"])
        self.assertIn("net_connectivity_hash", report["mismatches"])

    def test_parity_audit_includes_name_and_optional_metadata_diagnostics(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u_extra"],
            pin_names=["u0/A"],
            net_names=["n0"],
            flat_net2pin_map=[0],
            flat_net2pin_start_map=[0, 1],
            pin2node_map=[0],
            pin2net_map=[0],
        )
        pydb_b = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A"],
            net_names=["n0"],
            flat_net2pin_map=[0],
            flat_net2pin_start_map=[0, 1],
            pin2node_map=[0],
            pin2net_map=[0],
            node_master_names=["BUF_X1", "INV_X1"],
        )

        report = audit_pydb_parity(pydb_a, pydb_b)
        node_delta = report["diagnostics"]["set_deltas"]["node_names"]

        self.assertEqual(1, node_delta["only_lhs_count"])
        self.assertEqual(["u_extra"], node_delta["only_lhs_sample"])
        self.assertEqual(["u1"], node_delta["only_rhs_sample"])
        self.assertEqual(
            {
                "lhs": False,
                "rhs": True,
            },
            report["diagnostics"]["optional_sequence_presence"]["node_master_names"],
        )

    def test_parity_audit_ignores_unreferenced_backend_artifact_nodes(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u1", "VDD"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            node_size_x=[10, 20, 1],
            node_size_y=[30, 40, 1],
            num_nodes=3,
        )
        pydb_b = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            node_size_x=[10, 20],
            node_size_y=[30, 40],
            num_nodes=2,
        )

        report = audit_pydb_parity(pydb_a, pydb_b)

        self.assertTrue(report["passed"])
        self.assertEqual(2, report["lhs"]["counts"]["placement_relevant_nodes"])
        self.assertEqual(2, report["rhs"]["counts"]["placement_relevant_nodes"])
        self.assertEqual(
            1,
            report["diagnostics"]["set_deltas"]["node_names"]["only_lhs_count"],
        )
        self.assertEqual(
            0,
            report["diagnostics"]["placement_relevant_set_deltas"]["node_names"][
                "only_lhs_count"
            ],
        )

    def test_parity_audit_keeps_referenced_extra_nodes_as_mismatch(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        pydb_a = types.SimpleNamespace(
            node_names=["u0", "u1", "extra_driver"],
            pin_names=["u0/A", "u1/Y", "extra_driver/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1, 2],
            flat_net2pin_start_map=[0, 3],
            pin2node_map=[0, 1, 2],
            pin2net_map=[0, 0, 0],
        )
        pydb_b = types.SimpleNamespace(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y", "extra_driver/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1, 2],
            flat_net2pin_start_map=[0, 3],
            pin2node_map=[0, 1, 1],
            pin2net_map=[0, 0, 0],
        )

        report = audit_pydb_parity(pydb_a, pydb_b)

        self.assertFalse(report["passed"])
        self.assertIn("count_placement_relevant_nodes", report["mismatches"])

    def test_parity_audit_ignores_terminal_size_encoding_but_keeps_cell_size_mismatch(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        base = dict(
            node_names=["u0", "out_port"],
            pin_names=["u0/Y", "out_port"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
        )
        pydb_a = types.SimpleNamespace(
            **dict(base, node_size_x=[10, 1], node_size_y=[20, 1])
        )
        pydb_b = types.SimpleNamespace(
            **dict(base, node_size_x=[10, 0], node_size_y=[20, 0])
        )
        pydb_c = types.SimpleNamespace(
            **dict(base, node_size_x=[11, 0], node_size_y=[20, 0])
        )

        terminal_only_report = audit_pydb_parity(pydb_a, pydb_b)
        cell_size_report = audit_pydb_parity(pydb_a, pydb_c)

        self.assertTrue(terminal_only_report["passed"])
        self.assertEqual(1, terminal_only_report["lhs"]["counts"]["size_comparable_nodes"])
        self.assertFalse(cell_size_report["passed"])
        self.assertIn("node_size_x_hash", cell_size_report["mismatches"])

    def test_parity_audit_compares_optional_geometry_and_node_sizes(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        base = dict(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u1/Y"],
            net_names=["n0"],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            xl=0,
            yl=0,
            xh=1000,
            yh=2000,
            site_width=2,
            row_height=4,
            node_size_x=[10, 20],
            node_size_y=[30, 40],
        )
        pydb_a = types.SimpleNamespace(**base)
        pydb_b = types.SimpleNamespace(**base)
        pydb_c = types.SimpleNamespace(**dict(base, node_size_x=[10, 21]))

        passing_report = audit_pydb_parity(pydb_a, pydb_b)
        failing_report = audit_pydb_parity(pydb_a, pydb_c)

        self.assertTrue(passing_report["passed"])
        self.assertEqual(
            {"xl": 0.0, "yl": 0.0, "xh": 1000.0, "yh": 2000.0},
            passing_report["lhs"]["geometry"]["die_bbox"],
        )
        self.assertFalse(failing_report["passed"])
        self.assertIn("node_size_x_hash", failing_report["mismatches"])

    def test_parity_snapshot_accepts_numpy_array_fields(self):
        from dreamplace.ops.placeio_common.parity_audit import build_pydb_parity_snapshot

        pydb = types.SimpleNamespace(
            node_names=np.array([b"u0"]),
            pin_names=np.array([b"u0/A"]),
            net_names=np.array([b"n0"]),
            flat_net2pin_map=np.array([0]),
            flat_net2pin_start_map=np.array([0, 1]),
            pin2node_map=np.array([0]),
            pin2net_map=np.array([0]),
            node_size_x=np.array([10.0]),
            node_size_y=np.array([20.0]),
        )

        snapshot = build_pydb_parity_snapshot(pydb)

        self.assertEqual(1, snapshot["counts"]["nodes"])
        self.assertIsNotNone(snapshot["hashes"]["net_connectivity"])

    def test_parity_audit_reports_timing_sanity(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        base = dict(
            node_names=["u0", "u1"],
            pin_names=["u0/A", "u0/Y", "u1/D"],
            net_names=["n0"],
            flat_net2pin_map=[1, 2],
            flat_net2pin_start_map=[0, 2],
            pin2node_map=[0, 0, 1],
            pin2net_map=[0, 0, 0],
            node_size_x=[10, 20],
            node_size_y=[30, 40],
            start_points=[1],
            end_points=[2],
            endpoints_rRAT=[1.0],
            endpoints_fRAT=[1.0],
            endpoints_constraint_arcs=[0],
            endpoints_timing_check_arcs=[
                [0, 2, 0, 0, 1, 1, 1, 0],
                [0, 2, 0, -1, 1, 0, 3, -1],
                [0, 2, 0, -1, 1, 0, 4, -1],
            ],
        )
        passing_report = audit_pydb_parity(
            types.SimpleNamespace(**base),
            types.SimpleNamespace(**base),
        )
        failing_report = audit_pydb_parity(
            types.SimpleNamespace(**base),
            types.SimpleNamespace(
                **dict(
                    base,
                    end_points=[1],
                    endpoints_rRAT=[],
                    endpoints_fRAT=[1.0],
                )
            ),
        )

        self.assertTrue(passing_report["timing_passed"])
        self.assertEqual([], passing_report["timing_mismatches"])
        self.assertEqual(1, passing_report["lhs"]["timing"]["counts"]["start_points"])
        self.assertEqual(1, passing_report["lhs"]["timing"]["counts"]["end_points"])
        self.assertEqual(3, passing_report["lhs"]["timing"]["counts"]["endpoints_timing_check_arcs"])
        self.assertEqual(
            {"1": 1, "3": 1, "4": 1},
            passing_report["lhs"]["timing"]["metadata"]["timing_check_class_histogram"],
        )
        self.assertFalse(failing_report["passed"])
        self.assertTrue(failing_report["core_topology_passed"])
        self.assertFalse(failing_report["timing_passed"])
        self.assertIn("endpoint_rat_length_match", failing_report["timing_mismatches"])
        self.assertIn("end_point_pin_names_hash", failing_report["timing_mismatches"])

    def test_parity_audit_compares_optional_buffer_and_libcell_metadata(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        base = dict(
            node_names=["buf0"],
            pin_names=["buf0/A"],
            net_names=["n0"],
            flat_net2pin_map=[0],
            flat_net2pin_start_map=[0, 1],
            pin2node_map=[0],
            pin2net_map=[0],
            buffer_main_type_index=3,
            buffer_main_type_status="ok",
            buffer_main_type_candidate_indices=[3],
            node_is_buffer=[True],
            node_master_names=["BUF_X4"],
            inst_libcell_offset=[2],
            flat_libcell_names=["BUF_X1", "BUF_X2", "BUF_X4"],
            flat_libcell_info=[
                ["BUF_X1", 3, 1, 0],
                ["BUF_X2", 3, 2, 0],
                ["BUF_X4", 3, 4, 0],
            ],
            flat_libcell_width=[10, 20, 40],
            flat_libcell_height=[5, 5, 5],
            flat_libcell_leakage=[1.0, 2.0, 4.0],
            flat_libcell_main_id2size_vt_limit=[["BUF", 0, 2]],
        )
        pydb_a = types.SimpleNamespace(**base)
        pydb_b = types.SimpleNamespace(**base)
        pydb_c = types.SimpleNamespace(**dict(base, buffer_main_type_index=4))

        passing_report = audit_pydb_parity(pydb_a, pydb_b)
        failing_report = audit_pydb_parity(pydb_a, pydb_c)

        self.assertTrue(passing_report["passed"])
        self.assertEqual(3, passing_report["lhs"]["buffer"]["main_type_index"])
        self.assertEqual("ok", passing_report["lhs"]["buffer"]["main_type_status"])
        self.assertFalse(failing_report["passed"])
        self.assertTrue(failing_report["core_topology_passed"])
        self.assertFalse(failing_report["metadata_passed"])
        self.assertIn("buffer_metadata", failing_report["mismatches"])

    def test_parity_audit_reports_core_pass_with_missing_optional_metadata(self):
        from dreamplace.ops.placeio_common.parity_audit import audit_pydb_parity

        base = dict(
            node_names=["buf0"],
            pin_names=["buf0/A"],
            net_names=["n0"],
            flat_net2pin_map=[0],
            flat_net2pin_start_map=[0, 1],
            pin2node_map=[0],
            pin2net_map=[0],
            node_size_x=[10],
            node_size_y=[20],
        )
        pydb_ieda = types.SimpleNamespace(**base)
        pydb_openroad = types.SimpleNamespace(
            **dict(
                base,
                buffer_main_type_index=3,
                buffer_main_type_status="ok",
                buffer_main_type_candidate_indices=[3],
                node_is_buffer=[True],
            )
        )

        report = audit_pydb_parity(pydb_ieda, pydb_openroad)

        self.assertFalse(report["passed"])
        self.assertTrue(report["core_topology_passed"])
        self.assertFalse(report["metadata_passed"])
        self.assertEqual([], report["core_mismatches"])
        self.assertIn("buffer_metadata", report["metadata_mismatches"])
        self.assertIn("node_is_buffer_hash", report["metadata_mismatches"])


if __name__ == "__main__":
    unittest.main()
