import unittest
import json
from pathlib import Path

from dreamplace.ops.net_subgraph_timing.segment_transfer.calibration import (
    load_forced_z_delta,
    load_static_probe_row,
    summarize_forced_frame_probe_sanity,
    summarize_metadata_liberty_arc_parity,
    summarize_parent_visible_load_gap,
    summarize_top1_static_probe,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    alpha_sample_from_probe_row,
    fit_alpha_for_retained_cap_samples,
    summarize_retained_cap_candidate_errors,
)


def _find_checkout_root(start):
    candidate = None
    for parent in [start, *start.parents]:
        if (parent / "logs/regression").exists():
            candidate = parent
        nested_root = parent / "benchmark-admm-gatesizing"
        if (nested_root / "logs/regression").exists():
            candidate = nested_root
    return candidate


class SegmentTransferCalibrationTest(unittest.TestCase):
    def test_metadata_liberty_arc_parity_compares_master_name_luts(self):
        from types import SimpleNamespace

        metadata = SimpleNamespace(
            flat_libcell_names=["BUF_A", "BUF_B"],
            cell_id_2_arc_id_start=[0, 1, 2],
            f_delay_flat_luts_values=[
                [1.0, 3.0, 5.0, 7.0],
                [10.0, 30.0, 50.0, 70.0],
            ],
            f_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
            f_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
            f_delay_flat_luts_dim=[[2, 2], [2, 2]],
            r_delay_flat_luts_values=[
                [2.0, 4.0, 6.0, 8.0],
                [20.0, 40.0, 60.0, 80.0],
            ],
            r_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
            r_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
            r_delay_flat_luts_dim=[[2, 2], [2, 2]],
            f_trans_flat_luts_values=[
                [0.1, 0.3, 0.5, 0.7],
                [1.0, 3.0, 5.0, 7.0],
            ],
            f_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
            f_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
            f_trans_flat_luts_dim=[[2, 2], [2, 2]],
            r_trans_flat_luts_values=[
                [0.2, 0.4, 0.6, 0.8],
                [2.0, 4.0, 6.0, 8.0],
            ],
            r_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0]],
            r_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0]],
            r_trans_flat_luts_dim=[[2, 2], [2, 2]],
        )

        summary = summarize_metadata_liberty_arc_parity(
            metadata=metadata,
            master_name="BUF_B",
            input_slew=5.0,
            output_load=10.0,
            opensta_arc_samples=[
                {
                    "liberty_gate_delay": 45.0,
                    "liberty_output_slew": 4.5,
                }
            ],
        )

        self.assertEqual(summary["status"], "metadata_liberty_arc_parity")
        self.assertEqual(summary["cell_id"], 1)
        self.assertEqual(len(summary["arc_rows"]), 2)
        self.assertAlmostEqual(
            summary["closest_delay_to_max_opensta"]["metadata_value"],
            40.0,
        )
        self.assertAlmostEqual(
            summary["closest_slew_to_max_opensta"]["metadata_value"],
            4.0,
        )
        self.assertAlmostEqual(
            summary["closest_slew_to_max_opensta"]["abs_error"],
            0.5,
        )

    def test_metadata_liberty_arc_parity_extrapolates_out_of_range_load(self):
        from types import SimpleNamespace

        metadata = SimpleNamespace(
            flat_libcell_names=["BUF_A"],
            cell_id_2_arc_id_start=[0, 1],
            f_delay_flat_luts_values=[[10.0, 20.0, 30.0, 40.0]],
            f_delay_flat_luts_trans_table=[[0.0, 10.0]],
            f_delay_flat_luts_cap_table=[[0.0, 20.0]],
            f_delay_flat_luts_dim=[[2, 2]],
            r_delay_flat_luts_values=[[100.0, 200.0, 300.0, 400.0]],
            r_delay_flat_luts_trans_table=[[0.0, 10.0]],
            r_delay_flat_luts_cap_table=[[0.0, 20.0]],
            r_delay_flat_luts_dim=[[2, 2]],
            f_trans_flat_luts_values=[[1.0, 2.0, 3.0, 4.0]],
            f_trans_flat_luts_trans_table=[[0.0, 10.0]],
            f_trans_flat_luts_cap_table=[[0.0, 20.0]],
            f_trans_flat_luts_dim=[[2, 2]],
            r_trans_flat_luts_values=[[10.0, 20.0, 30.0, 40.0]],
            r_trans_flat_luts_trans_table=[[0.0, 10.0]],
            r_trans_flat_luts_cap_table=[[0.0, 20.0]],
            r_trans_flat_luts_dim=[[2, 2]],
        )

        summary = summarize_metadata_liberty_arc_parity(
            metadata=metadata,
            master_name="BUF_A",
            input_slew=10.0,
            output_load=40.0,
            opensta_arc_samples=[
                {
                    "liberty_gate_delay": 50.0,
                    "liberty_output_slew": 5.0,
                }
            ],
        )

        f_row = next(
            row
            for row in summary["arc_rows"]
            if row["arc_id"] == 0 and row["prefix"] == "f"
        )
        self.assertAlmostEqual(f_row["metadata_delay"], 50.0)
        self.assertAlmostEqual(f_row["metadata_output_slew"], 5.0)

    def test_top1_static_probe_summary_uses_explicit_opensta_inputs(self):
        root = _find_checkout_root(Path(__file__).resolve())
        if root is None:
            self.skipTest("checkout root with logs/regression is not available")
        probe_dir = (
            root
            / "logs/regression/iccad24/buffering_inner_loop"
            / "segment_device_probe_compactsdc_top1_native_20260623"
            / "NV_NVDLA_partition_m/probe"
        )
        if not probe_dir.exists():
            self.skipTest("top1 compact-SDC probe artifact is not available")

        row = load_static_probe_row(
            probe_dir / "segment_device_probe.json",
            segment_id=28067,
        )
        pysta_delta = load_forced_z_delta(
            probe_dir / "segment_force_z_probe.json",
            segment_id=28067,
        )
        summary = summarize_top1_static_probe(
            static_probe_row=row,
            pysta_delta_tns=pysta_delta,
            opensta_delta_tns=9174.470997374941,
            opensta_committed_arc={
                "buffer_master": "HB3xp67_ASAP7_75t_R",
                "input_slew_ps": [110.399, 95.170],
                "output_load_cap": 0.059257,
                "gate_delay_ps": [287.568, 272.230],
                "output_slew_ps": [522.672, 431.197],
                "pin_arrival_delta_ps": [266.087, 244.471],
            },
        )

        self.assertEqual(summary["segment_id"], 28067)
        self.assertEqual(summary["net_name"], "n_24317")
        self.assertAlmostEqual(
            summary["ratios"]["pysta_delta_tns_over_opensta_delta_tns"],
            1.9627961747424612,
            places=5,
        )
        self.assertGreater(
            summary["ratios"]["opensta_gate_delay_over_pysta_static_min"],
            7.0,
        )
        self.assertGreater(
            summary["ratios"]["opensta_output_slew_over_pysta_static_min"],
            30.0,
        )
        self.assertEqual(summary["status"], "static_probe_only")

    def test_current_top1_parent_visible_load_gap_is_explicit(self):
        root = _find_checkout_root(Path(__file__).resolve())
        if root is None:
            self.skipTest("checkout root with logs/regression is not available")
        probe_dir = (
            root
            / "logs/regression/iccad24/buffering_inner_loop"
            / "segment_device_probe_upstream_split_top1_20260623"
            / "NV_NVDLA_partition_m/probe"
        )
        if not probe_dir.exists():
            self.skipTest("current top1 upstream-split probe artifact is not available")

        row = load_static_probe_row(
            probe_dir / "segment_device_probe.json",
            segment_id=27429,
        )
        summary = summarize_parent_visible_load_gap(
            probe_row=row,
            opensta_upstream_net_cap=0.010885200,
            opensta_driver_slew_ps=[207.662569, 176.426860],
            opensta_buffer_input_slew_ps=[209.473731, 178.556670],
        )

        self.assertEqual(summary["segment_id"], 27429)
        self.assertEqual(summary["net_name"], "n_24317")
        self.assertEqual(summary["status"], "parent_visible_load_gap")
        self.assertAlmostEqual(
            summary["loads_pf"]["analytic_parent_visible_load"],
            0.0007313104579225183,
            places=12,
        )
        self.assertAlmostEqual(
            summary["loads_pf"]["required_retained_cap_over_analytic"],
            0.010153889542077482,
            places=12,
        )
        self.assertGreater(
            summary["ratios"]["opensta_over_analytic_parent_visible_load"],
            14.0,
        )
        self.assertGreater(
            summary["ratios"]["required_retained_over_local_upstream_wire_cap"],
            1000.0,
        )

    def test_current_top1_retained_cap_candidate_estimators_are_calibrated(self):
        root = _find_checkout_root(Path(__file__).resolve())
        if root is None:
            self.skipTest("checkout root with logs/regression is not available")
        probe_dir = (
            root
            / "logs/regression/iccad24/buffering_inner_loop"
            / "segment_device_probe_upstream_split_top1_20260623"
            / "NV_NVDLA_partition_m/probe"
        )
        if not probe_dir.exists():
            self.skipTest("current top1 upstream-split probe artifact is not available")

        row = load_static_probe_row(
            probe_dir / "segment_device_probe.json",
            segment_id=27429,
        )
        summary = summarize_retained_cap_candidate_errors(
            probe_row=row,
            opensta_upstream_net_cap=0.010885200,
        )

        self.assertEqual(summary["segment_id"], 27429)
        self.assertEqual(summary["best_candidate"]["name"], "edge_cap_sum_fraction")
        self.assertLess(summary["best_candidate"]["abs_error"], 0.003)
        zero = next(item for item in summary["candidates"] if item["name"] == "zero")
        self.assertGreater(zero["abs_error"], 0.010)

    def test_current_top1_edge_cap_fraction_alpha_fit_is_recorded(self):
        root = _find_checkout_root(Path(__file__).resolve())
        if root is None:
            self.skipTest("checkout root with logs/regression is not available")
        probe_dir = (
            root
            / "logs/regression/iccad24/buffering_inner_loop"
            / "segment_device_probe_upstream_split_top1_20260623"
            / "NV_NVDLA_partition_m/probe"
        )
        if not probe_dir.exists():
            self.skipTest("current top1 upstream-split probe artifact is not available")

        row = load_static_probe_row(
            probe_dir / "segment_device_probe.json",
            segment_id=27429,
        )
        sample = alpha_sample_from_probe_row(
            probe_row=row,
            target_parent_visible_load=0.010885200,
        )
        summary = fit_alpha_for_retained_cap_samples([sample])

        self.assertEqual(summary["sample_count"], 1)
        self.assertAlmostEqual(summary["alpha"], 0.8097938897171508, places=6)
        self.assertLess(summary["rmse"], 1e-12)

    def test_segment_transfer_python_gate_records_forced_frame_but_not_opensta_slew(self):
        root = _find_checkout_root(Path(__file__).resolve())
        if root is None:
            self.skipTest("checkout root with logs/regression is not available")
        probe_path = (
            root
            / "logs/regression/iccad24/buffering_inner_loop"
            / "segment_transfer_python_targetnet_gate_27429_20260624"
            / "NV_NVDLA_partition_m/probe/segment_device_probe_forced.json"
        )
        if not probe_path.exists():
            self.skipTest("segment-transfer-python forced-frame probe artifact is not available")

        with probe_path.open("r", encoding="utf8") as stream:
            artifact = json.load(stream)
        summary = summarize_forced_frame_probe_sanity(
            probe_rows=artifact["rows"],
            segment_id=27429,
            expected_net_name="n_24317",
            expected_z_value=1.0,
            opensta_buffer_input_slew_ps=[209.4737308654579, 178.5566701635801],
            opensta_output_load_pf=0.06437985723955347,
            opensta_gate_delay_ps=[321.6639745203815, 310.2097480184037],
            opensta_output_slew_ps=[569.861327072138, 468.38028087775274],
        )

        self.assertTrue(summary["matches_expected_frame"])
        self.assertEqual(summary["observed_net_names"], ["n_24317"])
        self.assertEqual(summary["observed_z_values"], [1.0])
        self.assertEqual(summary["status"], "forced_frame_sanity_only")
        self.assertGreater(
            summary["ratios"]["min_opensta_input_slew_over_max_pysta_input_slew"],
            8.0,
        )
        self.assertGreater(
            summary["ratios"]["min_opensta_gate_delay_over_max_pysta_delay"],
            2.0,
        )
        self.assertGreater(
            summary["ratios"]["min_opensta_output_slew_over_max_pysta_output_slew"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
