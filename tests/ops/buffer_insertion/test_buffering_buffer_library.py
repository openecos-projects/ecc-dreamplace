import unittest
from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.buffer_library import (
    build_buffer_library_from_metadata,
    lookup_buffer_delay_from_metadata,
    lookup_buffer_delay_for_bsu,
    lookup_buffer_delay_for_bsu_with_status,
    lookup_buffer_transition_from_metadata,
    lookup_buffer_transition_for_bsu,
    lookup_lut_value,
    lookup_lut_value_with_status,
)


def _metadata():
    return SimpleNamespace(
        flat_libcell_info=[
            [0, 7, 1.0, 0],
            [1, 7, 2.0, 0],
            [2, 9, 1.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2", "NAND_X1"],
        cell_id_2_libpin_id_start=[0, 2, 4, 6],
        flat_lib_pin_cap=[0.02, 0.0, 0.04, 0.0, 0.08, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2, 3],
        f_delay_flat_luts_values=[[1.0, 1.2], [2.0, 2.2], [5.0]],
        f_delay_flat_luts_trans_table=[[], [], []],
        f_delay_flat_luts_cap_table=[[], [], []],
        f_delay_flat_luts_dim=[[0, 0], [0, 0], [0, 0]],
        r_delay_flat_luts_values=[[1.4, 1.6], [2.4, 2.6], [5.5]],
        r_delay_flat_luts_trans_table=[[], [], []],
        r_delay_flat_luts_cap_table=[[], [], []],
        r_delay_flat_luts_dim=[[0, 0], [0, 0], [0, 0]],
    )


class DuBufferLibraryTest(unittest.TestCase):
    def test_builds_bsu_library_from_contract_legal_cells(self):
        contract_artifact = {
            "status": "ok",
            "legal_cell_ids": [0, 1],
            "legal_master_names": ["BUF_X1", "BUF_X2"],
            "legal_size_values": [1.0, 2.0],
        }

        library, summary = build_buffer_library_from_metadata(
            _metadata(),
            contract_artifact,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["source"], "contract_metadata_proxy")
        self.assertEqual(summary["entry_count"], 2)
        self.assertEqual(
            summary["delay_source_counts"],
            {"contract_metadata_proxy_average": 2},
        )
        self.assertEqual(library[0]["cell_id"], 0)
        self.assertEqual(library[0]["master_name"], "BUF_X1")
        self.assertAlmostEqual(library[0]["input_cap"], 0.02)
        self.assertAlmostEqual(library[0]["delay"], 1.3)
        self.assertEqual(library[0]["delay_source"], "contract_metadata_proxy_average")
        self.assertEqual(library[1]["cell_id"], 1)
        self.assertEqual(library[1]["master_name"], "BUF_X2")
        self.assertAlmostEqual(library[1]["input_cap"], 0.04)
        self.assertAlmostEqual(library[1]["delay"], 2.3)

    def test_rejects_missing_contract(self):
        with self.assertRaises(ValueError):
            build_buffer_library_from_metadata(_metadata(), {"status": "unsupported"})

    def test_rejects_missing_input_cap(self):
        metadata = _metadata()
        metadata.flat_lib_pin_cap = [0.0] * 6

        with self.assertRaises(ValueError):
            build_buffer_library_from_metadata(
                metadata,
                {
                    "status": "ok",
                    "legal_cell_ids": [0],
                    "legal_master_names": ["BUF_X1"],
                    "legal_size_values": [1.0],
                },
            )

    def test_bilinear_lut_lookup_uses_slew_and_cap_axes(self):
        value = lookup_lut_value(
            values=[1.0, 3.0, 5.0, 7.0],
            trans_axis=[0.0, 10.0],
            cap_axis=[0.0, 20.0],
            dim=[2, 2],
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertAlmostEqual(value, 4.0)

    def test_lut_lookup_clamps_to_axis_boundaries(self):
        low_corner = lookup_lut_value(
            values=[1.0, 3.0, 5.0, 7.0],
            trans_axis=[0.0, 10.0],
            cap_axis=[0.0, 20.0],
            dim=[2, 2],
            input_slew=-5.0,
            output_cap=-1.0,
        )
        high_corner = lookup_lut_value(
            values=[1.0, 3.0, 5.0, 7.0],
            trans_axis=[0.0, 10.0],
            cap_axis=[0.0, 20.0],
            dim=[2, 2],
            input_slew=50.0,
            output_cap=100.0,
        )

        self.assertAlmostEqual(low_corner, 1.0)
        self.assertAlmostEqual(high_corner, 7.0)

    def test_lut_lookup_status_reports_axis_clamps(self):
        value, status = lookup_lut_value_with_status(
            values=[1.0, 3.0, 5.0, 7.0],
            trans_axis=[0.0, 10.0],
            cap_axis=[0.0, 20.0],
            dim=[2, 2],
            input_slew=50.0,
            output_cap=-1.0,
        )

        self.assertAlmostEqual(value, 5.0)
        self.assertTrue(status["input_slew_clamped"])
        self.assertTrue(status["output_cap_clamped"])
        self.assertEqual(status["input_slew_clamp"], "high")
        self.assertEqual(status["output_cap_clamp"], "low")
        self.assertEqual(status["input_slew_axis_min"], 0.0)
        self.assertEqual(status["input_slew_axis_max"], 10.0)
        self.assertEqual(status["output_cap_axis_min"], 0.0)
        self.assertEqual(status["output_cap_axis_max"], 20.0)

    def test_lookup_buffer_delay_interpolates_rise_and_fall_luts(self):
        metadata = _metadata()
        metadata.f_delay_flat_luts_values = [[1.0, 3.0, 5.0, 7.0], [0.0], [0.0]]
        metadata.f_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.f_delay_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.f_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]
        metadata.r_delay_flat_luts_values = [[2.0, 4.0, 6.0, 8.0], [0.0], [0.0]]
        metadata.r_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.r_delay_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.r_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]

        delay, source = lookup_buffer_delay_from_metadata(
            metadata,
            cell_id=0,
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertEqual(source, "lut_bilinear_interpolation")
        self.assertAlmostEqual(delay, 4.5)

    def test_lookup_buffer_delay_for_bsu_uses_contract_legal_index(self):
        metadata = _metadata()
        metadata.f_delay_flat_luts_values = [[0.0], [1.0, 3.0, 5.0, 7.0], [0.0]]
        metadata.f_delay_flat_luts_trans_table = [[], [0.0, 10.0], []]
        metadata.f_delay_flat_luts_cap_table = [[], [0.0, 20.0], []]
        metadata.f_delay_flat_luts_dim = [[0, 0], [2, 2], [0, 0]]
        metadata.r_delay_flat_luts_values = [[0.0], [2.0, 4.0, 6.0, 8.0], [0.0]]
        metadata.r_delay_flat_luts_trans_table = [[], [0.0, 10.0], []]
        metadata.r_delay_flat_luts_cap_table = [[], [0.0, 20.0], []]
        metadata.r_delay_flat_luts_dim = [[0, 0], [2, 2], [0, 0]]

        delay, source = lookup_buffer_delay_for_bsu(
            metadata,
            {
                "status": "ok",
                "legal_cell_ids": [0, 1],
                "legal_master_names": ["BUF_X1", "BUF_X2"],
                "legal_size_values": [1.0, 2.0],
            },
            bsu=1,
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertEqual(source, "lut_bilinear_interpolation")
        self.assertAlmostEqual(delay, 4.5)

    def test_lookup_buffer_delay_for_bsu_rejects_missing_2d_lut(self):
        with self.assertRaises(ValueError):
            lookup_buffer_delay_for_bsu(
                _metadata(),
                {
                    "status": "ok",
                    "legal_cell_ids": [0],
                    "legal_master_names": ["BUF_X1"],
                    "legal_size_values": [1.0],
                },
                bsu=0,
                input_slew=5.0,
                output_cap=10.0,
            )

    def test_lookup_buffer_transition_interpolates_rise_and_fall_luts(self):
        metadata = _metadata()
        metadata.f_trans_flat_luts_values = [[0.1, 0.3, 0.5, 0.7], [0.0], [0.0]]
        metadata.f_trans_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.f_trans_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.f_trans_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]
        metadata.r_trans_flat_luts_values = [[0.2, 0.4, 0.6, 0.8], [0.0], [0.0]]
        metadata.r_trans_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.r_trans_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.r_trans_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]

        slew, source = lookup_buffer_transition_from_metadata(
            metadata,
            cell_id=0,
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertEqual(source, "lut_bilinear_interpolation")
        self.assertAlmostEqual(slew, 0.45)

    def test_lookup_buffer_transition_for_bsu_uses_contract_legal_index(self):
        metadata = _metadata()
        metadata.f_trans_flat_luts_values = [[0.0], [0.1, 0.3, 0.5, 0.7], [0.0]]
        metadata.f_trans_flat_luts_trans_table = [[], [0.0, 10.0], []]
        metadata.f_trans_flat_luts_cap_table = [[], [0.0, 20.0], []]
        metadata.f_trans_flat_luts_dim = [[0, 0], [2, 2], [0, 0]]
        metadata.r_trans_flat_luts_values = [[0.0], [0.2, 0.4, 0.6, 0.8], [0.0]]
        metadata.r_trans_flat_luts_trans_table = [[], [0.0, 10.0], []]
        metadata.r_trans_flat_luts_cap_table = [[], [0.0, 20.0], []]
        metadata.r_trans_flat_luts_dim = [[0, 0], [2, 2], [0, 0]]

        slew, source = lookup_buffer_transition_for_bsu(
            metadata,
            {
                "status": "ok",
                "legal_cell_ids": [0, 1],
                "legal_master_names": ["BUF_X1", "BUF_X2"],
                "legal_size_values": [1.0, 2.0],
            },
            bsu=1,
            input_slew=5.0,
            output_cap=10.0,
        )

        self.assertEqual(source, "lut_bilinear_interpolation")
        self.assertAlmostEqual(slew, 0.45)

    def test_lookup_buffer_delay_for_bsu_with_status_reports_candidate_clamp(self):
        metadata = _metadata()
        metadata.f_delay_flat_luts_values = [[1.0, 3.0, 5.0, 7.0], [0.0], [0.0]]
        metadata.f_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.f_delay_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.f_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]
        metadata.r_delay_flat_luts_values = [[2.0, 4.0, 6.0, 8.0], [0.0], [0.0]]
        metadata.r_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.r_delay_flat_luts_cap_table = [[0.0, 20.0], [], []]
        metadata.r_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]

        delay, source, status = lookup_buffer_delay_for_bsu_with_status(
            metadata,
            {
                "status": "ok",
                "legal_cell_ids": [0],
                "legal_master_names": ["BUF_X1"],
                "legal_size_values": [1.0],
            },
            bsu=0,
            input_slew=50.0,
            output_cap=10.0,
        )

        self.assertEqual(source, "lut_bilinear_interpolation")
        self.assertAlmostEqual(delay, 6.5)
        self.assertTrue(status["input_slew_clamped"])
        self.assertFalse(status["output_cap_clamped"])
        self.assertEqual(status["input_slew_clamp"], "high")
        self.assertEqual(status["output_cap_clamp"], "none")

    def test_build_summary_counts_lut_delay_sources(self):
        metadata = _metadata()
        metadata.f_delay_flat_luts_values = [[1.0, 3.0, 5.0, 7.0], [0.0], [0.0]]
        metadata.f_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.f_delay_flat_luts_cap_table = [[0.0, 0.1], [], []]
        metadata.f_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]
        metadata.r_delay_flat_luts_values = [[2.0, 4.0, 6.0, 8.0], [0.0], [0.0]]
        metadata.r_delay_flat_luts_trans_table = [[0.0, 10.0], [], []]
        metadata.r_delay_flat_luts_cap_table = [[0.0, 0.1], [], []]
        metadata.r_delay_flat_luts_dim = [[2, 2], [0, 0], [0, 0]]

        library, summary = build_buffer_library_from_metadata(
            metadata,
            {
                "status": "ok",
                "legal_cell_ids": [0, 1],
                "legal_master_names": ["BUF_X1", "BUF_X2"],
                "legal_size_values": [1.0, 2.0],
            },
        )

        self.assertEqual(library[0]["delay_source"], "lut_bilinear_interpolation")
        self.assertEqual(library[1]["delay_source"], "contract_metadata_proxy_average")
        self.assertEqual(
            summary["delay_source_counts"],
            {
                "lut_bilinear_interpolation": 1,
                "contract_metadata_proxy_average": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
