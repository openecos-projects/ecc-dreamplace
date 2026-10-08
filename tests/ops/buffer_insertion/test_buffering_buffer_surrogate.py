import unittest
from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.buffer_surrogate import build_buffer_surrogate


def _lut_metadata():
    return SimpleNamespace(
        cell_id_2_libpin_id_start=[0, 2],
        flat_lib_pin_cap=[0.01, 0.0],
        cell_id_2_arc_id_start=[0, 1],
        flat_libarc_info=[[0, 1]],
        f_delay_flat_luts_values=[[1.0, 3.0, 5.0, 7.0]],
        f_delay_flat_luts_trans_table=[[0.0, 10.0]],
        f_delay_flat_luts_cap_table=[[0.0, 20.0]],
        f_delay_flat_luts_dim=[[2, 2]],
        r_delay_flat_luts_values=[[2.0, 4.0, 6.0, 8.0]],
        r_delay_flat_luts_trans_table=[[0.0, 10.0]],
        r_delay_flat_luts_cap_table=[[0.0, 20.0]],
        r_delay_flat_luts_dim=[[2, 2]],
        f_trans_flat_luts_values=[[0.1, 0.3, 0.5, 0.7]],
        f_trans_flat_luts_trans_table=[[0.0, 10.0]],
        f_trans_flat_luts_cap_table=[[0.0, 20.0]],
        f_trans_flat_luts_dim=[[2, 2]],
        r_trans_flat_luts_values=[[0.2, 0.4, 0.6, 0.8]],
        r_trans_flat_luts_trans_table=[[0.0, 10.0]],
        r_trans_flat_luts_cap_table=[[0.0, 20.0]],
        r_trans_flat_luts_dim=[[2, 2]],
    )


class DuBufferSurrogateTest(unittest.TestCase):
    def test_surrogate_uses_candidate_coordinate_lut_when_available(self):
        surrogate = build_buffer_surrogate(
            {0: {"input_cap": 0.5, "delay": 1.0, "output_slew": 0.2}},
            metadata=_lut_metadata(),
            contract_artifact={
                "status": "ok",
                "legal_cell_ids": [0],
                "legal_master_names": ["BUF_X1"],
            },
        )

        result = surrogate({}, bsu=0, input_slew=20.0, output_cap=10.0)

        self.assertEqual(result["buffer_input_cap"], 0.5)
        self.assertEqual(result["delay_source"], "lut_bilinear_interpolation")
        self.assertEqual(result["transition_source"], "lut_bilinear_interpolation")
        self.assertAlmostEqual(result["buffer_delay"], 6.5)
        self.assertAlmostEqual(result["buffer_output_slew"], 0.65)
        self.assertEqual(result["input_slew"], 20.0)
        self.assertEqual(result["output_cap"], 10.0)
        self.assertTrue(result["delay_lut_status"]["input_slew_clamped"])
        self.assertFalse(result["delay_lut_status"]["output_cap_clamped"])
        self.assertTrue(result["transition_lut_status"]["input_slew_clamped"])
        self.assertFalse(result["transition_lut_status"]["output_cap_clamped"])

    def test_surrogate_falls_back_to_buffer_library_without_lut_contract(self):
        surrogate = build_buffer_surrogate(
            {
                1: {
                    "input_cap": 0.7,
                    "delay": 2.5,
                    "output_slew": 1.25,
                    "delay_source": "unit_test_library",
                    "transition_source": "unit_test_transition",
                }
            },
        )

        result = surrogate({}, bsu=1, input_slew=3.0, output_cap=4.0)

        self.assertEqual(result["buffer_input_cap"], 0.7)
        self.assertEqual(result["buffer_delay"], 2.5)
        self.assertEqual(result["buffer_output_slew"], 1.25)
        self.assertEqual(result["delay_source"], "unit_test_library")
        self.assertEqual(result["transition_source"], "unit_test_transition")
        self.assertEqual(result["delay_lut_status"]["source"], "buffer_library")
        self.assertEqual(result["transition_lut_status"]["source"], "buffer_library")


if __name__ == "__main__":
    unittest.main()
