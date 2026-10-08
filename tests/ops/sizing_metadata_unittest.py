import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.macroPlaceDB import MacroPlaceDB
from dreamplace.BasicPlace import PlaceDataCollection, compute_init_size_logits
sys.path.pop()


class SizingMetadataTest(unittest.TestCase):
    def _build_db(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        db.dtype = np.float32
        db.main_id_is_sizeable = None
        db.main_id_2_cell_id_start = np.array([0, 3, 5], dtype=np.int32)
        db.cell_id_2_arc_id_start = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        db.inst_main_id = np.array([0, 1], dtype=np.int32)
        db.inst_libcell_offset = np.array([1, 0], dtype=np.int32)
        db.flat_libcell_info = np.array(
            [
                [0, 0, 1.0, 0],
                [1, 0, 2.0, 1],
                [2, 0, 4.0, 1],
                [3, 1, 1.5, 0],
                [4, 1, 3.0, 2],
            ],
            dtype=np.float32,
        )
        db.flat_libcell_main_id2size_vt_limit = np.array(
            [
                [0, 3, 2],
                [1, 2, 2],
            ],
            dtype=np.int32,
        )
        db.flat_libcell_leakage = np.array([0.1, 0.2, 0.4, 0.15, 0.3], dtype=np.float32)
        return db

    def test_normalize_flat_libcell_info(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        db.dtype = np.float32
        names, numeric = db._normalize_flat_libcell_info(
            [
                ["INV_X1", 0, 1.0, 0],
                ["INV_X2", 0, 2.0, 1],
            ]
        )

        self.assertEqual(names.tolist(), [b"INV_X1", b"INV_X2"])
        np.testing.assert_array_equal(numeric[:, 0], np.array([0.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(numeric[:, 1], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(numeric[:, 3], np.array([0.0, 1.0], dtype=np.float32))

    def test_normalize_flat_libcell_info_with_string_vt(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        db.dtype = np.float32
        names, numeric = db._normalize_flat_libcell_info(
            [
                ["INV_X1", 0, 1.0, "HVT"],
                ["INV_X2", 0, 2.0, "LVT"],
                ["INV_X4", 0, 4.0, "HVT"],
            ]
        )

        self.assertEqual(names.tolist(), [b"INV_X1", b"INV_X2", b"INV_X4"])
        np.testing.assert_array_equal(numeric[:, 1], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(numeric[:, 2], np.array([1.0, 2.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(numeric[:, 3], np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def test_normalize_flat_libarc_info(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        names, numeric = db._normalize_flat_libarc_info(
            [
                ["A", "ZN", 7, 3, -1, 2],
                ["B", "Y", 8, 0],
                ["A", "Y", 9, 1, 1, 0],
            ]
        )

        self.assertEqual(names.tolist(), [[b"A", b"ZN"], [b"B", b"Y"], [b"A", b"Y"]])
        np.testing.assert_array_equal(
            numeric,
            np.array(
                [
                    [0, 1, 7, 3, -1, 2],
                    [2, 3, 8, 0, 0, 0],
                    [0, 3, 9, 1, 1, 0],
                ],
                dtype=np.int32,
            ),
        )

    def test_normalize_flat_libcell_limit_info(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        names, numeric = db._normalize_flat_libcell_limit_info(
            [
                ["AND2X1H7R", 4, 3],
                ["INVX1H7L", 2, 2],
            ]
        )

        self.assertEqual(names.tolist(), [b"AND2X1H7R", b"INVX1H7L"])
        np.testing.assert_array_equal(
            numeric,
            np.array(
                [
                    [0, 4, 3],
                    [1, 2, 2],
                ],
                dtype=np.int32,
            ),
        )

    def test_normalize_flat_libpin_offsets(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        db.dtype = np.float32

        offsets = db._normalize_flat_libpin_offsets(
            [1.5, -2.0, 3.25],
            "flat_lib_pin_offset_x",
            expected_num_libpins=3,
        )

        np.testing.assert_allclose(offsets, np.array([1.5, -2.0, 3.25], dtype=np.float32))

    def test_normalize_flat_libpin_offsets_rejects_length_mismatch(self):
        db = MacroPlaceDB.__new__(MacroPlaceDB)
        db.dtype = np.float32

        with self.assertRaisesRegex(ValueError, "must match number of lib pins"):
            db._normalize_flat_libpin_offsets(
                [1.0, 2.0],
                "flat_lib_pin_offset_y",
                expected_num_libpins=3,
            )

    def test_build_inst_sizing_metadata(self):
        db = self._build_db()
        (
            inst_cell_id,
            inst_size_init,
            inst_leakage_init,
            inst_vt_init,
            inst_size_lower,
            inst_size_upper,
            inst_vt_mask,
            inst_is_sizeable,
        ) = db._build_inst_sizing_metadata()

        np.testing.assert_array_equal(inst_cell_id, np.array([1, 3], dtype=np.int32))
        np.testing.assert_allclose(inst_size_init, np.array([2.0, 1.5], dtype=np.float32))
        np.testing.assert_allclose(inst_leakage_init, np.array([0.2, 0.15], dtype=np.float32))
        np.testing.assert_allclose(inst_size_lower, np.array([1.0, 1.5], dtype=np.float32))
        np.testing.assert_allclose(inst_size_upper, np.array([4.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(inst_vt_init[0], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(inst_vt_init[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(inst_vt_mask[0], np.array([True, True, False]))
        np.testing.assert_array_equal(inst_vt_mask[1], np.array([True, False, True]))
        np.testing.assert_array_equal(inst_is_sizeable, np.array([True, True]))

    def test_timing_coordinate_column_drives_size_metadata_and_size_var(self):
        db = self._build_db()
        db.flat_libcell_info[:, 2] = np.array([10.0, 20.0, 30.0, 100.0, 200.0], dtype=np.float32)
        (
            _inst_cell_id,
            inst_size_init,
            _inst_leakage_init,
            _inst_vt_init,
            inst_size_lower,
            inst_size_upper,
            _inst_vt_mask,
            _inst_is_sizeable,
        ) = db._build_inst_sizing_metadata()

        np.testing.assert_allclose(inst_size_init, np.array([20.0, 100.0], dtype=np.float32))
        np.testing.assert_allclose(inst_size_lower, np.array([10.0, 100.0], dtype=np.float32))
        np.testing.assert_allclose(inst_size_upper, np.array([30.0, 200.0], dtype=np.float32))

        collection = PlaceDataCollection.__new__(PlaceDataCollection)
        collection.sizing_parameterization = "logits"
        collection.inst_size_lower = torch.from_numpy(inst_size_lower)
        collection.inst_size_upper = torch.from_numpy(inst_size_upper)
        collection.size_logits = torch.from_numpy(
            compute_init_size_logits(
                inst_size_init=inst_size_init,
                inst_size_lower=inst_size_lower,
                inst_size_upper=inst_size_upper,
                dtype=np.float32,
            )
        )

        size_var = collection.get_size_var()
        self.assertTrue(torch.all(size_var >= collection.inst_size_lower))
        self.assertTrue(torch.all(size_var <= collection.inst_size_upper))
        self.assertTrue(torch.allclose(size_var[0], torch.tensor(20.0), atol=1e-3))
        self.assertGreater(float(size_var[1]), 99.0)

    def test_missing_limit_fails_fast(self):
        db = self._build_db()
        db.flat_libcell_main_id2size_vt_limit = np.array([[0, 3, 2]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "Missing size/vt limit"):
            db._build_inst_sizing_metadata()

    def test_negative_main_id_is_not_sizeable(self):
        db = self._build_db()
        db.inst_main_id = np.array([-1, 1], dtype=np.int32)
        (
            _inst_cell_id,
            _inst_size_init,
            _inst_leakage_init,
            inst_vt_init,
            _inst_size_lower,
            _inst_size_upper,
            inst_vt_mask,
            inst_is_sizeable,
        ) = db._build_inst_sizing_metadata()

        np.testing.assert_array_equal(inst_is_sizeable, np.array([False, True]))
        np.testing.assert_array_equal(inst_vt_mask[0], np.array([True, False, False]))
        np.testing.assert_array_equal(inst_vt_init[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_negative_offset_still_checks_candidate_limits(self):
        db = self._build_db()
        db.inst_libcell_offset = np.array([-1, 0], dtype=np.int32)
        db.flat_libcell_main_id2size_vt_limit = np.array(
            [
                [0, 1, 1],
                [1, 2, 2],
            ],
            dtype=np.int32,
        )
        with self.assertRaisesRegex(ValueError, "more sizes than declared size limit"):
            db._build_inst_sizing_metadata()

    def test_unsizable_family_keeps_static_mapping(self):
        db = self._build_db()
        db.main_id_is_sizeable = np.array([False, True], dtype=np.bool_)
        (
            inst_cell_id,
            inst_size_init,
            inst_leakage_init,
            inst_vt_init,
            _inst_size_lower,
            _inst_size_upper,
            _inst_vt_mask,
            inst_is_sizeable,
        ) = db._build_inst_sizing_metadata()

        np.testing.assert_array_equal(inst_cell_id, np.array([1, 3], dtype=np.int32))
        np.testing.assert_allclose(inst_size_init, np.array([2.0, 1.5], dtype=np.float32))
        np.testing.assert_allclose(inst_leakage_init, np.array([0.2, 0.15], dtype=np.float32))
        np.testing.assert_array_equal(inst_vt_init[0], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(inst_is_sizeable, np.array([False, True]))

    def test_place_data_collection_projects_size_and_vt(self):
        collection = PlaceDataCollection.__new__(PlaceDataCollection)
        collection.sizing_parameterization = "logits"
        collection.inst_size_lower = torch.tensor([1.0, 2.0], dtype=torch.float32)
        collection.inst_size_upper = torch.tensor([5.0, 6.0], dtype=torch.float32)
        collection.size_logits = torch.tensor([0.0, 2.0], dtype=torch.float32)
        collection.inst_vt_mask = torch.tensor(
            [[True, True, False], [False, True, True]], dtype=torch.bool
        )
        collection.inst_vt_init = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        collection.inst_is_sizeable = torch.tensor([True, False], dtype=torch.bool)
        collection.vt_logits = torch.tensor(
            [[0.0, 1.0, -10.0], [-3.0, 2.0, 0.0]], dtype=torch.float32
        )

        size_var = collection.get_size_var()
        vt_var = collection.get_vt_var()

        self.assertTrue(torch.all(size_var >= collection.inst_size_lower))
        self.assertTrue(torch.all(size_var <= collection.inst_size_upper))
        self.assertTrue(torch.allclose(vt_var.sum(dim=1), torch.ones(2)))
        self.assertEqual(vt_var[0, 2].item(), 0.0)
        self.assertEqual(vt_var[1, 0].item(), 0.0)
        self.assertEqual(vt_var[1, 1].item(), 1.0)

    def test_compute_init_size_logits_uses_default_lower_clip(self):
        init_size_logits = compute_init_size_logits(
            inst_size_init=np.array([1.0], dtype=np.float32),
            inst_size_lower=np.array([1.0], dtype=np.float32),
            inst_size_upper=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )

        expected = np.log(np.float32(1e-4) / np.float32(1.0 - 1e-4))
        np.testing.assert_allclose(init_size_logits, np.array([expected], dtype=np.float32))

    def test_compute_init_size_logits_respects_env_lower_clip_override(self):
        with mock.patch.dict(os.environ, {"AIMP_SIZE_INIT_NORM_LOWER_CLIP": "0.05"}):
            init_size_logits = compute_init_size_logits(
                inst_size_init=np.array([1.0, 3.0], dtype=np.float32),
                inst_size_lower=np.array([1.0, 1.0], dtype=np.float32),
                inst_size_upper=np.array([5.0, 5.0], dtype=np.float32),
                dtype=np.float32,
            )

        expected = np.array(
            [
                np.log(np.float32(0.05) / np.float32(0.95)),
                0.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(init_size_logits, expected)


if __name__ == "__main__":
    unittest.main()
