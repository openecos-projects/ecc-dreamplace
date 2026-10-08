#!/usr/bin/env python

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


AUTODMP_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(AUTODMP_ROOT))

from dreamplace.ops.cell_modeling import cell_modeling as cell_modeling_module  # noqa: E402
from dreamplace.ops.cell_modeling import cell_modeling_op  # noqa: E402
from dreamplace.ops.cell_modeling.cell_modeling import (  # noqa: E402
    CellModeling,
    LUT_BOUNDARY_MODE_CLAMP,
    LUT_BOUNDARY_MODE_EXTRAPOLATE,
    LOG_LEAKAGE_POLY12_SCHEMA,
    REGRESSION_SCHEMAS,
)


class LogLeakageCellModelingTest(unittest.TestCase):
    def _cell_model_shell(self):
        model = object.__new__(CellModeling)
        model.device = torch.device("cpu")
        model.cell_model_schema = LOG_LEAKAGE_POLY12_SCHEMA
        return model

    def test_log_leakage_coord_is_normalized_per_main_id_and_vt(self):
        model = self._cell_model_shell()
        flat_libcell_info = torch.tensor(
            [
                [0, 0, 1.0, 0],
                [1, 0, 2.0, 0],
                [2, 0, 4.0, 0],
            ],
            dtype=torch.float32,
        )
        data_collections = SimpleNamespace(
            flat_libcell_leakage=torch.tensor([1e-6, 1e-5, 1e-4], dtype=torch.float32)
        )

        coord = model._flat_libcell_log_leakage_coord(flat_libcell_info, data_collections)

        self.assertTrue(torch.allclose(coord, torch.tensor([1.0, 1.5, 2.0]), atol=1e-5))
        self.assertEqual(model.log_leakage_coord_stats["num_duplicate_leakage_values"], 0)

    def test_log_leakage_coord_is_normalized_separately_by_vt(self):
        model = self._cell_model_shell()
        flat_libcell_info = torch.tensor(
            [
                [0, 0, 1.0, 0],
                [1, 0, 2.0, 0],
                [2, 0, 1.0, 1],
                [3, 0, 2.0, 1],
            ],
            dtype=torch.float32,
        )
        data_collections = SimpleNamespace(
            flat_libcell_leakage=torch.tensor([1e-6, 1e-4, 1e-3, 1e-1], dtype=torch.float32)
        )

        coord = model._flat_libcell_log_leakage_coord(flat_libcell_info, data_collections)

        self.assertTrue(torch.allclose(coord, torch.tensor([1.0, 2.0, 1.0, 2.0]), atol=1e-5))

    def test_log_leakage_schema_selects_leakage_coord_for_fit(self):
        model = self._cell_model_shell()
        meta = {
            "sizes": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "log_leakage_coords": torch.tensor([2.0, 1.0], dtype=torch.float32),
        }

        selected = model._regression_size_coord_from_meta(meta, LOG_LEAKAGE_POLY12_SCHEMA)

        self.assertTrue(torch.equal(selected, meta["log_leakage_coords"]))
        self.assertNotEqual(selected.tolist(), meta["sizes"].tolist())

    def test_log_leakage_schema_remaps_forward_size_coord(self):
        model = self._cell_model_shell()
        model.log_leakage_coord_table = {
            (0, 0, 1.0): 1.75,
        }
        libcell_main_id = torch.tensor([0], dtype=torch.long)
        vt = torch.tensor([0.0], dtype=torch.float32)
        size = torch.tensor([1.0], dtype=torch.float32)

        coord = model._forward_size_coord(LOG_LEAKAGE_POLY12_SCHEMA, libcell_main_id, vt, size)

        self.assertTrue(torch.allclose(coord, torch.tensor([1.75]), atol=1e-6))

    def test_log_leakage_schema_analysis_payload_uses_direct_coord(self):
        model = self._cell_model_shell()
        payload = {
            "size": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "log_leakage_coord": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }

        size = model._analysis_forward_size_payload(payload)

        self.assertTrue(torch.equal(size, payload["log_leakage_coord"]))
        self.assertNotEqual(size.tolist(), payload["size"].tolist())

    def test_log_leakage_schema_analysis_flag_prevents_second_remap(self):
        model = self._cell_model_shell()
        model.log_leakage_coord_table = {
            (0, 0, 1.0): 1.25,
        }
        model._analysis_size_coord_is_precomputed = True
        libcell_main_id = torch.tensor([0], dtype=torch.long)
        vt = torch.tensor([0.0], dtype=torch.float32)
        precomputed_coord = torch.tensor([1.0], dtype=torch.float32)

        coord = model._forward_size_coord(
            LOG_LEAKAGE_POLY12_SCHEMA,
            libcell_main_id,
            vt,
            precomputed_coord,
        )

        self.assertTrue(torch.equal(coord, precomputed_coord))

    def test_log_leakage_schema_requires_leakage_tensor(self):
        model = self._cell_model_shell()
        flat_libcell_info = torch.tensor([[0, 0, 1.0, 0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires flat_libcell_leakage"):
            model._flat_libcell_log_leakage_coord(flat_libcell_info, SimpleNamespace())

    def test_log_leakage_schema_is_registered(self):
        self.assertIn(LOG_LEAKAGE_POLY12_SCHEMA, REGRESSION_SCHEMAS)


class PiecewiseSizeLutBoundaryModeTest(unittest.TestCase):
    def _assert_native_piecewise_size_lut_boundary_mode(
        self,
        device,
        boundary_mode,
        expected_value,
        expected_slew_grad=None,
        expected_cap_grad=None,
    ):
        size_table = torch.tensor([[1.0]], dtype=torch.float64, device=device)
        arc_table = torch.tensor([[0]], dtype=torch.long, device=device)
        size_count = torch.tensor([1], dtype=torch.long, device=device)
        trans_tables = torch.tensor(
            [
                [40.0, 80.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        cap_tables = torch.tensor(
            [
                [10.0, 20.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        lut_values = torch.tensor(
            [
                [100.0, 200.0, 110.0, 210.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        trans_dims = torch.tensor([2], dtype=torch.long, device=device)
        cap_dims = torch.tensor([2], dtype=torch.long, device=device)
        size = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
        input_slew = torch.tensor([60.0], dtype=torch.float64, device=device, requires_grad=True)
        out_cap = torch.tensor([30.0], dtype=torch.float64, device=device, requires_grad=True)

        value = cell_modeling_op.piecewise_size_forward(
            size_table,
            arc_table,
            size_count,
            trans_tables,
            cap_tables,
            lut_values,
            trans_dims,
            cap_dims,
            size,
            input_slew,
            out_cap,
            boundary_mode,
        )
        value.sum().backward()

        self.assertTrue(
            torch.allclose(value, torch.tensor([expected_value], dtype=torch.float64, device=device))
        )
        if expected_slew_grad is not None:
            self.assertTrue(
                torch.allclose(
                    input_slew.grad,
                    torch.tensor([expected_slew_grad], dtype=torch.float64, device=device),
                )
            )
        if expected_cap_grad is not None:
            self.assertTrue(
                torch.allclose(
                    out_cap.grad,
                    torch.tensor([expected_cap_grad], dtype=torch.float64, device=device),
                )
            )

    def test_native_piecewise_size_extrapolates_out_of_range_cap_and_keeps_gradient_cpu(self):
        self._assert_native_piecewise_size_lut_boundary_mode(
            "cpu",
            LUT_BOUNDARY_MODE_EXTRAPOLATE,
            305.0,
            expected_slew_grad=0.25,
            expected_cap_grad=10.0,
        )

    def test_native_piecewise_size_clamps_out_of_range_cap_cpu(self):
        self._assert_native_piecewise_size_lut_boundary_mode(
            "cpu",
            LUT_BOUNDARY_MODE_CLAMP,
            205.0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_native_piecewise_size_extrapolates_out_of_range_cap_and_keeps_gradient_cuda(self):
        self._assert_native_piecewise_size_lut_boundary_mode(
            "cuda",
            LUT_BOUNDARY_MODE_EXTRAPOLATE,
            305.0,
            expected_slew_grad=0.25,
            expected_cap_grad=10.0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_native_piecewise_size_clamps_out_of_range_cap_cuda(self):
        self._assert_native_piecewise_size_lut_boundary_mode(
            "cuda",
            LUT_BOUNDARY_MODE_CLAMP,
            205.0,
        )

    def test_python_lut_entry_2d_extrapolates_out_of_range_cap(self):
        model = object.__new__(CellModeling)
        model.cell_model_lut_boundary_mode = LUT_BOUNDARY_MODE_EXTRAPOLATE
        input_slew = torch.tensor([60.0], dtype=torch.float64)
        out_cap = torch.tensor([30.0], dtype=torch.float64)
        trans_tables = torch.tensor([[40.0, 80.0]], dtype=torch.float64)
        cap_tables = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
        lut_values = torch.tensor([[100.0, 200.0, 110.0, 210.0]], dtype=torch.float64)
        trans_dims = torch.tensor([2], dtype=torch.long)
        cap_dims = torch.tensor([2], dtype=torch.long)

        with mock.patch.object(cell_modeling_module, "lut_entry_2d_op", None):
            value = model._lut_entry_2d_vectorized(
                input_slew,
                out_cap,
                trans_tables,
                cap_tables,
                lut_values,
                trans_dims,
                cap_dims,
            )

        self.assertTrue(torch.allclose(value, torch.tensor([305.0], dtype=torch.float64)))

    def test_python_lut_entry_2d_clamps_out_of_range_cap(self):
        model = object.__new__(CellModeling)
        model.cell_model_lut_boundary_mode = LUT_BOUNDARY_MODE_CLAMP
        input_slew = torch.tensor([60.0], dtype=torch.float64)
        out_cap = torch.tensor([30.0], dtype=torch.float64)
        trans_tables = torch.tensor([[40.0, 80.0]], dtype=torch.float64)
        cap_tables = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
        lut_values = torch.tensor([[100.0, 200.0, 110.0, 210.0]], dtype=torch.float64)
        trans_dims = torch.tensor([2], dtype=torch.long)
        cap_dims = torch.tensor([2], dtype=torch.long)

        with mock.patch.object(cell_modeling_module, "lut_entry_2d_op", None):
            value = model._lut_entry_2d_vectorized(
                input_slew,
                out_cap,
                trans_tables,
                cap_tables,
                lut_values,
                trans_dims,
                cap_dims,
            )

        self.assertTrue(torch.allclose(value, torch.tensor([205.0], dtype=torch.float64)))

    def test_lut_boundary_mode_resolves_env_override(self):
        model = object.__new__(CellModeling)

        with mock.patch.dict("os.environ", {"AIMP_CELL_MODEL_LUT_BOUNDARY_MODE": "clamp"}):
            mode = model._resolve_lut_boundary_mode(None)

        self.assertEqual(mode, LUT_BOUNDARY_MODE_CLAMP)


if __name__ == "__main__":
    unittest.main()
