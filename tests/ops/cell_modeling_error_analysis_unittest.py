import os
import sys
import tempfile
import unittest
import math
import types
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.cell_modeling.cell_modeling import (
    CellModeling,
    LINEAR6_SCHEMA,
    LOGCROSS12_SCHEMA,
    PIECEWISE_LINEAR_SCHEMA,
    POLY12_SCHEMA,
)
sys.path.pop()


class CellModelingErrorAnalysisTest(unittest.TestCase):
    def _build_luts(self):
        return SimpleNamespace(
            flat_luts_values=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[2, 2]], dtype=torch.int64),
        )

    def _build_data_collections(self, cache_root):
        luts = self._build_luts()
        return SimpleNamespace(
            device=torch.device("cpu"),
            surrogate_cache_root=cache_root,
            flat_libcell_info=torch.tensor([[0.0, 0.0, 2.0, 1.0]], dtype=torch.float32),
            flat_libarc_info=torch.tensor([[0, 1, 0, 0]], dtype=torch.int64),
            arcs_info=SimpleNamespace(
                f_delay_luts=luts,
                r_delay_luts=luts,
                f_trans_luts=luts,
                r_trans_luts=luts,
            ),
        )

    def test_analyze_all_lut_errors_returns_cache_and_dataset_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = CellModeling(self._build_data_collections(tmpdir))
            report = model.analyze_all_lut_errors(top_k=2, include_all_points=True)

        self.assertIn("cache", report)
        self.assertEqual(report["cache"]["cache_event"], "write")
        self.assertEqual(report["aggregate"]["evaluated_points"], 16)
        self.assertIn("mean_rel_error", report["aggregate"])
        self.assertIn("r2", report["aggregate"])
        self.assertTrue(math.isfinite(report["aggregate"]["mean_rel_error"]))
        self.assertTrue(math.isfinite(report["aggregate"]["r2"]))
        self.assertLessEqual(report["aggregate"]["r2"], 1.0)
        self.assertEqual(set(report["datasets"].keys()), {"f_delay", "r_delay", "f_trans", "r_trans"})
        self.assertEqual(len(report["datasets"]["f_delay"]["topk_rows"]), 2)
        self.assertEqual(len(report["datasets"]["f_delay"]["all_rows"]), 4)
        self.assertIn("mean_rel_error", report["datasets"]["f_delay"]["summary"])
        self.assertIn("r2", report["datasets"]["f_delay"]["summary"])

    def test_analyze_lut_error_respects_sampling_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = CellModeling(self._build_data_collections(tmpdir))
            report = model.analyze_lut_error("f_delay", max_points=2, top_k=5, include_all_points=True)

        self.assertEqual(report["summary"]["total_points"], 4)
        self.assertEqual(report["summary"]["evaluated_points"], 2)
        self.assertEqual(len(report["topk_rows"]), 2)
        self.assertEqual(len(report["all_rows"]), 2)

    def test_piecewise_schema_beats_linear_on_intermediate_size_vt_query(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0, 11.0, 13.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 3.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
            )
            data.cell_model_schema = LINEAR6_SCHEMA
            linear_model = CellModeling(data)
            linear_value = linear_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )

            data.cell_model_schema = PIECEWISE_LINEAR_SCHEMA
            piecewise_model = CellModeling(data)
            piecewise_value = piecewise_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )

        target = torch.tensor([7.0], dtype=torch.float32)
        self.assertLess(
            (piecewise_value - target).abs().item(),
            (linear_value - target).abs().item(),
        )
        self.assertAlmostEqual(piecewise_value.item(), 7.0, places=5)

    def test_piecewise_schema_clamps_out_of_range_query_to_finite_value(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0, 11.0, 13.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.1], [0.1], [0.1], [0.1]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.2], [0.2], [0.2], [0.2]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 3.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
                cell_model_schema=PIECEWISE_LINEAR_SCHEMA,
            )
            piecewise_model = CellModeling(data)
            piecewise_value = piecewise_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([-1.0], dtype=torch.float32),
                torch.tensor([10.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([5.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )

        self.assertTrue(torch.isfinite(piecewise_value).all())
        self.assertAlmostEqual(piecewise_value.item(), 3.0, places=5)

    def test_piecewise_schema_uses_piecewise_forward_with_stable_gradient_backup(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0, 11.0, 13.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 3.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
                cell_model_schema=PIECEWISE_LINEAR_SCHEMA,
                piecewise_gradient_mode="linear6_ste",
            )
            piecewise_model = CellModeling(data)
            size = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
            vt = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
            output = piecewise_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                vt,
                size,
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )
            output.sum().backward()

        self.assertAlmostEqual(output.item(), 7.0, places=5)
        self.assertTrue(torch.isfinite(size.grad).all())
        self.assertTrue(torch.isfinite(vt.grad).all())

    def test_default_model_uses_piecewise_forward_with_native_gradient(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0, 11.0, 13.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 3.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
            )
            piecewise_model = CellModeling(data)
            size = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
            vt = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
            output = piecewise_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                vt,
                size,
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )
            output.sum().backward()

        self.assertAlmostEqual(output.item(), 7.0, places=5)
        self.assertAlmostEqual(size.grad.item(), 1.0, places=5)
        self.assertAlmostEqual(vt.grad.item(), 10.0, places=5)

    def test_piecewise_schema_rejects_unknown_gradient_mode(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0, 11.0, 13.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0, 1.0],
                        [3.0, 0.0, 3.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
                cell_model_schema=PIECEWISE_LINEAR_SCHEMA,
                piecewise_gradient_mode="bad_mode",
            )
            piecewise_model = CellModeling(data)

            with self.assertRaisesRegex(ValueError, "unsupported piecewise_gradient_mode"):
                piecewise_model(
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0.5], dtype=torch.float32),
                    torch.tensor([2.0], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                    arc_types=torch.tensor([0], dtype=torch.long),
                )

    def test_piecewise_schema_uses_dataset_local_arc_indices_when_global_arc_list_is_longer(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([1.0, 3.0], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[1, 1], [1, 1]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 3.0, 0.0],
                        [2.0, 0.0, 5.0, 0.0],
                        [3.0, 0.0, 7.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
                cell_model_schema=PIECEWISE_LINEAR_SCHEMA,
            )
            piecewise_model = CellModeling(data)
            piecewise_value = piecewise_model(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([6.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                arc_types=torch.tensor([0], dtype=torch.long),
            )

        self.assertTrue(torch.isfinite(piecewise_value).all())
        self.assertAlmostEqual(piecewise_value.item(), 3.0, places=5)

    def test_logcross_schema_beats_linear_on_log_nonlinear_pattern(self):
        trans_table = torch.tensor(
            [
                [0.1, 2.0, 8.0],
                [0.1, 2.0, 8.0],
                [0.1, 2.0, 8.0],
                [0.1, 2.0, 8.0],
            ],
            dtype=torch.float32,
        )
        cap_table = torch.tensor(
            [
                [0.001, 0.01, 0.03],
                [0.001, 0.01, 0.03],
                [0.001, 0.01, 0.03],
                [0.001, 0.01, 0.03],
            ],
            dtype=torch.float32,
        )
        flat_values = []
        libcell_info = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        for libcell in libcell_info:
            size = libcell[2]
            vt = libcell[3]
            slew_grid = trans_table[0].unsqueeze(1).expand(-1, 3)
            cap_grid = cap_table[0].unsqueeze(0).expand(3, -1)
            target = (
                1.7 * torch.log1p(slew_grid)
                + 2.3 * torch.log1p(cap_grid * 1e3)
                + 0.9 * (1.0 / size)
                + 1.2 * vt
                + 0.8 * torch.log1p(slew_grid) * torch.log1p(cap_grid * 1e3)
                + 0.6 * vt * (1.0 / size)
                + 0.4
            )
            flat_values.append(target)
        luts = SimpleNamespace(
            flat_luts_values=torch.stack(flat_values, dim=0),
            flat_luts_trans_table=trans_table,
            flat_luts_cap_table=cap_table,
            flat_luts_dim=torch.tensor([[3, 3], [3, 3], [3, 3], [3, 3]], dtype=torch.int64),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data = SimpleNamespace(
                device=torch.device("cpu"),
                surrogate_cache_root=tmpdir,
                flat_libcell_info=libcell_info,
                flat_libarc_info=torch.tensor(
                    [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 3, 0]],
                    dtype=torch.int64,
                ),
                arcs_info=SimpleNamespace(
                    f_delay_luts=luts,
                    r_delay_luts=luts,
                    f_trans_luts=luts,
                    r_trans_luts=luts,
                ),
            )
            data.cell_model_schema = LINEAR6_SCHEMA
            linear_model = CellModeling(data)
            linear_report = linear_model.analyze_lut_error("f_delay", include_all_points=False)

            data.cell_model_schema = LOGCROSS12_SCHEMA
            logcross_model = CellModeling(data)
            logcross_report = logcross_model.analyze_lut_error("f_delay", include_all_points=False)

            data.cell_model_schema = POLY12_SCHEMA
            poly_model = CellModeling(data)
            poly_report = poly_model.analyze_lut_error("f_delay", include_all_points=False)

        self.assertLess(
            logcross_report["summary"]["mae"],
            linear_report["summary"]["mae"],
        )
        self.assertLess(
            logcross_report["summary"]["mae"],
            0.05,
        )
        self.assertTrue(torch.isfinite(torch.tensor(poly_report["summary"]["mae"])))


class CellModelingArcTypeDispatchTest(unittest.TestCase):
    def _base_inputs(self):
        return (
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            torch.tensor([0.2, 0.4], dtype=torch.float32),
            torch.tensor([0.5, 0.7], dtype=torch.float32),
        )

    def test_regression_forward_with_uniform_arc_types_only_computes_requested_dataset(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.cell_model_schema = LINEAR6_SCHEMA
        model.coeff_f_delay = torch.tensor([[[1.0]]], dtype=torch.float32)
        model.coeff_r_delay = torch.tensor([[[2.0]]], dtype=torch.float32)
        model.coeff_f_trans = torch.tensor([[[3.0]]], dtype=torch.float32)
        model.coeff_r_trans = torch.tensor([[[4.0]]], dtype=torch.float32)
        dataset_calls = []

        def fake_linear(self, coeff_tensor, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, schema=None):
            del libcell_main_id, arc_offset, vt, size, out_cap, schema
            coeff_to_dataset = {
                id(self.coeff_f_delay): ("f_delay", 10.0),
                id(self.coeff_r_delay): ("r_delay", 20.0),
                id(self.coeff_f_trans): ("f_trans", 30.0),
                id(self.coeff_r_trans): ("r_trans", 40.0),
            }
            dataset_name, value = coeff_to_dataset[id(coeff_tensor)]
            dataset_calls.append(dataset_name)
            return torch.full_like(input_slew, value)

        model._linear_forward_dataset = types.MethodType(fake_linear, model)
        inputs = self._base_inputs()

        result = model.forward(*inputs, arc_types=torch.tensor([0, 0], dtype=torch.long))

        self.assertTrue(torch.equal(result, torch.tensor([10.0, 10.0], dtype=torch.float32)))
        self.assertEqual(dataset_calls, ["f_delay"])

    def test_forward_with_unknown_arc_type_keeps_legacy_zero_fill_behavior(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.cell_model_schema = LINEAR6_SCHEMA
        model.coeff_f_delay = torch.tensor([[[1.0]]], dtype=torch.float32)
        model.coeff_r_delay = torch.tensor([[[2.0]]], dtype=torch.float32)
        model.coeff_f_trans = torch.tensor([[[3.0]]], dtype=torch.float32)
        model.coeff_r_trans = torch.tensor([[[4.0]]], dtype=torch.float32)
        dataset_calls = []

        def fake_linear(self, coeff_tensor, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, schema=None):
            del libcell_main_id, arc_offset, vt, size, out_cap, schema
            coeff_to_dataset = {
                id(self.coeff_f_delay): ("f_delay", 10.0),
                id(self.coeff_r_delay): ("r_delay", 20.0),
                id(self.coeff_f_trans): ("f_trans", 30.0),
                id(self.coeff_r_trans): ("r_trans", 40.0),
            }
            dataset_name, value = coeff_to_dataset[id(coeff_tensor)]
            dataset_calls.append(dataset_name)
            return torch.full_like(input_slew, value)

        model._linear_forward_dataset = types.MethodType(fake_linear, model)
        inputs = self._base_inputs()

        result = model.forward(*inputs, arc_types=torch.tensor([0, 99], dtype=torch.long))

        self.assertTrue(torch.equal(result, torch.tensor([10.0, 0.0], dtype=torch.float32)))
        self.assertEqual(dataset_calls, ["f_delay"])

    def test_piecewise_forward_with_uniform_arc_types_only_computes_requested_dataset(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.cell_model_schema = PIECEWISE_LINEAR_SCHEMA
        model.piecewise_gradient_mode = "native_piecewise"
        dataset_calls = []

        def fake_piecewise(self, dataset_name, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
            del libcell_main_id, arc_offset, vt, size, out_cap
            dataset_calls.append(dataset_name)
            dataset_to_value = {
                "f_delay": 10.0,
                "r_delay": 20.0,
                "f_trans": 30.0,
                "r_trans": 40.0,
            }
            return torch.full_like(input_slew, dataset_to_value[dataset_name])

        def passthrough_gradient_mode(
            self,
            piecewise_value,
            dataset_name,
            libcell_main_id,
            arc_offset,
            vt,
            size,
            input_slew,
            out_cap,
        ):
            del self, dataset_name, libcell_main_id, arc_offset, vt, size, input_slew, out_cap
            return piecewise_value

        model._piecewise_forward_dataset = types.MethodType(fake_piecewise, model)
        model._apply_piecewise_gradient_mode = types.MethodType(passthrough_gradient_mode, model)
        inputs = self._base_inputs()

        result = model.forward(*inputs, arc_types=torch.tensor([2, 2], dtype=torch.long))

        self.assertTrue(torch.equal(result, torch.tensor([30.0, 30.0], dtype=torch.float32)))
        self.assertEqual(dataset_calls, ["f_trans"])


class CellModelingForwardProfileTest(unittest.TestCase):
    def test_forward_profile_logs_aggregate_latency(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.cell_model_schema = LINEAR6_SCHEMA
        model._cell_model_forward_profile_enabled = True
        model._cell_model_forward_profile_log_every = 2
        model._cell_model_forward_profile_calls = 0
        model._cell_model_forward_profile_total_seconds = 0.0
        model._cell_model_forward_profile_total_arcs = 0

        def fake_selected(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types):
            del self, libcell_main_id, arc_offset, vt, size, out_cap, arc_types
            return torch.full_like(input_slew, 1.0)

        model._forward_selected_arc_types = types.MethodType(fake_selected, model)
        inputs = (
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            torch.tensor([0.2, 0.4], dtype=torch.float32),
            torch.tensor([0.5, 0.7], dtype=torch.float32),
        )
        arc_types = torch.tensor([0, 0], dtype=torch.long)

        with self.assertLogs(level="INFO") as captured:
            model.forward(*inputs, arc_types=arc_types)
            model.forward(*inputs, arc_types=arc_types)

        self.assertEqual(model._cell_model_forward_profile_calls, 2)
        self.assertEqual(model._cell_model_forward_profile_total_arcs, 4)
        logs = "\n".join(captured.output)
        self.assertIn("CellModeling.forward profile", logs)
        self.assertIn("calls=2", logs)
        self.assertIn("batch=2", logs)
        self.assertIn("avg_us_per_arc=", logs)


class CellModelingLutLookupCacheTest(unittest.TestCase):
    def test_lut_lookup_uses_preexpanded_runtime_cache(self):
        luts = SimpleNamespace(
            flat_luts_values=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            flat_luts_trans_table=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            flat_luts_cap_table=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            flat_luts_dim=torch.tensor([[2, 2]], dtype=torch.int64),
        )
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.data_collections = SimpleNamespace(
            arcs_info=SimpleNamespace(
                f_delay_luts=luts,
                r_delay_luts=luts,
                f_trans_luts=luts,
                r_trans_luts=luts,
            )
        )
        model._lut_runtime_lookup_cache = {
            "f_delay": {
                "trans_dims": torch.tensor([2], dtype=torch.long),
                "cap_dims": torch.tensor([2], dtype=torch.long),
                "values_3d": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
                "values_flat": torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
                "trans_tables": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                "cap_tables": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            }
        }

        def fail_expand(self, values, dims):
            del self, values, dims
            raise AssertionError("runtime cache should bypass _expand_lut_values")

        model._expand_lut_values = types.MethodType(fail_expand, model)

        result = model._lut_lookup(
            "f_delay",
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0.15], dtype=torch.float32),
            torch.tensor([1.5], dtype=torch.float32),
        )

        self.assertTrue(torch.allclose(result, torch.tensor([2.5], dtype=torch.float32), atol=1e-6))


class CellModelingPiecewiseFastPathTest(unittest.TestCase):
    @staticmethod
    def _scalar_f_delay_lut_data_collections():
        # Arc slot 7 of the piecewise state references a 1x1 (scalar) f_delay LUT, so
        # _piecewise_size_interp must use its python interpolation path instead of the
        # fused 2D native op.
        luts = SimpleNamespace(
            flat_luts_values=torch.full((8, 1), 4.0, dtype=torch.float32),
            flat_luts_trans_table=torch.full((8, 1), 0.1, dtype=torch.float32),
            flat_luts_cap_table=torch.full((8, 1), 0.2, dtype=torch.float32),
            flat_luts_dim=torch.ones((8, 2), dtype=torch.int64),
        )
        return SimpleNamespace(
            arcs_info=SimpleNamespace(
                f_delay_luts=luts,
                r_delay_luts=luts,
                f_trans_luts=luts,
                r_trans_luts=luts,
            )
        )

    def test_piecewise_size_interp_single_candidate_bypasses_searchsorted(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.data_collections = self._scalar_f_delay_lut_data_collections()
        state = {
            "size_table": torch.tensor([[[[2.0]]]], dtype=torch.float32),
            "arc_index_table": torch.tensor([[[[7]]]], dtype=torch.long),
            "size_count": torch.tensor([[[1]]], dtype=torch.long),
        }

        def fake_lut_lookup(self, dataset_name, arc_indices, input_slew, out_cap):
            del self, dataset_name, input_slew, out_cap
            return arc_indices.to(dtype=torch.float32) + 0.25

        model._lut_lookup = types.MethodType(fake_lut_lookup, model)

        with mock.patch("torch.searchsorted", side_effect=AssertionError("single-candidate path should not search")):
            result = model._piecewise_size_interp(
                "f_delay",
                state,
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([0.1], dtype=torch.float32),
                torch.tensor([0.2], dtype=torch.float32),
            )

        self.assertTrue(torch.allclose(result, torch.tensor([7.25], dtype=torch.float32)))

    def test_piecewise_size_interp_skips_duplicate_lut_lookup_when_arc_slots_match(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.data_collections = self._scalar_f_delay_lut_data_collections()
        state = {
            "size_table": torch.tensor([[[[2.0]]]], dtype=torch.float32),
            "arc_index_table": torch.tensor([[[[7]]]], dtype=torch.long),
            "size_count": torch.tensor([[[1]]], dtype=torch.long),
        }
        call_counter = {"count": 0}

        def fake_lut_lookup(self, dataset_name, arc_indices, input_slew, out_cap):
            del self, dataset_name, input_slew, out_cap
            call_counter["count"] += 1
            return arc_indices.to(dtype=torch.float32) + 0.5

        model._lut_lookup = types.MethodType(fake_lut_lookup, model)

        result = model._piecewise_size_interp(
            "f_delay",
            state,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32),
            torch.tensor([0.2], dtype=torch.float32),
        )

        self.assertEqual(call_counter["count"], 1)
        self.assertTrue(torch.allclose(result, torch.tensor([7.5], dtype=torch.float32)))

    def test_piecewise_forward_skips_duplicate_size_interp_when_vt_slots_match(self):
        model = CellModeling.__new__(CellModeling)
        nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.vt_order_tensor = torch.tensor([0.0], dtype=torch.float32)
        model.piecewise_state = {
            "f_delay": {
                "size_table": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                "vt_available": torch.ones((1, 1, 1), dtype=torch.bool),
            }
        }
        call_counter = {"count": 0}

        def fake_vt_slots(self, state, mt, ao, vt):
            del self, state, mt, ao, vt
            zeros = torch.zeros(2, dtype=torch.long)
            return zeros, zeros

        def fake_size_interp(self, dataset_name, state, mt, ao, vt_slot, size, input_slew, out_cap):
            del self, dataset_name, state, mt, ao, vt_slot, size, input_slew, out_cap
            call_counter["count"] += 1
            return torch.tensor([7.0, 9.0], dtype=torch.float32)

        model._piecewise_vt_slots = types.MethodType(fake_vt_slots, model)
        model._piecewise_size_interp = types.MethodType(fake_size_interp, model)

        result = model._piecewise_forward_dataset(
            "f_delay",
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([0.3, 0.4], dtype=torch.float32),
        )

        self.assertEqual(call_counter["count"], 1)
        self.assertTrue(torch.allclose(result, torch.tensor([7.0, 9.0], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
