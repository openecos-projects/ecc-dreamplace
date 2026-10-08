import os
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
sys.path.pop()


class CellModelingCacheTest(unittest.TestCase):
    COEFF_DIM = 12

    def test_cache_disabled_without_explicit_root(self):
        data = self._build_data_collections(None)
        coeff_dim = self.COEFF_DIM

        with mock.patch.object(
            CellModeling,
            "_fit_dataset",
            return_value={(0, 0): torch.ones(coeff_dim, dtype=torch.float32)},
        ):
            model = CellModeling(data)

        self.assertEqual(model.cache_event, "disabled")
        self.assertEqual(model.cache_reason, "cache_root_unset")
        cache_meta = model._cache_metadata(
            data.flat_libcell_info, data.flat_libarc_info, data
        )
        self.assertIsNone(cache_meta["cache_dir"])
        self.assertIsNone(cache_meta["manifest_path"])
        self.assertIsNone(cache_meta["weights_path"])

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
            cell_model_schema="main_id_arc_offset_poly12",
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

    def test_build_writes_cache_and_reload_skips_refit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._build_data_collections(tmpdir)
            fitted = {"count": 0}
            coeff_dim = self.COEFF_DIM

            def fake_fit(self, luts_info, libcell_main_ids, arc_offsets, sizes, vts):
                fitted["count"] += 1
                return {(0, 0): torch.arange(coeff_dim, dtype=torch.float32)}

            with mock.patch.object(CellModeling, "_fit_dataset", new=fake_fit):
                model = CellModeling(data)

            self.assertEqual(fitted["count"], 4)
            cache_meta = model._cache_metadata(data.flat_libcell_info, data.flat_libarc_info, data)
            self.assertEqual(model.cache_event, "write")
            self.assertEqual(model.cache_history[0]["event"], "miss")
            self.assertEqual(model.cache_history[0]["reason"], "cache_dir_missing")
            self.assertTrue(os.path.isdir(cache_meta["cache_dir"]))
            with open(cache_meta["manifest_path"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["num_libcells"], 1)
            self.assertEqual(manifest["num_arcs"], 1)
            self.assertEqual(manifest["cell_model_schema"], "main_id_arc_offset_poly12")

            def should_not_fit(*args, **kwargs):
                raise AssertionError("cache hit should skip dataset fitting")

            with mock.patch.object(CellModeling, "_fit_dataset", side_effect=should_not_fit):
                reloaded = CellModeling(self._build_data_collections(tmpdir))

            self.assertEqual(reloaded.cache_event, "hit")
            self.assertIsNone(reloaded.cache_reason)
            self.assertTrue(torch.equal(model.coeff_f_delay.cpu(), reloaded.coeff_f_delay.cpu()))
            self.assertTrue(torch.equal(model.coeff_r_trans.cpu(), reloaded.coeff_r_trans.cpu()))

    def test_partial_cache_failure_refits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._build_data_collections(tmpdir)
            coeff_dim = self.COEFF_DIM

            with mock.patch.object(
                CellModeling,
                "_fit_dataset",
                return_value={(0, 0): torch.ones(coeff_dim, dtype=torch.float32)},
            ):
                model = CellModeling(data)

            cache_meta = model._cache_metadata(data.flat_libcell_info, data.flat_libarc_info, data)
            os.remove(cache_meta["weights_path"])

            fitted = {"count": 0}

            def fake_refit(self, luts_info, libcell_main_ids, arc_offsets, sizes, vts):
                fitted["count"] += 1
                return {(0, 0): torch.full((coeff_dim,), 2.0, dtype=torch.float32)}

            with mock.patch.object(CellModeling, "_fit_dataset", new=fake_refit):
                rebuilt = CellModeling(self._build_data_collections(tmpdir))

            self.assertEqual(fitted["count"], 4)
            self.assertTrue(os.path.isfile(cache_meta["weights_path"]))
            self.assertEqual(rebuilt.cache_event, "write")
            self.assertEqual(rebuilt.cache_history[0]["event"], "miss")
            self.assertEqual(rebuilt.cache_history[0]["reason"], "weights_missing")
            self.assertTrue(torch.equal(rebuilt.coeff_f_delay.cpu(), torch.full((1, 1, coeff_dim), 2.0)))

    def test_manifest_mismatch_forces_refit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._build_data_collections(tmpdir)
            coeff_dim = self.COEFF_DIM

            with mock.patch.object(
                CellModeling,
                "_fit_dataset",
                return_value={(0, 0): torch.ones(coeff_dim, dtype=torch.float32)},
            ):
                model = CellModeling(data)

            cache_meta = model._cache_metadata(data.flat_libcell_info, data.flat_libarc_info, data)
            with open(cache_meta["manifest_path"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["num_arcs"] = 99
            with open(cache_meta["manifest_path"], "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            fitted = {"count": 0}

            def fake_refit(self, luts_info, libcell_main_ids, arc_offsets, sizes, vts):
                fitted["count"] += 1
                return {(0, 0): torch.full((coeff_dim,), 3.0, dtype=torch.float32)}

            with mock.patch.object(CellModeling, "_fit_dataset", new=fake_refit):
                rebuilt = CellModeling(self._build_data_collections(tmpdir))

            self.assertEqual(fitted["count"], 4)
            self.assertEqual(rebuilt.cache_event, "write")
            self.assertEqual(rebuilt.cache_history[0]["event"], "miss")
            self.assertEqual(rebuilt.cache_history[0]["reason"], "manifest_mismatch:num_arcs")
            self.assertTrue(torch.equal(rebuilt.coeff_f_delay.cpu(), torch.full((1, 1, coeff_dim), 3.0)))

    def test_expand_lut_values_supports_1d_tables(self):
        model = CellModeling.__new__(CellModeling)
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=torch.float32)
        dims = torch.tensor([[3, 1], [1, 2]], dtype=torch.int64)

        expanded = model._expand_lut_values(values, dims)

        self.assertEqual(tuple(expanded.shape), (2, 3, 2))
        self.assertTrue(torch.equal(expanded[0, :3, 0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(expanded[1, 0, :2], torch.tensor([4.0, 5.0])))

    def test_expand_lut_values_supports_flattened_2d_tables(self):
        model = CellModeling.__new__(CellModeling)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        dims = torch.tensor([[2, 2]], dtype=torch.int64)

        expanded = model._expand_lut_values(values, dims)

        self.assertEqual(tuple(expanded.shape), (1, 2, 2))
        self.assertTrue(
            torch.equal(expanded[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
        )

    def test_regression_support_stack_matches_per_dataset_lookup(self):
        model = CellModeling.__new__(CellModeling)
        model.device = torch.device("cpu")
        model.cell_model_schema = "main_id_arc_offset_poly12"
        model.support_f_delay = torch.tensor([[False, True]], dtype=torch.bool)
        model.support_r_delay = torch.tensor([[True], [False]], dtype=torch.bool)
        model.support_f_trans = torch.tensor([[False, False, True]], dtype=torch.bool)
        model.support_r_trans = torch.tensor([[True, False]], dtype=torch.bool)

        model._rebuild_regression_support_stack()

        main_ids = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)
        arc_offsets = torch.tensor([1, 0, 0, 2, 9], dtype=torch.long)
        arc_types = torch.tensor([0, 1, 1, 2, 3], dtype=torch.long)

        support = model.supports_arc_types(main_ids, arc_offsets, arc_types)

        self.assertTrue(torch.equal(
            support,
            torch.tensor([True, True, False, True, False], dtype=torch.bool),
        ))


if __name__ == "__main__":
    unittest.main()
