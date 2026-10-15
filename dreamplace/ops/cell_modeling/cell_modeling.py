import logging
import hashlib
import json
import os
import math
import time
from datetime import datetime, timezone

import torch
from torch import nn

try:
    from dreamplace.ops.cell_modeling import cell_modeling_op
except ImportError:
    cell_modeling_op = None

try:
    from dreamplace.ops.timing_propagation import lut_entry_2d_op
except ImportError:
    lut_entry_2d_op = None


MODEL_VERSION = "v4"
FEATURE_VERSION = "physics_regression_v1"
LINEAR6_SCHEMA = "main_id_arc_offset_linear6"
POLY12_SCHEMA = "main_id_arc_offset_poly12"
LOGCROSS12_SCHEMA = "main_id_arc_offset_logcross12"
LOG_LEAKAGE_POLY12_SCHEMA = "main_id_arc_offset_log_leakage_poly12"
PIECEWISE_LINEAR_SCHEMA = "main_id_arc_offset_piecewise_linear"
CELL_MODEL_SCHEMA = PIECEWISE_LINEAR_SCHEMA
# Disk cache is opt-in: a fresh fit costs only a few seconds, so by default
# nothing is written to disk. Set params.surrogate_cache_root to enable.
DEFAULT_CACHE_ROOT = None
LINEAR6_STE_GRADIENT_MODE = "linear6_ste"
NATIVE_PIECEWISE_GRADIENT_MODE = "native_piecewise"
DEFAULT_PIECEWISE_GRADIENT_MODE = NATIVE_PIECEWISE_GRADIENT_MODE
LUT_BOUNDARY_MODE_CLAMP = "clamp"
LUT_BOUNDARY_MODE_EXTRAPOLATE = "extrapolate"
LUT_BOUNDARY_MODES = (LUT_BOUNDARY_MODE_CLAMP, LUT_BOUNDARY_MODE_EXTRAPOLATE)
DEFAULT_LUT_BOUNDARY_MODE = LUT_BOUNDARY_MODE_EXTRAPOLATE
LUT_BOUNDARY_MODE_ENV = "AIMP_CELL_MODEL_LUT_BOUNDARY_MODE"
DATASET_SPECS = {
    "f_delay": {
        "luts_attr": "f_delay_luts",
        "arc_type": 0,
        "metric_kind": "delay",
    },
    "r_delay": {
        "luts_attr": "r_delay_luts",
        "arc_type": 1,
        "metric_kind": "delay",
    },
    "f_trans": {
        "luts_attr": "f_trans_luts",
        "arc_type": 2,
        "metric_kind": "slew",
    },
    "r_trans": {
        "luts_attr": "r_trans_luts",
        "arc_type": 3,
        "metric_kind": "slew",
    },
}

REGRESSION_SCHEMAS = {
    LINEAR6_SCHEMA,
    POLY12_SCHEMA,
    LOGCROSS12_SCHEMA,
    LOG_LEAKAGE_POLY12_SCHEMA,
}
ARC_TYPE_TO_DATASET = {
    0: "f_delay",
    1: "r_delay",
    2: "f_trans",
    3: "r_trans",
}


def _compute_r2(sum_squared_error, target_sum, target_squared_sum, count, eps=1e-12):
    if count <= 0:
        return 0.0
    total_sum_squares = target_squared_sum - (target_sum * target_sum) / float(count)
    if total_sum_squares <= eps:
        return 1.0 if sum_squared_error <= eps else 0.0
    return 1.0 - (sum_squared_error / total_sum_squares)

class CellModeling(nn.Module):
    '''
    Cell modeling module implementing Physics-Aware Regression Model.
    Reference: analyze_lut_data.py (Model B)
    
    Formulation:
    Delay = w0 + w1*Slew + w2*Cap + w3*(1/Size) + w4*(Cap/Size) + w5*VT
    
    Supports differentiable delay calculation w.r.t VT, Size, Slew, Cap.
    '''
    def __init__(self, data_collections):
        super(CellModeling, self).__init__()
        self.data_collections = data_collections
        self.device = data_collections.device
        self.cache_root = (
            getattr(data_collections, "surrogate_cache_root", DEFAULT_CACHE_ROOT) or None
        )
        self.fit_num_threads = int(getattr(data_collections, "cell_modeling_num_threads", 1))
        self.cell_model_schema = getattr(data_collections, "cell_model_schema", CELL_MODEL_SCHEMA) or CELL_MODEL_SCHEMA
        self.piecewise_gradient_mode = (
            getattr(data_collections, "piecewise_gradient_mode", DEFAULT_PIECEWISE_GRADIENT_MODE)
            or DEFAULT_PIECEWISE_GRADIENT_MODE
        )
        self.cell_model_lut_boundary_mode = self._resolve_lut_boundary_mode(
            getattr(data_collections, "cell_model_lut_boundary_mode", None)
        )
        self._active_dataset_name = "unknown"
        self.cache_event = None
        self.cache_reason = None
        self.cache_history = []
        self.cache_meta = None
        self.vt_order_tensor = None
        self.piecewise_state = None
        self.piecewise_linear_coeffs = None
        self.use_poly12_op = self._profile_flag_enabled(os.environ.get("AIMP_USE_CELL_MODEL_POLY12_OP", ""))
        self.support_f_delay = None
        self.support_r_delay = None
        self.support_f_trans = None
        self.support_r_trans = None
        self.regression_support_stack = None
        self.log_leakage_coord = None
        self.log_leakage_coord_table = {}
        self.log_leakage_family_candidates = {}
        self.log_leakage_coord_stats = {}
        self._lut_runtime_lookup_cache = {}
        self._init_forward_profile(data_collections)
        
        # Load Data
        # We need to reconstruct the training data from the LUTs and Cell Info
        self.build_models(data_collections)

    @staticmethod
    def _profile_flag_enabled(raw_value):
        if isinstance(raw_value, bool):
            return raw_value
        text = str(raw_value or "").strip().lower()
        return text not in {"", "0", "false", "no", "off"}

    @staticmethod
    def _resolve_lut_boundary_mode(raw_value):
        mode = raw_value
        if mode is None or str(mode).strip() == "":
            mode = os.environ.get(LUT_BOUNDARY_MODE_ENV, DEFAULT_LUT_BOUNDARY_MODE)
        mode = str(mode).strip().lower()
        if mode not in LUT_BOUNDARY_MODES:
            raise ValueError(
                f"unsupported cell_model_lut_boundary_mode: {mode!r}; "
                f"expected one of {LUT_BOUNDARY_MODES}"
            )
        return mode

    @staticmethod
    def _profile_log_every(raw_value, default_value=100):
        text = str(raw_value or "").strip()
        if text == "":
            return default_value
        try:
            return max(1, int(text))
        except (TypeError, ValueError):
            logging.warning(
                "Invalid cell-model forward profile log-every value %r, fallback to %d",
                raw_value,
                default_value,
            )
            return default_value

    def _init_forward_profile(self, data_collections):
        enabled_value = getattr(data_collections, "cell_model_forward_profile", None)
        if enabled_value is None:
            enabled_value = os.environ.get("AUTODMP_CELL_MODEL_FORWARD_PROFILE", "")
        log_every_value = getattr(data_collections, "cell_model_forward_profile_log_every", None)
        if log_every_value is None:
            log_every_value = os.environ.get("AUTODMP_CELL_MODEL_FORWARD_PROFILE_LOG_EVERY", "")

        self._cell_model_forward_profile_config_enabled = self._profile_flag_enabled(enabled_value)
        self._cell_model_forward_profile_enabled = self._cell_model_forward_profile_config_enabled
        self._cell_model_forward_profile_log_every = self._profile_log_every(log_every_value)
        self.reset_forward_profile(enabled=self._cell_model_forward_profile_enabled)

    def reset_forward_profile(self, enabled=None):
        if enabled is not None:
            self._cell_model_forward_profile_enabled = (
                bool(enabled)
                or bool(getattr(self, "_cell_model_forward_profile_config_enabled", False))
            )
        self._cell_model_forward_profile_calls = 0
        self._cell_model_forward_profile_total_seconds = 0.0
        self._cell_model_forward_profile_total_arcs = 0
        self._cell_model_forward_profile_detail = {
            "cell_model_forward_calls": 0,
            "cell_model_forward_total_ms": 0.0,
            "cell_model_forward_total_arcs": 0,
            "cell_model_forward_selected_arc_type_calls": 0,
            "cell_model_forward_all_dataset_calls": 0,
            "cell_model_forward_selected_arc_type_unique_sum": 0,
            "cell_model_forward_dataset_calls": 0,
            "cell_model_forward_dataset_calls_by_dataset": {},
            "piecewise_forward_dataset_calls": 0,
            "piecewise_forward_dataset_calls_by_dataset": {},
            "piecewise_size_interp_calls": 0,
            "piecewise_size_interp_calls_by_dataset": {},
            "piecewise_size_interp_fused_calls": 0,
            "piecewise_size_interp_python_calls": 0,
            "piecewise_size_interp_arcs": 0,
            "piecewise_size_interp_fused_arcs": 0,
            "piecewise_size_interp_python_arcs": 0,
            "piecewise_vt_second_pass_calls": 0,
            "piecewise_vt_second_pass_arcs": 0,
        }

    def get_forward_profile(self):
        detail = dict(getattr(self, "_cell_model_forward_profile_detail", {}) or {})
        for key in (
            "cell_model_forward_dataset_calls_by_dataset",
            "piecewise_forward_dataset_calls_by_dataset",
            "piecewise_size_interp_calls_by_dataset",
        ):
            detail[key] = dict(detail.get(key, {}) or {})
        forward_calls = max(1, int(detail.get("cell_model_forward_calls", 0) or 0))
        forward_arcs = max(1, int(detail.get("cell_model_forward_total_arcs", 0) or 0))
        interp_calls = max(1, int(detail.get("piecewise_size_interp_calls", 0) or 0))
        interp_arcs = max(1, int(detail.get("piecewise_size_interp_arcs", 0) or 0))
        detail["cell_model_forward_avg_ms"] = (
            float(detail.get("cell_model_forward_total_ms", 0.0)) / float(forward_calls)
        )
        detail["cell_model_forward_avg_us_per_arc"] = (
            float(detail.get("cell_model_forward_total_ms", 0.0)) * 1000.0 / float(forward_arcs)
        )
        detail["piecewise_size_interp_avg_arcs_per_call"] = (
            float(detail.get("piecewise_size_interp_arcs", 0) or 0) / float(interp_calls)
        )
        detail["piecewise_size_interp_fused_call_ratio"] = (
            float(detail.get("piecewise_size_interp_fused_calls", 0) or 0) / float(interp_calls)
        )
        detail["piecewise_vt_second_pass_arc_ratio"] = (
            float(detail.get("piecewise_vt_second_pass_arcs", 0) or 0) / float(interp_arcs)
        )
        return detail

    def _profile_detail_enabled(self):
        return bool(getattr(self, "_cell_model_forward_profile_enabled", False))

    def _profile_increment(self, key, value=1):
        if not self._profile_detail_enabled():
            return
        detail = getattr(self, "_cell_model_forward_profile_detail", None)
        if detail is None:
            self.reset_forward_profile()
            detail = self._cell_model_forward_profile_detail
        detail[key] = detail.get(key, 0) + value

    def _profile_increment_dataset(self, key, dataset_name, value=1):
        if not self._profile_detail_enabled():
            return
        detail = getattr(self, "_cell_model_forward_profile_detail", None)
        if detail is None:
            self.reset_forward_profile()
            detail = self._cell_model_forward_profile_detail
        dataset_counts = detail.setdefault(key, {})
        dataset_counts[dataset_name] = int(dataset_counts.get(dataset_name, 0)) + int(value)

    def _record_forward_profile(self, batch_size, arc_types, elapsed_seconds):
        self._profile_increment("cell_model_forward_calls")
        self._profile_increment("cell_model_forward_total_ms", float(elapsed_seconds) * 1000.0)
        self._profile_increment("cell_model_forward_total_arcs", int(batch_size))
        if arc_types is None:
            self._profile_increment("cell_model_forward_all_dataset_calls")
        else:
            self._profile_increment("cell_model_forward_selected_arc_type_calls")
            try:
                self._profile_increment(
                    "cell_model_forward_selected_arc_type_unique_sum",
                    int(torch.unique(arc_types).numel()),
                )
            except RuntimeError:
                pass

        self._cell_model_forward_profile_calls = int(
            getattr(self, "_cell_model_forward_profile_calls", 0)
        ) + 1
        self._cell_model_forward_profile_total_seconds = float(
            getattr(self, "_cell_model_forward_profile_total_seconds", 0.0)
        ) + float(elapsed_seconds)
        self._cell_model_forward_profile_total_arcs = int(
            getattr(self, "_cell_model_forward_profile_total_arcs", 0)
        ) + int(batch_size)

        log_every = max(1, int(getattr(self, "_cell_model_forward_profile_log_every", 100)))
        if self._cell_model_forward_profile_calls % log_every != 0:
            return

        total_seconds = float(self._cell_model_forward_profile_total_seconds)
        total_arcs = max(1, int(self._cell_model_forward_profile_total_arcs))
        mode = "selected_arc_types" if arc_types is not None else "all_datasets"
        logging.info(
            "CellModeling.forward profile: schema=%s mode=%s batch=%d calls=%d "
            "last_ms=%.3f total_ms=%.3f avg_ms=%.3f batch_us_per_arc=%.3f avg_us_per_arc=%.3f",
            self.cell_model_schema,
            mode,
            int(batch_size),
            int(self._cell_model_forward_profile_calls),
            float(elapsed_seconds) * 1e3,
            total_seconds * 1e3,
            total_seconds * 1e3 / float(self._cell_model_forward_profile_calls),
            float(elapsed_seconds) * 1e6 / float(max(1, int(batch_size))),
            total_seconds * 1e6 / float(total_arcs),
        )

    def _dataset_arc_metadata(self, dataset_name):
        spec = self._dataset_spec(dataset_name)
        flat_libcell_info = self.data_collections.flat_libcell_info.to(self.device)
        flat_libarc_info = self.data_collections.flat_libarc_info.to(self.device)
        luts_info = getattr(self.data_collections.arcs_info, spec["luts_attr"])
        num_dataset_arcs = int(luts_info.flat_luts_dim.shape[0])
        if num_dataset_arcs > int(flat_libarc_info.shape[0]):
            raise ValueError(
                f"{dataset_name} LUT rows ({num_dataset_arcs}) exceed flat_libarc_info rows ({int(flat_libarc_info.shape[0])})"
            )
        dataset_arc_info = flat_libarc_info[:num_dataset_arcs]
        libcell_indices = dataset_arc_info[:, 2].long()
        arc_offsets = dataset_arc_info[:, 3].long()
        return {
            "libcell_indices": libcell_indices,
            "arc_offsets": arc_offsets,
            "libcell_main_ids": flat_libcell_info[libcell_indices, 1].long(),
            # Historical field name: values are the unified timing coordinate.
            "sizes": flat_libcell_info[libcell_indices, 2].float(),
            "log_leakage_coords": self._log_leakage_coords_for_libcells(libcell_indices),
            "vts": flat_libcell_info[libcell_indices, 3].float(),
            "num_dataset_arcs": num_dataset_arcs,
        }

    def _schema_uses_log_leakage_coord(self, schema=None):
        schema = schema or self.cell_model_schema
        return schema == LOG_LEAKAGE_POLY12_SCHEMA

    def _leakage_tensor(self, data_collections):
        leakage = getattr(data_collections, "flat_libcell_leakage", None)
        if leakage is None:
            raise ValueError(
                f"{LOG_LEAKAGE_POLY12_SCHEMA} requires flat_libcell_leakage"
            )
        if not isinstance(leakage, torch.Tensor):
            leakage = torch.tensor(leakage, device=self.device)
        return leakage.to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _size_key(value):
        return round(float(value), 9)

    def _flat_libcell_log_leakage_coord(self, flat_libcell_info, data_collections):
        leakage = self._leakage_tensor(data_collections)
        flat_libcell_info = flat_libcell_info.to(device=self.device)
        if int(leakage.numel()) != int(flat_libcell_info.shape[0]):
            raise ValueError(
                "flat_libcell_leakage length must match flat_libcell_info rows "
                f"({int(leakage.numel())} vs {int(flat_libcell_info.shape[0])})"
            )
        if not torch.isfinite(leakage).all():
            raise ValueError(f"{LOG_LEAKAGE_POLY12_SCHEMA} requires finite leakage values")
        if torch.any(leakage < 0):
            raise ValueError(f"{LOG_LEAKAGE_POLY12_SCHEMA} requires non-negative leakage values")

        positive = leakage[leakage > 0]
        if positive.numel() == 0:
            raise ValueError(f"{LOG_LEAKAGE_POLY12_SCHEMA} requires at least one positive leakage value")
        eps = max(float(positive.min().item()) * 1e-6, 1e-30)
        raw = torch.log(torch.clamp(leakage, min=eps))
        coord = torch.ones_like(raw)
        duplicate_count = 0
        family_count = 0
        tiny = 1e-12
        main_ids = flat_libcell_info[:, 1].long()
        vts = flat_libcell_info[:, 3].long()
        family_ids = main_ids * 100000 + vts
        for family_id in torch.unique(family_ids).tolist():
            family_mask = family_ids == int(family_id)
            if not torch.any(family_mask):
                continue
            family_count += 1
            family_raw = raw[family_mask]
            unique_raw = torch.unique(family_raw)
            duplicate_count += int(family_raw.numel() - unique_raw.numel())
            raw_min = family_raw.min()
            raw_max = family_raw.max()
            span = raw_max - raw_min
            if float(span.abs().item()) <= tiny:
                coord[family_mask] = 1.0
            else:
                coord[family_mask] = 1.0 + (family_raw - raw_min) / span
        self.log_leakage_coord_stats = {
            "log_leakage_coord_mode": "family_normalized_log_leakage_1_to_2",
            "leakage_eps": eps,
            "num_zero_leakage_values": int(torch.count_nonzero(leakage == 0).item()),
            "num_duplicate_leakage_values": duplicate_count,
            "num_log_leakage_families": family_count,
            "log_leakage_coord_min": float(coord.min().item()) if coord.numel() else 0.0,
            "log_leakage_coord_max": float(coord.max().item()) if coord.numel() else 0.0,
        }
        return coord

    def _build_log_leakage_coord_lookup(self, flat_libcell_info, log_leakage_coord):
        table = {}
        candidates = {}
        for libcell_idx in range(int(flat_libcell_info.shape[0])):
            row = flat_libcell_info[libcell_idx]
            main_id = int(row[1].item())
            size = self._size_key(row[2].item())
            vt = int(row[3].item())
            coord = float(log_leakage_coord[libcell_idx].item())
            key = (main_id, vt, size)
            table.setdefault(key, coord)
            candidates.setdefault((main_id, vt), []).append((size, coord))
        for key, values in candidates.items():
            values.sort(key=lambda item: item[0])
        return table, candidates

    def _prepare_log_leakage_coord_state(self, flat_libcell_info, data_collections):
        if not self._schema_uses_log_leakage_coord():
            self.log_leakage_coord = None
            self.log_leakage_coord_table = {}
            self.log_leakage_family_candidates = {}
            self.log_leakage_coord_stats = {}
            return
        coord = self._flat_libcell_log_leakage_coord(flat_libcell_info, data_collections)
        table, candidates = self._build_log_leakage_coord_lookup(flat_libcell_info, coord)
        self.log_leakage_coord = coord
        self.log_leakage_coord_table = table
        self.log_leakage_family_candidates = candidates

    def _log_leakage_coords_for_libcells(self, libcell_indices):
        if self.log_leakage_coord is None:
            return None
        return self.log_leakage_coord[libcell_indices.long()].float()

    def _regression_size_coord_from_meta(self, meta, schema=None):
        schema = schema or self.cell_model_schema
        if self._schema_uses_log_leakage_coord(schema):
            coord = meta.get("log_leakage_coords")
            if coord is None:
                raise ValueError(f"{schema} requires log_leakage_coords metadata")
            return coord
        return meta["sizes"]

    def _forward_size_coord(self, schema, libcell_main_id, vt, size):
        if not self._schema_uses_log_leakage_coord(schema):
            return size
        if getattr(self, "_analysis_size_coord_is_precomputed", False):
            return size
        result = torch.empty_like(size.float())
        main_cpu = libcell_main_id.detach().cpu().long().tolist()
        vt_cpu = vt.detach().cpu().round().long().tolist()
        size_cpu = size.detach().cpu().tolist()
        for idx, (main_id, vt_value, size_value) in enumerate(zip(main_cpu, vt_cpu, size_cpu)):
            key = (int(main_id), int(vt_value), self._size_key(size_value))
            coord = self.log_leakage_coord_table.get(key)
            if coord is None:
                family = self.log_leakage_family_candidates.get((int(main_id), int(vt_value)), [])
                if family:
                    coord = min(family, key=lambda item: abs(item[0] - float(size_value)))[1]
                else:
                    coord = float(size_value)
            result[idx] = coord
        return result.to(device=size.device, dtype=size.dtype)

    def build_models(self, data_collections):
        logging.info("Building Cell Models (Physics-Aware Regression)...")
        
        # 1. Gather Cell/Arc Info
        # flat_libcell_info: [libcell_name_id, main_type_id, timing_coordinate, vt_index]
        # Note: We assume these are numeric. If BasicPlace.py loaded them as Tensor, we are good.
        # If they are lists containing strings, we need to handle that. 
        # Assuming the user followed the README plan or BasicPlace handles it.
        # For robustness, we try to access them as attributes.
        
        try:
            flat_libcell_info = data_collections.flat_libcell_info
            if not isinstance(flat_libcell_info, torch.Tensor):
                # Fallback: Try to convert list to tensor (assuming numeric)
                # If it contains strings, this will fail, but we can't fix C++ here.
                flat_libcell_info = torch.tensor(flat_libcell_info, device=self.device)
            else:
                flat_libcell_info = flat_libcell_info.to(self.device)
                
            flat_libarc_info = data_collections.flat_libarc_info
            if not isinstance(flat_libarc_info, torch.Tensor):
                flat_libarc_info = torch.tensor(flat_libarc_info, device=self.device)
            else:
                flat_libarc_info = flat_libarc_info.to(self.device)
                
        except Exception as e:
            logging.error(f"Failed to load cell/arc info tensors: {e}")
            raise e

        self._prepare_log_leakage_coord_state(flat_libcell_info, data_collections)
        cache_meta = self._cache_metadata(flat_libcell_info, flat_libarc_info, data_collections)
        self.cache_meta = cache_meta
        if self._try_load_cache(cache_meta):
            return

        # 2. Map Global Arc ID to Properties
        # flat_libarc_info: [src, dst, libcell_idx, arc_offset]
        libcell_indices = flat_libarc_info[:, 2].long()
        arc_offsets = flat_libarc_info[:, 3].long()
        
        # flat_libcell_info: [name, main_type, timing_coordinate, vt]
        libcell_main_ids = flat_libcell_info[libcell_indices, 1].long()
        # Historical variable name: values are the unified timing coordinate.
        sizes = flat_libcell_info[libcell_indices, 2].float()
        vts = flat_libcell_info[libcell_indices, 3].float() # Treat VT index as continuous for regression
        
        self.num_libcell_main_ids = int(libcell_main_ids.max().item()) + 1
        self.max_arc_offset = int(arc_offsets.max().item()) + 1
        self.vt_order_tensor = torch.tensor(cache_meta["vt_order"], device=self.device, dtype=torch.float32)

        if self.cell_model_schema in REGRESSION_SCHEMAS:
            previous_threads = torch.get_num_threads()
            fit_threads = max(1, self.fit_num_threads)
            if previous_threads != fit_threads:
                logging.info(
                    "CellModeling fit temporarily sets torch threads from %d to %d",
                    previous_threads,
                    fit_threads,
                )
                torch.set_num_threads(fit_threads)
            try:
                self._active_dataset_name = "f_delay"
                meta = self._dataset_arc_metadata("f_delay")
                self.coeff_f_delay = self._fit_dataset(
                    data_collections.arcs_info.f_delay_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    self._regression_size_coord_from_meta(meta),
                    meta["vts"],
                )
                self._active_dataset_name = "r_delay"
                meta = self._dataset_arc_metadata("r_delay")
                self.coeff_r_delay = self._fit_dataset(
                    data_collections.arcs_info.r_delay_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    self._regression_size_coord_from_meta(meta),
                    meta["vts"],
                )
                self._active_dataset_name = "f_trans"
                meta = self._dataset_arc_metadata("f_trans")
                self.coeff_f_trans = self._fit_dataset(
                    data_collections.arcs_info.f_trans_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    self._regression_size_coord_from_meta(meta),
                    meta["vts"],
                )
                self._active_dataset_name = "r_trans"
                meta = self._dataset_arc_metadata("r_trans")
                self.coeff_r_trans = self._fit_dataset(
                    data_collections.arcs_info.r_trans_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    self._regression_size_coord_from_meta(meta),
                    meta["vts"],
                )
            finally:
                self._active_dataset_name = "unknown"
                if previous_threads != fit_threads:
                    torch.set_num_threads(previous_threads)

            self.coeff_f_delay = self._reshape_coeffs(self.coeff_f_delay, schema=self.cell_model_schema)
            self.coeff_r_delay = self._reshape_coeffs(self.coeff_r_delay, schema=self.cell_model_schema)
            self.coeff_f_trans = self._reshape_coeffs(self.coeff_f_trans, schema=self.cell_model_schema)
            self.coeff_r_trans = self._reshape_coeffs(self.coeff_r_trans, schema=self.cell_model_schema)
            self.support_f_delay = self._reshape_support_mask(self.coeff_f_delay)
            self.support_r_delay = self._reshape_support_mask(self.coeff_r_delay)
            self.support_f_trans = self._reshape_support_mask(self.coeff_f_trans)
            self.support_r_trans = self._reshape_support_mask(self.coeff_r_trans)
            self._rebuild_regression_support_stack()
        elif self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA:
            previous_threads = torch.get_num_threads()
            fit_threads = max(1, self.fit_num_threads)
            if previous_threads != fit_threads:
                logging.info(
                    "CellModeling fit temporarily sets torch threads from %d to %d",
                    previous_threads,
                    fit_threads,
                )
                torch.set_num_threads(fit_threads)
            try:
                self._active_dataset_name = "f_delay"
                meta = self._dataset_arc_metadata("f_delay")
                coeff_f_delay = self._fit_dataset(
                    data_collections.arcs_info.f_delay_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    meta["sizes"],
                    meta["vts"],
                    schema=LINEAR6_SCHEMA,
                )
                self._active_dataset_name = "r_delay"
                meta = self._dataset_arc_metadata("r_delay")
                coeff_r_delay = self._fit_dataset(
                    data_collections.arcs_info.r_delay_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    meta["sizes"],
                    meta["vts"],
                    schema=LINEAR6_SCHEMA,
                )
                self._active_dataset_name = "f_trans"
                meta = self._dataset_arc_metadata("f_trans")
                coeff_f_trans = self._fit_dataset(
                    data_collections.arcs_info.f_trans_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    meta["sizes"],
                    meta["vts"],
                    schema=LINEAR6_SCHEMA,
                )
                self._active_dataset_name = "r_trans"
                meta = self._dataset_arc_metadata("r_trans")
                coeff_r_trans = self._fit_dataset(
                    data_collections.arcs_info.r_trans_luts,
                    meta["libcell_main_ids"],
                    meta["arc_offsets"],
                    meta["sizes"],
                    meta["vts"],
                    schema=LINEAR6_SCHEMA,
                )
            finally:
                self._active_dataset_name = "unknown"
                if previous_threads != fit_threads:
                    torch.set_num_threads(previous_threads)
            self.piecewise_state = {}
            for dataset_name in DATASET_SPECS:
                meta = self._dataset_arc_metadata(dataset_name)
                self.piecewise_state[dataset_name] = self._build_piecewise_state(
                    libcell_main_ids=meta["libcell_main_ids"],
                    arc_offsets=meta["arc_offsets"],
                    sizes=meta["sizes"],
                    vts=meta["vts"],
                    num_main_ids=self.num_libcell_main_ids,
                    max_arc_offset=self.max_arc_offset,
                    vt_order=cache_meta["vt_order"],
                )
            self.piecewise_linear_coeffs = {
                "coeff_f_delay": self._reshape_coeffs(coeff_f_delay, schema=LINEAR6_SCHEMA),
                "coeff_r_delay": self._reshape_coeffs(coeff_r_delay, schema=LINEAR6_SCHEMA),
                "coeff_f_trans": self._reshape_coeffs(coeff_f_trans, schema=LINEAR6_SCHEMA),
                "coeff_r_trans": self._reshape_coeffs(coeff_r_trans, schema=LINEAR6_SCHEMA),
            }
        else:
            raise ValueError(f"unsupported cell_model_schema: {self.cell_model_schema}")

        self._save_cache(cache_meta)
        
        logging.info("Cell Models Built Successfully.")

    def _record_cache_event(self, event, cache_dir, reason=None):
        self.cache_event = event
        self.cache_reason = reason
        self.cache_history.append({"event": event, "reason": reason, "cache_dir": cache_dir})
        if reason:
            logging.info("CellModeling cache %s at %s (%s)", event, cache_dir, reason)
        else:
            logging.info("CellModeling cache %s at %s", event, cache_dir)

    def _tensor_payload_bytes(self, tensor):
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        return cpu_tensor.numpy().tobytes()

    def _hash_tensors(self, *tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(self._tensor_payload_bytes(tensor))
        return digest.hexdigest()

    def _vt_order(self, flat_libcell_info):
        unique_vts = torch.unique(flat_libcell_info[:, 3].long().to("cpu"), sorted=True)
        return [int(v.item()) for v in unique_vts]

    def _cache_metadata(self, flat_libcell_info, flat_libarc_info, data_collections):
        vt_order = self._vt_order(flat_libcell_info)
        arcs_info = data_collections.arcs_info
        hash_tensors = [
            flat_libcell_info,
            flat_libarc_info,
            arcs_info.f_delay_luts.flat_luts_values,
            arcs_info.f_delay_luts.flat_luts_trans_table,
            arcs_info.f_delay_luts.flat_luts_cap_table,
            arcs_info.f_delay_luts.flat_luts_dim,
            arcs_info.r_delay_luts.flat_luts_values,
            arcs_info.r_delay_luts.flat_luts_trans_table,
            arcs_info.r_delay_luts.flat_luts_cap_table,
            arcs_info.r_delay_luts.flat_luts_dim,
            arcs_info.f_trans_luts.flat_luts_values,
            arcs_info.f_trans_luts.flat_luts_trans_table,
            arcs_info.f_trans_luts.flat_luts_cap_table,
            arcs_info.f_trans_luts.flat_luts_dim,
            arcs_info.r_trans_luts.flat_luts_values,
            arcs_info.r_trans_luts.flat_luts_trans_table,
            arcs_info.r_trans_luts.flat_luts_cap_table,
            arcs_info.r_trans_luts.flat_luts_dim,
        ]
        if self._schema_uses_log_leakage_coord():
            hash_tensors.append(self._leakage_tensor(data_collections))
        lib_hash = self._hash_tensors(*hash_tensors)
        vt_order_hash = hashlib.sha256(json.dumps(vt_order).encode("utf-8")).hexdigest()
        if self.cache_root:
            if self.cell_model_schema == LINEAR6_SCHEMA:
                cache_dir = os.path.join(self.cache_root, lib_hash)
            else:
                cache_dir = os.path.join(self.cache_root, lib_hash, self.cell_model_schema)
            manifest_path = os.path.join(cache_dir, "manifest.json")
            weights_path = os.path.join(cache_dir, "weights.pt")
        else:
            cache_dir = None
            manifest_path = None
            weights_path = None
        return {
            "cache_dir": cache_dir,
            "manifest_path": manifest_path,
            "weights_path": weights_path,
            "lib_hash": lib_hash,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "vt_order": vt_order,
            "vt_order_hash": vt_order_hash,
            "num_main_ids": int(flat_libcell_info[:, 1].max().item()) + 1 if flat_libcell_info.numel() else 0,
            "num_vt_types": len(vt_order),
            "cell_model_schema": self.cell_model_schema,
            "num_libcells": int(flat_libcell_info.shape[0]),
            "num_arcs": int(flat_libarc_info.shape[0]),
            "log_leakage_coord_stats": dict(getattr(self, "log_leakage_coord_stats", {}) or {}),
        }

    def _coeff_state_dict(self):
        if self.cell_model_schema in REGRESSION_SCHEMAS:
            return {
                "coeff_f_delay": self.coeff_f_delay.detach().to("cpu"),
                "coeff_r_delay": self.coeff_r_delay.detach().to("cpu"),
                "coeff_f_trans": self.coeff_f_trans.detach().to("cpu"),
                "coeff_r_trans": self.coeff_r_trans.detach().to("cpu"),
                "support_f_delay": self.support_f_delay.detach().to("cpu"),
                "support_r_delay": self.support_r_delay.detach().to("cpu"),
                "support_f_trans": self.support_f_trans.detach().to("cpu"),
                "support_r_trans": self.support_r_trans.detach().to("cpu"),
            }
        if self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA:
            state_dict = {
                "piecewise_coeff_f_delay": self.piecewise_linear_coeffs["coeff_f_delay"].detach().to("cpu"),
                "piecewise_coeff_r_delay": self.piecewise_linear_coeffs["coeff_r_delay"].detach().to("cpu"),
                "piecewise_coeff_f_trans": self.piecewise_linear_coeffs["coeff_f_trans"].detach().to("cpu"),
                "piecewise_coeff_r_trans": self.piecewise_linear_coeffs["coeff_r_trans"].detach().to("cpu"),
            }
            for dataset_name in DATASET_SPECS:
                state = self.piecewise_state[dataset_name]
                prefix = f"piecewise_{dataset_name}"
                state_dict[f"{prefix}_size_table"] = state["size_table"].detach().to("cpu")
                state_dict[f"{prefix}_arc_index_table"] = state["arc_index_table"].detach().to("cpu")
                state_dict[f"{prefix}_size_count"] = state["size_count"].detach().to("cpu")
                state_dict[f"{prefix}_vt_available"] = state["vt_available"].detach().to("cpu")
            return state_dict
        raise ValueError(f"unsupported cell_model_schema: {self.cell_model_schema}")

    def _expected_weight_shapes(self):
        return {
            key: tuple(value.shape)
            for key, value in self._coeff_state_dict().items()
        }

    def _manifest_payload(self, cache_meta):
        return {
            "lib_hash": cache_meta["lib_hash"],
            "model_version": cache_meta["model_version"],
            "feature_version": cache_meta["feature_version"],
            "vt_order": cache_meta["vt_order"],
            "vt_order_hash": cache_meta["vt_order_hash"],
            "num_main_ids": cache_meta["num_main_ids"],
            "num_vt_types": cache_meta["num_vt_types"],
            "num_libcells": cache_meta["num_libcells"],
            "num_arcs": cache_meta["num_arcs"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cell_model_schema": cache_meta["cell_model_schema"],
            "log_leakage_coord_stats": cache_meta.get("log_leakage_coord_stats", {}),
            "weight_shapes": {
                key: list(value)
                for key, value in self._expected_weight_shapes().items()
            },
        }

    def _shape_matches(self, state_dict, manifest):
        manifest_shapes = manifest.get("weight_shapes")
        if not isinstance(manifest_shapes, dict):
            return False
        for key, expected_shape in manifest_shapes.items():
            tensor = state_dict.get(key)
            if tensor is None or tuple(tensor.shape) != tuple(expected_shape):
                return False
        return True

    def _manifest_matches(self, manifest, cache_meta):
        checks = (
            ("lib_hash", cache_meta["lib_hash"]),
            ("model_version", cache_meta["model_version"]),
            ("feature_version", cache_meta["feature_version"]),
            ("vt_order_hash", cache_meta["vt_order_hash"]),
            ("vt_order", cache_meta["vt_order"]),
            ("cell_model_schema", cache_meta["cell_model_schema"]),
            ("num_main_ids", cache_meta["num_main_ids"]),
            ("num_vt_types", cache_meta["num_vt_types"]),
            ("num_libcells", cache_meta["num_libcells"]),
            ("num_arcs", cache_meta["num_arcs"]),
        )
        if self._schema_uses_log_leakage_coord():
            checks = checks + (
                ("log_leakage_coord_stats", cache_meta.get("log_leakage_coord_stats", {})),
            )
        for key, expected in checks:
            if manifest.get(key) != expected:
                return False, key
        return True, None

    def _try_load_cache(self, cache_meta):
        manifest_path = cache_meta["manifest_path"]
        weights_path = cache_meta["weights_path"]
        if not cache_meta["cache_dir"]:
            self._record_cache_event("disabled", None, "cache_root_unset")
            return False
        if not os.path.isdir(cache_meta["cache_dir"]):
            self._record_cache_event("miss", cache_meta["cache_dir"], "cache_dir_missing")
            return False
        if not os.path.isfile(manifest_path) or not os.path.isfile(weights_path):
            missing = "manifest_missing" if not os.path.isfile(manifest_path) else "weights_missing"
            self._record_cache_event("miss", cache_meta["cache_dir"], missing)
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_ok, mismatch_key = self._manifest_matches(manifest, cache_meta)
            if not manifest_ok:
                self._record_cache_event("miss", cache_meta["cache_dir"], "manifest_mismatch:%s" % mismatch_key)
                return False
            state_dict = torch.load(weights_path, map_location="cpu")
            if not self._shape_matches(state_dict, manifest):
                self._record_cache_event("miss", cache_meta["cache_dir"], "weight_shape_mismatch")
                return False
            self.num_libcell_main_ids = int(cache_meta["num_main_ids"])
            self.vt_order_tensor = torch.tensor(cache_meta["vt_order"], device=self.device, dtype=torch.float32)
            if self.cell_model_schema in REGRESSION_SCHEMAS:
                self.coeff_f_delay = state_dict["coeff_f_delay"].to(self.device)
                self.coeff_r_delay = state_dict["coeff_r_delay"].to(self.device)
                self.coeff_f_trans = state_dict["coeff_f_trans"].to(self.device)
                self.coeff_r_trans = state_dict["coeff_r_trans"].to(self.device)
                self.support_f_delay = state_dict["support_f_delay"].to(self.device).bool()
                self.support_r_delay = state_dict["support_r_delay"].to(self.device).bool()
                self.support_f_trans = state_dict["support_f_trans"].to(self.device).bool()
                self.support_r_trans = state_dict["support_r_trans"].to(self.device).bool()
                self._rebuild_regression_support_stack()
                self.max_arc_offset = self.coeff_f_delay.shape[1]
            elif self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA:
                required_piecewise_keys = {
                    "piecewise_coeff_f_delay",
                    "piecewise_coeff_r_delay",
                    "piecewise_coeff_f_trans",
                    "piecewise_coeff_r_trans",
                }
                for dataset_name in DATASET_SPECS:
                    prefix = f"piecewise_{dataset_name}"
                    required_piecewise_keys.update(
                        {
                            f"{prefix}_size_table",
                            f"{prefix}_arc_index_table",
                            f"{prefix}_size_count",
                            f"{prefix}_vt_available",
                        }
                    )
                if not required_piecewise_keys.issubset(state_dict.keys()):
                    self._record_cache_event("miss", cache_meta["cache_dir"], "piecewise_aux_missing")
                    return False
                self.piecewise_state = {}
                for dataset_name in DATASET_SPECS:
                    prefix = f"piecewise_{dataset_name}"
                    self.piecewise_state[dataset_name] = {
                        "size_table": state_dict[f"{prefix}_size_table"].to(self.device),
                        "arc_index_table": state_dict[f"{prefix}_arc_index_table"].to(self.device),
                        "size_count": state_dict[f"{prefix}_size_count"].to(self.device),
                        "vt_available": state_dict[f"{prefix}_vt_available"].to(self.device),
                    }
                self.piecewise_linear_coeffs = {
                    "coeff_f_delay": state_dict["piecewise_coeff_f_delay"].to(self.device),
                    "coeff_r_delay": state_dict["piecewise_coeff_r_delay"].to(self.device),
                    "coeff_f_trans": state_dict["piecewise_coeff_f_trans"].to(self.device),
                    "coeff_r_trans": state_dict["piecewise_coeff_r_trans"].to(self.device),
                }
                sample_state = next(iter(self.piecewise_state.values()))
                self.max_arc_offset = sample_state["size_table"].shape[1]
            else:
                raise ValueError(f"unsupported cell_model_schema: {self.cell_model_schema}")
            self._record_cache_event("hit", cache_meta["cache_dir"])
            return True
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
            self._record_cache_event("miss", cache_meta["cache_dir"], "load_error:%s" % type(e).__name__)
            logging.warning("Failed to load surrogate cache from %s: %s", cache_meta["cache_dir"], e)
            return False

    def _save_cache(self, cache_meta):
        if not cache_meta["cache_dir"]:
            self._record_cache_event("disabled", None, "cache_root_unset")
            return
        os.makedirs(cache_meta["cache_dir"], exist_ok=True)
        try:
            torch.save(self._coeff_state_dict(), cache_meta["weights_path"])
            with open(cache_meta["manifest_path"], "w", encoding="utf-8") as f:
                json.dump(self._manifest_payload(cache_meta), f, indent=2, sort_keys=True)
            self._record_cache_event("write", cache_meta["cache_dir"])
        except OSError as e:
            logging.warning("Failed to write surrogate cache to %s: %s", cache_meta["cache_dir"], e)

    def cache_summary(self):
        cache_meta = self.cache_meta or {}
        manifest_path = cache_meta.get("manifest_path")
        weights_path = cache_meta.get("weights_path")
        cache_dir = cache_meta.get("cache_dir")
        return {
            "cache_event": self.cache_event,
            "cache_reason": self.cache_reason,
            "cache_history": list(self.cache_history),
            "cache_dir": cache_dir,
            "manifest_path": manifest_path,
            "weights_path": weights_path,
            "manifest_exists": bool(manifest_path and os.path.isfile(manifest_path)),
            "weights_exist": bool(weights_path and os.path.isfile(weights_path)),
            "cache_dir_exists": bool(cache_dir and os.path.isdir(cache_dir)),
            "lib_hash": cache_meta.get("lib_hash"),
            "model_version": cache_meta.get("model_version"),
            "feature_version": cache_meta.get("feature_version"),
            "vt_order_hash": cache_meta.get("vt_order_hash"),
            "num_main_ids": cache_meta.get("num_main_ids"),
            "num_vt_types": cache_meta.get("num_vt_types"),
            "num_libcells": cache_meta.get("num_libcells"),
            "num_arcs": cache_meta.get("num_arcs"),
            "cell_model_schema": cache_meta.get("cell_model_schema"),
            "log_leakage_coord_stats": cache_meta.get("log_leakage_coord_stats"),
        }

    def _expand_lut_values(self, values, dims):
        if values.ndim == 3:
            return values

        if dims.numel() == 0:
            return values.reshape(0, 1, 1)

        trans_dims = dims[:, 0].long()
        cap_dims = dims[:, 1].long()
        max_t = int(max(1, trans_dims.max().item()))
        max_c = int(max(1, cap_dims.max().item()))
        expanded = values.new_zeros((values.shape[0], max_t, max_c))

        if values.ndim == 1:
            expanded[:, 0, 0] = values
            return expanded

        if values.ndim != 2:
            raise ValueError(f"Unsupported LUT value rank {values.ndim}, expected 1D/2D/3D")

        for arc_idx in range(values.shape[0]):
            trans_dim = int(max(1, trans_dims[arc_idx].item()))
            cap_dim = int(max(1, cap_dims[arc_idx].item()))
            if trans_dim > 1 and cap_dim > 1:
                point_count = trans_dim * cap_dim
                expanded[arc_idx, :trans_dim, :cap_dim] = values[arc_idx, :point_count].reshape(
                    trans_dim, cap_dim
                )
                continue
            if trans_dim > 1:
                expanded[arc_idx, :trans_dim, 0] = values[arc_idx, :trans_dim]
            elif cap_dim > 1:
                expanded[arc_idx, 0, :cap_dim] = values[arc_idx, :cap_dim]
            else:
                expanded[arc_idx, 0, 0] = values[arc_idx, 0]
        return expanded

    def _ensure_lut_runtime_lookup_cache(self, dataset_name):
        cache = getattr(self, "_lut_runtime_lookup_cache", None)
        if cache is None:
            cache = {}
            self._lut_runtime_lookup_cache = cache
        if dataset_name in cache:
            return cache[dataset_name]

        spec = self._dataset_spec(dataset_name)
        luts_info = getattr(self.data_collections.arcs_info, spec["luts_attr"])

        dims = luts_info.flat_luts_dim[:, :2]
        if not isinstance(dims, torch.Tensor):
            dims = torch.tensor(dims, device=self.device)
        else:
            dims = dims.to(self.device)
        dims = dims.long()

        values = luts_info.flat_luts_values
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self.device)
        else:
            values = values.to(self.device)

        trans_tables = luts_info.flat_luts_trans_table
        if not isinstance(trans_tables, torch.Tensor):
            trans_tables = torch.tensor(trans_tables, device=self.device)
        else:
            trans_tables = trans_tables.to(self.device)

        cap_tables = luts_info.flat_luts_cap_table
        if not isinstance(cap_tables, torch.Tensor):
            cap_tables = torch.tensor(cap_tables, device=self.device)
        else:
            cap_tables = cap_tables.to(self.device)

        values_3d = self._expand_lut_values(values, dims)
        coeff_table = None
        coeff_op_requested = (
            os.environ.get("AIMP_USE_LUT_2D_COEFF_OP", "").lower()
            in ("1", "true", "yes", "on")
        )
        if (
            coeff_op_requested
            and
            lut_entry_2d_op is not None
            and values_3d.numel() > 0
            and trans_tables.dim() == 2
            and cap_tables.dim() == 2
            and trans_tables.shape[1] >= 2
            and cap_tables.shape[1] >= 2
        ):
            is_2d = (dims[:, 0] > 1) & (dims[:, 1] > 1)
            if bool(torch.any(is_2d).detach().cpu().item()):
                coeff_table = lut_entry_2d_op.build_2d_coefficients(
                    trans_tables,
                    cap_tables,
                    values_3d.reshape(values_3d.shape[0], -1),
                    dims[:, 0],
                    dims[:, 1],
                )
        cache[dataset_name] = {
            "trans_dims": dims[:, 0].long(),
            "cap_dims": dims[:, 1].long(),
            "values_3d": values_3d,
            "trans_tables": trans_tables,
            "cap_tables": cap_tables,
            "coeff_table": coeff_table,
        }
        return cache[dataset_name]

    @staticmethod
    def _lookup_state_on_device(lookup_state, device):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in lookup_state.items()
        }

    def _solve_ridge_regression(self, X_sub, y_sub, lambda_reg=1e-3):
        X64 = X_sub.to(dtype=torch.float64)
        y64 = y_sub.to(dtype=torch.float64)
        eye = torch.eye(X64.shape[1], device=self.device, dtype=torch.float64)
        ridge_scale = torch.sqrt(
            torch.tensor(lambda_reg, device=self.device, dtype=torch.float64)
        )
        aug_X = torch.cat([X64, ridge_scale * eye], dim=0)
        aug_y = torch.cat([y64, torch.zeros(X64.shape[1], device=self.device, dtype=torch.float64)], dim=0)
        solution = torch.linalg.lstsq(aug_X, aug_y).solution
        return solution.to(dtype=torch.float32)

    def _feature_width(self, schema=None):
        schema = schema or self.cell_model_schema
        widths = {
            LINEAR6_SCHEMA: 6,
            POLY12_SCHEMA: 12,
            LOGCROSS12_SCHEMA: 12,
            LOG_LEAKAGE_POLY12_SCHEMA: 12,
        }
        if schema not in widths:
            raise ValueError(f"unsupported regression cell_model_schema: {schema}")
        return widths[schema]

    def _schema_ridge_lambda(self, schema=None):
        schema = schema or self.cell_model_schema
        if schema == LINEAR6_SCHEMA:
            return 1e-3
        if schema in {POLY12_SCHEMA, LOG_LEAKAGE_POLY12_SCHEMA}:
            return 5e-3
        if schema == LOGCROSS12_SCHEMA:
            return 2e-3
        raise ValueError(f"unsupported regression cell_model_schema: {schema}")

    def _regression_features(self, schema, vt, size, input_slew, out_cap):
        vt = vt.float()
        size = size.float()
        input_slew = input_slew.float()
        out_cap = out_cap.float()
        size_safe = size + 1e-9
        inv_size = 1.0 / size_safe
        cap_div_size = out_cap / size_safe
        ones = torch.ones_like(input_slew)
        if schema == LINEAR6_SCHEMA:
            return torch.stack([input_slew, out_cap, inv_size, cap_div_size, vt, ones], dim=1)
        if schema in {POLY12_SCHEMA, LOG_LEAKAGE_POLY12_SCHEMA}:
            slew_cap = input_slew * out_cap
            slew_inv_size = input_slew * inv_size
            slew_vt = input_slew * vt
            cap_vt = out_cap * vt
            inv_size_vt = inv_size * vt
            inv_size_sq = inv_size.square()
            return torch.stack(
                [
                    input_slew,
                    out_cap,
                    inv_size,
                    cap_div_size,
                    vt,
                    slew_cap,
                    slew_inv_size,
                    slew_vt,
                    cap_vt,
                    inv_size_vt,
                    inv_size_sq,
                    ones,
                ],
                dim=1,
            )
        if schema == LOGCROSS12_SCHEMA:
            slew_log = torch.log1p(torch.clamp_min(input_slew, 0.0))
            cap_log = torch.log1p(torch.clamp_min(out_cap * 1e3, 0.0))
            slew_cap_log = slew_log * cap_log
            slew_log_inv_size = slew_log * inv_size
            cap_log_inv_size = cap_log * inv_size
            slew_log_vt = slew_log * vt
            cap_log_vt = cap_log * vt
            inv_size_vt = inv_size * vt
            return torch.stack(
                [
                    slew_log,
                    cap_log,
                    inv_size,
                    cap_div_size,
                    vt,
                    slew_cap_log,
                    slew_log_inv_size,
                    cap_log_inv_size,
                    slew_log_vt,
                    cap_log_vt,
                    inv_size_vt,
                    ones,
                ],
                dim=1,
            )
        raise ValueError(f"unsupported regression cell_model_schema: {schema}")

    def _fit_dataset(self, luts_info, libcell_main_ids, arc_offsets, sizes, vts, schema=None):
        """
        Fit Linear Regression for each (MainType, ArcOffset) group.
        Returns dictionary {(main_type, arc_offset): coefficients}
        """
        schema = schema or self.cell_model_schema
        # Expand LUTs to points
        # flat_luts_values: [NumArcs, MaxT, MaxC]
        dataset_name = self._active_dataset_name
        logging.info(
            "CellModeling prepare dataset=%s values_shape=%s trans_shape=%s cap_shape=%s dim_shape=%s",
            dataset_name,
            tuple(luts_info.flat_luts_values.shape),
            tuple(luts_info.flat_luts_trans_table.shape),
            tuple(luts_info.flat_luts_cap_table.shape),
            tuple(luts_info.flat_luts_dim.shape),
        )
        dims = luts_info.flat_luts_dim[:, :2]
        logging.info("CellModeling dataset=%s dims extracted shape=%s", dataset_name, tuple(dims.shape))
        values = self._expand_lut_values(luts_info.flat_luts_values, dims)
        logging.info("CellModeling dataset=%s expanded values shape=%s", dataset_name, tuple(values.shape))
        trans_table = luts_info.flat_luts_trans_table # [NumArcs, MaxT]
        cap_table = luts_info.flat_luts_cap_table     # [NumArcs, MaxC]
        dims = dims                                  # [NumArcs, 2]
        
        num_arcs, max_t, max_c = values.shape
        
        # Create meshgrid for Slew and Cap
        # slew_grid: [NumArcs, MaxT, MaxC]
        slew_grid = trans_table.unsqueeze(2).expand(-1, -1, max_c)
        cap_grid = cap_table.unsqueeze(1).expand(-1, max_t, -1)
        
        # Mask invalid entries (padding)
        # dims: [trans_dim, cap_dim]
        row_indices = torch.arange(max_t, device=self.device).unsqueeze(0).unsqueeze(2) # [1, MaxT, 1]
        col_indices = torch.arange(max_c, device=self.device).unsqueeze(0).unsqueeze(1) # [1, 1, MaxC]
        
        valid_mask = (row_indices < dims[:, 0].unsqueeze(1).unsqueeze(2)) & \
                     (col_indices < dims[:, 1].unsqueeze(1).unsqueeze(2))
        
        # Use explicit coordinates instead of flattening expanded tensors.
        # This avoids fragile expand/reshape/bool-indexing combinations on CPU.
        arc_idx, row_idx, col_idx = valid_mask.nonzero(as_tuple=True)
        logging.info(
            "CellModeling dataset=%s gathered valid coordinates count=%d",
            dataset_name,
            arc_idx.numel(),
        )
        y = values[arc_idx, row_idx, col_idx]
        slew = trans_table[arc_idx, row_idx]
        cap = cap_table[arc_idx, col_idx]
        size = sizes[arc_idx]
        vt = vts[arc_idx]
        mt = libcell_main_ids[arc_idx]
        ao = arc_offsets[arc_idx]
        
        # Construct Features for Physics Model
        # Features: [Slew, Cap, 1/Size, Cap/Size, VT, 1]
        X = self._regression_features(schema, vt, size, slew, cap)
        feature_width = self._feature_width(schema)
        ridge_lambda = self._schema_ridge_lambda(schema)
        
        # Group by (MainType, ArcOffset)
        # GroupID = MainType * 1000 + ArcOffset (Assuming MaxArcOffset < 1000)
        group_ids = mt * 1000 + ao
        unique_groups = torch.unique(group_ids)
        logging.info(
            "CellModeling fit start: dataset=%s num_groups=%d num_points=%d",
            dataset_name,
            unique_groups.numel(),
            X.shape[0],
        )
        
        coeffs_map = {}
        
        for group_idx, gid in enumerate(unique_groups):
            mask = (group_ids == gid)
            X_sub = X[mask]
            y_sub = y[mask]
            
            if X_sub.shape[0] < feature_width:
                # Not enough data, use zeros or fallback
                w = torch.zeros(feature_width, device=self.device)
            else:
                try:
                    if not torch.isfinite(X_sub).all() or not torch.isfinite(y_sub).all():
                        raise RuntimeError("non-finite regression inputs")
                    w = self._solve_ridge_regression(X_sub, y_sub, lambda_reg=ridge_lambda)
                except RuntimeError as exc:
                    logging.warning(
                        "CellModeling regression fallback: dataset=%s group_idx=%d points=%d reason=%s",
                        dataset_name,
                        group_idx,
                        X_sub.shape[0],
                        exc,
                    )
                    w = torch.zeros(feature_width, device=self.device)
            
            # Decode GID
            m_id = (gid // 1000).item()
            a_off = (gid % 1000).item()
            coeffs_map[(m_id, a_off)] = w
        logging.info(
            "CellModeling fit done: dataset=%s num_groups=%d",
            dataset_name,
            len(coeffs_map),
        )
            
        return coeffs_map

    def _reshape_coeffs(self, coeffs_map, schema=None):
        """
        Convert dict to tensor [NumMainTypes, MaxArcOffset, 6]
        """
        feature_width = self._feature_width(schema)
        # Determine max dimensions
        if not coeffs_map:
            return torch.zeros(1, 1, feature_width, device=self.device)
            
        max_mt = max(k[0] for k in coeffs_map.keys())
        max_ao = max(k[1] for k in coeffs_map.keys())
        
        tensor = torch.zeros(max_mt + 1, max_ao + 1, feature_width, device=self.device)
        
        for (m, a), w in coeffs_map.items():
            tensor[m, a] = w
            
        return tensor

    def _reshape_support_mask(self, coeff_tensor):
        if coeff_tensor.numel() == 0:
            return torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        return coeff_tensor.abs().sum(dim=-1) > 0

    def _rebuild_regression_support_stack(self):
        support_tensors = (
            self.support_f_delay,
            self.support_r_delay,
            self.support_f_trans,
            self.support_r_trans,
        )
        if any(tensor is None for tensor in support_tensors):
            self.regression_support_stack = None
            return
        max_main_ids = max(int(tensor.shape[0]) for tensor in support_tensors)
        max_arc_offsets = max(int(tensor.shape[1]) for tensor in support_tensors)
        stack = torch.zeros(
            4,
            max_main_ids,
            max_arc_offsets,
            dtype=torch.bool,
            device=self.device,
        )
        for arc_type, support_tensor in enumerate(support_tensors):
            stack[
                arc_type,
                : support_tensor.shape[0],
                : support_tensor.shape[1],
            ] = support_tensor.to(self.device).bool()
        self.regression_support_stack = stack

    def supports_arc_types(self, libcell_main_id, arc_offset, arc_types):
        mt = libcell_main_id.long()
        ao = arc_offset.long()
        arc_types = arc_types.to(device=mt.device)
        support = torch.zeros_like(mt, dtype=torch.bool, device=mt.device)

        if self.cell_model_schema in REGRESSION_SCHEMAS:
            support_stack = getattr(self, "regression_support_stack", None)
            if support_stack is None:
                self._rebuild_regression_support_stack()
                support_stack = self.regression_support_stack
            if support_stack is None:
                return support
            support_stack = support_stack.to(device=mt.device)
            valid_arc_type = (arc_types >= 0) & (arc_types < support_stack.shape[0])
            if not torch.any(valid_arc_type):
                return support
            safe_arc_types = arc_types.clamp(min=0, max=support_stack.shape[0] - 1)
            safe_mt = mt.clamp(max=support_stack.shape[1] - 1)
            safe_ao = ao.clamp(max=support_stack.shape[2] - 1)
            support = support_stack[safe_arc_types, safe_mt, safe_ao]
            return support & valid_arc_type

        if self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA and self.piecewise_state:
            support = torch.zeros_like(mt, dtype=torch.bool, device=mt.device)
            arc_state_map = {
                0: self.piecewise_state.get("f_delay"),
                1: self.piecewise_state.get("r_delay"),
                2: self.piecewise_state.get("f_trans"),
                3: self.piecewise_state.get("r_trans"),
            }
            for arc_type, state in arc_state_map.items():
                if state is None:
                    continue
                arc_mask = arc_types == arc_type
                if not torch.any(arc_mask):
                    continue
                vt_available = state["vt_available"].to(device=mt.device)
                mt_sub = mt[arc_mask].clamp(max=vt_available.shape[0] - 1)
                ao_sub = ao[arc_mask].clamp(max=vt_available.shape[1] - 1)
                support[arc_mask] = vt_available[mt_sub, ao_sub].any(dim=1)
            return support

        return torch.ones_like(mt, dtype=torch.bool, device=mt.device)

    def _build_piecewise_state(self, libcell_main_ids, arc_offsets, sizes, vts, num_main_ids, max_arc_offset, vt_order):
        num_vts = len(vt_order)
        vt_to_slot = {int(v): idx for idx, v in enumerate(vt_order)}
        grouped = {}
        for arc_idx in range(libcell_main_ids.numel()):
            main_id = int(libcell_main_ids[arc_idx].item())
            arc_offset = int(arc_offsets[arc_idx].item())
            vt_slot = vt_to_slot[int(vts[arc_idx].item())]
            size_value = float(sizes[arc_idx].item())
            grouped.setdefault((main_id, arc_offset, vt_slot), {})[size_value] = arc_idx

        max_sizes = max((len(entries) for entries in grouped.values()), default=1)
        size_table = torch.full(
            (num_main_ids, max_arc_offset, num_vts, max_sizes),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        arc_index_table = torch.full(
            (num_main_ids, max_arc_offset, num_vts, max_sizes),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        size_count = torch.zeros(
            (num_main_ids, max_arc_offset, num_vts),
            device=self.device,
            dtype=torch.long,
        )
        vt_available = torch.zeros(
            (num_main_ids, max_arc_offset, num_vts),
            device=self.device,
            dtype=torch.bool,
        )
        for (main_id, arc_offset, vt_slot), entries in grouped.items():
            sorted_items = sorted(entries.items(), key=lambda item: item[0])
            count = len(sorted_items)
            if count == 0:
                continue
            vt_available[main_id, arc_offset, vt_slot] = True
            size_count[main_id, arc_offset, vt_slot] = count
            size_table[main_id, arc_offset, vt_slot, :count] = torch.tensor(
                [item[0] for item in sorted_items],
                device=self.device,
                dtype=torch.float32,
            )
            arc_index_table[main_id, arc_offset, vt_slot, :count] = torch.tensor(
                [item[1] for item in sorted_items],
                device=self.device,
                dtype=torch.long,
            )
        return {
            "size_table": size_table,
            "arc_index_table": arc_index_table,
            "size_count": size_count,
            "vt_available": vt_available,
        }

    def _analysis_cache_identity(self):
        cache_meta = self.cache_meta or {}
        return {
            "cache_dir": cache_meta.get("cache_dir"),
            "lib_hash": cache_meta.get("lib_hash"),
            "model_version": cache_meta.get("model_version", MODEL_VERSION),
            "feature_version": cache_meta.get("feature_version", FEATURE_VERSION),
            "cell_model_schema": cache_meta.get("cell_model_schema", self.cell_model_schema),
            "cache_event": self.cache_event,
            "cache_reason": self.cache_reason,
            "cache_history": list(self.cache_history),
            "log_leakage_coord_stats": cache_meta.get("log_leakage_coord_stats"),
        }

    def _forward_dataset(self, dataset_name, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
        self._profile_increment("cell_model_forward_dataset_calls")
        self._profile_increment_dataset(
            "cell_model_forward_dataset_calls_by_dataset",
            dataset_name,
        )
        if self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA:
            piecewise_value = torch.nan_to_num(
                self._piecewise_forward_dataset(
                    dataset_name,
                    libcell_main_id.long(),
                    arc_offset.long(),
                    vt.float(),
                    size.float(),
                    input_slew.float(),
                    out_cap.float(),
                )
            )
            return self._apply_piecewise_gradient_mode(
                piecewise_value,
                dataset_name,
                libcell_main_id,
                arc_offset,
                vt.float(),
                size.float(),
                input_slew.float(),
                out_cap.float(),
            )

        coeff_tensor = {
            "f_delay": self.coeff_f_delay,
            "r_delay": self.coeff_r_delay,
            "f_trans": self.coeff_f_trans,
            "r_trans": self.coeff_r_trans,
        }[dataset_name]
        return self._linear_forward_dataset(
            coeff_tensor,
            libcell_main_id.long(),
            arc_offset.long(),
            vt,
            size,
            input_slew,
            out_cap,
        )

    def _forward_selected_arc_types(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types):
        if arc_types.numel() == 0:
            return torch.zeros_like(input_slew)

        arc_types = arc_types.long()
        unique_arc_types = torch.unique(arc_types)
        if unique_arc_types.numel() == 1:
            dataset_name = ARC_TYPE_TO_DATASET.get(int(unique_arc_types.item()))
            if dataset_name is None:
                return torch.zeros_like(input_slew)
            return self._forward_dataset(
                dataset_name,
                libcell_main_id,
                arc_offset,
                vt,
                size,
                input_slew,
                out_cap,
            )

        result = torch.zeros_like(input_slew)
        for arc_type in unique_arc_types.tolist():
            dataset_name = ARC_TYPE_TO_DATASET.get(int(arc_type))
            if dataset_name is None:
                continue
            mask = arc_types == arc_type
            result[mask] = self._forward_dataset(
                dataset_name,
                libcell_main_id[mask],
                arc_offset[mask],
                vt[mask],
                size[mask],
                input_slew[mask],
                out_cap[mask],
            )
        return result

    def _linear_forward_dataset(self, coeff_tensor, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, schema=None):
        schema = schema or self.cell_model_schema
        if schema == POLY12_SCHEMA and self.use_poly12_op and cell_modeling_op is not None:
            return cell_modeling_op.poly12_forward(
                coeff_tensor,
                libcell_main_id.long(),
                arc_offset.long(),
                vt,
                size,
                input_slew,
                out_cap,
            )
        feature_size = self._forward_size_coord(schema, libcell_main_id, vt, size)
        X = self._regression_features(schema, vt, feature_size, input_slew, out_cap).unsqueeze(2)
        mt = libcell_main_id.clamp(max=coeff_tensor.shape[0] - 1)
        ao = arc_offset.clamp(max=coeff_tensor.shape[1] - 1)
        w = coeff_tensor[mt, ao]
        return torch.bmm(w.unsqueeze(1), X).squeeze(2).squeeze(1)

    def _piecewise_backward_stabilized(self, piecewise_value, dataset_name, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
        if self.piecewise_linear_coeffs is None:
            return piecewise_value
        coeff_key = {
            "f_delay": "coeff_f_delay",
            "r_delay": "coeff_r_delay",
            "f_trans": "coeff_f_trans",
            "r_trans": "coeff_r_trans",
        }[dataset_name]
        linear_value = self._linear_forward_dataset(
            self.piecewise_linear_coeffs[coeff_key],
            libcell_main_id.long(),
            arc_offset.long(),
            vt,
            size,
            input_slew,
            out_cap,
            schema=LINEAR6_SCHEMA,
        )
        return piecewise_value.detach() + (linear_value - linear_value.detach())

    def _apply_piecewise_gradient_mode(
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
        if self.piecewise_gradient_mode == NATIVE_PIECEWISE_GRADIENT_MODE:
            return piecewise_value
        if self.piecewise_gradient_mode == LINEAR6_STE_GRADIENT_MODE:
            if torch.is_grad_enabled():
                requires_grad = (
                    vt.requires_grad
                    or size.requires_grad
                    or input_slew.requires_grad
                    or out_cap.requires_grad
                )
                if requires_grad:
                    return self._piecewise_backward_stabilized(
                        piecewise_value,
                        dataset_name,
                        libcell_main_id,
                        arc_offset,
                        vt,
                        size,
                        input_slew,
                        out_cap,
                    )
            return piecewise_value
        raise ValueError(
            f"unsupported piecewise_gradient_mode: {self.piecewise_gradient_mode}"
        )

    def _lut_entry_1d_vectorized(self, x, x_table, y_table, actual_dims):
        batch_size = x.shape[0]
        max_idx_actual = (actual_dims - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=x.device)
        x_min = x_table[:, 0]
        x_max = x_table[batch_indices, max_idx_actual]
        boundary_mode = self._resolve_lut_boundary_mode(
            getattr(self, "cell_model_lut_boundary_mode", None)
        )
        x_query = x
        if boundary_mode == LUT_BOUNDARY_MODE_CLAMP:
            x_query = torch.minimum(torch.maximum(x, x_min), x_max)
        idx_padded = torch.searchsorted(x_table, x_query.unsqueeze(1), right=True).squeeze(1)
        idx_high = idx_padded.clamp(min=1).clamp(max=max_idx_actual)
        idx_high = torch.where(x_query < x_min, torch.ones_like(idx_high), idx_high)
        idx_high = torch.where(x_query >= x_max, max_idx_actual, idx_high)
        idx_low = (idx_high - 1).clamp(min=0)

        x0 = x_table[batch_indices, idx_low]
        x1 = x_table[batch_indices, idx_high]
        y0 = y_table[batch_indices, idx_low]
        y1 = y_table[batch_indices, idx_high]
        denom = x1 - x0
        safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        factor = (x_query - x0) / safe_denom
        return torch.where(denom.abs() < 1e-12, y0, torch.lerp(y0, y1, factor))

    def _lut_entry_2d_vectorized(self, input_trans, output_caps, trans_tables_batch, cap_tables_batch, lut_values_batch, trans_dims_actual, cap_dims_actual, coeff_table_batch=None):
        boundary_mode = self._resolve_lut_boundary_mode(
            getattr(self, "cell_model_lut_boundary_mode", None)
        )
        use_extrapolate_boundary = boundary_mode == LUT_BOUNDARY_MODE_EXTRAPOLATE
        if use_extrapolate_boundary and lut_entry_2d_op is not None and coeff_table_batch is not None:
            return lut_entry_2d_op.lut_entry_2d_coeff(
                input_trans,
                output_caps,
                trans_tables_batch,
                cap_tables_batch,
                coeff_table_batch,
                trans_dims_actual,
                cap_dims_actual,
            )
        if (
            use_extrapolate_boundary
            and lut_entry_2d_op is not None
            and os.environ.get("AIMP_USE_LUT_2D_OP", "").lower() in ("1", "true", "yes", "on")
        ):
            return lut_entry_2d_op.lut_entry_2d(
                input_trans,
                output_caps,
                trans_tables_batch,
                cap_tables_batch,
                lut_values_batch,
                trans_dims_actual,
                cap_dims_actual,
            )
        batch_size = input_trans.shape[0]
        denom_epsilon = torch.tensor(1e-12, device=input_trans.device, dtype=input_trans.dtype)
        trans_min = trans_tables_batch[:, 0]
        trans_max = trans_tables_batch.gather(1, (trans_dims_actual - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        cap_min = cap_tables_batch[:, 0]
        cap_max = cap_tables_batch.gather(1, (cap_dims_actual - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        query_trans = input_trans
        query_caps = output_caps
        if boundary_mode == LUT_BOUNDARY_MODE_CLAMP:
            query_trans = torch.minimum(torch.maximum(input_trans, trans_min), trans_max)
            query_caps = torch.minimum(torch.maximum(output_caps, cap_min), cap_max)

        trans_idx_padded = torch.searchsorted(trans_tables_batch, query_trans.unsqueeze(1), right=True).squeeze(1)
        cap_idx_padded = torch.searchsorted(cap_tables_batch, query_caps.unsqueeze(1), right=True).squeeze(1)
        max_trans_idx_actual = (trans_dims_actual - 1).clamp(min=0)
        max_cap_idx_actual = (cap_dims_actual - 1).clamp(min=0)
        trans_idx_high = trans_idx_padded.clamp(min=1).clamp(max=max_trans_idx_actual)
        trans_idx_high = torch.where(query_trans < trans_min, torch.ones_like(trans_idx_high), trans_idx_high)
        trans_idx_high = torch.where(query_trans >= trans_max, max_trans_idx_actual, trans_idx_high)
        trans_idx_low = (trans_idx_high - 1).clamp(min=0)
        cap_idx_high = cap_idx_padded.clamp(min=1).clamp(max=max_cap_idx_actual)
        cap_idx_high = torch.where(query_caps < cap_min, torch.ones_like(cap_idx_high), cap_idx_high)
        cap_idx_high = torch.where(query_caps >= cap_max, max_cap_idx_actual, cap_idx_high)
        cap_idx_low = (cap_idx_high - 1).clamp(min=0)

        batch_indices = torch.arange(batch_size, device=input_trans.device)
        t0 = trans_tables_batch[batch_indices, trans_idx_low]
        t1 = trans_tables_batch[batch_indices, trans_idx_high]
        c0 = cap_tables_batch[batch_indices, cap_idx_low]
        c1 = cap_tables_batch[batch_indices, cap_idx_high]
        stride = cap_dims_actual
        idx00 = trans_idx_low * stride + cap_idx_low
        idx01 = trans_idx_low * stride + cap_idx_high
        idx10 = trans_idx_high * stride + cap_idx_low
        idx11 = trans_idx_high * stride + cap_idx_high
        corner_indices = torch.stack([idx00, idx01, idx10, idx11], dim=1)
        corner_values = lut_values_batch.gather(1, corner_indices)
        v00, v01, v10, v11 = corner_values[:, 0], corner_values[:, 1], corner_values[:, 2], corner_values[:, 3]

        t_interval = t1 - t0
        c_interval = c1 - c0
        is_t_degenerate = torch.abs(t_interval) < denom_epsilon
        is_c_degenerate = torch.abs(c_interval) < denom_epsilon
        t_interval_safe = torch.where(is_t_degenerate, denom_epsilon, t_interval)
        c_interval_safe = torch.where(is_c_degenerate, denom_epsilon, c_interval)
        safe_denominator = t_interval_safe * c_interval_safe
        wa = (t1 - query_trans) * (c1 - query_caps)
        wb = (t1 - query_trans) * (query_caps - c0)
        wc = (query_trans - t0) * (c1 - query_caps)
        wd = (query_trans - t0) * (query_caps - c0)
        bilinear_val = (v00 * wa + v01 * wb + v10 * wc + v11 * wd) / safe_denominator
        lerp_c_factor = (query_caps - c0) / c_interval_safe
        lerp_t_factor = (query_trans - t0) / t_interval_safe
        val_t_degenerate = torch.lerp(v00, v01, lerp_c_factor)
        val_c_degenerate = torch.lerp(v00, v10, lerp_t_factor)
        return torch.where(
            is_t_degenerate & is_c_degenerate,
            v00,
            torch.where(is_t_degenerate, val_t_degenerate, torch.where(is_c_degenerate, val_c_degenerate, bilinear_val)),
        )

    def _lut_lookup(self, dataset_name, arc_indices, input_slew, out_cap):
        lookup_state = self._lookup_state_on_device(
            self._ensure_lut_runtime_lookup_cache(dataset_name),
            input_slew.device,
        )
        result = torch.zeros_like(input_slew)
        valid_mask = arc_indices >= 0
        if not torch.any(valid_mask):
            return result
        valid_arc_indices = arc_indices[valid_mask].long()
        values = lookup_state["values_3d"][valid_arc_indices]
        trans_tables = lookup_state["trans_tables"][valid_arc_indices]
        cap_tables = lookup_state["cap_tables"][valid_arc_indices]
        trans_dims_actual = lookup_state["trans_dims"][valid_arc_indices]
        cap_dims_actual = lookup_state["cap_dims"][valid_arc_indices]
        local_slew = input_slew[valid_mask]
        local_cap = out_cap[valid_mask]
        local_result = torch.zeros_like(local_slew)
        is_scalar = (trans_dims_actual <= 1) & (cap_dims_actual <= 1)
        is_trans_1d = (trans_dims_actual > 1) & (cap_dims_actual <= 1)
        is_cap_1d = (trans_dims_actual <= 1) & (cap_dims_actual > 1)
        is_2d = (trans_dims_actual > 1) & (cap_dims_actual > 1)
        if torch.any(is_2d):
            idx = is_2d.nonzero(as_tuple=True)[0]
            local_result[idx] = self._lut_entry_2d_vectorized(
                local_slew[idx],
                local_cap[idx],
                trans_tables[idx],
                cap_tables[idx],
                values[idx].reshape(values[idx].shape[0], -1),
                trans_dims_actual[idx],
                cap_dims_actual[idx],
                coeff_table_batch=(
                    lookup_state["coeff_table"][valid_arc_indices[idx]]
                    if lookup_state.get("coeff_table") is not None
                    and os.environ.get("AIMP_USE_LUT_2D_COEFF_OP", "").lower()
                    in ("1", "true", "yes", "on")
                    else None
                ),
            )
        if torch.any(is_trans_1d):
            idx = is_trans_1d.nonzero(as_tuple=True)[0]
            local_result[idx] = self._lut_entry_1d_vectorized(
                local_slew[idx],
                trans_tables[idx],
                values[idx, :, 0],
                trans_dims_actual[idx],
            )
        if torch.any(is_cap_1d):
            idx = is_cap_1d.nonzero(as_tuple=True)[0]
            local_result[idx] = self._lut_entry_1d_vectorized(
                local_cap[idx],
                cap_tables[idx],
                values[idx, 0, :],
                cap_dims_actual[idx],
            )
        if torch.any(is_scalar):
            idx = is_scalar.nonzero(as_tuple=True)[0]
            local_result[idx] = values[idx, 0, 0]
        result[valid_mask] = local_result
        return result

    def _piecewise_vt_slots(self, state, mt, ao, vt):
        device = vt.device
        vt_available = state["vt_available"].to(device)[mt, ao]
        num_slots = vt_available.shape[1]
        slot_indices = torch.arange(num_slots, device=device, dtype=torch.long)
        vt_values = self.vt_order_tensor.to(device)
        low_mask = vt_available & (vt_values.unsqueeze(0) <= vt.unsqueeze(1))
        high_mask = vt_available & (vt_values.unsqueeze(0) >= vt.unsqueeze(1))
        low_idx = torch.where(
            low_mask,
            slot_indices.unsqueeze(0),
            torch.full((vt.shape[0], num_slots), -1, device=device, dtype=torch.long),
        ).max(dim=1).values
        high_idx = torch.where(
            high_mask,
            slot_indices.unsqueeze(0),
            torch.full((vt.shape[0], num_slots), num_slots, device=device, dtype=torch.long),
        ).min(dim=1).values
        high_idx = torch.where(high_idx >= num_slots, low_idx, high_idx)
        low_idx = torch.where(low_idx < 0, high_idx, low_idx)
        return low_idx.clamp(min=0), high_idx.clamp(min=0, max=max(0, num_slots - 1))

    def _piecewise_size_interp(self, dataset_name, state, mt, ao, vt_slot, size, input_slew, out_cap):
        device = size.device
        size_table = state["size_table"].to(device)[mt, ao, vt_slot]
        arc_index_table = state["arc_index_table"].to(device)[mt, ao, vt_slot]
        size_count = state["size_count"].to(device)[mt, ao, vt_slot]
        if cell_modeling_op is not None:
            lookup_state = self._lookup_state_on_device(
                self._ensure_lut_runtime_lookup_cache(dataset_name),
                device,
            )
            supports_2d_only = state.get("_all_luts_2d_only")
            if supports_2d_only is None:
                trans_dims = lookup_state["trans_dims"]
                cap_dims = lookup_state["cap_dims"]
                supports_2d_only = bool(
                    torch.all((trans_dims > 1) & (cap_dims > 1)).detach().cpu().item()
                )
                state["_all_luts_2d_only"] = supports_2d_only
            if not supports_2d_only:
                valid_arcs = arc_index_table[arc_index_table >= 0]
                if valid_arcs.numel() > 0:
                    valid_arcs = valid_arcs.long()
                    trans_dims = lookup_state["trans_dims"]
                    cap_dims = lookup_state["cap_dims"]
                    supports_2d_only = torch.all(
                        (trans_dims[valid_arcs] > 1) & (cap_dims[valid_arcs] > 1)
                    )
                    supports_2d_only = bool(supports_2d_only.detach().cpu().item())
                else:
                    supports_2d_only = True
            if supports_2d_only:
                self._profile_increment("piecewise_size_interp_calls")
                self._profile_increment_dataset(
                    "piecewise_size_interp_calls_by_dataset",
                    dataset_name,
                )
                self._profile_increment("piecewise_size_interp_fused_calls")
                self._profile_increment("piecewise_size_interp_arcs", int(size.numel()))
                self._profile_increment("piecewise_size_interp_fused_arcs", int(size.numel()))
                values = lookup_state["values_3d"]
                return cell_modeling_op.piecewise_size_forward(
                    size_table,
                    arc_index_table,
                    size_count,
                    lookup_state["trans_tables"],
                    lookup_state["cap_tables"],
                    values.reshape(values.shape[0], -1),
                    lookup_state["trans_dims"],
                    lookup_state["cap_dims"],
                    size,
                    input_slew,
                    out_cap,
                    getattr(self, "cell_model_lut_boundary_mode", None),
                )
        self._profile_increment("piecewise_size_interp_calls")
        self._profile_increment_dataset(
            "piecewise_size_interp_calls_by_dataset",
            dataset_name,
        )
        self._profile_increment("piecewise_size_interp_python_calls")
        self._profile_increment("piecewise_size_interp_arcs", int(size.numel()))
        self._profile_increment("piecewise_size_interp_python_arcs", int(size.numel()))
        has_candidates = size_count > 0
        result = torch.zeros_like(size)

        single_candidate_mask = has_candidates & (size_count <= 1)
        if torch.any(single_candidate_mask):
            single_arc = arc_index_table[single_candidate_mask, 0]
            result[single_candidate_mask] = self._lut_lookup(
                dataset_name,
                single_arc,
                input_slew[single_candidate_mask],
                out_cap[single_candidate_mask],
            )

        multi_candidate_mask = has_candidates & (size_count > 1)
        if not torch.any(multi_candidate_mask):
            return result

        multi_size_table = size_table[multi_candidate_mask]
        multi_arc_index_table = arc_index_table[multi_candidate_mask]
        multi_size_count = size_count[multi_candidate_mask]
        multi_size = size[multi_candidate_mask]
        multi_input_slew = input_slew[multi_candidate_mask]
        multi_out_cap = out_cap[multi_candidate_mask]

        max_idx = (multi_size_count - 1).clamp(min=0)
        batch_idx = torch.arange(multi_size.shape[0], device=device)
        size_min = multi_size_table[:, 0]
        size_max = multi_size_table[batch_idx, max_idx]
        size_clamped = torch.minimum(torch.maximum(multi_size, size_min), size_max)
        idx_high = torch.searchsorted(multi_size_table, size_clamped.unsqueeze(1), right=True).squeeze(1)
        idx_high = idx_high.clamp(min=1).clamp(max=max_idx)
        idx_low = (idx_high - 1).clamp(min=0)
        size_low = multi_size_table[batch_idx, idx_low]
        size_high = multi_size_table[batch_idx, idx_high]
        arc_low = multi_arc_index_table[batch_idx, idx_low]
        arc_high = multi_arc_index_table[batch_idx, idx_high]
        value_low = self._lut_lookup(dataset_name, arc_low, multi_input_slew, multi_out_cap)
        same_arc_slot = arc_low == arc_high
        if torch.all(same_arc_slot):
            value_high = value_low
        else:
            value_high = value_low.clone()
            diff_mask = ~same_arc_slot
            value_high[diff_mask] = self._lut_lookup(
                dataset_name,
                arc_high[diff_mask],
                multi_input_slew[diff_mask],
                multi_out_cap[diff_mask],
            )
        denom = size_high - size_low
        finite_mask = torch.isfinite(size_low) & torch.isfinite(size_high) & torch.isfinite(denom)
        safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        factor = (size_clamped - size_low) / safe_denom
        result[multi_candidate_mask] = torch.where(
            (~finite_mask) | (denom.abs() < 1e-12),
            value_low,
            torch.lerp(value_low, value_high, factor),
        )
        return result

    def _piecewise_forward_dataset(self, dataset_name, libcell_main_id, arc_offset, vt, size, input_slew, out_cap):
        self._profile_increment("piecewise_forward_dataset_calls")
        self._profile_increment_dataset(
            "piecewise_forward_dataset_calls_by_dataset",
            dataset_name,
        )
        state = self.piecewise_state[dataset_name]
        device = input_slew.device
        mt = libcell_main_id.to(device).clamp(max=state["size_table"].shape[0] - 1)
        ao = arc_offset.to(device).clamp(max=state["size_table"].shape[1] - 1)
        vt = vt.to(device)
        size = size.to(device)
        out_cap = out_cap.to(device)
        vt_available = state["vt_available"].to(device)[mt, ao]
        has_vt_candidates = vt_available.any(dim=1)
        vt_low, vt_high = self._piecewise_vt_slots(state, mt, ao, vt)
        value_low = self._piecewise_size_interp(dataset_name, state, mt, ao, vt_low, size, input_slew, out_cap)
        same_vt_slot = vt_low == vt_high
        if torch.all(same_vt_slot):
            return torch.where(has_vt_candidates, torch.nan_to_num(value_low), torch.zeros_like(value_low))

        value_high = value_low.clone()
        diff_mask = ~same_vt_slot
        if self._profile_detail_enabled():
            self._profile_increment("piecewise_vt_second_pass_calls")
            self._profile_increment(
                "piecewise_vt_second_pass_arcs",
                int(diff_mask.sum().detach().item()),
            )
        value_high[diff_mask] = self._piecewise_size_interp(
            dataset_name,
            state,
            mt[diff_mask],
            ao[diff_mask],
            vt_high[diff_mask],
            size[diff_mask],
            input_slew[diff_mask],
            out_cap[diff_mask],
        )
        vt_values = self.vt_order_tensor.to(device)
        vt_low_value = vt_values[vt_low]
        vt_high_value = vt_values[vt_high]
        denom = vt_high_value - vt_low_value
        safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        vt_clamped = torch.minimum(torch.maximum(vt, vt_low_value), vt_high_value)
        factor = (vt_clamped - vt_low_value) / safe_denom
        interpolated = torch.where(denom.abs() < 1e-12, value_low, torch.lerp(value_low, value_high, factor))
        return torch.where(has_vt_candidates, torch.nan_to_num(interpolated), torch.zeros_like(interpolated))

    def _dataset_spec(self, dataset_name):
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"unknown cell-model dataset: {dataset_name}")
        return DATASET_SPECS[dataset_name]

    def _flatten_lut_points(self, dataset_name):
        spec = self._dataset_spec(dataset_name)
        metadata = self._dataset_arc_metadata(dataset_name)
        luts_info = getattr(self.data_collections.arcs_info, spec["luts_attr"])

        dims = luts_info.flat_luts_dim[:, :2]
        values = self._expand_lut_values(luts_info.flat_luts_values, dims)
        trans_table = luts_info.flat_luts_trans_table
        cap_table = luts_info.flat_luts_cap_table
        _, max_t, max_c = values.shape

        row_indices = torch.arange(max_t, device=self.device).unsqueeze(0).unsqueeze(2)
        col_indices = torch.arange(max_c, device=self.device).unsqueeze(0).unsqueeze(1)
        valid_mask = (row_indices < dims[:, 0].unsqueeze(1).unsqueeze(2)) & (
            col_indices < dims[:, 1].unsqueeze(1).unsqueeze(2)
        )
        arc_idx, row_idx, col_idx = valid_mask.nonzero(as_tuple=True)

        return {
            "dataset_name": dataset_name,
            "metric_kind": spec["metric_kind"],
            "arc_type": spec["arc_type"],
            "arc_index": arc_idx,
            "libcell_index": metadata["libcell_indices"][arc_idx],
            "main_id": metadata["libcell_main_ids"][arc_idx],
            "arc_offset": metadata["arc_offsets"][arc_idx],
            "size": metadata["sizes"][arc_idx],
            "log_leakage_coord": (
                metadata["log_leakage_coords"][arc_idx]
                if metadata.get("log_leakage_coords") is not None
                else None
            ),
            "vt": metadata["vts"][arc_idx],
            "slew": trans_table[arc_idx, row_idx],
            "cap": cap_table[arc_idx, col_idx],
            "target": values[arc_idx, row_idx, col_idx],
            "row_index": row_idx,
            "col_index": col_idx,
        }

    def _sample_point_indices(self, total_points, max_points=None):
        if total_points <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if max_points is None or max_points >= total_points:
            return torch.arange(total_points, device=self.device, dtype=torch.long)
        sampled = torch.linspace(
            0,
            total_points - 1,
            steps=max_points,
            device=self.device,
            dtype=torch.float32,
        ).round().long()
        return torch.unique(sampled, sorted=True)

    def _slice_point_payload(self, payload, point_indices):
        sliced = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == payload["target"].shape[0]:
                sliced[key] = value[point_indices]
            else:
                sliced[key] = value
        return sliced

    def _analysis_forward_size_payload(self, payload):
        if self._schema_uses_log_leakage_coord():
            coord = payload.get("log_leakage_coord")
            if coord is None:
                raise ValueError(
                    f"{self.cell_model_schema} analysis requires log_leakage_coord payload"
                )
            return coord
        return payload["size"]

    def _point_rows(self, payload, predicted, abs_error, rel_error, point_indices):
        rows = []
        predicted_cpu = predicted.detach().cpu()
        abs_cpu = abs_error.detach().cpu()
        rel_cpu = rel_error.detach().cpu()
        for local_idx in point_indices.detach().cpu().tolist():
            rows.append(
                {
                    "dataset_name": payload["dataset_name"],
                    "metric_kind": payload["metric_kind"],
                    "arc_index": int(payload["arc_index"][local_idx].item()),
                    "libcell_index": int(payload["libcell_index"][local_idx].item()),
                    "main_id": int(payload["main_id"][local_idx].item()),
                    "arc_offset": int(payload["arc_offset"][local_idx].item()),
                    "row_index": int(payload["row_index"][local_idx].item()),
                    "col_index": int(payload["col_index"][local_idx].item()),
                    "size": float(payload["size"][local_idx].item()),
                    "vt": float(payload["vt"][local_idx].item()),
                    "input_slew": float(payload["slew"][local_idx].item()),
                    "output_cap": float(payload["cap"][local_idx].item()),
                    "target": float(payload["target"][local_idx].item()),
                    "predicted": float(predicted_cpu[local_idx].item()),
                    "abs_error": float(abs_cpu[local_idx].item()),
                    "rel_error": float(rel_cpu[local_idx].item()),
                }
            )
        return rows

    def analyze_lut_error(self, dataset_name, max_points=None, top_k=100, include_all_points=False):
        payload = self._flatten_lut_points(dataset_name)
        total_points = int(payload["target"].numel())
        if total_points == 0:
            summary = {
                "dataset_name": dataset_name,
                "metric_kind": payload["metric_kind"],
                "total_points": 0,
                "evaluated_points": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "max_abs_error": 0.0,
                "p95_abs_error": 0.0,
                "mean_rel_error": 0.0,
                "r2": 0.0,
                "sum_rel_error": 0.0,
                "sum_abs_error": 0.0,
                "sum_squared_error": 0.0,
                "sum_target": 0.0,
                "sum_target_squared": 0.0,
            }
            return {
                "summary": summary,
                "topk_rows": [],
                "all_rows": [] if include_all_points else None,
            }

        sample_idx = self._sample_point_indices(total_points, max_points=max_points)
        payload = self._slice_point_payload(payload, sample_idx)
        evaluated_points = int(payload["target"].numel())
        arc_types = torch.full(
            (evaluated_points,),
            payload["arc_type"],
            dtype=torch.long,
            device=self.device,
        )
        previous_precomputed = getattr(self, "_analysis_size_coord_is_precomputed", False)
        self._analysis_size_coord_is_precomputed = self._schema_uses_log_leakage_coord()
        try:
            predicted = self.forward(
                payload["main_id"],
                payload["arc_offset"],
                payload["vt"],
                self._analysis_forward_size_payload(payload),
                payload["slew"],
                payload["cap"],
                arc_types=arc_types,
            )
        finally:
            self._analysis_size_coord_is_precomputed = previous_precomputed
        abs_error = (predicted - payload["target"]).abs()
        rel_error = abs_error / payload["target"].abs().clamp_min(1e-9)
        squared_error = abs_error.square()
        target = payload["target"]
        sum_squared_error = float(squared_error.sum().item())
        sum_target = float(target.sum().item())
        sum_target_squared = float(target.square().sum().item())

        top_k = max(1, int(top_k))
        actual_top_k = min(top_k, evaluated_points)
        topk_indices = torch.topk(abs_error, k=actual_top_k).indices

        p95_abs_error = float(torch.quantile(abs_error, 0.95).item()) if evaluated_points > 1 else float(abs_error.max().item())
        summary = {
            "dataset_name": dataset_name,
            "metric_kind": payload["metric_kind"],
            "total_points": total_points,
            "evaluated_points": evaluated_points,
            "mae": float(abs_error.mean().item()),
            "rmse": float(torch.sqrt(squared_error.mean()).item()),
            "max_abs_error": float(abs_error.max().item()),
            "p95_abs_error": p95_abs_error,
            "mean_rel_error": float(rel_error.mean().item()),
            "r2": float(_compute_r2(sum_squared_error, sum_target, sum_target_squared, evaluated_points)),
            "sum_rel_error": float(rel_error.sum().item()),
            "sum_abs_error": float(abs_error.sum().item()),
            "sum_squared_error": sum_squared_error,
            "sum_target": sum_target,
            "sum_target_squared": sum_target_squared,
        }

        point_range = torch.arange(evaluated_points, device=self.device, dtype=torch.long)
        return {
            "summary": summary,
            "topk_rows": self._point_rows(payload, predicted, abs_error, rel_error, topk_indices),
            "all_rows": self._point_rows(payload, predicted, abs_error, rel_error, point_range)
            if include_all_points
            else None,
        }

    def analyze_all_lut_errors(self, max_points_per_dataset=None, top_k=100, include_all_points=False):
        dataset_reports = {}
        aggregate_points = 0
        aggregate_abs_error = 0.0
        aggregate_rel_error = 0.0
        aggregate_squared_error = 0.0
        aggregate_max_abs_error = 0.0
        aggregate_target_sum = 0.0
        aggregate_target_squared_sum = 0.0
        for dataset_name in DATASET_SPECS:
            report = self.analyze_lut_error(
                dataset_name,
                max_points=max_points_per_dataset,
                top_k=top_k,
                include_all_points=include_all_points,
            )
            dataset_reports[dataset_name] = report
            summary = report["summary"]
            aggregate_points += int(summary["evaluated_points"])
            aggregate_abs_error += float(summary.get("sum_abs_error", 0.0))
            aggregate_rel_error += float(summary.get("sum_rel_error", 0.0))
            aggregate_squared_error += float(summary.get("sum_squared_error", 0.0))
            aggregate_max_abs_error = max(aggregate_max_abs_error, float(summary["max_abs_error"]))
            aggregate_target_sum += float(summary.get("sum_target", 0.0))
            aggregate_target_squared_sum += float(summary.get("sum_target_squared", 0.0))

        aggregate = {
            "evaluated_points": aggregate_points,
            "mae": 0.0,
            "rmse": 0.0,
            "max_abs_error": aggregate_max_abs_error,
            "mean_rel_error": 0.0,
            "r2": 0.0,
        }
        if aggregate_points > 0:
            aggregate["mae"] = aggregate_abs_error / aggregate_points
            aggregate["mean_rel_error"] = aggregate_rel_error / aggregate_points
            aggregate["rmse"] = math.sqrt(aggregate_squared_error / aggregate_points)
            aggregate["r2"] = _compute_r2(
                aggregate_squared_error,
                aggregate_target_sum,
                aggregate_target_squared_sum,
                aggregate_points,
            )

        return {
            "cache": self._analysis_cache_identity(),
            "datasets": dataset_reports,
            "aggregate": aggregate,
        }

    def _forward_impl(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types=None):
        """
        Unified Inference.
        
        Args:
            libcell_main_id: [N] LongTensor
            arc_offset: [N] LongTensor
            vt: [N] Tensor (Float, differentiable)
            size: [N] Tensor (Float, differentiable)
            input_slew: [N] Tensor (Float, differentiable)
            out_cap: [N] Tensor (Float, differentiable)
            arc_types: [N] LongTensor (0: cell_fall, 1: cell_rise, 2: fall_trans, 3: rise_trans)
                       If None, returns a dict with all metrics.
        
        Returns:
            If arc_types is provided: Tensor [N] (Values)
            If arc_types is None: Dict {'cell_fall': ..., 'cell_rise': ...}
        """
        if self.cell_model_schema == PIECEWISE_LINEAR_SCHEMA:
            if arc_types is not None:
                return self._forward_selected_arc_types(
                    libcell_main_id,
                    arc_offset,
                    vt,
                    size,
                    input_slew,
                    out_cap,
                    arc_types,
                )
            return {
                "cell_fall": self._forward_dataset("f_delay", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                "cell_rise": self._forward_dataset("r_delay", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                "fall_transition": self._forward_dataset("f_trans", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                "rise_transition": self._forward_dataset("r_trans", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
            }

        if arc_types is not None:
            return self._forward_selected_arc_types(
                libcell_main_id,
                arc_offset,
                vt,
                size,
                input_slew,
                out_cap,
                arc_types,
            )
        else:
            return {
                'cell_fall': self._forward_dataset("f_delay", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                'cell_rise': self._forward_dataset("r_delay", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                'fall_transition': self._forward_dataset("f_trans", libcell_main_id, arc_offset, vt, size, input_slew, out_cap),
                'rise_transition': self._forward_dataset("r_trans", libcell_main_id, arc_offset, vt, size, input_slew, out_cap)
            }

    def forward(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types=None):
        if not getattr(self, "_cell_model_forward_profile_enabled", False):
            return self._forward_impl(libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types)

        start_time = time.perf_counter()
        result = self._forward_impl(libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types)
        elapsed_seconds = time.perf_counter() - start_time
        self._record_forward_profile(input_slew.numel(), arc_types, elapsed_seconds)
        return result
