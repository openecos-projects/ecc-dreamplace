"""Production master metadata and setup action probes on a native export.

This boundary also accepts legacy setup-only exports, which do not carry
min timing data or the later check-class column. Their original action matcher
handles that absent column; no min model or check class is fabricated here.
"""

from types import SimpleNamespace

import numpy as np
import torch
from dreamplace.macroPlaceDB import MacroPlaceDB
from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    build_quad_candidate_table,
)
from dreamplace.ops.timing_propagation.discrete_actions import build_discrete_action_context
from fixed_sta import production_timing


class NativeActionCase:
    def __init__(self, pydb, view, base, *, device="cpu", timing_factory=production_timing):
        placedb = MacroPlaceDB(None)
        placedb.dtype = np.float64
        for name in (
            "main_id_2_cell_id_start",
            "cell_id_2_arc_id_start",
            "inst_main_id",
            "inst_libcell_offset",
            "main_id_is_sizeable",
        ):
            setattr(placedb, name, np.asarray(getattr(pydb, name)))
        placedb.flat_libcell_names, placedb.flat_libcell_info = (
            placedb._normalize_flat_libcell_info(pydb.flat_libcell_info)
        )
        placedb.flat_libarc_names, placedb.flat_libarc_info = placedb._normalize_flat_libarc_info(
            pydb.flat_libarc_info
        )
        placedb.flat_libcell_leakage = placedb._normalize_flat_libcell_leakage(
            pydb.flat_libcell_leakage, len(placedb.flat_libcell_info)
        )
        _, placedb.flat_libcell_main_id2size_vt_limit = placedb._normalize_flat_libcell_limit_info(
            pydb.flat_libcell_main_id2size_vt_limit
        )
        metadata = placedb._build_inst_sizing_metadata()
        names = (
            "inst_cell_id",
            "inst_size_init",
            "inst_leakage_init",
            "inst_vt_init",
            "inst_size_lower",
            "inst_size_upper",
            "inst_vt_mask",
            "inst_is_sizeable",
        )
        for name, value in zip(names, metadata, strict=True):
            setattr(placedb, name, value)
        self.placedb = placedb
        self.data = SimpleNamespace()
        for name in (
            "flat_libcell_info",
            "flat_libcell_leakage",
            "flat_libarc_info",
            "main_id_2_cell_id_start",
            "cell_id_2_arc_id_start",
            "inst_main_id",
            "inst_libcell_offset",
            "inst_cell_id",
            "inst_is_sizeable",
            "inst_vt_mask",
        ):
            setattr(self.data, name, torch.from_numpy(getattr(placedb, name)))
        for name in (
            "flat_inst_arcs_by_level",
            "endpoints_timing_check_arcs",
            "endpoints_constraint_arcs",
            "pin2node_map",
            "pin2net_map",
            "net2driver_pin_map",
            "cell_id_2_libpin_id_start",
            "pin_2_libpin_offset",
            "end_points",
        ):
            value = np.asarray(getattr(pydb, name), dtype=np.int64)
            if name in ("endpoints_timing_check_arcs", "endpoints_constraint_arcs"):
                value = value.reshape(-1, 8)
            setattr(self.data, name, torch.from_numpy(value))
        if hasattr(pydb, "flat_libarc_check_types"):
            self.data.flat_libarc_check_types = torch.as_tensor(
                pydb.flat_libarc_check_types, dtype=torch.int64
            )
        device = torch.device(device)
        for name, value in vars(self.data).items():
            setattr(self.data, name, value.to(device))
        self.timing = timing_factory(view, base, device)
        self.timing.start_points = torch.tensor(pydb.start_points, dtype=torch.long, device=device)
        self.candidates = build_quad_candidate_table(
            self.data.flat_libcell_info, self.data.flat_libcell_leakage
        )
        self.context = build_discrete_action_context(
            self.data,
            self.candidates,
            action_names=ACTION_NAMES,
            pin_names=pydb.flat_lib_pin_names,
            device=device,
            dtype=torch.float64,
        )
        self.context.evaluate_targets = True
        self.timing._active_discrete_action_context = self.context
        nodes = self.data.pin2node_map
        offsets = self.data.pin_2_libpin_offset
        main = self.data.inst_main_id[nodes]
        pin_mask = (main >= 0) & (offsets >= 0)
        cells = self.data.main_id_2_cell_id_start[main[pin_mask]].long()
        cells += self.data.inst_libcell_offset[nodes[pin_mask]].long()
        libpins = self.data.cell_id_2_libpin_id_start[cells] + offsets[pin_mask]
        self.model = SimpleNamespace(
            params=SimpleNamespace(timing_action_pin_cap_gradient=1),
            data_collections=self.data,
            _active_discrete_action_context=self.context,
            _current_timing_surrogate_mode=lambda: "lut_only",
            pin_2_libpin_ids=lambda offset, data: (pin_mask, libpins),
        )
        self.zeros = torch.zeros(self.timing.num_pins, dtype=torch.float64, device=device)
