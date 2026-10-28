"""Shared cell-model queries for scalar pins and sparse timing states.

The caller supplies electrical vectors and physical output-pin identity;
size/VT lookup stays in TimingPropagation's existing cell-model boundary.
"""

from dataclasses import dataclass
import logging

import torch


@dataclass
class CellAatUpdates:
    lanes: tuple[torch.Tensor, ...]
    delays: tuple[torch.Tensor | None, ...]
    profile: dict | None


def query_cell_aat_updates(
    self,
    level_inst_arcs,
    pin_rAAT,
    pin_fAAT,
    pin_rtran,
    pin_ftran,
    pin_net_cap_rise,
    pin_net_cap_fall,
    *,
    use_surrogate,
    need_delay_updates=True,
    profile_level=None,
    physical_pins=None,
):
    from .timing_propagation import SMOOTH_MAX_ALPHA, smooth_max

    device = pin_rAAT.device
    static_level_data = (
        self._get_cell_aat_static_level_data(level_inst_arcs)
        if physical_pins is None and getattr(self, "use_cell_aat_static_cache", False)
        else None
    )

    if static_level_data is None:
        arc_in_pins = level_inst_arcs[:, 0]
        arc_out_pins = level_inst_arcs[:, 1]
        lib_cell_idxs = level_inst_arcs[:, 2]
        lib_arc_idxs = level_inst_arcs[:, 3]
        timing_senses = level_inst_arcs[:, 4]
    else:
        arc_in_pins = static_level_data["arc_in_pins"]
        arc_out_pins = static_level_data["arc_out_pins"]
        lib_cell_idxs = static_level_data["lib_cell_idxs"]
        lib_arc_idxs = static_level_data["lib_arc_idxs"]

    if (
        physical_pins is None
        and torch.logical_and(arc_out_pins == 2485, arc_in_pins == 3200).any()
    ):
        logging.warning(
            f"flat arc index is: {torch.where(torch.logical_and(arc_out_pins == 2485, arc_in_pins == 3200))}"
        )

    # 1. 准备输入条件 (Prepare inputs)
    pin_r_slew_in = pin_rtran[arc_in_pins]
    pin_f_slew_in = pin_ftran[arc_in_pins]
    pin_r_load_out = pin_net_cap_rise[arc_out_pins]
    pin_f_load_out = pin_net_cap_fall[arc_out_pins]

    # 2. 根据 Unate 类型进行分组 (Group arcs by unate type)
    if static_level_data is None:
        is_pos_unate = timing_senses == 1
        is_neg_unate = timing_senses == -1
        is_non_unate = ~(is_pos_unate | is_neg_unate)
        has_pos_unate = torch.any(is_pos_unate)
        has_neg_unate = torch.any(is_neg_unate)
        has_non_unate = torch.any(is_non_unate)
    else:
        is_pos_unate = static_level_data["pos_arc_indices"]
        is_neg_unate = static_level_data["neg_arc_indices"]
        is_non_unate = static_level_data["non_arc_indices"]
        has_pos_unate = static_level_data["has_pos_unate"]
        has_neg_unate = static_level_data["has_neg_unate"]
        has_non_unate = static_level_data["has_non_unate"]

    num_level_arcs = level_inst_arcs.shape[0]

    r_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
    f_trans = torch.zeros(num_level_arcs, device=device, dtype=self.dtype)
    # 初始化 AAT 更新张量
    r_aat_updates = torch.full_like(
        r_trans, -float("inf"), device=device, dtype=self.dtype
    )
    f_aat_updates = torch.full_like(
        r_trans, -float("inf"), device=device, dtype=self.dtype
    )
    # --- 3. 按需计算每种 Unate 类型的 Delay 和 Tran ---
    if need_delay_updates:
        delay_rr_updates = torch.full_like(
            r_trans, -float("inf"), device=device, dtype=self.dtype
        )
        delay_rf_updates = torch.full_like(
            r_trans, -float("inf"), device=device, dtype=self.dtype
        )
        delay_fr_updates = torch.full_like(
            r_trans, -float("inf"), device=device, dtype=self.dtype
        )
        delay_ff_updates = torch.full_like(
            r_trans, -float("inf"), device=device, dtype=self.dtype
        )
    else:
        delay_rr_updates = delay_rf_updates = None
        delay_fr_updates = delay_ff_updates = None
    profile_enabled = bool(getattr(self, "timing_propagation_profile", False))
    level_profile = None
    if profile_enabled:
        level_profile = {
            "level": None if profile_level is None else int(profile_level),
            "num_arcs": int(num_level_arcs),
            "query_count": 0,
            "query_build_ms": 0.0,
            "query_eval_ms": 0.0,
            "cell_update_ms": 0.0,
            "unique_net_ms": 0.0,
        }
        for key in self._cell_aat_detail_count_keys():
            level_profile[key] = 0
        profile_timer = self._sync_profile_clock(device)

    level_queries = []

    def register_level_query(name, mask, pin_slew, pin_load, arc_type, luts):
        level_queries.append(
            {
                "name": name,
                "lib_cell_idxs": lib_cell_idxs[mask],
                "pin_slew": pin_slew[mask],
                "pin_net_caps": pin_load[mask],
                "lib_arc_idxs": lib_arc_idxs[mask],
                "luts": luts,
                "node_ids": (
                    arc_out_pins
                    if physical_pins is None
                    else physical_pins[arc_out_pins]
                )[mask],
                "arc_type": arc_type,
            }
        )

    if has_pos_unate:
        register_level_query(
            "pos_delay_rr",
            is_pos_unate,
            pin_r_slew_in,
            pin_r_load_out,
            1,
            self.arcs_info.r_delay_luts,
        )
        register_level_query(
            "pos_tran_rr",
            is_pos_unate,
            pin_r_slew_in,
            pin_r_load_out,
            3,
            self.arcs_info.r_trans_luts,
        )
        register_level_query(
            "pos_delay_ff",
            is_pos_unate,
            pin_f_slew_in,
            pin_f_load_out,
            0,
            self.arcs_info.f_delay_luts,
        )
        register_level_query(
            "pos_tran_ff",
            is_pos_unate,
            pin_f_slew_in,
            pin_f_load_out,
            2,
            self.arcs_info.f_trans_luts,
        )

    if has_neg_unate:
        register_level_query(
            "neg_delay_fr",
            is_neg_unate,
            pin_f_slew_in,
            pin_r_load_out,
            1,
            self.arcs_info.r_delay_luts,
        )
        register_level_query(
            "neg_tran_fr",
            is_neg_unate,
            pin_f_slew_in,
            pin_r_load_out,
            3,
            self.arcs_info.r_trans_luts,
        )
        register_level_query(
            "neg_delay_rf",
            is_neg_unate,
            pin_r_slew_in,
            pin_f_load_out,
            0,
            self.arcs_info.f_delay_luts,
        )
        register_level_query(
            "neg_tran_rf",
            is_neg_unate,
            pin_r_slew_in,
            pin_f_load_out,
            2,
            self.arcs_info.f_trans_luts,
        )

    if has_non_unate:
        register_level_query(
            "non_delay_rr",
            is_non_unate,
            pin_r_slew_in,
            pin_r_load_out,
            1,
            self.arcs_info.r_delay_luts,
        )
        register_level_query(
            "non_delay_fr",
            is_non_unate,
            pin_f_slew_in,
            pin_r_load_out,
            1,
            self.arcs_info.r_delay_luts,
        )
        register_level_query(
            "non_tran_rr",
            is_non_unate,
            pin_r_slew_in,
            pin_r_load_out,
            3,
            self.arcs_info.r_trans_luts,
        )
        register_level_query(
            "non_tran_fr",
            is_non_unate,
            pin_f_slew_in,
            pin_r_load_out,
            3,
            self.arcs_info.r_trans_luts,
        )
        register_level_query(
            "non_delay_ff",
            is_non_unate,
            pin_f_slew_in,
            pin_f_load_out,
            0,
            self.arcs_info.f_delay_luts,
        )
        register_level_query(
            "non_delay_rf",
            is_non_unate,
            pin_r_slew_in,
            pin_f_load_out,
            0,
            self.arcs_info.f_delay_luts,
        )
        register_level_query(
            "non_tran_ff",
            is_non_unate,
            pin_f_slew_in,
            pin_f_load_out,
            2,
            self.arcs_info.f_trans_luts,
        )
        register_level_query(
            "non_tran_rf",
            is_non_unate,
            pin_r_slew_in,
            pin_f_load_out,
            2,
            self.arcs_info.f_trans_luts,
        )

    if level_profile is not None:
        profile_now = self._sync_profile_clock(device)
        level_profile["query_build_ms"] = (profile_now - profile_timer) * 1000.0
        level_profile["query_count"] = len(level_queries)
        profile_timer = profile_now

    query_results = {
        query["name"]: result
        for query, result in zip(
            level_queries,
            self._entry_queries_with_optional_surrogate(
                level_queries,
                use_surrogate,
                profile=level_profile,
            ),
        )
    }
    if level_profile is not None:
        profile_now = self._sync_profile_clock(device)
        level_profile["query_eval_ms"] = (profile_now - profile_timer) * 1000.0
        profile_timer = profile_now

    # A. Positive Unate (r->r, f->f)
    if has_pos_unate:
        mask = is_pos_unate
        delay_rr = query_results["pos_delay_rr"]
        tran_rr = query_results["pos_tran_rr"]
        delay_ff = query_results["pos_delay_ff"]
        tran_ff = query_results["pos_tran_ff"]

        if need_delay_updates:
            delay_rr_updates[mask] = delay_rr
            delay_ff_updates[mask] = delay_ff
        r_trans[mask] = tran_rr
        f_trans[mask] = tran_ff
        r_aat_updates[mask] = pin_rAAT[arc_in_pins[mask]] + delay_rr
        f_aat_updates[mask] = pin_fAAT[arc_in_pins[mask]] + delay_ff

    # B. Negative Unate (f->r, r->f)
    if has_neg_unate:
        mask = is_neg_unate
        delay_fr = query_results["neg_delay_fr"]
        tran_fr = query_results["neg_tran_fr"]
        delay_rf = query_results["neg_delay_rf"]
        tran_rf = query_results["neg_tran_rf"]

        if need_delay_updates:
            delay_fr_updates[mask] = delay_fr
            delay_rf_updates[mask] = delay_rf
        r_trans[mask] = tran_fr
        f_trans[mask] = tran_rf
        r_aat_updates[mask] = (
            pin_fAAT[arc_in_pins[mask]] + delay_fr
        )  # <-- 使用 pin_fAAT
        f_aat_updates[mask] = (
            pin_rAAT[arc_in_pins[mask]] + delay_rf
        )  # <-- 使用 pin_rAAT

    # C. Non-Unate (worst of all applicable cases)
    if has_non_unate:
        mask = is_non_unate
        delay_rr_non = query_results["non_delay_rr"]
        delay_fr_non = query_results["non_delay_fr"]
        tran_rr_non = query_results["non_tran_rr"]
        tran_fr_non = query_results["non_tran_fr"]
        if need_delay_updates:
            delay_fr_updates[mask] = delay_fr_non
            delay_rr_updates[mask] = delay_rr_non

        # unate_r_delays = torch.maximum(delay_rr_non, delay_fr_non)
        r_trans[mask] = smooth_max(tran_rr_non, tran_fr_non, alpha=SMOOTH_MAX_ALPHA)

        aat_rr = pin_rAAT[arc_in_pins[mask]] + delay_rr_non
        aat_fr = pin_fAAT[arc_in_pins[mask]] + delay_fr_non
        if level_inst_arcs.shape[1] > 8:
            transitions = level_inst_arcs[mask, 8]
            aat_rr = torch.where(transitions & 1 != 0, aat_rr, -2e8)
            aat_fr = torch.where(transitions & 4 != 0, aat_fr, -2e8)
        r_aat_updates[mask] = smooth_max(aat_rr, aat_fr, alpha=SMOOTH_MAX_ALPHA)

        # Fall output calculation (worst of f->f and r->f)
        delay_ff_non = query_results["non_delay_ff"]
        delay_rf_non = query_results["non_delay_rf"]
        tran_ff_non = query_results["non_tran_ff"]
        tran_rf_non = query_results["non_tran_rf"]

        f_trans[mask] = smooth_max(tran_ff_non, tran_rf_non, alpha=SMOOTH_MAX_ALPHA)

        if need_delay_updates:
            delay_ff_updates[mask] = delay_ff_non
            delay_rf_updates[mask] = delay_rf_non

        aat_ff = pin_fAAT[arc_in_pins[mask]] + delay_ff_non
        aat_rf = pin_rAAT[arc_in_pins[mask]] + delay_rf_non
        if level_inst_arcs.shape[1] > 8:
            aat_ff = torch.where(transitions & 8 != 0, aat_ff, -2e8)
            aat_rf = torch.where(transitions & 2 != 0, aat_rf, -2e8)
        f_aat_updates[mask] = smooth_max(aat_ff, aat_rf, alpha=SMOOTH_MAX_ALPHA)

    if level_inst_arcs.shape[1] > 8:
        r_aat_updates = torch.where(level_inst_arcs[:, 8] & 5 != 0, r_aat_updates, -2e8)
        f_aat_updates = torch.where(level_inst_arcs[:, 8] & 10 != 0, f_aat_updates, -2e8)

    return CellAatUpdates(
        (r_aat_updates, f_aat_updates, r_trans, f_trans),
        (delay_rr_updates, delay_fr_updates, delay_rf_updates, delay_ff_updates),
        level_profile,
    )
