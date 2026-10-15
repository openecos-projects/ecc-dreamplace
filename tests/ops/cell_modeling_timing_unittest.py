import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation
sys.path.pop()


class FakeCellModeling(torch.nn.Module):
    def forward(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types=None):
        value = (
            0.5 * libcell_main_id.float()
            + 0.25 * arc_offset.float()
            + 1.5 * vt.float()
            + 2.0 * size.float()
            + 0.75 * input_slew.float()
            + 0.5 * out_cap.float()
        )
        if arc_types is not None:
            value = value + 0.1 * arc_types.float()
        return value


class FakeCountingCellModeling(FakeCellModeling):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.forward_batch_sizes = []

    def forward(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types=None):
        self.forward_calls += 1
        self.forward_batch_sizes.append(int(input_slew.numel()))
        return super().forward(
            libcell_main_id,
            arc_offset,
            vt,
            size,
            input_slew,
            out_cap,
            arc_types=arc_types,
        )


class FakeSelectiveCellModeling(torch.nn.Module):
    def supports_arc_types(self, libcell_main_id, arc_offset, arc_types):
        del libcell_main_id, arc_types
        return arc_offset == 44

    def forward(self, libcell_main_id, arc_offset, vt, size, input_slew, out_cap, arc_types=None):
        del libcell_main_id, vt, size, input_slew, out_cap, arc_types
        return arc_offset.float() + 100.0


class FakeSupportCountingCellModeling(FakeCellModeling):
    def __init__(self):
        super().__init__()
        self.support_calls = 0

    def supports_arc_types(self, libcell_main_id, arc_offset, arc_types):
        del libcell_main_id, arc_types
        self.support_calls += 1
        return torch.ones(arc_offset.shape, dtype=torch.bool, device=arc_offset.device)


class CellModelingTimingHookTest(unittest.TestCase):
    def _build_tp(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.timing_aggregation_mode = "hard"
        tp.timing_aggregation_tau_ps = 1.0
        tp.cell_modeling_op = FakeCellModeling()
        tp.pin2node_map = torch.tensor([0, 1, 0], dtype=torch.int64)
        tp.lib_arc_offsets = None
        tp.inst_main_id = torch.tensor([0, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, True], dtype=torch.bool)
        tp.pin_net = torch.tensor([0, 0, 0], dtype=torch.int64)
        size_logits = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
        vt_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32, requires_grad=True)
        size_lower = torch.tensor([1.0, 1.0], dtype=torch.float32)
        size_upper = torch.tensor([4.0, 4.0], dtype=torch.float32)

        def get_size_var():
            size_norm = torch.sigmoid(size_logits)
            return size_lower + size_norm * (size_upper - size_lower)

        def get_vt_var():
            return torch.softmax(vt_logits, dim=1)

        tp.size_var_getter = get_size_var
        tp.vt_var_getter = get_vt_var
        tp.arcs_info = SimpleNamespace(
            r_delay_luts=object(),
            f_delay_luts=object(),
            r_trans_luts=object(),
            f_trans_luts=object(),
        )
        tp.lut_entry_vectorized = lambda lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts: torch.zeros_like(pin_slew)
        return tp, size_logits, vt_logits

    def _build_tp_with_counted_state_getters(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.cell_modeling_op = FakeCellModeling()
        tp.pin2node_map = torch.tensor([0, 1, 2], dtype=torch.int64)
        tp.lib_arc_offsets = None
        tp.inst_main_id = torch.tensor([0, 0, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, True, True], dtype=torch.bool)
        tp.pin_net = torch.tensor([0, 0, 0], dtype=torch.int64)
        tp._surrogate_state_cache = None
        call_counts = {"size": 0, "vt": 0}
        size_var = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)
        vt_var = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.25, 0.75],
            ],
            dtype=torch.float32,
        )

        def get_size_var():
            call_counts["size"] += 1
            return size_var

        def get_vt_var():
            call_counts["vt"] += 1
            return vt_var

        tp.size_var_getter = get_size_var
        tp.vt_var_getter = get_vt_var
        tp.arcs_info = SimpleNamespace(
            r_delay_luts=object(),
            f_delay_luts=object(),
            r_trans_luts=object(),
            f_trans_luts=object(),
        )
        tp.lut_entry_vectorized = (
            lambda lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts: torch.zeros_like(pin_slew)
        )
        return tp, call_counts

    def _build_tp_with_counted_surrogate_forward(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.timing_aggregation_mode = "hard"
        tp.timing_aggregation_tau_ps = 1.0
        tp.cell_modeling_op = FakeCountingCellModeling()
        tp.pin2node_map = torch.tensor([0, 1, 2], dtype=torch.int64)
        tp.lib_arc_offsets = None
        tp.inst_main_id = torch.tensor([0, 0, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, True, True], dtype=torch.bool)
        tp.pin_net = torch.tensor([0, 0, 0], dtype=torch.int64)
        tp.size_var_getter = lambda: torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)
        tp.vt_var_getter = lambda: torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.25, 0.75],
            ],
            dtype=torch.float32,
        )
        tp.arcs_info = SimpleNamespace(
            r_delay_luts=object(),
            f_delay_luts=object(),
            r_trans_luts=object(),
            f_trans_luts=object(),
        )
        tp.lut_entry_vectorized = (
            lambda lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts: torch.zeros_like(pin_slew)
        )
        return tp

    def test_surrogate_changes_delay_and_transition_outputs(self):
        tp, size_logits, vt_logits = self._build_tp()
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        arc_offsets = torch.tensor([1, 2], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4], dtype=torch.float32)
        cap = torch.tensor([1.0, 3.0], dtype=torch.float32)

        delay_before = tp.r_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
        tran_before = tp.f_tran_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)

        with torch.no_grad():
            size_logits.add_(0.75)
            vt_logits[0, 0] -= 1.5
            vt_logits[0, 1] += 1.5

        delay_after = tp.r_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
        tran_after = tp.f_tran_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)

        self.assertFalse(torch.allclose(delay_before, delay_after))
        self.assertFalse(torch.allclose(tran_before, tran_after))

    def test_surrogate_backprop_reaches_sizing_tensors(self):
        tp, size_logits, vt_logits = self._build_tp()
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        arc_offsets = torch.tensor([1, 2], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.3, 0.5], dtype=torch.float32)
        cap = torch.tensor([1.5, 2.5], dtype=torch.float32)

        delay = tp.r_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
        tran = tp.r_tran_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
        loss = delay.sum() + tran.sum()
        loss.backward()

        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(vt_logits.grad.abs().sum().item(), 0.0)

    def test_surrogate_uses_local_arc_offset_mapping_instead_of_global_arc_index(self):
        tp, _, _ = self._build_tp()
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        global_arc_idxs = torch.tensor([4, 5], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4], dtype=torch.float32)
        cap = torch.tensor([1.0, 3.0], dtype=torch.float32)

        delay = tp.r_delay_entry(lib_cell_idxs, slew, cap, global_arc_idxs, node_pins)

        size_var = tp.size_var_getter()
        vt_scalar = tp._vt_scalar(tp.vt_var_getter()[torch.tensor([0, 1], dtype=torch.int64)])
        expected = (
            0.5 * torch.tensor([0.0, 0.0], dtype=torch.float32)
            + 0.25 * torch.tensor([44.0, 55.0], dtype=torch.float32)
            + 1.5 * vt_scalar
            + 2.0 * size_var[torch.tensor([0, 1], dtype=torch.int64)]
            + 0.75 * slew
            + 0.5 * cap
            + 0.1 * 1.0
        )

        self.assertTrue(torch.allclose(delay.detach(), expected, atol=1e-4))

    def test_surrogate_falls_back_to_lut_for_unsupported_arc_offsets(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSelectiveCellModeling()
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.lut_entry_vectorized = (
            lambda lib_cell_idxs, pin_slew, pin_net_caps, lib_arc_idxs, luts: torch.full_like(pin_slew, 7.0)
        )
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        global_arc_idxs = torch.tensor([4, 5], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4], dtype=torch.float32)
        cap = torch.tensor([1.0, 3.0], dtype=torch.float32)

        delay = tp.r_delay_entry(lib_cell_idxs, slew, cap, global_arc_idxs, node_pins)

        self.assertEqual(delay[0].item(), 144.0)
        self.assertEqual(delay[1].item(), 7.0)

    def test_surrogate_supported_arc_skips_lut_lookup(self):
        tp, _, _ = self._build_tp()
        lut_calls = {"count": 0}

        def lut_counting(_lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts):
            lut_calls["count"] += 1
            return torch.full_like(pin_slew, -7.0)

        tp.lut_entry_vectorized = lut_counting
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        arc_offsets = torch.tensor([1, 2], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4], dtype=torch.float32)
        cap = torch.tensor([1.0, 3.0], dtype=torch.float32)

        delay = tp.r_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)

        self.assertEqual(lut_calls["count"], 0)
        self.assertTrue(torch.all(delay > 0.0))

    def test_surrogate_lut_fallback_only_computes_unsupported_subset(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSelectiveCellModeling()
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        lut_calls = {"count": 0, "batch_sizes": []}

        def lut_counting(_lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts):
            lut_calls["count"] += 1
            lut_calls["batch_sizes"].append(pin_slew.numel())
            return torch.full_like(pin_slew, 7.0)

        tp.lut_entry_vectorized = lut_counting
        lib_cell_idxs = torch.tensor([0, 0], dtype=torch.int64)
        global_arc_idxs = torch.tensor([4, 5], dtype=torch.int64)
        node_pins = torch.tensor([2, 1], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4], dtype=torch.float32)
        cap = torch.tensor([1.0, 3.0], dtype=torch.float32)

        delay = tp.r_delay_entry(lib_cell_idxs, slew, cap, global_arc_idxs, node_pins)

        self.assertEqual(delay[0].item(), 144.0)
        self.assertEqual(delay[1].item(), 7.0)
        self.assertEqual(lut_calls["count"], 1)
        self.assertEqual(lut_calls["batch_sizes"], [1])

    def test_main_cell_aat_level_uses_surrogate_intrinsic_gradients(self):
        tp, size_logits, vt_logits = self._build_tp()
        inst_arcs = torch.tensor([[0, 2, 0, 1, 1]], dtype=torch.int64)
        pin_rAAT = torch.zeros(3, dtype=torch.float32)
        pin_fAAT = torch.zeros(3, dtype=torch.float32)
        pin_rtran = torch.tensor([0.3, 0.4, 0.0], dtype=torch.float32)
        pin_ftran = torch.tensor([0.2, 0.1, 0.0], dtype=torch.float32)
        pin_net_cap_rise = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)
        pin_net_cap_fall = torch.tensor([0.0, 0.0, 2.5], dtype=torch.float32)

        _, out_rAAT, out_fAAT, out_rtran, out_ftran, *_ = tp.calculate_cell_aat_level(
            inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
        )
        loss = out_rAAT.sum() + out_fAAT.sum() + out_rtran.sum() + out_ftran.sum()
        loss.backward()

        self.assertGreater(size_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(vt_logits.grad.abs().sum().item(), 0.0)

    def test_clk2q_uses_exported_clock_slew_instead_of_runtime_pin_transition(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.device = torch.device("cpu")
        tp.dtype = torch.float32
        tp.timing_aggregation_mode = "hard"
        tp.timing_aggregation_tau_ps = 1.0
        tp.cell_modeling_op = None
        tp.pin2node_map = None
        tp.inst_main_id = None
        tp.inst_is_sizeable = None
        tp.size_var_getter = None
        tp.vt_var_getter = None
        tp.flat_inst_arcs_by_level_start = torch.tensor([0, 1], dtype=torch.int64)
        tp.flat_inst_arcs_by_level = torch.tensor([[0, 1, 0, 0, 1, 1]], dtype=torch.int64)
        tp.pin_net = torch.tensor([0, 0], dtype=torch.int64)
        tp.FF_ids = torch.tensor([0], dtype=torch.int64)
        tp.clk_pin_rtran = torch.tensor([125.0], dtype=torch.float32)
        tp.clk_pin_ftran = torch.tensor([75.0], dtype=torch.float32)

        def _r_delay_entry(_lib_cell_idxs, pin_rtrans, _pin_net_caps, _lib_arc_idxs, node_ids=None, use_surrogate=True):
            return pin_rtrans.clone()

        def _f_delay_entry(_lib_cell_idxs, pin_ftrans, _pin_net_caps, _lib_arc_idxs, node_ids=None, use_surrogate=True):
            return pin_ftrans.clone()

        def _r_tran_entry(_lib_cell_idxs, pin_rtrans, _pin_net_caps, _lib_arc_idxs, node_ids=None, use_surrogate=True):
            return pin_rtrans.clone()

        def _f_tran_entry(_lib_cell_idxs, pin_ftrans, _pin_net_caps, _lib_arc_idxs, node_ids=None, use_surrogate=True):
            return pin_ftrans.clone()

        tp.r_delay_entry = _r_delay_entry
        tp.f_delay_entry = _f_delay_entry
        tp.r_tran_entry = _r_tran_entry
        tp.f_tran_entry = _f_tran_entry

        pin_rAAT = torch.full((2,), -1.0, dtype=torch.float32)
        pin_fAAT = torch.full((2,), -1.0, dtype=torch.float32)
        pin_rtran = torch.zeros(2, dtype=torch.float32)
        pin_ftran = torch.zeros(2, dtype=torch.float32)
        pin_net_cap_rise = torch.ones(2, dtype=torch.float32)
        pin_net_cap_fall = torch.ones(2, dtype=torch.float32)

        _, out_rAAT, out_fAAT, out_rtran, out_ftran = tp.calculate_clk2q_aat(
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
        )

        self.assertEqual(out_rAAT[1].item(), 125.0)
        self.assertEqual(out_rtran[1].item(), 125.0)
        self.assertLess(out_fAAT[1].item(), 0.0)
        self.assertEqual(out_ftran[1].item(), 0.0)

    def test_surrogate_mode_switch_can_override_per_call_behavior(self):
        tp, _, _ = self._build_tp()

        clk2q_use_surrogate, cell_use_surrogate = tp._resolve_surrogate_mode("mixed")
        self.assertFalse(clk2q_use_surrogate)
        self.assertTrue(cell_use_surrogate)

        clk2q_use_surrogate, cell_use_surrogate = tp._resolve_surrogate_mode("lut_only")
        self.assertFalse(clk2q_use_surrogate)
        self.assertFalse(cell_use_surrogate)

        clk2q_use_surrogate, cell_use_surrogate = tp._resolve_surrogate_mode("surrogate_only")
        self.assertTrue(clk2q_use_surrogate)
        self.assertTrue(cell_use_surrogate)

    def test_surrogate_state_cache_reuses_size_and_vt_within_one_forward(self):
        tp, call_counts = self._build_tp_with_counted_state_getters()
        lib_cell_idxs = torch.tensor([0, 0, 0], dtype=torch.int64)
        arc_offsets = torch.tensor([1, 2, 3], dtype=torch.int64)
        node_pins = torch.tensor([0, 1, 2], dtype=torch.int64)
        slew = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
        cap = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        tp._begin_forward_runtime_state()
        try:
            tp.r_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
            tp.f_delay_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
            tp.r_tran_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
            tp.f_tran_entry(lib_cell_idxs, slew, cap, arc_offsets, node_pins)
        finally:
            tp._end_forward_runtime_state()

        self.assertEqual(call_counts["size"], 1)
        self.assertEqual(call_counts["vt"], 1)

    def test_cell_aat_level_batches_positive_unate_surrogate_queries_into_one_forward(self):
        tp = self._build_tp_with_counted_surrogate_forward()
        inst_arcs = torch.tensor(
            [
                [0, 2, 0, 1, 1],
                [1, 2, 0, 2, 1],
            ],
            dtype=torch.int64,
        )
        pin_rAAT = torch.zeros(3, dtype=torch.float32)
        pin_fAAT = torch.zeros(3, dtype=torch.float32)
        pin_rtran = torch.tensor([0.3, 0.4, 0.0], dtype=torch.float32)
        pin_ftran = torch.tensor([0.2, 0.1, 0.0], dtype=torch.float32)
        pin_net_cap_rise = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)
        pin_net_cap_fall = torch.tensor([0.0, 0.0, 2.5], dtype=torch.float32)

        tp._begin_forward_runtime_state()
        try:
            tp.calculate_cell_aat_level(
                inst_arcs,
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                pin_net_cap_rise,
                pin_net_cap_fall,
                use_surrogate=True,
            )
        finally:
            tp._end_forward_runtime_state()

        self.assertEqual(tp.cell_modeling_op.forward_calls, 1)
        self.assertEqual(tp.cell_modeling_op.forward_batch_sizes, [8])

    def test_entry_queries_profile_counts_surrogate_support_and_fallback_entries(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSelectiveCellModeling()
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.timing_propagation_profile = True
        lut_calls = {"count": 0, "batch_sizes": []}

        def lut_counting(_lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts):
            lut_calls["count"] += 1
            lut_calls["batch_sizes"].append(int(pin_slew.numel()))
            return torch.full_like(pin_slew, 7.0)

        tp.lut_entry_vectorized = lut_counting
        shared_luts = object()
        queries = [
            {
                "lib_cell_idxs": torch.zeros(2, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([4, 5], dtype=torch.long),
                "luts": shared_luts,
                "node_ids": torch.tensor([2, 1], dtype=torch.long),
                "arc_type": 1,
            },
            {
                "lib_cell_idxs": torch.zeros(1, dtype=torch.long),
                "pin_slew": torch.tensor([0.6], dtype=torch.float32),
                "pin_net_caps": torch.tensor([2.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([5], dtype=torch.long),
                "luts": shared_luts,
                "node_ids": torch.tensor([1], dtype=torch.long),
                "arc_type": 1,
            },
        ]
        profile = {}

        results = tp._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
            profile=profile,
        )

        self.assertEqual(lut_calls["count"], 2)
        self.assertEqual(lut_calls["batch_sizes"], [1, 1])
        self.assertEqual(results[0][0].item(), 144.0)
        self.assertEqual(results[0][1].item(), 7.0)
        self.assertEqual(results[1][0].item(), 7.0)
        self.assertEqual(profile["surrogate_candidate_count"], 3)
        self.assertEqual(profile["surrogate_supported_count"], 1)
        self.assertEqual(profile["surrogate_unsupported_count"], 2)
        self.assertEqual(profile["fallback_lut_query_count"], 2)
        self.assertEqual(profile["fallback_lut_entry_count"], 2)
        self.assertEqual(profile["direct_lut_query_count"], 0)
        self.assertEqual(profile["direct_lut_entry_count"], 0)
        self.assertEqual(profile["surrogate_unsupported_unique_key_count"], 1)
        self.assertEqual(
            profile["surrogate_unsupported_top_keys"],
            [
                {
                    "main_id": 0,
                    "arc_offset": 55,
                    "arc_type": 1,
                    "count": 2,
                }
            ],
        )
        for timing_key in (
            "fallback_lut_select_ms",
            "fallback_lut_eval_ms",
            "result_all_valid_check_ms",
            "result_clone_ms",
            "result_fallback_mask_ms",
            "result_fallback_assign_ms",
            "result_split_non_lut_ms",
        ):
            self.assertIn(timing_key, profile)
            self.assertGreaterEqual(profile[timing_key], 0.0)

    def test_entry_queries_profile_splits_surrogate_fallback_reasons(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSelectiveCellModeling()
        tp.pin2node_map = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        tp.inst_main_id = torch.tensor([0, 0, -1, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, False, True, True], dtype=torch.bool)
        tp.size_var_getter = lambda: torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        tp.vt_var_getter = lambda: torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.timing_propagation_profile = True
        tp.lut_entry_vectorized = (
            lambda _lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts: torch.full_like(pin_slew, 7.0)
        )
        queries = [
            {
                "lib_cell_idxs": torch.zeros(4, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([4, 4, 4, 5], dtype=torch.long),
                "luts": object(),
                "node_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "arc_type": 1,
            },
        ]
        profile = {}

        tp._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
            profile=profile,
        )

        self.assertEqual(profile["surrogate_candidate_count"], 4)
        self.assertEqual(profile["surrogate_supported_count"], 1)
        self.assertEqual(profile["surrogate_unsupported_count"], 3)
        self.assertEqual(profile["surrogate_invalid_main_id_count"], 1)
        self.assertEqual(profile["surrogate_non_sizeable_count"], 1)
        self.assertEqual(profile["surrogate_support_missing_count"], 1)
        self.assertEqual(profile["surrogate_unsupported_unique_key_count"], 1)
        self.assertEqual(
            profile["surrogate_unsupported_top_keys"],
            [
                {
                    "main_id": 0,
                    "arc_offset": 55,
                    "arc_type": 1,
                    "count": 1,
                }
            ],
        )

    def test_entry_queries_uses_fixed_init_surrogate_for_non_sizeable_cells(self):
        tp, _, _ = self._build_tp()
        tp.pin2node_map = torch.tensor([0, 1], dtype=torch.int64)
        tp.inst_main_id = torch.tensor([0, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, False], dtype=torch.bool)
        tp.inst_size_init = torch.tensor([1.0, 3.0], dtype=torch.float32)
        tp.inst_vt_init = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        tp.size_var_getter = lambda: torch.tensor([1.0, 99.0], dtype=torch.float32)
        tp.vt_var_getter = lambda: torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.timing_propagation_profile = True
        lut_calls = {"count": 0}

        def lut_counting(_lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts):
            lut_calls["count"] += 1
            return torch.full_like(pin_slew, -1.0)

        tp.lut_entry_vectorized = lut_counting
        queries = [
            {
                "lib_cell_idxs": torch.zeros(2, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([4, 4], dtype=torch.long),
                "luts": object(),
                "node_ids": torch.tensor([0, 1], dtype=torch.long),
                "arc_type": 1,
            },
        ]
        profile = {}

        results = tp._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
            profile=profile,
        )

        self.assertEqual(lut_calls["count"], 0)
        self.assertAlmostEqual(results[0][0].item(), 13.75, places=5)
        self.assertAlmostEqual(results[0][1].item(), 20.4, places=5)
        self.assertEqual(profile["surrogate_candidate_count"], 2)
        self.assertEqual(profile["surrogate_supported_count"], 2)
        self.assertEqual(profile["surrogate_unsupported_count"], 0)
        self.assertEqual(profile["surrogate_non_sizeable_candidate_count"], 1)
        self.assertEqual(profile["surrogate_fixed_non_sizeable_count"], 1)
        self.assertEqual(profile["surrogate_non_sizeable_count"], 0)
        self.assertEqual(profile["fallback_lut_query_count"], 0)
        self.assertEqual(profile["fallback_lut_entry_count"], 0)

    def test_full_support_cache_skips_repeated_support_checks(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSupportCountingCellModeling()
        tp.pin2node_map = torch.tensor([0, 1], dtype=torch.int64)
        tp.inst_main_id = torch.tensor([0, 0], dtype=torch.int64)
        tp.inst_is_sizeable = torch.tensor([True, False], dtype=torch.bool)
        tp.inst_size_init = torch.tensor([1.0, 3.0], dtype=torch.float32)
        tp.inst_vt_init = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        tp.size_var_getter = lambda: torch.tensor([1.0, 99.0], dtype=torch.float32)
        tp.vt_var_getter = lambda: torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.timing_propagation_profile = True
        tp._surrogate_support_cache_status = "all_supported"
        tp.lut_entry_vectorized = (
            lambda _lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts: torch.full_like(pin_slew, -1.0)
        )
        queries = [
            {
                "lib_cell_idxs": torch.zeros(2, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([4, 4], dtype=torch.long),
                "luts": object(),
                "node_ids": torch.tensor([0, 1], dtype=torch.long),
                "arc_type": 1,
            },
        ]
        profile = {}

        results = tp._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
            profile=profile,
        )

        self.assertEqual(tp.cell_modeling_op.support_calls, 0)
        self.assertEqual(profile["surrogate_supported_count"], 2)
        self.assertEqual(profile["surrogate_unsupported_count"], 0)
        self.assertEqual(profile["surrogate_support_skipped_count"], 2)
        self.assertEqual(profile["fallback_lut_entry_count"], 0)
        self.assertAlmostEqual(results[0][1].item(), 20.4, places=5)

    def test_entry_queries_all_valid_batch_split_checks_mask_once(self):
        tp, _, _ = self._build_tp()
        tp.timing_propagation_profile = True
        queries = [
            {
                "lib_cell_idxs": torch.zeros(2, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([1, 2], dtype=torch.long),
                "luts": object(),
                "node_ids": torch.tensor([0, 1], dtype=torch.long),
                "arc_type": 1,
            },
            {
                "lib_cell_idxs": torch.zeros(1, dtype=torch.long),
                "pin_slew": torch.tensor([0.6], dtype=torch.float32),
                "pin_net_caps": torch.tensor([2.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([3], dtype=torch.long),
                "luts": object(),
                "node_ids": torch.tensor([2], dtype=torch.long),
                "arc_type": 0,
            },
        ]

        with mock.patch.object(torch, "all", wraps=torch.all) as all_mock:
            results = tp._entry_queries_with_optional_surrogate(
                queries,
                use_surrogate=True,
                profile={},
            )

        self.assertEqual(all_mock.call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(tuple(results[0].shape), (2,))
        self.assertEqual(tuple(results[1].shape), (1,))

    def test_entry_queries_profile_counts_fallback_lut_table_types(self):
        tp, _, _ = self._build_tp()
        tp.cell_modeling_op = FakeSelectiveCellModeling()
        tp.lib_arc_offsets = torch.tensor([0, 11, 22, 33, 44, 55], dtype=torch.int64)
        tp.timing_propagation_profile = True
        luts = SimpleNamespace(
            is_scalar=torch.tensor([True, False, False, False, False, False]),
            is_trans_1d=torch.tensor([False, True, False, False, False, False]),
            is_cap_1d=torch.tensor([False, False, True, False, False, False]),
            is_2d=torch.tensor([False, False, False, True, False, False]),
        )

        tp.lut_entry_vectorized = (
            lambda _lib_cell_idxs, pin_slew, _pin_net_caps, _lib_arc_idxs, _luts: torch.full_like(pin_slew, 7.0)
        )
        queries = [
            {
                "lib_cell_idxs": torch.zeros(2, dtype=torch.long),
                "pin_slew": torch.tensor([0.2, 0.4], dtype=torch.float32),
                "pin_net_caps": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([4, 0], dtype=torch.long),
                "luts": luts,
                "node_ids": torch.tensor([2, 1], dtype=torch.long),
                "arc_type": 1,
            },
            {
                "lib_cell_idxs": torch.zeros(3, dtype=torch.long),
                "pin_slew": torch.tensor([0.6, 0.7, 0.8], dtype=torch.float32),
                "pin_net_caps": torch.tensor([2.0, 2.5, 3.0], dtype=torch.float32),
                "lib_arc_idxs": torch.tensor([1, 2, 3], dtype=torch.long),
                "luts": luts,
                "node_ids": torch.tensor([1, 1, 1], dtype=torch.long),
                "arc_type": 2,
            },
        ]
        profile = {}

        tp._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
            profile=profile,
        )

        self.assertEqual(profile["fallback_lut_entry_count"], 4)
        self.assertEqual(profile["fallback_lut_scalar_entry_count"], 1)
        self.assertEqual(profile["fallback_lut_trans_1d_entry_count"], 1)
        self.assertEqual(profile["fallback_lut_cap_1d_entry_count"], 1)
        self.assertEqual(profile["fallback_lut_2d_entry_count"], 1)
        self.assertEqual(profile["fallback_lut_f_delay_entry_count"], 0)
        self.assertEqual(profile["fallback_lut_r_delay_entry_count"], 1)
        self.assertEqual(profile["fallback_lut_f_trans_entry_count"], 3)
        self.assertEqual(profile["fallback_lut_r_trans_entry_count"], 0)


class TimingViolationStatsTest(unittest.TestCase):
    def test_cap_violation_uses_shared_prefix_when_graph_suffix_exists(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.pin_net_cap_rise = torch.tensor([1.5, 2.5, 9.0, 99.0, 100.0], dtype=torch.float32)
        tp.pin_net_cap_fall = torch.tensor([1.0, 3.0, 9.0, 88.0, 77.0], dtype=torch.float32)

        total_violation = tp.get_total_cap_violation(
            torch.tensor([True, True, False], dtype=torch.bool),
            torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32),
        )

        self.assertAlmostEqual(total_violation, 1500.0)

    def test_slew_violation_uses_max_of_rise_and_fall_per_pin(self):
        tp = TimingPropagation.__new__(TimingPropagation)
        nn.Module.__init__(tp)
        tp.pin_rtran = torch.tensor([120.0, 140.0, 999.0], dtype=torch.float32)
        tp.pin_ftran = torch.tensor([110.0, 150.0, 999.0], dtype=torch.float32)

        total_violation = tp.get_total_slew_violation(
            torch.tensor([True, True, False], dtype=torch.bool),
            torch.tensor([100.0, 130.0, 1.0], dtype=torch.float32),
        )

        self.assertAlmostEqual(total_violation, 0.04)


if __name__ == "__main__":
    unittest.main()
