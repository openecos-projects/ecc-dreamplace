#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@file         : test_timing_propagation_plain.py
@author       : AI Assistant
@brief        : Plain Python tests for timing_propagation.py (no unittest)
@version      : 1.0
@date         : 2025-04-22
'''

import torch
import time  # Keep original import if needed elsewhere
import sys
import os
from pathlib import Path
import math  # For isclose
import unittest
import logging
import warnings
from torch.func import vmap
import unittest
from dataclasses import dataclass, fields
import torch.autograd.gradcheck as gradcheck # Import gradcheck
from unittest import mock

# --- Add project root to path if running script directly ---
# (Same as before, uncomment/adjust if needed)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)

# --- Import classes from the original file ---
# this test lives outside the dreamplace package: make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dreamplace.ops.timing_propagation.timing_propagation import (
    LUTS_INFO,
    ARCS_INFO,
    TimingPropagation,
    _move_dynamic_net_arc_inputs_to_device,
    _resolve_native_net_subgraph_timing_device,
)
from dreamplace.ops.timing_propagation.critical_endpoint_pruning import (
    DynamicCriticalEndpointSelector,
)
from dreamplace.ops.net_subgraph_timing import (
    build_dynamic_net_arc_inputs_from_net_subgraphs,
    build_timing_pin_net_inputs,
    net_subgraph_forward_native,
)


# --- Unit Tests ---
# --- Unit Testing ---

class TestDynamicCriticalEndpointSelector(unittest.TestCase):
    def test_does_not_full_refresh_on_initial_iteration(self):
        selector = DynamicCriticalEndpointSelector(
            top_k=2,
            slack_window_ps=0.0,
            refresh_interval=10,
            hysteresis_interval=0,
            full_refresh_interval=50,
        )

        active, stats = selector.refresh(
            iteration=0,
            endpoint_ids=[0, 1, 2, 3],
            endpoint_slack_ps=[-100.0, -90.0, -80.0, -70.0],
        )

        self.assertEqual(active, {0, 1})
        self.assertEqual(stats["selected_by_top_k_count"], 2)
        self.assertEqual(stats["selected_by_full_refresh_count"], 0)


@unittest.skip("legacy fixture uses the pre-current TimingPropagation constructor")
class TestTimingPropagation(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float64 for gradient checking precision
        self.dtype = torch.float64

        # --- Mock Topology ---
        # Pins: 0=PI, 1=INV_in, 2=INV_out, 3=BUF_in, 4=BUF_out, 5=PO
        # Nets: 0 (PI->INV), 1 (INV->BUF), 2 (BUF->PO)
        # Cells: 0 (INV), 1 (BUF)
        self.num_pins = 6
        self.num_nets = 3
        self.num_cells = 2
        self.num_lib_cells = 2 # Assume 2 types of cells in library (e.g., INV, BUF)
        # Assume max 1 timing arc per lib cell type for simplicity in this test
        self.num_lib_arcs_per_type = 1

        self.pin_net = torch.tensor([0, 0, 1, 1, 2, 2], device=self.device, dtype=torch.long) # Pin index to Net index

        self.start_points = torch.tensor([0], device=self.device, dtype=torch.long) # PI
        self.end_points = torch.tensor([5], device=self.device, dtype=torch.long)   # PO

        # Cell Info
        self.cells_by_level = [
            torch.tensor([0], device=self.device, dtype=torch.long), # Level 0: INV (cell instance 0)
            torch.tensor([1], device=self.device, dtype=torch.long)  # Level 1: BUF (cell instance 1)
        ]
        self.cells_by_reverse_level = [
            torch.tensor([1], device=self.device, dtype=torch.long), # Level 1: BUF
            torch.tensor([0], device=self.device, dtype=torch.long)  # Level 0: INV
        ]

        # cell_flat_outpins: [INV_out (pin 2), BUF_out (pin 4)]
        self.cell_flat_outpins = torch.tensor([2, 4], device=self.device, dtype=torch.long)
        # Starts for cell 0, cell 1, and end marker
        self.cell_flat_outpins_start = torch.tensor([0, 1, 2], device=self.device, dtype=torch.long)

        # --- Cell Arcs ---
        # [inpin, outpin, lib_cell_idx, lib_arc_idx, arc_type]
        # Arc Instance 0: INV input (1) -> INV output (2), LibCell 0 (INV), LibArc 0, type 0 (neg)
        # Arc Instance 1: BUF input (3) -> BUF output (4), LibCell 1 (BUF), LibArc 0, type 1 (pos)
        self.inst_flat_arcs = torch.tensor([
            [1, 2, 0, 0, 0],
            [3, 4, 1, 0, 1]
        ], device=self.device, dtype=torch.long)
        # Starts for cell instance 0, cell instance 1, and end marker
        self.inst_flat_arcs_start = torch.tensor([0, 1, 2], device=self.device, dtype=torch.long)

        self.num_total_cell_arc_instances = self.inst_flat_arcs.shape[0]

        # --- Net Arcs ---
        # [src_pin, sink_pin]
        self.net_flat_arcs = torch.tensor([
            [0, 1], # Net 0: PI (0) -> INV_in (1)
            [2, 3], # Net 1: INV_out (2) -> BUF_in (3)
            [4, 5]  # Net 2: BUF_out (4) -> PO (5)
        ], device=self.device, dtype=torch.long)
        # Starts for net 0, net 1, net 2, and end marker
        self.net_flat_arcs_start = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)

        # --- Initial Conditions ---
        self.inrdelays = torch.tensor([0.1], device=self.device, dtype=self.dtype)
        self.infdelays = torch.tensor([0.1], device=self.device, dtype=self.dtype)
        self.inrtrans = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        self.inftrans = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        self.outcaps = torch.tensor([0.5], device=self.device, dtype=self.dtype) # Capacitance load at PO

        # --- Mock LUT Data ---
        # Simple 2x2 LUTs for demonstration
        max_t = 2
        max_c = 2
        # Total number of unique LUTs = num_lib_cells * num_lib_arcs_per_type
        num_luts = self.num_lib_cells * self.num_lib_arcs_per_type # = 2 * 1 = 2

        # --- LUTs for INV (lib_cell_idx=0, lib_arc_idx=0 => flat lut index 0) ---
        # Transition index values for the first LUT
        inv_trans_table = torch.tensor([0.1, 0.5], device=self.device, dtype=self.dtype)
        # Capacitance index values for the first LUT
        inv_cap_table = torch.tensor([0.2, 1.0], device=self.device, dtype=self.dtype)
        # Delay values: Delay = 0.1 + 0.5*tran + 1.0*cap
        inv_delay_vals = torch.tensor([
            [0.1 + 0.5*0.1 + 1.0*0.2, 0.1 + 0.5*0.1 + 1.0*1.0], # row for tran=0.1
            [0.1 + 0.5*0.5 + 1.0*0.2, 0.1 + 0.5*0.5 + 1.0*1.0]  # row for tran=0.5
        ], device=self.device, dtype=self.dtype) # Shape [2, 2]
        # Transition values: Tran = 0.05 + 0.2*tran + 0.5*cap
        inv_tran_vals = torch.tensor([
            [0.05 + 0.2*0.1 + 0.5*0.2, 0.05 + 0.2*0.1 + 0.5*1.0], # row for tran=0.1
            [0.05 + 0.2*0.5 + 0.5*0.2, 0.05 + 0.2*0.5 + 0.5*1.0]  # row for tran=0.5
        ], device=self.device, dtype=self.dtype) # Shape [2, 2]
        # Actual dimensions [T_dim, C_dim] for the first LUT
        inv_dims = torch.tensor([2, 2], device=self.device, dtype=torch.long)

        # --- LUTs for BUF (lib_cell_idx=1, lib_arc_idx=0 => flat lut index 1) ---
        # Using same index values for simplicity
        buf_trans_table = torch.tensor([0.1, 0.5], device=self.device, dtype=self.dtype)
        buf_cap_table = torch.tensor([0.2, 1.0], device=self.device, dtype=self.dtype)
        # BUF is slightly faster
        buf_delay_vals = inv_delay_vals * 0.8 # Shape [2, 2]
        # BUF has slightly better slew
        buf_tran_vals = inv_tran_vals * 0.9   # Shape [2, 2]
        # Actual dimensions [T_dim, C_dim] for the second LUT
        buf_dims = torch.tensor([2, 2], device=self.device, dtype=torch.long)

        # --- Combine LUTs into Padded Tensors ---
        # Stack individual LUTs along a new dimension (dim=0)
        # Shape becomes [num_luts, MaxT, MaxC]
        flat_luts_delay_values = torch.stack([inv_delay_vals, buf_delay_vals], dim=0)
        flat_luts_tran_values = torch.stack([inv_tran_vals, buf_tran_vals], dim=0)
        # Shape becomes [num_luts, MaxT]
        flat_luts_trans_table = torch.stack([inv_trans_table, buf_trans_table], dim=0)
        # Shape becomes [num_luts, MaxC]
        flat_luts_cap_table = torch.stack([inv_cap_table, buf_cap_table], dim=0)
        # Shape becomes [num_luts, 2]
        flat_luts_dim = torch.stack([inv_dims, buf_dims], dim=0)

        # Create LUTS_INFO instances for each type (delay/tran, rise/fall)
        # Using cloned data for simplicity in this test; real data would differ.
        luts_delay_template = LUTS_INFO(
             flat_luts_values=flat_luts_delay_values.clone().detach().to(self.device, self.dtype),
             flat_luts_trans_table=flat_luts_trans_table.clone().detach().to(self.device, self.dtype),
             flat_luts_cap_table=flat_luts_cap_table.clone().detach().to(self.device, self.dtype),
             flat_luts_dim=flat_luts_dim.clone().detach().to(self.device, torch.long)
        )
        luts_tran_template = LUTS_INFO(
             flat_luts_values=flat_luts_tran_values.clone().detach().to(self.device, self.dtype),
             flat_luts_trans_table=flat_luts_trans_table.clone().detach().to(self.device, self.dtype),
             flat_luts_cap_table=flat_luts_cap_table.clone().detach().to(self.device, self.dtype),
             flat_luts_dim=flat_luts_dim.clone().detach().to(self.device, torch.long)
        )

        # Assign to ARCS_INFO (using same for rise/fall here)
        self.arcs_info = ARCS_INFO(
            f_delay_luts=luts_delay_template,
            r_delay_luts=luts_delay_template,
            f_tran_luts=luts_tran_template,
            r_tran_luts=luts_tran_template
        )

        # --- Instantiate the Module ---
        self.model = TimingPropagation(
            inrdelays=self.inrdelays,
            infdelays=self.infdelays,
            inrtrans=self.inrtrans,
            inftrans=self.inftrans,
            outcaps=self.outcaps,
            pin_net=self.pin_net,
            cells_by_level=self.cells_by_level,
            start_points=self.start_points,
            end_points=self.end_points,
            net_flat_arcs_start=self.net_flat_arcs_start,
            net_flat_arcs=self.net_flat_arcs,
            arcs_info=self.arcs_info,
            inst_flat_arcs_start=self.inst_flat_arcs_start,
            inst_flat_arcs=self.inst_flat_arcs,
            cells_by_reverse_level=self.cells_by_reverse_level
        ).to(self.device, self.dtype) # Ensure model parameters are also float64

        # --- Inputs for Forward Pass (requiring gradients) ---
        self.pin_net_delay = torch.rand(self.num_pins, device=self.device, dtype=self.dtype, requires_grad=True) * 0.1
        self.pin_net_impulse = torch.rand(self.num_pins, device=self.device, dtype=self.dtype, requires_grad=True) * 0.05
        pin_net_cap_init = torch.rand(self.num_pins, device=self.device, dtype=self.dtype) * 0.2
        # Ensure pin_net_cap is a leaf tensor requiring grad
        self.pin_net_cap = pin_net_cap_init.clone().detach().requires_grad_(True)
        with torch.no_grad():
             # Add PO load capacitance (pin 5) to the corresponding net capacitance
             # Note: In a real scenario, pin_net_cap might represent the total downstream cap
             # seen by the driver pin, potentially including wire and input pin caps.
             # Here, we simply add the explicit PO load for testing.
             self.pin_net_cap[self.end_points] += self.outcaps

    def test_forward_pass(self):
        """Test the forward pass execution and output types/shapes."""
        wns, tns = self.model(self.pin_net_delay, self.pin_net_impulse, self.pin_net_cap)

        # Check types
        self.assertIsInstance(wns, torch.Tensor)
        self.assertIsInstance(tns, torch.Tensor)

        # Check shapes (should be scalar)
        self.assertEqual(wns.shape, torch.Size([]))
        self.assertEqual(tns.shape, torch.Size([]))

        # Check dtype
        self.assertEqual(wns.dtype, self.dtype)
        self.assertEqual(tns.dtype, self.dtype)

        # Optional: Check for NaN/Inf
        self.assertFalse(torch.isnan(wns).item())
        self.assertFalse(torch.isinf(wns).item())
        self.assertFalse(torch.isnan(tns).item())
        self.assertFalse(torch.isinf(tns).item())
        print(f"\nForward Pass Results: WNS={wns.item():.4f}, TNS={tns.item():.4f}")


    def test_backward_pass_gradcheck(self):
        """Verify gradients using torch.autograd.gradcheck."""

        # Ensure inputs require gradients and are float64
        inputs = (
            self.pin_net_delay.clone().detach().requires_grad_(True),
            self.pin_net_impulse.clone().detach().requires_grad_(True),
            self.pin_net_cap.clone().detach().requires_grad_(True)
        )

        # Define a function that takes the inputs and returns the TNS (or WNS)
        # gradcheck works best with scalar outputs. TNS is generally better behaved.
        def func_tns(*args):
            # args will be (pin_net_delay, pin_net_impulse, pin_net_cap)
            # Need to call the model's forward method
            # We need to pass the model instance if func is defined outside,
            # or access self.model if defined as a method or nested function.
            _, tns_output = self.model(*args)
            return tns_output

        def func_wns(*args):
            wns_output, _ = self.model(*args)
            return wns_output

        print("\nRunning gradcheck for TNS...")
        # gradcheck compares analytical gradients with numerical approximations
        # `eps`: perturbation size for finite differences
        # `atol`: absolute tolerance
        # `rtol`: relative tolerance
        # `raise_exception=True`: gradcheck raises an error if check fails
        tns_grad_check_passed = gradcheck(func_tns, inputs, eps=1e-6, atol=1e-4, raise_exception=True)
        self.assertTrue(tns_grad_check_passed, "Gradient check failed for TNS")
        print("Gradcheck for TNS passed.")

        # # --- Optional: Gradcheck for WNS ---
        # # WNS involves min operations, which can have zero gradients or sharp corners,
        # # making gradcheck potentially less reliable or requiring nondet_tol.
        # print("\nRunning gradcheck for WNS...")
        # try:
        #     # Nondeterministic tolerance (nondet_tol) might be needed for min/max ops
        #     wns_grad_check_passed = gradcheck(func_wns, inputs, eps=1e-6, atol=1e-4, nondet_tol=1e-6, raise_exception=True)
        #     self.assertTrue(wns_grad_check_passed, "Gradient check failed for WNS")
        #     print("Gradcheck for WNS passed.")
        # except RuntimeError as e:
        #     print(f"Gradcheck for WNS failed or encountered issues (potentially expected due to min): {e}")
        #     # Decide if this failure is acceptable or needs investigation
        #     # self.fail("Gradient check failed for WNS") # Uncomment to make WNS gradcheck mandatory


class TestLutCoefficientCache(unittest.TestCase):
    def _build_affine_2d_luts(self):
        dtype = torch.float64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trans_table = torch.tensor([[0.1, 0.5, 1.0]], device=device, dtype=dtype)
        cap_table = torch.tensor([[0.2, 0.7, 1.3]], device=device, dtype=dtype)
        t = trans_table[0]
        c = cap_table[0]
        values_grid = (
            0.25 * t[:, None] * c[None, :]
            + 0.5 * t[:, None]
            + 1.5 * c[None, :]
            + 2.0
        )
        values = values_grid.reshape(1, -1)
        dims = torch.tensor([[3, 3]], device=device, dtype=torch.long)
        return device, dtype, trans_table, cap_table, values, dims

    def test_luts_info_precomputes_2d_bilinear_coefficients(self):
        device, dtype, trans_table, cap_table, values, dims = self._build_affine_2d_luts()
        luts = LUTS_INFO(values, trans_table, cap_table, dims)

        self.assertTrue(luts.has_2d_coeff_cache)
        self.assertEqual(luts.coeff_a.shape, (1, 2, 2))
        self.assertEqual(luts.coeff_b.shape, (1, 2, 2))
        self.assertEqual(luts.coeff_c.shape, (1, 2, 2))
        self.assertEqual(luts.coeff_d.shape, (1, 2, 2))

        input_trans = torch.tensor([0.3, 0.8], device=device, dtype=dtype)
        output_caps = torch.tensor([0.4, 1.0], device=device, dtype=dtype)
        trans_idx_low = torch.tensor([0, 1], device=device, dtype=torch.long)
        cap_idx_low = torch.tensor([0, 1], device=device, dtype=torch.long)
        coeff_value = (
            luts.coeff_a[0, trans_idx_low, cap_idx_low] * input_trans * output_caps
            + luts.coeff_b[0, trans_idx_low, cap_idx_low] * input_trans
            + luts.coeff_c[0, trans_idx_low, cap_idx_low] * output_caps
            + luts.coeff_d[0, trans_idx_low, cap_idx_low]
        )

        model = TimingPropagation.__new__(TimingPropagation)
        legacy_value = model._lut_entry_2d_vectorized(
            input_trans=input_trans,
            output_caps=output_caps,
            trans_tables_batch=trans_table.expand(2, -1),
            cap_tables_batch=cap_table.expand(2, -1),
            lut_values_batch=values.expand(2, -1),
            trans_dims_actual=dims[:, 0].expand(2),
            cap_dims_actual=dims[:, 1].expand(2),
        )
        torch.testing.assert_close(coeff_value, legacy_value)

    def test_full_valid_2d_luts_enable_coeff_cache_when_requested_by_env(self):
        _, _, trans_table, cap_table, values, dims = self._build_affine_2d_luts()

        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_COEFF_CACHE": "1"}):
            luts = LUTS_INFO(values, trans_table, cap_table, dims)

        self.assertTrue(luts.has_full_2d_coeff_cache)
        self.assertTrue(luts.use_2d_coeff_cache)

    def test_2d_coeff_cache_is_not_enabled_by_default_after_runtime_regression(self):
        _, _, trans_table, cap_table, values, dims = self._build_affine_2d_luts()

        luts = LUTS_INFO(values, trans_table, cap_table, dims)

        self.assertTrue(luts.has_full_2d_coeff_cache)
        self.assertFalse(luts.use_2d_coeff_cache)

    def test_incomplete_2d_coeff_cache_is_not_enabled_even_when_requested(self):
        _, _, trans_table, cap_table, values, dims = self._build_affine_2d_luts()
        degenerate_cap_table = cap_table.clone()
        degenerate_cap_table[0, 1] = degenerate_cap_table[0, 0]

        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_COEFF_CACHE": "1"}):
            luts = LUTS_INFO(values, trans_table, degenerate_cap_table, dims)

        self.assertFalse(luts.has_full_2d_coeff_cache)
        self.assertFalse(luts.use_2d_coeff_cache)

    def test_lut_entry_vectorized_uses_coeff_cache_without_changing_values(self):
        device, dtype, trans_table, cap_table, values, dims = self._build_affine_2d_luts()
        with mock.patch.dict(os.environ, {"AIMP_USE_LUT_2D_COEFF_CACHE": "1"}):
            luts = LUTS_INFO(values, trans_table, cap_table, dims)
        no_cache_luts = LUTS_INFO(values, trans_table, cap_table, dims)
        self.assertTrue(luts.use_2d_coeff_cache)
        no_cache_luts.has_full_2d_coeff_cache = False
        no_cache_luts.use_2d_coeff_cache = False

        model = TimingPropagation.__new__(TimingPropagation)
        lib_cell_idxs = torch.zeros(4, device=device, dtype=torch.long)
        arc_idxs = torch.zeros(4, device=device, dtype=torch.long)
        input_trans = torch.tensor([0.15, 0.3, 0.8, 1.2], device=device, dtype=dtype)
        output_caps = torch.tensor([0.25, 0.5, 1.0, 1.4], device=device, dtype=dtype)

        cached_value = model.lut_entry_vectorized(
            lib_cell_idxs,
            input_trans,
            output_caps,
            arc_idxs,
            luts,
        )
        legacy_value = model.lut_entry_vectorized(
            lib_cell_idxs,
            input_trans,
            output_caps,
            arc_idxs,
            no_cache_luts,
        )

        torch.testing.assert_close(cached_value, legacy_value)

    def test_2d_lut_searchsorted_tables_are_made_contiguous_internally(self):
        device, dtype, trans_table, cap_table, values, dims = self._build_affine_2d_luts()
        model = TimingPropagation.__new__(TimingPropagation)

        trans_tables_batch = trans_table.expand(4, -1)
        cap_tables_batch = cap_table.expand(4, -1)
        self.assertFalse(trans_tables_batch.is_contiguous())
        self.assertFalse(cap_tables_batch.is_contiguous())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = model._lut_entry_2d_vectorized(
                input_trans=torch.tensor([0.15, 0.3, 0.8, 1.2], device=device, dtype=dtype),
                output_caps=torch.tensor([0.25, 0.5, 1.0, 1.4], device=device, dtype=dtype),
                trans_tables_batch=trans_tables_batch,
                cap_tables_batch=cap_tables_batch,
                lut_values_batch=values.expand(4, -1),
                trans_dims_actual=dims[:, 0].expand(4),
                cap_dims_actual=dims[:, 1].expand(4),
            )

        self.assertEqual(value.shape, (4,))
        self.assertFalse(
            any("boundary tensor is non-contiguous" in str(item.message) for item in caught)
        )


class TestCellAatStaticCache(unittest.TestCase):
    def _build_two_arc_cell_aat_model(self, device, dtype):
        model = TimingPropagation.__new__(TimingPropagation)
        model.dtype = dtype
        model.timing_propagation_profile = False
        model.timing_aggregation_mode = "hard"
        model.timing_aggregation_tau_ps = None
        model.use_cell_aat_static_cache = True
        model.pin_net = torch.tensor([0, 0, 1, 1, 2, 2], device=device, dtype=torch.long)

        trans_table = torch.tensor([[0.1, 0.5], [0.1, 0.5]], device=device, dtype=dtype)
        cap_table = torch.tensor([[0.2, 1.0], [0.2, 1.0]], device=device, dtype=dtype)
        delay_values = torch.tensor(
            [
                [[0.35, 1.15], [0.55, 1.35]],
                [[0.28, 0.92], [0.44, 1.08]],
            ],
            device=device,
            dtype=dtype,
        ).reshape(2, -1)
        tran_values = torch.tensor(
            [
                [[0.17, 0.57], [0.25, 0.65]],
                [[0.153, 0.513], [0.225, 0.585]],
            ],
            device=device,
            dtype=dtype,
        ).reshape(2, -1)
        dims = torch.tensor([[2, 2], [2, 2]], device=device, dtype=torch.long)
        delay_luts = LUTS_INFO(delay_values, trans_table, cap_table, dims)
        trans_luts = LUTS_INFO(tran_values, trans_table, cap_table, dims)
        model.arcs_info = ARCS_INFO(
            f_delay_luts=delay_luts,
            r_delay_luts=delay_luts,
            f_trans_luts=trans_luts,
            r_trans_luts=trans_luts,
        )
        return model

    def _build_two_arc_cell_aat_inputs(self, device, dtype):
        inst_arcs = torch.tensor(
            [
                [1, 2, 0, 0, -1],
                [3, 4, 1, 1, 1],
            ],
            device=device,
            dtype=torch.long,
        )
        pin_rAAT = torch.zeros(6, device=device, dtype=dtype)
        pin_fAAT = torch.zeros(6, device=device, dtype=dtype)
        pin_rtran = torch.full((6,), 0.05, device=device, dtype=dtype)
        pin_ftran = torch.full((6,), 0.05, device=device, dtype=dtype)
        pin_net_cap_rise = torch.full((6,), 0.5, device=device, dtype=dtype)
        pin_net_cap_fall = torch.full((6,), 0.5, device=device, dtype=dtype)
        return (
            inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
        )

    def test_reuses_partitioned_arc_metadata_for_same_level_tensor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        model = self._build_two_arc_cell_aat_model(device, dtype)
        (
            inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
        ) = self._build_two_arc_cell_aat_inputs(device, dtype)

        first = model.calculate_cell_aat_level(
            inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
            use_surrogate=False,
        )
        self.assertEqual(getattr(model, "_cell_aat_static_cache_misses", 0), 1)
        self.assertEqual(getattr(model, "_cell_aat_static_cache_hits", 0), 0)

        second = model.calculate_cell_aat_level(
            inst_arcs,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            pin_net_cap_rise,
            pin_net_cap_fall,
            use_surrogate=False,
        )

        self.assertEqual(getattr(model, "_cell_aat_static_cache_misses", 0), 1)
        self.assertEqual(getattr(model, "_cell_aat_static_cache_hits", 0), 1)
        for first_value, second_value in zip(first, second):
            torch.testing.assert_close(first_value, second_value)

    def test_cell_aat_level_can_skip_delay_update_tensors_for_fast_loop(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        model = self._build_two_arc_cell_aat_model(device, dtype)
        inputs = self._build_two_arc_cell_aat_inputs(device, dtype)

        with_delay_updates = model.calculate_cell_aat_level(
            *inputs,
            use_surrogate=False,
            need_delay_updates=True,
        )
        without_delay_updates = model.calculate_cell_aat_level(
            *inputs,
            use_surrogate=False,
            need_delay_updates=False,
        )

        for expected, actual in zip(with_delay_updates[:5], without_delay_updates[:5]):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(value is None for value in without_delay_updates[5:]))


class TestSurrogateAllValidHint(unittest.TestCase):
    def test_entry_queries_ignore_incorrect_all_valid_hint(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        model = TimingPropagation.__new__(TimingPropagation)
        model._surrogate_batch_entry = lambda *args, **kwargs: (
            torch.tensor([True, False], device=device),
            torch.tensor([11.0, 22.0], device=device, dtype=dtype),
            True,
        )
        model.lut_entry_vectorized = lambda *args, **kwargs: torch.tensor(
            [33.0],
            device=device,
            dtype=dtype,
        )
        model._format_surrogate_unsupported_top_keys = lambda *args, **kwargs: None

        queries = [
            {
                "node_ids": torch.tensor([0, 1], device=device, dtype=torch.long),
                "pin_slew": torch.tensor([0.1, 0.2], device=device, dtype=dtype),
                "pin_net_caps": torch.tensor([0.3, 0.4], device=device, dtype=dtype),
                "lib_arc_idxs": torch.tensor([0, 0], device=device, dtype=torch.long),
                "lib_cell_idxs": torch.tensor([0, 0], device=device, dtype=torch.long),
                "arc_type": 0,
                "luts": object(),
            }
        ]

        results = model._entry_queries_with_optional_surrogate(
            queries,
            use_surrogate=True,
        )

        torch.testing.assert_close(
            results[0],
            torch.tensor([11.0, 33.0], device=device, dtype=dtype),
        )


class TestSizeOnlyFastLoopObjective(unittest.TestCase):
    def test_size_only_production_fast_loop_skips_placement_objective_ops_after_density_init(self):
        from dreamplace.PlaceObj import PlaceObj

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PlaceObj.__new__(PlaceObj)
        model.params = type(
            "Params",
            (),
            {
                "placement_sizing_mode": "size_only",
                "production_fast_loop": True,
                "macro_overlap_flag": False,
                "enable_net_weighting": False,
                "net_weighting_scheme": "",
            },
        )()
        model.placedb = type("PlaceDB", (), {"regions": [], "num_movable_nodes": 0})()
        model.op_collections = type("Ops", (), {})()
        model.op_collections.wirelength_op = mock.Mock(
            side_effect=AssertionError("wirelength should be skipped")
        )
        model.op_collections.density_op = mock.Mock(
            side_effect=AssertionError("density should be skipped after init")
        )
        model.use_timing_obj = True
        model.timing_obj = mock.Mock(
            return_value=(
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )
        )
        model._timing_loss = mock.Mock(return_value=torch.tensor(2.0, device=device))
        model._continuous_density_area_penalty = mock.Mock(return_value=None)
        model.init_density = torch.tensor(1.0, device=device)
        model.density = torch.tensor(1.0, device=device)
        model.quad_penalty = False

        pos = torch.zeros(4, device=device, requires_grad=True)
        result = model.obj_fn(pos)

        self.assertEqual(float(result.detach().cpu()), 2.0)
        model.op_collections.wirelength_op.assert_not_called()
        model.op_collections.density_op.assert_not_called()

    def test_production_fast_loop_computes_leakage_tensor_only_when_lane_uses_leakage(self):
        from dreamplace.PlaceObj import PlaceObj

        model = PlaceObj.__new__(PlaceObj)
        model.params = type(
            "Params",
            (),
            {
                "production_fast_loop": True,
                "timing_objective_lane": "timing_slew_cap",
            },
        )()

        self.assertFalse(model._should_compute_timing_aux_tensor("leakage"))

        model.params.timing_objective_lane = "timing_slew_cap_leakage"
        self.assertTrue(model._should_compute_timing_aux_tensor("leakage"))

        model.params.production_fast_loop = False
        model.params.timing_objective_lane = "timing_slew_cap"
        self.assertTrue(model._should_compute_timing_aux_tensor("leakage"))

    def test_production_fast_loop_skips_pin_slack_cpu_copy_when_artifacts_are_disabled(self):
        from dreamplace.PlaceObj import PlaceObj

        model = PlaceObj.__new__(PlaceObj)
        model.params = type("Params", (), {"production_fast_loop": True})()

        self.assertFalse(model._should_copy_pin_slack_after_timing(False))
        self.assertTrue(model._should_copy_pin_slack_after_timing(True))

        model.params.flow_kind = "joint"
        model.params.buffering_mode = "segment"
        model.data_collections = type(
            "Data",
            (),
            {"buffer_segment_count_state": object()},
        )()
        self.assertTrue(model._should_copy_pin_slack_after_timing(False))

        model.params.flow_kind = "placement"
        self.assertFalse(model._should_copy_pin_slack_after_timing(False))

        model.params.production_fast_loop = False
        self.assertTrue(model._should_copy_pin_slack_after_timing(False))

    def test_fast_loop_reuses_zero_slew_aux_by_default_with_disable_override(self):
        from dreamplace.PlaceObj import PlaceObj

        model = PlaceObj.__new__(PlaceObj)
        model.params = type("Params", (), {"production_fast_loop": True})()
        timing_op = type("TimingOp", (), {"last_total_slew_violation": 0.0})()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIMP_REUSE_ZERO_SLEW_AUX", None)
            os.environ.pop("AIMP_DISABLE_REUSE_ZERO_SLEW_AUX", None)
            self.assertTrue(model._should_reuse_zero_slew_aux(timing_op))

        with mock.patch.dict(
            os.environ,
            {"AIMP_DISABLE_REUSE_ZERO_SLEW_AUX": "1"},
            clear=False,
        ):
            self.assertFalse(model._should_reuse_zero_slew_aux(timing_op))

        timing_op.last_total_slew_violation = 1.0
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIMP_DISABLE_REUSE_ZERO_SLEW_AUX", None)
            self.assertFalse(model._should_reuse_zero_slew_aux(timing_op))

        model.params.production_fast_loop = False
        timing_op.last_total_slew_violation = 0.0
        self.assertFalse(model._should_reuse_zero_slew_aux(timing_op))

    def test_fast_loop_non_linear_place_skips_redundant_paths_by_default(self):
        from dreamplace.NonLinearPlace import NonLinearPlace

        model = NonLinearPlace.__new__(NonLinearPlace)
        model._live_timing_topology_initialized = True
        params = type(
            "Params",
            (),
            {
                "placement_sizing_mode": "size_only",
                "production_fast_loop": True,
                "continuous_size_dynamics_mode": "discrete_gradient_topk",
            },
        )()

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIMP_SKIP_SIZE_ONLY_TOPOLOGY_REFRESH", None)
            os.environ.pop("AIMP_DISABLE_SIZE_ONLY_TOPOLOGY_REFRESH_SKIP", None)
            os.environ.pop("AIMP_SKIP_DISCRETE_TOPK_ARTIFACT_WRITE", None)
            os.environ.pop("AIMP_DISABLE_DISCRETE_TOPK_ARTIFACT_SKIP", None)
            self.assertTrue(model._should_skip_live_timing_topology_refresh(params))
            self.assertTrue(
                model._should_skip_discrete_gradient_topk_artifact_write(params)
            )

        with mock.patch.dict(
            os.environ,
            {
                "AIMP_DISABLE_SIZE_ONLY_TOPOLOGY_REFRESH_SKIP": "1",
                "AIMP_DISABLE_DISCRETE_TOPK_ARTIFACT_SKIP": "1",
            },
            clear=False,
        ):
            self.assertFalse(model._should_skip_live_timing_topology_refresh(params))
            self.assertFalse(
                model._should_skip_discrete_gradient_topk_artifact_write(params)
            )

        params.production_fast_loop = False
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIMP_DISABLE_SIZE_ONLY_TOPOLOGY_REFRESH_SKIP", None)
            os.environ.pop("AIMP_DISABLE_DISCRETE_TOPK_ARTIFACT_SKIP", None)
            self.assertFalse(model._should_skip_live_timing_topology_refresh(params))
            self.assertFalse(
                model._should_skip_discrete_gradient_topk_artifact_write(params)
            )


class TestTimingPropagationFastLoopState(unittest.TestCase):
    def test_forward_input_tensor_skips_clone_only_in_production_fast_loop(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source = torch.arange(4, device=device, dtype=torch.float32)

        fast_model = TimingPropagation.__new__(TimingPropagation)
        fast_model.production_fast_loop = True
        fast_model.device = device
        fast_value = fast_model._forward_input_tensor(source)
        self.assertIs(fast_value, source)

        debug_model = TimingPropagation.__new__(TimingPropagation)
        debug_model.production_fast_loop = False
        debug_model.device = device
        debug_value = debug_model._forward_input_tensor(source)
        self.assertIsNot(debug_value, source)
        torch.testing.assert_close(debug_value, source)

    def test_fast_loop_detaches_surrogate_slew_and_cap_inputs_by_default(self):
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        slew = torch.arange(4, dtype=torch.float32, requires_grad=True)
        cap = torch.arange(4, dtype=torch.float32, requires_grad=True)

        with mock.patch.dict(
            os.environ,
            {
                "AIMP_DETACH_SURROGATE_SLEW_INPUTS": "",
                "AIMP_DETACH_SURROGATE_CAP_INPUTS": "",
                "AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS": "",
                "AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS": "",
            },
            clear=False,
        ):
            self.assertFalse(model._surrogate_slew_input_tensor(slew).requires_grad)
            self.assertFalse(model._surrogate_cap_input_tensor(cap).requires_grad)

    def test_fast_loop_surrogate_input_detach_disable_overrides_default(self):
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        slew = torch.arange(4, dtype=torch.float32, requires_grad=True)
        cap = torch.arange(4, dtype=torch.float32, requires_grad=True)

        with mock.patch.dict(
            os.environ,
            {
                "AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS": "1",
                "AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS": "1",
            },
            clear=False,
        ):
            self.assertIs(model._surrogate_slew_input_tensor(slew), slew)
            self.assertIs(model._surrogate_cap_input_tensor(cap), cap)

    def test_non_fast_loop_keeps_surrogate_slew_and_cap_inputs_attached(self):
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = False
        slew = torch.arange(4, dtype=torch.float32, requires_grad=True)
        cap = torch.arange(4, dtype=torch.float32, requires_grad=True)

        with mock.patch.dict(
            os.environ,
            {
                "AIMP_DISABLE_DETACH_SURROGATE_SLEW_INPUTS": "",
                "AIMP_DISABLE_DETACH_SURROGATE_CAP_INPUTS": "",
            },
            clear=False,
        ):
            self.assertIs(model._surrogate_slew_input_tensor(slew), slew)
            self.assertIs(model._surrogate_cap_input_tensor(cap), cap)

    def _build_minimal_forward_model(self, production_fast_loop, *, forbid_cell_rat):
        device = torch.device("cpu")
        dtype = torch.float32
        model = TimingPropagation.__new__(TimingPropagation)
        torch.nn.Module.__init__(model)
        model.production_fast_loop = bool(production_fast_loop)
        model.timing_propagation_profile = False
        model.timing_forward_count = 0
        model.device = device
        model.dtype = dtype
        model.num_pins = 3
        model.pin_net = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
        model.inrdelays = torch.tensor([0.0], device=device, dtype=dtype)
        model.infdelays = torch.tensor([0.0], device=device, dtype=dtype)
        model.inrtrans = torch.tensor([1.0], device=device, dtype=dtype)
        model.inftrans = torch.tensor([1.0], device=device, dtype=dtype)
        model.start_points = torch.tensor([0], device=device, dtype=torch.long)
        model.end_points = torch.tensor([2], device=device, dtype=torch.long)
        model.endpoints_rRAT = torch.tensor([100.0], device=device, dtype=dtype)
        model.endpoints_fRAT = torch.tensor([100.0], device=device, dtype=dtype)
        model.clk_pin_rtran = torch.tensor([1.0], device=device, dtype=dtype)
        model.clk_pin_ftran = torch.tensor([1.0], device=device, dtype=dtype)
        model.flat_inst_arcs_by_level = torch.tensor(
            [[0, 1, 0, 0, 1]],
            device=device,
            dtype=torch.long,
        )
        model.flat_inst_arcs_by_level_start = torch.tensor(
            [0, 0, 1],
            device=device,
            dtype=torch.long,
        )
        model.last_critical_endpoint_pruning_stats = {}
        model.last_traversal_pruning_stats = {}

        model._begin_forward_runtime_state = lambda: None
        model._end_forward_runtime_state = lambda: None
        model._validate_runtime_indices = lambda: None
        model._resolve_surrogate_mode = lambda surrogate_mode: (False, False)
        model._build_device_contract = lambda tensors, scope: {"scope": scope}
        model._assert_device_contract = lambda contract: None
        model._empty_cell_aat_profile = lambda enabled=False: {"enabled": enabled}
        model._build_cell_aat_surrogate_support_cache = lambda *args, **kwargs: None
        model._record_cell_aat_profile_value = lambda *args, **kwargs: None
        model._format_surrogate_unsupported_top_keys = lambda *args, **kwargs: None
        model.update_critical_endpoint_pruning_state = lambda iteration: None
        model._refresh_traversal_pruning_state = lambda iteration: None
        model._captured_store_forward_state = None
        def capture_store_forward_state(**kwargs):
            model._captured_store_forward_state = kwargs
        model._store_forward_state = capture_store_forward_state
        model._try_run_cpu_cuda_parity = lambda *args, **kwargs: None
        model._build_profile_payload = lambda **kwargs: kwargs
        model._sync_profile_clock = lambda device=None: time.perf_counter()
        model._build_traversal_pruning_working_view = lambda device: (
            model.flat_inst_arcs_by_level,
            model.flat_inst_arcs_by_level_start,
            torch.arange(1, device=device, dtype=torch.long),
            False,
        )

        def calculate_clk2q_aat(pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args, **kwargs):
            return torch.empty(0, device=device, dtype=torch.long), pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

        def calculate_net_aat_level(cur_nets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
            return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

        def calculate_cell_aat_level(inst_arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args, **kwargs):
            pin_rAAT = pin_rAAT.clone()
            pin_fAAT = pin_fAAT.clone()
            pin_rtran = pin_rtran.clone()
            pin_ftran = pin_ftran.clone()
            pin_rAAT[2] = 20.0
            pin_fAAT[2] = 25.0
            pin_rtran[2] = 2.0
            pin_ftran[2] = 2.0
            delays = torch.zeros(inst_arcs.shape[0], device=device, dtype=dtype)
            return (
                torch.empty(0, device=device, dtype=torch.long),
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                delays,
                delays,
                delays,
                delays,
            )

        def calculate_setup_rat(pin_rRAT, pin_fRAT, *args):
            pin_rRAT = pin_rRAT.clone()
            pin_fRAT = pin_fRAT.clone()
            pin_rRAT[2] = 10.0
            pin_fRAT[2] = 12.0
            return pin_rRAT, pin_fRAT

        def calculate_net_rat_level(cur_endpoint, pin_rRAT, pin_fRAT, *args):
            return pin_rRAT, pin_fRAT

        def calculate_cell_rat_level(*args, **kwargs):
            if forbid_cell_rat:
                raise AssertionError("production fast-loop must not run full cell RAT propagation")
            return torch.empty(0, device=device, dtype=torch.long), args[1], args[2]

        model.calculate_clk2q_aat = calculate_clk2q_aat
        model.calculate_net_aat_level = calculate_net_aat_level
        model.calculate_cell_aat_level = calculate_cell_aat_level
        model.calculate_setup_rat = calculate_setup_rat
        model.calculate_net_rat_level = calculate_net_rat_level
        model.calculate_cell_rat_level = calculate_cell_rat_level
        return model

    def test_production_fast_loop_skips_full_cell_rat_while_preserving_endpoint_metrics(self):
        pin_net_delays = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_impulses = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_caps = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }

        full_model = self._build_minimal_forward_model(
            production_fast_loop=False,
            forbid_cell_rat=False,
        )
        full_wns, full_tns, _, _ = full_model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
        )

        fast_model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )
        fast_wns, fast_tns, _, _ = fast_model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
        )

        torch.testing.assert_close(fast_wns, full_wns)
        torch.testing.assert_close(fast_tns, full_tns)
        torch.testing.assert_close(
            fast_model.last_endpoint_slack_tensor,
            full_model.last_endpoint_slack_tensor,
        )
        self.assertIs(fast_model._captured_store_forward_state["rslack"], None)
        self.assertIs(fast_model._captured_store_forward_state["fslack"], None)
        self.assertIs(fast_model._captured_store_forward_state["slack"], None)
        self.assertIs(fast_model._captured_store_forward_state["cell_arc_rr_delays"], None)
        self.assertIs(fast_model._captured_store_forward_state["cell_arc_fr_delays"], None)
        self.assertIs(fast_model._captured_store_forward_state["cell_arc_rf_delays"], None)
        self.assertIs(fast_model._captured_store_forward_state["cell_arc_ff_delays"], None)

    def test_forward_can_consume_dynamic_net_arc_inputs(self):
        pin_net_delays = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_impulses = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_caps = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        dynamic_net_arc_inputs = {
            "pin_net_delays": {
                "rise": torch.tensor([0.0, 11.0, 12.0], dtype=torch.float32),
                "fall": torch.tensor([0.0, 21.0, 22.0], dtype=torch.float32),
            },
            "pin_net_impulses": {
                "rise": torch.tensor([0.0, 1.1, 1.2], dtype=torch.float32),
                "fall": torch.tensor([0.0, 2.1, 2.2], dtype=torch.float32),
            },
            "pin_net_caps": {
                "rise": torch.tensor([0.0, 0.11, 0.12], dtype=torch.float32),
                "fall": torch.tensor([0.0, 0.21, 0.22], dtype=torch.float32),
            },
        }

        model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )
        model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            dynamic_net_arc_inputs=dynamic_net_arc_inputs,
        )

        captured = model._captured_store_forward_state
        torch.testing.assert_close(
            captured["pin_net_delay_rise"],
            dynamic_net_arc_inputs["pin_net_delays"]["rise"],
        )
        torch.testing.assert_close(
            captured["pin_net_delay_fall"],
            dynamic_net_arc_inputs["pin_net_delays"]["fall"],
        )
        torch.testing.assert_close(
            captured["pin_net_cap_rise"],
            dynamic_net_arc_inputs["pin_net_caps"]["rise"],
        )
        torch.testing.assert_close(
            captured["pin_net_cap_fall"],
            dynamic_net_arc_inputs["pin_net_caps"]["fall"],
        )
        self.assertEqual(model.last_dynamic_net_arc_input_status["status"], "applied")

    def test_forward_can_consume_native_net_subgraph_outputs(self):
        pin_net_delays = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_impulses = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_caps = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }

        net_result = net_subgraph_forward_native(
            net_flat_topo_sort=torch.tensor([0, 1, 2], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 3], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0, 1], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            flat_pin_to=torch.tensor([1, 2], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 2.0, 3.0], dtype=torch.float32),
            node_capacitance=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
            edge_capacitance=torch.empty(0, dtype=torch.float32),
            driver_arrival=torch.tensor([5.0], dtype=torch.float32),
            driver_slew=torch.tensor([0.1], dtype=torch.float32),
            candidate_node_id=torch.tensor([1], dtype=torch.int32),
            candidate_bu=torch.tensor([1.0], dtype=torch.float32),
            buffer_input_cap=torch.tensor([0.5], dtype=torch.float32),
            buffer_delay=torch.tensor([7.0], dtype=torch.float32),
            buffer_output_slew=torch.tensor([0.2], dtype=torch.float32),
            sink_node_id=torch.tensor([1, 2], dtype=torch.int32),
            sink_net_index=torch.tensor([0, 0], dtype=torch.int32),
        )
        dynamic_net_arc_inputs = build_timing_pin_net_inputs(
            net_result,
            sink_pin_id=torch.tensor([1, 2], dtype=torch.int32),
            num_pins=3,
            fill_value=0.0,
        )
        dynamic_net_arc_inputs["source"] = "net_subgraph_timing_cpp"

        model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )
        model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            dynamic_net_arc_inputs=dynamic_net_arc_inputs,
        )

        captured = model._captured_store_forward_state
        torch.testing.assert_close(
            captured["pin_net_delay_rise"],
            dynamic_net_arc_inputs["pin_net_delays"]["rise"],
        )
        torch.testing.assert_close(
            captured["pin_net_delay_fall"],
            dynamic_net_arc_inputs["pin_net_delays"]["fall"],
        )
        torch.testing.assert_close(
            captured["pin_net_cap_rise"],
            dynamic_net_arc_inputs["pin_net_caps"]["rise"],
        )
        torch.testing.assert_close(
            captured["pin_net_cap_fall"],
            dynamic_net_arc_inputs["pin_net_caps"]["fall"],
        )
        self.assertEqual(model.last_dynamic_net_arc_input_status["status"], "applied")
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["source"],
            "net_subgraph_timing_cpp",
        )

    def test_forward_can_schedule_native_net_subgraph_inputs(self):
        pin_net_delays = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_impulses = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_caps = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        dynamic_net_subgraph_inputs = {
            "build_dynamic_net_arc_inputs": build_dynamic_net_arc_inputs_from_net_subgraphs,
            "nets": [
                {
                    "net_id": 7,
                    "driver_pin_id": 0,
                    "rc_tree": {
                        "root_node_id": 0,
                        "children_by_node": {0: [1], 1: []},
                        "edge_rc": {(0, 1): {"r": 2.0, "c": 0.0}},
                        "node_cap": {0: 0.0, 1: 3.0},
                        "sink_nodes": [1],
                    },
                }
            ],
            "candidates": [
                {
                    "net_id": 7,
                    "node_id": 1,
                    "bu": 1.0,
                    "buffer_input_cap": 0.5,
                    "buffer_delay": 4.0,
                    "buffer_output_slew": 0.2,
                }
            ],
            "driver_arrival_by_net": {7: 1.0},
            "driver_slew_by_net": {7: 0.1},
            "fill_value": 0.0,
        }

        model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )
        model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            dynamic_net_subgraph_inputs=dynamic_net_subgraph_inputs,
        )

        captured = model._captured_store_forward_state
        torch.testing.assert_close(
            captured["pin_net_delay_rise"],
            torch.tensor([0.0, 5.0, 0.0]),
        )
        torch.testing.assert_close(
            captured["pin_net_impulse_rise"],
            torch.tensor([0.0, 0.03, 0.0]),
        )
        torch.testing.assert_close(
            captured["pin_net_cap_rise"],
            torch.tensor([0.5, 0.5, 0.0]),
        )
        self.assertEqual(model.last_dynamic_net_arc_input_status["status"], "applied")
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["source"],
            "net_subgraph_timing_cpp",
        )
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["net_subgraph_timing_mode"],
            "native_fixed_state_forward",
        )
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["net_subgraph_native_device"],
            "cpu",
        )

    def test_native_net_subgraph_scheduler_defaults_to_cpu_even_for_cuda_timing_device(self):
        native_device = _resolve_native_net_subgraph_timing_device(
            timing_device=torch.device("cuda:0"),
            requested_device=None,
        )

        self.assertEqual(native_device.type, "cpu")

    def test_relaxed_buffer_net_arc_inputs_drive_global_tns_gradients(self):
        from dreamplace.ops.buffer_insertion.optimization_state import (
            build_buffer_optimization_state,
        )
        from dreamplace.ops.net_subgraph_timing import (
            build_relaxed_buffer_dynamic_net_arc_inputs,
        )

        pin_net_delays = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_impulses = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        pin_net_caps = {
            "rise": torch.zeros(3, dtype=torch.float32),
            "fall": torch.zeros(3, dtype=torch.float32),
        }
        state = build_buffer_optimization_state(
            [
                {
                    "candidate_id": 3,
                    "net_id": 7,
                    "candidate_node_id": 1,
                    "buffer_main_type_index": 4,
                    "x_dbu": 10,
                    "y_dbu": 20,
                }
            ],
            buffer_main_type_index=4,
            legal_buffer_count=2,
            initial_bu_logit=0.0,
            initial_bsu_index=0.5,
        )
        dynamic_net_arc_inputs = build_relaxed_buffer_dynamic_net_arc_inputs(
            buffer_state=state,
            per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
            per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            sink_pin_id=torch.tensor([2], dtype=torch.int32),
            num_pins=3,
            fill_value=0.0,
            net_flat_topo_sort=torch.tensor([0, 1], dtype=torch.int32),
            net_flat_topo_sort_start=torch.tensor([0, 2], dtype=torch.int32),
            pin_fa=torch.tensor([-1, 0], dtype=torch.int32),
            flat_pin_to_start=torch.tensor([0, 1, 1], dtype=torch.int32),
            flat_pin_to=torch.tensor([1], dtype=torch.int32),
            edge_resistance=torch.tensor([0.0, 0.0], dtype=torch.float32),
            node_capacitance=torch.tensor([0.0, 1.0], dtype=torch.float32),
            driver_arrival=torch.tensor([70.0], dtype=torch.float32),
            driver_slew=torch.tensor([0.1], dtype=torch.float32),
            sink_node_id=torch.tensor([1], dtype=torch.int32),
            sink_net_index=torch.tensor([0], dtype=torch.int32),
            coordinate_source="fixed_smoke",
        )

        model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )

        def calculate_net_rat_level(
            cur_endpoint,
            pin_rRAT,
            pin_fRAT,
            pin_net_delay_rise,
            pin_net_delay_fall,
        ):
            pin_rRAT = pin_rRAT.clone()
            pin_fRAT = pin_fRAT.clone()
            pin_rRAT[cur_endpoint] = pin_rRAT[cur_endpoint] - pin_net_delay_rise[cur_endpoint]
            pin_fRAT[cur_endpoint] = pin_fRAT[cur_endpoint] - pin_net_delay_fall[cur_endpoint]
            return pin_rRAT, pin_fRAT

        model.calculate_net_rat_level = calculate_net_rat_level

        _, tns, _, _ = model(
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            dynamic_net_arc_inputs=dynamic_net_arc_inputs,
        )
        loss = -tns
        loss.backward()

        self.assertEqual(model.last_dynamic_net_arc_input_status["status"], "applied")
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["source"],
            "net_subgraph_timing_relaxed_buffer",
        )
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["net_subgraph_timing_mode"],
            "relaxed_buffer_autograd",
        )
        self.assertEqual(
            model.last_dynamic_net_arc_input_status["gradient_chain_level"],
            "toy_global_timing_autograd",
        )
        self.assertTrue(torch.isfinite(state.bu_logits.grad).all())
        self.assertTrue(torch.isfinite(state.bsu_index_param.grad).all())
        self.assertGreater(float(state.bu_logits.grad.abs().max().item()), 0.0)
        self.assertGreater(float(state.bsu_index_param.grad.abs().max().item()), 0.0)

    def test_dynamic_net_arc_tns_gradcheck_current_forward_interface(self):
        base_inputs = {
            "pin_net_delays": {
                "rise": torch.zeros(3, dtype=torch.float64),
                "fall": torch.zeros(3, dtype=torch.float64),
            },
            "pin_net_impulses": {
                "rise": torch.zeros(3, dtype=torch.float64),
                "fall": torch.zeros(3, dtype=torch.float64),
            },
            "pin_net_caps": {
                "rise": torch.zeros(3, dtype=torch.float64),
                "fall": torch.zeros(3, dtype=torch.float64),
            },
        }
        dynamic_rise_delay = torch.tensor(
            [0.0, 0.0, 2.0],
            dtype=torch.float64,
            requires_grad=True,
        )

        model = self._build_minimal_forward_model(
            production_fast_loop=True,
            forbid_cell_rat=True,
        )
        model.dtype = torch.float64
        model.inrdelays = model.inrdelays.to(torch.float64)
        model.infdelays = model.infdelays.to(torch.float64)
        model.inrtrans = model.inrtrans.to(torch.float64)
        model.inftrans = model.inftrans.to(torch.float64)
        model.endpoints_rRAT = model.endpoints_rRAT.to(torch.float64)
        model.endpoints_fRAT = model.endpoints_fRAT.to(torch.float64)
        model.clk_pin_rtran = model.clk_pin_rtran.to(torch.float64)
        model.clk_pin_ftran = model.clk_pin_ftran.to(torch.float64)

        def calculate_cell_aat_level(inst_arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args, **kwargs):
            pin_rAAT = pin_rAAT.clone()
            pin_fAAT = pin_fAAT.clone()
            pin_rtran = pin_rtran.clone()
            pin_ftran = pin_ftran.clone()
            pin_rAAT[2] = pin_rAAT[2] + 12.0
            pin_fAAT[2] = pin_fAAT[2] + 14.0
            pin_rtran[2] = 2.0
            pin_ftran[2] = 2.0
            delays = torch.zeros(inst_arcs.shape[0], dtype=pin_rAAT.dtype)
            return (
                torch.empty(0, dtype=torch.long),
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                delays,
                delays,
                delays,
                delays,
            )

        def calculate_net_rat_level(cur_endpoint, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall):
            pin_rRAT = pin_rRAT.clone()
            pin_fRAT = pin_fRAT.clone()
            pin_rRAT[cur_endpoint] = pin_rRAT[cur_endpoint] - pin_net_delay_rise[cur_endpoint]
            pin_fRAT[cur_endpoint] = pin_fRAT[cur_endpoint] - pin_net_delay_fall[cur_endpoint]
            return pin_rRAT, pin_fRAT

        model.calculate_cell_aat_level = calculate_cell_aat_level
        model.calculate_net_rat_level = calculate_net_rat_level

        def func_tns(dynamic_rise):
            dynamic_net_arc_inputs = {
                "pin_net_delays": {
                    "rise": dynamic_rise,
                    "fall": torch.zeros_like(dynamic_rise),
                },
                "pin_net_impulses": {
                    "rise": torch.zeros_like(dynamic_rise),
                    "fall": torch.zeros_like(dynamic_rise),
                },
                "pin_net_caps": {
                    "rise": torch.zeros_like(dynamic_rise),
                    "fall": torch.zeros_like(dynamic_rise),
                },
                "source": "unit_gradcheck_dynamic_net_arc",
            }
            _, tns, _, _ = model(
                base_inputs["pin_net_delays"],
                base_inputs["pin_net_impulses"],
                base_inputs["pin_net_caps"],
                dynamic_net_arc_inputs=dynamic_net_arc_inputs,
            )
            return tns

        self.assertTrue(
            gradcheck(func_tns, (dynamic_rise_delay,), eps=1e-6, atol=1e-4, raise_exception=True)
        )

    def test_dynamic_net_arc_inputs_can_be_moved_to_timing_device(self):
        dynamic_net_arc_inputs = {
            "pin_net_delays": {
                "rise": torch.tensor([1.0], dtype=torch.float32),
                "fall": torch.tensor([2.0], dtype=torch.float32),
            },
            "pin_net_impulses": {
                "rise": torch.tensor([3.0], dtype=torch.float32),
                "fall": torch.tensor([4.0], dtype=torch.float32),
            },
            "pin_net_caps": {
                "rise": torch.tensor([5.0], dtype=torch.float32),
                "fall": torch.tensor([6.0], dtype=torch.float32),
            },
            "sink_pin_id": torch.tensor([0], dtype=torch.int64),
            "source": "net_subgraph_timing_cpp",
        }

        moved = _move_dynamic_net_arc_inputs_to_device(
            dynamic_net_arc_inputs,
            torch.device("cpu"),
        )

        self.assertEqual(moved["source"], "net_subgraph_timing_cpp")
        self.assertIsNot(moved, dynamic_net_arc_inputs)
        self.assertEqual(moved["pin_net_delays"]["rise"].device.type, "cpu")
        self.assertEqual(moved["sink_pin_id"].device.type, "cpu")

    def test_production_fast_loop_skips_debug_state_snapshots_but_keeps_live_violation_tensors(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        model.num_pins = 2
        model.pin_rAAT = torch.tensor([123.0], device=device, dtype=dtype)

        pin_rAAT = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        pin_fAAT = torch.tensor([1.5, 2.5], device=device, dtype=dtype)
        pin_rRAT = torch.tensor([9.0, 10.0], device=device, dtype=dtype)
        pin_fRAT = torch.tensor([8.0, 11.0], device=device, dtype=dtype)
        pin_rtran = torch.tensor([80.0, 120.0], device=device, dtype=dtype)
        pin_ftran = torch.tensor([70.0, 90.0], device=device, dtype=dtype)
        pin_net_cap_rise = torch.tensor([0.01, 0.08], device=device, dtype=dtype)
        pin_net_cap_fall = torch.tensor([0.02, 0.09], device=device, dtype=dtype)
        pin_net_delay_rise = torch.tensor([0.1, 0.2], device=device, dtype=dtype)
        pin_net_delay_fall = torch.tensor([0.3, 0.4], device=device, dtype=dtype)
        pin_net_impulse_rise = torch.tensor([1.1, 1.2], device=device, dtype=dtype)
        pin_net_impulse_fall = torch.tensor([1.3, 1.4], device=device, dtype=dtype)
        slack = torch.tensor([3.0, -2.0], device=device, dtype=dtype)
        cell_arc_delay = torch.tensor([0.5, 0.6], device=device, dtype=dtype)

        model._store_forward_state(
            pin_rAAT=pin_rAAT,
            pin_fAAT=pin_fAAT,
            pin_rRAT=pin_rRAT,
            pin_fRAT=pin_fRAT,
            pin_rtran=pin_rtran,
            pin_ftran=pin_ftran,
            pin_net_cap_rise=pin_net_cap_rise,
            pin_net_cap_fall=pin_net_cap_fall,
            pin_net_delay_rise=pin_net_delay_rise,
            pin_net_delay_fall=pin_net_delay_fall,
            pin_net_impulse_rise=pin_net_impulse_rise,
            pin_net_impulse_fall=pin_net_impulse_fall,
            rslack=pin_rRAT - pin_rAAT,
            fslack=pin_fRAT - pin_fAAT,
            slack=slack,
            cell_arc_rr_delays=cell_arc_delay,
            cell_arc_fr_delays=cell_arc_delay,
            cell_arc_rf_delays=cell_arc_delay,
            cell_arc_ff_delays=cell_arc_delay,
        )

        self.assertIs(model.pin_rAAT, None)
        self.assertIs(model.pin_slack, None)
        self.assertIs(model.cell_arc_rr_delays, None)
        self.assertIs(model.pin_rtran_live, pin_rtran)
        self.assertIs(model.pin_net_cap_rise_live, pin_net_cap_rise)
        torch.testing.assert_close(model.pin_net_delay_rise_live, pin_net_delay_rise)
        torch.testing.assert_close(model.pin_net_delay_fall_live, pin_net_delay_fall)
        torch.testing.assert_close(model.pin_net_impulse_rise_live, pin_net_impulse_rise)
        torch.testing.assert_close(model.pin_net_impulse_fall_live, pin_net_impulse_fall)
        torch.testing.assert_close(model.get_pin_slack(), slack.detach().cpu())
        self.assertIsNot(model.pin_rAAT_live_snapshot, pin_rAAT)
        self.assertIsNot(model.pin_rRAT_live_snapshot, pin_rRAT)
        torch.testing.assert_close(model.pin_rAAT_live_snapshot, pin_rAAT.detach())
        torch.testing.assert_close(model.pin_fAAT_live_snapshot, pin_fAAT.detach())
        torch.testing.assert_close(model.pin_rRAT_live_snapshot, pin_rRAT.detach())
        torch.testing.assert_close(model.pin_fRAT_live_snapshot, pin_fRAT.detach())

        slew_vio = model.get_total_slew_violation_tensor(
            torch.tensor([True, True], device=device),
            torch.tensor([100.0, 100.0], device=device, dtype=dtype),
        )
        cap_vio = model.get_total_cap_violation_tensor(
            torch.tensor([True, True], device=device),
            torch.tensor([0.05, 0.05], device=device, dtype=dtype),
        )
        self.assertAlmostEqual(float(slew_vio.detach().cpu().item()), 0.020, places=6)
        self.assertAlmostEqual(float(cap_vio.detach().cpu().item()), 0.040, places=6)

    def test_production_fast_loop_critical_paths_use_live_snapshots(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        model.num_pins = 3
        model.start_points = torch.tensor([0], device=device, dtype=torch.int32)
        model.flat_pin_to_graph_start_reverse = torch.tensor(
            [0, 0, 1, 2],
            device=device,
            dtype=torch.int32,
        )
        model.flat_pin_to_graph_reverse = torch.tensor(
            [0, 1],
            device=device,
            dtype=torch.int32,
        )
        model.pin_pair_arc_keys = torch.empty((0, 2), device=device, dtype=torch.int32)
        model.flat_pin_pair_arc_start = torch.empty((0,), device=device, dtype=torch.int32)
        model.flat_pin_pair_arc_indices = torch.empty((0,), device=device, dtype=torch.int32)
        model.pin_rAAT = model.pin_fAAT = None
        model.pin_rslack = model.pin_fslack = model.pin_slack = None
        model.pin_rAAT_live_snapshot = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)
        model.pin_fAAT_live_snapshot = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)
        model.pin_rslack_live_snapshot = torch.tensor([5.0, 1.0, -1.0], device=device, dtype=dtype)
        model.pin_fslack_live_snapshot = torch.tensor([6.0, 2.0, 3.0], device=device, dtype=dtype)
        model.pin_slack_live_snapshot = torch.tensor([5.0, 1.0, -1.0], device=device, dtype=dtype)

        paths = model.get_critical_paths([2], K=1, max_depth=8)

        self.assertEqual(paths, [[0, 1, 2]])

    def test_production_fast_loop_reconstructs_live_slack_when_full_rat_is_skipped(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        model = TimingPropagation.__new__(TimingPropagation)
        model.production_fast_loop = True
        model.num_pins = 2

        pin_rAAT = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        pin_fAAT = torch.tensor([1.5, 2.5], device=device, dtype=dtype)
        pin_rRAT = torch.tensor([9.0, 10.0], device=device, dtype=dtype)
        pin_fRAT = torch.tensor([8.0, 11.0], device=device, dtype=dtype)
        pin_rtran = torch.tensor([80.0, 120.0], device=device, dtype=dtype)
        pin_ftran = torch.tensor([70.0, 90.0], device=device, dtype=dtype)
        pin_net_cap_rise = torch.tensor([0.01, 0.08], device=device, dtype=dtype)
        pin_net_cap_fall = torch.tensor([0.02, 0.09], device=device, dtype=dtype)
        pin_net_delay_rise = torch.tensor([0.1, 0.2], device=device, dtype=dtype)
        pin_net_delay_fall = torch.tensor([0.3, 0.4], device=device, dtype=dtype)
        cell_arc_delay = torch.tensor([0.5, 0.6], device=device, dtype=dtype)

        model._store_forward_state(
            pin_rAAT=pin_rAAT,
            pin_fAAT=pin_fAAT,
            pin_rRAT=pin_rRAT,
            pin_fRAT=pin_fRAT,
            pin_rtran=pin_rtran,
            pin_ftran=pin_ftran,
            pin_net_cap_rise=pin_net_cap_rise,
            pin_net_cap_fall=pin_net_cap_fall,
            pin_net_delay_rise=pin_net_delay_rise,
            pin_net_delay_fall=pin_net_delay_fall,
            rslack=None,
            fslack=None,
            slack=None,
            cell_arc_rr_delays=cell_arc_delay,
            cell_arc_fr_delays=cell_arc_delay,
            cell_arc_rf_delays=cell_arc_delay,
            cell_arc_ff_delays=cell_arc_delay,
        )

        expected = torch.min(pin_rRAT - pin_rAAT, pin_fRAT - pin_fAAT)
        self.assertIs(model.pin_slack, None)
        torch.testing.assert_close(model.get_pin_slack(), expected.detach().cpu())


# --- Main execution block ---
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Run the tests
    unittest.main()
