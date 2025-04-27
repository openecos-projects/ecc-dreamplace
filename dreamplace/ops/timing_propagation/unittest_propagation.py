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
import math  # For isclose
import unittest
import logging
from torch.func import vmap
import unittest
from dataclasses import dataclass, fields
import torch.autograd.gradcheck as gradcheck # Import gradcheck

# --- Add project root to path if running script directly ---
# (Same as before, uncomment/adjust if needed)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)

# --- Import classes from the original file ---
try:
    from timing_propagation import LUTS_INFO, ARCS_INFO, TimingPropagation
except ImportError:
    print("Error: Could not import from timing_propagation.py.")
    print("Ensure the file is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)


# --- Unit Tests ---
# --- Unit Testing ---

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
        self.cell_flat_arcs = torch.tensor([
            [1, 2, 0, 0, 0],
            [3, 4, 1, 0, 1]
        ], device=self.device, dtype=torch.long)
        # Starts for cell instance 0, cell instance 1, and end marker
        self.cell_flat_arcs_start = torch.tensor([0, 1, 2], device=self.device, dtype=torch.long)
        self.num_total_cell_arc_instances = self.cell_flat_arcs.shape[0]

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
            cell_flat_arcs_start=self.cell_flat_arcs_start,
            cell_flat_arcs=self.cell_flat_arcs,
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


# --- Main execution block ---
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Run the tests
    unittest.main()