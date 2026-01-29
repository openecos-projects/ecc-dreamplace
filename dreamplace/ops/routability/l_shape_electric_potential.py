##
# @file   l_shape_electric_potential.py
# @brief  Compute electric potential for L-shape routing segments
#         Uses C++/CUDA backends for density map and force computation
#         Gradients flow back through segment positions to cell positions
#

from doctest import FAIL_FAST
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
import logging
import os

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform
import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

from .l_shape_electric_overflow import SegmentDensityMapFunction
from .plot_map import plot_density_map

logger = logging.getLogger(__name__)
_PLOT_ITER = 0


class SegmentElectricPotentialFunction(Function):
    """
    Compute electric potential for routing segments.
    
    Forward: Compute density map and potential energy using DCT.
    Backward: Compute gradients using electric field from Poisson equation.
    """
    
    @staticmethod
    def forward(
        ctx,
        segment_pos,
        segment_size_x,
        segment_size_y,
        segment_size_x_clamped,
        segment_size_y_clamped,
        offset_x,
        offset_y,
        ratio,
        bin_center_x,
        bin_center_y,
        initial_density_map,
        target_density,
        target_demand,
        area_per_track,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_segments,
        padding,
        padding_mask,
        num_bins_x, num_bins_y,
        num_impacted_bins_x,
        num_impacted_bins_y,
        deterministic_flag,
        sorted_segment_map,
        exact_expkM,
        exact_expkN,
        inv_wu2_plus_wv2,
        wu_by_wu2_plus_wv2_half,
        wv_by_wu2_plus_wv2_half,
        dct2,
        idct2,
        idct_idxst,
        idxst_idct,
        fast_mode
    ):
        """
        Compute electric potential energy for segments.
        """
        tt = time.time()
        
        # Compute density map using C++/CUDA backend
        density_map = SegmentDensityMapFunction.forward(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            bin_center_x,
            bin_center_y,
            initial_density_map,
            target_density,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_segments,
            padding,
            padding_mask,
            num_bins_x, num_bins_y,
            num_impacted_bins_x,
            num_impacted_bins_y,
            deterministic_flag,
            sorted_segment_map
        )

        # plot density map
        # plot_density_map(density_map, 
        #     title="Segment Density Map", 
        #     save_path=f"/home/sxr/workspace/test-benchmark/benchmark/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/segment_density_map.png")
        
        # exit(0)
        
        # Save for backward
        ctx.segment_pos = segment_pos
        ctx.segment_size_x_clamped = segment_size_x_clamped
        ctx.segment_size_y_clamped = segment_size_y_clamped
        ctx.offset_x = offset_x
        ctx.offset_y = offset_y
        ctx.ratio = ratio
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.target_density = target_density
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_segments = num_segments
        ctx.padding = padding
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_impacted_bins_x = num_impacted_bins_x
        ctx.num_impacted_bins_y = num_impacted_bins_y
        ctx.deterministic_flag = deterministic_flag
        ctx.sorted_segment_map = sorted_segment_map
        
        # DCT-based Poisson solver
        M = num_bins_x
        N = num_bins_y
        
        # Compute overflow map based on supply-demand relationship
        # target_density is the supply map (2D tensor from EGR, or scalar for uniform)
        bin_area = bin_size_x * bin_size_y
        
        if isinstance(target_density, torch.Tensor) and target_density.dim() == 2:
            # EGR supply map: target_density contains track counts per bin
            # 
            # Problem: density_map (segment area) and supply_map (track count) have 
            # different physical units. We need a calibration factor to convert
            # segment area to equivalent track demand.
            #
            # Calibration factor k (area_per_track):
            #   demand_in_tracks = density / k
            #   overflow = demand_in_tracks - supply
            #
            # How to determine k:
            # Method 1 (current): k = total_density / (target_utilization * total_supply)
            #   - target_utilization controls how aggressively we penalize congestion
            #   - target_utilization = 1.0: demand matches supply on average (no slack)
            #   - target_utilization = 0.8: 80% utilization, 20% slack (more conservative)
            #   - target_utilization < 1.0: easier to trigger overflow (more spreading)
            #
            # Method 2 (physical): k = wire_width * track_pitch
            #   - Requires technology parameters
            #
            # We use Method 1 with configurable target_utilization
            
            supply_map = target_density  
            
            if isinstance(target_demand, torch.Tensor) and target_demand.dim() == 2:
                # Scheme 3: Calibrate area_per_track once using EGR net_map (demand map)
                total_density = density_map.sum()
                total_demand = target_demand.sum()
                
                if total_demand > 0 and total_density > 0:
                    # Initialize area_per_track once (persistent buffer)
                    if isinstance(area_per_track, torch.Tensor) and area_per_track.numel() == 1:
                        if (area_per_track <= 0).all():
                            area_per_track.fill_(total_density / total_demand)
                        calibrated_area_per_track = area_per_track
                    else:
                        calibrated_area_per_track = total_density / total_demand
                    
                    # Convert density (area) to demand (track equivalents)
                    demand_in_tracks = density_map / calibrated_area_per_track
                    
                    # Compute utilization per bin for debugging
                    utilization = demand_in_tracks / supply_map.clamp(min=1e-6)
                    
                    # Overflow in track units: positive means congestion (utilization > 1)
                    overflow_in_tracks = (demand_in_tracks - supply_map).clamp(min=0)
                    
                    # Convert back to area units for gradient consistency
                    overflow_map = overflow_in_tracks * calibrated_area_per_track
                    
                    # Debug logging
                    logger.debug(f"Calibration(net_map): area_per_track={calibrated_area_per_track:.3f}, "
                                f"utilization: mean={utilization.mean():.2f}, max={utilization.max():.2f}, "
                                f"overflow_bins={(overflow_in_tracks > 0).sum().item()}/{overflow_map.numel()}")
                else:
                    # Fallback: use density map directly
                    overflow_map = density_map
                    logger.debug("Fallback: using density map directly (net_map or density is zero)")
            else:
                # Fallback to utilization-based calibration (legacy)
                # Target utilization: what fraction of supply should demand use on average
                # Lower value = more aggressive spreading (easier to trigger overflow)
                target_utilization = 0.8  # TODO: make this configurable
                
                # Compute calibration factor
                total_density = density_map.sum()
                total_supply = supply_map.sum()
                
                if total_supply > 0 and total_density > 0:
                    # area_per_track: how much segment area corresponds to 1 track
                    # At target_utilization, total demand_in_tracks = target_utilization * total_supply
                    # So: total_density / k = target_utilization * total_supply
                    # => k = total_density / (target_utilization * total_supply)
                    calibrated_area_per_track = total_density / (target_utilization * total_supply)
                    
                    # Convert density (area) to demand (track equivalents)
                    demand_in_tracks = density_map / calibrated_area_per_track
                    
                    # Compute utilization per bin for debugging
                    utilization = demand_in_tracks / supply_map.clamp(min=1e-6)
                    
                    # Overflow in track units: positive means congestion (utilization > 1)
                    overflow_in_tracks = (demand_in_tracks - supply_map).clamp(min=0)
                    
                    # Convert back to area units for gradient consistency
                    overflow_map = overflow_in_tracks * calibrated_area_per_track
                    
                    # Debug logging
                    logger.debug(f"Calibration(legacy): area_per_track={calibrated_area_per_track:.3f}, "
                                f"target_util={target_utilization}, "
                                f"utilization: mean={utilization.mean():.2f}, max={utilization.max():.2f}, "
                                f"overflow_bins={(overflow_in_tracks > 0).sum().item()}/{overflow_map.numel()}")
                else:
                    # Fallback: use density map directly
                    overflow_map = density_map
                    logger.debug("Fallback: using density map directly (supply or density is zero)")
        else:
            # Uniform target_density (scalar): use original density map
            # This is the standard electric potential without supply-aware adjustment
            overflow_map = density_map
        
        # Normalize for DCT
        overflow_map_normalized = overflow_map.clone()
        overflow_map_normalized.mul_(1.0 / bin_area)
        
        # DCT transform on overflow map
        auv = dct2.forward(overflow_map_normalized)
        
        # Compute electric field (gradient of potential)
        auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_half)
        auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_half)
        
        # IDCT/IDXST to get field maps
        ctx.field_map_x = idxst_idct.forward(auv_by_wu2_plus_wv2_wu)
        ctx.field_map_y = idct_idxst.forward(auv_by_wu2_plus_wv2_wv)
        

        # Compute energy
        if fast_mode:
            # Dummy energy for gradient computation
            energy = torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device)
        else:
            # Compute potential: phi = IDCT(auv / (wu^2 + wv^2))
            auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2)
            potential_map = idct2.forward(auv_by_wu2_plus_wv2)
            
            # Scale potential map by bin area to approximate continuous potential
            # This makes the values physical and independent of bin count.
            # We apply this in-place so it is reflected in the plot and energy.
            potential_map.mul_(bin_size_x * bin_size_y)

            # Energy = sum(overflow * potential)
            # Only overflow (demand - supply) contributes to energy
            # This encourages the optimizer to reduce overflow in congested regions
            energy = potential_map.mul(overflow_map_normalized).sum()
        

        # plot overflow map
        global _PLOT_ITER
        iter = _PLOT_ITER
        _PLOT_ITER += 1
        plot_root = "/home/sxr/workspace/test-benchmark/benchmark/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot"
        plot_dirs = [
            "overflow",
            "density",
            "egr_supply",
            "egr_overflow",
            "egr_netmap",
            "egr_supply_original",
            "potential",
        ]
        for d in plot_dirs:
            os.makedirs(os.path.join(plot_root, d), exist_ok=True)

        plot_density_map(overflow_map_normalized, 
            title="Overflow Map", 
            save_path=f"{plot_root}/overflow/overflow_map_iter{iter}.png")
        
        # plot density map (segment demand)
        plot_density_map(density_map, 
            title="Density Map (Segment Demand)", 
            save_path=f"{plot_root}/density/density_map_iter{iter}.png")
        
        # Debug: print statistics
        # logger.info(f"=== Debug Statistics ===")
        # logger.info(f"density_map: min={density_map.min():.2f}, max={density_map.max():.2f}, sum={density_map.sum():.2f}")
        # logger.info(f"supply_map (target_density): min={supply_map.min():.2f}, max={supply_map.max():.2f}, sum={supply_map.sum():.2f}")
        # logger.info(f"supply_map shape: {supply_map.shape}, density_map shape: {density_map.shape}")
        
        # plot EGR supply map
        plot_density_map(supply_map, 
            title="EGR Supply Map", 
            save_path=f"{plot_root}/egr_supply/egr_supply_map_iter{iter}.png")
        
        # plot EGR overflow map
        from .egr_resample import load_egr_csv_map
        egr_overflow_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/BM64/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/overflow_map_planar.csv"
        egr_overflow_map = load_egr_csv_map(egr_overflow_path)
        egr_overflow_tensor = torch.from_numpy(egr_overflow_map).to(segment_pos.device)
        plot_density_map(egr_overflow_tensor, 
            title="EGR Overflow Map (from CSV)", 
            save_path=f"{plot_root}/egr_overflow/egr_overflow_map_iter{iter}.png")

        egr_netmap_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/BM64/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/net_map_planar.csv"
        egr_netmap = load_egr_csv_map(egr_netmap_path)
        egr_netmap_tensor = torch.from_numpy(egr_netmap).to(segment_pos.device)
        plot_density_map(egr_netmap_tensor, 
            title="EGR Netmap (from CSV)", 
            save_path=f"{plot_root}/egr_netmap/egr_netmap_iter{iter}.png")
        
        # plot EGR ORIGINAL supply map (before resample) for comparison
        egr_supply_path = "/nfs/share/home/sxr/routability_benchmark/dataset_cx55/20251023/BM64/workspace/output/iEDA/data/rt/rt_temp_directory/early_router/supply_map_planar.csv"
        egr_supply_original = load_egr_csv_map(egr_supply_path)
        egr_supply_original_tensor = torch.from_numpy(egr_supply_original).to(segment_pos.device)
        logger.info(f"EGR ORIGINAL supply: shape={egr_supply_original.shape}, min={egr_supply_original.min():.2f}, max={egr_supply_original.max():.2f}")
        plot_density_map(egr_supply_original_tensor, 
            title="EGR Supply Map (ORIGINAL from CSV)", 
            save_path=f"{plot_root}/egr_supply_original/egr_supply_original_iter{iter}.png")
        
        # plot potential map
        plot_density_map(potential_map, 
            title="Potential Map", 
            save_path=f"{plot_root}/potential/potential_map_iter{iter}.png")
        # exit(0)

        if segment_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug(f"Segment electric potential forward: {(time.time() - tt) * 1000:.2f} ms")
        
        return energy
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients using electric force.
        """
        tt = time.time()
        
        # Use C++/CUDA backend for force computation
        num_movable_nodes = ctx.num_segments
        num_filler_nodes = 0
        
        if grad_output.is_cuda:
            output = -electric_potential_cuda.electric_force(
                grad_output,
                ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_impacted_bins_x, ctx.num_impacted_bins_y,
                0, 0,  # filler impacted bins
                ctx.field_map_x.view([-1]),
                ctx.field_map_y.view([-1]),
                ctx.segment_pos,
                ctx.segment_size_x_clamped,
                ctx.segment_size_y_clamped,
                ctx.offset_x,
                ctx.offset_y,
                ctx.ratio,
                ctx.bin_center_x,
                ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                num_movable_nodes, num_filler_nodes,
                ctx.deterministic_flag,
                ctx.sorted_segment_map
            )
        else:
            output = -electric_potential_cpp.electric_force(
                grad_output,
                ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_impacted_bins_x, ctx.num_impacted_bins_y,
                0, 0,  # filler impacted bins
                ctx.field_map_x.view([-1]),
                ctx.field_map_y.view([-1]),
                ctx.segment_pos,
                ctx.segment_size_x_clamped,
                ctx.segment_size_y_clamped,
                ctx.offset_x,
                ctx.offset_y,
                ctx.ratio,
                ctx.bin_center_x,
                ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                num_movable_nodes, num_filler_nodes
            )
        
        if grad_output.is_cuda:
            torch.cuda.synchronize()
        logger.debug(f"Segment electric potential backward: {(time.time() - tt) * 1000:.2f} ms")
        
        # Return gradients (only for segment_pos, others are None)
        return (output,) + (None,) * 38


class LShapeElectricPotential(nn.Module):
    """
    Compute electric potential for L-shape routing segments.
    
    This module:
    1. Computes density map for segments using C++/CUDA
    2. Solves Poisson equation using DCT
    3. Computes gradients using electric force with C++/CUDA
    
    The target_density can be:
    - A scalar (uniform target for all bins)
    - A 2D tensor (per-bin target, e.g., from EGR supply map)
    The target_demand (optional) can be:
    - A 2D tensor (per-bin demand, e.g., from EGR net map)
    """
    
    def __init__(
        self,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        target_density=1.0,
        target_demand=None,
        padding=0,
        deterministic_flag=True,
        fast_mode=False
    ):
        """
        Initialize L-shape electric potential module.
        
        Args:
            xl, yl, xh, yh: die boundaries
            bin_size_x, bin_size_y: bin sizes
            num_bins_x, num_bins_y: number of bins
            target_density: target routing density (scalar or 2D tensor)
                - If scalar: uniform target for all bins
                - If 2D tensor: per-bin target from EGR supply map
            target_demand: target routing demand (2D tensor from EGR net map)
            padding: bin padding
            deterministic_flag: whether to use deterministic routine
            fast_mode: if True, skip energy computation (only gradients)
        """
        super(LShapeElectricPotential, self).__init__()
        
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.padding = padding
        self.deterministic_flag = deterministic_flag
        self.fast_mode = fast_mode
        
        # Store target_density (scalar or 2D tensor)
        if isinstance(target_density, torch.Tensor):
            # EGR supply map: shape (num_bins_x, num_bins_y)
            self.register_buffer('target_density', target_density)
        else:
            # Scalar: will be expanded to tensor on first forward
            self.target_density = target_density

        # Store target_demand (2D tensor from EGR net map) or None
        if isinstance(target_demand, torch.Tensor):
            self.register_buffer('target_demand', target_demand)
        else:
            self.target_demand = target_demand

        # Persistent calibration factor (area per track), initialized on first forward
        self.register_buffer('area_per_track', torch.tensor(0.0))
        
        # Will be initialized on first forward pass
        self.bin_center_x = None
        self.bin_center_y = None
        self.padding_mask = None
        self.initial_density_map = None
        
        # DCT related
        self.exact_expkM = None
        self.exact_expkN = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None
        self.dct2 = None
        self.idct2 = None
        self.idct_idxst = None
        self.idxst_idct = None
    
    def _init_bins(self, device, dtype):
        """Initialize bin centers and padding mask."""
        # Bin centers
        self.bin_center_x = torch.arange(
            self.num_bins_x, device=device, dtype=dtype
        ).mul_(self.bin_size_x).add_(self.xl + self.bin_size_x / 2)
        
        self.bin_center_y = torch.arange(
            self.num_bins_y, device=device, dtype=dtype
        ).mul_(self.bin_size_y).add_(self.yl + self.bin_size_y / 2)
        
        # Padding mask (使用bool类型以兼容新版PyTorch的masked_fill_)
        if self.padding > 0:
            self.padding_mask = torch.ones(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.bool, device=device
            )
            self.padding_mask[
                self.padding:self.num_bins_x - self.padding,
                self.padding:self.num_bins_y - self.padding
            ] = False
        else:
            self.padding_mask = torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.bool, device=device
            )
        
        # Initial density map
        self.initial_density_map = torch.zeros(
            self.num_bins_x, self.num_bins_y,
            dtype=dtype, device=device
        )
        
        # Ensure target_density is properly initialized as tensor
        self._init_target_density(device, dtype)
        self._init_target_demand(device, dtype)
        self.area_per_track = self.area_per_track.to(device=device, dtype=dtype)
    
    def _init_target_density(self, device, dtype):
        """Initialize or convert target_density to proper tensor format."""
        if isinstance(self.target_density, torch.Tensor):
            # Already a tensor (e.g., EGR supply map)
            if self.target_density.shape != (self.num_bins_x, self.num_bins_y):
                logger.warning(f"target_density shape {self.target_density.shape} != "
                              f"expected ({self.num_bins_x}, {self.num_bins_y}), resizing...")
                # Resize using interpolation
                from torch.nn.functional import interpolate
                td = self.target_density.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                self.target_density = td.squeeze(0).squeeze(0).to(device=device, dtype=dtype)
            else:
                self.target_density = self.target_density.to(device=device, dtype=dtype)
        else:
            # Scalar: create uniform tensor
            self.target_density = torch.full(
                (self.num_bins_x, self.num_bins_y),
                self.target_density,
                dtype=dtype, device=device
            )

    def _init_target_demand(self, device, dtype):
        """Initialize or convert target_demand to proper tensor format."""
        if isinstance(self.target_demand, torch.Tensor):
            if self.target_demand.shape != (self.num_bins_x, self.num_bins_y):
                logger.warning(f"target_demand shape {self.target_demand.shape} != "
                              f"expected ({self.num_bins_x}, {self.num_bins_y}), resizing...")
                from torch.nn.functional import interpolate
                td = self.target_demand.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                self.target_demand = td.squeeze(0).squeeze(0).to(device=device, dtype=dtype)
            else:
                self.target_demand = self.target_demand.to(device=device, dtype=dtype)
        else:
            self.target_demand = None
    
    def set_target_density(self, target_density):
        """
        Set target density (supply map) from EGR.
        
        Args:
            target_density: scalar or 2D tensor (num_bins_x, num_bins_y)
                Values should be normalized to [0, 1] where:
                - 1.0 = full routing capacity
                - 0.0 = no routing capacity (blockage)
        """
        if isinstance(target_density, torch.Tensor):
            # Ensure correct shape
            if target_density.shape != (self.num_bins_x, self.num_bins_y):
                from torch.nn.functional import interpolate
                td = target_density.unsqueeze(0).unsqueeze(0)
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                target_density = td.squeeze(0).squeeze(0)
            
            # Move to correct device/dtype if bin_center_x is initialized
            if self.bin_center_x is not None:
                target_density = target_density.to(
                    device=self.bin_center_x.device, 
                    dtype=self.bin_center_x.dtype
                )
            self.target_density = target_density
            logger.info(f"Set target_density from EGR supply map: "
                       f"min={target_density.min():.3f}, max={target_density.max():.3f}, "
                       f"mean={target_density.mean():.3f}")
        else:
            # Scalar
            if self.bin_center_x is not None:
                self.target_density = torch.full(
                    (self.num_bins_x, self.num_bins_y),
                    target_density,
                    dtype=self.bin_center_x.dtype,
                    device=self.bin_center_x.device
                )
            else:
                self.target_density = target_density
            logger.info(f"Set uniform target_density: {target_density}")

    def set_target_demand(self, target_demand):
        """
        Set target demand (net map) from EGR.
        
        Args:
            target_demand: 2D tensor (num_bins_x, num_bins_y)
        """
        if isinstance(target_demand, torch.Tensor):
            if target_demand.shape != (self.num_bins_x, self.num_bins_y):
                from torch.nn.functional import interpolate
                td = target_demand.unsqueeze(0).unsqueeze(0)
                td = interpolate(td, size=(self.num_bins_x, self.num_bins_y), mode='bilinear', align_corners=False)
                target_demand = td.squeeze(0).squeeze(0)
            
            if self.bin_center_x is not None:
                target_demand = target_demand.to(
                    device=self.bin_center_x.device, 
                    dtype=self.bin_center_x.dtype
                )
            self.target_demand = target_demand
            if isinstance(self.area_per_track, torch.Tensor):
                self.area_per_track.zero_()
            logger.info(f"Set target_demand from EGR net map: "
                       f"min={target_demand.min():.3f}, max={target_demand.max():.3f}, "
                       f"mean={target_demand.mean():.3f}")
        else:
            self.target_demand = None
            if isinstance(self.area_per_track, torch.Tensor):
                self.area_per_track.zero_()
            logger.info("Cleared target_demand")
    
    def _init_dct(self, device, dtype):
        """Initialize DCT related parameters."""
        M = self.num_bins_x
        N = self.num_bins_y
        
        # Precompute exp(-j*pi*k/M) and exp(-j*pi*k/N)
        self.exact_expkM = precompute_expk(M, dtype=dtype, device=device)
        self.exact_expkN = precompute_expk(N, dtype=dtype, device=device)
        
        # DCT operators
        self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
        if not self.fast_mode:
            self.idct2 = dct.IDCT2(self.exact_expkM, self.exact_expkN)
        self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM, self.exact_expkN)
        self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM, self.exact_expkN)
        
        # wu and wv (angular frequencies)
        wu = torch.arange(M, dtype=dtype, device=device).mul(2 * np.pi / M).view([M, 1])
        # Scale wv for non-square bins
        wv = torch.arange(N, dtype=dtype, device=device).mul(2 * np.pi / N).view([1, N])
        wv = wv.mul_(self.bin_size_x / self.bin_size_y)
        
        # Precompute inverse of (wu^2 + wv^2)
        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0, 0] = 1.0  # Avoid division by zero
        
        self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        self.inv_wu2_plus_wv2[0, 0] = 0.0  # DC component has zero potential
        
        self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(0.5)
        self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(0.5)
    
    def _prepare_segment_data(self, segment_pos, segment_size_x, segment_size_y):
        """Prepare segment data for density computation."""
        sqrt2 = math.sqrt(2)
        
        # Clamp sizes to minimum bin size
        # Using sqrt(2) ensures that the segment "splats" cover enough bins to form a smooth
        # continuous potential field, which is critical for the optimizer to spread cells.
        # Reducing this factor makes the force too local (sharp), causing cells to clump/gather
        # because they don't feel the repulsion until they are right on top of each other.
        # segment_size_x_clamped = segment_size_x.clamp(min=self.bin_size_x * sqrt2)
        # segment_size_y_clamped = segment_size_y.clamp(min=self.bin_size_y * sqrt2)
        segment_size_x_clamped = segment_size_x
        segment_size_y_clamped = segment_size_y
        
        # Compute offsets
        offset_x = (segment_size_x - segment_size_x_clamped).mul(0.5)
        offset_y = (segment_size_y - segment_size_y_clamped).mul(0.5)
        
        # Compute area ratio
        segment_area = segment_size_x * segment_size_y
        clamped_area = segment_size_x_clamped * segment_size_y_clamped
        ratio = segment_area / clamped_area.clamp(min=1e-10)
        
        # Compute maximum impacted bins
        sqrt2_bin_x = sqrt2 * self.bin_size_x
        sqrt2_bin_y = sqrt2 * self.bin_size_y
        
        if segment_size_x.numel() > 0:
            num_impacted_bins_x = int(
                ((segment_size_x.max() + 2 * sqrt2_bin_x) / self.bin_size_x)
                .ceil().clamp(max=self.num_bins_x).item()
            )
            num_impacted_bins_y = int(
                ((segment_size_y.max() + 2 * sqrt2_bin_y) / self.bin_size_y)
                .ceil().clamp(max=self.num_bins_y).item()
            )
        else:
            num_impacted_bins_x = 0
            num_impacted_bins_y = 0
        
        # Sort segments by size
        if segment_size_x.numel() > 0:
            sorted_segment_map = torch.argsort(segment_size_x).to(torch.int32)
        else:
            sorted_segment_map = torch.tensor([], dtype=torch.int32, device=segment_pos.device)
        
        return (
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            num_impacted_bins_x,
            num_impacted_bins_y,
            sorted_segment_map
        )
    
    def forward(self, segment_pos, segment_size_x, segment_size_y):
        """
        Compute electric potential energy for routing segments.
        
        Args:
            segment_pos: [num_segments * 2] positions (llx then lly)
            segment_size_x: [num_segments] segment widths
            segment_size_y: [num_segments] segment heights
            
        Returns:
            energy: electric potential energy (scalar)
        """
        num_segments = segment_size_x.numel()
        
        if num_segments == 0:
            return torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device, requires_grad=True)
        
        # Initialize on first call
        if self.bin_center_x is None:
            self._init_bins(segment_pos.device, segment_pos.dtype)
        if self.dct2 is None:
            self._init_dct(segment_pos.device, segment_pos.dtype)
        
        # Prepare segment data
        (
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            num_impacted_bins_x,
            num_impacted_bins_y,
            sorted_segment_map
        ) = self._prepare_segment_data(segment_pos, segment_size_x, segment_size_y)
        
        # Compute electric potential
        energy = SegmentElectricPotentialFunction.apply(
            segment_pos,
            segment_size_x,
            segment_size_y,
            segment_size_x_clamped,
            segment_size_y_clamped,
            offset_x,
            offset_y,
            ratio,
            self.bin_center_x,
            self.bin_center_y,
            self.initial_density_map,
            self.target_density,
            self.target_demand,
            self.area_per_track,
            self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y,
            num_segments,
            self.padding,
            self.padding_mask,
            self.num_bins_x, self.num_bins_y,
            num_impacted_bins_x,
            num_impacted_bins_y,
            self.deterministic_flag,
            sorted_segment_map,
            self.exact_expkM,
            self.exact_expkN,
            self.inv_wu2_plus_wv2,
            self.wu_by_wu2_plus_wv2_half,
            self.wv_by_wu2_plus_wv2_half,
            self.dct2,
            self.idct2,
            self.idct_idxst,
            self.idxst_idct,
            self.fast_mode
        )
        
        return energy


class LShapeRoutabilityPotentialOp(nn.Module):
    """
    Complete L-shape routability operator using electric potential.
    
    This operator:
    1. Builds L-shape segments from Steiner tree edges
    2. Computes segment density using electric potential model
    3. Returns differentiable cost for optimization
    """
    
    def __init__(
        self,
        placedb,
        data_collections,
        steiner_topo_op,
        l_direction_resolver,
        num_bins_x=64,
        num_bins_y=64,
        target_density=1.0,
        target_demand=None,
        wire_width=0.0
    ):
        """
        Initialize routability operator.
        
        Args:
            placedb: placement database
            data_collections: data collections with pin/net info
            steiner_topo_op: Steiner topology operator
            l_direction_resolver: L-direction resolver
            num_bins_x, num_bins_y: number of bins
            target_density: target routing density
            target_demand: target routing demand (EGR net map)
            wire_width: wire width for segments
        """
        super(LShapeRoutabilityPotentialOp, self).__init__()
        
        self.placedb = placedb
        self.data_collections = data_collections
        self.steiner_topo_op = steiner_topo_op
        self.l_direction_resolver = l_direction_resolver
        self.wire_width = wire_width
        
        # Compute bin sizes
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        
        # Create electric potential operator
        self.electric_potential_op = LShapeElectricPotential(
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            target_density=target_density,
            target_demand=target_demand,
            padding=0,
            deterministic_flag=True,
            fast_mode=False
        )
        
        # Import segment builder
        from .l_shape_segment import LShapeSegmentOp
        self.segment_op = LShapeSegmentOp(wire_width=wire_width, use_vectorized=True)
        
        # Cached L-directions
        self.l_directions = None
    
    def resolve_l_directions(self):
        """Resolve L-directions for all edges."""
        if self.l_direction_resolver is not None:
            self.l_directions = self.l_direction_resolver.resolve_l_directions()
        else:
            # Default to V_FIRST
            num_edges = self.data_collections.flat_pin_from.numel()
            self.l_directions = torch.ones(
                num_edges,
                dtype=torch.int32,
                device=self.data_collections.flat_pin_from.device
            )
    
    def forward(self, pos):
        """
        Compute routability cost.
        
        Args:
            pos: cell positions [num_nodes * 2]
            
        Returns:
            cost: routability cost (scalar, differentiable)
        """
        # Resolve L-directions if not done
        if self.l_directions is None:
            self.resolve_l_directions()
        
        # Compute Steiner point coordinates
        newx, newy = self.steiner_topo_op(pos)
        
        # Build segments
        segment_result = self.segment_op(
            newx, newy,
            self.data_collections.flat_pin_from,
            self.data_collections.flat_pin_to,
            self.l_directions
        )
        
        if segment_result['num_segments'] == 0:
            return torch.zeros(1, dtype=pos.dtype, device=pos.device, requires_grad=True)
        
        # Compute electric potential cost
        segment_pos = segment_result['segment_pos']
        segment_size_x = segment_result['segment_size_x']
        segment_size_y = segment_result['segment_size_y']
        
        cost = self.electric_potential_op(segment_pos, segment_size_x, segment_size_y)
        
        return cost


def create_l_shape_electric_potential(
    placedb,
    num_bins_x=64,
    num_bins_y=64,
    target_density=1.0,
    target_demand=None,
    padding=0,
    deterministic_flag=True,
    fast_mode=False
):
    """
    Factory function to create LShapeElectricPotential.
    
    Args:
        placedb: placement database
        num_bins_x, num_bins_y: number of bins
        target_density: target routing density
        target_demand: target routing demand (EGR net map)
        padding: bin padding
        deterministic_flag: whether to use deterministic routine
        fast_mode: if True, skip energy computation
        
    Returns:
        LShapeElectricPotential
    """
    bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
    bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
    
    return LShapeElectricPotential(
        xl=placedb.xl,
        yl=placedb.yl,
        xh=placedb.xh,
        yh=placedb.yh,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        target_density=target_density,
        target_demand=target_demand,
        padding=padding,
        deterministic_flag=deterministic_flag,
        fast_mode=fast_mode
    )
