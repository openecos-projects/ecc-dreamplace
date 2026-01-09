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
        
        # Normalize density map
        density_map_normalized = density_map.clone()
        density_map_normalized.mul_(1.0 / (bin_size_x * bin_size_y))
        
        # DCT transform
        auv = dct2.forward(density_map_normalized)
        
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
            # Energy = sum(density * potential)
            energy = potential_map.mul(density_map_normalized).sum()
        

        # plot potential map
        # plot_density_map(potential_map, 
        #     title="Potential Map", 
        #     save_path=f"/home/sxr/workspace/test-benchmark/benchmark/AiEDA/third_party/AutoDMP/dreamplace/ops/routability/plot/potential_map.png")
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
        return (output,) + (None,) * 36


class LShapeElectricPotential(nn.Module):
    """
    Compute electric potential for L-shape routing segments.
    
    This module:
    1. Computes density map for segments using C++/CUDA
    2. Solves Poisson equation using DCT
    3. Computes gradients using electric force with C++/CUDA
    """
    
    def __init__(
        self,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        target_density=1.0,
        padding=0,
        deterministic_flag=False,
        fast_mode=False
    ):
        """
        Initialize L-shape electric potential module.
        
        Args:
            xl, yl, xh, yh: die boundaries
            bin_size_x, bin_size_y: bin sizes
            num_bins_x, num_bins_y: number of bins
            target_density: target routing density
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
        self.target_density = target_density
        self.padding = padding
        self.deterministic_flag = deterministic_flag
        self.fast_mode = fast_mode
        
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
        
        # Padding mask
        if self.padding > 0:
            self.padding_mask = torch.ones(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.uint8, device=device
            )
            self.padding_mask[
                self.padding:self.num_bins_x - self.padding,
                self.padding:self.num_bins_y - self.padding
            ].fill_(0)
        else:
            self.padding_mask = torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=torch.uint8, device=device
            )
        
        # Initial density map
        self.initial_density_map = torch.zeros(
            self.num_bins_x, self.num_bins_y,
            dtype=dtype, device=device
        )
    
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
            padding=0,
            deterministic_flag=False,
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
    padding=0,
    deterministic_flag=False,
    fast_mode=False
):
    """
    Factory function to create LShapeElectricPotential.
    
    Args:
        placedb: placement database
        num_bins_x, num_bins_y: number of bins
        target_density: target routing density
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
        padding=padding,
        deterministic_flag=deterministic_flag,
        fast_mode=fast_mode
    )
