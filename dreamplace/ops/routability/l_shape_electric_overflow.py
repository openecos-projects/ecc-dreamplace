##
# @file   l_shape_electric_overflow.py
# @brief  Compute routing density overflow from L-shape segments
#         Uses C++/CUDA backends for efficient density map computation
#

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import logging

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

logger = logging.getLogger(__name__)


class SegmentDensityMapFunction(Function):
    """
    Compute density map for routing segments using C++/CUDA backends.
    
    Segments are treated as virtual cells with given positions and sizes.
    
    Note: This function computes the raw density map only. 
    Supply-aware overflow computation should be done at a higher level.
    """
    @staticmethod
    def forward(
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
    ):
        """
        Compute density map for segments.
        
        Args:
            segment_pos: [num_segments * 2] positions (llx then lly)
            segment_size_x/y: [num_segments] segment sizes
            target_density: can be scalar or 2D tensor (only scalar passed to C++)
            Others: similar to ElectricDensityMapFunction
            
        Returns:
            density_map: [num_bins_x, num_bins_y]
        """
        num_movable_nodes = num_segments
        num_filler_nodes = 0
        
        # C++ backend expects scalar target_density
        # If we have a 2D supply map, use 1.0 for raw density computation
        # The supply-aware overflow is computed at a higher level
        if isinstance(target_density, torch.Tensor) and target_density.dim() == 2:
            cpp_target_density = 1.0
        else:
            cpp_target_density = float(target_density)
        
        if segment_pos.is_cuda:
            output = electric_potential_cuda.density_map(
                segment_pos.view(segment_pos.numel()),
                segment_size_x_clamped,
                segment_size_y_clamped,
                offset_x, offset_y, ratio,
                bin_center_x, bin_center_y,
                initial_density_map,
                cpp_target_density,  # scalar
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_movable_nodes, num_filler_nodes,
                padding,
                num_bins_x, num_bins_y,
                num_impacted_bins_x, num_impacted_bins_y,
                0, 0,  # filler impacted bins
                deterministic_flag,
                sorted_segment_map
            )
        else:
            output = electric_potential_cpp.density_map(
                segment_pos.view(segment_pos.numel()),
                segment_size_x_clamped,
                segment_size_y_clamped,
                offset_x, offset_y, ratio,
                bin_center_x, bin_center_y,
                initial_density_map,
                cpp_target_density,  # scalar
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_movable_nodes, num_filler_nodes,
                padding,
                num_bins_x, num_bins_y,
                num_impacted_bins_x, num_impacted_bins_y,
                0, 0,  # filler impacted bins
                deterministic_flag
            )
        
        density_map = output.view([num_bins_x, num_bins_y])
        
        # Set padding density (use scalar for consistency)
        if padding > 0:
            density_map.masked_fill_(padding_mask, cpp_target_density * bin_size_x * bin_size_y)
        
        return density_map


class LShapeElectricOverflow(nn.Module):
    """
    Compute routing density overflow for L-shape segments.
    
    This module treats routing segments as virtual cells and uses
    the electric potential C++/CUDA backends for density computation.
    """
    
    def __init__(
        self,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        target_density=1.0,
        padding=0,
        deterministic_flag=False
    ):
        """
        Initialize L-shape electric overflow module.
        
        Args:
            xl, yl, xh, yh: die boundaries
            bin_size_x, bin_size_y: bin sizes
            num_bins_x, num_bins_y: number of bins
            target_density: target routing density
            padding: bin padding
            deterministic_flag: whether to use deterministic routine
        """
        super(LShapeElectricOverflow, self).__init__()
        
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
        
        # Will be initialized on first forward pass
        self.bin_center_x = None
        self.bin_center_y = None
        self.padding_mask = None
        self.initial_density_map = None
    
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
        
        # Initial density map (for fixed obstacles, if any)
        self.initial_density_map = torch.zeros(
            self.num_bins_x, self.num_bins_y,
            dtype=dtype, device=device
        )
    
    def _prepare_segment_data(self, segment_pos, segment_size_x, segment_size_y):
        """
        Prepare segment data for density computation.
        
        Clamp segment sizes to minimum bin size (similar to cell clamping).
        """
        sqrt2 = math.sqrt(2)
        
        # Clamp sizes to minimum bin size
        segment_size_x_clamped = segment_size_x.clamp(min=self.bin_size_x * sqrt2)
        segment_size_y_clamped = segment_size_y.clamp(min=self.bin_size_y * sqrt2)
        
        # Compute offsets (for smooth density distribution)
        offset_x = (segment_size_x - segment_size_x_clamped).mul(0.5)
        offset_y = (segment_size_y - segment_size_y_clamped).mul(0.5)
        
        # Compute area ratio (original area / clamped area)
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
        
        # Sort segments by size for better memory access
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
        Compute density overflow for routing segments.
        
        Args:
            segment_pos: [num_segments * 2] positions (llx then lly)
            segment_size_x: [num_segments] segment widths
            segment_size_y: [num_segments] segment heights
            
        Returns:
            density_cost: sum of overflow
            max_density: maximum density ratio
        """
        num_segments = segment_size_x.numel()
        
        if num_segments == 0:
            return (
                torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device),
                torch.zeros(1, dtype=segment_pos.dtype, device=segment_pos.device)
            )
        
        # Initialize bins on first call
        if self.bin_center_x is None:
            self._init_bins(segment_pos.device, segment_pos.dtype)
        
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
        
        # Compute density map
        density_map = SegmentDensityMapFunction.forward(
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
            sorted_segment_map
        )
        
        # Compute overflow cost
        bin_area = self.bin_size_x * self.bin_size_y
        target_area = self.target_density * bin_area
        
        density_cost = (density_map - target_area).clamp_(min=0.0).sum().unsqueeze(0)
        max_density = (density_map.max() / bin_area).unsqueeze(0)
        
        return density_cost, max_density
    
    def compute_density_map(self, segment_pos, segment_size_x, segment_size_y):
        """
        Compute density map without overflow calculation.
        
        Returns:
            density_map: [num_bins_x, num_bins_y]
        """
        num_segments = segment_size_x.numel()
        
        if num_segments == 0:
            return torch.zeros(
                self.num_bins_x, self.num_bins_y,
                dtype=segment_pos.dtype, device=segment_pos.device
            )
        
        # Initialize bins on first call
        if self.bin_center_x is None:
            self._init_bins(segment_pos.device, segment_pos.dtype)
        
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
        
        # Compute density map
        density_map = SegmentDensityMapFunction.forward(
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
            sorted_segment_map
        )
        
        return density_map


def create_l_shape_electric_overflow(
    placedb,
    num_bins_x=64,
    num_bins_y=64,
    target_density=1.0,
    padding=0,
    deterministic_flag=False
):
    """
    Factory function to create LShapeElectricOverflow.
    
    Args:
        placedb: placement database
        num_bins_x, num_bins_y: number of bins
        target_density: target routing density
        padding: bin padding
        deterministic_flag: whether to use deterministic routine
        
    Returns:
        LShapeElectricOverflow
    """
    bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
    bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
    
    return LShapeElectricOverflow(
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
        deterministic_flag=deterministic_flag
    )
