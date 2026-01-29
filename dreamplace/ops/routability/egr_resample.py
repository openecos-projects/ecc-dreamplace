##
# @file   egr_resample.py
# @brief  Resample EGR non-uniform GCell supply/demand to uniform bin grids
#         For L-shape routability optimization
#
# EGR outputs:
#   - gcell.info: GCell coordinates (non-uniform sizes)
#   - supply_map_*.csv: routing capacity per GCell per layer
#   - net_map_*.csv: routing demand per GCell per layer
#   - overflow_map_*.csv: overflow per GCell per layer
#   - *_planar.csv: aggregated across all layers
#

import torch
import numpy as np
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_egr_csv_map(csv_path):
    """
    Load EGR map from CSV file.
    
    CSV format: comma-separated values, one row per y-coordinate.
    Note: 
    - Each row may have trailing comma, resulting in empty last column.
    - EGR writes CSV with y from large to small (row 0 = y_max, last row = y=0)
    
    Args:
        csv_path: path to CSV file
        
    Returns:
        numpy array of shape (num_x, num_y), index [x, y] gives value at GCell (x, y)
    """
    rows = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by comma, filter empty strings (from trailing comma)
            values = [float(v) for v in line.split(',') if v.strip()]
            rows.append(values)
    
    # Stack rows: shape (num_y, num_x)
    # Note: EGR writes y from large to small, so row[0] = y_max, row[-1] = y=0
    data = np.array(rows, dtype=np.float32)
    
    # Flip y-axis: now row[0] = y=0, row[-1] = y_max
    # Use .copy() to create contiguous array (flipud creates negative stride view)
    data = np.flipud(data).copy()
    
    # Transpose to (num_x, num_y): data[x, y] = GCell(x, y)
    data = data.T.copy()  # Also copy after transpose for contiguous memory
    
    logger.info(f"Loaded EGR map from {csv_path}: shape {data.shape}")
    return data


class EGRGCellInfo:
    """
    Load and manage EGR GCell information (non-uniform grid).
    
    GCell info file format: grid_x,grid_y,llx,lly,urx,ury (units: nm)
    """
    
    def __init__(self, gcell_info_path=None):
        """
        Initialize GCell info loader.
        
        Args:
            gcell_info_path: path to gcell.info file
        """
        self.gcell_info = {}  # (gx, gy) -> (llx, lly, urx, ury) in nm
        self.num_gcells_x = 0
        self.num_gcells_y = 0
        self.xl = None  # min x (nm)
        self.yl = None  # min y (nm)
        self.xh = None  # max x (nm)
        self.yh = None  # max y (nm)
        
        if gcell_info_path:
            self.load(gcell_info_path)
    
    def load(self, gcell_info_path):
        """
        Load GCell info from file.
        
        Args:
            gcell_info_path: path to gcell.info file
        """
        self.gcell_info = {}
        max_gx, max_gy = 0, 0
        
        try:
            with open(gcell_info_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        gx, gy = int(parts[0]), int(parts[1])
                        llx, lly = float(parts[2]), float(parts[3])
                        urx, ury = float(parts[4]), float(parts[5])
                        self.gcell_info[(gx, gy)] = (llx, lly, urx, ury)
                        max_gx = max(max_gx, gx)
                        max_gy = max(max_gy, gy)
                        
                        # Track bounding box
                        if self.xl is None or llx < self.xl:
                            self.xl = llx
                        if self.yl is None or lly < self.yl:
                            self.yl = lly
                        if self.xh is None or urx > self.xh:
                            self.xh = urx
                        if self.yh is None or ury > self.yh:
                            self.yh = ury
            
            self.num_gcells_x = max_gx + 1
            self.num_gcells_y = max_gy + 1
            logger.info(f"Loaded EGR GCell info: {len(self.gcell_info)} gcells, "
                       f"grid={self.num_gcells_x}x{self.num_gcells_y}, "
                       f"bbox=[{self.xl:.0f},{self.yl:.0f}]-[{self.xh:.0f},{self.yh:.0f}] nm")
            
        except Exception as e:
            logger.error(f"Failed to load GCell info: {e}")
            raise
    
    def get_gcell_bbox(self, gx, gy):
        """Get GCell bounding box (llx, lly, urx, ury) in nm."""
        return self.gcell_info.get((gx, gy), None)
    
    def get_gcell_area(self, gx, gy):
        """Get GCell area in nm^2."""
        bbox = self.get_gcell_bbox(gx, gy)
        if bbox is None:
            return 0.0
        llx, lly, urx, ury = bbox
        return (urx - llx) * (ury - lly)
    
    def get_average_gcell_size(self):
        """
        Get average GCell width and height in nm.
        
        Returns:
            (avg_width, avg_height) in nm
        """
        if not self.gcell_info:
            return 0.0, 0.0
        
        widths = []
        heights = []
        for (gx, gy), (llx, lly, urx, ury) in self.gcell_info.items():
            widths.append(urx - llx)
            heights.append(ury - lly)
        
        avg_width = np.mean(widths) if widths else 0.0
        avg_height = np.mean(heights) if heights else 0.0
        
        return avg_width, avg_height
    
    def get_min_gcell_dimension(self, scale_factor=1.0, shift_factor=(0, 0)):
        """
        Get minimum of average GCell width/height, converted to DREAMPlace coordinates.
        
        This is useful as wire_width for L-shape segments, as it represents
        approximately one routing track's physical extent.
        
        Args:
            scale_factor: DREAMPlace scale factor
            shift_factor: not used for dimension calculation, kept for API consistency
            
        Returns:
            min(avg_gcell_width, avg_gcell_height) in DREAMPlace coordinates
        """
        avg_width, avg_height = self.get_average_gcell_size()
        # Convert from nm to DREAMPlace coordinates
        dp_width = avg_width * scale_factor
        dp_height = avg_height * scale_factor
        
        min_dim = min(dp_width, dp_height)
        logger.info(f"GCell average size: {avg_width:.1f} x {avg_height:.1f} nm, "
                   f"min dimension in DP coords: {min_dim:.3f}")
        return min_dim


class EGRCapacityResampler:
    """
    Resample EGR non-uniform GCell supply map to uniform bin grid.
    
    This takes EGR's supply_map (routing capacity) and resamples it to
    the uniform bin grid used by L-shape routability optimization.
    """
    
    def __init__(self, gcell_info, target_xl, target_yl, target_xh, target_yh,
                 num_bins_x, num_bins_y, scale_factor=1.0, shift_factor=(0, 0)):
        """
        Initialize capacity resampler.
        
        Args:
            gcell_info: EGRGCellInfo instance
            target_xl, target_yl, target_xh, target_yh: target grid bounds (DREAMPlace coords)
            num_bins_x, num_bins_y: number of uniform bins
            scale_factor: DREAMPlace scale factor
            shift_factor: DREAMPlace shift factor (x, y) in nm
        """
        self.gcell_info = gcell_info
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        
        # Target grid in DREAMPlace coordinates
        self.target_xl = target_xl
        self.target_yl = target_yl
        self.target_xh = target_xh
        self.target_yh = target_yh
        
        # Compute uniform bin size
        self.bin_size_x = (target_xh - target_xl) / num_bins_x
        self.bin_size_y = (target_yh - target_yl) / num_bins_y
        
        # Precompute GCell-to-bin mapping
        self._precompute_mapping()
    
    def _gcell_to_dreamplace_coords(self, llx, lly, urx, ury):
        """Convert GCell coordinates (nm) to DREAMPlace coordinates."""
        dp_llx = (llx - self.shift_factor[0]) * self.scale_factor
        dp_lly = (lly - self.shift_factor[1]) * self.scale_factor
        dp_urx = (urx - self.shift_factor[0]) * self.scale_factor
        dp_ury = (ury - self.shift_factor[1]) * self.scale_factor
        return dp_llx, dp_lly, dp_urx, dp_ury
    
    def _precompute_mapping(self):
        """
        Precompute the mapping from GCells to uniform bins.
        
        For each bin, store list of (gx, gy, overlap_ratio) where overlap_ratio
        is the fraction of the bin covered by that GCell.
        """
        self.bin_to_gcell_overlaps = defaultdict(list)  # (bx, by) -> [(gx, gy, overlap_ratio)]
        
        bin_area = self.bin_size_x * self.bin_size_y
        
        for (gx, gy), (llx, lly, urx, ury) in self.gcell_info.gcell_info.items():
            # Convert to DREAMPlace coordinates
            dp_llx, dp_lly, dp_urx, dp_ury = self._gcell_to_dreamplace_coords(llx, lly, urx, ury)
            gcell_area = (dp_urx - dp_llx) * (dp_ury - dp_lly)
            
            if gcell_area <= 0:
                continue
            
            # Find overlapping bins
            bin_xl = int(max(0, (dp_llx - self.target_xl) / self.bin_size_x))
            bin_xh = int(min(self.num_bins_x - 1, (dp_urx - self.target_xl) / self.bin_size_x))
            bin_yl = int(max(0, (dp_lly - self.target_yl) / self.bin_size_y))
            bin_yh = int(min(self.num_bins_y - 1, (dp_ury - self.target_yl) / self.bin_size_y))
            
            for bx in range(bin_xl, bin_xh + 1):
                for by in range(bin_yl, bin_yh + 1):
                    # Compute overlap area
                    bin_llx = self.target_xl + bx * self.bin_size_x
                    bin_lly = self.target_yl + by * self.bin_size_y
                    bin_urx = bin_llx + self.bin_size_x
                    bin_ury = bin_lly + self.bin_size_y
                    
                    overlap_llx = max(dp_llx, bin_llx)
                    overlap_lly = max(dp_lly, bin_lly)
                    overlap_urx = min(dp_urx, bin_urx)
                    overlap_ury = min(dp_ury, bin_ury)
                    
                    if overlap_urx > overlap_llx and overlap_ury > overlap_lly:
                        overlap_area = (overlap_urx - overlap_llx) * (overlap_ury - overlap_lly)
                        # overlap_ratio: fraction of GCell that falls into this bin
                        overlap_ratio = overlap_area / gcell_area
                        self.bin_to_gcell_overlaps[(bx, by)].append((gx, gy, overlap_ratio))
        
        logger.info(f"Precomputed GCell-to-bin mapping: {len(self.bin_to_gcell_overlaps)} bins")
    
    def resample_map(self, value_map, device=None, dtype=None):
        """
        Resample any EGR GCell map to uniform bin grid.
        
        Each bin's value = sum of overlapping GCells' values weighted by overlap ratio.
        
        Args:
            value_map: numpy array (num_gcells_x, num_gcells_y) or torch tensor
            device: torch device
            dtype: torch dtype
            
        Returns:
            torch tensor of shape (num_bins_x, num_bins_y)
        """
        if isinstance(value_map, torch.Tensor):
            value_map = value_map.cpu().numpy()
        
        result = np.zeros((self.num_bins_x, self.num_bins_y), dtype=np.float32)
        
        for (bx, by), overlaps in self.bin_to_gcell_overlaps.items():
            total_value = 0.0
            for gx, gy, overlap_ratio in overlaps:
                if gx < value_map.shape[0] and gy < value_map.shape[1]:
                    total_value += value_map[gx, gy] * overlap_ratio
            result[bx, by] = total_value
        
        result_tensor = torch.from_numpy(result)
        if device is not None:
            result_tensor = result_tensor.to(device)
        if dtype is not None:
            result_tensor = result_tensor.to(dtype)
        
        return result_tensor

    def resample_supply(self, supply_map, device=None, dtype=None):
        """
        Resample EGR supply map to uniform bin grid.
        
        Each bin's supply = sum of overlapping GCells' supply weighted by overlap ratio.
        
        Args:
            supply_map: numpy array (num_gcells_x, num_gcells_y) from load_egr_csv_map
            device: torch device
            dtype: torch dtype
            
        Returns:
            torch tensor of shape (num_bins_x, num_bins_y)
        """
        return self.resample_map(supply_map, device=device, dtype=dtype)
    
    def get_capacity_map(self, supply_map=None, device=None, dtype=None, normalize=True):
        """
        Get the capacity map as target_density for L-shape routability.
        
        Args:
            supply_map: EGR supply map, or None to use unit supply
            device: torch device
            dtype: torch dtype
            normalize: if True, normalize to [0, 1] range
            
        Returns:
            capacity_map: torch tensor (num_bins_x, num_bins_y)
        """
        if supply_map is None:
            # Use unit supply (area-based capacity)
            supply_map = np.ones(
                (self.gcell_info.num_gcells_x, self.gcell_info.num_gcells_y),
                dtype=np.float32
            )
        
        result = self.resample_map(supply_map, device=device, dtype=dtype)
        
        if normalize and result.max() > 0:
            result = result / result.max()
        
        logger.info(f"Capacity map: min={result.min():.3f}, max={result.max():.3f}, "
                   f"mean={result.mean():.3f}")
        
        return result


def create_supply_map_from_egr(egr_dir, placedb, params, num_bins_x, num_bins_y,
                                device=None, dtype=None, layer='planar', normalize=True,
                                return_wire_width=False):
    """
    Create uniform supply/capacity map from EGR output directory.
    
    Usage:
        supply_map = create_supply_map_from_egr(
            egr_dir, placedb, params, num_bins_x, num_bins_y
        )
        # Use as target_density in LShapeElectricPotential
        
        # Or get wire_width as well:
        supply_map, wire_width = create_supply_map_from_egr(
            egr_dir, placedb, params, num_bins_x, num_bins_y,
            return_wire_width=True
        )
    
    Args:
        egr_dir: path to EGR output directory (containing gcell.info and supply_map_*.csv)
        placedb: placement database
        params: DREAMPlace parameters (for scale_factor, shift_factor)
        num_bins_x, num_bins_y: number of uniform bins
        device: torch device
        dtype: torch dtype
        layer: 'planar' (all layers) or specific layer like 'MET1', 'MET2', etc.
        normalize: if True, normalize to [0, 1] range
        return_wire_width: if True, also return recommended wire_width based on GCell size
        
    Returns:
        supply_map: torch tensor (num_bins_x, num_bins_y)
        wire_width: (optional) min(avg_gcell_width, avg_gcell_height) in DREAMPlace coords
    """
    # Load GCell info
    gcell_info_path = os.path.join(egr_dir, 'gcell.info')
    gcell_info = EGRGCellInfo(gcell_info_path)
    
    # Load supply map
    supply_csv_path = os.path.join(egr_dir, f'supply_map_{layer}.csv')
    supply_map = load_egr_csv_map(supply_csv_path)
    
    # Create resampler
    resampler = EGRCapacityResampler(
        gcell_info,
        target_xl=placedb.xl,
        target_yl=placedb.yl,
        target_xh=placedb.xh,
        target_yh=placedb.yh,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        scale_factor=params.scale_factor,
        shift_factor=params.shift_factor
    )
    
    result = resampler.get_capacity_map(supply_map, device=device, dtype=dtype, normalize=normalize)
    
    if return_wire_width:
        # Get wire_width from min GCell dimension (more physically meaningful)
        wire_width = gcell_info.get_min_gcell_dimension(
            scale_factor=params.scale_factor,
            shift_factor=params.shift_factor
        )
        return result, wire_width
    
    return result


def create_supply_and_demand_maps_from_egr(egr_dir, placedb, params, num_bins_x, num_bins_y,
                                           device=None, dtype=None, layer='planar',
                                           normalize_supply=False, normalize_demand=False,
                                           return_wire_width=False):
    """
    Create uniform supply (capacity) map and demand (net) map from EGR output directory.
    
    Args:
        egr_dir: path to EGR output directory (containing gcell.info and *_map_*.csv)
        placedb: placement database
        params: DREAMPlace parameters (for scale_factor, shift_factor)
        num_bins_x, num_bins_y: number of uniform bins
        device: torch device
        dtype: torch dtype
        layer: 'planar' (all layers) or specific layer like 'MET1', 'MET2', etc.
        normalize_supply: if True, normalize supply map to [0, 1]
        normalize_demand: if True, normalize demand map to [0, 1]
        return_wire_width: if True, also return recommended wire_width based on GCell size
        
    Returns:
        supply_map: torch tensor (num_bins_x, num_bins_y)
        demand_map: torch tensor (num_bins_x, num_bins_y)
        wire_width: (optional) min(avg_gcell_width, avg_gcell_height) in DREAMPlace coords
    """
    # Load GCell info
    gcell_info_path = os.path.join(egr_dir, 'gcell.info')
    gcell_info = EGRGCellInfo(gcell_info_path)
    
    # Load supply and demand maps
    supply_csv_path = os.path.join(egr_dir, f'supply_map_{layer}.csv')
    demand_csv_path = os.path.join(egr_dir, f'net_map_{layer}.csv')
    supply_map = load_egr_csv_map(supply_csv_path)
    demand_map = load_egr_csv_map(demand_csv_path)
    
    # Create resampler
    resampler = EGRCapacityResampler(
        gcell_info,
        target_xl=placedb.xl,
        target_yl=placedb.yl,
        target_xh=placedb.xh,
        target_yh=placedb.yh,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        scale_factor=params.scale_factor,
        shift_factor=params.shift_factor
    )
    
    supply_resampled = resampler.resample_map(supply_map, device=device, dtype=dtype)
    demand_resampled = resampler.resample_map(demand_map, device=device, dtype=dtype)
    
    if normalize_supply and supply_resampled.max() > 0:
        supply_resampled = supply_resampled / supply_resampled.max()
    if normalize_demand and demand_resampled.max() > 0:
        demand_resampled = demand_resampled / demand_resampled.max()
    
    logger.info(f"Supply map: min={supply_resampled.min():.3f}, max={supply_resampled.max():.3f}, "
                f"mean={supply_resampled.mean():.3f}")
    logger.info(f"Demand map: min={demand_resampled.min():.3f}, max={demand_resampled.max():.3f}, "
                f"mean={demand_resampled.mean():.3f}")
    
    if return_wire_width:
        wire_width = gcell_info.get_min_gcell_dimension(
            scale_factor=params.scale_factor,
            shift_factor=params.shift_factor
        )
        return supply_resampled, demand_resampled, wire_width
    
    return supply_resampled, demand_resampled
