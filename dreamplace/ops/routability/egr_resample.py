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
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


def _build_uniform_gcell_info(xl, yl, xh, yh, num_x, num_y):
    """Build a uniform-grid GCell info (in DREAMPlace coordinates)."""
    gcell_info = EGRGCellInfo()
    gcell_info.gcell_info = {}
    gcell_info.num_gcells_x = num_x
    gcell_info.num_gcells_y = num_y
    gcell_info.xl = xl
    gcell_info.yl = yl
    gcell_info.xh = xh
    gcell_info.yh = yh

    size_x = (xh - xl) / num_x
    size_y = (yh - yl) / num_y
    for gx in range(num_x):
        llx = xl + gx * size_x
        urx = llx + size_x
        for gy in range(num_y):
            lly = yl + gy * size_y
            ury = lly + size_y
            gcell_info.gcell_info[(gx, gy)] = (llx, lly, urx, ury)
    return gcell_info


def create_supply_map_from_placedb(placedb, params, num_bins_x, num_bins_y,
                                   device=None, dtype=None,
                                   wire_width=None, as_area=False,
                                   return_directional=False):
    """
    Create supply map from placedb routing grid.
    If as_area=True, convert track capacity to area capacity using wire_width.
    """

    bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
    bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

    cap_h = placedb.unit_horizontal_capacity * bin_size_y
    cap_v = placedb.unit_vertical_capacity * bin_size_x
    logger.info(
        f"placedb scale: dbu={getattr(placedb,'dbu',None)} "
        f"scale_factor={getattr(placedb,'scale_factor',None)}"
    )
    logger.info(
        f"unit_cap: H={placedb.unit_horizontal_capacity:.6g}, "
        f"V={placedb.unit_vertical_capacity:.6g}, "
        f"bin_size=({bin_size_x:.6g},{bin_size_y:.6g}), "
        f"wire_width={wire_width}"
    )
    if placedb.unit_horizontal_capacity > 0 and placedb.unit_vertical_capacity > 0:
        logger.info(
            f"pitch_est: H={1.0/placedb.unit_horizontal_capacity:.3f}, "
            f"V={1.0/placedb.unit_vertical_capacity:.3f}"
        )

    supply_h = np.full((num_bins_x, num_bins_y), cap_h, dtype=np.float32)
    supply_v = np.full((num_bins_x, num_bins_y), cap_v, dtype=np.float32)

    supply_h = np.maximum(supply_h, 0.0)
    supply_v = np.maximum(supply_v, 0.0)

    if as_area:
        if wire_width is None or wire_width <= 0:
            logger.warning("wire_width invalid; fallback to track-units supply map")
            supply_h_out = supply_h
            supply_v_out = supply_v
        else:
            # Convert track capacity to area capacity per direction
            supply_h_out = supply_h * bin_size_x * wire_width
            supply_v_out = supply_v * bin_size_y * wire_width
    else:
        supply_h_out = supply_h
        supply_v_out = supply_v

    supply = supply_h_out + supply_v_out

    supply_tensor = torch.from_numpy(supply)
    supply_h_tensor = torch.from_numpy(supply_h_out)
    supply_v_tensor = torch.from_numpy(supply_v_out)
    if device is not None:
        supply_tensor = supply_tensor.to(device)
        supply_h_tensor = supply_h_tensor.to(device)
        supply_v_tensor = supply_v_tensor.to(device)
    if dtype is not None:
        supply_tensor = supply_tensor.to(dtype)
        supply_h_tensor = supply_h_tensor.to(dtype)
        supply_v_tensor = supply_v_tensor.to(dtype)

    unit = "area" if as_area else "tracks"
    logger.info(f"Placedb supply map ({unit}): min={supply_tensor.min():.3f}, max={supply_tensor.max():.3f}, "
                f"mean={supply_tensor.mean():.3f}")
    if return_directional:
        return supply_tensor, supply_h_tensor, supply_v_tensor
    return supply_tensor


def _lef_value_to_micron(value, dbu_per_micron):
    if dbu_per_micron and value > dbu_per_micron:
        return value / dbu_per_micron
    return value


def _parse_lef_routing_layers(lef_path):
    """
    Parse LEF routing layers for pitch/width/direction.
    Returns: (layers, dbu_per_micron)
    """
    layers = []
    dbu_per_micron = None
    in_layer = False
    layer_name = None
    is_routing = False
    direction = None
    pitch_x = None
    pitch_y = None
    width = None

    def flush_layer():
        if in_layer and is_routing:
            layers.append(
                {
                    "name": layer_name,
                    "direction": direction,
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                    "width": width,
                }
            )

    with open(lef_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if "DATABASE MICRONS" in upper:
                m = re.search(r"DATABASE\s+MICRONS\s+([0-9.]+)", upper)
                if m:
                    try:
                        dbu_per_micron = float(m.group(1))
                    except ValueError:
                        pass
            if upper.startswith("LAYER "):
                flush_layer()
                parts = line.split()
                layer_name = parts[1] if len(parts) > 1 else None
                in_layer = True
                is_routing = False
                direction = None
                pitch_x = None
                pitch_y = None
                width = None
                continue
            if in_layer and upper.startswith("END"):
                flush_layer()
                in_layer = False
                layer_name = None
                continue
            if not in_layer:
                continue
            if upper.startswith("TYPE") and "ROUTING" in upper:
                is_routing = True
                continue
            if upper.startswith("DIRECTION"):
                parts = upper.split()
                if len(parts) >= 2:
                    direction = parts[1]
                continue
            if upper.startswith("PITCH"):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if nums:
                    pitch_x = float(nums[0])
                    if len(nums) > 1:
                        pitch_y = float(nums[1])
                continue
            if upper.startswith("WIDTH"):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if nums:
                    width = float(nums[0])
                continue

    flush_layer()
    return layers, dbu_per_micron


def create_supply_map_from_gcellinfo_and_lef(
    egr_dir,
    lef_path,
    placedb,
    params,
    num_bins_x,
    num_bins_y,
    device=None,
    dtype=None,
    normalize=False,
    wire_width=None,
    return_wire_width=False,
):
    """
    Create supply map from EGR gcell.info and LEF routing pitch/width (no EGR supply CSV).
    Supply is computed per GCell and resampled to place bins.
    """
    gcell_info_path = os.path.join(egr_dir, "gcell.info")
    gcell_info = EGRGCellInfo(gcell_info_path)

    layers, dbu_per_micron = _parse_lef_routing_layers(lef_path)
    if not layers:
        raise RuntimeError(f"No routing layers parsed from LEF: {lef_path}")

    h_pitches = []
    v_pitches = []
    min_width = None
    for layer in layers:
        px = layer.get("pitch_x")
        py = layer.get("pitch_y")
        if px is None and py is None:
            continue
        if px is None:
            px = py
        if py is None:
            py = px
        px = _lef_value_to_micron(px, dbu_per_micron)
        py = _lef_value_to_micron(py, dbu_per_micron)
        direction = layer.get("direction")
        if direction == "HORIZONTAL":
            h_pitches.append(py)
        elif direction == "VERTICAL":
            v_pitches.append(px)
        else:
            h_pitches.append(py)
            v_pitches.append(px)
        w = layer.get("width")
        if w is not None:
            w = _lef_value_to_micron(w, dbu_per_micron)
            if min_width is None or w < min_width:
                min_width = w

    if not h_pitches or not v_pitches:
        # Fallback: use all pitches for both directions
        all_pitches = h_pitches + v_pitches
        if not all_pitches:
            raise RuntimeError("No pitch parsed from LEF routing layers")
        h_pitches = all_pitches
        v_pitches = all_pitches

    micron_to_nm = 1000.0
    h_pitches_dp = [(p * micron_to_nm) * params.scale_factor for p in h_pitches]
    v_pitches_dp = [(p * micron_to_nm) * params.scale_factor for p in v_pitches]

    if wire_width is None:
        if min_width is None:
            raise RuntimeError("No wire width parsed from LEF")
        wire_width = (min_width * micron_to_nm) * params.scale_factor

    sum_inv_pitch_h = sum(1.0 / p for p in h_pitches_dp if p > 0)
    sum_inv_pitch_v = sum(1.0 / p for p in v_pitches_dp if p > 0)

    logger.info(
        f"LEF pitch (dp): H_mean={np.mean(h_pitches_dp):.3f}, V_mean={np.mean(v_pitches_dp):.3f}, "
        f"wire_width={wire_width:.3f}"
    )

    supply_map = np.zeros(
        (gcell_info.num_gcells_x, gcell_info.num_gcells_y), dtype=np.float32
    )
    for (gx, gy), (llx, lly, urx, ury) in gcell_info.gcell_info.items():
        w = (urx - llx) * params.scale_factor
        h = (ury - lly) * params.scale_factor
        if w <= 0 or h <= 0:
            continue
        tracks_h = h * sum_inv_pitch_h
        tracks_v = w * sum_inv_pitch_v
        supply_area = tracks_h * w * wire_width + tracks_v * h * wire_width
        supply_map[gx, gy] = supply_area

    resampler = EGRCapacityResampler(
        gcell_info,
        target_xl=placedb.xl,
        target_yl=placedb.yl,
        target_xh=placedb.xh,
        target_yh=placedb.yh,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        scale_factor=params.scale_factor,
        shift_factor=params.shift_factor,
    )
    supply_resampled = resampler.resample_map(supply_map, device=device, dtype=dtype)
    if normalize and supply_resampled.max() > 0:
        supply_resampled = supply_resampled / supply_resampled.max()

    logger.info(
        f"Supply map (gcell+lef): min={supply_resampled.min():.3f}, "
        f"max={supply_resampled.max():.3f}, mean={supply_resampled.mean():.3f}"
    )

    if return_wire_width:
        return supply_resampled, wire_width
    return supply_resampled


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
        if not self.gcell_info:
            return 0.0

        min_width = None
        min_height = None
        for (gx, gy), (llx, lly, urx, ury) in self.gcell_info.items():
            w = urx - llx
            h = ury - lly
            if min_width is None or w < min_width:
                min_width = w
            if min_height is None or h < min_height:
                min_height = h

        # Convert from nm to DREAMPlace coordinates
        dp_width = min_width * scale_factor
        dp_height = min_height * scale_factor
        min_dim = min(dp_width, dp_height)
        logger.info(f"GCell min size: {min_width:.1f} x {min_height:.1f} nm, "
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


def _normalize_lef_input(lef_input):
    if isinstance(lef_input, (list, tuple)):
        return [str(item) for item in lef_input if item]
    if lef_input:
        return [str(lef_input)]
    return []


def _collect_routing_layers_from_lefs(lef_paths):
    layer_by_name = {}
    order = []
    for lef_path in lef_paths:
        if not lef_path or not os.path.exists(lef_path):
            continue
        try:
            layers, _ = _parse_lef_routing_layers(lef_path)
        except Exception as exc:
            logger.warning("Failed to parse LEF routing layers from %s: %s", lef_path, exc)
            continue
        for layer in layers:
            name = layer.get("name")
            if not name:
                continue
            if name not in layer_by_name:
                order.append(name)
            layer_by_name[name] = layer
    return [layer_by_name[name] for name in order]


def create_directional_supply_and_demand_maps_from_egr(
    egr_dir,
    placedb,
    params,
    num_bins_x,
    num_bins_y,
    device=None,
    dtype=None,
    normalize_supply=False,
    normalize_demand=False,
    return_wire_width=False,
):
    """
    Create directional H/V supply and demand maps from EGR per-layer CSVs.

    Returns:
        supply_h, supply_v, demand_h, demand_v [, wire_width]
    """
    gcell_info_path = os.path.join(egr_dir, "gcell.info")
    gcell_info = EGRGCellInfo(gcell_info_path)
    resampler = EGRCapacityResampler(
        gcell_info,
        target_xl=placedb.xl,
        target_yl=placedb.yl,
        target_xh=placedb.xh,
        target_yh=placedb.yh,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        scale_factor=params.scale_factor,
        shift_factor=params.shift_factor,
    )

    lef_paths = _normalize_lef_input(getattr(params, "lef_input", None))
    routing_layers = _collect_routing_layers_from_lefs(lef_paths)

    shape = (gcell_info.num_gcells_x, gcell_info.num_gcells_y)
    supply_h_raw = np.zeros(shape, dtype=np.float32)
    supply_v_raw = np.zeros(shape, dtype=np.float32)
    demand_h_raw = np.zeros(shape, dtype=np.float32)
    demand_v_raw = np.zeros(shape, dtype=np.float32)
    count_h = 0
    count_v = 0

    for layer in routing_layers:
        name = layer.get("name")
        direction = str(layer.get("direction", "")).upper()
        if direction not in ("HORIZONTAL", "VERTICAL"):
            continue

        supply_csv_path = os.path.join(egr_dir, f"supply_map_{name}.csv")
        demand_csv_path = os.path.join(egr_dir, f"net_map_{name}.csv")
        if not (os.path.exists(supply_csv_path) and os.path.exists(demand_csv_path)):
            continue

        supply_layer = load_egr_csv_map(supply_csv_path)
        demand_layer = load_egr_csv_map(demand_csv_path)
        if direction == "HORIZONTAL":
            supply_h_raw += supply_layer
            demand_h_raw += demand_layer
            count_h += 1
        else:
            supply_v_raw += supply_layer
            demand_v_raw += demand_layer
            count_v += 1

    if count_h == 0 or count_v == 0:
        logger.warning(
            "Directional EGR maps incomplete (loaded H=%d, V=%d layers). Falling back to planar maps for missing directions.",
            count_h,
            count_v,
        )
        planar_supply, planar_demand, wire_width = create_supply_and_demand_maps_from_egr(
            egr_dir=egr_dir,
            placedb=placedb,
            params=params,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            device=device,
            dtype=dtype,
            layer="planar",
            normalize_supply=normalize_supply,
            normalize_demand=normalize_demand,
            return_wire_width=True,
        )
        if count_h == 0:
            supply_h = planar_supply
            demand_h = planar_demand
        else:
            supply_h = resampler.resample_map(supply_h_raw, device=device, dtype=dtype)
            demand_h = resampler.resample_map(demand_h_raw, device=device, dtype=dtype)
        if count_v == 0:
            supply_v = planar_supply
            demand_v = planar_demand
        else:
            supply_v = resampler.resample_map(supply_v_raw, device=device, dtype=dtype)
            demand_v = resampler.resample_map(demand_v_raw, device=device, dtype=dtype)
    else:
        supply_h = resampler.resample_map(supply_h_raw, device=device, dtype=dtype)
        supply_v = resampler.resample_map(supply_v_raw, device=device, dtype=dtype)
        demand_h = resampler.resample_map(demand_h_raw, device=device, dtype=dtype)
        demand_v = resampler.resample_map(demand_v_raw, device=device, dtype=dtype)
        wire_width = gcell_info.get_min_gcell_dimension(
            scale_factor=params.scale_factor,
            shift_factor=params.shift_factor,
        )

    if normalize_supply:
        if supply_h.max() > 0:
            supply_h = supply_h / supply_h.max()
        if supply_v.max() > 0:
            supply_v = supply_v / supply_v.max()
    if normalize_demand:
        if demand_h.max() > 0:
            demand_h = demand_h / demand_h.max()
        if demand_v.max() > 0:
            demand_v = demand_v / demand_v.max()

    logger.info(
        "Directional EGR maps: H layers=%d V layers=%d | supply_h[%.3f, %.3f] supply_v[%.3f, %.3f] "
        "demand_h[%.3f, %.3f] demand_v[%.3f, %.3f]",
        count_h,
        count_v,
        supply_h.min().item(),
        supply_h.max().item(),
        supply_v.min().item(),
        supply_v.max().item(),
        demand_h.min().item(),
        demand_h.max().item(),
        demand_v.min().item(),
        demand_v.max().item(),
    )

    if return_wire_width:
        return supply_h, supply_v, demand_h, demand_v, wire_width
    return supply_h, supply_v, demand_h, demand_v
