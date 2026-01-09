# @file   __init__.py
# @brief  Routability optimization module

from .l_shape_segment import (
    LShapeSegmentBuilder,
    LShapeSegmentOp,
    build_l_shape_segments_vectorized,
    build_segment_pos_tensor,
    H_FIRST, V_FIRST, STRAIGHT, FAKE_STRAIGHT, UNKNOWN
)

from .segment_density import (
    SegmentDensityOp,
    SegmentDensityFunction,
    compute_segment_rudy_density,
    compute_segment_overlap_density,
    create_segment_density_op
)

from .l_shape_electric_overflow import (
    LShapeElectricOverflow,
    SegmentDensityMapFunction,
    create_l_shape_electric_overflow
)

from .l_shape_electric_potential import (
    LShapeElectricPotential,
    SegmentElectricPotentialFunction,
    LShapeRoutabilityPotentialOp,
    create_l_shape_electric_potential
)

from .l_shape_routability import (
    LShapeRoutabilityOp,
    LShapeRoutabilityMixin,
    create_l_shape_routability_op,
    plot_l_shape_segments,
    plot_segment_density_map
)

__all__ = [
    # L-shape segment
    'LShapeSegmentBuilder',
    'LShapeSegmentOp',
    'build_l_shape_segments_vectorized',
    'build_segment_pos_tensor',
    'H_FIRST', 'V_FIRST', 'STRAIGHT', 'FAKE_STRAIGHT', 'UNKNOWN',
    
    # Segment density (Python RUDY)
    'SegmentDensityOp',
    'SegmentDensityFunction',
    'compute_segment_rudy_density',
    'compute_segment_overlap_density',
    'create_segment_density_op',
    
    # L-shape electric overflow (C++/CUDA)
    'LShapeElectricOverflow',
    'SegmentDensityMapFunction',
    'create_l_shape_electric_overflow',
    
    # L-shape electric potential (C++/CUDA)
    'LShapeElectricPotential',
    'SegmentElectricPotentialFunction',
    'LShapeRoutabilityPotentialOp',
    'create_l_shape_electric_potential',
    
    # L-shape routability
    'LShapeRoutabilityOp',
    'LShapeRoutabilityMixin',
    'create_l_shape_routability_op',
    'plot_l_shape_segments',
    'plot_segment_density_map',
]
