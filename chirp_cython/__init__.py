"""
Cython-accelerated CHIRP projection analysis modules.

This package provides Cython implementations of performance-critical functions
with automatic fallback to pure Python if Cython extensions are not available.

Usage:
    from chirp_cython import compute_bounds_cy, fill_array_cy
    
    # Check if Cython is available
    from chirp_cython import CYTHON_AVAILABLE
    if CYTHON_AVAILABLE:
        print("Using Cython-accelerated functions")
    else:
        print("Using pure Python fallback")
"""

# Try to import Cython versions
CYTHON_AVAILABLE = False

try:
    # Try importing Cython modules
    from chirp_cython.feature_transforms_cy import (
        apply_transform_vectorized_cy,
        set_epsilon_cy,
        get_epsilon_cy,
    )
    from chirp_cython.projection_vectorized_cy import (
        compute_bounds_cy,
        fill_array_cy,
    )
    from chirp_cython.validation_cy import (
        validate_projection_bins_cy,
        validate_2d_projection_bins_cy,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    # Fall back to Python versions
    from chirp_python.feature_transforms import (
        apply_feature_transform_vectorized as apply_transform_vectorized_cy,
        set_epsilon as set_epsilon_cy,
        get_epsilon as get_epsilon_cy,
    )
    from chirp_python.projection_vectorized import (
        compute_bounds as compute_bounds_cy,
        fill_array as fill_array_cy,
    )
    from chirp_python.validation import (
        validate_projection_bins as validate_projection_bins_cy,
        validate_2d_projection_bins as validate_2d_projection_bins_cy,
    )

# Export functions
__all__ = [
    'CYTHON_AVAILABLE',
    'apply_transform_vectorized_cy',
    'set_epsilon_cy',
    'get_epsilon_cy',
    'compute_bounds_cy',
    'fill_array_cy',
    'validate_projection_bins_cy',
    'validate_2d_projection_bins_cy',
]
