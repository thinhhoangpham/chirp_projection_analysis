"""
Cython-accelerated computational modules for CHIRP projection analysis.

This package provides high-performance Cython implementations of core
computational functions with automatic fallback to pure Python if Cython
modules are not available.

The package exports all functions from the optimized modules when available,
or falls back to the pure Python implementations from chirp_python.
"""

import sys
import warnings

# Flag to track if Cython is available
CYTHON_AVAILABLE = False

# Try to import Cython-compiled modules
try:
    # Import feature transform functions
    from chirp_cython.feature_transforms_cy import (
        apply_feature_transform_vectorized as _apply_feature_transform_vectorized_cy,
        set_epsilon as _set_epsilon_cy,
        get_epsilon as _get_epsilon_cy,
        AVAILABLE_TRANSFORMS as _AVAILABLE_TRANSFORMS_cy
    )
    
    # Import projection functions
    from chirp_cython.projection_vectorized_cy import (
        compute_bounds as _compute_bounds_cy,
        fill_array as _fill_array_cy,
        compute_projection_vectorized as _compute_projection_vectorized_cy,
        set_fuzz0 as _set_fuzz0_cy
    )
    
    # Import validation functions
    from chirp_cython.validation_cy import (
        validate_projection_bins as _validate_projection_bins_cy,
        validate_2d_projection_bins as _validate_2d_projection_bins_cy
    )
    
    CYTHON_AVAILABLE = True
    
    # Export Cython versions with wrappers for compatibility
    from chirp_python.data_source import DataSource
    from chirp_python.computation_cache import get_computation_cache
    from chirp_python.feature_transforms import EPSILON
    
    # Set FUZZ0 from DataSource
    _set_fuzz0_cy(DataSource.FUZZ0)
    _set_epsilon_cy(EPSILON)
    
    # Get global cache
    _computation_cache = get_computation_cache()
    
    def apply_feature_transform_vectorized(data, feature_idx, transform_type, cache=None):
        """Wrapper for Cython feature transform with caching support."""
        # Check cache first if available
        if cache is not None:
            cached_result = cache.get_transformed_features(data, feature_idx, transform_type)
            if cached_result is not None:
                return cached_result
        
        # Call Cython implementation
        result = _apply_feature_transform_vectorized_cy(data, feature_idx, transform_type)
        
        # Cache result if cache is available
        if cache is not None:
            cache.cache_transformed_features(data, feature_idx, transform_type, result)
        
        return result
    
    def compute_bounds(data_source, wi, transforms, n_pts):
        """Wrapper for Cython compute_bounds with caching."""
        import numpy as np
        
        # Get data hash from cache
        data_hash = _computation_cache.get_data_hash(data_source.data)
        
        # Check cache first
        cached_bounds = _computation_cache.get_projection_bounds(wi, transforms, data_hash)
        if cached_bounds is not None:
            return cached_bounds
        
        # Pre-compute all transformed features
        transformed_features = []
        for j in range(len(wi)):
            wij = abs(wi[j])
            transform_type = transforms[j]
            transformed_col = apply_feature_transform_vectorized(
                data_source.data, wij, transform_type, cache=_computation_cache
            )
            transformed_features.append(transformed_col)
        
        # Call Cython implementation with pre-transformed features
        wi_abs = np.array(wi, dtype=np.int32)
        bounds = _compute_bounds_cy(transformed_features, wi_abs, n_pts)
        
        # Cache the result
        _computation_cache.cache_projection_bounds(wi, transforms, data_hash, bounds)
        
        return bounds
    
    def fill_array(wi, transforms, bounds, data_source, n_pts):
        """Wrapper for Cython fill_array with caching."""
        import numpy as np
        
        # Get data hash from cache
        data_hash = _computation_cache.get_data_hash(data_source.data)
        
        # Check cache first
        cached_projection = _computation_cache.get_projection_array(wi, transforms, bounds, data_hash)
        if cached_projection is not None:
            return cached_projection
        
        # Validate bounds
        bounds_range = bounds[1] - bounds[0]
        if (not np.all(np.isfinite(bounds))) or (not np.isfinite(bounds_range)) or (bounds_range <= 0):
            print(f"Warning: Invalid bounds detected (bounds={bounds}, range={bounds_range}), returning zeros")
            return np.zeros(n_pts)
        
        # Pre-compute all transformed features
        transformed_features = []
        for j in range(len(wi)):
            wij = abs(wi[j])
            transform_type = transforms[j]
            transformed_col = apply_feature_transform_vectorized(
                data_source.data, wij, transform_type, cache=_computation_cache
            )
            transformed_features.append(transformed_col)
        
        # Call Cython implementation
        wi_abs = np.array(wi, dtype=np.int32)
        result = _fill_array_cy(transformed_features, wi_abs, bounds, n_pts)
        
        # Cache the result
        _computation_cache.cache_projection_array(wi, transforms, bounds, data_hash, result)
        
        return result
    
    def compute_projection_vectorized(data_source, wi, transforms, n_pts, normalize=True):
        """Wrapper for Cython compute_projection_vectorized with caching."""
        import numpy as np
        
        # Get data hash for cache key
        data_hash = _computation_cache.get_data_hash(data_source.data)
        
        # Create cache key
        wi_tuple = tuple(wi) if isinstance(wi, (list, np.ndarray)) else wi
        transforms_tuple = tuple(transforms)
        cache_key = (data_hash, wi_tuple, transforms_tuple, normalize, 'vectorized_cy')
        
        # Check cache
        if cache_key in _computation_cache.projection_cache:
            cached_result = _computation_cache.projection_cache[cache_key]
            _computation_cache.cache_hits += 1
            return cached_result
        
        _computation_cache.cache_misses += 1
        
        # Pre-compute all transformed features
        transformed_features = []
        for j in range(len(wi)):
            wij = abs(wi[j])
            transform_type = transforms[j]
            transformed_col = apply_feature_transform_vectorized(
                data_source.data, wij, transform_type, cache=_computation_cache
            )
            transformed_features.append(transformed_col)
        
        # Call Cython implementation
        wi_abs = np.array(wi, dtype=np.int32)
        result, bounds, valid_count = _compute_projection_vectorized_cy(
            transformed_features, wi_abs, n_pts
        )
        
        # Cache the result
        result_tuple = (result, bounds, valid_count)
        _computation_cache.projection_cache[cache_key] = result_tuple
        
        return result, bounds, valid_count
    
    def validate_projection_bins(proj_array, n_bins, min_occupancy_ratio=0.05):
        """Wrapper for Cython validate_projection_bins."""
        return _validate_projection_bins_cy(proj_array, n_bins, min_occupancy_ratio)
    
    def validate_2d_projection_bins(x_proj, y_proj, n_bins, min_occupancy_ratio=0.05):
        """Wrapper for Cython validate_2d_projection_bins."""
        return _validate_2d_projection_bins_cy(x_proj, y_proj, n_bins, min_occupancy_ratio)
    
    # Import non-optimized functions from Python
    from chirp_python.validation import validate_incremental_term
    from chirp_python.projection_vectorized import IncrementalProjection
    
    set_epsilon = _set_epsilon_cy
    get_epsilon = _get_epsilon_cy
    AVAILABLE_TRANSFORMS = _AVAILABLE_TRANSFORMS_cy
    
    print("✓ Cython-accelerated modules loaded successfully")

except ImportError as e:
    # Fallback to pure Python implementations
    CYTHON_AVAILABLE = False
    
    warnings.warn(
        f"Cython modules not available, falling back to pure Python implementation. "
        f"For better performance, build Cython extensions with: python setup.py build_ext --inplace\n"
        f"Import error: {e}",
        ImportWarning
    )
    
    # Import pure Python implementations
    from chirp_python.feature_transforms import (
        apply_feature_transform_vectorized,
        set_epsilon,
        get_epsilon,
        AVAILABLE_TRANSFORMS
    )
    from chirp_python.projection_vectorized import (
        compute_bounds,
        fill_array,
        compute_projection_vectorized,
        IncrementalProjection
    )
    from chirp_python.validation import (
        validate_projection_bins,
        validate_2d_projection_bins,
        validate_incremental_term
    )

# Export all functions
__all__ = [
    'CYTHON_AVAILABLE',
    'apply_feature_transform_vectorized',
    'set_epsilon',
    'get_epsilon',
    'AVAILABLE_TRANSFORMS',
    'compute_bounds',
    'fill_array',
    'compute_projection_vectorized',
    'IncrementalProjection',
    'validate_projection_bins',
    'validate_2d_projection_bins',
    'validate_incremental_term'
]
