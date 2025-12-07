# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-accelerated projection computation functions for CHIRP projection analysis.

This module provides optimized implementations of projection bounds computation
and array filling using typed memory views and C-level operations.

Expected speedup: 3-5x over pure Python NumPy version
"""

import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf, isfinite
cimport cython

# Initialize NumPy C API
np.import_array()

def compute_bounds_cy(data_source, wi, transforms, n_pts):
    """Compute bounds for projection with transformations (Cython-accelerated).
    
    Args:
        data_source: DataSource object containing the data
        wi: Weight array (feature indices with signs)
        transforms: List of transformation types for each weight
        n_pts: Number of data points
        
    Returns:
        Array [min_bound, max_bound]
    """
    from chirp_python.computation_cache import get_computation_cache
    from chirp_cython import apply_transform_vectorized_cy
    
    # Get computation cache
    _computation_cache = get_computation_cache()
    
    # Get data hash from cache (O(1) instead of O(N))
    data_hash = _computation_cache.get_data_hash(data_source.data)
    
    # Check cache first
    cached_bounds = _computation_cache.get_projection_bounds(wi, transforms, data_hash)
    if cached_bounds is not None:
        return cached_bounds
    
    # Declare C variables first
    cdef int i, j, nwt
    cdef double data_val, wt, xi_transformed
    cdef double min_val = 1.0 / 0.0  # Inf
    cdef double max_val = -1.0 / 0.0  # -Inf
    cdef int n_terms = len(wi)
    cdef double bounds_range, center
    
    # Pre-compute all transformed features for this projection
    transformed_features = []
    for j in range(len(wi)):
        wij = abs(wi[j])
        transform_type = transforms[j]
        transformed_col = apply_transform_vectorized_cy(
            data_source.data, wij, transform_type, cache=_computation_cache
        )
        transformed_features.append(transformed_col)
    
    # Convert to numpy arrays for efficient access
    cdef np.ndarray[np.float64_t, ndim=2] transform_matrix = np.stack(
        transformed_features, axis=0
    ).astype(np.float64)
    
    # Compute projection values for all points
    for i in range(n_pts):
        data_val = 0.0
        nwt = 0
        
        for j in range(n_terms):
            wt = 1.0 if wi[j] >= 0 else -1.0
            xi_transformed = transform_matrix[j, i]
            
            if not isnan(xi_transformed):
                data_val += wt * xi_transformed
                nwt += 1
        
        if nwt > 0:
            if nwt < n_terms:
                data_val = data_val * n_terms / nwt
            
            # Update bounds
            if data_val < min_val:
                min_val = data_val
            if data_val > max_val:
                max_val = data_val
    
    # Create bounds array
    cdef np.ndarray[np.float64_t, ndim=1] bounds = np.empty(2, dtype=np.float64)
    
    # Check if bounds are valid
    if isfinite(min_val) and isfinite(max_val):
        bounds_range = max_val - min_val
        if bounds_range > 0:
            # Use FUZZ0 from DataSource
            bounds[0] = min_val - data_source.FUZZ0 * bounds_range
            bounds[1] = max_val + data_source.FUZZ0 * bounds_range
        else:
            # All values are identical, create a small range
            center = min_val
            bounds[0] = center - data_source.FUZZ0
            bounds[1] = center + data_source.FUZZ0
    else:
        bounds[0] = 1.0 / 0.0  # Inf
        bounds[1] = -1.0 / 0.0  # -Inf
    
    # Cache the result
    _computation_cache.cache_projection_bounds(wi, transforms, data_hash, bounds)
    
    return bounds

def fill_array_cy(wi, transforms, bounds, data_source, n_pts):
    """Fill projection array with transformations (Cython-accelerated).
    
    Args:
        wi: Weight array (feature indices with signs)
        transforms: List of transformation types for each weight
        bounds: Array [min_bound, max_bound]
        data_source: DataSource object containing the data
        n_pts: Number of data points
        
    Returns:
        1D array of normalized projection values [0, 1]
    """
    from chirp_python.computation_cache import get_computation_cache
    from chirp_cython import apply_transform_vectorized_cy
    
    # Get computation cache
    _computation_cache = get_computation_cache()
    
    # Get data hash from cache (O(1) instead of O(N))
    data_hash = _computation_cache.get_data_hash(data_source.data)
    
    # Check cache first
    cached_projection = _computation_cache.get_projection_array(wi, transforms, bounds, data_hash)
    if cached_projection is not None:
        return cached_projection
    
    # Validate bounds to prevent division by zero
    cdef double bounds_range = bounds[1] - bounds[0]
    if (not isfinite(bounds[0])) or (not isfinite(bounds[1])) or (bounds_range <= 0):
        print(f"Warning: Invalid bounds detected (bounds={bounds}, range={bounds_range}), returning zeros")
        return np.zeros(n_pts)
    
    # Declare C variables first
    cdef int i, j, nwt
    cdef double projection_val, wt, xi_transformed
    cdef int n_terms = len(wi)
    cdef int total_invalid_points = 0
    cdef int partial_invalid_points = 0
    cdef int valid_points = 0
    
    # Pre-compute all transformed features for this projection
    transformed_features = []
    for j in range(len(wi)):
        wij = abs(wi[j])
        transform_type = transforms[j]
        transformed_col = apply_transform_vectorized_cy(
            data_source.data, wij, transform_type
        )
        transformed_features.append(transformed_col)
    
    # Convert to numpy arrays
    cdef np.ndarray[np.float64_t, ndim=2] transform_matrix = np.stack(
        transformed_features, axis=0
    ).astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n_pts, dtype=np.float64)
    
    # Compute projections
    for i in range(n_pts):
        projection_val = 0.0
        nwt = 0
        
        for j in range(n_terms):
            wt = 1.0 if wi[j] >= 0 else -1.0
            xi_transformed = transform_matrix[j, i]
            
            if not isnan(xi_transformed):
                projection_val += wt * xi_transformed
                nwt += 1
        
        if nwt > 0:
            if nwt < n_terms:
                partial_invalid_points += 1
                projection_val = projection_val * n_terms / nwt
            else:
                valid_points += 1
            # Normalize to [0, 1] based on bounds
            result[i] = (projection_val - bounds[0]) / bounds_range
        else:
            total_invalid_points += 1
    
    # Log projection statistics
    if total_invalid_points > 0 or partial_invalid_points > 0:
        total_invalid = total_invalid_points + partial_invalid_points
        invalid_percentage = (total_invalid / n_pts) * 100
        print(f"  [PROJECTION] Invalid points: {total_invalid}/{n_pts} ({invalid_percentage:.1f}%) - "
              f"Fully invalid: {total_invalid_points}, Partially invalid: {partial_invalid_points}")
    
    # Cache the result
    _computation_cache.cache_projection_array(wi, transforms, bounds, data_hash, result)
    
    return result
