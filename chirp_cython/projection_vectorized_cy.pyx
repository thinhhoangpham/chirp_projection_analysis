# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized projection computation utilities for CHIRP projection analysis.

This module provides high-performance implementations of projection bounds
computation and array filling using Cython's static typing.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan, isinf, isfinite

# Forward declare FUZZ0 - will be set from Python
cdef double FUZZ0 = 0.01

def set_fuzz0(double value):
    """Set the FUZZ0 value from DataSource.FUZZ0"""
    global FUZZ0
    FUZZ0 = value

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_bounds(list transformed_features, int[:] wi_abs, int n_pts):
    """Compute bounds for projection with pre-transformed features.
    
    Args:
        transformed_features: List of pre-transformed feature arrays
        wi_abs: Absolute values of weights (for sign extraction)
        n_pts: Number of points
        
    Returns:
        Array of [min_bound, max_bound]
    """
    cdef int n_terms = len(transformed_features)
    cdef np.ndarray[double, ndim=1] bounds_arr = np.array([float('inf'), float('-inf')])
    cdef double[:] bounds = bounds_arr
    cdef int i, j
    cdef double data_val, wt, xi_transformed
    cdef int nwt
    cdef double min_val = float('inf')
    cdef double max_val = float('-inf')
    cdef bint has_valid = False
    cdef double bounds_range, center
    
    # Process each point
    for i in range(n_pts):
        data_val = 0.0
        nwt = 0
        
        # Sum contributions from all terms
        for j in range(n_terms):
            wt = 1.0 if wi_abs[j] >= 0 else -1.0
            xi_transformed = transformed_features[j][i]
            
            if not isnan(xi_transformed):
                data_val += wt * xi_transformed
                nwt += 1
        
        # If at least one term contributed, update bounds
        if nwt > 0:
            if nwt < n_terms:
                data_val = data_val * n_terms / nwt
            
            if not has_valid:
                min_val = data_val
                max_val = data_val
                has_valid = True
            else:
                if data_val < min_val:
                    min_val = data_val
                if data_val > max_val:
                    max_val = data_val
    
    # Set bounds
    if has_valid:
        bounds_range = max_val - min_val
        if bounds_range > 0:
            bounds[0] = min_val - FUZZ0 * bounds_range
            bounds[1] = max_val + FUZZ0 * bounds_range
        else:
            # All values identical, create small range
            bounds[0] = min_val - FUZZ0
            bounds[1] = min_val + FUZZ0
    
    return bounds_arr

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_array(list transformed_features, int[:] wi_abs, double[:] bounds, int n_pts):
    """Fill projection array with pre-transformed features.
    
    Args:
        transformed_features: List of pre-transformed feature arrays
        wi_abs: Absolute values of weights (for sign extraction)
        bounds: [min_bound, max_bound]
        n_pts: Number of points
        
    Returns:
        Normalized projection array [0, 1]
    """
    cdef int n_terms = len(transformed_features)
    cdef np.ndarray[double, ndim=1] result_arr = np.zeros(n_pts, dtype=np.float64)
    cdef double[:] result = result_arr
    cdef int i, j
    cdef double projection_val, wt, xi_transformed
    cdef int nwt
    cdef double bounds_range = bounds[1] - bounds[0]
    
    # Validate bounds to prevent division by zero
    if not (isfinite(bounds[0]) and isfinite(bounds[1])) or bounds_range <= 0:
        # Invalid bounds, return zeros
        return result_arr
    
    # Process each point
    for i in range(n_pts):
        projection_val = 0.0
        nwt = 0
        
        # Sum contributions from all terms
        for j in range(n_terms):
            wt = 1.0 if wi_abs[j] >= 0 else -1.0
            xi_transformed = transformed_features[j][i]
            
            if not isnan(xi_transformed):
                projection_val += wt * xi_transformed
                nwt += 1
        
        # Normalize if at least one term contributed
        if nwt > 0:
            if nwt < n_terms:
                projection_val = projection_val * n_terms / nwt
            # Normalize to [0, 1]
            result[i] = (projection_val - bounds[0]) / bounds_range
    
    return result_arr

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_projection_vectorized(list transformed_features, int[:] wi_abs, int n_pts):
    """Compute projection with bounds in a single pass.
    
    This combines bounds computation and array filling for efficiency.
    
    Args:
        transformed_features: List of pre-transformed feature arrays
        wi_abs: Absolute values of weights (for sign extraction)
        n_pts: Number of points
        
    Returns:
        Tuple of (projection_array, bounds, valid_count)
    """
    cdef int n_terms = len(transformed_features)
    cdef np.ndarray[double, ndim=1] projection_values = np.zeros(n_pts, dtype=np.float64)
    cdef double[:] proj_view = projection_values
    cdef np.ndarray[int, ndim=1] weight_counts = np.zeros(n_pts, dtype=np.int32)
    cdef int[:] count_view = weight_counts
    cdef int i, j
    cdef double data_val, wt, xi_transformed
    cdef int nwt
    cdef double min_val = float('inf')
    cdef double max_val = float('-inf')
    cdef int valid_count = 0
    cdef bint has_valid = False
    cdef double bounds_range
    cdef np.ndarray[double, ndim=1] bounds_arr = np.array([float('inf'), float('-inf')])
    cdef double[:] bounds = bounds_arr
    cdef np.ndarray[double, ndim=1] result_arr
    cdef double[:] result
    
    # First pass: compute raw projection values
    for i in range(n_pts):
        data_val = 0.0
        nwt = 0
        
        for j in range(n_terms):
            wt = 1.0 if wi_abs[j] >= 0 else -1.0
            xi_transformed = transformed_features[j][i]
            
            if not isnan(xi_transformed):
                data_val += wt * xi_transformed
                nwt += 1
        
        if nwt > 0:
            if nwt < n_terms:
                data_val = data_val * n_terms / nwt
            proj_view[i] = data_val
            count_view[i] = nwt
            
            # Update bounds
            if not has_valid:
                min_val = data_val
                max_val = data_val
                has_valid = True
            else:
                if data_val < min_val:
                    min_val = data_val
                if data_val > max_val:
                    max_val = data_val
            
            valid_count += 1
    
    # Compute bounds
    if has_valid:
        bounds_range = max_val - min_val
        if bounds_range > 0:
            bounds[0] = min_val - FUZZ0 * bounds_range
            bounds[1] = max_val + FUZZ0 * bounds_range
        else:
            bounds[0] = min_val - FUZZ0
            bounds[1] = min_val + FUZZ0
    
    # Normalize to [0, 1]
    bounds_range = bounds[1] - bounds[0]
    result_arr = np.zeros(n_pts, dtype=np.float64)
    result = result_arr
    
    if bounds_range > 0:
        for i in range(n_pts):
            if count_view[i] > 0:
                result[i] = (proj_view[i] - bounds[0]) / bounds_range
    
    return result_arr, bounds_arr, valid_count
