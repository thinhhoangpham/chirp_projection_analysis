# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-accelerated validation functions for CHIRP projection analysis.

This module provides optimized implementations of bin occupancy validation
using typed memory views and efficient counting algorithms.

Expected speedup: 5-8x over pure Python NumPy version
"""

import numpy as np
cimport numpy as np
from libc.math cimport floor
cimport cython

# Initialize NumPy C API
np.import_array()

def validate_projection_bins_cy(
    np.ndarray[np.float64_t, ndim=1] proj_array,
    int n_bins,
    double min_occupancy_ratio = 0.05
):
    """Validate that a projection has sufficient bin occupancy (Cython-accelerated).
    
    Args:
        proj_array: Normalized projection values in [0, 1]
        n_bins: Number of bins
        min_occupancy_ratio: Minimum ratio of bins that must be occupied (default 0.05 = 5%)
    
    Returns:
        Tuple of (is_valid, occupied_bins_count)
    """
    cdef int n_pts = proj_array.shape[0]
    cdef int i, bin_idx
    cdef np.ndarray[np.int32_t, ndim=1] bin_indices = np.empty(n_pts, dtype=np.int32)
    
    # Convert normalized values to bin indices
    for i in range(n_pts):
        bin_idx = <int>(proj_array[i] * n_bins)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bin_indices[i] = bin_idx
    
    # Count unique occupied bins using NumPy's unique
    cdef int occupied_bins = len(np.unique(bin_indices))
    cdef int min_required_bins = <int>(n_bins * min_occupancy_ratio)
    
    cdef bint is_valid = occupied_bins >= min_required_bins
    return is_valid, occupied_bins

def validate_2d_projection_bins_cy(
    np.ndarray[np.float64_t, ndim=1] x_proj,
    np.ndarray[np.float64_t, ndim=1] y_proj,
    int n_bins,
    double min_occupancy_ratio = 0.05
):
    """Validate that a 2D projection has sufficient bin occupancy (Cython-accelerated).
    
    Uses linear indexing for efficient 2D bin counting.
    
    Args:
        x_proj: Normalized X projection values in [0, 1]
        y_proj: Normalized Y projection values in [0, 1]
        n_bins: Number of bins per dimension (creates n_bins x n_bins grid)
        min_occupancy_ratio: Minimum ratio of 2D bins that must be occupied (default 0.05 = 5%)
    
    Returns:
        Tuple of (is_valid, occupied_2d_bins_count)
    """
    cdef int n_pts = x_proj.shape[0]
    cdef int i, x_bin, y_bin, linear_idx
    cdef np.ndarray[np.int32_t, ndim=1] linear_indices = np.empty(n_pts, dtype=np.int32)
    
    # Convert normalized values to bin indices and compute linear index
    for i in range(n_pts):
        # X bin index
        x_bin = <int>(x_proj[i] * n_bins)
        if x_bin < 0:
            x_bin = 0
        elif x_bin >= n_bins:
            x_bin = n_bins - 1
        
        # Y bin index
        y_bin = <int>(y_proj[i] * n_bins)
        if y_bin < 0:
            y_bin = 0
        elif y_bin >= n_bins:
            y_bin = n_bins - 1
        
        # Linear index = x * n_bins + y
        linear_indices[i] = x_bin * n_bins + y_bin
    
    # Count unique occupied 2D bins using NumPy's unique
    cdef int occupied_2d_bins = len(np.unique(linear_indices))
    
    cdef int total_2d_bins = n_bins * n_bins
    cdef int min_required_2d_bins = <int>(total_2d_bins * min_occupancy_ratio)
    
    cdef bint is_valid = occupied_2d_bins >= min_required_2d_bins
    return is_valid, occupied_2d_bins
