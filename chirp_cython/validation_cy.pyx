# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized validation utilities for CHIRP projection analysis.

This module provides high-performance implementations of bin validation
functions using Cython's static typing and optimizations.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset

@cython.boundscheck(False)
@cython.wraparound(False)
def validate_projection_bins(double[:] proj_array, int n_bins, double min_occupancy_ratio=0.05):
    """
    Validate that a projection has sufficient bin occupancy.
    
    Optimized with Cython for faster execution.
    
    Args:
        proj_array: Normalized projection values in [0, 1]
        n_bins: Number of bins
        min_occupancy_ratio: Minimum ratio of bins that must be occupied
    
    Returns:
        Tuple of (is_valid, occupied_bins_count)
    """
    cdef int n_points = proj_array.shape[0]
    cdef int i, bin_idx
    cdef int* occupied = <int*>malloc(n_bins * sizeof(int))
    cdef int occupied_bins = 0
    cdef int min_required_bins = <int>(n_bins * min_occupancy_ratio)
    cdef bint is_valid
    
    # Initialize occupied array to zeros
    memset(occupied, 0, n_bins * sizeof(int))
    
    # Mark occupied bins
    for i in range(n_points):
        bin_idx = <int>(proj_array[i] * n_bins)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= n_bins:
            bin_idx = n_bins - 1
        occupied[bin_idx] = 1
    
    # Count occupied bins
    for i in range(n_bins):
        if occupied[i]:
            occupied_bins += 1
    
    free(occupied)
    
    is_valid = occupied_bins >= min_required_bins
    return is_valid, occupied_bins

@cython.boundscheck(False)
@cython.wraparound(False)
def validate_2d_projection_bins(double[:] x_proj, double[:] y_proj, int n_bins, 
                                 double min_occupancy_ratio=0.05):
    """
    Validate that a 2D projection has sufficient bin occupancy.
    
    Optimized with Cython using hash set for O(n) counting.
    
    Args:
        x_proj: Normalized X projection values in [0, 1]
        y_proj: Normalized Y projection values in [0, 1]
        n_bins: Number of bins per dimension
        min_occupancy_ratio: Minimum ratio of 2D bins that must be occupied
    
    Returns:
        Tuple of (is_valid, occupied_2d_bins_count)
    """
    cdef int n_points = x_proj.shape[0]
    cdef int i, x_bin, y_bin, linear_idx
    cdef int total_2d_bins = n_bins * n_bins
    cdef int* occupied = <int*>malloc(total_2d_bins * sizeof(int))
    cdef int occupied_2d_bins = 0
    cdef int min_required_2d_bins = <int>(total_2d_bins * min_occupancy_ratio)
    cdef bint is_valid
    
    # Initialize occupied array to zeros
    memset(occupied, 0, total_2d_bins * sizeof(int))
    
    # Mark occupied 2D bins using linear indexing
    for i in range(n_points):
        x_bin = <int>(x_proj[i] * n_bins)
        if x_bin < 0:
            x_bin = 0
        elif x_bin >= n_bins:
            x_bin = n_bins - 1
            
        y_bin = <int>(y_proj[i] * n_bins)
        if y_bin < 0:
            y_bin = 0
        elif y_bin >= n_bins:
            y_bin = n_bins - 1
        
        linear_idx = x_bin * n_bins + y_bin
        occupied[linear_idx] = 1
    
    # Count occupied bins
    for i in range(total_2d_bins):
        if occupied[i]:
            occupied_2d_bins += 1
    
    free(occupied)
    
    is_valid = occupied_2d_bins >= min_required_2d_bins
    return is_valid, occupied_2d_bins
