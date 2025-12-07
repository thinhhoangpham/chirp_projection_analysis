# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-accelerated feature transformation functions for CHIRP projection analysis.

This module provides optimized implementations of mathematical transformations
using C math functions and typed memory views for performance.

Expected speedup: 5-10x over pure Python NumPy version
"""

import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, exp, fabs, isnan, isinf, isfinite, NAN, INFINITY
cimport cython

# Initialize NumPy C API
np.import_array()

# Global epsilon constant for numerical stability
cdef double EPSILON = 1e-10

def set_epsilon_cy(double value):
    """Set the global epsilon value for numerical stability.
    
    Args:
        value: Small positive float to use as epsilon
    """
    global EPSILON
    EPSILON = value

def get_epsilon_cy():
    """Get the current global epsilon value.
    
    Returns:
        Current epsilon value
    """
    return EPSILON

def apply_transform_vectorized_cy(
    np.ndarray[np.float64_t, ndim=2] data,
    int feature_idx,
    str transform_type,
    cache=None
):
    """Apply transformation to entire feature column (Cython-accelerated).
    
    Args:
        data: 2D array of shape (n_samples, n_features)
        feature_idx: Index of feature column to transform
        transform_type: Type of transformation to apply
        cache: Optional ComputationCache instance for caching results
        
    Returns:
        1D array of transformed feature values
    """
    # Check cache first if available
    if cache is not None:
        cached_result = cache.get_transformed_features(data, feature_idx, transform_type)
        if cached_result is not None:
            return cached_result
    
    cdef int n_pts = data.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n_pts, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] feature_data = data[:, feature_idx].astype(np.float64)
    cdef double x, p, pos_result, neg_result
    cdef int i
    
    # Apply transformation using typed loops and C math functions
    if transform_type == 'square':
        for i in range(n_pts):
            x = feature_data[i]
            result[i] = x * x
    
    elif transform_type == 'sqrt':
        for i in range(n_pts):
            x = feature_data[i]
            if x >= 0:
                result[i] = sqrt(x)
            else:
                result[i] = 0.0  # sqrt of negative -> 0
    
    elif transform_type == 'log':
        for i in range(n_pts):
            x = feature_data[i]
            if x > 0:
                result[i] = log(x)
            else:
                result[i] = NAN
    
    elif transform_type == 'log_eps':
        for i in range(n_pts):
            x = feature_data[i]
            result[i] = log(x + EPSILON)
    
    elif transform_type == 'inverse':
        for i in range(n_pts):
            x = feature_data[i]
            if x != 0:
                result[i] = 1.0 / x
            else:
                result[i] = INFINITY
    
    elif transform_type == 'inverse_eps':
        for i in range(n_pts):
            x = feature_data[i]
            result[i] = 1.0 / (x + EPSILON)
    
    elif transform_type == 'logit':
        for i in range(n_pts):
            x = feature_data[i]
            # Stable sigmoid: branchless
            if x >= 0:
                pos_result = 1.0 / (1.0 + exp(-fabs(x)))
                p = pos_result
            else:
                neg_result = exp(x) / (1.0 + exp(x))
                p = neg_result
            # No clamping
            if p <= 0 or p >= 1:
                result[i] = INFINITY
            else:
                result[i] = log(p / (1.0 - p))
    
    elif transform_type == 'logit_eps':
        for i in range(n_pts):
            x = feature_data[i]
            # Stable sigmoid with clamping
            if x >= 0:
                pos_result = 1.0 / (1.0 + exp(-fabs(x)))
                p = pos_result
            else:
                neg_result = exp(x) / (1.0 + exp(x))
                p = neg_result
            # Clamp to [epsilon, 1-epsilon]
            if p < EPSILON:
                p = EPSILON
            elif p > 1.0 - EPSILON:
                p = 1.0 - EPSILON
            result[i] = log(p / (1.0 - p))
    
    elif transform_type == 'sigmoid':
        for i in range(n_pts):
            x = feature_data[i]
            # Stable sigmoid: branchless
            if x >= 0:
                result[i] = 1.0 / (1.0 + exp(-fabs(x)))
            else:
                neg_result = exp(x) / (1.0 + exp(x))
                result[i] = neg_result
    
    elif transform_type == 'sigmoid_eps':
        for i in range(n_pts):
            x = feature_data[i]
            # Stable sigmoid with clamping
            if x >= 0:
                pos_result = 1.0 / (1.0 + exp(-fabs(x)))
                p = pos_result
            else:
                neg_result = exp(x) / (1.0 + exp(x))
                p = neg_result
            # Clamp to [epsilon, 1-epsilon]
            if p < EPSILON:
                p = EPSILON
            elif p > 1.0 - EPSILON:
                p = 1.0 - EPSILON
            result[i] = p
    
    else:  # 'none'
        result = feature_data.copy()
    
    # Cache result if cache is available
    if cache is not None:
        cache.cache_transformed_features(data, feature_idx, transform_type, result)
    
    return result
