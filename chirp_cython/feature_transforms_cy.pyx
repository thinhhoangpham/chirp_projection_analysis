# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized feature transformation utilities for CHIRP projection analysis.

This module provides high-performance implementations of feature transformations
using Cython's static typing and C-level optimizations.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log, sqrt, exp, fabs, isnan, isinf, isfinite

# Global epsilon constant for numerical stability
cdef double EPSILON = 1e-10

def set_epsilon(double value):
    """Set the global epsilon value for numerical stability.
    
    Args:
        value: Small positive float to use as epsilon
    """
    global EPSILON
    EPSILON = value

def get_epsilon():
    """Get the current global epsilon value.
    
    Returns:
        Current epsilon value
    """
    return EPSILON

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _expit_stable(double z) nogil:
    """Numerically stable logistic sigmoid.
    
    Uses numerically stable computation to avoid overflow:
    - For z >= 0: 1/(1+exp(-z))
    - For z < 0: exp(z)/(1+exp(z))
    
    Args:
        z: Input value
        
    Returns:
        Sigmoid output clamped to [EPSILON, 1-EPSILON]
    """
    cdef double out, ez
    
    if z >= 0:
        # For z >= 0, exp(-z) <= 1 so 1/(1+exp(-z)) is stable
        out = 1.0 / (1.0 + exp(-z))
    else:
        # For z < 0, use exp(z)/(1+exp(z)) to avoid overflow
        ez = exp(z)
        out = ez / (1.0 + ez)
    
    # Clamp output to avoid exact 0 or 1 (for logit stability)
    if out < EPSILON:
        out = EPSILON
    elif out > 1.0 - EPSILON:
        out = 1.0 - EPSILON
        
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_feature_transform_vectorized(double[:, :] data, int feature_idx,
                                       str transform_type):
    """Apply transformation to entire feature column with Cython acceleration.
    
    Args:
        data: 2D array of shape (n_samples, n_features)
        feature_idx: Index of feature column to transform
        transform_type: Type of transformation to apply
        
    Returns:
        1D array of transformed feature values
    """
    cdef int n_samples = data.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.empty(n_samples, dtype=np.float64)
    cdef double[:] result_view = result
    cdef int i
    cdef double val
    
    # Extract transformation type and dispatch to appropriate function
    if transform_type == 'square':
        for i in range(n_samples):
            val = data[i, feature_idx]
            result_view[i] = val * val
    elif transform_type == 'sqrt':
        for i in range(n_samples):
            val = data[i, feature_idx]
            if val >= 0:
                result_view[i] = sqrt(val)
            else:
                result_view[i] = 0.0  # sqrt of max(val, 0)
    elif transform_type == 'log':
        for i in range(n_samples):
            val = data[i, feature_idx]
            if val > 0:
                result_view[i] = log(val)
            else:
                result_view[i] = -1.0 / 0.0  # -inf
    elif transform_type == 'log_eps':
        for i in range(n_samples):
            val = data[i, feature_idx]
            result_view[i] = log(val + EPSILON)
    elif transform_type == 'inverse':
        for i in range(n_samples):
            val = data[i, feature_idx]
            if val != 0:
                result_view[i] = 1.0 / val
            else:
                result_view[i] = 1.0 / 0.0  # inf
    elif transform_type == 'inverse_eps':
        for i in range(n_samples):
            val = data[i, feature_idx]
            result_view[i] = 1.0 / (val + EPSILON)
    elif transform_type == 'logit':
        for i in range(n_samples):
            val = data[i, feature_idx]
            # Stable sigmoid without clamping
            if val >= 0:
                p = 1.0 / (1.0 + exp(-val))
            else:
                ez = exp(val)
                p = ez / (1.0 + ez)
            # Compute logit without safety clamping
            if p > 0 and p < 1.0:
                result_view[i] = log(p / (1.0 - p))
            else:
                result_view[i] = 1.0 / 0.0 if p >= 1.0 else -1.0 / 0.0  # inf or -inf
    elif transform_type == 'logit_eps':
        for i in range(n_samples):
            val = data[i, feature_idx]
            p = _expit_stable(val)  # Already clamped in _expit_stable
            result_view[i] = log(p / (1.0 - p))
    elif transform_type == 'sigmoid':
        for i in range(n_samples):
            val = data[i, feature_idx]
            # Stable sigmoid without epsilon clamping
            if val >= 0:
                result_view[i] = 1.0 / (1.0 + exp(-val))
            else:
                ez = exp(val)
                result_view[i] = ez / (1.0 + ez)
    elif transform_type == 'sigmoid_eps':
        for i in range(n_samples):
            val = data[i, feature_idx]
            result_view[i] = _expit_stable(val)  # With epsilon clamping
    else:  # 'none'
        for i in range(n_samples):
            result_view[i] = data[i, feature_idx]
    
    return result

# List of available transformations
AVAILABLE_TRANSFORMS = [
    'none', 'square', 'sqrt', 'log', 'log_eps',
    'inverse', 'inverse_eps', 'logit', 'logit_eps',
    'sigmoid', 'sigmoid_eps'
]
