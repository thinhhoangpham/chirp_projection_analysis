"""
Feature transformation utilities for CHIRP projection analysis.

This module provides functions to apply various mathematical transformations
to feature data, including:
- Basic transforms: square, sqrt, log, inverse
- Sigmoid and logit transforms
- Epsilon-safe versions to avoid numerical instabilities

All vectorized transforms support caching for improved performance.
"""

import numpy as np
from typing import Optional

# Global epsilon constant for numerical stability
EPSILON = 1e-10  # Small value to avoid division by zero and log(0)


def set_epsilon(value: float):
    """Set the global epsilon value for numerical stability.

    Args:
        value: Small positive float to use as epsilon
    """
    global EPSILON
    EPSILON = value


def get_epsilon() -> float:
    """Get the current global epsilon value.

    Returns:
        Current epsilon value
    """
    return EPSILON


def _expit_stable(z):
    """Numerically stable logistic sigmoid for scalars or arrays.

    Uses numerically stable computation to avoid overflow:
    - For z >= 0: 1/(1+exp(-z))
    - For z < 0: exp(z)/(1+exp(z))

    Args:
        z: Input value or array

    Returns:
        Sigmoid output clamped to [EPSILON, 1-EPSILON]
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    pos = z_arr >= 0
    # For z >= 0, exp(-z) <= 1 so 1/(1+exp(-z)) is stable
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    # For z < 0, use exp(z)/(1+exp(z)) to avoid overflow
    neg = ~pos
    if np.any(neg):
        ez = np.exp(z_arr[neg])
        out[neg] = ez / (1.0 + ez)
    # Clamp output to avoid exact 0 or 1 (for logit stability)
    out = np.clip(out, EPSILON, 1.0 - EPSILON)
    # Return scalar for scalar input
    return out if out.shape != () else out.item()


def apply_feature_transform(x, transform_type: str):
    """Apply transformation to a single feature value.

    Args:
        x: Input value
        transform_type: One of 'square', 'sqrt', 'log', 'log_eps', 'inverse',
                       'inverse_eps', 'logit', 'logit_eps', 'sigmoid',
                       'sigmoid_eps', 'none'

    Returns:
        Transformed value (may be NaN if transformation is invalid)
    """
    if np.isnan(x):
        return x

    if transform_type == 'square':
        return x * x
    elif transform_type == 'sqrt':
        return np.sqrt(x) if x >= 0 else np.nan
    elif transform_type == 'log':
        return np.log(x) if x > 0 else np.nan
    elif transform_type == 'log_eps':
        # Safe log with epsilon to avoid log(0)
        return np.log(x + EPSILON)
    elif transform_type == 'inverse':
        return 1.0 / x if x != 0 else np.nan
    elif transform_type == 'inverse_eps':
        # Safe inverse with epsilon to avoid 1/0
        return 1.0 / (x + EPSILON)
    elif transform_type == 'logit':
        # Apply sigmoid first, then logit
        p = _expit_stable(x)
        if p <= 0 or p >= 1:
            return np.nan
        return np.log(p / (1 - p))
    elif transform_type == 'logit_eps':
        # Safe logit with epsilon-clamped sigmoid
        p = _expit_stable(x)  # Already clamped in _expit_stable
        # Additional safety: clamp p to avoid boundary issues
        p = np.clip(p, EPSILON, 1.0 - EPSILON)
        return np.log(p / (1.0 - p))
    elif transform_type == 'sigmoid':
        return _expit_stable(x)
    elif transform_type == 'sigmoid_eps':
        # Sigmoid with explicit epsilon clamping (same as regular sigmoid now)
        return _expit_stable(x)
    else:  # 'none'
        return x


def apply_feature_transform_vectorized(data: np.ndarray, feature_idx: int,
                                      transform_type: str,
                                      cache=None) -> np.ndarray:
    """Apply transformation to entire feature column with optional caching.

    Uses fully vectorized NumPy operations for performance. Epsilon-safe versions
    of log/inverse/logit/sigmoid are used to avoid numerical instabilities.

    Args:
        data: 2D array of shape (n_samples, n_features)
        feature_idx: Index of feature column to transform
        transform_type: Type of transformation to apply
        cache: Optional ComputationCache instance for caching results

    Returns:
        1D array of transformed feature values

    Notes:
        - Invalid values (NaN/Inf) may be produced by some transformations
        - When cache is provided, results are cached for reuse
        - Uses epsilon-safe transforms by default for numerical stability
    """
    # Check cache first if available
    if cache is not None:
        cached_result = cache.get_transformed_features(data, feature_idx, transform_type)
        if cached_result is not None:
            return cached_result

    # Extract feature column as float64 for consistent computation
    feature_data = data[:, feature_idx].astype(np.float64)
    total_points = len(feature_data)

    # Apply transformation using vectorized operations
    if transform_type == 'square':
        result = feature_data * feature_data
    elif transform_type == 'sqrt':
        # Vectorized sqrt with np.maximum to handle negatives
        result = np.sqrt(np.maximum(feature_data, 0.0))
    elif transform_type == 'log':
        # Unsafe log - can produce -inf/nan
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log(feature_data)
    elif transform_type == 'log_eps':
        # Always use epsilon-safe version for robustness
        result = np.log(feature_data + EPSILON)
    elif transform_type == 'inverse':
        # Unsafe inverse - can produce inf
        with np.errstate(divide='ignore'):
            result = 1.0 / feature_data
    elif transform_type == 'inverse_eps':
        # Always use epsilon-safe version for robustness
        result = 1.0 / (feature_data + EPSILON)
    elif transform_type == 'logit':
        # Unsafe logit - can produce inf/-inf if p=0 or p=1
        # Stable sigmoid: branchless using np.where
        pos_result = 1.0 / (1.0 + np.exp(-np.abs(feature_data)))
        neg_result = np.exp(feature_data) / (1.0 + np.exp(feature_data))
        p = np.where(feature_data >= 0, pos_result, neg_result)
        # No clamping
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log(p / (1.0 - p))
    elif transform_type == 'logit_eps':
        # Fully vectorized stable logit with epsilon
        # Stable sigmoid: branchless using np.where
        pos_result = 1.0 / (1.0 + np.exp(-np.abs(feature_data)))
        neg_result = np.exp(feature_data) / (1.0 + np.exp(feature_data))
        p = np.where(feature_data >= 0, pos_result, neg_result)
        # Clamp to [epsilon, 1-epsilon] for logit stability
        p = np.clip(p, EPSILON, 1.0 - EPSILON)
        result = np.log(p / (1.0 - p))
    elif transform_type == 'sigmoid':
        # Unsafe sigmoid - no clamping
        pos_result = 1.0 / (1.0 + np.exp(-np.abs(feature_data)))
        neg_result = np.exp(feature_data) / (1.0 + np.exp(feature_data))
        result = np.where(feature_data >= 0, pos_result, neg_result)
    elif transform_type == 'sigmoid_eps':
        # Fully vectorized stable sigmoid (branchless)
        pos_result = 1.0 / (1.0 + np.exp(-np.abs(feature_data)))
        neg_result = np.exp(feature_data) / (1.0 + np.exp(feature_data))
        result = np.where(feature_data >= 0, pos_result, neg_result)
        # Clamp to [epsilon, 1-epsilon]
        result = np.clip(result, EPSILON, 1.0 - EPSILON)
    else:  # 'none'
        result = feature_data.copy()

    # Cache result if cache is available
    if cache is not None:
        cache.cache_transformed_features(data, feature_idx, transform_type, result)

    return result


# List of available transformations
AVAILABLE_TRANSFORMS = [
    'none', 'square', 'sqrt', 'log', 'log_eps',
    'inverse', 'inverse_eps', 'logit', 'logit_eps',
    'sigmoid', 'sigmoid_eps'
]
