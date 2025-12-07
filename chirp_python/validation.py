import numpy as np
from typing import Tuple, List, Any, Optional
from chirp_python.projection_vectorized import compute_projection_vectorized

def validate_projection_bins(proj_array: np.ndarray, n_bins: int, min_occupancy_ratio: float = 0.05) -> Tuple[bool, int]:
    """
    Validate that a projection has sufficient bin occupancy.
    
    OPTIMIZATION: Uses single vectorized operation for bin computation.
    
    Args:
        proj_array: Normalized projection values in [0, 1]
        n_bins: Number of bins
        min_occupancy_ratio: Minimum ratio of bins that must be occupied (default 0.05 = 5%)
    
    Returns:
        Tuple of (is_valid, occupied_bins_count)
    """
    # Convert normalized values to bin indices (single vectorized operation)
    bin_indices = np.clip((proj_array * n_bins).astype(np.int32), 0, n_bins - 1)
    
    # Count unique occupied bins
    occupied_bins = len(np.unique(bin_indices))
    min_required_bins = int(n_bins * min_occupancy_ratio)
    
    is_valid = occupied_bins >= min_required_bins
    return is_valid, occupied_bins

def validate_2d_projection_bins(x_proj: np.ndarray, y_proj: np.ndarray, n_bins: int, min_occupancy_ratio: float = 0.05) -> Tuple[bool, int]:
    """
    Validate that a 2D projection has sufficient bin occupancy.
    
    OPTIMIZATION: Uses linear indexing + np.unique for O(n log n) counting
    instead of Python zip() + set() operations (~5x faster).
    
    Args:
        x_proj: Normalized X projection values in [0, 1]
        y_proj: Normalized Y projection values in [0, 1]
        n_bins: Number of bins per dimension (creates n_bins x n_bins grid)
        min_occupancy_ratio: Minimum ratio of 2D bins that must be occupied (default 0.05 = 5%)
    
    Returns:
        Tuple of (is_valid, occupied_2d_bins_count)
    """
    # Convert normalized values to bin indices (vectorized)
    x_bin_indices = np.clip((x_proj * n_bins).astype(np.int32), 0, n_bins - 1)
    y_bin_indices = np.clip((y_proj * n_bins).astype(np.int32), 0, n_bins - 1)
    
    # OPTIMIZATION: Use linear indexing for unique count (avoids Python zip + set)
    # Linear index = x * n_bins + y, then count unique values
    linear_indices = x_bin_indices * n_bins + y_bin_indices
    occupied_2d_bins = len(np.unique(linear_indices))
    
    total_2d_bins = n_bins * n_bins
    min_required_2d_bins = int(total_2d_bins * min_occupancy_ratio)
    
    is_valid = occupied_2d_bins >= min_required_2d_bins
    return is_valid, occupied_2d_bins

def validate_incremental_term(axis: str, term_idx: int, term_weight: int, term_transform: str, 
                              current_xwt: np.ndarray, current_ywt: np.ndarray,
                              current_x_transforms: list, current_y_transforms: list,
                              ds, n_bins: int, n_pts: int, min_occupancy_ratio: float = 0.05,
                              validation_mode: str = '1d') -> Tuple[bool, int, bool]:
    """
    Validate adding a single term to either X or Y axis incrementally.
    
    Special handling for first terms:
    - First term to X: Accepted without validation (Y is still empty, can't validate both)
    - First term to Y: Accepted without validation (X is still empty, can't validate both)
    - Subsequent terms: Both axes validated together
    - This allows bootstrapping projections without invalid bounds errors
    
    Args:
        axis: 'x' or 'y' - which axis to add the term to
        term_idx: Feature index to add
        term_weight: Weight for the term (positive or negative)
        term_transform: Transformation to apply
        current_xwt: Current X weights
        current_ywt: Current Y weights  
        current_x_transforms: Current X transformations
        current_y_transforms: Current Y transformations
        ds: DataSource
        n_bins: Number of bins
        n_pts: Number of points
        min_occupancy_ratio: Minimum occupancy ratio
        validation_mode: '1d' for independent X/Y validation, '2d' for combined 2D grid validation
    
    Returns:
        Tuple of (is_valid, occupied_bins_count, has_invalid_values)
        - is_valid: Whether the projection meets occupancy requirements
        - occupied_bins_count: Number of occupied bins
        - has_invalid_values: Whether NaN/Inf values were produced by transformations
    """
    # Create temporary copies
    temp_xwt = current_xwt.copy()
    temp_ywt = current_ywt.copy()
    temp_x_transforms = current_x_transforms.copy()
    temp_y_transforms = current_y_transforms.copy()
    
    # Add the new term to the specified axis
    if axis == 'x':
        temp_xwt = np.append(temp_xwt, term_weight)
        temp_x_transforms.append(term_transform)
    else:  # axis == 'y'
        temp_ywt = np.append(temp_ywt, term_weight)
        temp_y_transforms.append(term_transform)
    
    # Check if either axis is empty before adding the new term
    x_was_empty = len(current_xwt) == 0
    y_was_empty = len(current_ywt) == 0
    
    # OPTIMIZATION: Use vectorized projection (combines bounds + array in single pass)
    # OLD: compute_bounds(X) + fill_array(X) + compute_bounds(Y) + fill_array(Y) = 4 passes
    # NEW: compute_projection_vectorized(X) + compute_projection_vectorized(Y) = 2 passes
    if len(temp_xwt) > 0:
        x_proj, x_bounds, _ = compute_projection_vectorized(ds, temp_xwt, temp_x_transforms, n_pts, normalize=True)
    else:
        x_proj = np.zeros(n_pts)
        x_bounds = np.array([0.0, 1.0])

    if len(temp_ywt) > 0:
        y_proj, y_bounds, _ = compute_projection_vectorized(ds, temp_ywt, temp_y_transforms, n_pts, normalize=True)
    else:
        y_proj = np.zeros(n_pts)
        y_bounds = np.array([0.0, 1.0])
    
    # Check for invalid values (NaN/Inf) produced by unsafe transforms
    has_invalid_values = False
    if len(temp_xwt) > 0:
        if np.any(np.isnan(x_proj)) or np.any(np.isinf(x_proj)):
            has_invalid_values = True
    if len(temp_ywt) > 0:
        if np.any(np.isnan(y_proj)) or np.any(np.isinf(y_proj)):
            has_invalid_values = True
            
    if has_invalid_values:
        return False, 0, True
    
    # Validate based on mode
    if validation_mode == '1d':
        # Validate X and Y independently
        x_valid, x_occupied = validate_projection_bins(x_proj, n_bins, min_occupancy_ratio)
        y_valid, y_occupied = validate_projection_bins(y_proj, n_bins, min_occupancy_ratio)
        
        # Special handling for first terms:
        # - First term to X: Accept without validation (Y is still empty)
        # - First term to Y: Now validate both axes together
        if axis == 'x' and y_was_empty:
            # Adding first term to X, Y is still empty - accept without validation
            is_valid = True
            occupied_bins = x_occupied
        elif axis == 'y' and x_was_empty:
            # Adding first term to Y, but X is still empty - accept without validation
            is_valid = True
            occupied_bins = y_occupied
        else:
            # Both axes now have terms, validate both together
            is_valid = x_valid and y_valid
            occupied_bins = x_occupied + y_occupied
        
        return is_valid, occupied_bins, False
    else:
        # For 2D mode, skip validation until both axes have terms
        if len(temp_xwt) == 0 or len(temp_ywt) == 0:
            # At least one axis still empty - accept without validation
            return True, 1, False
        
        # Both axes have terms, validate 2D projection
        is_valid, occupied_bins = validate_2d_projection_bins(x_proj, y_proj, n_bins, min_occupancy_ratio)
        return is_valid, occupied_bins, False
