"""
Test equivalence between Cython and Python implementations.

This test suite verifies that Cython-accelerated implementations produce
identical results to the pure Python implementations for all core functions.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pure Python implementations
from chirp_python.feature_transforms import (
    apply_feature_transform_vectorized as py_apply_feature_transform_vectorized,
    EPSILON
)
from chirp_python.projection_vectorized import (
    compute_bounds as py_compute_bounds,
    fill_array as py_fill_array,
    compute_projection_vectorized as py_compute_projection_vectorized
)
from chirp_python.validation import (
    validate_projection_bins as py_validate_projection_bins,
    validate_2d_projection_bins as py_validate_2d_projection_bins
)
from chirp_python.data_source import DataSource

# Try to import Cython implementations
try:
    from chirp_cython import (
        apply_feature_transform_vectorized as cy_apply_feature_transform_vectorized,
        compute_bounds as cy_compute_bounds,
        fill_array as cy_fill_array,
        compute_projection_vectorized as cy_compute_projection_vectorized,
        validate_projection_bins as cy_validate_projection_bins,
        validate_2d_projection_bins as cy_validate_2d_projection_bins,
        CYTHON_AVAILABLE
    )
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython modules not available. Skipping Cython equivalence tests.")

def create_test_data(n_samples=100, n_features=6):
    """Create test data for equivalence testing."""
    np.random.seed(42)
    data = np.random.randn(n_samples, n_features)
    # Add some edge cases
    data[0, 0] = 0.0  # Zero value
    data[1, 1] = -1.0  # Negative value
    data[2, 2] = 1e-12  # Very small positive
    data[3, 3] = 1000.0  # Large value
    return data

def create_test_datasource(data):
    """Create a DataSource object for testing."""
    import pandas as pd
    
    # Create class values array
    n_samples = data.shape[0]
    class_values = np.zeros(n_samples, dtype=int)
    
    # Create DataSource directly with data and class_values
    ds = DataSource(data=data, class_values=class_values, class_names=['0'])
    return ds

def test_feature_transforms():
    """Test all feature transformations for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_feature_transforms (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing Feature Transformations")
    print("="*70)
    
    data = create_test_data(n_samples=100, n_features=6)
    
    transforms = [
        'none', 'square', 'sqrt', 'log', 'log_eps',
        'inverse', 'inverse_eps', 'logit', 'logit_eps',
        'sigmoid', 'sigmoid_eps'
    ]
    
    all_passed = True
    for feature_idx in range(data.shape[1]):
        for transform in transforms:
            # Compute with Python
            py_result = py_apply_feature_transform_vectorized(data, feature_idx, transform)
            
            # Compute with Cython
            cy_result = cy_apply_feature_transform_vectorized(data, feature_idx, transform)
            
            # Compare results
            # Handle NaN and Inf values specially
            py_finite = np.isfinite(py_result)
            cy_finite = np.isfinite(cy_result)
            
            if not np.array_equal(py_finite, cy_finite):
                print(f"✗ FAILED: {transform} on feature {feature_idx} - finite masks differ")
                all_passed = False
                continue
            
            # Compare finite values
            if np.any(py_finite):
                finite_match = np.allclose(py_result[py_finite], cy_result[cy_finite], rtol=1e-10, atol=1e-10)
                if not finite_match:
                    print(f"✗ FAILED: {transform} on feature {feature_idx} - finite values differ")
                    print(f"  Max diff: {np.max(np.abs(py_result[py_finite] - cy_result[cy_finite]))}")
                    all_passed = False
                    continue
            
            # Compare NaN/Inf values
            py_isnan = np.isnan(py_result)
            cy_isnan = np.isnan(cy_result)
            py_isinf = np.isinf(py_result)
            cy_isinf = np.isinf(cy_result)
            
            if not np.array_equal(py_isnan, cy_isnan):
                print(f"✗ FAILED: {transform} on feature {feature_idx} - NaN patterns differ")
                all_passed = False
                continue
            
            if not np.array_equal(py_isinf, cy_isinf):
                print(f"✗ FAILED: {transform} on feature {feature_idx} - Inf patterns differ")
                all_passed = False
                continue
    
    if all_passed:
        print(f"✓ PASSED: All {len(transforms)} transformations on {data.shape[1]} features")
    else:
        print("✗ Some feature transformation tests failed")
    
    return all_passed

def test_projection_bounds():
    """Test projection bounds computation for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_projection_bounds (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing Projection Bounds Computation")
    print("="*70)
    
    data = create_test_data(n_samples=100, n_features=6)
    ds = create_test_datasource(data)
    
    # Test cases: different weight patterns and transforms
    test_cases = [
        (np.array([0, 1, 2]), ['none', 'square', 'sqrt']),
        (np.array([0, -1, 2]), ['none', 'log_eps', 'inverse_eps']),
        (np.array([3, 4]), ['sigmoid_eps', 'logit_eps']),
    ]
    
    all_passed = True
    for wi, transforms in test_cases:
        # Compute with Python
        py_bounds = py_compute_bounds(ds, wi, transforms, len(data))
        
        # Compute with Cython  
        cy_bounds = cy_compute_bounds(ds, wi, transforms, len(data))
        
        # Compare results
        if not np.allclose(py_bounds, cy_bounds, rtol=1e-10, atol=1e-10):
            print(f"✗ FAILED: Bounds differ for wi={wi}, transforms={transforms}")
            print(f"  Python bounds: {py_bounds}")
            print(f"  Cython bounds: {cy_bounds}")
            all_passed = False
        else:
            print(f"✓ PASSED: wi={wi}, transforms={transforms}")
    
    return all_passed

def test_fill_array():
    """Test array filling for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_fill_array (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing Array Filling")
    print("="*70)
    
    data = create_test_data(n_samples=100, n_features=6)
    ds = create_test_datasource(data)
    
    # Test cases
    test_cases = [
        (np.array([0, 1, 2]), ['none', 'square', 'sqrt'], np.array([0.0, 10.0])),
        (np.array([0, -1, 2]), ['none', 'log_eps', 'inverse_eps'], np.array([-5.0, 5.0])),
    ]
    
    all_passed = True
    for wi, transforms, bounds in test_cases:
        # Compute with Python
        py_result = py_fill_array(wi, transforms, bounds, ds, len(data))
        
        # Compute with Cython
        cy_result = cy_fill_array(wi, transforms, bounds, ds, len(data))
        
        # Compare results
        if not np.allclose(py_result, cy_result, rtol=1e-10, atol=1e-10):
            print(f"✗ FAILED: Results differ for wi={wi}, transforms={transforms}")
            print(f"  Max diff: {np.max(np.abs(py_result - cy_result))}")
            all_passed = False
        else:
            print(f"✓ PASSED: wi={wi}, transforms={transforms}")
    
    return all_passed

def test_projection_vectorized():
    """Test vectorized projection computation for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_projection_vectorized (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing Vectorized Projection Computation")
    print("="*70)
    
    data = create_test_data(n_samples=100, n_features=6)
    ds = create_test_datasource(data)
    
    # Test cases
    test_cases = [
        (np.array([0, 1, 2]), ['none', 'square', 'sqrt']),
        (np.array([0, -1, 2]), ['none', 'log_eps', 'inverse_eps']),
    ]
    
    all_passed = True
    for wi, transforms in test_cases:
        # Compute with Python
        py_proj, py_bounds, py_valid = py_compute_projection_vectorized(ds, wi, transforms, len(data), normalize=True)
        
        # Compute with Cython
        cy_proj, cy_bounds, cy_valid = cy_compute_projection_vectorized(ds, wi, transforms, len(data), normalize=True)
        
        # Compare results
        proj_match = np.allclose(py_proj, cy_proj, rtol=1e-10, atol=1e-10)
        bounds_match = np.allclose(py_bounds, cy_bounds, rtol=1e-10, atol=1e-10)
        valid_match = py_valid == cy_valid
        
        if not (proj_match and bounds_match and valid_match):
            print(f"✗ FAILED: Results differ for wi={wi}, transforms={transforms}")
            if not proj_match:
                print(f"  Projection max diff: {np.max(np.abs(py_proj - cy_proj))}")
            if not bounds_match:
                print(f"  Bounds: Python={py_bounds}, Cython={cy_bounds}")
            if not valid_match:
                print(f"  Valid count: Python={py_valid}, Cython={cy_valid}")
            all_passed = False
        else:
            print(f"✓ PASSED: wi={wi}, transforms={transforms}")
    
    return all_passed

def test_validate_bins_1d():
    """Test 1D bin validation for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_validate_bins_1d (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing 1D Bin Validation")
    print("="*70)
    
    # Create test projection arrays
    test_cases = [
        (np.linspace(0, 1, 100), 20, 0.05),  # Uniform distribution
        (np.random.rand(100), 20, 0.05),  # Random distribution
        (np.zeros(100), 20, 0.05),  # All in one bin
    ]
    
    all_passed = True
    for proj_array, n_bins, min_ratio in test_cases:
        # Validate with Python
        py_valid, py_count = py_validate_projection_bins(proj_array, n_bins, min_ratio)
        
        # Validate with Cython
        cy_valid, cy_count = cy_validate_projection_bins(proj_array, n_bins, min_ratio)
        
        # Compare results
        if py_valid != cy_valid or py_count != cy_count:
            print(f"✗ FAILED: Results differ")
            print(f"  Python: valid={py_valid}, count={py_count}")
            print(f"  Cython: valid={cy_valid}, count={cy_count}")
            all_passed = False
        else:
            print(f"✓ PASSED: valid={py_valid}, count={py_count}")
    
    return all_passed

def test_validate_bins_2d():
    """Test 2D bin validation for equivalence."""
    if not CYTHON_AVAILABLE:
        print("SKIP: test_validate_bins_2d (Cython not available)")
        return
    
    print("\n" + "="*70)
    print("Testing 2D Bin Validation")
    print("="*70)
    
    # Create test projection arrays
    test_cases = [
        (np.linspace(0, 1, 100), np.linspace(0, 1, 100), 20, 0.05),  # Diagonal
        (np.random.rand(100), np.random.rand(100), 20, 0.05),  # Random
        (np.zeros(100), np.zeros(100), 20, 0.05),  # All in one bin
    ]
    
    all_passed = True
    for x_proj, y_proj, n_bins, min_ratio in test_cases:
        # Validate with Python
        py_valid, py_count = py_validate_2d_projection_bins(x_proj, y_proj, n_bins, min_ratio)
        
        # Validate with Cython
        cy_valid, cy_count = cy_validate_2d_projection_bins(x_proj, y_proj, n_bins, min_ratio)
        
        # Compare results
        if py_valid != cy_valid or py_count != cy_count:
            print(f"✗ FAILED: Results differ")
            print(f"  Python: valid={py_valid}, count={py_count}")
            print(f"  Cython: valid={cy_valid}, count={cy_count}")
            all_passed = False
        else:
            print(f"✓ PASSED: valid={py_valid}, count={py_count}")
    
    return all_passed

def main():
    """Run all equivalence tests."""
    print("\n" + "="*70)
    print("CYTHON EQUIVALENCE TEST SUITE")
    print("="*70)
    
    if not CYTHON_AVAILABLE:
        print("\n⚠ Cython modules not available. All tests will be skipped.")
        print("To build Cython modules, run: python setup.py build_ext --inplace")
        return
    
    print("✓ Cython modules loaded successfully")
    
    results = []
    results.append(("Feature Transforms", test_feature_transforms()))
    results.append(("Projection Bounds", test_projection_bounds()))
    results.append(("Array Filling", test_fill_array()))
    results.append(("Vectorized Projection", test_projection_vectorized()))
    results.append(("1D Bin Validation", test_validate_bins_1d()))
    results.append(("2D Bin Validation", test_validate_bins_2d()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All equivalence tests PASSED!")
        print("Cython and Python implementations produce identical results.")
        return 0
    else:
        print("\n✗ Some equivalence tests FAILED!")
        print("Cython and Python implementations have differences.")
        return 1

if __name__ == '__main__':
    exit(main())
