"""
Equivalence tests for Cython-accelerated CHIRP modules.

These tests verify that Cython implementations produce identical results
to the pure Python implementations for:
- Feature transformation functions
- Projection computation functions
- Validation functions

Run with: pytest tests/test_cython_equivalence.py
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Python versions
from chirp_python.feature_transforms import (
    apply_feature_transform_vectorized,
    set_epsilon,
    get_epsilon,
    EPSILON as PY_EPSILON
)
from chirp_python.projection_vectorized import (
    compute_bounds,
    fill_array,
)
from chirp_python.validation import (
    validate_projection_bins,
    validate_2d_projection_bins,
)

# Try to import Cython versions
try:
    from chirp_cython import (
        apply_transform_vectorized_cy,
        set_epsilon_cy,
        get_epsilon_cy,
        compute_bounds_cy,
        fill_array_cy,
        validate_projection_bins_cy,
        validate_2d_projection_bins_cy,
        CYTHON_AVAILABLE,
    )
    CYTHON_COMPILED = CYTHON_AVAILABLE
except ImportError:
    CYTHON_COMPILED = False

# Skip all tests if Cython is not compiled
pytestmark = pytest.mark.skipif(
    not CYTHON_COMPILED,
    reason="Cython extensions not compiled"
)


class MockDataSource:
    """Mock DataSource for testing"""
    FUZZ0 = 0.01
    
    def __init__(self, data):
        self.data = data


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    data = np.random.randn(n_samples, n_features) * 10 + 50
    # Ensure some positive values for transforms
    data = np.abs(data)
    return data


@pytest.fixture
def sample_data_source(sample_data):
    """Create a mock DataSource"""
    return MockDataSource(sample_data)


class TestFeatureTransforms:
    """Test feature transformation functions"""
    
    def test_epsilon_management(self):
        """Test epsilon set/get functions"""
        # Set epsilon in both versions
        test_epsilon = 1e-8
        set_epsilon(test_epsilon)
        set_epsilon_cy(test_epsilon)
        
        # Verify they match
        assert get_epsilon() == test_epsilon
        assert get_epsilon_cy() == test_epsilon
        
        # Reset to default
        set_epsilon(1e-10)
        set_epsilon_cy(1e-10)
    
    @pytest.mark.parametrize("transform_type", [
        'none', 'square', 'sqrt', 'log', 'log_eps',
        'inverse', 'inverse_eps', 'logit', 'logit_eps',
        'sigmoid', 'sigmoid_eps'
    ])
    def test_transform_equivalence(self, sample_data, transform_type):
        """Test that transforms produce identical results"""
        feature_idx = 0
        
        # Apply Python transform
        py_result = apply_feature_transform_vectorized(
            sample_data, feature_idx, transform_type, cache=None
        )
        
        # Apply Cython transform
        cy_result = apply_transform_vectorized_cy(
            sample_data, feature_idx, transform_type, cache=None
        )
        
        # Check shape matches
        assert py_result.shape == cy_result.shape
        
        # Check values match (allowing for numerical precision)
        # For transforms that can produce NaN/Inf, check those separately
        py_finite = np.isfinite(py_result)
        cy_finite = np.isfinite(cy_result)
        
        # Finite masks should match
        assert np.array_equal(py_finite, cy_finite), \
            f"Finite masks differ for {transform_type}"
        
        # Where both are finite, values should be very close
        if np.any(py_finite):
            np.testing.assert_allclose(
                py_result[py_finite],
                cy_result[cy_finite],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Values differ for {transform_type}"
            )
        
        # Where both are non-finite, check they're the same kind of non-finite
        non_finite = ~py_finite
        if np.any(non_finite):
            py_nan = np.isnan(py_result[non_finite])
            cy_nan = np.isnan(cy_result[non_finite])
            assert np.array_equal(py_nan, cy_nan), \
                f"NaN patterns differ for {transform_type}"


class TestProjectionFunctions:
    """Test projection computation functions"""
    
    def test_compute_bounds_equivalence(self, sample_data_source):
        """Test that compute_bounds produces identical results"""
        n_pts = sample_data_source.data.shape[0]
        wi = np.array([0, 1, 2])
        transforms = ['none', 'square', 'sqrt']
        
        # Compute with Python
        py_bounds = compute_bounds(sample_data_source, wi, transforms, n_pts)
        
        # Compute with Cython
        cy_bounds = compute_bounds_cy(sample_data_source, wi, transforms, n_pts)
        
        # Check shapes match
        assert py_bounds.shape == cy_bounds.shape
        
        # Check values match
        np.testing.assert_allclose(
            py_bounds,
            cy_bounds,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Bounds differ between Python and Cython"
        )
    
    def test_fill_array_equivalence(self, sample_data_source):
        """Test that fill_array produces identical results"""
        n_pts = sample_data_source.data.shape[0]
        wi = np.array([0, 1, 2])
        transforms = ['none', 'square', 'sqrt']
        
        # First compute bounds
        bounds = compute_bounds(sample_data_source, wi, transforms, n_pts)
        
        # Fill array with Python
        py_array = fill_array(wi, transforms, bounds, sample_data_source, n_pts)
        
        # Fill array with Cython
        cy_array = fill_array_cy(wi, transforms, bounds, sample_data_source, n_pts)
        
        # Check shapes match
        assert py_array.shape == cy_array.shape
        
        # Check values match
        np.testing.assert_allclose(
            py_array,
            cy_array,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Arrays differ between Python and Cython"
        )
    
    def test_projection_with_transforms(self, sample_data_source):
        """Test projection with various transforms"""
        n_pts = sample_data_source.data.shape[0]
        wi = np.array([0, -1, 2])  # Include negative weight
        transforms = ['log_eps', 'inverse_eps', 'sigmoid_eps']
        
        # Compute with Python
        py_bounds = compute_bounds(sample_data_source, wi, transforms, n_pts)
        py_array = fill_array(wi, transforms, py_bounds, sample_data_source, n_pts)
        
        # Compute with Cython
        cy_bounds = compute_bounds_cy(sample_data_source, wi, transforms, n_pts)
        cy_array = fill_array_cy(wi, transforms, cy_bounds, sample_data_source, n_pts)
        
        # Check bounds match
        np.testing.assert_allclose(py_bounds, cy_bounds, rtol=1e-10, atol=1e-12)
        
        # Check arrays match
        np.testing.assert_allclose(py_array, cy_array, rtol=1e-10, atol=1e-12)


class TestValidationFunctions:
    """Test validation functions"""
    
    def test_validate_projection_bins_equivalence(self):
        """Test 1D bin validation produces identical results"""
        np.random.seed(42)
        proj_array = np.random.rand(1000)
        n_bins = 50
        min_occupancy_ratio = 0.05
        
        # Validate with Python
        py_valid, py_count = validate_projection_bins(
            proj_array, n_bins, min_occupancy_ratio
        )
        
        # Validate with Cython
        cy_valid, cy_count = validate_projection_bins_cy(
            proj_array, n_bins, min_occupancy_ratio
        )
        
        # Check results match
        assert py_valid == cy_valid, "Valid flags differ"
        assert py_count == cy_count, f"Counts differ: Python={py_count}, Cython={cy_count}"
    
    def test_validate_2d_projection_bins_equivalence(self):
        """Test 2D bin validation produces identical results"""
        np.random.seed(42)
        x_proj = np.random.rand(1000)
        y_proj = np.random.rand(1000)
        n_bins = 20
        min_occupancy_ratio = 0.05
        
        # Validate with Python
        py_valid, py_count = validate_2d_projection_bins(
            x_proj, y_proj, n_bins, min_occupancy_ratio
        )
        
        # Validate with Cython
        cy_valid, cy_count = validate_2d_projection_bins_cy(
            x_proj, y_proj, n_bins, min_occupancy_ratio
        )
        
        # Check results match
        assert py_valid == cy_valid, "Valid flags differ"
        assert py_count == cy_count, f"Counts differ: Python={py_count}, Cython={cy_count}"
    
    def test_edge_cases_validation(self):
        """Test validation edge cases"""
        # All points in one bin
        proj_array = np.ones(100) * 0.5
        n_bins = 50
        
        py_valid, py_count = validate_projection_bins(proj_array, n_bins, 0.05)
        cy_valid, cy_count = validate_projection_bins_cy(proj_array, n_bins, 0.05)
        
        assert py_valid == cy_valid
        assert py_count == cy_count
        
        # Uniform distribution
        proj_array = np.linspace(0, 1, 1000, endpoint=False)
        py_valid, py_count = validate_projection_bins(proj_array, n_bins, 0.05)
        cy_valid, cy_count = validate_projection_bins_cy(proj_array, n_bins, 0.05)
        
        assert py_valid == cy_valid
        assert py_count == cy_count


class TestFullPipeline:
    """Test full projection pipeline end-to-end"""
    
    def test_full_projection_pipeline(self, sample_data_source):
        """Test complete projection pipeline"""
        n_pts = sample_data_source.data.shape[0]
        wi = np.array([0, 1, -2, 3])
        transforms = ['log_eps', 'sqrt', 'inverse_eps', 'sigmoid_eps']
        n_bins = 30
        
        # Python pipeline
        py_bounds = compute_bounds(sample_data_source, wi, transforms, n_pts)
        py_array = fill_array(wi, transforms, py_bounds, sample_data_source, n_pts)
        py_valid, py_count = validate_projection_bins(py_array, n_bins, 0.05)
        
        # Cython pipeline
        cy_bounds = compute_bounds_cy(sample_data_source, wi, transforms, n_pts)
        cy_array = fill_array_cy(wi, transforms, cy_bounds, sample_data_source, n_pts)
        cy_valid, cy_count = validate_projection_bins_cy(cy_array, n_bins, 0.05)
        
        # Check all results match
        np.testing.assert_allclose(py_bounds, cy_bounds, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(py_array, cy_array, rtol=1e-10, atol=1e-12)
        assert py_valid == cy_valid
        assert py_count == cy_count


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
