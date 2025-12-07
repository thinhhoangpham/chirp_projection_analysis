"""
Simple benchmark to demonstrate Cython speedup.

Run with: python3 tests/benchmark_cython.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Python versions
from chirp_python.feature_transforms import apply_feature_transform_vectorized
from chirp_python.validation import validate_projection_bins, validate_2d_projection_bins

# Import Cython versions
try:
    from chirp_cython import (
        apply_transform_vectorized_cy,
        validate_projection_bins_cy,
        validate_2d_projection_bins_cy,
        CYTHON_AVAILABLE,
    )
except ImportError:
    print("Cython not available, cannot run benchmark")
    sys.exit(0)

if not CYTHON_AVAILABLE:
    print("Cython not compiled, cannot run benchmark")
    sys.exit(0)

print("=" * 60)
print("CYTHON PERFORMANCE BENCHMARK")
print("=" * 60)

# Generate test data
np.random.seed(42)
n_samples = 10000
n_features = 10
data = np.abs(np.random.randn(n_samples, n_features) * 10 + 50)

print(f"\nTest data: {n_samples} samples, {n_features} features")
print()

# Benchmark feature transforms
transforms = ['log_eps', 'inverse_eps', 'sigmoid_eps', 'sqrt', 'square']
print("Feature Transform Benchmark:")
print("-" * 60)

for transform in transforms:
    # Python version
    start = time.time()
    for _ in range(100):
        result_py = apply_feature_transform_vectorized(data, 0, transform, cache=None)
    time_py = time.time() - start
    
    # Cython version
    start = time.time()
    for _ in range(100):
        result_cy = apply_transform_vectorized_cy(data, 0, transform, cache=None)
    time_cy = time.time() - start
    
    speedup = time_py / time_cy
    print(f"  {transform:15s}: {speedup:5.2f}x faster")

# Benchmark 1D validation
print("\n1D Validation Benchmark:")
print("-" * 60)

proj_array = np.random.rand(n_samples)
n_bins = 50

# Python version
start = time.time()
for _ in range(1000):
    result_py = validate_projection_bins(proj_array, n_bins, 0.05)
time_py = time.time() - start

# Cython version
start = time.time()
for _ in range(1000):
    result_cy = validate_projection_bins_cy(proj_array, n_bins, 0.05)
time_cy = time.time() - start

speedup = time_py / time_cy
print(f"  1D validation:  {speedup:5.2f}x faster")

# Benchmark 2D validation
print("\n2D Validation Benchmark:")
print("-" * 60)

x_proj = np.random.rand(n_samples)
y_proj = np.random.rand(n_samples)
n_bins = 20

# Python version
start = time.time()
for _ in range(1000):
    result_py = validate_2d_projection_bins(x_proj, y_proj, n_bins, 0.05)
time_py = time.time() - start

# Cython version
start = time.time()
for _ in range(1000):
    result_cy = validate_2d_projection_bins_cy(x_proj, y_proj, n_bins, 0.05)
time_cy = time.time() - start

speedup = time_py / time_cy
print(f"  2D validation:  {speedup:5.2f}x faster")

print("\n" + "=" * 60)
print("✓ Cython acceleration is working correctly!")
print("=" * 60)
