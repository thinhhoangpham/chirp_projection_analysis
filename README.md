# CHIRP Projection Analysis

A high-performance implementation of CHIRP (Convex Hull Iterative Random Projection) analysis for classification and visualization of high-dimensional data.

## Features

- **Fast 2D Projection Generation**: Generate optimal 2D projection pairs for visualization
- **Multiple Validation Modes**: Support for incremental and final validation strategies
- **Cython Acceleration**: Optional 3-8x performance improvements with Cython
- **Automatic Fallback**: Gracefully falls back to pure Python if Cython unavailable
- **Flexible Transforms**: Support for 11+ feature transformations (log, sqrt, sigmoid, etc.)
- **Robust Caching**: Intelligent computation caching for improved performance

## Installation

### Basic Installation (Pure Python)

```bash
# Install dependencies
pip install numpy pandas matplotlib psutil

# Run the visualizer
python 2d_pairs_visualizer.py --help
```

### Installation with Cython Acceleration (Recommended)

For 3-8x performance improvements, build the Cython extensions:

```bash
# Install Cython build requirements
pip install -r requirements-cython.txt

# Build Cython extensions
python setup.py build_ext --inplace

# Verify installation
python -c "from chirp_cython import CYTHON_AVAILABLE; print(f'Cython available: {CYTHON_AVAILABLE}')"
```

Expected output:
```
✓ Cython-accelerated modules loaded successfully
Cython available: True
```

### Installation via pip

```bash
# Install with Cython support
pip install -e .

# Or install without Cython (pure Python only)
pip install -e . --no-build-isolation
```

## Usage

### Basic Usage

```bash
python 2d_pairs_visualizer.py \
    --csv data.csv \
    --class-column class \
    --output-dir output \
    --n-bins 20 \
    --validation-mode incremental
```

### Command Line Options

```
--csv FILE              Input CSV file with features and class labels
--class-column NAME     Name of the class column in CSV
--output-dir DIR        Directory for output files (default: output)
--n-bins N              Number of bins for validation (default: 20)
--validation-mode MODE  Validation strategy: 'incremental' or 'final' (default: incremental)
--min-occupancy RATIO   Minimum bin occupancy ratio (default: 0.05)
--epsilon VALUE         Epsilon for numerical stability (default: 1e-10)
```

### Python API

```python
from chirp_cython import (
    compute_projection_vectorized,
    validate_2d_projection_bins,
    CYTHON_AVAILABLE
)
from chirp_python.data_source import DataSource
import numpy as np

# Load your data
data = np.loadtxt('data.csv', delimiter=',')
ds = DataSource(data=data[:, :-1], class_values=data[:, -1].astype(int))

# Compute projection
wi = np.array([0, 1, 2])  # Feature indices
transforms = ['none', 'square', 'sqrt']
projection, bounds, valid_count = compute_projection_vectorized(
    ds, wi, transforms, len(data), normalize=True
)

# Validate projection
is_valid, occupied_bins = validate_2d_projection_bins(
    projection_x, projection_y, n_bins=20, min_occupancy_ratio=0.05
)

print(f"Using Cython: {CYTHON_AVAILABLE}")
print(f"Valid projection: {is_valid}, Occupied bins: {occupied_bins}")
```

## Performance

### Cython Speedups

When Cython extensions are built, you can expect the following performance improvements:

| Component | Speedup |
|-----------|---------|
| Feature Transforms | 5-10x |
| Projection Bounds | 3-5x |
| Array Filling | 3-5x |
| Bin Validation (1D) | 5-8x |
| Bin Validation (2D) | 5-8x |
| **Overall Pipeline** | **3-8x** |

Actual speedup depends on:
- Data size (larger datasets see better speedups)
- Number of features and transformations
- Validation complexity
- CPU architecture

### Benchmarking

To benchmark your specific workload:

```python
import time
from chirp_cython import CYTHON_AVAILABLE

print(f"Using Cython: {CYTHON_AVAILABLE}")

# Run your analysis
start = time.time()
# ... your code here ...
elapsed = time.time() - start

print(f"Elapsed time: {elapsed:.2f}s")
```

## Testing

### Run Equivalence Tests

Verify that Cython and Python implementations produce identical results:

```bash
python tests/test_cython_equivalence.py
```

Expected output:
```
✓ PASSED: Feature Transforms
✓ PASSED: Projection Bounds
✓ PASSED: Array Filling
✓ PASSED: Vectorized Projection
✓ PASSED: 1D Bin Validation
✓ PASSED: 2D Bin Validation

✓ All equivalence tests PASSED!
```

## Architecture

```
chirp_projection_analysis/
├── chirp_python/              # Pure Python implementations
│   ├── feature_transforms.py  # Feature transformations
│   ├── projection_vectorized.py  # Projection computations
│   ├── validation.py          # Bin validation
│   ├── data_source.py         # Data management
│   └── ...                    # Other modules
├── chirp_cython/              # Cython-accelerated implementations
│   ├── __init__.py           # Auto-fallback wrapper
│   ├── feature_transforms_cy.pyx
│   ├── projection_vectorized_cy.pyx
│   ├── validation_cy.pyx
│   └── README.md             # Cython-specific docs
├── tests/
│   └── test_cython_equivalence.py  # Equivalence tests
├── 2d_pairs_visualizer.py    # Main visualization script
├── setup.py                  # Cython build configuration
├── pyproject.toml            # Modern packaging config
└── requirements-cython.txt   # Cython build dependencies
```

## Compatibility

### Python Versions
- Python 3.7+
- Tested on Python 3.8, 3.9, 3.10, 3.11, 3.12

### Operating Systems
- Linux (recommended for best Cython performance)
- macOS
- Windows (requires Visual C++ Build Tools for Cython)

### Dependencies
- **Required**: numpy, pandas, matplotlib, psutil
- **Optional**: Cython (for acceleration)

## Troubleshooting

### Cython Not Building

**Error: `Cython not found`**
```bash
pip install Cython
```

**Error: `numpy not found`**
```bash
pip install numpy
```

**Error: Compiler not found (Windows)**
- Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/)

**Error: Compiler not found (Linux)**
```bash
sudo apt-get install build-essential python3-dev
```

**Error: Compiler not found (macOS)**
```bash
xcode-select --install
```

### Cython Modules Not Loading

Check that `.so` files exist:
```bash
ls chirp_cython/*.so      # Linux/macOS
ls chirp_cython/*.pyd     # Windows
```

Rebuild with force flag:
```bash
python setup.py build_ext --inplace --force
```

### Different Results

If you see different results between Cython and Python, run equivalence tests:
```bash
python tests/test_cython_equivalence.py
```

All tests should pass. If not, please file an issue.

## Advanced Usage

### Custom Transformations

Add custom transformations by modifying `chirp_python/feature_transforms.py`:

```python
def apply_feature_transform_vectorized(data, feature_idx, transform_type, cache=None):
    # ... existing code ...
    elif transform_type == 'my_custom_transform':
        result = custom_function(feature_data)
    # ... rest of code ...
```

### Caching Configuration

Adjust cache behavior in `chirp_python/computation_cache.py`:

```python
# Increase cache size
cache.max_cache_size = 10000

# Clear cache
cache.clear()

# Print cache statistics
print(f"Cache hits: {cache.cache_hits}")
print(f"Cache misses: {cache.cache_misses}")
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `python tests/test_cython_equivalence.py`
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/thinhhoangpham/chirp_projection_analysis.git
cd chirp_projection_analysis

# Install in development mode
pip install -e .

# Build Cython extensions
python setup.py build_ext --inplace

# Run tests
python tests/test_cython_equivalence.py
```

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]

## Acknowledgments

This implementation includes optimizations for:
- Vectorized NumPy operations
- Intelligent computation caching
- Optional Cython acceleration
- Robust numerical stability handling
