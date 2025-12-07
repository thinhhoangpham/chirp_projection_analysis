# Cython Acceleration for CHIRP Projection Analysis

This directory contains Cython-accelerated versions of performance-critical computation modules for CHIRP projection analysis.

## Performance Improvements

Expected speedups for Cython-accelerated modules:
- **feature_transforms_cy**: 1.5-3x faster for complex transforms (sigmoid, logit)
- **validation_cy**: 1.3-1.5x faster for bin validation
- **projection_vectorized_cy**: 1.5-2x faster for projection computation
- **Overall pipeline**: 2-5x faster depending on data and configuration

## Installation

### Building Cython Extensions

1. **Install dependencies:**
   ```bash
   pip install -r requirements-cython.txt
   ```

2. **Build extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

   This will compile the `.pyx` files into `.so` shared libraries.

3. **Verify installation:**
   ```bash
   python -c "from chirp_cython import CYTHON_AVAILABLE; print(f'Cython available: {CYTHON_AVAILABLE}')"
   ```

### Alternative Installation

You can also install in development mode:
```bash
pip install -e .
```

## Usage

The Cython modules are designed to be drop-in replacements for the Python versions:

```python
# Automatic detection and fallback
from chirp_cython import (
    apply_transform_vectorized_cy,
    compute_bounds_cy,
    fill_array_cy,
    validate_projection_bins_cy,
    validate_2d_projection_bins_cy,
    CYTHON_AVAILABLE
)

if CYTHON_AVAILABLE:
    print("Using Cython-accelerated functions")
else:
    print("Using pure Python fallback")
```

The `2d_pairs_visualizer.py` script automatically uses Cython functions if available.

## Graceful Fallback

If Cython extensions are not compiled, the code automatically falls back to pure Python implementations from `chirp_python/`. This ensures:
- ✓ No breaking changes to existing workflows
- ✓ Works on systems without Cython or C compiler
- ✓ Development continues smoothly without compilation

## Testing

### Run Equivalence Tests

Verify that Cython implementations produce identical results to Python:
```bash
pytest tests/test_cython_equivalence.py -v
```

All 19 tests should pass, confirming exact behavioral compatibility.

### Run Performance Benchmark

Compare Cython vs Python performance:
```bash
python tests/benchmark_cython.py
```

## Module Details

### feature_transforms_cy.pyx

Cython implementation of feature transformation functions:
- `apply_transform_vectorized_cy()` - Vectorized transforms
- `set_epsilon_cy()` / `get_epsilon_cy()` - Epsilon management

**Transforms supported:**
- Basic: `square`, `sqrt`, `none`
- Logarithmic: `log`, `log_eps`
- Inverse: `inverse`, `inverse_eps`
- Sigmoid: `sigmoid`, `sigmoid_eps`
- Logit: `logit`, `logit_eps`

**Optimizations:**
- C math functions (`log`, `sqrt`, `exp`)
- Typed memory views for NumPy arrays
- Eliminated Python loops where possible

### projection_vectorized_cy.pyx

Cython implementation of projection computation:
- `compute_bounds_cy()` - Compute min/max bounds
- `fill_array_cy()` - Fill normalized projection array [0,1]

**Optimizations:**
- C-level loops for projection computation
- Direct array access via typed memoryviews
- Integrated with existing caching system

### validation_cy.pyx

Cython implementation of bin validation:
- `validate_projection_bins_cy()` - 1D bin occupancy validation
- `validate_2d_projection_bins_cy()` - 2D bin occupancy validation

**Optimizations:**
- C-level loops for bin index computation
- Efficient linear indexing for 2D bins
- NumPy unique for fast counting

## Compiler Directives

The Cython modules use these optimization directives:
```cython
# cython: boundscheck=False      # Skip array bounds checking
# cython: wraparound=False        # Disable negative indexing
# cython: cdivision=True          # C-style division (no zero check)
# cython: initializedcheck=False  # Skip initialization checks
```

These provide significant speedups but require careful coding to avoid bugs.

## Build Artifacts

The following files are auto-generated during compilation:
- `*.c` - Generated C code (ignored by git)
- `*.so` - Compiled shared libraries (ignored by git)
- `build/` - Build directory (ignored by git)

These are excluded from version control via `.gitignore`.

## Troubleshooting

### Build Errors

If you encounter build errors:

1. **Check compiler installation:**
   ```bash
   gcc --version
   ```

2. **Check NumPy installation:**
   ```bash
   python -c "import numpy; print(numpy.get_include())"
   ```

3. **Clean and rebuild:**
   ```bash
   rm -rf build chirp_cython/*.c chirp_cython/*.so
   python setup.py build_ext --inplace
   ```

### Import Errors

If Cython modules fail to import, the code automatically falls back to Python. Check:
```bash
python -c "from chirp_cython import CYTHON_AVAILABLE; print(CYTHON_AVAILABLE)"
```

If `False`, either:
- Cython extensions were not compiled
- Compilation failed silently
- `.so` files are missing or incompatible

## Development

### Adding New Cython Modules

1. Create `.pyx` file in `chirp_cython/`
2. Add extension to `setup.py`
3. Update `chirp_cython/__init__.py` with imports
4. Add equivalence tests to `tests/test_cython_equivalence.py`
5. Rebuild: `python setup.py build_ext --inplace`

### Testing Changes

After modifying `.pyx` files:
```bash
# Rebuild
python setup.py build_ext --inplace

# Run tests
pytest tests/test_cython_equivalence.py -v

# Benchmark
python tests/benchmark_cython.py
```

## Requirements

- Python >= 3.8
- NumPy >= 1.19.0
- Cython >= 0.29.0 (for building)
- C compiler (gcc, clang, or MSVC)

## License

Same license as the main CHIRP project.
