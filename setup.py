"""
Setup script for building Cython extensions for CHIRP projection analysis.

This setup file configures compilation of performance-critical modules:
- feature_transforms_cy: Vectorized feature transformations (5-10x speedup)
- projection_vectorized_cy: Projection computation (3-5x speedup)
- validation_cy: Bin validation functions (5-8x speedup)

Installation:
    python setup.py build_ext --inplace
    
Or with pip:
    pip install -e .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define Cython extensions
extensions = [
    Extension(
        "chirp_cython.feature_transforms_cy",
        ["chirp_cython/feature_transforms_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "chirp_cython.projection_vectorized_cy",
        ["chirp_cython/projection_vectorized_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
    Extension(
        "chirp_cython.validation_cy",
        ["chirp_cython/validation_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
]

setup(
    name="chirp_cython",
    version="1.0.0",
    description="Cython-accelerated CHIRP projection analysis modules",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        }
    ),
    zip_safe=False,
)
