"""
Setup script for building Cython extensions for CHIRP projection analysis.

To build the extensions in-place (for development):
    python setup.py build_ext --inplace

To build and install:
    pip install .

To build with optimization:
    python setup.py build_ext --inplace --force
"""

import sys
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Check if Cython is available
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython not found. Cython extensions will not be built.")
    print("Install Cython with: pip install cython")

# Check if NumPy is available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    numpy_include = np.get_include()
except ImportError:
    NUMPY_AVAILABLE = False
    numpy_include = ""
    print("Warning: NumPy not found. Please install NumPy before building.")
    print("Install NumPy with: pip install numpy")

# Define extensions only if Cython and NumPy are available
ext_modules = []

if CYTHON_AVAILABLE and NUMPY_AVAILABLE:
    # Compiler arguments for optimization
    extra_compile_args = ['-O3']
    if sys.platform != 'win32':
        extra_compile_args.append('-march=native')  # Use native CPU optimizations on Unix
    
    # Define Cython extensions
    extensions = [
        Extension(
            "chirp_cython.feature_transforms_cy",
            ["chirp_cython/feature_transforms_cy.pyx"],
            include_dirs=[numpy_include],
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        ),
        Extension(
            "chirp_cython.projection_vectorized_cy",
            ["chirp_cython/projection_vectorized_cy.pyx"],
            include_dirs=[numpy_include],
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        ),
        Extension(
            "chirp_cython.validation_cy",
            ["chirp_cython/validation_cy.pyx"],
            include_dirs=[numpy_include],
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        ),
    ]
    
    # Cythonize with compiler directives for optimization
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'embedsignature': True,
        },
        annotate=False,  # Set to True to generate HTML annotation files
    )

# Custom build_ext command to handle build failures gracefully
class BuildExtGraceful(build_ext):
    """Custom build_ext that doesn't fail the entire build if extensions fail."""
    
    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f"\nWarning: Failed to build Cython extensions: {e}")
            print("The package will still work using pure Python implementations.")
    
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
            print(f"✓ Successfully built {ext.name}")
        except Exception as e:
            print(f"✗ Failed to build {ext.name}: {e}")

# Setup configuration
setup(
    name='chirp_projection_analysis',
    version='1.0.0',
    description='CHIRP Projection Analysis with Cython Acceleration',
    author='CHIRP Team',
    packages=['chirp_python', 'chirp_cython'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtGraceful},
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'psutil>=5.8.0',
    ],
    extras_require={
        'cython': ['Cython>=0.29.0'],
    },
    python_requires='>=3.7',
    zip_safe=False,
)
