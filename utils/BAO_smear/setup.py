
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extension = Extension(
    "cython_filter",
    sources=["filter_funcs.pyx", "interpolation.c"],
    libraries=["gsl", "gslcblas", "m"],
    library_dirs=["/home/grads/miniconda/envs/joao/lib/python3.11/site-packages/numpy/core/lib"],
    include_dirs=["/home/grads/miniconda/envs/joao/lib/python3.11/site-packages/numpy/core/include"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='Filter Functions',
    ext_modules=cythonize([extension]),
    zip_safe=False,
)
