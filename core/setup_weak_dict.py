# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="cython_weakdict",
    ext_modules=cythonize("cython_weakdict.pyx"),
)
# pip install cython
# python setup_weak_dict.py build_ext --inplace  # Generates a `.so` (Linux) or `.pyd` (Windows)