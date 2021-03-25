from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name="bvh2Geometric",
    ext_modules=cythonize("bvh2geometric.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)