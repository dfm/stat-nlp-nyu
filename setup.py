#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
    name="nlp",
    ext_modules=[
        Extension("nlp._edit", ["nlp/_edit_dist.c"]),
        Extension("nlp._maxent", ["nlp/_maxent.c"],
                  include_dirs=get_numpy_include_dirs(),
                  libraries=["lbfgs"],
                  library_dirs=["/usr/local/lib"],
                  extra_compile_args=["-fopenmp"],
                  extra_link_args=["-fopenmp", "-Wl,-rpath",
                                   "-Wl,/usr/local/lib"]),
    ],
)
