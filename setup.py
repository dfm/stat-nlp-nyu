#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension

setup(
    name="nlp",
    ext_modules=[Extension("nlp._edit", ["nlp/_edit_dist.c"])],
)
