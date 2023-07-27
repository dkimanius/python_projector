#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from torch.utils import cpp_extension

project_root = os.path.realpath(os.path.dirname(__file__))

include_dirs = [project_root]

ext_module = cpp_extension.CUDAExtension(
    'diffractive_cpp',
    [
        'diffractive/cpp/dense_real_linear.cpp',
        'diffractive/cpp/dense_real_linear_kernels.cpp',
        'diffractive/cpp/dense_real_linear_kernels_cuda.cu',
        'diffractive/cpp/dense_complex_linear.cpp',
        'diffractive/cpp/dense_complex_linear_kernels.cpp',
        'diffractive/cpp/dense_complex_linear_kernels_cuda.cu',
        'diffractive/cpp/vtk.cpp',
        'diffractive/cpp/shift_center.cpp',
        'diffractive/cpp/main.cpp'
    ],
    include_dirs=include_dirs
)

setup(
    name='diffractive_cpp',
    ext_modules=[ext_module],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
