#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: setup.py

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy as np

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入配置
from dreamplace.configure import configure

# 设置头文件目录
include_dirs = [
    current_dir + "/src",
    np.get_include()
]

# 设置源文件目录
src_cpp = [
    current_dir + "/src/rc_timing.cpp"
]

# 编译选项
extra_compile_args = {
    "cxx": ["-std=c++14", "-O3"],
    "nvcc": ["-std=c++14", "-O3"]
}

# 创建扩展模块
modules = []

# C++扩展模块
modules.append(
    CppExtension(
        name="dreamplace.ops.rc_timing.rc_timing_cpp",
        sources=src_cpp,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args["cxx"]
    )
)

# 如果有CUDA支持，添加CUDA扩展模块
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    src_cuda = [
        current_dir + "/src/rc_timing_cuda.cpp",
        current_dir + "/src/rc_timing_cuda_kernel.cu"
    ]
    
    modules.append(
        CUDAExtension(
            name="dreamplace.ops.rc_timing.rc_timing_cuda",
            sources=src_cuda,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": extra_compile_args["cxx"],
                "nvcc": extra_compile_args["nvcc"]
            }
        )
    )

# 设置安装
setup(
    name="rc_timing",
    ext_modules=modules,
    cmdclass={
        "build_ext": BuildExtension
    }
) 