"""
custom_ops/build.py
编译 CUDA 算子: .cu + .cpp → .so 动态库
用法: python -m custom_ops.build
"""

import os
import subprocess
import sys
import torch
from torch.utils.cpp_extension import CUDAExtension, load

EXT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(EXT_DIR, "build")
MARKER = os.path.join(EXT_DIR, ".built")


def build():
    """编译定制 CUDA 算子, 返回加载路径"""
    module = load(
        name="fused_cuda_ops",
        sources=[
            os.path.join(EXT_DIR, "fused_ops.cpp"),
            os.path.join(EXT_DIR, "fused_layernorm.cu"),
            os.path.join(EXT_DIR, "fused_softmax_mask.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        build_directory=BUILD_DIR,
        verbose=True,
    )
    so_path = module.__file__
    with open(MARKER, "w") as f:
        f.write(so_path + "\n")
    print(f"\n编译成功: {so_path}")
    return so_path


if __name__ == "__main__":
    build()
