"""
custom_ops 包入口
自动编译并加载 CUDA 算子, 提供 Python 接口
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_MARKER = os.path.join(_PKG_DIR, ".built")


def _load():
    if not os.path.exists(_BUILD_MARKER):
        from custom_ops.build import build
        build()

    with open(_BUILD_MARKER) as f:
        so_path = f.readline().strip()

    if not os.path.exists(so_path):
        from custom_ops.build import build
        so_path = build()

    import importlib.util
    spec = importlib.util.spec_from_file_location("fused_cuda_ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C = None


def _get_module():
    global _C
    if _C is None:
        _C = _load()
    return _C


def fused_layernorm(input, weight, bias, eps=1e-5):
    return _get_module().fused_layernorm(input, weight, bias, eps)


def fused_softmax_mask(logits):
    return _get_module().fused_softmax_mask(logits)
