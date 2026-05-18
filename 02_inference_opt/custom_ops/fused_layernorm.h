/**
 * fused_layernorm.h
 * Fused LayerNorm CUDA 算子头文件
 * 将 均值→方差→归一化→缩放偏移 合并为单次 kernel
 */

#pragma once
#include <torch/extension.h>

torch::Tensor fused_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);
