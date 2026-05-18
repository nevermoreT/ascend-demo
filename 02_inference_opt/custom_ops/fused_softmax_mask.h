/**
 * fused_softmax_mask.h
 * Fused Causal Softmax CUDA 算子头文件
 * 将上三角 mask (-inf) + softmax 合并为单次 kernel
 */

#pragma once
#include <torch/extension.h>

torch::Tensor fused_softmax_mask_cuda(torch::Tensor logits);
