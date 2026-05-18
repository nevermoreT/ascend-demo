/**
 * fused_layernorm.cpp
 * PyTorch C++ 扩展绑定层
 * 将 CUDA kernel 注册为 Python 可调用的算子
 */

#include "fused_layernorm.h"
#include "fused_softmax_mask.h"

// 注册 FusedLayerNorm 算子
// Python 端调用: custom_ops.fused_layernorm(input, weight, bias, eps)
torch::Tensor fused_layernorm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    // 参数检查
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(input.size(-1) == weight.size(0), "input last dim must match weight size");
    TORCH_CHECK(input.size(-1) == bias.size(0), "input last dim must match bias size");

    return fused_layernorm_cuda(input, weight, bias, eps);
}

// 注册 FusedSoftmaxMask 算子
// Python 端调用: custom_ops.fused_softmax_mask(logits)
torch::Tensor fused_softmax_mask(
    torch::Tensor logits
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() >= 2, "logits must be at least 2D");

    return fused_softmax_mask_cuda(logits);
}

// 使用 PYBIND11_MODULE 将 C++ 函数暴露给 Python
// 通过 torch.ops.load_library() 加载 .so 后可直接 import 使用
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm", &fused_layernorm,
          "Fused LayerNorm: mean + var + normalize + scale + bias in one kernel");
    m.def("fused_softmax_mask", &fused_softmax_mask,
          "Fused Causal Softmax: apply upper-triangular mask + softmax in one kernel");
}
