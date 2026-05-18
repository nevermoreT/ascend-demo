/**
 * fused_layernorm.cu
 * Fused LayerNorm CUDA kernel 实现
 *
 * PyTorch 标准实现需要 4 次 kernel 启动:
 *   1. 计算 mean
 *   2. 计算 variance
 *   3. (x - mean) / sqrt(var + eps)
 *   4. * weight + bias
 * 每次都有全局显存读写的开销
 *
 * 本实现: 单次 kernel 完成全部计算
 *   每个 thread block 处理一行向量
 *   利用 shared memory 存储中间结果, 避免重复读写全局显存
 */

#include "fused_layernorm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

/**
 * FusedLayerNorm 前向 kernel
 *
 * 每个 block 处理一行 (hidden_size 个元素)
 * grid: (batch * seq_len) 个 block, 每个 block 有 hidden_size 个 thread
 *
 * 计算步骤 (全部在 register + shared memory 中完成):
 *   1. 每个 thread 读取自己负责的元素 → shared memory 求和得到 mean
 *   2. 每个 thread 计算 (x - mean)^2 → shared memory 求和得到 variance
 *   3. 每个 thread 计算 (x - mean) / sqrt(var + eps) * weight + bias
 *   4. 写回结果到全局显存
 */
template <typename scalar_t>
__global__ void fused_layernorm_kernel(
    const scalar_t* __restrict__ input,    // (N, H) 展平为 1D
    const scalar_t* __restrict__ weight,   // (H,)
    const scalar_t* __restrict__ bias,     // (H,)
    scalar_t* __restrict__ output,         // (N, H)
    const int hidden_size,
    const double eps
) {
    // 每个 block 处理一行, row_idx = 哪一行
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row_idx * hidden_size;

    // shared memory: 同一个 block 内所有 thread 共享
    // 用于归约求和 (mean 和 variance)
    extern __shared__ char smem_char[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_char);

    // --- Step 1: 计算 mean ---
    // 每个 thread 累加自己负责的元素
    scalar_t thread_sum = static_cast<scalar_t>(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        thread_sum += input[row_offset + i];
    }
    smem[tid] = thread_sum;
    __syncthreads();  // 等待所有 thread 完成写入

    // Block-level 归约求和 (二叉树归约)
    // 每轮减半参与归约的 thread 数, 最终 smem[0] = 总和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    scalar_t mean = smem[0] / hidden_size;
    __syncthreads();

    // --- Step 2: 计算 variance ---
    // 每个 thread 累加 (x - mean)^2
    scalar_t thread_var = static_cast<scalar_t>(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        scalar_t diff = input[row_offset + i] - mean;
        thread_var += diff * diff;
    }
    smem[tid] = thread_var;
    __syncthreads();

    // 再次归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    scalar_t inv_std = rsqrt(smem[0] / hidden_size + static_cast<scalar_t>(eps));

    // --- Step 3+4: 归一化 + 缩放偏移 + 写回 ---
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        scalar_t x_hat = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = x_hat * weight[i] + bias[i];
    }
}


torch::Tensor fused_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    // 输入形状: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
    const int hidden_size = input.size(input.dim() - 1);
    const int64_t rows = input.numel() / hidden_size;

    // 确保 contiguous (内存连续)
    auto input_contiguous = input.contiguous();
    auto weight_contiguous = weight.contiguous();
    auto bias_contiguous = bias.contiguous();

    // 分配输出 tensor
    auto output = torch::empty_like(input_contiguous);

    // CUDA launch 配置
    // grid: rows 个 block (每行一个)
    // block: hidden_size 个 thread (如果 hidden_size > 1024 则分批处理)
    const int threads = std::min(hidden_size, 1024);
    const int blocks = rows;
    // shared memory 大小: threads 个 scalar_t 元素
    const int smem_size = threads * input.element_size();

    // 使用 AT_DISPATCH 浮点类型宏, 自动处理 float/half/bfloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input_contiguous.scalar_type(), "fused_layernorm_kernel", ([&] {
            fused_layernorm_kernel<scalar_t><<<blocks, threads, smem_size>>>(
                input_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                hidden_size,
                eps
            );
        })
    );

    // 检查 CUDA 错误
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
