/**
 * fused_softmax_mask.cu
 * Fused Causal Softmax CUDA kernel 实现
 *
 * PyTorch 标准实现需要多步:
 *   1. masked_fill(上三角, -inf)  → 写回全局显存
 *   2. exp()                      → 写回全局显存
 *   3. sum()                      → 写回全局显存
 *   4. div()                      → 写回全局显存
 *
 * 本实现: 单次 kernel 完成 mask + exp + sum + div
 *   利用寄存器和 shared memory, 零额外显存读写
 */

#include "fused_softmax_mask.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

/**
 * Fused Causal Softmax kernel
 *
 * 每个 block 处理一行的 softmax
 * grid: (batch * num_heads * seq_len) 个 block
 *
 * 对于因果 mask: 位置 j 只有在 j <= i 时才参与 softmax (i=行号, j=列号)
 * 即: 对角线及以下保留, 以上设为 -inf
 */
template <typename scalar_t>
__global__ void fused_softmax_mask_kernel(
    const scalar_t* __restrict__ input,    // (N, S, S) 展平为 1D
    scalar_t* __restrict__ output,         // (N, S, S)
    const int seq_len
) {
    const int row_idx = blockIdx.x;   // 哪一行 (对应 query 位置 i)
    const int tid = threadIdx.x;
    const int row_offset = row_idx * seq_len;

    // 因果 mask: 只有 j <= row_in_seq 的位置有效
    // row_in_seq = 这一行在序列中的位置 (第几个 query)
    const int row_in_seq = row_idx % seq_len;

    extern __shared__ char smem_char[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_char);

    // --- Step 1: 找到有效位置中的最大值 (数值稳定的 softmax) ---
    // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    // 先找 max 防止 exp 溢出
    scalar_t thread_max = static_cast<scalar_t>(-FLT_MAX);
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {
            scalar_t val = input[row_offset + j];
            thread_max = fmaxf(thread_max, val);
        }
        // j > row_in_seq 的位置被 mask 掉, 不参与计算
    }
    smem[tid] = thread_max;
    __syncthreads();

    // Block 归约找最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    scalar_t max_val = smem[0];
    __syncthreads();

    // --- Step 2: 计算 exp(x - max) 并求和 ---
    scalar_t thread_sum = static_cast<scalar_t>(0);
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {
            scalar_t val = expf(input[row_offset + j] - max_val);
            thread_sum += val;
        }
    }
    smem[tid] = thread_sum;
    __syncthreads();

    // Block 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    scalar_t sum_exp = smem[0];
    scalar_t inv_sum = static_cast<scalar_t>(1) / (sum_exp + static_cast<scalar_t>(1e-12));

    // --- Step 3: 计算 softmax 结果并写回 ---
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {
            output[row_offset + j] = expf(input[row_offset + j] - max_val) * inv_sum;
        } else {
            output[row_offset + j] = static_cast<scalar_t>(0);  // mask 位置输出 0
        }
    }
}


torch::Tensor fused_softmax_mask_cuda(torch::Tensor logits) {
    // 输入形状: (batch*num_heads, seq_len, seq_len) 或 (seq_len, seq_len)
    const int seq_len = logits.size(-1);
    const int rows = logits.numel() / (seq_len * seq_len) * seq_len;

    auto logits_contiguous = logits.contiguous();
    auto output = torch::empty_like(logits_contiguous);

    const int threads = std::min(seq_len, 1024);
    const int blocks = rows;
    const int smem_size = threads * logits.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits_contiguous.scalar_type(), "fused_softmax_mask_kernel", ([&] {
            fused_softmax_mask_kernel<scalar_t><<<blocks, threads, smem_size>>>(
                logits_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                seq_len
            );
        })
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
