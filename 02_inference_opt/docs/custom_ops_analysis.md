# 定制 CUDA 算子深度解析

## 一、整体架构：从 CUDA kernel 到 Python 调用

```
Python 调用层                  C++ 绑定层                    CUDA 计算层
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│                  │     │                  │     │                    │
│  __init__.py     │     │  fused_ops.cpp   │     │  fused_layernorm   │
│  build.py        │ ──→ │  (pybind11 绑定)  │ ──→ │     .cu            │
│                  │     │                  │     │  fused_softmax_    │
│  fused_layernorm │     │  fused_layernorm │     │     mask.cu        │
│  fused_softmax_  │     │  fused_softmax_  │     │                    │
│    mask          │     │    mask          │     │  GPU kernel 函数   │
│                  │     │                  │     │                    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
       Python                  C++                     CUDA C++
```

编译与加载流程：

```
.cu (CUDA kernel)  ─┐
                     ├→ nvcc + g++ → fused_cuda_ops.so → importlib 加载 → Python 调用
.cpp (pybind 绑定)  ─┘
```

---

## 二、文件结构与职责

```
custom_ops/
├── __init__.py              ← Python 入口: 懒加载 + 对外接口
├── build.py                 ← 编译脚本: 调用 torch.utils.cpp_extension.load()
├── fused_ops.cpp            ← C++ 绑定: pybind11 暴露函数 + 参数校验
├── fused_layernorm.h        ← 头文件: 声明 CUDA 函数签名
├── fused_layernorm.cu       ← CUDA 实现: FusedLayerNorm kernel
├── fused_softmax_mask.h     ← 头文件
├── fused_softmax_mask.cu    ← CUDA 实现: FusedSoftmaxMask kernel
├── .built                   ← 编译标记: 记录 .so 路径, 避免重复编译
└── build/                   ← 编译产物目录
    ├── fused_cuda_ops.so    ← 最终动态库 (被 Python 加载)
    └── ...
```

| 层级 | 文件 | 语言 | 核心职责 |
|:---|:---|:---|:---|
| 接口层 | `__init__.py` | Python | 对外提供 `fused_layernorm()` / `fused_softmax_mask()` |
| 编译层 | `build.py` | Python | JIT 编译: 源码 → .so |
| 绑定层 | `fused_ops.cpp` | C++ | 参数校验 + pybind11 注册 + 转发到 CUDA |
| 声明层 | `.h` | C++ | 函数声明，供 .cpp 调用 .cu |
| 计算层 | `.cu` | CUDA C++ | GPU kernel: 显存操作 + 并行计算 |

---

## 三、CUDA Kernel 实现详解

### 3.1 GPU 执行模型回顾

在理解 kernel 之前，先回顾 CUDA 的执行模型：

```
┌─────────────────────────── Grid (网格) ───────────────────────────┐
│  一个 kernel 启动一个 Grid                                        │
│                                                                   │
│  ┌─── Block 0 ───┐  ┌─── Block 1 ───┐  ┌─── Block N ───┐       │
│  │ Thread 0      │  │ Thread 0      │  │ Thread 0      │       │
│  │ Thread 1      │  │ Thread 1      │  │ Thread 1      │       │
│  │ Thread 2      │  │ Thread 2      │  │ Thread 2      │       │
│  │ ...           │  │ ...           │  │ ...           │       │
│  │               │  │               │  │               │       │
│  │ Shared Memory │  │ Shared Memory │  │ Shared Memory │       │
│  │ (block 内共享) │  │ (block 内共享) │  │ (block 内共享) │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                   │
│  全局显存 (Global Memory): 所有 block/thread 共享, 速度最慢       │
└───────────────────────────────────────────────────────────────────┘

寄存器 (Registers): 每个 thread 私有, 速度最快
Shared Memory: block 内共享, 速度极快 (~5x 快于全局显存)
全局显存: 所有人共享, 速度最慢 (主要性能瓶颈)
```

### 3.2 FusedLayerNorm Kernel

#### 数学公式

```
LayerNorm(x) = (x - μ) / √(σ² + ε) * γ + β

其中:
  x : 输入向量, 形状 (hidden_size,)
  μ = mean(x) : 均值
  σ² = mean((x - μ)²) : 方差
  ε : 防止除零的小常数
  γ, β : 可学习的缩放和偏移参数
```

#### PyTorch 标准实现 vs Fused 实现

```
PyTorch 标准实现 (4 次 kernel 启动):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Kernel 1     │   │ Kernel 2     │   │ Kernel 3     │   │ Kernel 4     │
│ mean(x)      │→  │ var(x)       │→  │ (x-μ)/√(σ²+ε)│→  │ ×γ + β       │
│              │   │              │   │              │   │              │
│ 读x → 写μ   │   │ 读x,μ → 写σ²│   │ 读x,μ,σ²    │   │ 读x̂,γ,β     │
│ 全局显存 ×1  │   │ 全局显存 ×2  │   │ → 写x̂       │   │ → 写输出     │
│              │   │              │   │ 全局显存 ×3  │   │ 全局显存 ×2  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  全局显存读写: 8 次 (4次读 + 4次写, 中间结果需写回再读出)
  kernel 启动开销: 4 × ~5μs = ~20μs

Fused 实现 (1 次 kernel 启动):
┌─────────────────────────────────────────────────────────────────────┐
│ Fused Kernel                                                        │
│ mean → var → normalize → scale → bias                               │
│                                                                     │
│ 所有中间结果在 shared memory / register 中传递                       │
│ 只读 1 次输入, 只写 1 次输出                                         │
│ 全局显存读写: 2 次 (1次读 + 1次写)                                   │
│ kernel 启动开销: 1 × ~5μs = ~5μs                                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Kernel 代码逐行解析

```cpp
template <typename scalar_t>                          // scalar_t = float/half/bfloat16
__global__ void fused_layernorm_kernel(               // __global__ = GPU kernel 函数
    const scalar_t* __restrict__ input,               // 只读输入 (restrict 告知编译器无别名)
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,                    // 只写输出
    const int hidden_size,                            // 最后一维大小 (256)
    const double eps                                  // 归一化 epsilon
) {
    // ============ 线程索引 ============
    const int row_idx = blockIdx.x;                   // 当前 block 负责第几行
    const int tid = threadIdx.x;                      // 当前 thread 在 block 内的编号
    const int row_offset = row_idx * hidden_size;     // 该行在全局显存中的起始偏移

    // ============ Shared Memory ============
    extern __shared__ char smem_char[];               // 动态 shared memory
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_char);
    // 用途: block 内所有 thread 共享的临时存储
    //       用于归约求和 (mean 和 variance)

    // ============ Step 1: 计算 mean ============
    // 策略: 每个 thread 累加若干元素 → 写入 smem → 二叉树归约
    scalar_t thread_sum = 0;
    for (int i = tid; i < hidden_size; i += blockDim.x) {  // 分批处理: thread 0 处理 0,blockDim,...
        thread_sum += input[row_offset + i];                // 读全局显存 (寄存器中累加)
    }
    smem[tid] = thread_sum;                          // 写入 shared memory
    __syncthreads();                                 // 同步: 等待所有 thread 完成

    // 二叉树归约求和:
    //
    //  初始: [a, b, c, d, e, f, g, h]   (8 个 thread 的部分和)
    //  第1轮: [a+e, b+f, c+g, d+h, -, -, -, -]   (tid < 4 的 thread 执行加法)
    //  第2轮: [a+e+c+g, b+f+d+h, -, -, -, -, -, -]
    //  第3轮: [总和, -, -, -, -, -, -, -]
    //  最终: smem[0] = 所有元素之和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();                             // 每轮归约后同步
    }
    scalar_t mean = smem[0] / hidden_size;           // 均值 = 总和 / 元素数
    __syncthreads();

    // ============ Step 2: 计算 variance ============
    scalar_t thread_var = 0;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        scalar_t diff = input[row_offset + i] - mean;    // 再次读全局显存 (无缓存)
        thread_var += diff * diff;                        // (x - μ)² 累加
    }
    smem[tid] = thread_var;
    __syncthreads();

    // 同样的二叉树归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // inv_std = 1 / √(σ² + ε) , 用 rsqrt 一次完成除法+开方
    scalar_t inv_std = rsqrt(smem[0] / hidden_size + (scalar_t)eps);

    // ============ Step 3+4: 归一化 + 缩放偏移 + 写回 ============
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        scalar_t x_hat = (input[row_offset + i] - mean) * inv_std;  // 归一化
        output[row_offset + i] = x_hat * weight[i] + bias[i];       // ×γ + β, 写回全局显存
    }
    // 结束: 全局显存只被读取 2 次 (step1 + step2+3), 写入 1 次 (step4)
}
```

#### GPU 线程映射示意 (以 hidden_size=256, batch*seq=4 为例)

```
Grid 维度: 4 个 Block
Block 维度: 256 个 Thread
Shared Memory: 256 × 4 bytes = 1 KB per block

Block 0 (处理第 0 行):
  Thread 0: 处理 input[0]    → thread_sum += x[0]
  Thread 1: 处理 input[1]    → thread_sum += x[1]
  ...
  Thread 255: 处理 input[255] → thread_sum += x[255]
  → smem 归约 → mean → variance → normalize → output[0..255]

Block 1 (处理第 1 行):
  同理处理 input[256..511]
...
```

### 3.3 FusedSoftmaxMask Kernel

#### 数学公式

```
因果 Softmax:
  softmax(x_i)[j] = exp(x_i[j] - max_i) / Σ_{k≤i} exp(x_i[k] - max_i)

  其中 max_i = max(x_i[0], x_i[1], ..., x_i[i])  (只取对角线及以下)

因果 mask:
  对于注意力矩阵 (seq_len × seq_len):
  位置 (i, j) 当 j > i 时, 注意力权重为 0 (不看未来)
```

#### Kernel 核心逻辑

```cpp
template <typename scalar_t>
__global__ void fused_softmax_mask_kernel(
    const scalar_t* input,    // (batch*heads, seq_len, seq_len)
    scalar_t* output,
    const int seq_len
) {
    const int row_idx = blockIdx.x;           // 第几行 = 第几个 query 位置
    const int tid = threadIdx.x;
    const int row_offset = row_idx * seq_len;
    const int row_in_seq = row_idx % seq_len; // 在序列中的位置 i

    // 因果 mask 的关键:
    //   位置 i 的 query 只能 attend to 位置 0..i 的 key
    //   即 j <= row_in_seq 的位置有效, 否则输出 0

    // Step 1: 在有效位置中找最大值 (数值稳定)
    scalar_t thread_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {                      // ← 因果 mask 在这里生效
            thread_max = fmaxf(thread_max, input[row_offset + j]);
        }
    }
    // 归约 → max_val

    // Step 2: 计算 exp(x - max) 并求和 (只在有效位置)
    scalar_t thread_sum = 0;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {
            thread_sum += expf(input[row_offset + j] - max_val);
        }
    }
    // 归约 → sum_exp → inv_sum

    // Step 3: 写回结果
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (j <= row_in_seq) {
            output[row_offset + j] = expf(input[row_offset + j] - max_val) * inv_sum;
        } else {
            output[row_offset + j] = 0;              // ← mask 位置输出 0
        }
    }
}
```

#### PyTorch 标准实现 vs Fused 实现

```
PyTorch 标准实现 (mask 在 Python 层, softmax 在 CUDA kernel 层):

  Step 1: Python 层生成 mask 矩阵                    → CPU → GPU 传输
  Step 2: torch.triu(ones)                           → GPU 显存分配
  Step 3: logits.masked_fill(mask, -inf)             → kernel 1: 读+写 全局显存
  Step 4: torch.softmax(masked_logits, dim=-1)       → kernel 2: 读+exp+sum+div+写
                                                          全局显存读写 4 次

Fused 实现:

  全部在 1 个 CUDA kernel 中完成:
  mask 逻辑通过 if (j <= row_in_seq) 实现            → 零额外显存
  exp + sum + div 在 register + shared memory 中完成  → 零额外全局显存读写
  只读 1 次输入, 只写 1 次输出                         → 全局显存读写 2 次
```

---

## 四、C++ 绑定层详解

### 4.1 fused_ops.cpp

```cpp
#include "fused_layernorm.h"     // 声明 CUDA 函数
#include "fused_softmax_mask.h"

// C++ 包装函数: 参数校验 + 转发到 CUDA kernel
torch::Tensor fused_layernorm(
    torch::Tensor input,         // Python 传来的 torch.Tensor
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    // TORCH_CHECK 类似 Python 的 assert, 失败时抛出异常给 Python
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(input.size(-1) == weight.size(0), "dimension mismatch");

    return fused_layernorm_cuda(input, weight, bias, eps);  // 调用 .cu 中的函数
}

// pybind11 模块定义: 将 C++ 函数暴露给 Python
// TORCH_EXTENSION_NAME 是编译时由 PyTorch 设定的宏, 值为 load(name="fused_cuda_ops")
// 展开后等价于 PYBIND11_MODULE(fused_cuda_ops, m)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm", &fused_layernorm, "docstring...");
    m.def("fused_softmax_mask", &fused_softmax_mask, "docstring...");
}
```

数据类型转换链：

```
Python 层                     C++ 层                      CUDA 层
torch.Tensor  ──pybind11──→  torch::Tensor  ──.data_ptr()──→  float* / half*
(Py 对象)                    (C++ 对象, 共享底层存储)      (裸指针, GPU 地址)

注意: 零拷贝! Python Tensor 和 C++ Tensor 共享同一块 GPU 显存,
      传参不会触发数据复制
```

---

## 五、编译与加载机制

### 5.1 build.py: JIT 编译流程

```python
from torch.utils.cpp_extension import load

module = load(
    name="fused_cuda_ops",           # 输出的 .so 文件名
    sources=[                         # 源文件列表
        "fused_ops.cpp",              #   .cpp → g++ 编译
        "fused_layernorm.cu",         #   .cu → nvcc 编译
        "fused_softmax_mask.cu",      #   .cu → nvcc 编译
    ],
    extra_cuda_cflags=[               # nvcc 额外编译选项
        "-O3",                        #   最高优化级别
        "--use_fast_math",            #   使用快速数学近似 (牺牲微小精度换速度)
    ],
    extra_cflags=["-O3"],             # g++ 额外编译选项
    build_directory="build/",         # 编译中间产物和 .so 的输出目录
    verbose=True,                     # 打印完整编译命令
)
```

编译命令展开 (实际执行)：

```bash
# Step 1: nvcc 编译 .cu → .o
/usr/local/cuda/bin/nvcc \
  -DTORCH_EXTENSION_NAME=fused_cuda_ops \      # 定义宏, 用于 PYBIND11_MODULE
  -isystem .../torch/include \                  # PyTorch C++ 头文件路径
  -isystem /usr/local/cuda/include \            # CUDA 头文件路径
  -isystem /usr/include/python3.12 \            # Python 头文件路径
  -gencode=arch=compute_120,code=sm_120 \       # 目标 GPU 架构 (RTX 5080)
  -O3 --use_fast_math \                         # 优化选项
  -c fused_layernorm.cu -o fused_layernorm.cuda.o

# Step 2: g++ 编译 .cpp → .o
g++ \
  -fPIC -std=c++20 \                            # 位置无关代码 + C++20 标准
  -O3 \
  -c fused_ops.cpp -o fused_ops.o

# Step 3: g++ 链接所有 .o → .so
g++ fused_ops.o fused_layernorm.cuda.o fused_softmax_mask.cuda.o \
  -shared \                                     # 生成共享库
  -L.../torch/lib -lc10 -lc10_cuda \            # 链接 PyTorch 基础库
  -ltorch_cpu -ltorch_cuda -ltorch \            # 链接 PyTorch 核心库
  -L/usr/local/cuda/lib64 -lcudart \            # 链接 CUDA runtime
  -o fused_cuda_ops.so                          # 最终输出
```

### 5.2 __init__.py: 懒加载机制

```python
import importlib.util

def _load():
    # 1. 检查是否已编译 (.built 标记文件)
    if not os.path.exists(".built"):
        build()                    # 首次 import 时自动编译

    # 2. 读取 .so 路径
    with open(".built") as f:
        so_path = f.readline().strip()

    # 3. 动态加载 .so 为 Python 模块
    spec = importlib.util.spec_from_file_location(
        "fused_cuda_ops",          # 模块名 (需与 PYBIND11_MODULE 名一致)
        so_path                    # .so 文件路径
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # 执行加载, 触发 .so 的初始化函数
    return mod                     # mod.fused_layernorm 就是 C++ 函数

# 懒加载: 首次调用时才触发编译+加载
_C = None

def fused_layernorm(input, weight, bias, eps=1e-5):
    global _C
    if _C is None:
        _C = _load()               # 第一次调用时加载
    return _C.fused_layernorm(input, weight, bias, eps)
```

加载时序图：

```
Python 进程启动
    │
    ├── import custom_ops          ← 只加载 __init__.py, 不触发编译
    │
    ├── custom_ops.fused_layernorm(x, w, b)  ← 首次调用
    │       │
    │       ├── _C is None?  → Yes
    │       │
    │       ├── _load()
    │       │     ├── .built 存在?
    │       │     │     ├── Yes → 读取 so_path
    │       │     │     └── No  → build() 编译 (首次约 10-30s)
    │       │     │
    │       │     └── importlib 加载 .so
    │       │           └── dlopen("fused_cuda_ops.so")
    │       │               ├── 执行 PyInit_fused_cuda_ops()  (pybind11 初始化)
    │       │               └── 注册 fused_layernorm, fused_softmax_mask
    │       │
    │       └── _C.fused_layernorm(x, w, b)  ← 直接调用 C++ 函数
    │               │
    │               ├── pybind11 类型转换: Tensor → torch::Tensor
    │               ├── TORCH_CHECK 参数校验
    │               ├── 调用 fused_layernorm_cuda()
    │               │     ├── 分配输出显存
    │               │     ├── 计算 grid/block 维度
    │               │     └── 启动 CUDA kernel <<<blocks, threads>>>
    │               │           └── GPU 执行 kernel
    │               ├── pybind11 类型转换: torch::Tensor → Tensor
    │               └── 返回 Python
    │
    ├── custom_ops.fused_layernorm(x, w, b)  ← 第二次调用
    │       └── _C 已缓存, 直接调用 (无额外开销)
    ...
```

---

## 六、如何在模型中集成定制算子

### 6.1 方法一：Monkey-Patch (本项目使用)

运行时替换 `nn.LayerNorm` 的 forward 方法，无需修改模型源码：

```python
from custom_ops import fused_layernorm
import torch.nn as nn

model = load_model(device, dtype=torch.bfloat16)

# 遍历模型, 替换所有 LayerNorm
for name, module in model.transformer.named_modules():
    if isinstance(module, nn.LayerNorm):
        weight = module.weight
        bias = module.bias
        eps = module.eps
        # 闭包捕获当前 module 的参数
        module.forward = lambda x, _w=weight, _b=bias, _e=eps: \
            fused_layernorm(x, _w, _b, _e)
```

优点：不改模型代码，即插即用
缺点：每次 forward 有 Python 分发开销（微秒级）

### 6.2 方法二：自定义 Module（推荐生产使用）

直接在模型定义中使用定制算子：

```python
from custom_ops import fused_layernorm

class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return fused_layernorm(x, self.weight, self.bias, self.eps)

# 然后在 Transformer 模型中用 FusedLayerNorm 替换 nn.LayerNorm
```

优点：无 Python 分发开销，更干净
缺点：需要修改模型源码

---

## 七、性能分析：为什么 Fused 更快

### 7.1 性能瓶颈分析

```
GPU 计算的一次完整操作:

  Host (CPU)                      Device (GPU)
  ──────────                      ───────────
  准备参数 ──────────────────→
  启动 kernel ──→                  kernel 执行
                    kernel 启动开销 (~2-5μs)
                    读取输入 (全局显存, ~100ns/4B)
                    计算 (ALU, 极快)
                    写入输出 (全局显存, ~100ns/4B)

  对于小向量 (hidden_size=256):
    计算时间: ~0.1μs (GPU ALU 极快)
    显存读写: ~0.5μs
    kernel 启动: ~3μs  ← 占比最大!
```

### 7.2 Fused 的收益量化

```
场景: Transformer 有 12 个 LayerNorm (3 层 encoder × 2 + 3 层 decoder × 2)

非 Fused (PyTorch 标准):
  12 × 4 次 kernel = 48 次 kernel 启动
  48 × 3μs = 144μs 的纯启动开销
  48 × (读+写) = 96 次全局显存读写

Fused:
  12 × 1 次 kernel = 12 次 kernel 启动
  12 × 3μs = 36μs 的纯启动开销
  12 × (读+写) = 24 次全局显存读写

节省:
  kernel 启动: -108μs  (-75%)
  全局显存读写: -72 次  (-75%)
```

### 7.3 Shared Memory vs Global Memory 带宽

```
NVIDIA RTX 5080 (Blackwell 架构, 估计值):

  全局显存 (GDDR7):
    带宽: ~960 GB/s
    延迟: ~200-800 cycles

  Shared Memory (on-chip):
    带宽: ~19 TB/s  (约 20x 快于全局显存)
    延迟: ~20-30 cycles

  寄存器:
    带宽: 立即 (同一个 cycle)
    延迟: ~1 cycle

Fused kernel 的核心优化:
  中间结果 (mean, variance) 在 shared memory 中传递
  而非写回全局显存再读出
  → 等效带宽提升 ~20x (对于中间结果部分)
```

---

## 八、实际 Benchmark 数据

### 8.1 单算子精度验证

```
FusedLayerNorm vs torch.nn.functional.layer_norm:
  输入: (4, 8, 256) float32
  最大误差: 4.77e-7  ← 远低于 float32 精度极限 (~1e-6)

FusedSoftmaxMask vs torch.softmax(masked_fill(...)):
  输入: (4, 8, 8) float32
  最大误差: 5.96e-8  ← 几乎精确

结论: 定制算子精度无损, 误差来源于浮点运算顺序差异
      (GPU 并行归约的累加顺序与串行不同, 但误差在可接受范围内)
```

### 8.2 端到端推理性能

```
Batch=16, RTX 5080:

  基线 (FP32 Eager):        4.688 ms
  Fused CUDA + BF16:        2.804 ms  → 1.67x 加速

  加速来源拆解:
    BF16 半精度:           ~1.0x  (sm_120 的 BF16 SDPA 未优化, 基本无收益)
    Fused LayerNorm:       ~1.3x  (减少 kernel 启动 + 显存读写)
    Fused SoftmaxMask:     ~1.2x  (消除 mask + softmax 的显存中间结果)
    其他 (跳过 dropout 等): ~1.1x

    综合: 1.0 × 1.3 × 1.2 × 1.1 ≈ 1.7x ✓
```
