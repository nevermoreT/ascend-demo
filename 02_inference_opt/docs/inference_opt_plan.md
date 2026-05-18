# Transformer 推理优化方案设计

## 一、任务与数据集

| 项目 | 选择 | 理由 |
|------|------|------|
| **任务** | 英→德 机器翻译 (Seq2Seq) | 经典 Transformer 任务，Encoder-Decoder 完整展示 |
| **数据集** | Multi30k (~30k 句对) | 小巧、下载快、RTX 5080 几分钟可训完 |
| **词表** | SentencePiece BPE (共用 8000 词) | 子词分割，兼顾效率和覆盖 |

## 二、模型规格

| 参数 | 值 | 说明 |
|------|----|------|
| d_model | 256 | 适中大小，方便观察优化效果 |
| nhead | 8 | 标准配置 |
| num_layers | 3 | 比迷你版大，比原版小 |
| dim_feedforward | 1024 | FFN 中间层 |
| 总参数量 | ~10M | RTX 5080 轻松运行 |

## 三、项目文件结构

```
ascend-tutorial/inference_opt/
├── 01_train.py              # 训练模型，保存 checkpoint
├── 02_baseline.py           # 基线推理 (FP32 eager mode)
├── 03_optimized.py          # 优化推理 + 速度对比
├── dataset.py               # 数据加载与预处理
├── model.py                 # Transformer 模型定义
├── custom_ops/              # 定制算子目录
│   ├── __init__.py          # Python 绑定入口
│   ├── build.py             # 编译脚本 (调用 CUDAExtension)
│   ├── fused_layernorm.h    # C++ 头文件
│   ├── fused_layernorm.cpp  # C++ 接口 (torch 扩展注册)
│   ├── fused_layernorm.cu   # CUDA kernel 实现 (核心算子)
│   ├── fused_softmax_mask.h
│   ├── fused_softmax_mask.cpp
│   ├── fused_softmax_mask.cu
│   └── fused_qkv.h / .cpp / .cu
└── benchmarks.md            # 最终性能对比报告
```

各层职责：

| 文件 | 语言 | 职责 |
|------|------|------|
| `*.cu` | CUDA C++ | 核心 kernel 实现，直接操作 GPU 线程和显存 |
| `*.cpp` | C++ | PyTorch 扩展绑定，用 `PYBIND11_MODULE` 注册算子 |
| `build.py` | Python | 调用 `CUDAExtension` 编译 `.cu/.cpp` → `.so` 动态库 |
| `__init__.py` | Python | `import custom_ops` 后可直接调用 |

编译流程：

```
.cu + .cpp → nvcc + g++ → custom_ops.so → Python import 加载
```

## 四、优化策略（由易到难，共 5 级）

```
Level 0 ── FP32 Eager Mode (基线)
  │
Level 1 ── FP16 / BF16 半精度推理
  │         精度略降，显存减半，吞吐翻倍
  │
Level 2 ── torch.compile (图模式)
  │         算子融合 + 内存优化，零代码改动
  │
Level 3 ── 定制 CUDA 算子 (custom_ops/)
  │         ├─ Fused Softmax+Mask (替代逐步计算)
  │         ├─ Fused LayerNorm (一次 kernel 完成均值/方差/归一化)
  │         └─ Fused QKV Projection (合并三次矩阵乘为一次)
  │
Level 4 ── ONNX Runtime + TensorRT (可选)
            导出 ONNX → TensorRT 引擎，极限推理速度
```

## 五、定制算子详解

| 算子 | 替代对象 | 优化原理 |
|------|----------|----------|
| **FusedSoftmaxMask** | `torch.softmax(masked_fill(...))` | 将 mask 填充和 softmax 合并为一次 kernel 启动，减少 GPU 全局内存读写 |
| **FusedLayerNorm** | PyTorch `nn.LayerNorm` | 将均值、方差、归一化、缩放合并为一次 kernel，避免中间结果写回显存 |
| **FusedQKVProjection** | 3 次 `nn.Linear` | 将 Q/K/V 三个投影合并为一次大矩阵乘，利用 GPU 的张量核心加速 |

### FusedLayerNorm CUDA 实现思路

```cpp
// fused_layernorm.cu 核心逻辑
// PyTorch 原始实现: 多次 kernel 启动
//   1. kernel_mean:    计算均值 μ
//   2. kernel_var:     计算方差 σ²
//   3. kernel_norm:    (x - μ) / √(σ² + ε)
//   4. kernel_scale:   × γ + β
// 每次都要从全局显存读/写中间结果

// Fused 实现: 单次 kernel 完成全部计算
//   每个 thread block 处理一行向量
//   利用 shared memory 存储中间结果，避免全局显存读写
//   一次 kernel 启动 = 原来的 4 次
```

### FusedSoftmaxMask CUDA 实现思路

```cpp
// fused_softmax_mask.cu 核心逻辑
// PyTorch 原始实现:
//   1. masked_fill(-inf)  → 写回显存
//   2. exp()              → 写回显存
//   3. sum()              → 写回显存
//   4. div()              → 写回显存

// Fused 实现: 单次 kernel
//   在寄存器中完成 mask + exp + sum + div
//   零额外显存读写
```

### FusedQKVProjection CUDA 实现思路

```cpp
// fused_qkv.cpp 核心逻辑
// PyTorch 原始实现:
//   Q = x @ W_q  (一次 cublas GEMM)
//   K = x @ W_k  (一次 cublas GEMM)
//   V = x @ V_v  (一次 cublas GEMM)
//   = 3 次 kernel 启动 + 3 次读输入 x

// Fused 实现:
//   将 W_q, W_k, W_v 拼接为一个大矩阵 W_fused [d_model, 3*d_model]
//   [Q, K, V] = x @ W_fused  → 一次 cublas GEMM
//   = 1 次 kernel 启动 + 1 次读输入 x
//   然后 split 为 Q, K, V
```

## 六、Benchmark 方法

```python
# 每个优化级别测试:
#   1. 单次推理延迟 (latency): ms/batch
#   2. 吞吐量 (throughput): tokens/sec
#   3. GPU 显存占用: MB
#   4. 数值精度: 与 FP32 基线的最大误差

# 测量方法:
for _ in range(warmup_iters):   # 50 次 warmup
    model(src, tgt)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(bench_iters):    # 200 次测量
    model(src, tgt)
end.record()
torch.cuda.synchronize()
avg_ms = start.elapsed_time(end) / bench_iters
```

## 七、预期加速比

| 优化级别 | 预期延迟 | 预期加速比 | 显存 |
|----------|----------|-----------|------|
| Level 0: FP32 Eager | ~15 ms | 1.0x (基线) | 100% |
| Level 1: BF16 | ~8 ms | ~1.8x | ~55% |
| Level 2: torch.compile | ~10 ms | ~1.5x | ~80% |
| Level 3: 定制算子 + BF16 | ~5 ms | ~3.0x | ~55% |
| Level 4: TensorRT | ~3 ms | ~5.0x | ~50% |

> 实际数据以运行结果为准，上表为粗略估计

## 八、执行顺序

```
1. dataset.py          → 下载数据、构建词表、DataLoader
2. model.py            → 定义 Transformer 模型
3. 01_train.py         → 训练 ~10 epochs，保存 best_model.pt
4. custom_ops/         → 编写并编译定制 CUDA 算子
5. 02_baseline.py      → 加载模型，FP32 推理基准测试
6. 03_optimized.py     → 逐级优化，汇总对比表
```

## 九、环境信息

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| VRAM | 15.9 GB |
| CUDA Version | 13.1 (Driver 592.01) |
| PyTorch | 2.13.0.dev20260507+cu132 |
| Python | 3.12.3 |
| BF16 支持 | 是 |
