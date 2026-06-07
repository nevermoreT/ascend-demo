# Ascend Tutorial — Transformer 推理优化实战

> 在 NVIDIA RTX 5080 (Blackwell) 上，从零实现 Transformer 翻译模型，编写定制 CUDA 算子进行推理加速，并完成全链路性能对比。

## 目录

- [项目概览](#项目概览)
- [环境信息](#环境信息)
- [项目结构](#项目结构)
- [Part 1：最小 Transformer](#part-1)
- [Part 2：推理优化](#part-2)
  - [训练](#训练)
  - [交互式翻译](#交互式翻译)
  - [性能基准测试](#性能基准测试)
  - [定制 CUDA 算子](#定制-cuda-算子)
- [性能对比结果](#性能对比结果)
- [文档索引](#文档索引)
- [快速开始](#快速开始)

---

## 项目概览

本项目包含两个渐进式实战模块：

| 模块 | 内容 | 关键技术 |
|:---|:---|:---|
| **Part 1** | 最小 Transformer (字符级复制) | Encoder-Decoder、Self-Attention、因果掩码、位置编码 |
| **Part 2** | 英→德翻译 + 推理优化 | Multi30k、Fused CUDA Kernel、BF16、torch.compile |

核心成果：通过手写 CUDA 算子 (FusedLayerNorm + FusedSoftmaxMask) + BF16 半精度，在 RTX 5080 上实现 **1.67x 推理加速 + 50% 显存节省**。

---

## 环境信息

| 项目 | 值 |
|:---|:---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU (16GB, sm_120) |
| CUDA | 13.2 (Driver 592.01) |
| PyTorch | 2.13.0.dev+cu132 (nightly) |
| Python | 3.12.3 |
| OS | Ubuntu 24.04 |
| nvcc | 13.2, V13.2.78 |

---

## 项目结构

```
ascend-tutorial/
├── README.md                               ← 你在这里
├── .gitignore
│
├── 01_mini_transformer/                    ← Part 1: 最小 Transformer
│   ├── mini_transformer.py                 #   完整可运行代码 (含训练+推理)
│   ├── mini_transformer_architecture.md    #   网络架构详解
│   └── embedding_explained.md              #   Embedding 原理详解
│
├── 02_inference_opt/                       ← Part 2: 推理优化
│   ├── dataset.py                          #   Multi30k 数据加载 (HuggingFace)
│   ├── model.py                            #   Transformer 翻译模型 (12.4M 参数)
│   ├── 01_train.py                         #   训练脚本 (10 epochs)
│   ├── 02_baseline.py                      #   FP32 基线性能测试
│   ├── 03_optimized.py                     #   多级优化推理对比
│   ├── translate.py                        #   交互式英→德翻译器
│   ├── best_model.pt                       #   训练好的权重 (gitignore)
│   │
│   ├── custom_ops/                         #   定制 CUDA 算子
│   │   ├── __init__.py                     #     Python 入口 (懒加载)
│   │   ├── build.py                        #     JIT 编译脚本
│   │   ├── fused_ops.cpp                   #     C++ pybind11 绑定
│   │   ├── fused_layernorm.h               #     FusedLayerNorm 头文件
│   │   ├── fused_layernorm.cu              #     FusedLayerNorm CUDA kernel
│   │   ├── fused_softmax_mask.h            #     FusedSoftmaxMask 头文件
│   │   └── fused_softmax_mask.cu           #     FusedSoftmaxMask CUDA kernel
│   │
│   └── docs/                               #   详细文档
│       ├── inference_opt_plan.md           #     方案设计
│       ├── benchmarks.md                   #     性能对比 + 实现问题总结
│       ├── custom_ops_analysis.md          #     算子深度解析
│       ├── end_to_end_example.md           #     端到端数据流计算实例
│       └── gpu_thread_mapping_detail.md    #     GPU 线程映射详解
│
├── vector_add.cu                           #   CUDA 向量加法示例
└── Makefile
```

---

<a id="part-1"></a>

## Part 1：最小 Transformer

一个完整的 Encoder-Decoder Transformer，任务是将输入数字序列原样复制输出。

**模型参数**：d_model=64, nhead=4, layers=2, ~0.1M 参数

```bash
cd 01_mini_transformer/
python3 mini_transformer.py
```

输出：
```
Step    0 | Loss: 2.7927
Step  500 | Loss: 0.7797
Step 1000 | Loss: 0.3263
Step 1500 | Loss: 0.1821
Step 2000 | Loss: 0.0944
Input: 1134  =>  Predicted: 1134  ✓
Input: 9431  =>  Predicted: 9431  ✓
Input: 7881  =>  Predicted: 7881  ✓
```

| 文档 | 内容 |
|:---|:---|
| [mini_transformer_architecture.md](01_mini_transformer/mini_transformer_architecture.md) | 架构图示、各组件详解、数据流、与原论文参数对比 |
| [embedding_explained.md](01_mini_transformer/embedding_explained.md) | Embedding 原理、与 one-hot 对比、为什么有效 |

---

<a id="part-2"></a>

## Part 2：推理优化

### 训练

使用 Multi30k 英→德翻译数据集，训练 10 个 epoch：

```bash
cd 02_inference_opt/
python3 01_train.py
```

训练结果：
```
Epoch  1 | Train Loss: 4.6781 | Val Loss: 3.7718 | Time: 15.1s
Epoch  5 | Train Loss: 2.9109 | Val Loss: 2.9840 | Time: 14.2s
Epoch 10 | Train Loss: 2.3010 | Val Loss: 2.8404 | Time: 14.4s
模型参数量: 12.43M
```

### 交互式翻译

```bash
# 标准推理 (FP32)
python3 translate.py

# 自定义 CUDA 算子 + BF16 加速
python3 translate.py --fused
```

示例：
```
英文> a man is running
德语> ein mann springt über einen berg.

英文> two children play in the water
德语> zwei kinder spielen im wasser.
```

### 性能基准测试

```bash
# 基线测试
python3 02_baseline.py

# 全部优化级别对比
python3 03_optimized.py
```

### 定制 CUDA 算子

编译并验证：
```bash
python3 -c "from custom_ops import fused_layernorm, fused_softmax_mask; print('OK')"
```

算子自动 JIT 编译，首次 import 时触发，后续调用无额外开销。

---

## 性能对比结果

**测试条件**：RTX 5080, batch_size=16, 序列长度 ~16 tokens

### 延迟 (ms, 越低越好)

| 优化级别 | Batch=1 | Batch=4 | Batch=8 | Batch=16 | Batch=32 |
|:---|:---:|:---:|:---:|:---:|:---:|
| FP32 Eager (基线) | 4.18 | 4.49 | 4.85 | 4.69 | 4.68 |
| BF16 半精度 | 4.28 | 4.87 | 5.39 | 5.17 | 6.20 |
| torch.compile | 2.90 | 3.39 | 3.24 | 3.27 | 3.65 |
| **Fused CUDA + BF16** | **2.72** | **2.74** | **2.88** | **2.80** | **3.11** |

### 加速比 (vs FP32 基线)

| 优化级别 | Batch=1 | Batch=16 | 说明 |
|:---|:---:|:---:|:---|
| BF16 半精度 | 0.98x | 0.91x | sm_120 的 BF16 SDPA 未优化，反而变慢 |
| torch.compile | 1.44x | 1.44x | 算子融合 + 图优化，零代码改动 |
| **Fused CUDA + BF16** | **1.54x** | **1.67x** | 手写 kernel + 半精度，最优方案 |

### 显存节省

| 优化级别 | Batch=16 显存 | 节省 |
|:---|:---:|:---:|
| FP32 Eager | 0.5 MB | - |
| BF16 | 0.3 MB | 50% |
| Fused CUDA + BF16 | 0.3 MB | 50% |

### 算子精度验证

| 算子 | 最大误差 (vs PyTorch 原生) | 影响 |
|:---|:---:|:---|
| FusedLayerNorm | 4.77e-7 | 无损 |
| FusedSoftmaxMask | 5.96e-8 | 无损 |

---

## 文档索引

### Part 1 文档

| 文档 | 内容 |
|:---|:---|
| [网络架构详解](01_mini_transformer/mini_transformer_architecture.md) | ASCII 架构图、Self-Attention 公式、Encoder/Decoder 各子层、数据流示例 |
| [Embedding 详解](01_mini_transformer/embedding_explained.md) | 为什么需要 Embedding、查找表机制、与 one-hot 对比、训练过程 |

### Part 2 文档

| 文档 | 内容 |
|:---|:---|
| [方案设计](02_inference_opt/docs/inference_opt_plan.md) | 整体方案、模型规格、5 级优化策略、定制算子设计、预期加速比 |
| [性能对比 + 问题总结](02_inference_opt/docs/benchmarks.md) | 最终性能数据、实现过程 6 大问题、最终评估结论 |
| [算子深度解析](02_inference_opt/docs/custom_ops_analysis.md) | 三层调用链、.cu kernel 逐行注释、pybind11 绑定、JIT 编译加载流程、monkey-patch 集成 |
| [端到端数据流实例](02_inference_opt/docs/end_to_end_example.md) | 从 `"A man runs"` 出发，追踪 token→Embedding→QKV→Attention→LayerNorm 全流程，含具体数值计算 |
| [GPU 线程映射详解](02_inference_opt/docs/gpu_thread_mapping_detail.md) | Grid/Block/Thread 映射、Shared Memory 二叉树归约动画、显存访问模式对比、大 hidden_size 处理 |

---

## 快速开始

### 1. 安装依赖

```bash
# PyTorch (需 nightly 以支持 RTX 5080 Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu132

# HuggingFace datasets (Multi30k 数据加载)
pip install datasets

# CUDA 扩展编译工具
pip install ninja
sudo apt-get install python3-dev
```

### 2. 运行 Part 1

```bash
cd 01_mini_transformer/
python3 mini_transformer.py
```

### 3. 运行 Part 2

```bash
cd 02_inference_opt/

# 训练 (约 2 分钟)
python3 01_train.py

# 交互式翻译
python3 translate.py --fused

# 性能基准测试
python3 03_optimized.py
```

### 4. 阅读文档

从 [端到端数据流实例](02_inference_opt/docs/end_to_end_example.md) 开始，这是最直观的入门文档。
