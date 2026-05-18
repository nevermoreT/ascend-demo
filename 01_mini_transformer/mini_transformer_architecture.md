# MiniTransformer 网络架构详解

## 整体结构（Encoder-Decoder）

```
┌─────────────────────────────────────────────────────┐
│                    Encoder 侧                        │
│                                                     │
│  src ids ──→ Embedding ──→ ×√d_model ──→ + PosEnc  │
│                                    │                │
│                          ┌─────────▼──────────┐     │
│                          │ Encoder Layer ×2    │     │
│                          │  ├─ Multi-Head      │     │
│                          │  │  Self-Attention   │     │
│                          │  ├─ Add & LayerNorm │     │
│                          │  ├─ FFN (64→128→64) │     │
│                          │  └─ Add & LayerNorm │     │
│                          └─────────┬──────────┘     │
│                                    │                │
│                              encoder_output         │
└────────────────────────────────────┬────────────────┘
                                     │
                              Cross-Attention
                              (decoder 关注 encoder)
                                     │
┌────────────────────────────────────▼────────────────┐
│                    Decoder 侧                        │
│                                                     │
│  tgt ids ──→ Embedding ──→ ×√d_model ──→ + PosEnc  │
│                                    │                │
│                          ┌─────────▼──────────┐     │
│                          │ Decoder Layer ×2    │     │
│                          │  ├─ Masked Multi-   │     │
│                          │  │  Head Self-Attn  │     │
│                          │  ├─ Add & LayerNorm │     │
│                          │  ├─ Cross-Attention │     │
│                          │  │  (attend encoder)│     │
│                          │  ├─ Add & LayerNorm │     │
│                          │  ├─ FFN (64→128→64) │     │
│                          │  └─ Add & LayerNorm │     │
│                          └─────────┬──────────┘     │
│                                    │                │
│                             Linear(64 → 13)         │
│                                    │                │
│                              logits / softmax        │
│                                    │                │
│                            predicted token           │
└─────────────────────────────────────────────────────┘
```

## 各组件详解

### 1. Embedding 层

将离散的 token id（整数）映射为 64 维连续向量。`padding_idx=0` 让 PAD token 的向量保持为零且不参与梯度更新。

### 2. 缩放 + 位置编码

- `×√d_model`：Embedding 初始值较小（~1/√d_model），乘以 √64=8 使其与位置编码量级匹配
- 位置编码用正弦/余弦函数生成，让模型知道每个 token 的位置。不同维度用不同频率，低维度变化快（捕捉局部位置），高维度变化慢（捕捉全局位置）

### 3. Encoder（2 层）

每层包含两个子层：

```
┌─────────────────────────────────────────┐
│ Multi-Head Self-Attention (4 heads)     │  每个位置关注序列中所有位置
│   Q = K = V = encoder 输入              │  4个头分别关注不同的模式
│   head_dim = 64/4 = 16                  │
├─────────────────────────────────────────┤
│ Add & Layer Norm                        │  残差连接 + 归一化
├─────────────────────────────────────────┤
│ FFN: Linear(64→128) → ReLU → Linear(128→64)  逐位置的前馈网络
├─────────────────────────────────────────┤
│ Add & Layer Norm                        │  残差连接 + 归一化
└─────────────────────────────────────────┘
```

Self-Attention 的核心计算：

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

- Q、K、V 由输入线性变换得到
- `QKᵀ` 计算每对位置间的相似度（注意力分数）
- 除以 `√d_k` 防止数值过大导致 softmax 梯度消失
- softmax 归一化后作为权重，对 V 做加权求和

### 4. Decoder（2 层）

每层包含**三个**子层，比 Encoder 多一个 Cross-Attention：

```
┌─────────────────────────────────────────┐
│ Masked Multi-Head Self-Attention        │  decoder 关注自己，但只能看过去
│   + 因果掩码 (causal mask)              │  未来位置设为 -inf → softmax后为0
├─────────────────────────────────────────┤
│ Add & Layer Norm                        │
├─────────────────────────────────────────┤
│ Cross-Attention                         │  Q来自decoder, K/V来自encoder
│   Q = decoder 输出                       │  让decoder"看到"输入序列
│   K = V = encoder 输出                   │
├─────────────────────────────────────────┤
│ Add & Layer Norm                        │
├─────────────────────────────────────────┤
│ FFN: Linear(64→128) → ReLU → Linear(128→64)
├─────────────────────────────────────────┤
│ Add & Layer Norm                        │
└─────────────────────────────────────────┘
```

### 5. 因果掩码（Causal Mask）

```
位置:  0    1    2    3
     ┌──────────────────┐
  0  │  0  -inf -inf -inf│   位置0只能看自己
  1  │  0    0  -inf -inf│   位置1能看0,1
  2  │  0    0    0  -inf│   位置2能看0,1,2
  3  │  0    0    0    0 │   位置3能看所有
     └──────────────────┘
```

`-inf` 经过 softmax 后变为 0，确保预测第 t 个 token 时不会"偷看"第 t+1 个及之后的 token。

### 6. 输出投影

`Linear(64 → 13)` 将每个位置的 64 维向量映射到 13 维（词表大小），得到每个 token 的原始分数（logits），再通过交叉熵损失隐式做 softmax。

## 数据流总结（以输入 `[3, 7, 5]` 为例）

```
训练阶段 (Teacher Forcing):
  src        = [3, 7, 5]              → Encoder
  tgt_in     = [SOS, 3, 7, 5, EOS]   → Decoder 输入
  tgt_out    = [3, 7, 5, EOS]         → 训练目标

  Encoder:  [3,7,5] → Embed → PosEnc → 2层SelfAttn+FFN → memory
  Decoder:  [SOS,3,7,5,EOS] → Embed → PosEnc
            → MaskedSelfAttn (看自己过去的)
            → CrossAttn (看 encoder memory)
            → FFN → Linear → logits [batch, 5, 13]

推理阶段 (自回归):
  步骤1: tgt=[SOS]        → 模型预测 3
  步骤2: tgt=[SOS, 3]     → 模型预测 7
  步骤3: tgt=[SOS, 3, 7]  → 模型预测 5
  步骤4: tgt=[SOS,3,7,5]  → 模型预测 EOS → 停止
```

## 与原论文的参数对比

| 参数 | 原论文 (Attention Is All You Need) | 本实例 |
|------|-----------------------------------|--------|
| d_model | 512 | 64 |
| nhead | 8 | 4 |
| head_dim | 64 | 16 |
| num_layers | 6 | 2 |
| dim_feedforward | 2048 | 128 |
| 总参数量 | ~65M | ~0.1M |

缩小了约 600 倍，使得在几秒内就能在单卡上完成训练，但架构完全一致。
