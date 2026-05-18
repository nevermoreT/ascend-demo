# Embedding 详解

## 一句话理解

Embedding 就是把离散的符号（如整数 id）映射成连续的浮点向量。

## 为什么需要 Embedding

神经网络只能处理**连续数值**，不能直接处理"第3个词"这样的离散 id。如果用 one-hot 编码：

```
词表大小 = 13
token id = 5  →  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   ← 13维，只有1个1
```

问题：
- **维度灾难**：词表 10 万词就需要 10 万维向量
- **语义缺失**：id=3 和 id=4 的向量距离，与 id=3 和 id=12 的距离完全相同（都是 √2），无法表达"相似性"

## Embedding 怎么做

用一个**可学习的矩阵** `W`，形状 `(vocab_size, d_model)`：

```
W = 一个 13×64 的矩阵（本例中）

token id = 5  →  直接取 W 的第 5 行  →  一个 64 维浮点向量
                          ↓
               [0.23, -0.71, 0.05, ..., 0.42]   ← 64维，每个值都是可学习的参数
```

对应代码 (`mini_transformer.py:110`)：

```python
self.src_emb = nn.Embedding(13, 64)  # 13个token, 每个映射到64维
```

## Embedding 的本质

```
id: 3 (数字0)  →  [0.23, -0.71, 0.05, ...]   ─┐
id: 4 (数字1)  →  [0.25, -0.68, 0.09, ...]   ─┤  相似的数字 → 向量相近
id: 5 (数字2)  →  [0.22, -0.70, 0.06, ...]   ─┘

id: 0  (PAD)   →  [0.00,  0.00, 0.00, ...]      ← padding_idx 固定为零向量
```

训练过程中，这个矩阵会**自动学习**，使得：

1. 语义/功能相似的 token → 向量距离近
2. 不同的 token → 向量距离远
3. 64 维足以编码丰富的信息（比 13 维 one-hot 更紧凑、更有表达力）

## 直觉类比

把 Embedding 想象成给每个 token 分配一个"身份证"，上面有 64 个特征描述（语义、语法、位置倾向等）。这些特征不是手工设计的，而是**通过训练自动发现的**。

## 本项目中的 Embedding

在本项目的 MiniTransformer 中，有两组独立的 Embedding：

```python
# 源序列 Embedding (Encoder 端)
self.src_emb = nn.Embedding(src_vocab_size=13, d_model=64, padding_idx=0)

# 目标序列 Embedding (Decoder 端)
self.tgt_emb = nn.Embedding(tgt_vocab_size=13, d_model=64, padding_idx=0)
```

参数说明：
- `src_vocab_size=13`：词表大小（PAD + SOS + EOS + 数字0~9）
- `d_model=64`：每个 token 映射为 64 维向量
- `padding_idx=0`：id=0 (PAD) 的嵌入始终为零向量，不参与梯度更新

数据流：

```
输入:  [3, 7, 5]   (整数 token ids)
        ↓
Embedding: 取矩阵的第3、7、5行
        ↓
输出:  [[0.23, -0.71, ...],   ← token 3 的 64 维向量
        [0.55,  0.12, ...],   ← token 7 的 64 维向量
        [0.41, -0.33, ...]]   ← token 5 的 64 维向量
        ↓
乘以 √d_model = √64 = 8  (缩放，匹配位置编码量级)
        ↓
加上 Positional Encoding
        ↓
送入 Transformer
```

## 进阶：为什么 Embedding 有效

从数学角度看，Embedding 矩阵是一个**查找表 (Lookup Table)**：

```
W = | w₀  (PAD, 零向量)        |
    | w₁  (SOS)                |
    | w₂  (EOS)                |
    | w₃  (数字0)              |
    | w₄  (数字1)              |
    | ...                      |
    | w₁₂ (数字9)              |

维度: 13 × 64
```

`nn.Embedding` 的前向传播就是 `W[input]` — 根据索引取行向量。

训练时，通过反向传播，每一行向量会被调整到"对任务最有利"的位置。
向量空间中，功能相似的 token 会自然聚类，这就是 Embedding 能表达语义的根本原因。

## 与 One-Hot 的对比

| 特性 | One-Hot | Embedding |
|------|---------|-----------|
| 维度 | = 词表大小 (13) | = d_model (64)，与词表无关 |
| 稀疏性 | 极度稀疏 (只有1个1) | 稠密 (每个维度都有值) |
| 相似性 | 任意两个向量距离相同 | 相似token距离近 |
| 可学习 | 否 (固定的) | 是 (随训练更新) |
| 存储 | 词表大时浪费空间 | 固定大小，高效 |
