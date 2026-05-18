"""
最小 Transformer 完整示例
========================
任务: 字符级序列复制 (Copy Task)
  - 输入一段随机数字序列, 模型学会将输入原样输出
  - 这是验证 Transformer 能否正常工作的经典 toy task
  - 包含完整的 Encoder-Decoder 架构、训练、推理流程

词表定义:
  0 = PAD  (填充符, 用于对齐不同长度的序列)
  1 = SOS  (Start Of Sentence, 解码器起始标记)
  2 = EOS  (End Of Sentence, 解码器终止标记)
  3~12 = 数字 0~9

数据流:
  Encoder 输入:  [3, 7, 5, 9]          (原始数字序列)
  Decoder 输入:  [SOS, 3, 7, 5, 9, EOS] (teacher forcing, 训练时使用真实目标)
  Decoder 目标:  [3, 7, 5, 9, EOS]       (模型需要预测的序列)
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================================
# 位置编码 (Positional Encoding)
# ----------------------------------------------------------------------
# Transformer 本身没有顺序概念 (与 RNN 不同), 需要显式注入位置信息。
# 使用原始论文中的正弦-余弦编码:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# 其中 pos 是位置索引, i 是维度索引。
# 这种编码的好处: 模型可以通过学习线性组合来推断相对位置。
# ======================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 模型维度 (每个 token 的嵌入向量长度)
            max_len: 预计算的最大序列长度
        """
        super().__init__()
        # pe: (max_len, d_model) — 预计算所有位置编码
        pe = torch.zeros(max_len, d_model)
        # position: (max_len, 1) — 位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term: (d_model/2,) — 频率项, 控制不同维度的正弦波频率
        # 偶数维度和奇数维度共享同一组频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数列: sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列: cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: (1, max_len, d_model) — 增加 batch 维度
        # register_buffer: 不参与梯度更新, 但会随模型一起保存/加载
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — 输入嵌入
        Returns:
            (batch, seq_len, d_model) — 输入嵌入 + 位置编码
        """
        # 取前 seq_len 个位置编码, 加到输入上 (广播机制)
        return x + self.pe[:, :x.size(1)]


# ======================================================================
# 最小 Transformer 模型 (Encoder-Decoder 架构)
# ----------------------------------------------------------------------
# 结构:
#   Encoder: Embedding + PositionalEncoding → N 层 Encoder Layer
#     每层: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
#   Decoder: Embedding + PositionalEncoding → N 层 Decoder Layer
#     每层: Masked Multi-Head Self-Attention → Add & Norm
#           → Cross-Attention(attend to encoder output) → Add & Norm
#           → FFN → Add & Norm
#   Output: Linear projection → softmax → 词表上的概率分布
#
# 关键超参数:
#   d_model=64: 嵌入维度 (原论文 512, 这里缩小以加速训练)
#   nhead=4: 注意力头数 (原论文 8)
#   num_layers=2: Encoder/Decoder 层数 (原论文 6)
#   dim_feedforward=128: FFN 中间层维度 (原论文 2048)
# ======================================================================
class MiniTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, pad_id=0):
        """
        Args:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型嵌入维度
            nhead: 多头注意力的头数 (d_model 必须能被 nhead 整除)
            num_layers: Encoder 和 Decoder 的层数
            dim_feedforward: FFN (前馈网络) 隐藏层维度
            pad_id: 填充 token 的 id, 用于 embedding 层和 loss 忽略
        """
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # --- Embedding 层: 将 token id 映射为 d_model 维向量 ---
        # padding_idx=pad_id: PAD token 的嵌入始终为零向量, 且不更新梯度
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        # 位置编码 (源和目标共享同一个 PositionalEncoding 实例)
        self.pos_enc = PositionalEncoding(d_model)

        # --- Transformer 核心模块 (PyTorch 内置实现) ---
        # batch_first=True: 输入格式为 (batch, seq_len, d_model)
        # batch_first=False (默认): 输入格式为 (seq_len, batch, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        # --- 输出投影层: 将 d_model 维向量映射回词表大小的 logits ---
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def _make_mask(self, tgt):
        """
        生成因果掩码 (Causal Mask / Look-ahead Mask)
        防止 decoder 在预测第 t 个 token 时看到第 t+1, t+2, ... 的信息

        生成的掩码是一个上三角矩阵 (不含对角线), 形状 (seq_len, seq_len):
          [[0, -inf, -inf],
           [0,    0, -inf],
           [0,    0,    0]]
        其中 0 表示允许关注, -inf 表示禁止关注 (softmax 后变为 0)
        """
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        return tgt_mask

    def forward(self, src, tgt):
        """
        Args:
            src: (batch, src_seq_len) — 源序列 token ids
            tgt: (batch, tgt_seq_len) — 目标序列 token ids (decoder 输入)
        Returns:
            logits: (batch, tgt_seq_len, tgt_vocab_size) — 每个位置上词表的原始分数
        """
        # 1. Embedding + 缩放 + 位置编码
        # 缩放因子 sqrt(d_model): 原论文中的技巧, 使嵌入值与位置编码量级匹配
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        # 2. 生成因果掩码
        tgt_mask = self._make_mask(tgt)

        # 3. 通过 Transformer: Encoder 处理 src_emb, Decoder 基于 src_emb 和 tgt_emb 生成输出
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)

        # 4. 线性投影到词表大小
        return self.fc_out(out)


# ======================================================================
# 数据准备: 字符级复制任务
# ======================================================================

# 特殊 token 定义
PAD = 0   # 填充符 — 用于 batch 中不同长度序列的对齐
SOS = 1   # Start Of Sentence — decoder 的起始输入
EOS = 2   # End Of Sentence — 标记生成结束

# 词表大小: 3个特殊token + 10个数字(0-9) = 13
vocab_size = 13
# 最大序列长度 (随机序列的最大长度)
max_seq_len = 6


def generate_batch(batch_size=32):
    """
    随机生成一个 batch 的训练数据

    示例 (seq_len=4):
      src:        [3, 7, 5, 9]              — 随机数字序列
      tgt_input:  [1, 3, 7, 5, 9, 2]       — SOS + src + EOS (decoder 输入)
      tgt_output: [3, 7, 5, 9, 2]           — src + EOS (训练目标)

    训练时使用 Teacher Forcing:
      - 将真实目标序列(而非模型自己的预测)作为 decoder 的输入
      - 这样每一步预测只需要做 "一步" 的预测, 大大加速收敛

    Returns:
        src:        (batch, seq_len) — encoder 输入
        tgt_input:  (batch, seq_len+2) — decoder 输入 (SOS + src + EOS)
        tgt_output: (batch, seq_len+1) — 训练目标 (src + EOS)
    """
    # 随机序列长度: [3, max_seq_len) 之间
    seq_len = torch.randint(3, max_seq_len, (1,)).item()
    # 随机数字: token ids 在 [3, vocab_size) 范围内, 即数字 0~9
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    # decoder 输入: SOS + 原始序列 + EOS
    tgt_input = torch.cat([
        torch.full((batch_size, 1), SOS),   # 开头加 SOS
        src,                                  # 原始序列
        torch.full((batch_size, 1), EOS),   # 结尾加 EOS
    ], dim=1)
    # 训练目标: 原始序列 + EOS
    tgt_output = torch.cat([
        src,                                  # 原始序列
        torch.full((batch_size, 1), EOS),   # 结尾加 EOS
    ], dim=1)
    return src, tgt_input, tgt_output


# ======================================================================
# 训练循环
# ======================================================================

# 自动选择设备: GPU 优先, 没有 GPU 则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型 (源和目标词表大小相同, 因为是复制任务)
model = MiniTransformer(vocab_size, vocab_size).to(device)

# Adam 优化器 — 自适应学习率, 适合 Transformer 训练
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 交叉熵损失函数
# ignore_index=PAD: 忽略 PAD token 的损失, 避免填充位置影响梯度
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# 切换到训练模式 (启用 dropout 等训练专用层)
model.train()

# 训练 2001 步
for step in range(2001):
    # 生成一个 batch 的随机训练数据
    src, tgt_in, tgt_out = generate_batch()
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

    # 前向传播: 得到每个位置上词表的 logits
    logits = model(src, tgt_in)  # (batch, tgt_seq_len, vocab_size)

    # 计算损失: logits 长度与 tgt_input 一致, 但 tgt_output 少一个 SOS, 所以截掉 logits 最后一列
    # logits[:, :-1] → 与 tgt_out 长度对齐
    loss = criterion(logits[:, :-1].reshape(-1, vocab_size), tgt_out.reshape(-1))

    # 反向传播三件套:
    optimizer.zero_grad()  # 1. 清空上一步的梯度
    loss.backward()        # 2. 反向传播, 计算梯度
    optimizer.step()       # 3. 更新参数

    # 每 500 步打印一次损失
    if step % 500 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")


# ======================================================================
# 推理: 贪心解码 (Greedy Decoding)
# ----------------------------------------------------------------------
# 与训练时不同, 推理时没有真实目标可用 (没有 teacher forcing)
# 模型需要自回归地逐步生成:
#   1. 输入 SOS, 预测第一个 token
#   2. 将预测结果拼接到已有序列, 再预测下一个 token
#   3. 重复直到生成 EOS 或达到最大长度
# ======================================================================

model.eval()  # 切换到评估模式 (关闭 dropout)


def greedy_decode(src, max_len=10):
    """
    贪心解码: 每步选择概率最高的 token

    Args:
        src: (1, src_seq_len) — 输入序列
        max_len: 最大生成长度 (防止无限循环)
    Returns:
        ys: (1, generated_len) — 生成的 token 序列 (包含 SOS 和 EOS)
    """
    src = src.to(device)
    # 初始化 decoder 输入为 SOS
    ys = torch.tensor([[SOS]], device=device)  # (1, 1)

    for _ in range(max_len):
        # 前向传播: 得到所有位置的 logits
        logits = model(src, ys)  # (1, cur_len, vocab)
        # 取最后一个位置的 logits, 选择概率最大的 token
        next_token = logits[:, -1].argmax(-1).unsqueeze(1)  # (1, 1)
        # 将新 token 拼接到已有序列
        ys = torch.cat([ys, next_token], dim=1)  # (1, cur_len+1)
        # 如果生成了 EOS, 停止解码
        if next_token.item() == EOS:
            break
    return ys


# 测试 3 个样本, 验证模型是否能正确复制输入
for _ in range(3):
    # 随机生成一个测试样本
    src = torch.randint(3, vocab_size, (1, torch.randint(3, max_seq_len, (1,)).item()))
    # 贪心解码
    pred = greedy_decode(src)
    # 将 token id 转回数字字符 (id-3 对应数字 0~9)
    src_str = "".join(str(t - 3) for t in src[0].tolist())
    pred_str = "".join(str(t - 3) for t in pred[0].tolist() if t not in (SOS, EOS))
    print(f"Input: {src_str}  =>  Predicted: {pred_str}  {'✓' if src_str == pred_str else '✗'}")
