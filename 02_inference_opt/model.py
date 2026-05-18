"""
Transformer 英→德翻译模型
Encoder-Decoder 架构，支持导出为优化推理模式
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        emb = self.tgt_emb(tgt)
        return mask.to(device=tgt.device, dtype=emb.dtype)

    def _make_padding_mask(self, x):
        return (x == self.pad_id)

    def forward(self, src, tgt):
        scale = math.sqrt(self.d_model)
        src_emb = self.pos_enc(self.src_emb(src) * scale)
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * scale)

        tgt_mask = self._make_tgt_mask(tgt)
        src_key_padding_mask = self._make_padding_mask(src)
        tgt_key_padding_mask = self._make_padding_mask(tgt)

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=80, sos_id=1, eos_id=2):
        self.eval()
        src = src.to(next(self.parameters()).device)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits = self(src, ys)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_id)
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return ys
