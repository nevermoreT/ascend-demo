"""
交互式英→德翻译 (支持自定义算子加速)
用法:
  python3 translate.py            ← 标准推理
  python3 translate.py --fused    ← 自定义 CUDA 算子 + BF16 加速
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_data, SOS, EOS, PAD
from model import TranslationTransformer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FUSED = "--fused" in sys.argv


def load_model_and_data():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    _, _, src_vocab, tgt_vocab = load_data(batch_size=1, max_len=64)

    dtype = torch.bfloat16 if USE_FUSED else torch.float32
    model = TranslationTransformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=1024, dropout=0.1, pad_id=PAD,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if USE_FUSED:
        from custom_ops import fused_layernorm
        model.pos_enc.pe = model.pos_enc.pe.to(torch.bfloat16)
        for name, module in model.transformer.named_modules():
            if isinstance(module, nn.LayerNorm):
                weight = module.weight
                bias = module.bias
                eps = module.eps
                module.forward = lambda x, _w=weight, _b=bias, _e=eps: \
                    fused_layernorm(x, _w, _b, _e)
        print("  ✓ 自定义 CUDA 算子 (FusedLayerNorm) 已启用")
        print("  ✓ BF16 半精度已启用")

    return model, src_vocab, tgt_vocab


def translate(model, src_vocab, tgt_vocab, sentence, max_len=80):
    src_ids = src_vocab.encode(sentence.lower(), max_len=64)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        t0 = time.time()
        output_ids = model.greedy_decode(src_tensor, max_len=max_len, sos_id=SOS, eos_id=EOS)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000

    result = tgt_vocab.decode(output_ids[0].tolist())
    return result, elapsed


def main():
    print("加载模型中...")
    model, src_vocab, tgt_vocab = load_model_and_data()
    mode = "Fused CUDA + BF16" if USE_FUSED else "FP32 Eager (基线)"
    print(f"设备: {device}")
    print(f"GPU:  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
    print(f"模式: {mode}")
    print()
    print("=" * 60)
    print("  英→德 翻译器 (Multi30k)")
    print(f"  推理模式: {mode}")
    print("  输入英文句子, 回车获得德语翻译")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    total_time = 0
    count = 0

    while True:
        try:
            sentence = input("\n英文> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not sentence:
            continue
        if sentence.lower() in ("quit", "exit", "q"):
            if count > 0:
                print(f"\n平均推理延迟: {total_time/count:.1f} ms ({count} 句)")
            print("再见!")
            break

        result, elapsed = translate(model, src_vocab, tgt_vocab, sentence)
        total_time += elapsed
        count += 1
        print(f"德语> {result}")
        print(f"耗时: {elapsed:.1f} ms")


if __name__ == "__main__":
    main()
