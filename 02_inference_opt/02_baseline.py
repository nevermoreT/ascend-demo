"""
02_baseline.py
基线推理性能测试: FP32 Eager Mode (Level 0)
"""

import os
import sys
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_data, PAD
from model import TranslationTransformer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")


def load_model(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = TranslationTransformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=1024, dropout=0.1, pad_id=PAD,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def benchmark(model, src, tgt_in, warmup=50, iters=200):
    """使用 CUDA Event 精确计时"""
    device = src.device

    for _ in range(warmup):
        _ = model(src, tgt_in)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = model(src, tgt_in)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    peak_mem = (torch.cuda.max_memory_allocated(device) - mem_before) / 1024 / 1024
    return avg_ms, peak_mem


def main():
    device = torch.device("cuda")
    print(f"=== 基线推理 (FP32 Eager Mode) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    model = load_model(device)
    _, val_loader, src_vocab, tgt_vocab = load_data(batch_size=1, max_len=64)
    src, tgt_in, tgt_out = next(iter(val_loader))
    src, tgt_in = src.to(device), tgt_in.to(device)

    print(f"输入形状: src={src.shape}, tgt={tgt_in.shape}")

    # 正确性验证
    with torch.no_grad():
        logits = model(src, tgt_in)
        pred = logits.argmax(-1)
        src_str = src_vocab.decode(src[0].tolist())
        tgt_str = tgt_vocab.decode(tgt_out[0].tolist())
        pred_str = tgt_vocab.decode(pred[0].tolist())
    print(f"源文:   {src_str}")
    print(f"参考:   {tgt_str}")
    print(f"预测:   {pred_str}")

    # 性能测试: 不同 batch size
    print(f"\n{'Batch':>6} | {'延迟(ms)':>10} | {'吞吐(tok/s)':>12} | {'显存(MB)':>10}")
    print("-" * 50)

    results = {}
    for bs in [1, 4, 8, 16, 32]:
        src_batch = src.repeat(bs, 1)
        tgt_batch = tgt_in.repeat(bs, 1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            avg_ms, peak_mem = benchmark(model, src_batch, tgt_batch)
        total_tokens = tgt_batch.numel()
        throughput = total_tokens / (avg_ms / 1000)
        results[bs] = {"latency": avg_ms, "throughput": throughput, "memory": peak_mem}
        print(f"{bs:>6} | {avg_ms:>10.3f} | {throughput:>12.0f} | {peak_mem:>10.1f}")

    print("\n基线测试完成")
    return results


if __name__ == "__main__":
    main()
